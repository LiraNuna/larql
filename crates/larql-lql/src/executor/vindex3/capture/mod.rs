//! `CAPTURE <site> [AT LAYER <n> | AT LAYERS (…)] FOR "<prompt>" [CONTINUE … | GENERATE <n>] [POSITIONS …] TO
//! "<file>.npy"`: the values the session's program computed at one site, written as a NumPy array with a sidecar
//! that names the model state that computed them.
//!
//! Nothing here computes a forward pass of its own. Every value is one the executor produced while running the
//! session's program, over the operands `INFER` resolves (the stored weights, or the patched bytes of the
//! stored-weight ops in effect, or a refusal where INFER refuses), observed directly from the one-shot batch
//! prefill traversal INFER runs ([`batch`]).

mod batch;
mod geometry;
mod provenance;
mod sequence;

pub use geometry::CaptureGeometry;
pub(crate) use provenance::AppliedPatchFile;

use super::npy::{npy_sha256, NpyDescr, NpyWriter};
use super::npy_output::{
    container_index, container_path, index_list, literal, output_paths, remove_partial,
    write_sidecar, LARQL_VERSION,
};
use super::{prepared_operands_for_session, v3_continuation};
use crate::ast::{CaptureContinuation, CaptureLayers, CapturePositions, CaptureSite};
use crate::error::LqlError;
use crate::executor::{Backend, Session};

const STATEMENT: &str = "CAPTURE";
/// The sidecar's schema id.
const SIDECAR_SCHEMA: &str = "larql-capture/v1";
/// The array type every capture writes: the f32 values the executor held.
const DESCR: NpyDescr = NpyDescr::F4;
/// The execution path a capture reads, as the sidecar names it.
const PATH_INFER_PASS: &str = "the one-shot traversal INFER runs, observed";
/// How a prompt is encoded, as the sidecar says it.
const PROMPT_ENCODING: &str =
    "INFER's raw-prompt encoder: the tokenizer's own special tokens, plus the \
     container's declared BOS where the tokenizer did not add it; no chat template";
const CONTINUATION_ENCODING: &str = "no special tokens";

/// The rows of one block: one `[width]` vector per captured position.
type Rows = Vec<Vec<f32>>;

fn refused(e: impl std::fmt::Display) -> LqlError {
    super::npy_output::refused(STATEMENT, e)
}

/// What one capture computes: the site, its layers, the sequence and the positions kept. The arguments of a
/// `CAPTURE` statement other than its output.
#[derive(Debug, Clone, Copy)]
pub struct CaptureSpec<'a> {
    pub site: CaptureSite,
    pub layers: Option<&'a CaptureLayers>,
    pub prompt: &'a str,
    pub continuation: Option<&'a CaptureContinuation>,
    pub positions: &'a CapturePositions,
}

/// The arguments of one single-site `CAPTURE`, as the statement gives them.
pub(crate) struct CaptureRequest<'a> {
    pub spec: CaptureSpec<'a>,
    pub output: &'a str,
}

/// The arguments of one multi-site `CAPTURE (...) TO "<dir>"`.
pub(crate) struct CaptureMultiRequest<'a> {
    pub sites: &'a [CaptureSite],
    pub layers: Option<&'a CaptureLayers>,
    pub prompt: &'a str,
    pub continuation: Option<&'a CaptureContinuation>,
    pub positions: &'a CapturePositions,
    pub output_dir: &'a str,
}

/// The values one capture computed, in memory, with the provenance `CAPTURE`'s sidecar records.
#[derive(Debug, Clone)]
pub struct Captured {
    /// `[positions, width]`, or `[layers, positions, width]` for a list of layers.
    pub shape: Vec<usize>,
    /// The sidecar's fields for the same capture, with `statement` null and `npy_sha256` the digest of the
    /// `.npy` file the statement would write.
    pub provenance: serde_json::Value,
    blocks: Vec<Rows>,
    summary: CaptureSummary,
}

impl Captured {
    /// The values, C order, in the array's shape.
    pub fn values(&self) -> Vec<f32> {
        self.blocks.iter().flatten().flatten().copied().collect()
    }
}

/// What the statement's reply says beyond the provenance.
#[derive(Debug, Clone)]
struct CaptureSummary {
    sequence_len: usize,
    prompt_len: usize,
    positions: Vec<usize>,
    executes: &'static str,
    path: &'static str,
    start: std::time::Instant,
}

/// The values as the `.npy` body holds them: little-endian f32, C order.
fn body_bytes(blocks: &[Rows]) -> Vec<u8> {
    blocks
        .iter()
        .flatten()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect()
}

/// The statement as LQL text, in its canonical form: what the sidecar records.
fn statement_text(request: &CaptureRequest<'_>) -> String {
    let spec = &request.spec;
    let mut text = format!("{STATEMENT} {}", spec.site.name());
    match spec.layers {
        Some(CaptureLayers::One(layer)) => text.push_str(&format!(" AT LAYER {layer}")),
        Some(CaptureLayers::Many(layers)) => {
            text.push_str(&format!(" AT LAYERS ({})", index_list(layers)))
        }
        None => {}
    }
    text.push_str(&format!(" FOR {}", literal(spec.prompt)));
    match spec.continuation {
        Some(CaptureContinuation::Text(t)) => text.push_str(&format!(" CONTINUE {}", literal(t))),
        Some(CaptureContinuation::Tokens(ids)) => {
            text.push_str(&format!(" CONTINUE TOKENS ({})", index_list(ids)))
        }
        Some(CaptureContinuation::Generate(n)) => text.push_str(&format!(" GENERATE {n}")),
        None => {}
    }
    text.push_str(&format!(
        " POSITIONS {} TO {};",
        spec.positions.spelled(),
        literal(request.output)
    ));
    text
}

/// Every block holds one row per position, and every row has the same width.
/// For `attn.weights`, each raw row holds `[num_q_heads, seq_len]` flattened.
/// - Single-position captures (`POSITIONS LAST`) export `[n_layers, n_q_heads, seq_len]`
///   (or `[n_q_heads, seq_len]` for a single layer).
/// - Multi-position captures (`POSITIONS (a..b)`, `POSITIONS ALL`) transpose each layer's
///   `[n_pos, n_q_heads, seq_len]` into `[n_q_heads, n_pos, seq_len]` so the exported shape is
///   `[n_layers, n_q_heads, n_pos, seq_len]` (or `[n_q_heads, n_pos, seq_len]` for a single layer).
fn finalize_blocks_and_shape(
    site: CaptureSite,
    blocks: &mut [Rows],
    positions: usize,
    stacked: bool,
    seq_len: usize,
) -> Result<Vec<usize>, LqlError> {
    let width = blocks.first().and_then(|b| b.first()).map_or(0, Vec::len);
    let consistent = !blocks.is_empty()
        && width > 0
        && blocks
            .iter()
            .all(|b| b.len() == positions && b.iter().all(|row| row.len() == width));
    if !consistent {
        return Err(LqlError::Execution(format!(
            "{STATEMENT}: the executor did not report one value of a single width at every requested layer \
             and position"
        )));
    }
    if site == CaptureSite::AttnWeights {
        if seq_len == 0 || !width.is_multiple_of(seq_len) {
            return Err(LqlError::Execution(format!(
                "{STATEMENT}: attn.weights row width {width} is not a multiple of sequence length {seq_len}"
            )));
        }
        let n_heads = width / seq_len;
        if positions > 1 {
            // Transpose each layer block from `[positions, n_heads, seq_len]` to
            // `[n_heads, positions, seq_len]` (stored as `n_heads` rows of `positions * seq_len`).
            for block in blocks.iter_mut() {
                let mut transposed: Rows = Vec::with_capacity(n_heads);
                for h in 0..n_heads {
                    let mut head_plane = Vec::with_capacity(positions * seq_len);
                    for pos_row in block.iter() {
                        head_plane.extend_from_slice(&pos_row[h * seq_len..(h + 1) * seq_len]);
                    }
                    transposed.push(head_plane);
                }
                *block = transposed;
            }
        }
        let mut shape = Vec::with_capacity(4);
        if stacked {
            shape.push(blocks.len());
        }
        shape.push(n_heads);
        if positions > 1 {
            shape.push(positions);
        }
        shape.push(seq_len);
        return Ok(shape);
    }
    Ok(if stacked {
        vec![blocks.len(), positions, width]
    } else {
        vec![positions, width]
    })
}

/// Write the rows, block after block, as one C-order array.
fn write_npy(path: &std::path::Path, shape: &[usize], blocks: &[Rows]) -> Result<String, LqlError> {
    let mut writer = NpyWriter::create_new(path, DESCR, shape)
        .map_err(|e| LqlError::exec(format!("create {}", path.display()), e))?;
    let body: std::io::Result<()> = blocks.iter().flatten().try_for_each(|row| {
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        writer.write(&bytes)
    });
    let finished = body.and_then(|()| writer.finish());
    finished.map_err(|e| {
        remove_partial(
            &[path],
            LqlError::exec(format!("write {}", path.display()), e),
        )
    })
}

impl Session {
    /// The typed operation behind `CAPTURE`: the values the session's program computes at one site, in memory,
    /// with their provenance. Nothing is written.
    pub fn capture(&self, spec: &CaptureSpec<'_>) -> Result<Captured, LqlError> {
        let (mut captured, ()) = self.compute_capture(spec, || Ok(()))?;
        let npy_sha256 = npy_sha256(DESCR, &captured.shape, &body_bytes(&captured.blocks))
            .map_err(|e| LqlError::exec("describe the array", e))?;
        captured.provenance["npy_sha256"] = npy_sha256.into();
        Ok(captured)
    }

    /// Compute a capture; the provenance's `statement` and `npy_sha256` are left null. `check_output` runs once the site and layers are checked and before anything is
    /// computed, where the statement checks its output, so a refusal names the same first problem either way.
    fn compute_capture<T>(
        &self,
        spec: &CaptureSpec<'_>,
        check_output: impl FnOnce() -> Result<T, LqlError>,
    ) -> Result<(Captured, T), LqlError> {
        let (root, runtime, tokenizer, overlay, bos, patch_files, prepared_cache) = match &self
            .backend
        {
            Backend::Vindex3 {
                path,
                runtime,
                tokenizer,
                overlay,
                bos_token,
                patch_files,
                prepared_cache,
                ..
            } => (
                path,
                runtime,
                tokenizer,
                overlay,
                *bos_token,
                patch_files,
                prepared_cache,
            ),
            Backend::None => return Err(LqlError::NoBackend),
            _ => return Err(refused(
                "CAPTURE runs a VINDEX3 container's program, and this session is not bound to one",
            )),
        };
        let tokenizer = tokenizer.as_ref().ok_or_else(|| {
            LqlError::Execution(
                "CAPTURE needs a tokenizer and this container carries no tokenizer.json — \
                 token-id capability only"
                    .into(),
            )
        })?;
        let site = spec.site;
        let layers = sequence::layers(site, spec.layers, runtime.plan().layers.len())?;
        let checked_output = check_output()?;
        let (index, index_sha256) = container_index(root)?;
        let start = std::time::Instant::now();

        let sequence = sequence::build(
            runtime,
            overlay,
            prepared_cache,
            tokenizer,
            bos,
            spec.prompt,
            spec.continuation,
        )?;
        let positions = sequence::positions(spec.positions, &sequence)?;
        let (ops, overrides, cache_key) =
            prepared_operands_for_session(runtime, overlay, prepared_cache)?;
        let continuation = v3_continuation(runtime)?;
        let mut blocks = batch::capture(
            runtime,
            &ops,
            cache_key.as_ref().map(|k| (prepared_cache.prefix(), k)),
            &continuation,
            &sequence.ids,
            site,
            &layers,
            &positions,
        )?;
        let path = PATH_INFER_PASS;
        let cross_check = serde_json::Value::Null;
        let stacked = matches!(spec.layers, Some(CaptureLayers::Many(_)));
        let shape = finalize_blocks_and_shape(
            site,
            &mut blocks,
            positions.len(),
            stacked,
            sequence.ids.len(),
        )?;

        let decode = |id: u32| tokenizer.decode(&[id], false).unwrap_or_default();
        let position_records: Vec<serde_json::Value> = positions
            .iter()
            .map(|&p| {
                serde_json::json!({
                    "position": p,
                    "part": if p < sequence.prompt_len { "prompt" } else { "continuation" },
                    "token_id": sequence.ids[p],
                    "token": decode(sequence.ids[p]),
                    "predicts_token_id": sequence.predicts(p),
                })
            })
            .collect();
        let executes = provenance::executes(overlay, overrides.is_some());
        let provenance = serde_json::json!({
            "schema": SIDECAR_SCHEMA,
            "statement": serde_json::Value::Null,
            "container": container_path(root),
            "index_sha256": index_sha256,
            "model": index.model,
            "site": site.name(),
            "layers": if site.per_layer() { serde_json::json!(layers) } else { serde_json::Value::Null },
            "positions": position_records,
            "sequence": {
                "token_ids": sequence.ids,
                "prompt_tokens": sequence.prompt_len,
                "prompt": spec.prompt,
                "prompt_encoding": PROMPT_ENCODING,
                "bos_token_id": bos,
                "continuation": sequence.continuation,
                "continuation_encoding": CONTINUATION_ENCODING,
            },
            "executes": executes,
            "patches": provenance::patches(overlay, patch_files),
            "live_tensors": provenance::live_tensors(overlay),
            "execution": {
                "path": path,
                "binding": provenance::binding(runtime),
                "cross_check": cross_check,
            },
            "dtype": DESCR.label(),
            "shape": shape,
            // Filled in by whoever holds the array: the file's digest, or the digest of the file it would be.
            "npy_sha256": serde_json::Value::Null,
            "larql_version": LARQL_VERSION,
        });
        let captured = Captured {
            shape,
            provenance,
            summary: CaptureSummary {
                sequence_len: sequence.ids.len(),
                prompt_len: sequence.prompt_len,
                positions,
                executes,
                path,
                start,
            },
            blocks,
        };
        Ok((captured, checked_output))
    }

    pub(crate) fn exec_capture(
        &self,
        request: &CaptureRequest<'_>,
    ) -> Result<Vec<String>, LqlError> {
        let (captured, (npy_path, sidecar_path)) =
            self.compute_capture(&request.spec, || output_paths(STATEMENT, request.output))?;
        let Captured {
            shape,
            mut provenance,
            blocks,
            summary,
        } = captured;
        let npy_sha256 = write_npy(&npy_path, &shape, &blocks)?;
        let elapsed_ms = summary.start.elapsed().as_secs_f64() * 1000.0;

        provenance["statement"] = statement_text(request).into();
        provenance["npy_sha256"] = npy_sha256.clone().into();
        write_sidecar(&sidecar_path, &provenance, &[&npy_path])?;

        Ok(vec![
            format!(
                "Captured {} {:?} ({}) to {}",
                request.spec.site.name(),
                shape,
                DESCR.label(),
                npy_path.display()
            ),
            format!(
                "  sequence: {} token(s), {} from the prompt; positions {}",
                summary.sequence_len,
                summary.prompt_len,
                index_list(&summary.positions)
            ),
            format!("  executes: {} weights; {}", summary.executes, summary.path),
            format!("  sha256:  {npy_sha256}"),
            format!("  sidecar: {}", sidecar_path.display()),
            format!("  {elapsed_ms:.0}ms"),
        ])
    }

    /// Execute `CAPTURE (<site1>, <site2>, …) … TO "<dir>"` in a single batch prefill traversal,
    /// writing `<dir>/<site>.npy` and `<dir>/<site>.npy.json` for every requested site.
    pub(crate) fn exec_capture_multi(
        &self,
        request: &CaptureMultiRequest<'_>,
    ) -> Result<Vec<String>, LqlError> {
        let (root, runtime, tokenizer, overlay, bos, patch_files, prepared_cache) = match &self
            .backend
        {
            Backend::Vindex3 {
                path,
                runtime,
                tokenizer,
                overlay,
                bos_token,
                patch_files,
                prepared_cache,
                ..
            } => (
                path,
                runtime,
                tokenizer,
                overlay,
                *bos_token,
                patch_files,
                prepared_cache,
            ),
            Backend::None => return Err(LqlError::NoBackend),
            _ => return Err(refused(
                "CAPTURE runs a VINDEX3 container's program, and this session is not bound to one",
            )),
        };
        let tokenizer = tokenizer.as_ref().ok_or_else(|| {
            LqlError::Execution(
                "CAPTURE needs a tokenizer and this container carries no tokenizer.json — \
                 token-id capability only"
                    .into(),
            )
        })?;
        if request.sites.is_empty() {
            return Err(refused("CAPTURE (...) requires at least one site"));
        }
        let mut seen = std::collections::BTreeSet::new();
        if let Some(dup) = request.sites.iter().find(|s| !seen.insert(**s)) {
            return Err(refused(format!("site {} is listed twice", dup.name())));
        }
        let plan_layers = runtime.plan().layers.len();
        let any_per_layer = request.sites.iter().any(|s| s.per_layer());
        let layers = if any_per_layer {
            let first_per_layer = request
                .sites
                .iter()
                .copied()
                .find(|s| s.per_layer())
                .unwrap();
            sequence::layers(first_per_layer, request.layers, plan_layers)?
        } else {
            sequence::layers(request.sites[0], request.layers, plan_layers)?
        };

        let out_dir = std::path::Path::new(request.output_dir);
        if out_dir.is_file() {
            return Err(refused(format!(
                "output {} is an existing file, expected a directory for multi-site CAPTURE",
                out_dir.display()
            )));
        }
        let mut planned_paths: Vec<(CaptureSite, std::path::PathBuf, std::path::PathBuf)> =
            Vec::with_capacity(request.sites.len());
        for &site in request.sites {
            let npy = out_dir.join(format!("{}.npy", site.name()));
            let sidecar = std::path::PathBuf::from(format!("{}.json", npy.display()));
            if npy.exists() {
                return Err(refused(format!("{} already exists", npy.display())));
            }
            if sidecar.exists() {
                return Err(refused(format!("{} already exists", sidecar.display())));
            }
            planned_paths.push((site, npy, sidecar));
        }

        let (index, index_sha256) = container_index(root)?;
        let start = std::time::Instant::now();
        let sequence = sequence::build(
            runtime,
            overlay,
            prepared_cache,
            tokenizer,
            bos,
            request.prompt,
            request.continuation,
        )?;
        let positions = sequence::positions(request.positions, &sequence)?;
        let (ops, overrides, cache_key) =
            prepared_operands_for_session(runtime, overlay, prepared_cache)?;
        let continuation = v3_continuation(runtime)?;
        let mut all_blocks = batch::capture_many(
            runtime,
            &ops,
            cache_key.as_ref().map(|k| (prepared_cache.prefix(), k)),
            &continuation,
            &sequence.ids,
            request.sites,
            &layers,
            &positions,
        )?;

        std::fs::create_dir_all(out_dir)
            .map_err(|e| LqlError::exec(format!("create directory {}", out_dir.display()), e))?;

        let decode = |id: u32| tokenizer.decode(&[id], false).unwrap_or_default();
        let position_records: Vec<serde_json::Value> = positions
            .iter()
            .map(|&p| {
                serde_json::json!({
                    "position": p,
                    "part": if p < sequence.prompt_len { "prompt" } else { "continuation" },
                    "token_id": sequence.ids[p],
                    "token": decode(sequence.ids[p]),
                    "predicts_token_id": sequence.predicts(p),
                })
            })
            .collect();
        let executes = provenance::executes(overlay, overrides.is_some());
        let stacked = matches!(request.layers, Some(CaptureLayers::Many(_)));

        let mut written_files: Vec<std::path::PathBuf> = Vec::new();
        let mut out_lines = vec![format!(
            "Captured {} site(s) in 1 pass to {}:",
            planned_paths.len(),
            out_dir.display()
        )];
        for (site, npy_path, sidecar_path) in &planned_paths {
            let mut blocks = all_blocks.remove(site).unwrap_or_default();
            let site_stacked = site.per_layer() && stacked;
            let shape = match finalize_blocks_and_shape(
                *site,
                &mut blocks,
                positions.len(),
                site_stacked,
                sequence.ids.len(),
            ) {
                Ok(s) => s,
                Err(e) => {
                    let refs: Vec<&std::path::Path> = written_files
                        .iter()
                        .map(std::path::PathBuf::as_path)
                        .collect();
                    return Err(remove_partial(&refs, e));
                }
            };
            let npy_sha256 = match write_npy(npy_path, &shape, &blocks) {
                Ok(digest) => {
                    written_files.push(npy_path.clone());
                    digest
                }
                Err(e) => {
                    let refs: Vec<&std::path::Path> = written_files
                        .iter()
                        .map(std::path::PathBuf::as_path)
                        .collect();
                    return Err(remove_partial(&refs, e));
                }
            };
            let single_req = CaptureRequest {
                spec: CaptureSpec {
                    site: *site,
                    layers: if site.per_layer() {
                        request.layers
                    } else {
                        None
                    },
                    prompt: request.prompt,
                    continuation: request.continuation,
                    positions: request.positions,
                },
                output: &npy_path.display().to_string(),
            };
            let provenance = serde_json::json!({
                "schema": SIDECAR_SCHEMA,
                "statement": statement_text(&single_req),
                "container": container_path(root),
                "index_sha256": index_sha256,
                "model": index.model,
                "site": site.name(),
                "layers": if site.per_layer() { serde_json::json!(layers) } else { serde_json::Value::Null },
                "positions": position_records,
                "sequence": {
                    "token_ids": sequence.ids,
                    "prompt_tokens": sequence.prompt_len,
                    "prompt": request.prompt,
                    "prompt_encoding": PROMPT_ENCODING,
                    "bos_token_id": bos,
                    "continuation": sequence.continuation,
                    "continuation_encoding": CONTINUATION_ENCODING,
                },
                "executes": executes,
                "patches": provenance::patches(overlay, patch_files),
                "live_tensors": provenance::live_tensors(overlay),
                "execution": {
                    "path": PATH_INFER_PASS,
                    "binding": provenance::binding(runtime),
                    "cross_check": serde_json::Value::Null,
                },
                "dtype": DESCR.label(),
                "shape": shape,
                "npy_sha256": npy_sha256,
                "larql_version": LARQL_VERSION,
            });
            let refs: Vec<&std::path::Path> = written_files
                .iter()
                .map(std::path::PathBuf::as_path)
                .collect();
            write_sidecar(sidecar_path, &provenance, &refs)?;
            written_files.push(sidecar_path.clone());
            out_lines.push(format!(
                "  {:<20} {:?} ({}) → {}  sha256 {}",
                site.name(),
                shape,
                DESCR.label(),
                npy_path.display(),
                &npy_sha256[..16],
            ));
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        out_lines.push(format!(
            "  sequence: {} token(s), {} from the prompt; positions {}; executes: {} weights; {elapsed_ms:.0}ms",
            sequence.ids.len(),
            sequence.prompt_len,
            index_list(&positions),
            executes,
        ));
        Ok(out_lines)
    }
}
