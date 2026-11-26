//! `ADD LOW RANK`: stage a structured `add_low_rank_columns` (VLP numerics v2) on the bound VINDEX3 container.

use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::weight_patch::structured::{OperandBits, Profile};
use larql_vindex::format::vindex3::weight_patch::LowRankColumnsOp;
use serde::Deserialize;

use super::npy::{read_npy_floats, NpyFloatData};
use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// Hex digits of one operand element in an f64-working profile, and in the f32 one.
const F64_HEX_DIGITS: usize = 16;
const F32_HEX_DIGITS: usize = 8;

/// One factor inside a JSON `ADD LOW RANK` operand file: either an array of fixed-width lowercase
/// hex strings (exact bits) or a path to a C-order `<f4`/`<f8` `.npy` file.
#[derive(Deserialize)]
#[serde(untagged)]
enum FactorSpec {
    Hex(Vec<String>),
    NpyPath(String),
}

/// The operand file an `ADD LOW RANK` reads: the rank, both factors as stored bits (or `.npy` file
/// paths) in the profile's working width, and optionally where they came from.
/// `a` is `rows × rank` and `b` is `rank × |columns|`, both row-major.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LowRankFile {
    rank: usize,
    a: FactorSpec,
    b: FactorSpec,
    #[serde(default)]
    origin: Option<String>,
}

fn refused(e: impl std::fmt::Display) -> LqlError {
    LqlError::Execution(format!("ADD LOW RANK refused: {e}"))
}

/// Fixed-width lowercase hex, as the profile's working format.
fn operand_bits(texts: &[String], f64_width: bool, which: &str) -> Result<OperandBits, LqlError> {
    let width = if f64_width {
        F64_HEX_DIGITS
    } else {
        F32_HEX_DIGITS
    };
    let bits = texts
        .iter()
        .map(|h| {
            (h.len() == width && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
                .then(|| u64::from_str_radix(h, 16).ok())
                .flatten()
                .ok_or_else(|| {
                    refused(format!(
                        "`{h}` in `{which}` is not {width} lowercase hex digits"
                    ))
                })
        })
        .collect::<Result<Vec<u64>, _>>()?;
    Ok(if f64_width {
        OperandBits::F64(bits)
    } else {
        OperandBits::F32(bits.into_iter().map(|b| b as u32).collect())
    })
}

/// Operand values handed over in memory rather than in an operand file.
#[derive(Debug, Clone, Copy)]
pub enum OperandValues<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
}

/// The operand bits `values` are in the working width of the profile `numerics`: f64 values for an f64-working
/// profile (f32 values are widened, exactly), f32 values for the f32 one (f64 values are refused, since
/// narrowing rounds). The same [`OperandBits`] an operand file's hex parses to.
pub fn operand_bits_from_values(
    numerics: &str,
    values: OperandValues<'_>,
    which: &str,
) -> Result<OperandBits, LqlError> {
    let profile = Profile::parse(Some(numerics)).map_err(refused)?;
    match (profile.works_in_f64(), values) {
        (true, OperandValues::F64(v)) => Ok(OperandBits::F64(v.iter().map(|x| x.to_bits()).collect())),
        (true, OperandValues::F32(v)) => Ok(OperandBits::F64(
            v.iter().map(|&x| f64::from(x).to_bits()).collect(),
        )),
        (false, OperandValues::F32(v)) => Ok(OperandBits::F32(v.iter().map(|x| x.to_bits()).collect())),
        (false, OperandValues::F64(_)) => Err(refused(format!(
            "`{which}` holds f64 values and `{numerics}` works in f32; narrowing would round them, so pass f32 \
             values"
        ))),
    }
}

/// The origin an op records for operands handed over in memory: where they came from, each factor's shape and
/// working width, and the SHA-256 of its bits as little-endian bytes in that width (for f64 operands, the bytes
/// of a C-order `<f8` array of the values).
pub fn operand_origin(
    source: &str,
    a: &OperandBits,
    a_shape: [usize; 2],
    b: &OperandBits,
    b_shape: [usize; 2],
) -> String {
    use sha2::{Digest, Sha256};
    let describe = |bits: &OperandBits, [rows, cols]: [usize; 2]| {
        let (width, bytes): (&str, Vec<u8>) = match bits {
            OperandBits::F64(v) => ("f64", v.iter().flat_map(|b| b.to_le_bytes()).collect()),
            OperandBits::F32(v) => ("f32", v.iter().flat_map(|b| b.to_le_bytes()).collect()),
        };
        format!("{rows}x{cols} {width} sha256 {:x}", Sha256::digest(&bytes))
    };
    format!(
        "{source}: a {}; b {}",
        describe(a, a_shape),
        describe(b, b_shape)
    )
}

/// What staging one `add_low_rank_columns` did.
#[derive(Debug, Clone)]
pub struct StagedLowRank {
    /// The op as staged, sealed: what a recording saves into a `.vlp`.
    pub op: LowRankColumnsOp,
    /// How many elements of the tensor the op changes, resolved by the frozen interpreter.
    pub changed_elements: usize,
    /// Every term has a zero factor: the op compiles to the base's exact bytes.
    pub sham: bool,
    /// Structured ops pending COMPILE in this session after staging.
    pub pending_structured_ops: usize,
    /// The live-weights report the statement prints after staging.
    live: Vec<String>,
}

fn vindex3_path(session: &Session) -> Result<std::path::PathBuf, LqlError> {
    match &session.backend {
        Backend::Vindex3 { path, .. } => Ok(path.clone()),
        Backend::Vindex { .. } => Err(LqlError::Execution(
            "ADD LOW RANK edits the stored weights of a VINDEX3 container; \
             this session is bound to a VINDEX2 vindex"
                .into(),
        )),
        _ => Err(LqlError::NoBackend),
    }
}

impl Session {
    /// The typed operation behind `ADD LOW RANK`: stage `a · b` (`a` is `rows × rank`, `b` is
    /// `rank × |columns|`, both row-major, in the profile's working width) on `columns` of `tensor`. Everything
    /// checkable is checked before anything is staged: the profile, the tensor, the columns, and that the op,
    /// resolved by the frozen interpreter, edits at least one element. A `BEGIN PATCH` recording records it.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_low_rank(
        &mut self,
        tensor: &str,
        columns: &[usize],
        numerics: &str,
        rank: usize,
        a: &OperandBits,
        b: &OperandBits,
        origin: Option<String>,
    ) -> Result<StagedLowRank, LqlError> {
        let path = vindex3_path(self)?;
        Profile::parse(Some(numerics)).map_err(refused)?;
        let mut op = LowRankColumnsOp::bind(&path, tensor, numerics, rank, a, b, columns.to_vec())
            .map_err(refused)?;
        if let Some(origin) = origin {
            op = op.with_origin(origin);
        }
        // Resolve now, so a bad op is refused when it is written down and not when it is compiled.
        let resolved = op.resolve_tensor(&path).map_err(refused)?;
        let sham = resolved.is_sham().map_err(refused)?;
        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("matched above")
        };
        overlay
            .stage_low_rank_columns_op(op.clone())
            .map_err(refused)?;
        let pending_structured_ops = overlay.structured_op_count();
        if let Some(recording) = &mut self.patch_recording {
            recording
                .operations
                .push(larql_vindex::PatchOp::LowRankColumnsWeight(op.clone()));
        }
        let live = self.refresh_live_weights();
        Ok(StagedLowRank {
            op,
            changed_elements: resolved.changed_elements,
            sham,
            pending_structured_ops,
            live,
        })
    }

    /// `ADD LOW RANK`: the operand file (`.json` or `.npy`) is read and parsed, then staged by [`Self::stage_low_rank`].
    pub(crate) fn exec_add_low_rank(
        &mut self,
        operands: &str,
        tensor: &str,
        columns: &[usize],
        heads: Option<&[usize]>,
        numerics: &str,
    ) -> Result<Vec<String>, LqlError> {
        vindex3_path(self)?;
        let resolved_cols: Vec<usize> = match heads {
            Some(hs) => {
                let Backend::Vindex3 { runtime, .. } = &self.backend else {
                    unreachable!("checked above")
                };
                super::weights::resolve_attention_heads(runtime.plan(), tensor, hs, false)
                    .map_err(refused)?
            }
            None => columns.to_vec(),
        };
        let columns = &resolved_cols;
        let profile = Profile::parse(Some(numerics)).map_err(refused)?;
        let (rank, a, b, origin) = if Path::new(operands)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("npy"))
        {
            load_direct_npy_low_rank(
                Path::new(operands),
                operands,
                columns.len(),
                numerics,
                &profile,
            )?
        } else {
            let text = std::fs::read_to_string(operands)
                .map_err(|e| refused(format!("cannot read operand file {operands:?}: {e}")))?;
            let file: LowRankFile = serde_json::from_str(&text)
                .map_err(|e| refused(format!("operand file {operands:?}: {e}")))?;
            let base_dir = Path::new(operands).parent();
            let (a, a_note) = load_factor_spec(&file.a, base_dir, numerics, &profile, "a")?;
            let (b, b_note) = load_factor_spec(&file.b, base_dir, numerics, &profile, "b")?;
            let origin = match (file.origin, a_note, b_note) {
                (Some(o), _, _) => Some(o),
                (None, Some(an), Some(bn)) => Some(format!("json:{operands} ({an}; {bn})")),
                (None, Some(note), None) | (None, None, Some(note)) => {
                    Some(format!("json:{operands} ({note})"))
                }
                (None, None, None) => None,
            };
            (file.rank, a, b, origin)
        };
        let staged = self.stage_low_rank(tensor, columns, numerics, rank, &a, &b, origin)?;
        let mut out = vec![
            format!(
                "Staged: add rank-{rank} update to {tensor}, {} column(s) {}..={} under {numerics}; \
                 {} element(s) would change",
                columns.len(),
                columns.first().copied().unwrap_or_default(),
                columns.last().copied().unwrap_or_default(),
                staged.changed_elements
            ),
            format!("  op digest {}", &staged.op.digest[..16]),
            if staged.sham {
                "  zero-valued (every term has a zero factor): a declared sham, compiled to the base's exact bytes".into()
            } else {
                "  results are BF16 as the profile defines them; an element whose terms all have a zero factor keeps its bits".into()
            },
            format!(
                "  {} structured op(s) pending COMPILE in this session",
                staged.pending_structured_ops
            ),
        ];
        out.extend(staged.live);
        Ok(out)
    }
}

fn load_factor_spec(
    spec: &FactorSpec,
    base_dir: Option<&Path>,
    numerics: &str,
    profile: &Profile,
    which: &str,
) -> Result<(OperandBits, Option<String>), LqlError> {
    match spec {
        FactorSpec::Hex(hex) => Ok((operand_bits(hex, profile.works_in_f64(), which)?, None)),
        FactorSpec::NpyPath(raw) => {
            let resolved = resolve_factor_path(raw, base_dir);
            let npy = read_npy_floats(&resolved).map_err(refused)?;
            let dtype = npy.data.dtype_label();
            let bits = npy_data_to_bits(&npy.data, numerics, which)?;
            let note = format!(
                "{which}=npy:{raw} shape={:?} dtype={dtype} sha256={}",
                npy.shape, npy.sha256
            );
            Ok((bits, Some(note)))
        }
    }
}

fn resolve_factor_path(raw: &str, base_dir: Option<&Path>) -> PathBuf {
    let p = Path::new(raw);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    if let Some(dir) = base_dir {
        let candidate = dir.join(p);
        if candidate.exists() {
            return candidate;
        }
    }
    p.to_path_buf()
}

fn npy_data_to_bits(
    data: &NpyFloatData,
    numerics: &str,
    which: &str,
) -> Result<OperandBits, LqlError> {
    match data {
        NpyFloatData::F64(v) => operand_bits_from_values(numerics, OperandValues::F64(v), which),
        NpyFloatData::F32(v) => operand_bits_from_values(numerics, OperandValues::F32(v), which),
    }
}

/// Interpret a direct `.npy` file passed to `ADD LOW RANK`:
/// - 1D `[rows]` (or singleton-prefixed `[1, rows]` / `[1, 1, rows]`): rank-1 update with
///   `a = vec` (`rows × 1`) and `b = [1.0; n_cols]` (`1 × n_cols`).
/// - 2D `[rows, n_cols]` (or `[1, rows, n_cols]`): exact column delta matrix with
///   `rank = n_cols`, `a = mat` (`rows × n_cols`), and `b = I` (`n_cols × n_cols`).
fn load_direct_npy_low_rank(
    path: &Path,
    display: &str,
    n_cols: usize,
    numerics: &str,
    profile: &Profile,
) -> Result<(usize, OperandBits, OperandBits, Option<String>), LqlError> {
    if n_cols == 0 {
        return Err(refused("COLUMNS must not be empty"));
    }
    let npy = read_npy_floats(path).map_err(refused)?;
    let dtype = npy.data.dtype_label();
    // Strip leading singleton dimensions (e.g. CAPTURE `[1, hidden]` -> `[hidden]`).
    let mut trimmed = npy.shape.as_slice();
    while trimmed.len() > 1 && trimmed[0] == 1 {
        trimmed = &trimmed[1..];
    }
    let a = npy_data_to_bits(&npy.data, numerics, "a")?;
    let (rank, b) = match trimmed {
        [_rows] => {
            let b = if profile.works_in_f64() {
                OperandBits::F64(vec![1.0f64.to_bits(); n_cols])
            } else {
                OperandBits::F32(vec![1.0f32.to_bits(); n_cols])
            };
            (1usize, b)
        }
        [_rows, cols] if *cols == n_cols => {
            let rank = n_cols;
            if rank > 256 {
                return Err(refused(format!(
                    "2D .npy matrix {display:?} has {rank} columns, exceeding max rank 256; \
                     pass a JSON file referencing factored `a` and `b` .npy arrays"
                )));
            }
            let b = if profile.works_in_f64() {
                let mut id = vec![0.0f64.to_bits(); rank * rank];
                for i in 0..rank {
                    id[i * rank + i] = 1.0f64.to_bits();
                }
                OperandBits::F64(id)
            } else {
                let mut id = vec![0.0f32.to_bits(); rank * rank];
                for i in 0..rank {
                    id[i * rank + i] = 1.0f32.to_bits();
                }
                OperandBits::F32(id)
            };
            (rank, b)
        }
        _ => {
            return Err(refused(format!(
                "operand .npy file {display:?} has shape {:?}; expected [rows] (rank-1 broadcast \
                 to {n_cols} column(s)) or [rows, {n_cols}] (per-column delta matrix)",
                npy.shape
            )));
        }
    };
    let origin = format!(
        "npy:{display} shape={:?} dtype={dtype} sha256={}",
        npy.shape, npy.sha256
    );
    Ok((rank, a, b, Some(origin)))
}
