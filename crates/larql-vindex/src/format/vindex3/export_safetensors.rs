//! Exact, SHA-256-verified safetensors export for canonical VINDEX3 containers.
//!
//! Streams raw canonical tensor bytes directly from VINDEX3 segments into standard
//! Hugging Face `.safetensors` files (single-file `model.safetensors` or sharded
//! `model-00001-of-NNNNN.safetensors` + `model.safetensors.index.json`), reconstructing
//! every original HF tensor name from `system_graph.json`'s source bindings.
//!
//! Zero floating-point conversion, zero full-model RAM buffering, and independent
//! read-back SHA-256 verification of every exported tensor (including cross-checking
//! any `weight_patches.json` ledger).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::VindexError;
use crate::format::filenames::{INDEX_JSON, WEIGHT_PATCHES_JSON};
use crate::format::fresh_output::check_fresh_destination;
use crate::format::vindex3::encode::checkpoint::CHECKPOINT_CAPABILITY_FILES;
use crate::format::vindex3::encode::segment::{read_exact_at, read_segment_header};
use crate::format::vindex3::encode::SYSTEM_GRAPH_JSON;
use crate::format::vindex3::graph::{LogicalObject, SystemGraph};
use crate::format::vindex3::index::{ContainerAuthority, Vindex3Index};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::weight_patch::parse_ledger;

/// Schema tag written into `export_manifest.json`.
pub const EXPORT_MANIFEST_SCHEMA: &str = "larql-safetensors-export/v1";
/// Filename of the verification manifest written alongside exported safetensors.
pub const EXPORT_MANIFEST_JSON: &str = "export_manifest.json";
/// Default shard ceiling: 4 GiB per `.safetensors` file.
pub const DEFAULT_MAX_SHARD_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Streaming I/O chunk size (1 MiB).
const COPY_CHUNK_BYTES: usize = 1 << 20;

/// Valid Hugging Face safetensors dtype labels.
const SAFETENSORS_DTYPES: &[&str] = &[
    "BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "I16", "U16", "F16", "BF16", "I32", "U32", "F32",
    "I64", "U64", "F64",
];

/// Options controlling a safetensors export.
#[derive(Debug, Clone)]
pub struct SafetensorsExportOptions<'a> {
    /// Optional source HF checkpoint directory to copy `config.json` (and any missing
    /// capability files) from.
    pub source_checkpoint: Option<&'a Path>,
    /// Maximum payload bytes per `.safetensors` shard before sharding.
    pub max_shard_bytes: u64,
}

impl Default for SafetensorsExportOptions<'_> {
    fn default() -> Self {
        Self {
            source_checkpoint: None,
            max_shard_bytes: DEFAULT_MAX_SHARD_BYTES,
        }
    }
}

/// One exported shard file recorded in [`SafetensorsExportReport`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedShard {
    pub filename: String,
    pub tensor_count: usize,
    pub payload_bytes: u64,
    pub file_bytes: u64,
    pub sha256: String,
}

/// One exported tensor's provenance and SHA-256 verification record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportedTensorRecord {
    pub name: String,
    pub object: String,
    pub segment_tensor: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
    pub shard: String,
    pub sha256: String,
}

/// Manifest written to `export_manifest.json` and returned by [`export_safetensors`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SafetensorsExportReport {
    pub schema: String,
    pub model: String,
    pub source_index_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_patches_sha256: Option<String>,
    pub total_tensors: usize,
    pub total_bytes: u64,
    pub shards: Vec<ExportedShard>,
    pub tensors: Vec<ExportedTensorRecord>,
    pub copied_files: Vec<String>,
    pub verified: bool,
}

struct PlannedExportTensor {
    hf_name: String,
    object_id: String,
    segment_path: PathBuf,
    segment_tensor: String,
    dtype: String,
    shape: Vec<usize>,
    payload_offset: u64,
    len: u64,
}

struct ScratchDir(Option<PathBuf>);

impl Drop for ScratchDir {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::remove_dir_all(path);
        }
    }
}

/// Export a canonical VINDEX3 container to a Hugging Face safetensors directory with
/// full read-back SHA-256 verification.
///
/// `output_dir` must not exist. On any error, nothing is left at `output_dir`.
pub fn export_safetensors(
    container: &Path,
    output_dir: &Path,
    options: &SafetensorsExportOptions<'_>,
) -> Result<SafetensorsExportReport, VindexError> {
    if options.max_shard_bytes == 0 {
        return Err(VindexError::Parse(
            "max_shard_bytes must be greater than zero".into(),
        ));
    }
    if output_dir.exists() {
        return Err(VindexError::Parse(format!(
            "export destination `{}` already exists; refusing to overwrite",
            output_dir.display()
        )));
    }
    check_fresh_destination(container, output_dir)?;

    if let Some(ckpt) = options.source_checkpoint {
        if !ckpt.join("config.json").is_file() {
            return Err(VindexError::Parse(format!(
                "--source-checkpoint `{}` does not contain config.json",
                ckpt.display()
            )));
        }
    }

    let inspection = inspect_container(container, false)?;
    if !inspection.is_coherent() {
        return Err(VindexError::Parse(format!(
            "refusing to export an incoherent VINDEX3 container: {:?}",
            inspection.defects
        )));
    }

    let raw_index = std::fs::read_to_string(container.join(INDEX_JSON))?;
    let source_index_sha256 = hex_digest(raw_index.as_bytes());
    let index: Vindex3Index = serde_json::from_str(&raw_index)
        .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;

    if index.authority != ContainerAuthority::Canonical {
        return Err(VindexError::Parse(format!(
            "safetensors export requires a canonical VINDEX3 container, found authority `{:?}`",
            index.authority
        )));
    }

    let graph_name = index.system_graph.as_deref().unwrap_or(SYSTEM_GRAPH_JSON);
    let graph_raw = std::fs::read_to_string(container.join(graph_name))?;
    let graph: SystemGraph = serde_json::from_str(&graph_raw)
        .map_err(|e| VindexError::Parse(format!("parse {graph_name}: {e}")))?;

    // If a weight-patch ledger is present, parse it so we can cross-check result_tensor_sha256.
    let (weight_patches_sha256, ledger_tensor_hashes) = load_ledger_hashes(container, &index)?;

    let objects_by_id: BTreeMap<&str, &LogicalObject> =
        graph.objects.iter().map(|o| (o.id.as_str(), o)).collect();

    // Gather every canonical representation (one per logical object).
    let mut canonical_by_object: BTreeMap<
        &str,
        (&str, &crate::format::vindex3::index::RepresentationEntry),
    > = BTreeMap::new();
    for (rep_id, entry) in index
        .representations
        .iter()
        .filter(|(_, e)| e.compiled_from.is_none())
    {
        if canonical_by_object
            .insert(entry.object.as_str(), (rep_id.as_str(), entry))
            .is_some()
        {
            return Err(VindexError::Parse(format!(
                "object `{}` has more than one canonical representation",
                entry.object
            )));
        }
    }

    let mut planned: Vec<PlannedExportTensor> = Vec::new();
    let mut seen_hf_names: BTreeSet<String> = BTreeSet::new();
    let mut expected_total_bytes = 0u64;

    for object in &graph.objects {
        let Some((rep_id, entry)) = canonical_by_object.get(object.id.as_str()) else {
            return Err(VindexError::Parse(format!(
                "object `{}` has no canonical representation to export",
                object.id
            )));
        };
        let seg_path = container.join(&entry.segment);
        if !seg_path.is_file() {
            return Err(VindexError::Parse(format!(
                "canonical segment `{}` for `{rep_id}` is not resident on disk",
                entry.segment
            )));
        }
        let (header, payload_start) = read_segment_header(&seg_path)?;
        let strip = common_segment_prefix(
            &object
                .source_bindings
                .iter()
                .map(|b| b.tensor_prefix.as_str())
                .collect::<Vec<_>>(),
        );

        let mut seg_payload_bytes = 0u64;
        for tensor in &header.tensors {
            if !SAFETENSORS_DTYPES.contains(&tensor.dtype.as_str()) {
                return Err(VindexError::Parse(format!(
                    "tensor `{}/{}` has dtype `{}`, which is not a standard safetensors dtype",
                    object.id, tensor.name, tensor.dtype
                )));
            }
            let hf_name = reconstruct_hf_name(&strip, &tensor.name);
            if !seen_hf_names.insert(hf_name.clone()) {
                return Err(VindexError::Parse(format!(
                    "duplicate reconstructed HF tensor name `{hf_name}`"
                )));
            }
            planned.push(PlannedExportTensor {
                hf_name,
                object_id: object.id.clone(),
                segment_path: seg_path.clone(),
                segment_tensor: tensor.name.clone(),
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                payload_offset: payload_start + tensor.offset,
                len: tensor.len,
            });
            seg_payload_bytes += tensor.len;
        }
        if seg_payload_bytes != entry.payload_bytes {
            return Err(VindexError::Parse(format!(
                "segment `{}` tensor lengths sum to {seg_payload_bytes}, index declares {}",
                entry.segment, entry.payload_bytes
            )));
        }
        expected_total_bytes += seg_payload_bytes;
    }

    if planned.is_empty() {
        return Err(VindexError::Parse(
            "container holds no canonical tensors to export".into(),
        ));
    }

    // Verify all canonical representations belong to objects in the graph.
    for obj_id in canonical_by_object.keys() {
        if !objects_by_id.contains_key(obj_id) {
            return Err(VindexError::Parse(format!(
                "canonical representation references unknown object `{obj_id}`"
            )));
        }
    }

    // Deterministic order by HF tensor name.
    planned.sort_by(|a, b| a.hf_name.cmp(&b.hf_name));

    // Partition into shards.
    let shard_groups = partition_shards(&planned, options.max_shard_bytes);
    let num_shards = shard_groups.len();
    let shard_names: Vec<String> = if num_shards == 1 {
        vec!["model.safetensors".to_string()]
    } else {
        (1..=num_shards)
            .map(|i| format!("model-{i:05}-of-{num_shards:05}.safetensors"))
            .collect()
    };

    let stage_path = sibling_stage(output_dir)?;
    std::fs::create_dir_all(&stage_path)?;
    let mut scratch = ScratchDir(Some(stage_path.clone()));

    let mut exported_shards = Vec::with_capacity(num_shards);
    let mut exported_tensors = Vec::with_capacity(planned.len());
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();

    for (shard_idx, group) in shard_groups.iter().enumerate() {
        let filename = &shard_names[shard_idx];
        let shard_path = stage_path.join(filename);
        let (shard_record, tensor_records) = write_and_verify_shard(
            &shard_path,
            filename,
            group,
            &index.model,
            &source_index_sha256,
        )?;
        for rec in &tensor_records {
            // Cross-check against weight_patches.json ledger result_tensor_sha256 if recorded.
            if let Some(expected_hash) =
                ledger_tensor_hashes.get(&(rec.object.clone(), rec.segment_tensor.clone()))
            {
                if &rec.sha256 != expected_hash {
                    return Err(VindexError::Parse(format!(
                        "exported tensor `{}` ({}/{}) SHA-256 {} disagrees with weight-patch ledger {}",
                        rec.name, rec.object, rec.segment_tensor, rec.sha256, expected_hash
                    )));
                }
            }
            weight_map.insert(rec.name.clone(), filename.clone());
        }
        exported_shards.push(shard_record);
        exported_tensors.extend(tensor_records);
    }

    let total_bytes: u64 = exported_shards.iter().map(|s| s.payload_bytes).sum();
    if total_bytes != expected_total_bytes {
        return Err(VindexError::Parse(format!(
            "exported {total_bytes} payload bytes, expected {expected_total_bytes}"
        )));
    }

    // Write model.safetensors.index.json when sharded.
    if num_shards > 1 {
        let shard_index = serde_json::json!({
            "metadata": {
                "total_size": total_bytes,
            },
            "weight_map": weight_map,
        });
        std::fs::write(
            stage_path.join("model.safetensors.index.json"),
            serde_json::to_string_pretty(&shard_index)
                .map_err(|e| VindexError::Parse(format!("serialise shard index: {e}")))?,
        )?;
    }

    // Copy capability files (tokenizer, templates, generation_config) from container,
    // and config.json (+ any missing capability files) from source_checkpoint.
    let mut copied_files = Vec::new();
    let mut copied_set = BTreeSet::new();
    for name in CHECKPOINT_CAPABILITY_FILES {
        let src = container.join(name);
        if src.is_file() {
            std::fs::copy(&src, stage_path.join(name))?;
            copied_files.push(name.to_string());
            copied_set.insert(name.to_string());
        }
    }
    if let Some(ckpt) = options.source_checkpoint {
        let cfg = ckpt.join("config.json");
        std::fs::copy(&cfg, stage_path.join("config.json"))?;
        copied_files.push("config.json".to_string());
        copied_set.insert("config.json".to_string());
        for name in CHECKPOINT_CAPABILITY_FILES {
            if copied_set.contains(name) {
                continue;
            }
            let src = ckpt.join(name);
            if src.is_file() {
                std::fs::copy(&src, stage_path.join(name))?;
                copied_files.push(name.to_string());
                copied_set.insert(name.to_string());
            }
        }
    }

    let report = SafetensorsExportReport {
        schema: EXPORT_MANIFEST_SCHEMA.to_string(),
        model: index.model,
        source_index_sha256,
        weight_patches_sha256,
        total_tensors: exported_tensors.len(),
        total_bytes,
        shards: exported_shards,
        tensors: exported_tensors,
        copied_files,
        verified: true,
    };

    std::fs::write(
        stage_path.join(EXPORT_MANIFEST_JSON),
        serde_json::to_string_pretty(&report)
            .map_err(|e| VindexError::Parse(format!("serialise {EXPORT_MANIFEST_JSON}: {e}")))?,
    )?;

    std::fs::rename(&stage_path, output_dir)?;
    scratch.0 = None;
    Ok(report)
}

fn write_and_verify_shard(
    shard_path: &Path,
    filename: &str,
    tensors: &[&PlannedExportTensor],
    model_name: &str,
    source_index_sha256: &str,
) -> Result<(ExportedShard, Vec<ExportedTensorRecord>), VindexError> {
    let mut header_map = serde_json::Map::new();
    header_map.insert(
        "__metadata__".to_string(),
        serde_json::json!({
            "format": "pt",
            "larql_source_model": model_name,
            "larql_source_index_sha256": source_index_sha256,
        }),
    );

    let mut offset = 0u64;
    let mut planned_offsets: Vec<(u64, u64)> = Vec::with_capacity(tensors.len());
    for t in tensors {
        let start = offset;
        let end = start + t.len;
        planned_offsets.push((start, end));
        header_map.insert(
            t.hf_name.clone(),
            serde_json::json!({
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [start, end],
            }),
        );
        offset = end;
    }
    let payload_bytes = offset;

    let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header_map))
        .map_err(|e| VindexError::Parse(format!("serialise safetensors header: {e}")))?;
    let header_len = header_bytes.len() as u64;

    let out_file = File::create(shard_path)?;
    let mut writer = BufWriter::new(out_file);
    let mut file_hasher = Sha256::new();

    let len_prefix = header_len.to_le_bytes();
    writer.write_all(&len_prefix)?;
    file_hasher.update(len_prefix);
    writer.write_all(&header_bytes)?;
    file_hasher.update(&header_bytes);

    let mut buf = vec![0u8; COPY_CHUNK_BYTES];
    let mut source_hashes: Vec<String> = Vec::with_capacity(tensors.len());
    let mut open_segments: BTreeMap<&Path, File> = BTreeMap::new();

    for t in tensors {
        let seg_file = match open_segments.get_mut(t.segment_path.as_path()) {
            Some(f) => f,
            None => {
                let f = File::open(&t.segment_path)?;
                open_segments.entry(t.segment_path.as_path()).or_insert(f)
            }
        };
        seg_file.seek(SeekFrom::Start(t.payload_offset))?;
        let mut tensor_hasher = Sha256::new();
        let mut remaining = t.len;
        while remaining > 0 {
            let take = remaining.min(buf.len() as u64) as usize;
            seg_file.read_exact(&mut buf[..take])?;
            writer.write_all(&buf[..take])?;
            tensor_hasher.update(&buf[..take]);
            file_hasher.update(&buf[..take]);
            remaining -= take as u64;
        }
        source_hashes.push(hex_bytes(&tensor_hasher.finalize()));
    }
    writer.flush()?;
    drop(writer);

    let written_shard_sha256 = hex_bytes(&file_hasher.finalize());
    let file_bytes = 8 + header_len + payload_bytes;

    // Independent read-back verification: re-open the written .safetensors file, parse its header,
    // and verify every tensor's bytes hash to the exact SHA-256 read from the VINDEX3 segment.
    let readback_file = File::open(shard_path)?;
    let actual_len = readback_file.metadata()?.len();
    if actual_len != file_bytes {
        return Err(VindexError::Parse(format!(
            "safetensors verification failed for `{filename}`: file length {actual_len} != expected {file_bytes}"
        )));
    }
    let mut rb_len_prefix = [0u8; 8];
    read_exact_at(&readback_file, &mut rb_len_prefix, 0)?;
    let rb_header_len = u64::from_le_bytes(rb_len_prefix);
    if rb_header_len != header_len {
        return Err(VindexError::Parse(format!(
            "safetensors verification failed for `{filename}`: header length mismatch"
        )));
    }
    let mut rb_header_bytes = vec![0u8; rb_header_len as usize];
    read_exact_at(&readback_file, &mut rb_header_bytes, 8)?;
    let rb_header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&rb_header_bytes)
            .map_err(|e| VindexError::Parse(format!("verify `{filename}` header: {e}")))?;

    let data_start = 8 + rb_header_len;
    let mut records = Vec::with_capacity(tensors.len());
    for ((t, &(exp_start, exp_end)), src_hash) in
        tensors.iter().zip(&planned_offsets).zip(source_hashes)
    {
        let entry = rb_header.get(&t.hf_name).ok_or_else(|| {
            VindexError::Parse(format!(
                "safetensors verification failed: `{}` missing from `{filename}` header",
                t.hf_name
            ))
        })?;
        let offsets = entry
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                VindexError::Parse(format!(
                    "safetensors verification failed: `{}` missing data_offsets",
                    t.hf_name
                ))
            })?;
        let got_start = offsets.first().and_then(|v| v.as_u64()).unwrap_or(u64::MAX);
        let got_end = offsets.get(1).and_then(|v| v.as_u64()).unwrap_or(u64::MAX);
        if (got_start, got_end) != (exp_start, exp_end) {
            return Err(VindexError::Parse(format!(
                "safetensors verification failed: `{}` offsets [{got_start}, {got_end}] != [{exp_start}, {exp_end}]",
                t.hf_name
            )));
        }

        let mut rb_hasher = Sha256::new();
        let mut pos = data_start + got_start;
        let mut remaining = got_end - got_start;
        while remaining > 0 {
            let take = remaining.min(buf.len() as u64) as usize;
            read_exact_at(&readback_file, &mut buf[..take], pos)?;
            rb_hasher.update(&buf[..take]);
            pos += take as u64;
            remaining -= take as u64;
        }
        let rb_hash = hex_bytes(&rb_hasher.finalize());
        if rb_hash != src_hash {
            return Err(VindexError::Parse(format!(
                "safetensors SHA-256 verification failed for `{}`: wrote {src_hash}, read back {rb_hash}",
                t.hf_name
            )));
        }
        records.push(ExportedTensorRecord {
            name: t.hf_name.clone(),
            object: t.object_id.clone(),
            segment_tensor: t.segment_tensor.clone(),
            dtype: t.dtype.clone(),
            shape: t.shape.clone(),
            bytes: t.len,
            shard: filename.to_string(),
            sha256: rb_hash,
        });
    }

    Ok((
        ExportedShard {
            filename: filename.to_string(),
            tensor_count: tensors.len(),
            payload_bytes,
            file_bytes,
            sha256: written_shard_sha256,
        },
        records,
    ))
}

type LedgerTensorHashes = BTreeMap<(String, String), String>;

fn load_ledger_hashes(
    container: &Path,
    index: &Vindex3Index,
) -> Result<(Option<String>, LedgerTensorHashes), VindexError> {
    let Some(name) = index.weight_patches.as_deref() else {
        return Ok((None, BTreeMap::new()));
    };
    let text = std::fs::read_to_string(container.join(name))?;
    let digest = hex_digest(text.as_bytes());
    let ledger = parse_ledger(&text)
        .map_err(|e| VindexError::Parse(format!("parse {WEIGHT_PATCHES_JSON}: {e}")))?;
    let mut map = BTreeMap::new();
    for target in &ledger.targets {
        if let Some(hash) = &target.result_tensor_sha256 {
            map.insert((target.object.clone(), target.tensor.clone()), hash.clone());
        }
    }
    Ok((Some(digest), map))
}

fn partition_shards<'a>(
    tensors: &'a [PlannedExportTensor],
    max_shard_bytes: u64,
) -> Vec<Vec<&'a PlannedExportTensor>> {
    let mut shards: Vec<Vec<&'a PlannedExportTensor>> = Vec::new();
    let mut current: Vec<&'a PlannedExportTensor> = Vec::new();
    let mut current_bytes = 0u64;
    for t in tensors {
        if !current.is_empty() && current_bytes.saturating_add(t.len) > max_shard_bytes {
            shards.push(current);
            current = Vec::new();
            current_bytes = 0;
        }
        current_bytes = current_bytes.saturating_add(t.len);
        current.push(t);
    }
    if !current.is_empty() {
        shards.push(current);
    }
    shards
}

/// Exact inverse of `encode::common_segment_prefix`.
fn common_segment_prefix(prefixes: &[&str]) -> String {
    let Some(first) = prefixes.first() else {
        return String::new();
    };
    let mut common: Vec<&str> = first.split('.').collect();
    for prefix in &prefixes[1..] {
        let segments: Vec<&str> = prefix.split('.').collect();
        let shared = common
            .iter()
            .zip(&segments)
            .take_while(|(a, b)| a == b)
            .count();
        common.truncate(shared);
    }
    common.join(".")
}

fn reconstruct_hf_name(strip: &str, relative: &str) -> String {
    if strip.is_empty() {
        relative.to_string()
    } else if relative.is_empty() {
        strip.to_string()
    } else {
        format!("{strip}.{relative}")
    }
}

fn sibling_stage(out: &Path) -> Result<PathBuf, VindexError> {
    let name = out
        .file_name()
        .ok_or_else(|| VindexError::Parse(format!("`{}` has no file name", out.display())))?;
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    Ok(parent.join(format!(
        ".{}.st-export-{}-{stamp}",
        name.to_string_lossy(),
        std::process::id()
    )))
}

fn hex_digest(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
