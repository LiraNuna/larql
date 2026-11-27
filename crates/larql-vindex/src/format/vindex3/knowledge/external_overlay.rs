//! Fast in-memory delta safetensors & PEFT LoRA overlay loading over a bound read-only VINDEX3 container.
//!
//! Instead of re-encoding a full multi-GB checkpoint into a fresh `.vindex3` on every training
//! checkpoint or adapter, [`load_safetensors_overlay`] scans a candidate `.safetensors` file or
//! HF directory, skips every tensor that is byte-identical to the bound container, and installs
//! `OperandOverrides::replace_bf16` ONLY for modified tensors. [`load_lora_overlay`] merges a
//! PEFT `adapter_config.json` + `adapter_model.safetensors` adapter directly in RAM onto the
//! bound container's BF16 weights with parallel row-wise `B @ A` accumulation.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::live_weights::{LiveTensor, LiveWeightView};
use super::KnowledgeOverlay;
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::encode::segment::{read_exact_at, read_segment_header};
use crate::format::vindex3::encode::SYSTEM_GRAPH_JSON;
use crate::format::vindex3::graph::SystemGraph;
use crate::format::vindex3::index::{ContainerAuthority, Vindex3Index};
use crate::format::vindex3::opplan::exec::operands::OperandOverrides;
use crate::format::vindex3::opplan::{ComponentOpPlan, LayerAttention, LayerFfn, OperandRef};

/// Structured summary returned by `OVERLAY SAFETENSORS` / `OVERLAY LORA` (and JSON output).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalOverlaySummary {
    pub kind: String,
    pub source: String,
    pub overlaid_tensors: usize,
    pub total_scanned: usize,
    pub unchanged_skipped: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_scale: Option<f64>,
    pub tensors: Vec<String>,
    pub elapsed_ms: f64,
}

/// Active external weight overlay mounted over a read-only VINDEX3 container.
#[derive(Debug)]
pub struct ExternalWeightOverlay {
    pub kind: &'static str,
    pub source: String,
    pub total_scanned: usize,
    pub unchanged_skipped: usize,
    pub lora_scale: Option<f64>,
    pub elapsed_ms: f64,
    pub(super) view: LiveWeightView,
}

impl ExternalWeightOverlay {
    pub fn summary(&self) -> ExternalOverlaySummary {
        ExternalOverlaySummary {
            kind: self.kind.to_string(),
            source: self.source.clone(),
            overlaid_tensors: self.view.tensors().len(),
            total_scanned: self.total_scanned,
            unchanged_skipped: self.unchanged_skipped,
            lora_scale: self.lora_scale,
            tensors: self
                .view
                .tensors()
                .iter()
                .map(|t| t.tensor.clone())
                .collect(),
            elapsed_ms: self.elapsed_ms,
        }
    }

    pub fn tensors(&self) -> &[LiveTensor] {
        self.view.tensors()
    }
}

impl KnowledgeOverlay {
    /// Mount an external safetensors or LoRA overlay, replacing any previous external overlay.
    pub fn set_external_overlay(&mut self, overlay: ExternalWeightOverlay) {
        self.external = Some(overlay);
        self.invalidate_live_weights();
    }

    /// Drop any mounted external overlay. Returns whether one was active.
    pub fn clear_external_overlay(&mut self) -> bool {
        let had = self.external.take().is_some();
        if had {
            self.invalidate_live_weights();
        }
        had
    }

    /// The currently mounted external overlay, if any.
    pub fn external_overlay(&self) -> Option<&ExternalWeightOverlay> {
        self.external.as_ref()
    }
}

struct BaseTensorLocation {
    segment_path: PathBuf,
    payload_offset: u64,
    len: u64,
    dtype: String,
}

struct PlanTensorCatalog {
    /// Maps HF tensor name (and bare segment tensor name) -> `(object_id, segment_tensor, shape, Vec<OperandRef>)`.
    by_name: BTreeMap<String, (String, String, Vec<usize>, Vec<OperandRef>)>,
    /// Canonical segment byte location for every `(object_id, segment_tensor)`.
    locations: BTreeMap<(String, String), BaseTensorLocation>,
}

fn build_catalog(
    container: &Path,
    plan: &ComponentOpPlan,
) -> Result<PlanTensorCatalog, VindexError> {
    let raw_index = std::fs::read_to_string(container.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&raw_index)
        .map_err(|e| VindexError::Parse(format!("parse {INDEX_JSON}: {e}")))?;
    if index.authority != ContainerAuthority::Canonical {
        return Err(VindexError::Parse(format!(
            "external weight overlay requires a canonical VINDEX3 container, found {:?}",
            index.authority
        )));
    }
    let graph_name = index.system_graph.as_deref().unwrap_or(SYSTEM_GRAPH_JSON);
    let graph_raw = std::fs::read_to_string(container.join(graph_name))?;
    let graph: SystemGraph = serde_json::from_str(&graph_raw)
        .map_err(|e| VindexError::Parse(format!("parse {graph_name}: {e}")))?;

    let mut strip_by_object: BTreeMap<String, String> = BTreeMap::new();
    for obj in &graph.objects {
        let prefixes: Vec<&str> = obj
            .source_bindings
            .iter()
            .map(|b| b.tensor_prefix.as_str())
            .collect();
        strip_by_object.insert(obj.id.clone(), common_segment_prefix(&prefixes));
    }

    let mut locations: BTreeMap<(String, String), BaseTensorLocation> = BTreeMap::new();
    for entry in index
        .representations
        .values()
        .filter(|e| e.compiled_from.is_none())
    {
        let seg_path = container.join(&entry.segment);
        if !seg_path.is_file() {
            continue;
        }
        let (header, payload_start) = read_segment_header(&seg_path)?;
        for t in header.tensors {
            locations.insert(
                (entry.object.clone(), t.name),
                BaseTensorLocation {
                    segment_path: seg_path.clone(),
                    payload_offset: payload_start + t.offset,
                    len: t.len,
                    dtype: t.dtype,
                },
            );
        }
    }

    // Collect every OperandRef the executable plan reads (matrices, norms, conv kernels, biases).
    let mut grouped: BTreeMap<(String, String), (Vec<usize>, Vec<OperandRef>)> = BTreeMap::new();
    let mut record = |op: &OperandRef| {
        let entry = grouped
            .entry((op.object.clone(), op.tensor.clone()))
            .or_insert_with(|| (op.shape.clone(), Vec::new()));
        if !entry.1.contains(op) {
            entry.1.push(op.clone());
        }
    };
    for planned in plan.planned_operands() {
        record(&planned.operand);
    }
    if let Some(norm) = &plan.final_norm {
        record(&norm.weight);
    }
    for layer in &plan.layers {
        for norm in [
            &layer.pre_attention_norm,
            &layer.post_attention_norm,
            &layer.pre_ffn_norm,
            &layer.post_ffn_norm,
        ]
        .into_iter()
        .flatten()
        {
            record(&norm.weight);
        }
        if let Some(s) = &layer.layer_scale {
            record(s);
        }
        match &layer.attention {
            LayerAttention::Softmax(attn) => {
                if let Some(qk) = &attn.qk_norm {
                    record(&qk.q);
                    record(&qk.k);
                }
                for b in [&attn.q_bias, &attn.k_bias, &attn.v_bias, &attn.o_bias]
                    .into_iter()
                    .flatten()
                {
                    record(b);
                }
            }
            LayerAttention::ShortConv(sc) => {
                record(&sc.conv_weight);
                for b in [&sc.in_proj_bias, &sc.conv_bias, &sc.out_proj_bias]
                    .into_iter()
                    .flatten()
                {
                    record(b);
                }
            }
            _ => {}
        }
        if let Some(LayerFfn::Dense(ffn)) = &layer.ffn {
            record(&ffn.down);
            record(&ffn.up);
            if let Some(g) = &ffn.gate {
                record(g);
            }
        }
    }

    let mut by_name: BTreeMap<String, (String, String, Vec<usize>, Vec<OperandRef>)> =
        BTreeMap::new();
    for ((obj_id, seg_tensor), (shape, refs)) in grouped {
        let strip = strip_by_object
            .get(&obj_id)
            .map(String::as_str)
            .unwrap_or("");
        let hf_name = reconstruct_hf_name(strip, &seg_tensor);
        by_name.insert(
            hf_name,
            (
                obj_id.clone(),
                seg_tensor.clone(),
                shape.clone(),
                refs.clone(),
            ),
        );
        by_name
            .entry(seg_tensor.clone())
            .or_insert((obj_id, seg_tensor, shape, refs));
    }

    Ok(PlanTensorCatalog { by_name, locations })
}

struct StTensorMeta {
    dtype: String,
    shape: Vec<usize>,
    start: u64,
    end: u64,
}

fn read_safetensors_header(
    path: &Path,
) -> Result<(u64, BTreeMap<String, StTensorMeta>), VindexError> {
    let file = File::open(path)?;
    let mut len_buf = [0u8; 8];
    read_exact_at(&file, &mut len_buf, 0)?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len > 256 * 1024 * 1024 {
        return Err(VindexError::Parse(format!(
            "safetensors `{}` header length {header_len} is implausibly large",
            path.display()
        )));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    read_exact_at(&file, &mut header_bytes, 8)?;
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&header_bytes)
        .map_err(|e| {
            VindexError::Parse(format!(
                "parse safetensors header `{}`: {e}",
                path.display()
            ))
        })?;
    let mut out = BTreeMap::new();
    for (name, val) in map {
        if name == "__metadata__" {
            continue;
        }
        let dtype = val
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| VindexError::Parse(format!("safetensors `{name}` missing dtype")))?
            .to_string();
        let shape: Vec<usize> = val
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| VindexError::Parse(format!("safetensors `{name}` missing shape")))?
            .iter()
            .map(|v| {
                v.as_u64().map(|n| n as usize).ok_or_else(|| {
                    VindexError::Parse(format!("safetensors `{name}` invalid shape"))
                })
            })
            .collect::<Result<_, _>>()?;
        let offsets = val
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                VindexError::Parse(format!("safetensors `{name}` missing data_offsets"))
            })?;
        let start = offsets.first().and_then(|v| v.as_u64()).ok_or_else(|| {
            VindexError::Parse(format!("safetensors `{name}` invalid start offset"))
        })?;
        let end = offsets.get(1).and_then(|v| v.as_u64()).ok_or_else(|| {
            VindexError::Parse(format!("safetensors `{name}` invalid end offset"))
        })?;
        if end < start {
            return Err(VindexError::Parse(format!(
                "safetensors `{name}` has inverted offsets [{start}, {end}]"
            )));
        }
        out.insert(
            name,
            StTensorMeta {
                dtype,
                shape,
                start,
                end,
            },
        );
    }
    Ok((8 + header_len, out))
}

fn collect_safetensors_files(path: &Path) -> Result<Vec<PathBuf>, VindexError> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    if !path.is_dir() {
        return Err(VindexError::Parse(format!(
            "safetensors overlay path `{}` does not exist",
            path.display()
        )));
    }
    let index_path = path.join("model.safetensors.index.json");
    if index_path.is_file() {
        let raw = std::fs::read_to_string(&index_path)?;
        let val: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| VindexError::Parse(format!("parse `{}`: {e}", index_path.display())))?;
        let weight_map = val
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                VindexError::Parse("model.safetensors.index.json missing weight_map".into())
            })?;
        let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for v in weight_map.values() {
            if let Some(s) = v.as_str() {
                names.insert(s.to_string());
            }
        }
        return Ok(names.into_iter().map(|n| path.join(n)).collect());
    }
    let single = path.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(VindexError::Parse(format!(
            "directory `{}` contains no .safetensors files",
            path.display()
        )));
    }
    Ok(files)
}

fn raw_to_bf16_bytes(dtype: &str, raw: &[u8], name: &str) -> Result<Vec<u8>, VindexError> {
    match dtype {
        "BF16" => Ok(raw.to_vec()),
        "F32" => {
            if !raw.len().is_multiple_of(4) {
                return Err(VindexError::Parse(format!(
                    "`{name}` F32 byte length invalid"
                )));
            }
            let vals: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(larql_models::quant::half::encode_bf16(&vals))
        }
        "F16" => {
            let vals = larql_models::quant::half::decode_f16(raw);
            Ok(larql_models::quant::half::encode_bf16(&vals))
        }
        other => Err(VindexError::Parse(format!(
            "tensor `{name}` has unsupported overlay dtype `{other}` (expected BF16, F16, or F32)"
        ))),
    }
}

fn raw_to_f32_vec(dtype: &str, raw: &[u8], name: &str) -> Result<Vec<f32>, VindexError> {
    match dtype {
        "F32" => Ok(raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" => Ok(raw
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()),
        "F16" => Ok(larql_models::quant::half::decode_f16(raw)),
        other => Err(VindexError::Parse(format!(
            "LoRA tensor `{name}` has unsupported dtype `{other}`"
        ))),
    }
}

/// Mount a delta or full Hugging Face `.safetensors` file/directory over a bound canonical
/// VINDEX3 container. Tensors byte-identical to the container's canonical bytes are skipped.
pub fn load_safetensors_overlay(
    container: &Path,
    plan: &ComponentOpPlan,
    source_path: &Path,
) -> Result<ExternalWeightOverlay, VindexError> {
    let start = std::time::Instant::now();
    let catalog = build_catalog(container, plan)?;
    let files = collect_safetensors_files(source_path)?;

    let mut open_segs: BTreeMap<PathBuf, File> = BTreeMap::new();
    let mut tensors: Vec<LiveTensor> = Vec::new();
    let mut overrides = OperandOverrides::new();
    let mut seen_keys: std::collections::BTreeSet<(String, String)> =
        std::collections::BTreeSet::new();
    let mut total_scanned = 0usize;
    let mut unchanged_skipped = 0usize;

    for shard_path in &files {
        let (data_start, entries) = read_safetensors_header(shard_path)?;
        let shard_file = File::open(shard_path)?;
        for (name, meta) in entries {
            total_scanned += 1;
            let Some((obj_id, seg_tensor, expected_shape, op_refs)) = catalog.by_name.get(&name)
            else {
                continue;
            };
            if meta.shape != *expected_shape {
                return Err(VindexError::Parse(format!(
                    "overlay tensor `{name}` has shape {:?}, expected {:?}",
                    meta.shape, expected_shape
                )));
            }
            let key = (obj_id.clone(), seg_tensor.clone());
            if !seen_keys.insert(key.clone()) {
                continue;
            }
            let byte_len = (meta.end - meta.start) as usize;
            let mut raw = vec![0u8; byte_len];
            read_exact_at(&shard_file, &mut raw, data_start + meta.start)?;
            let bf16_bytes = raw_to_bf16_bytes(&meta.dtype, &raw, &name)?;

            // Compare directly against the container's canonical BF16 segment bytes: skip if identical!
            if let Some(loc) = catalog.locations.get(&key) {
                if loc.dtype == "BF16" && loc.len as usize == bf16_bytes.len() {
                    let seg_file = match open_segs.get(&loc.segment_path) {
                        Some(f) => f,
                        None => {
                            let f = File::open(&loc.segment_path)?;
                            open_segs.entry(loc.segment_path.clone()).or_insert(f)
                        }
                    };
                    let mut base_bytes = vec![0u8; bf16_bytes.len()];
                    read_exact_at(seg_file, &mut base_bytes, loc.payload_offset)?;
                    if base_bytes == bf16_bytes {
                        unchanged_skipped += 1;
                        continue;
                    }
                }
            }

            let sha256 = hex_digest(&bf16_bytes);
            let arc_bytes: Arc<[u8]> = Arc::from(bf16_bytes);
            for operand in op_refs {
                overrides.replace_bf16(operand, Arc::clone(&arc_bytes))?;
            }
            tensors.push(LiveTensor {
                object: obj_id.clone(),
                tensor: seg_tensor.clone(),
                shape: expected_shape.clone(),
                sha256,
                bytes: arc_bytes,
            });
        }
    }

    Ok(ExternalWeightOverlay {
        kind: "safetensors",
        source: source_path.display().to_string(),
        total_scanned,
        unchanged_skipped,
        lora_scale: None,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        view: LiveWeightView::from_parts(tensors, overrides),
    })
}

#[derive(Deserialize)]
struct PeftAdapterConfig {
    r: f64,
    lora_alpha: f64,
    #[serde(default)]
    alpha_pattern: BTreeMap<String, f64>,
    #[serde(default)]
    fan_in_fan_out: bool,
}

/// Mount a PEFT LoRA adapter directory (`adapter_config.json` + `adapter_model.safetensors`)
/// in memory over a bound canonical VINDEX3 container, computing `W + scale * (alpha/r) * (B @ A)`
/// directly onto the base BF16 matrices without rewriting a `.vindex3` container.
pub fn load_lora_overlay(
    container: &Path,
    plan: &ComponentOpPlan,
    adapter_path: &Path,
    scale_override: Option<f64>,
) -> Result<ExternalWeightOverlay, VindexError> {
    let start = std::time::Instant::now();
    let (adapter_dir, st_path) = if adapter_path.is_dir() {
        (
            adapter_path.to_path_buf(),
            adapter_path.join("adapter_model.safetensors"),
        )
    } else {
        let dir = adapter_path
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();
        (dir, adapter_path.to_path_buf())
    };
    let cfg_path = adapter_dir.join("adapter_config.json");
    if !cfg_path.is_file() {
        return Err(VindexError::Parse(format!(
            "LoRA overlay requires `{}`",
            cfg_path.display()
        )));
    }
    if !st_path.is_file() {
        return Err(VindexError::Parse(format!(
            "LoRA overlay requires `{}`",
            st_path.display()
        )));
    }
    let cfg: PeftAdapterConfig = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)
        .map_err(|e| VindexError::Parse(format!("parse `{}`: {e}", cfg_path.display())))?;
    if !cfg.r.is_finite() || cfg.r <= 0.0 {
        return Err(VindexError::Parse(format!(
            "adapter_config.json has invalid rank r={}",
            cfg.r
        )));
    }

    let catalog = build_catalog(container, plan)?;
    let (data_start, entries) = read_safetensors_header(&st_path)?;
    let st_file = File::open(&st_path)?;

    let mut pairs: BTreeMap<String, (Option<StTensorMeta>, Option<StTensorMeta>)> = BTreeMap::new();
    let total_scanned = entries.len();
    for (name, meta) in entries {
        if let Some(prefix) = name.strip_suffix(".lora_A.weight") {
            pairs.entry(prefix.to_string()).or_default().0 = Some(meta);
        } else if let Some(prefix) = name.strip_suffix(".lora_B.weight") {
            pairs.entry(prefix.to_string()).or_default().1 = Some(meta);
        }
    }
    if pairs.is_empty() {
        return Err(VindexError::Parse(format!(
            "`{}` contains no *.lora_A.weight / *.lora_B.weight pairs",
            st_path.display()
        )));
    }

    let multiplier = scale_override.unwrap_or(1.0);
    let mut open_segs: BTreeMap<PathBuf, File> = BTreeMap::new();
    let mut tensors: Vec<LiveTensor> = Vec::new();
    let mut overrides = OperandOverrides::new();
    let mut unchanged_skipped = 0usize;

    for (prefix, (a_opt, b_opt)) in pairs {
        let (Some(a_meta), Some(b_meta)) = (a_opt, b_opt) else {
            return Err(VindexError::Parse(format!(
                "LoRA module `{prefix}` is missing either lora_A or lora_B"
            )));
        };
        let stripped = prefix
            .strip_prefix("base_model.model.")
            .or_else(|| prefix.strip_prefix("base_model."))
            .unwrap_or(&prefix);
        let target_name = format!("{stripped}.weight");
        let Some((obj_id, seg_tensor, expected_shape, op_refs)) = catalog.by_name.get(&target_name)
        else {
            return Err(VindexError::Parse(format!(
                "LoRA target `{target_name}` does not match any tensor in the bound container"
            )));
        };
        if expected_shape.len() != 2 {
            return Err(VindexError::Parse(format!(
                "LoRA target `{target_name}` has shape {:?}, expected 2-D matrix",
                expected_shape
            )));
        }
        let (out_dim, in_dim) = (expected_shape[0], expected_shape[1]);
        if a_meta.shape.len() != 2 || b_meta.shape.len() != 2 {
            return Err(VindexError::Parse(format!(
                "LoRA `{prefix}` A/B tensors must be 2-D"
            )));
        }
        let rank = a_meta.shape[0];
        if a_meta.shape[1] != in_dim || b_meta.shape[0] != out_dim || b_meta.shape[1] != rank {
            return Err(VindexError::Parse(format!(
                "LoRA `{prefix}` shapes A={:?} B={:?} do not match target [{out_dim}, {in_dim}]",
                a_meta.shape, b_meta.shape
            )));
        }

        let mut a_raw = vec![0u8; (a_meta.end - a_meta.start) as usize];
        let mut b_raw = vec![0u8; (b_meta.end - b_meta.start) as usize];
        read_exact_at(&st_file, &mut a_raw, data_start + a_meta.start)?;
        read_exact_at(&st_file, &mut b_raw, data_start + b_meta.start)?;
        let a_vals = raw_to_f32_vec(&a_meta.dtype, &a_raw, &format!("{prefix}.lora_A"))?;
        let b_vals = raw_to_f32_vec(&b_meta.dtype, &b_raw, &format!("{prefix}.lora_B"))?;

        let key = (obj_id.clone(), seg_tensor.clone());
        let loc = catalog.locations.get(&key).ok_or_else(|| {
            VindexError::Parse(format!(
                "base tensor `{obj_id}/{seg_tensor}` has no canonical segment bytes"
            ))
        })?;
        let seg_file = match open_segs.get(&loc.segment_path) {
            Some(f) => f,
            None => {
                let f = File::open(&loc.segment_path)?;
                open_segs.entry(loc.segment_path.clone()).or_insert(f)
            }
        };
        let mut base_raw = vec![0u8; loc.len as usize];
        read_exact_at(seg_file, &mut base_raw, loc.payload_offset)?;
        let base_f32 = raw_to_f32_vec(&loc.dtype, &base_raw, &target_name)?;

        let alpha = cfg
            .alpha_pattern
            .iter()
            .find(|(k, _)| stripped.ends_with(k.as_str()))
            .map(|(_, &v)| v)
            .unwrap_or(cfg.lora_alpha);
        let eff_scale = multiplier * (alpha / cfg.r);

        // Compute merged BF16 rows in parallel: W[o, i] = RNE_bf16(W_base[o, i] + scale * (B @ A)[o, i]).
        let merged_f32: Vec<f32> = (0..out_dim)
            .into_par_iter()
            .flat_map_iter(|o| {
                let base_row = &base_f32[o * in_dim..(o + 1) * in_dim];
                let b_row = &b_vals[o * rank..(o + 1) * rank];
                let mut row = Vec::with_capacity(in_dim);
                for (i, &w0) in base_row.iter().enumerate() {
                    let mut acc = 0.0f64;
                    if !cfg.fan_in_fan_out {
                        for (k, &bk) in b_row.iter().enumerate() {
                            acc += (bk as f64) * (a_vals[k * in_dim + i] as f64);
                        }
                    } else {
                        for (k, &bk) in b_row.iter().enumerate() {
                            acc += (bk as f64) * (a_vals[i * rank + k] as f64);
                        }
                    }
                    row.push(((w0 as f64) + eff_scale * acc) as f32);
                }
                row
            })
            .collect();
        let bf16_bytes = larql_models::quant::half::encode_bf16(&merged_f32);

        if loc.dtype == "BF16" && base_raw == bf16_bytes {
            unchanged_skipped += 1;
            continue;
        }
        let sha256 = hex_digest(&bf16_bytes);
        let arc_bytes: Arc<[u8]> = Arc::from(bf16_bytes);
        for operand in op_refs {
            overrides.replace_bf16(operand, Arc::clone(&arc_bytes))?;
        }
        tensors.push(LiveTensor {
            object: obj_id.clone(),
            tensor: seg_tensor.clone(),
            shape: expected_shape.clone(),
            sha256,
            bytes: arc_bytes,
        });
    }

    Ok(ExternalWeightOverlay {
        kind: "lora",
        source: adapter_path.display().to_string(),
        total_scanned,
        unchanged_skipped,
        lora_scale: Some(multiplier),
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        view: LiveWeightView::from_parts(tensors, overrides),
    })
}

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

fn hex_digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
