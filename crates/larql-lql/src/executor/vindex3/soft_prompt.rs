//! Native continuous soft-prompt prefix loading (`.pt`, `.npy`, `.safetensors`)
//! and execution state for VINDEX3 sessions (`LOAD SOFT_PROMPT` / `CLEAR SOFT_PROMPT`).

use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// Loaded continuous soft-prompt prefix held in a VINDEX3 session's [`super::PreparedCache`].
#[derive(Debug, Clone)]
pub struct SoftPromptState {
    pub path: String,
    pub n_soft: usize,
    pub hidden: usize,
    pub sha256: String,
    pub mean_l2_norm: f32,
    pub embeds: Vec<Vec<f32>>,
}

/// Structured summary returned by `LOAD SOFT_PROMPT`.
#[derive(Debug, Clone, Serialize)]
pub struct SoftPromptSummary {
    pub path: String,
    pub n_soft: usize,
    pub hidden: usize,
    pub sha256: String,
    pub mean_l2_norm: f32,
}

impl SoftPromptState {
    pub fn summary(&self) -> SoftPromptSummary {
        SoftPromptSummary {
            path: self.path.clone(),
            n_soft: self.n_soft,
            hidden: self.hidden,
            sha256: self.sha256.clone(),
            mean_l2_norm: self.mean_l2_norm,
        }
    }
}

impl Session {
    /// Load a continuous soft-prompt prefix (`.pt`, `.npy`, or `.safetensors`) into the bound
    /// VINDEX3 session. Subsequent `INFER`, `GENERATE`, `DLA`, and `CAPTURE` calls prepend these
    /// continuous vectors directly before token 0 in the model's traversal.
    pub fn load_soft_prompt(&mut self, path: &str) -> Result<SoftPromptSummary, LqlError> {
        let (runtime, prepared_cache) = match &self.backend {
            Backend::Vindex3 {
                runtime,
                prepared_cache,
                ..
            } => (runtime, prepared_cache),
            Backend::None => return Err(LqlError::NoBackend),
            _ => {
                return Err(LqlError::Execution(
                    "LOAD SOFT_PROMPT requires a bound canonical VINDEX3 container".into(),
                ));
            }
        };
        let hidden = runtime
            .plan()
            .embedding
            .as_ref()
            .map(|e| e.table.shape[1])
            .ok_or_else(|| {
                LqlError::Execution("bound component declares no embedding hidden dimension".into())
            })?;
        let state = read_soft_prompt_file(Path::new(path), hidden)?;
        let summary = state.summary();
        prepared_cache.set_soft_prompt(state);
        Ok(summary)
    }

    /// Remove any active soft-prompt prefix from the bound VINDEX3 session (0ms, weights stay hot).
    pub fn clear_soft_prompt(&mut self) -> Result<bool, LqlError> {
        let prepared_cache = match &self.backend {
            Backend::Vindex3 { prepared_cache, .. } => prepared_cache,
            Backend::None => return Err(LqlError::NoBackend),
            _ => {
                return Err(LqlError::Execution(
                    "CLEAR SOFT_PROMPT requires a bound canonical VINDEX3 container".into(),
                ));
            }
        };
        Ok(prepared_cache.clear_soft_prompt())
    }

    pub(crate) fn exec_load_soft_prompt(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let s = self.load_soft_prompt(path)?;
        Ok(vec![
            format!(
                "Loaded soft prompt from {} ([{}, {}], mean L2 norm {:.4}):",
                s.path, s.n_soft, s.hidden, s.mean_l2_norm
            ),
            format!("  sha256: {}", s.sha256),
            "  active: INFER, GENERATE, DLA, and CAPTURE now prepend this prefix before token 0."
                .into(),
        ])
    }

    pub(crate) fn exec_clear_soft_prompt(&mut self) -> Result<Vec<String>, LqlError> {
        let had = self.clear_soft_prompt()?;
        if had {
            Ok(vec![
                "Cleared soft prompt prefix (restored unprefixed prompt execution).".into(),
            ])
        } else {
            Ok(vec!["No soft prompt prefix was active.".into()])
        }
    }
}

fn read_soft_prompt_file(path: &Path, expected_hidden: usize) -> Result<SoftPromptState, LqlError> {
    let bytes = std::fs::read(path)
        .map_err(|e| LqlError::exec(format!("read soft prompt {}", path.display()), e))?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let flat: Vec<f32> = if bytes.starts_with(b"\x93NUMPY") || ext == "npy" {
        parse_npy_soft_prompt(&bytes, expected_hidden)?
    } else if bytes.starts_with(b"PK\x03\x04") || matches!(ext.as_str(), "pt" | "pth" | "bin") {
        parse_pytorch_zip_soft_prompt(&bytes, expected_hidden)?
    } else if ext == "safetensors" {
        parse_safetensors_soft_prompt(&bytes, expected_hidden)?
    } else {
        return Err(LqlError::Execution(format!(
            "unsupported soft-prompt format `{}` (expected .pt, .npy, or .safetensors)",
            path.display()
        )));
    };

    if flat.is_empty() || !flat.len().is_multiple_of(expected_hidden) {
        return Err(LqlError::Execution(format!(
            "soft prompt element count {} is empty or not divisible by model hidden_size {expected_hidden}",
            flat.len()
        )));
    }
    if let Some((i, &v)) = flat.iter().enumerate().find(|(_, v)| !v.is_finite()) {
        return Err(LqlError::Execution(format!(
            "soft prompt contains non-finite value {v} at flat index {i}"
        )));
    }

    let n_soft = flat.len() / expected_hidden;
    let mut embeds = Vec::with_capacity(n_soft);
    let mut norm_sum = 0.0f64;
    let mut hasher = Sha256::new();
    for chunk in flat.chunks_exact(expected_hidden) {
        let mut sq = 0.0f64;
        for &v in chunk {
            sq += (v as f64) * (v as f64);
            hasher.update(v.to_le_bytes());
        }
        norm_sum += sq.sqrt();
        embeds.push(chunk.to_vec());
    }
    let sha256 = format!("{:x}", hasher.finalize());
    let mean_l2_norm = (norm_sum / n_soft as f64) as f32;

    Ok(SoftPromptState {
        path: path.display().to_string(),
        n_soft,
        hidden: expected_hidden,
        sha256,
        mean_l2_norm,
        embeds,
    })
}

fn validate_soft_shape(shape: &[usize], expected_hidden: usize) -> Result<usize, LqlError> {
    let (n_soft, hidden) = match shape {
        [n, h] => (*n, *h),
        [1, n, h] => (*n, *h),
        other => {
            return Err(LqlError::Execution(format!(
                "soft prompt shape {other:?} must be [n_soft, {expected_hidden}] or [1, n_soft, {expected_hidden}]"
            )));
        }
    };
    if n_soft == 0 || hidden != expected_hidden {
        return Err(LqlError::Execution(format!(
            "soft prompt shape {shape:?} does not match hidden_size {expected_hidden} (or n_soft == 0)"
        )));
    }
    Ok(n_soft)
}

fn decode_f32_le(raw: &[u8]) -> Result<Vec<f32>, LqlError> {
    if !raw.len().is_multiple_of(4) {
        return Err(LqlError::Execution(format!(
            "F32 buffer length {} is not a multiple of 4",
            raw.len()
        )));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn decode_f64_le(raw: &[u8]) -> Result<Vec<f32>, LqlError> {
    if !raw.len().is_multiple_of(8) {
        return Err(LqlError::Execution(format!(
            "F64 buffer length {} is not a multiple of 8",
            raw.len()
        )));
    }
    Ok(raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
        .collect())
}

fn decode_bf16_le(raw: &[u8]) -> Result<Vec<f32>, LqlError> {
    if !raw.len().is_multiple_of(2) {
        return Err(LqlError::Execution(format!(
            "BF16 buffer length {} is not a multiple of 2",
            raw.len()
        )));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect())
}

fn decode_f16_le(raw: &[u8]) -> Result<Vec<f32>, LqlError> {
    if !raw.len().is_multiple_of(2) {
        return Err(LqlError::Execution(format!(
            "F16 buffer length {} is not a multiple of 2",
            raw.len()
        )));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|c| larql_models::quant::half::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect())
}

fn parse_npy_soft_prompt(bytes: &[u8], expected_hidden: usize) -> Result<Vec<f32>, LqlError> {
    if bytes.len() < 10 || !bytes.starts_with(b"\x93NUMPY") {
        return Err(LqlError::Execution(
            "invalid .npy file (missing NUMPY magic)".into(),
        ));
    }
    let major = bytes[6];
    let (hdr_len, hdr_start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize),
        2 | 3 => {
            if bytes.len() < 12 {
                return Err(LqlError::Execution("truncated .npy v2 header".into()));
            }
            (
                u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
                12usize,
            )
        }
        other => {
            return Err(LqlError::Execution(format!(
                "unsupported .npy format version {other}"
            )));
        }
    };
    let data_start = hdr_start + hdr_len;
    if bytes.len() < data_start {
        return Err(LqlError::Execution("truncated .npy header body".into()));
    }
    let hdr = std::str::from_utf8(&bytes[hdr_start..data_start])
        .map_err(|_| LqlError::Execution("non-UTF-8 .npy header".into()))?;
    if hdr.contains("fortran_order': True") || hdr.contains("fortran_order\": True") {
        return Err(LqlError::Execution(
            "soft prompt .npy must be C-order (fortran_order: False)".into(),
        ));
    }

    let descr = extract_py_quoted_after(hdr, "'descr'")
        .or_else(|| extract_py_quoted_after(hdr, "\"descr\""))
        .ok_or_else(|| LqlError::Execution("missing 'descr' in .npy header".into()))?;
    let shape = extract_npy_shape(hdr)?;
    let n_soft = validate_soft_shape(&shape, expected_hidden)?;
    let raw = &bytes[data_start..];
    let vals = match descr.as_str() {
        "<f4" | "f4" => decode_f32_le(raw)?,
        "<f8" | "f8" => decode_f64_le(raw)?,
        "<f2" | "f2" => decode_f16_le(raw)?,
        "<u2" | "u2" => decode_bf16_le(raw)?,
        other => {
            return Err(LqlError::Execution(format!(
                "unsupported .npy soft-prompt dtype `{other}` (expected <f4, <f8, <f2, or <u2 BF16)"
            )));
        }
    };
    if vals.len() != n_soft * expected_hidden {
        return Err(LqlError::Execution(format!(
            ".npy data has {} elements, expected {}",
            vals.len(),
            n_soft * expected_hidden
        )));
    }
    Ok(vals)
}

fn extract_py_quoted_after(hdr: &str, key: &str) -> Option<String> {
    let idx = hdr.find(key)?;
    let rest = &hdr[idx + key.len()..];
    let colon = rest.find(':')?;
    let after = rest[colon + 1..].trim_start();
    let quote = after.chars().next()?;
    if quote != '\'' && quote != '"' {
        return None;
    }
    let inner = &after[1..];
    let end = inner.find(quote)?;
    Some(inner[..end].to_string())
}

fn extract_npy_shape(hdr: &str) -> Result<Vec<usize>, LqlError> {
    let idx = hdr
        .find("'shape'")
        .or_else(|| hdr.find("\"shape\""))
        .ok_or_else(|| LqlError::Execution("missing 'shape' in .npy header".into()))?;
    let rest = &hdr[idx..];
    let open = rest
        .find('(')
        .ok_or_else(|| LqlError::Execution("malformed 'shape' tuple in .npy header".into()))?;
    let close = rest[open..]
        .find(')')
        .ok_or_else(|| LqlError::Execution("unterminated 'shape' tuple in .npy header".into()))?;
    let inner = &rest[open + 1..open + close];
    let mut dims = Vec::new();
    for part in inner.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let d: usize = p
            .parse()
            .map_err(|_| LqlError::Execution(format!("invalid dimension `{p}` in .npy shape")))?;
        dims.push(d);
    }
    Ok(dims)
}

fn parse_safetensors_soft_prompt(
    bytes: &[u8],
    expected_hidden: usize,
) -> Result<Vec<f32>, LqlError> {
    if bytes.len() < 8 {
        return Err(LqlError::Execution("truncated safetensors file".into()));
    }
    let hdr_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let data_base = 8 + hdr_len;
    if bytes.len() < data_base {
        return Err(LqlError::Execution("truncated safetensors header".into()));
    }
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..data_base])
            .map_err(|e| LqlError::exec("parse safetensors header", e))?;

    let mut candidates: Vec<(&str, &serde_json::Value)> = header
        .iter()
        .filter(|(k, _)| k.as_str() != "__metadata__")
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    if candidates.is_empty() {
        return Err(LqlError::Execution(
            "safetensors file contains no tensors".into(),
        ));
    }
    let (_, entry) = candidates
        .iter()
        .find(|(k, _)| matches!(*k, "soft_emb" | "soft_prompt" | "prompt_embeddings"))
        .copied()
        .or_else(|| (candidates.len() == 1).then(|| candidates.remove(0)))
        .ok_or_else(|| {
            LqlError::Execution(
                "safetensors file holds multiple tensors and none is named soft_emb/soft_prompt"
                    .into(),
            )
        })?;

    let dtype = entry["dtype"]
        .as_str()
        .ok_or_else(|| LqlError::Execution("missing dtype in safetensors tensor".into()))?;
    let shape: Vec<usize> = entry["shape"]
        .as_array()
        .ok_or_else(|| LqlError::Execution("missing shape in safetensors tensor".into()))?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();
    let offsets = entry["data_offsets"]
        .as_array()
        .ok_or_else(|| LqlError::Execution("missing data_offsets in safetensors tensor".into()))?;
    let start = offsets.first().and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let end = offsets.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    if end < start || data_base + end > bytes.len() {
        return Err(LqlError::Execution(
            "safetensors tensor data_offsets out of bounds".into(),
        ));
    }
    let n_soft = validate_soft_shape(&shape, expected_hidden)?;
    let raw = &bytes[data_base + start..data_base + end];
    let vals = match dtype {
        "F32" => decode_f32_le(raw)?,
        "BF16" => decode_bf16_le(raw)?,
        "F16" => decode_f16_le(raw)?,
        "F64" => decode_f64_le(raw)?,
        other => {
            return Err(LqlError::Execution(format!(
                "unsupported safetensors soft-prompt dtype `{other}`"
            )));
        }
    };
    if vals.len() != n_soft * expected_hidden {
        return Err(LqlError::Execution(format!(
            "safetensors tensor has {} elements, expected {}",
            vals.len(),
            n_soft * expected_hidden
        )));
    }
    Ok(vals)
}

/// Pure-Rust reader for PyTorch `.pt` ZIP archives (uncompressed STORED entries, the default
/// emitted by `torch.save` for soft-prompt checkpoints like `sp16_mf103_step160.pt`).
fn parse_pytorch_zip_soft_prompt(
    bytes: &[u8],
    expected_hidden: usize,
) -> Result<Vec<f32>, LqlError> {
    let entries = read_stored_zip_entries(bytes)?;
    let pkl = entries
        .iter()
        .find(|(name, _)| name.ends_with("/data.pkl") || name == "data.pkl")
        .map(|(_, data)| *data)
        .ok_or_else(|| LqlError::Execution("PyTorch .pt archive missing data.pkl entry".into()))?;

    // Find the primary tensor storage payload (`*/data/0` or sole `*/data/*` entry).
    let storage_entries: Vec<(&str, &[u8])> = entries
        .iter()
        .filter(|(name, _)| name.contains("/data/") || name.starts_with("data/"))
        .map(|(n, d)| (n.as_str(), *d))
        .collect();
    let (_, raw) = storage_entries
        .iter()
        .find(|(n, _)| n.ends_with("/data/0") || *n == "data/0")
        .copied()
        .or_else(|| (storage_entries.len() == 1).then(|| storage_entries[0]))
        .ok_or_else(|| {
            LqlError::Execution(
                "PyTorch .pt archive must contain a single tensor storage (data/0) for soft_emb"
                    .into(),
            )
        })?;

    // Detect storage dtype directly from PyTorch's pickle global class name.
    let vals = if bytes_contains(pkl, b"BFloat16Storage") {
        decode_bf16_le(raw)?
    } else if bytes_contains(pkl, b"HalfStorage") {
        decode_f16_le(raw)?
    } else if bytes_contains(pkl, b"DoubleStorage") {
        decode_f64_le(raw)?
    } else if bytes_contains(pkl, b"FloatStorage") {
        decode_f32_le(raw)?
    } else {
        return Err(LqlError::Execution(
            "could not identify PyTorch tensor storage dtype in data.pkl \
             (expected FloatStorage, BFloat16Storage, HalfStorage, or DoubleStorage)"
                .into(),
        ));
    };

    if vals.is_empty() || !vals.len().is_multiple_of(expected_hidden) {
        return Err(LqlError::Execution(format!(
            "PyTorch .pt storage has {} elements, which is not a positive multiple of hidden_size {expected_hidden}",
            vals.len()
        )));
    }
    Ok(vals)
}

fn bytes_contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Read all STORED (uncompressed, method 0) entries from a ZIP archive via its Central Directory.
fn read_stored_zip_entries(bytes: &[u8]) -> Result<Vec<(String, &[u8])>, LqlError> {
    if bytes.len() < 22 {
        return Err(LqlError::Execution("truncated ZIP archive".into()));
    }
    // Locate End of Central Directory record (PK\x05\x06) near the tail.
    let search_start = bytes.len().saturating_sub(65_557);
    let eocd_pos = (search_start..=bytes.len() - 22)
        .rev()
        .find(|&i| &bytes[i..i + 4] == b"PK\x05\x06")
        .ok_or_else(|| {
            LqlError::Execution("not a valid PyTorch ZIP archive (missing EOCD signature)".into())
        })?;

    let cd_entries = u16::from_le_bytes([bytes[eocd_pos + 10], bytes[eocd_pos + 11]]) as usize;
    let cd_size = u32::from_le_bytes([
        bytes[eocd_pos + 12],
        bytes[eocd_pos + 13],
        bytes[eocd_pos + 14],
        bytes[eocd_pos + 15],
    ]) as usize;
    let cd_offset = u32::from_le_bytes([
        bytes[eocd_pos + 16],
        bytes[eocd_pos + 17],
        bytes[eocd_pos + 18],
        bytes[eocd_pos + 19],
    ]) as usize;
    if cd_offset + cd_size > bytes.len() {
        return Err(LqlError::Execution(
            "corrupt ZIP central directory bounds".into(),
        ));
    }

    let mut out = Vec::with_capacity(cd_entries);
    let mut pos = cd_offset;
    for _ in 0..cd_entries {
        if pos + 46 > bytes.len() || &bytes[pos..pos + 4] != b"PK\x01\x02" {
            return Err(LqlError::Execution(
                "corrupt ZIP central directory entry".into(),
            ));
        }
        let method = u16::from_le_bytes([bytes[pos + 10], bytes[pos + 11]]);
        let uncomp_size = u32::from_le_bytes([
            bytes[pos + 24],
            bytes[pos + 25],
            bytes[pos + 26],
            bytes[pos + 27],
        ]) as usize;
        let name_len = u16::from_le_bytes([bytes[pos + 28], bytes[pos + 29]]) as usize;
        let extra_len = u16::from_le_bytes([bytes[pos + 30], bytes[pos + 31]]) as usize;
        let comment_len = u16::from_le_bytes([bytes[pos + 32], bytes[pos + 33]]) as usize;
        let local_off = u32::from_le_bytes([
            bytes[pos + 42],
            bytes[pos + 43],
            bytes[pos + 44],
            bytes[pos + 45],
        ]) as usize;

        if pos + 46 + name_len > bytes.len() {
            return Err(LqlError::Execution("truncated ZIP filename".into()));
        }
        let name = std::str::from_utf8(&bytes[pos + 46..pos + 46 + name_len])
            .map_err(|_| LqlError::Execution("non-UTF-8 ZIP entry name".into()))?
            .to_string();

        if local_off + 30 > bytes.len() || &bytes[local_off..local_off + 4] != b"PK\x03\x04" {
            return Err(LqlError::Execution("corrupt ZIP local file header".into()));
        }
        let local_name_len =
            u16::from_le_bytes([bytes[local_off + 26], bytes[local_off + 27]]) as usize;
        let local_extra_len =
            u16::from_le_bytes([bytes[local_off + 28], bytes[local_off + 29]]) as usize;
        let data_start = local_off + 30 + local_name_len + local_extra_len;
        let data_end = data_start + uncomp_size;
        if data_end > bytes.len() {
            return Err(LqlError::Execution("truncated ZIP entry payload".into()));
        }
        if method == 0 {
            out.push((name, &bytes[data_start..data_end]));
        } else if name.ends_with("data.pkl") || name.contains("/data/") || name.starts_with("data/")
        {
            return Err(LqlError::Execution(format!(
                "PyTorch .pt entry `{name}` uses ZIP compression method {method} (only STORED/uncompressed PyTorch archives are supported; convert to .npy or .safetensors)"
            )));
        }

        pos += 46 + name_len + extra_len + comment_len;
    }
    Ok(out)
}
