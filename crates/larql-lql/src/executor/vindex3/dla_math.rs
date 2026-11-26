//! Exact linear-algebra and token-resolution helpers for Direct Logit Attribution (`DLA`).

use std::collections::BTreeSet;

use larql_models::config::NormType;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandSource;
use larql_vindex::format::vindex3::opplan::exec::Plane;
use larql_vindex::format::vindex3::opplan::{LayerAttention, NormOp, OperandRef};
use larql_vindex::VindexError;

use crate::ast::{CaptureLayers, DlaTarget};
use crate::error::LqlError;

pub(super) fn resolve_dla_layers(
    spec: Option<&CaptureLayers>,
    num_layers: usize,
) -> Result<Vec<usize>, LqlError> {
    let raw: Vec<usize> = match spec {
        None => (0..num_layers).collect(),
        Some(CaptureLayers::One(l)) => vec![*l],
        Some(CaptureLayers::Many(ls)) => ls.clone(),
    };
    let mut seen = BTreeSet::new();
    for &l in &raw {
        if l >= num_layers {
            return Err(LqlError::Execution(format!(
                "DLA: layer {l} is outside the program's 0..{num_layers} layers"
            )));
        }
        if !seen.insert(l) {
            return Err(LqlError::Execution(format!(
                "DLA: layer {l} is listed more than once"
            )));
        }
    }
    Ok(raw)
}

pub(super) fn resolve_dla_target(
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    target: &DlaTarget,
    clause: &str,
    vocab_size: usize,
) -> Result<(u32, String), LqlError> {
    match target {
        DlaTarget::Id(id) => {
            if (*id as usize) >= vocab_size {
                return Err(LqlError::Execution(format!(
                    "DLA {clause} ID {id} is outside vocabulary size {vocab_size}"
                )));
            }
            let label = tokenizer
                .decode(&[*id], false)
                .unwrap_or_else(|_| format!("<id:{id}>"));
            Ok((*id, label))
        }
        DlaTarget::Text(text) => {
            let encoding = tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| LqlError::exec("DLA tokenize target", e))?;
            let ids = encoding.get_ids();
            if ids.len() != 1 {
                let pieces: Vec<String> = ids
                    .iter()
                    .map(|&id| {
                        format!(
                            "{:?} [id {id}]",
                            tokenizer.decode(&[id], false).unwrap_or_default()
                        )
                    })
                    .collect();
                let hint = ids
                    .first()
                    .map(|first| format!("; use `{clause} ID {first}` to target its first token"))
                    .unwrap_or_default();
                return Err(LqlError::Execution(format!(
                    "DLA {clause} {text:?} must encode to exactly 1 token, got {} ({}){hint}",
                    ids.len(),
                    pieces.join(", ")
                )));
            }
            let id = ids[0];
            if (id as usize) >= vocab_size {
                return Err(LqlError::Execution(format!(
                    "DLA {clause} {text:?} resolved to token id {id}, outside vocabulary size {vocab_size}"
                )));
            }
            Ok((id, text.clone()))
        }
    }
}

pub(super) fn plane_rows<'a>(plane: &'a Plane, label: &str) -> Result<&'a [Vec<f32>], VindexError> {
    plane.try_rows().map_err(|e| {
        VindexError::Parse(format!(
            "DLA: {label} has no single [hidden] row per position: {e}"
        ))
    })
}

pub(super) fn mixer_label(attn: &LayerAttention) -> &'static str {
    match attn {
        LayerAttention::Softmax(op) if op.window.is_some() => "sliding",
        LayerAttention::Softmax(_) => "full",
        LayerAttention::ShortConv(_) => "short-conv",
        LayerAttention::GatedDelta(_) => "gated-delta",
        LayerAttention::Kda(_) => "kda",
        LayerAttention::Mamba2(_) => "mamba2",
        LayerAttention::Mla(_) => "mla",
        LayerAttention::ConvQkv(_) => "conv-qkv",
    }
}

pub(super) fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum()
}

pub(super) fn dot_diff_f64(u: &[f32], after: &[f32], before: &[f32]) -> f64 {
    u.iter()
        .zip(after.iter().zip(before.iter()))
        .map(|(&ui, (&a, &b))| (ui as f64) * ((a as f64) - (b as f64)))
        .sum()
}

/// Pull a linear direction `u` back through `norm` at operating point `x`, so that
/// `dot(u_eff, x) == dot(u, norm(x))` holds exactly for any additive decomposition of `x`.
pub(super) fn linearize_norm(
    source: &OperandSource<'_>,
    norm: &NormOp,
    x: &[f32],
    u: &[f32],
) -> Result<Vec<f32>, VindexError> {
    let w = source.load(&norm.weight)?;
    let n = x.len();
    if w.len() != n || u.len() != n {
        return Err(VindexError::Parse(format!(
            "norm `{}` width mismatch: x={}, w={}, u={}",
            norm.weight.tensor,
            n,
            w.len(),
            u.len()
        )));
    }
    let inv_n = 1.0f64 / (n as f64);
    match norm.kind {
        NormType::RmsNorm => {
            let mean_sq: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() * inv_n;
            let inv_rms = 1.0f64 / (mean_sq + norm.eps).sqrt();
            let offset = norm.weight_offset as f64;
            Ok(u.iter()
                .zip(w.iter())
                .map(|(&ui, &wi)| ((ui as f64) * ((wi as f64) + offset) * inv_rms) as f32)
                .collect())
        }
        NormType::LayerNorm => {
            let mean: f64 = x.iter().map(|&v| v as f64).sum::<f64>() * inv_n;
            let var: f64 = x
                .iter()
                .map(|&v| {
                    let d = (v as f64) - mean;
                    d * d
                })
                .sum::<f64>()
                * inv_n;
            let inv_std = 1.0f64 / (var + norm.eps).sqrt();
            let offset = norm.weight_offset as f64;
            let uw: Vec<f64> = u
                .iter()
                .zip(w.iter())
                .map(|(&ui, &wi)| (ui as f64) * ((wi as f64) + offset) * inv_std)
                .collect();
            let mean_uw: f64 = uw.iter().sum::<f64>() * inv_n;
            Ok(uw.into_iter().map(|v| (v - mean_uw) as f32).collect())
        }
    }
}

/// Read exactly two rows (`row_a`, `row_b`) of a `[vocab, hidden]` matrix. Uses `load_raw_range`
/// on unpatched float tensors (4 KB instead of 250+ MB) and falls back to `source.load` when
/// patched, overridden, or stored in a packed quantised codec.
pub(super) fn load_two_rows(
    source: &OperandSource<'_>,
    operand: &OperandRef,
    row_a: usize,
    row_b: usize,
    vocab: usize,
    hidden: usize,
) -> Result<(Vec<f32>, Vec<f32>), VindexError> {
    if !source.is_overridden(operand) && !source.is_patched(operand) {
        if let Some(dtype) = source.stored_dtype(operand) {
            let elem_bytes: Option<usize> = match dtype {
                "BF16" | "F16" => Some(2),
                "F32" => Some(4),
                _ => None,
            };
            if let Some(eb) = elem_bytes {
                let row_bytes = (hidden * eb) as u64;
                let total_bytes = (vocab as u64) * row_bytes;
                let store = source.store();
                let raw_a = store.load_raw_range(
                    operand,
                    total_bytes,
                    (row_a as u64) * row_bytes,
                    row_bytes,
                )?;
                let raw_b = store.load_raw_range(
                    operand,
                    total_bytes,
                    (row_b as u64) * row_bytes,
                    row_bytes,
                )?;
                return Ok((
                    widen_float_row(&raw_a.dtype, &raw_a.bytes)?,
                    widen_float_row(&raw_b.dtype, &raw_b.bytes)?,
                ));
            }
        }
    }
    let full = source.load(operand)?;
    let start_a = row_a * hidden;
    let start_b = row_b * hidden;
    Ok((
        full[start_a..start_a + hidden].to_vec(),
        full[start_b..start_b + hidden].to_vec(),
    ))
}

fn widen_float_row(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, VindexError> {
    match dtype {
        "F32" => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "BF16" => Ok(bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
            .collect()),
        "F16" => Ok(larql_models::quant::half::decode_f16(bytes)),
        other => Err(VindexError::Parse(format!(
            "unsupported float row dtype `{other}`"
        ))),
    }
}
