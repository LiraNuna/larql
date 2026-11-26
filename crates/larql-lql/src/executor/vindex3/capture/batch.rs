//! The sites `INFER`'s own pass exposes: one fresh causal traversal of the whole sequence, over the operands
//! `INFER` resolves, observed.
//!
//! The per-layer sites are the planes the traversal hands its subscriber (`PlaneEvent::Layer`); `final` and
//! `logits` are the traversal's exit at each requested position. At the sequence's last position that exit is
//! the traversal's own output, checked bit for bit by the executor, so a `logits` capture there IS the vector
//! `INFER` ranks for the same tokens.

use std::collections::BTreeMap;

use larql_inference::vindex3::{PlaneEvent, SelectedContinuation};
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::opplan::exec::{
    execute_prepared_position_exits_with_taps, requires_continuation, BatchTapRequest, Plane,
};
use larql_vindex::VindexError;

use super::super::V3Runtime;
use super::Rows;
use crate::ast::CaptureSite;
use crate::error::LqlError;

/// What INFER says of a component without an output head; a capture refuses in the same words.
const NO_HEAD: &str = "the component carries no output head";

/// One traversal of `ids`, keeping `site` at `layers` (or at the exit) and `positions`. One block of rows per
/// layer, in the order given; one block for a site after the last layer.
pub(super) fn capture(
    runtime: &V3Runtime,
    ops: &PreparedOperands,
    continuation: &SelectedContinuation,
    ids: &[u32],
    site: CaptureSite,
    layers: &[usize],
    positions: &[usize],
) -> Result<Vec<Rows>, LqlError> {
    let mut results = capture_many(runtime, ops, continuation, ids, &[site], layers, positions)?;
    Ok(results.remove(&site).unwrap_or_default())
}

/// One traversal of `ids` collecting every site in `sites` simultaneously.
/// Per-layer sites are captured at `layers`; exit sites (`final`, `logits`)
/// are captured at `positions` after the last layer.
pub(super) fn capture_many(
    runtime: &V3Runtime,
    ops: &PreparedOperands,
    continuation: &SelectedContinuation,
    ids: &[u32],
    sites: &[CaptureSite],
    layers: &[usize],
    positions: &[usize],
) -> Result<BTreeMap<CaptureSite, Vec<Rows>>, LqlError> {
    let mut taps = BatchTapRequest::default();
    let mut want_residual = false;
    let mut want_post_attn = false;
    let mut want_ffn_in = false;
    let mut want_exit = false;
    for &site in sites {
        match site {
            CaptureSite::Residual => want_residual = true,
            CaptureSite::PostAttention => want_post_attn = true,
            CaptureSite::FfnInput => want_ffn_in = true,
            CaptureSite::FfnDownInput => taps.ffn_down_input.extend(layers.iter().copied()),
            CaptureSite::FfnDownOutput => taps.ffn_down_output.extend(layers.iter().copied()),
            CaptureSite::AttnHeadOut => taps.attn_head_out.extend(layers.iter().copied()),
            CaptureSite::AttnWeights => taps.attn_weights.extend(layers.iter().copied()),
            CaptureSite::ShortConvGateB
            | CaptureSite::ShortConvGateC
            | CaptureSite::ShortConvConvOut
            | CaptureSite::ShortConvGatedY => taps.short_conv.extend(layers.iter().copied()),
            CaptureSite::Final | CaptureSite::Logits => want_exit = true,
        }
    }

    let mut kept_residual: BTreeMap<usize, Rows> = BTreeMap::new();
    let mut kept_post_attn: BTreeMap<usize, Rows> = BTreeMap::new();
    let mut kept_ffn_in: BTreeMap<usize, Rows> = BTreeMap::new();
    let mut sink = |event: PlaneEvent| {
        if let PlaneEvent::Layer { index, trace } = event {
            if layers.contains(&index) {
                if want_residual {
                    let rows = plane_rows(&trace.post_layer, index)?;
                    kept_residual
                        .insert(index, positions.iter().map(|&p| rows[p].clone()).collect());
                }
                if want_post_attn {
                    let rows = plane_rows(&trace.post_attention, index)?;
                    kept_post_attn
                        .insert(index, positions.iter().map(|&p| rows[p].clone()).collect());
                }
                if want_ffn_in {
                    let rows = &trace.ffn_input[..];
                    kept_ffn_in.insert(index, positions.iter().map(|&p| rows[p].clone()).collect());
                }
            }
        }
        Ok(())
    };
    let exit_positions = if want_exit { positions } else { &[][..] };
    let plan = runtime.plan();
    // A stack that keeps state beyond softmax attention runs over a fresh provider, as INFER's pass does.
    let state = requires_continuation(plan).then(|| continuation.build());
    let (observed, tap_out) = execute_prepared_position_exits_with_taps(
        plan,
        ops,
        ids,
        runtime.backend(),
        state,
        exit_positions,
        &taps,
        &mut sink,
    )
    .map_err(|e| LqlError::exec("capture pass failed", e))?;
    if observed.output.logits.is_none() {
        return Err(LqlError::Execution(NO_HEAD.into()));
    }

    let pick_positions =
        |rows: &[Vec<f32>]| -> Rows { positions.iter().map(|&p| rows[p].clone()).collect() };
    let mut out = BTreeMap::new();
    for &site in sites {
        let blocks = match site {
            CaptureSite::Residual => layers
                .iter()
                .map(|l| kept_residual.get(l).cloned().unwrap_or_default())
                .collect(),
            CaptureSite::PostAttention => layers
                .iter()
                .map(|l| kept_post_attn.get(l).cloned().unwrap_or_default())
                .collect(),
            CaptureSite::FfnInput => layers
                .iter()
                .map(|l| kept_ffn_in.get(l).cloned().unwrap_or_default())
                .collect(),
            CaptureSite::FfnDownInput => layers
                .iter()
                .map(|l| {
                    tap_out
                        .ffn_down_input
                        .get(l)
                        .map(|rows| pick_positions(rows))
                        .unwrap_or_default()
                })
                .collect(),
            CaptureSite::FfnDownOutput => layers
                .iter()
                .map(|l| {
                    tap_out
                        .ffn_down_output
                        .get(l)
                        .map(|rows| pick_positions(rows))
                        .unwrap_or_default()
                })
                .collect(),
            CaptureSite::AttnHeadOut => layers
                .iter()
                .map(|l| {
                    tap_out
                        .attn_head_out
                        .get(l)
                        .map(|rows| pick_positions(rows))
                        .unwrap_or_default()
                })
                .collect(),
            CaptureSite::AttnWeights => layers
                .iter()
                .map(|l| {
                    tap_out
                        .attn_weights
                        .get(l)
                        .map(|rows| pick_positions(rows))
                        .unwrap_or_default()
                })
                .collect(),
            CaptureSite::ShortConvGateB => layers
                .iter()
                .map(|l| {
                    tap_out
                        .short_conv
                        .get(l)
                        .map(|p| pick_positions(&p.gate_b))
                        .unwrap_or_default()
                })
                .collect(),
            CaptureSite::ShortConvGateC => layers
                .iter()
                .map(|l| {
                    tap_out
                        .short_conv
                        .get(l)
                        .map(|p| pick_positions(&p.gate_c))
                        .unwrap_or_default()
                })
                .collect(),
            CaptureSite::ShortConvConvOut => layers
                .iter()
                .map(|l| {
                    tap_out
                        .short_conv
                        .get(l)
                        .map(|p| pick_positions(&p.conv_out))
                        .unwrap_or_default()
                })
                .collect(),
            CaptureSite::ShortConvGatedY => layers
                .iter()
                .map(|l| {
                    tap_out
                        .short_conv
                        .get(l)
                        .map(|p| pick_positions(&p.gated_y))
                        .unwrap_or_default()
                })
                .collect(),
            CaptureSite::Final => {
                vec![observed.exits.iter().map(|e| e.hidden.clone()).collect()]
            }
            CaptureSite::Logits => {
                vec![observed.exits.iter().map(|e| e.logits.clone()).collect()]
            }
        };
        out.insert(site, blocks);
    }
    Ok(out)
}

/// A plane's `[hidden]` rows, refused on a component whose residual is not single-stream.
fn plane_rows(plane: &Plane, layer: usize) -> Result<&[Vec<f32>], VindexError> {
    plane.try_rows().map_err(|e| {
        VindexError::Parse(format!(
            "layer {layer}'s residual has no single [hidden] row per position to capture: {e}"
        ))
    })
}
