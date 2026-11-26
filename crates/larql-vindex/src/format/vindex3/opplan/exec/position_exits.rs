//! One fresh causal prefill, observed: every plane the traversal produces, and the exit at chosen positions.
//!
//! The one-shot traversal `INFER` runs computes a residual for every position at every layer and keeps only the
//! last position's exit. A caller that wants to see more of that same run (the planes at a layer, the final-normed
//! row or the logits at an earlier position) subscribes here instead of running anything else: plane events are
//! forwarded to the caller's sink as they occur, and the exit [`super::position_logits`] defines (final norm, then
//! the head) is applied to each requested position of the last layer's plane.
//!
//! Fail-closed self-check: the last position's exit, whenever any position is requested, must equal the
//! traversal's own exit bit for bit, both the final-normed row and the logits, or the call refuses.

use std::collections::{BTreeMap, BTreeSet};

use super::super::{LayerAttention, LayerFfn};
use super::continuation_registry::BoxedContinuation;
use super::hyper_connection::Mutation;
use super::kv::KvState;
use super::operands::OperandSource;
use super::position_logits::{exit_rows, row_exit, same_as_exit};
use super::prepared::{ExecutionSlice, PreparedOperands};
use super::short_conv::ShortConvBatchPlanes;
use super::{backend::PlanBackend, ComponentOpPlan, FinalOutput, FinalState, Plane, PlaneEvent};
use crate::error::VindexError;

/// Extra per-layer batch-prefill taps to collect during a traversal.
/// Empty by default so unobserved prefill pays zero extra work.
#[derive(Debug, Default, Clone)]
pub struct BatchTapRequest {
    /// Layers at which to capture dense FFN `down_input` (`[T][intermediate]`).
    pub ffn_down_input: BTreeSet<usize>,
    /// Layers at which to capture dense FFN `down_output` before post-FFN norm (`[T][hidden]`).
    pub ffn_down_output: BTreeSet<usize>,
    /// Layers at which to capture `ShortConv` internal planes (`gate_b`, `gate_c`, `conv_out`, `gated_y`).
    pub short_conv: BTreeSet<usize>,
}

impl BatchTapRequest {
    pub fn is_empty(&self) -> bool {
        self.ffn_down_input.is_empty()
            && self.ffn_down_output.is_empty()
            && self.short_conv.is_empty()
    }
}

/// Captured per-layer batch-prefill tap planes (populated only for layers in [`BatchTapRequest`]).
#[derive(Debug, Default)]
pub struct BatchTapOutput {
    pub ffn_down_input: BTreeMap<usize, Vec<Vec<f32>>>,
    pub ffn_down_output: BTreeMap<usize, Vec<Vec<f32>>>,
    pub short_conv: BTreeMap<usize, ShortConvBatchPlanes>,
}

/// What the exit computed at one position.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionExit {
    /// The `[hidden]` vector the head read: the final-normed residual, or the residual itself where the image
    /// declares no final norm.
    pub hidden: Vec<f32>,
    /// The logits predicting the token after this position.
    pub logits: Vec<f32>,
}

/// What [`execute_streaming_position_exits`] returns: the traversal's own output, as `INFER` receives it, and
/// one exit per requested position, in the order requested.
#[derive(Debug)]
pub struct ObservedPrefill {
    pub output: FinalOutput,
    pub exits: Vec<PositionExit>,
}

const LAST_HIDDEN_DIFFERS: &str =
    "the last position's final-normed row differs from the traversal's own \
     exit; the per-position exit is not the exit the traversal runs";

/// Execute `tokens` once over the whole stack, forward every [`PlaneEvent`] to `sink`, and apply the exit at each
/// of `positions`. With no positions requested, no plane is retained and this is the plain streaming traversal.
///
/// `state` is the fresh continuation provider a stateful stack needs, `None` on a softmax-only stack (the rule
/// [`super::requires_continuation`] states). Per-position exits need a single-stream stack.
pub fn execute_streaming_position_exits<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    state: Option<BoxedContinuation>,
    positions: &[usize],
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<ObservedPrefill, VindexError> {
    let (observed, _) = execute_streaming_position_exits_with_taps(
        plan,
        store,
        tokens,
        backend,
        state,
        positions,
        &BatchTapRequest::default(),
        sink,
    )?;
    Ok(observed)
}

/// [`execute_streaming_position_exits`], additionally collecting any requested
/// batch-prefill taps (`ffn.down_input`, `ffn.down_output`, `short_conv.*`)
/// in the exact same single traversal.
#[allow(clippy::too_many_arguments)]
pub fn execute_streaming_position_exits_with_taps<'s, B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    store: impl Into<OperandSource<'s>>,
    tokens: &[u32],
    backend: &B,
    mut state: Option<BoxedContinuation>,
    positions: &[usize],
    taps: &BatchTapRequest,
    sink: &mut dyn FnMut(PlaneEvent) -> Result<(), VindexError>,
) -> Result<(ObservedPrefill, BatchTapOutput), VindexError> {
    if let Some(bad) = positions.iter().find(|&&p| p >= tokens.len()) {
        return Err(VindexError::Parse(format!(
            "position {bad} is outside a {}-token sequence",
            tokens.len()
        )));
    }
    if !taps.ffn_down_input.is_empty() && !backend.serves_ffn_down_input() {
        return Err(VindexError::Parse(format!(
            "FFN down-input capture is not served by the {} backend",
            backend.name()
        )));
    }
    for &layer in taps.ffn_down_input.iter().chain(&taps.ffn_down_output) {
        let Some(lp) = plan.layers.get(layer) else {
            return Err(VindexError::Parse(format!(
                "layer {layer} is outside the plan's {} layers",
                plan.layers.len()
            )));
        };
        if !matches!(lp.ffn, Some(LayerFfn::Dense(_))) {
            return Err(VindexError::Parse(format!(
                "FFN down capture at layer {layer} requires a dense FFN"
            )));
        }
    }
    for &layer in &taps.short_conv {
        let Some(lp) = plan.layers.get(layer) else {
            return Err(VindexError::Parse(format!(
                "layer {layer} is outside the plan's {} layers",
                plan.layers.len()
            )));
        };
        if !matches!(lp.attention, LayerAttention::ShortConv(_)) {
            return Err(VindexError::Parse(format!(
                "ShortConv capture at layer {layer} needs a ShortConv layer: this layer's mixer \
                 is not ShortConv"
            )));
        }
    }
    let ops = PreparedOperands::load(plan, store, backend, ExecutionSlice::Full)?;
    let keep_exit_plane = !positions.is_empty();
    // Only the newest layer's plane is kept: each boundary replaces the last.
    let mut exit_plane: Option<Plane> = None;
    let mut forward = |event: PlaneEvent| {
        if keep_exit_plane {
            if let PlaneEvent::Layer { trace, .. } = &event {
                exit_plane = Some(trace.post_layer.clone());
            }
        }
        sink(event)
    };
    let mut tap_output = BatchTapOutput::default();
    let tap_arg = (!taps.is_empty()).then_some((taps, &mut tap_output));
    let output = super::streaming::execute_prepared_streaming_with_taps(
        plan,
        &ops,
        tokens,
        backend,
        None,
        &mut forward,
        state.as_mut().map(|s| &mut **s as &mut dyn KvState),
        Mutation::None,
        tap_arg,
    )?;
    if !keep_exit_plane {
        return Ok((
            ObservedPrefill {
                output,
                exits: Vec::new(),
            },
            tap_output,
        ));
    }
    let rows = exit_rows(exit_plane)?;
    let mut exits = Vec::with_capacity(positions.len());
    for &p in positions {
        let (hidden, logits) = row_exit(&ops, backend, &rows[p])?;
        if p + 1 == tokens.len() {
            same_as_exit(&logits, output.logits.as_deref())?;
            same_hidden(&hidden, &output.exit)?;
        }
        exits.push(PositionExit { hidden, logits });
    }
    Ok((ObservedPrefill { output, exits }, tap_output))
}

/// The final-normed row at the last position is the traversal's own exit state, bit for bit.
pub(super) fn same_hidden(hidden: &[f32], exit: &FinalState) -> Result<(), VindexError> {
    let same = exit.try_hidden().is_ok_and(|exit| {
        exit.len() == hidden.len()
            && exit
                .iter()
                .zip(hidden)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    });
    match same {
        true => Ok(()),
        false => Err(VindexError::Parse(LAST_HIDDEN_DIFFERS.into())),
    }
}
