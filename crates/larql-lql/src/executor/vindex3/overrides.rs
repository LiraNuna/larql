//! Override composition, residual capture and continuation for VINDEX3.

use std::sync::{Arc, Mutex};

use crate::error::LqlError;
use larql_kv::{shipped_continuations, CanonicalFactory};
use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::{
    ContinuationFactory, SelectedContinuation,
};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandSource;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::tokenizers::Tokenizer;

/// Cache key uniquely identifying the resident `PreparedOperands` for a VINDEX3 session:
/// every live patched tensor's `(name, sha256)` in order (empty for unpatched base weights).
#[derive(Clone, PartialEq, Eq)]
struct PreparedCacheKey {
    patched_tensors: Vec<(String, String)>,
}

pub(crate) struct CachedPrepared {
    key: PreparedCacheKey,
    operands: Arc<PreparedOperands>,
}

/// Session-resident cache of lowered `PreparedOperands`, shared across `INFER`, `GENERATE`,
/// `CAPTURE`, and `DLA` so weights are loaded and Q8-quantised once per patch state.
#[derive(Default)]
pub(crate) struct PreparedCache {
    slot: Mutex<Option<CachedPrepared>>,
}

impl PreparedCache {
    pub(crate) fn clear(&self) {
        if let Ok(mut guard) = self.slot.lock() {
            *guard = None;
        }
    }
}

/// Resolve and return resident [`PreparedOperands`] for the session's current weight state,
/// reusing the cached image when the live patched-tensor digests match and rebuilding (and
/// verifying every patched tensor was read through the operand seam) on a cache miss.
/// Compose vector edits bypass the cache so every vector mutation executes fresh.
pub(crate) fn prepared_operands_for_session(
    runtime: &V3Runtime,
    overlay: &KnowledgeOverlay,
    cache: &PreparedCache,
) -> Result<
    (
        Arc<PreparedOperands>,
        Option<larql_inference::vindex3::OperandOverrides>,
    ),
    LqlError,
> {
    let overrides = execution_overrides(runtime, overlay)?;
    if overlay.has_vector_state() {
        let source = match &overrides {
            Some(ov) => OperandSource::overlaid(runtime.store(), ov),
            None => runtime.operands(),
        };
        let ops = PreparedOperands::load(
            runtime.plan(),
            source,
            runtime.backend(),
            ExecutionSlice::Full,
        )
        .map_err(|e| LqlError::exec("prepare operands failed", e))?;
        confirm_patched_weights_ran(overrides.as_ref())?;
        return Ok((Arc::new(ops), overrides));
    }

    let patched_tensors = match overlay.live_weights() {
        Ok(Some(view)) => view
            .tensors()
            .iter()
            .map(|t| (t.tensor.clone(), t.sha256.clone()))
            .collect(),
        Ok(None) => Vec::new(),
        Err(refusal) => {
            return Err(LqlError::Execution(format!(
                "refused: stored-weight ops are pending in this session and cannot be executed, \
                 and the stored weights will not be executed in their place: {refusal}"
            )));
        }
    };
    let key = PreparedCacheKey { patched_tensors };

    {
        let mut guard = cache.slot.lock().unwrap();
        if let Some(entry) = guard.as_ref().filter(|e| e.key == key) {
            return Ok((Arc::clone(&entry.operands), overrides));
        }
        // Release the previous resident image before allocating the replacement so peak RAM
        // stays bounded to one prepared image during patch sweeps.
        *guard = None;
    }

    let source = match &overrides {
        Some(ov) => OperandSource::overlaid(runtime.store(), ov),
        None => runtime.operands(),
    };
    let ops = Arc::new(
        PreparedOperands::load(
            runtime.plan(),
            source,
            runtime.backend(),
            ExecutionSlice::Full,
        )
        .map_err(|e| LqlError::exec("prepare operands failed", e))?,
    );
    confirm_patched_weights_ran(overrides.as_ref())?;
    *cache.slot.lock().unwrap() = Some(CachedPrepared {
        key,
        operands: Arc::clone(&ops),
    });
    Ok((ops, overrides))
}

/// The one raw-prompt encoder every V3 surface calls — the BOS rule
/// lives with it in larql-inference, not here.
pub(crate) use larql_inference::vindex3::encode_raw_prompt;

#[allow(unused_imports)]
use super::*;

/// One observed pass over the runtime's plan, returning the last
/// position's **normed FFN input** at `layer` — V2's exact install
/// statistic (its walk-FFN trace captures the post-norm vector the
/// gates multiply), so a gate built from it fires on the prompt that
/// produced it (V3-LQL-3B capture: the KNN key and the compose gate
/// direction). The pass is the same canonical traversal INFER runs;
/// the tap is a subscription, never a second executor.
/// The overlay's compose edits as executor operand overrides — `None`
/// while no compose state exists, so callers keep the plain (bit-for-
/// bit identical) execution path.
pub(crate) fn compose_overrides(
    runtime: &V3Runtime,
    overlay: &larql_vindex::format::vindex3::knowledge::KnowledgeOverlay,
) -> Result<Option<larql_inference::vindex3::OperandOverrides>, LqlError> {
    if !overlay.has_vector_state() {
        return Ok(None);
    }
    overlay
        .operand_overrides(runtime.plan())
        .map(Some)
        .map_err(|e| LqlError::exec("failed to derive operand overrides", e))
}

/// What an EXECUTION of the session's program runs over: the replacements for its pending stored-weight ops,
/// or else its compose edits, or `None` when the overlay changes nothing execution computes.
///
/// Every statement that runs the program asks this, never [`compose_overrides`] — which is what CURRENT means
/// to `DIFF` and to the knowledge COMPILE, and deliberately leaves stored-weight ops out. With such an op in
/// effect the answer is its patched bytes or a refusal that says why; it is never the stored weights.
pub(crate) fn execution_overrides(
    runtime: &V3Runtime,
    overlay: &larql_vindex::format::vindex3::knowledge::KnowledgeOverlay,
) -> Result<Option<larql_inference::vindex3::OperandOverrides>, LqlError> {
    match overlay.live_weight_overrides() {
        Ok(Some(overrides)) => Ok(Some(overrides)),
        Ok(None) => compose_overrides(runtime, overlay),
        Err(refusal) => Err(LqlError::Execution(format!(
            "refused: stored-weight ops are pending in this session and cannot be executed, and the stored \
             weights will not be executed in their place: {refusal}"
        ))),
    }
}

/// End an execution by refusing its result unless every patched tensor was read through the operand seam:
/// a tensor loaded from the container instead means the run was not the patched model.
pub(crate) fn confirm_patched_weights_ran(
    overrides: Option<&larql_inference::vindex3::OperandOverrides>,
) -> Result<(), LqlError> {
    match overrides {
        Some(overrides) => overrides
            .ensure_patched_read()
            .map_err(|e| LqlError::exec("refused: no result is reported", e)),
        None => Ok(()),
    }
}

pub(crate) fn capture_layer_residual(
    runtime: &V3Runtime,
    tokenizer: &Tokenizer,
    prompt: &str,
    layer: usize,
    overrides: Option<&larql_inference::vindex3::OperandOverrides>,
    bos: Option<u32>,
) -> Result<Vec<f32>, LqlError> {
    let prompt_ids = encode_raw_prompt(tokenizer, prompt, bos)?;
    let mut captured: Option<Vec<f32>> = None;
    let mut sink = |event: larql_inference::vindex3::PlaneEvent| {
        if let larql_inference::vindex3::PlaneEvent::Layer { index, trace } = event {
            if index == layer {
                captured = trace.ffn_input.last().cloned();
            }
        }
        Ok(())
    };
    // The capture runs over the same effective program INFER runs —
    // V2's contract (its capture forward observes the patch overlay).
    let continuation = v3_continuation(runtime)?;
    match overrides {
        Some(overrides) => {
            runtime.execute_streaming_overlaid(&prompt_ids, overrides, &continuation, &mut sink)
        }
        None => runtime.execute_streaming(&prompt_ids, &continuation, &mut sink),
    }
    .map_err(|e| LqlError::exec("v3 capture pass failed", e))?;
    confirm_patched_weights_ran(overrides)?;
    captured.ok_or_else(|| LqlError::Execution(format!("no residual captured at layer {layer}")))
}

/// LQL's continuation provider — the canonical cache, named by its
/// factory's identity (CONTINUATION-PLUGIN-1, C3) and selected against the
/// runtime's program before anything runs.
pub(crate) fn v3_continuation(runtime: &V3Runtime) -> Result<SelectedContinuation, LqlError> {
    runtime
        .select_continuation(
            &shipped_continuations(),
            &CanonicalFactory.identity(),
            &ContinuationConfig::empty(),
        )
        .map_err(|e| LqlError::exec("v3 continuation selection failed", e))
}

/// Softmax over the logits, then the top-k `(token_id, probability)`
/// pairs, ties keeping the lower id (the greedy sampler's rule).
pub(crate) fn top_k_probs(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut scored: Vec<(u32, f32)> = exps
        .iter()
        .enumerate()
        .map(|(id, &e)| (id as u32, e / sum))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}
