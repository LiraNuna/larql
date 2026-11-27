//! Native `DIFF DLA` execution: compares the current VINDEX3 session state (active `OVERLAY`,
//! live weight edits, and/or active `SOFT_PROMPT`) against the untouched bound Base container
//! in a single hot session (reusing resident `cache.base` in 0ms), computing exact per-layer,
//! per-head, per-column/neuron, and per-channel attribution deltas (`candidate - base`).

use std::collections::BTreeMap;

use serde::Serialize;

use super::dla::compute_dla_on;
use super::dla_math::{resolve_dla_layers, resolve_dla_target};
use super::{
    encode_raw_prompt, prepared_operands_for_base, prepared_operands_for_session, BaseDlaCacheKey,
};
use crate::ast::{CaptureLayers, DlaTarget};
use crate::error::LqlError;
use crate::executor::Session;

/// Per-layer attribution delta (`candidate - base`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DlaLayerDelta {
    pub layer: usize,
    pub mixer: &'static str,
    pub base_operator_dla: f32,
    pub candidate_operator_dla: f32,
    pub delta_operator_dla: f32,
    pub base_ffn_dla: f32,
    pub candidate_ffn_dla: f32,
    pub delta_ffn_dla: f32,
    pub base_layer_dla: f32,
    pub candidate_layer_dla: f32,
    pub delta_layer_dla: f32,
}

/// Per-head attention attribution delta (`candidate - base`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DlaHeadDelta {
    pub layer: usize,
    pub head: usize,
    pub base: f32,
    pub candidate: f32,
    pub delta: f32,
}

/// Per-column/neuron FFN attribution delta (`candidate - base`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DlaColumnDelta {
    pub layer: usize,
    pub column: usize,
    pub base: f32,
    pub candidate: f32,
    pub delta: f32,
}

/// Per-channel ShortConv attribution delta (`candidate - base`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DlaChannelDelta {
    pub layer: usize,
    pub channel: usize,
    pub base: f32,
    pub candidate: f32,
    pub delta: f32,
}

/// Structured result of `DIFF DLA "<prompt>" FOR ... VS ...;`.
#[derive(Debug, Clone, Serialize)]
pub struct DlaDiffReport {
    pub prompt_ids: Vec<u32>,
    pub target_a: String,
    pub token_id_a: u32,
    pub target_b: String,
    pub token_id_b: u32,
    pub base_logit_diff: f32,
    pub candidate_logit_diff: f32,
    pub delta_logit_diff: f32,
    pub base_embedding_dla: f32,
    pub candidate_embedding_dla: f32,
    pub delta_embedding_dla: f32,
    pub candidate_has_weight_edits: bool,
    pub candidate_has_soft_prompt: bool,
    pub base_cached: bool,
    pub layer_deltas: Vec<DlaLayerDelta>,
    pub head_deltas: Vec<DlaHeadDelta>,
    pub column_deltas: Vec<DlaColumnDelta>,
    pub channel_deltas: Vec<DlaChannelDelta>,
    pub elapsed_ms: f64,
}

impl Session {
    /// Compare DLA on the session's active state (weight overlay/edits + soft prompt) against
    /// the untouched base container without reloading base weights.
    #[allow(clippy::too_many_arguments)]
    pub fn diff_dla(
        &self,
        prompt: &str,
        target_a: impl Into<DlaTarget>,
        target_b: impl Into<DlaTarget>,
        layers: Option<&CaptureLayers>,
        top_columns: Option<u32>,
        top_heads: Option<u32>,
        top_channels: Option<u32>,
    ) -> Result<DlaDiffReport, LqlError> {
        let target_a = target_a.into();
        let target_b = target_b.into();
        let (runtime, tokenizer, overlay, bos_token, prepared_cache) =
            self.v3_text_program("DIFF DLA")?;
        let start = std::time::Instant::now();

        // Build immutable Base DLA cache key so multi-checkpoint overlay sweeps reuse Base DLA!
        let plan = runtime.plan();
        let output_op = plan
            .output
            .as_ref()
            .ok_or_else(|| LqlError::Execution("the component carries no output head".into()))?;
        let vocab_size = output_op.projection.shape[0];
        let prompt_ids_for_key = encode_raw_prompt(tokenizer, prompt, bos_token)?;
        let (id_a, _) = resolve_dla_target(tokenizer, &target_a, "FOR", vocab_size)?;
        let (id_b, _) = resolve_dla_target(tokenizer, &target_b, "VS", vocab_size)?;
        let req_layers = resolve_dla_layers(layers, plan.layers.len())?;
        let base_dla_key = BaseDlaCacheKey {
            prompt_ids: prompt_ids_for_key,
            token_id_a: id_a,
            token_id_b: id_b,
            layers: req_layers,
            want_columns: top_columns.unwrap_or(0) > 0,
            want_heads: top_heads.unwrap_or(0) > 0,
            want_channels: top_channels.unwrap_or(0) > 0,
        };

        // 1. Run Base (untouched weights, no soft prompt) using resident `cache.base` + `base_dla_cache`.
        let (base_report, base_cached) =
            if let Some(cached) = prepared_cache.get_base_dla(&base_dla_key) {
                (cached, true)
            } else {
                let (base_ops, base_key) = prepared_operands_for_base(runtime, prepared_cache)?;
                let report = compute_dla_on(
                    runtime,
                    tokenizer,
                    bos_token,
                    prepared_cache,
                    &base_ops,
                    None,
                    Some(base_key),
                    None,
                    prompt,
                    &target_a,
                    &target_b,
                    layers,
                    top_columns,
                    top_heads,
                    top_channels,
                    false,
                    start,
                )?;
                prepared_cache.put_base_dla(base_dla_key, report.clone());
                (report, false)
            };

        // 2. Run Candidate (active overlay/weight edits + active soft prompt).
        let (cand_ops, cand_overrides, cand_key) =
            prepared_operands_for_session(runtime, overlay, prepared_cache)?;
        let soft_prompt = prepared_cache.soft_prompt();
        let candidate_has_weight_edits = cand_overrides.is_some();
        let candidate_has_soft_prompt = soft_prompt.is_some();
        let cand_report = compute_dla_on(
            runtime,
            tokenizer,
            bos_token,
            prepared_cache,
            &cand_ops,
            cand_overrides.as_ref(),
            cand_key,
            soft_prompt.as_deref(),
            prompt,
            &target_a,
            &target_b,
            layers,
            top_columns,
            top_heads,
            top_channels,
            false,
            start,
        )?;

        let layer_deltas: Vec<DlaLayerDelta> = base_report
            .layers
            .iter()
            .zip(cand_report.layers.iter())
            .map(|(b, c)| DlaLayerDelta {
                layer: b.layer,
                mixer: b.mixer,
                base_operator_dla: b.operator_dla,
                candidate_operator_dla: c.operator_dla,
                delta_operator_dla: c.operator_dla - b.operator_dla,
                base_ffn_dla: b.ffn_dla,
                candidate_ffn_dla: c.ffn_dla,
                delta_ffn_dla: c.ffn_dla - b.ffn_dla,
                base_layer_dla: b.layer_dla,
                candidate_layer_dla: c.layer_dla,
                delta_layer_dla: c.layer_dla - b.layer_dla,
            })
            .collect();

        let top_h = top_heads.unwrap_or(0) as usize;
        let mut head_deltas = Vec::new();
        if top_h > 0 {
            let base_map: BTreeMap<(usize, usize), f32> = base_report
                .top_heads
                .iter()
                .map(|r| ((r.layer, r.head), r.contribution))
                .collect();
            for c in &cand_report.top_heads {
                let b_val = base_map.get(&(c.layer, c.head)).copied().unwrap_or(0.0);
                head_deltas.push(DlaHeadDelta {
                    layer: c.layer,
                    head: c.head,
                    base: b_val,
                    candidate: c.contribution,
                    delta: c.contribution - b_val,
                });
            }
            head_deltas.sort_by(|a, b| {
                b.delta
                    .abs()
                    .total_cmp(&a.delta.abs())
                    .then_with(|| a.layer.cmp(&b.layer))
                    .then_with(|| a.head.cmp(&b.head))
            });
            head_deltas.truncate(top_h);
        }

        let top_k = top_columns.unwrap_or(0) as usize;
        let mut column_deltas = Vec::new();
        if top_k > 0 {
            let base_map: BTreeMap<(usize, usize), f32> = base_report
                .top_columns
                .iter()
                .map(|r| ((r.layer, r.column), r.contribution))
                .collect();
            for c in &cand_report.top_columns {
                let b_val = base_map.get(&(c.layer, c.column)).copied().unwrap_or(0.0);
                column_deltas.push(DlaColumnDelta {
                    layer: c.layer,
                    column: c.column,
                    base: b_val,
                    candidate: c.contribution,
                    delta: c.contribution - b_val,
                });
            }
            column_deltas.sort_by(|a, b| {
                b.delta
                    .abs()
                    .total_cmp(&a.delta.abs())
                    .then_with(|| a.layer.cmp(&b.layer))
                    .then_with(|| a.column.cmp(&b.column))
            });
            column_deltas.truncate(top_k);
        }

        let top_c = top_channels.unwrap_or(0) as usize;
        let mut channel_deltas = Vec::new();
        if top_c > 0 {
            let base_map: BTreeMap<(usize, usize), f32> = base_report
                .top_channels
                .iter()
                .map(|r| ((r.layer, r.channel), r.contribution))
                .collect();
            for c in &cand_report.top_channels {
                let b_val = base_map.get(&(c.layer, c.channel)).copied().unwrap_or(0.0);
                channel_deltas.push(DlaChannelDelta {
                    layer: c.layer,
                    channel: c.channel,
                    base: b_val,
                    candidate: c.contribution,
                    delta: c.contribution - b_val,
                });
            }
            channel_deltas.sort_by(|a, b| {
                b.delta
                    .abs()
                    .total_cmp(&a.delta.abs())
                    .then_with(|| a.layer.cmp(&b.layer))
                    .then_with(|| a.channel.cmp(&b.channel))
            });
            channel_deltas.truncate(top_c);
        }

        Ok(DlaDiffReport {
            prompt_ids: cand_report.prompt_ids,
            target_a: cand_report.target_a,
            token_id_a: cand_report.token_id_a,
            target_b: cand_report.target_b,
            token_id_b: cand_report.token_id_b,
            base_logit_diff: base_report.exact_logit_diff,
            candidate_logit_diff: cand_report.exact_logit_diff,
            delta_logit_diff: cand_report.exact_logit_diff - base_report.exact_logit_diff,
            base_embedding_dla: base_report.embedding_dla,
            candidate_embedding_dla: cand_report.embedding_dla,
            delta_embedding_dla: cand_report.embedding_dla - base_report.embedding_dla,
            candidate_has_weight_edits,
            candidate_has_soft_prompt,
            base_cached,
            layer_deltas,
            head_deltas,
            column_deltas,
            channel_deltas,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_diff_dla(
        &self,
        prompt: &str,
        target_a: &DlaTarget,
        target_b: &DlaTarget,
        layers: Option<&CaptureLayers>,
        top_columns: Option<u32>,
        top_heads: Option<u32>,
        top_channels: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let r = self.diff_dla(
            prompt,
            target_a.clone(),
            target_b.clone(),
            layers,
            top_columns,
            top_heads,
            top_channels,
        )?;
        let mut out = vec![
            format!(
                "DIFF DLA ({} prompt tokens; candidate weights_edited={} soft_prompt={}):",
                r.prompt_ids.len(),
                r.candidate_has_weight_edits,
                r.candidate_has_soft_prompt
            ),
            format!(
                "  Target A: {:?} [id {}]  vs  Target B: {:?} [id {}]",
                r.target_a, r.token_id_a, r.target_b, r.token_id_b
            ),
            format!(
                "  Logit diff: base {:+.4} -> cand {:+.4}  (delta {:+.4})",
                r.base_logit_diff, r.candidate_logit_diff, r.delta_logit_diff
            ),
            String::new(),
            format!(
                "  {:<6} {:<12} {:>10} {:>10} {:>10}",
                "Layer", "Mixer", "ΔOp DLA", "ΔFFN DLA", "ΔLayer"
            ),
            format!("  {}", "-".repeat(52)),
        ];
        for row in &r.layer_deltas {
            out.push(format!(
                "  L{:<5} {:<12} {:>+10.4} {:>+10.4} {:>+10.4}",
                row.layer,
                row.mixer,
                row.delta_operator_dla,
                row.delta_ffn_dla,
                row.delta_layer_dla
            ));
        }
        if !r.head_deltas.is_empty() {
            out.push(String::new());
            out.push(format!(
                "Top {} head deltas (|cand - base|):",
                r.head_deltas.len()
            ));
            for (i, h) in r.head_deltas.iter().enumerate() {
                out.push(format!(
                    "  {:>2}. L{:02}.H{:02}  base {:>+8.4} -> cand {:>+8.4}  (delta {:>+8.4})",
                    i + 1,
                    h.layer,
                    h.head,
                    h.base,
                    h.candidate,
                    h.delta
                ));
            }
        }
        if !r.column_deltas.is_empty() {
            out.push(String::new());
            out.push(format!(
                "Top {} neuron/column deltas (|cand - base|):",
                r.column_deltas.len()
            ));
            for (i, c) in r.column_deltas.iter().enumerate() {
                out.push(format!(
                    "  {:>2}. L{:02}:{:<5} base {:>+8.4} -> cand {:>+8.4}  (delta {:>+8.4})",
                    i + 1,
                    c.layer,
                    c.column,
                    c.base,
                    c.candidate,
                    c.delta
                ));
            }
        }
        if !r.channel_deltas.is_empty() {
            out.push(String::new());
            out.push(format!(
                "Top {} ShortConv channel deltas (|cand - base|):",
                r.channel_deltas.len()
            ));
            for (i, ch) in r.channel_deltas.iter().enumerate() {
                out.push(format!(
                    "  {:>2}. L{:02}:ch{:<4} base {:>+8.4} -> cand {:>+8.4}  (delta {:>+8.4})",
                    i + 1,
                    ch.layer,
                    ch.channel,
                    ch.base,
                    ch.candidate,
                    ch.delta
                ));
            }
        }
        out.push(format!("  {:.0}ms", r.elapsed_ms));
        Ok(out)
    }
}
