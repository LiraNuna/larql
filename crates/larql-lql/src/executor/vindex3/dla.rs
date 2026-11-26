//! Native Direct Logit Attribution (`DLA`) on VINDEX3 containers.
//!
//! Runs ONE batch prefill pass over the prompt (honoring any pending live weight edits),
//! linearizes the final norm at the prompt's last position to form `u_eff`, and computes:
//! - Exact additive layer attribution: `emb + sum_l (op_dla[l] + ffn_dla[l]) == pre_softcap_logit_diff`.
//! - Optional exact per-column FFN attribution (`TOP <k> COLUMNS`), linearizing `post_ffn_norm`
//!   (and `residual_scale` / `layer_scale` when declared) so `sum_c col_dla[l, c] == ffn_dla[l]`.

use std::collections::BTreeSet;

use larql_inference::vindex3::PlaneEvent;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandSource;
use larql_vindex::format::vindex3::opplan::exec::{
    execute_prepared_position_exits_with_taps, execute_prepared_prefill_position_exits_with_taps,
    requires_continuation, BatchTapRequest,
};
use larql_vindex::format::vindex3::opplan::{LayerAttention, LayerFfn};
use serde::Serialize;

use super::dla_math::{
    dot_diff_f64, dot_f64, format_dla_report, linearize_norm, load_two_rows, mixer_label,
    plane_rows, resolve_dla_layers, resolve_dla_target,
};
use super::{encode_raw_prompt, prefix_cache, prepared_operands_for_session, v3_continuation};
use crate::ast::{CaptureLayers, DlaTarget};
use crate::error::LqlError;
use crate::executor::Session;

/// One layer's direct logit attribution row.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DlaLayerRow {
    pub layer: usize,
    pub mixer: &'static str,
    pub operator_dla: f32,
    pub ffn_dla: f32,
    pub layer_dla: f32,
    pub cumulative_dla: f32,
}

/// One dense FFN column/neuron's direct logit attribution.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DlaColumnRow {
    pub layer: usize,
    pub column: usize,
    pub activation: f32,
    pub projection: f32,
    pub contribution: f32,
}

/// One softmax attention head's direct logit attribution.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DlaHeadRow {
    pub layer: usize,
    pub head: usize,
    pub head_norm: f32,
    pub contribution: f32,
}

/// One `ShortConv` channel's direct logit attribution (`gated_y[c] * (u_conv · W_out[:, c])`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DlaChannelRow {
    pub layer: usize,
    pub channel: usize,
    pub activation: f32,
    pub projection: f32,
    pub contribution: f32,
}

/// Typed result of `DLA "<prompt>" FOR "<token_a>" VS "<token_b>" ...;`.
#[derive(Debug, Clone, Serialize)]
pub struct DlaReport {
    pub prompt_ids: Vec<u32>,
    pub target_a: String,
    pub token_id_a: u32,
    pub logit_a: f32,
    pub target_b: String,
    pub token_id_b: u32,
    pub logit_b: f32,
    pub exact_logit_diff: f32,
    pub pre_softcap_diff: f32,
    pub softcapping: Option<f32>,
    pub embedding_dla: f32,
    pub total_dla: f32,
    pub layers: Vec<DlaLayerRow>,
    pub top_columns: Vec<DlaColumnRow>,
    pub dense_ffn_layers_attributed: usize,
    pub top_heads: Vec<DlaHeadRow>,
    pub softmax_layers_attributed: usize,
    pub top_channels: Vec<DlaChannelRow>,
    pub short_conv_layers_attributed: usize,
    pub executes_patched: bool,
    pub elapsed_ms: f64,
}

impl Session {
    /// Typed direct logit attribution for `target_a` vs `target_b` (text literal or explicit token ID)
    /// at the prompt's final position.
    #[allow(clippy::too_many_arguments)]
    pub fn dla(
        &self,
        prompt: &str,
        target_a: impl Into<DlaTarget>,
        target_b: impl Into<DlaTarget>,
        layers: Option<&CaptureLayers>,
        top_columns: Option<u32>,
        top_heads: Option<u32>,
        top_channels: Option<u32>,
    ) -> Result<DlaReport, LqlError> {
        let target_a = target_a.into();
        let target_b = target_b.into();
        let (runtime, tokenizer, overlay, bos_token, prepared_cache) =
            self.v3_text_program("DLA")?;
        let plan = runtime.plan();
        if !plan.residual_topology.is_single_stream() {
            return Err(LqlError::Execution(format!(
                "DLA requires a single-stream residual topology, and `{}` declares {:?}",
                plan.component, plan.residual_topology
            )));
        }
        let output_op = plan
            .output
            .as_ref()
            .ok_or_else(|| LqlError::Execution("the component carries no output head".into()))?;
        let vocab_size = output_op.projection.shape[0];
        let hidden = output_op.projection.shape[1];

        let requested_layers = resolve_dla_layers(layers, plan.layers.len())?;
        let top_k = top_columns.unwrap_or(0) as usize;
        let top_h = top_heads.unwrap_or(0) as usize;
        let top_c = top_channels.unwrap_or(0) as usize;
        if top_columns == Some(0) {
            return Err(LqlError::Execution(
                "DLA: TOP column/neuron count must be at least 1".into(),
            ));
        }
        if top_heads == Some(0) {
            return Err(LqlError::Execution(
                "DLA: TOP head count must be at least 1".into(),
            ));
        }
        if top_channels == Some(0) {
            return Err(LqlError::Execution(
                "DLA: TOP channel count must be at least 1".into(),
            ));
        }

        let prompt_ids = encode_raw_prompt(tokenizer, prompt, bos_token)?;
        if prompt_ids.is_empty() {
            return Err(LqlError::Execution(
                "DLA: prompt encoded to zero tokens".into(),
            ));
        }
        let (token_id_a, label_a) = resolve_dla_target(tokenizer, &target_a, "FOR", vocab_size)?;
        let (token_id_b, label_b) = resolve_dla_target(tokenizer, &target_b, "VS", vocab_size)?;

        let start = std::time::Instant::now();
        let (ops, overrides, cache_key) =
            prepared_operands_for_session(runtime, overlay, prepared_cache)?;
        let executes_patched = overrides.is_some();
        let source = match &overrides {
            Some(ov) => OperandSource::overlaid(runtime.store(), ov),
            None => runtime.operands(),
        };

        let last = prompt_ids.len() - 1;
        let mut taps = BatchTapRequest::default();
        let mut dense_layers: Vec<usize> = Vec::new();
        let mut softmax_layers: Vec<usize> = Vec::new();
        let mut short_conv_layers: Vec<usize> = Vec::new();
        if top_k > 0 {
            for &l in &requested_layers {
                if plan.layers[l]
                    .ffn
                    .as_ref()
                    .and_then(LayerFfn::dense)
                    .is_some()
                {
                    taps.ffn_down_input.insert(l);
                    if plan.layers[l].post_ffn_norm.is_some() {
                        taps.ffn_down_output.insert(l);
                    }
                    dense_layers.push(l);
                }
            }
        }
        if top_h > 0 {
            for &l in &requested_layers {
                if plan.layers[l].attention.softmax().is_some() {
                    taps.attn_head_out.insert(l);
                    softmax_layers.push(l);
                }
            }
        }
        if top_c > 0 {
            for &l in &requested_layers {
                if matches!(plan.layers[l].attention, LayerAttention::ShortConv(_)) {
                    taps.short_conv.insert(l);
                    short_conv_layers.push(l);
                }
            }
        }

        let mut h_emb: Option<Vec<f32>> = None;
        let mut post_attn: Vec<Option<Vec<f32>>> = vec![None; plan.layers.len()];
        let mut post_layer: Vec<Option<Vec<f32>>> = vec![None; plan.layers.len()];
        let mut sink = |event: PlaneEvent| {
            match event {
                PlaneEvent::Embedded(plane) => {
                    let rows = plane_rows(plane, "embedding")?;
                    h_emb = rows.last().cloned();
                }
                PlaneEvent::Layer { index, trace } => {
                    let pa = plane_rows(&trace.post_attention, "post_attention")?;
                    let pl = plane_rows(&trace.post_layer, "post_layer")?;
                    post_attn[index] = pa.last().cloned();
                    post_layer[index] = pl.last().cloned();
                }
                _ => {}
            }
            Ok(())
        };

        let can_prefix_cache = prefix_cache::plan_supports_prefix_cache(runtime);
        let (observed, tap_out) = if let Some(ref key) = cache_key.filter(|_| can_prefix_cache) {
            let (mut kv, prefix_len) = prefix_cache::acquire_prefix_state(
                runtime,
                &ops,
                prepared_cache.prefix(),
                key,
                &prompt_ids,
                last,
            )?;
            let local_last = last - prefix_len;
            let (obs, to) = execute_prepared_prefill_position_exits_with_taps(
                plan,
                &ops,
                &prompt_ids[prefix_len..],
                runtime.backend(),
                &mut kv,
                &[local_last],
                &taps,
                &mut sink,
            )
            .map_err(|e| LqlError::exec("DLA forward pass failed", e))?;
            prepared_cache.prefix().record_snapshot(
                key,
                prompt_ids.clone(),
                kv,
                obs.output.logits.clone(),
            );
            (obs, to)
        } else {
            let continuation = v3_continuation(runtime)?;
            let state = requires_continuation(plan).then(|| continuation.build());
            execute_prepared_position_exits_with_taps(
                plan,
                &ops,
                &prompt_ids,
                runtime.backend(),
                state,
                &[last],
                &taps,
                &mut sink,
            )
            .map_err(|e| LqlError::exec("DLA forward pass failed", e))?
        };

        let exit = observed
            .exits
            .first()
            .ok_or_else(|| LqlError::Execution("DLA: traversal returned no exit logits".into()))?;
        let logit_a = exit.logits[token_id_a as usize];
        let logit_b = exit.logits[token_id_b as usize];
        let exact_logit_diff = logit_a - logit_b;

        let h_emb = h_emb
            .ok_or_else(|| LqlError::Execution("DLA: embedding state was not captured".into()))?;
        let pre_final = if plan.layers.is_empty() {
            h_emb.clone()
        } else {
            post_layer[plan.layers.len() - 1].clone().ok_or_else(|| {
                LqlError::Execution("DLA: final layer residual was not captured".into())
            })?
        };

        // Read the two vocabulary rows from the resident prepared output head (matching the
        // exact Q8 or BF16 realization that computed `exit.logits`), falling back to a
        // 2-row range read from `source` for exotic packed head representations.
        let (row_a, row_b) = if token_id_a == token_id_b {
            match ops
                .select_output_head(&[token_id_a])
                .and_then(|h| h.row_f32(0))
            {
                Ok(r) => (r.clone(), r),
                Err(_) => load_two_rows(
                    &source,
                    &output_op.projection,
                    token_id_a as usize,
                    token_id_b as usize,
                    vocab_size,
                    hidden,
                )
                .map_err(|e| LqlError::exec("DLA failed to read output head rows", e))?,
            }
        } else {
            match ops
                .select_output_head(&[token_id_a, token_id_b])
                .and_then(|h| Ok((h.row_f32(0)?, h.row_f32(1)?)))
            {
                Ok(pair) => pair,
                Err(_) => load_two_rows(
                    &source,
                    &output_op.projection,
                    token_id_a as usize,
                    token_id_b as usize,
                    vocab_size,
                    hidden,
                )
                .map_err(|e| LqlError::exec("DLA failed to read output head rows", e))?,
            }
        };

        let mult = output_op.multiplier.unwrap_or(1.0) as f32;
        let u_diff: Vec<f32> = row_a
            .iter()
            .zip(row_b.iter())
            .map(|(&a, &b)| (a - b) * mult)
            .collect();

        // Pre-softcap exact difference is `dot(u_diff, final_hidden)`.
        let pre_softcap_diff = dot_f64(&u_diff, &exit.hidden) as f32;

        // Pull `u_diff` back through `final_norm` at the operating point `pre_final`.
        let u_eff = match &plan.final_norm {
            Some(norm) => linearize_norm(&source, norm, &pre_final, &u_diff)
                .map_err(|e| LqlError::exec("DLA failed to linearize final norm", e))?,
            None => u_diff,
        };

        let embedding_dla = dot_f64(&u_eff, &h_emb) as f32;
        let requested_set: BTreeSet<usize> = requested_layers.iter().copied().collect();

        let mut cum = embedding_dla as f64;
        let mut all_layer_rows: Vec<DlaLayerRow> = Vec::with_capacity(requested_layers.len());
        let mut prev = &h_emb;
        for (l, layer_plan) in plan.layers.iter().enumerate() {
            let pa = post_attn[l].as_ref().ok_or_else(|| {
                LqlError::Execution(format!("DLA: layer {l} post_attention missing"))
            })?;
            let pl = post_layer[l]
                .as_ref()
                .ok_or_else(|| LqlError::Execution(format!("DLA: layer {l} post_layer missing")))?;

            // Note: if `layer_scale` is present, `pl = layer_scale * (pa + ffn_delta)`,
            // so `pl - prev` still telescopes cleanly to `pre_final - h_emb`.
            let op_dla = dot_diff_f64(&u_eff, pa, prev) as f32;
            let ffn_dla = dot_diff_f64(&u_eff, pl, pa) as f32;
            let layer_dla = op_dla + ffn_dla;
            cum += (op_dla as f64) + (ffn_dla as f64);

            if requested_set.contains(&l) {
                all_layer_rows.push(DlaLayerRow {
                    layer: l,
                    mixer: mixer_label(&layer_plan.attention),
                    operator_dla: op_dla,
                    ffn_dla,
                    layer_dla,
                    cumulative_dla: cum as f32,
                });
            }
            prev = pl;
        }
        let total_dla = cum as f32;

        // Optional exact per-column FFN attribution across requested dense FFN layers.
        let mut columns: Vec<DlaColumnRow> = Vec::new();
        for &l in &dense_layers {
            let layer_plan = &plan.layers[l];
            let Some(ffn) = layer_plan.ffn.as_ref().and_then(LayerFfn::dense) else {
                continue;
            };
            let down_in_rows = tap_out.ffn_down_input.get(&l).ok_or_else(|| {
                LqlError::Execution(format!("DLA: layer {l} ffn.down_input was not captured"))
            })?;
            let act = down_in_rows.last().ok_or_else(|| {
                LqlError::Execution(format!("DLA: layer {l} ffn.down_input empty"))
            })?;

            // Effective direction entering `W_down` output space (`down_output` before post_ffn_norm).
            let mut u_ffn = u_eff.clone();
            if let Some(scale) = layer_plan.residual_scale {
                for v in &mut u_ffn {
                    *v *= scale;
                }
            }
            if let Some(layer_scale_ref) = &layer_plan.layer_scale {
                let s = source
                    .load(layer_scale_ref)
                    .map_err(|e| LqlError::exec("DLA failed to load layer_scale", e))?;
                if let Some(&scalar) = s.first() {
                    for v in &mut u_ffn {
                        *v *= scalar;
                    }
                }
            }
            if let Some(post_norm) = &layer_plan.post_ffn_norm {
                let down_out_rows = tap_out.ffn_down_output.get(&l).ok_or_else(|| {
                    LqlError::Execution(format!("DLA: layer {l} ffn.down_output was not captured"))
                })?;
                let down_out_last = down_out_rows.last().ok_or_else(|| {
                    LqlError::Execution(format!("DLA: layer {l} ffn.down_output empty"))
                })?;
                u_ffn = linearize_norm(&source, post_norm, down_out_last, &u_ffn)
                    .map_err(|e| LqlError::exec("DLA failed to linearize post_ffn_norm", e))?;
            }

            let w_down = ops
                .dense_ffn_down_f32(plan, l)
                .or_else(|_| source.load(&ffn.down))
                .map_err(|e| LqlError::exec("DLA failed to load FFN down projection", e))?;
            let intermediate = ffn.intermediate_size;
            if w_down.len() != hidden * intermediate || act.len() != intermediate {
                return Err(LqlError::Execution(format!(
                    "DLA: layer {l} down projection shape mismatch (got {} weights, {} activations, expected {hidden}x{intermediate})",
                    w_down.len(),
                    act.len()
                )));
            }

            // Compute `col_proj[c] = sum_r u_ffn[r] * W_down[r, c]` in row-major order (cache-friendly!).
            let mut col_proj = vec![0.0f64; intermediate];
            for (r, &ur) in u_ffn.iter().enumerate() {
                let ur = ur as f64;
                if ur == 0.0 {
                    continue;
                }
                let row = &w_down[r * intermediate..(r + 1) * intermediate];
                for (acc, &w) in col_proj.iter_mut().zip(row.iter()) {
                    *acc += ur * (w as f64);
                }
            }

            for (c, (&a, &p)) in act.iter().zip(col_proj.iter()).enumerate() {
                let proj = p as f32;
                let contrib = ((a as f64) * p) as f32;
                columns.push(DlaColumnRow {
                    layer: l,
                    column: c,
                    activation: a,
                    projection: proj,
                    contribution: contrib,
                });
            }
        }

        columns.sort_by(|a, b| {
            b.contribution
                .abs()
                .total_cmp(&a.contribution.abs())
                .then_with(|| a.layer.cmp(&b.layer))
                .then_with(|| a.column.cmp(&b.column))
        });
        columns.truncate(top_k);

        // Optional exact per-head softmax attention attribution across requested softmax layers.
        let mut heads: Vec<DlaHeadRow> = Vec::new();
        for &l in &softmax_layers {
            let layer_plan = &plan.layers[l];
            let Some(attn_op) = layer_plan.attention.softmax() else {
                continue;
            };
            let head_out_rows = tap_out.attn_head_out.get(&l).ok_or_else(|| {
                LqlError::Execution(format!("DLA: layer {l} attn.head_out was not captured"))
            })?;
            let head_out = head_out_rows.last().ok_or_else(|| {
                LqlError::Execution(format!("DLA: layer {l} attn.head_out empty"))
            })?;
            let q_rows = attn_op.num_q_heads * attn_op.head_dim;
            let w_o = ops
                .attention_o_proj_f32(plan, l)
                .or_else(|_| source.load(&attn_op.o))
                .map_err(|e| LqlError::exec("DLA failed to load attention o_proj", e))?;
            if w_o.len() != hidden * q_rows || head_out.len() != q_rows {
                return Err(LqlError::Execution(format!(
                    "DLA: layer {l} o_proj shape mismatch (got {} weights, {} head_out, expected {hidden}x{q_rows})",
                    w_o.len(),
                    head_out.len()
                )));
            }

            let mut u_attn = u_eff.clone();
            if let Some(scale) = layer_plan.residual_scale {
                for v in &mut u_attn {
                    *v *= scale;
                }
            }
            if let Some(layer_scale_ref) = &layer_plan.layer_scale {
                let s = source
                    .load(layer_scale_ref)
                    .map_err(|e| LqlError::exec("DLA failed to load layer_scale", e))?;
                if let Some(&scalar) = s.first() {
                    for v in &mut u_attn {
                        *v *= scalar;
                    }
                }
            }
            if let Some(post_norm) = &layer_plan.post_attention_norm {
                // Reconstruct exact pre-post_attention_norm output `W_o @ head_out + o_bias`
                // as the linearization operating point.
                let o_bias = attn_op
                    .o_bias
                    .as_ref()
                    .map(|b| source.load(b))
                    .transpose()
                    .map_err(|e| LqlError::exec("DLA failed to load o_proj bias", e))?;
                let mut attn_out = vec![0.0f32; hidden];
                for (r, dst) in attn_out.iter_mut().enumerate() {
                    let row = &w_o[r * q_rows..(r + 1) * q_rows];
                    let mut sum: f64 = row
                        .iter()
                        .zip(head_out.iter())
                        .map(|(&w, &x)| (w as f64) * (x as f64))
                        .sum();
                    if let Some(ref bias) = o_bias {
                        sum += bias[r] as f64;
                    }
                    *dst = sum as f32;
                }
                u_attn = linearize_norm(&source, post_norm, &attn_out, &u_attn).map_err(|e| {
                    LqlError::exec("DLA failed to linearize post_attention_norm", e)
                })?;
            }

            // Compute `in_proj[j] = sum_r u_attn[r] * W_o[r, j]` in cache-friendly row-major order.
            let mut in_proj = vec![0.0f64; q_rows];
            for (r, &ur) in u_attn.iter().enumerate() {
                let ur = ur as f64;
                if ur == 0.0 {
                    continue;
                }
                let row = &w_o[r * q_rows..(r + 1) * q_rows];
                for (acc, &w) in in_proj.iter_mut().zip(row.iter()) {
                    *acc += ur * (w as f64);
                }
            }

            let hd = attn_op.head_dim;
            for h in 0..attn_op.num_q_heads {
                let ho = &head_out[h * hd..(h + 1) * hd];
                let pr = &in_proj[h * hd..(h + 1) * hd];
                let contrib: f64 = ho
                    .iter()
                    .zip(pr.iter())
                    .map(|(&x, &p)| (x as f64) * p)
                    .sum();
                let norm: f64 = ho
                    .iter()
                    .map(|&x| (x as f64) * (x as f64))
                    .sum::<f64>()
                    .sqrt();
                heads.push(DlaHeadRow {
                    layer: l,
                    head: h,
                    head_norm: norm as f32,
                    contribution: contrib as f32,
                });
            }
        }
        heads.sort_by(|a, b| {
            b.contribution
                .abs()
                .total_cmp(&a.contribution.abs())
                .then_with(|| a.layer.cmp(&b.layer))
                .then_with(|| a.head.cmp(&b.head))
        });
        heads.truncate(top_h);

        // Optional exact per-channel ShortConv attribution across requested ShortConv layers.
        let mut channels: Vec<DlaChannelRow> = Vec::new();
        for &l in &short_conv_layers {
            let layer_plan = &plan.layers[l];
            let LayerAttention::ShortConv(sc_op) = &layer_plan.attention else {
                continue;
            };
            let planes = tap_out.short_conv.get(&l).ok_or_else(|| {
                LqlError::Execution(format!(
                    "DLA: layer {l} short_conv.gated_y was not captured"
                ))
            })?;
            let gated_y = planes.gated_y.last().ok_or_else(|| {
                LqlError::Execution(format!("DLA: layer {l} short_conv.gated_y empty"))
            })?;
            let conv_dim = sc_op.out_proj.shape[1];
            let w_out = ops
                .short_conv_out_proj_f32(plan, l)
                .or_else(|_| source.load(&sc_op.out_proj))
                .map_err(|e| LqlError::exec("DLA failed to load ShortConv out_proj", e))?;
            if w_out.len() != hidden * conv_dim || gated_y.len() != conv_dim {
                return Err(LqlError::Execution(format!(
                    "DLA: layer {l} ShortConv out_proj shape mismatch (got {} weights, {} gated_y, expected {hidden}x{conv_dim})",
                    w_out.len(),
                    gated_y.len()
                )));
            }

            let mut u_conv = u_eff.clone();
            if let Some(scale) = layer_plan.residual_scale {
                for v in &mut u_conv {
                    *v *= scale;
                }
            }
            if let Some(layer_scale_ref) = &layer_plan.layer_scale {
                let s = source
                    .load(layer_scale_ref)
                    .map_err(|e| LqlError::exec("DLA failed to load layer_scale", e))?;
                if let Some(&scalar) = s.first() {
                    for v in &mut u_conv {
                        *v *= scalar;
                    }
                }
            }
            if let Some(post_norm) = &layer_plan.post_attention_norm {
                let out_bias = sc_op
                    .out_proj_bias
                    .as_ref()
                    .map(|b| source.load(b))
                    .transpose()
                    .map_err(|e| LqlError::exec("DLA failed to load ShortConv out_proj bias", e))?;
                let mut conv_out = vec![0.0f32; hidden];
                for (r, dst) in conv_out.iter_mut().enumerate() {
                    let row = &w_out[r * conv_dim..(r + 1) * conv_dim];
                    let mut sum: f64 = row
                        .iter()
                        .zip(gated_y.iter())
                        .map(|(&w, &x)| (w as f64) * (x as f64))
                        .sum();
                    if let Some(ref bias) = out_bias {
                        sum += bias[r] as f64;
                    }
                    *dst = sum as f32;
                }
                u_conv = linearize_norm(&source, post_norm, &conv_out, &u_conv)
                    .map_err(|e| LqlError::exec("DLA failed to linearize post_operator_norm", e))?;
            }

            let mut chan_proj = vec![0.0f64; conv_dim];
            for (r, &ur) in u_conv.iter().enumerate() {
                let ur = ur as f64;
                if ur == 0.0 {
                    continue;
                }
                let row = &w_out[r * conv_dim..(r + 1) * conv_dim];
                for (acc, &w) in chan_proj.iter_mut().zip(row.iter()) {
                    *acc += ur * (w as f64);
                }
            }

            for (c, (&y, &p)) in gated_y.iter().zip(chan_proj.iter()).enumerate() {
                channels.push(DlaChannelRow {
                    layer: l,
                    channel: c,
                    activation: y,
                    projection: p as f32,
                    contribution: ((y as f64) * p) as f32,
                });
            }
        }
        channels.sort_by(|a, b| {
            b.contribution
                .abs()
                .total_cmp(&a.contribution.abs())
                .then_with(|| a.layer.cmp(&b.layer))
                .then_with(|| a.channel.cmp(&b.channel))
        });
        channels.truncate(top_c);

        Ok(DlaReport {
            prompt_ids,
            target_a: label_a,
            token_id_a,
            logit_a,
            target_b: label_b,
            token_id_b,
            logit_b,
            exact_logit_diff,
            pre_softcap_diff,
            softcapping: output_op.softcapping,
            embedding_dla,
            total_dla,
            layers: all_layer_rows,
            top_columns: columns,
            dense_ffn_layers_attributed: dense_layers.len(),
            top_heads: heads,
            softmax_layers_attributed: softmax_layers.len(),
            top_channels: channels,
            short_conv_layers_attributed: short_conv_layers.len(),
            executes_patched,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_dla(
        &self,
        prompt: &str,
        target_a: &DlaTarget,
        target_b: &DlaTarget,
        layers: Option<&CaptureLayers>,
        top_columns: Option<u32>,
        top_heads: Option<u32>,
        top_channels: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let report = self.dla(
            prompt,
            target_a,
            target_b,
            layers,
            top_columns,
            top_heads,
            top_channels,
        )?;
        Ok(format_dla_report(&report))
    }
}
