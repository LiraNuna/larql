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
    execute_prepared_position_exits_with_taps, requires_continuation, BatchTapRequest,
};
use larql_vindex::format::vindex3::opplan::LayerFfn;

use super::dla_math::{
    dot_diff_f64, dot_f64, linearize_norm, load_two_rows, mixer_label, plane_rows,
    resolve_dla_layers, resolve_single_token,
};
use super::{encode_raw_prompt, prepared_operands_for_session, v3_continuation};
use crate::ast::CaptureLayers;
use crate::error::LqlError;
use crate::executor::Session;

/// One layer's direct logit attribution row.
#[derive(Debug, Clone, PartialEq)]
pub struct DlaLayerRow {
    pub layer: usize,
    pub mixer: &'static str,
    pub operator_dla: f32,
    pub ffn_dla: f32,
    pub layer_dla: f32,
    pub cumulative_dla: f32,
}

/// One dense FFN column's direct logit attribution.
#[derive(Debug, Clone, PartialEq)]
pub struct DlaColumnRow {
    pub layer: usize,
    pub column: usize,
    pub activation: f32,
    pub projection: f32,
    pub contribution: f32,
}

/// One softmax attention head's direct logit attribution.
#[derive(Debug, Clone, PartialEq)]
pub struct DlaHeadRow {
    pub layer: usize,
    pub head: usize,
    pub head_norm: f32,
    pub contribution: f32,
}

/// Typed result of `DLA "<prompt>" FOR "<token_a>" VS "<token_b>" ...;`.
#[derive(Debug, Clone)]
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
    pub executes_patched: bool,
    pub elapsed_ms: f64,
}

impl Session {
    /// Typed direct logit attribution for `token_a` vs `token_b` at the prompt's final position.
    pub fn dla(
        &self,
        prompt: &str,
        target_a: &str,
        target_b: &str,
        layers: Option<&CaptureLayers>,
        top_columns: Option<u32>,
        top_heads: Option<u32>,
    ) -> Result<DlaReport, LqlError> {
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
        if top_columns == Some(0) {
            return Err(LqlError::Execution(
                "DLA: TOP column count must be at least 1".into(),
            ));
        }
        if top_heads == Some(0) {
            return Err(LqlError::Execution(
                "DLA: TOP head count must be at least 1".into(),
            ));
        }

        let prompt_ids = encode_raw_prompt(tokenizer, prompt, bos_token)?;
        if prompt_ids.is_empty() {
            return Err(LqlError::Execution(
                "DLA: prompt encoded to zero tokens".into(),
            ));
        }
        let token_id_a = resolve_single_token(tokenizer, target_a, "FOR", vocab_size)?;
        let token_id_b = resolve_single_token(tokenizer, target_b, "VS", vocab_size)?;

        let start = std::time::Instant::now();
        let (ops, overrides) = prepared_operands_for_session(runtime, overlay, prepared_cache)?;
        let executes_patched = overrides.is_some();
        let source = match &overrides {
            Some(ov) => OperandSource::overlaid(runtime.store(), ov),
            None => runtime.operands(),
        };

        let last = prompt_ids.len() - 1;
        let mut taps = BatchTapRequest::default();
        let mut dense_layers: Vec<usize> = Vec::new();
        let mut softmax_layers: Vec<usize> = Vec::new();
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

        let mut h_emb: Option<Vec<f32>> = None;
        let mut post_attn: Vec<Option<Vec<f32>>> = vec![None; plan.layers.len()];
        let mut post_layer: Vec<Option<Vec<f32>>> = vec![None; plan.layers.len()];
        let mut sink = |event: PlaneEvent| {
            match event {
                PlaneEvent::Embedded(plane) => {
                    let rows = plane_rows(plane, "embedding")?;
                    h_emb = Some(rows[last].clone());
                }
                PlaneEvent::Layer { index, trace } => {
                    let pa = plane_rows(&trace.post_attention, "post_attention")?;
                    let pl = plane_rows(&trace.post_layer, "post_layer")?;
                    post_attn[index] = Some(pa[last].clone());
                    post_layer[index] = Some(pl[last].clone());
                }
                _ => {}
            }
            Ok(())
        };

        let continuation = v3_continuation(runtime)?;
        let state = requires_continuation(plan).then(|| continuation.build());
        let (observed, tap_out) = execute_prepared_position_exits_with_taps(
            plan,
            &ops,
            &prompt_ids,
            runtime.backend(),
            state,
            &[last],
            &taps,
            &mut sink,
        )
        .map_err(|e| LqlError::exec("DLA forward pass failed", e))?;

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
            let act = &down_in_rows[last];

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
                u_ffn = linearize_norm(&source, post_norm, &down_out_rows[last], &u_ffn)
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
            let head_out = &head_out_rows[last];
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

        Ok(DlaReport {
            prompt_ids,
            target_a: target_a.to_string(),
            token_id_a,
            logit_a,
            target_b: target_b.to_string(),
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
            executes_patched,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub(crate) fn exec_dla(
        &self,
        prompt: &str,
        target_a: &str,
        target_b: &str,
        layers: Option<&CaptureLayers>,
        top_columns: Option<u32>,
        top_heads: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let report = self.dla(prompt, target_a, target_b, layers, top_columns, top_heads)?;
        let mode = if report.executes_patched {
            "patched"
        } else {
            "stored"
        };
        let mut out = vec![
            format!(
                "Direct Logit Attribution ({} prompt tokens, executes: {mode}):",
                report.prompt_ids.len()
            ),
            format!(
                "  Target A: {:?} [id {}]  logit = {:+.4}",
                report.target_a, report.token_id_a, report.logit_a
            ),
            format!(
                "  Target B: {:?} [id {}]  logit = {:+.4}",
                report.target_b, report.token_id_b, report.logit_b
            ),
            format!(
                "  Logit diff (A - B): {:+.4}   (DLA sum: {:+.4}, residual: {:.2e})",
                report.exact_logit_diff,
                report.total_dla,
                (report.total_dla - report.pre_softcap_diff).abs()
            ),
        ];
        if let Some(cap) = report.softcapping {
            out.push(format!(
                "  Note: output softcapping = {cap}; DLA decomposes pre-softcap diff ({:+.4}).",
                report.pre_softcap_diff
            ));
        }
        out.push(String::new());
        out.push(format!(
            "  {:<6} {:<12} {:>10} {:>10} {:>10} {:>10}",
            "Layer", "Mixer", "Op DLA", "FFN DLA", "Layer DLA", "Cum DLA"
        ));
        out.push(format!("  {}", "-".repeat(62)));
        out.push(format!(
            "  {:<6} {:<12} {:>+10.4} {:>10} {:>+10.4} {:>+10.4}",
            "emb", "-", report.embedding_dla, "-", report.embedding_dla, report.embedding_dla
        ));
        for row in &report.layers {
            out.push(format!(
                "  L{:<5} {:<12} {:>+10.4} {:>+10.4} {:>+10.4} {:>+10.4}",
                row.layer,
                row.mixer,
                row.operator_dla,
                row.ffn_dla,
                row.layer_dla,
                row.cumulative_dla
            ));
        }

        if !report.top_columns.is_empty() {
            out.push(String::new());
            out.push(format!(
                "Top {} FFN columns (across {} dense FFN layer(s)):",
                report.top_columns.len(),
                report.dense_ffn_layers_attributed
            ));
            out.push(format!(
                "  {:>4}  {:<6} {:>6} {:>10} {:>12} {:>10}",
                "Rank", "Layer", "Col", "Act", "Proj (u·w)", "Col DLA"
            ));
            out.push(format!("  {}", "-".repeat(54)));
            for (i, col) in report.top_columns.iter().enumerate() {
                out.push(format!(
                    "  {:>4}  L{:<5} {:>6} {:>+10.4} {:>+12.4} {:>+10.4}",
                    i + 1,
                    col.layer,
                    col.column,
                    col.activation,
                    col.projection,
                    col.contribution
                ));
            }
        }

        if !report.top_heads.is_empty() {
            out.push(String::new());
            out.push(format!(
                "Top {} attention heads (across {} softmax layer(s)):",
                report.top_heads.len(),
                report.softmax_layers_attributed
            ));
            out.push(format!(
                "  {:>4}  {:<6} {:<6} {:>10} {:>10}",
                "Rank", "Layer", "Head", "||h_out||", "Head DLA"
            ));
            out.push(format!("  {}", "-".repeat(42)));
            for (i, h) in report.top_heads.iter().enumerate() {
                out.push(format!(
                    "  {:>4}  L{:<5} H{:02}    {:>10.4} {:>+10.4}",
                    i + 1,
                    h.layer,
                    h.head,
                    h.head_norm,
                    h.contribution
                ));
            }
        }

        out.push(format!("  {:.0}ms", report.elapsed_ms));
        Ok(out)
    }
}
