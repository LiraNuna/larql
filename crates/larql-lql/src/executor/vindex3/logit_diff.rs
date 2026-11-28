//! Fast forward-only `LOGIT DIFF` and `DIFF LOGIT DIFF` on VINDEX3 containers.
//!
//! Runs a single zero-tap forward pass stopping at the final-normed hidden state
//! (`prefill_prepared_hidden`), skipping all per-layer taps, weight dequantization, and the
//! full 128K-row vocabulary projection — evaluating ONLY the 2 requested target rows via
//! `PreparedOperands::head_over_normed_for_tokens` with bit-exact parity against `DLA` and `INFER`.

use larql_vindex::format::vindex3::opplan::exec::operands::OperandSource;
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::opplan::exec::{
    prefill_prepared_embeds, prefill_prepared_hidden, FinalState,
};
use serde::Serialize;

use super::dla_math::{dot_f64, load_two_rows, resolve_dla_target};
use super::{
    encode_raw_prompt, prefix_cache, prepared_operands_for_base, prepared_operands_for_session,
    v3_continuation, PrefixCacheKey, PreparedCache, PreparedCacheKey, SoftPromptState, V3Runtime,
};
use crate::ast::DlaTarget;
use crate::error::LqlError;
use crate::executor::Session;

/// Structured result of `LOGIT DIFF "<prompt>" FOR <A> VS <B>;`.
#[derive(Debug, Clone, Serialize)]
pub struct LogitDiffReport {
    pub prompt_ids: Vec<u32>,
    pub target_a: String,
    pub token_id_a: u32,
    pub logit_a: f32,
    pub target_b: String,
    pub token_id_b: u32,
    pub logit_b: f32,
    pub logit_diff: f32,
    pub executes_patched: bool,
    pub has_soft_prompt: bool,
    pub elapsed_ms: f64,
}

/// Structured result of `DIFF LOGIT DIFF "<prompt>" FOR <A> VS <B>;`.
#[derive(Debug, Clone, Serialize)]
pub struct DiffLogitDiffReport {
    pub prompt_ids: Vec<u32>,
    pub target_a: String,
    pub token_id_a: u32,
    pub target_b: String,
    pub token_id_b: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_target_a: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_token_id_a: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_target_b: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_token_id_b: Option<u32>,
    pub base_logit_a: f32,
    pub base_logit_b: f32,
    pub base_logit_diff: f32,
    pub candidate_logit_a: f32,
    pub candidate_logit_b: f32,
    pub candidate_logit_diff: f32,
    pub delta_logit_diff: f32,
    pub candidate_has_weight_edits: bool,
    pub candidate_has_soft_prompt: bool,
    pub base_cached: bool,
    pub elapsed_ms: f64,
}

enum ContrastivePlan {
    Fixed {
        id_a: u32,
        label_a: String,
        id_b: u32,
        label_b: String,
    },
    MaxOtherVsFixed {
        excluded: std::collections::HashSet<u32>,
        id_b: u32,
        label_b: String,
    },
    FixedVsMaxOther {
        id_a: u32,
        label_a: String,
        excluded: std::collections::HashSet<u32>,
    },
}

impl ContrastivePlan {
    fn uses_max_other(&self) -> bool {
        !matches!(self, Self::Fixed { .. })
    }
}

struct EvaluatedPair {
    id_a: u32,
    label_a: String,
    logit_a: f32,
    id_b: u32,
    label_b: String,
    logit_b: f32,
}

fn plan_contrastive_targets(
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    target_a: &DlaTarget,
    target_b: &DlaTarget,
    vocab_size: usize,
) -> Result<ContrastivePlan, LqlError> {
    let build_excluded =
        |fixed_id: u32, except: &[DlaTarget]| -> Result<std::collections::HashSet<u32>, LqlError> {
            let mut set = std::collections::HashSet::with_capacity(except.len() + 1);
            set.insert(fixed_id);
            for item in except {
                let (eid, _) = resolve_dla_target(tokenizer, item, "EXCEPT", vocab_size)?;
                set.insert(eid);
            }
            Ok(set)
        };
    match (target_a, target_b) {
        (DlaTarget::MaxOther { .. }, DlaTarget::MaxOther { .. }) => Err(LqlError::Execution(
            "LOGIT DIFF cannot use MAX_OTHER on both FOR and VS sides".into(),
        )),
        (DlaTarget::MaxOther { except }, fixed_b) => {
            let (id_b, label_b) = resolve_dla_target(tokenizer, fixed_b, "VS", vocab_size)?;
            let excluded = build_excluded(id_b, except)?;
            Ok(ContrastivePlan::MaxOtherVsFixed {
                excluded,
                id_b,
                label_b,
            })
        }
        (fixed_a, DlaTarget::MaxOther { except }) => {
            let (id_a, label_a) = resolve_dla_target(tokenizer, fixed_a, "FOR", vocab_size)?;
            let excluded = build_excluded(id_a, except)?;
            Ok(ContrastivePlan::FixedVsMaxOther {
                id_a,
                label_a,
                excluded,
            })
        }
        (fixed_a, fixed_b) => {
            let (id_a, label_a) = resolve_dla_target(tokenizer, fixed_a, "FOR", vocab_size)?;
            let (id_b, label_b) = resolve_dla_target(tokenizer, fixed_b, "VS", vocab_size)?;
            Ok(ContrastivePlan::Fixed {
                id_a,
                label_a,
                id_b,
                label_b,
            })
        }
    }
}

fn evaluate_contrastive_pair(
    runtime: &V3Runtime,
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    ops: &PreparedOperands,
    source: &OperandSource<'_>,
    final_hidden: &[f32],
    plan: &ContrastivePlan,
) -> Result<EvaluatedPair, LqlError> {
    match plan {
        ContrastivePlan::Fixed {
            id_a,
            label_a,
            id_b,
            label_b,
        } => {
            let (logit_a, logit_b) =
                project_two_logits(runtime, ops, source, final_hidden, *id_a, *id_b)?;
            Ok(EvaluatedPair {
                id_a: *id_a,
                label_a: label_a.clone(),
                logit_a,
                id_b: *id_b,
                label_b: label_b.clone(),
                logit_b,
            })
        }
        ContrastivePlan::MaxOtherVsFixed {
            excluded,
            id_b,
            label_b,
        } => {
            let logits = ops
                .head_over_normed(runtime.backend(), final_hidden)
                .map_err(|e| LqlError::exec("LOGIT DIFF MAX_OTHER output head failed", e))?
                .ok_or_else(|| {
                    LqlError::Execution("the component carries no output head".into())
                })?;
            let (best_id, best_logit) = argmax_excluding(&logits, excluded)?;
            let best_label = tokenizer
                .decode(&[best_id], false)
                .unwrap_or_else(|_| format!("<id:{best_id}>"));
            let logit_b = logits[*id_b as usize];
            Ok(EvaluatedPair {
                id_a: best_id,
                label_a: best_label,
                logit_a: best_logit,
                id_b: *id_b,
                label_b: label_b.clone(),
                logit_b,
            })
        }
        ContrastivePlan::FixedVsMaxOther {
            id_a,
            label_a,
            excluded,
        } => {
            let logits = ops
                .head_over_normed(runtime.backend(), final_hidden)
                .map_err(|e| LqlError::exec("LOGIT DIFF MAX_OTHER output head failed", e))?
                .ok_or_else(|| {
                    LqlError::Execution("the component carries no output head".into())
                })?;
            let (best_id, best_logit) = argmax_excluding(&logits, excluded)?;
            let best_label = tokenizer
                .decode(&[best_id], false)
                .unwrap_or_else(|_| format!("<id:{best_id}>"));
            let logit_a = logits[*id_a as usize];
            Ok(EvaluatedPair {
                id_a: *id_a,
                label_a: label_a.clone(),
                logit_a,
                id_b: best_id,
                label_b: best_label,
                logit_b: best_logit,
            })
        }
    }
}

fn argmax_excluding(
    logits: &[f32],
    excluded: &std::collections::HashSet<u32>,
) -> Result<(u32, f32), LqlError> {
    let mut best: Option<(u32, f32)> = None;
    for (idx, &val) in logits.iter().enumerate() {
        let id = idx as u32;
        if excluded.contains(&id) || !val.is_finite() {
            continue;
        }
        if best.is_none_or(|(_, bval)| val > bval) {
            best = Some((id, val));
        }
    }
    best.ok_or_else(|| {
        LqlError::Execution("MAX_OTHER: no eligible vocabulary token left after exclusions".into())
    })
}

impl Session {
    /// Fast forward-only `logit(target_a) - logit(target_b)` at the prompt's final position.
    pub fn logit_diff(
        &self,
        prompt: &str,
        target_a: impl Into<DlaTarget>,
        target_b: impl Into<DlaTarget>,
    ) -> Result<LogitDiffReport, LqlError> {
        let target_a = target_a.into();
        let target_b = target_b.into();
        let (runtime, tokenizer, overlay, bos_token, prepared_cache) =
            self.v3_text_program("LOGIT DIFF")?;
        let output_op =
            runtime.plan().output.as_ref().ok_or_else(|| {
                LqlError::Execution("the component carries no output head".into())
            })?;
        let vocab_size = output_op.projection.shape[0];
        let prompt_ids = encode_raw_prompt(tokenizer, prompt, bos_token)?;
        if prompt_ids.is_empty() {
            return Err(LqlError::Execution(
                "LOGIT DIFF: prompt encoded to zero tokens".into(),
            ));
        }
        let plan = plan_contrastive_targets(tokenizer, &target_a, &target_b, vocab_size)?;

        let start = std::time::Instant::now();
        let (ops, overrides, cache_key) =
            prepared_operands_for_session(runtime, overlay, prepared_cache)?;
        let soft_prompt = prepared_cache.soft_prompt();
        let executes_patched = overrides.is_some();
        let has_soft_prompt = soft_prompt.is_some();

        let final_hidden = if !executes_patched && !has_soft_prompt {
            if let Some(cached) = prepared_cache.get_base_hidden(&prompt_ids) {
                cached
            } else {
                let h = forward_final_hidden(
                    runtime,
                    &ops,
                    prepared_cache,
                    cache_key.as_ref(),
                    None,
                    &prompt_ids,
                )?;
                prepared_cache.put_base_hidden(prompt_ids.clone(), h.clone());
                h
            }
        } else {
            forward_final_hidden(
                runtime,
                &ops,
                prepared_cache,
                cache_key.as_ref(),
                soft_prompt.as_deref(),
                &prompt_ids,
            )?
        };
        let source = match &overrides {
            Some(ov) => OperandSource::overlaid(runtime.store(), ov),
            None => runtime.operands(),
        };
        let eval =
            evaluate_contrastive_pair(runtime, tokenizer, &ops, &source, &final_hidden, &plan)?;

        Ok(LogitDiffReport {
            prompt_ids,
            target_a: eval.label_a,
            token_id_a: eval.id_a,
            logit_a: eval.logit_a,
            target_b: eval.label_b,
            token_id_b: eval.id_b,
            logit_b: eval.logit_b,
            logit_diff: eval.logit_a - eval.logit_b,
            executes_patched,
            has_soft_prompt,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    /// Fast forward-only `DIFF LOGIT DIFF` comparing the session's active state against the
    /// untouched Base container (using resident `cache.base` and skipping full-vocab GEMV for
    /// concrete token pairs, or resolving `MAX_OTHER [EXCEPT (...)]` over the cached hidden state).
    pub fn diff_logit_diff(
        &self,
        prompt: &str,
        target_a: impl Into<DlaTarget>,
        target_b: impl Into<DlaTarget>,
    ) -> Result<DiffLogitDiffReport, LqlError> {
        let target_a = target_a.into();
        let target_b = target_b.into();
        let (runtime, tokenizer, overlay, bos_token, prepared_cache) =
            self.v3_text_program("DIFF LOGIT DIFF")?;
        let output_op =
            runtime.plan().output.as_ref().ok_or_else(|| {
                LqlError::Execution("the component carries no output head".into())
            })?;
        let vocab_size = output_op.projection.shape[0];
        let prompt_ids = encode_raw_prompt(tokenizer, prompt, bos_token)?;
        if prompt_ids.is_empty() {
            return Err(LqlError::Execution(
                "DIFF LOGIT DIFF: prompt encoded to zero tokens".into(),
            ));
        }
        let plan = plan_contrastive_targets(tokenizer, &target_a, &target_b, vocab_size)?;

        let start = std::time::Instant::now();
        let (base_ops, base_key) = prepared_operands_for_base(runtime, prepared_cache)?;
        let (base_hidden, base_cached) =
            if let Some(cached) = prepared_cache.get_base_hidden(&prompt_ids) {
                (cached, true)
            } else {
                let h = forward_final_hidden(
                    runtime,
                    &base_ops,
                    prepared_cache,
                    Some(&base_key),
                    None,
                    &prompt_ids,
                )?;
                prepared_cache.put_base_hidden(prompt_ids.clone(), h.clone());
                (h, false)
            };
        let base_eval = evaluate_contrastive_pair(
            runtime,
            tokenizer,
            &base_ops,
            &runtime.operands(),
            &base_hidden,
            &plan,
        )?;
        let base_logit_diff = base_eval.logit_a - base_eval.logit_b;

        let (cand_ops, cand_overrides, cand_key) =
            prepared_operands_for_session(runtime, overlay, prepared_cache)?;
        let soft_prompt = prepared_cache.soft_prompt();
        let candidate_has_weight_edits = cand_overrides.is_some();
        let candidate_has_soft_prompt = soft_prompt.is_some();

        let cand_eval = if !candidate_has_weight_edits && !candidate_has_soft_prompt {
            EvaluatedPair {
                id_a: base_eval.id_a,
                label_a: base_eval.label_a.clone(),
                logit_a: base_eval.logit_a,
                id_b: base_eval.id_b,
                label_b: base_eval.label_b.clone(),
                logit_b: base_eval.logit_b,
            }
        } else {
            let cand_hidden = forward_final_hidden(
                runtime,
                &cand_ops,
                prepared_cache,
                cand_key.as_ref(),
                soft_prompt.as_deref(),
                &prompt_ids,
            )?;
            let cand_source = match &cand_overrides {
                Some(ov) => OperandSource::overlaid(runtime.store(), ov),
                None => runtime.operands(),
            };
            evaluate_contrastive_pair(
                runtime,
                tokenizer,
                &cand_ops,
                &cand_source,
                &cand_hidden,
                &plan,
            )?
        };
        let candidate_logit_diff = cand_eval.logit_a - cand_eval.logit_b;
        let show_base_targets = plan.uses_max_other();

        Ok(DiffLogitDiffReport {
            prompt_ids,
            target_a: cand_eval.label_a,
            token_id_a: cand_eval.id_a,
            target_b: cand_eval.label_b,
            token_id_b: cand_eval.id_b,
            base_target_a: show_base_targets.then_some(base_eval.label_a),
            base_token_id_a: show_base_targets.then_some(base_eval.id_a),
            base_target_b: show_base_targets.then_some(base_eval.label_b),
            base_token_id_b: show_base_targets.then_some(base_eval.id_b),
            base_logit_a: base_eval.logit_a,
            base_logit_b: base_eval.logit_b,
            base_logit_diff,
            candidate_logit_a: cand_eval.logit_a,
            candidate_logit_b: cand_eval.logit_b,
            candidate_logit_diff,
            delta_logit_diff: candidate_logit_diff - base_logit_diff,
            candidate_has_weight_edits,
            candidate_has_soft_prompt,
            base_cached,
            elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }

    pub(crate) fn exec_logit_diff(
        &self,
        prompt: &str,
        target_a: &DlaTarget,
        target_b: &DlaTarget,
    ) -> Result<Vec<String>, LqlError> {
        let r = self.logit_diff(prompt, target_a.clone(), target_b.clone())?;
        let mode = if r.executes_patched {
            "patched"
        } else {
            "stored"
        };
        Ok(vec![
            format!(
                "Logit Diff ({} prompt tokens, executes: {mode}, soft_prompt: {}):",
                r.prompt_ids.len(),
                r.has_soft_prompt
            ),
            format!(
                "  Target A: {:?} [id {}]  logit = {:+.4}",
                r.target_a, r.token_id_a, r.logit_a
            ),
            format!(
                "  Target B: {:?} [id {}]  logit = {:+.4}",
                r.target_b, r.token_id_b, r.logit_b
            ),
            format!("  Logit diff (A - B): {:+.4}", r.logit_diff),
            format!("  {:.0}ms", r.elapsed_ms),
        ])
    }

    pub(crate) fn exec_diff_logit_diff(
        &self,
        prompt: &str,
        target_a: &DlaTarget,
        target_b: &DlaTarget,
    ) -> Result<Vec<String>, LqlError> {
        let r = self.diff_logit_diff(prompt, target_a.clone(), target_b.clone())?;
        Ok(vec![
            format!(
                "DIFF LOGIT DIFF ({} prompt tokens; weights_edited={} soft_prompt={} base_cached={}):",
                r.prompt_ids.len(),
                r.candidate_has_weight_edits,
                r.candidate_has_soft_prompt,
                r.base_cached
            ),
            format!(
                "  Target A: {:?} [id {}]  vs  Target B: {:?} [id {}]",
                r.target_a, r.token_id_a, r.target_b, r.token_id_b
            ),
            format!(
                "  Logit diff: base {:+.4} -> cand {:+.4}  (delta {:+.4})",
                r.base_logit_diff, r.candidate_logit_diff, r.delta_logit_diff
            ),
            format!("  {:.0}ms", r.elapsed_ms),
        ])
    }
}

pub(super) fn forward_final_hidden(
    runtime: &V3Runtime,
    ops: &PreparedOperands,
    prepared_cache: &PreparedCache,
    weight_key: Option<&PreparedCacheKey>,
    soft_prompt: Option<&SoftPromptState>,
    prompt_ids: &[u32],
) -> Result<Vec<f32>, LqlError> {
    let plan = runtime.plan();
    if let Some(wkey) = weight_key.filter(|_| prefix_cache::plan_supports_prefix_cache(runtime)) {
        let pkey = PrefixCacheKey {
            weights: wkey.clone(),
            soft_prompt_sha256: soft_prompt.map(|sp| sp.sha256.clone()),
        };
        if let Some(hit) = prepared_cache.prefix().find_exact_hidden(&pkey, prompt_ids) {
            return Ok(hit);
        }
        let (mut kv, prefix_len) = prefix_cache::acquire_prefix_state(
            runtime,
            ops,
            prepared_cache.prefix(),
            &pkey,
            soft_prompt,
            prompt_ids,
            prompt_ids.len().saturating_sub(1),
        )?;
        let out = prefill_prepared_hidden(
            plan,
            ops,
            &prompt_ids[prefix_len..],
            runtime.backend(),
            &mut kv,
        )
        .map_err(|e| LqlError::exec("LOGIT DIFF forward pass failed", e))?;
        let hidden = extract_final_hidden(out.exit)?;
        prepared_cache.prefix().record_snapshot_hidden(
            &pkey,
            prompt_ids.to_vec(),
            kv,
            hidden.clone(),
        );
        return Ok(hidden);
    }

    let mut kv = v3_continuation(runtime)?.build();
    if let Some(sp) = soft_prompt {
        prefill_prepared_embeds(plan, ops, &sp.embeds, runtime.backend(), &mut *kv)
            .map_err(|e| LqlError::exec("soft-prompt prefix prefill failed", e))?;
    }
    let out = prefill_prepared_hidden(plan, ops, prompt_ids, runtime.backend(), &mut *kv)
        .map_err(|e| LqlError::exec("LOGIT DIFF forward pass failed", e))?;
    extract_final_hidden(out.exit)
}

fn extract_final_hidden(state: FinalState) -> Result<Vec<f32>, LqlError> {
    match state {
        FinalState::Hidden(h) => Ok(h),
        other => Err(LqlError::Execution(format!(
            "LOGIT DIFF requires a single-stream hidden exit, got {other:?}"
        ))),
    }
}

pub(super) fn project_two_logits(
    runtime: &V3Runtime,
    ops: &PreparedOperands,
    source: &OperandSource<'_>,
    final_hidden: &[f32],
    id_a: u32,
    id_b: u32,
) -> Result<(f32, f32), LqlError> {
    if let Ok(Some(pair)) =
        ops.head_over_normed_for_tokens(runtime.backend(), final_hidden, &[id_a, id_b])
    {
        if pair.len() == 2 {
            return Ok((pair[0], pair[1]));
        }
    }
    let output_op = runtime
        .plan()
        .output
        .as_ref()
        .ok_or_else(|| LqlError::Execution("the component carries no output head".into()))?;
    let vocab = output_op.projection.shape[0];
    let hidden = output_op.projection.shape[1];
    let (row_a, row_b) = load_two_rows(
        source,
        &output_op.projection,
        id_a as usize,
        id_b as usize,
        vocab,
        hidden,
    )
    .map_err(|e| LqlError::exec("LOGIT DIFF failed to read output head rows", e))?;
    let mult = output_op.multiplier.unwrap_or(1.0);
    let apply = |raw: f64| -> f32 {
        let scaled = (raw * mult) as f32;
        match output_op.softcapping {
            Some(cap) if cap > 0.0 => cap * (scaled / cap).tanh(),
            _ => scaled,
        }
    };
    Ok((
        apply(dot_f64(&row_a, final_hidden)),
        apply(dot_f64(&row_b, final_hidden)),
    ))
}
