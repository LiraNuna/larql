//! Exact sandwich-norm-aware Rank-1 Nullspace Projection (`PROJECT OUT <tokA> - <tokB> ON LAYERS (...);`).
//!
//! Computes the unembedding logit difference direction `u_res = final_norm_weight * (W_out[tokA] - W_out[tokB])`,
//! folds each targeted sublayer's sandwich `post_ffn_norm` / `post_attention_norm` weight
//! (`u_y = gamma_post * u_res`, automatically preserving zero-gamma RMS ballast dims with zero blowup!),
//! normalizes `u_hat = u_y / ||u_y||_2`, and stages an exact rank-1 `LowRankColumnsOp`
//! (`W <- W + (-scale * u_hat) * (u_hat^T W_eff)`) recorded as a `weight-patches/v4` operation target
//! (no `MAX_PATCH_ELEMENTS` cap!) that composes directly over `OVERLAY SAFETENSORS` /
//! `OVERLAY LORA` and exports cleanly via `EXPORT SAFETENSORS`.

use larql_models::config::NormType;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandSource;
use larql_vindex::format::vindex3::opplan::{LayerAttention, LayerFfn, NormOp, OperandRef};
use larql_vindex::format::vindex3::weight_patch::structured::OperandBits;
use larql_vindex::format::vindex3::weight_patch::LowRankColumnsOp;
use serde::Serialize;

use super::dla_math::{load_two_rows, resolve_dla_layers, resolve_dla_target};
use super::execution_overrides;
use crate::ast::{CaptureLayers, DlaTarget, ProjectOutTarget};
use crate::error::LqlError;
use crate::executor::{Backend, Session};

const NUMERICS: &str = "f64-seq-rne1/v1";

fn refused_project(e: impl std::fmt::Display) -> LqlError {
    LqlError::Execution(format!("PROJECT OUT refused: {e}"))
}

/// Structured summary returned by `PROJECT OUT` and `PROJECT OUT DELTA`.
#[derive(Debug, Clone, Serialize)]
pub struct ProjectOutSummary {
    pub delta_only: bool,
    pub target_a: String,
    pub token_id_a: u32,
    pub target_b: String,
    pub token_id_b: u32,
    pub scale: f64,
    pub target_scope: &'static str,
    pub projected_tensors: Vec<String>,
    pub skipped_unchanged_tensors: Vec<String>,
    pub total_changed_elements: usize,
}

impl Session {
    /// Typed operation behind `PROJECT OUT [DELTA] <A> - <B> ON LAYERS (...) [TARGET ...] [SCALE <f>];`.
    pub fn project_out(
        &mut self,
        delta_only: bool,
        target_a: &DlaTarget,
        target_b: &DlaTarget,
        layers: &CaptureLayers,
        target: ProjectOutTarget,
        scale: f64,
    ) -> Result<ProjectOutSummary, LqlError> {
        if !scale.is_finite() || scale == 0.0 {
            return Err(refused_project(format!(
                "SCALE must be finite and non-zero, got {scale}"
            )));
        }

        // Gather all plan geometry, unembedding direction, and per-tensor rank-1 factors
        // under an immutable borrow before staging ops on `self.backend`.
        struct PlannedProjection {
            tensor: String,
            cols: usize,
            a_bits: Vec<u64>,
            b_bits: Vec<u64>,
        }

        let (path, label_a, id_a, label_b, id_b, planned, skipped_unchanged_tensors) = {
            let (path, runtime, tokenizer, overlay) = match &self.backend {
                Backend::Vindex3 {
                    path,
                    runtime,
                    tokenizer,
                    overlay,
                    ..
                } => (path, runtime, tokenizer, overlay),
                Backend::None => return Err(LqlError::NoBackend),
                _ => {
                    return Err(refused_project(
                        "PROJECT OUT requires a bound canonical VINDEX3 container",
                    ));
                }
            };
            let tokenizer = tokenizer.as_ref().ok_or_else(|| {
                refused_project(
                    "PROJECT OUT needs a tokenizer and this container carries no tokenizer.json",
                )
            })?;
            let plan = runtime.plan();
            if !plan.residual_topology.is_single_stream() {
                return Err(refused_project(format!(
                    "PROJECT OUT requires a single-stream residual topology, got {:?}",
                    plan.residual_topology
                )));
            }
            let output_op = plan
                .output
                .as_ref()
                .ok_or_else(|| refused_project("the component carries no output head"))?;
            let vocab_size = output_op.projection.shape[0];
            let hidden = output_op.projection.shape[1];

            let (id_a, label_a) =
                resolve_dla_target(tokenizer, target_a, "PROJECT OUT", vocab_size)
                    .map_err(refused_project)?;
            let (id_b, label_b) =
                resolve_dla_target(tokenizer, target_b, "PROJECT OUT", vocab_size)
                    .map_err(refused_project)?;
            if id_a == id_b {
                return Err(refused_project(format!(
                    "targets {:?} and {:?} resolve to the same token ID {id_a} (zero direction)",
                    label_a, label_b
                )));
            }
            let req_layers =
                resolve_dla_layers(Some(layers), plan.layers.len()).map_err(refused_project)?;

            // Read effective output rows & norms (honoring any mounted OVERLAY SAFETENSORS/LORA).
            let overrides = execution_overrides(runtime, overlay)?;
            if delta_only && overrides.is_none() {
                return Err(refused_project(
                    "PROJECT OUT DELTA requires an active OVERLAY SAFETENSORS or OVERLAY LORA \
                     (ΔW = W_overlay - W_base is zero without an overlay)",
                ));
            }
            let base_source = runtime.operands();
            let source = match &overrides {
                Some(ov) => OperandSource::overlaid(runtime.store(), ov),
                None => base_source,
            };
            let (row_a, row_b) = load_two_rows(
                &source,
                &output_op.projection,
                id_a as usize,
                id_b as usize,
                vocab_size,
                hidden,
            )
            .map_err(|e| refused_project(format!("read output head rows: {e}")))?;

            let mult = output_op.multiplier.unwrap_or(1.0);
            let mut u_res: Vec<f64> = row_a
                .iter()
                .zip(row_b.iter())
                .map(|(&a, &b)| ((a as f64) - (b as f64)) * mult)
                .collect();
            if let Some(final_norm) = &plan.final_norm {
                u_res = fold_norm_direction(&source, final_norm, &u_res)
                    .map_err(|e| refused_project(format!("fold final norm: {e}")))?;
            }

            let want_w2 = matches!(target, ProjectOutTarget::W2 | ProjectOutTarget::All);
            let want_o = matches!(target, ProjectOutTarget::OProj | ProjectOutTarget::All);
            let want_conv = matches!(target, ProjectOutTarget::ConvOut | ProjectOutTarget::All);

            let mut planned = Vec::new();
            let mut skipped_unchanged = Vec::new();
            let mut plan_matrix = |operand: &OperandRef,
                                   norm: Option<&NormOp>,
                                   label: &str|
             -> Result<(), LqlError> {
                let u_y = match norm {
                    Some(n) => fold_norm_direction(&source, n, &u_res)
                        .map_err(|e| refused_project(format!("{label}: {e}")))?,
                    None => u_res.clone(),
                };
                let u_hat = normalize_unit(&u_y)?;
                let base_opt = delta_only.then_some(&base_source);
                let Some((a_bits, b_bits)) =
                    compute_rank1_factors(&source, base_opt, operand, &u_hat, scale)?
                else {
                    skipped_unchanged.push(operand.tensor.clone());
                    return Ok(());
                };
                planned.push(PlannedProjection {
                    tensor: operand.tensor.clone(),
                    cols: operand.shape[1],
                    a_bits,
                    b_bits,
                });
                Ok(())
            };

            for &l in &req_layers {
                let layer_plan = &plan.layers[l];
                if want_w2 {
                    if let Some(ffn) = layer_plan.ffn.as_ref().and_then(LayerFfn::dense) {
                        plan_matrix(
                            &ffn.down,
                            layer_plan.post_ffn_norm.as_ref(),
                            &format!("layer {l} post_ffn_norm"),
                        )?;
                    }
                }
                if want_o {
                    if let Some(attn) = layer_plan.attention.softmax() {
                        plan_matrix(
                            &attn.o,
                            layer_plan.post_attention_norm.as_ref(),
                            &format!("layer {l} post_attention_norm"),
                        )?;
                    }
                }
                if want_conv {
                    if let LayerAttention::ShortConv(sc) = &layer_plan.attention {
                        plan_matrix(
                            &sc.out_proj,
                            layer_plan.post_attention_norm.as_ref(),
                            &format!("layer {l} post_operator_norm"),
                        )?;
                    }
                }
            }

            if planned.is_empty() {
                if delta_only && !skipped_unchanged.is_empty() {
                    return Err(refused_project(format!(
                        "none of the {} targeted tensor(s) had a non-zero delta ΔW = W_overlay - W_base \
                         along the projected direction",
                        skipped_unchanged.len()
                    )));
                }
                return Err(refused_project(
                    "no matching projection matrices found on the requested layer(s)",
                ));
            }
            (
                path.clone(),
                label_a,
                id_a,
                label_b,
                id_b,
                planned,
                skipped_unchanged,
            )
        };

        let mode_tag = if delta_only {
            "project_out_delta"
        } else {
            "project_out"
        };
        let origin = format!("{mode_tag}:{label_a}[{id_a}]-{label_b}[{id_b}]");
        let mut ops_to_stage = Vec::with_capacity(planned.len());
        let mut total_changed_elements = 0usize;
        for item in planned {
            let op = LowRankColumnsOp::bind(
                &path,
                &item.tensor,
                NUMERICS,
                1,
                &OperandBits::F64(item.a_bits),
                &OperandBits::F64(item.b_bits),
                (0..item.cols).collect(),
            )
            .map_err(refused_project)?
            .with_origin(origin.clone());
            let resolved = op.resolve_tensor(&path).map_err(refused_project)?;
            total_changed_elements += resolved.changed_elements;
            ops_to_stage.push((item.tensor, op));
        }

        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("matched above")
        };
        let mut projected_tensors = Vec::with_capacity(ops_to_stage.len());
        for (tensor, op) in ops_to_stage {
            overlay
                .stage_low_rank_columns_op(op.clone())
                .map_err(refused_project)?;
            if let Some(recording) = &mut self.patch_recording {
                recording
                    .operations
                    .push(larql_vindex::PatchOp::LowRankColumnsWeight(op));
            }
            projected_tensors.push(tensor);
        }
        let _ = self.refresh_live_weights();

        Ok(ProjectOutSummary {
            delta_only,
            target_a: label_a,
            token_id_a: id_a,
            target_b: label_b,
            token_id_b: id_b,
            scale,
            target_scope: target.label(),
            projected_tensors,
            skipped_unchanged_tensors,
            total_changed_elements,
        })
    }

    pub(crate) fn exec_project_out(
        &mut self,
        delta_only: bool,
        target_a: &DlaTarget,
        target_b: &DlaTarget,
        layers: &CaptureLayers,
        target: ProjectOutTarget,
        scale: f64,
    ) -> Result<Vec<String>, LqlError> {
        let s = self.project_out(delta_only, target_a, target_b, layers, target, scale)?;
        let mode = if s.delta_only {
            "delta (W_base + P_perp * ΔW)"
        } else {
            "full (P_perp * W)"
        };
        let mut out = vec![
            format!(
                "Projected out rank-1 direction {:?} [id {}] - {:?} [id {}] (mode={mode}, target={}, scale={}) across {} tensor(s):",
                s.target_a,
                s.token_id_a,
                s.target_b,
                s.token_id_b,
                s.target_scope,
                s.scale,
                s.projected_tensors.len()
            ),
            format!(
                "  changed_elements: {} (skipped_unchanged={})",
                s.total_changed_elements,
                s.skipped_unchanged_tensors.len()
            ),
        ];
        for t in s.projected_tensors.iter().take(12) {
            out.push(format!("  projected:        {t}"));
        }
        if s.projected_tensors.len() > 12 {
            out.push(format!(
                "  ... and {} more tensor(s)",
                s.projected_tensors.len() - 12
            ));
        }
        out.push(
            "  active: INFER, LOGIT DIFF, DLA, CAPTURE, and EXPORT SAFETENSORS reflect this projection (DISCARD WEIGHT EDITS; to undo)."
                .into(),
        );
        Ok(out)
    }
}

fn fold_norm_direction(
    source: &OperandSource<'_>,
    norm: &NormOp,
    u: &[f64],
) -> Result<Vec<f64>, larql_vindex::VindexError> {
    let w = source.load(&norm.weight)?;
    let offset = norm.weight_offset as f64;
    let mut out: Vec<f64> = u
        .iter()
        .zip(w.iter())
        .map(|(&ui, &wi)| ui * ((wi as f64) + offset))
        .collect();
    if norm.kind == NormType::LayerNorm && !out.is_empty() {
        let mean = out.iter().sum::<f64>() / (out.len() as f64);
        for v in &mut out {
            *v -= mean;
        }
    }
    Ok(out)
}

fn normalize_unit(u: &[f64]) -> Result<Vec<f64>, LqlError> {
    let norm = u.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return Err(refused_project(
            "folded projection direction has near-zero norm (cannot normalize)",
        ));
    }
    Ok(u.iter().map(|&v| v / norm).collect())
}

type Rank1Factors = (Vec<u64>, Vec<u64>);

/// Compute exact rank-1 factors `(a, b)` such that:
/// - Full mode (`base_source = None`): `W_eff + a * b^T = W_eff - scale * u_hat * (u_hat^T W_eff)`.
/// - Delta mode (`base_source = Some(base)`): `W_eff + a * b^T = W_base + ΔW - scale * u_hat * (u_hat^T ΔW)`
///   where `ΔW = W_eff - W_base`. Returns `Ok(None)` when `u_hat^T ΔW` is identically zero
///   (e.g. a layer untouched by the overlay).
fn compute_rank1_factors(
    source: &OperandSource<'_>,
    base_source: Option<&OperandSource<'_>>,
    operand: &OperandRef,
    u_hat: &[f64],
    scale: f64,
) -> Result<Option<Rank1Factors>, LqlError> {
    if operand.shape.len() != 2 {
        return Err(refused_project(format!(
            "tensor `{}` has rank {}, expected 2D matrix",
            operand.tensor,
            operand.shape.len()
        )));
    }
    let rows = operand.shape[0];
    let cols = operand.shape[1];
    if u_hat.len() != rows {
        return Err(refused_project(format!(
            "tensor `{}` has {rows} rows, expected {}",
            operand.tensor,
            u_hat.len()
        )));
    }
    let w = source
        .load(operand)
        .map_err(|e| refused_project(format!("read `{}`: {e}", operand.tensor)))?;
    let mut b = vec![0.0f64; cols];
    if let Some(base) = base_source {
        let w_base = base
            .load(operand)
            .map_err(|e| refused_project(format!("read base `{}`: {e}", operand.tensor)))?;
        for (r, &ur) in u_hat.iter().enumerate() {
            let eff_row = &w[r * cols..(r + 1) * cols];
            let base_row = &w_base[r * cols..(r + 1) * cols];
            for (c, (&we, &wb)) in eff_row.iter().zip(base_row.iter()).enumerate() {
                b[c] += ur * ((we as f64) - (wb as f64));
            }
        }
        let max_b = b.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
        if max_b < 1e-12 {
            return Ok(None);
        }
    } else {
        for (r, &ur) in u_hat.iter().enumerate() {
            let row = &w[r * cols..(r + 1) * cols];
            for (c, &wrc) in row.iter().enumerate() {
                b[c] += ur * (wrc as f64);
            }
        }
    }
    let a_bits: Vec<u64> = u_hat.iter().map(|&ur| (-scale * ur).to_bits()).collect();
    let b_bits: Vec<u64> = b.iter().map(|&bc| bc.to_bits()).collect();
    Ok(Some((a_bits, b_bits)))
}
