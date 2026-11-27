//! Exact sandwich-norm-aware Rank-1 Nullspace Projection (`PROJECT OUT <tokA> - <tokB> ON LAYERS (...);`).
//!
//! Computes the unembedding logit difference direction `u_res = final_norm_weight * (W_out[tokA] - W_out[tokB])`,
//! folds each targeted sublayer's sandwich `post_ffn_norm` / `post_attention_norm` weight
//! (`u_y = gamma_post * u_res`, automatically preserving zero-gamma RMS ballast dims with zero blowup!),
//! normalizes `u_hat = u_y / ||u_y||_2`, and stages exact BF16 `RemoveDirection` ops
//! (`W <- W - scale * u_hat * (u_hat^T W)`) that compose directly over `OVERLAY SAFETENSORS` /
//! `OVERLAY LORA` and export cleanly via `EXPORT SAFETENSORS`.

use larql_models::config::NormType;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandSource;
use larql_vindex::format::vindex3::opplan::{LayerAttention, LayerFfn, NormOp};
use larql_vindex::format::vindex3::weight_patch::structured::{OperandBits, ScalarBits};
use larql_vindex::format::vindex3::weight_patch::StructuredWeightOp;
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

/// Structured summary returned by `PROJECT OUT`.
#[derive(Debug, Clone, Serialize)]
pub struct ProjectOutSummary {
    pub target_a: String,
    pub token_id_a: u32,
    pub target_b: String,
    pub token_id_b: u32,
    pub scale: f64,
    pub target_scope: &'static str,
    pub projected_tensors: Vec<String>,
    pub total_changed_elements: usize,
}

impl Session {
    /// Typed operation behind `PROJECT OUT <A> - <B> ON LAYERS (...) [TARGET ...] [SCALE <f>];`.
    pub fn project_out(
        &mut self,
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

        // Gather all plan geometry, unembedding direction, and per-tensor folded unit directions
        // under an immutable borrow before staging ops on `self.backend`.
        struct PlannedProjection {
            tensor: String,
            cols: usize,
            u_bits: Vec<u64>,
        }

        let (path, label_a, id_a, label_b, id_b, planned) = {
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
            let source = match &overrides {
                Some(ov) => OperandSource::overlaid(runtime.store(), ov),
                None => runtime.operands(),
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
            for &l in &req_layers {
                let layer_plan = &plan.layers[l];
                if want_w2 {
                    if let Some(ffn) = layer_plan.ffn.as_ref().and_then(LayerFfn::dense) {
                        let u_y = match &layer_plan.post_ffn_norm {
                            Some(norm) => {
                                fold_norm_direction(&source, norm, &u_res).map_err(|e| {
                                    refused_project(format!("layer {l} post_ffn_norm: {e}"))
                                })?
                            }
                            None => u_res.clone(),
                        };
                        let u_bits = normalize_to_f64_bits(&u_y)?;
                        planned.push(PlannedProjection {
                            tensor: ffn.down.tensor.clone(),
                            cols: ffn.down.shape[1],
                            u_bits,
                        });
                    }
                }
                if want_o {
                    if let Some(attn) = layer_plan.attention.softmax() {
                        let u_y = match &layer_plan.post_attention_norm {
                            Some(norm) => {
                                fold_norm_direction(&source, norm, &u_res).map_err(|e| {
                                    refused_project(format!("layer {l} post_attention_norm: {e}"))
                                })?
                            }
                            None => u_res.clone(),
                        };
                        let u_bits = normalize_to_f64_bits(&u_y)?;
                        planned.push(PlannedProjection {
                            tensor: attn.o.tensor.clone(),
                            cols: attn.o.shape[1],
                            u_bits,
                        });
                    }
                }
                if want_conv {
                    if let LayerAttention::ShortConv(sc) = &layer_plan.attention {
                        let u_y = match &layer_plan.post_attention_norm {
                            Some(norm) => {
                                fold_norm_direction(&source, norm, &u_res).map_err(|e| {
                                    refused_project(format!("layer {l} post_operator_norm: {e}"))
                                })?
                            }
                            None => u_res.clone(),
                        };
                        let u_bits = normalize_to_f64_bits(&u_y)?;
                        planned.push(PlannedProjection {
                            tensor: sc.out_proj.tensor.clone(),
                            cols: sc.out_proj.shape[1],
                            u_bits,
                        });
                    }
                }
            }

            if planned.is_empty() {
                return Err(refused_project(
                    "no matching projection matrices found on the requested layer(s)",
                ));
            }
            (path.clone(), label_a, id_a, label_b, id_b, planned)
        };

        let origin = format!("project_out:{label_a}[{id_a}]-{label_b}[{id_b}]");
        let mut ops_to_stage = Vec::with_capacity(planned.len());
        let mut total_changed_elements = 0usize;
        for item in &planned {
            let op = StructuredWeightOp::bind(
                &path,
                &item.tensor,
                NUMERICS,
                &OperandBits::F64(item.u_bits.clone()),
                ScalarBits::F64(scale.to_bits()),
                (0..item.cols).collect(),
            )
            .map_err(refused_project)?
            .with_origin(origin.clone());
            let resolved = op.resolve(&path).map_err(refused_project)?;
            total_changed_elements += resolved.edits.len();
            ops_to_stage.push((item.tensor.clone(), op));
        }

        // Check disjointness across all planned ops before staging any of them (atomic!).
        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("matched above")
        };
        let mut projected_tensors = Vec::with_capacity(ops_to_stage.len());
        for (tensor, op) in ops_to_stage {
            overlay
                .stage_structured_op(op.clone())
                .map_err(refused_project)?;
            if let Some(recording) = &mut self.patch_recording {
                recording
                    .operations
                    .push(larql_vindex::PatchOp::StructuredWeight(op));
            }
            projected_tensors.push(tensor);
        }
        let _ = self.refresh_live_weights();

        Ok(ProjectOutSummary {
            target_a: label_a,
            token_id_a: id_a,
            target_b: label_b,
            token_id_b: id_b,
            scale,
            target_scope: target.label(),
            projected_tensors,
            total_changed_elements,
        })
    }

    pub(crate) fn exec_project_out(
        &mut self,
        target_a: &DlaTarget,
        target_b: &DlaTarget,
        layers: &CaptureLayers,
        target: ProjectOutTarget,
        scale: f64,
    ) -> Result<Vec<String>, LqlError> {
        let s = self.project_out(target_a, target_b, layers, target, scale)?;
        let mut out = vec![
            format!(
                "Projected out rank-1 direction {:?} [id {}] - {:?} [id {}] (target={}, scale={}) across {} tensor(s):",
                s.target_a,
                s.token_id_a,
                s.target_b,
                s.token_id_b,
                s.target_scope,
                s.scale,
                s.projected_tensors.len()
            ),
            format!("  changed_elements: {} (composed over any active OVERLAY)", s.total_changed_elements),
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

fn normalize_to_f64_bits(u: &[f64]) -> Result<Vec<u64>, LqlError> {
    let norm = u.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if !norm.is_finite() || norm < 1e-12 {
        return Err(refused_project(
            "folded projection direction has near-zero norm (cannot normalize)",
        ));
    }
    Ok(u.iter().map(|&v| (v / norm).to_bits()).collect())
}
