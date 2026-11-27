//! Execution of counterfactual `PATCH HEAD / COLUMN / NEURON / CHANNEL AT LAYER <l> SCALE <s>;`.

use larql_vindex::format::vindex3::opplan::{LayerAttention, LayerFfn};
use larql_vindex::format::vindex3::weight_patch::structured::ScalarBits;
use larql_vindex::format::vindex3::weight_patch::StructuredWeightOp;

use crate::ast::PatchUnitTarget;
use crate::error::LqlError;
use crate::executor::{Backend, Session};

const NUMERICS: &str = "f64-seq-rne1/v1";

fn refused_patch(e: impl std::fmt::Display) -> LqlError {
    LqlError::Execution(format!("PATCH refused: {e}"))
}

impl Session {
    /// Stage a counterfactual column-scaling/ablation op on one layer's attention head(s),
    /// dense FFN neuron(s)/column(s), or ShortConv channel(s).
    pub(crate) fn exec_patch_unit(
        &mut self,
        target: &PatchUnitTarget,
        layer: usize,
        scale: f64,
    ) -> Result<Vec<String>, LqlError> {
        if !scale.is_finite() {
            return Err(refused_patch(format!("SCALE must be finite, got {scale}")));
        }
        let (path, tensor, columns, unit_label) = {
            let Backend::Vindex3 { path, runtime, .. } = &self.backend else {
                return match &self.backend {
                    Backend::None => Err(LqlError::NoBackend),
                    _ => Err(refused_patch(
                        "PATCH requires a bound canonical VINDEX3 container",
                    )),
                };
            };
            let plan = runtime.plan();
            let layer_plan = plan.layers.get(layer).ok_or_else(|| {
                refused_patch(format!(
                    "layer {layer} is outside the program's 0..{} layers",
                    plan.layers.len()
                ))
            })?;
            let (tensor, columns, unit_label) = match target {
                PatchUnitTarget::Heads(heads) => {
                    let attn = layer_plan.attention.softmax().ok_or_else(|| {
                        refused_patch(format!(
                            "layer {layer} is not a softmax attention layer (cannot PATCH HEAD)"
                        ))
                    })?;
                    let sorted = validate_unique_indices(heads, attn.num_q_heads, "head")?;
                    let mut cols = Vec::with_capacity(sorted.len() * attn.head_dim);
                    for h in &sorted {
                        cols.extend(h * attn.head_dim..(h + 1) * attn.head_dim);
                    }
                    (attn.o.tensor.clone(), cols, format!("head(s) {sorted:?}"))
                }
                PatchUnitTarget::Columns(cols_in) => {
                    let ffn = layer_plan
                        .ffn
                        .as_ref()
                        .and_then(LayerFfn::dense)
                        .ok_or_else(|| {
                            refused_patch(format!(
                                "layer {layer} does not have a dense FFN (cannot PATCH COLUMN/NEURON)"
                            ))
                        })?;
                    let sorted =
                        validate_unique_indices(cols_in, ffn.intermediate_size, "column/neuron")?;
                    let label = format!("column(s)/neuron(s) {sorted:?}");
                    (ffn.down.tensor.clone(), sorted, label)
                }
                PatchUnitTarget::Channels(chans_in) => {
                    let LayerAttention::ShortConv(sc) = &layer_plan.attention else {
                        return Err(refused_patch(format!(
                            "layer {layer} is not a ShortConv layer (cannot PATCH CHANNEL)"
                        )));
                    };
                    let width = sc.out_proj.shape[1];
                    let sorted = validate_unique_indices(chans_in, width, "channel")?;
                    let label = format!("ShortConv channel(s) {sorted:?}");
                    (sc.out_proj.tensor.clone(), sorted, label)
                }
            };
            (path.clone(), tensor, columns, unit_label)
        };

        let op = StructuredWeightOp::bind_scale_columns(
            &path,
            &tensor,
            NUMERICS,
            ScalarBits::F64(scale.to_bits()),
            columns.clone(),
        )
        .map_err(refused_patch)?;
        let resolved = op.resolve(&path).map_err(refused_patch)?;
        let digest = op.digest.clone();

        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("matched above")
        };
        overlay
            .stage_structured_op(op.clone())
            .map_err(refused_patch)?;
        let pending = overlay.structured_op_count();
        if let Some(recording) = &mut self.patch_recording {
            recording
                .operations
                .push(larql_vindex::PatchOp::StructuredWeight(op));
        }

        let mut out = vec![format!(
            "Staged counterfactual PATCH on layer {layer} {unit_label} ({tensor}, {} column(s), scale {scale}): {} element(s) changed",
            columns.len(),
            resolved.edits.len()
        )];
        out.push(format!("  op digest {}", &digest[..16]));
        out.push(format!(
            "  {pending} structured op(s) pending (use DISCARD WEIGHT EDITS; to restore in 0ms)"
        ));
        out.extend(self.refresh_live_weights());
        Ok(out)
    }
}

fn validate_unique_indices(
    raw: &[usize],
    limit: usize,
    label: &str,
) -> Result<Vec<usize>, LqlError> {
    if raw.is_empty() {
        return Err(refused_patch(format!("{label} list must not be empty")));
    }
    let mut sorted = raw.to_vec();
    sorted.sort_unstable();
    if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
        return Err(refused_patch(format!(
            "{label} {} is listed more than once",
            w[0]
        )));
    }
    if let Some(&bad) = sorted.iter().find(|&&i| i >= limit) {
        return Err(refused_patch(format!(
            "{label} {bad} is outside 0..{limit}"
        )));
    }
    Ok(sorted)
}
