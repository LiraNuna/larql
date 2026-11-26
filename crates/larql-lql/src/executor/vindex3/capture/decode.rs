//! The FFN's down projection, which only the decode step exposes: its input as the executing backend fed it
//! (`StepObserver::ffn_down_input`, through [`DownInputCapture`]) and its output before any post-FFN norm
//! (`StepObserver::operand_input` at [`InputSite::FfnOutput`]).
//!
//! The steps are `TRACE`'s: a fresh session over the operands `INFER` resolves, one observed step per token from
//! position 0. The same steps also record the FFN's normed input, which the one-shot pass `INFER` runs exposes
//! too; the two are compared bit for bit, and the sidecar says whether the decode steps computed what that pass
//! computes at the FFN's input.

use std::collections::{BTreeMap, BTreeSet};

use larql_inference::vindex3::{OperandOverrides, SelectedContinuation};
use larql_vindex::format::vindex3::opplan::exec::down_input_capture::DownInputCapture;
use larql_vindex::format::vindex3::opplan::exec::observe::{
    InputSite, ShortConvRecord, StepEvent, StepObserver,
};
use larql_vindex::format::vindex3::opplan::LayerAttention;

use super::super::{confirm_patched_weights_ran, V3Runtime};
use super::{batch, refused, Rows};
use crate::ast::CaptureSite;
use crate::error::LqlError;

/// What the decode steps captured, and how their FFN input compares with INFER's pass.
pub(super) struct DecodeCapture {
    pub blocks: Vec<Rows>,
    pub ffn_input_matches_infer_pass: bool,
}

/// Which internal `ShortConv` signal to copy from [`ShortConvRecord`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShortConvField {
    GateB,
    GateC,
    ConvOut,
    GatedY,
}

impl ShortConvField {
    fn of(site: CaptureSite) -> Option<Self> {
        match site {
            CaptureSite::ShortConvGateB => Some(Self::GateB),
            CaptureSite::ShortConvGateC => Some(Self::GateC),
            CaptureSite::ShortConvConvOut => Some(Self::ConvOut),
            CaptureSite::ShortConvGatedY => Some(Self::GatedY),
            _ => None,
        }
    }
}

/// Records the requested sublayer values at the requested `(layer, position)` pairs.
struct Tap {
    down: Option<DownInputCapture>,
    want_output: bool,
    short_conv: Option<ShortConvField>,
    layers: BTreeSet<usize>,
    positions: BTreeSet<usize>,
    position: Option<usize>,
    outputs: BTreeMap<(usize, usize), Vec<f32>>,
    ffn_inputs: BTreeMap<(usize, usize), Vec<f32>>,
}

impl Tap {
    fn wanted(&self, layer: usize) -> Option<(usize, usize)> {
        let position = self.position?;
        (self.layers.contains(&layer) && self.positions.contains(&position))
            .then_some((layer, position))
    }
}

impl StepObserver for Tap {
    fn event(&mut self, event: StepEvent) {
        if let StepEvent::Embedded { position } = event {
            self.position = Some(position);
        }
        if let Some(down) = &mut self.down {
            down.observe_event(&event);
        }
    }

    fn wants_ffn_down_input(&self, layer: usize) -> bool {
        self.down.as_ref().is_some_and(|d| d.wants_layer(layer))
    }

    fn ffn_down_input(&mut self, layer: usize, values: &[f32]) {
        if let Some(down) = &mut self.down {
            down.observe_down_input(layer, values);
        }
    }

    fn wants_short_conv(&self) -> bool {
        self.short_conv.is_some()
    }

    fn wants_short_conv_at(&self, layer: usize, position: usize) -> bool {
        self.short_conv.is_some()
            && self.layers.contains(&layer)
            && self.positions.contains(&position)
    }

    fn short_conv(&mut self, record: ShortConvRecord<'_>) {
        let Some(field) = self.short_conv else {
            return;
        };
        if !self.layers.contains(&record.layer) || !self.positions.contains(&record.position) {
            return;
        }
        let values = match field {
            ShortConvField::GateB => record.gate_b,
            ShortConvField::GateC => record.gate_c,
            ShortConvField::ConvOut => record.conv_out,
            ShortConvField::GatedY => record.gated_y,
        };
        self.outputs
            .insert((record.layer, record.position), values.to_vec());
    }

    fn operand_input(&mut self, layer: usize, site: InputSite, values: &[f32]) {
        let Some(key) = self.wanted(layer) else {
            return;
        };
        match site {
            InputSite::Ffn => {
                self.ffn_inputs.insert(key, values.to_vec());
            }
            InputSite::FfnOutput if self.want_output => {
                self.outputs.insert(key, values.to_vec());
            }
            _ => {}
        }
    }
}

/// Step every token up to the last requested position, keeping `site` at `layers` and `positions`.
pub(super) fn capture(
    runtime: &V3Runtime,
    overrides: Option<&OperandOverrides>,
    continuation: &SelectedContinuation,
    ids: &[u32],
    site: CaptureSite,
    layers: &[usize],
    positions: &[usize],
) -> Result<DecodeCapture, LqlError> {
    if site == CaptureSite::FfnDownOutput {
        // The FFN output is the down projection's output only on a dense FFN.
        if let Some(&layer) = layers.iter().find(|&&l| {
            runtime.plan().layers[l]
                .ffn
                .as_ref()
                .and_then(|ffn| ffn.dense())
                .is_none()
        }) {
            return Err(refused(format!(
                "ffn.down_output at layer {layer} needs a dense FFN: this layer's FFN is not one down projection"
            )));
        }
    }
    let short_conv = ShortConvField::of(site);
    if short_conv.is_some() {
        if let Some(&layer) = layers.iter().find(|&&l| {
            !matches!(
                runtime.plan().layers[l].attention,
                LayerAttention::ShortConv(_)
            )
        }) {
            return Err(refused(format!(
                "{} at layer {layer} needs a ShortConv layer: this layer's mixer is not ShortConv",
                site.name()
            )));
        }
    }
    let addresses = layers
        .iter()
        .flat_map(|&l| positions.iter().map(move |&p| (l, p)));
    let mut tap = Tap {
        down: (site == CaptureSite::FfnDownInput).then(|| DownInputCapture::at(addresses)),
        want_output: site == CaptureSite::FfnDownOutput,
        short_conv,
        layers: layers.iter().copied().collect(),
        positions: positions.iter().copied().collect(),
        position: None,
        outputs: BTreeMap::new(),
        ffn_inputs: BTreeMap::new(),
    };
    let mut session = match overrides {
        Some(overrides) => runtime.session_overlaid(overrides, continuation),
        None => runtime.session(continuation),
    }
    .map_err(|e| LqlError::exec("v3 session failed", e))?;
    // The session captured its operands at construction.
    confirm_patched_weights_ran(overrides)?;
    let last = positions.iter().copied().max().unwrap_or(0);
    for &token in &ids[..=last] {
        session
            .step_observed(token, &mut tap)
            .map_err(|e| LqlError::exec("v3 step failed", e))?;
    }

    let take = |layer: usize, position: usize| -> Option<Vec<f32>> {
        match &tap.down {
            Some(down) => down.get(layer, position).map(<[f32]>::to_vec),
            None => tap.outputs.get(&(layer, position)).cloned(),
        }
    };
    let mut blocks = Vec::with_capacity(layers.len());
    for &layer in layers {
        let mut rows = Vec::with_capacity(positions.len());
        for &position in positions {
            rows.push(take(layer, position).ok_or_else(|| {
                LqlError::Execution(format!(
                    "the executor reported no {} at layer {layer}, position {position}",
                    site.name()
                ))
            })?);
        }
        blocks.push(rows);
    }

    let infer_pass = batch::capture(
        runtime,
        overrides,
        continuation,
        ids,
        batch::PassSite::Layer(batch::LayerPlane::FfnInput),
        layers,
        positions,
    )?;
    let matches = layers.iter().zip(&infer_pass).all(|(&layer, rows)| {
        positions.iter().zip(rows).all(|(&position, row)| {
            tap.ffn_inputs
                .get(&(layer, position))
                .is_some_and(|seen| same_bits(seen, row))
        })
    });
    Ok(DecodeCapture {
        blocks,
        ffn_input_matches_infer_pass: matches,
    })
}

fn same_bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
