//! Per-channel interventions on a `ShortConv` layer's gated output
//! `gated_y = C * conv(B * x)`, immediately before `out_proj`.
//!
//! Mirrors [`super::intervene_heads`] in discipline:
//! - Every intervention is a declared object `(layer, channel, positions)`
//!   with a kind (`Zero`, `Scale(α)`, `Replace(v)`).
//! - Plans are admitted fail-closed before the first token executes:
//!   out-of-range layers, non-`ShortConv` layers, out-of-range channels,
//!   overlapping declarations, non-finite scalars, and non-`Single` residual
//!   carriers are all refused upfront.
//! - Declaring `ShortConvInterventionPlan::none()` adds zero overhead to the
//!   unintervened path; `Scale(1.0)` is a bit-exact no-op on `gated_y[c]`.
//! - When observation is also armed on the same layer, the observer receives
//!   the **pre-intervention** `ShortConvRecord`, followed by
//!   `StepEvent::ShortConvObserved`, followed by one
//!   `StepEvent::ShortConvIntervened` per channel that fired.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use super::super::{ComponentOpPlan, LayerAttention};
use super::prepared::PreparedOperands;
use crate::error::VindexError;

/// What a channel intervention does to `gated_y[channel]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ShortConvInterventionKind {
    /// `gated_y[channel]` becomes `0.0`.
    Zero,
    /// `gated_y[channel]` is multiplied by a declared finite scalar.
    Scale,
    /// `gated_y[channel]` is replaced by a declared finite scalar.
    Replace,
}

impl ShortConvInterventionKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Scale => "scale",
            Self::Replace => "replace",
        }
    }
}

/// One `ShortConv` layer's one channel, at declared positions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortConvAddress {
    pub layer: usize,
    pub channel: usize,
    positions: BTreeSet<usize>,
}

impl ShortConvAddress {
    pub fn new(
        layer: usize,
        channel: usize,
        positions: impl IntoIterator<Item = usize>,
    ) -> Result<Self, VindexError> {
        let positions: BTreeSet<usize> = positions.into_iter().collect();
        if positions.is_empty() {
            return Err(VindexError::Parse(format!(
                "short-conv intervention at layer {layer} channel {channel} names no position"
            )));
        }
        Ok(Self {
            layer,
            channel,
            positions,
        })
    }

    pub fn positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.positions.iter().copied()
    }

    pub fn covers(&self, position: usize) -> bool {
        self.positions.contains(&position)
    }

    fn overlaps(&self, other: &Self) -> bool {
        self.layer == other.layer
            && self.channel == other.channel
            && self.positions.iter().any(|p| other.positions.contains(p))
    }
}

/// One declared change to `gated_y[channel]` before `out_proj`.
#[derive(Debug, Clone, PartialEq)]
pub struct ShortConvIntervention {
    address: ShortConvAddress,
    kind: ShortConvInterventionKind,
    scalar: Option<f32>,
}

impl ShortConvIntervention {
    /// `gated_y[channel]` becomes `0.0` at the address.
    pub fn zero(address: ShortConvAddress) -> Self {
        Self {
            address,
            kind: ShortConvInterventionKind::Zero,
            scalar: None,
        }
    }

    /// `gated_y[channel]` is multiplied by `factor` at the address. `factor`
    /// must be finite; `Scale(1.0)` is the bit-exact no-op law.
    pub fn scale(address: ShortConvAddress, factor: f32) -> Result<Self, VindexError> {
        if !factor.is_finite() {
            return Err(VindexError::Parse(format!(
                "short-conv intervention `scale` at layer {} channel {} carries a non-finite \
                 factor",
                address.layer, address.channel
            )));
        }
        Ok(Self {
            address,
            kind: ShortConvInterventionKind::Scale,
            scalar: Some(factor),
        })
    }

    /// `gated_y[channel]` becomes `value` at the address. `value` must be finite.
    pub fn replace(address: ShortConvAddress, value: f32) -> Result<Self, VindexError> {
        if !value.is_finite() {
            return Err(VindexError::Parse(format!(
                "short-conv intervention `replace` at layer {} channel {} carries a non-finite \
                 value",
                address.layer, address.channel
            )));
        }
        Ok(Self {
            address,
            kind: ShortConvInterventionKind::Replace,
            scalar: Some(value),
        })
    }

    pub fn address(&self) -> &ShortConvAddress {
        &self.address
    }

    pub fn kind(&self) -> ShortConvInterventionKind {
        self.kind
    }

    /// Apply this intervention in place to `gated_y[channel]`.
    pub(super) fn apply(&self, value: &mut f32) {
        match self.kind {
            ShortConvInterventionKind::Zero => *value = 0.0,
            ShortConvInterventionKind::Scale => {
                *value *= self.scalar.expect("scale carries its factor");
            }
            ShortConvInterventionKind::Replace => {
                *value = self.scalar.expect("replace carries its value");
            }
        }
    }

    fn hash_into(&self, hasher: &mut Sha256) {
        hasher.update((self.address.layer as u64).to_le_bytes());
        hasher.update((self.address.channel as u64).to_le_bytes());
        hasher.update((self.address.positions.len() as u64).to_le_bytes());
        for p in &self.address.positions {
            hasher.update((*p as u64).to_le_bytes());
        }
        hasher.update(self.kind.name().as_bytes());
        if let Some(s) = self.scalar {
            hasher.update(s.to_le_bytes());
        }
    }
}

/// One ShortConv channel firing, as the step reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortConvFiring {
    pub layer: usize,
    pub channel: usize,
    pub position: usize,
    pub kind: ShortConvInterventionKind,
}

/// A declared ShortConv channel address that the run never reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShortConvUnreached {
    pub layer: usize,
    pub channel: usize,
    pub position: usize,
}

/// Every ShortConv channel intervention a run declares.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ShortConvInterventionPlan {
    interventions: Vec<ShortConvIntervention>,
}

impl ShortConvInterventionPlan {
    pub const fn none() -> Self {
        Self {
            interventions: Vec::new(),
        }
    }

    pub fn is_none(&self) -> bool {
        self.interventions.is_empty()
    }

    /// Declare one more channel intervention. Two interventions sharing a
    /// `(layer, channel, position)` are refused so execution never depends
    /// on declaration order.
    pub fn with(mut self, intervention: ShortConvIntervention) -> Result<Self, VindexError> {
        if let Some(existing) = self
            .interventions
            .iter()
            .find(|i| i.address.overlaps(&intervention.address))
        {
            return Err(VindexError::Parse(format!(
                "two short-conv interventions share layer {} channel {} at a position — `{}` \
                 and `{}` — and which applied first would depend on declaration order",
                intervention.address.layer,
                intervention.address.channel,
                existing.kind.name(),
                intervention.kind.name()
            )));
        }
        self.interventions.push(intervention);
        Ok(self)
    }

    pub fn declared(&self) -> usize {
        self.interventions.len()
    }

    pub fn interventions(&self) -> &[ShortConvIntervention] {
        &self.interventions
    }

    /// Refuse before the first token executes: non-`Single` carrier, layer
    /// outside executed range, non-`ShortConv` layer, or channel `>= hidden_size`.
    pub fn admit(&self, plan: &ComponentOpPlan, ops: &PreparedOperands) -> Result<(), VindexError> {
        if self.is_none() {
            return Ok(());
        }
        if ops.hyper_connection().is_some() {
            return Err(VindexError::Parse(
                "short-conv interventions act on a `Single` carrier; this component carries a \
                 `Bundle` (hyper-connected)"
                    .to_string(),
            ));
        }
        if ops.attention_residual_block_size().is_some() {
            return Err(VindexError::Parse(
                "short-conv interventions act on a `Single` carrier; this component carries a \
                 `History` (attention-residual)"
                    .to_string(),
            ));
        }
        let first = ops.first_layer();
        let executed = first..first + ops.layers().len();
        for intervention in &self.interventions {
            let address = &intervention.address;
            if !executed.contains(&address.layer) {
                return Err(VindexError::Parse(format!(
                    "short-conv intervention `{}` addresses layer {} but this image executes \
                     layers {}..{}",
                    intervention.kind.name(),
                    address.layer,
                    executed.start,
                    executed.end
                )));
            }
            let LayerAttention::ShortConv(op) = &plan.layers[address.layer].attention else {
                return Err(VindexError::Parse(format!(
                    "short-conv intervention `{}` addresses layer {}, which is not a ShortConv \
                     layer",
                    intervention.kind.name(),
                    address.layer
                )));
            };
            if address.channel >= op.hidden_size {
                return Err(VindexError::Parse(format!(
                    "short-conv intervention `{}` addresses channel {} but layer {} has {} \
                     channels",
                    intervention.kind.name(),
                    address.channel,
                    address.layer,
                    op.hidden_size
                )));
            }
        }
        Ok(())
    }

    pub(super) fn touches_layer(&self, layer: usize) -> bool {
        self.interventions.iter().any(|i| i.address.layer == layer)
    }

    pub(super) fn at(
        &self,
        layer: usize,
        channel: usize,
        position: usize,
    ) -> Option<&ShortConvIntervention> {
        self.interventions.iter().find(|i| {
            i.address.layer == layer && i.address.channel == channel && i.address.covers(position)
        })
    }

    pub fn declaration_sha256(&self) -> Option<String> {
        if self.is_none() {
            return None;
        }
        let mut hasher = Sha256::new();
        for i in &self.interventions {
            i.hash_into(&mut hasher);
        }
        let bytes: [u8; 32] = hasher.finalize().into();
        Some(bytes.iter().fold(String::with_capacity(64), |mut s, b| {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
            s
        }))
    }

    pub fn unreached(&self, positions_executed: usize) -> Vec<ShortConvUnreached> {
        self.interventions
            .iter()
            .flat_map(|i| {
                i.address
                    .positions()
                    .filter(move |&p| p >= positions_executed)
                    .map(move |position| ShortConvUnreached {
                        layer: i.address.layer,
                        channel: i.address.channel,
                        position,
                    })
            })
            .collect()
    }
}
