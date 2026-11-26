//! Short gated causal depthwise-conv1d recurrence (`Lfm3ShortConv`).
//!
//! Each `conv` layer carries a fused `in_proj` (`[3*D, D]`) split into
//! `(B, C, x)`, a causal depthwise 1D convolution of width `conv_l_cache`
//! over `B * x` (no nonlinearity), multiplicative output gating `C * conv_out`,
//! and `out_proj` (`[D, D]`). Unlike Mamba2/ConvQkv, each ShortConv layer sits
//! inside a standard transformer block with pre/post norms and a dense FFN.

use serde::Serialize;

use super::OperandRef;

/// One layer's short gated depthwise-conv1d operator.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ShortConvOp {
    /// Model hidden dimension (`D`).
    pub hidden_size: usize,
    /// Causal convolution kernel width (`conv_L_cache`).
    pub conv_l_cache: usize,
    /// Fused B|C|x input projection, `[3·hidden_size, hidden_size]`.
    pub in_proj: OperandRef,
    /// Optional additive bias on `in_proj`, `[3·hidden_size]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_proj_bias: Option<OperandRef>,
    /// Depthwise causal conv1d kernel, `[hidden_size, 1, conv_l_cache]`.
    pub conv_weight: OperandRef,
    /// Optional additive bias on `conv`, `[hidden_size]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conv_bias: Option<OperandRef>,
    /// Output projection, `[hidden_size, hidden_size]`.
    pub out_proj: OperandRef,
    /// Optional additive bias on `out_proj`, `[hidden_size]`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub out_proj_bias: Option<OperandRef>,
}

impl ShortConvOp {
    /// Elements in this layer's rolling convolution cache: `hidden_size * conv_l_cache`.
    pub fn state_elements(&self) -> usize {
        self.hidden_size * self.conv_l_cache
    }
}
