//! Declared geometry of a gated short-convolution mixer (`Lfm3ShortConv`).
//!
//! ```text
//! BCx      = in_proj(x)                     [3 * hidden, hidden]
//! B, C, x' = chunk(BCx, 3)                  each [hidden]
//! conv_out = depthwise_conv1d(B * x', w)    w: [hidden, 1, conv_l_cache], no SiLU
//! y        = out_proj(C * conv_out)         [hidden, hidden]
//! ```
//!
//! Kept as its own geometry struct rather than reusing Mamba2 or GatedDelta
//! fields: short-conv has no heads, no state matrix, no decay gate, and no
//! SiLU activation — only the causal depthwise window width (`conv_L_cache`)
//! and the bias flag (`conv_bias`) governing `in_proj`, `conv`, and `out_proj`.

use serde::{Deserialize, Serialize};

use super::ModelConfig;

/// Declared geometry of a gated short-convolution layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShortConvGeometry {
    /// Causal depthwise convolution kernel width (`conv_L_cache`, 7 on LFM3-1.7B).
    pub conv_l_cache: usize,
    /// Whether `in_proj`, `conv`, and `out_proj` carry additive biases (`conv_bias`).
    pub conv_bias: bool,
}

impl ShortConvGeometry {
    /// Read the declared geometry from a parsed [`ModelConfig`], returning
    /// `None` unless `conv_L_cache` was explicitly declared.
    pub fn from_config(cfg: &ModelConfig) -> Option<Self> {
        Some(Self {
            conv_l_cache: cfg.conv_l_cache?,
            conv_bias: cfg.conv_bias.unwrap_or(false),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect_from_json;

    #[test]
    fn from_config_reads_declared_fields_and_returns_none_when_absent() {
        let absent = detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2
        }));
        assert_eq!(ShortConvGeometry::from_config(absent.config()), None);

        let present = detect_from_json(&serde_json::json!({
            "model_type": "lfm3",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "conv_L_cache": 7,
            "conv_bias": true
        }));
        assert_eq!(
            ShortConvGeometry::from_config(present.config()),
            Some(ShortConvGeometry {
                conv_l_cache: 7,
                conv_bias: true,
            })
        );
    }
}
