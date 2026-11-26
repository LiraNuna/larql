//! LFM3 (Liquid AI) — hybrid 3-way stack interleaving gated short-convolution
//! layers (`conv`), sliding-window attention (`sliding_attention`), and full
//! attention (`full_attention`) with zero-tail partial RoPE, per-head scalar
//! sigmoid attention output gating, per-head QK RMSNorm, sandwich norms
//! (`use_sandwich_norm`), and dense SwiGLU FFNs (`feed_forward.{w1,w3,w2}`).
//!
//! Reference: `configuration_lfm3.py`, `modeling_lfm3.py`, `lfm3_conv.py`
//! (`LiquidAI/LFM3-1.7B`).

use crate::config::{
    AttentionGateSpec, ModelArchitecture, ModelConfig, PositionPolicy, PostNormEps, QkNormScope,
    RotaryFrequencyBasis, ShortConvGeometry, LAYER_TYPE_CONV, LAYER_TYPE_FULL_ATTENTION,
    LAYER_TYPE_SLIDING_ATTENTION,
};
use crate::tensor_keys::qk_norm;

/// `Lfm3Config` class default for `norm_eps` (`0.00001`).
const LFM3_DEFAULT_NORM_EPS: f32 = 1e-5;
/// `Lfm3Config` class default for `conv_L_cache`.
const LFM3_DEFAULT_CONV_L_CACHE: usize = 7;
/// `Lfm3Config` class default for `rope_parameters.sliding_attention.rope_theta`.
const LFM3_DEFAULT_SLIDING_ROPE_THETA: f64 = 10_000.0;

pub struct Lfm3Arch {
    config: ModelConfig,
}

impl Lfm3Arch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }

    /// Per-layer operator spelling: declared `layer_types[layer]`, or
    /// `Lfm3Config.__post_init__`'s default schedule (`full_attention` when
    /// `i % 4 == 2`, `sliding_attention` when `i % 4 == 0 && i > 0`, else
    /// `conv`).
    fn layer_type(&self, layer: usize) -> &str {
        if let Some(entry) = self
            .config
            .layer_types
            .as_ref()
            .and_then(|types| types.get(layer))
        {
            return entry.as_str();
        }
        if layer % 4 == 2 {
            LAYER_TYPE_FULL_ATTENTION
        } else if layer > 0 && layer.is_multiple_of(4) {
            LAYER_TYPE_SLIDING_ATTENTION
        } else {
            LAYER_TYPE_CONV
        }
    }

    fn is_conv_layer(&self, layer: usize) -> bool {
        self.layer_type(layer).eq_ignore_ascii_case(LAYER_TYPE_CONV)
    }

    fn is_full_attention_layer(&self, layer: usize) -> bool {
        self.layer_type(layer)
            .eq_ignore_ascii_case(LAYER_TYPE_FULL_ATTENTION)
    }

    fn is_sliding_layer(&self, layer: usize) -> bool {
        self.layer_type(layer)
            .eq_ignore_ascii_case(LAYER_TYPE_SLIDING_ATTENTION)
    }
}

use crate::config::architecture_prelude::*;

impl ArchitectureCore for Lfm3Arch {
    /// Preserves the checkpoint's own `model_type` (`lfm3`, `lfm3_text`, …).
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }
}

impl TensorKeys for Lfm3Arch {
    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::q(&self.layer_prefix(layer))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::k(&self.layer_prefix(layer))
    }

    fn ffn_gate_key(&self, layer: usize) -> String {
        format!("{}feed_forward.w1.weight", self.layer_prefix(layer))
    }

    fn ffn_up_key(&self, layer: usize) -> String {
        format!("{}feed_forward.w3.weight", self.layer_prefix(layer))
    }

    fn ffn_down_key(&self, layer: usize) -> String {
        format!("{}feed_forward.w2.weight", self.layer_prefix(layer))
    }

    fn input_layernorm_key(&self, layer: usize) -> String {
        format!("{}operator_norm.weight", self.layer_prefix(layer))
    }

    fn post_attention_layernorm_key(&self, layer: usize) -> String {
        if self.has_post_norms() {
            format!("{}post_operator_norm.weight", self.layer_prefix(layer))
        } else {
            format!("{}ffn_norm.weight", self.layer_prefix(layer))
        }
    }

    fn pre_feedforward_layernorm_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}ffn_norm.weight", self.layer_prefix(layer)))
    }

    fn post_feedforward_layernorm_key(&self, layer: usize) -> Option<String> {
        Some(format!("{}post_ffn_norm.weight", self.layer_prefix(layer)))
    }

    fn final_norm_key(&self) -> &str {
        "model.embedding_norm.weight"
    }
}

impl Norms for Lfm3Arch {
    fn default_norm_eps(&self) -> f32 {
        LFM3_DEFAULT_NORM_EPS
    }

    fn qk_norm_scope(&self) -> QkNormScope {
        QkNormScope::PerHead
    }

    /// `Lfm3Config.use_sandwich_norm` defaults to `True`.
    fn has_post_norms(&self) -> bool {
        self.config.use_sandwich_norm.unwrap_or(true)
    }

    /// `Lfm3DecoderLayer` builds `post_operator_norm` and `post_ffn_norm`
    /// with `eps=config.norm_eps`, shared with the pre-norms.
    fn post_norm_eps(&self) -> Option<PostNormEps> {
        self.config
            .post_norm_eps
            .map(PostNormEps::Value)
            .or(Some(PostNormEps::Shared))
    }
}

impl Position for Lfm3Arch {
    fn rope_base_for_layer(&self, layer: usize) -> f64 {
        if self.is_sliding_layer(layer) {
            self.config
                .rope_local_base
                .unwrap_or(LFM3_DEFAULT_SLIDING_ROPE_THETA)
        } else {
            self.config.rope_base
        }
    }

    fn rotary_fraction_for_layer(&self, layer: usize) -> f64 {
        if self.is_full_attention_layer(layer) {
            self.config.partial_rotary_factor.unwrap_or(1.0)
        } else {
            1.0
        }
    }

    /// `conv` layers carry no RoPE; `sliding_attention` rotates the full head
    /// at the local base (`10_000`); `full_attention` uses LFM3's zero-tail
    /// partial RoPE (`RotaryFrequencyBasis::HeadWidth`) when
    /// `partial_rotary_factor < 1.0`, else full RoPE at `rope_base`.
    fn position_policy_for_layer(&self, layer: usize) -> PositionPolicy {
        if self.is_conv_layer(layer) {
            return PositionPolicy::None;
        }
        let theta = self.rope_base_for_layer(layer);
        if self.is_full_attention_layer(layer) {
            if let Some(rotary_fraction) = self.config.partial_rotary_factor.filter(|f| *f < 1.0) {
                return PositionPolicy::PartialRope {
                    theta,
                    rotary_fraction,
                    basis: RotaryFrequencyBasis::HeadWidth,
                };
            }
        }
        PositionPolicy::Rope { theta }
    }
}

impl Attention for Lfm3Arch {
    /// `Lfm3Attention` unconditionally applies a per-head scalar sigmoid gate
    /// `gate_proj: Linear(hidden_size, num_attention_heads, bias=False)` to
    /// the pre-attention normed hidden states before `o_proj`.
    fn attention_output_gate(&self) -> Option<AttentionGateSpec> {
        Some(AttentionGateSpec::from_attention_input_per_head_sigmoid_before_output_projection())
    }

    fn is_sliding_window_layer(&self, layer: usize) -> bool {
        self.sliding_window_size().is_some() && self.is_sliding_layer(layer)
    }

    fn short_conv_geometry(&self) -> Option<ShortConvGeometry> {
        Some(ShortConvGeometry {
            conv_l_cache: self
                .config
                .conv_l_cache
                .unwrap_or(LFM3_DEFAULT_CONV_L_CACHE),
            conv_bias: self.config.conv_bias.unwrap_or(false),
        })
    }
}

impl FeedForward for Lfm3Arch {}
impl LatentAttention for Lfm3Arch {}
impl Embeddings for Lfm3Arch {}
impl ModelArchitecture for Lfm3Arch {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GateGranularity;

    fn lfm3_config_json(use_sandwich_norm: bool) -> serde_json::Value {
        serde_json::json!({
            "model_type": "lfm3",
            "hidden_size": 2048,
            "intermediate_size": 7680,
            "num_hidden_layers": 6,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "vocab_size": 128000,
            "conv_L_cache": 7,
            "conv_bias": false,
            "sliding_window": 256,
            "use_sandwich_norm": use_sandwich_norm,
            "tie_word_embeddings": true,
            "layer_types": [
                "conv",
                "conv",
                "full_attention",
                "conv",
                "sliding_attention",
                "conv"
            ],
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.5
                },
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0
                }
            }
        })
    }

    #[test]
    fn resolves_three_way_schedule_norms_gate_and_rope() {
        let arch = crate::detect_from_json(&lfm3_config_json(true));
        assert_eq!(arch.family(), "lfm3");
        assert!((arch.default_norm_eps() - 1e-5).abs() < 1e-12);
        assert!((arch.norm_eps() - 1e-5).abs() < 1e-12);
        assert!(arch.has_post_norms());
        assert_eq!(arch.post_norm_eps(), Some(PostNormEps::Shared));
        assert_eq!(arch.qk_norm_scope(), QkNormScope::PerHead);

        let gate = arch.attention_output_gate().expect("LFM3 carries gate");
        assert_eq!(gate.granularity, GateGranularity::PerHead);

        assert_eq!(
            arch.short_conv_geometry(),
            Some(ShortConvGeometry {
                conv_l_cache: 7,
                conv_bias: false,
            })
        );

        // Layer 0: conv → NoPE, not sliding.
        assert_eq!(arch.position_policy_for_layer(0), PositionPolicy::None);
        assert!(!arch.is_sliding_window_layer(0));
        assert!((arch.rotary_fraction_for_layer(0) - 1.0).abs() < 1e-12);

        // Layer 2: full_attention → HeadWidth zero-tail partial RoPE at 1e6.
        assert_eq!(
            arch.position_policy_for_layer(2),
            PositionPolicy::PartialRope {
                theta: 1_000_000.0,
                rotary_fraction: 0.5,
                basis: RotaryFrequencyBasis::HeadWidth,
            }
        );
        assert!(!arch.is_sliding_window_layer(2));
        assert!((arch.rotary_fraction_for_layer(2) - 0.5).abs() < 1e-12);

        // Layer 4: sliding_attention → full RoPE at 10_000, sliding window 256.
        assert_eq!(
            arch.position_policy_for_layer(4),
            PositionPolicy::Rope { theta: 10_000.0 }
        );
        assert!(arch.is_sliding_window_layer(4));
        assert_eq!(arch.sliding_window_size(), Some(256));

        // Tensor keys match LFM3 checkpoint spellings.
        assert_eq!(
            arch.attn_q_norm_key(2).as_deref(),
            Some("layers.2.self_attn.q_norm.weight")
        );
        assert_eq!(
            arch.attn_k_norm_key(2).as_deref(),
            Some("layers.2.self_attn.k_norm.weight")
        );
        assert_eq!(arch.attn_o_key(2), "layers.2.self_attn.o_proj.weight");
        assert_eq!(arch.ffn_gate_key(0), "layers.0.feed_forward.w1.weight");
        assert_eq!(arch.ffn_up_key(0), "layers.0.feed_forward.w3.weight");
        assert_eq!(arch.ffn_down_key(0), "layers.0.feed_forward.w2.weight");
        assert_eq!(arch.input_layernorm_key(0), "layers.0.operator_norm.weight");
        assert_eq!(
            arch.post_attention_layernorm_key(0),
            "layers.0.post_operator_norm.weight"
        );
        assert_eq!(
            arch.pre_feedforward_layernorm_key(0).as_deref(),
            Some("layers.0.ffn_norm.weight")
        );
        assert_eq!(
            arch.post_feedforward_layernorm_key(0).as_deref(),
            Some("layers.0.post_ffn_norm.weight")
        );
        assert_eq!(arch.final_norm_key(), "model.embedding_norm.weight");
    }

    #[test]
    fn sandwich_norm_false_switches_to_two_norm_mode() {
        let arch = crate::detect_from_json(&lfm3_config_json(false));
        assert!(!arch.has_post_norms());
        assert_eq!(
            arch.post_attention_layernorm_key(0),
            "layers.0.ffn_norm.weight"
        );
    }

    #[test]
    fn omitted_layer_types_falls_back_to_lfm3_default_schedule() {
        let arch = crate::detect_from_json(&serde_json::json!({
            "model_type": "lfm3_text",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "sliding_window": 64
        }));
        assert_eq!(arch.family(), "lfm3_text");
        assert_eq!(arch.position_policy_for_layer(0), PositionPolicy::None);
        assert_eq!(arch.position_policy_for_layer(1), PositionPolicy::None);
        assert_eq!(
            arch.position_policy_for_layer(2),
            PositionPolicy::Rope { theta: 1_000_000.0 }
        );
        assert_eq!(arch.position_policy_for_layer(3), PositionPolicy::None);
        assert!(arch.is_sliding_window_layer(4));
        assert_eq!(
            arch.position_policy_for_layer(4),
            PositionPolicy::Rope { theta: 10_000.0 }
        );
        assert_eq!(arch.position_policy_for_layer(5), PositionPolicy::None);
    }
}
