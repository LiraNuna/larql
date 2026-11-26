//! Which planned operands a stored-weight op may stand in for: the allow-list, over every kind of layer the
//! fixtures build.

use std::path::Path;

use larql_vindex::format::vindex3::fixtures::{encode_fixture_container, hybrid_lllf_f32_model};
use larql_vindex::format::vindex3::fixtures_kimi::{
    hybrid_kda_mla_f32_model, kimi_per_expert_moe_f32_model,
};
use larql_vindex::format::vindex3::fixtures_qwen::miniature_qwen2_moe;
use larql_vindex::format::vindex3::fixtures_routed::miniature_routed;
use larql_vindex::format::vindex3::knowledge::live_weights::read_path::{
    seam_operands, seam_refusal, WHY_EXPERT_BANK, WHY_OTHER_MIXER, WHY_ROUTED_FFN,
    WHY_SHARED_EXPERT, WHY_UNPLANNED_CLASS,
};
use larql_vindex::format::vindex3::knowledge::LiveWeightRefusal;
use larql_vindex::format::vindex3::opplan::exec::backend::MatrixClass;
use larql_vindex::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use larql_vindex::format::vindex3::opplan::{ComponentOpPlan, LayerAttention, LayerFfn};

use crate::fixture::*;
use crate::live_support::*;

const ATTENTION: Operation = Operation::Project(MatrixClass::AttentionProjection);
const FFN: Operation = Operation::Project(MatrixClass::FfnProjection);

fn plan_of(write_checkpoint: impl FnOnce(&Path)) -> ComponentOpPlan {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        write_checkpoint,
        checkpoint.path(),
        container.path(),
        "read-path",
    );
    bind(container.path()).plan
}

/// The refusals of every planned operand of `plan` matching `pick`; panics if none matches.
fn refusals(
    plan: &ComponentOpPlan,
    pick: impl Fn(&ComponentOpPlan, &PlannedOperand) -> bool,
) -> Vec<Option<&'static str>> {
    let found: Vec<_> = plan
        .planned_operands()
        .iter()
        .filter(|p| pick(plan, p))
        .map(|p| seam_refusal(plan, p))
        .collect();
    assert!(!found.is_empty(), "the fixture carries no such operand");
    found
}

fn all(found: Vec<Option<&'static str>>, want: Option<&'static str>) -> bool {
    found.iter().all(|r| *r == want)
}

#[test]
fn every_matrix_of_a_dense_softmax_model_reads_through_the_seam() {
    let src = source_with_pack();
    let plan = bind(&src.container).plan;
    assert!(all(refusals(&plan, |_, _| true), None));
    let operands = seam_operands(&plan, TENSOR).unwrap();
    assert_eq!(operands.len(), 1);
    assert_eq!(
        (operands[0].object.as_str(), operands[0].tensor.as_str()),
        (DECODER, TENSOR)
    );
    assert_eq!(
        seam_operands(&plan, "no.such.weight").unwrap_err(),
        LiveWeightRefusal::NotExecuted {
            tensor: "no.such.weight".into()
        }
    );
}

#[test]
fn expert_banks_are_outside_the_seam_packed_or_per_expert() {
    let per_expert = plan_of(kimi_per_expert_moe_f32_model);
    assert!(all(
        refusals(&per_expert, |_, p| matches!(
            p.operation,
            Operation::ExpertProject { .. }
        )),
        Some(WHY_EXPERT_BANK)
    ));
    let packed = plan_of(miniature_routed);
    let slices =
        |_: &ComponentOpPlan, p: &PlannedOperand| p.operation == Operation::ExpertBankSlice;
    assert!(all(refusals(&packed, slices), Some(WHY_EXPERT_BANK)));
    let bank = packed
        .planned_operands()
        .into_iter()
        .find(|p| p.operation == Operation::ExpertBankSlice)
        .unwrap();
    assert!(matches!(
        seam_operands(&packed, &bank.operand.tensor),
        Err(LiveWeightRefusal::OutsideSeam { why, operation, .. })
            if why == WHY_EXPERT_BANK && operation == Operation::ExpertBankSlice.name()
    ));
}

#[test]
fn a_shared_expert_is_outside_the_seam() {
    let plan = plan_of(miniature_qwen2_moe);
    assert!(all(
        refusals(&plan, |_, p| matches!(
            p.operation,
            Operation::SharedExpertProject | Operation::SharedExpertBranchGate
        )),
        Some(WHY_SHARED_EXPERT)
    ));
}

#[test]
fn only_softmax_attention_is_inside_the_seam() {
    let softmax = |plan: &ComponentOpPlan, p: &PlannedOperand| {
        p.operation == ATTENTION
            && matches!(
                plan.layers[p.layer.unwrap()].attention,
                LayerAttention::Softmax(_)
            )
    };
    let other = |plan: &ComponentOpPlan, p: &PlannedOperand| {
        p.operation == ATTENTION
            && !matches!(
                plan.layers[p.layer.unwrap()].attention,
                LayerAttention::Softmax(_)
            )
    };
    // Three Gated DeltaNet layers and one softmax layer.
    let lllf = plan_of(hybrid_lllf_f32_model);
    assert!(all(refusals(&lllf, softmax), None));
    assert!(all(refusals(&lllf, other), Some(WHY_OTHER_MIXER)));
    // KDA and MLA layers, no softmax.
    let kkkm = plan_of(hybrid_kda_mla_f32_model);
    assert!(all(refusals(&kkkm, other), Some(WHY_OTHER_MIXER)));
}

#[test]
fn a_projection_with_no_qualified_layer_or_class_is_outside_the_seam() {
    let src = source_with_pack();
    let plan = bind(&src.container).plan;
    let planned = plan.planned_operands();
    let of = |operation: Operation| {
        planned
            .iter()
            .find(|p| p.operation == operation)
            .cloned()
            .unwrap()
    };

    // A projection the plan gives no layer: nothing says which loader reads it.
    let mut attention = of(ATTENTION);
    attention.layer = None;
    assert_eq!(seam_refusal(&plan, &attention), Some(WHY_OTHER_MIXER));
    let mut ffn = of(FFN);
    ffn.layer = None;
    assert_eq!(seam_refusal(&plan, &ffn), Some(WHY_ROUTED_FFN));

    // The classes no projection is planned under.
    for class in [MatrixClass::OutputHead, MatrixClass::RoutedExpertBank] {
        let mut unplanned = of(FFN);
        unplanned.operation = Operation::Project(class);
        assert_eq!(seam_refusal(&plan, &unplanned), Some(WHY_UNPLANNED_CLASS));
    }

    // A dense projection of a layer whose FFN is routed.
    let routed = plan_of(miniature_routed);
    let layer = routed
        .layers
        .iter()
        .position(|l| matches!(l.ffn, Some(LayerFfn::Routed(_))))
        .expect("the fixture routes a layer");
    let mut dense_in_routed = of(FFN);
    dense_in_routed.layer = Some(layer);
    assert_eq!(
        seam_refusal(&routed, &dense_in_routed),
        Some(WHY_ROUTED_FFN)
    );
}

#[test]
fn hybrid_short_conv_and_dense_ffn_matrices_read_through_the_seam() {
    use larql_vindex::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};

    let plan = plan_of(|dir| {
        const H: usize = 8;
        const I: usize = 12;
        const V: usize = 16;
        let config = serde_json::json!({
            "architectures": ["Lfm3ForCausalLM"],
            "model_type": "lfm3",
            "torch_dtype": "float32",
            "hidden_size": H,
            "intermediate_size": I,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "vocab_size": V,
            "max_position_embeddings": 32,
            "norm_eps": 1e-5,
            "conv_L_cache": 3,
            "conv_bias": false,
            "use_sandwich_norm": true,
            "tie_word_embeddings": true,
            "layer_types": ["conv", "full_attention"],
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.5
                }
            }
        });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let mut s = ShardBuilder::new();
        s.push("model.embed_tokens.weight", &[V, H], &lcg_values(V * H, 1));
        s.push("model.embedding_norm.weight", &[H], &norm_values(H, 2));
        for (layer, kind) in [(0, "conv"), (1, "full_attention")] {
            let p = format!("model.layers.{layer}");
            let seed = 10 + layer as u64 * 20;
            s.push(
                &format!("{p}.operator_norm.weight"),
                &[H],
                &norm_values(H, seed),
            );
            s.push(
                &format!("{p}.post_operator_norm.weight"),
                &[H],
                &norm_values(H, seed + 1),
            );
            s.push(
                &format!("{p}.post_ffn_norm.weight"),
                &[H],
                &norm_values(H, seed + 2),
            );
            s.push(
                &format!("{p}.ffn_norm.weight"),
                &[H],
                &norm_values(H, seed + 3),
            );
            if kind == "conv" {
                s.push(
                    &format!("{p}.conv.in_proj.weight"),
                    &[3 * H, H],
                    &lcg_values(3 * H * H, seed + 4),
                );
                s.push(
                    &format!("{p}.conv.conv.weight"),
                    &[H, 1, 3],
                    &lcg_values(H * 3, seed + 5),
                );
                s.push(
                    &format!("{p}.conv.out_proj.weight"),
                    &[H, H],
                    &lcg_values(H * H, seed + 6),
                );
            } else {
                s.push(
                    &format!("{p}.self_attn.q_proj.weight"),
                    &[H, H],
                    &lcg_values(H * H, seed + 4),
                );
                s.push(
                    &format!("{p}.self_attn.k_proj.weight"),
                    &[H, H],
                    &lcg_values(H * H, seed + 5),
                );
                s.push(
                    &format!("{p}.self_attn.v_proj.weight"),
                    &[H, H],
                    &lcg_values(H * H, seed + 6),
                );
                s.push(
                    &format!("{p}.self_attn.o_proj.weight"),
                    &[H, H],
                    &lcg_values(H * H, seed + 7),
                );
                s.push(
                    &format!("{p}.self_attn.q_norm.weight"),
                    &[4],
                    &norm_values(4, seed + 8),
                );
                s.push(
                    &format!("{p}.self_attn.k_norm.weight"),
                    &[4],
                    &norm_values(4, seed + 9),
                );
                s.push(
                    &format!("{p}.self_attn.gate_proj.weight"),
                    &[2, H],
                    &lcg_values(2 * H, seed + 10),
                );
            }
            s.push(
                &format!("{p}.feed_forward.w1.weight"),
                &[I, H],
                &lcg_values(I * H, seed + 11),
            );
            s.push(
                &format!("{p}.feed_forward.w3.weight"),
                &[I, H],
                &lcg_values(I * H, seed + 12),
            );
            s.push(
                &format!("{p}.feed_forward.w2.weight"),
                &[H, I],
                &lcg_values(H * I, seed + 13),
            );
        }
        s.write(dir);
    });

    // Every planned matrix (embedding, ShortConv in/out_proj, softmax q/k/v/o_proj,
    // dense FFN w1/w2/w3, and tied output head) must read through the live-weight seam.
    assert!(all(refusals(&plan, |_, _| true), None));
    assert!(seam_operands(&plan, "0.conv.in_proj.weight").is_ok());
    assert!(seam_operands(&plan, "0.conv.out_proj.weight").is_ok());
    assert!(seam_operands(&plan, "0.feed_forward.w2.weight").is_ok());
    assert!(seam_operands(&plan, "1.feed_forward.w2.weight").is_ok());
}
