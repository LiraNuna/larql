//! End-to-end LFM3 (hybrid gated ShortConv + GQA with per-head output gate,
//! partial RoPE, and sandwich norms) acceptance tests in CI miniature.
//!
//! Covers:
//! - Canonical 4-layer LFM3 (`["conv", "sliding_attention", "conv", "full_attention"]`)
//!   with `sliding_window = 3`, `use_sandwich_norm = true`, `conv_bias = false`,
//!   `tie_word_embeddings = true`, `model.embedding_norm.weight`, per-head sigmoid
//!   `output_gate`, and partial RoPE on `full_attention`:
//!   - Plan admission (0 blocking, attention census, operators, surfaces)
//!   - VINDEX3 encode, container verification, opplan closure
//!   - Continuation geometry (`Recurrent([D, L-1])` + `Kv`)
//!   - Prompt longer than `sliding_window` (8 tokens > window 3):
//!     Reference vs Production parity, batch prefill vs step-by-step decode parity,
//!     continuation split/resume parity across ShortConv + Sliding + Full layers
//!   - Per-head observation (`StepObserver` with `wants_attention_heads`) verifying
//!     per-head gate broadcast on gated attention layers and bit-identical observed logits
//!   - Per-head intervention (`HeadInterventionPlan`) firing on gated attention layers
//! - Derivative variant A: `use_sandwich_norm = false` (2-norm `PreOnly` placement)
//! - Derivative variant B: `conv_bias = true` + `tie_word_embeddings = false` (untied `lm_head`)
//! - Completeness defect when the `short_conv` operation surface is stripped.

use std::path::Path;

use larql_models::inventory::build_inventory;

use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
use crate::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};
use crate::format::vindex3::graph::{
    build_from_inventories, execution_completeness, CompletenessDefect, LayerOperator,
};
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::continuation::{
    plan_continuation_geometry, LayerContinuationGeometry,
};
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::intervene::InterventionPlan;
use crate::format::vindex3::opplan::exec::intervene_heads::{
    HeadAddress, HeadIntervention, HeadInterventionPlan,
};
use crate::format::vindex3::opplan::exec::kv::{ContinuationProvider, RowKvState};
use crate::format::vindex3::opplan::exec::observe::{
    AttentionHeadRecord, NoopObserver, StepEvent, StepObserver,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prefill_plan;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::{plan_component_ops, LayerAttention};
use crate::format::vindex3::plan::plan_system;

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 24;
const VOCAB: usize = 31;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const CONV_L_CACHE: usize = 3;
const SLIDING_WINDOW: usize = 3;

#[derive(Clone, Copy)]
struct Lfm3FixtureOpts {
    use_sandwich_norm: bool,
    conv_bias: bool,
    tie_word_embeddings: bool,
}

impl Default for Lfm3FixtureOpts {
    fn default() -> Self {
        Self {
            use_sandwich_norm: true,
            conv_bias: false,
            tie_word_embeddings: true,
        }
    }
}

fn write_miniature_lfm3(dir: &Path, opts: Lfm3FixtureOpts) {
    let config = serde_json::json!({
        "architectures": ["Lfm3ForCausalLM"],
        "model_type": "lfm3",
        "torch_dtype": "float32",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": 4,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "norm_eps": 1e-5,
        "conv_L_cache": CONV_L_CACHE,
        "conv_bias": opts.conv_bias,
        "use_sandwich_norm": opts.use_sandwich_norm,
        "sliding_window": SLIDING_WINDOW,
        "tie_word_embeddings": opts.tie_word_embeddings,
        "layer_types": ["conv", "sliding_attention", "conv", "full_attention"],
        "rope_parameters": {
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0
            }
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let mut shard = ShardBuilder::new();
    shard.push(
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 1),
    );
    shard.push(
        "model.embedding_norm.weight",
        &[HIDDEN],
        &norm_values(HIDDEN, 2),
    );
    if !opts.tie_word_embeddings {
        shard.push(
            "lm_head.weight",
            &[VOCAB, HIDDEN],
            &lcg_values(VOCAB * HIDDEN, 3),
        );
    }

    let layer_kinds = ["conv", "sliding_attention", "conv", "full_attention"];
    for (layer, &kind) in layer_kinds.iter().enumerate() {
        let seed = 100 + layer as u64 * 30;
        let p = format!("model.layers.{layer}");

        shard.push(
            &format!("{p}.operator_norm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed),
        );
        if opts.use_sandwich_norm {
            shard.push(
                &format!("{p}.post_operator_norm.weight"),
                &[HIDDEN],
                &norm_values(HIDDEN, seed + 1),
            );
            shard.push(
                &format!("{p}.post_ffn_norm.weight"),
                &[HIDDEN],
                &norm_values(HIDDEN, seed + 2),
            );
        }
        shard.push(
            &format!("{p}.ffn_norm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 3),
        );

        if kind == "conv" {
            shard.push(
                &format!("{p}.conv.in_proj.weight"),
                &[3 * HIDDEN, HIDDEN],
                &lcg_values(3 * HIDDEN * HIDDEN, seed + 4),
            );
            shard.push(
                &format!("{p}.conv.conv.weight"),
                &[HIDDEN, 1, CONV_L_CACHE],
                &lcg_values(HIDDEN * CONV_L_CACHE, seed + 5),
            );
            shard.push(
                &format!("{p}.conv.out_proj.weight"),
                &[HIDDEN, HIDDEN],
                &lcg_values(HIDDEN * HIDDEN, seed + 6),
            );
            if opts.conv_bias {
                shard.push(
                    &format!("{p}.conv.in_proj.bias"),
                    &[3 * HIDDEN],
                    &lcg_values(3 * HIDDEN, seed + 7),
                );
                shard.push(
                    &format!("{p}.conv.conv.bias"),
                    &[HIDDEN],
                    &lcg_values(HIDDEN, seed + 8),
                );
                shard.push(
                    &format!("{p}.conv.out_proj.bias"),
                    &[HIDDEN],
                    &lcg_values(HIDDEN, seed + 9),
                );
            }
        } else {
            shard.push(
                &format!("{p}.self_attn.q_proj.weight"),
                &[Q_HEADS * HEAD_DIM, HIDDEN],
                &lcg_values(Q_HEADS * HEAD_DIM * HIDDEN, seed + 4),
            );
            shard.push(
                &format!("{p}.self_attn.k_proj.weight"),
                &[KV_HEADS * HEAD_DIM, HIDDEN],
                &lcg_values(KV_HEADS * HEAD_DIM * HIDDEN, seed + 5),
            );
            shard.push(
                &format!("{p}.self_attn.v_proj.weight"),
                &[KV_HEADS * HEAD_DIM, HIDDEN],
                &lcg_values(KV_HEADS * HEAD_DIM * HIDDEN, seed + 6),
            );
            shard.push(
                &format!("{p}.self_attn.o_proj.weight"),
                &[HIDDEN, Q_HEADS * HEAD_DIM],
                &lcg_values(HIDDEN * Q_HEADS * HEAD_DIM, seed + 7),
            );
            shard.push(
                &format!("{p}.self_attn.q_norm.weight"),
                &[HEAD_DIM],
                &norm_values(HEAD_DIM, seed + 8),
            );
            shard.push(
                &format!("{p}.self_attn.k_norm.weight"),
                &[HEAD_DIM],
                &norm_values(HEAD_DIM, seed + 9),
            );
            // Per-head sigmoid output gate: [num_attention_heads, hidden_size].
            shard.push(
                &format!("{p}.self_attn.gate_proj.weight"),
                &[Q_HEADS, HIDDEN],
                &lcg_values(Q_HEADS * HIDDEN, seed + 10),
            );
        }

        // SwiGLU FFN: w1 (gate), w3 (up), w2 (down).
        shard.push(
            &format!("{p}.feed_forward.w1.weight"),
            &[INTERMEDIATE, HIDDEN],
            &lcg_values(INTERMEDIATE * HIDDEN, seed + 11),
        );
        shard.push(
            &format!("{p}.feed_forward.w3.weight"),
            &[INTERMEDIATE, HIDDEN],
            &lcg_values(INTERMEDIATE * HIDDEN, seed + 12),
        );
        shard.push(
            &format!("{p}.feed_forward.w2.weight"),
            &[HIDDEN, INTERMEDIATE],
            &lcg_values(HIDDEN * INTERMEDIATE, seed + 13),
        );
    }

    shard.write(dir);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[derive(Default)]
struct HeadCapture {
    records: Vec<(usize, usize, usize, Option<Vec<f32>>)>,
}

impl StepObserver for HeadCapture {
    fn event(&mut self, _event: StepEvent) {}

    fn wants_attention_heads(&self) -> bool {
        true
    }

    fn attention_head(&mut self, layer: usize, record: AttentionHeadRecord<'_>) {
        self.records.push((
            layer,
            record.position,
            record.head,
            record.gate.map(<[f32]>::to_vec),
        ));
    }
}

#[test]
fn lfm3_canonical_admits_executes_observes_and_intervenes_beyond_sliding_window() {
    let dir = tempfile::tempdir().unwrap();
    write_miniature_lfm3(dir.path(), Lfm3FixtureOpts::default());

    let inventory = build_inventory(dir.path()).unwrap();
    let plan = plan_system(&[("lfm3-mini".to_string(), inventory)]);
    let blocking: Vec<String> = plan
        .artifacts
        .iter()
        .flat_map(|a| &a.findings)
        .filter(|f| f.blocks())
        .map(|f| format!("{}: {}", f.subject, f.detail))
        .collect();
    assert!(plan.admissible, "blocking findings: {blocking:?}");

    let census = plan.artifacts[0]
        .findings
        .iter()
        .find(|f| f.subject == "attention_policy")
        .expect("attention_policy finding");
    assert!(census.detail.contains("1 full"), "{}", census.detail);
    assert!(census.detail.contains("1 sliding"), "{}", census.detail);
    assert!(
        census.detail.contains("2 short-conv recurrent"),
        "{}",
        census.detail
    );

    let comp = plan
        .graph
        .components
        .iter()
        .find(|c| c.id == "target")
        .expect("target component");
    let ops: Vec<LayerOperator> = comp
        .attention
        .as_ref()
        .unwrap()
        .iter()
        .map(|l| l.operator)
        .collect();
    assert_eq!(
        ops,
        vec![
            LayerOperator::ShortConv,
            LayerOperator::Softmax,
            LayerOperator::ShortConv,
            LayerOperator::Softmax,
        ]
    );

    let out = tempfile::tempdir().unwrap();
    let container = out.path().join("lfm3-mini.vindex3");
    encode_checkpoint(dir.path(), &container).expect("encode LFM3");

    let inspection = inspect_container(&container, true).unwrap();
    let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
    assert!(outcome.defects.is_empty(), "{:?}", outcome.defects);
    let op_plan = outcome.plan.expect("op plan closed");
    assert!(op_plan.layers[0].post_attention_norm.is_some());
    assert!(op_plan.layers[0].post_ffn_norm.is_some());
    assert!(matches!(
        op_plan.layers[0].attention,
        LayerAttention::ShortConv(_)
    ));
    assert!(matches!(
        op_plan.layers[1].attention,
        LayerAttention::Softmax(_)
    ));
    assert!(matches!(
        op_plan.layers[2].attention,
        LayerAttention::ShortConv(_)
    ));
    assert!(matches!(
        op_plan.layers[3].attention,
        LayerAttention::Softmax(_)
    ));

    // Continuation geometry: ShortConv layers declare Recurrent([HIDDEN, CONV_L_CACHE]).
    let geometry = plan_continuation_geometry(&op_plan).expect("continuation geometry");
    for (idx, layer_geom) in geometry.iter().enumerate() {
        if idx % 2 == 0 {
            let LayerContinuationGeometry::Recurrent(rec) = layer_geom else {
                panic!("layer {idx} should be Recurrent: {layer_geom:?}");
            };
            assert_eq!(rec.buffers.len(), 1);
            assert_eq!(rec.buffers[0].shape, vec![HIDDEN, CONV_L_CACHE]);
        } else {
            assert!(
                matches!(layer_geom, LayerContinuationGeometry::Kv(_)),
                "layer {idx} should be Kv: {layer_geom:?}"
            );
        }
    }

    let store = OperandStore::open(&container, &inspection).unwrap();
    // 8 tokens > SLIDING_WINDOW (3) so sliding window eviction & truncation are active.
    let tokens = [2u32, 7, 13, 4, 19, 1, 25, 11];

    // 1. Batch prefill: Reference vs Production.
    let mut ref_kv = RowKvState::default();
    ref_kv.prepare_continuation(&geometry).unwrap();
    let ref_prefill = prefill_plan(&op_plan, &store, &tokens, &ReferenceBackend, &mut ref_kv)
        .unwrap()
        .logits
        .unwrap();
    let prod = ProductionBackend::new();
    let mut prod_kv = RowKvState::default();
    prod_kv.prepare_continuation(&geometry).unwrap();
    let prod_prefill = prefill_plan(&op_plan, &store, &tokens, &prod, &mut prod_kv)
        .unwrap()
        .logits
        .unwrap();
    assert_eq!(ref_prefill.len(), VOCAB);
    assert!(ref_prefill.iter().all(|v| v.is_finite()));
    assert!(
        max_abs_diff(&ref_prefill, &prod_prefill) < 1e-4,
        "ref vs prod prefill max_abs={}",
        max_abs_diff(&ref_prefill, &prod_prefill)
    );

    // 2. Prefill vs step-by-step decode on Reference and Production.
    let mut ref_sess = DecodeSession::new(
        &op_plan,
        &store,
        &ReferenceBackend,
        Box::new(RowKvState::default()),
    )
    .unwrap();
    let ref_step = ref_sess.step_many(&tokens).unwrap().logits.unwrap();
    assert!(
        max_abs_diff(&ref_prefill, &ref_step) < 1e-5,
        "ref prefill vs step max_abs={}",
        max_abs_diff(&ref_prefill, &ref_step)
    );

    let mut prod_sess =
        DecodeSession::new(&op_plan, &store, &prod, Box::new(RowKvState::default())).unwrap();
    let prod_step = prod_sess.step_many(&tokens).unwrap().logits.unwrap();
    assert!(
        max_abs_diff(&prod_prefill, &prod_step) < 1e-5,
        "prod prefill vs step max_abs={}",
        max_abs_diff(&prod_prefill, &prod_step)
    );

    // 3. Continuation split/resume across ShortConv + Sliding + Full layers:
    //    prefill first 4 tokens into a RowKvState, then resume for the remaining 4.
    let mut split_kv = RowKvState::default();
    split_kv.prepare_continuation(&geometry).unwrap();
    prefill_plan(&op_plan, &store, &tokens[..4], &prod, &mut split_kv).unwrap();
    let mut resumed = DecodeSession::with_kv_state(&op_plan, &store, &prod, &mut split_kv).unwrap();
    let split_logits = resumed.step_many(&tokens[4..]).unwrap().logits.unwrap();
    assert!(
        max_abs_diff(&prod_step, &split_logits) < 1e-5,
        "split/resume diverged: max_abs={}",
        max_abs_diff(&prod_step, &split_logits)
    );

    // 4. Per-head observation: verifies per-head sigmoid gate broadcast on attention
    //    layers (1 and 3), no head records on ShortConv layers (0 and 2), and
    //    bit-identical logits between observed and unobserved execution.
    let mut obs_sess =
        DecodeSession::new(&op_plan, &store, &prod, Box::new(RowKvState::default())).unwrap();
    let mut capture = HeadCapture::default();
    let mut obs_logits = None;
    for &t in &tokens {
        obs_logits = obs_sess.step_observed(t, &mut capture).unwrap().logits;
    }
    assert_eq!(
        obs_logits.unwrap(),
        prod_step,
        "head observation must not perturb logits"
    );
    // 2 attention layers * 8 positions * Q_HEADS records.
    assert_eq!(capture.records.len(), 2 * tokens.len() * Q_HEADS);
    for (layer, _pos, _head, gate) in &capture.records {
        assert!(
            *layer == 1 || *layer == 3,
            "unexpected head record on ShortConv layer {layer}"
        );
        let g = gate.as_ref().expect("per-head gate must be captured");
        assert_eq!(g.len(), HEAD_DIM);
        // PerHead granularity broadcasts one scalar sigmoid per head across all head_dim elements.
        assert!(
            g.iter().all(|&v| v == g[0] && (0.0..=1.0).contains(&v)),
            "per-head gate must be constant in (0, 1) across head_dim: {g:?}"
        );
    }

    // 5. Per-head intervention: zeroing head 0 on layer 1 (sliding) and layer 3 (full)
    //    at the final position fires twice and changes the logits.
    let last_pos = tokens.len() - 1;
    let head_plan = HeadInterventionPlan::none()
        .with(HeadIntervention::zero(
            HeadAddress::new(1, 0, [last_pos]).unwrap(),
        ))
        .unwrap()
        .with(HeadIntervention::zero(
            HeadAddress::new(3, 0, [last_pos]).unwrap(),
        ))
        .unwrap();
    let mut int_sess =
        DecodeSession::new(&op_plan, &store, &prod, Box::new(RowKvState::default())).unwrap();
    for &t in &tokens[..last_pos] {
        int_sess.step(t).unwrap();
    }
    let int_out = int_sess
        .step_intervened(
            tokens[last_pos],
            &mut NoopObserver,
            &InterventionPlan::none(),
            &head_plan,
        )
        .unwrap();
    assert_eq!(int_out.head_firings.len(), 2);
    assert!(
        max_abs_diff(&int_out.logits.unwrap(), &prod_step) > 1e-4,
        "head intervention must measurably alter logits"
    );
}

#[test]
fn lfm3_derivative_without_sandwich_norm_closes_and_executes() {
    let dir = tempfile::tempdir().unwrap();
    write_miniature_lfm3(
        dir.path(),
        Lfm3FixtureOpts {
            use_sandwich_norm: false,
            conv_bias: false,
            tie_word_embeddings: true,
        },
    );
    let out = tempfile::tempdir().unwrap();
    let container = out.path().join("lfm3-preonly.vindex3");
    encode_checkpoint(dir.path(), &container).expect("encode 2-norm LFM3 derivative");

    let inspection = inspect_container(&container, false).unwrap();
    let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
    assert!(outcome.defects.is_empty(), "{:?}", outcome.defects);
    let plan = outcome.plan.expect("op plan closed");
    assert!(plan.layers[0].post_attention_norm.is_none());
    assert!(plan.layers[0].post_ffn_norm.is_none());

    let store = OperandStore::open(&container, &inspection).unwrap();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let tokens = [1u32, 5, 9, 12, 3, 8];
    let mut r_kv = RowKvState::default();
    r_kv.prepare_continuation(&geometry).unwrap();
    let r = prefill_plan(&plan, &store, &tokens, &ReferenceBackend, &mut r_kv)
        .unwrap()
        .logits
        .unwrap();
    let mut p_kv = RowKvState::default();
    p_kv.prepare_continuation(&geometry).unwrap();
    let p = prefill_plan(&plan, &store, &tokens, &ProductionBackend::new(), &mut p_kv)
        .unwrap()
        .logits
        .unwrap();
    assert!(r.iter().all(|v| v.is_finite()));
    assert!(max_abs_diff(&r, &p) < 1e-4);
}

#[test]
fn lfm3_derivative_with_conv_bias_and_untied_head_closes_and_executes() {
    let dir = tempfile::tempdir().unwrap();
    write_miniature_lfm3(
        dir.path(),
        Lfm3FixtureOpts {
            use_sandwich_norm: true,
            conv_bias: true,
            tie_word_embeddings: false,
        },
    );
    let out = tempfile::tempdir().unwrap();
    let container = out.path().join("lfm3-biased-untied.vindex3");
    encode_checkpoint(dir.path(), &container).expect("encode biased untied LFM3 derivative");

    let inspection = inspect_container(&container, false).unwrap();
    let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
    assert!(outcome.defects.is_empty(), "{:?}", outcome.defects);
    let plan = outcome.plan.expect("op plan closed");
    for idx in [0, 2] {
        let LayerAttention::ShortConv(op) = &plan.layers[idx].attention else {
            panic!("layer {idx} must be ShortConv");
        };
        assert!(op.in_proj_bias.is_some());
        assert!(op.conv_bias.is_some());
        assert!(op.out_proj_bias.is_some());
    }

    let store = OperandStore::open(&container, &inspection).unwrap();
    let geometry = plan_continuation_geometry(&plan).unwrap();
    let tokens = [4u32, 10, 2, 18, 7, 15];
    let prod = ProductionBackend::new();
    let mut kv = RowKvState::default();
    kv.prepare_continuation(&geometry).unwrap();
    let batch = prefill_plan(&plan, &store, &tokens, &prod, &mut kv)
        .unwrap()
        .logits
        .unwrap();
    let mut sess =
        DecodeSession::new(&plan, &store, &prod, Box::new(RowKvState::default())).unwrap();
    let stepped = sess.step_many(&tokens).unwrap().logits.unwrap();
    assert!(batch.iter().all(|v| v.is_finite()));
    assert!(
        max_abs_diff(&batch, &stepped) < 1e-5,
        "conv_bias batch vs step max_abs={}",
        max_abs_diff(&batch, &stepped)
    );
}

#[test]
fn stripping_short_conv_surface_reports_named_completeness_defect() {
    let dir = tempfile::tempdir().unwrap();
    write_miniature_lfm3(dir.path(), Lfm3FixtureOpts::default());
    let inventory = build_inventory(dir.path()).unwrap();
    let mut graph = build_from_inventories(&[("lfm3-mini".to_string(), inventory)]).graph;
    let comp = graph
        .components
        .iter_mut()
        .find(|c| c.id == "target")
        .unwrap();
    comp.execution.as_mut().unwrap().short_conv = None;

    let defects: Vec<String> = execution_completeness(&graph)
        .into_iter()
        .filter(|d| {
            matches!(
                d,
                CompletenessDefect::MissingOperationSurface {
                    operation: "short_conv",
                    ..
                }
            )
        })
        .map(|d| d.to_string())
        .collect();
    assert_eq!(
        defects.len(),
        1,
        "expected named short-conv defect: {defects:?}"
    );
}
