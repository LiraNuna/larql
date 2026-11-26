//! Integration tests for the native `DLA` LQL verb.

use std::path::Path;

use larql_lql::{parse, Session, Statement};
use larql_vindex::format::vindex3::encode::checkpoint::encode_checkpoint;
use larql_vindex::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 24;
const VOCAB: usize = 32;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const CONV_L_CACHE: usize = 3;

fn lql_path(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn push_bf16(shard: &mut ShardBuilder, name: &str, shape: &[usize], values: &[f32]) {
    let bytes = larql_models::quant::half::encode_bf16(values);
    shard.push_bytes(name, "BF16", shape, &bytes);
}

fn lfm3_session() -> (tempfile::TempDir, Session) {
    let dir = tempfile::tempdir().unwrap();
    let ckpt = dir.path().join("ckpt");
    std::fs::create_dir_all(&ckpt).unwrap();

    let config = serde_json::json!({
        "architectures": ["Lfm3ForCausalLM"],
        "model_type": "lfm3",
        "torch_dtype": "bfloat16",
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
        "conv_bias": false,
        "use_sandwich_norm": true,
        "sliding_window": 3,
        "tie_word_embeddings": true,
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
    std::fs::write(ckpt.join("config.json"), config.to_string()).unwrap();

    let mut shard = ShardBuilder::new();
    push_bf16(
        &mut shard,
        "model.embed_tokens.weight",
        &[VOCAB, HIDDEN],
        &lcg_values(VOCAB * HIDDEN, 1),
    );
    push_bf16(
        &mut shard,
        "model.embedding_norm.weight",
        &[HIDDEN],
        &norm_values(HIDDEN, 2),
    );

    let layer_kinds = ["conv", "sliding_attention", "conv", "full_attention"];
    for (layer, &kind) in layer_kinds.iter().enumerate() {
        let seed = 100 + layer as u64 * 30;
        let p = format!("model.layers.{layer}");
        push_bf16(
            &mut shard,
            &format!("{p}.operator_norm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.post_operator_norm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 1),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.post_ffn_norm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 2),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.ffn_norm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 3),
        );
        if kind == "conv" {
            push_bf16(
                &mut shard,
                &format!("{p}.conv.in_proj.weight"),
                &[3 * HIDDEN, HIDDEN],
                &lcg_values(3 * HIDDEN * HIDDEN, seed + 4),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.conv.conv.weight"),
                &[HIDDEN, 1, CONV_L_CACHE],
                &lcg_values(HIDDEN * CONV_L_CACHE, seed + 5),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.conv.out_proj.weight"),
                &[HIDDEN, HIDDEN],
                &lcg_values(HIDDEN * HIDDEN, seed + 6),
            );
        } else {
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.q_proj.weight"),
                &[Q_HEADS * HEAD_DIM, HIDDEN],
                &lcg_values(Q_HEADS * HEAD_DIM * HIDDEN, seed + 4),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.k_proj.weight"),
                &[KV_HEADS * HEAD_DIM, HIDDEN],
                &lcg_values(KV_HEADS * HEAD_DIM * HIDDEN, seed + 5),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.v_proj.weight"),
                &[KV_HEADS * HEAD_DIM, HIDDEN],
                &lcg_values(KV_HEADS * HEAD_DIM * HIDDEN, seed + 6),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.o_proj.weight"),
                &[HIDDEN, Q_HEADS * HEAD_DIM],
                &lcg_values(HIDDEN * Q_HEADS * HEAD_DIM, seed + 7),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.q_norm.weight"),
                &[HEAD_DIM],
                &norm_values(HEAD_DIM, seed + 8),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.k_norm.weight"),
                &[HEAD_DIM],
                &norm_values(HEAD_DIM, seed + 9),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.gate_proj.weight"),
                &[Q_HEADS, HIDDEN],
                &lcg_values(Q_HEADS * HIDDEN, seed + 10),
            );
        }
        push_bf16(
            &mut shard,
            &format!("{p}.feed_forward.w1.weight"),
            &[INTERMEDIATE, HIDDEN],
            &lcg_values(INTERMEDIATE * HIDDEN, seed + 11),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.feed_forward.w2.weight"),
            &[HIDDEN, INTERMEDIATE],
            &lcg_values(HIDDEN * INTERMEDIATE, seed + 12),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.feed_forward.w3.weight"),
            &[INTERMEDIATE, HIDDEN],
            &lcg_values(INTERMEDIATE * HIDDEN, seed + 13),
        );
    }
    shard.write(&ckpt);

    let container = dir.path().join("lfm3-bf16.vindex3");
    encode_checkpoint(&ckpt, &container).expect("encode BF16 LFM3 container");

    let vocab: serde_json::Map<String, serde_json::Value> =
        (0..VOCAB).map(|i| (format!("[{i}]"), i.into())).collect();
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[0]"}
    });
    std::fs::write(container.join("tokenizer.json"), tokenizer.to_string()).unwrap();

    let mut session = Session::new();
    let bind = format!("USE \"{}\";", lql_path(&container));
    session.execute(&parse(&bind).unwrap()).unwrap();
    (dir, session)
}

#[test]
fn parses_all_dla_forms() {
    for src in [
        r#"DLA "[1] [2]" FOR "[5]" VS "[6]";"#,
        r#"DLA "[1] [2]" FOR "[5]" VS "[6]" LAYERS 0-3;"#,
        r#"DLA "[1] [2]" FOR "[5]" VS "[6]" LAYERS 0-3 TOP 8 COLUMNS;"#,
        r#"DLA "[1] [2]" FOR "[5]" VS "[6]" AT LAYER 1 TOP 4;"#,
        r#"DLA "[1] [2]" FOR "[5]" VS "[6]" AT LAYERS (0, 2, 3) TOP 6 COLUMNS;"#,
        r#"DLA "[1] [2]" FOR "[5]" VS "[6]" LAYERS ALL TOP 3 COLUMNS;"#,
        r#"DLA "[1] [2]" FOR "[5]" VS "[6]" AT LAYERS (1, 3) TOP 4 HEADS;"#,
        r#"DLA "[1] [2]" FOR "[5]" VS "[6]" LAYERS 0-3 TOP 6 COLUMNS TOP 4 HEADS;"#,
        r#"DLA "[1] [2]" FOR ID 5 VS ID 12;"#,
        r#"DLA "[1] [2]" FOR "[5]" VS ID 12 AT LAYERS (1, 3) TOP 4 HEADS;"#,
        r#"DLA "[1] [2]" FOR ID 5 VS "[12]" LAYERS 0-3 TOP 6 COLUMNS;"#,
    ] {
        let stmt = parse(src).unwrap_or_else(|e| panic!("parse {src}: {e}"));
        assert!(matches!(stmt, Statement::Dla { .. }), "{stmt:?}");
    }
}

#[test]
fn dla_decomposition_is_exact_at_layer_and_column_level_on_hybrid_short_conv() {
    let (_dir, mut session) = lfm3_session();

    // Ask for all columns across all 4 layers (4 * 24 = 96 total) AND all heads across the 2 softmax layers (2 * 4 = 8).
    let report = session
        .dla(
            "[3] [7] [11] [15]",
            "[5]",
            "[12]",
            None,
            Some(200),
            Some(100),
        )
        .expect(
            "DLA on hybrid ShortConv + Softmax with sandwich post_ffn_norm and post_operator_norm",
        );

    assert_eq!(report.layers.len(), 4);
    assert_eq!(report.layers[0].mixer, "short-conv");
    assert_eq!(report.layers[1].mixer, "sliding");
    assert_eq!(report.layers[2].mixer, "short-conv");
    assert_eq!(report.layers[3].mixer, "full");
    assert_eq!(report.dense_ffn_layers_attributed, 4);
    assert_eq!(report.softmax_layers_attributed, 2);
    assert_eq!(report.top_heads.len(), 2 * Q_HEADS);

    // Exact additive identity across the stack: emb + sum(op + ffn) == pre_softcap_diff == exact_logit_diff.
    assert!(
        (report.total_dla - report.exact_logit_diff).abs() < 1e-3,
        "total DLA {} must match exact logit diff {}",
        report.total_dla,
        report.exact_logit_diff
    );

    // Exact additive identity at the column level through sandwich post_ffn_norm:
    // sum_c col_dla[l, c] == ffn_dla[l] for every dense FFN layer.
    for layer_row in &report.layers {
        let col_sum: f32 = report
            .top_columns
            .iter()
            .filter(|c| c.layer == layer_row.layer)
            .map(|c| c.contribution)
            .sum();
        assert!(
            (col_sum - layer_row.ffn_dla).abs() < 1e-3,
            "layer {}: column DLA sum {col_sum} must match ffn_dla {}",
            layer_row.layer,
            layer_row.ffn_dla
        );
    }

    // Exact additive identity at the attention-head level through sandwich post_operator_norm
    // and per-head sigmoid output_gate: sum_h head_dla[l, h] == operator_dla[l] for every softmax layer!
    for &softmax_layer in &[1usize, 3usize] {
        let head_sum: f32 = report
            .top_heads
            .iter()
            .filter(|h| h.layer == softmax_layer)
            .map(|h| h.contribution)
            .sum();
        let op_dla = report.layers[softmax_layer].operator_dla;
        assert!(
            (head_sum - op_dla).abs() < 1e-3,
            "layer {softmax_layer}: head DLA sum {head_sum} must match operator_dla {op_dla}"
        );
    }

    // Heads and columns are ranked strictly by |contribution| descending.
    for w in report.top_heads.windows(2) {
        assert!(
            w[0].contribution.abs() >= w[1].contribution.abs(),
            "heads not sorted by |contribution|: {:?} vs {:?}",
            w[0],
            w[1]
        );
    }
    for w in report.top_columns.windows(2) {
        assert!(
            w[0].contribution.abs() >= w[1].contribution.abs(),
            "columns not sorted by |contribution|: {:?} vs {:?}",
            w[0],
            w[1]
        );
    }

    // Statement execution formats cleanly with both columns and heads.
    let lines = session
        .execute(
            &parse(r#"DLA "[3] [7] [11] [15]" FOR "[5]" VS "[12]" LAYERS 0-3 TOP 5 COLUMNS TOP 4 HEADS;"#)
                .unwrap(),
        )
        .unwrap();
    let text = lines.join("\n");
    assert!(text.contains("Direct Logit Attribution"), "{text}");
    assert!(text.contains("short-conv"), "{text}");
    assert!(text.contains("Top 5 FFN columns"), "{text}");
    assert!(text.contains("Top 4 attention heads"), "{text}");
}

#[test]
fn dla_executes_live_patched_weights_and_reverts_on_discard() {
    let (_dir, mut session) = lfm3_session();
    let base = session
        .dla("[3] [7] [11] [15]", "[5]", "[12]", None, Some(10), Some(4))
        .unwrap();
    assert!(!base.executes_patched);

    let stage = "UPDATE WEIGHTS SET DELTA = 16.0 WHERE TENSOR = \"3.feed_forward.w2.weight\" AND ROW = 0 AND COL IN (0, 1, 2, 3);";
    session.execute(&parse(stage).unwrap()).unwrap();

    let patched = session
        .dla("[3] [7] [11] [15]", "[5]", "[12]", None, Some(10), Some(4))
        .unwrap();
    assert!(patched.executes_patched);
    assert_ne!(
        patched.exact_logit_diff.to_bits(),
        base.exact_logit_diff.to_bits(),
        "patched DLA must reflect staged weight edits"
    );
    assert!(
        (patched.total_dla - patched.exact_logit_diff).abs() < 1e-2,
        "patched DLA sum {} must still equal patched exact diff {}",
        patched.total_dla,
        patched.exact_logit_diff
    );

    session
        .execute(&parse("DISCARD WEIGHT EDITS;").unwrap())
        .unwrap();
    let restored = session
        .dla("[3] [7] [11] [15]", "[5]", "[12]", None, Some(10), Some(4))
        .unwrap();
    assert!(!restored.executes_patched);
    assert_eq!(
        restored.exact_logit_diff.to_bits(),
        base.exact_logit_diff.to_bits()
    );
}

#[test]
fn dla_runs_under_auto_source_with_hybrid_live_patches_and_cached_operands() {
    // Build a fresh container and bind under RepresentationSource::Auto.
    let (_auto_dir, mut auto_session) = {
        let (d, s) = lfm3_session();
        let container = d.path().join("lfm3-bf16.vindex3");
        drop(s);
        let mut auto_s = Session::new();
        auto_s.set_representation_source(larql_lql::RepresentationSource::Auto);
        let bind = format!("USE \"{}\";", lql_path(&container));
        auto_s.execute(&parse(&bind).unwrap()).unwrap();
        (d, auto_s)
    };

    let base_auto = auto_session
        .dla(
            "[3] [7] [11] [15]",
            "[5]",
            "[12]",
            None,
            Some(200),
            Some(100),
        )
        .expect("DLA runs cleanly under auto representation source");
    let base_auto_cached = auto_session
        .dla(
            "[3] [7] [11] [15]",
            "[5]",
            "[12]",
            None,
            Some(200),
            Some(100),
        )
        .expect("second DLA reuses resident prepared operands");
    assert_eq!(
        base_auto.exact_logit_diff.to_bits(),
        base_auto_cached.exact_logit_diff.to_bits()
    );
    assert!(
        (base_auto.total_dla - base_auto.exact_logit_diff).abs() < 1e-3,
        "auto DLA sum {} must match exact diff {}",
        base_auto.total_dla,
        base_auto.exact_logit_diff
    );

    // Stage a live patch under auto (hybrid exact BF16 on patched tensor + auto on unpatched).
    let stage = "UPDATE WEIGHTS SET DELTA = 16.0 WHERE TENSOR = \"3.feed_forward.w2.weight\" AND ROW = 0 AND COL IN (0, 1, 2, 3);";
    auto_session.execute(&parse(stage).unwrap()).unwrap();
    let patched_auto = auto_session
        .dla(
            "[3] [7] [11] [15]",
            "[5]",
            "[12]",
            None,
            Some(200),
            Some(100),
        )
        .expect("hybrid auto patched DLA succeeds");
    assert!(patched_auto.executes_patched);
    assert_ne!(
        patched_auto.exact_logit_diff.to_bits(),
        base_auto.exact_logit_diff.to_bits()
    );
    assert!(
        (patched_auto.total_dla - patched_auto.exact_logit_diff).abs() < 1e-2,
        "hybrid auto patched DLA sum {} must match exact diff {}",
        patched_auto.total_dla,
        patched_auto.exact_logit_diff
    );
}

#[test]
fn dla_token_id_targets_match_text_targets_bit_for_bit_and_refuse_oob() {
    let (_dir, mut session) = lfm3_session();
    let by_text = session
        .dla("[3] [7] [11] [15]", "[5]", "[12]", None, Some(10), Some(8))
        .unwrap();
    let by_id = session
        .dla("[3] [7] [11] [15]", 5u32, 12u32, None, Some(10), Some(8))
        .unwrap();
    assert_eq!(by_id.token_id_a, 5);
    assert_eq!(by_id.token_id_b, 12);
    assert_eq!(by_id.target_a, "[5]");
    assert_eq!(by_id.target_b, "[12]");
    assert_eq!(
        by_id.exact_logit_diff.to_bits(),
        by_text.exact_logit_diff.to_bits()
    );
    assert_eq!(by_id.total_dla.to_bits(), by_text.total_dla.to_bits());
    assert_eq!(by_id.top_heads, by_text.top_heads);
    assert_eq!(by_id.top_columns, by_text.top_columns);

    let lines = session
        .execute(
            &parse(r#"DLA "[3] [7] [11] [15]" FOR ID 5 VS ID 12 AT LAYERS (1, 3) TOP 4 HEADS;"#)
                .unwrap(),
        )
        .unwrap();
    let rendered = lines.join("\n");
    assert!(
        rendered.contains("[id 5]") && rendered.contains("[id 12]"),
        "{rendered}"
    );

    let err = session
        .execute(&parse(r#"DLA "[3] [7]" FOR ID 9999 VS ID 12;"#).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("outside vocabulary"), "{err}");
}

#[test]
fn dla_refuses_multi_token_targets_and_bad_layers() {
    let (_dir, mut session) = lfm3_session();
    let err = session
        .dla("[3] [7]", "[5] [6]", "[12]", None, None, None)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("exactly 1 token") && err.contains("FOR ID 5"),
        "{err}"
    );

    let err = session
        .dla("[3] [7]", "[5]", "", None, None, None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exactly 1 token"), "{err}");

    let err = session
        .execute(&parse(r#"DLA "[3] [7]" FOR "[5]" VS "[12]" AT LAYER 99;"#).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("outside"), "{err}");

    let err = session
        .execute(&parse(r#"DLA "[3] [7]" FOR "[5]" VS "[12]" TOP 0 COLUMNS;"#).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("at least 1"), "{err}");

    let err = session
        .execute(&parse(r#"DLA "[3] [7]" FOR "[5]" VS "[12]" TOP 0 HEADS;"#).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("at least 1"), "{err}");
}
