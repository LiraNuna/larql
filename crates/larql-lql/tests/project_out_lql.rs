//! Integration tests for exact sandwich-norm-aware `PROJECT OUT <tokA> - <tokB> ON LAYERS (...)`
//! including live composition over `OVERLAY SAFETENSORS` and export via `EXPORT SAFETENSORS`.

use std::path::Path;

use larql_lql::ast::{DlaTarget, ProjectOutTarget};
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

fn q(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn push_bf16(shard: &mut ShardBuilder, name: &str, shape: &[usize], values: &[f32]) {
    let bytes = larql_models::quant::half::encode_bf16(values);
    shard.push_bytes(name, "BF16", shape, &bytes);
}

fn write_config_and_tok(ckpt: &Path) {
    let config = serde_json::json!({
        "architectures": ["Lfm3ForCausalLM"],
        "model_type": "lfm3",
        "torch_dtype": "bfloat16",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": 2,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "norm_eps": 1e-5,
        "conv_L_cache": CONV_L_CACHE,
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
    std::fs::write(ckpt.join("config.json"), config.to_string()).unwrap();

    let mut vocab = serde_json::Map::new();
    for id in 0..VOCAB {
        let tok = match id {
            0 => "<bos>".to_string(),
            1 => "alpha".to_string(),
            2 => "beta".to_string(),
            3 => "gamma".to_string(),
            other => format!("tok{other}"),
        };
        vocab.insert(tok, serde_json::json!(id));
    }
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "WhitespaceSplit" },
        "post_processor": null,
        "decoder": null,
        "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "<bos>" }
    });
    std::fs::write(ckpt.join("tokenizer.json"), tokenizer.to_string()).unwrap();
}

fn write_shard(ckpt: &Path, w2_l1_delta: f32) {
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
    for (layer, &kind) in ["conv", "full_attention"].iter().enumerate() {
        let seed = 200 + layer as u64 * 30;
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
            &format!("{p}.feed_forward.w3.weight"),
            &[INTERMEDIATE, HIDDEN],
            &lcg_values(INTERMEDIATE * HIDDEN, seed + 12),
        );
        let mut w2 = lcg_values(HIDDEN * INTERMEDIATE, seed + 13);
        if layer == 1 && w2_l1_delta != 0.0 {
            for (i, v) in w2.iter_mut().enumerate() {
                if i % 5 == 0 {
                    *v += w2_l1_delta;
                }
            }
        }
        push_bf16(
            &mut shard,
            &format!("{p}.feed_forward.w2.weight"),
            &[HIDDEN, INTERMEDIATE],
            &w2,
        );
    }
    shard.write(ckpt);
}

#[test]
fn parses_project_out_forms() {
    let s1 = parse(r#"PROJECT OUT "gamma" - "alpha" ON LAYERS (0..2);"#).unwrap();
    assert!(!s1.writes_artifact());
    assert_eq!(s1.verb(), "PROJECT OUT");
    assert!(matches!(
        s1,
        Statement::ProjectOut {
            target: ProjectOutTarget::W2,
            scale,
            ..
        } if (scale - 1.0).abs() < 1e-12
    ));

    let s2 = parse(r#"PROJECT OUT ID 3 - ID 1 AT LAYER 1 TARGET ALL SCALE 0.75;"#).unwrap();
    assert!(matches!(
        s2,
        Statement::ProjectOut {
            target: ProjectOutTarget::All,
            scale,
            ..
        } if (scale - 0.75).abs() < 1e-12
    ));
}

#[test]
fn project_out_zeroes_sandwich_norm_ffn_dla_and_composes_over_overlay_and_export() {
    let dir = tempfile::tempdir().unwrap();
    let base_ckpt = dir.path().join("base_ckpt");
    let cand_ckpt = dir.path().join("cand_ckpt");
    std::fs::create_dir_all(&base_ckpt).unwrap();
    std::fs::create_dir_all(&cand_ckpt).unwrap();
    write_config_and_tok(&base_ckpt);
    write_config_and_tok(&cand_ckpt);
    write_shard(&base_ckpt, 0.0);
    write_shard(&cand_ckpt, 0.25);

    let base_vindex = dir.path().join("base.vindex3");
    encode_checkpoint(&base_ckpt, &base_vindex).unwrap();

    let mut sess = Session::new();
    sess.execute(&parse(&format!(r#"USE "{}";"#, q(&base_vindex))).unwrap())
        .unwrap();

    // Mount candidate checkpoint as an in-memory overlay!
    sess.execute(&parse(&format!(r#"OVERLAY SAFETENSORS "{}";"#, q(&cand_ckpt))).unwrap())
        .unwrap();

    let before_dla = sess
        .dla(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
            None,
            None,
            None,
            None,
        )
        .unwrap();
    assert!(
        before_dla.layers[1].ffn_dla.abs() > 1e-3,
        "L1 FFN DLA should be nonzero before projection: {}",
        before_dla.layers[1].ffn_dla
    );

    // Project out `gamma - alpha` on L1 W2 (composing directly over the overlaid w2 bytes!).
    sess.execute(
        &parse(r#"PROJECT OUT "gamma" - "alpha" ON LAYER 1 TARGET W2 SCALE 1.0;"#).unwrap(),
    )
    .unwrap();

    let after_dla = sess
        .dla(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
            None,
            None,
            None,
            None,
        )
        .unwrap();
    // Because of BF16 rounding across 24 columns, L1 FFN DLA must collapse to near-zero!
    assert!(
        after_dla.layers[1].ffn_dla.abs() < 0.02,
        "projected L1 FFN DLA should be near zero (was {}, now {})",
        before_dla.layers[1].ffn_dla,
        after_dla.layers[1].ffn_dla
    );
    assert!(
        after_dla.layers[1].ffn_dla.abs() < before_dla.layers[1].ffn_dla.abs() * 0.05,
        "expected >=95% reduction in L1 FFN DLA (before={}, after={})",
        before_dla.layers[1].ffn_dla,
        after_dla.layers[1].ffn_dla
    );

    // Export composed (OVERLAY + PROJECT OUT) directly to HF safetensors and verify bit-exact parity!
    let live_ld = sess
        .logit_diff(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
        )
        .unwrap();
    let exported_hf = dir.path().join("exported_projected");
    sess.execute(
        &parse(&format!(
            r#"EXPORT SAFETENSORS "{}" FROM BASE "{}";"#,
            q(&exported_hf),
            q(&cand_ckpt)
        ))
        .unwrap(),
    )
    .unwrap();
    let reenc = dir.path().join("reenc_projected.vindex3");
    encode_checkpoint(&exported_hf, &reenc).unwrap();

    let mut fresh = Session::new();
    fresh
        .execute(&parse(&format!(r#"USE "{}";"#, q(&reenc))).unwrap())
        .unwrap();
    let reenc_ld = fresh
        .logit_diff(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
        )
        .unwrap();
    assert_eq!(live_ld.logit_diff.to_bits(), reenc_ld.logit_diff.to_bits());
}
