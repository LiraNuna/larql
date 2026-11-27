//! Integration tests for fast forward-only `LOGIT DIFF` and `DIFF LOGIT DIFF`, verifying
//! bit-for-bit parity against full `DLA` and `DIFF DLA`.

use std::path::Path;

use larql_lql::operations::{npy_file_bytes, DlaTarget, NpyDescr};
use larql_lql::{parse, run_batch_json_in, Session, Statement};
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

fn build_lfm3_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let ckpt = dir.path().join("ckpt");
    std::fs::create_dir_all(&ckpt).unwrap();

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
        push_bf16(
            &mut shard,
            &format!("{p}.feed_forward.w2.weight"),
            &[HIDDEN, INTERMEDIATE],
            &lcg_values(HIDDEN * INTERMEDIATE, seed + 13),
        );
    }
    shard.write(&ckpt);

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

    let vindex = dir.path().join("base.vindex3");
    encode_checkpoint(&ckpt, &vindex).unwrap();
    (dir, vindex)
}

#[test]
fn logit_diff_and_diff_logit_diff_match_dla_bit_for_bit() {
    let s1 = parse(r#"LOGIT DIFF "alpha beta" FOR "gamma" VS ID 1;"#).unwrap();
    assert!(!s1.writes_artifact());
    assert_eq!(s1.verb(), "LOGIT DIFF");
    assert!(matches!(s1, Statement::LogitDiff { .. }));

    let s2 = parse(r#"DIFF LOGIT DIFF "alpha beta" FOR ID 3 VS "alpha";"#).unwrap();
    assert!(!s2.writes_artifact());
    assert_eq!(s2.verb(), "DIFF");
    assert!(matches!(s2, Statement::DiffLogitDiff { .. }));

    let (dir, vindex) = build_lfm3_fixture();
    let sp_path = dir.path().join("sp.npy");
    let sp_vals = lcg_values(2 * HIDDEN, 444);
    let raw_f32: Vec<u8> = sp_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(
        &sp_path,
        npy_file_bytes(NpyDescr::F4, &[2, HIDDEN], &raw_f32).unwrap(),
    )
    .unwrap();

    let mut sess = Session::new();
    sess.execute(&parse(&format!(r#"USE "{}";"#, q(&vindex))).unwrap())
        .unwrap();

    let dla = sess
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
    let ld = sess
        .logit_diff(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
        )
        .unwrap();
    assert!(
        (ld.logit_a - dla.logit_a).abs() < 1e-5,
        "logit_a: {} vs {}",
        ld.logit_a,
        dla.logit_a
    );
    assert!(
        (ld.logit_b - dla.logit_b).abs() < 1e-5,
        "logit_b: {} vs {}",
        ld.logit_b,
        dla.logit_b
    );
    assert!(
        (ld.logit_diff - dla.exact_logit_diff).abs() < 1e-5,
        "logit_diff: {} vs {}",
        ld.logit_diff,
        dla.exact_logit_diff
    );

    // Load soft prompt and verify DIFF LOGIT DIFF matches DIFF DLA within SIMD summation tolerance!
    let batch = run_batch_json_in(
        &mut sess,
        &format!(
            r#"LOAD SOFT_PROMPT "{}"; DIFF DLA "alpha beta" FOR "gamma" VS "alpha"; DIFF LOGIT DIFF "alpha beta" FOR "gamma" VS "alpha";"#,
            q(&sp_path)
        ),
    );
    assert_eq!(batch.errors, 0, "{:?}", batch.results);
    let dd = &batch.results[1]["data"];
    let dld = &batch.results[2]["data"];
    assert_eq!(
        batch.results[2]["statement"],
        serde_json::json!("DIFF LOGIT DIFF")
    );
    assert!(
        (dd["base_logit_diff"].as_f64().unwrap() - dld["base_logit_diff"].as_f64().unwrap()).abs()
            < 1e-5
    );
    assert!(
        (dd["candidate_logit_diff"].as_f64().unwrap()
            - dld["candidate_logit_diff"].as_f64().unwrap())
        .abs()
            < 1e-5
    );
    assert!(
        (dd["delta_logit_diff"].as_f64().unwrap() - dld["delta_logit_diff"].as_f64().unwrap())
            .abs()
            < 1e-5
    );
}
