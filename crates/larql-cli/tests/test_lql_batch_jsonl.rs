//! Integration test for `larql lql --json` and `larql lql --prompts-jsonl <file>` through the real binary.

use std::process::Command;

use larql_vindex::format::vindex3::encode::checkpoint::encode_checkpoint;
use larql_vindex::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 24;
const VOCAB: usize = 32;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const CONV_L_CACHE: usize = 3;

fn push_bf16(shard: &mut ShardBuilder, name: &str, shape: &[usize], values: &[f32]) {
    let bytes = larql_models::quant::half::encode_bf16(values);
    shard.push_bytes(name, "BF16", shape, &bytes);
}

fn build_fixture() -> (tempfile::TempDir, std::path::PathBuf) {
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
fn lql_json_flag_and_prompts_jsonl_run_raw_probes_without_escaping() {
    let (dir, vindex) = build_fixture();
    let jsonl = dir.path().join("probes.jsonl");
    let lines = [
        serde_json::json!({
            "id": "p_dla",
            "op": "dla",
            "prompt": "alpha \"quoted\"\nbeta",
            "for": "gamma",
            "vs": 1,
            "layers": "0..2",
            "top_heads": 2,
            "top_neurons": 4
        })
        .to_string(),
        serde_json::json!({
            "id": "p_diff",
            "op": "diff_dla",
            "prompt": "alpha beta",
            "for": 3,
            "vs": 1,
            "layers": [0, 1],
            "top_heads": 2
        })
        .to_string(),
        serde_json::json!({
            "id": "p_infer",
            "op": "infer",
            "prompt": "alpha beta",
            "top": 3
        })
        .to_string(),
    ];
    std::fs::write(&jsonl, lines.join("\n")).unwrap();

    let setup = format!(
        r#"USE "{}";"#,
        vindex.display().to_string().replace('\\', "\\\\")
    );

    // 1. Test --json shorthand flag on regular LQL statement
    let out_json = Command::new(env!("CARGO_BIN_EXE_larql"))
        .env("LARQL_VERIFIER", "")
        .args([
            "lql",
            "--json",
            &format!(r#"{setup} INFER "alpha beta" TOP 3;"#),
        ])
        .output()
        .expect("runs larql --json");
    assert_eq!(out_json.status.code(), Some(0));
    let parsed_json: serde_json::Value = serde_json::from_slice(&out_json.stdout).unwrap();
    assert_eq!(parsed_json["ok"], serde_json::json!(true));
    assert_eq!(
        parsed_json["results"][1]["statement"],
        serde_json::json!("INFER")
    );

    // 2. Test --prompts-jsonl batch execution
    let out_batch = Command::new(env!("CARGO_BIN_EXE_larql"))
        .env("LARQL_VERIFIER", "")
        .args(["lql", "--prompts-jsonl", jsonl.to_str().unwrap(), &setup])
        .output()
        .expect("runs larql --prompts-jsonl");
    assert_eq!(
        out_batch.status.code(),
        Some(0),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out_batch.stdout),
        String::from_utf8_lossy(&out_batch.stderr)
    );
    let doc: serde_json::Value = serde_json::from_slice(&out_batch.stdout).unwrap();
    assert_eq!(doc["ok"], serde_json::json!(true));
    let probes = doc["probes"].as_array().unwrap();
    assert_eq!(probes.len(), 3);
    assert_eq!(probes[0]["id"], serde_json::json!("p_dla"));
    assert_eq!(probes[0]["data"]["top_heads"].as_array().unwrap().len(), 2);
    assert_eq!(probes[1]["id"], serde_json::json!("p_diff"));
    assert_eq!(
        probes[1]["data"]["delta_logit_diff"],
        serde_json::json!(0.0)
    );
    assert_eq!(probes[2]["id"], serde_json::json!("p_infer"));
    assert_eq!(probes[2]["data"]["top"].as_array().unwrap().len(), 3);
}
