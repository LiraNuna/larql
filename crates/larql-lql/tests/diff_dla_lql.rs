//! Integration tests for native `DIFF DLA` (comparing active overlay / soft prompt against
//! the untouched bound Base container in one hot session).

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
fn parses_diff_dla_and_computes_exact_deltas_against_base() {
    let stmt = parse(
        r#"DIFF DLA "alpha beta" FOR "gamma" VS ID 1 AT LAYERS (0..2) TOP 4 NEURONS TOP 2 HEADS TOP 2 CHANNELS;"#,
    )
    .unwrap();
    assert!(!stmt.writes_artifact());
    assert_eq!(stmt.verb(), "DIFF");
    assert!(matches!(stmt, Statement::DiffDla { .. }));

    let (dir, vindex) = build_lfm3_fixture();
    let sp_path = dir.path().join("sp.npy");
    let sp_vals = lcg_values(3 * HIDDEN, 333);
    let raw_f32: Vec<u8> = sp_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(
        &sp_path,
        npy_file_bytes(NpyDescr::F4, &[3, HIDDEN], &raw_f32).unwrap(),
    )
    .unwrap();

    let mut sess = Session::new();
    sess.execute(&parse(&format!(r#"USE "{}";"#, q(&vindex))).unwrap())
        .unwrap();

    // Unmodified session: all deltas must be exact 0.0!
    let zero_diff = sess
        .diff_dla(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
            None,
            Some(4),
            Some(4),
            Some(4),
        )
        .unwrap();
    assert!(!zero_diff.candidate_has_weight_edits);
    assert!(!zero_diff.candidate_has_soft_prompt);
    assert_eq!(zero_diff.delta_logit_diff, 0.0);
    assert!(zero_diff.head_deltas.iter().all(|h| h.delta == 0.0));

    // Load soft prompt and run DIFF DLA through JSON batch mode!
    let batch = run_batch_json_in(
        &mut sess,
        &format!(
            r#"LOAD SOFT_PROMPT "{}"; DIFF DLA "alpha beta" FOR "gamma" VS "alpha" AT LAYERS (0..2) TOP 4 NEURONS TOP 4 HEADS TOP 4 CHANNELS;"#,
            q(&sp_path)
        ),
    );
    assert_eq!(batch.errors, 0, "{:?}", batch.results);
    let diff_entry = &batch.results[1];
    assert_eq!(diff_entry["statement"], serde_json::json!("DIFF DLA"));
    let data = &diff_entry["data"];
    assert_eq!(data["candidate_has_soft_prompt"], serde_json::json!(true));
    assert_eq!(data["candidate_has_weight_edits"], serde_json::json!(false));
    let base_ld = data["base_logit_diff"].as_f64().unwrap();
    let cand_ld = data["candidate_logit_diff"].as_f64().unwrap();
    let delta_ld = data["delta_logit_diff"].as_f64().unwrap();
    assert!((delta_ld - (cand_ld - base_ld)).abs() < 1e-5);
    assert!(delta_ld.abs() > 1e-4, "soft prompt should move logit diff");
    assert_eq!(data["head_deltas"].as_array().unwrap().len(), 4);
    assert_eq!(data["column_deltas"].as_array().unwrap().len(), 4);
    assert_eq!(data["channel_deltas"].as_array().unwrap().len(), 4);
}
