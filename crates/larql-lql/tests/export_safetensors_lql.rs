//! Integration tests for in-session `EXPORT SAFETENSORS` (including live staged edits
//! without an intermediate `COMPILE CURRENT INTO VINDEX`).

use std::path::Path;

use larql_lql::operations::{npy_file_bytes, NpyDescr};
use larql_lql::{parse, run_batch_outcome_in, Session, Statement};
use larql_vindex::format::vindex3::encode::checkpoint::encode_checkpoint;
use larql_vindex::format::vindex3::export_safetensors::EXPORT_MANIFEST_JSON;
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

fn build_lfm3_fixture() -> (tempfile::TempDir, std::path::PathBuf, std::path::PathBuf) {
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
    (dir, ckpt, vindex)
}

fn write_f32_npy(path: &Path, values: &[f32]) {
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = npy_file_bytes(NpyDescr::F4, &[values.len()], &raw).unwrap();
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn parses_export_safetensors_forms_and_marks_artifact_write() {
    let a = parse(r#"EXPORT SAFETENSORS "out_hf";"#).unwrap();
    assert!(a.writes_artifact());
    assert_eq!(a.verb(), "EXPORT SAFETENSORS");
    assert!(matches!(
        a,
        Statement::ExportSafetensors {
            ref output,
            base_checkpoint: None
        } if output == "out_hf"
    ));

    let b = parse(r#"EXPORT SAFETENSORS TO "out_hf" FROM BASE "base_hf";"#).unwrap();
    assert!(matches!(
        b,
        Statement::ExportSafetensors {
            ref output,
            base_checkpoint: Some(ref base),
        } if output == "out_hf" && base == "base_hf"
    ));

    let c = parse(r#"EXPORT SAFETENSORS "out_hf" FROM "base_hf";"#).unwrap();
    assert!(matches!(
        c,
        Statement::ExportSafetensors {
            ref output,
            base_checkpoint: Some(ref base),
        } if output == "out_hf" && base == "base_hf"
    ));
}

#[test]
fn live_staged_edit_exports_directly_to_hf_safetensors_with_bit_exact_infer_parity() {
    let (dir, ckpt, vindex) = build_lfm3_fixture();
    let vec_path = dir.path().join("v.npy");
    let mut v: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    write_f32_npy(&vec_path, &v);

    let hf_out = dir.path().join("exported_hf");
    let mut session = Session::new();
    session
        .execute(&parse(&format!(r#"USE "{}";"#, q(&vindex))).unwrap())
        .unwrap();

    // Stage a live REMOVE INPUT DIRECTION edit (NO intermediate COMPILE!).
    session
        .execute(
            &parse(&format!(
                r#"REMOVE INPUT DIRECTION "{}" FROM "1.self_attn.k_proj.weight" HEAD 1 SCALE 1.0 NUMERICS "f64-seq-rne1/v1";"#,
                q(&vec_path)
            ))
            .unwrap(),
        )
        .unwrap();

    let live_pred = session.predict("alpha beta", 4).unwrap();

    // Export directly from the live session into a complete HF checkpoint directory.
    let export_lines = session
        .execute(
            &parse(&format!(
                r#"EXPORT SAFETENSORS "{}" FROM BASE "{}";"#,
                q(&hf_out),
                q(&ckpt)
            ))
            .unwrap(),
        )
        .unwrap();
    let joined = export_lines.join("\n");
    assert!(joined.contains("verified:         true"), "{joined}");
    assert!(joined.contains("live_patched:     1 tensor(s)"), "{joined}");

    assert!(hf_out.join("model.safetensors").is_file());
    assert!(hf_out.join("config.json").is_file());
    assert!(hf_out.join("tokenizer.json").is_file());
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(hf_out.join(EXPORT_MANIFEST_JSON)).unwrap())
            .unwrap();
    assert_eq!(manifest["verified"], serde_json::json!(true));
    assert_eq!(manifest["live_overridden_tensors"], serde_json::json!(1));

    // Re-encode the exported HF directory into a brand-new VINDEX3 container and verify
    // INFER matches the live patched session bit-for-bit!
    let reencoded = dir.path().join("reencoded.vindex3");
    encode_checkpoint(&hf_out, &reencoded).unwrap();

    let mut fresh = Session::new();
    fresh
        .execute(&parse(&format!(r#"USE "{}";"#, q(&reencoded))).unwrap())
        .unwrap();
    let reencoded_pred = fresh.predict("alpha beta", 4).unwrap();
    assert_eq!(live_pred.top.len(), reencoded_pred.top.len());
    for (a, b) in live_pred.top.iter().zip(reencoded_pred.top.iter()) {
        assert_eq!(a.token_id, b.token_id);
        assert_eq!(a.probability.to_bits(), b.probability.to_bits());
    }

    // Refuses overwriting an existing output directory.
    let err = session
        .execute(&parse(&format!(r#"EXPORT SAFETENSORS "{}";"#, q(&hf_out))).unwrap())
        .unwrap_err()
        .to_string();
    assert!(err.contains("already exists"), "{err}");

    // Batch write guard: refused after an earlier statement fails.
    let blocked_out = dir.path().join("blocked_hf");
    let mut batch_sess = Session::new();
    let outcome = run_batch_outcome_in(
        &mut batch_sess,
        &format!(
            r#"USE "{}"; REMOVE DIRECTION "missing.npy" FROM "1.self_attn.o_proj.weight" HEAD 0 SCALE 1.0 NUMERICS "f64-seq-rne1/v1"; EXPORT SAFETENSORS "{}";"#,
            q(&vindex),
            q(&blocked_out)
        ),
    );
    assert!(outcome.errors >= 2);
    assert!(
        !blocked_out.exists(),
        "write guard must prevent export after earlier failure"
    );
}
