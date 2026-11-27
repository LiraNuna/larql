//! Integration tests for in-memory `OVERLAY SAFETENSORS`, `OVERLAY LORA`, and `OVERLAY CLEAR`.

use std::path::Path;

use larql_lql::ast::{DlaTarget, OverlayKind};
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

fn write_lfm3_config_and_tok(ckpt: &Path) {
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

struct BaseWeights {
    w2_l1: Vec<f32>,
    o_proj_l1: Vec<f32>,
}

fn write_lfm3_shard(
    ckpt: &Path,
    override_w2_l1: Option<&[f32]>,
    override_o_l1: Option<&[f32]>,
) -> BaseWeights {
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

    let mut w2_l1 = Vec::new();
    let mut o_proj_l1 = Vec::new();

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
            let base_o = lcg_values(HIDDEN * Q_HEADS * HEAD_DIM, seed + 7);
            o_proj_l1 = base_o.clone();
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.o_proj.weight"),
                &[HIDDEN, Q_HEADS * HEAD_DIM],
                override_o_l1.unwrap_or(&base_o),
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
        let base_w2 = lcg_values(HIDDEN * INTERMEDIATE, seed + 13);
        let w2_vals = if layer == 1 {
            w2_l1 = base_w2.clone();
            override_w2_l1.unwrap_or(&base_w2)
        } else {
            &base_w2
        };
        push_bf16(
            &mut shard,
            &format!("{p}.feed_forward.w2.weight"),
            &[HIDDEN, INTERMEDIATE],
            w2_vals,
        );
    }
    shard.write(ckpt);
    BaseWeights { w2_l1, o_proj_l1 }
}

#[test]
fn parses_overlay_forms() {
    let s1 = parse(r#"OVERLAY SAFETENSORS "cand_dir";"#).unwrap();
    assert!(!s1.writes_artifact());
    assert_eq!(s1.verb(), "OVERLAY");
    assert!(matches!(
        s1,
        Statement::Overlay {
            kind: OverlayKind::Safetensors { ref path }
        } if path == "cand_dir"
    ));

    let s2 = parse(r#"OVERLAY LORA "adapter_dir" SCALE 0.5;"#).unwrap();
    assert!(matches!(
        s2,
        Statement::Overlay {
            kind: OverlayKind::Lora { ref path, scale: Some(s) }
        } if path == "adapter_dir" && (s - 0.5).abs() < 1e-12
    ));

    let s3 = parse("OVERLAY CLEAR;").unwrap();
    assert!(matches!(
        s3,
        Statement::Overlay {
            kind: OverlayKind::Clear
        }
    ));
}

#[test]
fn overlay_safetensors_skips_unchanged_and_matches_full_reencode_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    let base_ckpt = dir.path().join("base_ckpt");
    let cand_ckpt = dir.path().join("cand_ckpt");
    std::fs::create_dir_all(&base_ckpt).unwrap();
    std::fs::create_dir_all(&cand_ckpt).unwrap();
    write_lfm3_config_and_tok(&base_ckpt);
    write_lfm3_config_and_tok(&cand_ckpt);

    let base_w = write_lfm3_shard(&base_ckpt, None, None);

    // Mutate only 2 tensors (w2 and o_proj on layer 1) in candidate checkpoint.
    let mut mod_w2 = base_w.w2_l1.clone();
    for (i, v) in mod_w2.iter_mut().enumerate() {
        if i % 3 == 0 {
            *v += 0.15;
        }
    }
    let mut mod_o = base_w.o_proj_l1.clone();
    for (i, v) in mod_o.iter_mut().enumerate() {
        if i % 4 == 0 {
            *v -= 0.12;
        }
    }
    write_lfm3_shard(&cand_ckpt, Some(&mod_w2), Some(&mod_o));

    let base_vindex = dir.path().join("base.vindex3");
    let cand_vindex = dir.path().join("cand.vindex3");
    encode_checkpoint(&base_ckpt, &base_vindex).unwrap();
    encode_checkpoint(&cand_ckpt, &cand_vindex).unwrap();

    // Baseline session on base.vindex3
    let mut base_sess = Session::new();
    base_sess
        .execute(&parse(&format!(r#"USE "{}";"#, q(&base_vindex))).unwrap())
        .unwrap();
    let base_pred = base_sess.predict("alpha beta", 4).unwrap();
    let base_dla = base_sess
        .dla(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
            None,
            Some(4),
            Some(2),
            Some(2),
        )
        .unwrap();

    // Candidate session on freshly encoded cand.vindex3
    let mut cand_sess = Session::new();
    cand_sess
        .execute(&parse(&format!(r#"USE "{}";"#, q(&cand_vindex))).unwrap())
        .unwrap();
    let cand_pred = cand_sess.predict("alpha beta", 4).unwrap();
    let cand_dla = cand_sess
        .dla(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
            None,
            Some(4),
            Some(2),
            Some(2),
        )
        .unwrap();

    // Mount candidate checkpoint as an in-memory overlay on top of base_sess!
    let summary = base_sess
        .apply_external_overlay(&OverlayKind::Safetensors {
            path: cand_ckpt.display().to_string(),
        })
        .unwrap()
        .expect("overlay summary");
    assert_eq!(summary.overlaid_tensors, 2);
    assert!(summary.unchanged_skipped >= 15, "{summary:?}");

    let ov_pred = base_sess.predict("alpha beta", 4).unwrap();
    let ov_dla = base_sess
        .dla(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
            None,
            Some(4),
            Some(2),
            Some(2),
        )
        .unwrap();

    // Verify bit-exact parity with freshly encoded candidate container.
    assert_ne!(
        base_dla.exact_logit_diff.to_bits(),
        ov_dla.exact_logit_diff.to_bits(),
        "overlay must actually change model output"
    );
    assert_eq!(
        ov_dla.exact_logit_diff.to_bits(),
        cand_dla.exact_logit_diff.to_bits()
    );
    for (a, b) in ov_pred.top.iter().zip(cand_pred.top.iter()) {
        assert_eq!(a.token_id, b.token_id);
        assert_eq!(a.probability.to_bits(), b.probability.to_bits());
    }

    // Exporting safetensors while overlay is active writes the overlaid weights!
    let exported = dir.path().join("exported_overlaid");
    base_sess
        .execute(
            &parse(&format!(
                r#"EXPORT SAFETENSORS "{}" FROM BASE "{}";"#,
                q(&exported),
                q(&base_ckpt)
            ))
            .unwrap(),
        )
        .unwrap();
    let reenc = dir.path().join("reenc_overlaid.vindex3");
    encode_checkpoint(&exported, &reenc).unwrap();
    let mut check_sess = Session::new();
    check_sess
        .execute(&parse(&format!(r#"USE "{}";"#, q(&reenc))).unwrap())
        .unwrap();
    let reenc_pred = check_sess.predict("alpha beta", 4).unwrap();
    for (a, b) in reenc_pred.top.iter().zip(cand_pred.top.iter()) {
        assert_eq!(a.token_id, b.token_id);
        assert_eq!(a.probability.to_bits(), b.probability.to_bits());
    }

    // OVERLAY CLEAR restores exact untouched base predictions!
    base_sess
        .execute(&parse("OVERLAY CLEAR;").unwrap())
        .unwrap();
    let restored_pred = base_sess.predict("alpha beta", 4).unwrap();
    for (a, b) in restored_pred.top.iter().zip(base_pred.top.iter()) {
        assert_eq!(a.token_id, b.token_id);
        assert_eq!(a.probability.to_bits(), b.probability.to_bits());
    }
}

#[test]
fn overlay_lora_matches_explicitly_merged_checkpoint_bit_for_bit() {
    let dir = tempfile::tempdir().unwrap();
    let base_ckpt = dir.path().join("base_ckpt");
    let merged_ckpt = dir.path().join("merged_ckpt");
    let lora_dir = dir.path().join("adapter");
    std::fs::create_dir_all(&base_ckpt).unwrap();
    std::fs::create_dir_all(&merged_ckpt).unwrap();
    std::fs::create_dir_all(&lora_dir).unwrap();
    write_lfm3_config_and_tok(&base_ckpt);
    write_lfm3_config_and_tok(&merged_ckpt);

    let base_w = write_lfm3_shard(&base_ckpt, None, None);

    // Build a rank-4 PEFT LoRA adapter targeting 1.self_attn.o_proj.weight (16 x 16).
    let rank = 4usize;
    let alpha = 8.0f64;
    let scale_arg = 0.5f64;
    let factor = scale_arg * (alpha / rank as f64);
    let lora_a = lcg_values(rank * HIDDEN, 901);
    let lora_b = lcg_values(HIDDEN * rank, 902);

    std::fs::write(
        lora_dir.join("adapter_config.json"),
        serde_json::json!({
            "peft_type": "LORA",
            "r": rank,
            "lora_alpha": alpha,
            "target_modules": ["o_proj"]
        })
        .to_string(),
    )
    .unwrap();

    // Round A and B to BF16 in the adapter file (matching PyTorch PEFT safetensors).
    let a_bf16 = larql_models::quant::half::encode_bf16(&lora_a);
    let b_bf16 = larql_models::quant::half::encode_bf16(&lora_b);
    let a_f32: Vec<f32> = a_bf16
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect();
    let b_f32: Vec<f32> = b_bf16
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect();

    let mut adapter_shard = ShardBuilder::new();
    adapter_shard.push_bytes(
        "base_model.model.model.layers.1.self_attn.o_proj.lora_A.weight",
        "BF16",
        &[rank, HIDDEN],
        &a_bf16,
    );
    adapter_shard.push_bytes(
        "base_model.model.model.layers.1.self_attn.o_proj.lora_B.weight",
        "BF16",
        &[HIDDEN, rank],
        &b_bf16,
    );
    // Write as adapter_model.safetensors
    adapter_shard.write(&lora_dir);
    std::fs::rename(
        lora_dir.join("model.safetensors"),
        lora_dir.join("adapter_model.safetensors"),
    )
    .unwrap();

    // Compute exact reference merged BF16 matrix: base_bf16 + factor * (B @ A).
    let base_o_bf16 = larql_models::quant::half::encode_bf16(&base_w.o_proj_l1);
    let base_o_f32: Vec<f32> = base_o_bf16
        .chunks_exact(2)
        .map(|c| f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16))
        .collect();
    let mut merged_o = vec![0.0f32; HIDDEN * HIDDEN];
    for i in 0..HIDDEN {
        for j in 0..HIDDEN {
            let mut acc = 0.0f64;
            for r in 0..rank {
                acc += f64::from(b_f32[i * rank + r]) * f64::from(a_f32[r * HIDDEN + j]);
            }
            merged_o[i * HIDDEN + j] =
                (f64::from(base_o_f32[i * HIDDEN + j]) + factor * acc) as f32;
        }
    }
    write_lfm3_shard(&merged_ckpt, None, Some(&merged_o));

    let base_vindex = dir.path().join("base.vindex3");
    let merged_vindex = dir.path().join("merged.vindex3");
    encode_checkpoint(&base_ckpt, &base_vindex).unwrap();
    encode_checkpoint(&merged_ckpt, &merged_vindex).unwrap();

    let mut ov_sess = Session::new();
    ov_sess
        .execute(&parse(&format!(r#"USE "{}";"#, q(&base_vindex))).unwrap())
        .unwrap();
    let lines = ov_sess
        .execute(
            &parse(&format!(
                r#"OVERLAY LORA "{}" SCALE {scale_arg};"#,
                q(&lora_dir)
            ))
            .unwrap(),
        )
        .unwrap();
    assert!(lines[0].contains("1 modified tensor(s)"), "{lines:?}");

    let mut ref_sess = Session::new();
    ref_sess
        .execute(&parse(&format!(r#"USE "{}";"#, q(&merged_vindex))).unwrap())
        .unwrap();

    let ov_pred = ov_sess.predict("alpha beta", 4).unwrap();
    let ref_pred = ref_sess.predict("alpha beta", 4).unwrap();
    for (a, b) in ov_pred.top.iter().zip(ref_pred.top.iter()) {
        assert_eq!(a.token_id, b.token_id);
        assert_eq!(a.probability.to_bits(), b.probability.to_bits());
    }
}
