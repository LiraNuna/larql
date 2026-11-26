//! `CAPTURE` on a BF16 hybrid `ShortConv` + softmax GQA container (LFM3-shaped):
//! internal `short_conv.{gate_b, gate_c, conv_out, gated_y}` sites, multi-layer
//! stacking, geometry bounds, and fail-closed refusal on softmax layers.

use std::path::{Path, PathBuf};

use larql_lql::ast::CaptureSite;
use larql_lql::Session;
use larql_vindex::format::vindex3::encode::checkpoint::encode_checkpoint;
use larql_vindex::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};

use crate::support::*;

pub(crate) const SC_HIDDEN: usize = 16;
pub(crate) const SC_INTERMEDIATE: usize = 24;
const SC_VOCAB: usize = 32;
const SC_Q_HEADS: usize = 4;
const SC_KV_HEADS: usize = 2;
const SC_HEAD_DIM: usize = 4;
const SC_CONV_L_CACHE: usize = 3;

fn push_bf16(shard: &mut ShardBuilder, name: &str, shape: &[usize], values: &[f32]) {
    let bytes = larql_models::quant::half::encode_bf16(values);
    shard.push_bytes(name, "BF16", shape, &bytes);
}

fn sc_tokenizer_json() -> String {
    let words = ["The", "capital", "of", "France", "is", "Paris"];
    let entries: serde_json::Map<String, serde_json::Value> = (0..SC_VOCAB)
        .map(|i| {
            let tok = match i.checked_sub(10) {
                Some(w) if w < words.len() => words[w].to_string(),
                _ => format!("[{i}]"),
            };
            (tok, i.into())
        })
        .collect();
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": entries, "unk_token": "[0]"}
    })
    .to_string()
}

pub(crate) struct HybridShortConvFixture {
    pub dir: tempfile::TempDir,
    pub container: PathBuf,
}

impl HybridShortConvFixture {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let ckpt = dir.path().join("ckpt");
        std::fs::create_dir_all(&ckpt).unwrap();
        write_bf16_lfm3_checkpoint(&ckpt);

        let container = dir.path().join("lfm3-bf16.vindex3");
        encode_checkpoint(&ckpt, &container).expect("encode BF16 LFM3 container");
        std::fs::write(container.join("tokenizer.json"), sc_tokenizer_json()).unwrap();
        Self { dir, container }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    pub fn session(&self) -> Session {
        session_on(&self.container)
    }
}

fn write_bf16_lfm3_checkpoint(dir: &Path) {
    let config = serde_json::json!({
        "architectures": ["Lfm3ForCausalLM"],
        "model_type": "lfm3",
        "torch_dtype": "bfloat16",
        "hidden_size": SC_HIDDEN,
        "intermediate_size": SC_INTERMEDIATE,
        "num_hidden_layers": 4,
        "num_attention_heads": SC_Q_HEADS,
        "num_key_value_heads": SC_KV_HEADS,
        "head_dim": SC_HEAD_DIM,
        "vocab_size": SC_VOCAB,
        "max_position_embeddings": 64,
        "norm_eps": 1e-5,
        "conv_L_cache": SC_CONV_L_CACHE,
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
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let mut shard = ShardBuilder::new();
    push_bf16(
        &mut shard,
        "model.embed_tokens.weight",
        &[SC_VOCAB, SC_HIDDEN],
        &lcg_values(SC_VOCAB * SC_HIDDEN, 1),
    );
    push_bf16(
        &mut shard,
        "model.embedding_norm.weight",
        &[SC_HIDDEN],
        &norm_values(SC_HIDDEN, 2),
    );

    let layer_kinds = ["conv", "sliding_attention", "conv", "full_attention"];
    for (layer, &kind) in layer_kinds.iter().enumerate() {
        let seed = 100 + layer as u64 * 30;
        let p = format!("model.layers.{layer}");
        push_bf16(
            &mut shard,
            &format!("{p}.operator_norm.weight"),
            &[SC_HIDDEN],
            &norm_values(SC_HIDDEN, seed),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.post_operator_norm.weight"),
            &[SC_HIDDEN],
            &norm_values(SC_HIDDEN, seed + 1),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.post_ffn_norm.weight"),
            &[SC_HIDDEN],
            &norm_values(SC_HIDDEN, seed + 2),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.ffn_norm.weight"),
            &[SC_HIDDEN],
            &norm_values(SC_HIDDEN, seed + 3),
        );
        if kind == "conv" {
            push_bf16(
                &mut shard,
                &format!("{p}.conv.in_proj.weight"),
                &[3 * SC_HIDDEN, SC_HIDDEN],
                &lcg_values(3 * SC_HIDDEN * SC_HIDDEN, seed + 4),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.conv.conv.weight"),
                &[SC_HIDDEN, 1, SC_CONV_L_CACHE],
                &lcg_values(SC_HIDDEN * SC_CONV_L_CACHE, seed + 5),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.conv.out_proj.weight"),
                &[SC_HIDDEN, SC_HIDDEN],
                &lcg_values(SC_HIDDEN * SC_HIDDEN, seed + 6),
            );
        } else {
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.q_proj.weight"),
                &[SC_Q_HEADS * SC_HEAD_DIM, SC_HIDDEN],
                &lcg_values(SC_Q_HEADS * SC_HEAD_DIM * SC_HIDDEN, seed + 4),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.k_proj.weight"),
                &[SC_KV_HEADS * SC_HEAD_DIM, SC_HIDDEN],
                &lcg_values(SC_KV_HEADS * SC_HEAD_DIM * SC_HIDDEN, seed + 5),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.v_proj.weight"),
                &[SC_KV_HEADS * SC_HEAD_DIM, SC_HIDDEN],
                &lcg_values(SC_KV_HEADS * SC_HEAD_DIM * SC_HIDDEN, seed + 6),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.o_proj.weight"),
                &[SC_HIDDEN, SC_Q_HEADS * SC_HEAD_DIM],
                &lcg_values(SC_HIDDEN * SC_Q_HEADS * SC_HEAD_DIM, seed + 7),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.q_norm.weight"),
                &[SC_HEAD_DIM],
                &norm_values(SC_HEAD_DIM, seed + 8),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.k_norm.weight"),
                &[SC_HEAD_DIM],
                &norm_values(SC_HEAD_DIM, seed + 9),
            );
            push_bf16(
                &mut shard,
                &format!("{p}.self_attn.gate_proj.weight"),
                &[SC_Q_HEADS, SC_HIDDEN],
                &lcg_values(SC_Q_HEADS * SC_HIDDEN, seed + 10),
            );
        }
        push_bf16(
            &mut shard,
            &format!("{p}.feed_forward.w1.weight"),
            &[SC_INTERMEDIATE, SC_HIDDEN],
            &lcg_values(SC_INTERMEDIATE * SC_HIDDEN, seed + 11),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.feed_forward.w3.weight"),
            &[SC_INTERMEDIATE, SC_HIDDEN],
            &lcg_values(SC_INTERMEDIATE * SC_HIDDEN, seed + 12),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.feed_forward.w2.weight"),
            &[SC_HIDDEN, SC_INTERMEDIATE],
            &lcg_values(SC_HIDDEN * SC_INTERMEDIATE, seed + 13),
        );
    }
    shard.write(dir);
}

#[test]
fn short_conv_internal_sites_capture_exact_signals_and_refuse_softmax_layers() {
    let fx = HybridShortConvFixture::new();
    let mut s = fx.session();

    // Geometry: bounded on ShortConv layers (0, 2), unbounded on softmax layer (1).
    for site in [
        CaptureSite::ShortConvGateB,
        CaptureSite::ShortConvGateC,
        CaptureSite::ShortConvConvOut,
        CaptureSite::ShortConvGatedY,
    ] {
        assert_eq!(
            s.capture_geometry(site, &[0, 2]).unwrap().width,
            Some(SC_HIDDEN),
            "{}",
            site.name()
        );
        assert_eq!(
            s.capture_geometry(site, &[1]).unwrap().width,
            None,
            "{}",
            site.name()
        );
        assert_eq!(
            s.capture_geometry(site, &[0, 1]).unwrap().width,
            None,
            "{}",
            site.name()
        );
    }

    // Multi-layer capture across both ShortConv layers (0, 2) and all prompt positions.
    let gate_b_path = fx.path("gate_b.npy");
    let gate_c_path = fx.path("gate_c.npy");
    let conv_out_path = fx.path("conv_out.npy");
    let gated_y_path = fx.path("gated_y.npy");

    capture(
        &mut s,
        &format!("short_conv.gate_b AT LAYERS (0, 2) FOR \"{PROMPT}\" POSITIONS ALL"),
        &gate_b_path,
    );
    capture(
        &mut s,
        &format!("short_conv.gate_c AT LAYERS (0, 2) FOR \"{PROMPT}\" POSITIONS ALL"),
        &gate_c_path,
    );
    capture(
        &mut s,
        &format!("short_conv.conv_out AT LAYERS (0, 2) FOR \"{PROMPT}\" POSITIONS ALL"),
        &conv_out_path,
    );
    capture(
        &mut s,
        &format!("short_conv.gated_y AT LAYERS (0, 2) FOR \"{PROMPT}\" POSITIONS ALL"),
        &gated_y_path,
    );

    let gate_b = read_npy(&gate_b_path);
    let gate_c = read_npy(&gate_c_path);
    let conv_out = read_npy(&conv_out_path);
    let gated_y = read_npy(&gated_y_path);
    let expected_shape = vec![2, PROMPT_TOKENS, SC_HIDDEN];
    assert_eq!(gate_b.shape, expected_shape);
    assert_eq!(gate_c.shape, expected_shape);
    assert_eq!(conv_out.shape, expected_shape);
    assert_eq!(gated_y.shape, expected_shape);

    // Exact elementwise law: gated_y == gate_c * conv_out.
    for i in 0..gated_y.values.len() {
        assert_eq!(
            gated_y.values[i].to_bits(),
            (gate_c.values[i] * conv_out.values[i]).to_bits(),
            "mismatch at flat index {i}"
        );
    }
    assert!(sidecar(&gated_y_path)["execution"]["path"]
        .as_str()
        .unwrap()
        .contains("one-shot traversal"));
    assert_eq!(sidecar(&gated_y_path)["site"], "short_conv.gated_y");

    // Fail-closed refusal when targeting a softmax layer (layer 1).
    let bad = fx.path("bad.npy");
    let err = try_capture(
        &mut s,
        &format!("short_conv.gated_y AT LAYER 1 FOR \"{PROMPT}\""),
        &bad,
    )
    .unwrap_err();
    assert!(err.contains("needs a ShortConv layer"), "{err}");
    assert!(!bad.exists(), "a refused capture writes nothing");

    // Softmax attention per-head output (`attn.head_out`) on sliding (L1) & full (L3) layers.
    assert_eq!(
        s.capture_geometry(CaptureSite::AttnHeadOut, &[1, 3])
            .unwrap()
            .width,
        Some(SC_Q_HEADS * SC_HEAD_DIM)
    );
    assert_eq!(
        s.capture_geometry(CaptureSite::AttnHeadOut, &[0])
            .unwrap()
            .width,
        None
    );
    let head_out_path = fx.path("attn_head_out.npy");
    capture(
        &mut s,
        &format!("attn.head_out AT LAYERS (1, 3) FOR \"{PROMPT}\" POSITIONS ALL"),
        &head_out_path,
    );
    let head_out = read_npy(&head_out_path);
    assert_eq!(
        head_out.shape,
        vec![2, PROMPT_TOKENS, SC_Q_HEADS * SC_HEAD_DIM]
    );
    assert!(head_out.values.iter().any(|&v| v != 0.0));
    let bad_attn = fx.path("bad_attn.npy");
    let err = try_capture(
        &mut s,
        &format!("attn.head_out AT LAYER 0 FOR \"{PROMPT}\""),
        &bad_attn,
    )
    .unwrap_err();
    assert!(err.contains("needs a softmax attention layer"), "{err}");
    assert!(!bad_attn.exists());
}

fn hex_f64(values: &[f64]) -> Vec<String> {
    values
        .iter()
        .map(|v| format!("{:016x}", v.to_bits()))
        .collect()
}

#[test]
fn staged_remove_direction_and_add_low_rank_execute_live_for_infer_and_capture_on_hybrid_short_conv(
) {
    let fx = HybridShortConvFixture::new();
    let mut s = fx.session();

    // Baseline captures: batch (`logits`, `post_attention`) and decode (`ffn.down_output`, `short_conv.gated_y`).
    let base_logits = fx.path("base_logits.npy");
    let base_post_attn = fx.path("base_post_attn.npy");
    let base_down_out = fx.path("base_down_out.npy");
    let base_gated_y = fx.path("base_gated_y.npy");
    capture(&mut s, &format!("logits FOR \"{PROMPT}\""), &base_logits);
    capture(
        &mut s,
        &format!("post_attention AT LAYER 0 FOR \"{PROMPT}\""),
        &base_post_attn,
    );
    capture(
        &mut s,
        &format!("ffn.down_output AT LAYER 0 FOR \"{PROMPT}\""),
        &base_down_out,
    );
    capture(
        &mut s,
        &format!("short_conv.gated_y AT LAYER 0 FOR \"{PROMPT}\""),
        &base_gated_y,
    );
    let base_logits_bits = bits(&read_npy(&base_logits).values);
    let base_post_attn_bits = bits(&read_npy(&base_post_attn).values);
    let base_down_out_bits = bits(&read_npy(&base_down_out).values);
    let base_gated_y_bits = bits(&read_npy(&base_gated_y).values);

    // 1. Write operand files for REMOVE DIRECTION on dense FFN `feed_forward.w2.weight`
    //    (both on ShortConv layer 0 and softmax layer 1) and ADD LOW RANK on ShortConv
    //    `0.conv.in_proj.weight` + `2.conv.out_proj.weight`.
    let u_path = fx.path("u.json");
    let mut u = vec![0.0f64; SC_HIDDEN];
    u[0] = 1.0;
    std::fs::write(
        &u_path,
        serde_json::json!({ "bits": hex_f64(&u), "origin": "test-unit-0" }).to_string(),
    )
    .unwrap();

    // `0.conv.in_proj.weight` has shape `[3 * SC_HIDDEN, SC_HIDDEN]`, rank 1 on columns (0, 1).
    let ab_in_path = fx.path("ab_in.json");
    let a_in = vec![0.5f64; 3 * SC_HIDDEN];
    let b_in = vec![1.0f64, -1.0f64];
    std::fs::write(
        &ab_in_path,
        serde_json::json!({
            "rank": 1,
            "a": hex_f64(&a_in),
            "b": hex_f64(&b_in),
            "origin": "test-sc-in"
        })
        .to_string(),
    )
    .unwrap();

    // `2.conv.out_proj.weight` has shape `[SC_HIDDEN, SC_HIDDEN]`, rank 1 on columns (0, 1).
    let ab_out_path = fx.path("ab_out.json");
    let a_out = vec![0.5f64; SC_HIDDEN];
    let b_out = vec![1.0f64, 0.5f64];
    std::fs::write(
        &ab_out_path,
        serde_json::json!({
            "rank": 1,
            "a": hex_f64(&a_out),
            "b": hex_f64(&b_out),
            "origin": "test-sc-out"
        })
        .to_string(),
    )
    .unwrap();

    // Stage all four ops in one session.
    run(
        &mut s,
        &format!(
            "REMOVE DIRECTION \"{}\" FROM \"0.feed_forward.w2.weight\" COLUMNS (0, 1, 2, 3) \
             SCALE 1.0 NUMERICS \"f64-seq-rne1/v1\";",
            q(&u_path)
        ),
    );
    run(
        &mut s,
        &format!(
            "REMOVE DIRECTION \"{}\" FROM \"1.feed_forward.w2.weight\" COLUMNS (0, 1, 2, 3) \
             SCALE 1.0 NUMERICS \"f64-seq-rne1/v1\";",
            q(&u_path)
        ),
    );
    run(
        &mut s,
        &format!(
            "ADD LOW RANK \"{}\" TO \"0.conv.in_proj.weight\" COLUMNS (0, 1) \
             NUMERICS \"f64-seq-rne1/v1\";",
            q(&ab_in_path)
        ),
    );
    run(
        &mut s,
        &format!(
            "ADD LOW RANK \"{}\" TO \"2.conv.out_proj.weight\" COLUMNS (0, 1) \
             NUMERICS \"f64-seq-rne1/v1\";",
            q(&ab_out_path)
        ),
    );

    // 2. Live INFER must execute cleanly on patched weights (no hybrid/non-softmax refusal).
    let infer_out = run(&mut s, &format!("INFER \"{PROMPT}\" TOP 3;"));
    assert!(
        infer_out.iter().any(|l| l.contains("Predictions")),
        "expected patched INFER predictions: {infer_out:?}"
    );

    // 3. Live CAPTURE across batch and decode sites must execute with `"executes": "patched"`
    //    and measurably alter every affected site.
    let pat_logits = fx.path("pat_logits.npy");
    let pat_post_attn = fx.path("pat_post_attn.npy");
    let pat_down_out = fx.path("pat_down_out.npy");
    let pat_gated_y = fx.path("pat_gated_y.npy");
    capture(&mut s, &format!("logits FOR \"{PROMPT}\""), &pat_logits);
    capture(
        &mut s,
        &format!("post_attention AT LAYER 0 FOR \"{PROMPT}\""),
        &pat_post_attn,
    );
    capture(
        &mut s,
        &format!("ffn.down_output AT LAYER 0 FOR \"{PROMPT}\""),
        &pat_down_out,
    );
    capture(
        &mut s,
        &format!("short_conv.gated_y AT LAYER 0 FOR \"{PROMPT}\""),
        &pat_gated_y,
    );

    let pat_logits_bits = bits(&read_npy(&pat_logits).values);
    let pat_post_attn_bits = bits(&read_npy(&pat_post_attn).values);
    let pat_down_out_bits = bits(&read_npy(&pat_down_out).values);
    let pat_gated_y_bits = bits(&read_npy(&pat_gated_y).values);

    assert_ne!(pat_logits_bits, base_logits_bits);
    assert_ne!(pat_post_attn_bits, base_post_attn_bits);
    assert_ne!(pat_down_out_bits, base_down_out_bits);
    assert_ne!(pat_gated_y_bits, base_gated_y_bits);
    for p in [&pat_logits, &pat_post_attn, &pat_down_out, &pat_gated_y] {
        assert_eq!(sidecar(p)["executes"], "patched", "{}", p.display());
    }
    assert!(sidecar(&pat_gated_y)["execution"]["path"]
        .as_str()
        .unwrap()
        .contains("one-shot traversal"));

    // 4. Compiled container matches live patched CAPTURE bit for bit.
    let compiled = fx.path("compiled_hybrid.vindex3");
    run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&compiled)),
    );
    let mut fresh = session_on(&compiled);
    let cmp_logits = fx.path("cmp_logits.npy");
    let cmp_gated_y = fx.path("cmp_gated_y.npy");
    capture(&mut fresh, &format!("logits FOR \"{PROMPT}\""), &cmp_logits);
    capture(
        &mut fresh,
        &format!("short_conv.gated_y AT LAYER 0 FOR \"{PROMPT}\""),
        &cmp_gated_y,
    );
    assert_eq!(bits(&read_npy(&cmp_logits).values), pat_logits_bits);
    assert_eq!(bits(&read_npy(&cmp_gated_y).values), pat_gated_y_bits);

    // 5. DISCARD WEIGHT EDITS restores base output bit for bit.
    run(&mut s, "DISCARD WEIGHT EDITS;");
    let after_logits = fx.path("after_logits.npy");
    capture(&mut s, &format!("logits FOR \"{PROMPT}\""), &after_logits);
    assert_eq!(bits(&read_npy(&after_logits).values), base_logits_bits);
}
