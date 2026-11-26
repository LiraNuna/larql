//! Integration tests for session-scoped prefix KV & recurrent continuation-state caching.

use std::path::Path;

use larql_lql::operations::{CaptureLayers, CapturePositions, CaptureSite, CaptureSpec};
use larql_lql::{parse, Session};
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

fn build_lfm3_container() -> (tempfile::TempDir, std::path::PathBuf) {
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

    let container = dir.path().join("lfm3.vindex3");
    encode_checkpoint(&ckpt, &container).expect("encode hybrid LFM3 container");

    let vocab: serde_json::Map<String, serde_json::Value> =
        (0..VOCAB).map(|i| (format!("[{i}]"), i.into())).collect();
    let tokenizer = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[0]"}
    });
    std::fs::write(container.join("tokenizer.json"), tokenizer.to_string()).unwrap();
    (dir, container)
}

fn open(container: &Path) -> Session {
    let mut s = Session::new();
    s.execute(&parse(&format!("USE \"{}\";", q(container))).unwrap())
        .unwrap();
    s
}

#[test]
fn shared_prefix_and_extension_match_fresh_uncached_sessions_on_hybrid_lfm3() {
    let (_dir, container) = build_lfm3_container();

    // Shared 5-token prefix across three distinct suffixes + one direct extension.
    let prefix = "[1] [2] [3] [4] [5]";
    let p1 = "[1] [2] [3] [4] [5] [10] [11]";
    let p2 = "[1] [2] [3] [4] [5] [12] [13] [14]";
    let p3 = "[1] [2] [3] [4] [5] [15] [16]";
    let ext = "[1] [2] [3] [4] [5] [15] [16] [17] [18]";

    // One persistent session warming & reusing the prefix cache across every operation.
    let mut cached = open(&container);
    let _ = cached.predict(prefix, 5).unwrap();
    let c_p1 = cached.predict(p1, 5).unwrap();
    let c_p2 = cached.predict(p2, 5).unwrap();
    let c_p3 = cached.predict(p3, 5).unwrap();
    let c_ext = cached.predict(ext, 5).unwrap();

    // Independent fresh sessions (cold cache) must match bit-for-bit on every prompt!
    for (label, prompt, got) in [
        ("p1", p1, c_p1),
        ("p2", p2, c_p2),
        ("p3", p3, c_p3),
        ("ext", ext, c_ext),
    ] {
        let fresh = open(&container).predict(prompt, 5).unwrap();
        assert_eq!(got.top, fresh.top, "INFER mismatch for {label}");
    }

    // GENERATE right after INFER on `ext` (exact cache hit) AND on `p2` (prefix hit).
    for prompt in [ext, p2] {
        let got = cached.generate(prompt, 4).unwrap();
        let fresh = open(&container).generate(prompt, 4).unwrap();
        assert_eq!(
            got.token_ids, fresh.token_ids,
            "GENERATE mismatch on {prompt}"
        );
    }

    // DLA with TOP COLUMNS over prefix-cached prompts matches fresh session.
    for prompt in [p1, p2, ext] {
        let got = cached
            .dla(prompt, 5u32, 12u32, None, Some(20), None)
            .unwrap();
        let fresh = open(&container)
            .dla(prompt, 5u32, 12u32, None, Some(20), None)
            .unwrap();
        assert_eq!(
            got.exact_logit_diff.to_bits(),
            fresh.exact_logit_diff.to_bits(),
            "DLA exact_logit_diff mismatch on {prompt}"
        );
        assert_eq!(
            got.total_dla.to_bits(),
            fresh.total_dla.to_bits(),
            "DLA total_dla mismatch on {prompt}"
        );
        assert_eq!(got.top_columns, fresh.top_columns);
    }

    // CAPTURE across residual, ffn.down_input, short_conv.gated_y, and logits at POSITIONS LAST
    // on a prefix-cached prompt matches a fresh session bit-for-bit.
    let cases = [
        (CaptureSite::Residual, Some(CaptureLayers::One(3))),
        (CaptureSite::FfnDownInput, Some(CaptureLayers::One(2))),
        (CaptureSite::ShortConvGatedY, Some(CaptureLayers::One(2))),
        (CaptureSite::Logits, None),
    ];
    for (site, layers) in cases {
        let spec = CaptureSpec {
            site,
            layers: layers.as_ref(),
            prompt: ext,
            continuation: None,
            positions: &CapturePositions::Last,
        };
        let got = cached.capture(&spec).unwrap();
        let fresh = open(&container).capture(&spec).unwrap();
        assert_eq!(got.shape, fresh.shape, "shape mismatch on {:?}", site);
        assert_eq!(
            got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            fresh
                .values()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            "values mismatch on {:?}",
            site
        );
    }

    // Staging a live weight edit immediately invalidates the prefix cache, and DISCARD restores it.
    let base_p2 = cached.predict(p2, 5).unwrap().top;
    cached
        .execute(
            &parse(
                "UPDATE WEIGHTS SET DELTA = 16.0 WHERE TENSOR = \"3.feed_forward.w2.weight\" AND ROW = 0 AND COL IN (0, 1, 2, 3);",
            )
            .unwrap(),
        )
        .unwrap();
    let patched_p2 = cached.predict(p2, 5).unwrap().top;
    assert_ne!(
        patched_p2, base_p2,
        "live weight patch must invalidate cached prefix KV/conv snapshots"
    );

    cached
        .execute(&parse("DISCARD WEIGHT EDITS;").unwrap())
        .unwrap();
    assert_eq!(
        cached.predict(p2, 5).unwrap().top,
        base_p2,
        "discarding weight edits must restore exact base predictions"
    );
}
