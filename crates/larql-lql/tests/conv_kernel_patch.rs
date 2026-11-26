//! End-to-end tests for 3D depthwise-conv1d kernel patching (`[channels, 1, kernel_size]`)
//! and `UPDATE WEIGHTS SET VALUE = <f>`: live execution, `CAPTURE short_conv.conv_out`,
//! `.vlp` save/apply, `COMPILE CURRENT INTO VINDEX`, and ledger coherence.

use std::path::Path;

use larql_lql::ast::{CaptureLayers, CapturePositions, CaptureSite};
use larql_lql::operations::CaptureSpec;
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::encode::checkpoint::encode_checkpoint;
use larql_vindex::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::weight_patch::read_bf16_tensor;

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

    for (layer, kind) in ["conv", "full_attention"].iter().enumerate() {
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
        if *kind == "conv" {
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

    (dir, container)
}

fn open_session(container: &Path) -> Session {
    let mut s = Session::new();
    s.execute(&parse(&format!("USE \"{}\";", lql_path(container))).unwrap())
        .unwrap();
    s
}

#[test]
fn zeroing_3d_conv_kernel_channel_zeroes_conv_out_live_and_compiles_bit_identically() {
    let (dir, container) = build_lfm3_container();
    let mut s = open_session(&container);

    let prompt = "[3] [7] [11] [15]";
    let cap_conv_out = |sess: &Session| {
        sess.capture(&CaptureSpec {
            site: CaptureSite::ShortConvConvOut,
            layers: Some(&CaptureLayers::One(0)),
            prompt,
            continuation: None,
            positions: &CapturePositions::Last,
        })
        .unwrap()
    };

    let base_out = cap_conv_out(&s);
    assert_ne!(
        base_out.values()[3],
        0.0,
        "channel 3 must start non-zero before zeroing its kernel"
    );
    let base_logits = s
        .capture(&CaptureSpec {
            site: CaptureSite::Logits,
            layers: None,
            prompt,
            continuation: None,
            positions: &CapturePositions::Last,
        })
        .unwrap();

    // Record into a .vlp patch and stage exact zeroing of channel 3's 3-tap temporal kernel.
    let vlp = dir.path().join("zero_ch3.vlp");
    s.execute(&parse(&format!("BEGIN PATCH \"{}\";", lql_path(&vlp))).unwrap())
        .unwrap();
    let staged = s
        .execute(
            &parse(
                r#"UPDATE WEIGHTS SET VALUE = 0.0 WHERE TENSOR = "0.conv.conv.weight" AND ROW = 3 AND COL IN (0, 1, 2);"#,
            )
            .unwrap(),
        )
        .unwrap();
    let staged_text = staged.join("\n");
    assert!(
        staged_text.contains("[16, 1, 3]"),
        "live report must preserve 3D shape [16, 1, 3]:\n{staged_text}"
    );
    assert!(
        staged_text.contains(larql_lql::live_scope::IN_EFFECT),
        "3D conv kernel patch must execute live:\n{staged_text}"
    );
    s.execute(&parse("SAVE PATCH;").unwrap()).unwrap();

    // Live execution: channel 3's conv_out MUST be exact 0.0, while other channels stay identical!
    let live_out = cap_conv_out(&s);
    assert_eq!(
        live_out.values()[3],
        0.0,
        "zeroed channel 3 must produce exact 0.0 conv_out"
    );
    for ch in 0..HIDDEN {
        if ch != 3 {
            assert_eq!(
                live_out.values()[ch].to_bits(),
                base_out.values()[ch].to_bits(),
                "unmodified channel {ch} must be bit-identical"
            );
        }
    }
    let live_logits = s
        .capture(&CaptureSpec {
            site: CaptureSite::Logits,
            layers: None,
            prompt,
            continuation: None,
            positions: &CapturePositions::Last,
        })
        .unwrap();
    assert_ne!(live_logits.values(), base_logits.values());

    // Apply the saved .vlp in a fresh session and compile into a new VINDEX3 container.
    let mut fresh = open_session(&container);
    fresh
        .execute(&parse(&format!("APPLY PATCH \"{}\";", lql_path(&vlp))).unwrap())
        .unwrap();
    let compiled = dir.path().join("compiled.vindex3");
    fresh
        .execute(
            &parse(&format!(
                "COMPILE CURRENT INTO VINDEX \"{}\";",
                lql_path(&compiled)
            ))
            .unwrap(),
        )
        .unwrap();

    // Container + weight_patches.json ledger must be 100% coherent.
    let inspected = inspect_container(&compiled, false).unwrap();
    assert!(
        inspected.is_coherent(),
        "compiled 3D kernel container has defects: {:?}",
        inspected.defects
    );

    // Stored BF16 bits of channel 3 in compiled container must be exact 0x0000.
    let (shape, bits) = read_bf16_tensor(&compiled, "0.conv.conv.weight").unwrap();
    assert_eq!(shape, [HIDDEN, CONV_L_CACHE]);
    for tap in 0..CONV_L_CACHE {
        assert_eq!(
            bits[3 * CONV_L_CACHE + tap],
            0x0000,
            "tap {tap} of channel 3 must be exact BF16 0x0000"
        );
    }

    // Compiled container's logits and conv_out must match the live patched session bit-for-bit!
    let compiled_sess = open_session(&compiled);
    let compiled_out = cap_conv_out(&compiled_sess);
    assert_eq!(compiled_out.values(), live_out.values());
    let compiled_logits = compiled_sess
        .capture(&CaptureSpec {
            site: CaptureSite::Logits,
            layers: None,
            prompt,
            continuation: None,
            positions: &CapturePositions::Last,
        })
        .unwrap();
    assert_eq!(compiled_logits.values(), live_logits.values());
}

#[test]
fn set_value_nonzero_lands_on_exact_bf16_bits() {
    let (_dir, container) = build_lfm3_container();
    let mut s = open_session(&container);
    let staged = s
        .stage_weight_value("0.conv.conv.weight", 1, &[0, 2], 0.5)
        .unwrap();
    // BF16 0.5 is 0x3f00.
    for el in &staged.elements {
        assert_eq!(el.new_bits, 0x3f00);
    }
    // Out-of-bounds tap is refused cleanly.
    let err = s
        .stage_weight_value("0.conv.conv.weight", 1, &[CONV_L_CACHE], 0.0)
        .unwrap_err()
        .to_string();
    assert!(err.contains("outside") || err.contains("bounds"), "{err}");
}
