//! Round-trip integration tests for `export_safetensors`:
//! BF16 hybrid LFM3 checkpoint -> encode -> weight patch (2D matrix + 3D conv kernel) ->
//! export_safetensors (single-file & sharded) -> re-encode -> prove bit-exact SHA-256 parity!

use std::path::Path;

use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::vindex3::encode::checkpoint::encode_checkpoint;
use larql_vindex::format::vindex3::export_safetensors::{
    export_safetensors, SafetensorsExportOptions, EXPORT_MANIFEST_JSON,
};
use larql_vindex::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};
use larql_vindex::format::vindex3::index::Vindex3Index;
use larql_vindex::format::vindex3::weight_patch::{
    apply_weight_patches, bf16::bf16_to_f32, read_bf16_tensor, WeightEdit, WeightPatch,
};

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

fn write_lfm3_checkpoint(ckpt: &Path) {
    std::fs::create_dir_all(ckpt).unwrap();
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
    std::fs::write(ckpt.join("tokenizer.json"), r#"{"version":"1.0"}"#).unwrap();

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
    shard.write(ckpt);
}

fn canonical_payload_hashes(container: &Path) -> std::collections::BTreeMap<String, String> {
    let raw = std::fs::read_to_string(container.join(INDEX_JSON)).unwrap();
    let idx: Vindex3Index = serde_json::from_str(&raw).unwrap();
    idx.representations
        .into_iter()
        .filter(|(_, e)| e.compiled_from.is_none())
        .map(|(id, e)| (id, e.payload_sha256))
        .collect()
}

#[test]
fn patched_vindex3_exports_to_safetensors_and_round_trips_bit_identically() {
    let dir = tempfile::tempdir().unwrap();
    let ckpt = dir.path().join("ckpt");
    write_lfm3_checkpoint(&ckpt);

    let base = dir.path().join("base.vindex3");
    encode_checkpoint(&ckpt, &base).unwrap();

    // Patch BOTH a 3D conv kernel (zero out channel 3) AND a 2D FFN matrix!
    let (_, conv_bits) = read_bf16_tensor(&base, "0.conv.conv.weight").unwrap();
    let conv_edits: Vec<WeightEdit> = (0..CONV_L_CACHE)
        .map(|tap| WeightEdit {
            row: 3,
            col: tap,
            delta: -bf16_to_f32(conv_bits[3 * CONV_L_CACHE + tap]),
        })
        .collect();
    let patches = vec![
        WeightPatch {
            tensor: "0.conv.conv.weight".to_string(),
            edits: conv_edits,
        },
        WeightPatch {
            tensor: "1.feed_forward.w2.weight".to_string(),
            edits: vec![WeightEdit {
                row: 0,
                col: 2,
                delta: 0.5,
            }],
        },
    ];
    let patched = dir.path().join("patched.vindex3");
    apply_weight_patches(&base, &patches, &[], &patched).unwrap();

    // 1) Single-file safetensors export with --source-checkpoint.
    let single_out = dir.path().join("export_single");
    let report = export_safetensors(
        &patched,
        &single_out,
        &SafetensorsExportOptions {
            source_checkpoint: Some(&ckpt),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(report.verified);
    assert_eq!(report.shards.len(), 1);
    assert_eq!(report.shards[0].filename, "model.safetensors");
    assert!(report.weight_patches_sha256.is_some());
    assert!(single_out.join("model.safetensors").is_file());
    assert!(single_out.join("config.json").is_file());
    assert!(single_out.join("tokenizer.json").is_file());
    assert!(single_out.join(EXPORT_MANIFEST_JSON).is_file());

    // Re-encode the exported safetensors checkpoint into a fresh VINDEX3 container:
    // EVERY canonical representation's payload_sha256 must match `patched` BIT FOR BIT!
    let reencoded = dir.path().join("reencoded_single.vindex3");
    encode_checkpoint(&single_out, &reencoded).unwrap();
    assert_eq!(
        canonical_payload_hashes(&reencoded),
        canonical_payload_hashes(&patched),
        "single-file safetensors round-trip must preserve every canonical payload SHA-256 bit-for-bit"
    );

    // Confirm the zeroed 3D conv kernel survived export -> re-encode as exact 0x0000!
    let (_, re_conv_bits) = read_bf16_tensor(&reencoded, "0.conv.conv.weight").unwrap();
    for tap in 0..CONV_L_CACHE {
        assert_eq!(re_conv_bits[3 * CONV_L_CACHE + tap], 0x0000);
    }

    // 2) Forced multi-shard export (max_shard_bytes = 4096) with model.safetensors.index.json!
    let sharded_out = dir.path().join("export_sharded");
    let sharded = export_safetensors(
        &patched,
        &sharded_out,
        &SafetensorsExportOptions {
            source_checkpoint: Some(&ckpt),
            max_shard_bytes: 4096,
        },
    )
    .unwrap();
    assert!(sharded.verified);
    assert!(sharded.shards.len() > 1, "expected multiple shards");
    assert!(sharded_out.join("model.safetensors.index.json").is_file());

    let reencoded_sharded = dir.path().join("reencoded_sharded.vindex3");
    encode_checkpoint(&sharded_out, &reencoded_sharded).unwrap();
    assert_eq!(
        canonical_payload_hashes(&reencoded_sharded),
        canonical_payload_hashes(&patched),
        "sharded safetensors round-trip must preserve every canonical payload SHA-256 bit-for-bit"
    );

    // 3) Refuses overwriting an existing output directory.
    let err = export_safetensors(&patched, &single_out, &SafetensorsExportOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("already exists"), "{err}");
}
