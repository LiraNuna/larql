//! Integration tests for index range syntax (`a..b`, `a..=b`), attention `HEAD`/`HEADS`
//! shorthands, and row-space `REMOVE INPUT DIRECTION` edits.

use std::path::Path;

use larql_lql::operations::{npy_file_bytes, NpyDescr};
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::encode::checkpoint::encode_checkpoint;
use larql_vindex::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};
use larql_vindex::format::vindex3::weight_patch::read_bf16_tensor;

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
    for (layer, &kind) in ["conv", "sliding_attention", "conv", "full_attention"]
        .iter()
        .enumerate()
    {
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
    encode_checkpoint(&ckpt, &container).expect("encode LFM3 container");

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

fn write_unit_vec_npy(path: &Path, len: usize) -> Vec<f64> {
    let raw: Vec<f64> = (0..len).map(|i| ((i as f64) + 1.0) * 0.25).collect();
    let norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
    let unit: Vec<f64> = raw.into_iter().map(|x| x / norm).collect();
    let body: Vec<u8> = unit.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, npy_file_bytes(NpyDescr::F8, &[len], &body).unwrap()).unwrap();
    unit
}

#[test]
fn range_syntax_and_head_shorthand_match_explicit_columns_bit_for_bit() {
    let (dir, container) = build_lfm3_container();
    let u_npy = dir.path().join("u.npy");
    write_unit_vec_npy(&u_npy, HIDDEN);

    // HEAD 2 on o_proj (head_dim=4) == COLUMNS (8..12) == COLUMNS (8..=11) == COLUMNS (8, 9, 10, 11).
    let mut s_head = open(&container);
    s_head
        .execute(
            &parse(&format!(
                "REMOVE DIRECTION \"{}\" FROM \"3.self_attn.o_proj.weight\" HEAD 2 SCALE 0.75 NUMERICS \"f64-seq-rne1/v1\";",
                q(&u_npy)
            ))
            .unwrap(),
        )
        .unwrap();
    let pred_head = s_head.predict("[1] [2] [3] [4]", 5).unwrap().top;

    for col_clause in [
        "COLUMNS (8..12)",
        "COLUMNS (8..=11)",
        "COLUMNS (8, 9..11, 11)",
    ] {
        let mut s = open(&container);
        s.execute(
            &parse(&format!(
                "REMOVE DIRECTION \"{}\" FROM \"3.self_attn.o_proj.weight\" {col_clause} SCALE 0.75 NUMERICS \"f64-seq-rne1/v1\";",
                q(&u_npy)
            ))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            s.predict("[1] [2] [3] [4]", 5).unwrap().top,
            pred_head,
            "mismatch for {col_clause}"
        );
    }
}

#[test]
fn remove_input_direction_projects_selected_kv_head_rows_and_matches_compiled_container() {
    let (dir, container) = build_lfm3_container();
    let v_npy = dir.path().join("v.npy");
    let v = write_unit_vec_npy(&v_npy, HIDDEN);

    let (_, base_bits) = read_bf16_tensor(&container, "3.self_attn.k_proj.weight").unwrap();
    let base_pred = open(&container).predict("[1] [2] [3] [4]", 5).unwrap().top;

    // Stage REMOVE INPUT DIRECTION on KV HEAD 1 (rows 4..8) of 3.self_attn.k_proj.weight,
    // record into a .vlp patch, and compile into a fresh container.
    let vlp_path = dir.path().join("k_head1.vlp");
    let out_vindex = dir.path().join("compiled.vindex3");
    let mut s = open(&container);
    s.execute(&parse(&format!("BEGIN PATCH \"{}\";", q(&vlp_path))).unwrap())
        .unwrap();
    s.execute(
        &parse(&format!(
            "REMOVE INPUT DIRECTION \"{}\" FROM \"3.self_attn.k_proj.weight\" HEAD 1 SCALE 1.0 NUMERICS \"f64-seq-rne1/v1\";",
            q(&v_npy)
        ))
        .unwrap(),
    )
    .unwrap();
    s.execute(&parse("SAVE PATCH;").unwrap()).unwrap();
    let live_pred = s.predict("[1] [2] [3] [4]", 5).unwrap().top;
    assert_ne!(live_pred, base_pred);

    s.execute(
        &parse(&format!(
            "COMPILE CURRENT INTO VINDEX \"{}\";",
            q(&out_vindex)
        ))
        .unwrap(),
    )
    .unwrap();

    // Verify compiled container matches live execution bit-for-bit!
    assert_eq!(
        open(&out_vindex).predict("[1] [2] [3] [4]", 5).unwrap().top,
        live_pred
    );

    // Verify exact row-space projection geometry on 3.self_attn.k_proj.weight ([8, 16]):
    // - Head 0 (rows 0..4) is bit-identical to base.
    // - Head 1 (rows 4..8) has (W_new[r, :] · v) ≈ 0 (stripped along unit vector v!).
    let ([rows, cols], new_bits) =
        read_bf16_tensor(&out_vindex, "3.self_attn.k_proj.weight").unwrap();
    assert_eq!((rows, cols), (KV_HEADS * HEAD_DIM, HIDDEN));
    for r in 0..HEAD_DIM {
        assert_eq!(
            &new_bits[r * cols..(r + 1) * cols],
            &base_bits[r * cols..(r + 1) * cols],
            "untouched KV head 0 row {r} must keep exact base bits"
        );
    }
    for r in HEAD_DIM..2 * HEAD_DIM {
        let dot_after: f64 = (0..cols)
            .map(|j| {
                let w = f32::from_bits(u32::from(new_bits[r * cols + j]) << 16) as f64;
                w * v[j]
            })
            .sum();
        assert!(
            dot_after.abs() < 1e-2,
            "KV head 1 row {r} projection onto unit v should be stripped near 0, got {dot_after}"
        );
    }

    // Applying the saved .vlp in a fresh session matches bit-for-bit.
    let mut fresh = open(&container);
    fresh
        .execute(&parse(&format!("APPLY PATCH \"{}\";", q(&vlp_path))).unwrap())
        .unwrap();
    assert_eq!(fresh.predict("[1] [2] [3] [4]", 5).unwrap().top, live_pred);

    // Disjoint KV head (HEAD 0 alongside HEAD 1) composes cleanly on the same k_proj tensor.
    fresh
        .execute(
            &parse(&format!(
                "REMOVE INPUT DIRECTION \"{}\" FROM \"3.self_attn.k_proj.weight\" HEAD 0 SCALE 0.5 NUMERICS \"f64-seq-rne1/v1\";",
                q(&v_npy)
            ))
            .unwrap(),
        )
        .unwrap();
    assert_ne!(fresh.predict("[1] [2] [3] [4]", 5).unwrap().top, live_pred);
}

#[test]
fn head_shorthand_and_input_direction_fail_closed_on_invalid_targets() {
    let (dir, container) = build_lfm3_container();
    let v_npy = dir.path().join("v.npy");
    write_unit_vec_npy(&v_npy, HIDDEN);
    let mut s = open(&container);

    // KV head index 2 out of range for KV_HEADS=2.
    let err = s
        .execute(
            &parse(&format!(
                "REMOVE INPUT DIRECTION \"{}\" FROM \"3.self_attn.k_proj.weight\" HEAD 2 SCALE 1.0 NUMERICS \"f64-seq-rne1/v1\";",
                q(&v_npy)
            ))
            .unwrap(),
        )
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("outside") && err.contains("2 KV heads"),
        "{err}"
    );

    // HEAD on a non-attention tensor refused.
    let err = s
        .execute(
            &parse(&format!(
                "REMOVE DIRECTION \"{}\" FROM \"3.feed_forward.w2.weight\" HEAD 0 SCALE 1.0 NUMERICS \"f64-seq-rne1/v1\";",
                q(&v_npy)
            ))
            .unwrap(),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("softmax o_proj"), "{err}");

    // Empty / reversed range rejected at parse time.
    assert!(parse("REMOVE DIRECTION \"u.npy\" FROM \"t\" COLUMNS (10..10) SCALE 1 NUMERICS \"f64-seq-rne1/v1\";").is_err());
    assert!(parse("REMOVE DIRECTION \"u.npy\" FROM \"t\" COLUMNS (12..8) SCALE 1 NUMERICS \"f64-seq-rne1/v1\";").is_err());
}
