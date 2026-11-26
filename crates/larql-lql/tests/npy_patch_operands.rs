//! Integration tests for `.npy` vector and matrix operands in `REMOVE DIRECTION` and `ADD LOW RANK`.

use std::path::Path;

use larql_lql::operations::{npy_file_bytes, NpyDescr};
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::encode::checkpoint::encode_checkpoint;
use larql_vindex::format::vindex3::fixtures::{lcg_values, norm_values, ShardBuilder};

const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 24;
const VOCAB: usize = 32;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 4;

fn q(path: &Path) -> String {
    path.display().to_string().replace('\\', "\\\\")
}

fn push_bf16(shard: &mut ShardBuilder, name: &str, shape: &[usize], values: &[f32]) {
    let bytes = larql_models::quant::half::encode_bf16(values);
    shard.push_bytes(name, "BF16", shape, &bytes);
}

fn build_container() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let ckpt = dir.path().join("ckpt");
    std::fs::create_dir_all(&ckpt).unwrap();

    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "torch_dtype": "bfloat16",
        "hidden_size": HIDDEN,
        "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": 2,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true
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
        "model.norm.weight",
        &[HIDDEN],
        &norm_values(HIDDEN, 2),
    );
    for layer in 0..2 {
        let seed = 100 + layer as u64 * 30;
        let p = format!("model.layers.{layer}");
        push_bf16(
            &mut shard,
            &format!("{p}.input_layernorm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.post_attention_layernorm.weight"),
            &[HIDDEN],
            &norm_values(HIDDEN, seed + 1),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.self_attn.q_proj.weight"),
            &[Q_HEADS * HEAD_DIM, HIDDEN],
            &lcg_values(Q_HEADS * HEAD_DIM * HIDDEN, seed + 2),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.self_attn.k_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
            &lcg_values(KV_HEADS * HEAD_DIM * HIDDEN, seed + 3),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.self_attn.v_proj.weight"),
            &[KV_HEADS * HEAD_DIM, HIDDEN],
            &lcg_values(KV_HEADS * HEAD_DIM * HIDDEN, seed + 4),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, Q_HEADS * HEAD_DIM],
            &lcg_values(HIDDEN * Q_HEADS * HEAD_DIM, seed + 5),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.mlp.gate_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
            &lcg_values(INTERMEDIATE * HIDDEN, seed + 6),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.mlp.up_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
            &lcg_values(INTERMEDIATE * HIDDEN, seed + 7),
        );
        push_bf16(
            &mut shard,
            &format!("{p}.mlp.down_proj.weight"),
            &[HIDDEN, INTERMEDIATE],
            &lcg_values(HIDDEN * INTERMEDIATE, seed + 8),
        );
    }
    shard.write(&ckpt);

    let container = dir.path().join("model.vindex3");
    encode_checkpoint(&ckpt, &container).expect("encode BF16 container");

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
    let stmt = format!("USE \"{}\";", q(container));
    s.execute(&parse(&stmt).unwrap()).unwrap();
    s
}

fn write_f32_npy(path: &Path, shape: &[usize], values: &[f32]) {
    let body: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = npy_file_bytes(NpyDescr::F4, shape, &body).unwrap();
    std::fs::write(path, bytes).unwrap();
}

fn write_f64_npy(path: &Path, shape: &[usize], values: &[f64]) {
    let body: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = npy_file_bytes(NpyDescr::F8, shape, &body).unwrap();
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn captured_npy_pipes_directly_into_remove_direction_and_matches_hex_json() {
    let (dir, container) = build_container();
    let mut s_npy = open_session(&container);

    // Capture a `[1, hidden]` residual directly to `.npy` and pipe it into REMOVE DIRECTION.
    let cap_npy = dir.path().join("dir.npy");
    s_npy
        .execute(
            &parse(&format!(
                "CAPTURE residual AT LAYER 0 FOR \"[1] [2] [3]\" POSITIONS LAST TO \"{}\";",
                q(&cap_npy)
            ))
            .unwrap(),
        )
        .unwrap();

    let captured_values = {
        let spec = larql_lql::operations::CaptureSpec {
            site: larql_lql::operations::CaptureSite::Residual,
            layers: Some(&larql_lql::operations::CaptureLayers::One(0)),
            prompt: "[1] [2] [3]",
            continuation: None,
            positions: &larql_lql::operations::CapturePositions::Last,
        };
        s_npy.capture(&spec).unwrap().values()
    };
    assert_eq!(captured_values.len(), HIDDEN);

    // Stage REMOVE DIRECTION directly from the captured `[1, hidden]` `.npy` file.
    s_npy
        .execute(
            &parse(&format!(
                "REMOVE DIRECTION \"{}\" FROM \"1.mlp.down_proj.weight\" COLUMNS (0, 1, 2) SCALE 0.5 NUMERICS \"f64-seq-rne1/v1\";",
                q(&cap_npy)
            ))
            .unwrap(),
        )
        .unwrap();
    let pred_npy = s_npy.predict("[1] [2] [3]", 5).unwrap();

    // Build the exact same direction as hex-JSON and verify bit-for-bit identical predictions.
    let json_path = dir.path().join("dir.json");
    let hex_bits: Vec<String> = captured_values
        .iter()
        .map(|&v| format!("{:016x}", f64::from(v).to_bits()))
        .collect();
    std::fs::write(
        &json_path,
        serde_json::json!({ "bits": hex_bits }).to_string(),
    )
    .unwrap();
    let mut s_json = open_session(&container);
    s_json
        .execute(
            &parse(&format!(
                "REMOVE DIRECTION \"{}\" FROM \"1.mlp.down_proj.weight\" COLUMNS (0, 1, 2) SCALE 0.5 NUMERICS \"f64-seq-rne1/v1\";",
                q(&json_path)
            ))
            .unwrap(),
        )
        .unwrap();
    let pred_json = s_json.predict("[1] [2] [3]", 5).unwrap();
    assert_eq!(pred_npy.top, pred_json.top);

    // Also test `<f8` `.npy` 1D vector in REMOVE DIRECTION.
    let f64_vec: Vec<f64> = (0..HIDDEN).map(|i| (i as f64 + 1.0) * 0.125).collect();
    let f64_npy = dir.path().join("dir_f64.npy");
    write_f64_npy(&f64_npy, &[HIDDEN], &f64_vec);
    let mut s_f64 = open_session(&container);
    s_f64
        .execute(
            &parse(&format!(
                "REMOVE DIRECTION \"{}\" FROM \"1.mlp.down_proj.weight\" COLUMNS (3, 4) SCALE 0.25 NUMERICS \"f64-seq-rne2/v1\";",
                q(&f64_npy)
            ))
            .unwrap(),
        )
        .unwrap();
    assert_ne!(
        s_f64.predict("[1] [2] [3]", 5).unwrap().top,
        open_session(&container)
            .predict("[1] [2] [3]", 5)
            .unwrap()
            .top
    );
}

#[test]
fn add_low_rank_accepts_direct_1d_and_2d_npy_and_json_npy_paths() {
    let (dir, container) = build_container();
    let base_pred = open_session(&container)
        .predict("[1] [2] [3]", 5)
        .unwrap()
        .top;

    // 1) Direct 1D `[rows]` `.npy` (rank-1 broadcast across listed columns).
    let vec_vals: Vec<f32> = (0..HIDDEN).map(|i| (i as f32 - 7.5) * 0.25).collect();
    let vec_npy = dir.path().join("vec.npy");
    write_f32_npy(&vec_npy, &[1, HIDDEN], &vec_vals);
    let mut s_1d = open_session(&container);
    s_1d.execute(
        &parse(&format!(
            "ADD LOW RANK \"{}\" TO \"1.mlp.down_proj.weight\" COLUMNS (0, 2) NUMERICS \"f64-seq-rne1/v1\";",
            q(&vec_npy)
        ))
        .unwrap(),
    )
    .unwrap();
    let pred_1d = s_1d.predict("[1] [2] [3]", 5).unwrap().top;
    assert_ne!(pred_1d, base_pred);

    // 2) Direct 2D `[rows, 2]` `.npy` with identical columns matches the rank-1 broadcast!
    let mut mat_vals = Vec::with_capacity(HIDDEN * 2);
    for &v in &vec_vals {
        mat_vals.push(v);
        mat_vals.push(v);
    }
    let mat_npy = dir.path().join("mat.npy");
    write_f32_npy(&mat_npy, &[HIDDEN, 2], &mat_vals);
    let mut s_2d = open_session(&container);
    s_2d.execute(
        &parse(&format!(
            "ADD LOW RANK \"{}\" TO \"1.mlp.down_proj.weight\" COLUMNS (0, 2) NUMERICS \"f64-seq-rne1/v1\";",
            q(&mat_npy)
        ))
        .unwrap(),
    )
    .unwrap();
    assert_eq!(s_2d.predict("[1] [2] [3]", 5).unwrap().top, pred_1d);

    // 3) JSON file referencing `.npy` factor paths (`a.npy` and `b.npy`).
    let a_npy = dir.path().join("a.npy");
    let b_npy = dir.path().join("b.npy");
    write_f32_npy(&a_npy, &[HIDDEN, 1], &vec_vals);
    write_f32_npy(&b_npy, &[1, 2], &[1.0, 1.0]);
    let json_path = dir.path().join("factors.json");
    std::fs::write(
        &json_path,
        serde_json::json!({
            "rank": 1,
            "a": "a.npy",
            "b": "b.npy",
        })
        .to_string(),
    )
    .unwrap();
    let mut s_json_npy = open_session(&container);
    s_json_npy
        .execute(
            &parse(&format!(
                "ADD LOW RANK \"{}\" TO \"1.mlp.down_proj.weight\" COLUMNS (0, 2) NUMERICS \"f64-seq-rne1/v1\";",
                q(&json_path)
            ))
            .unwrap(),
        )
        .unwrap();
    assert_eq!(s_json_npy.predict("[1] [2] [3]", 5).unwrap().top, pred_1d);
}

#[test]
fn npy_operands_fail_closed_on_bad_dtype_order_truncation_or_precision_narrowing() {
    let (dir, container) = build_container();
    let mut s = open_session(&container);

    // Unsupported dtype (<u4).
    let u4_npy = dir.path().join("u4.npy");
    let u4_body: Vec<u8> = (0u32..HIDDEN as u32)
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(
        &u4_npy,
        npy_file_bytes(NpyDescr::U4, &[HIDDEN], &u4_body).unwrap(),
    )
    .unwrap();
    let err = s
        .execute(
            &parse(&format!(
                "REMOVE DIRECTION \"{}\" FROM \"1.mlp.down_proj.weight\" COLUMNS (0) SCALE 0.5 NUMERICS \"f64-seq-rne1/v1\";",
                q(&u4_npy)
            ))
            .unwrap(),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("unsupported .npy dtype"), "{err}");

    // Fortran-order header refused.
    let fortran_npy = dir.path().join("fortran.npy");
    let header_dict =
        format!("{{'descr': '<f4', 'fortran_order': True, 'shape': ({HIDDEN},), }}   \n");
    let mut raw = Vec::new();
    raw.extend_from_slice(b"\x93NUMPY\x01\x00");
    raw.extend_from_slice(&(header_dict.len() as u16).to_le_bytes());
    raw.extend_from_slice(header_dict.as_bytes());
    raw.extend_from_slice(&[0u8; HIDDEN * 4]);
    std::fs::write(&fortran_npy, raw).unwrap();
    let err = s
        .execute(
            &parse(&format!(
                "REMOVE DIRECTION \"{}\" FROM \"1.mlp.down_proj.weight\" COLUMNS (0) SCALE 0.5 NUMERICS \"f64-seq-rne1/v1\";",
                q(&fortran_npy)
            ))
            .unwrap(),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("Fortran-order"), "{err}");

    // `<f8` `.npy` refused under `f32-seq-rne1/v1` (narrowing would round).
    let f64_npy = dir.path().join("f64.npy");
    write_f64_npy(&f64_npy, &[HIDDEN], &[0.5f64; HIDDEN]);
    let err = s
        .execute(
            &parse(&format!(
                "ADD LOW RANK \"{}\" TO \"1.mlp.down_proj.weight\" COLUMNS (0) NUMERICS \"f32-seq-rne1/v1\";",
                q(&f64_npy)
            ))
            .unwrap(),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("narrowing"), "{err}");

    // Wrong 2D shape in REMOVE DIRECTION refused.
    let bad_shape = dir.path().join("bad_shape.npy");
    write_f32_npy(&bad_shape, &[2, HIDDEN], &[0.5f32; 2 * HIDDEN]);
    let err = s
        .execute(
            &parse(&format!(
                "REMOVE DIRECTION \"{}\" FROM \"1.mlp.down_proj.weight\" COLUMNS (0) SCALE 0.5 NUMERICS \"f64-seq-rne1/v1\";",
                q(&bad_shape)
            ))
            .unwrap(),
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("expected a 1D vector"), "{err}");
}
