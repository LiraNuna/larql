//! Integration tests for native continuous soft-prompt prefix injection
//! (`LOAD SOFT_PROMPT` / `CLEAR SOFT_PROMPT`) across `.npy`, `.safetensors`, and PyTorch `.pt`.

use std::path::Path;

use larql_lql::ast::{CapturePositions, CaptureSite, DlaTarget};
use larql_lql::operations::{npy_file_bytes, CaptureSpec, NpyDescr};
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

/// Build a minimal uncompressed STORED ZIP archive matching `torch.save({"soft_emb": ...})`.
fn write_synthetic_pytorch_zip(path: &Path, values: &[f32]) {
    let pkl: &[u8] = b"\x80\x02}q\x00X\x08\x00\x00\x00soft_embq\x01ctorch.FloatStorage\nq\x02u.";
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let entries: Vec<(&str, &[u8])> = vec![("sp/data.pkl", pkl), ("sp/data/0", &raw)];

    let mut buf = Vec::new();
    let mut cd = Vec::new();
    for (name, data) in &entries {
        let local_off = buf.len() as u32;
        let name_b = name.as_bytes();
        let sz = data.len() as u32;
        // Local file header
        buf.extend_from_slice(b"PK\x03\x04");
        buf.extend_from_slice(&20u16.to_le_bytes()); // version
        buf.extend_from_slice(&0u16.to_le_bytes()); // flags
        buf.extend_from_slice(&0u16.to_le_bytes()); // method = STORED
        buf.extend_from_slice(&0u16.to_le_bytes()); // mod time
        buf.extend_from_slice(&0u16.to_le_bytes()); // mod date
        buf.extend_from_slice(&0u32.to_le_bytes()); // crc32
        buf.extend_from_slice(&sz.to_le_bytes());
        buf.extend_from_slice(&sz.to_le_bytes());
        buf.extend_from_slice(&(name_b.len() as u16).to_le_bytes());
        buf.extend_from_slice(&0u16.to_le_bytes()); // extra len
        buf.extend_from_slice(name_b);
        buf.extend_from_slice(data);

        // Central directory header
        cd.extend_from_slice(b"PK\x01\x02");
        cd.extend_from_slice(&20u16.to_le_bytes());
        cd.extend_from_slice(&20u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes()); // STORED
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u32.to_le_bytes());
        cd.extend_from_slice(&sz.to_le_bytes());
        cd.extend_from_slice(&sz.to_le_bytes());
        cd.extend_from_slice(&(name_b.len() as u16).to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u16.to_le_bytes());
        cd.extend_from_slice(&0u32.to_le_bytes());
        cd.extend_from_slice(&local_off.to_le_bytes());
        cd.extend_from_slice(name_b);
    }

    let cd_offset = buf.len() as u32;
    let cd_size = cd.len() as u32;
    buf.extend_from_slice(&cd);
    // EOCD
    buf.extend_from_slice(b"PK\x05\x06");
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    buf.extend_from_slice(&cd_size.to_le_bytes());
    buf.extend_from_slice(&cd_offset.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    std::fs::write(path, buf).unwrap();
}

#[test]
fn parses_soft_prompt_statements() {
    let s1 = parse(r#"LOAD SOFT_PROMPT "sp16.pt";"#).unwrap();
    assert!(!s1.writes_artifact());
    assert_eq!(s1.verb(), "LOAD SOFT_PROMPT");
    assert!(matches!(s1, Statement::LoadSoftPrompt { ref path } if path == "sp16.pt"));

    let s2 = parse("CLEAR SOFT_PROMPT;").unwrap();
    assert!(!s2.writes_artifact());
    assert_eq!(s2.verb(), "CLEAR SOFT_PROMPT");
    assert!(matches!(s2, Statement::ClearSoftPrompt));
}

#[test]
fn soft_prompt_npy_safetensors_and_pt_match_bit_for_bit_and_clear_restores_base() {
    let (dir, vindex) = build_lfm3_fixture();
    let n_soft = 4usize;
    let vals = lcg_values(n_soft * HIDDEN, 777);

    // 1. Write .npy ([1, 4, 16] <f4)
    let npy_path = dir.path().join("sp.npy");
    let raw_f32: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(
        &npy_path,
        npy_file_bytes(NpyDescr::F4, &[1, n_soft, HIDDEN], &raw_f32).unwrap(),
    )
    .unwrap();

    // 2. Write .safetensors (F32, key "soft_emb")
    let st_dir = dir.path().join("st_dir");
    std::fs::create_dir_all(&st_dir).unwrap();
    let mut shard = ShardBuilder::new();
    shard.push_bytes("soft_emb", "F32", &[n_soft, HIDDEN], &raw_f32);
    shard.write(&st_dir);
    let st_path = st_dir.join("model.safetensors");

    // 3. Write synthetic PyTorch .pt ZIP archive (STORED FloatStorage)
    let pt_path = dir.path().join("sp.pt");
    write_synthetic_pytorch_zip(&pt_path, &vals);

    let mut sess = Session::new();
    sess.execute(&parse(&format!(r#"USE "{}";"#, q(&vindex))).unwrap())
        .unwrap();

    let base_pred = sess.predict("alpha beta", 4).unwrap();
    let base_dla = sess
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

    // Load .pt soft prompt
    let pt_summary = sess.load_soft_prompt(&q(&pt_path)).unwrap();
    assert_eq!(pt_summary.n_soft, n_soft);
    assert_eq!(pt_summary.hidden, HIDDEN);

    let pt_pred = sess.predict("alpha beta", 4).unwrap();
    let pt_dla = sess
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
    assert_ne!(
        base_dla.exact_logit_diff.to_bits(),
        pt_dla.exact_logit_diff.to_bits(),
        "soft prompt must change model predictions"
    );
    assert!(
        (pt_dla.total_dla - pt_dla.pre_softcap_diff).abs() < 1e-3,
        "DLA decomposition must remain exact with soft prompt: total={} pre_softcap={}",
        pt_dla.total_dla,
        pt_dla.pre_softcap_diff
    );

    // CAPTURE attn.weights includes the n_soft prefix tokens in seq_len!
    let cap = sess
        .capture(&CaptureSpec {
            site: CaptureSite::AttnWeights,
            layers: Some(&larql_lql::ast::CaptureLayers::One(1)),
            prompt: "alpha beta",
            continuation: None,
            positions: &CapturePositions::Last,
        })
        .unwrap();
    assert_eq!(cap.shape, vec![Q_HEADS, n_soft + pt_pred.prompt_ids.len()]);
    let w = cap.values();
    let seq_len = n_soft + pt_pred.prompt_ids.len();
    for h in 0..Q_HEADS {
        let sum: f32 = w[h * seq_len..(h + 1) * seq_len].iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "head {h} attn sum = {sum}");
    }

    // GENERATE runs cleanly with active soft prompt
    let gen = sess.generate("alpha beta", 3).unwrap();
    assert!(!gen.token_ids.is_empty());

    // .npy and .safetensors produce bit-for-bit identical predictions & DLA!
    for other in [&npy_path, &st_path] {
        sess.load_soft_prompt(&q(other)).unwrap();
        let pred = sess.predict("alpha beta", 4).unwrap();
        let dla = sess
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
        assert_eq!(
            dla.exact_logit_diff.to_bits(),
            pt_dla.exact_logit_diff.to_bits()
        );
        for (a, b) in pred.top.iter().zip(pt_pred.top.iter()) {
            assert_eq!(a.token_id, b.token_id);
            assert_eq!(a.probability.to_bits(), b.probability.to_bits());
        }
    }

    // CLEAR SOFT_PROMPT restores exact untouched base predictions bit-for-bit!
    sess.execute(&parse("CLEAR SOFT_PROMPT;").unwrap()).unwrap();
    let restored = sess.predict("alpha beta", 4).unwrap();
    for (a, b) in restored.top.iter().zip(base_pred.top.iter()) {
        assert_eq!(a.token_id, b.token_id);
        assert_eq!(a.probability.to_bits(), b.probability.to_bits());
    }
}
