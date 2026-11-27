//! Integration tests for counterfactual `PATCH HEAD / COLUMN / NEURON / CHANNEL AT LAYER <l> SCALE <s>;`.

use std::path::Path;

use larql_lql::ast::{DlaTarget, PatchUnitTarget};
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

#[test]
fn parses_patch_unit_forms() {
    let s1 = parse("PATCH HEAD 2 AT LAYER 1 SCALE 0.0;").unwrap();
    assert!(!s1.writes_artifact());
    assert_eq!(s1.verb(), "PATCH");
    assert!(matches!(
        s1,
        Statement::PatchUnit {
            target: PatchUnitTarget::Heads(ref h),
            layer: 1,
            scale,
        } if h == &[2] && scale == 0.0
    ));

    let s2 = parse("PATCH NEURONS (3, 5..7) AT LAYER 0 SCALE 0.5;").unwrap();
    assert!(matches!(
        s2,
        Statement::PatchUnit {
            target: PatchUnitTarget::Columns(ref c),
            layer: 0,
            scale,
        } if c == &[3, 5, 6] && (scale - 0.5).abs() < 1e-12
    ));

    let s3 = parse("PATCH CHANNEL 4 AT LAYER 0 SCALE -1.5;").unwrap();
    assert!(matches!(
        s3,
        Statement::PatchUnit {
            target: PatchUnitTarget::Channels(ref c),
            layer: 0,
            scale,
        } if c == &[4] && (scale + 1.5).abs() < 1e-12
    ));
}

#[test]
fn counterfactual_patch_zeroes_exact_head_neuron_and_channel_and_discards_cleanly() {
    let (_dir, vindex) = build_lfm3_fixture();
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
            Some(INTERMEDIATE as u32 * 2),
            Some(Q_HEADS as u32),
            Some(HIDDEN as u32),
        )
        .unwrap();

    // 1. Ablate L1.H01 -> exact 0.0000 DLA contribution!
    let base_h1 = base_dla
        .top_heads
        .iter()
        .find(|h| h.layer == 1 && h.head == 1)
        .unwrap()
        .contribution;
    assert!(base_h1.abs() > 1e-4);

    sess.execute(&parse("PATCH HEAD 1 AT LAYER 1 SCALE 0.0;").unwrap())
        .unwrap();
    let ablated_head = sess
        .dla(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
            None,
            None,
            Some(Q_HEADS as u32),
            None,
        )
        .unwrap();
    let h1_after = ablated_head
        .top_heads
        .iter()
        .find(|h| h.layer == 1 && h.head == 1)
        .unwrap()
        .contribution;
    assert_eq!(h1_after, 0.0, "ablated head must contribute exact 0.0");

    // DISCARD restores exact base bit-for-bit!
    sess.execute(&parse("DISCARD WEIGHT EDITS;").unwrap())
        .unwrap();
    let restored = sess.predict("alpha beta", 4).unwrap();
    for (a, b) in restored.top.iter().zip(base_pred.top.iter()) {
        assert_eq!(a.token_id, b.token_id);
        assert_eq!(a.probability.to_bits(), b.probability.to_bits());
    }

    // 2. Ablate L1 neuron 5 -> exact 0.0000 DLA contribution!
    sess.execute(&parse("PATCH NEURON 5 AT LAYER 1 SCALE 0.0;").unwrap())
        .unwrap();
    let ablated_neuron = sess
        .dla(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
            None,
            Some(INTERMEDIATE as u32 * 2),
            None,
            None,
        )
        .unwrap();
    let n5_after = ablated_neuron
        .top_columns
        .iter()
        .find(|c| c.layer == 1 && c.column == 5)
        .unwrap()
        .contribution;
    assert_eq!(n5_after, 0.0, "ablated neuron must contribute exact 0.0");
    sess.execute(&parse("DISCARD WEIGHT EDITS;").unwrap())
        .unwrap();

    // 3. Ablate L0 ShortConv channel 2 -> exact 0.0000 DLA contribution!
    sess.execute(&parse("PATCH CHANNEL 2 AT LAYER 0 SCALE 0.0;").unwrap())
        .unwrap();
    let ablated_chan = sess
        .dla(
            "alpha beta",
            DlaTarget::Text("gamma".into()),
            DlaTarget::Text("alpha".into()),
            None,
            None,
            None,
            Some(HIDDEN as u32),
        )
        .unwrap();
    let ch2_after = ablated_chan
        .top_channels
        .iter()
        .find(|c| c.layer == 0 && c.channel == 2)
        .unwrap()
        .contribution;
    assert_eq!(ch2_after, 0.0, "ablated channel must contribute exact 0.0");
}
