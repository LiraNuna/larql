//! Single-pass multi-site `CAPTURE (<site1>, <site2>, …) … TO "<dir>";`
//! and `AT LAYERS <lo>-<hi>` range shorthand.

use larql_lql::ast::{CaptureLayers, CaptureSite};
use larql_lql::{parse, Statement};

use crate::short_conv::HybridShortConvFixture;
use crate::support::*;

#[test]
fn multi_site_capture_matches_single_site_captures_bit_for_bit() {
    let fx = HybridShortConvFixture::new();
    let mut s = fx.session();

    // Single-pass capture across per-layer + ShortConv + exit sites.
    let dir = fx.path("multi_out");
    let reply = run(
        &mut s,
        &format!(
            "CAPTURE (post_attention, residual, ffn.down_input, short_conv.gated_y, logits) \
             AT LAYERS (0, 2) FOR \"{PROMPT}\" POSITIONS ALL TO \"{}\";",
            q(&dir)
        ),
    );
    assert!(
        has(&reply, "Captured 5 site(s) in 1 pass"),
        "expected single-pass summary: {reply:?}"
    );

    // Compare every written .npy file bit-for-bit against an individual single-site capture.
    for (site, layers_clause) in [
        ("post_attention", " AT LAYERS (0, 2)"),
        ("residual", " AT LAYERS (0, 2)"),
        ("ffn.down_input", " AT LAYERS (0, 2)"),
        ("short_conv.gated_y", " AT LAYERS (0, 2)"),
        ("logits", ""),
    ] {
        let single_path = fx.path(&format!("single_{site}.npy"));
        capture(
            &mut s,
            &format!("{site}{layers_clause} FOR \"{PROMPT}\" POSITIONS ALL"),
            &single_path,
        );
        let multi_path = dir.join(format!("{site}.npy"));
        let single = read_npy(&single_path);
        let multi = read_npy(&multi_path);
        assert_eq!(multi.shape, single.shape, "{site} shape");
        assert_eq!(bits(&multi.values), bits(&single.values), "{site} values");
        assert_eq!(sidecar(&multi_path)["site"], site);
        assert_eq!(
            sidecar(&multi_path)["npy_sha256"],
            sha256_hex(&multi.bytes),
            "{site} sha256"
        );
    }

    // `AT LAYERS <lo>-<hi>` range shorthand works across all 4 layers.
    let range_dir = fx.path("range_out");
    run(
        &mut s,
        &format!(
            "CAPTURE (post_attention, residual) AT LAYERS 0-3 FOR \"{PROMPT}\" TO \"{}\";",
            q(&range_dir)
        ),
    );
    assert_eq!(
        read_npy(&range_dir.join("residual.npy")).shape,
        vec![4, 1, crate::short_conv::SC_HIDDEN]
    );

    // Exit-only multi-site capture needs no AT LAYER clause.
    let exit_dir = fx.path("exit_out");
    run(
        &mut s,
        &format!(
            "CAPTURE (final, logits) FOR \"{PROMPT}\" TO \"{}\";",
            q(&exit_dir)
        ),
    );
    assert!(exit_dir.join("final.npy").exists());
    assert!(exit_dir.join("logits.npy").exists());

    // Fail-closed refusals: duplicate site, existing file in dir, missing AT LAYER.
    let dup_dir = fx.path("dup_out");
    let err = try_run(
        &mut s,
        &format!(
            "CAPTURE (logits, logits) FOR \"{PROMPT}\" TO \"{}\";",
            q(&dup_dir)
        ),
    )
    .unwrap_err();
    assert!(err.contains("listed twice"), "{err}");
    assert!(!dup_dir.exists());

    let err = try_run(
        &mut s,
        &format!(
            "CAPTURE (final, logits) FOR \"{PROMPT}\" TO \"{}\";",
            q(&exit_dir)
        ),
    )
    .unwrap_err();
    assert!(err.contains("already exists"), "{err}");

    let missing_layers_dir = fx.path("missing_layers_out");
    let err = try_run(
        &mut s,
        &format!(
            "CAPTURE (residual, logits) FOR \"{PROMPT}\" TO \"{}\";",
            q(&missing_layers_dir)
        ),
    )
    .unwrap_err();
    assert!(err.contains("AT LAYER"), "{err}");
    assert!(!missing_layers_dir.exists());
}

#[test]
fn parse_multi_site_and_layer_range_shorthand() {
    let Statement::CaptureMulti { sites, layers, .. } =
        parse(r#"CAPTURE (post_attention, residual, logits) AT LAYERS 0-3 FOR "p" TO "out_dir";"#)
            .unwrap()
    else {
        panic!("expected CaptureMulti");
    };
    assert_eq!(
        sites,
        vec![
            CaptureSite::PostAttention,
            CaptureSite::Residual,
            CaptureSite::Logits
        ]
    );
    assert_eq!(layers, Some(CaptureLayers::Many(vec![0, 1, 2, 3])));
}
