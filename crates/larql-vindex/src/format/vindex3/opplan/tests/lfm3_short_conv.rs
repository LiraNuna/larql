//! Per-channel `ShortConv` observation and channel intervention gates on LFM3.

use super::lfm3::{max_abs_diff, write_miniature_lfm3, Lfm3FixtureOpts, HIDDEN};
use crate::format::vindex3::encode::checkpoint::encode_checkpoint;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::intervene::InterventionPlan;
use crate::format::vindex3::opplan::exec::intervene_heads::HeadInterventionPlan;
use crate::format::vindex3::opplan::exec::intervene_short_conv::{
    ShortConvAddress, ShortConvIntervention, ShortConvInterventionKind, ShortConvInterventionPlan,
};
use crate::format::vindex3::opplan::exec::kv::RowKvState;
use crate::format::vindex3::opplan::exec::observe::{
    NoopObserver, ShortConvRecord, StepEvent, StepObserver,
};
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::plan_component_ops;

struct CapturedShortConv {
    layer: usize,
    position: usize,
    gate_b: Vec<f32>,
    gate_c: Vec<f32>,
    conv_out: Vec<f32>,
    gated_y: Vec<f32>,
}

#[derive(Default)]
struct ScCapture {
    records: Vec<CapturedShortConv>,
    events: Vec<StepEvent>,
}

impl StepObserver for ScCapture {
    fn event(&mut self, event: StepEvent) {
        if matches!(
            event,
            StepEvent::ShortConvObserved { .. } | StepEvent::ShortConvIntervened { .. }
        ) {
            self.events.push(event);
        }
    }

    fn wants_short_conv(&self) -> bool {
        true
    }

    fn short_conv(&mut self, record: ShortConvRecord<'_>) {
        self.records.push(CapturedShortConv {
            layer: record.layer,
            position: record.position,
            gate_b: record.gate_b.to_vec(),
            gate_c: record.gate_c.to_vec(),
            conv_out: record.conv_out.to_vec(),
            gated_y: record.gated_y.to_vec(),
        });
    }
}

#[test]
fn lfm3_short_conv_observation_and_channel_interventions() {
    let dir = tempfile::tempdir().unwrap();
    write_miniature_lfm3(dir.path(), Lfm3FixtureOpts::default());
    let out = tempfile::tempdir().unwrap();
    let container = out.path().join("lfm3-sc-obs.vindex3");
    encode_checkpoint(dir.path(), &container).expect("encode LFM3");

    let inspection = inspect_container(&container, false).unwrap();
    let op_plan = plan_component_ops(&inspection, &container, "target")
        .unwrap()
        .plan
        .unwrap();
    let store = OperandStore::open(&container, &inspection).unwrap();
    let prod = ProductionBackend::new();
    let tokens = [2u32, 7, 14, 5, 19];
    let last_pos = tokens.len() - 1;

    // 1. Baseline unobserved logits.
    let mut base_sess =
        DecodeSession::new(&op_plan, &store, &prod, Box::new(RowKvState::default())).unwrap();
    let mut base_logits = None;
    for &t in &tokens {
        base_logits = base_sess.step(t).unwrap().logits;
    }
    let base_logits = base_logits.unwrap();

    // 2. Observed decode: bit-exact logits, records only on ShortConv layers (0 & 2),
    //    and `gated_y[c] == gate_c[c] * conv_out[c]` for every channel.
    let mut obs_sess =
        DecodeSession::new(&op_plan, &store, &prod, Box::new(RowKvState::default())).unwrap();
    let mut capture = ScCapture::default();
    let mut obs_logits = None;
    for &t in &tokens {
        obs_logits = obs_sess.step_observed(t, &mut capture).unwrap().logits;
    }
    assert_eq!(
        obs_logits.unwrap(),
        base_logits,
        "observation must be bit-exact"
    );
    assert_eq!(capture.records.len(), 2 * tokens.len());
    for rec in &capture.records {
        assert!(
            rec.layer == 0 || rec.layer == 2,
            "only ShortConv layers: {}",
            rec.layer
        );
        assert_eq!(rec.gate_b.len(), HIDDEN);
        for c in 0..HIDDEN {
            assert_eq!(rec.gated_y[c], rec.gate_c[c] * rec.conv_out[c]);
        }
    }
    assert_eq!(
        capture
            .events
            .iter()
            .filter(|e| matches!(
                e,
                StepEvent::ShortConvObserved {
                    channels: HIDDEN,
                    ..
                }
            ))
            .count(),
        2 * tokens.len()
    );

    // 3. Admission refusals: non-ShortConv layer, out-of-range channel, overlap, non-finite.
    assert!(
        ShortConvIntervention::scale(ShortConvAddress::new(0, 0, [0]).unwrap(), f32::NAN).is_err()
    );
    assert!(ShortConvIntervention::replace(
        ShortConvAddress::new(0, 0, [0]).unwrap(),
        f32::INFINITY
    )
    .is_err());
    let overlap = ShortConvInterventionPlan::none()
        .with(ShortConvIntervention::zero(
            ShortConvAddress::new(0, 0, [0]).unwrap(),
        ))
        .unwrap()
        .with(
            ShortConvIntervention::scale(ShortConvAddress::new(0, 0, [0]).unwrap(), 0.5).unwrap(),
        );
    assert!(overlap.is_err());

    let bad_layer = ShortConvInterventionPlan::none()
        .with(ShortConvIntervention::zero(
            ShortConvAddress::new(1, 0, [0]).unwrap(),
        ))
        .unwrap();
    let mut ref_sess =
        DecodeSession::new(&op_plan, &store, &prod, Box::new(RowKvState::default())).unwrap();
    let err = ref_sess
        .step_short_conv_intervened(
            tokens[0],
            &mut NoopObserver,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
            &bad_layer,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("not a ShortConv layer"), "{err}");

    let bad_ch = ShortConvInterventionPlan::none()
        .with(ShortConvIntervention::zero(
            ShortConvAddress::new(0, HIDDEN, [0]).unwrap(),
        ))
        .unwrap();
    let err = ref_sess
        .step_short_conv_intervened(
            tokens[0],
            &mut NoopObserver,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
            &bad_ch,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("has 16 channels"), "{err}");

    // 4. Scale(1.0) is bit-exact no-op; Zero + Scale(0.25) + Replace(2.5) fire and alter logits.
    let noop_plan = ShortConvInterventionPlan::none()
        .with(
            ShortConvIntervention::scale(ShortConvAddress::new(0, 3, [last_pos]).unwrap(), 1.0)
                .unwrap(),
        )
        .unwrap();
    let mut noop_sess =
        DecodeSession::new(&op_plan, &store, &prod, Box::new(RowKvState::default())).unwrap();
    for &t in &tokens[..last_pos] {
        noop_sess.step(t).unwrap();
    }
    let noop_out = noop_sess
        .step_short_conv_intervened(
            tokens[last_pos],
            &mut NoopObserver,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
            &noop_plan,
        )
        .unwrap();
    assert_eq!(noop_out.short_conv_firings.len(), 1);
    assert_eq!(
        noop_out.logits.unwrap(),
        base_logits,
        "Scale(1.0) must be bit-exact"
    );

    let active_plan = ShortConvInterventionPlan::none()
        .with(ShortConvIntervention::zero(
            ShortConvAddress::new(0, 0, [last_pos]).unwrap(),
        ))
        .unwrap()
        .with(
            ShortConvIntervention::scale(ShortConvAddress::new(0, 1, [last_pos]).unwrap(), 0.25)
                .unwrap(),
        )
        .unwrap()
        .with(
            ShortConvIntervention::replace(ShortConvAddress::new(2, 5, [last_pos]).unwrap(), 2.5)
                .unwrap(),
        )
        .unwrap();
    let mut int_sess =
        DecodeSession::new(&op_plan, &store, &prod, Box::new(RowKvState::default())).unwrap();
    for &t in &tokens[..last_pos] {
        int_sess.step(t).unwrap();
    }
    let mut int_obs = ScCapture::default();
    let int_out = int_sess
        .step_short_conv_intervened(
            tokens[last_pos],
            &mut int_obs,
            &InterventionPlan::none(),
            &HeadInterventionPlan::none(),
            &active_plan,
        )
        .unwrap();
    assert_eq!(int_out.short_conv_firings.len(), 3);
    assert_eq!(
        int_out.short_conv_firings[0].kind,
        ShortConvInterventionKind::Zero
    );
    assert_eq!(
        int_out.short_conv_firings[1].kind,
        ShortConvInterventionKind::Scale
    );
    assert_eq!(
        int_out.short_conv_firings[2].kind,
        ShortConvInterventionKind::Replace
    );
    // Layer 0 record captured on the intervened step must still reflect pre-intervention gated_y.
    let base_l0_last = capture
        .records
        .iter()
        .find(|r| r.layer == 0 && r.position == last_pos)
        .unwrap();
    let int_l0_last = int_obs
        .records
        .iter()
        .find(|r| r.layer == 0 && r.position == last_pos)
        .unwrap();
    assert_eq!(
        int_l0_last.gated_y, base_l0_last.gated_y,
        "ShortConvRecord must capture pre-intervention gated_y"
    );
    assert_eq!(
        int_obs
            .events
            .iter()
            .filter(|e| matches!(e, StepEvent::ShortConvIntervened { .. }))
            .count(),
        3
    );
    assert!(
        max_abs_diff(&int_out.logits.unwrap(), &base_logits) > 1e-3,
        "ShortConv channel interventions must measurably alter logits"
    );
}
