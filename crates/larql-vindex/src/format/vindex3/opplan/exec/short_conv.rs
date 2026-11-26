//! Reference ShortConv (`Lfm3ShortConv`): gated causal depthwise-conv1d recurrence.
//!
//! Transcribed directly from `Lfm3ShortConv.forward` (`lfm3_conv.py`):
//! ```text
//! BCx      = in_proj(x) [+ in_proj_bias]       // [T, 3 * D]
//! B, C, x' = chunk(BCx, 3)                     // each [T, D]
//! Bx       = B * x'                            // [T, D]
//! conv_out = depthwise_causal_conv1d(Bx, w)    // width L_cache, NO nonlinearity
//! y        = C * conv_out                      // [T, D]
//! out      = out_proj(y) [+ out_proj_bias]     // [T, D]
//! ```
//!
//! State: one durable rolling buffer of shape `[D, L_cache]` in fp32
//! (`StateInitialization::Zeros`), holding the last `L_cache` values of `Bx`.
//! Every token shifts the rolling window left by 1, writes `Bx[t]` at the
//! trailing slot, and computes the causal dot product — so prefill, chunked
//! prefill, and token-by-token decode are bit-identical.

#![allow(clippy::needless_range_loop)]

use super::super::ShortConvOp;
use super::continuation::{
    RecurrentBufferGeometry, RecurrentGeometry, RecurrentState, StateInitialization,
};
use super::cpu::WeightRows;
use super::gated_delta::DenseProjections;
use super::timing::{timed, OpClass};

/// Buffer index for the rolling `[hidden_size, conv_l_cache]` conv cache.
pub const CONV_STATE: usize = 0;

/// Continuation state geometry for one ShortConv layer.
pub fn state_geometry(op: &ShortConvOp) -> RecurrentGeometry {
    RecurrentGeometry::single(RecurrentBufferGeometry {
        shape: vec![op.hidden_size, op.conv_l_cache],
        dtype: super::super::gated_delta::StateDtype::Float32,
        initialization: StateInitialization::Zeros,
    })
}

/// Resident weights of one ShortConv layer.
pub struct ShortConvWeights<'a> {
    /// `[3 * hidden_size, hidden_size]`
    pub in_proj: WeightRows<'a>,
    /// Optional `[3 * hidden_size]`
    pub in_proj_bias: Option<&'a [f32]>,
    /// `[hidden_size, 1, conv_l_cache]` flattened row-major `[d * L + i]`
    pub conv_weight: &'a [f32],
    /// Optional `[hidden_size]`
    pub conv_bias: Option<&'a [f32]>,
    /// `[hidden_size, hidden_size]`
    pub out_proj: WeightRows<'a>,
    /// Optional `[hidden_size]`
    pub out_proj_bias: Option<&'a [f32]>,
}

/// Output of one ShortConv forward pass.
#[derive(Debug, Default)]
pub struct ShortConvPlanes {
    /// `[T][hidden_size]`
    pub output: Vec<Vec<f32>>,
}

/// Execute one ShortConv layer over `hidden` (`[T][D]`), advancing `state` in place.
pub fn layer_forward_with(
    op: &ShortConvOp,
    w: &ShortConvWeights<'_>,
    hidden: &[Vec<f32>],
    state: &mut RecurrentState,
    proj: &dyn DenseProjections,
) -> ShortConvPlanes {
    let d = op.hidden_size;
    let l_cache = op.conv_l_cache;
    let t_len = hidden.len();
    if t_len == 0 {
        return ShortConvPlanes::default();
    }
    let _site = super::cpu::ledger::in_site(super::cpu::ledger::Site::Recurrent);

    // Stage 1: fused input projection `[T, 3 * D]`.
    let rows: Vec<&[f32]> = hidden.iter().map(Vec::as_slice).collect();
    let mut mixed = proj.project_many(w.in_proj, &rows, 3 * d);
    if let Some(bias) = w.in_proj_bias {
        for row in &mut mixed {
            for (v, &b) in row.iter_mut().zip(bias) {
                *v += b;
            }
        }
    }

    // Stage 2: gated causal depthwise conv1d + output gating `y = C * conv(B * x)`.
    let mut gated: Vec<Vec<f32>> = vec![vec![0.0; d]; t_len];
    let convolution = timed(OpClass::DeltaConv);
    let cells = state.buffer_mut(CONV_STATE).cells_mut();
    for t in 0..t_len {
        let b_vec = &mixed[t][..d];
        let c_vec = &mixed[t][d..2 * d];
        let x_vec = &mixed[t][2 * d..3 * d];
        let y_row = &mut gated[t];
        for ch in 0..d {
            let bx = b_vec[ch] * x_vec[ch];
            let state_row = &mut cells[ch * l_cache..(ch + 1) * l_cache];
            if l_cache > 1 {
                state_row.copy_within(1..l_cache, 0);
            }
            state_row[l_cache - 1] = bx;
            let taps = &w.conv_weight[ch * l_cache..(ch + 1) * l_cache];
            let mut acc = w.conv_bias.map_or(0.0, |b| b[ch]);
            for i in 0..l_cache {
                acc += state_row[i] * taps[i];
            }
            y_row[ch] = c_vec[ch] * acc;
        }
    }
    drop(convolution);

    // Stage 3: output projection `[T, D]`.
    let gated_refs: Vec<&[f32]> = gated.iter().map(Vec::as_slice).collect();
    let mut output = proj.project_many(w.out_proj, &gated_refs, d);
    if let Some(bias) = w.out_proj_bias {
        for row in &mut output {
            for (v, &b) in row.iter_mut().zip(bias) {
                *v += b;
            }
        }
    }

    ShortConvPlanes { output }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::vindex3::opplan::exec::gated_delta::ScalarProjections;
    use crate::format::vindex3::opplan::OperandRef;

    fn sample_op(d: usize, l_cache: usize) -> ShortConvOp {
        let op_ref = |name: &str, shape: Vec<usize>| OperandRef {
            object: "target.decoder_stack".into(),
            tensor: name.into(),
            dtype: "F32".into(),
            shape,
        };
        ShortConvOp {
            hidden_size: d,
            conv_l_cache: l_cache,
            in_proj: op_ref("0.conv.in_proj.weight", vec![3 * d, d]),
            in_proj_bias: None,
            conv_weight: op_ref("0.conv.conv.weight", vec![d, 1, l_cache]),
            conv_bias: None,
            out_proj: op_ref("0.conv.out_proj.weight", vec![d, d]),
            out_proj_bias: None,
        }
    }

    #[test]
    fn prefill_matches_step_by_step_decode_and_analytical_reference() {
        let d = 3;
        let l_cache = 4;
        let op = sample_op(d, l_cache);
        let in_proj_w: Vec<f32> = (0..(3 * d * d))
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        let in_proj_b: Vec<f32> = (0..(3 * d)).map(|i| 0.05 * (i as f32 - 4.0)).collect();
        let conv_w: Vec<f32> = (0..(d * l_cache))
            .map(|i| ((i as f32) * 0.27).cos())
            .collect();
        let conv_b: Vec<f32> = vec![0.1, -0.2, 0.3];
        let out_proj_w: Vec<f32> = (0..(d * d)).map(|i| ((i as f32) * 0.19).sin()).collect();
        let out_proj_b: Vec<f32> = vec![-0.05, 0.02, 0.07];

        let weights = ShortConvWeights {
            in_proj: WeightRows::F32(&in_proj_w),
            in_proj_bias: Some(&in_proj_b),
            conv_weight: &conv_w,
            conv_bias: Some(&conv_b),
            out_proj: WeightRows::F32(&out_proj_w),
            out_proj_bias: Some(&out_proj_b),
        };

        let seq: Vec<Vec<f32>> = (0..7)
            .map(|t| (0..d).map(|j| ((t * d + j) as f32 * 0.31).cos()).collect())
            .collect();

        // Full batch prefill.
        let mut state_batch = RecurrentState::zeros(&state_geometry(&op));
        let batch_out =
            layer_forward_with(&op, &weights, &seq, &mut state_batch, &ScalarProjections).output;

        // Token-by-token decode.
        let mut state_step = RecurrentState::zeros(&state_geometry(&op));
        let mut step_out = Vec::new();
        for tok in &seq {
            let out = layer_forward_with(
                &op,
                &weights,
                std::slice::from_ref(tok),
                &mut state_step,
                &ScalarProjections,
            )
            .output;
            step_out.push(out[0].clone());
        }

        // Chunked prefill (3 + 4).
        let mut state_chunk = RecurrentState::zeros(&state_geometry(&op));
        let mut chunk_out = layer_forward_with(
            &op,
            &weights,
            &seq[..3],
            &mut state_chunk,
            &ScalarProjections,
        )
        .output;
        chunk_out.extend(
            layer_forward_with(
                &op,
                &weights,
                &seq[3..],
                &mut state_chunk,
                &ScalarProjections,
            )
            .output,
        );

        assert_eq!(batch_out, step_out);
        assert_eq!(batch_out, chunk_out);
        assert_eq!(
            state_batch.buffer(CONV_STATE).cells(),
            state_step.buffer(CONV_STATE).cells()
        );
        assert_eq!(
            state_batch.buffer(CONV_STATE).cells(),
            state_chunk.buffer(CONV_STATE).cells()
        );
    }
}
