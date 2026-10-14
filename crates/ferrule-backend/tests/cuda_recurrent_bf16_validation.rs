#![cfg(feature = "cuda")]
use ferrule_backend::cuda::operator_owner::{CudaBf16Buffer, CudaOperators};
use ferrule_backend::cuda::operators::recurrent::*;
use ferrule_backend::cuda::operators::recurrent_bf16::*;

fn buffers(
    op: &CudaOperators,
    foreign: &CudaOperators,
    lengths: &[usize],
    bad: usize,
    wrong_owner: bool,
) -> Vec<CudaBf16Buffer> {
    lengths
        .iter()
        .enumerate()
        .map(|(i, &n)| {
            let owner = if i == bad && wrong_owner { foreign } else { op };
            owner
                .upload_bf16_words(&vec![
                    0x3f80;
                    if i == bad && !wrong_owner { n - 1 } else { n }
                ])
                .unwrap()
        })
        .collect()
}
fn rejected(op: &CudaOperators, result: ferrule_common::Result<()>) {
    assert!(result.is_err());
    assert_eq!(
        op.counters(),
        Default::default(),
        "preflight must not submit GPU work"
    );
}
fn unchanged(owner: &CudaOperators, b: &CudaBf16Buffer) {
    assert_eq!(owner.download_bf16_buffer(b).unwrap(), vec![1.0; b.len()]);
}

#[test]
#[ignore = "requires CUDA GPU; every typed input/output owner and extent, same ordinal different custodians"]
fn every_bf16_and_f32_operand_is_validated_before_launch() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let cl = CausalConv1dLayout {
        rows: 3,
        channels: 5,
        kernel_size: 4,
    };
    let dl = GatedDeltaNetLayout {
        rows: 3,
        key_heads: 2,
        value_heads: 4,
        key_dim: 3,
        value_dim: 2,
    };
    for wrong_owner in [false, true] {
        for bad in 0..5 {
            let mut b = buffers(&op, &foreign, &[15, 20, 5, 20, 15], bad, wrong_owner);
            let (reads, writes) = b.split_at_mut(3);
            let (history, output) = writes.split_at_mut(1);
            op.reset_counters();
            rejected(
                &op,
                op.causal_depthwise_conv1d_silu_bf16_into(
                    Bf16CausalConv1dBuffers {
                        input: &reads[0],
                        weight: &reads[1],
                        bias: Some(&reads[2]),
                        history: &mut history[0],
                        output: &mut output[0],
                    },
                    cl,
                ),
            );
            for i in [3, 4] {
                unchanged(
                    if i == bad && wrong_owner {
                        &foreign
                    } else {
                        &op
                    },
                    &b[i],
                );
            }
        }
        // Slots 0..5 are BF16; slot 6 is the F32 recurrent state.
        for bad in 0..7 {
            let mut b = buffers(&op, &foreign, &[60, 12, 12, 4, 4, 24], bad, wrong_owner);
            let state_owner = if bad == 6 && wrong_owner {
                &foreign
            } else {
                &op
            };
            let mut state = state_owner
                .upload_f32_buffer(&vec![0.37; if bad == 6 && !wrong_owner { 23 } else { 24 }])
                .unwrap();
            let (reads, out) = b.split_at_mut(5);
            op.reset_counters();
            rejected(
                &op,
                op.gated_delta_net_bf16_into(
                    Bf16GatedDeltaNetBuffers {
                        qkv: &reads[0],
                        a: &reads[1],
                        b: &reads[2],
                        a_log: &reads[3],
                        dt_bias: &reads[4],
                        state: &mut state,
                        output: &mut out[0],
                    },
                    dl,
                ),
            );
            unchanged(
                if bad == 5 && wrong_owner {
                    &foreign
                } else {
                    &op
                },
                &b[5],
            );
            assert_eq!(
                state_owner.download_f32_buffer(&state).unwrap(),
                vec![0.37; state.len()]
            );
        }
        for bad in 0..4 {
            let mut b = buffers(&op, &foreign, &[15, 5, 15, 15], bad, wrong_owner);
            let (reads, out) = b.split_at_mut(3);
            op.reset_counters();
            rejected(
                &op,
                op.gated_rms_norm_bf16_into(
                    &reads[0],
                    &reads[1],
                    &reads[2],
                    &mut out[0],
                    F32RowsLayout { rows: 3, width: 5 },
                    1e-6,
                ),
            );
            unchanged(
                if bad == 3 && wrong_owner {
                    &foreign
                } else {
                    &op
                },
                &b[3],
            );
        }
    }
    let mut b = buffers(&op, &foreign, &[15, 5, 15, 15], usize::MAX, false);
    let (reads, out) = b.split_at_mut(3);
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        op.reset_counters();
        rejected(
            &op,
            op.gated_rms_norm_bf16_into(
                &reads[0],
                &reads[1],
                &reads[2],
                &mut out[0],
                F32RowsLayout { rows: 3, width: 5 },
                epsilon,
            ),
        );
    }
    for layout in [
        F32RowsLayout { rows: 0, width: 5 },
        F32RowsLayout {
            rows: usize::MAX,
            width: 5,
        },
    ] {
        op.reset_counters();
        rejected(
            &op,
            op.gated_rms_norm_bf16_into(&reads[0], &reads[1], &reads[2], &mut out[0], layout, 1e-6),
        );
    }
    unchanged(&op, &out[0]);
}
