#![cfg(feature = "cuda")]

use ferrule_backend::cuda::context::{CudaF32Buffer, CudaOperators};
use ferrule_backend::cuda::operators::recurrent::*;

fn buffers(
    op: &CudaOperators,
    foreign: &CudaOperators,
    lengths: &[usize],
    bad: usize,
    wrong_owner: bool,
) -> Vec<CudaF32Buffer> {
    lengths
        .iter()
        .enumerate()
        .map(|(i, &len)| {
            let owner = if i == bad && wrong_owner { foreign } else { op };
            let len = if i == bad && !wrong_owner {
                len - 1
            } else {
                len
            };
            owner.upload_f32_buffer(&vec![0.37; len]).unwrap()
        })
        .collect()
}
fn rejected(op: &CudaOperators, result: ferrule_common::Result<()>) {
    assert!(result.is_err());
    assert_eq!(
        op.counters(),
        Default::default(),
        "validation must precede all GPU work"
    );
}
fn unchanged(owner: &CudaOperators, buffer: &CudaF32Buffer) {
    assert_eq!(
        owner.download_f32_buffer(buffer).unwrap(),
        vec![0.37; buffer.len()]
    );
}

#[test]
#[ignore = "requires actual CUDA GPU; distinct context owners on the SAME ordinal"]
fn every_conv_and_delta_buffer_checks_owner_and_exact_shape_before_launch() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let conv = CausalConv1dLayout {
        rows: 3,
        channels: 5,
        kernel_size: 4,
    };
    for wrong_owner in [false, true] {
        for bad in 0..5 {
            let mut b = buffers(&op, &foreign, &[15, 20, 5, 20, 15], bad, wrong_owner);
            let (reads, writes) = b.split_at_mut(3);
            let (history, output) = writes.split_at_mut(1);
            op.reset_counters();
            rejected(
                &op,
                op.causal_depthwise_conv1d_silu_into(
                    CausalConv1dBuffers {
                        input: &reads[0],
                        weight: &reads[1],
                        bias: Some(&reads[2]),
                        history: &mut history[0],
                        output: &mut output[0],
                    },
                    conv,
                ),
            );
            for i in [3, 4] {
                unchanged(
                    if bad == i && wrong_owner {
                        &foreign
                    } else {
                        &op
                    },
                    &b[i],
                );
            }
        }
    }
    let delta = GatedDeltaNetLayout {
        rows: 3,
        key_heads: 2,
        value_heads: 4,
        key_dim: 5,
        value_dim: 7,
    };
    let lengths = [
        delta.qkv_elements().unwrap(),
        12,
        12,
        4,
        4,
        delta.state_elements().unwrap(),
        delta.output_elements().unwrap(),
    ];
    for wrong_owner in [false, true] {
        for bad in 0..7 {
            let mut b = buffers(&op, &foreign, &lengths, bad, wrong_owner);
            let (reads, writes) = b.split_at_mut(5);
            let (state, output) = writes.split_at_mut(1);
            op.reset_counters();
            rejected(
                &op,
                op.gated_delta_net_into(
                    GatedDeltaNetBuffers {
                        qkv: &reads[0],
                        a: &reads[1],
                        b: &reads[2],
                        a_log: &reads[3],
                        dt_bias: &reads[4],
                        state: &mut state[0],
                        output: &mut output[0],
                    },
                    delta,
                ),
            );
            for i in [5, 6] {
                unchanged(
                    if bad == i && wrong_owner {
                        &foreign
                    } else {
                        &op
                    },
                    &b[i],
                );
            }
        }
    }
    // Valid buffers but invalid layouts must also be rejected without submission.
    let mut b = buffers(&op, &foreign, &lengths, usize::MAX, false);
    for layout in [
        GatedDeltaNetLayout { rows: 0, ..delta },
        GatedDeltaNetLayout {
            key_heads: 3,
            ..delta
        },
        GatedDeltaNetLayout {
            value_dim: usize::MAX,
            ..delta
        },
    ] {
        let (reads, writes) = b.split_at_mut(5);
        let (state, output) = writes.split_at_mut(1);
        op.reset_counters();
        rejected(
            &op,
            op.gated_delta_net_into(
                GatedDeltaNetBuffers {
                    qkv: &reads[0],
                    a: &reads[1],
                    b: &reads[2],
                    a_log: &reads[3],
                    dt_bias: &reads[4],
                    state: &mut state[0],
                    output: &mut output[0],
                },
                layout,
            ),
        );
    }
    unchanged(&op, &b[5]);
    unchanged(&op, &b[6]);
    let mut b = buffers(&op, &foreign, &[15, 20, 5, 20, 15], usize::MAX, false);
    for layout in [
        CausalConv1dLayout { rows: 0, ..conv },
        CausalConv1dLayout {
            kernel_size: 0,
            ..conv
        },
        CausalConv1dLayout {
            channels: usize::MAX,
            ..conv
        },
    ] {
        let (reads, writes) = b.split_at_mut(3);
        let (history, output) = writes.split_at_mut(1);
        op.reset_counters();
        rejected(
            &op,
            op.causal_depthwise_conv1d_silu_into(
                CausalConv1dBuffers {
                    input: &reads[0],
                    weight: &reads[1],
                    bias: Some(&reads[2]),
                    history: &mut history[0],
                    output: &mut output[0],
                },
                layout,
            ),
        );
    }
    unchanged(&op, &b[3]);
    unchanged(&op, &b[4]);
}

#[test]
#[ignore = "requires actual CUDA GPU; distinct context owners on the SAME ordinal"]
fn split_gate_and_offset_norm_validate_all_inputs_before_launch() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let split = QueryGateLayout {
        rows: 3,
        heads: 2,
        head_dim: 5,
    };
    let rows = F32RowsLayout { rows: 6, width: 5 };
    for wrong_owner in [false, true] {
        for bad in 0..3 {
            let mut b = buffers(&op, &foreign, &[60, 30, 30], bad, wrong_owner);
            let (input, writes) = b.split_at_mut(1);
            let (query, gate) = writes.split_at_mut(1);
            op.reset_counters();
            rejected(
                &op,
                op.split_query_gate_f32_into(&input[0], &mut query[0], &mut gate[0], split),
            );
            for i in [1, 2] {
                unchanged(
                    if bad == i && wrong_owner {
                        &foreign
                    } else {
                        &op
                    },
                    &b[i],
                );
            }
            let mut b = buffers(&op, &foreign, &[30, 30, 30], bad, wrong_owner);
            let (reads, writes) = b.split_at_mut(2);
            for activation in [GateActivation::Sigmoid, GateActivation::Silu] {
                op.reset_counters();
                rejected(
                    &op,
                    op.elementwise_gate_f32_into(
                        &reads[0],
                        &reads[1],
                        &mut writes[0],
                        rows,
                        activation,
                    ),
                );
            }
            unchanged(
                if bad == 2 && wrong_owner {
                    &foreign
                } else {
                    &op
                },
                &b[2],
            );
            let mut b = buffers(&op, &foreign, &[30, 5, 30], bad, wrong_owner);
            let (reads, writes) = b.split_at_mut(2);
            op.reset_counters();
            rejected(
                &op,
                op.offset_rms_norm_f32_into(&reads[0], &reads[1], &mut writes[0], rows, 1e-6),
            );
            unchanged(
                if bad == 2 && wrong_owner {
                    &foreign
                } else {
                    &op
                },
                &b[2],
            );
        }
    }
    let mut b = buffers(&op, &foreign, &[30, 5, 30], usize::MAX, false);
    let (reads, writes) = b.split_at_mut(2);
    for epsilon in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        op.reset_counters();
        rejected(
            &op,
            op.offset_rms_norm_f32_into(&reads[0], &reads[1], &mut writes[0], rows, epsilon),
        );
    }
    for layout in [
        F32RowsLayout { rows: 0, ..rows },
        F32RowsLayout {
            width: usize::MAX,
            ..rows
        },
    ] {
        op.reset_counters();
        rejected(
            &op,
            op.offset_rms_norm_f32_into(&reads[0], &reads[1], &mut writes[0], layout, 1e-6),
        );
        rejected(
            &op,
            op.elementwise_gate_f32_into(
                &reads[0],
                &reads[0],
                &mut writes[0],
                layout,
                GateActivation::Silu,
            ),
        );
    }
    unchanged(&op, &b[2]);
    let mut b = buffers(&op, &foreign, &[60, 30, 30], usize::MAX, false);
    let (input, writes) = b.split_at_mut(1);
    let (query, gate) = writes.split_at_mut(1);
    for layout in [
        QueryGateLayout { rows: 0, ..split },
        QueryGateLayout {
            head_dim: usize::MAX,
            ..split
        },
    ] {
        op.reset_counters();
        rejected(
            &op,
            op.split_query_gate_f32_into(&input[0], &mut query[0], &mut gate[0], layout),
        );
    }
    unchanged(&op, &b[1]);
    unchanged(&op, &b[2]);
}
