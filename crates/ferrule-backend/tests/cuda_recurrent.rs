#![cfg(feature = "cuda")]

#[path = "support/recurrent_reference.rs"]
mod reference;

use ferrule_backend::cuda::context::{CudaF32Buffer, CudaOperators};
use ferrule_backend::cuda::operators::recurrent::*;
use ferrule_backend::cuda::standard::SplitHalfRopeLayout;

fn values(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 + phase) * 0.37).sin() * 0.7)
        .collect()
}
fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: actual={a}, expected={e}"
        );
    }
}
fn resident(op: &CudaOperators, launches: u64) {
    let c = op.counters();
    assert_eq!(c.kernel_launches, launches);
    assert_eq!(c.device_allocation_attempts, 0);
    assert_eq!(c.device_to_host_copies, 0);
    assert_eq!(c.host_to_device_copies, 0);
    assert_eq!(c.stream_wide_syncs, 0);
}
fn upload(op: &CudaOperators, v: &[f32]) -> CudaF32Buffer {
    op.upload_f32_buffer(v).unwrap()
}

#[test]
fn layouts_reject_zero_overflow_and_invalid_head_repeat() {
    for l in [
        CausalConv1dLayout {
            rows: 0,
            channels: 3,
            kernel_size: 4,
        },
        CausalConv1dLayout {
            rows: 1,
            channels: 0,
            kernel_size: 4,
        },
        CausalConv1dLayout {
            rows: 1,
            channels: 3,
            kernel_size: 0,
        },
        CausalConv1dLayout {
            rows: usize::MAX,
            channels: 3,
            kernel_size: 4,
        },
        CausalConv1dLayout {
            rows: 1,
            channels: 3,
            kernel_size: usize::MAX,
        },
    ] {
        assert!(l.validate().is_err());
    }
    let l = GatedDeltaNetLayout {
        rows: 3,
        key_heads: 2,
        value_heads: 6,
        key_dim: 7,
        value_dim: 5,
    };
    l.validate().unwrap();
    for bad in [
        GatedDeltaNetLayout { rows: 0, ..l },
        GatedDeltaNetLayout { key_heads: 0, ..l },
        GatedDeltaNetLayout {
            value_heads: 0,
            ..l
        },
        GatedDeltaNetLayout {
            value_heads: 3,
            ..l
        },
        GatedDeltaNetLayout {
            value_heads: 1,
            ..l
        },
        GatedDeltaNetLayout { key_dim: 0, ..l },
        GatedDeltaNetLayout { value_dim: 0, ..l },
        GatedDeltaNetLayout {
            rows: usize::MAX,
            ..l
        },
        GatedDeltaNetLayout {
            key_dim: usize::MAX,
            ..l
        },
        GatedDeltaNetLayout {
            value_dim: usize::MAX,
            ..l
        },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
    }
    assert!(
        QueryGateLayout {
            rows: 1,
            heads: 2,
            head_dim: usize::MAX
        }
        .validate()
        .is_err()
    );
    assert!(
        QueryGateLayout {
            rows: 1,
            heads: 0,
            head_dim: 4
        }
        .validate()
        .is_err()
    );
    assert!(F32RowsLayout { rows: 1, width: 0 }.validate().is_err());
    assert!(
        F32RowsLayout {
            rows: i32::MAX as usize,
            width: 2
        }
        .validate()
        .is_err()
    );
}

#[test]
#[ignore = "requires actual native CUDA GPU; run explicitly (no silent skip)"]
fn conv_prefill_decode_matches_cpu_and_every_raw_history() {
    let op = CudaOperators::new_on_device(0).unwrap();
    for (rows, channels, kernel) in [(1, 5, 4), (2, 5, 7), (17, 5, 1), (257, 5, 4), (5, 259, 4)] {
        for use_bias in [false, true] {
            let l = CausalConv1dLayout {
                rows,
                channels,
                kernel_size: kernel,
            };
            let x = values(l.row_elements().unwrap(), 0.3);
            let w = values(l.history_elements().unwrap(), 1.7);
            let b = values(channels, -0.4);
            let initial = values(l.history_elements().unwrap(), 9.1);
            let (expected, history) =
                reference::conv(&x, &w, use_bias.then_some(&b), &initial, channels, kernel);
            let input = upload(&op, &x);
            let weight = upload(&op, &w);
            let bias = upload(&op, &b);
            let mut state = upload(&op, &initial);
            let mut output = op.zero_f32_buffer(x.len()).unwrap();
            op.reset_counters();
            op.enable_capture_safe();
            op.causal_depthwise_conv1d_silu_into(
                CausalConv1dBuffers {
                    input: &input,
                    weight: &weight,
                    bias: use_bias.then_some(&bias),
                    history: &mut state,
                    output: &mut output,
                },
                l,
            )
            .unwrap();
            op.disable_capture_safe();
            resident(&op, 1);
            let actual = op.download_f32_buffer(&output).unwrap();
            close(&actual, &expected, 2e-6);
            close(&op.download_f32_buffer(&state).unwrap(), &history, 0.0);
            // Token calls retain the same GPU history; independently check it after EVERY row.
            let mut token_state = upload(&op, &initial);
            let mut token_output = op.zero_f32_buffer(channels).unwrap();
            for row in 0..rows {
                let token = upload(&op, &x[row * channels..(row + 1) * channels]);
                op.causal_depthwise_conv1d_silu_into(
                    CausalConv1dBuffers {
                        input: &token,
                        weight: &weight,
                        bias: use_bias.then_some(&bias),
                        history: &mut token_state,
                        output: &mut token_output,
                    },
                    CausalConv1dLayout { rows: 1, ..l },
                )
                .unwrap();
                close(
                    &op.download_f32_buffer(&token_output).unwrap(),
                    &actual[row * channels..(row + 1) * channels],
                    0.0,
                );
                let (_, prefix_history) = reference::conv(
                    &x[..(row + 1) * channels],
                    &w,
                    None,
                    &initial,
                    channels,
                    kernel,
                );
                close(
                    &op.download_f32_buffer(&token_state).unwrap(),
                    &prefix_history,
                    0.0,
                );
            }
            // A multirow decode continuation after prefill must not reset history.
            let next = values(3 * channels, 2.8);
            let (next_expected, next_history) = reference::conv(
                &next,
                &w,
                use_bias.then_some(&b),
                &history,
                channels,
                kernel,
            );
            let next = upload(&op, &next);
            let mut next_output = op.zero_f32_buffer(3 * channels).unwrap();
            op.causal_depthwise_conv1d_silu_into(
                CausalConv1dBuffers {
                    input: &next,
                    weight: &weight,
                    bias: use_bias.then_some(&bias),
                    history: &mut state,
                    output: &mut next_output,
                },
                CausalConv1dLayout { rows: 3, ..l },
            )
            .unwrap();
            close(
                &op.download_f32_buffer(&next_output).unwrap(),
                &next_expected,
                2e-6,
            );
            close(&op.download_f32_buffer(&state).unwrap(), &next_history, 0.0);
        }
    }
}

#[test]
#[ignore = "requires actual native CUDA GPU; run explicitly (no silent skip)"]
fn delta_prefill_and_decode_match_independent_matrix_reference() {
    let op = CudaOperators::new_on_device(0).unwrap();
    for (prefill, kh, vh, dk, dv) in [
        (1, 2, 6, 7, 5),
        (2, 3, 3, 5, 11),
        (17, 1, 2, 129, 3),
        (257, 2, 6, 7, 5),
        (7, 2, 6, 11, 47),
    ] {
        let l = GatedDeltaNetLayout {
            rows: prefill + 3,
            key_heads: kh,
            value_heads: vh,
            key_dim: dk,
            value_dim: dv,
        };
        let width = l.qkv_width().unwrap();
        let mut qkv = values(l.qkv_elements().unwrap(), 0.7);
        // Include exact zero and tiny norms to detect max(norm,eps) vs sum(x*x)+eps.
        qkv[..dk].fill(0.0);
        qkv[kh * dk..(kh + 1) * dk].fill(1e-5);
        qkv[width..width + dk].fill(1e-5);
        let mut a = values(l.gate_elements().unwrap(), 2.3);
        let mut b = values(l.gate_elements().unwrap(), -1.9);
        a[0] = 100.0;
        a[1] = -100.0;
        b[0] = -100.0;
        b[1] = 100.0;
        let alog = values(vh, 5.0);
        let bias = values(vh, 3.0);
        let initial = values(l.state_elements().unwrap(), -3.7);
        let (expected, expected_state) = reference::delta(reference::DeltaInput {
            qkv: &qkv,
            a: &a,
            b: &b,
            a_log: &alog,
            dt_bias: &bias,
            initial: &initial,
            key_heads: kh,
            value_heads: vh,
            dk,
            dv,
        });
        let alog_gpu = upload(&op, &alog);
        let bias_gpu = upload(&op, &bias);
        // Whole sequence, prefill + decode, and all single-token calls share one contract.
        for chunks in [vec![l.rows], vec![prefill, 1, 2], vec![1; l.rows]] {
            let mut state = upload(&op, &initial);
            let mut actual = Vec::new();
            let mut start = 0;
            for rows in chunks {
                let qkv_gpu = upload(&op, &qkv[start * width..(start + rows) * width]);
                let a_gpu = upload(&op, &a[start * vh..(start + rows) * vh]);
                let b_gpu = upload(&op, &b[start * vh..(start + rows) * vh]);
                let mut output = op.zero_f32_buffer(rows * vh * dv).unwrap();
                op.reset_counters();
                op.enable_capture_safe();
                op.gated_delta_net_into(
                    GatedDeltaNetBuffers {
                        qkv: &qkv_gpu,
                        a: &a_gpu,
                        b: &b_gpu,
                        a_log: &alog_gpu,
                        dt_bias: &bias_gpu,
                        state: &mut state,
                        output: &mut output,
                    },
                    GatedDeltaNetLayout { rows, ..l },
                )
                .unwrap();
                op.disable_capture_safe();
                resident(&op, 1);
                actual.extend(op.download_f32_buffer(&output).unwrap());
                start += rows;
            }
            close(&actual, &expected, 5e-6);
            close(
                &op.download_f32_buffer(&state).unwrap(),
                &expected_state,
                5e-6,
            );
        }
    }
}

#[test]
#[ignore = "requires actual native CUDA GPU; run explicitly (no silent skip)"]
fn split_gate_offset_norm_and_existing_partial_rope_match_cpu() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let l = QueryGateLayout {
        rows: 3,
        heads: 3,
        head_dim: 31,
    };
    let packed_values = values(l.input_elements().unwrap(), 0.8);
    let packed = upload(&op, &packed_values);
    let n = l.output_elements().unwrap();
    let mut query = op.zero_f32_buffer(n).unwrap();
    let mut gate = op.zero_f32_buffer(n).unwrap();
    op.reset_counters();
    op.split_query_gate_f32_into(&packed, &mut query, &mut gate, l)
        .unwrap();
    resident(&op, 1);
    let mut expected_query = Vec::new();
    let mut expected_gate = Vec::new();
    for head in packed_values.chunks_exact(2 * l.head_dim) {
        expected_query.extend_from_slice(&head[..l.head_dim]);
        expected_gate.extend_from_slice(&head[l.head_dim..]);
    }
    close(
        &op.download_f32_buffer(&query).unwrap(),
        &expected_query,
        0.0,
    );
    close(&op.download_f32_buffer(&gate).unwrap(), &expected_gate, 0.0);
    let layout = F32RowsLayout {
        rows: l.rows * l.heads,
        width: l.head_dim,
    };
    let mut output = op.zero_f32_buffer(n).unwrap();
    // Include saturated tails, negative gate, non-integer values, and exact zeros.
    let gate_values: Vec<_> = [-100.0, -7.3, -0.0, 0.37, 100.0]
        .into_iter()
        .cycle()
        .take(n)
        .collect();
    let gates = upload(&op, &gate_values);
    for activation in [GateActivation::Sigmoid, GateActivation::Silu] {
        op.reset_counters();
        op.elementwise_gate_f32_into(&query, &gates, &mut output, layout, activation)
            .unwrap();
        resident(&op, 1);
        let expected: Vec<_> = expected_query
            .iter()
            .zip(&gate_values)
            .map(|(&x, &g)| {
                let factor = if activation == GateActivation::Silu {
                    f64::from(g)
                } else {
                    1.0
                };
                (f64::from(x) * factor / (1.0 + (-f64::from(g)).exp())) as f32
            })
            .collect();
        close(&op.download_f32_buffer(&output).unwrap(), &expected, 2e-6);
    }
    let weights: Vec<_> = [0.0, -1.0, 0.37]
        .into_iter()
        .cycle()
        .take(layout.width)
        .collect();
    let weight = upload(&op, &weights);
    for offset in [true, false] {
        op.reset_counters();
        if offset {
            op.offset_rms_norm_f32_into(&query, &weight, &mut output, layout, 1e-6)
                .unwrap();
        } else {
            op.rms_norm_f32_into(&query, layout.rows, &weight, 1e-6, &mut output)
                .unwrap();
        }
        resident(&op, 1);
        let mut expected = Vec::new();
        for row in expected_query.chunks_exact(layout.width) {
            let inv = 1.0
                / (row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / layout.width as f64
                    + 1e-6)
                    .sqrt();
            expected.extend(row.iter().zip(&weights).map(|(&x, &w)| {
                (f64::from(x) * inv * (f64::from(w) + if offset { 1.0 } else { 0.0 })) as f32
            }));
        }
        close(&op.download_f32_buffer(&output).unwrap(), &expected, 2e-6);
    }
    let rope = SplitHalfRopeLayout {
        rows: l.rows,
        heads: l.heads,
        head_dim: l.head_dim,
        rope_dim: 6,
        table_positions: 5,
        restore_bf16_boundary: false,
    };
    let cos: Vec<_> = (0..15).map(|i| (i as f32 * 0.19).cos()).collect();
    let sin: Vec<_> = (0..15).map(|i| (i as f32 * 0.19).sin()).collect();
    op.split_half_rope_f32(
        &mut query,
        &upload(&op, &cos),
        &upload(&op, &sin),
        &[4, 0, 2],
        rope,
    )
    .unwrap();
    let mut expected = expected_query.clone();
    for (row, pos) in [4, 0, 2].into_iter().enumerate() {
        for head in 0..l.heads {
            let base = (row * l.heads + head) * l.head_dim;
            for pair in 0..3 {
                let (x, y) = (expected_query[base + pair], expected_query[base + pair + 3]);
                expected[base + pair] = x * cos[pos * 3 + pair] - y * sin[pos * 3 + pair];
                expected[base + pair + 3] = x * sin[pos * 3 + pair] + y * cos[pos * 3 + pair];
            }
        }
    }
    close(&op.download_f32_buffer(&query).unwrap(), &expected, 2e-6);
}

#[test]
#[ignore = "requires actual CUDA GPU; composed single-sequence caller loop"]
fn conv_delta_norm_silu_pipeline_keeps_two_sequences_independent() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let base = GatedDeltaNetLayout {
        rows: 1,
        key_heads: 2,
        value_heads: 4,
        key_dim: 5,
        value_dim: 7,
    };
    let channels = base.qkv_width().unwrap();
    let kernel = 4;
    let weight = values(channels * kernel, 2.5);
    let a_log = values(base.value_heads, 0.7);
    let dt_bias = values(base.value_heads, 1.1);
    let norm_weight = values(base.value_dim, 7.3);
    let weight_gpu = upload(&op, &weight);
    let a_log_gpu = upload(&op, &a_log);
    let dt_bias_gpu = upload(&op, &dt_bias);
    let norm_weight_gpu = upload(&op, &norm_weight);
    // Distinct state allocations, caller loops sequences on one owner/stream.
    let mut histories = [
        op.zero_f32_buffer(channels * kernel).unwrap(),
        upload(&op, &values(channels * kernel, 5.2)),
    ];
    let mut states = [
        op.zero_f32_buffer(base.state_elements().unwrap()).unwrap(),
        upload(&op, &values(base.state_elements().unwrap(), 6.3)),
    ];
    let mut cpu_histories = [vec![0.0; channels * kernel], values(channels * kernel, 5.2)];
    let mut cpu_states = [
        vec![0.0; base.state_elements().unwrap()],
        values(base.state_elements().unwrap(), 6.3),
    ];
    for lengths in [[2, 133], [1, 1], [3, 2]] {
        for (sequence, rows) in lengths.into_iter().enumerate() {
            let layout = GatedDeltaNetLayout { rows, ..base };
            let raw = values(
                layout.qkv_elements().unwrap(),
                rows as f32 + sequence as f32,
            );
            let a = values(layout.gate_elements().unwrap(), 1.2);
            let b = values(layout.gate_elements().unwrap(), 5.7);
            let z = values(layout.output_elements().unwrap(), 3.2);
            let (qkv, history) = reference::conv(
                &raw,
                &weight,
                None,
                &cpu_histories[sequence],
                channels,
                kernel,
            );
            let (delta, state) = reference::delta(reference::DeltaInput {
                qkv: &qkv,
                a: &a,
                b: &b,
                a_log: &a_log,
                dt_bias: &dt_bias,
                initial: &cpu_states[sequence],
                key_heads: base.key_heads,
                value_heads: base.value_heads,
                dk: base.key_dim,
                dv: base.value_dim,
            });
            cpu_histories[sequence] = history;
            cpu_states[sequence] = state;
            let mut expected = Vec::new();
            for (row, gates) in delta
                .chunks_exact(base.value_dim)
                .zip(z.chunks_exact(base.value_dim))
            {
                let inv = 1.0
                    / (row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
                        / base.value_dim as f64
                        + 1e-6)
                        .sqrt();
                expected.extend(
                    row.iter()
                        .zip(gates)
                        .zip(&norm_weight)
                        .map(|((&x, &g), &w)| {
                            (f64::from(x) * inv * f64::from(w) * f64::from(g)
                                / (1.0 + (-f64::from(g)).exp())) as f32
                        }),
                );
            }
            let raw_gpu = upload(&op, &raw);
            let a_gpu = upload(&op, &a);
            let b_gpu = upload(&op, &b);
            let z_gpu = upload(&op, &z);
            let mut qkv_gpu = op.zero_f32_buffer(raw.len()).unwrap();
            let mut delta_gpu = op.zero_f32_buffer(delta.len()).unwrap();
            let mut norm_gpu = op.zero_f32_buffer(delta.len()).unwrap();
            let mut output = op.zero_f32_buffer(delta.len()).unwrap();
            op.reset_counters();
            op.enable_capture_safe();
            op.causal_depthwise_conv1d_silu_into(
                CausalConv1dBuffers {
                    input: &raw_gpu,
                    weight: &weight_gpu,
                    bias: None,
                    history: &mut histories[sequence],
                    output: &mut qkv_gpu,
                },
                CausalConv1dLayout {
                    rows,
                    channels,
                    kernel_size: kernel,
                },
            )
            .unwrap();
            op.gated_delta_net_into(
                GatedDeltaNetBuffers {
                    qkv: &qkv_gpu,
                    a: &a_gpu,
                    b: &b_gpu,
                    a_log: &a_log_gpu,
                    dt_bias: &dt_bias_gpu,
                    state: &mut states[sequence],
                    output: &mut delta_gpu,
                },
                layout,
            )
            .unwrap();
            op.rms_norm_f32_into(
                &delta_gpu,
                rows * base.value_heads,
                &norm_weight_gpu,
                1e-6,
                &mut norm_gpu,
            )
            .unwrap();
            op.elementwise_gate_f32_into(
                &norm_gpu,
                &z_gpu,
                &mut output,
                F32RowsLayout {
                    rows: rows * base.value_heads,
                    width: base.value_dim,
                },
                GateActivation::Silu,
            )
            .unwrap();
            op.disable_capture_safe();
            resident(&op, 4);
            // Retirement must fence intermediates even if their wrappers are dropped
            // before the consumer's completion is observed on the host.
            drop((raw_gpu, a_gpu, b_gpu, z_gpu, qkv_gpu, delta_gpu, norm_gpu));
            close(&op.download_f32_buffer(&output).unwrap(), &expected, 1e-5);
            for seq in 0..2 {
                close(
                    &op.download_f32_buffer(&histories[seq]).unwrap(),
                    &cpu_histories[seq],
                    0.0,
                );
                close(
                    &op.download_f32_buffer(&states[seq]).unwrap(),
                    &cpu_states[seq],
                    5e-6,
                );
            }
        }
    }
}
