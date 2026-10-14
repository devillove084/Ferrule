#[path = "support/recurrent_bf16_reference.rs"]
mod reference;
use reference::*;

#[test]
fn independent_scalar_matches_native_hf_bf16_loading() {
    for c in cases() {
        let (conv, histories) = conv(&c, true);
        assert_eq!(conv, c.get("conv"));
        assert_eq!(histories, c.get("history_steps"));
        for t in 0..c.rows {
            for h in 0..c.vh {
                let q = t * c.width() + (h / (c.vh / c.kh)) * c.dk;
                let k = q + c.kh * c.dk;
                let idx = (t * c.vh + h) * c.dk;
                assert_eq!(
                    normalized(&c.get("conv")[q..q + c.dk], true),
                    &c.get("qnorm")[idx..idx + c.dk],
                    "qnorm rows={} t={t} h={h}",
                    c.rows
                );
                assert_eq!(
                    normalized(&c.get("conv")[k..k + c.dk], true),
                    &c.get("knorm")[idx..idx + c.dk],
                    "knorm rows={} t={t} h={h}",
                    c.rows
                );
                assert_eq!(
                    round(1.0 / (1.0 + (-c.get("b")[t * c.vh + h]).exp())),
                    c.get("beta")[t * c.vh + h]
                );
            }
        }
        let (out, states) = delta(&c, true, false);
        close(&out, c.get("output"), 2e-6, 0.008);
        close(&states, c.get("state_steps"), 2e-6, 2e-5);
        assert_eq!(norm(&c, true), c.get("norm"));
        assert!(
            c.get("state_final").iter().any(|v| *v != round(*v)),
            "SSM must not be BF16 storage"
        );
        close(c.get("chunk_output"), c.get("output"), 2e-5, 0.008);
        for name in [
            "raw", "weight", "history", "conv", "output", "norm", "a_log", "dt_bias",
        ] {
            assert!(c.get(name).iter().all(|v| *v == round(*v)), "{name}");
        }
    }
}
#[test]
fn fixture_detects_missing_boundaries_and_false_bf16_state() {
    let c = cases().pop().unwrap();
    assert_ne!(conv(&c, false).0, c.get("conv"));
    assert_ne!(delta(&c, false, false).0, c.get("output"));
    assert_ne!(delta(&c, true, true).0, c.get("output"));
    assert_ne!(norm(&c, false), c.get("norm"));
}

// Q normalization below is CPU-derived, not an observed CUDA intermediate.
#[test]
fn frozen_cached_decode_final_reduction_is_order_sensitive() {
    let q = CACHED_Q.map(f32::from_bits);
    let normalized_q = CACHED_QNORM.map(f32::from_bits);
    assert_eq!(normalized(&q, true), normalized_q);
    let state = CACHED_FINAL_STATE.map(f32::from_bits);
    assert!(state.iter().any(|v| round(*v) != *v));
    let scale = 1.0 / 128_f32.sqrt();
    assert_eq!(scale.to_bits(), 0x3db5_04f3);
    let terms: [f32; 128] = std::array::from_fn(|i| state[i] * (normalized_q[i] * scale));
    let serial = terms.iter().fold(0.0_f32, |acc, v| acc + v);
    // Candidate Torch Reduce.cuh order: four y lanes, four accumulators each.
    // Diagnostic only: the actual CUDA reduction dispatch was not observed.
    let lanes: [f32; 4] = std::array::from_fn(|lane| {
        let acc: [f32; 4] = std::array::from_fn(|a| {
            (lane + 4 * a..128)
                .step_by(16)
                .fold(0.0_f32, |sum, i| sum + terms[i])
        });
        ((acc[0] + acc[1]) + acc[2]) + acc[3]
    });
    let tree = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
    assert_eq!(serial.to_bits(), 0x3911_8001);
    assert_eq!(tree.to_bits(), 0x3911_7fff);
    assert_eq!(round(serial).to_bits(), 0x3912_0000); // captured production
    assert_eq!(round(tree).to_bits(), 0x3911_0000); // captured HF
    assert_eq!(round(f32::from_bits(0x3911_8000)).to_bits(), 0x3912_0000);
}

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use ferrule_backend::cuda::operator_owner::{CudaBf16Buffer, CudaOperators};
    use ferrule_backend::cuda::operators::recurrent::*;
    use ferrule_backend::cuda::operators::recurrent_bf16::*;

    fn upload(op: &CudaOperators, values: &[f32]) -> CudaBf16Buffer {
        op.upload_bf16_words(
            &values
                .iter()
                .map(|v| {
                    assert_eq!(*v, round(*v));
                    (v.to_bits() >> 16) as u16
                })
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
    fn resident(op: &CudaOperators, launches: u64) {
        let c = op.counters();
        assert_eq!(c.kernel_launches, launches);
        assert_eq!(c.device_allocation_attempts, 0);
        assert_eq!(c.host_to_device_copies, 0);
        assert_eq!(c.device_to_host_copies, 0);
        assert_eq!(c.stream_wide_syncs, 0);
    }
    #[test]
    #[ignore = "requires CUDA GPU; isolated frozen cached GDN reduction, no model/linear ops"]
    fn frozen_cached_decode_final_reduction_observation() {
        let op = CudaOperators::new_on_device(0).unwrap();
        let layout = GatedDeltaNetLayout {
            rows: 1,
            key_heads: 16,
            value_heads: 32,
            key_dim: 128,
            value_dim: 128,
        };
        // Retain the real layout and affected head/column. Neutral K/V/gates
        // disable updates: frozen final S is input to normalization + final dot.
        let mut qkv = vec![0.0; 8192];
        qkv[10 * 128..11 * 128].copy_from_slice(&CACHED_Q.map(f32::from_bits));
        let qkv = upload(&op, &qkv);
        let negative = upload(&op, &[-1000.0; 32]);
        let zero = upload(&op, &[0.0; 32]);
        let mut initial = vec![0.0; 32 * 128 * 128];
        for (d, bits) in CACHED_FINAL_STATE.iter().enumerate() {
            initial[(20 * 128 + d) * 128 + 43] = f32::from_bits(*bits);
        }
        let mut state = op.upload_f32_buffer(&initial).unwrap();
        let mut output = op.zero_bf16_buffer(4096).unwrap();
        op.reset_counters();
        op.gated_delta_net_bf16_into(
            Bf16GatedDeltaNetBuffers {
                qkv: &qkv,
                a: &negative,
                b: &negative,
                a_log: &zero,
                dt_bias: &zero,
                state: &mut state,
                output: &mut output,
            },
            layout,
        )
        .unwrap();
        resident(&op, 1);
        let actual = op.download_bf16_buffer(&output).unwrap();
        let after = op.download_f32_buffer(&state).unwrap();
        assert!(
            after
                .iter()
                .zip(&initial)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        eprintln!(
            "cached GDN final-dot: index=2603 actual=0x{:04x} captured_production=0x3912 captured_hf=0x3911",
            actual[2603].to_bits() >> 16
        );
        // The source-derived reduction order now matches the frozen HF word.
        assert_eq!(actual[2603].to_bits(), 0x3911_0000);
        assert!(
            actual
                .iter()
                .enumerate()
                .all(|(i, v)| i == 2603 || *v == 0.0)
        );
    }

    #[test]
    #[ignore = "requires CUDA GPU and pinned actual_replay.py HF export; no full model"]
    fn actual_cached_decode_matches_pinned_hf_state_and_output_exactly() {
        let directory = std::path::PathBuf::from(
            std::env::var_os("FERRULE_GDN_ACTUAL_FIXTURE").expect("actual_replay.py HF export"),
        );
        // Bind every actual operand and independent HF result, not merely a
        // user-supplied directory/manifest. No production output is an oracle.
        use std::io::Write;
        let mut pin_check = std::process::Command::new("sha256sum")
            .args(["--check", "--strict", "-"])
            .current_dir(&directory)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        pin_check
            .stdin
            .take()
            .unwrap()
            .write_all(include_bytes!("recurrent_bf16/actual.sha256"))
            .unwrap();
        assert!(
            pin_check.wait().unwrap().success(),
            "actual GDN fixture hash mismatch"
        );
        let read = |name: &str| std::fs::read(directory.join(name)).unwrap();
        let f32_values = |name: &str| {
            let bytes = read(name);
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .collect::<Vec<_>>()
        };
        let op = CudaOperators::new_on_device(0).unwrap();
        let bf16 = |name: &str| {
            let bytes = read(name);
            assert_eq!(bytes.len() % 2, 0);
            op.upload_bf16_words(
                &bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| u16::from_le_bytes(*v))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let raw = bf16("raw.bf16");
        let weight = bf16("weight.bf16");
        let mut history = bf16("history.bf16");
        let a = bf16("a.bf16");
        let b = bf16("b.bf16");
        let a_log = bf16("a_log.bf16");
        let dt_bias = bf16("dt_bias.bf16");
        let initial = f32_values("state.f32");
        let mut state = op.upload_f32_buffer(&initial).unwrap();
        let mut conv = op.zero_bf16_buffer(8192).unwrap();
        let mut output = op.zero_bf16_buffer(4096).unwrap();
        op.reset_counters();
        op.causal_depthwise_conv1d_silu_bf16_into(
            Bf16CausalConv1dBuffers {
                input: &raw,
                weight: &weight,
                bias: None,
                history: &mut history,
                output: &mut conv,
            },
            CausalConv1dLayout {
                rows: 1,
                channels: 8192,
                kernel_size: 4,
            },
        )
        .unwrap();
        op.gated_delta_net_bf16_into(
            Bf16GatedDeltaNetBuffers {
                qkv: &conv,
                a: &a,
                b: &b,
                a_log: &a_log,
                dt_bias: &dt_bias,
                state: &mut state,
                output: &mut output,
            },
            GatedDeltaNetLayout {
                rows: 1,
                key_heads: 16,
                value_heads: 32,
                key_dim: 128,
                value_dim: 128,
            },
        )
        .unwrap();
        resident(&op, 2);
        for (name, buffer) in [
            ("expected-conv.f32", &conv),
            ("expected-history.f32", &history),
            ("expected-output.f32", &output),
        ] {
            assert_eq!(
                op.download_bf16_buffer(buffer)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                f32_values(name)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{name}"
            );
        }
        let actual = op.download_f32_buffer(&state).unwrap();
        let expected = f32_values("expected-state.f32");
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(actual.iter().any(|v| *v != round(*v)));
        assert_ne!(actual, initial);
        eprintln!(
            "actual cached GDN: output=4096/4096 exact F32 SSM=524288/524288 exact; resident guards passed"
        );
    }

    fn layouts(c: &Case, rows: usize) -> (CausalConv1dLayout, GatedDeltaNetLayout) {
        (
            CausalConv1dLayout {
                rows,
                channels: c.width(),
                kernel_size: c.kernel,
            },
            GatedDeltaNetLayout {
                rows,
                key_heads: c.kh,
                value_heads: c.vh,
                key_dim: c.dk,
                value_dim: c.dv,
            },
        )
    }
    #[test]
    fn typed_budget_and_layout_validation() {
        let c = cases().pop().unwrap();
        let (conv, delta) = layouts(&c, c.rows);
        let budget = Bf16RecurrentStorage::for_layouts(conv, delta).unwrap();
        assert_eq!(budget.conv_history_bytes, c.width() * c.kernel * 2);
        assert_eq!(budget.recurrent_state_bytes, c.vh * c.dk * c.dv * 4);
        assert_eq!(
            budget.total_bytes().unwrap(),
            c.width() * c.kernel * 2 + c.vh * c.dk * c.dv * 4
        );
        for bad in [
            CausalConv1dLayout { rows: 0, ..conv },
            CausalConv1dLayout {
                kernel_size: usize::MAX,
                ..conv
            },
            CausalConv1dLayout {
                channels: 1,
                ..conv
            },
        ] {
            assert!(Bf16RecurrentStorage::for_layouts(bad, delta).is_err());
        }
        for bad in [
            GatedDeltaNetLayout {
                key_heads: 0,
                ..delta
            },
            GatedDeltaNetLayout {
                value_heads: 3,
                ..delta
            },
            GatedDeltaNetLayout {
                value_dim: usize::MAX,
                ..delta
            },
        ] {
            assert!(Bf16RecurrentStorage::for_layouts(conv, bad).is_err());
        }
    }
    #[test]
    #[ignore = "requires CUDA GPU; independent HF native BF16 oracle, not a profile/perf test"]
    fn native_hf_oracle_prefill_decode_and_segmented_continuation() {
        let op = CudaOperators::new_on_device(0).unwrap();
        for c in cases() {
            // Isolate each stage against captured HF input, not preceding production output.
            let weight = upload(&op, c.get("weight"));
            let bias = c.bias.then(|| upload(&op, c.get("bias")));
            let a_log = upload(&op, c.get("a_log"));
            let dt_bias = upload(&op, c.get("dt_bias"));
            let norm_weight = upload(&op, c.get("norm_weight"));
            for segment in [c.rows, 1, 3, 64] {
                let mut history = upload(&op, c.get("history"));
                let mut state = op.upload_f32_buffer(c.get("state")).unwrap();
                for start in (0..c.rows).step_by(segment) {
                    let end = (start + segment).min(c.rows);
                    let n = end - start;
                    let raw = upload(&op, &c.get("raw")[start * c.width()..end * c.width()]);
                    let qkv = upload(&op, &c.get("conv")[start * c.width()..end * c.width()]);
                    let a = upload(&op, &c.get("a")[start * c.vh..end * c.vh]);
                    let b = upload(&op, &c.get("b")[start * c.vh..end * c.vh]);
                    let norm_input = upload(
                        &op,
                        &c.get("output")[start * c.vh * c.dv..end * c.vh * c.dv],
                    );
                    let z = upload(&op, &c.get("z")[start * c.vh * c.dv..end * c.vh * c.dv]);
                    let mut conv_out = op.zero_bf16_buffer(n * c.width()).unwrap();
                    let mut delta_out = op.zero_bf16_buffer(n * c.vh * c.dv).unwrap();
                    let mut norm_out = op.zero_bf16_buffer(n * c.vh * c.dv).unwrap();
                    let (cl, dl) = layouts(&c, n);
                    op.reset_counters();
                    op.causal_depthwise_conv1d_silu_bf16_into(
                        Bf16CausalConv1dBuffers {
                            input: &raw,
                            weight: &weight,
                            bias: bias.as_ref(),
                            history: &mut history,
                            output: &mut conv_out,
                        },
                        cl,
                    )
                    .unwrap();
                    op.gated_delta_net_bf16_into(
                        Bf16GatedDeltaNetBuffers {
                            qkv: &qkv,
                            a: &a,
                            b: &b,
                            a_log: &a_log,
                            dt_bias: &dt_bias,
                            state: &mut state,
                            output: &mut delta_out,
                        },
                        dl,
                    )
                    .unwrap();
                    op.gated_rms_norm_bf16_into(
                        &norm_input,
                        &norm_weight,
                        &z,
                        &mut norm_out,
                        F32RowsLayout {
                            rows: n * c.vh,
                            width: c.dv,
                        },
                        1e-6,
                    )
                    .unwrap();
                    resident(&op, 3);
                    // BF16 outputs permit one rounding step for CUDA transcendental/reduction differences;
                    // F32 state has a separate tight tolerance and must retain sub-BF16 information.
                    close(
                        &op.download_bf16_buffer(&conv_out).unwrap(),
                        &c.get("conv")[start * c.width()..end * c.width()],
                        1e-7,
                        0.008,
                    );
                    close(
                        &op.download_bf16_buffer(&delta_out).unwrap(),
                        &c.get("output")[start * c.vh * c.dv..end * c.vh * c.dv],
                        2e-6,
                        0.008,
                    );
                    close(
                        &op.download_bf16_buffer(&norm_out).unwrap(),
                        &c.get("norm")[start * c.vh * c.dv..end * c.vh * c.dv],
                        1e-7,
                        0.008,
                    );
                    let hs = c.width() * c.kernel;
                    assert_eq!(
                        op.download_bf16_buffer(&history).unwrap(),
                        &c.get("history_steps")[(end - 1) * hs..end * hs]
                    );
                    let ss = c.vh * c.dk * c.dv;
                    close(
                        &op.download_f32_buffer(&state).unwrap(),
                        &c.get("state_steps")[(end - 1) * ss..end * ss],
                        3e-6,
                        3e-5,
                    );
                }
                assert!(
                    op.download_f32_buffer(&state)
                        .unwrap()
                        .iter()
                        .any(|v| *v != round(*v))
                );
            }
        }
    }
}
