//! Opt-in backend API/resource gate. No model or default profile change.
#[cfg(feature = "cuda")]
mod gpu {
    use ferrule_backend::cuda::operator_owner::CudaOperators;
    use ferrule_backend::cuda::operators::recurrent_bf16::Bf16GatedDeltaNetBuffers;
    use ferrule_backend::cuda::operators::recurrent_bf16_chunk::CudaBf16ChunkWorkspace;
    use ferrule_backend::cuda_plan::{Bf16ChunkPlan, GatedDeltaNetLayout};
    use std::io::Write;

    fn resident(op: &CudaOperators, launches: u64) {
        let c = op.counters();
        assert_eq!(c.kernel_launches, launches);
        assert_eq!(c.device_allocation_attempts, 0);
        assert_eq!(c.host_to_device_copies, 0);
        assert_eq!(c.device_to_host_copies, 0);
        assert_eq!(c.stream_wide_syncs, 0);
    }
    #[test]
    #[ignore = "sole GPU owner; pinned actual capital layer0 chunk backend only"]
    fn actual_chunk_exact_and_resident_workspace_contract() {
        actual_case(
            "FERRULE_GDN_CHUNK_FIXTURE",
            5,
            false,
            include_bytes!("recurrent_bf16/chunk.sha256"),
        );
    }

    #[test]
    #[ignore = "sole GPU owner; frozen Hello one-token prefill, independent HF chunk oracle"]
    fn explicit_prefill_one_row_matches_hf_and_preserves_cached_semantics() {
        actual_case(
            "FERRULE_GDN_PREFILL_ONE_FIXTURE",
            1,
            true,
            include_bytes!("recurrent_bf16/prefill_one.sha256"),
        );
    }

    fn actual_case(environment: &str, rows: usize, prefill: bool, pins_text: &[u8]) {
        let dir = std::path::PathBuf::from(
            std::env::var_os(environment).expect("prepared actual directory"),
        );
        let mut pins = std::process::Command::new("sha256sum")
            .args(["--check", "--strict", "-"])
            .current_dir(&dir)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        pins.stdin.take().unwrap().write_all(pins_text).unwrap();
        assert!(pins.wait().unwrap().success());
        let op = CudaOperators::new_on_device(0).unwrap();
        let bf16 = |name: &str| {
            let bytes = std::fs::read(dir.join(name)).unwrap();
            let (words, remainder) = bytes.as_chunks::<2>();
            assert!(remainder.is_empty(), "incomplete BF16 word in {name}");
            op.upload_bf16_words(
                &words
                    .iter()
                    .map(|b| u16::from_le_bytes(*b))
                    .collect::<Vec<_>>(),
            )
            .unwrap()
        };
        let f32s = |name: &str| {
            let bytes = std::fs::read(dir.join(name)).unwrap();
            let (words, remainder) = bytes.as_chunks::<4>();
            assert!(remainder.is_empty(), "incomplete F32 word in {name}");
            words
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        let qkv = bf16("qkv.bf16");
        let a = bf16("a.bf16");
        let b = bf16("b.bf16");
        let al = bf16("a_log.bf16");
        let dt = bf16("dt_bias.bf16");
        let mut state = op.upload_f32_buffer(&f32s("state.f32")).unwrap();
        let mut output = op.zero_bf16_buffer(rows * 4096).unwrap();
        let layout = GatedDeltaNetLayout {
            rows,
            key_heads: 16,
            value_heads: 32,
            key_dim: 128,
            value_dim: 128,
        };
        let plan = if prefill {
            Bf16ChunkPlan::new_prefill(layout, 17_850_368).unwrap()
        } else {
            Bf16ChunkPlan::new(layout, 17_850_368).unwrap()
        };
        let wrong_plan = Bf16ChunkPlan::new(
            GatedDeltaNetLayout {
                rows: rows + 1,
                ..layout
            },
            17_850_368,
        )
        .unwrap();
        let mut scratch = op.zero_f32_buffer(plan.workspace_elements()).unwrap();
        let mut short = op.zero_f32_buffer(plan.workspace_elements() - 1).unwrap();
        let other = CudaOperators::new_on_device(0).unwrap();
        let mut foreign_scratch = other.zero_f32_buffer(plan.workspace_elements()).unwrap();
        let foreign_qkv = other.zero_bf16_buffer(qkv.len()).unwrap();
        op.reset_counters();
        assert!(CudaBf16ChunkWorkspace::new(&mut short, plan).is_err());
        for foreign_operand in [false, true] {
            let (input, scratch) = if foreign_operand {
                (&foreign_qkv, &mut scratch)
            } else {
                (&qkv, &mut foreign_scratch)
            };
            assert!(
                op.gated_delta_net_bf16_chunked_into(
                    Bf16GatedDeltaNetBuffers {
                        qkv: input,
                        a: &a,
                        b: &b,
                        a_log: &al,
                        dt_bias: &dt,
                        state: &mut state,
                        output: &mut output,
                    },
                    Some(CudaBf16ChunkWorkspace::new(scratch, plan).unwrap()),
                    plan,
                )
                .is_err()
            );
        }
        for wrong in [false, true] {
            let work = if wrong {
                Some(CudaBf16ChunkWorkspace::new(&mut scratch, wrong_plan).unwrap())
            } else {
                None
            };
            assert!(
                op.gated_delta_net_bf16_chunked_into(
                    Bf16GatedDeltaNetBuffers {
                        qkv: &qkv,
                        a: &a,
                        b: &b,
                        a_log: &al,
                        dt_bias: &dt,
                        state: &mut state,
                        output: &mut output
                    },
                    work,
                    plan
                )
                .is_err()
            );
        }
        resident(&op, 0);
        op.gated_delta_net_bf16_chunked_into(
            Bf16GatedDeltaNetBuffers {
                qkv: &qkv,
                a: &a,
                b: &b,
                a_log: &al,
                dt_bias: &dt,
                state: &mut state,
                output: &mut output,
            },
            Some(CudaBf16ChunkWorkspace::new(&mut scratch, plan).unwrap()),
            plan,
        )
        .unwrap();
        resident(&op, plan.launch_count());
        let actual = op.download_bf16_buffer(&output).unwrap();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            f32s("expected-output.f32")
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        let actual = op.download_f32_buffer(&state).unwrap();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            f32s("expected-state.f32")
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert!(actual.iter().any(|x| x.to_bits() & 65535 != 0));
        if prefill {
            let cached = Bf16ChunkPlan::new(layout, 0).unwrap();
            assert_ne!(cached, plan);
            assert!(CudaBf16ChunkWorkspace::new(&mut scratch, cached).is_err());
            let mut cached_state = op.upload_f32_buffer(&f32s("state.f32")).unwrap();
            let mut cached_output = op.zero_bf16_buffer(rows * 4096).unwrap();
            op.reset_counters();
            // A valid chunk scratch wrapper cannot be submitted to the cached kind.
            assert!(
                op.gated_delta_net_bf16_chunked_into(
                    Bf16GatedDeltaNetBuffers {
                        qkv: &qkv,
                        a: &a,
                        b: &b,
                        a_log: &al,
                        dt_bias: &dt,
                        state: &mut cached_state,
                        output: &mut cached_output
                    },
                    Some(CudaBf16ChunkWorkspace::new(&mut scratch, plan).unwrap()),
                    cached,
                )
                .is_err()
            );
            resident(&op, 0);
            op.gated_delta_net_bf16_chunked_into(
                Bf16GatedDeltaNetBuffers {
                    qkv: &qkv,
                    a: &a,
                    b: &b,
                    a_log: &al,
                    dt_bias: &dt,
                    state: &mut cached_state,
                    output: &mut cached_output,
                },
                None,
                cached,
            )
            .unwrap();
            resident(&op, 1);
            let cached_output = op.download_bf16_buffer(&cached_output).unwrap();
            let expected = f32s("expected-output.f32");
            assert_eq!(
                cached_output
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count(),
                2
            );
            let captured = std::fs::read(dir.join("actual.bf16")).unwrap();
            let (captured, remainder) = captured.as_chunks::<2>();
            assert!(remainder.is_empty());
            assert_eq!(
                cached_output
                    .iter()
                    .map(|v| (v.to_bits() >> 16) as u16)
                    .collect::<Vec<_>>(),
                captured
                    .iter()
                    .map(|b| u16::from_le_bytes([b[0], b[1]]))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                op.download_f32_buffer(&cached_state)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                f32s("expected-state.f32")
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "actual M1: chunk exact4096/state524288; cached retains original2-word difference; 15 vs1 resident launches"
            );
        }

        eprintln!(
            "actual chunk output/state bit-exact; {} launches; no allocation/copy/sync during submission; missing/short/mismatched/foreign-owner scratch and foreign inputs rejected",
            plan.launch_count()
        );
    }

    #[test]
    #[ignore = "sole GPU owner; chunk tail launch/resource accounting, no model"]
    fn multichunk_launch_accounting_stays_resident() {
        let op = CudaOperators::new_on_device(0).unwrap();
        for rows in [2, 64, 65, 129] {
            let layout = GatedDeltaNetLayout {
                rows,
                key_heads: 1,
                value_heads: 2,
                key_dim: 17,
                value_dim: 65,
            };
            let plan = Bf16ChunkPlan::new(layout, usize::MAX).unwrap();
            let qkv = op.zero_bf16_buffer(layout.qkv_elements().unwrap()).unwrap();
            let a = op
                .zero_bf16_buffer(layout.gate_elements().unwrap())
                .unwrap();
            let b = op
                .zero_bf16_buffer(layout.gate_elements().unwrap())
                .unwrap();
            let al = op.zero_bf16_buffer(layout.value_heads).unwrap();
            let dt = op.zero_bf16_buffer(layout.value_heads).unwrap();
            let initial = vec![0.1234567; layout.state_elements().unwrap()];
            let mut state = op.upload_f32_buffer(&initial).unwrap();
            let mut output = op
                .zero_bf16_buffer(layout.output_elements().unwrap())
                .unwrap();
            // Dirty scratch must be completely overwritten before every read.
            let dirty = vec![f32::NAN; plan.workspace_elements()];
            let mut scratch = op.upload_f32_buffer(&dirty).unwrap();
            op.reset_counters();
            op.gated_delta_net_bf16_chunked_into(
                Bf16GatedDeltaNetBuffers {
                    qkv: &qkv,
                    a: &a,
                    b: &b,
                    a_log: &al,
                    dt_bias: &dt,
                    state: &mut state,
                    output: &mut output,
                },
                Some(CudaBf16ChunkWorkspace::new(&mut scratch, plan).unwrap()),
                plan,
            )
            .unwrap();
            resident(&op, plan.launch_count());
            assert!(
                op.download_bf16_buffer(&output)
                    .unwrap()
                    .iter()
                    .all(|x| x.is_finite())
            );
            assert!(
                op.download_f32_buffer(&state)
                    .unwrap()
                    .iter()
                    .all(|x| x.is_finite())
            );
        }
    }

    #[test]
    #[ignore = "sole GPU owner; one-row passthrough remains existing native path"]
    fn single_row_delegates_without_workspace() {
        let op = CudaOperators::new_on_device(0).unwrap();
        let l = GatedDeltaNetLayout {
            rows: 1,
            key_heads: 1,
            value_heads: 1,
            key_dim: 4,
            value_dim: 3,
        };
        let q = op.upload_bf16_words(&[0x3f80; 11]).unwrap();
        let gate = op.upload_bf16_words(&[0]).unwrap();
        let initial: Vec<f32> = (0..12).map(|i| i as f32 / 17.0).collect();
        let mut sa = op.upload_f32_buffer(&initial).unwrap();
        let mut sb = op.upload_f32_buffer(&initial).unwrap();
        let mut oa = op.zero_bf16_buffer(3).unwrap();
        let mut ob = op.zero_bf16_buffer(3).unwrap();
        op.reset_counters();
        op.gated_delta_net_bf16_into(
            Bf16GatedDeltaNetBuffers {
                qkv: &q,
                a: &gate,
                b: &gate,
                a_log: &gate,
                dt_bias: &gate,
                state: &mut sa,
                output: &mut oa,
            },
            l,
        )
        .unwrap();
        op.gated_delta_net_bf16_chunked_into(
            Bf16GatedDeltaNetBuffers {
                qkv: &q,
                a: &gate,
                b: &gate,
                a_log: &gate,
                dt_bias: &gate,
                state: &mut sb,
                output: &mut ob,
            },
            None,
            Bf16ChunkPlan::new(l, 0).unwrap(),
        )
        .unwrap();
        resident(&op, 2);
        assert_eq!(
            op.download_bf16_buffer(&oa).unwrap(),
            op.download_bf16_buffer(&ob).unwrap()
        );
        assert_eq!(
            op.download_f32_buffer(&sa).unwrap(),
            op.download_f32_buffer(&sb).unwrap()
        );
    }
}
