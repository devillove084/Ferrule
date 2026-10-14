use super::oracle::*;
use ferrule_backend::cuda::operators::linear::*;
use ferrule_backend::cuda::providers::DeviceBuffer;
const PROFILE: NumericFp8Precision = NumericFp8Precision::W8A16Bf16F32Accumulate;

#[test]
#[ignore = "GPU owner required; independent W8A16 oracle, M1/2/4/8/32 ragged operands and canaries"]
fn fused_multirow_f32_and_bf16_input_oracles_and_residency() {
    let op = CudaOperators::new_on_device(0).unwrap();
    for m in [1, 2, 4, 8, 32] {
        for (n, k) in [(1, 1), (7, 3), (33, 35), (131, 129), (17, 259), (7, 8193)] {
            for ty in [NumericFp8ScaleType::F32, NumericFp8ScaleType::Bf16] {
                let l = layout(n, k, ty);
                let w = weight(l);
                let (sb, sv) = scales(l);
                let artifact = op.upload_numeric_fp8_linear(l, &w, &sb).unwrap();
                assert_eq!(artifact.storage_bytes(), w.len() + sb.len());
                let lda = k + 3;
                let ldd = n + 5;
                let a: Vec<_> = (0..m * lda)
                    .map(|i| (i as i32 % 37 - 18) as f32 * 0.03173)
                    .collect();
                let input = op.upload_f32_buffer(&a).unwrap();
                let input_bf16 = op
                    .upload_bf16_words(&a.iter().copied().map(bf16_bits).collect::<Vec<_>>())
                    .unwrap();
                let mut out = op.upload_f32_buffer(&vec![7331.0; m * ldd + 7]).unwrap();
                let mut bfout = op.upload_f32_buffer(&vec![7331.0; m * ldd + 7]).unwrap();
                let plan = NumericFp8LinearPlan::new(l, m, 0, PROFILE).unwrap();
                let before_workspace = op.allocator_metrics();
                let mut ws = op.numeric_fp8_linear_workspace(plan).unwrap();
                assert_eq!(ws.allocated_bytes(), 0);
                assert_eq!(
                    op.allocator_metrics().live_granted_bytes,
                    before_workspace.live_granted_bytes
                );
                op.reset_counters();
                let before = op.allocator_metrics();
                op.numeric_fp8_linear_into(&artifact, &input, &mut out, &mut ws, plan, lda, ldd)
                    .unwrap();
                op.numeric_fp8_bf16_linear_into(
                    &artifact,
                    &input_bf16,
                    &mut bfout,
                    &mut ws,
                    plan,
                    lda,
                    ldd,
                )
                .unwrap();
                let counters = op.counters();
                assert_eq!(counters.compute_kernel_launches, 2);
                assert_eq!(counters.device_allocation_attempts, 0);
                assert_eq!(counters.stream_wide_syncs, 0);
                assert_eq!(
                    op.allocator_metrics().live_granted_bytes,
                    before.live_granted_bytes
                );
                assert_eq!(
                    op.allocator_metrics().driver_allocations,
                    before.driver_allocations
                );
                let got = op.download_f32_buffer(&out).unwrap();
                assert_eq!(got, op.download_f32_buffer(&bfout).unwrap());
                for row in 0..m {
                    for col in 0..n {
                        let (expected, norm, _) = dot(l, &w, &sv, &a, lda, row, col);
                        let actual = f64::from(got[row * ldd + col]);
                        assert!(
                            actual.is_finite() && (actual - expected).abs() <= 1e-5 + 2e-6 * norm,
                            "{m}x{n}x{k}, {ty:?}, ({row},{col}): {actual} vs {expected}"
                        );
                    }
                    assert!(
                        got[row * ldd + n..(row + 1) * ldd]
                            .iter()
                            .all(|v| *v == 7331.0)
                    );
                }
                assert!(got[m * ldd..].iter().all(|v| *v == 7331.0));
                assert_eq!(op.download_f32_buffer(&input).unwrap(), a);
            }
        }
    }
}

#[test]
#[ignore = "GPU owner required; capture/replay, exact event consumer, no staging/upload/growth"]
fn graph_replay_and_cross_stream_reuse_without_residency_growth() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let stream = op.stream_clone();
    let consumer = stream.context().new_stream().unwrap();
    for m in [1, 2, 4, 8, 32] {
        let l = layout(33, 129, NumericFp8ScaleType::F32);
        let w = weight(l);
        let (sb, sv) = scales(l);
        let artifact = op.upload_numeric_fp8_linear(l, &w, &sb).unwrap();
        let values: Vec<_> = (0..m * l.k)
            .map(|i| (i as i32 % 19 - 9) as f32 * 0.117)
            .collect();
        let mut input = op.upload_f32_buffer(&values).unwrap();
        let mut output = op.zero_f32_buffer(m * l.n).unwrap();
        let mut cross = DeviceBuffer::<f32>::zeroed(&consumer, m * l.n).unwrap();
        let p = NumericFp8LinearPlan::new(l, m, 0, PROFILE).unwrap();
        let mut ws = op.numeric_fp8_linear_workspace(p).unwrap();
        op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut ws, p, l.k, l.n)
            .unwrap();
        op.record_compute_event().unwrap().synchronize().unwrap();
        op.reset_counters();
        let before = op.allocator_metrics();
        let graph = op
            .capture_decode_graph(|| {
                op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut ws, p, l.k, l.n)
            })
            .unwrap();
        assert_eq!(op.counters().compute_kernel_launches, 1);
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(op.counters().stream_wide_syncs, 0);
        for factor in [1.0, -0.71, 1.0317] {
            let a: Vec<_> = values.iter().map(|v| v * factor).collect();
            op.overwrite_f32_buffer(&a, &mut input).unwrap();
            op.launch_graph(&graph).unwrap();
            consumer.wait(&stream.record_event(None).unwrap()).unwrap();
            numeric_fp8_linear(
                &consumer,
                &artifact,
                input.as_device_buffer(),
                &mut cross,
                &mut ws,
                p,
                l.k,
                l.n,
            )
            .unwrap();
            consumer.record_event(None).unwrap().synchronize().unwrap();
            let got = op.download_f32_buffer(&output).unwrap();
            assert_eq!(got, cross.to_host_vec(&consumer).unwrap());
            for row in 0..m {
                for col in 0..l.n {
                    let (expected, norm, _) = dot(l, &w, &sv, &a, l.k, row, col);
                    assert!(
                        (f64::from(got[row * l.n + col]) - expected).abs() <= 1e-5 + 2e-6 * norm
                    );
                }
            }
        }
        assert_eq!(
            before.live_granted_bytes,
            op.allocator_metrics().live_granted_bytes
        );
        assert_eq!(
            before.driver_allocations,
            op.allocator_metrics().driver_allocations
        );
        assert!(!ws.is_poisoned());
    }
}

#[test]
#[ignore = "GPU owner required; failure atomicity, alias, owner, retained scratch and zero budget"]
fn rejects_owner_alias_layout_profile_stride_and_retained_workspace_before_launch() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let other = CudaOperators::new_on_device(0).unwrap();
    let l = layout(3, 3, NumericFp8ScaleType::F32);
    let (sb, _) = scales(l);
    let artifact = op.upload_numeric_fp8_linear(l, &[0x38; 9], &sb).unwrap();
    let foreign_artifact = other.upload_numeric_fp8_linear(l, &[0x38; 9], &sb).unwrap();
    let input = op.upload_f32_buffer(&[1.0; 6]).unwrap();
    let bfinput = op.upload_bf16_words(&[0x3f80; 6]).unwrap();
    let foreign_input = other.upload_f32_buffer(&[1.0; 6]).unwrap();
    let foreign_bfinput = other.upload_bf16_words(&[0x3f80; 6]).unwrap();
    let mut output = op.upload_f32_buffer(&[73.0; 6]).unwrap();
    let mut foreign_output = other.upload_f32_buffer(&[73.0; 6]).unwrap();
    let p = NumericFp8LinearPlan::new(l, 2, 0, PROFILE).unwrap();
    let mut ws = op.numeric_fp8_linear_workspace(p).unwrap();
    let mut foreign_ws = other.numeric_fp8_linear_workspace(p).unwrap();
    let mut oversized = CudaNumericFp8Workspace::from_buffer_with_precision(
        DeviceBuffer::<u8>::zeroed(&op.stream_clone(), 1).unwrap(),
        PROFILE,
    );
    let old =
        NumericFp8LinearPlan::new(l, 2, 128, NumericFp8Precision::Bf16RneF32Accumulate).unwrap();
    let mut wrong_profile = op.numeric_fp8_linear_workspace(old).unwrap();
    op.reset_counters();
    assert!(
        op.numeric_fp8_linear_into(&foreign_artifact, &input, &mut output, &mut ws, p, 3, 3)
            .is_err()
    );
    assert!(
        op.numeric_fp8_linear_into(&artifact, &foreign_input, &mut output, &mut ws, p, 3, 3)
            .is_err()
    );
    assert!(
        op.numeric_fp8_bf16_linear_into(&artifact, &foreign_bfinput, &mut output, &mut ws, p, 3, 3)
            .is_err()
    );
    assert!(
        op.numeric_fp8_linear_into(&artifact, &input, &mut foreign_output, &mut ws, p, 3, 3)
            .is_err()
    );
    for scratch in [&mut foreign_ws, &mut oversized, &mut wrong_profile] {
        assert!(
            op.numeric_fp8_linear_into(&artifact, &input, &mut output, scratch, p, 3, 3)
                .is_err()
        );
        assert!(!scratch.is_poisoned());
    }
    assert!(
        op.numeric_fp8_bf16_linear_into(&artifact, &bfinput, &mut output, &mut ws, old, 3, 3)
            .is_err()
    );
    for (lda, ldd) in [(2, 3), (3, 2), (usize::MAX, 3), (3, usize::MAX)] {
        assert!(
            op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut ws, p, lda, ldd)
                .is_err()
        );
    }
    let wrong = NumericFp8LinearPlan::new(
        NumericFp8Layout {
            row_origin: 254,
            ..l
        },
        2,
        0,
        PROFILE,
    )
    .unwrap();
    assert!(
        op.numeric_fp8_linear_into(&artifact, &input, &mut output, &mut ws, wrong, 3, 3)
            .is_err()
    );
    let mut alias = input.as_device_buffer().slice(0, 6).unwrap();
    assert!(
        numeric_fp8_linear(
            &op.stream_clone(),
            &artifact,
            input.as_device_buffer(),
            &mut alias,
            &mut ws,
            p,
            3,
            3
        )
        .is_err()
    );
    assert_eq!(op.counters().compute_kernel_launches, 0);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    assert!(!ws.is_poisoned());
    assert_eq!(op.download_f32_buffer(&output).unwrap(), [73.0; 6]);
    assert_eq!(
        other.download_f32_buffer(&foreign_output).unwrap(),
        [73.0; 6]
    );
    // Bind a foreign context immediately before submission: provider must bind
    // the exact correct owner after validation, not just accept equal ordinal.
    other.stream_clone().context().bind_to_thread().unwrap();
    op.numeric_fp8_bf16_linear_into(&artifact, &bfinput, &mut output, &mut ws, p, 3, 3)
        .unwrap();
    op.record_compute_event().unwrap().synchronize().unwrap();
}
