use super::*;

fn span_args(m: u32, scale_bytes: u32) -> NumericArgs {
    NumericArgs {
        m,
        n: 259,
        k: 256,
        padded_k: 256,
        tile_rows: 17,
        scale_cols: 3,
        row_origin: 127,
        column_origin: 125,
        scale_bytes,
        lda: 259,
        ldd: 264,
        reserved: 0,
        activation: 0x1000,
        weight: 0x2000,
        scales: 0x3000,
        output: 0x4000,
        workspace: 0x10000,
        workspace_bytes: u64::from(m + 17) * 256 * 2,
        stream: 0,
    }
}

#[test]
#[allow(
    unsafe_code,
    reason = "host-only preflight never dereferences fake addresses"
)]
fn tiled_and_sliced_native_preflight_reject_wrapped_spans() {
    type Preflight = unsafe extern "C" fn(*const NumericArgs) -> i32;
    for (m, input_bytes, preflight) in [
        (3, 4, ferrule_cutlass_numeric_fp8_can_implement as Preflight),
        (
            3,
            2,
            ferrule_cutlass_numeric_fp8_bf16_can_implement as Preflight,
        ),
        (
            1,
            4,
            ferrule_cutlass_numeric_fp8_sliced_k2_can_implement as Preflight,
        ),
        (
            1,
            2,
            ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement as Preflight,
        ),
        (
            5,
            4,
            ferrule_cutlass_numeric_fp8_sliced_k2_can_implement as Preflight,
        ),
        (
            5,
            2,
            ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement as Preflight,
        ),
    ] {
        for scale_bytes in [2, 4] {
            let mut a = span_args(m, scale_bytes);
            assert_eq!(unsafe { preflight(&a) }, 0);
            let spans = [
                (
                    (u64::from(m - 1) * u64::from(a.lda) + u64::from(a.k)) * input_bytes,
                    input_bytes,
                ),
                (u64::from(a.n) * u64::from(a.k), 1),
                (
                    u64::from((a.row_origin + a.n).div_ceil(128))
                        * u64::from(a.scale_cols)
                        * u64::from(scale_bytes),
                    u64::from(scale_bytes),
                ),
                (
                    (u64::from(m - 1) * u64::from(a.ldd) + u64::from(a.n)) * 4,
                    4,
                ),
                (a.workspace_bytes, 16),
            ];
            for (operand, (bytes, alignment)) in spans.into_iter().enumerate() {
                for overflow in [false, true] {
                    let mut a = span_args(m, scale_bytes);
                    let last_valid = (u64::MAX - bytes) & !(alignment - 1);
                    let address = last_valid + if overflow { alignment } else { 0 };
                    match operand {
                        0 => a.activation = address,
                        1 => a.weight = address,
                        2 => a.scales = address,
                        3 => a.output = address,
                        4 => a.workspace = address,
                        _ => unreachable!(),
                    }
                    assert_eq!(
                        unsafe { preflight(&a) },
                        i32::from(overflow),
                        "m={m} input_bytes={input_bytes} scale_bytes={scale_bytes} operand={operand} overflow={overflow}"
                    );
                }
            }
            // The ABI only accesses required scratch, not unused declared capacity.
            a.workspace_bytes = u64::MAX;
            assert_eq!(unsafe { preflight(&a) }, 0);
        }
    }
}

#[test]
#[allow(
    unsafe_code,
    reason = "host-only preflight never dereferences fake addresses"
)]
fn tiled_native_preflight_bounds_strided_byte_extents() {
    for preflight in [
        ferrule_cutlass_numeric_fp8_can_implement,
        ferrule_cutlass_numeric_fp8_bf16_can_implement,
    ] {
        // Both packed operands fit; only the stride makes the output span exceed
        // INT64_MAX. No allocation or launch is needed to check this limit.
        let mut a = span_args(i32::MAX as u32 - 64, 4);
        a.ldd = i32::MAX as u32;
        assert_eq!(unsafe { preflight(&a) }, 1);
    }
    let mut a = span_args(i32::MAX as u32 - 64, 4);
    a.lda = i32::MAX as u32;
    assert_eq!(unsafe { ferrule_cutlass_numeric_fp8_can_implement(&a) }, 1);
    // BF16's two-byte extent still fits with the same large element stride.
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_bf16_can_implement(&a) },
        0
    );
}

#[test]
#[allow(
    unsafe_code,
    reason = "host-only native preflight validates fake addresses without dereferencing or launching"
)]
fn typed_tiled_native_preflight_alignment_and_scratch() {
    let mut a = NumericArgs {
        m: 3,
        n: 19,
        k: 259,
        padded_k: 264,
        tile_rows: 17,
        scale_cols: 3,
        row_origin: 0,
        column_origin: 0,
        scale_bytes: 2,
        lda: 262,
        ldd: 24,
        reserved: 0,
        activation: 0x1002,
        weight: 0x2000,
        scales: 0x3000,
        output: 0x4000,
        workspace: 0x10000,
        workspace_bytes: 20 * 264 * 2,
        stream: 0,
    };
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_bf16_can_implement(&a) },
        0
    );
    assert_eq!(unsafe { ferrule_cutlass_numeric_fp8_can_implement(&a) }, 1);
    a.activation = 0x1000;
    assert_eq!(unsafe { ferrule_cutlass_numeric_fp8_can_implement(&a) }, 0);
    a.workspace_bytes -= 1;
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_bf16_can_implement(&a) },
        1
    );
    a.workspace_bytes += 1;
    a.activation += 1;
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_bf16_can_implement(&a) },
        1
    );
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_bf16_can_implement(std::ptr::null()) },
        1
    );
}

#[test]
#[ignore = "GPU owner only; typed tiled injected quarantine checks, not a device fault"]
fn typed_tiled_poison_fails_before_submission() {
    use crate::cuda::operators::linear::{CudaOperators, NumericFp8Layout, NumericFp8ScaleType};
    let op = CudaOperators::new_on_device(0).unwrap();
    for (m, k, budget, precision) in [
        (2, 3, 64, NumericFp8Precision::Bf16RneF32Accumulate),
        (5, 64, 1024, NumericFp8Precision::Bf16SlicedK2F32Accumulate),
    ] {
        let l = NumericFp8Layout {
            n: 3,
            k,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        };
        let artifact = op
            .upload_numeric_fp8_linear(l, &vec![0x38; 3 * k], &0x3f80_u16.to_le_bytes())
            .unwrap();
        let input = op.upload_bf16_words(&vec![0x3f80; m * k]).unwrap();
        let plan = NumericFp8LinearPlan::new(l, m, budget, precision).unwrap();
        let mut output = op.upload_f32_buffer(&vec![73.; m * 3]).unwrap();
        let mut ws = op.numeric_fp8_linear_workspace(plan).unwrap();
        ws.poison();
        op.reset_counters();
        assert!(
            op.numeric_fp8_bf16_linear_into(&artifact, &input, &mut output, &mut ws, plan, k, 3)
                .unwrap_err()
                .to_string()
                .contains("quarantined")
        );
        assert!(
            op.capture_decode_graph(|| op.numeric_fp8_bf16_linear_into(
                &artifact,
                &input,
                &mut output,
                &mut ws,
                plan,
                k,
                3
            ))
            .is_err()
        );
        assert_eq!(op.counters().compute_kernel_launches, 0);
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert!(ws.is_poisoned());
        assert_eq!(op.download_f32_buffer(&output).unwrap(), vec![73.; m * 3]);
    }
}

#[test]
fn typed_tiled_byte_ranges_reject_aliases() {
    assert!(no_overlap(&[(0x1000, 0x100c)], &[(0x1000, 0x1018), (0x2000, 0x2040)]).is_err());
    assert!(no_overlap(&[(0x1000, 0x100c)], &[(0x2000, 0x2018), (0x2000, 0x2040)]).is_err());
    assert!(no_overlap(&[(0x1000, 0x100c)], &[(0x2000, 0x2018), (0x3000, 0x3040)]).is_ok());
}

#[test]
#[allow(
    unsafe_code,
    reason = "host-only native preflight with fake aligned addresses"
)]
fn sliced_k2_preflight_bounds_match_explicit_profile() {
    let mut a = NumericArgs {
        m: 1,
        n: 4096,
        k: 2048,
        padded_k: 2048,
        tile_rows: 2047,
        scale_cols: 16,
        row_origin: 0,
        column_origin: 0,
        scale_bytes: 2,
        lda: 2048,
        ldd: 4096,
        reserved: 0,
        activation: 0x1002,
        weight: 0x2000,
        scales: 0x3000,
        output: 0x4000,
        workspace: 0x10000,
        workspace_bytes: 8 << 20,
        stream: 0,
    };
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement(&a) },
        0
    );
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_sliced_k2_can_implement(&a) },
        1
    );
    a.activation = 0x1000;
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_sliced_k2_can_implement(&a) },
        0
    );
    a.workspace_bytes -= 1;
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement(&a) },
        1
    );
    a.workspace_bytes += 1;
    a.m = 5;
    a.n = 8192;
    a.ldd = 8192;
    a.tile_rows = 257;
    a.workspace_bytes = (5 + 257) * 2048 * 2;
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement(&a) },
        0
    );
    for mutate in [
        |a: &mut NumericArgs| a.lda = a.k - 1,
        |a: &mut NumericArgs| a.ldd = a.n - 1,
        |a: &mut NumericArgs| a.workspace_bytes -= 1,
        |a: &mut NumericArgs| a.activation = 0,
        |a: &mut NumericArgs| a.output = 0,
    ] {
        let mut invalid = span_args(5, 2);
        mutate(&mut invalid);
        assert_eq!(
            unsafe { ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement(&invalid) },
            1
        );
    }
    a.m = 0;
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement(&a) },
        1
    );
    a.m = 1;
    a.k = 2040;
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement(&a) },
        1
    );
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_sliced_k2_bf16_can_implement(std::ptr::null()) },
        1
    );
}
