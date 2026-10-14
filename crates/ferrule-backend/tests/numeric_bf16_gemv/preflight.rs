use super::*;
use crate::cuda::ffi::cutlass::{
    ferrule_numeric_bf16_gemv_can_implement, ferrule_numeric_bf16_gemv_create,
};
use crate::cuda_plan::NUMERIC_BF16_GEMV_SCRATCH_BYTES;

#[test]
#[allow(
    unsafe_code,
    reason = "invalid spans are rejected before any CUDA API call"
)]
fn gemv_create_rejects_wrapped_warmup_storage_without_device_calls() {
    let required = NUMERIC_BF16_GEMV_SCRATCH_BYTES as u64;
    for (address, bytes) in [
        (0, required),
        (0x1001, required),
        (0x1000, required - 1),
        (u64::MAX - 15, required),
        // Staging fits, but the additional F32 warmup output wraps.
        ((u64::MAX - (required - 2048 * 4)) & !15, required),
    ] {
        let (mut handle, mut growth, mut version) = (0, 0, 0);
        assert_eq!(
            unsafe {
                ferrule_numeric_bf16_gemv_create(
                    0,
                    address,
                    bytes,
                    16 << 20,
                    &mut handle,
                    &mut growth,
                    &mut version,
                )
            },
            1
        );
        assert_eq!((handle, growth, version), (0, 0, 0));
    }
}

#[test]
#[allow(
    unsafe_code,
    reason = "native layout preflight never dereferences fake addresses"
)]
fn gemv_preflight_checks_operand_spans_and_initialization_output() {
    fn args() -> NumericArgs {
        NumericArgs {
            m: 1,
            n: 2048,
            k: 4096,
            padded_k: 4096,
            tile_rows: 257,
            scale_cols: 32,
            row_origin: 0,
            column_origin: 0,
            scale_bytes: 2,
            lda: 4096,
            ldd: 2048,
            reserved: 0,
            activation: 0x1000,
            weight: 0x2000,
            scales: 0x3000,
            output: 0x4000,
            workspace: 0x10000,
            workspace_bytes: NUMERIC_BF16_GEMV_SCRATCH_BYTES as u64,
            stream: 0,
        }
    }
    assert!(matches!(
        unsafe { ferrule_numeric_bf16_gemv_can_implement(&args()) },
        0 | 2
    ));
    for mutate in [
        |a: &mut NumericArgs| a.activation = u64::MAX - 1,
        |a: &mut NumericArgs| a.weight = u64::MAX,
        |a: &mut NumericArgs| a.scales = u64::MAX - 1,
        |a: &mut NumericArgs| a.output = u64::MAX - 3,
        |a: &mut NumericArgs| a.workspace = (u64::MAX - (a.workspace_bytes - 8192)) & !15,
        |a: &mut NumericArgs| a.workspace_bytes -= 1,
        |a: &mut NumericArgs| a.tile_rows = 256,
        |a: &mut NumericArgs| a.scale_bytes = 4,
        |a: &mut NumericArgs| a.row_origin = 1,
    ] {
        let mut a = args();
        mutate(&mut a);
        assert_eq!(unsafe { ferrule_numeric_bf16_gemv_can_implement(&a) }, 1);
    }
}
