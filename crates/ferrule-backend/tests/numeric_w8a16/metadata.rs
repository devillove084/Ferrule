use super::oracle::layout;
use ferrule_backend::cuda::operators::linear::*;
use ferrule_backend::cuda::providers::cutlass::{
    CutlassKernelId, CutlassProviderManifest, numeric_fp8_kernel_available,
};
const PROFILE: NumericFp8Precision = NumericFp8Precision::W8A16Bf16F32Accumulate;

#[test]
fn plan_selected_kernel_launches_bounded_physical_storage_and_legacy_unchanged() {
    for m in [1, 2, 3, 4, 7, 8, 9, 16, 17, 32, 33] {
        for (n, k) in [
            (1, 1),
            (7, 3),
            (33, 35),
            (131, 129),
            (17, 8193),
            (1, 16384),
            (1, 16385),
        ] {
            let l = layout(n, k, NumericFp8ScaleType::Bf16);
            for budget in [0, 1, 256, usize::MAX] {
                let p = NumericFp8LinearPlan::new(l, m, budget, PROFILE).unwrap();
                assert_eq!(p.precision(), PROFILE);
                assert_eq!(
                    p.selected_kernel(),
                    if m <= 8 {
                        NumericFp8SelectedKernel::W8A16FusedBf16Vector
                    } else {
                        NumericFp8SelectedKernel::W8A16FusedBf16Mma
                    }
                );
                assert_eq!(
                    p.selected_kernel().name(),
                    if m <= 8 {
                        "numeric-w8a16-fused-bf16-vector"
                    } else {
                        "numeric-w8a16-fused-bf16-mma"
                    }
                );
                assert_eq!(p.kernel_launches(), 1);
                assert_eq!(p.workspace_requirements().bytes, 0);
                assert_eq!(p.workspace_allocation_grant_bytes().unwrap(), 0);
                assert_eq!(p.activation_bytes(), 0);
                assert_eq!(p.weight_tile_bytes(), 0);
                assert_eq!(p.tile_count(), 0);
                assert_eq!(p.tile_rows(), 0);
                assert_eq!(p.padded_k(), k);
                assert_eq!(
                    p.fused_shared_bytes_per_cta(),
                    Some(if m <= 8 {
                        0
                    } else if k <= 2048 {
                        13312
                    } else if k <= 16384 {
                        19456
                    } else {
                        48128
                    })
                );
                assert_eq!(p.scratch_budget_bytes(), budget);
                let (w, s) = l.storage_lengths().unwrap();
                assert_eq!(
                    p.artifact_allocation_grant_bytes().unwrap(),
                    ferrule_backend::cuda_allocation_grant_bytes(w).unwrap()
                        + ferrule_backend::cuda_allocation_grant_bytes(s).unwrap()
                );
            }
            let f = NumericFp8LinearPlan::new(l, m, k * 4, NumericFp8Precision::F32Tf32x3).unwrap();
            assert_eq!(
                f.selected_kernel(),
                if m == 1 {
                    NumericFp8SelectedKernel::F32FusedM1
                } else {
                    NumericFp8SelectedKernel::F32TiledTf32x3
                }
            );
            assert_eq!(
                f.workspace_requirements().bytes as usize,
                if m == 1 { 0 } else { k * 4 }
            );
            assert_eq!(f.kernel_launches(), if m == 1 { 1 } else { 2 * n });
            let b = NumericFp8LinearPlan::new(
                l,
                m,
                (m + 1) * k.div_ceil(8) * 16,
                NumericFp8Precision::Bf16RneF32Accumulate,
            )
            .unwrap();
            assert_eq!(b.selected_kernel(), NumericFp8SelectedKernel::Bf16TiledGemm);
            assert_eq!(b.kernel_launches(), 1 + 2 * b.tile_count());
        }
    }
    let l = layout(3, 3, NumericFp8ScaleType::F32);
    // Two small independent allocations each consume a 256-byte grant.
    let small = NumericFp8LinearPlan::new(l, 2, 0, PROFILE).unwrap();
    assert_eq!(small.artifact_allocation_grant_bytes().unwrap(), 512);

    for m in [0, i32::MAX as usize, usize::MAX] {
        assert!(NumericFp8LinearPlan::new(l, m, 0, PROFILE).is_err());
    }
    for invalid in [
        NumericFp8Layout { n: 0, ..l },
        NumericFp8Layout { k: usize::MAX, ..l },
        NumericFp8Layout {
            row_origin: usize::MAX,
            ..l
        },
    ] {
        assert!(NumericFp8LinearPlan::new(invalid, 2, 0, PROFILE).is_err());
    }
    let huge = NumericFp8Layout {
        n: 1,
        k: i32::MAX as usize - 32,
        row_origin: 0,
        column_origin: 0,
        ..l
    };
    assert!(NumericFp8LinearPlan::new(huge, i32::MAX as usize - 64, 0, PROFILE).is_err());
}

#[test]
fn manifest_uses_independent_capability_not_bf16_fallback() {
    let legacy = CutlassProviderManifest {
        kernel_mask: CutlassKernelId::Bf16Gemm.mask()
            | CutlassKernelId::F32Gemm.mask()
            | CutlassKernelId::NumericFp8FusedM1.mask(),
    };
    assert!(!legacy.supports(CutlassKernelId::NumericFp8W8A16Fused));
    let fused = CutlassProviderManifest {
        kernel_mask: CutlassKernelId::NumericFp8W8A16Fused.mask(),
    };
    assert!(!fused.supports(CutlassKernelId::Bf16Gemm));
    assert!(fused.supports(CutlassKernelId::NumericFp8W8A16Fused));
    // This is a compile capability query; it does not create a CUDA context.
    for kernel in [
        NumericFp8SelectedKernel::W8A16FusedBf16Mma,
        NumericFp8SelectedKernel::W8A16FusedBf16Vector,
    ] {
        assert_eq!(
            numeric_fp8_kernel_available(kernel).unwrap(),
            ferrule_backend::cuda::providers::compiled_capabilities().bf16_mma_sync
        );
    }
}

// Mirror only the private numeric POD to test native preflight without pointers
// to real memory or a CUDA context. It must never be sent to a launch function.
#[derive(Clone, Copy)]
#[repr(C)]
struct Args {
    m: u32,
    n: u32,
    k: u32,
    padded_k: u32,
    tile_rows: u32,
    scale_cols: u32,
    row_origin: u32,
    column_origin: u32,
    scale_bytes: u32,
    lda: u32,
    ldd: u32,
    reserved: u32,
    activation: u64,
    weight: u64,
    scales: u64,
    output: u64,
    workspace: u64,
    workspace_bytes: u64,
    stream: u64,
}
#[allow(unsafe_code)]
unsafe extern "C" {
    fn ferrule_cutlass_numeric_fp8_w8a16_can_implement(a: *const Args) -> i32;
    fn ferrule_cutlass_numeric_fp8_w8a16_bf16_can_implement(a: *const Args) -> i32;
}
fn args() -> Args {
    Args {
        m: 32,
        n: 131,
        k: 129,
        padded_k: 129,
        tile_rows: 0,
        scale_cols: 2,
        row_origin: 127,
        column_origin: 125,
        scale_bytes: 4,
        lda: 132,
        ldd: 136,
        reserved: 0,
        activation: 0x100000,
        weight: 0x200000,
        scales: 0x300000,
        output: 0x400000,
        workspace: 0,
        workspace_bytes: 0,
        stream: 0,
    }
}
#[test]
#[allow(unsafe_code)]
fn native_cpu_preflight_shape_alias_zero_alignment_and_address_overflow() {
    assert_eq!(std::mem::size_of::<Args>(), 104);
    let supported = if ferrule_backend::cuda::providers::compiled_capabilities().bf16_mma_sync {
        0
    } else {
        2
    };
    for check in [
        ferrule_cutlass_numeric_fp8_w8a16_can_implement,
        ferrule_cutlass_numeric_fp8_w8a16_bf16_can_implement,
    ] {
        assert_eq!(unsafe { check(std::ptr::null()) }, 1);
        for m in [1, 2, 3, 4, 7, 8, 9, 16, 17, 32, 33] {
            let a = Args { m, ..args() };
            assert_eq!(unsafe { check(&a) }, supported);
        }
        for change in [
            |a: &mut Args| a.m = 0,
            |a: &mut Args| a.n = 0,
            |a: &mut Args| a.k = 0,
            |a: &mut Args| a.m = u32::MAX,
            |a: &mut Args| a.padded_k += 1,
            |a: &mut Args| a.tile_rows = 1,
            |a: &mut Args| a.workspace_bytes = 1,
            |a: &mut Args| a.reserved = 1,
            |a: &mut Args| a.row_origin = 128,
            |a: &mut Args| a.column_origin = 128,
            |a: &mut Args| a.scale_cols += 1,
            |a: &mut Args| a.scale_bytes = 1,
            |a: &mut Args| a.lda = 128,
            |a: &mut Args| a.ldd = 130,
            |a: &mut Args| a.lda = u32::MAX,
            |a: &mut Args| a.activation = 0,
            |a: &mut Args| a.weight = 0,
            |a: &mut Args| a.scales = 0,
            |a: &mut Args| a.output = 0,
            |a: &mut Args| a.activation += 1,
            |a: &mut Args| a.output += 1,
            |a: &mut Args| a.scales += 1,
            |a: &mut Args| a.activation = u64::MAX - 3,
            |a: &mut Args| a.output = u64::MAX - 3,
            |a: &mut Args| a.weight = u64::MAX,
            |a: &mut Args| a.scales = u64::MAX - 3,
            |a: &mut Args| a.output = a.activation + 4,
            |a: &mut Args| a.output = a.weight + 4,
            |a: &mut Args| a.output = a.scales,
        ] {
            let mut a = args();
            change(&mut a);
            assert_eq!(unsafe { check(&a) }, 1);
        }
    }
    let a = Args {
        activation: args().activation + 2,
        ..args()
    };
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_w8a16_can_implement(&a) },
        1
    );
    assert_eq!(
        unsafe { ferrule_cutlass_numeric_fp8_w8a16_bf16_can_implement(&a) },
        supported
    );
}
