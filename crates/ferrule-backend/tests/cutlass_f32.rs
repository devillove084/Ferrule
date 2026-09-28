#![cfg(feature = "cuda")]

use ferrule_backend::cuda::compile_model_plan;
use ferrule_backend::cuda::operators::linear::{
    F32GemmError, F32GemmLayout, f32_gemm, f32_gemm_can_implement, f32_gemm_workspace_requirements,
};
use ferrule_backend::cuda::providers::cutlass::{self, CutlassKernelId};
use ferrule_backend::cuda::providers::{COMPILED_TARGET, CudaContext, CudaTarget, DeviceBuffer};
use ferrule_backend::plan::{
    ExecutionMode, KernelOperation, KernelPhase, KernelProviderId, LayerKernelRequirements,
    LinearBundleRequirement, OperationRequirement, WeightLayout,
};

#[test]
fn f32_sm86_exact_capability_and_plan_policy() {
    let provider = cutlass::discover_provider().unwrap();
    let execution = provider.execution_manifest().unwrap();
    assert!(provider.supports(CutlassKernelId::F32Gemm));
    assert!(execution.supports(KernelOperation::LinearF32, ExecutionMode::Inference));
    if COMPILED_TARGET == "sm_86" {
        // Exact native capability set includes the independent BF16 GEMM bit.
        assert_eq!(provider.manifest().kernel_mask, 0xd86);
        assert!(provider.supports(CutlassKernelId::Bf16Gemm));
        let caps = CudaTarget::parse(COMPILED_TARGET).unwrap().capabilities();
        assert!(caps.tf32_mma_sync);
        assert!(caps.bf16_mma_sync);
        assert!(!caps.fp8_mma_sync);
        assert!(!provider.supports(CutlassKernelId::Fp8QueryAKv));
        assert!(!provider.supports(CutlassKernelId::Fp8Projection));
        assert!(!execution.supports(KernelOperation::MlaQueryAKv, ExecutionMode::Inference));
        assert!(!execution.supports(KernelOperation::MlaQueryB, ExecutionMode::Inference));
        // Generic BF16 GEMM adds a native bit, not a semantic plan binding.
        let expected_operations = [
            KernelOperation::MainCompressorProjection,
            KernelOperation::IndexerCompressorProjection,
            KernelOperation::AttentionHcPre,
            KernelOperation::FeedForwardHcPre,
            KernelOperation::HybridMlaAttention,
            KernelOperation::ProposalHead,
            KernelOperation::LinearF32,
        ];
        assert_eq!(execution.operations.len(), expected_operations.len());
        for operation in expected_operations {
            assert!(execution.supports(operation, ExecutionMode::Inference));
        }
        eprintln!(
            "sm_86 native/Rust mask=0xD86: F32 TF32x3=true, BF16 GEMM=true, FP8 QueryAKv/Projection=false, inference operations=7"
        );
    }
    let mut bundle = LayerKernelRequirements::default();
    bundle.add_linear_bundle(LinearBundleRequirement::new(
        KernelOperation::LinearF32,
        ExecutionMode::Inference,
        11,
        [7],
        WeightLayout::RowMajor,
    ));
    let mut semantic = LayerKernelRequirements::default();
    semantic.require(OperationRequirement::new(
        KernelOperation::LinearF32,
        ExecutionMode::Inference,
    ));
    let plan = compile_model_plan(&[bundle, semantic]).unwrap();
    for layer in &plan.layers {
        let launch = layer
            .operation(KernelOperation::LinearF32, ExecutionMode::Inference)
            .unwrap();
        assert_eq!(launch.kernel.provider, KernelProviderId::CUDA_CUTLASS);
        assert_eq!(launch.kernel.phase(), KernelPhase::Linear);
        assert_eq!(launch.kernel.variant, 0);
        assert!(launch.is_provider_managed() && launch.is_capture_safe());
    }
    for mode in [
        ExecutionMode::TrainingForward,
        ExecutionMode::Backward,
        ExecutionMode::Optimizer,
    ] {
        assert!(!execution.supports(KernelOperation::LinearF32, mode));
        let mut layer = LayerKernelRequirements::default();
        layer.require(OperationRequirement::new(KernelOperation::LinearF32, mode));
        assert!(compile_model_plan(&[layer]).is_err());
    }
    // Deterministic opt-in must not be invented for the new operation.
    let mut layer = LayerKernelRequirements::default();
    layer.require(
        OperationRequirement::new(KernelOperation::LinearF32, ExecutionMode::Inference)
            .deterministic(),
    );
    assert!(compile_model_plan(&[layer]).is_err());
    for (k, outputs, layout) in [
        (11, vec![7], WeightLayout::Bf16RowMajor),
        (11, vec![7], WeightLayout::Fp8E4m3BlockScaled),
        (11, vec![7, 3], WeightLayout::RowMajor),
        (0, vec![7], WeightLayout::RowMajor),
        (11, vec![64 * 65535 + 1], WeightLayout::RowMajor),
    ] {
        let mut layer = LayerKernelRequirements::default();
        layer.add_linear_bundle(LinearBundleRequirement::new(
            KernelOperation::LinearF32,
            ExecutionMode::Inference,
            k,
            outputs,
            layout,
        ));
        assert!(compile_model_plan(&[layer]).is_err());
    }
}

#[test]
fn f32_unsupported_architecture_metadata_does_not_gain_tf32_or_fp8() {
    for target in ["sm_50", "sm_70", "sm_75"] {
        let caps = CudaTarget::parse(target).unwrap().capabilities();
        assert!(!caps.tf32_mma_sync, "{target}");
        assert!(!caps.fp8_mma_sync, "{target}");
    }
    for target in ["sm_80", "sm_86", "sm_87"] {
        let caps = CudaTarget::parse(target).unwrap().capabilities();
        assert!(caps.tf32_mma_sync, "{target}");
        assert!(!caps.fp8_mma_sync, "{target}");
    }
    assert!(
        CudaTarget::parse("sm_89")
            .unwrap()
            .capabilities()
            .fp8_mma_sync
    );
}

#[test]
fn f32_metadata_tile_grid_and_signed_span_boundaries() {
    for m in [63, 64, 65] {
        for n in [63, 64, 65] {
            for k in [15, 16, 17] {
                f32_gemm_workspace_requirements(F32GemmLayout::contiguous(m, n, k)).unwrap();
            }
        }
    }
    let grid_limit = 64 * 65535;
    for n in [grid_limit - 1, grid_limit] {
        f32_gemm_workspace_requirements(F32GemmLayout::contiguous(1, n, 1)).unwrap();
    }
    assert!(matches!(
        f32_gemm_workspace_requirements(F32GemmLayout::contiguous(1, grid_limit + 1, 1)),
        Err(F32GemmError::UnsupportedShape)
    ));
    for limit in [i32::MAX as usize - 64, i32::MAX as usize - 16] {
        for delta in [0, 1] {
            let shape = if limit == i32::MAX as usize - 64 {
                F32GemmLayout::contiguous(limit - delta, 1, 1)
            } else {
                F32GemmLayout::contiguous(1, 1, limit - delta)
            };
            shape.validate().unwrap();
        }
        let shape = if limit == i32::MAX as usize - 64 {
            F32GemmLayout::contiguous(limit + 1, 1, 1)
        } else {
            F32GemmLayout::contiguous(1, 1, limit + 1)
        };
        assert!(matches!(shape.validate(), Err(F32GemmError::InvalidShape)));
    }
    // Dimensions/strides remain individually valid; only the byte span crosses
    // isize::MAX. No allocation is needed to test this boundary.
    let stride = i32::MAX as usize;
    let last_valid_rows = ((isize::MAX as usize / 4 - 1) / stride) + 1;
    for rows in [last_valid_rows - 1, last_valid_rows] {
        F32GemmLayout {
            activation_stride: stride,
            ..F32GemmLayout::contiguous(rows, 1, 1)
        }
        .validate()
        .unwrap();
    }
    for operand in 0..3 {
        let mut shape = F32GemmLayout::contiguous(last_valid_rows + 1, 1, 1);
        match operand {
            0 => shape.activation_stride = stride,
            1 => {
                shape.rows = 1;
                shape.n = last_valid_rows + 1;
                shape.output_stride = shape.n;
                shape.weight_stride = stride;
            }
            _ => shape.output_stride = stride,
        }
        assert!(matches!(shape.validate(), Err(F32GemmError::InvalidShape)));
    }
}

fn samples(count: usize, mut seed: u32) -> Vec<f32> {
    (0..count)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (f64::from(seed) / f64::from(u32::MAX) * 2.0 - 1.0) as f32
        })
        .collect()
}

#[test]
fn f32_gemm_metadata_and_workspace_contract() {
    let layout = F32GemmLayout::contiguous(3, 7, 11);
    layout.validate().unwrap();
    for shape in [
        F32GemmLayout { rows: 0, ..layout },
        F32GemmLayout { n: 0, ..layout },
        F32GemmLayout { k: 0, ..layout },
        F32GemmLayout {
            rows: usize::MAX,
            ..layout
        },
        F32GemmLayout {
            k: i32::MAX as usize,
            ..layout
        },
    ] {
        assert!(matches!(shape.validate(), Err(F32GemmError::InvalidShape)));
    }
    for (shape, operand) in [
        (
            F32GemmLayout {
                activation_stride: 10,
                ..layout
            },
            "activation",
        ),
        (
            F32GemmLayout {
                weight_stride: 0,
                ..layout
            },
            "weight",
        ),
        (
            F32GemmLayout {
                output_stride: 6,
                ..layout
            },
            "output",
        ),
        (
            F32GemmLayout {
                activation_stride: usize::MAX,
                ..layout
            },
            "activation",
        ),
    ] {
        assert!(
            matches!(shape.validate(), Err(F32GemmError::InvalidStride { operand: actual }) if actual == operand)
        );
    }
    let provider = cutlass::discover_provider().unwrap();
    assert!(provider.supports(CutlassKernelId::F32Gemm));
    let workspace = f32_gemm_workspace_requirements(layout).unwrap();
    assert_eq!((workspace.bytes, workspace.alignment), (0, 1));
    assert!(matches!(
        f32_gemm_workspace_requirements(F32GemmLayout::contiguous(1, 64 * 65535 + 1, 1)),
        Err(F32GemmError::UnsupportedShape)
    ));
}

#[test]
#[ignore = "requires native CUDA TensorOp GPU; run with FERRULE_CUDA_ARCH=sm_86"]
fn f32_gemm_numerics_strides_and_large_k() {
    eprintln!("CUTLASS F32 GPU test compiled target: {COMPILED_TARGET}");
    assert!(
        cutlass::discover_provider()
            .unwrap()
            .supports(CutlassKernelId::F32Gemm)
    );
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    for (m, n, k, padding) in [
        (1, 1, 1, 0),
        (3, 7, 11, 0),
        (17, 19, 65, 3),
        (65, 131, 257, 0),
        (129, 67, 129, 5),
        (3, 13, 8193, 7),
        (2, 9, 65537, 3),
    ] {
        let layout = F32GemmLayout {
            activation_stride: k + padding,
            weight_stride: k + padding + 1,
            output_stride: n + padding,
            ..F32GemmLayout::contiguous(m, n, k)
        };
        let a = samples(m * layout.activation_stride, 19);
        let w = samples(n * layout.weight_stride, 91);
        // Offset by one F32 to prove that 16-byte alignment is not required.
        let mut a_guarded = vec![f32::NAN];
        a_guarded.extend_from_slice(&a);
        let a_root = DeviceBuffer::from_host(&stream, &a_guarded).unwrap();
        let da = a_root.slice(1, a.len()).unwrap();
        let mut w_guarded = vec![f32::NAN];
        w_guarded.extend_from_slice(&w);
        let w_root = DeviceBuffer::from_host(&stream, &w_guarded).unwrap();
        let dw = w_root.slice(1, w.len()).unwrap();
        let mut initial = vec![-777.0; m * layout.output_stride + 2];
        for row in 0..m {
            initial[1 + row * layout.output_stride..1 + row * layout.output_stride + n]
                .fill(f32::NAN);
        }
        let root = DeviceBuffer::from_host(&stream, &initial).unwrap();
        let mut output = root.slice(1, initial.len() - 2).unwrap();
        f32_gemm_can_implement(&stream, &da, &dw, &output, layout).unwrap();
        f32_gemm(&stream, &da, &dw, &mut output, layout).unwrap();
        let actual = root.to_host_vec(&stream).unwrap();
        for row in 0..m {
            for col in 0..n {
                let (mut expected, mut sum_abs) = (0.0f64, 0.0f64);
                for t in 0..k {
                    let product = f64::from(a[row * layout.activation_stride + t])
                        * f64::from(w[col * layout.weight_stride + t]);
                    expected += product;
                    sum_abs += product.abs();
                }
                let got = f64::from(actual[1 + row * layout.output_stride + col]);
                let tolerance = 1e-6 + 3e-7 * sum_abs + 3e-6 * expected.abs();
                assert!(
                    got.is_finite() && (got - expected).abs() <= tolerance,
                    "{m}x{n}x{k} [{row},{col}]: {got} vs {expected}, tolerance={tolerance}"
                );
            }
            for col in n..layout.output_stride {
                assert_eq!(actual[1 + row * layout.output_stride + col], -777.0);
            }
        }
        assert_eq!(actual[0], -777.0);
        assert_eq!(*actual.last().unwrap(), -777.0);
        assert_eq!(da.to_host_vec(&stream).unwrap(), a);
        assert_eq!(dw.to_host_vec(&stream).unwrap(), w);
    }
    // A single TF32 product or BF16 boundary loses this cancellation residual.
    let a = DeviceBuffer::from_host(&stream, &[1.000123f32, 1.0]).unwrap();
    let w = DeviceBuffer::from_host(&stream, &[1.000217f32, -1.0]).unwrap();
    let mut d = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();
    f32_gemm(&stream, &a, &w, &mut d, F32GemmLayout::contiguous(1, 1, 2)).unwrap();
    let expected = f64::from(1.000123f32) * f64::from(1.000217f32) - 1.0;
    assert!((f64::from(d.to_host_vec(&stream).unwrap()[0]) - expected).abs() < 2e-7);
}

#[test]
#[ignore = "requires native CUDA GPU; validates rejection before any output write"]
fn f32_gemm_rejects_bad_buffers_owners_and_aliases() {
    let ctx = CudaContext::new(0).unwrap();
    let other = CudaContext::new(0).unwrap();
    let stream = ctx.new_stream().unwrap();
    let foreign_stream = other.new_stream().unwrap();
    let layout = F32GemmLayout::contiguous(2, 3, 5);
    let a = DeviceBuffer::from_host(&stream, &[1.0f32; 10]).unwrap();
    let w = DeviceBuffer::from_host(&stream, &[2.0f32; 15]).unwrap();
    let mut d = DeviceBuffer::from_host(&stream, &[73.0f32; 6]).unwrap();
    for (bad, operand) in [
        (
            F32GemmLayout {
                activation_stride: 4,
                ..layout
            },
            "activation",
        ),
        (
            F32GemmLayout {
                weight_stride: 4,
                ..layout
            },
            "weight",
        ),
        (
            F32GemmLayout {
                output_stride: 2,
                ..layout
            },
            "output",
        ),
    ] {
        assert!(
            matches!(f32_gemm(&stream, &a, &w, &mut d, bad), Err(F32GemmError::InvalidStride { operand: actual }) if actual == operand)
        );
    }
    let short = DeviceBuffer::from_host(&stream, &[1.0f32]).unwrap();
    let mut short_d = DeviceBuffer::from_host(&stream, &[73.0f32]).unwrap();
    assert!(matches!(
        f32_gemm(&stream, &short, &w, &mut d, layout),
        Err(F32GemmError::BufferTooSmall {
            operand: "activation",
            ..
        })
    ));
    assert!(matches!(
        f32_gemm(&stream, &a, &short, &mut d, layout),
        Err(F32GemmError::BufferTooSmall {
            operand: "weight",
            ..
        })
    ));
    assert!(matches!(
        f32_gemm(&stream, &a, &w, &mut short_d, layout),
        Err(F32GemmError::BufferTooSmall {
            operand: "output",
            ..
        })
    ));
    let foreign = DeviceBuffer::from_host(&foreign_stream, &[1.0f32; 15]).unwrap();
    let mut foreign_d = DeviceBuffer::from_host(&foreign_stream, &[73.0f32; 6]).unwrap();
    assert!(matches!(
        f32_gemm(&stream, &foreign, &w, &mut d, layout),
        Err(F32GemmError::OwnerMismatch {
            operand: "activation"
        })
    ));
    assert!(matches!(
        f32_gemm(&stream, &a, &foreign, &mut d, layout),
        Err(F32GemmError::OwnerMismatch { operand: "weight" })
    ));
    assert!(matches!(
        f32_gemm(&stream, &a, &w, &mut foreign_d, layout),
        Err(F32GemmError::OwnerMismatch { operand: "output" })
    ));
    assert!(matches!(
        f32_gemm(&foreign_stream, &a, &w, &mut d, layout),
        Err(F32GemmError::OwnerMismatch { .. })
    ));
    for mut alias in [a.slice(1, 6).unwrap(), w.slice(0, 6).unwrap()] {
        assert!(matches!(
            f32_gemm(&stream, &a, &w, &mut alias, layout),
            Err(F32GemmError::Aliasing)
        ));
    }
    assert!(matches!(
        f32_gemm(&stream, &a, &w, &mut d, F32GemmLayout { rows: 0, ..layout }),
        Err(F32GemmError::InvalidShape)
    ));
    assert_eq!(d.to_host_vec(&stream).unwrap(), [73.0; 6]);
    assert_eq!(short_d.to_host_vec(&stream).unwrap(), [73.0]);
    assert_eq!(foreign_d.to_host_vec(&foreign_stream).unwrap(), [73.0; 6]);
    assert_eq!(a.to_host_vec(&stream).unwrap(), [1.0; 10]);
    assert_eq!(w.to_host_vec(&stream).unwrap(), [2.0; 15]);
}

#[test]
#[ignore = "requires native CUDA GPU; cross-stream producer/consumer/quiescence events"]
fn f32_gemm_uses_caller_stream_and_retirement_fences() {
    let ctx = CudaContext::new(0).unwrap();
    let upload = ctx.new_stream().unwrap();
    let compute = ctx.new_stream().unwrap();
    let consumer = ctx.new_stream().unwrap();
    let a = DeviceBuffer::from_host(&upload, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
    let w = DeviceBuffer::from_host(&upload, &[2.0f32, -1.0, 3.0, 2.0]).unwrap();
    let mut d = DeviceBuffer::<f32>::zeroed(&upload, 4).unwrap();
    compute.wait(&upload.record_event(None).unwrap()).unwrap();
    f32_gemm(&compute, &a, &w, &mut d, F32GemmLayout::contiguous(2, 2, 2)).unwrap();
    consumer.wait(&compute.record_event(None).unwrap()).unwrap();
    drop(a);
    drop(w);
    // Scratch can be reused only after the allocator's registered-stream fences.
    let _scratch = DeviceBuffer::<f32>::zeroed(&upload, 8).unwrap();
    assert_eq!(d.to_host_vec(&consumer).unwrap(), [0.0, 7.0, 2.0, 17.0]);
    upload.wait(&consumer.record_event(None).unwrap()).unwrap();
    d.copy_from_host(&upload, &[73.0; 4]).unwrap();
    assert_eq!(d.to_host_vec(&upload).unwrap(), [73.0; 4]);
}
