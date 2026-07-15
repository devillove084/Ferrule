#![cfg(feature = "cutlass")]

use cuda_core::{CudaContext, DeviceBuffer};
use ferrule_cuda::cutlass;
use ferrule_cuda::{KernelPhase, KernelProviderId, RowBucket};

#[test]
fn cutlass_plan_resolves_all_row_buckets_during_prepare() {
    let plan = ferrule_cuda::compile_cuda_model_plan(2, 64).expect("compile CUTLASS plan");
    assert_eq!(plan.layers.len(), 2);
    for layer in &plan.layers {
        for rows in RowBucket::ALL {
            let phase = layer
                .plan(rows)
                .and_then(|plan| plan.phase(KernelPhase::CompressorProjection))
                .expect("CUTLASS compressor phase");
            assert_eq!(phase.kernel.provider, KernelProviderId::CutlassCubin);
            assert!(phase.is_provider_managed());
            assert!(phase.is_capture_safe());
        }
    }
}

#[test]
fn cutlass_f32_provider_launches_on_ferrule_stream() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    // A is [2, 3], B is Ferrule row-major [4, 3]. The provider computes
    // A * transpose(B) into row-major [2, 4].
    let a = DeviceBuffer::from_host(&stream, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("upload CUTLASS A");
    let b = DeviceBuffer::from_host(
        &stream,
        &[
            1.0f32, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0,
        ],
    )
    .expect("upload CUTLASS B");
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, 8).expect("allocate CUTLASS C");

    assert!(
        cutlass::gemm_f32_can_implement(&stream, &a, &b, &mut c, 2, 4, 3)
            .expect("CUTLASS can_implement")
    );
    cutlass::gemm_f32(&stream, &a, &b, &mut c, 2, 4, 3, 1.0, 0.0).expect("CUTLASS GEMM launch");

    let actual = c.to_host_vec(&stream).expect("download CUTLASS C");
    assert_eq!(actual, vec![1.0, 2.0, 3.0, 6.0, 4.0, 5.0, 6.0, 15.0]);

    let manifest = cutlass::provider_manifest().expect("CUTLASS provider manifest");
    assert_eq!(manifest.abi_version, cutlass::CUTLASS_ABI_VERSION);
    assert_eq!(manifest.cutlass_version, 461);
    assert_eq!(manifest.kernel_count, 2);
    assert!(manifest.supports(cutlass::CutlassKernelId::F32Simt));
    assert!(manifest.supports(cutlass::CutlassKernelId::Bf16MmaSync));
}

#[test]
fn cutlass_bf16_projection_uses_caller_owned_workspace() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let stream = context.default_stream();

    let a_host = [
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, //
        2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0,
    ];
    let b_values = [
        1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    ];
    let b_host = b_values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_ne_bytes())
        .collect::<Vec<_>>();

    let a = DeviceBuffer::from_host(&stream, &a_host).expect("upload CUTLASS BF16 A");
    let b = DeviceBuffer::from_host(&stream, &b_host).expect("upload CUTLASS BF16 B");
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, 8).expect("allocate CUTLASS BF16 C");
    let mut workspace = cutlass::CutlassBf16Workspace::new(&stream, 2, 8).expect("BF16 workspace");

    assert!(
        cutlass::gemm_bf16_f32_can_implement(&stream, &a, &b, &mut c, &mut workspace, 2, 4, 8,)
            .expect("CUTLASS BF16 can_implement")
    );
    cutlass::gemm_bf16_f32(&stream, &a, &b, &mut c, &mut workspace, 2, 4, 8, 1.0, 0.0)
        .expect("CUTLASS BF16 launch");

    let actual = c.to_host_vec(&stream).expect("download CUTLASS BF16 C");
    assert_eq!(actual, vec![1.0, 2.0, 36.0, 16.0, 2.0, 4.0, 72.0, 32.0]);
}
