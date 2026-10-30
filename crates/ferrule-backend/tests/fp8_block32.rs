//! Static contracts for the SM80+ E4M3/UE8M0 block-32 baseline.

const RUST_IMPL: &str = include_str!("../src/cuda/operators/fp8_block32.rs");
const CORE_ABI: &str = include_str!("../native/cuda/core/abi.h");
const CORE_BINDINGS: &str = include_str!("../native/cuda/core/bindings.cuh");
const CORE_KERNEL: &str = include_str!("../native/cuda/core/fp8_block32.cuh");

#[test]
fn baseline_is_explicitly_unfused_and_stream_ordered() {
    for source in [RUST_IMPL, CORE_ABI, CORE_BINDINGS, CORE_KERNEL] {
        assert!(!source.contains("cudaDeviceSynchronize"));
    }
    assert!(RUST_IMPL.contains("FP8_BLOCK32_BASELINE_PATH"));
    assert!(RUST_IMPL.contains("bf16_gemm("));
    assert!(RUST_IMPL.contains("record_compute_event()"));
    assert!(CORE_BINDINGS.contains("cudaGetLastError()"));
    assert!(CORE_BINDINGS.contains("cuCtxGetDevice"));
    assert!(CORE_BINDINGS.contains("cuDeviceGetAttribute"));
    assert!(CORE_BINDINGS.contains("FERRULE_CUDA_TARGET_SM < 80"));
    assert!(CORE_KERNEL.contains("fp8_value(code)"));
    assert!(CORE_KERNEL.contains("e8m0_scale(scale_code)"));
    assert!(CORE_KERNEL.contains("bf16_rne(dequantized)"));
}

#[test]
fn abi_keeps_decode_and_activation_padding_as_plain_pod_contracts() {
    assert!(CORE_ABI.contains("FerruleCoreFp8Block32DecodeArgs"));
    assert!(CORE_ABI.contains("FerruleCoreBf16Block32PackArgs"));
    assert!(CORE_ABI.contains("FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreFp8Block32DecodeArgs"));
    assert!(CORE_ABI.contains("FERRULE_CORE_ASSERT_LAYOUT(FerruleCoreBf16Block32PackArgs"));
    assert!(CORE_BINDINGS.contains("ferrule_core_fp8_block32_device_capability"));
    assert!(CORE_BINDINGS.contains("ferrule_core_fp8_block32_decode_launch"));
    assert!(CORE_BINDINGS.contains("ferrule_core_bf16_block32_pack_launch"));
    assert!(CORE_KERNEL.contains("args->reserved"));
}

#[test]
fn host_contract_rejects_nan_and_ue8m0_ff_before_cuda_upload() {
    use ferrule_backend::operator_contract::{Fp8Block32ScaleLayout, Fp8E4m3Ue8m0LinearPlan};

    let layout = Fp8Block32ScaleLayout::contiguous(3, 5, 0, 0).unwrap();
    let plan = Fp8E4m3Ue8m0LinearPlan::new(2, 3, 5, layout).unwrap();
    assert!(plan.validate_payload(&[0x7f; 15], &[127]).is_err());
    assert!(plan.validate_payload(&[0x38; 15], &[255]).is_err());
}

#[test]
fn static_baseline_launch_count_reflects_optional_k_padding() {
    assert!(RUST_IMPL.contains("let launches = if needs_activation_padding { 3 } else { 2 }"));
    assert!(RUST_IMPL.contains("activation_padding_bytes"));
    assert!(RUST_IMPL.contains("decoded_weight_bytes"));
    assert!(RUST_IMPL.contains("workspace_bytes_for_plan(plan, gemm_k)?"));
    assert!(!RUST_IMPL.contains("saturating_add(self.activation_padding_bytes())"));
}

#[test]
fn preflight_order_covers_graph_owner_capability_and_bf16_rne() {
    let linear_start = RUST_IMPL.find("pub fn fp8_block32_linear_into(").unwrap();
    let linear = &RUST_IMPL[linear_start..];
    let graph = linear.find("self.fp8_block32_graph_preflight()?").unwrap();
    let owners = linear
        .find("self.check_fp8_block32_owners(artifact")
        .unwrap();
    let capability = linear
        .find("self.fp8_block32_runtime_capability()?")
        .unwrap();
    let span = linear.find("device_span(input.as_device_buffer()").unwrap();
    let submit = linear.find("self.submit_operator(launches").unwrap();
    assert!(graph < owners && owners < capability && capability < span && span < submit);
    assert!(CORE_KERNEL.contains("result = bf16_rne(dequantized)"));
    assert!(RUST_IMPL.contains("self.check_buffer_owner(&artifact.weight"));
    assert!(RUST_IMPL.contains("self.check_buffer_owner(&artifact.scales"));
    assert!(RUST_IMPL.contains("check_graph_ready()"));
}

#[cfg(feature = "cuda")]
#[test]
fn workspace_total_overflow_is_rejected_before_allocation() {
    use ferrule_backend::cuda::operators::linear::fp8_block32_workspace_requirements;
    use ferrule_backend::operator_contract::{Fp8Block32ScaleLayout, Fp8E4m3Ue8m0LinearPlan};

    let layout = Fp8Block32ScaleLayout::contiguous(1, 1, 0, 0).unwrap();
    let plan = Fp8E4m3Ue8m0LinearPlan::new(usize::MAX, 1, 1, layout).unwrap();
    assert!(fp8_block32_workspace_requirements(plan).is_err());
}

#[test]
fn cpu_bf16_rne_boundary_matches_the_native_decode_contract() {
    fn bf16_rne(value: f32) -> u16 {
        let bits = value.to_bits();
        ((bits.wrapping_add(0x7fff + ((bits >> 16) & 1))) >> 16) as u16
    }
    assert_eq!(bf16_rne(1.0), 0x3f80);
    assert_eq!(bf16_rne(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(bf16_rne(f32::from_bits(0x3f81_8000)), 0x3f82);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an SM80+ CUDA GPU; no GPU runtime test is run in this environment"]
fn ragged_block32_gpu_matches_cpu_oracle_for_unit_values() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_backend::operator_contract::{Fp8Block32ScaleLayout, Fp8E4m3Ue8m0LinearPlan};

    let layout = Fp8Block32ScaleLayout::contiguous(3, 5, 0, 0).unwrap();
    let plan = Fp8E4m3Ue8m0LinearPlan::new(2, 3, 5, layout).unwrap();
    let operators = CudaOperators::new_on_device(0).unwrap();
    let artifact = operators
        .upload_fp8_e4m3_ue8m0_block32(plan, &[0x38; 15], &[127])
        .unwrap();
    let input = operators.upload_bf16_words(&[0x3f80; 10]).unwrap();
    let mut output = operators.zero_f32_buffer(6).unwrap();
    let mut workspace = operators.fp8_block32_workspace(plan).unwrap();

    operators
        .fp8_block32_linear_into(&artifact, &input, &mut output, &mut workspace, plan)
        .unwrap();
    let actual = operators.download_f32_buffer(&output).unwrap();
    assert_eq!(actual, vec![5.0; 6]);
}
