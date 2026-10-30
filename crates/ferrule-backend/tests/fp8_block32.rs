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
fn decoded_cache_is_explicit_and_steady_path_has_no_decode_or_weight_workspace() {
    assert!(RUST_IMPL.contains("CudaDecodedFp8Block32Artifact"));
    assert!(RUST_IMPL.contains("pub fn prepare_fp8_block32_decoded_weight"));
    assert!(RUST_IMPL.contains("pub fn fp8_block32_linear_from_decoded"));
    assert!(RUST_IMPL.contains("pub fn fp8_block32_linear_from_decoded_with_completion"));
    assert!(RUST_IMPL.contains("source: CudaFp8Block32Artifact"));
    assert!(RUST_IMPL.contains("ready_event: CudaComputeEvent"));
    assert!(RUST_IMPL.contains("allocate_fp8_block32_decoded_weight_buffer"));
    assert!(RUST_IMPL.contains("fp8_block32_activation_workspace_requirements"));

    let cached_start = RUST_IMPL
        .find("pub fn fp8_block32_linear_from_decoded")
        .unwrap();
    let baseline_start = RUST_IMPL.find("pub fn fp8_block32_linear_into(").unwrap();
    let cached = &RUST_IMPL[cached_start..baseline_start];
    assert!(cached.contains("self.fp8_block32_graph_preflight()"));
    assert!(cached.contains("self.check_fp8_block32_decoded_owners"));
    assert!(cached.contains("bf16_gemm("));
    assert!(!cached.contains("launch_fp8_block32_decode"));
    assert!(!cached.contains("allocate_fp8_block32_decoded_weight_buffer"));
}

#[test]
fn decoded_cache_documents_explicit_lifetime_and_quarantine() {
    assert!(RUST_IMPL.contains("No implicit cache lookup or eviction"));
    assert!(RUST_IMPL.contains("caller's memory budget permits"));
    assert!(RUST_IMPL.contains("is_ready"));
    assert!(RUST_IMPL.contains("decoded artifact is quarantined"));
    assert!(RUST_IMPL.contains("check_graph_ready()"));
    assert!(RUST_IMPL.contains("ready_event.event().check_context"));
    assert!(RUST_IMPL.contains("owner_context: Arc::clone"));
    assert!(RUST_IMPL.contains("activation workspace owner mismatch"));
    assert!(RUST_IMPL.contains("padded BF16 activation launch grid exceeds CUDA limit"));
    assert!(RUST_IMPL.contains("Fp8Block32Completion"));
    assert!(RUST_IMPL.contains("owner_stream: self.stream_clone()"));
    assert!(RUST_IMPL.contains("completion owner/stream mismatch"));
    let removed_raw_finalize = ["finalize_fp8_block32", "decoded"].concat();
    assert!(!RUST_IMPL.contains(&removed_raw_finalize));
    assert!(!RUST_IMPL.contains("completion: &CudaComputeEvent"));
    assert!(RUST_IMPL.contains("completion was already finalized"));
    assert!(RUST_IMPL.contains("CaptureActive"));
    assert!(RUST_IMPL.contains("CompletionUnknown"));
    assert!(RUST_IMPL.contains("if context.graph_completion_unknown()"));
    assert!(RUST_IMPL.contains("if context.is_capturing()"));
    assert!(RUST_IMPL.contains("artifact.poison();\n                workspace.poison();"));
    assert!(RUST_IMPL.contains("partial native"));
    assert!(RUST_IMPL.contains("record_compute_event()"));
    assert!(RUST_IMPL.contains("artifact.poison();"));
}

#[test]
fn decoded_event_failures_classify_capture_unknown_and_workspace() {
    assert!(RUST_IMPL.contains("classify_fp8_block32_event_failure"));
    assert!(RUST_IMPL.contains("self.decoded_weight.context(),"));
    assert!(RUST_IMPL.contains("artifact.decoded_weight.context(),"));
    assert!(RUST_IMPL.contains("Some(workspace),"));
    assert!(RUST_IMPL.contains("Capture-active is the only retryable class."));
    assert!(RUST_IMPL.contains("artifact and activation workspace"));
    assert!(RUST_IMPL.contains("consume_after_terminal_failure"));
    assert!(
        !RUST_IMPL.contains("artifact.ready_event.is_complete() {\n            artifact.poison();")
    );
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
    use ferrule_backend::cuda::operators::linear::{
        fp8_block32_activation_workspace_requirements, fp8_block32_workspace_requirements,
    };
    use ferrule_backend::operator_contract::{Fp8Block32ScaleLayout, Fp8E4m3Ue8m0LinearPlan};

    let layout = Fp8Block32ScaleLayout::contiguous(1, 1, 0, 0).unwrap();
    let plan = Fp8E4m3Ue8m0LinearPlan::new(usize::MAX, 1, 1, layout).unwrap();
    assert!(fp8_block32_workspace_requirements(plan).is_err());

    let ragged_layout = Fp8Block32ScaleLayout::contiguous(1, 1, 0, 0).unwrap();
    let ragged_m = (256usize * 65_535) / 8 + 1;
    let ragged_plan = Fp8E4m3Ue8m0LinearPlan::new(ragged_m, 1, 1, ragged_layout).unwrap();
    assert!(fp8_block32_activation_workspace_requirements(ragged_plan).is_err());
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

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an SM80+ CUDA GPU; CUDA12.4 short cache correctness test"]
fn decoded_block32_cache_ragged_matches_cpu_and_quarantines_owner_mismatch() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_backend::operator_contract::{Fp8Block32ScaleLayout, Fp8E4m3Ue8m0LinearPlan};

    fn bf16(value: f32) -> u16 {
        ((value
            .to_bits()
            .wrapping_add(0x7fff + ((value.to_bits() >> 16) & 1)))
            >> 16) as u16
    }
    fn bf16_to_f32(value: u16) -> f32 {
        f32::from_bits(u32::from(value) << 16)
    }

    let layout = Fp8Block32ScaleLayout::contiguous(3, 5, 0, 0).unwrap();
    let plan = Fp8E4m3Ue8m0LinearPlan::new(2, 3, 5, layout).unwrap();
    let input_host = [
        bf16(1.0),
        bf16(0.5),
        bf16(-1.0),
        bf16(2.0),
        bf16(1.0),
        bf16(0.5),
        bf16(-1.0),
        bf16(2.0),
        bf16(1.0),
        bf16(0.5),
    ];
    let operators = CudaOperators::new_on_device(0).unwrap();
    let baseline_artifact = operators
        .upload_fp8_e4m3_ue8m0_block32(plan, &[0x38; 15], &[127])
        .unwrap();
    let source = operators
        .upload_fp8_e4m3_ue8m0_block32(plan, &[0x38; 15], &[127])
        .unwrap();
    let input = operators.upload_bf16_words(&input_host).unwrap();
    let mut baseline_output = operators.zero_f32_buffer(6).unwrap();
    let mut baseline_workspace = operators.fp8_block32_workspace(plan).unwrap();
    let mut output = operators.zero_f32_buffer(6).unwrap();
    let mut activation_workspace = operators.fp8_block32_activation_workspace(plan).unwrap();

    let cpu_decoded = plan
        .reference_dequantize(&[0x38; 15], &[127])
        .unwrap()
        .into_iter()
        .map(bf16)
        .map(bf16_to_f32)
        .collect::<Vec<_>>();
    let expected = (0..plan.m)
        .flat_map(|row| {
            let cpu_decoded = &cpu_decoded;
            (0..plan.n).map(move |column| {
                (0..plan.k)
                    .map(|inner| {
                        bf16_to_f32(input_host[row * plan.k + inner])
                            * cpu_decoded[column * plan.k + inner]
                    })
                    .sum::<f32>()
            })
        })
        .collect::<Vec<_>>();
    operators
        .fp8_block32_linear_into(
            &baseline_artifact,
            &input,
            &mut baseline_output,
            &mut baseline_workspace,
            plan,
        )
        .unwrap();
    assert_eq!(
        operators.download_f32_buffer(&baseline_output).unwrap(),
        expected
    );

    operators.reset_counters();
    let decoded = operators
        .prepare_fp8_block32_decoded_weight(source, plan)
        .unwrap();
    assert_eq!(operators.counters().compute_kernel_launches, 1);
    assert!(decoded.source_artifact().weight_bytes() > 0);
    decoded.wait_until_ready().unwrap();
    assert!(decoded.is_ready().unwrap());
    assert_eq!(decoded.decoded_weight_bytes(), 3 * 8 * 2);
    assert_eq!(activation_workspace.workspace_bytes(), 2 * 8 * 2);

    for _ in 0..2 {
        let before = operators.counters().compute_kernel_launches;
        operators
            .fp8_block32_linear_from_decoded(
                &decoded,
                &input,
                &mut output,
                &mut activation_workspace,
                plan,
            )
            .unwrap();
        let after = operators.counters().compute_kernel_launches;
        assert_eq!(after - before, 2, "ragged K is pack plus one GEMM");
        assert_eq!(operators.download_f32_buffer(&output).unwrap(), expected);
    }

    let wrong_layout = Fp8Block32ScaleLayout::contiguous(3, 5, 0, 0).unwrap();
    let wrong_plan = Fp8E4m3Ue8m0LinearPlan::new(1, 3, 5, wrong_layout).unwrap();
    assert!(
        operators
            .fp8_block32_linear_from_decoded(
                &decoded,
                &input,
                &mut output,
                &mut activation_workspace,
                wrong_plan,
            )
            .is_err()
    );

    let other = CudaOperators::new_on_device(0).unwrap();
    let other_input = other.upload_bf16_words(&input_host).unwrap();
    let mut other_output = other.zero_f32_buffer(6).unwrap();
    let mut other_workspace = other.fp8_block32_activation_workspace(plan).unwrap();
    assert!(
        other
            .fp8_block32_linear_from_decoded(
                &decoded,
                &other_input,
                &mut other_output,
                &mut other_workspace,
                plan,
            )
            .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an SM80+ CUDA GPU; CUDA12.4 one-GEMM cache counter test"]
fn decoded_block32_cache_k8_steady_calls_are_one_gemm() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_backend::operator_contract::{Fp8Block32ScaleLayout, Fp8E4m3Ue8m0LinearPlan};

    let layout = Fp8Block32ScaleLayout::contiguous(3, 8, 0, 0).unwrap();
    let plan = Fp8E4m3Ue8m0LinearPlan::new(1, 3, 8, layout).unwrap();
    let operators = CudaOperators::new_on_device(0).unwrap();
    let source = operators
        .upload_fp8_e4m3_ue8m0_block32(plan, &[0x38; 24], &[127])
        .unwrap();
    let input = operators.upload_bf16_words(&[0x3f80; 8]).unwrap();
    let mut output = operators.zero_f32_buffer(3).unwrap();
    let mut workspace = operators.fp8_block32_activation_workspace(plan).unwrap();
    assert_eq!(workspace.workspace_bytes(), 0);

    let other = CudaOperators::new_on_device(0).unwrap();
    let mut other_workspace = other.fp8_block32_activation_workspace(plan).unwrap();
    let owner_mismatch_decoded = operators
        .prepare_fp8_block32_decoded_weight(
            operators
                .upload_fp8_e4m3_ue8m0_block32(plan, &[0x38; 24], &[127])
                .unwrap(),
            plan,
        )
        .unwrap();
    assert!(
        operators
            .fp8_block32_linear_from_decoded(
                &owner_mismatch_decoded,
                &input,
                &mut output,
                &mut other_workspace,
                plan,
            )
            .is_err()
    );
    owner_mismatch_decoded.wait_until_ready().unwrap();

    operators.reset_counters();
    let decoded = operators
        .prepare_fp8_block32_decoded_weight(source, plan)
        .unwrap();
    assert_eq!(operators.counters().compute_kernel_launches, 1);
    decoded.wait_until_ready().unwrap();
    for _ in 0..2 {
        let before = operators.counters().compute_kernel_launches;
        let mut completion = operators
            .fp8_block32_linear_from_decoded_with_completion(
                &decoded,
                &input,
                &mut output,
                &mut workspace,
                plan,
            )
            .unwrap();
        let after = operators.counters().compute_kernel_launches;
        assert_eq!(after - before, 1);
        let generation = completion.generation();
        assert_ne!(generation, 0);
        assert!(completion.finalize(&other).is_err());
        completion.finalize(&operators).unwrap();
        assert!(completion.is_finalized());
        assert!(completion.finalize(&operators).is_err());
        assert_eq!(operators.download_f32_buffer(&output).unwrap(), [8.0; 3]);
    }
    decoded.quarantine();
    workspace.quarantine();
    assert!(
        operators
            .fp8_block32_linear_from_decoded(&decoded, &input, &mut output, &mut workspace, plan)
            .is_err()
    );
}
