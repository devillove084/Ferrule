use super::*;
use crate::model_factory::qwen35::fixture::Fixture;

fn request(fixture: &Fixture, backend: ModelExecutionBackend) -> ResolvedModelRequest {
    std::fs::write(
        fixture.dir.join("config.json"),
        include_str!("../../../../../../ferrule-model/tests/qwen35_08b_config.json"),
    )
    .unwrap();
    let mut descriptor =
        crate::model_factory::tests::descriptor(ModelFamily::Qwen35, "qwen3_5", 24);
    descriptor.path = fixture.dir.clone();
    let config = AutoConfig::from_descriptor(descriptor);
    let mut input = crate::model_factory::tests::options();
    input.max_tensor_mebibytes = 1024;
    input.driver_config.enable_native_proposals = false;
    input.driver_config.ctx_size = 32;
    input.scheduler_config.max_active_sequences = 3;
    input.scheduler_config.max_batch_tokens = 8;
    input.scheduler_config.prefill_chunk_size = 8;
    let mut pipeline = PipelineBuildOptions::default();
    pipeline.parallelism.pipeline_parallel = 2;
    if backend == ModelExecutionBackend::Cuda {
        pipeline.devices = Some(vec![5, 5]);
    }
    let mut request = ResidentModelPlanner
        .prepare_pipeline(&config, backend.into(), None, input, pipeline)
        .unwrap()
        .request;
    request.max_layers = 4;
    request.model_info.num_layers = 4;
    request.model_info.hidden_size = 8;
    request.model_info.vocab_size = 11;
    request
}

fn verify(backend: ModelExecutionBackend) {
    let fixture = Fixture::new();
    let resources = fixture.resources();
    let request = request(&fixture, backend);
    let admission = Admission::new(&request, &resources).unwrap();
    let c = admission.config;
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Qwen35,
        WeightSource::Safetensors,
        c.page_size,
        c.max_positions,
        c.max_batch_tokens,
        c.session_capacity,
        c.max_parameter_bytes,
        c.precision,
    )
    .unwrap();
    let estimates = admission
        .plans
        .iter()
        .map(|p| HybridSegmentMemoryEstimate::for_segment(&resources, &options, p).unwrap())
        .collect::<Vec<_>>();
    let metadata =
        HybridSegmentMemoryEstimate::binding_metadata_bytes(&resources).unwrap() as u64 * 3;
    let rows = estimates
        .iter()
        .map(|e| e.boundary_rows_bytes as u64)
        .max()
        .unwrap()
        * 8;
    let custody = admission.kv.configured_pages as u64 * 3 * 4096
        + c.session_capacity as u64 * (8192 + request.max_layers as u64 * 256);
    let mut expected_host = metadata + rows + custody + MARGIN;
    for (e, stage) in estimates.iter().zip(&admission.stages) {
        let memory = if backend == ModelExecutionBackend::Cpu {
            e.host
        } else {
            e.device
        };
        assert_eq!(stage.weights, memory.weight_bytes().unwrap() as u64);
        assert_eq!(stage.workspace, memory.workspace_bytes as u64);
        assert_eq!(
            stage.kv,
            e.per_page_kv_bytes as u64 * admission.kv.configured_pages as u64
        );
        assert_eq!(
            stage.state,
            e.per_sequence_state_bytes as u64 * (c.session_capacity + 1) as u64
        );
        expected_host += e.host.weight_bytes().unwrap() as u64
            + e.host.workspace_bytes as u64
            + e.host_startup_temporary_bytes as u64
            + MARGIN;
        if backend == ModelExecutionBackend::Cpu {
            expected_host += stage.state + stage.kv;
        }
    }
    assert_eq!(admission.host_required, expected_host);
    admission.admit_host(expected_host).unwrap();
    assert!(admission.admit_host(expected_host - 1).is_err());
    let total = admission
        .stages
        .iter()
        .map(|s| s.total().unwrap())
        .sum::<u64>();
    let device = admission.stages[0].device;
    assert_eq!(
        HybridDeviceBudget::admit(&admission.stages, &[(device, total)]).unwrap()[0].required,
        total
    );
    assert!(HybridDeviceBudget::admit(&admission.stages, &[(device, total - 1)]).is_err());
}

#[test]
fn cpu_admission_composes_exact_model_host_state_and_physical_page_charges() {
    verify(ModelExecutionBackend::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_admission_retains_host_bindings_beside_colocated_device_budgets_without_gpu() {
    verify(ModelExecutionBackend::Cuda);
}
