//! Real factory -> boxed engine -> driver -> scoped native executor regression.
use super::*;
use crate::model_factory::resident_f32_bound;
use ferrule_runtime::{HostKvDecodeConfig, KvExecutionPolicy};
fn run(
    backend: BackendSelection,
    policy: KvExecutionPolicy,
) -> (Vec<u32>, ferrule_runtime::KvExecutionStats) {
    let fixture = host_kv_fixture::Fixture::new();
    let descriptor =
        super::super::super::tests::descriptor(ModelFamily::Qwen35, "host-kv-two-gqa", 2);
    let mut options = super::super::super::tests::options();
    options.driver_config.ctx_size = 16;
    options.driver_config.enable_native_proposals = false;
    options.scheduler_config.max_active_sequences = 1;
    options.scheduler_config.max_decode_batch = 1;
    options.scheduler_config.prefill_chunk_size = 2;
    options.scheduler_config.max_batch_tokens = 2;
    let entry = resolve(&descriptor, backend).unwrap();
    let request = configure_request(&descriptor, entry, options).unwrap();
    let mut engine = resident_f32_bound::build_resident_f32_bound(
        request,
        fixture.resources(),
        fixture.tokenizer(),
    )
    .unwrap();
    engine.configure_kv_execution(policy).unwrap();
    engine
        .try_submit(GenerateRequest {
            id: RequestId(1),
            session_id: Some(SessionId(1)),
            prompt_tokens: vec![1, 3, 2],
            max_new_tokens: 3,
            stop: vec![],
            ignore_eos: true,
        })
        .unwrap();
    let mut tokens = Vec::new();
    let mut done = false;
    for _ in 0..32 {
        let step = engine
            .step(&mut |event| {
                tokens.push(event.token);
                Ok(())
            })
            .unwrap();
        assert!(!matches!(
            step,
            ResidentDriverStep::WaitingForModelProgress(_) | ResidentDriverStep::Blocked
        ));
        if !engine.drain_finished().is_empty() {
            done = true;
            break;
        }
    }
    assert!(done);
    let counts = engine.observability_snapshot().driver.kv_execution;
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
    (tokens, counts)
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires an explicitly assigned CUDA device; compile-only in no-GPU validation"]
fn host_kv_factory_prefill_warming_then_decode_matches_resident_and_cpu() {
    let (expected, _) = run(BackendSelection::Cpu, KvExecutionPolicy::Resident);
    let (resident, default_counts) = run(BackendSelection::Cuda, KvExecutionPolicy::Resident);
    let (host, counts) = run(
        BackendSelection::Cuda,
        KvExecutionPolicy::HostKvDecode(HostKvDecodeConfig::default()),
    );
    assert_eq!(resident, expected);
    assert_eq!(host, expected);
    assert_eq!(default_counts, ferrule_runtime::KvExecutionStats::default());
    assert_eq!(counts.resident_warming, 2);
    assert!(counts.host_kv_decode >= 2);
}

#[test]
fn host_kv_factory_cpu_reference_remains_resident() {
    let (tokens, counts) = run(BackendSelection::Cpu, KvExecutionPolicy::Resident);
    assert_eq!(tokens.len(), 3);
    assert_eq!(counts, ferrule_runtime::KvExecutionStats::default());
}
#[test]
fn host_kv_startup_rejects_gdn_numeric_and_invalid_scratch_without_gpu() {
    use ferrule_runtime::engine::validate_host_kv_spec;
    let dense = host_kv_fixture::Fixture::new();
    let resources = dense.resources();
    validate_host_kv_spec(resources.spec(), false).unwrap();
    assert!(validate_host_kv_spec(resources.spec(), true).is_err());
    let hybrid = crate::hybrid_fixture::Fixture::new();
    assert!(validate_host_kv_spec(hybrid.resources().spec(), false).is_err());
    let config = HostKvDecodeConfig::default();
    config.validate().unwrap();
    assert!(
        HostKvDecodeConfig {
            block_tokens: 0,
            ..config
        }
        .validate()
        .is_err()
    );
    assert!(
        HostKvDecodeConfig {
            capacity: Default::default(),
            ..config
        }
        .validate()
        .is_err()
    );
}
