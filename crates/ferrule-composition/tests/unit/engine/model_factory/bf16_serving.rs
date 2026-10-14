use super::tests::{descriptor, options};
use super::*;
use ferrule_runtime::engine::KvExecutionPolicy;

#[test]
fn native_bf16_is_fail_closed_for_cpu_and_host_kv() {
    let config = AutoConfig::from_descriptor(descriptor(ModelFamily::Qwen3, "qwen3", 2));
    let serving = ServingModelOptions {
        model_precision: Some(ModelPrecision::Bf16),
        ..Default::default()
    };
    let cpu_error = ResidentModelPlanner::new()
        .prepare_serving(
            &config,
            BackendSelection::Cpu,
            None,
            options(),
            serving.clone(),
        )
        .err()
        .expect("must reject")
        .to_string();
    assert!(cpu_error.contains("CUDA"));

    let host = ServingModelOptions {
        model_precision: Some(ModelPrecision::Bf16),
        kv_execution_policy: KvExecutionPolicy::HostKvDecode(Default::default()),
        ..Default::default()
    };
    let host_error = ResidentModelPlanner::new()
        .prepare_serving(&config, BackendSelection::Cuda, None, options(), host)
        .err()
        .expect("must reject")
        .to_string();
    assert!(host_error.contains("typed BF16 KV"));
}

#[test]
fn bf16_rejects_hybrid_and_moe_before_opening_checkpoint() {
    for family in [
        ModelFamily::Qwen35,
        ModelFamily::Qwen35Moe,
        ModelFamily::QwenMoe,
    ] {
        let config = AutoConfig::from_descriptor(descriptor(family, "metadata-only", 2));
        let serving = ServingModelOptions {
            model_precision: Some(ModelPrecision::Bf16),
            ..Default::default()
        };
        let error = ResidentModelPlanner
            .prepare_serving(&config, BackendSelection::Cuda, None, options(), serving)
            .err()
            .expect("reject family");
        assert!(error.to_string().contains("no GDN/MoE/Qwen3.5"));
    }
}

#[allow(dead_code)]
#[path = "../../../dense_bf16/fixture.rs"]
mod fixture;

#[test]
fn frozen_builder_uses_standard_pipeline_bf16_config_not_hybrid_f32() {
    let f = fixture::Fixture::new();
    let plan = ResidentModelPlanner
        .prepare_serving(
            &f.config(),
            BackendSelection::Cuda,
            None,
            fixture::options(),
            fixture::serving(),
        )
        .unwrap();
    assert_eq!(
        plan.selected_implementation().kind,
        SelectedImplementationKind::Pipeline
    );
    assert_eq!(plan.topology_placement().logical.pipeline_parallel, 1);
    let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> = build_pipeline;
    assert!(std::ptr::fn_addr_eq(
        plan.request.execution.construction.builder(),
        expected
    ));
    plan.build_with(|request| -> Result<()> {
        let request = request.resolved;
        assert_eq!(
            request.execution.pipeline_precision(),
            ferrule_model::execution::ExecutionPrecisionPolicy::bf16_compatibility()
        );
        assert_eq!(
            request.execution.kv_execution_policy,
            KvExecutionPolicy::Resident
        );
        assert_eq!(request.backend, ModelExecutionBackend::Cuda);
        assert_eq!(request.pipeline.unwrap().parallelism, Default::default());
        Ok(())
    })
    .unwrap();
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "sole GPU owner only; actual full-depth 35B public numeric BF16 factory lifecycle, NOT strict full40 oracle acceptance"]
fn numeric_bf16_35b_public_factory_two_token_retained_decode_fork_cancel_release() {
    use ferrule_model::transformer::NumericFp8Bf16HybridCudaProfile as Profile;
    use ferrule_runtime::scheduling::RequestTerminal;
    use ferrule_runtime::{GenerateRequest, InferenceShutdownProgress, RequestId, SessionId};
    use std::time::{Duration, Instant};
    assert_eq!(
        std::env::var("FERRULE_NUMERIC_BF16_FACTORY_GPU").as_deref(),
        Ok("1"),
        "sole owner opt-in required"
    );
    let model = std::env::var_os("FERRULE_NUMERIC_BF16_MODEL")
        .expect("explicit real 35B checkpoint required");
    let config = AutoConfig::from_pretrained(std::path::Path::new(&model)).unwrap();
    let limits = Qwen35MoeCapacityLimits {
        max_experts: 8,
        max_bytes: 320 << 20,
        scratch_bytes: 0,
        workspace_bytes: 2 << 30,
        max_positions: 128,
        max_sequences: 2,
        max_batch_tokens: 8,
        ..Default::default()
    };
    let mut o = options();
    o.max_layers = None;
    o.max_tensor_mebibytes = 1024;
    o.output_head_chunk_rows = 4096;
    o.expert_reader_max_tensor_mebibytes = 64;
    o.expert_cache = Default::default();
    o.numeric_fp8_profile = None;
    o.qwen35_host_cache = None;
    o.qwen35_moe_capacity = Some(limits);
    o.moe_hotset_experts = 0;
    o.kv_cache_mebibytes = Some(1024);
    o.driver_config.ctx_size = 128;
    o.driver_config.enable_native_proposals = false;
    o.driver_config.stop_at_eos = false;
    o.scheduler_config = ResidentSchedulerConfig {
        max_active_sequences: 2,
        max_decode_batch: 1,
        max_batch_tokens: 8,
        prefill_chunk_size: 8,
        prefix_cache_capacity_pages: 0,
        ..Default::default()
    };
    let plan = ResidentModelPlanner
        .prepare_numeric_fp8_bf16_serving(
            &config,
            BackendSelection::Cuda,
            None,
            o,
            ServingModelOptions {
                model_precision: Some(ModelPrecision::Bf16),
                max_tensor_mebibytes: Some(1024),
                ..Default::default()
            },
            NumericStoragePolicy::E4M3FnBf16Scales,
        )
        .unwrap();
    assert_eq!(plan.backend_profile(), Profile::NAME);
    assert_eq!(plan.numeric_fp8_profile(), None);
    let memory = plan.numeric_bf16_memory_plan().unwrap().unwrap();
    let estimate = memory.model_estimate();
    let budget = memory.model_budget();
    eprintln!(
        "NUMERIC_BF16_FACTORY metadata_only=false strict_full40=false profile={} kv={:?} conv={:?} ssm={:?} rows={:?} source_gdn={:?} estimate={estimate:?} budget={budget:?} total_with_margin={}",
        Profile::NAME,
        Profile::KV_DTYPE,
        Profile::CONV_DTYPE,
        Profile::SSM_DTYPE,
        Profile::ROW_DTYPE,
        Profile::GDN_SOURCE_DTYPE,
        memory.total_device_bytes()
    );
    drop(memory);
    let mut engine = plan.build().unwrap();
    let device = super::bf16_serving::NUMERIC_BF16_TEST_DEVICE
        .with(|slot| slot.borrow_mut().take())
        .expect("factory owner observation");
    assert_eq!(engine.bound_layer_count(), Some(40));
    assert_eq!(engine.model_info().num_layers, 40);
    assert_eq!(engine.model_info().backend, Profile::NAME);
    assert_eq!(
        engine
            .observability_snapshot()
            .kv_cache
            .unwrap()
            .configured_bytes,
        Some(estimate.kv_bytes as u64)
    );
    let initial_state = device.live_state_bytes();
    assert_eq!(initial_state, estimate.per_sequence_state_bytes);
    let request = |id, session, prompt, count| GenerateRequest {
        id: RequestId(id),
        session_id: Some(SessionId(session)),
        prompt_tokens: prompt,
        max_new_tokens: count,
        stop: vec![],
        ignore_eos: true,
    };
    let drain = |engine: &mut BoxedSessionInferenceEngine, id: RequestId, cancelled: bool| {
        let deadline = Instant::now() + Duration::from_secs(600);
        let mut events = Vec::new();
        loop {
            assert!(Instant::now() < deadline, "factory request deadline");
            engine
                .step(&mut |e| {
                    events.push(e.clone());
                    Ok(())
                })
                .unwrap();
            assert!(engine.drain_failed().is_empty());
            if let Some(terminal) = engine.take_request_terminal(id) {
                if cancelled {
                    assert!(matches!(terminal, RequestTerminal::Cancelled(_)));
                } else {
                    assert!(matches!(terminal, RequestTerminal::Finished(_)));
                }
                break;
            }
        }
        for e in &events {
            let v = e.logit.expect("selected LM-head value");
            assert!(v.is_finite());
            assert_eq!(
                v.to_bits() & 0xffff,
                0,
                "BF16 logits exported exactly as F32"
            );
        }
        events
    };
    engine.retain_session(SessionId(40)).unwrap();
    for (id, prompt) in [(100, vec![9419]), (101, vec![760])] {
        let before = engine.observability_snapshot();
        let start = engine.retained_session_position(SessionId(40)).unwrap_or(0);
        engine
            .try_submit(request(id, 40, prompt.clone(), 2))
            .unwrap();
        let events = drain(&mut engine, RequestId(id), false);
        assert_eq!(events.len(), 2);
        assert_eq!(
            engine.retained_session_position(SessionId(40)),
            Some(start + prompt.len() + 2)
        );
        let delta = engine.observability_snapshot().delta_since(&before);
        assert_eq!(
            delta.driver.prefill_tokens,
            prompt.len(),
            "retained turn must not replay its prefix"
        );
        assert!(delta.driver.decode_steps >= 1);
        eprintln!(
            "NUMERIC_BF16_FACTORY request={id} start={start} events={events:?} actual_state={} allocator={:?}",
            device.live_state_bytes(),
            device.operators().allocator_metrics()
        );
        assert!(device.live_state_bytes() <= budget.state_bytes);
    }
    engine.release_session(SessionId(40)).unwrap();
    assert_eq!(engine.retained_session_position(SessionId(40)), None);
    assert_eq!(device.live_state_bytes(), initial_state);
    // Stop after a nonterminal prefill chunk: no uncommitted early token, an
    // active source frontier, and an odd partial page shared by the fork.
    engine
        .try_submit(request(102, 50, vec![9419; 9], 16))
        .unwrap();
    let before = engine.observability_snapshot();
    engine
        .step(&mut |_| panic!("first 8/9 prefill tokens cannot emit a token"))
        .unwrap();
    assert_eq!(
        engine
            .observability_snapshot()
            .delta_since(&before)
            .driver
            .prefill_tokens,
        8
    );
    assert_eq!(
        engine
            .fork_session_exact(SessionId(50), request(103, 51, vec![760], 16), 8)
            .unwrap(),
        SessionId(51)
    );
    engine.cancel_request(RequestId(102)).unwrap();
    let mut saw_child = false;
    let deadline = Instant::now() + Duration::from_secs(600);
    while !saw_child {
        assert!(Instant::now() < deadline);
        engine
            .step(&mut |e| {
                assert_eq!(e.session_id, SessionId(51));
                saw_child = true;
                Ok(())
            })
            .unwrap();
        assert!(engine.drain_failed().is_empty());
    }
    engine.cancel_request(RequestId(103)).unwrap();
    for id in [RequestId(102), RequestId(103)] {
        drain(&mut engine, id, true);
    }
    assert_eq!(
        device.live_state_bytes(),
        initial_state,
        "both cancelled model forks released"
    );
    assert_eq!(
        engine
            .capacity_snapshot()
            .kv
            .unwrap()
            .logical_allocated_pages,
        0
    );
    eprintln!(
        "NUMERIC_BF16_FACTORY released allocator={:?} actual_state={} budget={budget:?}; lifecycle-only, independent full40 reference NOT compared",
        device.operators().allocator_metrics(),
        device.live_state_bytes()
    );
    let metrics = device.operators().allocator_metrics();
    assert!(
        metrics.peak_live_requested_bytes
            <= budget.weight_bytes
                + budget.workspace_bytes
                + budget.state_bytes
                + estimate.kv_bytes
    );
    assert!(metrics.peak_reserved_bytes <= limits.device_bytes);
    loop {
        if engine.shutdown().unwrap() == InferenceShutdownProgress::Complete {
            break;
        }
        assert!(Instant::now() < deadline);
    }
    drop(engine);
    assert_eq!(device.live_state_bytes(), 0);
    assert!(!device.needs_quarantine());
}

#[cfg(feature = "cuda")]
#[path = "bf16_full40.rs"]
mod full40;
