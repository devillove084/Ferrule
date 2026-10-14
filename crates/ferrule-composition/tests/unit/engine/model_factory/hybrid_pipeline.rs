use super::*;
use ferrule_common::execution::{ForwardPhase, KvLayoutSchema};
use ferrule_model::models::qwen35::Qwen35Recipe;
use ferrule_runtime::engine::{InferenceEngine, InferenceShutdownProgress, SessionInferenceEngine};
use ferrule_runtime::scheduling::{GenerateRequest, RequestId, SessionId};
use std::sync::atomic::AtomicBool;

use crate::model_factory::qwen35::fixture::Fixture;

const STRICT_CONFIG: &str =
    include_str!("../../../../../ferrule-model/tests/qwen35_08b_config.json");
fn options() -> PipelineBuildOptions {
    PipelineBuildOptions {
        parallelism: ParallelismPlan {
            pipeline_parallel: 2,
            ..Default::default()
        },
        ..Default::default()
    }
}
fn public_plan(fixture: &Fixture, backend: BackendSelection) -> ResidentModelBuildPlan {
    std::fs::write(fixture.dir.join("config.json"), STRICT_CONFIG).unwrap();
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
    ResidentModelPlanner
        .prepare_pipeline(&config, backend, None, input, options())
        .unwrap()
}
pub(super) fn request(fixture: &Fixture, backend: ModelExecutionBackend) -> ResolvedModelRequest {
    let mut request = public_plan(fixture, backend.into()).request;
    // Only this private, already-bound fixture seam has tiny dimensions. The
    // public strict parser and production binder are not overridden.
    request.max_layers = 4;
    request.model_info.num_layers = 4;
    request.model_info.hidden_size = 8;
    request.model_info.vocab_size = 11;
    request
}
fn close(actual: &[f32], expected: &serde_json::Value, tolerance: f32) {
    let expected = expected.as_array().unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        let b = b.as_f64().unwrap() as f32;
        assert!(
            (a - b).abs() <= tolerance + tolerance * b.abs(),
            "{a} != {b}"
        );
    }
}
fn real_executor(backend: ModelExecutionBackend, devices: Option<Vec<usize>>) {
    let fixture = Fixture::new();
    let mut request = request(&fixture, backend);
    request.pipeline.as_mut().unwrap().devices = devices;
    let resources = fixture.resources();
    let admission = Admission::new(&request, &resources).unwrap();
    admission.admit_live(backend).unwrap();
    let bytes = admission.kv.configured_bytes.unwrap();
    let logical = admission.config.max_pages;
    let mut pp = executor(&request, resources, admission).unwrap();
    assert_eq!(pp.physical_kv_bytes().unwrap(), bytes);
    for description in pp.stage_descriptions() {
        assert_eq!(description.physical_pages, 2 * logical);
    }
    let output = pp
        .forward(SessionId(1), &[1, 3, 2], ForwardPhase::Prefill)
        .unwrap();
    let tolerance = if backend == ModelExecutionBackend::Cpu {
        3e-5
    } else {
        3e-4
    };
    for i in 0..3 {
        close(
            output.logits.row(i).unwrap(),
            &fixture.oracle["logits"][i],
            tolerance,
        );
    }
    pp.fork_session(SessionId(1), SessionId(2)).unwrap();
    let cancelled = AtomicBool::new(true);
    assert!(
        pp.forward_cancellable(SessionId(1), &[9], ForwardPhase::Decode, &cancelled)
            .is_err()
    );
    assert_eq!(pp.coordinator().publication_count(), 1);
    for session in [SessionId(1), SessionId(2)] {
        for (token, row) in [(5, 3), (7, 4)] {
            let output = pp.forward(session, &[token], ForwardPhase::Decode).unwrap();
            close(
                output.logits.row(0).unwrap(),
                &fixture.oracle["logits"][row],
                tolerance,
            );
        }
        pp.release_session(session).unwrap();
    }
    assert_eq!(pp.page_manager().allocated_pages(), 0);
    pp.shutdown().unwrap();
}
#[test]
fn cpu_factory_executor_matches_oracle_fork_cancel_and_release() {
    real_executor(ModelExecutionBackend::Cpu, None);
}
fn generate(id: u64, session: u64, tokens: Vec<u32>, max: usize) -> GenerateRequest {
    GenerateRequest {
        id: RequestId(id),
        session_id: Some(SessionId(session)),
        prompt_tokens: tokens,
        max_new_tokens: max,
        stop: vec![],
        ignore_eos: true,
    }
}
fn finish(engine: &mut PipelineInferenceEngine, id: u64) {
    for _ in 0..100 {
        engine.step(&mut |_| Ok(())).unwrap();
        if engine.take_request_terminal(RequestId(id)).is_some() {
            return;
        }
    }
    panic!("factory engine did not finish within the step budget");
}
fn real_engine(backend: ModelExecutionBackend, devices: Option<Vec<usize>>) {
    let fixture = Fixture::new();
    let mut request = request(&fixture, backend);
    request.pipeline.as_mut().unwrap().devices = devices;
    let mut engine = build_bound(
        request,
        fixture.resources(),
        TokenizerHandle::load(&fixture.dir).unwrap(),
    )
    .unwrap();
    engine.retain_session(SessionId(7)).unwrap();
    engine.try_submit(generate(1, 7, vec![1, 3, 2], 1)).unwrap();
    finish(&mut engine, 1);
    let position = engine.retained_session_position(SessionId(7)).unwrap();
    assert!(position >= 3);
    engine
        .fork_session_exact(SessionId(7), SessionId(8), position)
        .unwrap();
    engine.try_submit(generate(2, 8, vec![5], 2)).unwrap();
    finish(&mut engine, 2);
    assert!(engine.retained_session_position(SessionId(8)).unwrap() > 3);
    engine.release_session(SessionId(8)).unwrap();
    engine.try_submit(generate(3, 7, vec![5], 2)).unwrap();
    engine.cancel_request(RequestId(3)).unwrap();
    assert_eq!(engine.drain_cancelled().len(), 1);
    assert_eq!(engine.retained_session_position(SessionId(7)), None);
    engine.try_submit(generate(4, 9, vec![1, 3, 2], 2)).unwrap();
    finish(&mut engine, 4);
    assert_eq!(engine.pipeline().page_manager().allocated_pages(), 0);
    assert!(matches!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    ));
    assert!(engine.try_submit(generate(5, 10, vec![1], 0)).is_err());
}
#[test]
fn cpu_factory_engine_generate_retain_fork_cancel_reset_shutdown() {
    real_engine(ModelExecutionBackend::Cpu, None);
}
#[test]
fn public_dense_plan_selects_f32_but_strict_binding_rejects_tiny_inventory() {
    let fixture = Fixture::new();
    let plan = public_plan(&fixture, BackendSelection::Cpu);
    assert_eq!(
        plan.selected_implementation().kind,
        SelectedImplementationKind::Pipeline
    );
    let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
        build_hybrid_pipeline;
    assert!(std::ptr::fn_addr_eq(
        plan.request.execution.construction.builder(),
        expected
    ));
    assert_eq!(plan.support().arithmetic, ArithmeticPrecision::F32);
    assert!(plan.support().static_constraints.contains("PP2 TP1 EP1"));
    assert!(
        !plan
            .effective_options()
            .scheduler_config
            .allow_mixed_batches
    );
    assert_eq!(
        plan.effective_options().scheduler_config.max_decode_batch,
        1
    );
    assert!(
        plan.build().is_err(),
        "tiny fixture must not pass strict production binding"
    );
}
#[test]
fn public_hybrid_policy_rejects_unfinished_topologies_and_overrides() {
    for case in 0..7 {
        let mut p = options();
        match case {
            0 => p.rank_backend = PipelineRankBackend::Process,
            1 => p.parallelism.tensor_parallel = 2,
            2 => p.parallelism.expert_parallel = 2,
            3 => p.parallelism.pipeline_parallel = 1,
            4 => p.parallelism.pipeline_parallel = 4,
            5 => p.rank_restarts = 1,
            _ => p.devices = Some(vec![0, 0]),
        }
        assert!(
            p.hybrid_precision(&ModelFamily::Qwen35, ModelExecutionBackend::Cpu)
                .is_err()
        );
    }
    assert!(
        options()
            .hybrid_cpu_precision(&ModelFamily::Qwen35Moe)
            .is_err()
    );
    let fixture = Fixture::new();
    let mut request = request(&fixture, ModelExecutionBackend::Cpu);
    request.max_layers = 3;
    assert!(Admission::new(&request, &fixture.resources()).is_err());
}
#[test]
fn factory_stage_host_kv_and_colocated_budgets_fail_closed() {
    let fixture = Fixture::new();
    let mut request = request(&fixture, ModelExecutionBackend::Cpu);
    let resources = fixture.resources();
    let a = Admission::new(&request, &resources).unwrap();
    assert_eq!(a.layout.stage_kv_layers(0), 1);
    assert_eq!(a.layout.stage_kv_layers(1), 1);
    assert_eq!(a.kv.configured_pages, 2 * a.kv.full_capacity_pages);
    assert!(a.host_required > a.stages.iter().map(|s| s.total().unwrap()).sum::<u64>());
    a.admit_host(a.host_required).unwrap();
    assert!(a.admit_host(a.host_required - 1).is_err());
    let total = a.stages.iter().map(|s| s.total().unwrap()).sum::<u64>();
    assert!(HybridDeviceBudget::admit(&a.stages, &[(0, total - 1)]).is_err());
    let admitted = HybridDeviceBudget::admit(&a.stages, &[(0, total)]).unwrap();
    assert_eq!(admitted.len(), 1);
    assert_eq!(admitted[0].required, total);
    request.kv_cache_bytes = Some(a.kv.configured_bytes.unwrap() - 1);
    assert!(Admission::new(&request, &resources).is_err());
    request.kv_cache_bytes = a.kv.configured_bytes;
    Admission::new(&request, &resources).unwrap();
    request.max_tensor_bytes = 1;
    assert!(Admission::new(&request, &resources).is_err());
    request.max_tensor_bytes = u64::MAX;
    request.driver_config.ctx_size = 33;
    assert!(Admission::new(&request, &resources).is_err());
    assert!(mul(u64::MAX, 2).is_err());
    assert!(add(u64::MAX, 1).is_err());
}
#[test]
fn strict_08b_metadata_has_local_kv_3_plus_3_and_tied_endpoints() {
    let spec = Qwen35Recipe::new()
        .build_spec(&serde_json::from_str(STRICT_CONFIG).unwrap())
        .unwrap();
    assert!(spec.tie_word_embeddings());
    let plans = segment_plans(24, 2).unwrap();
    let layout = HybridPipelineLayout::new(&spec, &plans).unwrap();
    assert_eq!(plans[0].layers(), 0..12);
    assert_eq!(plans[1].layers(), 12..24);
    assert!(plans[0].owns_embedding() && !plans[0].owns_output());
    assert!(!plans[1].owns_embedding() && plans[1].owns_output());
    assert_eq!(layout.kv_layers(), 6);
    for (i, schema) in layout.schemas().iter().enumerate() {
        assert_eq!(schema.per_sequence_state_bytes().unwrap(), 10_321_920);
        let entries: Vec<_> = schema
            .entries()
            .iter()
            .filter(|e| e.local_kv_layer.is_some())
            .collect();
        assert_eq!(
            entries
                .iter()
                .map(|e| e.local_kv_layer.unwrap())
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            entries
                .iter()
                .map(|e| e.global_kv_layer.unwrap())
                .collect::<Vec<_>>(),
            (3 * i..3 * i + 3).collect::<Vec<_>>()
        );
        assert_eq!(
            schema.kv_planes(16, 32).unwrap().checked_page_bytes(),
            Some(196_608)
        );
    }
}
#[cfg(feature = "cuda")]
#[test]
fn cuda_factory_metadata_plan_and_colocation_without_device_access() {
    let fixture = Fixture::new();
    let mut request = request(&fixture, ModelExecutionBackend::Cuda);
    request.pipeline.as_mut().unwrap().devices = Some(vec![3, 3]);
    let a = Admission::new(&request, &fixture.resources()).unwrap();
    assert_eq!(a.stages[0].device, 3);
    assert_eq!(a.stages[1].device, 3);
    let total = a.stages.iter().map(|s| s.total().unwrap()).sum();
    assert!(HybridDeviceBudget::admit(&a.stages, &[(3, total)]).is_ok());
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires one CUDA GPU; explicit opt-in only"]
fn cuda_factory_tiny_colocated_oracle_fork_cancel() {
    real_executor(ModelExecutionBackend::Cuda, Some(vec![0, 0]));
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires two CUDA GPUs; explicit opt-in only"]
fn cuda_factory_tiny_two_cards_engine_lifecycle() {
    real_engine(ModelExecutionBackend::Cuda, Some(vec![0, 1]));
}

#[test]
fn tied_output_keeps_canonical_embedding_storage_on_second_owner() {
    use ferrule_model::transformer::{
        DecoderModelParts, ExactNameMapper, HybridDecoderSegment, NameMapping, StateDictBinder,
        StateDictSchema,
    };
    let fixture = Fixture::new();
    let resources = fixture.resources();
    let spec = resources.spec();
    let tied_spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: spec.architecture().into(),
        hidden_size: spec.hidden_size(),
        vocab_size: spec.vocab_size(),
        max_sequence_length: spec.max_sequence_length(),
        token_embedding: spec.token_embedding().clone(),
        layers: spec.layers().to_vec(),
        final_norm: spec.final_norm().clone(),
        output: spec.output().clone(),
        tie_word_embeddings: true,
    })
    .unwrap();
    let embedding = resources
        .state_dict()
        .parameters()
        .iter()
        .find(|p| *p.role() == TensorRole::TokenEmbedding)
        .unwrap();
    let mut schema = StateDictSchema::builder();
    let mut mapper = ExactNameMapper::new();
    let mut slices = Vec::new();
    for p in resources.state_dict().parameters() {
        let mut parameter = p.spec().clone();
        if *p.role() == TensorRole::OutputHead {
            parameter = parameter.with_alias(embedding.id());
        } else {
            mapper
                .insert(
                    &p.weight().slice().name,
                    NameMapping::weight(p.path().clone()),
                )
                .unwrap();
            slices.push(p.weight().slice().clone());
        }
        schema
            .register_with_role(parameter, p.role().clone())
            .unwrap();
    }
    let schema = schema.build().unwrap();
    let tied = BoundDecoderResources::new(
        tied_spec,
        std::sync::Arc::new(
            StateDictBinder::new(&schema, &mapper)
                .bind_slices(slices)
                .unwrap(),
        ),
    )
    .unwrap();
    let request = request(&fixture, ModelExecutionBackend::Cpu);
    let a = Admission::new(&request, &tied).unwrap();
    let untied = Admission::new(&request, &resources).unwrap();
    for (stage, plan) in a.plans.iter().enumerate() {
        // Equal endpoint shapes still consume one physical copy per owner.
        assert_eq!(a.stages[stage].weights, untied.stages[stage].weights);
        let segment = HybridDecoderSegment::prepare(
            &tied,
            plan.clone(),
            a.config.precision,
            a.config.max_positions,
            a.config.max_parameter_bytes,
        )
        .unwrap();
        let canonical: Vec<_> = segment
            .parameters()
            .iter()
            .filter(|p| p.canonical_id() == embedding.id())
            .collect();
        assert_eq!(canonical.len(), 1);
        assert_eq!(
            *canonical[0].role(),
            if stage == 0 {
                TensorRole::TokenEmbedding
            } else {
                TensorRole::OutputHead
            }
        );
    }
    a.admit_live(ModelExecutionBackend::Cpu).unwrap();
    let mut pp = executor(&request, tied, a).unwrap();
    let output = pp
        .forward(SessionId(1), &[1, 3, 2], ForwardPhase::Prefill)
        .unwrap();
    assert!(output.logits.row(2).unwrap().iter().all(|v| v.is_finite()));
    pp.release_session(SessionId(1)).unwrap();
    pp.shutdown().unwrap();
}
