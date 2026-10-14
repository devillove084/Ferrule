//! Experimental production-worker gates. GPU cases are compile-only unless explicitly granted.
#![cfg(feature = "cuda")]
#[path = "tp2_mixed_execution/capture.rs"]
mod capture;
#[path = "tp2_mixed_execution/decoded_cache.rs"]
mod decoded_cache;
#[path = "tp2_mixed_execution/lifecycle.rs"]
mod lifecycle;
#[path = "tp2_mixed_execution/oracle.rs"]
mod oracle;
#[path = "tp2_mixed_execution/quality_path.rs"]
mod quality_path;
#[path = "tp2_mixed_execution/tiny.rs"]
mod tiny;
use ferrule_common::{ParallelismPlan, execution::ForwardPhase};
use ferrule_composition::model_factory::*;
use ferrule_model::{
    execution::ExecutionPrecisionPolicy, transformer::hybrid_parallel::HybridRankPrecision,
};
use ferrule_runtime::{
    SessionId,
    parallel::{
        collective::HostCollectiveLimits,
        pipeline::{PipelineConfig, PipelineParallelExecutor},
    },
};

fn placement() -> PipelineBuildOptions {
    PipelineBuildOptions {
        parallelism: ParallelismPlan {
            tensor_parallel: 2,
            ..Default::default()
        },
        devices: Some(vec![0, 1]),
        ..Default::default()
    }
}
fn contract() -> ParallelExecutionContract {
    ParallelExecutionContract::new(
        HybridRankPrecision::MixedBf16F32,
        ParallelExpertMode::LocalTensorShards,
    )
}
fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: 16,
        max_pages: 32,
        max_positions: 128,
        max_batch_tokens: 8,
        session_capacity: 2,
        max_parameter_bytes: 2 << 30,
        max_ack_polls: 8,
        precision: ExecutionPrecisionPolicy::bf16_compatibility(),
    }
}
fn grant() {
    ferrule_common::observability::init_tracing();
    assert_eq!(
        std::env::var("FERRULE_MIXED_GPU_RUN").as_deref(),
        Ok("1"),
        "fresh two-GPU owner grant required"
    );
    assert!(
        !std::env::var("CUDA_VISIBLE_DEVICES")
            .unwrap_or_default()
            .is_empty(),
        "GPUs must be explicitly assigned by owner"
    );
    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
}
fn build(
    resources: ferrule_model::transformer::BoundDecoderResources,
    config: PipelineConfig,
) -> PipelineParallelExecutor {
    build_with_trace(resources, config, None)
}
fn build_with_trace(
    resources: ferrule_model::transformer::BoundDecoderResources,
    config: PipelineConfig,
    trace: Option<ferrule_runtime::parallel::pipeline::MixedTraceConfig>,
) -> PipelineParallelExecutor {
    build_with_trace_and_cache(resources, config, trace, None)
}
fn build_with_trace_and_cache(
    resources: ferrule_model::transformer::BoundDecoderResources,
    config: PipelineConfig,
    trace: Option<ferrule_runtime::parallel::pipeline::MixedTraceConfig>,
    cache_grants: Option<usize>,
) -> PipelineParallelExecutor {
    let plan = ParallelExecutionPlan::describe(resources.spec(), &placement(), contract(), vec![])
        .unwrap();
    assert_eq!(plan.root_count(), 2);
    assert_eq!(plan.external_owner_count(), 0);
    assert!(
        plan.validate_execution().is_err(),
        "experimental gate must not advertise quality"
    );
    let max_elements = config.max_batch_tokens
        * resources
            .spec()
            .vocab_size()
            .div_ceil(2)
            .max(resources.spec().hidden_size());
    // Keep the base execution host allowance; separately admit the complete
    // optional capture, including a writer stalled for all four invocations.
    let capacity = ParallelRankCapacity {
        page_size: config.page_size,
        max_positions: config.max_positions,
        max_rows: config.max_batch_tokens,
        sessions: config.session_capacity,
        logical_pages: config.max_pages,
    };
    let trace_host = trace
        .as_ref()
        .map(|t| {
            t.host_reserve_with_pages(plan.root_count(), capacity.physical_pages().unwrap())
                .unwrap()
        })
        .unwrap_or(0);
    let host_capacity = (256u64 << 20).checked_add(trace_host).unwrap();
    assert!(host_capacity <= 128u64 << 30, "supervised host target");
    plan.start_internal_with_decoded_cache(
        resources,
        config,
        ParallelOwnerMargins {
            context_bytes: 64 << 20,
            allocator_bytes: 64 << 20,
        },
        &[(0, 22 << 30), (1, 22 << 30)],
        host_capacity,
        HostCollectiveLimits {
            max_ranks: 2,
            max_elements_per_rank: max_elements,
            max_host_bytes: 64 << 20,
        },
        std::time::Duration::from_secs(60),
        trace,
        cache_grants,
    )
    .unwrap()
}

#[test]
#[ignore = "GPU owner only: real full40 checkpoint and pinned independent paper oracle; no CLI enablement"]
fn full40_independent_logits_prefill_one_decode() {
    full40(false);
}

#[test]
#[ignore = "owner permission required: candidate boundary export; not full inventory acceptance"]
fn full40_candidate_boundaries_two_cases_two_phases() {
    full40(true);
}

fn full40(capture_enabled: bool) {
    full40_with_cache(capture_enabled, None);
}
fn full40_with_cache(capture_enabled: bool, cache_grants: Option<usize>) {
    grant();
    let path = std::path::PathBuf::from(
        std::env::var("FERRULE_MIXED_MODEL").expect("explicit original checkpoint"),
    );
    let reference = oracle::Oracle::open(
        &std::path::PathBuf::from(
            std::env::var("FERRULE_MIXED_ORACLE_DIR")
                .expect("completed independent paper artifact"),
        ),
        &std::env::var("FERRULE_MIXED_ORACLE_SHA256").expect("explicit manifest pin"),
        &path,
    );
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(&path).unwrap();
    assert_eq!(resources.spec().layers().len(), 40);
    assert_eq!(resources.spec().vocab_size(), 248_320);
    // Authenticate all artifacts and source files before opening either owner.
    let cases = [reference.case("hello"), reference.case("capital")];
    let path = quality_path::QualityPath::from_capture(capture_enabled);
    let evidence = std::path::PathBuf::from(std::env::var("FERRULE_MIXED_EVIDENCE_DIR").unwrap());
    capture::write_new(
        evidence.join("execution-path.json"),
        &path.contract(cache_grants),
    );
    let capture = path.captures().then(capture::CandidateCapture::new);
    let trace = capture.as_ref().map(|capture| capture.config());
    let mut pipeline = build_with_trace_and_cache(resources, config(), trace, cache_grants);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for (index, case) in cases.iter().enumerate() {
            let session = SessionId(index as u64 + 1);
            for (stage, phase) in [
                ("prefill", ForwardPhase::Prefill),
                ("decode.0", ForwardPhase::Decode),
            ] {
                let tokens = case.tokens(stage);
                if stage == "decode.0" {
                    assert_eq!(tokens.len(), 1);
                }
                let start = quality_path::frontier(&pipeline, session);
                assert_eq!(case.call(stage)["position_start"], start);
                let before = pipeline.coordinator().publication_count();
                let result = pipeline.forward(session, &tokens, phase);
                if let Some(capture) = &capture {
                    capture.receipt(index, stage, session, &tokens, &result);
                }
                let actual = result.unwrap();
                let publication =
                    quality_path::publication(&pipeline, session, before, start, tokens.len());
                capture::write_new(
                    evidence.join(format!("case-{index}.{stage}.publication.json")),
                    &publication,
                );
                if let Some(capture) = &capture {
                    let export = pipeline.export_committed_trace(session, actual.transaction);
                    capture.kv_receipt(index, stage, session, actual.transaction, &export);
                    export.unwrap();
                }
                case.compare(stage, actual.logits.values());
            }
            pipeline.release_session(session).unwrap();
        }
        let owners = pipeline.owner_stats().unwrap();
        assert_eq!(owners.len(), 2);
        assert!(
            owners
                .iter()
                .all(|o| o.layers == (0..40) && o.experts.is_empty() && o.expert_outstanding == 0)
        );
        assert_eq!(pipeline.outstanding(), 0);
    }));
    let cleanup = pipeline.shutdown();
    if (cleanup.is_err() || result.is_err())
        && let Some(capture) = &capture
    {
        capture.abort();
    }
    let capture_complete = capture.as_ref().is_none_or(|capture| capture.finish());
    assert!(
        cleanup.is_ok(),
        "owner cleanup must prove quiescence: {cleanup:?}"
    );
    if let Err(panic) = result {
        std::panic::resume_unwind(panic);
    }
    assert!(
        capture_complete,
        "available trace capture incomplete; inspect capture-status.json"
    );
    eprintln!(
        "Full40 path={path:?} completed; independent full-boundary comparison and serving acceptance remain separate"
    );
}
