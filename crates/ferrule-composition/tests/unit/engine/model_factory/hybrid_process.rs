use super::*;
use crate::model_factory::qwen35::fixture::Fixture;
use ferrule_runtime::parallel::process::ProcessFrameLimits;

fn request(f: &Fixture) -> ResolvedModelRequest {
    let mut request = super::super::tests::request(f, ModelExecutionBackend::Cpu);
    let options = request.pipeline.as_mut().unwrap();
    options.rank_backend = PipelineRankBackend::Process;
    options.process_launch =
        Some(ProcessLaunch::new("must-not-launch-before-admission").arg("__rank-worker"));
    options.rank_timeout = std::time::Duration::from_millis(6789);
    request
}

#[test]
fn hybrid_process_admission_charges_all_child_and_parent_ipc_domains() {
    let f = Fixture::new();
    let request = request(&f);
    let resources = f.resources();
    let process = Admission::new(&request, &resources).unwrap();
    let p = process.process.as_ref().unwrap();
    let mut thread_request = request.clone();
    let options = thread_request.pipeline.as_mut().unwrap();
    options.rank_backend = PipelineRankBackend::Thread;
    options.process_launch = None;
    options.rank_timeout = std::time::Duration::from_secs(30);
    let thread = Admission::new(&thread_request, &resources).unwrap();
    let metadata = HybridSegmentMemoryEstimate::binding_metadata_bytes(&resources).unwrap() as u64;
    let frames = p.options.frame_limits.max_frame_bytes as u64;
    assert_eq!(
        p.host_bytes,
        2 * (metadata * 4 + (16 << 20) + frames * 32) + frames * 16
    );
    assert_eq!(process.host_required, thread.host_required + p.host_bytes);
    assert!(process.admit_host(thread.host_required).is_err());
    assert!(process.admit_host(process.host_required - 1).is_err());
    process.admit_host(process.host_required).unwrap();
    assert_eq!(
        p.options.command_timeout,
        std::time::Duration::from_millis(6789)
    );
    assert_eq!(p.options.startup_timeout, p.options.command_timeout);
    for (index, (identity, boot)) in p.boots.iter().enumerate() {
        assert_eq!(identity.rank.get(), index as u32);
        assert_eq!(boot.kv.physical_pages, process.config.max_pages * 2);
        assert_eq!(boot.recipe, DecoderRecipeKind::Qwen35DenseHybrid);
        assert!(boot.experts.is_none());
        boot.hybrid
            .as_ref()
            .unwrap()
            .validate_resources(boot, &resources)
            .unwrap();
    }
    assert!(process_host_bytes(u64::MAX, 1, 2).is_err());
    assert!(process_host_bytes(1, u64::MAX, 2).is_err());
    assert!(process_host_bytes(1, 1, u64::MAX).is_err());
}

#[test]
fn hybrid_full_vocab_frames_fit_both_peers_or_fail_before_launch() {
    let f = Fixture::new();
    let request = request(&f);
    let admission = Admission::new(&request, &f.resources()).unwrap();
    let mut boot = admission.process.as_ref().unwrap().boots[1].1.clone();
    boot.hybrid.as_mut().unwrap().vocabulary = 248_320;
    boot.kv.max_batch_tokens = 8;
    let defaults = ProcessFrameLimits::default();
    assert!(boot.validate_frame_limits(defaults).is_err());
    let fit = boot.fit_frame_limits(defaults).unwrap();
    assert_eq!(fit.max_output_bytes, 248_320 * 8 * 32 + 4096);
    assert_eq!(fit.max_frame_bytes, fit.max_output_bytes + 1024);
    assert_eq!(fit.max_config_bytes, defaults.max_config_bytes);
    assert!(fit.max_frame_bytes <= ProcessFrameLimits::MAX_FRAME_BYTES);
    boot.validate_frame_limits(fit).unwrap();
    let mut too_small = fit;
    too_small.max_output_bytes -= 1;
    assert!(boot.validate_frame_limits(too_small).is_err());
    boot.kv.max_batch_tokens = 16;
    assert!(boot.fit_frame_limits(defaults).is_ok());
    boot.kv.max_batch_tokens = 32;
    let error = boot.fit_frame_limits(defaults).unwrap_err();
    assert!(error.to_string().contains("128 MiB"));
    boot.kv.max_batch_tokens = 8;
    boot.hybrid.as_mut().unwrap().hidden = usize::MAX;
    assert!(boot.fit_frame_limits(defaults).is_err());
}

#[test]
fn hybrid_process_public_policy_preserves_launch_and_rejects_moe_and_restarts() {
    let f = Fixture::new();
    let request = request(&f);
    let options = request.pipeline.as_ref().unwrap();
    assert_eq!(
        options
            .hybrid_precision(&ModelFamily::Qwen35, ModelExecutionBackend::Cpu)
            .unwrap(),
        ExecutionPrecisionPolicy::f32()
    );
    assert!(
        options
            .hybrid_precision(&ModelFamily::Qwen35Moe, ModelExecutionBackend::Cpu)
            .is_err()
    );
    for axis in 0..6 {
        let mut bad = options.clone();
        match axis {
            0 => bad.parallelism.expert_parallel = 2,
            1 => bad.parallelism.tensor_parallel = 2,
            2 => bad.parallelism.pipeline_parallel = 4,
            3 => bad.process_launch = None,
            4 => bad.rank_timeout = std::time::Duration::ZERO,
            _ => bad.rank_restarts = 1,
        }
        assert!(
            bad.hybrid_precision(&ModelFamily::Qwen35, ModelExecutionBackend::Cpu)
                .is_err()
        );
    }
}

#[cfg(feature = "cpu-reference")]
#[test]
fn hybrid_production_factory_executes_real_children_and_session_lifecycle() {
    use ferrule_common::execution::ForwardPhase;
    use ferrule_runtime::engine::{
        InferenceEngine, InferenceShutdownProgress, SessionInferenceEngine,
    };
    use ferrule_runtime::scheduling::{GenerateRequest, RequestId, SessionId};
    let f = Fixture::new();
    let mut request = request(&f);
    static CHILD: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    let child = CHILD.get_or_init(|| {
        crate::build_process_child::build("ferrule-runtime", "example", "process_rank_child")
    });
    request.pipeline.as_mut().unwrap().process_launch =
        Some(ProcessLaunch::new(child).arg("hybrid_decoder"));
    let admission = Admission::new(&request, &f.resources()).unwrap();
    admission.admit_live(ModelExecutionBackend::Cpu).unwrap();
    let mut pipeline = executor(&request, f.resources(), admission).unwrap();
    let out = pipeline
        .forward(SessionId(1), &[1, 3, 2], ForwardPhase::Prefill)
        .unwrap();
    for row in 0..3 {
        for (a, b) in out
            .logits
            .row(row)
            .unwrap()
            .iter()
            .zip(f.oracle["logits"][row].as_array().unwrap())
        {
            let b = b.as_f64().unwrap() as f32;
            assert!((a - b).abs() <= 3e-5 + 3e-5 * b.abs());
        }
    }
    pipeline.fork_session(SessionId(1), SessionId(2)).unwrap();
    for session in [SessionId(2), SessionId(1)] {
        pipeline
            .forward(session, &[5], ForwardPhase::Decode)
            .unwrap();
        pipeline.release_session(session).unwrap();
    }
    assert_eq!(pipeline.page_manager().allocated_pages(), 0);
    pipeline.shutdown().unwrap();
    let mut engine = build_bound(
        request,
        f.resources(),
        TokenizerHandle::load(&f.dir).unwrap(),
    )
    .unwrap();
    engine.retain_session(SessionId(7)).unwrap();
    engine
        .try_submit(GenerateRequest {
            id: RequestId(1),
            session_id: Some(SessionId(7)),
            prompt_tokens: vec![1, 3, 2],
            max_new_tokens: 1,
            stop: vec![],
            ignore_eos: true,
        })
        .unwrap();
    let mut finished = false;
    for _ in 0..100 {
        engine.step(&mut |_| Ok(())).unwrap();
        if engine.take_request_terminal(RequestId(1)).is_some() {
            finished = true;
            break;
        }
    }
    assert!(finished);
    let position = engine.retained_session_position(SessionId(7)).unwrap();
    engine
        .fork_session_exact(SessionId(7), SessionId(8), position)
        .unwrap();
    engine.release_session(SessionId(8)).unwrap();
    engine.release_session(SessionId(7)).unwrap();
    assert!(matches!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn hybrid_process_cuda_metadata_preserves_device_order_and_owner_budgets() {
    let f = Fixture::new();
    let mut request = super::super::tests::request(&f, ModelExecutionBackend::Cuda);
    let options = request.pipeline.as_mut().unwrap();
    options.rank_backend = PipelineRankBackend::Process;
    options.process_launch = Some(ProcessLaunch::new("must-not-launch"));
    options.devices = Some(vec![3, 1]);
    let admission = Admission::new(&request, &f.resources()).unwrap();
    let plan = admission.process.as_ref().unwrap();
    for (index, (_, boot)) in plan.boots.iter().enumerate() {
        assert_eq!(
            boot.device,
            DecoderDevice::Cuda {
                ordinal: [3, 1][index]
            }
        );
        let actual = boot.hybrid.as_ref().unwrap().cuda_budget.as_ref().unwrap();
        let expected = &admission.stages[index];
        assert_eq!(actual.state_bytes as u64, expected.state);
        assert_eq!(actual.weight_bytes as u64, expected.weights);
        assert_eq!(actual.workspace_bytes as u64, expected.workspace);
        assert_eq!(actual.margin_bytes as u64, expected.margin);
    }
}
