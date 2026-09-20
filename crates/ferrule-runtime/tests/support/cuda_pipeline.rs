//! GPU PP/EP integration; shared fixtures/oracle only, no test-owned worker loop.
use super::*;
use ferrule_backend::cuda::operators::linear::CudaOperators;
use ferrule_model::nn::ParameterResidency;
use ferrule_model::transformer::expert_parallel::{ExpertDispatchLimits, ExpertPlacement};
use ferrule_runtime::parallel::expert::{ExpertGroup, ExpertParallelExecutor, ExpertRankWorker};

fn group(layers: std::ops::Range<usize>) -> ExpertGroup {
    let base = 100 + layers.start as u32 * 100;
    let members = vec![rank(base + 9), rank(base + 2)];
    ExpertGroup {
        source_rank: members[1],
        placement: ExpertPlacement::new(
            layers
                .clone()
                .flat_map(|layer| [(layer, 0, members[1]), (layer, 1, members[0])]),
        )
        .unwrap(),
        members,
        layers,
        limits: ExpertDispatchLimits {
            max_tokens: MAX * 2,
            max_bytes: MAX * 2 * 4 * 4,
        },
    }
}

fn gpu_pipeline(fixture: &Fixture, ep: bool) -> PipelineParallelExecutor {
    let path = fixture.0.clone();
    let cfg = config(ExecutionPrecisionPolicy::f32());
    PipelineParallelExecutor::new_with_program(topology(2), plans(2), cfg, move |pp, plan| {
        let resources = load(&path)?;
        // Device objects are created here, never in the coordinator/factory capture.
        let ops = Rc::new(CudaOperators::new_on_device(pp.local.get() as usize)?);
        if ep {
            let path = path.clone();
            let experts =
                ExpertParallelExecutor::new_cuda(group(plan.layers()), move |owner, group| {
                    let resources = load(&path)?;
                    let ops = Rc::new(CudaOperators::new_on_device(
                        2 + pp.local.get() as usize * 2 + owner.slot.get() as usize,
                    )?);
                    ExpertRankWorker::prepare_cuda(
                        &resources,
                        owner,
                        group,
                        cfg.max_parameter_bytes,
                        ops,
                    )
                })?;
            PipelineStage::prepare_cuda_with_experts(&resources, plan, cfg, ops, experts)
        } else {
            PipelineStage::prepare_cuda(&resources, plan, cfg, ops)
        }
    })
    .unwrap()
}

fn close(actual: &DenseLogits, expected: &ExecutionOutput) {
    assert_eq!(actual.rows(), expected.logits.len());
    for (row, expected) in expected.logits.iter().enumerate() {
        let LogitsOutput::Full(expected) = &expected.logits else {
            panic!("missing CPU logits")
        };
        assert_eq!(actual.width(), expected.len());
        for (&a, &e) in actual.row(row).unwrap().iter().zip(expected) {
            assert!(
                a.is_finite() && (a - e).abs() <= 2e-5 + 2e-4 * e.abs(),
                "CUDA={a}, CPU={e}"
            );
        }
    }
}

fn run(ep: bool) {
    let fixture = Fixture::with_top_k(true, 2);
    let precision = ExecutionPrecisionPolicy::f32();
    let mut source_oracle = Oracle::new(&fixture, precision);
    let mut branch_oracle = Oracle::new(&fixture, precision);
    let prefix = source_oracle.forward(&[1, 2, 3], ForwardPhase::Prefill);
    branch_oracle.forward(&[1, 2, 3], ForwardPhase::Prefill);
    let source_next = source_oracle.forward(&[4], ForwardPhase::Decode);
    let branch_next = branch_oracle.forward(&[5], ForwardPhase::Decode);
    let source_end = source_oracle.forward(&[6], ForwardPhase::Decode);
    let mut pipeline = gpu_pipeline(&fixture, ep);
    let initial = pipeline.owner_stats().unwrap();
    assert_eq!(initial.len(), 2);
    for owner in &initial {
        assert_ne!(owner.thread, thread::current().id());
        assert_eq!(owner.experts.len(), if ep { 2 } else { 0 });
        for expert in &owner.experts {
            assert_ne!(expert.thread, owner.thread);
            assert_eq!(expert.owned_experts.len(), 1);
        }
    }
    if ep {
        // Resident expert owners are authoritative. A local fallback/read after
        // startup now fails instead of accidentally making this test pass.
        for parameter in load(&fixture.0).unwrap().state_dict().parameters() {
            if matches!(parameter.residency(), ParameterResidency::Expert { .. }) {
                std::fs::remove_file(&parameter.weight().slice().path).unwrap();
            }
        }
    }
    let source = SessionId(11);
    let branch = SessionId(12);
    close(
        &pipeline
            .forward(source, &[1, 2, 3], ForwardPhase::Prefill)
            .unwrap()
            .logits,
        &prefix,
    );
    drained(&pipeline);
    pipeline.fork_session(source, branch).unwrap();
    let source_slot = pipeline.session_slot(source).unwrap();
    let original = pipeline
        .page_manager()
        .block_table(source_slot)
        .unwrap()
        .pages()
        .to_vec();
    assert_eq!(pipeline.page_manager().page_refcount(original[1]), 2);

    // Cancel after all real GPU work and readiness, but before commit decision.
    let cancel = AtomicBool::new(false);
    assert!(
        pipeline
            .forward_observed(source, &[7], ForwardPhase::Decode, &cancel, |progress| {
                if progress.state == TransactionState::Preparing {
                    cancel.store(true, Ordering::Release);
                }
            })
            .is_err()
    );
    drained(&pipeline);
    assert_eq!(pipeline.coordinator().publication_count(), 1);
    assert_eq!(
        pipeline
            .page_manager()
            .block_table(source_slot)
            .unwrap()
            .pages(),
        original
    );
    assert_eq!(
        pipeline
            .page_manager()
            .block_table(source_slot)
            .unwrap()
            .committed_tokens(),
        3
    );
    assert_eq!(pipeline.page_manager().allocated_pages(), 2);

    close(
        &pipeline
            .forward(source, &[4], ForwardPhase::Decode)
            .unwrap()
            .logits,
        &source_next,
    );
    close(
        &pipeline
            .forward(branch, &[5], ForwardPhase::Decode)
            .unwrap()
            .logits,
        &branch_next,
    );
    close(
        &pipeline
            .forward(source, &[6], ForwardPhase::Decode)
            .unwrap()
            .logits,
        &source_end,
    );
    drained(&pipeline);
    assert_eq!(pipeline.coordinator().publication_count(), 4);
    for (before, after) in initial.iter().zip(pipeline.owner_stats().unwrap()) {
        assert_eq!(before.thread, after.thread);
        assert_eq!(before.layers, after.layers);
        for (before, after) in before.experts.iter().zip(after.experts) {
            assert_eq!(before.thread, after.thread);
            assert_eq!(before.owned_experts, after.owned_experts);
            assert!(after.calls > 0 && after.tokens > 0);
        }
    }
    pipeline.release_session(source).unwrap();
    pipeline.release_session(branch).unwrap();
    zero(&pipeline);
    pipeline.shutdown().unwrap();
}

#[test]
#[ignore = "requires two CUDA devices; use an external timeout and --test-threads=1"]
fn cuda_pp2_prefill_decode_cow_and_cancel_use_the_existing_kv_cohort() {
    run(false);
}

#[test]
#[ignore = "requires six CUDA devices; use an external timeout and --test-threads=1"]
fn cuda_pp2_ep2_resident_experts_gpu_combine_and_kv_cohort() {
    run(true);
}

#[test]
fn cuda_expert_factory_rejects_cpu_workers_without_initializing_cuda() {
    let fixture = Fixture::with_top_k(false, 2);
    let path = fixture.0.clone();
    let result = ExpertParallelExecutor::new_cuda(group(0..1), move |rank, group| {
        ExpertRankWorker::prepare_cpu(&load(&path)?, rank, group, 4096)
    });
    let Err(error) = result else {
        panic!("CPU expert owner was accepted")
    };
    assert!(
        error.to_string().contains("CPU fallback is disabled"),
        "{error}"
    );
}
