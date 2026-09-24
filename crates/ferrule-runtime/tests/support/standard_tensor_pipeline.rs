//! Real GPU tests drive only PipelineParallelExecutor. Fixtures are inherited
//! from the draft, not its manual physical-KV orchestration.
use super::*;
use ferrule_common::{ParallelismPlan, Result, ValidatedParallelTopology};
use ferrule_runtime::parallel::pipeline::{
    PipelineConfig, PipelineParallelExecutor, PipelineStage, StandardCudaTensorConfig,
};
use ferrule_runtime::{SessionId, TransactionState};
use std::sync::atomic::AtomicBool;

const PARAMETER_LIMIT: u64 = 1024 * 1024 * 1024;
fn config() -> PipelineConfig {
    PipelineConfig {
        page_size: 2,
        max_pages: 16,
        max_positions: 16,
        max_batch_tokens: 8,
        session_capacity: 3,
        max_parameter_bytes: PARAMETER_LIMIT,
        precision: ExecutionPrecisionPolicy::f32(),
        max_ack_polls: 8,
    }
}
fn pipeline<F>(
    resources: &BoundDecoderResources,
    pp: usize,
    tp: usize,
    load: F,
) -> PipelineParallelExecutor
where
    F: FnOnce(ferrule_runtime::parallel::pipeline::PipelineRank) -> Result<BoundDecoderResources>
        + Clone
        + Send
        + 'static,
{
    let layers = resources.spec().layers().len();
    let plans = (0..pp)
        .map(|stage| {
            LayerSegmentPlan::new(
                layers,
                stage * layers / pp..(stage + 1) * layers / pp,
                stage == 0,
                stage + 1 == pp,
            )
            .unwrap()
        })
        .collect();
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(501),
        (pp * tp) as u32,
        ParallelRankId::new(0),
        ParallelismPlan::validated(1, tp, 1, 1, 1, pp).unwrap(),
    )
    .unwrap();
    let config = config();
    let result = if tp == 1 {
        PipelineParallelExecutor::new_with_program(topology, plans, config, move |rank, plan| {
            let resources = load(rank)?;
            let ops = Rc::new(CudaOperators::new_on_device(rank.local.get() as usize)?);
            PipelineStage::prepare_cuda(&resources, plan, config, ops)
        })
    } else {
        let max_elements = config.max_batch_tokens
            * resources
                .spec()
                .hidden_size()
                .max(resources.spec().vocab_size().div_ceil(tp));
        PipelineParallelExecutor::new_standard_cuda_tensor(
            topology,
            plans,
            config,
            resources.spec(),
            StandardCudaTensorConfig {
                devices: (0..pp * tp).collect(),
                collective_limits: HostCollectiveLimits {
                    max_ranks: tp,
                    max_elements_per_rank: max_elements,
                    max_host_bytes: max_elements * 4 * tp * (tp + 1),
                },
                collective_timeout: Duration::from_secs(30),
            },
            load,
        )
    };
    let pipeline = result.expect("production CUDA pipeline startup");
    let owners = pipeline.owner_stats().unwrap();
    assert_eq!(owners.len(), pp * tp);
    for (index, owner) in owners.iter().enumerate() {
        assert_eq!(owner.rank.global.get() as usize, index);
        assert_eq!(
            owner.layers,
            index / tp * layers / pp..(index / tp + 1) * layers / pp
        );
        assert_ne!(owner.thread, std::thread::current().id());
        assert!(
            owners[..index]
                .iter()
                .all(|peer| peer.thread != owner.thread)
        );
    }
    for description in pipeline.stage_descriptions() {
        let Attention::Gqa(gqa) = resources.spec().layers()[0].attention() else {
            panic!("GQA")
        };
        assert_eq!(description.kv_heads, gqa.num_kv_heads() / tp);
    }
    pipeline
}
fn drained(p: &PipelineParallelExecutor) {
    assert!(!p.is_quarantined());
    assert_eq!(p.outstanding(), 0);
    assert_eq!(p.coordinator().in_use_credits(), 0);
    assert_eq!(p.coordinator().retained_transaction_count(), 0);
    assert_eq!(p.page_manager().stats().retiring_pages, 0);
    for owner in p.owner_stats().unwrap() {
        assert_eq!(owner.kv.active_transactions, 0);
    }
}
fn exercise_pipeline(p: &mut PipelineParallelExecutor) -> Vec<DenseLogits> {
    let source = SessionId(11);
    let branch = SessionId(12);
    let mut outputs = vec![
        p.forward(source, &[1, 2, 3], ForwardPhase::Prefill)
            .unwrap()
            .logits,
    ];
    drained(p);
    p.fork_session(source, branch).unwrap();
    let source_slot = p.session_slot(source).unwrap();
    let branch_slot = p.session_slot(branch).unwrap();
    let before = p
        .page_manager()
        .block_table(source_slot)
        .unwrap()
        .pages()
        .to_vec();
    assert_eq!(
        p.page_manager().block_table(branch_slot).unwrap().pages(),
        before
    );
    let pages = p.page_manager().allocated_pages();
    let publications = p.coordinator().publication_count();
    let cancel = AtomicBool::new(false);
    let owners = p.owner_stats().unwrap().len();
    let mut saw_cohort = false;
    assert!(
        p.forward_observed(branch, &[7], ForwardPhase::Decode, &cancel, |progress| {
            if progress.state == TransactionState::Preparing {
                assert_eq!(progress.pending_ranks.len(), owners);
                assert_eq!(progress.publications, publications);
                saw_cohort = true;
                cancel.store(true, Ordering::Release);
            }
        })
        .is_err()
    );
    assert!(
        saw_cohort,
        "cancel after every physical owner actually executed"
    );
    assert_eq!(p.coordinator().publication_count(), publications);
    assert_eq!(p.page_manager().allocated_pages(), pages);
    assert_eq!(
        p.page_manager()
            .block_table(branch_slot)
            .unwrap()
            .committed_tokens(),
        3
    );
    assert_eq!(
        p.page_manager().block_table(branch_slot).unwrap().pages(),
        before
    );
    drained(p);
    outputs.push(
        p.forward(branch, &[4], ForwardPhase::Decode)
            .unwrap()
            .logits,
    );
    let after = p.page_manager().block_table(branch_slot).unwrap().pages();
    assert_eq!(after[0], before[0]);
    assert_ne!(
        after[1], before[1],
        "partial tail is COW on every local-head owner"
    );
    assert_eq!(
        p.page_manager().block_table(source_slot).unwrap().pages(),
        before
    );
    for token in [5, 6] {
        outputs.push(
            p.forward(source, &[token], ForwardPhase::Decode)
                .unwrap()
                .logits,
        );
        drained(p);
    }
    // Prefill is also allowed to extend an existing prefix.
    outputs.push(
        p.forward(source, &[8, 9], ForwardPhase::Prefill)
            .unwrap()
            .logits,
    );
    drained(p);
    p.release_session(branch).unwrap();
    p.release_session(source).unwrap();
    assert_eq!(p.page_manager().allocated_pages(), 0);
    for owner in p.owner_stats().unwrap() {
        assert_eq!(owner.kv.free_pages, config().max_pages);
        assert_eq!(owner.sessions, 0);
    }
    p.shutdown().unwrap();
    outputs
}
fn compare(label: &str, actual: &[DenseLogits], expected: &[DenseLogits], atol: f32, rtol: f32) {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut max_scaled = 0.0f32;
    for (step, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert_eq!((a.rows(), a.width()), (e.rows(), e.width()));
        for (index, (&a, &e)) in a.values().iter().zip(e.values()).enumerate() {
            let delta = (a - e).abs();
            max_abs = max_abs.max(delta);
            max_scaled = max_scaled.max(delta / (atol + rtol * e.abs()));
            assert!(
                a.is_finite() && e.is_finite() && delta <= atol + rtol * e.abs(),
                "{label} step={step} index={index} actual={a} GPU_TP1={e} delta={delta} tolerance={}",
                atol + rtol * e.abs()
            );
        }
    }
    eprintln!(
        "PASS {label}: ALL logits; max_abs={max_abs:e} max_tolerance_fraction={max_scaled:e}; prefill/decode/fork/COW/cancel/retry/release"
    );
}

fn dense_matrix(topologies: &[(usize, usize)]) {
    for (tied, bf16) in [(true, true), (false, true), (false, false)] {
        let fixture = Fixture::new(true, tied, bf16);
        let load = {
            let resources = fixture.resources.clone();
            move |_| Ok(resources.clone())
        };
        let expected = exercise_pipeline(&mut pipeline(&fixture.resources, 1, 1, load.clone()));
        for &(pp, tp) in topologies {
            let actual = exercise_pipeline(&mut pipeline(&fixture.resources, pp, tp, load.clone()));
            compare(
                &format!("dense PP{pp}TP{tp} tied={tied} BF16storage={bf16}"),
                &actual,
                &expected,
                2e-5,
                2e-4,
            );
        }
    }
}

fn load_nas() -> Result<BoundDecoderResources> {
    let path = std::path::Path::new("/mnt/nas1/hf/Qwen3-0.6B");
    let value = serde_json::from_slice(&std::fs::read(path.join("config.json"))?).map_err(|e| {
        ferrule_common::Error::Execution {
            message: e.to_string(),
        }
    })?;
    let resources = DecoderLoadOptions::new(&Qwen3DenseRecipe::new(), &value)
        .open_hf(path, ModelFamily::Qwen3)?;
    assert_eq!(resources.spec().layers().len(), 28, "no truncation");
    assert_eq!(resources.spec().vocab_size(), 151936);
    assert_eq!(resources.spec().hidden_size(), 1024);
    assert_eq!(resources.state_dict().len(), 3 + 28 * 11);
    assert!(resources.state_dict().validate_source_identities());
    Ok(resources)
}
fn nas_matrix(topologies: &[(usize, usize)]) {
    let resources = load_nas().expect("NAS checkpoint is required, never skip");
    let expected = exercise_pipeline(&mut pipeline(&resources, 1, 1, |_| load_nas()));
    // F32 full-vocabulary comparison against the unsharded GPU kernel path.
    // Small absolute slack near zero, relative slack for reassociated sums.
    for &(pp, tp) in topologies {
        let actual = exercise_pipeline(&mut pipeline(&resources, pp, tp, |_| load_nas()));
        compare(
            &format!("Qwen3-0.6B 28 layers PP{pp}TP{tp}"),
            &actual,
            &expected,
            2e-4,
            2e-4,
        );
    }
}

#[test]
#[ignore = "requires four real CUDA GPUs; --test-threads=1"]
fn production_dense_fixture_tp2_tp4() {
    dense_matrix(&[(1, 2), (1, 4)]);
}
#[test]
#[ignore = "requires four real CUDA GPUs; --test-threads=1"]
fn production_dense_fixture_pp2tp2() {
    dense_matrix(&[(2, 2)]);
}
#[test]
#[ignore = "requires NAS Qwen3-0.6B and four real CUDA GPUs; --test-threads=1"]
fn production_nas_qwen3_28_layers_tp2_tp4() {
    nas_matrix(&[(1, 2), (1, 4)]);
}
#[test]
#[ignore = "requires NAS Qwen3-0.6B and four real CUDA GPUs; --test-threads=1"]
fn production_nas_qwen3_28_layers_pp2tp2() {
    nas_matrix(&[(2, 2)]);
}

#[test]
#[ignore = "requires eight real CUDA GPUs; --test-threads=1"]
fn production_dense_fixture_pp2tp4() {
    dense_matrix(&[(2, 4)]);
}
#[test]
#[ignore = "requires NAS Qwen3-0.6B and eight real CUDA GPUs; --test-threads=1"]
fn production_nas_qwen3_28_layers_pp2tp4() {
    nas_matrix(&[(2, 4)]);
}
