//! Full-logit and Unknown fixtures on the production runner/engine, not test KV.
use super::*;
use ferrule_backend::cuda::context::CudaOperators;
use ferrule_common::execution::*;
use ferrule_model::decoder::*;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::{
    MultiSessionBatchProgress, MultiSessionRunner, TransactionEndIntent, TransactionEndProgress,
    WeightSource,
};
use std::rc::Rc;
type Runner = GenericDecoderRunner<HybridCudaDecoder>;
fn runner(
    reclaim: bool,
) -> (
    host_kv_fixture::Fixture,
    Runner,
    StandardGqaPlanes,
    Rc<CudaOperators>,
) {
    let f = host_kv_fixture::Fixture::new();
    let resources = f.resources();
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Qwen3,
        WeightSource::Safetensors,
        2,
        32,
        8,
        2,
        1 << 20,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap();
    let schema = HybridStateSchema::from_spec(resources.spec(), 2)
        .unwrap()
        .kv_planes(2, 32)
        .unwrap();
    let config = IdleKvReclaimConfig {
        staging_page_limit: 8,
        ..config()
    };
    let estimate = if reclaim {
        HybridCudaMemoryEstimate::for_resources_with_idle_kv_reclaim(
            &resources, &options, 32, config,
        )
    } else {
        HybridCudaMemoryEstimate::for_resources(&resources, &options, 32)
    }
    .unwrap();
    let device = HybridCudaDevice::new_on_device(
        0,
        HybridCudaMemoryBudget {
            kv_pages: 32,
            state_bytes: 1.max(estimate.per_sequence_state_bytes * 8),
            weight_bytes: estimate.weight_bytes_upper_bound,
            workspace_bytes: estimate.workspace_bytes_upper_bound,
        },
    )
    .unwrap();
    let ops = device.operators().clone();
    let runner = if reclaim {
        Runner::hybrid_cuda_with_idle_kv_reclaim(resources, f.tokenizer(), options, device, config)
    } else {
        Runner::hybrid_cuda(resources, f.tokenizer(), options, device)
    }
    .unwrap();
    (f, runner, schema, ops)
}
fn forward(
    e: &mut ferrule_runtime::RunnerMultiSessionExecutor<Runner>,
    ops: &CudaOperators,
    states: &mut [CudaHybridSequenceState],
    manager: &mut ferrule_runtime::KvPageManager,
    sequence: usize,
    tokens: &[u32],
    id: u64,
) -> Vec<Vec<f32>> {
    let slot = StateSlot::new(sequence as u32);
    let r = manager.reserve(slot, 0, tokens.len()).unwrap();
    let start = r.positions.start;
    let b = manager.reservation_bindings(r.view()).unwrap();
    let phase = if start == 0 {
        ForwardPhase::Prefill
    } else {
        ForwardPhase::Decode
    };
    let mode = if start == 0 {
        ForwardMode::Prefill
    } else {
        ForwardMode::Decode
    };
    let batch = ExecutionBatch::new(
        mode,
        tokens.to_vec(),
        (start as u32..(start + tokens.len()) as u32).collect(),
        b.write_slots.into_iter().map(Some).collect(),
        vec![LogitsRequest::Full; tokens.len()],
        vec![ExecutionSequence::new(
            slot,
            phase,
            0..tokens.len() as u32,
            start as u32,
            (start + tokens.len()) as u32,
            0..b.block_ids.len() as u32,
        )],
        b.block_ids,
    );
    let tx = ExecutionTransactionId::new(id).unwrap();
    e.prepare_batch_with_kv(tx, states, &batch, std::slice::from_ref(r.view()))
        .unwrap();
    let MultiSessionBatchProgress::Complete(output) =
        e.execute_prepared_batch(tx, states, &batch).unwrap()
    else {
        panic!("unexpected continuation")
    };
    ops.record_compute_event().unwrap().synchronize().unwrap();
    assert_eq!(
        e.end_transaction(tx, states, TransactionEndIntent::Publish)
            .unwrap(),
        TransactionEndProgress::Complete
    );
    let prepared = manager
        .prepare_commit(vec![ferrule_runtime::KvReservationCommit::new(
            r,
            tokens.len(),
        )])
        .unwrap();
    let retired = manager.publish_commit(prepared);
    manager.confirm_page_retirement(retired).unwrap();
    output
        .logits
        .into_iter()
        .map(|r| match r.logits {
            LogitsOutput::Full(v) => v,
            _ => panic!("full logits required"),
        })
        .collect()
}
fn close(a: &[Vec<f32>], b: &[Vec<f32>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(b) {
            assert!(
                (a - b).abs() < 3e-5 + b.abs() * 5e-4 && a.is_finite() && b.is_finite(),
                "{a} != {b}"
            );
        }
    }
}
#[test]
#[ignore = "CUDA 12.4 idle owner only; compile-only in implementation"]
fn full_logits_two_sequences_reload_and_cow_match_fixed_pool() {
    let (_fixture, r, schema, ops) = runner(true);
    let mut e = ferrule_runtime::RunnerMultiSessionExecutor::new(r);
    let (_reference_fixture, r, rschema, rops) = runner(false);
    let mut reference = ferrule_runtime::RunnerMultiSessionExecutor::new(r);
    let mut manager = ferrule_runtime::KvPageManager::new(Box::new(schema), 32);
    let mut rm = ferrule_runtime::KvPageManager::new(Box::new(rschema), 32);
    let mut states = vec![
        e.create_sequence_state().unwrap(),
        e.create_sequence_state().unwrap(),
    ];
    let mut rs = vec![
        reference.create_sequence_state().unwrap(),
        reference.create_sequence_state().unwrap(),
    ];
    for i in 0..2 {
        manager.alloc_sequence(StateSlot::new(i), 0).unwrap();
        rm.alloc_sequence(StateSlot::new(i), 0).unwrap();
    }
    for i in 0..2 {
        close(
            &forward(
                &mut e,
                &ops,
                &mut states,
                &mut manager,
                i,
                &[1, 3, 2],
                i as u64 + 1,
            ),
            &forward(
                &mut reference,
                &rops,
                &mut rs,
                &mut rm,
                i,
                &[1, 3, 2],
                i as u64 + 1,
            ),
        );
    }
    let source = manager
        .committed_kv_source(StateSlot::new(0), 0, 0..3)
        .unwrap();
    let suspension = manager.preempt_sequence(StateSlot::new(0)).unwrap();
    let replay = suspension.clone();
    let pages = suspension.evicted_pages().to_vec();
    e.preempt_kv_pages(&pages).unwrap();
    let off = e.idle_kv_physical_stats().unwrap();
    assert!(off.host_snapshot_bytes > 0 && off.live_payload_bytes > 0);
    close(
        &forward(&mut e, &ops, &mut states, &mut manager, 1, &[4], 3),
        &forward(&mut reference, &rops, &mut rs, &mut rm, 1, &[4], 3),
    );
    e.restore_kv_pages(&pages).unwrap();
    manager
        .restore_sequence(StateSlot::new(0), suspension)
        .unwrap();
    assert!(manager.restore_sequence(StateSlot::new(2), replay).is_err());
    assert!(manager.validate_committed_source(&source).is_err());
    assert_eq!(
        manager.block_table(StateSlot::new(0)).unwrap().pages(),
        source.pages
    );
    // Pin an actual manager snapshot to force COW of the restored partial tail.
    let snapshot = manager
        .capture_prefix_snapshot(StateSlot::new(0), 3)
        .unwrap();
    let rsnapshot = rm.capture_prefix_snapshot(StateSlot::new(0), 3).unwrap();
    close(
        &forward(&mut e, &ops, &mut states, &mut manager, 0, &[5], 4),
        &forward(&mut reference, &rops, &mut rs, &mut rm, 0, &[5], 4),
    );
    assert_ne!(
        manager.block_table(StateSlot::new(0)).unwrap().pages(),
        source.pages
    );
    assert_eq!(states[0].core().position(), 4);
    // Snapshots remain runtime-owned; dropping observations is never a release ACK.
    assert!(snapshot.page_count() > 0);
    assert!(rsnapshot.page_count() > 0);
}

#[test]
#[ignore = "CUDA 12.4 idle owner only; intentionally retains Unknown custody"]
fn actual_engine_cancel_after_unknown_dma_retains_physical_and_logical_credits() {
    let (_fixture, r, schema, ops) = runner(true);
    let config = r.idle_kv_reclaim_config().unwrap();
    let scheduler = ferrule_runtime::ResidentSchedulerConfig {
        max_active_sequences: 2,
        max_decode_batch: 2,
        prefill_chunk_size: 2,
        max_batch_tokens: 4,
        ..Default::default()
    };
    let driver = ferrule_runtime::ResidentTopKDriverConfig {
        ctx_size: 16,
        enable_native_proposals: false,
        ..Default::default()
    };
    // Match the runner's 32 physical slots: 16 context pages plus 16 shadows.
    let accounting = ferrule_runtime::engine::ResidentKvPageAccounting::TransactionCapacity {
        page_bytes: schema.checked_page_bytes().unwrap() as u64,
        budget_bytes: None,
    };
    assert_eq!(
        ferrule_runtime::engine::plan_resident_kv_pages(&schema, accounting, scheduler, driver)
            .unwrap()
            .configured_pages,
        32
    );
    let mut engine = ferrule_runtime::engine::build_resident_engine(
        r,
        Box::new(schema),
        accounting,
        scheduler,
        driver,
    )
    .unwrap();
    engine
        .configure_kv_execution(KvExecutionPolicy::IdleReclaim(config))
        .unwrap();
    submit(&mut engine, 1, 1, vec![1, 3, 2]);
    engine.step(&mut |_| Ok(())).unwrap();
    ops.record_compute_event().unwrap().synchronize().unwrap();
    // At the first prefill chunk the active session owns real committed pages.
    let before = engine.observability_snapshot().kv_cache.unwrap();
    ops.failpoints().arm_copy_event();
    assert!(engine.suspend_session_kv(SessionId(1)).is_err());
    let unknown = engine.observability_snapshot().kv_cache.unwrap();
    assert!(unknown.idle_physical.unwrap().unknown);
    assert!(unknown.idle_physical.unwrap().pinned_bytes > 0);
    assert_eq!(
        unknown.idle_physical.unwrap().live_payload_bytes,
        before.idle_physical.unwrap().live_payload_bytes
    );
    assert_eq!(unknown.stats.allocated_pages, before.stats.allocated_pages);
    let _ = engine.cancel_request(RequestId(1));
    ops.record_compute_event().unwrap().synchronize().unwrap();
    assert!(engine.restore_session_kv(SessionId(1)).is_err());
    assert!(engine.trim_idle_kv().is_err());
    assert!(engine.shutdown().is_err());
    let retained = engine.observability_snapshot().kv_cache.unwrap();
    assert!(retained.idle_physical.unwrap().unknown);
    assert_eq!(
        retained.idle_physical.unwrap().admission,
        unknown.idle_physical.unwrap().admission
    );
    assert_eq!(
        retained.stats.allocated_pages,
        unknown.stats.allocated_pages
    );
}
