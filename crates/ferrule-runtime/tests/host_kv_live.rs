//! Logical protocol tests and an opt-in real model/runtime committed-read fixture.
use ferrule_common::execution::{KvElementType, KvLayoutSchema, KvPlaneDescriptor, StateSlot};
use ferrule_runtime::{KvPageManager, KvReservationCommit};

#[derive(Debug)]
struct Schema;
impl KvLayoutSchema for Schema {
    fn planes(&self) -> &[KvPlaneDescriptor] {
        static PLANES: [KvPlaneDescriptor; 2] = [
            KvPlaneDescriptor::new("standard_gqa.key", 2, 1, KvElementType::F32),
            KvPlaneDescriptor::new("standard_gqa.value", 2, 1, KvElementType::F32),
        ];
        &PLANES
    }
    fn page_size(&self) -> usize {
        4
    }
    fn max_sequence_len(&self) -> usize {
        16
    }
}
#[test]
fn subrange_reader_freezes_entire_committed_frontier_until_explicit_release() {
    let mut manager = KvPageManager::new(Box::new(Schema), 4);
    let slot = StateSlot::new(0);
    manager.alloc_sequence(slot, 0).unwrap();
    let reservation = manager.reserve(slot, 0, 7).unwrap();
    let prepared = manager
        .prepare_commit(vec![KvReservationCommit::new(reservation, 7)])
        .unwrap();
    let retired = manager.publish_commit(prepared);
    manager.confirm_page_retirement(retired).unwrap();
    let source = manager.committed_kv_source(slot, 0, 1..2).unwrap();
    let lease = manager.acquire_committed_read(&source).unwrap();
    assert_eq!(lease.source().pages.len(), 2);
    assert!(manager.reserve(slot, 0, 1).is_err());
    assert!(manager.capture_prefix_snapshot(slot, 4).is_err());
    assert!(
        manager
            .prepare_fork_sequence_exact(slot, StateSlot::new(1), 0, 7)
            .is_err()
    );
    assert!(manager.preempt_sequence(slot).is_err());
    assert!(manager.free_sequence_pages(slot).is_err());
    manager.release_committed_read(&lease).unwrap();
    assert!(manager.release_committed_read(&lease).is_err());
    manager.validate_committed_source(&source).unwrap();
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use ferrule_backend::cpu::{
        StreamingF32GqaConfig, StreamingF32GqaRequest, streaming_f32_gqa_partial,
    };
    use ferrule_backend::cuda::context::CudaOperators;
    use ferrule_backend::cuda::operators::kv::{
        committed_gqa_credits, host_placement::CommittedGqaProgress,
    };
    use ferrule_common::execution::*;
    use ferrule_common::host_kv::placement::{PlacementBudget, PlacementCredits};
    use ferrule_model::decoder::*;
    use ferrule_model::execution::ExecutionPrecisionPolicy;
    use ferrule_model::transformer::{
        CommittedGqaRequest, CudaRows, CudaStandardDecoderOperators, Rows, RowsShape,
    };
    use ferrule_runtime::io::host_kv_adapters::LiveCommittedGqa;
    use std::rc::Rc;
    use std::time::{Duration, Instant};
    type State = DecoderSequenceState<(), ()>;

    #[test]
    #[ignore = "real committed decoder KV DMA; requires separately selected idle GPU"]
    fn live_runtime_lease_to_model_readonly_rows_and_cancel_reaper() {
        let ops = Rc::new(CudaOperators::new().unwrap());
        let mut operators =
            CudaStandardDecoderOperators::new(ops.clone(), ExecutionPrecisionPolicy::f32(), &[])
                .unwrap();
        let pool = TypedCudaPagedKvPool::<State>::new(
            ops.clone(),
            KvLayoutSchema::planes(&Schema).iter().copied(),
            4,
            4,
        )
        .unwrap();
        let mut backend = PagedKvBackend::new(pool);
        let mut states = vec![State::new((), ())];
        let mut manager = KvPageManager::new(Box::new(Schema), 4);
        let slot = StateSlot::new(0);
        manager.alloc_sequence(slot, 0).unwrap();
        let reservation = manager.reserve(slot, 0, 3).unwrap();
        let bindings = manager.reservation_bindings(reservation.view()).unwrap();
        let batch = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![1, 2, 3],
            vec![0, 1, 2],
            bindings.write_slots.into_iter().map(Some).collect(),
            vec![LogitsRequest::None; 3],
            vec![ExecutionSequence::new(
                slot,
                ForwardPhase::Prefill,
                0..3,
                0,
                3,
                0..1,
            )],
            bindings.block_ids,
        );
        let capabilities = ExecutionCapabilities {
            max_batch_tokens: 4,
            max_sequences: 1,
            max_prefill_query_tokens_per_sequence: 4,
            max_decode_query_tokens_per_sequence: 1,
            max_top_k: None,
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: false,
            full_logits_width: None,
            kv_binding_mode: KvBindingMode::Paged,
            logits_row_policy: LogitsRowPolicy::None,
        };
        let packed = PackedDecoderBatch::lower(
            &batch,
            &[reservation.view().clone()],
            &states,
            &capabilities,
            4,
            &backend,
        )
        .unwrap();
        let write_custody = DecoderKvSequenceCustody {
            source_index: 0,
            topology_id: states[0].topology_id(),
            page_state_slot: slot,
            page_generation: 0,
            execution_generation: 0,
            context_len: 0,
            query_len: 3,
        };
        let mut tx = backend
            .prepare(DecoderKvPrepare {
                transaction: ExecutionTransactionId::new(1).unwrap(),
                sequences: &[write_custody],
                new_pages: packed.new_pages(),
                writable_pages: packed.writable_pages(),
                cow_replacements: packed.cow_replacements(),
                protected_pages: packed.protected_pages(),
                capacity: backend.capacity(),
                page_statuses: &[],
            })
            .unwrap();
        backend.enter(&mut tx, &packed, &mut states).unwrap();
        let key = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let value = [2.0, 3.0, 4.0, 7.0, 5.0, 11.0];
        backend
            .active_view(&mut tx)
            .unwrap()
            .with_f32_gqa_planes(&ops, 0, &packed, packed.protected_pages(), |p| {
                for row in 0..3 {
                    let offset = p
                        .layout
                        .resolve_row_offset(0, row, p.key.len(), p.block_slots, p.block_offsets)
                        .unwrap();
                    ops.overwrite_f32_range(&key[row * 2..row * 2 + 2], p.key, offset)?;
                    ops.overwrite_f32_range(&value[row * 2..row * 2 + 2], p.value, offset)?;
                }
                Ok(())
            })
            .unwrap();
        backend.leave(&mut tx).unwrap();
        ops.record_compute_event().unwrap().synchronize().unwrap();
        let logical = manager
            .prepare_commit(vec![KvReservationCommit::new(reservation, 3)])
            .unwrap();
        assert_eq!(
            backend.commit(&mut Some(tx)).unwrap(),
            KvEndProgress::Complete
        );
        let step = states[0].core().begin_step().unwrap();
        states[0].core_mut().commit_step(step, 3).unwrap();
        let retired = manager.publish_commit(logical);
        manager.confirm_page_retirement(retired).unwrap();
        let source = manager.committed_kv_source(slot, 0, 0..3).unwrap();
        let custody = DecoderKvSequenceCustody {
            context_len: 3,
            query_len: 0,
            ..write_custody
        };
        let c = StreamingF32GqaConfig {
            rows: 1,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            block_tokens: 2,
            softmax_scale: 0.5,
            sliding_window: Some(2),
        };
        let credits = committed_gqa_credits(c, source.pages.len()).unwrap();
        let budget = PlacementBudget::new(credits);
        let q = [0.5, 1.0, 1.0, -0.5];
        for cancel in [false, true] {
            let request = CommittedGqaRequest {
                query: Rows::Cuda(
                    CudaRows::f32(
                        RowsShape::new(1, 4).unwrap(),
                        None,
                        ops.upload_f32_buffer(&q).unwrap(),
                    )
                    .unwrap(),
                ),
                layer: 0,
                config: c,
                query_position: 2,
                cpu_tokens: 1,
            };
            let mut live = LiveCommittedGqa::begin(
                &mut manager,
                &mut operators,
                &mut backend,
                &mut states[0],
                &custody,
                source.clone(),
                request,
                &budget,
            )
            .unwrap();
            assert!(live.finish().is_err());
            if cancel {
                live.poll().unwrap();
                live.cancel();
                assert!(live.finish().is_err());
                assert_eq!(budget.in_use(), credits);
                live.reap().unwrap();
            } else {
                let deadline = Instant::now() + Duration::from_secs(10);
                while live.poll().unwrap() != CommittedGqaProgress::Ready {
                    assert!(Instant::now() < deadline);
                    std::thread::yield_now();
                }
            }
            let mut output = live.finish().unwrap();
            assert_eq!(output.receipt().canceled(), cancel);
            assert!(live.finish().is_err());
            drop(live);
            if cancel {
                assert!(output.rows().is_none());
            } else {
                let actual = ops
                    .download_f32_range(
                        output.rows().unwrap().cuda().unwrap().f32_buffer().unwrap(),
                        0,
                        4,
                    )
                    .unwrap();
                let expected = streaming_f32_gqa_partial(StreamingF32GqaRequest {
                    config: c,
                    query: &q,
                    key: &key,
                    value: &value,
                    query_start: 2,
                    kv_start: 0,
                })
                .unwrap()
                .output();
                for (a, b) in actual.iter().zip(expected) {
                    assert!((a - b).abs() < 3e-5);
                }
                assert_eq!(budget.in_use(), credits);
            }
            output.release().unwrap();
            assert_eq!(budget.in_use(), PlacementCredits::default());
            manager.validate_committed_source(&source).unwrap();
            assert_eq!(states[0].core().position(), 3);
        }
        let retired = manager.free_sequence_pages(slot).unwrap();
        backend.release(retired.pages()).unwrap();
        manager.confirm_page_retirement(retired).unwrap();
        backend.shutdown().unwrap();
    }
}

#[test]
fn lifecycle_fork_export_requires_actual_cow_not_an_alias_bypass() {
    use ferrule_model::decoder::StandardGqaPlanes;
    fn append(manager: &mut KvPageManager, slot: StateSlot, count: usize) {
        let reservation = manager.reserve(slot, 0, count).unwrap();
        let prepared = manager
            .prepare_commit(vec![KvReservationCommit::new(reservation, count)])
            .unwrap();
        let retired = manager.publish_commit(prepared);
        manager.confirm_page_retirement(retired).unwrap();
    }
    for prefix in [1, 16] {
        let schema = StandardGqaPlanes::new(1, 1, 1, 16, 128, KvElementType::Bf16).unwrap();
        let mut manager = KvPageManager::new(Box::new(schema), 32);
        let parent = StateSlot::new(0);
        let left = StateSlot::new(1);
        let right = StateSlot::new(2);
        manager.alloc_sequence(parent, 0).unwrap();
        append(&mut manager, parent, prefix);
        for child in [left, right] {
            let fork = manager
                .prepare_fork_sequence_exact(parent, child, 0, prefix)
                .unwrap();
            manager.publish_fork_sequence_exact(fork).unwrap();
        }
        assert!(manager.committed_kv_source(parent, 0, 0..prefix).is_err());
        assert!(manager.committed_kv_source(left, 0, 0..prefix).is_err());
        append(&mut manager, left, 1);
        let source = manager.committed_kv_source(left, 0, 0..prefix + 1);
        if prefix == 1 {
            // Canonical Hello fork: the one partial shared page was physically COWed.
            let lease = manager.acquire_committed_read(&source.unwrap()).unwrap();
            manager.validate_committed_read(&lease).unwrap();
            manager.release_committed_read(&lease).unwrap();
        } else {
            // A full prefix page still aliases. Never manufacture its snapshot.
            assert!(source.is_err());
        }
        assert!(manager.committed_kv_source(parent, 0, 0..prefix).is_err());
    }
}
