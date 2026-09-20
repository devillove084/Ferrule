#![cfg(feature = "cuda")]
//! KV-only GPU tests: no model, checkpoint, pipeline, or CLI execution.
use ferrule_backend::cuda::context::CudaOperators;
use ferrule_common::execution::*;
use ferrule_common::{
    ParallelRankId, ParallelTopologyId, ParallelismPlan, ParticipantSet, ValidatedParallelTopology,
};
use ferrule_model::decoder::{
    CudaPagedKvBackend, CudaPagedKvPool, DecoderKvBackend, DecoderKvCommitBackend,
    DecoderKvPageSnapshot, DecoderKvPrepare, DecoderKvSequenceCustody, GenericDecoderSequenceState,
    KvCommitBinding, KvCommitOwner, KvEndProgress, PackedDecoderBatch, PagedKvBackend,
    PagedKvTransactionHandle, PhysicalKvPool, PreparedKvCommit, StandardGqaPlanes,
};
use std::num::NonZeroU32;
use std::rc::Rc;

fn tx(n: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(n).unwrap()
}
fn rank(n: u32) -> ParallelRankId {
    ParallelRankId::new(n)
}
fn binding() -> KvCommitBinding {
    let topology = ValidatedParallelTopology::new(
        ParallelTopologyId::new(1),
        2,
        rank(0),
        ParallelismPlan::validated(1, 2, 1, 1, 1, 1).unwrap(),
    )
    .unwrap();
    KvCommitBinding::new(
        tx(2),
        topology.topology_id(),
        ParticipantSet::new(&topology, [rank(0), rank(1)]).unwrap(),
        7,
    )
    .unwrap()
}
struct Owner {
    ops: Rc<CudaOperators>,
    backend: CudaPagedKvBackend,
    states: Vec<GenericDecoderSequenceState>,
    handle: Option<PagedKvTransactionHandle>,
    layer: usize,
}
impl Owner {
    fn new(layers: usize) -> Self {
        let ops = Rc::new(CudaOperators::new().unwrap());
        let schema = StandardGqaPlanes::new(layers, 1, 1, 4, 16, KvElementType::F32).unwrap();
        let pool = CudaPagedKvPool::from_strategy(ops.clone(), &schema, 6).unwrap();
        Self {
            ops,
            backend: PagedKvBackend::new(pool),
            states: vec![GenericDecoderSequenceState::new((), ())],
            handle: None,
            layer: layers - 1,
        }
    }
    fn execute(&mut self, id: u64, page: u32, new: bool, value: Option<f32>) -> f32 {
        let context = self.states[0].core().position();
        let position = u32::try_from(context).unwrap();
        let public = ExecutionBatch::new(
            ForwardMode::Prefill,
            vec![17],
            vec![position],
            vec![Some(KvWriteSlot::new(page * 4 + position % 4))],
            vec![LogitsRequest::None],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                ForwardPhase::Prefill,
                0..1,
                position,
                position + 1,
                0..1,
            )],
            vec![KvBlockId::new(page)],
        );
        let reservations = [KvReservationView {
            state_slot: StateSlot::new(3),
            execution_state_slot: StateSlot::new(0),
            positions: context..context + 1,
            newly_allocated: if new { vec![KvPageId(page)] } else { vec![] },
            generation: 7,
            execution_generation: self.states[0].core().generation(),
            cow_replacement: None,
        }];
        let caps = ExecutionCapabilities {
            max_batch_tokens: 8,
            max_sequences: 2,
            max_prefill_query_tokens_per_sequence: 4,
            max_decode_query_tokens_per_sequence: 4,
            max_top_k: NonZeroU32::new(4),
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: true,
            full_logits_width: NonZeroU32::new(4),
            kv_binding_mode: KvBindingMode::Paged,
            logits_row_policy: LogitsRowPolicy::Any,
        };
        let batch = PackedDecoderBatch::lower(
            &public,
            &reservations,
            &self.states,
            &caps,
            4,
            &self.backend,
        )
        .unwrap();
        let custody: Vec<_> = batch
            .sequences()
            .iter()
            .map(|s| DecoderKvSequenceCustody {
                source_index: s.state_index(),
                topology_id: s.topology_id(),
                page_state_slot: s.page_state_slot(),
                page_generation: s.page_generation(),
                execution_generation: s.execution_generation(),
                context_len: s.context_len(),
                query_len: s.query_len(),
            })
            .collect();
        let snapshots: Vec<_> = batch
            .protected_pages()
            .iter()
            .map(|p| DecoderKvPageSnapshot {
                page: *p,
                status: self.backend.page_status(*p),
            })
            .collect();
        let mut handle = self
            .backend
            .prepare(DecoderKvPrepare {
                transaction: tx(id),
                sequences: &custody,
                new_pages: batch.new_pages(),
                writable_pages: batch.writable_pages(),
                cow_replacements: batch.cow_replacements(),
                protected_pages: batch.protected_pages(),
                capacity: self.backend.capacity(),
                page_statuses: &snapshots,
            })
            .unwrap();
        self.backend
            .enter(&mut handle, &batch, &mut self.states.clone())
            .unwrap();
        let mut view = self.backend.active_view(&mut handle).unwrap();
        // The explicit read-set spelling is the standard GPU operator interface.
        assert!(
            view.with_f32_gqa_planes_mut(&self.ops, self.layer, &batch, &[], |_| ())
                .is_err()
        );
        let read = view
            .with_f32_gqa_planes_mut(
                &self.ops,
                self.layer,
                &batch,
                batch.protected_pages(),
                |planes| {
                    assert_eq!(planes.key_layout.layer_count, self.layer + 1);
                    let offset = planes
                        .key_layout
                        .resolve_row_offset(
                            0,
                            0,
                            planes.key.len(),
                            planes.block_slots,
                            planes.block_offsets,
                        )
                        .unwrap();
                    if let Some(value) = value {
                        let write_offset = planes
                            .key_layout
                            .resolve_row_offset(
                                0,
                                context,
                                planes.key.len(),
                                planes.block_slots,
                                planes.block_offsets,
                            )
                            .unwrap();
                        self.ops
                            .overwrite_f32_range(&[value], planes.key, write_offset)
                            .unwrap();
                        self.ops
                            .overwrite_f32_range(&[value + 10.0], planes.value, write_offset)
                            .unwrap();
                    }
                    // Probe committed token zero even when the new query starts later.
                    self.ops.download_f32_range(planes.key, offset, 1).unwrap()[0]
                },
            )
            .unwrap();
        self.backend.leave(&mut handle).unwrap();
        assert!(view.validate_batch(&batch).is_err());
        self.ops
            .record_compute_event()
            .unwrap()
            .synchronize()
            .unwrap();
        self.handle = Some(handle);
        read
    }
    fn owner(&mut self, rank: u32) -> KvCommitOwner<'_, CudaPagedKvBackend> {
        KvCommitOwner::new(
            ParallelRankId::new(rank),
            &mut self.backend,
            &mut self.handle,
            &self.states,
        )
    }
}

#[test]
#[ignore = "requires CUDA device and native providers"]
fn same_logical_ids_local_planes_share_the_existing_sealed_cohort_protocol() {
    let mut a = Owner::new(1);
    let mut b = Owner::new(3);
    // Make the physical slot mapping owner-local as well as the layer count.
    b.execute(1, 4, true, Some(1.0));
    b.backend.commit(&mut b.handle).unwrap();
    a.execute(2, 7, true, Some(3.0));
    b.execute(2, 7, true, Some(9.0));
    let mut a_clone = a.backend.pool().clone();
    let mut b_clone = b.backend.pool().clone();
    let scope = binding();
    assert!(
        a.backend
            .preflight_commit_ready(b.handle.as_ref().unwrap(), &scope, rank(0), &a.states)
            .is_err()
    );
    assert!(
        b.backend
            .preflight_commit_ready(a.handle.as_ref().unwrap(), &scope, rank(1), &b.states)
            .is_err()
    );
    let err =
        PreparedKvCommit::prepare_commit_ready(scope.clone(), (), vec![a.owner(0), b.owner(0)])
            .err()
            .unwrap();
    drop(err.into_parts());
    assert!(a.handle.is_some() && b.handle.is_some());
    let mut ready =
        PreparedKvCommit::prepare_commit_ready(scope.clone(), (), vec![a.owner(0), b.owner(1)])
            .unwrap();
    ready.install_commit(&scope, rank(0)).unwrap();
    assert!(ready.install_commit(&scope, rank(0)).is_err());
    assert!(ready.publish_logical(&scope, |()| ((), vec![])).is_err());
    assert!(a_clone.release(&[KvPageId(7)]).is_err());
    assert!(b_clone.preempt(&[KvPageId(7)]).is_err());
    assert!(a_clone.shutdown().is_err());
    ready.install_commit(&scope, rank(1)).unwrap();
    let mut retired = ready.publish_logical(&scope, |()| ((), vec![])).unwrap();
    drop(ready);
    retired.retire_owner(&scope, rank(0)).unwrap();
    assert!(retired.finish_retirement(&scope).is_err());
    retired.retire_owner(&scope, rank(1)).unwrap();
    retired.finish_retirement(&scope).unwrap();
    drop(retired);
    // This KV-only fixture supplies the logical sequence publication normally
    // performed by the decoder. Physical KV commit alone does not advance it.
    for owner in [&mut a, &mut b] {
        let step = owner.states[0].core().begin_step().unwrap();
        owner.states[0].core_mut().commit_step(step, 1).unwrap();
        assert_eq!(owner.states[0].core().position(), 1);
    }
    assert_eq!(a.execute(9, 7, false, None), 3.0);
    assert_eq!(b.execute(9, 7, false, None), 9.0);
    a.backend.rollback(&mut a.handle).unwrap();
    b.backend.rollback(&mut b.handle).unwrap();
    a.backend.release(&[KvPageId(7)]).unwrap();
    b.backend.release(&[KvPageId(4), KvPageId(7)]).unwrap();
}

#[test]
#[ignore = "requires CUDA device and native providers"]
fn abort_ready_retains_shared_pins_until_every_owner_finishes() {
    let mut a = Owner::new(1);
    let mut b = Owner::new(2);
    a.execute(2, 7, true, Some(3.0));
    b.execute(2, 7, true, Some(9.0));
    let mut clone = a.backend.pool().clone();
    let free = clone.free_slot_count();
    let scope = binding();
    let mut ready =
        PreparedKvCommit::prepare_commit_ready(scope.clone(), (), vec![a.owner(0), b.owner(1)])
            .unwrap();
    assert_eq!(
        ready.abort_prepared(&scope, rank(0)).unwrap().progress(),
        KvEndProgress::Complete
    );
    assert!(ready.abort_logical(&scope, |()| ()).is_err());
    assert_eq!(clone.free_slot_count(), free);
    assert!(clone.shutdown().is_err());
    ready.abort_prepared(&scope, rank(1)).unwrap();
    ready.abort_logical(&scope, |()| ()).unwrap();
    drop(ready);
    assert_eq!(clone.free_slot_count(), free + 1);
    a.backend.shutdown().unwrap();
    b.backend.shutdown().unwrap();
}
