//! Observation over the real owner pool/lifecycle; never a parent protocol driver.
use super::*;
use ferrule_common::execution::{
    ExecutionBatch, ExecutionSequence, ForwardMode, KvBlockId, KvReservationView, KvWriteSlot,
    LogitsRequest,
};
use ferrule_model::decoder::{
    DecoderKvCapacity, DecoderKvPageStatus, DecoderKvPrepare, DecoderSequenceCheckout,
    DecoderSequenceLifecycle, GenericDecoderSequenceState, PhysicalKvPool,
};
use ferrule_model::runner::SequenceStateReleaseError;

pub(super) type Pool = TypedCpuPagedKvPool<HybridTensorSequenceState>;
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PhysicalSnapshot {
    pub(super) slots: Vec<(KvPageId, usize)>,
    planes: Vec<Vec<Vec<u32>>>,
    pub(super) free: usize,
}
pub(super) type ReleasedState = (usize, BTreeMap<usize, GatedDeltaNetState>);

pub(super) fn physical_snapshot(
    pool: &mut Pool,
    description: &PipelineStageDescription,
    context: usize,
) -> PhysicalSnapshot {
    assert_eq!(pool.active_transaction_count(), 0);
    let slots = (0..description.config.max_pages)
        .filter_map(|p| {
            let page = KvPageId(p as u32);
            pool.physical_slot(page).map(|slot| (page, slot))
        })
        .collect::<Vec<_>>();
    let free = pool.free_slot_count();
    if slots.is_empty() {
        return PhysicalSnapshot {
            slots,
            planes: vec![],
            free,
        };
    }
    assert_eq!(slots.len(), context.div_ceil(description.config.page_size));
    // Lower the last row of the last EXISTING page. For a half page this is its
    // unused tail, for a full page a historical row. No model executes this row,
    // no page is allocated and no bytes are written. The view only snapshots.
    let position = slots.len() * description.config.page_size - 1;
    let batch = ExecutionBatch::new(
        ForwardMode::Decode,
        vec![0],
        vec![position as u32],
        vec![Some(KvWriteSlot::new(position as u32))],
        vec![LogitsRequest::None],
        vec![ExecutionSequence::new(
            StateSlot::new(0),
            ForwardPhase::Decode,
            0..1,
            position as u32,
            position as u32 + 1,
            0..slots.len() as u32,
        )],
        slots
            .iter()
            .map(|(page, _)| KvBlockId::new(page.0))
            .collect(),
    );
    let reservation = KvReservationView {
        state_slot: StateSlot::new(0),
        execution_state_slot: StateSlot::new(0),
        positions: position..position + 1,
        newly_allocated: vec![],
        generation: 0,
        execution_generation: 0,
        cow_replacement: None,
    };
    let state = GenericDecoderSequenceState::with_position(position, (), ());
    let packed = PackedDecoderBatch::lower(
        &batch,
        &[reservation],
        &[state],
        &description.execution_capabilities().unwrap(),
        description.config.page_size,
        &|_| DecoderKvPageStatus::Resident,
    )
    .unwrap();
    assert!(packed.new_pages().is_empty());
    let mut read = Some(
        pool.prepare(DecoderKvPrepare {
            transaction: ExecutionTransactionId::new(1_000_000).unwrap(),
            sequences: &[],
            new_pages: &[],
            writable_pages: packed.writable_pages(),
            cow_replacements: &[],
            protected_pages: packed.protected_pages(),
            capacity: DecoderKvCapacity::default(),
            page_statuses: &[],
        })
        .unwrap(),
    );
    pool.enter(read.as_mut().unwrap(), &packed, &mut [])
        .unwrap();
    let view = pool.active_view(read.as_mut().unwrap(), &packed).unwrap();
    let planes = slots
        .iter()
        .map(|(page, _)| {
            (0..pool.planes().len())
                .map(|plane| {
                    view.with_f32_page(*page, plane, |v| v.iter().map(|n| n.to_bits()).collect())
                        .unwrap()
                })
                .collect()
        })
        .collect();
    pool.leave(read.as_mut().unwrap()).unwrap();
    assert_eq!(pool.abort(&mut read).unwrap(), KvEndProgress::Complete);
    assert!(read.is_none());
    assert_eq!(pool.active_transaction_count(), 0);
    assert_eq!(pool.free_slot_count(), free);
    for (page, slot) in &slots {
        assert_eq!(pool.physical_slot(*page), Some(*slot));
    }
    PhysicalSnapshot {
        slots,
        planes,
        free,
    }
}
pub(super) struct Lifecycle {
    pub(super) inner: HybridTensorSequenceLifecycle,
    pub(super) rank: u32,
    pub(super) evidence: Arc<Evidence>,
}
impl DecoderSequenceLifecycle<HybridTensorSequenceState> for Lifecycle {
    fn create(&mut self) -> Result<HybridTensorSequenceState> {
        self.inner.create()
    }
    fn checkout(
        &mut self,
        request: DecoderSequenceCheckout,
        source: &HybridTensorSequenceState,
    ) -> Result<HybridTensorSequenceState> {
        self.inner.checkout(request, source)
    }
    fn logical_fork(
        &mut self,
        source: &HybridTensorSequenceState,
        position: usize,
    ) -> Result<HybridTensorSequenceState> {
        self.inner.logical_fork(source, position)
    }
    fn reset(&mut self, state: &mut HybridTensorSequenceState) -> Result<()> {
        self.inner.reset(state)
    }
    fn try_release(
        &mut self,
        state: HybridTensorSequenceState,
    ) -> std::result::Result<(), SequenceStateReleaseError<HybridTensorSequenceState>> {
        let snapshot = (state.core().position(), state.kv_state().recurrent());
        self.inner.try_release(state)?;
        self.evidence
            .released
            .lock()
            .unwrap()
            .entry(self.rank)
            .or_default()
            .push(snapshot);
        Ok(())
    }
}
