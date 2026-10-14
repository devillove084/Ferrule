//! Test-only local physical custody. No participants, rank remapping, vote or
//! global publication authority. Local commit merely seeds the next math step.
use ferrule_common::execution::*;
use ferrule_model::{
    ModelFamily, WeightSource, decoder::*, execution::ExecutionPrecisionPolicy, transformer::*,
};

pub fn tx(id: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(id).unwrap()
}
pub struct LocalTransaction<S: DecoderSequence> {
    pub backend: PagedKvBackend<TypedCpuPagedKvPool<S>>,
    pub states: Vec<S>,
    pages: Vec<KvPageId>,
    capabilities: ExecutionCapabilities,
}
pub struct Work<S> {
    pub batch: PackedDecoderBatch,
    pub view: CpuKvView,
    pub states: Vec<S>,
    handle: PagedKvTransactionHandle,
    pages: Vec<KvPageId>,
}
impl<S: DecoderSequence> LocalTransaction<S> {
    pub fn new(
        resources: &BoundDecoderResources,
        planes: StandardGqaPlanes,
        states: Vec<S>,
    ) -> Self {
        Self {
            backend: PagedKvBackend::new(TypedCpuPagedKvPool::from_strategy(&planes, 32).unwrap()),
            states,
            pages: vec![],
            capabilities: GenericDecoderOptions::standard_cpu(
                resources.spec(),
                ModelFamily::Unknown("hybrid-contract".into()),
                WeightSource::Safetensors,
                2,
                32,
                32,
                4,
                1 << 20,
                ExecutionPrecisionPolicy::f32(),
            )
            .unwrap()
            .capabilities(),
        }
    }
    pub fn enter(
        &mut self,
        lifecycle: &mut impl DecoderSequenceLifecycle<S>,
        id: u64,
        tokens: &[u32],
    ) -> Work<S> {
        assert_eq!(self.states.len(), 1);
        let context = self.states[0].core().position();
        let end = context + tokens.len();
        let pages = (0..end.div_ceil(2))
            .map(|p| KvPageId(p as u32))
            .collect::<Vec<_>>();
        let source = ExecutionBatch::new(
            if context == 0 {
                ForwardMode::Prefill
            } else {
                ForwardMode::Decode
            },
            tokens.to_vec(),
            (context..end).map(|p| p as u32).collect(),
            (context..end)
                .map(|p| Some(KvWriteSlot::new(p as u32)))
                .collect(),
            vec![LogitsRequest::Full; tokens.len()],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                if context == 0 {
                    ForwardPhase::Prefill
                } else {
                    ForwardPhase::Decode
                },
                0..tokens.len() as u32,
                context as u32,
                end as u32,
                0..pages.len() as u32,
            )],
            pages.iter().map(|p| KvBlockId::new(p.0)).collect(),
        );
        let reservations = [KvReservationView {
            state_slot: StateSlot::new(10),
            execution_state_slot: StateSlot::new(0),
            positions: context..end,
            newly_allocated: pages[self.pages.len()..].to_vec(),
            generation: 0,
            execution_generation: self.states[0].core().generation(),
            cow_replacement: None,
        }];
        self.enter_batch(lifecycle, id, &source, &reservations, pages)
    }
    pub fn enter_batch(
        &mut self,
        lifecycle: &mut impl DecoderSequenceLifecycle<S>,
        id: u64,
        source: &ExecutionBatch,
        reservations: &[KvReservationView],
        pages: Vec<KvPageId>,
    ) -> Work<S> {
        let batch = PackedDecoderBatch::lower(
            source,
            reservations,
            &self.states,
            &self.capabilities,
            2,
            &|p| DecoderKvPageView::page_status(&self.backend, p),
        )
        .unwrap();
        let mut states = self
            .states
            .iter()
            .enumerate()
            .map(|(i, s)| {
                lifecycle
                    .checkout(DecoderSequenceCheckout::new(tx(id), i, i), s)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let custody = batch
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
            .collect::<Vec<_>>();
        let statuses = batch
            .protected_pages()
            .iter()
            .map(|&page| DecoderKvPageSnapshot {
                page,
                status: DecoderKvPageView::page_status(&self.backend, page),
            })
            .collect::<Vec<_>>();
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
                page_statuses: &statuses,
            })
            .unwrap();
        self.backend
            .enter(&mut handle, &batch, &mut states)
            .unwrap();
        let view = self.backend.active_view(&mut handle).unwrap();
        Work {
            batch,
            view,
            states,
            handle,
            pages,
        }
    }
    pub fn commit_local(
        &mut self,
        lifecycle: &mut impl DecoderSequenceLifecycle<S>,
        mut work: Work<S>,
    ) {
        self.backend.leave(&mut work.handle).unwrap();
        for (state, seq) in work.states.iter_mut().zip(work.batch.sequences()) {
            let binding = state.core().begin_step().unwrap();
            state
                .core_mut()
                .commit_step(binding, seq.query_len())
                .unwrap();
        }
        assert_eq!(
            self.backend.commit(&mut Some(work.handle)).unwrap(),
            KvEndProgress::Complete
        );
        std::mem::swap(&mut self.states, &mut work.states);
        self.pages = work.pages;
        for s in work.states {
            assert!(lifecycle.try_release(s).is_ok());
        }
    }
    pub fn rollback_local(
        &mut self,
        lifecycle: &mut impl DecoderSequenceLifecycle<S>,
        mut work: Work<S>,
    ) {
        self.backend.leave(&mut work.handle).unwrap();
        assert_eq!(
            self.backend.rollback(&mut Some(work.handle)).unwrap(),
            KvEndProgress::Complete
        );
        for s in work.states {
            assert!(lifecycle.try_release(s).is_ok());
        }
    }
    pub fn release(&mut self, lifecycle: &mut impl DecoderSequenceLifecycle<S>) {
        self.backend.release(&self.pages).unwrap();
        self.pages.clear();
        for s in self.states.drain(..) {
            assert!(lifecycle.try_release(s).is_ok());
        }
        let c = self.backend.capacity();
        assert_eq!(c.active_transactions, 0);
        assert_eq!(c.free_pages, c.physical_pages);
    }
}
