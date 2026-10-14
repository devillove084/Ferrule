//! Model-only segment arithmetic/lifecycle tests, not an alternate PP runtime.
#[path = "hybrid_cpu.rs"]
mod reference;
use ferrule_common::execution::*;
use ferrule_model::decoder::*;
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::transformer::*;
use ferrule_model::{ModelFamily, WeightSource};
use reference::{Fixture, close};

type Backend = PagedKvBackend<TypedCpuPagedKvPool<HybridDecoderSequenceState>>;
fn tx(id: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(id).unwrap()
}
fn options(resources: &BoundDecoderResources) -> GenericDecoderOptions {
    GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Unknown("hybrid".into()),
        WeightSource::Safetensors,
        2,
        32,
        32,
        4,
        1 << 20,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap()
}
fn prepare(
    resources: &BoundDecoderResources,
    range: std::ops::Range<usize>,
) -> HybridDecoderSegment {
    HybridDecoderSegment::prepare(
        resources,
        LayerSegmentPlan::new(4, range.clone(), range.start == 0, range.end == 4).unwrap(),
        ExecutionPrecisionPolicy::f32(),
        32,
        1 << 20,
    )
    .unwrap()
}
struct Stage {
    segment: HybridDecoderSegment,
    lifecycle: HybridSequenceLifecycle,
    backend: Backend,
    states: Vec<HybridDecoderSequenceState>,
    pages: Vec<KvPageId>,
    capabilities: ExecutionCapabilities,
}
struct Working {
    batch: PackedDecoderBatch,
    handle: PagedKvTransactionHandle,
    view: CpuKvView,
    states: Vec<HybridDecoderSequenceState>,
    pages: Vec<KvPageId>,
}
impl Stage {
    fn new(resources: &BoundDecoderResources, range: std::ops::Range<usize>) -> Self {
        let segment = prepare(resources, range);
        let mut lifecycle = HybridSequenceLifecycle::new(segment.schema().clone());
        let backend = PagedKvBackend::new(
            TypedCpuPagedKvPool::from_strategy(&segment.schema().kv_planes(2, 32).unwrap(), 32)
                .unwrap(),
        );
        let states = vec![lifecycle.create().unwrap()];
        Self {
            segment,
            lifecycle,
            backend,
            states,
            pages: vec![],
            capabilities: options(resources).capabilities(),
        }
    }
    fn enter(&mut self, id: u64, tokens: &[u32]) -> Working {
        self.enter_with_cow(id, tokens, false)
    }
    fn enter_with_cow(&mut self, id: u64, tokens: &[u32], cow: bool) -> Working {
        let context = self.states[0].core().position();
        let end = context + tokens.len();
        let mut pages = self.pages.clone();
        let mut new_pages = vec![];
        let cow_replacement = if cow && !context.is_multiple_of(2) {
            let logical_page = context / 2;
            let source = pages[logical_page];
            let replacement = KvPageId(id as u32 * 32 + 31);
            pages[logical_page] = replacement;
            Some(KvCowReplacement {
                logical_page,
                source,
                replacement,
            })
        } else {
            None
        };
        while pages.len() < end.div_ceil(2) {
            let page = KvPageId(id as u32 * 32 + pages.len() as u32);
            pages.push(page);
            new_pages.push(page);
        }
        let phase = if context == 0 {
            ForwardPhase::Prefill
        } else {
            ForwardPhase::Decode
        };
        let batch = ExecutionBatch::new(
            if context == 0 {
                ForwardMode::Prefill
            } else {
                ForwardMode::Decode
            },
            tokens.to_vec(),
            (context..end).map(|p| p as u32).collect(),
            (context..end)
                .map(|p| Some(KvWriteSlot::new(pages[p / 2].0 * 2 + (p % 2) as u32)))
                .collect(),
            vec![LogitsRequest::Full; tokens.len()],
            vec![ExecutionSequence::new(
                StateSlot::new(0),
                phase,
                0..tokens.len() as u32,
                context as u32,
                end as u32,
                0..pages.len() as u32,
            )],
            pages.iter().map(|p| KvBlockId::new(p.0)).collect(),
        );
        let reservations = [KvReservationView {
            state_slot: StateSlot::new(0),
            execution_state_slot: StateSlot::new(0),
            positions: context..end,
            newly_allocated: new_pages,
            generation: self.states[0].core().generation(),
            execution_generation: self.states[0].core().generation(),
            cow_replacement,
        }];
        let packed = PackedDecoderBatch::lower(
            &batch,
            &reservations,
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
                self.lifecycle
                    .checkout(DecoderSequenceCheckout::new(tx(id), i, i), s)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let custody = packed
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
        let statuses = packed
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
                new_pages: packed.new_pages(),
                writable_pages: packed.writable_pages(),
                cow_replacements: packed.cow_replacements(),
                protected_pages: packed.protected_pages(),
                capacity: self.backend.capacity(),
                page_statuses: &statuses,
            })
            .unwrap();
        self.backend
            .enter(&mut handle, &packed, &mut states)
            .unwrap();
        let view = self.backend.active_view(&mut handle).unwrap();
        Working {
            batch: packed,
            handle,
            view,
            states,
            pages,
        }
    }
    fn execute(&self, working: &mut Working, input: SegmentInput) -> SegmentOutput {
        self.segment
            .execute(
                &working.batch,
                input,
                &mut working.states,
                &mut working.view,
            )
            .unwrap()
    }
    fn finish(&mut self, mut work: Working, publish: bool) {
        self.backend.leave(&mut work.handle).unwrap();
        if publish {
            self.segment
                .validate_frontier(&work.batch, &work.states)
                .unwrap();
            let binding = work.states[0].core().begin_step().unwrap();
            work.states[0]
                .core_mut()
                .commit_step(binding, work.batch.len())
                .unwrap();
            assert_eq!(
                self.backend.commit(&mut Some(work.handle)).unwrap(),
                KvEndProgress::Complete
            );
            std::mem::swap(&mut self.states, &mut work.states);
            self.pages = work.pages;
        } else {
            assert_eq!(
                self.backend.rollback(&mut Some(work.handle)).unwrap(),
                KvEndProgress::Complete
            );
        }
        for state in work.states {
            self.lifecycle.try_release(state).unwrap();
        }
    }
}
fn transfer(output: SegmentOutput) -> SegmentInput {
    match output {
        SegmentOutput::Hidden { next_layer, rows } => SegmentInput::Hidden { next_layer, rows },
        _ => panic!("expected hidden"),
    }
}
fn logits(output: SegmentOutput) -> Vec<f32> {
    match output {
        SegmentOutput::Logits(rows) => rows.values().to_vec(),
        _ => panic!("expected logits"),
    }
}
fn compare_states(stages: &[Stage; 2], full: &HybridDecoderSequenceState) {
    for stage in stages {
        assert_eq!(stage.states[0].core().position(), full.core().position());
        for entry in stage.segment.schema().entries() {
            if let HybridLayerState::GatedDeltaNet(actual) =
                &stage.states[0].kv_state().layers()[entry.state_slot]
            {
                let HybridLayerState::GatedDeltaNet(expected) =
                    &full.kv_state().layers()[entry.global_layer]
                else {
                    panic!("kind")
                };
                assert_eq!(actual.position(), expected.position());
                close(
                    actual.conv_history(),
                    expected.conv_history(),
                    "segment conv",
                );
                close(
                    actual.recurrent(),
                    expected.recurrent(),
                    "segment recurrent",
                );
            }
        }
    }
}
#[test]
fn pp2_prefill_continuation_decode_match_full_runner_and_oracle() {
    let fixture = Fixture::new();
    let resources = fixture.resources();
    let mut stages = [Stage::new(&resources, 0..2), Stage::new(&resources, 2..4)];
    let mut runner = fixture.runner();
    use ferrule_model::runner::MultiSessionRunner;
    let mut full = vec![runner.create_sequence_state().unwrap()];
    let mut pages = vec![vec![]];
    for (i, tokens) in [&[1, 3][..], &[2][..], &[5][..], &[7][..]]
        .into_iter()
        .enumerate()
    {
        let id = i as u64 + 1;
        let expected = reference::step(
            &mut runner,
            &mut full,
            &mut pages,
            &[(
                tokens,
                if i == 0 {
                    ForwardPhase::Prefill
                } else {
                    ForwardPhase::Decode
                },
            )],
            id,
            true,
        )
        .concat();
        let mut a = stages[0].enter(id, tokens);
        let hidden = stages[0].execute(&mut a, SegmentInput::Tokens);
        let mut b = stages[1].enter(id, tokens);
        let actual = logits(stages[1].execute(&mut b, transfer(hidden)));
        close(&actual, &expected, "PP2 logits");
        assert_eq!(
            stages[0].states[0].core().position(),
            a.states[0].core().position()
        );
        stages[0].finish(a, true);
        stages[1].finish(b, true);
        compare_states(&stages, &full[0]);
    }
}
#[test]
fn partial_gdn_cancel_dirty_retry_rejection_abort_and_fresh_checkout() {
    let fixture = Fixture::new();
    let resources = fixture.resources();
    let mut stage = Stage::new(&resources, 0..2);
    let source = stage.states.clone();
    let mut work = stage.enter(1, &[1, 3]);
    let mut checks = 0;
    let error = stage
        .segment
        .execute_checked(
            &work.batch,
            SegmentInput::Tokens,
            &mut work.states,
            &mut work.view,
            &mut |_| {
                checks += 1;
                if checks == 3 {
                    Err(ferrule_common::Error::Execution {
                        message: "cancel after GDN".into(),
                    })
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("cancel after GDN"));
    assert_eq!(
        work.states[0]
            .kv_state_mut()
            .linear_mut(0)
            .unwrap()
            .position(),
        2
    );
    assert!(
        stage
            .segment
            .validate_execution(&work.batch, &work.states)
            .is_err()
    );
    assert_eq!(stage.states, source);
    stage.finish(work, false);
    assert_eq!(stage.states, source);
    let mut retry = stage.enter(2, &[1, 3]);
    stage.execute(&mut retry, SegmentInput::Tokens);
    stage.finish(retry, true);
    assert_eq!(stage.states[0].core().position(), 2);
}
#[test]
fn segment_input_schema_boundary_and_precision_fail_before_mutation() {
    let fixture = Fixture::new();
    let resources = fixture.resources();
    let mut stage = Stage::new(&resources, 2..4);
    let mut work = stage.enter(1, &[1]);
    let before = work.states.clone();
    let input = SegmentInput::Hidden {
        next_layer: 0,
        rows: HostRows::new(
            RowsShape::new(1, 8).unwrap(),
            RowsDType::F32,
            None,
            vec![0.; 8],
        )
        .unwrap(),
    };
    assert!(matches!(
        stage
            .segment
            .execute(&work.batch, input, &mut work.states, &mut work.view),
        Err(SegmentError::HiddenBoundary { .. })
    ));
    assert_eq!(work.states, before);
    assert!(
        prepare(&resources, 0..2)
            .validate_execution(&work.batch, &work.states)
            .is_err()
    );
    assert!(
        stage
            .segment
            .validate_frontier(&work.batch, &work.states)
            .is_err()
    );
    stage.finish(work, false);
    assert!(
        HybridDecoderSegment::prepare(
            &resources,
            LayerSegmentPlan::new(4, 0..2, true, false).unwrap(),
            ExecutionPrecisionPolicy::bf16_compatibility(),
            32,
            1 << 20
        )
        .is_err()
    );
}

fn tied_fixture() -> (Fixture, BoundDecoderResources) {
    let original = Fixture::new();
    let mut oracle = original.oracle.clone();
    oracle["tensors"]
        .as_object_mut()
        .unwrap()
        .remove("output.weight");
    let fixture = Fixture::with_oracle(oracle);
    let spec = reference::spec();
    let spec = DecoderModelSpec::new(DecoderModelParts {
        architecture: "tied-hybrid".into(),
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
    let resources = fixture.resources_for(spec);
    (fixture, resources)
}

#[test]
fn tied_output_stage_retains_canonical_embedding_without_embedding_operation() {
    let (_fixture, resources) = tied_fixture();
    let embedding = resources
        .state_dict()
        .parameters()
        .iter()
        .find(|p| p.role() == &ferrule_model::TensorRole::TokenEmbedding)
        .unwrap();
    let mut first = Stage::new(&resources, 0..2);
    let mut last = Stage::new(&resources, 2..4);
    assert!(!last.segment.plan().owns_embedding());
    assert!(
        last.segment
            .parameters()
            .iter()
            .all(|p| p.role() != &ferrule_model::TensorRole::TokenEmbedding)
    );
    let head = last
        .segment
        .parameters()
        .iter()
        .find(|p| p.role() == &ferrule_model::TensorRole::OutputHead)
        .unwrap();
    assert_eq!(head.canonical_id(), embedding.canonical_id());
    assert!(head.shares_storage_with(embedding));
    let mut unsplit = Stage::new(&resources, 0..4);
    for (id, tokens) in [(1, &[1, 3][..]), (2, &[2][..]), (3, &[5][..])] {
        let mut a = first.enter(id, tokens);
        let mut b = last.enter(id, tokens);
        let hidden = first.execute(&mut a, SegmentInput::Tokens);
        let actual = logits(last.execute(&mut b, transfer(hidden)));
        let mut full = unsplit.enter(id, tokens);
        let expected = logits(unsplit.execute(&mut full, SegmentInput::Tokens));
        close(&actual, &expected, "tied split vs unsplit");
        first.finish(a, true);
        last.finish(b, true);
        unsplit.finish(full, true);
    }
}

#[test]
fn segment_metadata_budget_is_range_endpoint_and_tied_dependency_aware() {
    let (_fixture, resources) = tied_fixture();
    let options = options(&resources);
    let first = LayerSegmentPlan::new(4, 0..2, true, false).unwrap();
    let last = LayerSegmentPlan::new(4, 2..4, false, true).unwrap();
    let a = HybridMemoryPlan::for_segment(&resources, &options, &first, 32).unwrap();
    let b = HybridMemoryPlan::for_segment(&resources, &options, &last, 32).unwrap();
    let whole = HybridMemoryPlan::for_resources(&resources, &options, 32).unwrap();
    assert_eq!(
        a.per_sequence_state_bytes + b.per_sequence_state_bytes,
        whole.per_sequence_state_bytes
    );
    assert_eq!(a.kv_bytes + b.kv_bytes, whole.kv_bytes);
    assert_eq!(
        a.weight_bytes_upper_bound + b.weight_bytes_upper_bound,
        whole.weight_bytes_upper_bound
    );
    assert_eq!(a.per_sequence_state_bytes, 168);
    assert_eq!(a.kv_bytes, 32 * 2 * 8 * 2 * 4);
    // Independent metadata arithmetic includes selected F32 vector+linear
    // representation (8 bytes/element upper bound), plus one local RoPE table.
    for (plan, estimate) in [(&first, a), (&last, b)] {
        let selected = prepare(&resources, plan.layers());
        let parameter_bytes: usize = selected
            .parameters()
            .iter()
            .map(|p| p.weight().slice().shape.iter().product::<usize>() * 8)
            .sum();
        assert_eq!(
            estimate.weight_bytes_upper_bound,
            parameter_bytes + 32 * 4 * 4
        );
    }
    let no_head = LayerSegmentPlan::new(4, 2..4, false, false).unwrap();
    let c = HybridMemoryPlan::for_segment(&resources, &options, &no_head, 32).unwrap();
    assert_eq!(
        b.weight_bytes_upper_bound - c.weight_bytes_upper_bound,
        (11 * 8 + 8) * 8
    );
    assert!(HybridMemoryPlan::for_segment(&resources, &options, &last, usize::MAX).is_err());
    assert!(HybridMemoryPlan::for_segment(&resources, &options, &last, 0).is_err());
    assert!(
        HybridMemoryPlan::for_segment(
            &resources,
            &options,
            &LayerSegmentPlan::new(4, 0..1, true, false).unwrap(),
            32
        )
        .is_err()
    );
}

#[test]
fn segment_exact_fork_cow_abort_preserves_parent_and_release_reclaims_pages() {
    let fixture = Fixture::new();
    let resources = fixture.resources();
    let mut stage = Stage::new(&resources, 0..4);
    let mut prefill = stage.enter(1, &[1, 3, 2]);
    stage.execute(&mut prefill, SegmentInput::Tokens);
    stage.finish(prefill, true);
    let parent_before = stage.states[0].clone();
    let parent_pages = stage.pages.clone();
    assert!(stage.lifecycle.logical_fork(&stage.states[0], 2).is_err());
    let child = stage.lifecycle.logical_fork(&stage.states[0], 3).unwrap();
    let parent = std::mem::replace(&mut stage.states[0], child);
    let mut aborted = stage.enter_with_cow(2, &[9], true);
    stage.execute(&mut aborted, SegmentInput::Tokens);
    stage.finish(aborted, false);
    assert_eq!(stage.states[0].kv_state(), parent.kv_state());
    let mut child = stage.enter_with_cow(3, &[5], true);
    let actual = logits(stage.execute(&mut child, SegmentInput::Tokens));
    close(
        &actual,
        &reference::floats(&fixture.oracle["logits"][3]),
        "fork child",
    );
    stage.finish(child, true);
    assert_eq!(parent, parent_before);
    let child_pages = stage.pages.clone();
    let child = std::mem::replace(&mut stage.states[0], parent);
    stage.pages = parent_pages;
    let mut parent_work = stage.enter(4, &[7]);
    stage.execute(&mut parent_work, SegmentInput::Tokens);
    stage.finish(parent_work, true);
    assert_ne!(stage.states[0].kv_state(), child.kv_state());
    stage.lifecycle.try_release(child).unwrap();
    let pages = stage
        .pages
        .iter()
        .chain(&child_pages)
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    stage
        .backend
        .release(&pages.into_iter().collect::<Vec<_>>())
        .unwrap();
    for state in stage.states.drain(..) {
        stage.lifecycle.try_release(state).unwrap();
    }
    assert_eq!(stage.backend.capacity().active_transactions, 0);
    assert_eq!(
        stage.backend.capacity().free_pages,
        stage.backend.capacity().physical_pages
    );
}
