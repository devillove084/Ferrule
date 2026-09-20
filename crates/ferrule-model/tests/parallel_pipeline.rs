//! Real checkpoint-backed PP=2, using the same standard math as the full runner.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::time::Duration;

use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionOutput, ExecutionSequence,
    ExecutionTransactionId, ForwardMode, ForwardPhase, KvBlockId, KvElementType, KvPageId,
    KvReservationView, KvWriteSlot, LogitsOutput, LogitsRequest, StateSlot,
};
use ferrule_common::{Error, ParallelRankId, Result};
use ferrule_model::checkpoint::{CheckpointDType, CheckpointTensorSlice};
use ferrule_model::decoder::{
    CpuKvView, CpuPagedKvBackend, CpuPagedKvPool, DecoderKvBackend, DecoderKvPageSnapshot,
    DecoderKvPageStatus, DecoderKvPrepare, DecoderKvSequenceCustody, GenericDecoderOptions,
    GenericDecoderRunner, GenericDecoderSequenceState, KvEndProgress, PackedDecoderBatch,
    PagedKvBackend, PagedKvTransactionHandle, StandardGqaPlanes,
};
use ferrule_model::execution::ExecutionPrecisionPolicy;
use ferrule_model::moe::ExpertId;
use ferrule_model::nn::ParameterResidency;
use ferrule_model::runner::{
    MultiSessionBatchProgress, MultiSessionRunner, TransactionEndIntent, TransactionEndProgress,
};
use ferrule_model::transformer::expert_parallel::{
    CpuExpertResultExecutor, CpuReferenceExpertWorker, ExpertDispatchLimits,
    ExpertParallelRoutedExecutor, ExpertPlacement, ExpertResult, ExpertResultExecutor,
    ExpertTokenBucket, RoutedSwiGluExecutor, RoutedSwiGluRequest,
};
use ferrule_model::transformer::{
    BoundDecoderResources, BoundParameter, DecoderLoadOptions, DecoderRecipe, ExpertAvailability,
    ExpertProvider, HostRows, KvHistory, KvView, LayerSegmentPlan, OperatorProgress,
    PreparedLinear, PreparedSwiGlu, Rows, RowsDType, RowsShape, SegmentError, SegmentInput,
    SegmentOutput, SegmentStage, StandardDecoderSegment, StateDictMaterializer,
    SyntheticDecoderRecipe,
};
use ferrule_model::{ModelFamily, TensorRole, TokenizerHandle, WeightSource};

const PAGE_SIZE: usize = 2;
const MAX_POSITIONS: usize = 16;
const READ_LIMIT: u64 = 4096;

struct Fixture {
    directory: PathBuf,
    resources: BoundDecoderResources,
}

impl Fixture {
    fn new(tied: bool) -> Self {
        Self::with_top_k(tied, 1)
    }

    fn with_top_k(tied: bool, top_k: usize) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let directory = std::env::temp_dir().join(format!(
            "ferrule-pipeline-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::default());
        tokenizer
            .add_tokens(
                ["a", "b", "c", "d", "e", "f", "g", "h"]
                    .into_iter()
                    .map(|token| tokenizers::AddedToken::from(token, false)),
            )
            .unwrap();
        tokenizer
            .save(directory.join("tokenizer.json"), false)
            .unwrap();
        std::fs::write(directory.join("config.json"), r#"{"eos_token_id":7}"#).unwrap();
        let config = serde_json::json!({
            "vocab_size": 8, "hidden_size": 4,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 2,
            "intermediate_size": 4, "num_experts": 2, "experts_per_token": top_k,
            "max_position_embeddings": MAX_POSITIONS, "rms_norm_eps": 0.00001,
            "rope_theta": 10000.0, "tie_word_embeddings": tied
        });
        let recipe = SyntheticDecoderRecipe::new();
        let output = recipe.build(&config).unwrap();
        let slices = output
            .schema()
            .parameters()
            .iter()
            .filter(|parameter| parameter.alias_of().is_none())
            .map(|parameter| {
                // Non-collinear token rows and distinct layer/expert matrices make
                // skipped/repeated layers and missing KV history numerically visible.
                let seed = parameter
                    .path()
                    .as_str()
                    .bytes()
                    .fold(0usize, |sum, byte| (sum * 31 + usize::from(byte)) % 251);
                let values = (0..parameter.shape().iter().product::<usize>())
                    .map(|index| {
                        if parameter.shape().len() == 1 {
                            0.9 + index as f32 * 0.03
                        } else {
                            ((seed + index * 17 + index * index * 3) % 41) as f32 * 0.012 - 0.24
                        }
                    })
                    .collect::<Vec<_>>();
                let payload = values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>();
                let path = directory.join(format!("{}.bin", parameter.path()));
                std::fs::write(&path, &payload).unwrap();
                CheckpointTensorSlice {
                    name: SyntheticDecoderRecipe::external_name(parameter.path()),
                    role: TensorRole::Unknown,
                    path,
                    offset: 0,
                    bytes: payload.len() as u64,
                    dtype: CheckpointDType::F32,
                    shape: parameter.shape().to_vec(),
                }
            })
            .collect::<Vec<_>>();
        let resources = DecoderLoadOptions::new(&recipe, &config)
            .bind_slices(slices)
            .unwrap();
        assert_eq!(resources.spec().layers().len(), 2);
        Self {
            directory,
            resources,
        }
    }

    fn segment(
        &self,
        plan: LayerSegmentPlan,
        precision: ExecutionPrecisionPolicy,
    ) -> StandardDecoderSegment {
        StandardDecoderSegment::prepare(&self.resources, plan, precision, MAX_POSITIONS, READ_LIMIT)
            .unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn plans() -> [LayerSegmentPlan; 2] {
    [
        LayerSegmentPlan::new(2, 0..1, true, false).unwrap(),
        LayerSegmentPlan::new(2, 1..2, false, true).unwrap(),
    ]
}

fn options(
    resources: &BoundDecoderResources,
    precision: ExecutionPrecisionPolicy,
) -> GenericDecoderOptions {
    GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Unknown("synthetic".into()),
        WeightSource::Safetensors,
        PAGE_SIZE,
        MAX_POSITIONS,
        MAX_POSITIONS,
        2,
        READ_LIMIT,
        precision,
    )
    .unwrap()
}

fn backend(layers: usize, precision: ExecutionPrecisionPolicy) -> CpuPagedKvBackend {
    let dtype = if precision == ExecutionPrecisionPolicy::bf16_compatibility() {
        KvElementType::Bf16
    } else {
        KvElementType::F32
    };
    let planes = StandardGqaPlanes::new(layers, 1, 2, PAGE_SIZE, MAX_POSITIONS, dtype).unwrap();
    PagedKvBackend::new(CpuPagedKvPool::from_strategy(&planes, MAX_POSITIONS / PAGE_SIZE).unwrap())
}

fn oracle(
    fixture: &Fixture,
    precision: ExecutionPrecisionPolicy,
) -> GenericDecoderRunner<CpuPagedKvBackend> {
    GenericDecoderRunner::new(
        fixture.resources.clone(),
        TokenizerHandle::load(&fixture.directory).unwrap(),
        backend(2, precision),
        options(&fixture.resources, precision),
    )
    .unwrap()
}

fn tx(id: u64) -> ExecutionTransactionId {
    ExecutionTransactionId::new(id).unwrap()
}

fn batch(
    state: &GenericDecoderSequenceState,
    tokens: &[u32],
    page_base: u32,
) -> (ExecutionBatch, Vec<KvReservationView>) {
    cohort_batch(std::slice::from_ref(state), &[tokens], page_base)
}

fn cohort_batch(
    states: &[GenericDecoderSequenceState],
    tokens: &[&[u32]],
    page_base: u32,
) -> (ExecutionBatch, Vec<KvReservationView>) {
    assert_eq!(states.len(), tokens.len());
    let prefill = states.iter().all(|state| state.core().position() == 0);
    let (mode, phase) = if prefill {
        (ForwardMode::Prefill, ForwardPhase::Prefill)
    } else {
        (ForwardMode::Decode, ForwardPhase::Decode)
    };
    let mut token_ids = Vec::new();
    let mut positions = Vec::new();
    let mut slots = Vec::new();
    let mut sequences = Vec::new();
    let mut blocks = Vec::new();
    let mut reservations = Vec::new();
    for (index, (state, tokens)) in states.iter().zip(tokens).enumerate() {
        let start = state.core().position();
        let end = start + tokens.len();
        let pages = (0..end.div_ceil(PAGE_SIZE))
            .map(|page| KvPageId(page_base + index as u32 * 10 + page as u32))
            .collect::<Vec<_>>();
        let slot = StateSlot::new(index as u32);
        let query_start = token_ids.len() as u32;
        let block_start = blocks.len() as u32;
        token_ids.extend_from_slice(tokens);
        positions.extend((start..end).map(|p| p as u32));
        slots.extend((start..end).map(|p| {
            Some(KvWriteSlot::new(
                pages[p / PAGE_SIZE].0 * PAGE_SIZE as u32 + (p % PAGE_SIZE) as u32,
            ))
        }));
        blocks.extend(pages.iter().map(|page| KvBlockId::new(page.0)));
        sequences.push(ExecutionSequence::new(
            slot,
            phase,
            query_start..token_ids.len() as u32,
            start as u32,
            end as u32,
            block_start..blocks.len() as u32,
        ));
        reservations.push(KvReservationView {
            state_slot: slot,
            execution_state_slot: slot,
            positions: start..end,
            newly_allocated: pages[start.div_ceil(PAGE_SIZE)..].to_vec(),
            generation: state.core().generation(),
            execution_generation: state.core().generation(),
            cow_replacement: None,
        });
    }
    let logits = vec![LogitsRequest::Full; token_ids.len()];
    (
        ExecutionBatch::new(mode, token_ids, positions, slots, logits, sequences, blocks),
        reservations,
    )
}

fn execute_oracle(
    runner: &mut GenericDecoderRunner<CpuPagedKvBackend>,
    states: &mut [GenericDecoderSequenceState],
    id: u64,
    tokens: &[u32],
) -> ExecutionOutput {
    execute_oracle_cohort(runner, states, id, &[tokens])
}

fn execute_oracle_cohort(
    runner: &mut GenericDecoderRunner<CpuPagedKvBackend>,
    states: &mut [GenericDecoderSequenceState],
    id: u64,
    tokens: &[&[u32]],
) -> ExecutionOutput {
    let (batch, reservations) = cohort_batch(states, tokens, 100);
    runner
        .prepare_multi_session_batch(tx(id), states, &batch, &reservations)
        .unwrap();
    let MultiSessionBatchProgress::Complete(output) = runner
        .execute_multi_session_batch_progress(tx(id), states, &batch)
        .unwrap()
    else {
        panic!("CPU oracle suspended")
    };
    assert_eq!(
        runner
            .end_transaction(tx(id), states, TransactionEndIntent::Publish)
            .unwrap(),
        TransactionEndProgress::Complete
    );
    output
}

struct StageKv {
    backend: CpuPagedKvBackend,
    state: GenericDecoderSequenceState,
    capabilities: ExecutionCapabilities,
    page_base: u32,
}

impl StageKv {
    fn new(fixture: &Fixture, precision: ExecutionPrecisionPolicy, page_base: u32) -> Self {
        Self {
            backend: backend(1, precision),
            state: GenericDecoderSequenceState::new((), ()),
            capabilities: options(&fixture.resources, precision).capabilities(),
            page_base,
        }
    }

    fn enter(
        &mut self,
        id: u64,
        tokens: &[u32],
    ) -> (PackedDecoderBatch, PagedKvTransactionHandle, CpuKvView) {
        enter_cohort(
            &mut self.backend,
            std::slice::from_mut(&mut self.state),
            self.capabilities,
            tx(id),
            &[tokens],
            self.page_base,
        )
    }

    fn finish(&mut self, transaction: PagedKvTransactionHandle, rows: usize, publish: bool) {
        finish_cohort(
            &mut self.backend,
            std::slice::from_mut(&mut self.state),
            transaction,
            &[rows],
            publish,
        );
    }
}

fn enter_cohort(
    backend: &mut CpuPagedKvBackend,
    states: &mut [GenericDecoderSequenceState],
    capabilities: ExecutionCapabilities,
    transaction: ExecutionTransactionId,
    tokens: &[&[u32]],
    page_base: u32,
) -> (PackedDecoderBatch, PagedKvTransactionHandle, CpuKvView) {
    let (batch, reservations) = cohort_batch(states, tokens, page_base);
    let packed = PackedDecoderBatch::lower(
        &batch,
        &reservations,
        states,
        &capabilities,
        PAGE_SIZE,
        &|page| backend.page_status(page),
    )
    .unwrap();
    let custody = packed
        .sequences()
        .iter()
        .map(|sequence| DecoderKvSequenceCustody {
            source_index: sequence.state_index(),
            topology_id: sequence.topology_id(),
            page_state_slot: sequence.page_state_slot(),
            page_generation: sequence.page_generation(),
            execution_generation: sequence.execution_generation(),
            context_len: sequence.context_len(),
            query_len: sequence.query_len(),
        })
        .collect::<Vec<_>>();
    let statuses = packed
        .protected_pages()
        .iter()
        .map(|&page| DecoderKvPageSnapshot {
            page,
            status: backend.page_status(page),
        })
        .collect::<Vec<_>>();
    let mut transaction = backend
        .prepare(DecoderKvPrepare {
            transaction,
            sequences: &custody,
            new_pages: packed.new_pages(),
            writable_pages: packed.writable_pages(),
            cow_replacements: packed.cow_replacements(),
            protected_pages: packed.protected_pages(),
            capacity: backend.capacity(),
            page_statuses: &statuses,
        })
        .unwrap();
    backend.enter(&mut transaction, &packed, states).unwrap();
    let view = backend.active_view(&mut transaction).unwrap();
    (packed, transaction, view)
}

fn finish_cohort(
    backend: &mut CpuPagedKvBackend,
    states: &mut [GenericDecoderSequenceState],
    mut transaction: PagedKvTransactionHandle,
    rows: &[usize],
    publish: bool,
) {
    assert_eq!(states.len(), rows.len());
    let bindings = states
        .iter()
        .map(|state| state.core().begin_step().unwrap())
        .collect::<Vec<_>>();
    backend.leave(&mut transaction).unwrap();
    let mut transaction = Some(transaction);
    if publish {
        assert_eq!(
            backend.commit(&mut transaction).unwrap(),
            KvEndProgress::Complete
        );
        for ((state, binding), &rows) in states.iter_mut().zip(bindings).zip(rows) {
            state.core_mut().commit_step(binding, rows).unwrap();
        }
    } else {
        assert_eq!(
            backend.rollback(&mut transaction).unwrap(),
            KvEndProgress::Complete
        );
    }
    assert!(transaction.is_none());
}

fn transfer(output: SegmentOutput) -> SegmentInput {
    let SegmentOutput::Hidden { next_layer, rows } = output else {
        panic!("non-final stage returned logits")
    };
    let shape = rows.shape();
    let dtype = rows.dtype();
    // Actual host Vec ownership roundtrip; no module or hidden-state handle crosses.
    let wire: Vec<f32> = rows.into_values();
    let received = HostRows::new(shape, dtype, None, wire).unwrap();
    SegmentInput::Hidden {
        next_layer,
        rows: received,
    }
}

fn assert_values(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite() && expected.is_finite());
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "element {index}: {actual} != {expected}"
        );
    }
}

#[test]
fn pp2_prefill_and_decode_match_full_runner_elementwise() {
    for tied in [false, true] {
        for precision in [
            ExecutionPrecisionPolicy::f32(),
            ExecutionPrecisionPolicy::bf16_compatibility(),
        ] {
            let fixture = Fixture::new(tied);
            let plans = plans();
            LayerSegmentPlan::validate_pipeline(&plans).unwrap();
            let segments = plans.map(|plan| fixture.segment(plan, precision));
            let mut kvs = [
                StageKv::new(&fixture, precision, 10),
                StageKv::new(&fixture, precision, 20),
            ];
            let mut runner = oracle(&fixture, precision);
            let mut states = vec![runner.create_sequence_state().unwrap()];
            let mut histories: [Option<KvHistory>; 2] = [None, None];
            let mut final_values = Vec::new();
            for (step, tokens) in [vec![1, 2], vec![3], vec![4], vec![5]].iter().enumerate() {
                let expected = execute_oracle(&mut runner, &mut states, step as u64 + 1, tokens);
                let mut input = SegmentInput::Tokens;
                for (index, (segment, kv)) in segments.iter().zip(kvs.iter_mut()).enumerate() {
                    let before = kv.state.core().position();
                    let (packed, transaction, mut view) = kv.enter(step as u64 + 1, tokens);
                    assert_eq!(
                        packed.positions(),
                        &(before..before + tokens.len()).collect::<Vec<_>>()
                    );
                    assert_eq!(
                        packed.sequences()[0].block_table()[0],
                        KvPageId(kv.page_base)
                    );
                    assert!(view.physical_slot(KvPageId(kv.page_base)).is_some());
                    let output = segment.execute(&packed, input, &mut view).unwrap();
                    let history =
                        KvView::history(&view, 0, 0, before + tokens.len() - 1, 1, 2).unwrap();
                    assert_eq!(history.tokens, before + tokens.len());
                    assert!(
                        KvView::history(&view, 1, 0, before + tokens.len() - 1, 1, 2).is_err(),
                        "each stage must have only one local KV layer"
                    );
                    if let Some(previous) = &histories[index] {
                        assert_values(&history.key[..previous.key.len()], &previous.key);
                        assert_values(&history.value[..previous.value.len()], &previous.value);
                    }
                    histories[index] = Some(history);
                    assert_eq!(
                        kv.state.core().position(),
                        before,
                        "segment must not publish a cursor"
                    );
                    if index == 0 {
                        input = transfer(output);
                    } else {
                        let SegmentOutput::Logits(actual) = output else {
                            panic!("final stage returned hidden")
                        };
                        assert_eq!(actual.rows(), tokens.len());
                        assert_eq!(expected.logits.len(), tokens.len());
                        assert_eq!(actual.width(), 8);
                        for (row, expected) in expected.logits.iter().enumerate() {
                            let LogitsOutput::Full(expected) = &expected.logits else {
                                panic!("oracle did not return full logits")
                            };
                            assert_values(actual.row(row).unwrap(), expected);
                        }
                        final_values = actual.row(tokens.len() - 1).unwrap().to_vec();
                        input = SegmentInput::Tokens;
                    }
                    drop(view);
                    kv.finish(transaction, tokens.len(), true);
                    assert_eq!(kv.state.core().position(), states[0].core().position());
                    assert_eq!(
                        kv.backend.capacity().resident_pages,
                        states[0].core().position().div_ceil(PAGE_SIZE)
                    );
                }
            }
            // A fresh full causal prefill is an independent continuation oracle.
            let mut fresh = oracle(&fixture, precision);
            let mut fresh_states = vec![fresh.create_sequence_state().unwrap()];
            let full = execute_oracle(&mut fresh, &mut fresh_states, 20, &[1, 2, 3, 4, 5]);
            let LogitsOutput::Full(last) = &full.logits.last().unwrap().logits else {
                panic!("missing full logits")
            };
            assert_values(&final_values, last);
        }
    }
}

mod capability_preflight {
    use super::*;
    use ferrule_model::transformer::{
        Attention, CpuGqaMoeModule, DecoderLayer, DecoderModelParts, DecoderModelSpec, FeedForward,
        GqaAttention, HyperConnectionHeadSpec, Linear, Moe, MoeRouterSpec, RotaryEmbedding,
        RotaryScaling, RouterScoreFunction, RouterSelection, UnsupportedOperator,
    };

    fn parts(spec: &DecoderModelSpec) -> DecoderModelParts {
        DecoderModelParts {
            architecture: spec.architecture().into(),
            hidden_size: spec.hidden_size(),
            vocab_size: spec.vocab_size(),
            max_sequence_length: spec.max_sequence_length(),
            token_embedding: spec.token_embedding().clone(),
            layers: spec.layers().to_vec(),
            final_norm: spec.final_norm().clone(),
            output: spec.output().clone(),
            tie_word_embeddings: spec.tie_word_embeddings(),
        }
    }

    fn router(
        score: RouterScoreFunction,
        selection: RouterSelection,
        normalize: bool,
        bias: bool,
    ) -> MoeRouterSpec {
        MoeRouterSpec::new(2, 1, score, selection, normalize, 1.0)
            .unwrap()
            .with_selection_bias(bias)
    }

    fn yarn(attention_factor: Option<f32>) -> RotaryScaling {
        RotaryScaling::YaRN {
            factor: 2.0,
            original_max_position_embeddings: MAX_POSITIONS,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor,
        }
    }

    // Change the second layer to prove preflight happens before earlier weights
    // are read, not merely when preparation eventually reaches the bad layer.
    fn layer_spec(
        spec: &DecoderModelSpec,
        scaling: Option<RotaryScaling>,
        policy: Option<MoeRouterSpec>,
    ) -> DecoderModelSpec {
        let mut parts = parts(spec);
        let layer = &spec.layers()[1];
        let attention = if let Some(scaling) = scaling {
            let Attention::Gqa(gqa) = layer.attention() else {
                panic!("GQA fixture")
            };
            let rotary = gqa.rotary();
            Attention::Gqa(
                GqaAttention::new(
                    spec.hidden_size(),
                    gqa.num_heads(),
                    gqa.num_kv_heads(),
                    gqa.head_dim(),
                    false,
                    RotaryEmbedding::new(
                        rotary.head_dim(),
                        rotary.theta(),
                        rotary.pairing(),
                        rotary.region(),
                        scaling,
                    )
                    .unwrap(),
                )
                .unwrap(),
            )
        } else {
            layer.attention().clone()
        };
        let feed_forward = policy.map_or_else(
            || layer.feed_forward().clone(),
            |policy| FeedForward::Moe(Moe::new(spec.hidden_size(), 4, policy, false).unwrap()),
        );
        parts.layers[1] = DecoderLayer::new(
            layer.index(),
            layer.input_norm().clone(),
            attention,
            layer.attention_residual().clone(),
            layer.post_attention_norm().clone(),
            feed_forward,
            layer.feed_forward_residual().clone(),
        )
        .unwrap();
        DecoderModelSpec::new(parts).unwrap()
    }

    fn rebind(fixture: &Fixture, spec: DecoderModelSpec) -> BoundDecoderResources {
        BoundDecoderResources::new(spec, Arc::new(fixture.resources.state_dict().clone())).unwrap()
    }

    fn assert_unsupported(error: Error, operator: &str, reason: &str) {
        let Error::ModelSource { source } = error else {
            panic!("expected typed model source, got {error:?}")
        };
        let unsupported = source.downcast::<UnsupportedOperator>().unwrap();
        assert_eq!(unsupported.operator, operator);
        assert!(unsupported.reason.contains(reason), "{unsupported}");
    }

    #[test]
    fn full_prefix_and_segment_prepare_reject_before_any_weight_read() {
        let fixture = Fixture::new(false);
        let base = fixture.resources.spec();
        let mut biased = parts(base);
        biased.output = Linear::new(base.hidden_size(), base.vocab_size(), true).unwrap();
        let hyper = base
            .clone()
            .with_output_hyper_connection(
                HyperConnectionHeadSpec::new(2, base.hidden_size(), 0.00001).unwrap(),
            )
            .unwrap();
        let mut cases = vec![
            (
                DecoderModelSpec::new(biased).unwrap(),
                "lm_head",
                "output bias",
            ),
            (hyper, "output", "output_hyper_connection"),
        ];
        for factor in [0.5, 1.1, f32::from_bits(1.0f32.to_bits() + 1)] {
            cases.push((
                layer_spec(base, Some(yarn(Some(factor))), None),
                "rope",
                "attention_factor",
            ));
        }
        for (score, selection, normalize, bias) in [
            (
                RouterScoreFunction::Sigmoid,
                RouterSelection::TopK,
                true,
                false,
            ),
            (
                RouterScoreFunction::SqrtSoftplus,
                RouterSelection::TopK,
                true,
                false,
            ),
            (
                RouterScoreFunction::Softmax,
                RouterSelection::GroupLimitedTopK {
                    groups: 2,
                    selected_groups: 1,
                },
                true,
                false,
            ),
            (
                RouterScoreFunction::Softmax,
                RouterSelection::HashAssistedTopK { hash_layers: 1 },
                true,
                false,
            ),
            (
                RouterScoreFunction::Softmax,
                RouterSelection::TopK,
                false,
                false,
            ),
            (
                RouterScoreFunction::Softmax,
                RouterSelection::TopK,
                true,
                true,
            ),
        ] {
            cases.push((
                layer_spec(base, None, Some(router(score, selection, normalize, bias))),
                "router",
                "standard router requires",
            ));
        }
        let cases = cases
            .into_iter()
            .map(|(spec, operator, reason)| (rebind(&fixture, spec), operator, reason))
            .collect::<Vec<_>>();
        for path in fixture
            .resources
            .state_dict()
            .parameters()
            .iter()
            .map(|parameter| parameter.weight().slice().path.clone())
            .collect::<BTreeSet<_>>()
        {
            std::fs::remove_file(path).unwrap();
        }
        for (resources, operator, reason) in cases {
            let resources = Arc::new(resources);
            for precision in [
                ExecutionPrecisionPolicy::f32(),
                ExecutionPrecisionPolicy::bf16_compatibility(),
            ] {
                assert_unsupported(
                    CpuGqaMoeModule::prepare(
                        Arc::clone(&resources),
                        Arc::new(StateDictMaterializer::new(READ_LIMIT).unwrap()),
                        precision,
                        MAX_POSITIONS,
                    )
                    .unwrap_err(),
                    operator,
                    reason,
                );
                assert_unsupported(
                    CpuGqaMoeModule::prepare_prefix(
                        Arc::clone(&resources),
                        Arc::new(StateDictMaterializer::new(READ_LIMIT).unwrap()),
                        precision,
                        MAX_POSITIONS,
                        1,
                    )
                    .unwrap_err(),
                    operator,
                    reason,
                );
                let mut segment_plans = vec![
                    LayerSegmentPlan::new(2, 0..2, true, true).unwrap(),
                    plans()[1].clone(),
                ];
                if operator == "lm_head" || operator == "output" {
                    segment_plans.push(plans()[0].clone());
                }
                for plan in segment_plans {
                    let error = StandardDecoderSegment::prepare(
                        &resources,
                        plan,
                        precision,
                        MAX_POSITIONS,
                        READ_LIMIT,
                    )
                    .unwrap_err();
                    let SegmentError::Execution {
                        stage: SegmentStage::Prepare,
                        source,
                    } = error
                    else {
                        panic!("expected preparation rejection, got {error:?}")
                    };
                    assert_unsupported(source, operator, reason);
                }
            }
        }
    }

    #[test]
    fn local_defaults_and_unit_yarn_factor_remain_supported() {
        for tied in [false, true] {
            let fixture = Fixture::new(tied);
            for scaling in [
                RotaryScaling::None,
                RotaryScaling::Linear { factor: 2.0 },
                yarn(None),
                yarn(Some(1.0)),
            ] {
                let spec = layer_spec(fixture.resources.spec(), Some(scaling), None);
                let resources = Arc::new(rebind(&fixture, spec));
                for precision in [
                    ExecutionPrecisionPolicy::f32(),
                    ExecutionPrecisionPolicy::bf16_compatibility(),
                ] {
                    CpuGqaMoeModule::prepare(
                        Arc::clone(&resources),
                        Arc::new(StateDictMaterializer::new(READ_LIMIT).unwrap()),
                        precision,
                        MAX_POSITIONS,
                    )
                    .unwrap();
                    for plan in plans() {
                        StandardDecoderSegment::prepare(
                            &resources,
                            plan,
                            precision,
                            MAX_POSITIONS,
                            READ_LIMIT,
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
}

#[test]
fn plans_reject_invalid_ranges_and_ownership() {
    for (total, range) in [
        (0, 0..0),
        (2, 1..1),
        (2, 2..1),
        (2, 0..3),
        (2, usize::MAX..usize::MAX),
    ] {
        assert!(matches!(
            LayerSegmentPlan::new(total, range, false, false),
            Err(SegmentError::InvalidRange { .. })
        ));
    }
    assert!(matches!(
        LayerSegmentPlan::new(2, 1..2, true, true),
        Err(SegmentError::EmbeddingOwnership)
    ));
    assert!(matches!(
        LayerSegmentPlan::new(2, 0..1, true, true),
        Err(SegmentError::OutputOwnership)
    ));
    let [first, last] = plans();
    assert_eq!(last.local_layer(1), Some(0));
    assert_eq!(last.local_layer(0), None);
    assert_eq!(last.local_layer(2), None);
    assert_eq!(last.global_layer(0), Some(1));
    assert_eq!(last.global_layer(1), None);
    assert_eq!(last.global_layer(usize::MAX), None);
    assert!(LayerSegmentPlan::validate_pipeline(&[]).is_err());
    assert!(LayerSegmentPlan::validate_pipeline(&[last.clone(), first.clone()]).is_err());
    assert!(LayerSegmentPlan::validate_pipeline(&[first.clone(), first, last]).is_err());
    assert!(
        LayerSegmentPlan::validate_pipeline(&[
            LayerSegmentPlan::new(3, 0..1, true, false).unwrap(),
            LayerSegmentPlan::new(3, 2..3, false, true).unwrap(),
        ])
        .is_err()
    );
    assert!(matches!(
        LayerSegmentPlan::validate_pipeline(&[
            LayerSegmentPlan::new(2, 0..2, false, true).unwrap(),
        ]),
        Err(SegmentError::EmbeddingOwnership)
    ));
    assert!(matches!(
        LayerSegmentPlan::validate_pipeline(&[
            LayerSegmentPlan::new(2, 0..2, true, false).unwrap(),
        ]),
        Err(SegmentError::OutputOwnership)
    ));
    LayerSegmentPlan::validate_pipeline(&[LayerSegmentPlan::new(2, 0..2, true, true).unwrap()])
        .unwrap();
    let fixture = Fixture::new(false);
    assert!(matches!(
        StandardDecoderSegment::prepare(
            &fixture.resources,
            LayerSegmentPlan::new(3, 0..1, true, false).unwrap(),
            ExecutionPrecisionPolicy::f32(),
            MAX_POSITIONS,
            READ_LIMIT
        ),
        Err(SegmentError::LayerCount {
            planned: 3,
            actual: 2
        })
    ));
}

#[test]
fn segments_own_only_selected_resources_and_preserve_tied_head_dependency() {
    for tied in [false, true] {
        for (stage, plan) in plans().into_iter().enumerate() {
            let fixture = Fixture::new(tied);
            let precision = ExecutionPrecisionPolicy::f32();
            let mut kv = StageKv::new(&fixture, precision, 10);
            // Remove every unowned payload BEFORE preparation, including unowned
            // experts. Tied output's embedding file is a required exception.
            let owned_files = fixture
                .resources
                .state_dict()
                .parameters()
                .iter()
                .filter(|p| match p.residency() {
                    ParameterResidency::Layer { layer }
                    | ParameterResidency::Expert { layer, .. } => *layer == stage,
                    ParameterResidency::Static => {
                        if stage == 0 {
                            p.role() == &TensorRole::TokenEmbedding
                        } else {
                            matches!(p.role(), TensorRole::OutputHead | TensorRole::OutputNorm)
                        }
                    }
                    ParameterResidency::Attachment { .. } => false,
                })
                .map(|p| p.weight().slice().path.clone())
                .collect::<BTreeSet<_>>();
            for file in std::fs::read_dir(&fixture.directory).unwrap() {
                let path = file.unwrap().path();
                if !owned_files.contains(&path) {
                    std::fs::remove_file(path).unwrap();
                }
            }
            let segment = fixture.segment(plan.clone(), precision);
            assert_eq!(segment.plan(), &plan);
            for parameter in segment.parameters() {
                match parameter.residency() {
                    ParameterResidency::Layer { layer }
                    | ParameterResidency::Expert { layer, .. } => assert_eq!(*layer, stage),
                    ParameterResidency::Static => {
                        if stage == 0 {
                            assert_eq!(parameter.role(), &TensorRole::TokenEmbedding);
                        } else {
                            assert!(matches!(
                                parameter.role(),
                                TensorRole::OutputNorm | TensorRole::OutputHead
                            ));
                        }
                    }
                    ParameterResidency::Attachment { .. } => panic!("segment retained attachment"),
                }
            }
            if stage == 1 && tied {
                let head = segment
                    .parameters()
                    .iter()
                    .find(|p| p.role() == &TensorRole::OutputHead)
                    .unwrap();
                let embedding = fixture
                    .resources
                    .require_static(TensorRole::TokenEmbedding)
                    .unwrap();
                assert!(head.is_alias());
                assert_eq!(head.canonical_id(), embedding.canonical_id());
                assert!(head.shares_storage_with(embedding));
            }
            let (packed, transaction, mut view) = kv.enter(1, &[1, 2]);
            let input = if stage == 0 {
                SegmentInput::Tokens
            } else {
                SegmentInput::Hidden {
                    next_layer: 1,
                    rows: HostRows::new(
                        RowsShape::new(2, 4).unwrap(),
                        RowsDType::F32,
                        None,
                        vec![0.1; 8],
                    )
                    .unwrap(),
                }
            };
            let output = segment.execute(&packed, input, &mut view).unwrap();
            assert_eq!(matches!(output, SegmentOutput::Logits(_)), stage == 1);
            drop(view);
            kv.finish(transaction, 2, true);
        }
    }
}

#[test]
fn input_errors_are_typed_and_caller_controls_cancellation() {
    let fixture = Fixture::new(false);
    let precision = ExecutionPrecisionPolicy::f32();
    let [first, last] = plans().map(|plan| fixture.segment(plan, precision));
    let mut kv = StageKv::new(&fixture, precision, 10);
    let (packed, transaction, mut view) = kv.enter(1, &[1, 2]);
    assert!(matches!(
        last.execute(&packed, SegmentInput::Tokens, &mut view),
        Err(SegmentError::EmbeddingOwnership)
    ));
    let hidden = |next_layer, width, dtype| SegmentInput::Hidden {
        next_layer,
        rows: HostRows::zeros(RowsShape::new(2, width).unwrap(), dtype, None),
    };
    assert!(matches!(
        first.execute(&packed, hidden(0, 4, RowsDType::F32), &mut view),
        Err(SegmentError::EmbeddingOwnership)
    ));
    assert!(matches!(
        last.execute(&packed, hidden(0, 4, RowsDType::F32), &mut view),
        Err(SegmentError::HiddenBoundary { .. })
    ));
    assert!(matches!(
        last.execute(&packed, hidden(1, 3, RowsDType::F32), &mut view),
        Err(SegmentError::HiddenLayout { .. })
    ));
    assert!(matches!(
        last.execute(&packed, hidden(1, 4, RowsDType::Bf16), &mut view),
        Err(SegmentError::HiddenLayout { .. })
    ));
    assert!(matches!(
        last.execute(
            &packed,
            SegmentInput::Hidden {
                next_layer: 1,
                rows: HostRows::zeros(RowsShape::new(1, 4).unwrap(), RowsDType::F32, None),
            },
            &mut view,
        ),
        Err(SegmentError::HiddenLayout { .. })
    ));
    let short = StandardDecoderSegment::prepare(
        &fixture.resources,
        plans()[0].clone(),
        precision,
        1,
        READ_LIMIT,
    )
    .unwrap();
    assert!(matches!(
        short.execute(&packed, SegmentInput::Tokens, &mut view),
        Err(SegmentError::Position { position: 1, .. })
    ));
    let output = first
        .execute(&packed, SegmentInput::Tokens, &mut view)
        .unwrap();
    drop(output); // The caller cancels at the boundary instead of executing stage two.
    drop(view);
    kv.finish(transaction, 2, false);
    assert_eq!(kv.state.core().position(), 0);
    assert_eq!(kv.backend.capacity().resident_pages, 0);
    assert_eq!(
        kv.backend.page_status(KvPageId(10)),
        DecoderKvPageStatus::Vacant
    );
    // The same immutable segment can be used after caller-owned rollback.
    let (packed, transaction, mut view) = kv.enter(2, &[1, 2]);
    first
        .execute(&packed, SegmentInput::Tokens, &mut view)
        .unwrap();
    drop(view);
    kv.finish(transaction, 2, true);
    assert_eq!(kv.state.core().position(), 2);
    let (stale, reservations) = batch(&GenericDecoderSequenceState::new((), ()), &[3], 10);
    assert!(
        PackedDecoderBatch::lower(
            &stale,
            &reservations,
            std::slice::from_ref(&kv.state),
            &kv.capabilities,
            PAGE_SIZE,
            &|page| kv.backend.page_status(page)
        )
        .is_err()
    );
}

#[test]
fn lazy_expert_failure_reports_global_and_local_layer_and_can_be_rolled_back() {
    let fixture = Fixture::new(false);
    let precision = ExecutionPrecisionPolicy::f32();
    let last = fixture.segment(plans()[1].clone(), precision);
    for parameter in last.parameters() {
        if matches!(parameter.residency(), ParameterResidency::Expert { .. }) {
            std::fs::remove_file(&parameter.weight().slice().path).unwrap();
        }
    }
    let mut kv = StageKv::new(&fixture, precision, 10);
    let (packed, transaction, mut view) = kv.enter(1, &[1]);
    let input = SegmentInput::Hidden {
        next_layer: 1,
        rows: HostRows::new(
            RowsShape::new(1, 4).unwrap(),
            RowsDType::F32,
            None,
            vec![0.1, 0.2, 0.3, 0.4],
        )
        .unwrap(),
    };
    let error = last.execute(&packed, input, &mut view).unwrap_err();
    assert!(matches!(
        error,
        SegmentError::Execution {
            stage: SegmentStage::Layer {
                global: 1,
                local: 0
            },
            ..
        }
    ));
    assert!(std::error::Error::source(&error).is_some());
    drop(view);
    kv.finish(transaction, 1, false);
    assert_eq!(kv.state.core().position(), 0);
    assert_eq!(kv.backend.capacity().resident_pages, 0);
}

fn ep_error(message: &str) -> Error {
    Error::Execution {
        message: message.into(),
    }
}

fn owner_rank(layer: usize, expert: usize) -> ParallelRankId {
    ParallelRankId::new((layer * 2 + expert) as u32)
}

fn owner_bindings(resources: &BoundDecoderResources, id: ExpertId) -> [BoundParameter; 3] {
    [
        TensorRole::RoutedExpertGate,
        TensorRole::RoutedExpertUp,
        TensorRole::RoutedExpertDown,
    ]
    .map(|role| {
        resources
            .experts()
            .require(id.layer, id.expert, role)
            .unwrap()
            .clone()
    })
}

/// An owner receives only its assigned bindings at provisioning time, never the
/// full resource directory. No prepared weights are returned through transport.
struct OwnerProvider {
    id: ExpertId,
    prepared: Arc<PreparedSwiGlu>,
    calls: Vec<ExpertId>,
}

impl OwnerProvider {
    fn new(id: ExpertId, bindings: [BoundParameter; 3]) -> Self {
        let materializer = StateDictMaterializer::new(READ_LIMIT).unwrap();
        let linear = |binding: &BoundParameter| {
            assert_eq!(
                binding.residency(),
                &ParameterResidency::Expert {
                    layer: id.layer,
                    expert: id.expert,
                }
            );
            PreparedLinear::from_parameter(
                materializer
                    .expert_parameter(id.layer, id.expert, binding)
                    .unwrap(),
                binding.role().clone(),
            )
            .unwrap()
        };
        let prepared = Arc::new(
            PreparedSwiGlu::new(
                linear(&bindings[0]),
                linear(&bindings[1]),
                linear(&bindings[2]),
                None,
            )
            .unwrap(),
        );
        Self {
            id,
            prepared,
            calls: Vec::new(),
        }
    }
}

impl ExpertProvider for OwnerProvider {
    fn expert(&mut self, layer: usize, expert: usize) -> Result<ExpertAvailability> {
        let id = ExpertId::new(layer, expert);
        if id != self.id {
            return Err(ep_error("owner was asked to fetch non-owned weights"));
        }
        self.calls.push(id);
        Ok(ExpertAvailability::Ready(Arc::clone(&self.prepared)))
    }
}

struct ExpectedDispatch {
    transaction: ExecutionTransactionId,
    source_rank: ParallelRankId,
    layer: usize,
    row_sequences: Vec<u64>,
}

#[derive(Default)]
struct ThreadExpertResults {
    requests: BTreeMap<ParallelRankId, SyncSender<ExpertTokenBucket>>,
    replies: BTreeMap<ParallelRankId, Receiver<Result<Vec<ExpertResult>>>>,
    expected: Option<ExpectedDispatch>,
    calls: Vec<ParallelRankId>,
}

impl ExpertResultExecutor for ThreadExpertResults {
    fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
        let expected = self
            .expected
            .as_ref()
            .expect("caller admitted this decoder step");
        let owner = bucket.owner_rank;
        // top_k=2 with separate owners means each owner receives every source row.
        assert_eq!(bucket.tokens.len(), expected.row_sequences.len());
        for token in &bucket.tokens {
            assert_eq!(token.transaction, expected.transaction);
            assert_eq!(token.source_rank, expected.source_rank);
            assert_eq!(token.expert.layer, expected.layer);
            assert_eq!(owner_rank(token.expert.layer, token.expert.expert), owner);
            assert_eq!(token.sequence, expected.row_sequences[token.source_row]);
            assert_eq!(token.payload.len(), 4);
            assert!(token.route_slot < 2);
        }
        self.calls.push(owner);
        self.requests
            .get(&owner)
            .unwrap()
            .send(bucket)
            .map_err(|_| ep_error("owner request disconnected"))?;
        let mut results = self
            .replies
            .get(&owner)
            .unwrap()
            .recv_timeout(Duration::from_secs(5))
            .map_err(|_| ep_error("owner reply timeout/disconnected"))??;
        for result in &results {
            assert_eq!(result.owner_rank, owner);
            assert_eq!(result.output.len(), 4);
        }
        // Deliberately different arrival order; the existing EP combine owns weighting.
        results.reverse();
        Ok(results)
    }
}

#[test]
fn pp2_ep2_threaded_decoder_topk2_prefill_decode_matches_local_runner_bitwise() {
    for tied in [false, true] {
        let fixture = Fixture::with_top_k(tied, 2);
        let precision = ExecutionPrecisionPolicy::f32();
        let segments = plans().map(|plan| fixture.segment(plan, precision));
        let mut runner = oracle(&fixture, precision);
        let mut oracle_states = vec![
            runner.create_sequence_state().unwrap(),
            runner.create_sequence_state().unwrap(),
        ];
        let sequence_ids = oracle_states
            .iter()
            .map(|state| state.topology_id().get())
            .collect::<Vec<_>>();
        assert_ne!(sequence_ids[0], sequence_ids[1]);
        let initial_states = oracle_states.clone();
        let steps = [
            vec![vec![1, 2], vec![3]],
            vec![vec![4], vec![5]],
            vec![vec![6], vec![1]],
            vec![vec![2], vec![4]],
        ];
        // Run the same real full decoder before removing expert payloads. The EP
        // source then cannot accidentally use its checkpoint-backed local provider.
        let expected = steps
            .iter()
            .enumerate()
            .map(|(step, tokens)| {
                let tokens = tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
                execute_oracle_cohort(&mut runner, &mut oracle_states, 100 + step as u64, &tokens)
            })
            .collect::<Vec<_>>();
        drop(runner);
        let placement = ExpertPlacement::new((0..2).flat_map(|layer| {
            (0..2).map(move |expert| (layer, expert, owner_rank(layer, expert)))
        }))
        .unwrap();
        for layer in 0..2 {
            let first = owner_bindings(&fixture.resources, ExpertId::new(layer, 0));
            let second = owner_bindings(&fixture.resources, ExpertId::new(layer, 1));
            assert_ne!(
                std::fs::read(&first[0].weight().slice().path).unwrap(),
                std::fs::read(&second[0].weight().slice().path).unwrap(),
                "experts must have distinct checkpoint weights"
            );
        }
        std::thread::scope(|scope| {
            let mut remote = ThreadExpertResults::default();
            let mut workers = Vec::new();
            let mut prepared = Vec::new();
            for layer in 0..2 {
                for expert in 0..2 {
                    let id = ExpertId::new(layer, expert);
                    let owner = owner_rank(layer, expert);
                    let bindings = owner_bindings(&fixture.resources, id);
                    let (send, receive) = sync_channel::<ExpertTokenBucket>(1);
                    let (reply, result) = sync_channel::<Result<Vec<ExpertResult>>>(1);
                    let (ready, ready_rx) = sync_channel(1);
                    remote.requests.insert(owner, send);
                    remote.replies.insert(owner, result);
                    prepared.push(ready_rx);
                    let placement = &placement;
                    workers.push(scope.spawn(move || {
                        let mut provider = OwnerProvider::new(id, bindings);
                        assert!(provider.expert(layer, 1 - expert).is_err());
                        ready.send(()).unwrap();
                        {
                            let mut worker =
                                CpuReferenceExpertWorker::new(owner, placement, &mut provider);
                            let mut executor =
                                CpuExpertResultExecutor::new(vec![&mut worker]).unwrap();
                            while let Ok(bucket) = receive.recv() {
                                assert_eq!(bucket.owner_rank, owner);
                                reply.send(executor.execute(bucket)).unwrap();
                            }
                        }
                        assert_eq!(provider.calls, vec![id; 4]);
                    }));
                }
            }
            for ready in prepared {
                ready.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            for parameter in fixture.resources.state_dict().parameters() {
                if matches!(parameter.residency(), ParameterResidency::Expert { .. }) {
                    std::fs::remove_file(&parameter.weight().slice().path).unwrap();
                }
            }
            let mut pools = [backend(1, precision), backend(1, precision)];
            let mut states = [initial_states.clone(), initial_states.clone()];
            let mut previous: [Vec<Option<KvHistory>>; 2] = [vec![None, None], vec![None, None]];
            let mut positions = [0usize, 0];
            for (step, tokens) in steps.iter().enumerate() {
                let transaction = tx(100 + step as u64);
                let token_rows = tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let counts = tokens.iter().map(Vec::len).collect::<Vec<_>>();
                let mut input = SegmentInput::Tokens;
                let mut handles = Vec::new();
                for layer in 0..2 {
                    let (packed, handle, mut view) = enter_cohort(
                        &mut pools[layer],
                        &mut states[layer],
                        options(&fixture.resources, precision).capabilities(),
                        transaction,
                        &token_rows,
                        10 + layer as u32 * 40,
                    );
                    assert_eq!(view.transaction(), transaction);
                    assert_eq!(
                        packed.row_to_sequence(),
                        if step == 0 {
                            &[0, 0, 1][..]
                        } else {
                            &[0, 1][..]
                        }
                    );
                    let packed_ids = packed
                        .sequences()
                        .iter()
                        .map(|sequence| sequence_ids[sequence.state_index()])
                        .collect::<Vec<_>>();
                    remote.expected = Some(ExpectedDispatch {
                        transaction,
                        source_rank: owner_rank(layer, 0),
                        layer,
                        row_sequences: packed
                            .row_to_sequence()
                            .iter()
                            .map(|&index| packed_ids[index])
                            .collect(),
                    });
                    let mut executor = ExpertParallelRoutedExecutor::new(
                        vec![owner_rank(layer, 0), owner_rank(layer, 1)],
                        &placement,
                        ExpertDispatchLimits {
                            max_tokens: 32,
                            max_bytes: 4096,
                        },
                        &mut remote,
                    );
                    let mut active = |id| {
                        if id == transaction {
                            Ok(())
                        } else {
                            Err(ep_error("unknown caller transaction"))
                        }
                    };
                    let output = segments[layer]
                        .execute_with_experts(
                            &packed,
                            input,
                            &mut view,
                            (transaction, owner_rank(layer, 0), &packed_ids),
                            &mut executor,
                            &mut active,
                        )
                        .unwrap();
                    for sequence in 0..2 {
                        assert_eq!(
                            states[layer][sequence].core().position(),
                            positions[sequence]
                        );
                        let end = positions[sequence] + counts[sequence];
                        let history = KvView::history(&view, 0, sequence, end - 1, 1, 2).unwrap();
                        assert_eq!(history.tokens, end);
                        assert!(KvView::history(&view, 1, sequence, end - 1, 1, 2).is_err());
                        if let Some(previous) = &previous[layer][sequence] {
                            assert_values(&history.key[..previous.key.len()], &previous.key);
                            assert_values(&history.value[..previous.value.len()], &previous.value);
                        }
                        previous[layer][sequence] = Some(history);
                    }
                    if layer == 0 {
                        input = transfer(output);
                    } else {
                        let SegmentOutput::Logits(logits) = output else {
                            panic!("missing decoder logits")
                        };
                        assert_eq!(logits.rows(), packed.len());
                        assert_eq!(logits.width(), 8);
                        assert_eq!(expected[step].logits.len(), logits.rows());
                        for (row, expected) in expected[step].logits.iter().enumerate() {
                            assert_eq!(expected.input_row as usize, row);
                            let LogitsOutput::Full(values) = &expected.logits else {
                                panic!("oracle must return full logits")
                            };
                            assert_values(logits.row(row).unwrap(), values);
                        }
                        input = SegmentInput::Tokens;
                    }
                    drop(view);
                    handles.push(handle);
                }
                // Only the caller publishes, after both segment outputs succeeded.
                for (layer, handle) in handles.into_iter().enumerate() {
                    finish_cohort(&mut pools[layer], &mut states[layer], handle, &counts, true);
                    for sequence in 0..2 {
                        assert_eq!(
                            states[layer][sequence].core().position(),
                            positions[sequence] + counts[sequence]
                        );
                        assert_eq!(
                            states[layer][sequence].topology_id().get(),
                            sequence_ids[sequence]
                        );
                    }
                }
                for sequence in 0..2 {
                    positions[sequence] += counts[sequence];
                }
                let pages = positions
                    .iter()
                    .map(|p| p.div_ceil(PAGE_SIZE))
                    .sum::<usize>();
                for pool in &pools {
                    assert_eq!(pool.capacity().resident_pages, pages);
                }
            }
            assert_eq!(positions, [5, 4]);
            assert_eq!(
                remote.calls,
                [
                    owner_rank(0, 0),
                    owner_rank(0, 1),
                    owner_rank(1, 0),
                    owner_rank(1, 1)
                ]
                .repeat(4)
            );
            drop(remote);
            for worker in workers {
                worker.join().unwrap();
            }
        });
    }
}

#[derive(Default)]
struct FailingRoutedExecutor {
    calls: usize,
}

impl RoutedSwiGluExecutor for FailingRoutedExecutor {
    fn routed_swiglu(
        &mut self,
        _: RoutedSwiGluRequest<'_>,
        _: &mut dyn FnMut(ExecutionTransactionId) -> Result<()>,
    ) -> Result<OperatorProgress<Rows>> {
        self.calls += 1;
        Err(ep_error("injected routed failure"))
    }
}

#[test]
fn decoder_ep_rejects_bf16_bad_context_and_cancellation_without_local_fallback() {
    let fixture = Fixture::with_top_k(false, 2);
    for precision in [
        ExecutionPrecisionPolicy::bf16_compatibility(),
        ExecutionPrecisionPolicy::f32(),
    ] {
        let segment = fixture.segment(plans()[0].clone(), precision);
        let mut kv = StageKv::new(&fixture, precision, 10);
        let sequence_ids = [kv.state.topology_id().get()];
        let (packed, handle, mut view) = kv.enter(300, &[1, 2]);
        let mut executor = FailingRoutedExecutor::default();
        let mut active = |id| {
            assert_eq!(id, tx(300));
            Ok(())
        };
        if precision == ExecutionPrecisionPolicy::bf16_compatibility() {
            assert!(matches!(
                segment.execute_with_experts(
                    &packed,
                    SegmentInput::Tokens,
                    &mut view,
                    (tx(300), owner_rank(0, 0), &sequence_ids),
                    &mut executor,
                    &mut active
                ),
                Err(SegmentError::ExpertPrecision)
            ));
            assert_eq!(executor.calls, 0);
            // The ordinary BF16 local path remains available, explicitly chosen.
            segment
                .execute(&packed, SegmentInput::Tokens, &mut view)
                .unwrap();
        } else {
            assert!(matches!(
                segment.execute_with_experts(
                    &packed,
                    SegmentInput::Tokens,
                    &mut view,
                    (tx(301), owner_rank(0, 0), &sequence_ids),
                    &mut executor,
                    &mut active
                ),
                Err(SegmentError::ExpertTransaction { .. })
            ));
            assert!(matches!(
                segment.execute_with_experts(
                    &packed,
                    SegmentInput::Tokens,
                    &mut view,
                    (tx(300), owner_rank(0, 0), &[]),
                    &mut executor,
                    &mut active
                ),
                Err(SegmentError::ExpertSequenceCount {
                    expected: 1,
                    actual: 0
                })
            ));
            for reason in ["cancelled", "unknown caller transaction"] {
                let error = segment
                    .execute_with_experts(
                        &packed,
                        SegmentInput::Tokens,
                        &mut view,
                        (tx(300), owner_rank(0, 0), &sequence_ids),
                        &mut executor,
                        &mut |id| {
                            assert_eq!(id, tx(300));
                            Err(ep_error(reason))
                        },
                    )
                    .unwrap_err();
                assert!(matches!(
                    error,
                    SegmentError::Execution {
                        stage: SegmentStage::Input,
                        ..
                    }
                ));
                assert!(error.to_string().contains(reason));
            }
            assert_eq!(executor.calls, 0);
            let error = segment
                .execute_with_experts(
                    &packed,
                    SegmentInput::Tokens,
                    &mut view,
                    (tx(300), owner_rank(0, 0), &sequence_ids),
                    &mut executor,
                    &mut active,
                )
                .unwrap_err();
            assert!(matches!(
                error,
                SegmentError::Execution {
                    stage: SegmentStage::Layer {
                        global: 0,
                        local: 0
                    },
                    ..
                }
            ));
            assert!(error.to_string().contains("injected routed failure"));
            assert_eq!(executor.calls, 1);
        }
        drop(view);
        kv.finish(handle, 2, false);
        assert_eq!(kv.state.core().position(), 0);
        assert_eq!(kv.backend.capacity().resident_pages, 0);
    }
}

#[test]
fn decoder_ep_cancel_after_real_owner_reply_returns_no_output_and_rolls_back() {
    use std::cell::Cell;
    struct CancelReply<'a> {
        inner: CpuExpertResultExecutor<'a>,
        active: &'a Cell<bool>,
        calls: usize,
    }
    impl ExpertResultExecutor for CancelReply<'_> {
        fn execute(&mut self, bucket: ExpertTokenBucket) -> Result<Vec<ExpertResult>> {
            self.calls += 1;
            let reply = self.inner.execute(bucket)?;
            self.active.set(false);
            Ok(reply)
        }
    }
    let fixture = Fixture::with_top_k(false, 2);
    let precision = ExecutionPrecisionPolicy::f32();
    let segment = fixture.segment(plans()[1].clone(), precision);
    let placement =
        ExpertPlacement::new([(1, 0, owner_rank(1, 0)), (1, 1, owner_rank(1, 1))]).unwrap();
    let mut provider0 = OwnerProvider::new(
        ExpertId::new(1, 0),
        owner_bindings(&fixture.resources, ExpertId::new(1, 0)),
    );
    let mut provider1 = OwnerProvider::new(
        ExpertId::new(1, 1),
        owner_bindings(&fixture.resources, ExpertId::new(1, 1)),
    );
    let mut worker0 = CpuReferenceExpertWorker::new(owner_rank(1, 0), &placement, &mut provider0);
    let mut worker1 = CpuReferenceExpertWorker::new(owner_rank(1, 1), &placement, &mut provider1);
    let active = Cell::new(true);
    let mut results = CancelReply {
        inner: CpuExpertResultExecutor::new(vec![&mut worker0, &mut worker1]).unwrap(),
        active: &active,
        calls: 0,
    };
    let mut executor = ExpertParallelRoutedExecutor::new(
        vec![owner_rank(1, 0), owner_rank(1, 1)],
        &placement,
        ExpertDispatchLimits {
            max_tokens: 32,
            max_bytes: 4096,
        },
        &mut results,
    );
    let mut kv = StageKv::new(&fixture, precision, 10);
    let sequence_ids = [kv.state.topology_id().get()];
    let (packed, handle, mut view) = kv.enter(400, &[1, 2]);
    let input = SegmentInput::Hidden {
        next_layer: 1,
        rows: HostRows::new(
            RowsShape::new(2, 4).unwrap(),
            RowsDType::F32,
            None,
            vec![0.1, 0.2, 0.3, 0.4, -0.2, 0.1, 0.4, 0.3],
        )
        .unwrap(),
    };
    let error = segment
        .execute_with_experts(
            &packed,
            input,
            &mut view,
            (tx(400), owner_rank(1, 0), &sequence_ids),
            &mut executor,
            &mut |id| {
                assert_eq!(id, tx(400));
                if active.get() {
                    Ok(())
                } else {
                    Err(ep_error("cancelled after owner reply"))
                }
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        SegmentError::Execution {
            stage: SegmentStage::Layer {
                global: 1,
                local: 0
            },
            ..
        }
    ));
    assert!(error.to_string().contains("cancelled after owner reply"));
    assert_eq!(results.calls, 1);
    assert_eq!(provider0.calls, [ExpertId::new(1, 0)]);
    assert!(provider1.calls.is_empty());
    drop(view);
    kv.finish(handle, 2, false);
    assert_eq!(kv.state.core().position(), 0);
    assert_eq!(kv.backend.capacity().resident_pages, 0);
}
