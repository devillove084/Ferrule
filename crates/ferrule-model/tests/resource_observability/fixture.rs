//! CPU physical-owner mock used through the production forwarding chain.

use super::model::decoder::*;
use super::model::resource_observability::*;
use super::model::transformer::*;
use super::model::{AttentionKind, ModelFamily, ModelInfo, TokenizerHandle, WeightSource};
use ferrule_common::execution::*;
use ferrule_common::{CompletionHub, Error, Result};
use std::cell::Cell;
use std::convert::Infallible;
use std::num::NonZeroU32;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

type State = DecoderSequenceState<(), Vec<u8>>;
type Backend = PagedKvBackend<TypedCpuPagedKvPool<State>>;

#[derive(Default)]
pub struct Control {
    pub unsupported: Cell<bool>,
    pub reads: Cell<usize>,
    pub shutdown_calls: Cell<usize>,
    pub failures: Cell<usize>,
    pub unknown: Cell<bool>,
    pub lose_completion: Cell<bool>,
    pub closed: Cell<bool>,
    pub omit_quiescence_proof: Cell<bool>,
}

pub struct ReportingModule {
    control: Rc<Control>,
    // Real mock-owned allocations, retained after close until this object drops.
    resident: Vec<u8>,
    pending: Vec<u8>,
    scratch: Vec<u8>,
}

impl TransformerModule for ReportingModule {
    type State = State;
    type KvView = CpuKvView;
    type Hidden = ();
    type ArenaKey = usize;
    type Arena = ();
    type EmbeddingPending = Infallible;
    type LayerPending = Infallible;
    type OutputPending = Infallible;
    type Quiescence = ();
    type Event = Infallible;
    type TerminalGuard = NoTerminalGuard;

    fn physical_resource_snapshot(&self, state: &State) -> PhysicalResourceObservation {
        self.control.reads.set(self.control.reads.get() + 1);
        if self.control.unsupported.get() {
            return PhysicalResourceObservation::default();
        }
        let charged = self.resident.len() + self.pending.len() + self.scratch.len();
        let mut observation = PhysicalResourceObservation {
            image_generation: ResourceEvidence::Observed(47),
            default_state: ResourceEvidence::Observed(DefaultStateObservation {
                topology_id: state.topology_id().get(),
                generation: state.core().generation(),
                position: state.core().position(),
                poisoned: state.core().is_poisoned(),
                resident_bytes: ResourceEvidence::Observed(state.kv_state().len()),
            }),
            resident_parameter_bytes: ResourceEvidence::Observed(self.resident.len()),
            live_model_state_bytes: ResourceEvidence::Observed(state.kv_state().len()),
            expert_cache: ResourceEvidence::Observed(ExpertCacheObservation {
                resident_entries: 1,
                resident_bytes: self.resident.len(),
                pending_upload_bytes: self.pending.len(),
                scratch_bytes: self.scratch.len(),
                charged_bytes: charged,
                unknown_quarantine_bytes: if self.control.unknown.get() {
                    charged
                } else {
                    0
                },
                peak_bytes: charged,
                quarantined: self.control.unknown.get(),
                uploads: 1,
                hits: 0,
                evictions: 0,
            }),
            numeric_workspace: ResourceEvidence::Observed(WorkspaceObservation {
                image_generation: 47,
                kind: "numeric",
                reserved_bytes: self.scratch.len(),
                allocated_bytes: self.scratch.len(),
                creations: 1,
                reuses: 0,
                submissions: 0,
            }),
            quarantine: ResourceEvidence::Observed(self.control.unknown.get()),
            residency: ResourceEvidence::Observed(ResourceResidency::Retained),
            ..Default::default()
        };
        if self.control.unknown.get() {
            observation.mark_unknown();
        } else if self.control.closed.get() && !self.control.omit_quiescence_proof.get() {
            // This CPU mock supplies its explicit synchronous close receipt.
            observation.quiescence = ResourceEvidence::Verified(ResourceQuiescence::Quiescent);
        }
        observation
    }

    fn layer_count(&self) -> usize {
        1
    }
    fn validate(
        &mut self,
        _: &DecoderTransactionContext,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
    ) -> Result<()> {
        panic!("observation entered execution")
    }
    fn arena_key(&self, _: &PackedDecoderBatch) -> Result<usize> {
        panic!("observation requested an arena")
    }
    fn build_arena(&mut self, _: &usize) -> Result<()> {
        panic!("observation allocated an arena")
    }
    fn submit_embedding(
        &mut self,
        _: &DecoderTransactionContext,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &mut (),
    ) -> Result<Step<(), Infallible>> {
        unreachable!()
    }
    fn poll_embedding(
        &mut self,
        _: &DecoderTransactionContext,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &mut (),
        _: &mut Infallible,
    ) -> Result<Poll<(), Infallible>> {
        unreachable!()
    }
    fn start_layer(
        &mut self,
        _: &DecoderTransactionContext,
        _: usize,
        _: LayerRequest<'_, State, CpuKvView>,
        _: &mut (),
        _: &mut (),
    ) -> Result<Step<Vec<Infallible>, Infallible>> {
        unreachable!()
    }
    fn poll_layer(
        &mut self,
        _: &DecoderTransactionContext,
        _: usize,
        _: LayerRequest<'_, State, CpuKvView>,
        _: &mut (),
        _: &mut (),
        _: &mut Infallible,
    ) -> Result<Poll<Vec<Infallible>, Infallible>> {
        unreachable!()
    }
    fn post_layer(&mut self, _: usize, _: &PackedDecoderBatch, _: &(), _: &mut ()) -> Result<()> {
        unreachable!()
    }
    fn submit_output(
        &mut self,
        _: &DecoderTransactionContext,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &(),
        _: &mut (),
    ) -> Result<Step<DecoderLogits, Infallible>> {
        unreachable!()
    }
    fn poll_output(
        &mut self,
        _: &DecoderTransactionContext,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &(),
        _: &mut (),
        _: &mut Infallible,
    ) -> Result<Poll<DecoderLogits, Infallible>> {
        unreachable!()
    }
    fn poll_embedding_cancel(&mut self, _: &mut Infallible) -> Result<bool> {
        unreachable!()
    }
    fn poll_layer_cancel(&mut self, _: usize, _: &mut Infallible) -> Result<bool> {
        unreachable!()
    }
    fn poll_output_cancel(&mut self, _: &mut Infallible) -> Result<bool> {
        unreachable!()
    }
    fn cancel_embedding(&mut self, pending: Infallible) -> Result<()> {
        match pending {}
    }
    fn cancel_layer(&mut self, _: usize, pending: Infallible) -> Result<()> {
        match pending {}
    }
    fn cancel_output(&mut self, pending: Infallible) -> Result<()> {
        match pending {}
    }
    fn begin_quiescence(&mut self) -> Result<()> {
        panic!("observation began quiescence")
    }
    fn poll_quiescence(&mut self, _: &mut ()) -> Result<bool> {
        panic!("observation polled quiescence")
    }
    fn finish(
        &mut self,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &mut (),
        _: Vec<Infallible>,
    ) -> Result<NoTerminalGuard> {
        unreachable!()
    }
    fn abort(
        &mut self,
        _: &PackedDecoderBatch,
        _: &mut [State],
        _: &mut CpuKvView,
        _: &mut (),
    ) -> Result<()> {
        panic!("observation aborted work")
    }
    fn shutdown(&mut self) -> Result<()> {
        self.control
            .shutdown_calls
            .set(self.control.shutdown_calls.get() + 1);
        if self.control.failures.get() != 0 {
            self.control.failures.set(self.control.failures.get() - 1);
            self.control.unknown.set(self.control.lose_completion.get());
            return Err(Error::Execution {
                message: "mock physical close failed".into(),
            });
        }
        if self.control.unknown.get() {
            return Err(Error::Execution {
                message: "mock physical completion remains unknown".into(),
            });
        }
        self.control.closed.set(true);
        Ok(())
    }
}

pub struct Composition;
impl DecoderComposition for Composition {
    type Resources = ();
    type SequenceState = State;
    type KvBackend = Backend;
    type ForwardExecutor = TransformerForwardExecutor<ReportingModule>;
    type Proposal = NoProposal;
    type SequenceLifecycle = StandardSequenceLifecycle<(), Vec<u8>>;
    type ResourceManager = NoResourceManager;
    type Observer = StandardDecoderObserver;
    type Snapshot = GenericDecoderObservabilitySnapshot;
    type Continuation = TransformerContinuation<ReportingModule>;
    type ProposalContinuation = Infallible;
    type TerminalGuard = NoTerminalGuard;
}

fn tokenizer() -> TokenizerHandle {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    let path = std::env::temp_dir().join(format!(
        "ferrule-resource-observation-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ));
    std::fs::create_dir_all(&path).unwrap();
    std::fs::write(path.join("tokenizer.json"), r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"x":1},"unk_token":"<unk>"}}"#).unwrap();
    let tokenizer = TokenizerHandle::load(&path).unwrap();
    std::fs::remove_dir_all(path).unwrap();
    tokenizer
}

pub fn runner(control: Rc<Control>) -> GenericDecoderRunner<Composition> {
    let strategy = StandardGqaPlanes::new(1, 1, 1, 2, 4, KvElementType::F32).unwrap();
    GenericDecoderRunner::from_runtime(DecoderComponents {
        resources: (),
        model_info: ModelInfo {
            family: ModelFamily::Unknown("resource-observation-mock".into()),
            architecture: None,
            attention: AttentionKind::DenseMha,
            weight_source: WeightSource::Safetensors,
            hidden_size: 2,
            num_layers: 1,
            num_experts: 0,
            num_experts_per_tok: 0,
            vocab_size: 2,
            backend: "cpu",
        },
        tokenizer: tokenizer(),
        capabilities: ExecutionCapabilities {
            max_batch_tokens: 2,
            max_sequences: 1,
            max_prefill_query_tokens_per_sequence: 2,
            max_decode_query_tokens_per_sequence: 1,
            max_top_k: NonZeroU32::new(2),
            supports_prefill: true,
            supports_decode: true,
            supports_mixed: true,
            full_logits_width: NonZeroU32::new(2),
            kv_binding_mode: KvBindingMode::Paged,
            logits_row_policy: LogitsRowPolicy::Any,
        },
        page_size: 2,
        prepared_plan_id: 41,
        forward_executor: TransformerForwardExecutor::new(ReportingModule {
            control,
            resident: vec![0; 64],
            pending: vec![0; 32],
            scratch: vec![0; 16],
        }),
        backend: Backend::new(TypedCpuPagedKvPool::from_strategy(&strategy, 4).unwrap()),
        default_state: State::new((), vec![0; 24]),
        sequence_lifecycle: StandardSequenceLifecycle::new(),
        proposal: NoProposal,
        resource_manager: NoResourceManager,
        observer: StandardDecoderObserver,
        completion_hub: CompletionHub::new(),
        completion_reactors: vec![],
    })
    .unwrap()
}
