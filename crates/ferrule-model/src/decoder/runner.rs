use super::{
    CpuKvView, DecoderCancelProgress, DecoderComponents, DecoderComposition,
    DecoderContinuationWait, DecoderControlPlane, DecoderForwardExecutor, DecoderKvBackend,
    DecoderObserver, DecoderProposalExecutor, DecoderProposalProgress,
    DecoderProposalResumeProgress, DecoderResolverHandle, DecoderResourceManager, DecoderSequence,
    DecoderSequenceLifecycle, DecoderSequenceState, DecoderTransactionContext,
    DecoderTransactionProgress, DecoderWait, KvEndProgress, NoProposal, NoResourceManager,
    NoTerminalGuard, PackedDecoderBatch, PackedTransactionRegistry, PagedKvBackend, PhysicalKvPool,
    StandardDecoderObserver, StandardSequenceLifecycle,
};
use crate::execution::ExecutionPrecisionPolicy;
use crate::materialization::{
    MaterializationProvider, MaterializationRequest, MaterializationResolver,
};
use crate::runner::{
    ModelCompletionReactor, ModelInfo, ModelRunner, MultiSessionBatchProgress, MultiSessionRunner,
    NativeProposal, NativeProposalProgress, NativeProposalSource, PendingModelProgress,
    ResidentModelRunner, SequenceStateReleaseError, TransactionEndIntent, TransactionEndProgress,
};
use crate::spec::{AttentionKind, ModelFamily, WeightSource};
use crate::tokenizer::TokenizerHandle;
use crate::transformer::{
    BoundDecoderResources, CpuGqaMoeModule, FeedForward, PreparedDecoderGeneration,
    StateDictMaterializer, TransformerForwardExecutor,
};
use ferrule_common::execution::{
    ExecutionBatch, ExecutionCapabilities, ExecutionTransactionId, KvBindingMode, KvPageId,
    KvReservationView, LogitsRowPolicy,
};
use ferrule_common::expert_residency::{ExpertResidencyControl, ExpertResidencyRequirements};
use ferrule_common::{CompletionHub, ContinuationId, Error, ResidencyLeaseSet, Result};
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::sync::Arc;

/// Sequence state used by the standard model-independent decoder runner.
pub type GenericDecoderSequenceState = DecoderSequenceState<(), ()>;
/// Construction policy for a standard model-independent decoder.
#[derive(Debug, Clone)]
pub struct GenericDecoderOptions {
    family: ModelFamily,
    weight_source: WeightSource,
    capabilities: ExecutionCapabilities,
    page_size: usize,
    max_positions: usize,
    max_parameter_bytes: u64,
    active_layers: Option<usize>,
    precision: ExecutionPrecisionPolicy,
}
impl GenericDecoderOptions {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        family: ModelFamily,
        weight_source: WeightSource,
        capabilities: ExecutionCapabilities,
        page_size: usize,
        max_positions: usize,
        max_parameter_bytes: u64,
        precision: ExecutionPrecisionPolicy,
    ) -> Self {
        Self {
            family,
            weight_source,
            capabilities,
            page_size,
            max_positions,
            max_parameter_bytes,
            active_layers: None,
            precision,
        }
    }
    /// Restricts execution to a non-empty prefix of the described decoder.
    pub fn with_active_layers(mut self, active_layers: usize) -> Result<Self> {
        if active_layers == 0 {
            return Err(runner_error("decoder active layer count must be non-zero"));
        }
        self.active_layers = Some(active_layers);
        Ok(self)
    }
    /// Builds ordinary packed CPU capabilities from a decoder descriptor.
    #[allow(clippy::too_many_arguments)]
    pub fn standard_cpu(
        spec: &crate::transformer::DecoderModelSpec,
        family: ModelFamily,
        weight_source: WeightSource,
        page_size: usize,
        max_positions: usize,
        max_batch_tokens: usize,
        max_sequences: usize,
        max_parameter_bytes: u64,
        precision: ExecutionPrecisionPolicy,
    ) -> Result<Self> {
        let vocabulary = u32::try_from(spec.vocab_size())
            .map_err(|_| runner_error("decoder vocabulary exceeds the u32 execution protocol"))?;
        let full_logits_width = NonZeroU32::new(vocabulary)
            .ok_or_else(|| runner_error("decoder vocabulary must be non-zero"))?;
        Ok(Self::new(
            family,
            weight_source,
            ExecutionCapabilities {
                max_batch_tokens,
                max_sequences,
                max_prefill_query_tokens_per_sequence: max_batch_tokens,
                max_decode_query_tokens_per_sequence: 1,
                max_top_k: Some(full_logits_width),
                supports_prefill: true,
                supports_decode: true,
                supports_mixed: true,
                full_logits_width: Some(full_logits_width),
                kv_binding_mode: KvBindingMode::Paged,
                logits_row_policy: LogitsRowPolicy::Any,
            },
            page_size,
            max_positions,
            max_parameter_bytes,
            precision,
        ))
    }
    pub const fn family(&self) -> &ModelFamily {
        &self.family
    }
    pub const fn weight_source(&self) -> WeightSource {
        self.weight_source
    }
    pub const fn capabilities(&self) -> ExecutionCapabilities {
        self.capabilities
    }
    pub const fn page_size(&self) -> usize {
        self.page_size
    }
    pub const fn max_positions(&self) -> usize {
        self.max_positions
    }
    pub const fn max_parameter_bytes(&self) -> u64 {
        self.max_parameter_bytes
    }
    pub const fn active_layers(&self) -> Option<usize> {
        self.active_layers
    }
    pub const fn precision(&self) -> ExecutionPrecisionPolicy {
        self.precision
    }
    fn validate(&self, resources: &BoundDecoderResources) -> Result<()> {
        let spec = resources.spec();
        if self.page_size == 0 {
            return Err(runner_error("decoder KV page size must be non-zero"));
        }
        if self.max_positions == 0
            || spec
                .max_sequence_length()
                .is_some_and(|maximum| self.max_positions > maximum)
        {
            return Err(runner_error(format!(
                "invalid decoder maximum position count {}",
                self.max_positions
            )));
        }
        if self.max_parameter_bytes == 0 {
            return Err(runner_error(
                "decoder state-dict materialization limit must be non-zero",
            ));
        }
        if self
            .active_layers
            .is_some_and(|layers| layers == 0 || layers > spec.layers().len())
        {
            return Err(runner_error(format!(
                "decoder active layer count {:?} exceeds {} described layers",
                self.active_layers,
                spec.layers().len()
            )));
        }
        validate_capabilities(self.capabilities, self.max_positions)?;
        if self
            .capabilities
            .full_logits_width
            .is_some_and(|width| width.get() as usize != spec.vocab_size())
        {
            return Err(runner_error(format!(
                "full-logits capability does not match vocabulary size {}",
                spec.vocab_size()
            )));
        }
        if self
            .capabilities
            .max_top_k
            .is_some_and(|width| width.get() as usize > spec.vocab_size())
        {
            return Err(runner_error(
                "top-k capability exceeds decoder vocabulary size",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GenericDecoderObservabilitySnapshot {
    pub backend: &'static str,
    pub bound_layers: usize,
    pub physical_kv_pages: usize,
    pub resident_kv_pages: usize,
    pub preempted_kv_pages: usize,
    pub free_kv_pages: usize,
    pub active_transactions: usize,
    pub active_proposals: usize,
    pub completed_batches: u64,
    pub waits: u64,
    pub resumes: u64,
    pub proposal_waits: u64,
    pub proposal_resumes: u64,
    pub commits: u64,
    pub aborts: u64,
    pub shutdown: bool,
}
struct ProposalSlot<C> {
    transaction: ExecutionTransactionId,
    topology_id: crate::execution::SequenceTopologyId,
    continuation: C,
    wait: Option<DecoderContinuationWait>,
    leases: Vec<ResidencyLeaseSet>,
}
/// Fully bound decoder over one composition spec.
///
/// Every model-specific attachment is named by `D`; the transaction machinery
/// never exposes a list of independent generic parameters.
/// Borrowed inspection view over runner-owned model services.
pub struct GenericDecoderModelView<'a> {
    pub tokenizer: &'a TokenizerHandle,
}

pub struct GenericDecoderRunner<D>
where
    D: DecoderComposition,
{
    resources: D::Resources,
    model_info: ModelInfo,
    tokenizer: TokenizerHandle,
    capabilities: ExecutionCapabilities,
    page_size: usize,
    forward_executor: D::ForwardExecutor,
    backend: D::KvBackend,
    default_state: D::SequenceState,
    sequence_lifecycle: D::SequenceLifecycle,
    transactions: PackedTransactionRegistry<D>,
    proposal_continuations: HashMap<ContinuationId, ProposalSlot<D::ProposalContinuation>>,
    prepared_plan_id: u64,
    control: DecoderControlPlane,
    completion_reactors: Vec<ModelCompletionReactor>,
    observability: GenericDecoderObservabilitySnapshot,
    proposal: D::Proposal,
    resource_manager: D::ResourceManager,
    resolver: DecoderResolverHandle,
    observer: D::Observer,
    shutdown: bool,
}
impl<P> DecoderComposition for PagedKvBackend<P>
where
    P: PhysicalKvPool<SequenceState = GenericDecoderSequenceState, KvView = CpuKvView>,
{
    type Resources = Arc<BoundDecoderResources>;
    type SequenceState = GenericDecoderSequenceState;
    type KvBackend = Self;
    type ForwardExecutor = TransformerForwardExecutor<CpuGqaMoeModule>;
    type Proposal = NoProposal;
    type SequenceLifecycle = StandardSequenceLifecycle<(), ()>;
    type ResourceManager = NoResourceManager;
    type Observer = StandardDecoderObserver;
    type Snapshot = GenericDecoderObservabilitySnapshot;
    type Continuation = crate::transformer::TransformerContinuation<CpuGqaMoeModule>;
    type ProposalContinuation = std::convert::Infallible;
    type TerminalGuard = NoTerminalGuard;
}
impl<P> GenericDecoderRunner<PagedKvBackend<P>>
where
    P: PhysicalKvPool<SequenceState = GenericDecoderSequenceState, KvView = CpuKvView>,
{
    /// Standard CPU constructor used by Qwen adapters.
    pub fn new(
        resources: BoundDecoderResources,
        tokenizer: TokenizerHandle,
        backend: PagedKvBackend<P>,
        options: GenericDecoderOptions,
    ) -> Result<Self> {
        Self::standard_cpu(resources, tokenizer, backend, options)
    }
    /// Prepares the standard forward executor and injects a fresh completion hub once.
    pub fn standard_cpu(
        resources: BoundDecoderResources,
        tokenizer: TokenizerHandle,
        backend: PagedKvBackend<P>,
        options: GenericDecoderOptions,
    ) -> Result<Self> {
        Self::standard_cpu_with_completion(
            resources,
            tokenizer,
            backend,
            options,
            CompletionHub::new(),
            Vec::new(),
        )
    }
    /// Standard constructor for producers that must share a pre-created hub.
    /// The hub cannot be replaced after construction.
    pub fn standard_cpu_with_completion(
        resources: BoundDecoderResources,
        tokenizer: TokenizerHandle,
        backend: PagedKvBackend<P>,
        options: GenericDecoderOptions,
        completion_hub: CompletionHub,
        completion_reactors: Vec<ModelCompletionReactor>,
    ) -> Result<Self> {
        options.validate(&resources)?;
        let resources = Arc::new(resources);
        let materializer = Arc::new(StateDictMaterializer::new(options.max_parameter_bytes)?);
        let module = CpuGqaMoeModule::prepare_prefix(
            Arc::clone(&resources),
            materializer,
            options.precision,
            options.max_positions,
            options
                .active_layers
                .unwrap_or(resources.spec().layers().len()),
        )
        .map_err(|source| Error::Model {
            message: format!("cannot prepare CPU transformer module: {source}"),
        })?;
        let forward_executor = TransformerForwardExecutor::new(module);
        let model_info = standard_model_info(&resources, &options);
        let prepared_plan_id = PreparedDecoderGeneration::take()?.get();
        Self::from_runtime(DecoderComponents {
            resources,
            model_info,
            tokenizer,
            capabilities: options.capabilities,
            page_size: options.page_size,
            prepared_plan_id,
            forward_executor,
            backend,
            default_state: DecoderSequenceState::new((), ()),
            sequence_lifecycle: StandardSequenceLifecycle::new(),
            proposal: NoProposal,
            resource_manager: NoResourceManager,
            observer: StandardDecoderObserver,
            completion_hub,
            completion_reactors,
        })
    }
}
impl<D> GenericDecoderRunner<D>
where
    D: DecoderComposition,
{
    /// Constructs a runner from one complete model composition.
    pub fn from_runtime(components: DecoderComponents<D>) -> Result<Self> {
        let DecoderComponents {
            resources,
            model_info,
            tokenizer,
            capabilities,
            page_size,
            prepared_plan_id,
            forward_executor,
            backend,
            default_state,
            sequence_lifecycle,
            proposal,
            resource_manager,
            observer,
            completion_hub,
            completion_reactors,
        } = components;
        validate_capabilities(capabilities, usize::MAX)?;
        if page_size == 0 {
            return Err(runner_error("decoder KV page size must be non-zero"));
        }
        if prepared_plan_id == 0 {
            return Err(runner_error(
                "decoder prepared-plan identity must be non-zero",
            ));
        }
        let capacity = backend.capacity();
        if capacity.active_transactions != 0 {
            return Err(runner_error(
                "generic decoder backend already owns active transactions",
            ));
        }
        let control = DecoderControlPlane::new(completion_hub);
        let observability = GenericDecoderObservabilitySnapshot {
            backend: model_info.backend,
            bound_layers: model_info.num_layers,
            physical_kv_pages: capacity.physical_pages,
            resident_kv_pages: capacity.resident_pages,
            preempted_kv_pages: capacity.preempted_pages,
            free_kv_pages: capacity.free_pages,
            ..GenericDecoderObservabilitySnapshot::default()
        };
        let resolver = DecoderResolverHandle::default();
        let mut transactions =
            PackedTransactionRegistry::with_control("generic decoder", control.clone());
        transactions.set_resolver(resolver.clone())?;
        Ok(Self {
            resources,
            model_info,
            tokenizer,
            capabilities,
            page_size,
            forward_executor,
            backend,
            default_state,
            sequence_lifecycle,
            transactions,
            proposal_continuations: HashMap::new(),
            prepared_plan_id,
            control,
            completion_reactors,
            observability,
            proposal,
            resource_manager,
            resolver,
            observer,
            shutdown: false,
        })
    }
    pub const fn resources(&self) -> &D::Resources {
        &self.resources
    }
    pub const fn model(&self) -> GenericDecoderModelView<'_> {
        GenericDecoderModelView {
            tokenizer: &self.tokenizer,
        }
    }
    pub const fn forward_executor(&self) -> &D::ForwardExecutor {
        &self.forward_executor
    }
    pub fn backend(&self) -> &D::KvBackend {
        &self.backend
    }
    pub fn backend_mut(&mut self) -> &mut D::KvBackend {
        &mut self.backend
    }
    pub const fn default_state(&self) -> &D::SequenceState {
        &self.default_state
    }
    pub const fn proposal_executor(&self) -> &D::Proposal {
        &self.proposal
    }
    pub const fn resource_manager(&self) -> &D::ResourceManager {
        &self.resource_manager
    }
    #[cfg(test)]
    pub fn proposal_held_resume_lease_count(&self, continuation: ContinuationId) -> Result<usize> {
        self.proposal_continuations
            .get(&continuation)
            .map(|slot| slot.leases.len())
            .ok_or_else(|| {
                runner_error(format!(
                    "decoder has no native proposal continuation {}",
                    continuation.get()
                ))
            })
    }
    pub fn shutdown(&mut self) -> Result<()> {
        if !self.transactions.is_empty() || !self.proposal_continuations.is_empty() {
            return Err(runner_error(
                "cannot shut down generic decoder while transactions or proposals retain custody",
            ));
        }
        if self.shutdown {
            return Ok(());
        }
        DecoderForwardExecutor::shutdown(&mut self.forward_executor)?;
        self.resource_manager.shutdown()?;
        self.backend.shutdown()?;
        self.resolver.clear()?;
        self.control.completion_hub().close();
        self.shutdown = true;
        self.observability.shutdown = true;
        self.refresh_observability();
        Ok(())
    }
    fn ensure_running(&self) -> Result<()> {
        if self.shutdown {
            return Err(runner_error("generic decoder runner is shut down"));
        }
        Ok(())
    }
    fn ensure_sequence_available(&self, state: &D::SequenceState, operation: &str) -> Result<()> {
        self.transactions
            .ensure_sequence_available(state, operation)?;
        if let Some((continuation, slot)) = self
            .proposal_continuations
            .iter()
            .find(|(_, slot)| slot.topology_id == state.topology_id())
        {
            return Err(runner_error(format!(
                "cannot {operation}: decoder sequence topology {} is owned by proposal continuation {} in transaction {}",
                state.topology_id().get(),
                continuation.get(),
                slot.transaction.get()
            )));
        }
        Ok(())
    }
    fn lower_batch(
        &self,
        states: &[D::SequenceState],
        batch: &ExecutionBatch,
        reservations: &[KvReservationView],
    ) -> Result<PackedDecoderBatch> {
        PackedDecoderBatch::lower(
            batch,
            reservations,
            states,
            &self.capabilities,
            self.page_size,
            &|page| self.backend.page_status(page),
        )
    }
    fn map_progress(
        &mut self,
        transaction: ExecutionTransactionId,
        progress: DecoderTransactionProgress,
    ) -> Result<MultiSessionBatchProgress> {
        match progress {
            DecoderTransactionProgress::Complete(output) => {
                self.observability.completed_batches =
                    self.observability.completed_batches.saturating_add(1);
                Ok(MultiSessionBatchProgress::Complete(output))
            }
            DecoderTransactionProgress::Waiting { continuation, wait } => {
                self.observability.waits = self.observability.waits.saturating_add(1);
                let pending = match wait {
                    DecoderWait::Dependencies(dependencies) => {
                        PendingModelProgress::for_decoder_graph(
                            transaction,
                            continuation,
                            dependencies,
                        )?
                    }
                    DecoderWait::ResolvedStage(stage) => {
                        PendingModelProgress::for_resolved_stage(transaction, continuation, stage)?
                    }
                };
                Ok(MultiSessionBatchProgress::Waiting(pending))
            }
        }
    }
    fn refresh_observability(&mut self) {
        let capacity = self.backend.capacity();
        self.observability.physical_kv_pages = capacity.physical_pages;
        self.observability.resident_kv_pages = capacity.resident_pages;
        self.observability.preempted_kv_pages = capacity.preempted_pages;
        self.observability.free_kv_pages = capacity.free_pages;
        self.observability.active_transactions = self.transactions.len();
        self.observability.active_proposals = self.proposal_continuations.len();
    }
    fn validate_native_proposal(
        proposal: &NativeProposal,
        source: NativeProposalSource,
    ) -> Result<()> {
        proposal.validate_for_source(source)
    }
    fn insert_proposal_slot(
        &mut self,
        transaction: ExecutionTransactionId,
        topology_id: crate::execution::SequenceTopologyId,
        id: ContinuationId,
        continuation: D::ProposalContinuation,
        wait: Option<DecoderWait>,
        leases: Vec<ResidencyLeaseSet>,
    ) -> Result<Option<PendingModelProgress>> {
        let wait = match wait
            .map(|wait| DecoderContinuationWait::from_wait(id, wait))
            .transpose()
        {
            Ok(wait) => wait,
            Err(error) => {
                let replaced = self.proposal_continuations.insert(
                    id,
                    ProposalSlot {
                        transaction,
                        topology_id,
                        continuation,
                        wait: None,
                        leases,
                    },
                );
                debug_assert!(replaced.is_none(), "shared decoder continuation ID reused");
                self.refresh_observability();
                return Err(error);
            }
        };
        let pending = match wait
            .as_ref()
            .map(|wait| Self::proposal_pending(transaction, wait))
            .transpose()
        {
            Ok(pending) => pending,
            Err(error) => {
                let replaced = self.proposal_continuations.insert(
                    id,
                    ProposalSlot {
                        transaction,
                        topology_id,
                        continuation,
                        wait: None,
                        leases,
                    },
                );
                debug_assert!(replaced.is_none(), "shared decoder continuation ID reused");
                self.refresh_observability();
                return Err(error);
            }
        };
        let replaced = self.proposal_continuations.insert(
            id,
            ProposalSlot {
                transaction,
                topology_id,
                continuation,
                wait,
                leases,
            },
        );
        debug_assert!(replaced.is_none(), "shared decoder continuation ID reused");
        self.refresh_observability();
        Ok(pending)
    }
    fn proposal_pending(
        transaction: ExecutionTransactionId,
        wait: &DecoderContinuationWait,
    ) -> Result<PendingModelProgress> {
        match wait.wait() {
            DecoderWait::Dependencies(dependencies) => PendingModelProgress::for_decoder_graph(
                transaction,
                wait.continuation_id(),
                dependencies.clone(),
            ),
            DecoderWait::ResolvedStage(stage) => PendingModelProgress::for_resolved_stage(
                transaction,
                wait.continuation_id(),
                stage.clone(),
            ),
        }
    }
    fn cancel_proposals(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [D::SequenceState],
    ) -> Result<TransactionEndProgress> {
        loop {
            let Some(id) = self
                .proposal_continuations
                .iter()
                .find_map(|(id, slot)| (slot.transaction == transaction).then_some(*id))
            else {
                return Ok(TransactionEndProgress::Complete);
            };
            let mut slot = self
                .proposal_continuations
                .remove(&id)
                .expect("proposal continuation was selected above");
            let owns_default_state = self.default_state.topology_id() == slot.topology_id;
            let explicit_state_index = states
                .iter()
                .position(|state| state.topology_id() == slot.topology_id);
            if !owns_default_state && explicit_state_index.is_none() {
                let topology_id = slot.topology_id;
                self.proposal_continuations.insert(id, slot);
                return Err(runner_error(format!(
                    "proposal continuation {} transaction {} lost sequence topology {}",
                    id.get(),
                    transaction.get(),
                    topology_id.get()
                )));
            }
            let mut context = DecoderTransactionContext::with_services(
                transaction,
                self.control.clone(),
                std::mem::take(&mut slot.leases),
                false,
                Some(id),
                self.resolver.clone(),
            );
            let result = if owns_default_state {
                match self
                    .backend
                    .proposal_view(transaction, &mut self.default_state)
                {
                    Ok(mut kv) => self.proposal.cancel(
                        &mut context,
                        &mut self.default_state,
                        &mut kv,
                        &mut slot.continuation,
                    ),
                    Err(error) => Err(error),
                }
            } else {
                let state =
                    &mut states[explicit_state_index.expect("validated explicit proposal state")];
                match self.backend.proposal_view(transaction, state) {
                    Ok(mut kv) => {
                        self.proposal
                            .cancel(&mut context, state, &mut kv, &mut slot.continuation)
                    }
                    Err(error) => Err(error),
                }
            };
            slot.leases = context.into_leases();
            match result {
                Ok(DecoderCancelProgress::Waiting) => {
                    self.proposal_continuations.insert(id, slot);
                    self.refresh_observability();
                    return Ok(TransactionEndProgress::Pending);
                }
                Ok(DecoderCancelProgress::Complete) => {
                    drop(slot);
                    self.refresh_observability();
                }
                Err(error) => {
                    self.proposal_continuations.insert(id, slot);
                    self.refresh_observability();
                    return Err(error);
                }
            }
        }
    }
}
impl<D> ModelRunner for GenericDecoderRunner<D>
where
    D: DecoderComposition,
{
    fn model_info(&self) -> ModelInfo {
        self.model_info.clone()
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens)
    }
    fn reset_session(&mut self) -> Result<()> {
        self.ensure_running()?;
        self.ensure_sequence_available(&self.default_state, "reset the default session")?;
        self.sequence_lifecycle.reset(&mut self.default_state)
    }
    fn eos_token_id(&self) -> Option<u32> {
        self.tokenizer.eos_token_id()
    }
    fn eos_token_ids(&self) -> Vec<u32> {
        self.tokenizer.eos_token_ids().to_vec()
    }
    fn is_eos_token(&self, token_id: u32) -> bool {
        self.tokenizer.is_eos_token(token_id)
    }
    fn bound_layer_count(&self) -> Option<usize> {
        Some(self.model_info.num_layers)
    }
}
impl<D> MultiSessionRunner for GenericDecoderRunner<D>
where
    D: DecoderComposition,
{
    type SequenceState = D::SequenceState;
    fn sequence_generation(&self, state: &Self::SequenceState) -> u64 {
        state.core().generation()
    }
    fn prefix_cache_plan_identity(&self) -> u64 {
        self.prepared_plan_id
    }
    fn expert_residency_requirements(&self) -> Option<ExpertResidencyRequirements> {
        self.resource_manager.expert_residency_requirements()
    }
    fn expert_residency_control_installed(&self) -> bool {
        self.resource_manager.expert_residency_control_installed()
    }
    fn install_expert_residency_control(
        &mut self,
        control: Box<dyn ExpertResidencyControl>,
    ) -> Result<()> {
        if !self.transactions.is_empty() || !self.proposal_continuations.is_empty() {
            return Err(runner_error(
                "cannot replace decoder residency control while transactions are active",
            ));
        }
        self.resource_manager
            .install_expert_residency_control(control)
    }
    fn take_materialization_provider(&mut self) -> Option<Box<dyn MaterializationProvider>> {
        self.resource_manager.take_provider()
    }
    fn take_warmup_requests(&mut self) -> Result<Vec<MaterializationRequest>> {
        self.resource_manager.take_warmup_requests()
    }
    fn transaction_prefetch_requests(
        &self,
        transaction: ExecutionTransactionId,
        batch: &ExecutionBatch,
    ) -> Result<Vec<MaterializationRequest>> {
        self.resource_manager
            .transaction_prefetch_requests(transaction, batch)
    }
    fn materialization_resolver_installed(&self) -> bool {
        self.resolver.installed()
    }
    fn install_materialization_resolver(
        &mut self,
        resolver: Box<dyn MaterializationResolver>,
    ) -> Result<()> {
        if !self.transactions.is_empty() || !self.proposal_continuations.is_empty() {
            return Err(runner_error(
                "cannot replace decoder resolver while transactions are active",
            ));
        }
        self.resolver
            .install(resolver, self.resource_manager.materialization_placement())
    }
    fn materialization_resolver(&mut self) -> Result<&mut (dyn MaterializationResolver + '_)> {
        if !self.resolver.installed() {
            return Err(runner_error(
                "decoder materialization resolver is not installed",
            ));
        }
        Ok(&mut self.resolver)
    }
    fn with_sequence_state<T>(
        &mut self,
        state: &mut Self::SequenceState,
        execute: impl FnOnce(&mut Self) -> Result<T>,
    ) -> Result<T> {
        self.ensure_running()?;
        self.ensure_sequence_available(state, "execute explicit sequence work")?;
        self.ensure_sequence_available(&self.default_state, "swap the default sequence")?;
        std::mem::swap(&mut self.default_state, state);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(self)));
        std::mem::swap(&mut self.default_state, state);
        match result {
            Ok(result) => result,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }
    fn create_sequence_state(&mut self) -> Result<Self::SequenceState> {
        self.ensure_running()?;
        self.sequence_lifecycle.create()
    }
    fn fork_sequence_state(&mut self) -> Result<Self::SequenceState> {
        self.ensure_running()?;
        self.ensure_sequence_available(&self.default_state, "fork the default sequence")?;
        self.sequence_lifecycle
            .logical_fork(&self.default_state, self.default_state.core().position())
    }
    fn fork_sequence_state_from(
        &mut self,
        source: &Self::SequenceState,
        expected_position: usize,
    ) -> Result<Self::SequenceState> {
        self.ensure_running()?;
        self.ensure_sequence_available(source, "fork an explicit sequence")?;
        self.sequence_lifecycle
            .logical_fork(source, expected_position)
    }
    fn reset_sequence_state(&mut self, state: &mut Self::SequenceState) -> Result<()> {
        self.ensure_running()?;
        self.ensure_sequence_available(state, "reset sequence state")?;
        self.sequence_lifecycle.reset(state)
    }
    fn try_release_sequence_state(
        &mut self,
        state: Self::SequenceState,
    ) -> std::result::Result<(), SequenceStateReleaseError<Self::SequenceState>> {
        if let Err(error) = self
            .ensure_running()
            .and_then(|()| self.ensure_sequence_available(&state, "release sequence state"))
        {
            return Err(SequenceStateReleaseError::new(error, state));
        }
        self.sequence_lifecycle.try_release(state)
    }
    fn configure_kv_page_capacity(&mut self, max_pages: usize) -> Result<()> {
        self.ensure_running()?;
        if !self.transactions.is_empty() {
            return Err(runner_error(
                "cannot configure decoder KV capacity while transactions are active",
            ));
        }
        self.backend.configure_capacity(max_pages)?;
        self.refresh_observability();
        Ok(())
    }
    fn release_kv_pages(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.ensure_running()?;
        self.transactions.ensure_pages_available(pages, "release")?;
        self.backend.release(pages)?;
        self.refresh_observability();
        Ok(())
    }
    fn preempt_kv_pages(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.ensure_running()?;
        self.transactions.ensure_pages_available(pages, "preempt")?;
        self.backend.preempt(pages)?;
        self.refresh_observability();
        Ok(())
    }
    fn restore_kv_pages(&mut self, pages: &[KvPageId]) -> Result<()> {
        self.ensure_running()?;
        self.transactions.ensure_pages_available(pages, "restore")?;
        self.backend.restore(pages)?;
        self.refresh_observability();
        Ok(())
    }
    fn prepare_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
        kv_reservations: &[KvReservationView],
    ) -> Result<()> {
        self.ensure_running()?;
        let lowered = self.lower_batch(states, batch, kv_reservations)?;
        for sequence in lowered.sequences() {
            let state = &states[sequence.state_index()];
            self.ensure_sequence_available(state, "prepare packed decoder work")?;
        }
        self.transactions.prepare_with_lifecycle(
            transaction,
            lowered,
            states,
            &mut self.backend,
            &mut self.sequence_lifecycle,
        )?;
        self.refresh_observability();
        Ok(())
    }
    fn end_transaction(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        intent: TransactionEndIntent,
    ) -> Result<TransactionEndProgress> {
        self.ensure_running()?;
        match intent {
            TransactionEndIntent::Publish => {
                if self
                    .proposal_continuations
                    .values()
                    .any(|slot| slot.transaction == transaction)
                {
                    return Err(runner_error(format!(
                        "cannot publish decoder transaction {} with active proposal continuations",
                        transaction.get()
                    )));
                }
            }
            TransactionEndIntent::Abort => {
                if self.cancel_proposals(transaction, states)? == TransactionEndProgress::Pending {
                    return Ok(TransactionEndProgress::Pending);
                }
                if !self.transactions.contains(transaction) {
                    self.observability.aborts = self.observability.aborts.saturating_add(1);
                    self.refresh_observability();
                    return Ok(TransactionEndProgress::Complete);
                }
            }
        }
        let progress = match intent {
            TransactionEndIntent::Publish => {
                self.transactions
                    .publish(transaction, states, &mut self.backend)?
            }
            TransactionEndIntent::Abort => self.transactions.abort(
                transaction,
                &mut self.backend,
                &mut self.forward_executor,
            )?,
        };
        let mapped = match progress {
            KvEndProgress::Pending => TransactionEndProgress::Pending,
            KvEndProgress::Complete => {
                match intent {
                    TransactionEndIntent::Publish => {
                        self.observability.commits = self.observability.commits.saturating_add(1)
                    }
                    TransactionEndIntent::Abort => {
                        self.observability.aborts = self.observability.aborts.saturating_add(1)
                    }
                }
                TransactionEndProgress::Complete
            }
            KvEndProgress::ConsumedRejected => {
                self.refresh_observability();
                return Err(runner_error(format!(
                    "decoder transaction {} terminal KV transaction was consumed and rejected during {intent:?}",
                    transaction.get()
                )));
            }
        };
        self.refresh_observability();
        Ok(mapped)
    }
    fn retain_provisional_prefixes(
        &mut self,
        transaction: ExecutionTransactionId,
        sources: &[Self::SequenceState],
        branches: &mut [Self::SequenceState],
        executed_rows: &[usize],
        retained_rows: &[usize],
    ) -> Result<()> {
        self.ensure_running()?;
        self.transactions.retain_provisional(
            transaction,
            sources,
            branches,
            executed_rows,
            retained_rows,
            &mut self.backend,
            &mut self.proposal,
        )
    }
    fn execute_multi_session_batch_progress(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
    ) -> Result<MultiSessionBatchProgress> {
        self.ensure_running()?;
        let progress = self.transactions.execute_batch(
            transaction,
            states,
            batch,
            &mut self.backend,
            &mut self.forward_executor,
        )?;
        self.map_progress(transaction, progress)
    }
    fn resume_multi_session_batch(
        &mut self,
        transaction: ExecutionTransactionId,
        states: &mut [Self::SequenceState],
        batch: &ExecutionBatch,
        continuation: ContinuationId,
        leases: ResidencyLeaseSet,
    ) -> Result<MultiSessionBatchProgress> {
        self.ensure_running()?;
        let progress = self.transactions.resume_batch(
            transaction,
            states,
            batch,
            continuation,
            leases,
            &mut self.backend,
            &mut self.forward_executor,
        )?;
        self.observability.resumes = self.observability.resumes.saturating_add(1);
        self.map_progress(transaction, progress)
    }
    fn multi_session_capabilities(&self) -> ExecutionCapabilities {
        self.capabilities
    }
}
impl<D> ResidentModelRunner for GenericDecoderRunner<D>
where
    D: DecoderComposition,
{
    type ObservabilitySnapshot = D::Snapshot;
    fn observability_snapshot(&self) -> Self::ObservabilitySnapshot {
        let mut core = self.observability.clone();
        core.active_transactions = self.transactions.len();
        core.active_proposals = self.proposal_continuations.len();
        let mut resources = self.resource_manager.observer_snapshot();
        resources.resolver_installed = self.resolver.installed();
        self.observer.snapshot(&core, &self.backend, &resources)
    }
    fn completion_hub(&self) -> CompletionHub {
        self.control.completion_hub().clone()
    }
    fn take_completion_reactors(&mut self) -> Vec<ModelCompletionReactor> {
        std::mem::take(&mut self.completion_reactors)
    }
    fn native_proposal_source(&self) -> Result<Option<NativeProposalSource>> {
        self.proposal.source()
    }
    fn begin_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        anchor_token_id: u32,
    ) -> Result<NativeProposalProgress> {
        self.ensure_running()?;
        let source = self.proposal.source()?.ok_or_else(|| {
            runner_error(format!(
                "decoder transaction {} requested a native proposal from a target-only composition",
                transaction.get()
            ))
        })?;
        source.validate()?;
        self.ensure_sequence_available(&self.default_state, "begin a native proposal")?;
        let topology_id = self.default_state.topology_id();
        if self
            .proposal_continuations
            .values()
            .any(|slot| slot.topology_id == topology_id)
        {
            return Err(runner_error(format!(
                "decoder sequence topology {} already owns a native proposal continuation",
                topology_id.get()
            )));
        }
        let continuation_id = self.control.continuations().allocate()?;
        let mut context = DecoderTransactionContext::with_services(
            transaction,
            self.control.clone(),
            Vec::new(),
            false,
            Some(continuation_id),
            self.resolver.clone(),
        );
        let mut kv = self
            .backend
            .proposal_view(transaction, &mut self.default_state)?;
        let progress = self.proposal.start(
            &mut context,
            &mut self.default_state,
            &mut kv,
            anchor_token_id,
        );
        let leases = context.into_leases();
        match progress {
            Ok(DecoderProposalProgress::Waiting { continuation, wait }) => {
                let pending = self
                    .insert_proposal_slot(
                        transaction,
                        topology_id,
                        continuation_id,
                        continuation,
                        Some(wait),
                        leases,
                    )?
                    .expect("waiting proposal has dependency progress");
                self.observability.proposal_waits =
                    self.observability.proposal_waits.saturating_add(1);
                Ok(NativeProposalProgress::Waiting(pending))
            }
            Ok(DecoderProposalProgress::Complete(proposal)) => {
                drop(leases);
                Self::validate_native_proposal(&proposal, source)?;
                Ok(NativeProposalProgress::Complete(proposal))
            }
            Ok(DecoderProposalProgress::FailedActive {
                continuation,
                error,
            }) => {
                self.insert_proposal_slot(
                    transaction,
                    topology_id,
                    continuation_id,
                    continuation,
                    None,
                    leases,
                )?;
                Err(error)
            }
            Ok(DecoderProposalProgress::FailedQuiescent(error)) | Err(error) => {
                drop(leases);
                Err(error)
            }
        }
    }
    fn resume_native_proposal(
        &mut self,
        transaction: ExecutionTransactionId,
        continuation_id: ContinuationId,
        leases: ResidencyLeaseSet,
    ) -> Result<NativeProposalProgress> {
        self.ensure_running()?;
        let source = self.proposal.source()?.ok_or_else(|| {
            runner_error("target-only decoder has no native proposal continuation")
        })?;
        let mut slot = self
            .proposal_continuations
            .remove(&continuation_id)
            .ok_or_else(|| {
                runner_error(format!(
                    "decoder has no native proposal continuation {}",
                    continuation_id.get()
                ))
            })?;
        if slot.transaction != transaction || slot.topology_id != self.default_state.topology_id() {
            let owner_transaction = slot.transaction;
            let owner_topology = slot.topology_id;
            self.proposal_continuations.insert(continuation_id, slot);
            return Err(runner_error(format!(
                "proposal continuation {} belongs to transaction {}/topology {}, not {}/{}",
                continuation_id.get(),
                owner_transaction.get(),
                owner_topology.get(),
                transaction.get(),
                self.default_state.topology_id().get()
            )));
        }
        let Some(wait) = slot.wait.as_ref() else {
            self.proposal_continuations.insert(continuation_id, slot);
            return Err(runner_error(format!(
                "proposal continuation {} failed active and must be cancelled",
                continuation_id.get()
            )));
        };
        if let Err(error) = wait.validate_resume_leases(&leases) {
            self.proposal_continuations.insert(continuation_id, slot);
            return Err(error);
        }
        slot.leases.push(leases);
        let mut context = DecoderTransactionContext::with_services(
            transaction,
            self.control.clone(),
            std::mem::take(&mut slot.leases),
            true,
            Some(continuation_id),
            self.resolver.clone(),
        );
        let progress = match self
            .backend
            .proposal_view(transaction, &mut self.default_state)
        {
            Ok(mut kv) => self.proposal.resume(
                &mut context,
                &mut self.default_state,
                &mut kv,
                &mut slot.continuation,
            ),
            Err(error) => {
                slot.leases = context.into_leases();
                self.proposal_continuations.insert(continuation_id, slot);
                self.refresh_observability();
                return Err(error);
            }
        };
        slot.leases = context.into_leases();
        self.observability.proposal_resumes = self.observability.proposal_resumes.saturating_add(1);
        match progress {
            Ok(DecoderProposalResumeProgress::Waiting(wait)) => {
                let wait = match DecoderContinuationWait::from_wait(continuation_id, wait) {
                    Ok(wait) => wait,
                    Err(error) => {
                        slot.wait = None;
                        self.proposal_continuations.insert(continuation_id, slot);
                        self.refresh_observability();
                        return Err(error);
                    }
                };
                let pending = match Self::proposal_pending(transaction, &wait) {
                    Ok(pending) => pending,
                    Err(error) => {
                        slot.wait = None;
                        self.proposal_continuations.insert(continuation_id, slot);
                        self.refresh_observability();
                        return Err(error);
                    }
                };
                slot.wait = Some(wait);
                self.proposal_continuations.insert(continuation_id, slot);
                self.observability.proposal_waits =
                    self.observability.proposal_waits.saturating_add(1);
                self.refresh_observability();
                Ok(NativeProposalProgress::Waiting(pending))
            }
            Ok(DecoderProposalResumeProgress::Complete(proposal)) => {
                drop(slot);
                Self::validate_native_proposal(&proposal, source)?;
                self.refresh_observability();
                Ok(NativeProposalProgress::Complete(proposal))
            }
            Ok(DecoderProposalResumeProgress::FailedActive(error)) | Err(error) => {
                slot.wait = None;
                self.proposal_continuations.insert(continuation_id, slot);
                self.refresh_observability();
                Err(error)
            }
            Ok(DecoderProposalResumeProgress::FailedQuiescent(error)) => {
                drop(slot);
                self.refresh_observability();
                Err(error)
            }
        }
    }
}
fn validate_capabilities(capabilities: ExecutionCapabilities, max_positions: usize) -> Result<()> {
    if capabilities.max_sequences == 0
        || capabilities.max_batch_tokens == 0
        || capabilities.max_prefill_query_tokens_per_sequence == 0
        || capabilities.max_decode_query_tokens_per_sequence == 0
    {
        return Err(runner_error(
            "decoder execution capabilities must admit sequences and tokens",
        ));
    }
    if capabilities.kv_binding_mode != KvBindingMode::Paged {
        return Err(runner_error(
            "generic decoder requires authoritative paged KV bindings",
        ));
    }
    if !capabilities.supports_prefill || !capabilities.supports_decode {
        return Err(runner_error(
            "generic decoder options must support both prefill and decode",
        ));
    }
    if capabilities.max_prefill_query_tokens_per_sequence > max_positions
        || capabilities.max_decode_query_tokens_per_sequence > max_positions
    {
        return Err(runner_error(
            "decoder query capability exceeds the prepared position table",
        ));
    }
    Ok(())
}
fn standard_model_info(
    resources: &BoundDecoderResources,
    options: &GenericDecoderOptions,
) -> ModelInfo {
    let spec = resources.spec();
    let mut num_experts = 0;
    let mut num_experts_per_tok = 0;
    for layer in spec.layers() {
        if let FeedForward::Moe(moe) = layer.feed_forward() {
            num_experts = num_experts.max(moe.router_spec().num_experts());
            num_experts_per_tok = num_experts_per_tok.max(moe.router_spec().experts_per_token());
        }
    }
    ModelInfo {
        family: options.family.clone(),
        architecture: Some(spec.architecture().to_string()),
        attention: AttentionKind::GroupedQuery,
        weight_source: options.weight_source,
        hidden_size: spec.hidden_size(),
        num_layers: options.active_layers.unwrap_or(spec.layers().len()),
        num_experts,
        num_experts_per_tok,
        vocab_size: spec.vocab_size(),
        backend: "cpu",
    }
}
fn runner_error(message: impl Into<String>) -> Error {
    Error::Execution {
        message: message.into(),
    }
}
