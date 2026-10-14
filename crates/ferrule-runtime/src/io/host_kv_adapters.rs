//! Memory-source admission seam. This does not allocate buffers, touch the
//! checkpoint registry, or expose a physical-completion success path.
use super::RuntimeMaterializationProvider;
use crate::cache::{CommittedKvReadLease, KvPageManager};
use ferrule_common::host_kv::placement::{
    HostKvPlacementHandle, HostKvPlacementOwner, PlacementBudget, PlacementCredits,
};
use ferrule_common::host_kv::{
    CommittedKvSource, HostKvError, HostKvFence, HostKvPlacementState, HostKvPlan, HostKvResult,
    HostKvStage,
};

/// Classifies only the returned value. `Rejected` is NOT no-reference evidence:
/// safe provider implementations can return it after retaining work. Only the
/// explicit unsafe integration boundary may rely on the stronger contract.
#[derive(Debug, PartialEq, Eq)]
enum HostKvSubmission {
    Rejected(HostKvError),
    Accepted,
    Unknown,
}
impl From<HostKvResult<()>> for HostKvSubmission {
    fn from(result: HostKvResult<()>) -> Self {
        match result {
            Ok(()) => Self::Accepted,
            Err(HostKvError::Unknown) => Self::Unknown,
            Err(error) => Self::Rejected(error),
        }
    }
}

/// Owns one logical read pin and six-domain charge until explicit cancellation.
/// Accepted/Unknown commands deliberately cannot be released through this seam:
/// a future backend adapter must supply opaque physical quiescence evidence.
#[derive(Debug)]
pub struct ReservedHostKvPlacement {
    owner: HostKvPlacementOwner<CommittedKvReadLease>,
    handle: HostKvPlacementHandle,
    source: CommittedKvSource,
    credits: PlacementCredits,
}
impl ReservedHostKvPlacement {
    pub fn reserve(
        manager: &mut KvPageManager,
        source: CommittedKvSource,
        credits: PlacementCredits,
        budget: PlacementBudget,
    ) -> HostKvResult<Self> {
        let mut owner = HostKvPlacementOwner::new(budget)?;
        let handle = owner.reserve(manager, &source, credits)?;
        Ok(Self {
            owner,
            handle,
            source,
            credits,
        })
    }
    pub fn state(&self) -> HostKvResult<HostKvPlacementState> {
        self.owner.state(self.handle)
    }
    /// Safe provider calls do not certify rejection. Any returned error retains
    /// the logical pin/charge as sticky Unknown; the original error is reported.
    /// Use `submit_trusted` only at an audited provider integration boundary.
    pub fn submit<P: RuntimeMaterializationProvider + ?Sized>(
        &mut self,
        manager: &KvPageManager,
        provider: &mut P,
        generation: u64,
        stream: u64,
        event: u64,
    ) -> HostKvResult<()> {
        self.submit_with_trust(manager, provider, (generation, stream, event), false)
    }
    /// Submit using a provider whose no-retained-reference rejection contract
    /// has been audited for this exact operation and backend integration.
    /// Acceptance still isn't completion and Unknown never becomes rejection.
    ///
    /// # Safety
    /// The caller must ensure every non-Unknown error from this provider means
    /// no work/queue item/DMA/consumer/callback or buffer reference remains for
    /// this exact plan/source. Partial submission must return Unknown unless
    /// ALL physical work/references were positively retired before returning.
    /// A safe `RuntimeMaterializationProvider` implementation or error variant
    /// alone does not establish this guarantee. Do not use an untrusted provider
    /// or adapt a generic backend error into a no-submit assertion.
    ///
    /// ```compile_fail,E0133
    /// use ferrule_runtime::{KvPageManager, io::{RuntimeMaterializationProvider, host_kv_adapters::ReservedHostKvPlacement}};
    /// fn unreviewed<P: RuntimeMaterializationProvider>(r: &mut ReservedHostKvPlacement, m: &KvPageManager, p: &mut P) {
    ///     r.submit_trusted(m, p, 1, 0, 1).unwrap();
    /// }
    /// ```
    #[expect(
        unsafe_code,
        reason = "Audited provider no-reference attestation; see Safety contract"
    )]
    pub unsafe fn submit_trusted<P: RuntimeMaterializationProvider + ?Sized>(
        &mut self,
        manager: &KvPageManager,
        provider: &mut P,
        generation: u64,
        stream: u64,
        event: u64,
    ) -> HostKvResult<()> {
        self.submit_with_trust(manager, provider, (generation, stream, event), true)
    }
    fn submit_with_trust<P: RuntimeMaterializationProvider + ?Sized>(
        &mut self,
        manager: &KvPageManager,
        provider: &mut P,
        identity: (u64, u64, u64),
        trusted_rejection: bool,
    ) -> HostKvResult<()> {
        let (generation, stream, event) = identity;
        let fence = HostKvFence {
            owner: self.handle.owner(),
            operation: self.handle.operation(),
            generation,
            stage: HostKvStage::HostCopy,
            stream,
            event,
        };
        self.owner.submit(manager, self.handle, fence)?;
        let plan = HostKvPlan {
            source: self.source.clone(),
            credits: self.credits,
            fence,
        };
        // The registration precedes invocation. A caught provider unwind must
        // not leave a pending operation eligible for a future rejection retry.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            provider.submit_host_kv_plan(&plan)
        }));
        let result = match result {
            Ok(result) => result,
            Err(panic) => {
                let _ = self.owner.mark_unknown(self.handle);
                std::panic::resume_unwind(panic);
            }
        };
        match HostKvSubmission::from(result) {
            HostKvSubmission::Accepted => Ok(()),
            HostKvSubmission::Unknown => {
                self.owner.mark_unknown(self.handle)?;
                Err(HostKvError::Unknown)
            }
            HostKvSubmission::Rejected(error) => {
                if trusted_rejection {
                    // SAFETY: only submit_trusted selects this branch. Its
                    // caller certifies this provider's exact no-reference Err
                    // contract; generic safe submissions can never arrive here.
                    #[expect(
                        unsafe_code,
                        reason = "Only the audited submit_trusted caller certifies this rejection"
                    )]
                    unsafe {
                        self.owner.reject_submission(self.handle, fence)?;
                    }
                } else {
                    self.owner.mark_unknown(self.handle)?;
                }
                Err(error)
            }
        }
    }
    pub fn cancel(&mut self, manager: &mut KvPageManager) -> HostKvResult<()> {
        self.owner.cancel(manager, self.handle)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/io/host_kv_adapters.rs"]
mod tests;

/// Live memory-source adapter. It bypasses checkpoint providers and LoadRegistry;
/// logical pins remain owned here until the exact physical operation is reaped.
/// `begin` explicitly establishes the supplied manager/backend pair on first use;
/// callers must supply their real decoder composition, not infer it from page IDs.
/// This is a quiescent committed-snapshot operation, not ordinary append-attend:
/// it does not evict resident pages or publish a decoder frontier. Integrating it
/// with append-attend requires a transaction-derived physical read capability and
/// mergeable prefix/tail partials; this adapter must not be used as a bypass.
#[cfg(feature = "cuda")]
mod live {
    use super::*;
    use ferrule_backend::cuda::operators::kv::host_placement::CommittedGqaProgress;
    use ferrule_common::{Error, Result};
    use ferrule_model::decoder::{
        DecoderKvSequenceCustody, DecoderSequence, PagedKvBackend, TypedCudaPagedKvPool,
    };
    use ferrule_model::transformer::{
        CommittedGqa, CommittedGqaOutput, CommittedGqaRequest, CudaStandardDecoderOperators,
    };

    #[doc = include_str!("../../tests/compile/host_kv/live_custody.md")]
    #[must_use = "finish or cancel and reap; drop retains logical/physical custody"]
    pub struct LiveCommittedGqa<'a, S: DecoderSequence> {
        manager: &'a mut KvPageManager,
        lease: Option<CommittedKvReadLease>,
        operation: CommittedGqa,
        completed: Option<CommittedGqaOutput>,
        _backend: &'a mut PagedKvBackend<TypedCudaPagedKvPool<S>>,
        _sequence: &'a mut S,
    }
    fn logical_error(error: HostKvError) -> Error {
        Error::Execution {
            message: error.to_string(),
        }
    }
    impl<'a, S: DecoderSequence> LiveCommittedGqa<'a, S> {
        pub fn begin(
            manager: &'a mut KvPageManager,
            operators: &mut CudaStandardDecoderOperators,
            backend: &'a mut PagedKvBackend<TypedCudaPagedKvPool<S>>,
            sequence: &'a mut S,
            custody: &DecoderKvSequenceCustody,
            source: CommittedKvSource,
            request: CommittedGqaRequest,
            budget: &PlacementBudget,
        ) -> Result<Self> {
            let lease = manager
                .acquire_committed_read(&source)
                .map_err(logical_error)?;
            if let Err(error) = backend.bind_committed_logical_owner(lease.source().owner) {
                manager
                    .release_committed_read(&lease)
                    .map_err(logical_error)?;
                return Err(error);
            }
            let query_position = request.query_position;
            let cpu_tokens = request.cpu_tokens;
            let operation = match operators.committed_gqa(
                backend,
                sequence,
                custody,
                lease.source(),
                request,
                budget,
            ) {
                Ok(operation) => operation,
                Err(error) => {
                    // begin has submitted no DMA; ordinary rejection is retryable.
                    manager
                        .release_committed_read(&lease)
                        .map_err(logical_error)?;
                    return Err(error);
                }
            };
            tracing::debug!(
                target: "ferrule::host_kv",
                event = "committed_gqa_begin",
                owner = lease.source().owner,
                state_slot = lease.source().state_slot.get(),
                generation = lease.source().generation,
                incarnation = lease.source().incarnation,
                positions_start = lease.source().positions.start,
                positions_end = lease.source().positions.end,
                pages = lease.source().pages.len(),
                query_position,
                cpu_tokens,
                budget_in_use = ?budget.in_use(),
                "started quiescent committed GQA"
            );
            Ok(Self {
                manager,
                lease: Some(lease),
                operation,
                completed: None,
                _backend: backend,
                _sequence: sequence,
            })
        }
        pub fn poll(&mut self) -> Result<CommittedGqaProgress> {
            self.manager
                .validate_committed_read(
                    self.lease
                        .as_ref()
                        .ok_or_else(|| logical_error(HostKvError::UnknownLease))?,
                )
                .map_err(logical_error)?;
            let was_quarantined = self.operation.is_quarantined();
            let result = self.operation.poll();
            if !was_quarantined && self.operation.is_quarantined() {
                let source = self.lease.as_ref().expect("validated lease").source();
                tracing::debug!(
                    target: "ferrule::host_kv",
                    event = "committed_gqa_unknown",
                    owner = source.owner,
                    state_slot = source.state_slot.get(),
                    "physical completion unknown; logical pin and placement charge retained"
                );
            }
            result
        }
        pub fn cancel(&mut self) {
            if let Some(lease) = self.lease.as_ref() {
                tracing::debug!(
                    target: "ferrule::host_kv",
                    event = "committed_gqa_cancel",
                    owner = lease.source().owner,
                    state_slot = lease.source().state_slot.get(),
                    "cancellation requested; custody retained"
                );
            }
            self.operation.cancel();
        }
        pub fn reap(&mut self) -> Result<()> {
            let result = self.operation.reap();
            if let Some(lease) = self.lease.as_ref() {
                tracing::debug!(
                    target: "ferrule::host_kv",
                    event = "committed_gqa_reap",
                    owner = lease.source().owner,
                    state_slot = lease.source().state_slot.get(),
                    success = result.is_ok(),
                    "reap attempted across DMA and consumer custody"
                );
            }
            result
        }
        pub fn is_quarantined(&self) -> bool {
            self.operation.is_quarantined()
        }
        /// A failed logical release keeps the opaque result for retry. No public
        /// HostKvCompletion tuple can release this operation or its lease.
        pub fn finish(&mut self) -> Result<CommittedGqaOutput> {
            let lease = self
                .lease
                .as_ref()
                .ok_or_else(|| logical_error(HostKvError::UnknownLease))?;
            self.manager
                .validate_committed_read(lease)
                .map_err(logical_error)?;
            if self.completed.is_none() {
                self.completed = Some(self.operation.finish()?);
            }
            if self
                .completed
                .as_ref()
                .expect("physical receipt")
                .receipt()
                .source()
                != lease.source()
            {
                return Err(logical_error(HostKvError::StaleSource));
            }
            self.manager
                .release_committed_read(lease)
                .map_err(logical_error)?;
            self.lease = None;
            let output = self.completed.take().expect("physically completed output");
            tracing::debug!(
                target: "ferrule::host_kv",
                event = "committed_gqa_finish",
                owner = output.receipt().source().owner,
                state_slot = output.receipt().source().state_slot.get(),
                operation = output.receipt().operation(),
                canceled = output.receipt().canceled(),
                cpu_chunks = output.receipt().cpu_chunks(),
                gpu_chunks = output.receipt().gpu_chunks(),
                output_charge_retained = !output.receipt().canceled(),
                "finished committed GQA and released logical custody"
            );
            Ok(output)
        }
    }
}
#[cfg(feature = "cuda")]
pub use live::LiveCommittedGqa;

/// The only production logical-authority implementation for transaction append.
#[cfg(feature = "cuda")]
mod append {
    use super::AppendGqaBinding;
    use crate::cache::kv_read_authority::ReservedAppendPrefix;
    use ferrule_backend::cuda::operators::kv::transaction::append_bridge::AppendPrefixAuthority;
    use ferrule_common::{Error, Result};
    use ferrule_model::decoder::{DecoderSequence, PagedKvBackend, TypedCudaPagedKvPool};
    use ferrule_model::transformer::{
        CudaStandardDecoderOperators, CudaStandardKvBinding, GqaRequest, OperatorProgress, Rows,
        StandardDecoderOperators,
    };

    // SAFETY: this type borrows the real manager and non-cloneable reservation;
    // validate re-derives the source/ABA/COW projection from that same authority.
    // Neither borrow can be detached by the backend or replaced by metadata.
    #[expect(
        unsafe_code,
        reason = "Retains and revalidates the live manager/reservation borrow; audited below"
    )]
    unsafe impl AppendPrefixAuthority for ReservedAppendPrefix<'_> {
        fn witness(&self) -> &ferrule_common::host_kv::HostKvAppendReadWitness {
            self.witness()
        }
        fn layer(&self) -> usize {
            self.layer()
        }
        fn validate(&self) -> Result<()> {
            self.validate().map_err(|e| Error::Execution {
                message: e.to_string(),
            })
        }
    }
    impl ReservedAppendPrefix<'_> {
        /// Opt-in prepare for the ordinary runner. Bind the actual owner pair
        /// before entering the backend transaction; this does not publish KV.
        pub fn prepare_cuda_decode(
            &self,
            runner: &mut ferrule_model::decoder::GenericDecoderRunner<
                ferrule_model::decoder::HybridCudaDecoder,
            >,
            states: &mut [ferrule_model::decoder::CudaHybridSequenceState],
            batch: &ferrule_common::execution::ExecutionBatch,
        ) -> Result<()> {
            use ferrule_model::MultiSessionRunner;
            AppendPrefixAuthority::validate(self)?;
            if batch.mode() != ferrule_common::execution::ForwardMode::Decode
                || batch.len() != 1
                || batch.sequences().len() != 1
            {
                return Err(Error::Execution {
                    message:
                        "HostKV requires single-token single-sequence decode; no prefill fallback"
                            .into(),
                });
            }
            runner.bind_host_kv_owner(self.witness().source().owner)?;
            runner.prepare_multi_session_batch(
                self.witness().transaction(),
                states,
                batch,
                std::slice::from_ref(self.witness().reservation()),
            )
        }
        /// Runs the existing prepared runner forward/layer loop with all GQA
        /// layer authorities borrowed from this ONE live reservation. No layer
        /// loop, copied authorization or independent transaction lives here.
        /// On error the caller must retain the reservation and abort the backend
        /// before logical retirement; Unknown is not an abort acknowledgment.
        pub fn execute_cuda_decode(
            &self,
            runner: &mut ferrule_model::decoder::GenericDecoderRunner<
                ferrule_model::decoder::HybridCudaDecoder,
            >,
            states: &mut [ferrule_model::decoder::CudaHybridSequenceState],
            batch: &ferrule_common::execution::ExecutionBatch,
            budget: &ferrule_common::host_kv::placement::PlacementBudget,
            block_tokens: usize,
        ) -> Result<ferrule_model::MultiSessionBatchProgress> {
            AppendPrefixAuthority::validate(self)?;
            let layers = self.witness().source().schema.planes()[0].layer_count;
            let authorities = (0..layers)
                .map(|layer| {
                    self.for_layer(layer).map_err(|e| Error::Execution {
                        message: e.to_string(),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let context = ferrule_model::decoder::CudaHostKvAppendContext::new(
                authorities
                    .iter()
                    .map(|a| a as &dyn AppendPrefixAuthority)
                    .collect(),
                budget,
                block_tokens,
            )?;
            runner.execute_host_kv_decode(self.witness().transaction(), states, batch, &context)
        }
        /// Establish the exact manager/backend pair while physical custody is
        /// quiescent, BEFORE preparing this append transaction. Not a lazy rebind.
        pub fn bind_cuda_pool<S: DecoderSequence>(
            &self,
            backend: &mut PagedKvBackend<TypedCudaPagedKvPool<S>>,
        ) -> Result<()> {
            AppendPrefixAuthority::validate(self)?;
            backend.bind_committed_logical_owner(self.witness().source().owner)
        }
        /// Executes the real model append/attend hook and holds its output grant
        /// through a scoped downstream consumer. The callback must enqueue all
        /// output consumers on this owner and must not retain output aliases.
        /// Cancellation/consumer failure is fail-stop, never a cleanup ACK.
        pub fn with_cuda_append(
            &self,
            binding: AppendGqaBinding<'_>,
            operators: &mut CudaStandardDecoderOperators,
            request: GqaRequest<'_>,
            consume: impl FnOnce(&Rows) -> Result<()>,
        ) -> Result<()> {
            let mut kv = CudaStandardKvBinding::with_append_prefix(
                binding.view,
                binding.batch,
                self,
                binding.budget,
                binding.block_tokens,
            )?;
            let output = match operators.paged_gqa(&mut kv, request)? {
                OperatorProgress::Ready(rows) => rows,
                _ => {
                    kv.cancel();
                    return Err(Error::Execution {
                        message: "append GQA did not return synchronous output".into(),
                    });
                }
            };
            if let Err(e) = consume(&output) {
                kv.cancel();
                return Err(e);
            }
            kv.return_output(output)
        }
    }
}

#[cfg(feature = "cuda")]
pub struct AppendGqaBinding<'a> {
    pub view: &'a mut ferrule_model::decoder::CudaKvView,
    pub batch: &'a ferrule_model::decoder::PackedDecoderBatch,
    pub budget: &'a ferrule_common::host_kv::placement::PlacementBudget,
    pub block_tokens: usize,
}
