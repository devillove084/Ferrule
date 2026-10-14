//! Typed dispatch installed by the concrete local composition, not by metadata.
use super::*;
use crate::cache::kv_read_authority::ReservedAppendPrefix;
use crate::cache::{KvPageManager, KvReservation};
use crate::engine::kv_policy::invalid;
use crate::engine::{HostKvDecodeConfig, KvExecutionMode, KvExecutionPolicy};
use ferrule_common::host_kv::placement::PlacementBudget;

type Prepare<R> = fn(
    &mut RunnerMultiSessionExecutor<R>,
    &ReservedAppendPrefix<'_>,
    &mut [<R as MultiSessionRunner>::SequenceState],
    &ExecutionBatch,
) -> Result<()>;
type Execute<R> = fn(
    &mut RunnerMultiSessionExecutor<R>,
    &ReservedAppendPrefix<'_>,
    &mut [<R as MultiSessionRunner>::SequenceState],
    &ExecutionBatch,
    &PlacementBudget,
    usize,
) -> Result<MultiSessionBatchProgress>;

pub(crate) struct HostKvDispatch<R: MultiSessionRunner> {
    pub(crate) validate: fn(&R, HostKvDecodeConfig, usize) -> Result<()>,
    pub(crate) prepare: Prepare<R>,
    pub(crate) execute: Execute<R>,
}

impl<R: MultiSessionRunner> RunnerMultiSessionExecutor<R> {
    #[cfg_attr(
        not(feature = "cuda"),
        allow(
            dead_code,
            reason = "concrete dispatch installers require CUDA; registration does not"
        )
    )]
    pub(crate) fn install_host_kv_dispatch(&mut self, dispatch: HostKvDispatch<R>) {
        self.host_kv_dispatch = Some(dispatch);
    }
    pub(crate) fn configure_kv_execution(
        &mut self,
        policy: KvExecutionPolicy,
        max_pages: usize,
    ) -> Result<()> {
        let storage = self.runner.idle_kv_reclaim_config();
        match (policy, storage) {
            (KvExecutionPolicy::IdleReclaim(expected), Some(actual)) if expected == actual => {}
            (KvExecutionPolicy::IdleReclaim(_), _) => {
                return Err(invalid(
                    "idle reclaim policy requires matching chunked runner construction",
                ));
            }
            (KvExecutionPolicy::HostKvDecode(_), Some(_)) => {
                return Err(invalid(
                    "idle reclaim and HostKV prefix decode are mutually exclusive",
                ));
            }
            (KvExecutionPolicy::Resident, Some(_)) => {
                return Err(invalid(
                    "cannot silently switch chunked storage to fixed/resident policy",
                ));
            }
            _ => {}
        }
        let budget = match policy {
            KvExecutionPolicy::Resident | KvExecutionPolicy::IdleReclaim(_) => None,
            KvExecutionPolicy::HostKvDecode(config) => {
                config.validate()?;
                let dispatch = self.host_kv_dispatch.as_ref().ok_or_else(|| invalid(
                    "HostKV decode requires installed local CUDA dense F32 all-GQA dispatch; no GDN/numeric/TP/EP fallback"))?;
                (dispatch.validate)(&self.runner, config, max_pages)?;
                Some(PlacementBudget::new(config.capacity))
            }
        };
        self.host_kv_budget = budget;
        self.kv_policy = policy;
        Ok(())
    }
    pub(crate) fn host_kv_enabled(&self) -> bool {
        matches!(self.kv_policy, KvExecutionPolicy::HostKvDecode(_))
    }
    pub(crate) fn kv_execution_mode(&self, batch: &ExecutionBatch) -> Result<KvExecutionMode> {
        self.kv_policy.select(batch)
    }
    pub(crate) fn prepare_scoped_batch(
        &mut self,
        mode: KvExecutionMode,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
        manager: Option<&KvPageManager>,
        reservations: &[KvReservation],
        views: &[KvReservationView],
    ) -> Result<()> {
        if mode != self.kv_execution_mode(batch)? {
            return Err(invalid("KV execution mode changed during admission"));
        }
        if mode != KvExecutionMode::HostKvDecode {
            return self.prepare_batch_with_kv(transaction, states, batch, views);
        }
        let prefix = prefix(manager, reservations, transaction)?;
        let prepare = self
            .host_kv_dispatch
            .as_ref()
            .ok_or_else(|| invalid("HostKV dispatch absent"))?
            .prepare;
        prepare(self, &prefix, states, batch)
    }
    pub(crate) fn execute_scoped_batch(
        &mut self,
        mode: KvExecutionMode,
        transaction: ExecutionTransactionId,
        states: &mut [R::SequenceState],
        batch: &ExecutionBatch,
        manager: Option<&KvPageManager>,
        reservations: &[KvReservation],
    ) -> Result<MultiSessionBatchProgress> {
        if mode != self.kv_execution_mode(batch)? {
            return Err(invalid("KV execution mode changed after prepare"));
        }
        if mode != KvExecutionMode::HostKvDecode {
            return self.execute_prepared_batch(transaction, states, batch);
        }
        let prefix = prefix(manager, reservations, transaction)?;
        let execute = self
            .host_kv_dispatch
            .as_ref()
            .ok_or_else(|| invalid("HostKV dispatch absent"))?
            .execute;
        let KvExecutionPolicy::HostKvDecode(config) = self.kv_policy else {
            unreachable!()
        };
        // A budget clone shares charges; it is NOT a reservation authority.
        let budget = self
            .host_kv_budget
            .as_ref()
            .expect("configured budget")
            .clone();
        let progress = execute(self, &prefix, states, batch, &budget, config.block_tokens)?;
        // Only synchronous HostKV is admitted. The model already rejects Waiting
        // before its context can escape. This boundary never registers one either.
        if matches!(progress, MultiSessionBatchProgress::Waiting(_)) {
            return Err(invalid(
                "HostKV decode returned Waiting; abort the existing transaction",
            ));
        }
        Self::validate_progress(transaction, batch, &self.capabilities, progress)
    }
}
fn prefix<'a>(
    manager: Option<&'a KvPageManager>,
    reservations: &'a [KvReservation],
    transaction: ExecutionTransactionId,
) -> Result<ReservedAppendPrefix<'a>> {
    let manager = manager.ok_or_else(|| invalid("HostKV requires the live page manager"))?;
    let [reservation] = reservations else {
        return Err(invalid("HostKV requires exactly one live reservation"));
    };
    manager
        .derive_append_prefix(reservation, transaction, 0)
        .map_err(|e| invalid(&format!("HostKV live reservation: {e}")))
}

#[cfg(feature = "cuda")]
impl
    RunnerMultiSessionExecutor<
        ferrule_model::decoder::GenericDecoderRunner<ferrule_model::decoder::HybridCudaDecoder>,
    >
{
    /// Explicit opt-in ordinary decode route. It borrows the existing runtime
    /// reservation authority and enters the already-prepared model transaction;
    /// the default executor path is unchanged.
    pub fn prepare_host_kv_decode(
        &mut self,
        prefix: &ReservedAppendPrefix<'_>,
        states: &mut [ferrule_model::decoder::CudaHybridSequenceState],
        batch: &ferrule_common::execution::ExecutionBatch,
    ) -> Result<()> {
        if self.runner.idle_kv_reclaim_config().is_some() {
            return Err(invalid(
                "idle KV reclaim and active HostKV prefix decode are mutually exclusive",
            ));
        }
        self.ensure_execution_ready()?;
        batch.validate(states.len(), &self.capabilities)?;
        Ok(prefix.prepare_cuda_decode(&mut self.runner, states, batch)?)
    }

    /// Execute the same runner-owned layer loop with per-layer borrowed
    /// authorities. Waiting is rejected by the model policy before a borrowed
    /// authority could enter a continuation; abort retains all normal custody.
    pub fn execute_host_kv_decode(
        &mut self,
        prefix: &ReservedAppendPrefix<'_>,
        states: &mut [ferrule_model::decoder::CudaHybridSequenceState],
        batch: &ferrule_common::execution::ExecutionBatch,
        budget: &ferrule_common::host_kv::placement::PlacementBudget,
        block_tokens: usize,
    ) -> Result<MultiSessionBatchProgress> {
        batch.validate(states.len(), &self.capabilities)?;
        let progress =
            prefix.execute_cuda_decode(&mut self.runner, states, batch, budget, block_tokens)?;
        Self::validate_progress(
            prefix.witness().transaction(),
            batch,
            &self.capabilities,
            progress,
        )
    }

    pub(crate) fn install_cuda_host_kv_dispatch(&mut self) {
        self.install_host_kv_dispatch(HostKvDispatch {
            validate: |runner, config, max_pages| {
                use ferrule_model::transformer::Attention;
                let spec = runner.resources().spec();
                crate::engine::kv_policy::validate_host_kv_spec(
                    spec,
                    runner
                        .forward_executor()
                        .module()
                        .numeric_fp8_precision()
                        .is_some(),
                )?;
                // Reject insufficient scratch at startup using the same physical
                // six-domain formula as append DMA, conservatively for every page.
                let budget = PlacementBudget::new(config.capacity);
                for layer in spec.layers() {
                    let Attention::Gqa(gqa) = layer.attention() else {
                        unreachable!()
                    };
                    let geometry = ferrule_backend::cpu::StreamingF32GqaConfig {
                        rows: 1,
                        q_heads: gqa.num_heads(),
                        kv_heads: gqa.num_kv_heads(),
                        head_dim: gqa.head_dim(),
                        block_tokens: config.block_tokens,
                        softmax_scale: 1.0 / (gqa.head_dim() as f32).sqrt(),
                        sliding_window: None,
                    };
                    let credits = ferrule_backend::cuda::operators::kv::committed_gqa_credits(
                        geometry, max_pages,
                    )?;
                    budget
                        .reserve_resources(credits)
                        .map_err(|e| invalid(&format!("HostKV scratch admission: {e}")))?
                        .release();
                }
                Ok(())
            },
            prepare: Self::prepare_host_kv_decode,
            execute: Self::execute_host_kv_decode,
        });
    }
}
