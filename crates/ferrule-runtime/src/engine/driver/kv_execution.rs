//! Startup selection and observations; reservation custody stays in admission.
use super::*;
use crate::engine::kv_policy::invalid;
use crate::engine::{KvExecutionPolicy, KvExecutionStats};

impl<R: MultiSessionRunner, C: SequenceSlotPool> ResidentTopKDriver<R, C> {
    pub fn suspend_session_kv(&mut self, session: crate::scheduling::SessionId) -> Result<()> {
        if !self.executor.idle_kv_reclaim_enabled() {
            return Err(invalid("idle KV reclaim is not selected"));
        }
        self.preempt_session(session)?;
        // Suspension is already durably owned. A trim error is not rollback or
        // permission to replay preempt; the session remains suspended.
        self.executor.trim_idle_kv()
    }
    pub fn restore_session_kv(&mut self, session: crate::scheduling::SessionId) -> Result<()> {
        if !self.executor.idle_kv_reclaim_enabled() {
            return Err(invalid("idle KV reclaim is not selected"));
        }
        self.restore_session(session)
    }
    pub fn trim_idle_kv(&mut self) -> Result<()> {
        self.executor.trim_idle_kv()
    }
    pub fn idle_kv_physical_stats(&self) -> Option<crate::engine::IdleKvPhysicalStats> {
        self.executor.idle_kv_physical_stats()
    }
    pub fn configure_kv_execution(&mut self, policy: KvExecutionPolicy) -> Result<()> {
        if self.admission_started
            || self.scheduler.waiting_len() != 0
            || self.scheduler.active_len() != 0
            || self.shutting_down
            || self
                .page_manager
                .as_ref()
                .is_some_and(|m| m.active_sequences() != 0)
            || !self.resident_transactions.is_empty()
            || !self.speculative_transactions.is_empty()
        {
            return Err(invalid("KV execution policy is startup-only"));
        }
        let max_pages = self
            .page_manager
            .as_ref()
            .map_or(0, KvPageManager::max_pages);
        if let KvExecutionPolicy::HostKvDecode(_) = policy {
            let scheduler = self.scheduler.config();
            if scheduler.max_active_sequences != 1
                || scheduler.max_decode_batch != 1
                || max_pages == 0
            {
                return Err(invalid(
                    "HostKV requires one active sequence, one decode row and the existing bounded page manager",
                ));
            }
        }
        self.executor.configure_kv_execution(policy, max_pages)?;
        tracing::info!(
            target: "ferrule_runtime::engine::driver::host_kv",
            ?policy,
            "KV execution policy installed; prefill warming remains resident"
        );
        Ok(())
    }
    pub fn kv_execution_stats(&self) -> KvExecutionStats {
        self.observability.stats.kv_execution
    }
}

#[cfg(feature = "cuda")]
impl<C: SequenceSlotPool>
    ResidentTopKDriver<
        ferrule_model::decoder::GenericDecoderRunner<ferrule_model::decoder::HybridCudaDecoder>,
        C,
    >
{
    pub(crate) fn install_cuda_host_kv_dispatch(&mut self) {
        self.executor.install_cuda_host_kv_dispatch();
    }
}
