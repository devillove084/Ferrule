//! Reporting only: the wrapped runtime engine retains all execution and custody.
use super::inputs::{ExpertPlacementConfig, NumericCapacityLimits};
use crate::Result;
use crate::model_factory::{NumericFp8Profile, PipelineRankBackend};
#[cfg(feature = "cuda")]
use ferrule_model::decoder::GenericDecoderRunner;

#[cfg(feature = "cuda")]
type MoeEngine = ferrule_runtime::engine::ResidentInferenceEngine<
    GenericDecoderRunner<ferrule_model::decoder::HybridCudaDecoder>,
    ferrule_runtime::scheduling::FixedSequenceSlotPool,
>;

#[cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "metadata/admission wrapper is independent of CUDA construction"
    )
)]
pub(in crate::model_factory) trait MoeShutdownReport:
    ferrule_runtime::engine::SessionInferenceEngine
{
    fn report_shutdown(&self);
}

/// Only adds reports to the existing engine; does not create another runner,
/// scheduler, worker or KV pool. Read model cache stats after shutdown custody.
#[cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "metadata/admission wrapper is independent of CUDA construction"
    )
)]
pub(in crate::model_factory) struct ReportedMoeEngine<E: MoeShutdownReport> {
    pub(in crate::model_factory) engine: E,
    pub(in crate::model_factory) model_label: &'static str,
    pub(in crate::model_factory) host_experts:
        Option<std::sync::Arc<ferrule_model::transformer::host_experts::HostExpertCache>>,
    pub(in crate::model_factory) limits: NumericCapacityLimits,
    pub(in crate::model_factory) numeric_profile: NumericFp8Profile,
    pub(in crate::model_factory) expert_placement: Option<ExpertPlacementConfig>,
    pub(in crate::model_factory) shutdown_attempted: bool,
    pub(in crate::model_factory) close_reported: bool,
}

#[cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "metadata/admission wrapper is independent of CUDA construction"
    )
)]
impl<E: MoeShutdownReport> ReportedMoeEngine<E> {
    pub(in crate::model_factory) fn inner(&self) -> &E {
        &self.engine
    }
    pub(in crate::model_factory) fn inner_mut(&mut self) -> &mut E {
        &mut self.engine
    }
}

#[cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "metadata/admission wrapper is independent of CUDA construction"
    )
)]
impl<E: MoeShutdownReport> ferrule_runtime::engine::InferenceEngine for ReportedMoeEngine<E> {
    fn enable_kv_trace(&mut self, config: ferrule_runtime::RuntimeKvTraceConfig) -> Result<()> {
        self.inner_mut().enable_kv_trace(config)
    }

    fn drain_kv_trace(&mut self) -> Option<ferrule_runtime::RuntimeKvTraceSnapshot> {
        self.inner_mut().drain_kv_trace()
    }

    fn finalize_kv_trace(&mut self) -> Option<ferrule_runtime::RuntimeKvTraceSnapshot> {
        self.inner_mut().finalize_kv_trace()
    }

    fn completion_hub(&self) -> ferrule_common::CompletionHub {
        self.inner().completion_hub()
    }
    fn take_completion_reactors(
        &mut self,
    ) -> Vec<ferrule_runtime::engine::InferenceCompletionReactor> {
        self.inner_mut().take_completion_reactors()
    }
    fn has_background_work(&self) -> bool {
        self.inner().has_background_work()
    }
    fn has_pending_async_work(&self) -> bool {
        self.inner().has_pending_async_work()
    }
    fn start_background_work(&mut self) -> Result<()> {
        self.inner_mut().start_background_work()
    }
    fn shutdown(&mut self) -> Result<ferrule_runtime::engine::InferenceShutdownProgress> {
        self.shutdown_attempted = true;
        let progress = self.inner_mut().shutdown()?;
        if progress == ferrule_runtime::engine::InferenceShutdownProgress::Complete
            && !self.close_reported
        {
            self.engine.report_shutdown();
            self.close_reported = true;
        }
        Ok(progress)
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner().encode(text)
    }
    fn submit(&mut self, request: ferrule_runtime::GenerateRequest) {
        self.inner_mut().submit(request);
    }
    fn try_submit(&mut self, request: ferrule_runtime::GenerateRequest) -> Result<()> {
        self.inner_mut().try_submit(request)
    }
    fn request_cleanup(
        &self,
        request_id: ferrule_runtime::RequestId,
    ) -> ferrule_runtime::InferenceRequestCleanup {
        self.inner().request_cleanup(request_id)
    }
    fn admission_snapshot(&self) -> Option<ferrule_runtime::RuntimeAdmissionSnapshot> {
        self.inner().admission_snapshot()
    }
    fn capacity_snapshot(&self) -> ferrule_runtime::engine::InferenceCapacitySnapshot {
        self.inner().capacity_snapshot()
    }
    fn set_admission_options(
        &mut self,
        options: ferrule_runtime::RuntimeAdmissionOptions,
    ) -> Result<()> {
        self.inner_mut().set_admission_options(options)
    }
    fn close_admission(&mut self) -> Result<()> {
        self.inner_mut().close_admission()
    }
    fn step(
        &mut self,
        emit: &mut dyn FnMut(&ferrule_runtime::ResidentTokenEvent) -> Result<()>,
    ) -> Result<ferrule_runtime::ResidentDriverStep> {
        self.inner_mut().step(emit)
    }
    fn cancel_request(
        &mut self,
        id: ferrule_runtime::RequestId,
    ) -> Result<ferrule_runtime::InferenceCancelProgress> {
        self.inner_mut().cancel_request(id)
    }
    fn drain_finished(&mut self) -> Vec<ferrule_runtime::scheduling::SequenceState> {
        self.inner_mut().drain_finished()
    }
    fn drain_cancelled(&mut self) -> Vec<ferrule_runtime::scheduling::SequenceState> {
        self.inner_mut().drain_cancelled()
    }
    fn drain_failed(&mut self) -> Vec<ferrule_runtime::scheduling::SequenceState> {
        self.inner_mut().drain_failed()
    }
}

#[cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "metadata/admission wrapper is independent of CUDA construction"
    )
)]
impl<E: MoeShutdownReport> ferrule_runtime::engine::SessionInferenceEngine
    for ReportedMoeEngine<E>
{
    fn model_info(&self) -> ferrule_model::ModelInfo {
        self.inner().model_info()
    }
    fn observability_snapshot(&self) -> ferrule_runtime::ResidentEngineObservability {
        self.inner().observability_snapshot()
    }
    fn bound_layer_count(&self) -> Option<usize> {
        self.inner().bound_layer_count()
    }
    fn expert_report(&self) -> Option<String> {
        if let Some(placement) = &self.expert_placement {
            return Some(format!(
                "{} {} CUDA {} EP{}; root={} experts={:?}; all owned experts eager resident, no local routed cache; root-only KV; TP/PP/restarts unsupported; host={}\n",
                self.model_label,
                self.numeric_profile.arithmetic_description(),
                match placement.rank_backend() {
                    PipelineRankBackend::Thread => "thread",
                    PipelineRankBackend::Process => "process",
                },
                placement.expert_parallel(),
                placement.root_device(),
                placement.expert_devices(),
                self.host_experts.as_ref().map_or_else(
                    || if placement.rank_backend() == PipelineRankBackend::Process {
                        "owned partitions in expert children; root has no routed payload".into()
                    } else {
                        "missing".into()
                    },
                    |host| format!(
                        "full ready generation={} stats={:?} hits={}",
                        host.generation(),
                        host.stats(),
                        host.hits()
                    )
                ),
            ));
        }
        Some(format!(
            "{}: compressed E4M3FN + numeric BF16 scales; {}; device cache entries={} bytes={} scratch reservation={}; host={}\n",
            self.model_label,
            self.numeric_profile.arithmetic_description(),
            self.limits.max_experts,
            self.limits.max_bytes,
            self.limits.scratch_bytes,
            self.host_experts.as_ref().map_or_else(
                || "lazy (NOT prewarmed)".into(),
                |cache| format!(
                    "full ready generation={} stats={:?} hits={}",
                    cache.generation(),
                    cache.stats(),
                    cache.hits()
                )
            )
        ))
    }
    fn retain_session(&mut self, id: ferrule_runtime::SessionId) -> Result<()> {
        self.inner_mut().retain_session(id)
    }
    fn retained_session_position(&self, id: ferrule_runtime::SessionId) -> Option<usize> {
        self.inner().retained_session_position(id)
    }
    fn reset_session(&mut self, id: ferrule_runtime::SessionId) -> Result<()> {
        self.inner_mut().reset_session(id)
    }
    fn take_request_terminal(
        &mut self,
        id: ferrule_runtime::RequestId,
    ) -> Option<ferrule_runtime::scheduling::RequestTerminal> {
        self.inner_mut().take_request_terminal(id)
    }
}

#[cfg(feature = "cuda")]
impl MoeShutdownReport for MoeEngine {
    fn report_shutdown(&self) {
        use ferrule_runtime::engine::SessionInferenceEngine;
        let snapshot = self.observability_snapshot();
        let physically_closed = self.driver().physical_shutdown_complete();
        tracing::info!(kv = ?snapshot.kv_cache, driver = ?snapshot.driver, physically_closed,
            "numeric provider resident owner final runtime stats");
        let module = self.runner_for_report().forward_executor().module();
        tracing::info!(resident_bytes = module.resident_parameter_bytes(),
            cache = ?module.expert_cache_stats(), workspace = ?module.numeric_workspace_usage(),
            precision = ?module.numeric_fp8_precision(), physically_closed,
            "numeric provider owner final model stats");
    }
}

#[cfg_attr(
    not(feature = "cuda"),
    allow(
        dead_code,
        reason = "metadata/admission wrapper is independent of CUDA construction"
    )
)]
impl<E: MoeShutdownReport> Drop for ReportedMoeEngine<E> {
    fn drop(&mut self) {
        use ferrule_runtime::engine::InferenceEngine;
        // Never retry a reported failure behind the caller's back. The same
        // engine still owns the runner and any unknown-completion quarantine.
        if !self.shutdown_attempted {
            match self.shutdown() {
                Ok(ferrule_runtime::engine::InferenceShutdownProgress::Complete) => {}
                Ok(ferrule_runtime::engine::InferenceShutdownProgress::Pending) => {
                    tracing::warn!("numeric provider dropped before shutdown completed");
                }
                Err(error) => {
                    tracing::error!(%error, "numeric provider best-effort shutdown failed")
                }
            }
        }
        if !self.close_reported {
            self.engine.report_shutdown();
        }
    }
}
