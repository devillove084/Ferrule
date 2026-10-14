//! Factory-owned policy and raw-engine overrides use the same owner-local setter,
//! but only one of them may call it. No model, CUDA context or device is needed.
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ferrule_common::{CompletionHub, WorkerStartError};
use ferrule_runtime::{
    CancelRequestResult, Error, GenerateRequest, HostKvDecodeConfig, InferenceCancelProgress,
    InferenceCompletionReactor, InferenceEngine, KvExecutionPolicy, RequestId, ResidentDriverStep,
    ResidentTokenEvent, Result, SequenceState,
};
use ferrule_server::{WorkerConfig, spawn_model_worker_with};

struct PolicyEngine {
    owner: std::thread::ThreadId,
    hub: CompletionHub,
    calls: Arc<AtomicUsize>,
    starts: Arc<AtomicUsize>,
    policy: KvExecutionPolicy,
    expected: KvExecutionPolicy,
    reject: bool,
}

impl InferenceEngine for PolicyEngine {
    fn configure_kv_execution(&mut self, policy: KvExecutionPolicy) -> Result<()> {
        assert_eq!(self.owner, std::thread::current().id());
        assert_eq!(
            self.calls.fetch_add(1, Ordering::SeqCst),
            0,
            "second policy owner"
        );
        if self.reject {
            return Err(Error::InvalidRequest {
                message: "test policy rejected".into(),
            });
        }
        self.policy = policy;
        Ok(())
    }

    fn start_background_work(&mut self) -> Result<()> {
        assert_eq!(self.owner, std::thread::current().id());
        assert_eq!(
            self.policy, self.expected,
            "policy changed or silently discarded"
        );
        self.starts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn completion_hub(&self) -> CompletionHub {
        self.hub.clone()
    }
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        Vec::new()
    }
    fn has_pending_async_work(&self) -> bool {
        false
    }
    fn encode(&self, _prompt: &str) -> Result<Vec<u32>> {
        unreachable!()
    }
    fn submit(&mut self, _request: GenerateRequest) {
        unreachable!()
    }
    fn step(
        &mut self,
        _on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<()>,
    ) -> Result<ResidentDriverStep> {
        Ok(ResidentDriverStep::Idle)
    }
    fn cancel_request(&mut self, request_id: RequestId) -> Result<InferenceCancelProgress> {
        Ok(InferenceCancelProgress::Complete(
            CancelRequestResult::NotFound { request_id },
        ))
    }
    fn drain_finished(&mut self) -> Vec<SequenceState> {
        Vec::new()
    }
    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        Vec::new()
    }
    fn drain_failed(&mut self) -> Vec<SequenceState> {
        Vec::new()
    }
}

#[tokio::test]
async fn factory_owned_and_raw_engine_policies_are_each_configured_exactly_once() {
    let policy = KvExecutionPolicy::HostKvDecode(HostKvDecodeConfig::default());
    for factory_owned in [true, false] {
        let calls = Arc::new(AtomicUsize::new(0));
        let starts = Arc::new(AtomicUsize::new(0));
        let engine_calls = calls.clone();
        let engine_starts = starts.clone();
        let config = if factory_owned {
            // Same handoff as ResidentModelBuildPlan::build + CLI worker_config:
            // the factory validates/installs policy before returning the engine.
            WorkerConfig::default()
        } else {
            WorkerConfig {
                kv_execution_policy: policy,
                ..Default::default()
            }
        };
        let worker = spawn_model_worker_with(
            move || {
                let mut engine = PolicyEngine {
                    owner: std::thread::current().id(),
                    hub: CompletionHub::new(),
                    calls: engine_calls,
                    starts: engine_starts,
                    policy: KvExecutionPolicy::Resident,
                    expected: policy,
                    reject: false,
                };
                if factory_owned {
                    engine.configure_kv_execution(policy)?;
                }
                Ok::<_, Error>(engine)
            },
            config,
        )
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        worker.shutdown().await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

#[test]
fn rejected_factory_or_worker_policy_never_starts_background_work() {
    let policy = KvExecutionPolicy::HostKvDecode(HostKvDecodeConfig::default());
    for factory_owned in [true, false] {
        let calls = Arc::new(AtomicUsize::new(0));
        let starts = Arc::new(AtomicUsize::new(0));
        let engine_calls = calls.clone();
        let engine_starts = starts.clone();
        let result = spawn_model_worker_with(
            move || {
                let mut engine = PolicyEngine {
                    owner: std::thread::current().id(),
                    hub: CompletionHub::new(),
                    calls: engine_calls,
                    starts: engine_starts,
                    policy: KvExecutionPolicy::Resident,
                    expected: policy,
                    reject: true,
                };
                if factory_owned {
                    engine.configure_kv_execution(policy)?;
                }
                Ok::<_, Error>(engine)
            },
            WorkerConfig {
                kv_execution_policy: if factory_owned {
                    KvExecutionPolicy::Resident
                } else {
                    policy
                },
                ..Default::default()
            },
        );
        match result {
            Err(WorkerStartError::EngineFactory { .. }) if factory_owned => {}
            Err(WorkerStartError::Startup { .. }) if !factory_owned => {}
            _ => panic!("rejected policy must fail at its configuring boundary"),
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(starts.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn default_worker_preserves_legacy_engines_without_calling_policy_setter() {
    let calls = Arc::new(AtomicUsize::new(0));
    let engine_calls = calls.clone();
    let worker = spawn_model_worker_with(
        move || {
            Ok::<_, Error>(PolicyEngine {
                owner: std::thread::current().id(),
                hub: CompletionHub::new(),
                calls: engine_calls,
                starts: Arc::new(AtomicUsize::new(0)),
                policy: KvExecutionPolicy::Resident,
                expected: KvExecutionPolicy::Resident,
                reject: true,
            })
        },
        WorkerConfig::default(),
    )
    .unwrap();
    worker.shutdown().await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
