use std::convert::Infallible;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ferrule_common::execution::ExecutionTransactionId;
use ferrule_common::{CompletionHub, DependencySet, LogicalDependency, OperationId};
use ferrule_model::{ContinuationId, PendingModelProgress};
use ferrule_runtime::{
    CancelRequestResult, GenerateRequest, InferenceCancelProgress, InferenceCompletionReactor,
    InferenceEngine, RequestId, ResidentDriverStep, ResidentTokenEvent, Result as RuntimeResult,
    SequenceState,
};
use ferrule_server::{WorkerConfig, spawn_model_worker_with};

#[test]
fn cross_thread_api_exposes_send_control_values() {
    fn assert_send<T: Send>() {}

    assert_send::<WorkerConfig>();
    assert_send::<ferrule_server::ModelWorkerHandle>();
}

#[test]
fn prebuilt_engine_worker_api_is_absent() {
    let public_api = include_str!("../src/lib.rs");
    let worker = include_str!("../src/worker.rs");

    assert!(!public_api.contains("spawn_model_worker,"));
    assert!(!worker.contains("pub fn spawn_model_worker<E>"));
}

struct ThreadBoundEngine {
    owner: std::thread::ThreadId,
    completion_hub: CompletionHub,
    shutdown_called: Arc<AtomicBool>,
    drop_sender: std::sync::mpsc::SyncSender<std::thread::ThreadId>,
    _not_send: Rc<()>,
}

impl ThreadBoundEngine {
    fn assert_owner(&self) {
        assert_eq!(self.owner, std::thread::current().id());
    }
}

impl Drop for ThreadBoundEngine {
    fn drop(&mut self) {
        let _ = self.drop_sender.send(std::thread::current().id());
    }
}

impl InferenceEngine for ThreadBoundEngine {
    fn completion_hub(&self) -> CompletionHub {
        self.assert_owner();
        self.completion_hub.clone()
    }

    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        self.assert_owner();
        Vec::new()
    }

    fn has_pending_async_work(&self) -> bool {
        self.assert_owner();
        false
    }

    fn shutdown(&mut self) -> RuntimeResult<ferrule_runtime::InferenceShutdownProgress> {
        self.assert_owner();
        self.shutdown_called.store(true, Ordering::Release);
        Ok(ferrule_runtime::InferenceShutdownProgress::Complete)
    }

    fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
        self.assert_owner();
        Ok(prompt.bytes().map(u32::from).collect())
    }

    fn submit(&mut self, _request: GenerateRequest) {
        self.assert_owner();
    }

    fn step(
        &mut self,
        _on_token: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
    ) -> RuntimeResult<ResidentDriverStep> {
        self.assert_owner();
        Ok(ResidentDriverStep::Idle)
    }

    fn cancel_request(&mut self, request_id: RequestId) -> RuntimeResult<InferenceCancelProgress> {
        self.assert_owner();
        Ok(InferenceCancelProgress::Complete(
            CancelRequestResult::NotFound { request_id },
        ))
    }

    fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.assert_owner();
        Vec::new()
    }

    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        self.assert_owner();
        Vec::new()
    }

    fn drain_failed(&mut self) -> Vec<SequenceState> {
        self.assert_owner();
        Vec::new()
    }
}

struct BackgroundEngine {
    completion_hub: CompletionHub,
    ready: Arc<AtomicBool>,
    phase: Arc<AtomicUsize>,
    step_calls: Arc<AtomicUsize>,
}

impl InferenceEngine for BackgroundEngine {
    fn completion_hub(&self) -> CompletionHub {
        self.completion_hub.clone()
    }

    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        Vec::new()
    }

    fn has_background_work(&self) -> bool {
        self.phase.load(Ordering::Acquire) < 2
    }

    fn has_pending_async_work(&self) -> bool {
        self.phase.load(Ordering::Acquire) == 1
    }

    fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
        Ok(prompt.bytes().map(u32::from).collect())
    }

    fn submit(&mut self, _request: GenerateRequest) {
        unreachable!("background-only engine does not accept requests")
    }

    fn step(
        &mut self,
        _on_token: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
    ) -> RuntimeResult<ResidentDriverStep> {
        self.step_calls.fetch_add(1, Ordering::AcqRel);
        match self.phase.load(Ordering::Acquire) {
            0 => {
                self.phase.store(1, Ordering::Release);
                let continuation = ContinuationId::new(2);
                let dependency = LogicalDependency::operation_retired(OperationId::new(2))?;
                let dependencies = DependencySet::new([dependency])?;
                let transaction = ExecutionTransactionId::new(2)?;
                let pending = PendingModelProgress::new(transaction, continuation, dependencies)?;
                Ok(ResidentDriverStep::WaitingForModelProgress(vec![pending]))
            }
            1 if self.ready.load(Ordering::Acquire) => {
                self.phase.store(2, Ordering::Release);
                Ok(ResidentDriverStep::Executed {
                    action_kind: ferrule_runtime::ResidentActionKind::Prefill,
                    rows: 0,
                    staged: 0,
                    finished: 0,
                })
            }
            1 => Ok(ResidentDriverStep::Blocked),
            _ => Ok(ResidentDriverStep::Idle),
        }
    }

    fn cancel_request(&mut self, request_id: RequestId) -> RuntimeResult<InferenceCancelProgress> {
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
async fn factory_shutdown_and_drop_share_the_owner_thread() {
    let caller = std::thread::current().id();
    let shutdown_called = Arc::new(AtomicBool::new(false));
    let (owner_sender, owner_receiver) = std::sync::mpsc::sync_channel(1);
    let (drop_sender, drop_receiver) = std::sync::mpsc::sync_channel(1);
    let worker_shutdown_called = Arc::clone(&shutdown_called);
    let worker = spawn_model_worker_with(
        move || {
            let owner = std::thread::current().id();
            owner_sender.send(owner).unwrap();
            Ok::<Box<dyn InferenceEngine>, Infallible>(Box::new(ThreadBoundEngine {
                owner,
                completion_hub: CompletionHub::new(),
                shutdown_called: worker_shutdown_called,
                drop_sender,
                _not_send: Rc::new(()),
            }))
        },
        WorkerConfig::default(),
    )
    .unwrap();

    let owner = owner_receiver.recv().unwrap();
    assert_ne!(owner, caller);
    worker.shutdown().await.unwrap();
    assert!(shutdown_called.load(Ordering::Acquire));
    assert_eq!(drop_receiver.recv().unwrap(), owner);
}

#[tokio::test]
async fn idle_worker_drives_background_work_from_completion_wakes() {
    let completion_hub = CompletionHub::new();
    let ready = Arc::new(AtomicBool::new(false));
    let phase = Arc::new(AtomicUsize::new(0));
    let step_calls = Arc::new(AtomicUsize::new(0));
    let worker_completion_hub = completion_hub.clone();
    let worker_ready = Arc::clone(&ready);
    let worker_phase = Arc::clone(&phase);
    let worker_step_calls = Arc::clone(&step_calls);
    let worker = spawn_model_worker_with(
        move || {
            Ok::<BackgroundEngine, Infallible>(BackgroundEngine {
                completion_hub: worker_completion_hub,
                ready: worker_ready,
                phase: worker_phase,
                step_calls: worker_step_calls,
            })
        },
        WorkerConfig::default(),
    )
    .unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while step_calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(step_calls.load(Ordering::Acquire), 1);

    ready.store(true, Ordering::Release);
    completion_hub.notify();
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while phase.load(Ordering::Acquire) != 2 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(step_calls.load(Ordering::Acquire), 2);
    worker.shutdown().await.unwrap();
}

#[tokio::test]
async fn shutdown_preempts_a_waiting_background_worker() {
    let completion_hub = CompletionHub::new();
    let step_calls = Arc::new(AtomicUsize::new(0));
    let worker_step_calls = Arc::clone(&step_calls);
    let worker = spawn_model_worker_with(
        move || {
            Ok::<BackgroundEngine, Infallible>(BackgroundEngine {
                completion_hub,
                ready: Arc::new(AtomicBool::new(false)),
                phase: Arc::new(AtomicUsize::new(0)),
                step_calls: worker_step_calls,
            })
        },
        WorkerConfig::default(),
    )
    .unwrap();

    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        while step_calls.load(Ordering::Acquire) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(1), worker.shutdown())
        .await
        .unwrap()
        .unwrap();
}
