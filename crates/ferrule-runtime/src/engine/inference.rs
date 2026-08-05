//! Model-neutral inference engine owned by runtime.

use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::pin::Pin;
use std::thread::{self, ThreadId};

use ferrule_common::{CompletionHub, CompletionListener, CompletionWake};
use ferrule_model::ResidentModelRunner;

use crate::scheduling::{GenerateRequest, RequestId, SequenceSlotPool, SequenceState};
use crate::{CancelRequestResult, Error, ResidentDriverStep, ResidentTokenEvent, Result};

use super::ResidentTopKDriver;

/// Owner-affine storage for model state that may contain intentionally `!Send`
/// completion reactors. The allocation never moves; only this guarded pointer may
/// cross a thread boundary to satisfy the legacy prebuilt-engine worker API.
struct OwnerLocal<T> {
    owner: ThreadId,
    value: Option<Box<T>>,
}

impl<T> OwnerLocal<T> {
    fn new(value: T) -> Self {
        Self {
            owner: thread::current().id(),
            value: Some(Box::new(value)),
        }
    }

    fn assert_owner(&self) {
        assert_eq!(
            self.owner,
            thread::current().id(),
            "owner-local inference engine was accessed from a different thread"
        );
    }

    fn get(&self) -> &T {
        self.assert_owner();
        self.value
            .as_deref()
            .expect("owner-local inference value is present")
    }

    fn get_mut(&mut self) -> &mut T {
        self.assert_owner();
        self.value
            .as_deref_mut()
            .expect("owner-local inference value is present")
    }

    fn into_inner(mut self) -> T {
        self.assert_owner();
        *self
            .value
            .take()
            .expect("owner-local inference value is present")
    }
}

impl<T> Deref for OwnerLocal<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T> DerefMut for OwnerLocal<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

// SAFETY: `T` remains in its original allocation and can only be dereferenced on
// `owner`. If the guard itself is transferred incorrectly, every accessor panics
// before touching `T`, and `Drop` below deliberately leaves `T` allocated rather
// than running an owner-affine destructor on the wrong thread.
#[allow(
    unsafe_code,
    reason = "owner affinity prevents access or destruction of T after a guard transfer"
)]
unsafe impl<T> Send for OwnerLocal<T> {}

impl<T> Drop for OwnerLocal<T> {
    fn drop(&mut self) {
        if self.owner != thread::current().id()
            && let Some(value) = self.value.take()
        {
            std::mem::forget(value);
        }
    }
}

/// Completion reactors and wake coordination owned by one local inference task.
///
/// Attach this object on the CUDA owner's [`tokio::task::LocalSet`]. Reactor
/// futures never borrow the engine, and their producer callbacks only publish to
/// the shared [`CompletionHub`].
pub struct InferenceCompletionOwner {
    completion_hub: CompletionHub,
    reactor_errors: tokio::sync::mpsc::UnboundedReceiver<Error>,
    reactor_tasks: Vec<tokio::task::JoinHandle<()>>,
    reactor_errors_open: bool,
}

impl InferenceCompletionOwner {
    /// Transfer and spawn all completion reactors exposed by `engine`.
    pub fn attach(engine: &mut impl InferenceEngine) -> Self {
        let completion_hub = engine.completion_hub();
        let (reactor_errors, reactor_error_rx) = tokio::sync::mpsc::unbounded_channel();
        let reactor_tasks = engine
            .take_completion_reactors()
            .into_iter()
            .map(|reactor| {
                let reactor_errors = reactor_errors.clone();
                tokio::task::spawn_local(async move {
                    let error = match reactor.await {
                        Ok(()) => Error::CompletionReactorStopped,
                        Err(error) => error,
                    };
                    let _ = reactor_errors.send(error);
                })
            })
            .collect::<Vec<_>>();
        drop(reactor_errors);
        let reactor_errors_open = !reactor_tasks.is_empty();
        Self {
            completion_hub,
            reactor_errors: reactor_error_rx,
            reactor_tasks,
            reactor_errors_open,
        }
    }

    /// Arm a race-free listener before inspecting or advancing model state.
    pub fn listen(&self) -> CompletionListener {
        self.completion_hub.listen()
    }

    /// Resolve with the next reactor failure. If the engine has no reactor, this
    /// remains pending so it can safely be used as a `tokio::select!` branch.
    pub async fn reactor_failure(&mut self) -> Error {
        loop {
            if !self.reactor_errors_open {
                std::future::pending::<()>().await;
            }
            match self.reactor_errors.recv().await {
                Some(error) => return error,
                None => self.reactor_errors_open = false,
            }
        }
    }

    /// Await an already-armed completion listener or a terminal reactor error.
    pub async fn wait(&mut self, completion: CompletionListener) -> Result<()> {
        tokio::select! {
            wake = completion => match wake {
                CompletionWake::Progress(_) => Ok(()),
                CompletionWake::Closed => Err(Error::CompletionSourceClosed),
            },
            error = self.reactor_failure() => Err(error),
        }
    }

    /// Start model-lifecycle background warmup without delaying request admission.
    /// The first owner step creates ordinary prefetch operations; later execution
    /// demand may join and promote them through the same materialization registry.
    pub async fn initialize(&mut self, engine: &mut impl InferenceEngine) -> Result<()> {
        engine.start_background_work()
    }

    /// Advance one engine step and, when it suspends, await a real completion
    /// before returning. This is the event-driven primitive used by local CLI
    /// owners; it never retries the engine or polls a timer internally.
    pub async fn step<F>(
        &mut self,
        engine: &mut impl InferenceEngine,
        on_token: &mut F,
    ) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()> + ?Sized,
    {
        let completion = self.listen();
        let mut adapter = |event: &ResidentTokenEvent| on_token(event);
        let step = engine.step(&mut adapter)?;
        if matches!(
            step,
            ResidentDriverStep::WaitingForModelProgress(_) | ResidentDriverStep::Blocked
        ) {
            if !engine.has_pending_async_work() {
                return Err(Error::Invariant {
                    message: "runtime reported blocked work without an owned async continuation"
                        .into(),
                });
            }
            self.wait(completion).await?;
        }
        Ok(step)
    }
}

impl Drop for InferenceCompletionOwner {
    fn drop(&mut self) {
        for task in &self.reactor_tasks {
            task.abort();
        }
    }
}

/// Local, event-driven owner for a concrete resident driver.
///
/// This is intended for command-line and diagnostic frontends that own the
/// inference lane directly. Serving frontends use [`InferenceCompletionOwner`]
/// separately so commands and cancellation can participate in their `select!`.
pub struct LocalResidentInferenceEngine<R, C>
where
    R: ResidentModelRunner,
    C: SequenceSlotPool,
{
    engine: ResidentInferenceEngine<R, C>,
    completion_owner: InferenceCompletionOwner,
}

impl<R, C> LocalResidentInferenceEngine<R, C>
where
    R: ResidentModelRunner + 'static,
    R::SequenceState: 'static,
    C: SequenceSlotPool + 'static,
{
    pub fn new(driver: ResidentTopKDriver<R, C>) -> Self {
        let mut engine = ResidentInferenceEngine::new(driver);
        let completion_owner = InferenceCompletionOwner::attach(&mut engine);
        Self {
            engine,
            completion_owner,
        }
    }

    pub fn driver(&self) -> &ResidentTopKDriver<R, C> {
        self.engine.driver()
    }

    pub fn model_info(&self) -> ferrule_model::ModelInfo {
        self.engine.driver().model_info()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.engine.driver().encode(text)
    }

    pub fn bound_layer_count(&self) -> Option<usize> {
        self.engine.driver().bound_layer_count()
    }

    pub fn expert_report(&self) -> Option<String> {
        self.engine.driver().expert_report()
    }

    pub fn model_observability_snapshot(&self) -> R::ObservabilitySnapshot {
        self.engine.driver().model_observability_snapshot()
    }

    pub fn stats(&self) -> &super::ResidentTopKDriverStats {
        self.engine.driver().stats()
    }

    pub fn retain_session(&mut self, session_id: crate::scheduling::SessionId) -> Result<()> {
        self.engine.driver_mut().retain_session(session_id)
    }

    pub fn retained_session_position(
        &self,
        session_id: crate::scheduling::SessionId,
    ) -> Option<usize> {
        self.engine.driver.retained_session_position(session_id)
    }

    pub fn reset_session(&mut self, session_id: crate::scheduling::SessionId) -> Result<()> {
        self.engine.driver_mut().reset_session(session_id)
    }

    pub async fn initialize(&mut self) -> Result<()> {
        self.completion_owner.initialize(&mut self.engine).await
    }

    /// Wait until model-lifecycle materialization has reached its planned ready
    /// state. Request-serving owners may omit this and overlap background warmup
    /// with admission; latency-sensitive offline owners can call it explicitly.
    pub async fn wait_for_model_warmup(&mut self) -> Result<()> {
        while self.engine.driver().warmup_pending() {
            let step = self.step(&mut |_| Ok(())).await?;
            if matches!(step, ResidentDriverStep::Idle) && self.engine.driver().warmup_pending() {
                return Err(Error::Invariant {
                    message: "runtime became idle before model warmup completed".into(),
                });
            }
        }
        Ok(())
    }

    pub fn submit(&mut self, request: GenerateRequest) {
        self.engine.driver_mut().submit(request);
    }

    pub fn take_request_terminal(
        &mut self,
        request_id: crate::scheduling::RequestId,
    ) -> Option<crate::scheduling::RequestTerminal> {
        self.engine.driver_mut().take_request_terminal(request_id)
    }

    pub fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.engine.driver_mut().drain_finished()
    }

    pub async fn step<F>(&mut self, on_token: &mut F) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()> + ?Sized,
    {
        self.completion_owner.step(&mut self.engine, on_token).await
    }

    /// Cancel background warmup and drain every submitted physical operation before
    /// releasing the model owner. Logical shutdown never releases resources while a
    /// provider still owns physical work.
    pub async fn shutdown(&mut self) -> Result<()> {
        loop {
            let completion = self.completion_owner.listen();
            let mut discard = |_event: &ResidentTokenEvent| Ok(());
            match self.engine.driver_mut().shutdown_progress(&mut discard)? {
                super::driver::ResidentShutdownProgress::Complete(report) => {
                    debug_assert!(report.registry.drained);
                    return Ok(());
                }
                super::driver::ResidentShutdownProgress::Pending => {
                    self.completion_owner.wait(completion).await?;
                }
            }
        }
    }
}

/// Owned completion reactor driven on the inference owner's local task set.
pub type InferenceCompletionReactor = Pin<Box<dyn Future<Output = Result<()>> + 'static>>;

/// Progress of a request cancellation owned by the inference runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceCancelProgress {
    /// The request is quiescent and scheduler cancellation has completed.
    Complete(CancelRequestResult),
    /// Cancellation was accepted and the owner will drive backend quiescence on
    /// subsequent ticks. The caller retains request ownership but must not resubmit
    /// the cancellation request.
    Pending,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceShutdownProgress {
    Pending,
    Complete,
}

/// Execution lifecycle consumed by serving frontends.
///
/// Engines are owner-local in production: `spawn_model_worker_with` constructs
/// them on the dedicated model thread, and completion reactors may intentionally
/// be `!Send`. The `Send` bound remains for the legacy prebuilt-engine worker API;
/// concrete runtime engines enforce owner-thread access internally.
/// Protocol crates depend on this model-neutral boundary and never select model
/// capabilities or scheduling algorithms themselves.
pub trait InferenceEngine: Send + 'static {
    /// Shared allocation-free wake source for all storage, staging, and device
    /// completion producers owned by this engine.
    fn completion_hub(&self) -> CompletionHub;

    /// Transfer completion reactors to the dedicated inference owner exactly once.
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor>;

    /// Whether model-lifecycle work should continue while no requests are active.
    /// Implementations must return false once one idle step cannot make progress
    /// without an external command or completion.
    fn has_background_work(&self) -> bool {
        false
    }

    /// Whether a returned Waiting/Blocked step has owned work that can be woken by
    /// the completion hub.
    fn has_pending_async_work(&self) -> bool;

    fn start_background_work(&mut self) -> Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<InferenceShutdownProgress> {
        Ok(InferenceShutdownProgress::Complete)
    }

    fn encode(&self, prompt: &str) -> Result<Vec<u32>>;
    fn submit(&mut self, request: GenerateRequest);
    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<()>,
    ) -> Result<ResidentDriverStep>;
    fn cancel_request(&mut self, request_id: RequestId) -> Result<InferenceCancelProgress>;
    fn drain_finished(&mut self) -> Vec<SequenceState>;
    fn drain_cancelled(&mut self) -> Vec<SequenceState>;
    fn drain_failed(&mut self) -> Vec<SequenceState>;
}

/// Runtime-owned resident inference engine.
///
/// `R` supplies model capabilities. The driver selects target-only or optional
/// native-proposal execution from those capabilities; the serving frontend sees
/// only [`InferenceEngine`].
pub struct ResidentInferenceEngine<R, C>
where
    R: ResidentModelRunner,
    C: SequenceSlotPool,
{
    driver: OwnerLocal<ResidentTopKDriver<R, C>>,
}

impl<R, C> ResidentInferenceEngine<R, C>
where
    R: ResidentModelRunner,
    C: SequenceSlotPool,
{
    pub fn new(driver: ResidentTopKDriver<R, C>) -> Self {
        Self {
            driver: OwnerLocal::new(driver),
        }
    }

    pub fn driver(&self) -> &ResidentTopKDriver<R, C> {
        self.driver.get()
    }

    fn driver_mut(&mut self) -> &mut ResidentTopKDriver<R, C> {
        self.driver.get_mut()
    }

    pub fn into_driver(self) -> ResidentTopKDriver<R, C> {
        self.driver.into_inner()
    }
}

impl<R, C> InferenceEngine for ResidentInferenceEngine<R, C>
where
    R: ResidentModelRunner + 'static,
    R::SequenceState: 'static,
    C: SequenceSlotPool + 'static,
{
    fn completion_hub(&self) -> CompletionHub {
        self.driver.completion_hub()
    }

    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        self.driver
            .take_completion_reactors()
            .into_iter()
            .map(|reactor| {
                Box::pin(async move { Ok(reactor.await?) }) as InferenceCompletionReactor
            })
            .collect()
    }

    fn has_background_work(&self) -> bool {
        self.driver.has_background_work()
    }

    fn has_pending_async_work(&self) -> bool {
        self.driver.has_pending_async_work()
    }

    fn start_background_work(&mut self) -> Result<()> {
        self.driver.start_background_work()
    }

    fn shutdown(&mut self) -> Result<InferenceShutdownProgress> {
        let mut discard = |_event: &ResidentTokenEvent| Ok(());
        self.driver
            .shutdown_progress(&mut discard)
            .map(|progress| match progress {
                super::driver::ResidentShutdownProgress::Pending => {
                    InferenceShutdownProgress::Pending
                }
                super::driver::ResidentShutdownProgress::Complete(_) => {
                    InferenceShutdownProgress::Complete
                }
            })
    }

    fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        self.driver.encode(prompt)
    }

    fn submit(&mut self, request: GenerateRequest) {
        self.driver.submit(request);
    }

    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<()>,
    ) -> Result<ResidentDriverStep> {
        let mut adapter = |event: &ResidentTokenEvent| on_token(event);
        self.driver.step(&mut adapter)
    }

    fn cancel_request(&mut self, request_id: RequestId) -> Result<InferenceCancelProgress> {
        match self.driver.cancel_request(request_id) {
            Ok(super::driver::ResidentCancelProgress::Complete(result)) => {
                Ok(InferenceCancelProgress::Complete(result))
            }
            Ok(super::driver::ResidentCancelProgress::Pending) => {
                Ok(InferenceCancelProgress::Pending)
            }
            Err(error) => Err(error),
        }
    }

    fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.driver.drain_finished()
    }

    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        self.driver.drain_cancelled()
    }

    fn drain_failed(&mut self) -> Vec<SequenceState> {
        self.driver.drain_failed()
    }
}
