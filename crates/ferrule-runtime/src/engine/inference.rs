//! Model-neutral inference engine owned by runtime.

use std::future::Future;
use std::pin::Pin;

use ferrule_common::{CompletionHub, CompletionListener, CompletionWake};
use ferrule_model::ResidentModelRunner;

use crate::scheduling::{GenerateRequest, RequestId, SequenceSlotPool, SequenceState};
use crate::{CancelRequestResult, Error, ResidentDriverStep, ResidentTokenEvent, Result};

use super::{ResidentEngineObservability, ResidentKvPagePlan, ResidentTopKDriver};

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

/// Local, event-driven owner for an object-safe session engine.
///
/// Interactive frontends use this owner without naming a concrete model runner.
pub struct LocalSessionInferenceEngine {
    engine: BoxedSessionInferenceEngine,
    completion_owner: InferenceCompletionOwner,
}

impl LocalSessionInferenceEngine {
    pub fn new(mut engine: BoxedSessionInferenceEngine) -> Self {
        let completion_owner = InferenceCompletionOwner::attach(&mut engine);
        Self {
            engine,
            completion_owner,
        }
    }

    pub fn model_info(&self) -> ferrule_model::ModelInfo {
        self.engine.model_info()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.engine.encode(text)
    }

    pub fn bound_layer_count(&self) -> Option<usize> {
        self.engine.bound_layer_count()
    }

    pub fn expert_report(&self) -> Option<String> {
        self.engine.expert_report()
    }

    pub fn observability_snapshot(&self) -> ResidentEngineObservability {
        self.engine.observability_snapshot()
    }

    pub fn retain_session(&mut self, session_id: crate::scheduling::SessionId) -> Result<()> {
        self.engine.retain_session(session_id)
    }

    pub fn retained_session_position(
        &self,
        session_id: crate::scheduling::SessionId,
    ) -> Option<usize> {
        self.engine.retained_session_position(session_id)
    }

    pub fn reset_session(&mut self, session_id: crate::scheduling::SessionId) -> Result<()> {
        self.engine.reset_session(session_id)
    }

    pub async fn initialize(&mut self) -> Result<()> {
        self.completion_owner.initialize(&mut self.engine).await
    }

    /// Drive model-lifecycle materialization until the runtime reports no
    /// remaining background work.
    pub async fn wait_for_model_warmup(&mut self) -> Result<()> {
        while self.engine.has_background_work() {
            let step = self.step(&mut |_| Ok(())).await?;
            if matches!(step, ResidentDriverStep::Idle) && self.engine.has_background_work() {
                return Err(Error::Invariant {
                    message: "runtime became idle before model warmup completed".into(),
                });
            }
        }
        Ok(())
    }

    pub fn submit(&mut self, request: GenerateRequest) {
        self.engine.submit(request);
    }

    pub fn take_request_terminal(
        &mut self,
        request_id: crate::scheduling::RequestId,
    ) -> Option<crate::scheduling::RequestTerminal> {
        self.engine.take_request_terminal(request_id)
    }

    pub async fn step<F>(&mut self, on_token: &mut F) -> Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> Result<()> + ?Sized,
    {
        self.completion_owner.step(&mut self.engine, on_token).await
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        loop {
            let completion = self.completion_owner.listen();
            match self.engine.shutdown()? {
                InferenceShutdownProgress::Complete => return Ok(()),
                InferenceShutdownProgress::Pending => {
                    self.completion_owner.wait(completion).await?;
                }
            }
        }
    }
}

/// Local, event-driven owner for a concrete resident driver.
///
/// This typed interface remains available to benchmarks and diagnostics that
/// need concrete driver statistics or model-specific observability snapshots.
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

    pub fn prefix_cache_stats(&self) -> &super::ResidentPrefixCacheStats {
        self.engine.driver().prefix_cache_stats()
    }

    pub fn prefix_hits(&self) -> usize {
        self.engine.driver().prefix_hits()
    }

    pub fn prefix_misses(&self) -> usize {
        self.engine.driver().prefix_misses()
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

/// Owner-local execution lifecycle consumed by serving frontends.
///
/// Engines may intentionally be `!Send`. A dedicated owner must construct the
/// engine locally from a `Send` build plan or factory and retain it until shutdown
/// and destruction complete. Protocol crates depend on this model-neutral boundary
/// and never select model capabilities or scheduling algorithms themselves.
///
/// The object-safe owner boundary deliberately does not implement `Send`:
///
/// ```compile_fail
/// use ferrule_runtime::InferenceEngine;
///
/// fn require_send<T: Send>() {}
/// require_send::<Box<dyn InferenceEngine>>();
/// ```
pub trait InferenceEngine: 'static {
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

impl<T> InferenceEngine for Box<T>
where
    T: InferenceEngine + ?Sized,
{
    fn completion_hub(&self) -> CompletionHub {
        (**self).completion_hub()
    }

    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        (**self).take_completion_reactors()
    }

    fn has_background_work(&self) -> bool {
        (**self).has_background_work()
    }

    fn has_pending_async_work(&self) -> bool {
        (**self).has_pending_async_work()
    }

    fn start_background_work(&mut self) -> Result<()> {
        (**self).start_background_work()
    }

    fn shutdown(&mut self) -> Result<InferenceShutdownProgress> {
        (**self).shutdown()
    }

    fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        (**self).encode(prompt)
    }

    fn submit(&mut self, request: GenerateRequest) {
        (**self).submit(request);
    }

    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<()>,
    ) -> Result<ResidentDriverStep> {
        (**self).step(on_token)
    }

    fn cancel_request(&mut self, request_id: RequestId) -> Result<InferenceCancelProgress> {
        (**self).cancel_request(request_id)
    }

    fn drain_finished(&mut self) -> Vec<SequenceState> {
        (**self).drain_finished()
    }

    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        (**self).drain_cancelled()
    }

    fn drain_failed(&mut self) -> Vec<SequenceState> {
        (**self).drain_failed()
    }
}

/// Object-safe session lifecycle used by model-neutral interactive frontends.
pub trait SessionInferenceEngine: InferenceEngine {
    fn model_info(&self) -> ferrule_model::ModelInfo;
    fn observability_snapshot(&self) -> ResidentEngineObservability;
    fn bound_layer_count(&self) -> Option<usize>;
    fn expert_report(&self) -> Option<String>;
    fn retain_session(&mut self, session_id: crate::scheduling::SessionId) -> Result<()>;
    fn retained_session_position(&self, session_id: crate::scheduling::SessionId) -> Option<usize>;
    fn reset_session(&mut self, session_id: crate::scheduling::SessionId) -> Result<()>;
    fn take_request_terminal(
        &mut self,
        request_id: crate::scheduling::RequestId,
    ) -> Option<crate::scheduling::RequestTerminal>;
}

pub type BoxedSessionInferenceEngine = Box<dyn SessionInferenceEngine>;

impl<T> SessionInferenceEngine for Box<T>
where
    T: SessionInferenceEngine + ?Sized,
{
    fn model_info(&self) -> ferrule_model::ModelInfo {
        (**self).model_info()
    }

    fn observability_snapshot(&self) -> ResidentEngineObservability {
        (**self).observability_snapshot()
    }

    fn bound_layer_count(&self) -> Option<usize> {
        (**self).bound_layer_count()
    }

    fn expert_report(&self) -> Option<String> {
        (**self).expert_report()
    }

    fn retain_session(&mut self, session_id: crate::scheduling::SessionId) -> Result<()> {
        (**self).retain_session(session_id)
    }

    fn retained_session_position(&self, session_id: crate::scheduling::SessionId) -> Option<usize> {
        (**self).retained_session_position(session_id)
    }

    fn reset_session(&mut self, session_id: crate::scheduling::SessionId) -> Result<()> {
        (**self).reset_session(session_id)
    }

    fn take_request_terminal(
        &mut self,
        request_id: crate::scheduling::RequestId,
    ) -> Option<crate::scheduling::RequestTerminal> {
        (**self).take_request_terminal(request_id)
    }
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
    driver: ResidentTopKDriver<R, C>,
    kv_page_plan: Option<ResidentKvPagePlan>,
}

impl<R, C> ResidentInferenceEngine<R, C>
where
    R: ResidentModelRunner,
    C: SequenceSlotPool,
{
    pub fn new(driver: ResidentTopKDriver<R, C>) -> Self {
        Self {
            driver,
            kv_page_plan: None,
        }
    }

    pub(crate) fn with_kv_page_plan(
        driver: ResidentTopKDriver<R, C>,
        kv_page_plan: ResidentKvPagePlan,
    ) -> Self {
        Self {
            driver,
            kv_page_plan: Some(kv_page_plan),
        }
    }

    pub fn driver(&self) -> &ResidentTopKDriver<R, C> {
        &self.driver
    }

    fn driver_mut(&mut self) -> &mut ResidentTopKDriver<R, C> {
        &mut self.driver
    }

    pub fn into_driver(self) -> ResidentTopKDriver<R, C> {
        self.driver
    }
}

impl<R, C> SessionInferenceEngine for ResidentInferenceEngine<R, C>
where
    R: ResidentModelRunner + 'static,
    R::SequenceState: 'static,
    C: SequenceSlotPool + 'static,
{
    fn model_info(&self) -> ferrule_model::ModelInfo {
        self.driver.model_info()
    }

    fn observability_snapshot(&self) -> ResidentEngineObservability {
        super::observability::snapshot_driver(self.driver(), self.kv_page_plan)
    }

    fn bound_layer_count(&self) -> Option<usize> {
        self.driver.bound_layer_count()
    }

    fn expert_report(&self) -> Option<String> {
        self.driver.expert_report()
    }

    fn retain_session(&mut self, session_id: crate::scheduling::SessionId) -> Result<()> {
        self.driver_mut().retain_session(session_id)
    }

    fn retained_session_position(&self, session_id: crate::scheduling::SessionId) -> Option<usize> {
        self.driver.retained_session_position(session_id)
    }

    fn reset_session(&mut self, session_id: crate::scheduling::SessionId) -> Result<()> {
        self.driver_mut().reset_session(session_id)
    }

    fn take_request_terminal(
        &mut self,
        request_id: crate::scheduling::RequestId,
    ) -> Option<crate::scheduling::RequestTerminal> {
        self.driver_mut().take_request_terminal(request_id)
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
