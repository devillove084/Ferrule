//! Model-neutral inference engine owned by runtime.

use std::future::Future;
use std::pin::Pin;

use ferrule_common::{CompletionHub, CompletionListener, CompletionWake};
use ferrule_model::ResidentModelRunner;

use crate::scheduling::{GenerateRequest, RequestId, SequenceSlotPool, SequenceState};
use crate::{CancelRequestResult, ExpertIoBudget, ResidentDriverStep, ResidentTokenEvent};

use super::ResidentTopKDriver;

/// Completion reactors and wake coordination owned by one local inference task.
///
/// Attach this object on the CUDA owner's [`tokio::task::LocalSet`]. Reactor
/// futures never borrow the engine, and their producer callbacks only publish to
/// the shared [`CompletionHub`].
pub struct InferenceCompletionOwner {
    completion_hub: CompletionHub,
    reactor_errors: tokio::sync::mpsc::UnboundedReceiver<String>,
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
                    let message = match reactor.await {
                        Ok(()) => "model completion reactor stopped unexpectedly".to_owned(),
                        Err(error) => format!("model completion reactor failed: {error}"),
                    };
                    let _ = reactor_errors.send(message);
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
    pub async fn reactor_failure(&mut self) -> String {
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
    pub async fn wait(&mut self, completion: CompletionListener) -> Result<(), String> {
        tokio::select! {
            wake = completion => match wake {
                CompletionWake::Progress(_) => Ok(()),
                CompletionWake::Closed => {
                    Err("model completion source closed with live async work".to_owned())
                }
            },
            error = self.reactor_failure() => Err(error),
        }
    }

    /// Advance one engine step and, when it suspends, await a real completion
    /// before returning. This is the event-driven primitive used by local CLI
    /// owners; it never retries the engine or polls a timer internally.
    pub async fn step(
        &mut self,
        engine: &mut impl InferenceEngine,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<(), String>,
    ) -> Result<ResidentDriverStep, String> {
        let completion = self.listen();
        let step = engine.step(on_token)?;
        if matches!(
            step,
            ResidentDriverStep::WaitingForModelProgress(_) | ResidentDriverStep::Blocked
        ) {
            if !engine.has_pending_async_work() {
                return Err(
                    "runtime reported blocked work without an owned async continuation".to_owned(),
                );
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
    R: ResidentModelRunner + Send + 'static,
    R::SequenceState: Send + 'static,
    C: SequenceSlotPool + Send + 'static,
{
    pub fn new(driver: ResidentTopKDriver<R, C>, expert_budget: ExpertIoBudget) -> Self {
        let mut engine = ResidentInferenceEngine::new(driver, expert_budget);
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

    pub fn encode(&self, text: &str) -> ferrule_common::Result<Vec<u32>> {
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

    pub fn retain_session(
        &mut self,
        session_id: crate::scheduling::SessionId,
    ) -> ferrule_common::Result<()> {
        self.engine.driver.retain_session(session_id)
    }

    pub fn retained_session_position(
        &self,
        session_id: crate::scheduling::SessionId,
    ) -> Option<usize> {
        self.engine.driver.retained_session_position(session_id)
    }

    pub fn reset_session(
        &mut self,
        session_id: crate::scheduling::SessionId,
    ) -> ferrule_common::Result<()> {
        self.engine.driver.reset_session(session_id)
    }

    pub fn submit(&mut self, request: GenerateRequest) {
        self.engine.driver.submit(request);
    }

    pub fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.engine.driver.drain_finished()
    }

    pub async fn step<F>(&mut self, on_token: &mut F) -> ferrule_common::Result<ResidentDriverStep>
    where
        F: FnMut(&ResidentTokenEvent) -> ferrule_common::Result<()> + ?Sized,
    {
        let mut callback_error = None;
        let step = {
            let mut adapter = |event: &ResidentTokenEvent| match on_token(event) {
                Ok(()) => Ok(()),
                Err(error) => {
                    let message = error.to_string();
                    callback_error = Some(error);
                    Err(message)
                }
            };
            self.completion_owner
                .step(&mut self.engine, &mut adapter)
                .await
        };
        match (callback_error, step) {
            (Some(error), _) => Err(error),
            (None, Ok(step)) => Ok(step),
            (None, Err(error)) => Err(ferrule_common::Error::Execution(error)),
        }
    }
}

/// Owned completion reactor driven on the inference owner's local task set.
pub type InferenceCompletionReactor = Pin<Box<dyn Future<Output = Result<(), String>> + 'static>>;

/// Progress of a request cancellation owned by the inference runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceCancelProgress {
    /// The request is quiescent and scheduler cancellation has completed.
    Complete(CancelRequestResult),
    /// Model work still owns an asynchronous continuation. The caller must retain
    /// request ownership and retry after the engine's completion hub wakes.
    WaitingForModelProgress,
}

/// Execution lifecycle consumed by serving frontends.
///
/// Protocol crates depend on this model-neutral boundary and never select model
/// capabilities or scheduling algorithms themselves.
pub trait InferenceEngine: Send + 'static {
    /// Shared allocation-free wake source for all storage, staging, and device
    /// completion producers owned by this engine.
    fn completion_hub(&self) -> CompletionHub;

    /// Transfer completion reactors to the dedicated inference owner exactly once.
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor>;

    /// Whether a returned Waiting/Blocked step has an owned continuation that can
    /// be woken by the completion hub.
    fn has_pending_async_work(&self) -> bool;

    fn encode(&self, prompt: &str) -> Result<Vec<u32>, String>;
    fn submit(&mut self, request: GenerateRequest);
    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<(), String>,
    ) -> Result<ResidentDriverStep, String>;
    fn cancel_request(&mut self, request_id: RequestId) -> Result<InferenceCancelProgress, String>;
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
    driver: ResidentTopKDriver<R, C>,
    expert_budget: ExpertIoBudget,
}

impl<R, C> ResidentInferenceEngine<R, C>
where
    R: ResidentModelRunner,
    C: SequenceSlotPool,
{
    pub fn new(driver: ResidentTopKDriver<R, C>, expert_budget: ExpertIoBudget) -> Self {
        Self {
            driver,
            expert_budget,
        }
    }

    pub fn driver(&self) -> &ResidentTopKDriver<R, C> {
        &self.driver
    }

    pub fn into_driver(self) -> ResidentTopKDriver<R, C> {
        self.driver
    }
}

impl<R, C> InferenceEngine for ResidentInferenceEngine<R, C>
where
    R: ResidentModelRunner + Send + 'static,
    R::SequenceState: Send + 'static,
    C: SequenceSlotPool + Send + 'static,
{
    fn completion_hub(&self) -> CompletionHub {
        self.driver.completion_hub()
    }

    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        self.driver
            .take_completion_reactors()
            .into_iter()
            .map(|reactor| {
                Box::pin(async move { reactor.await.map_err(|error| error.to_string()) })
                    as InferenceCompletionReactor
            })
            .collect()
    }

    fn has_pending_async_work(&self) -> bool {
        self.driver.has_pending_async_work()
    }

    fn encode(&self, prompt: &str) -> Result<Vec<u32>, String> {
        self.driver
            .encode(prompt)
            .map_err(|error| error.to_string())
    }

    fn submit(&mut self, request: GenerateRequest) {
        self.driver.submit(request);
    }

    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> Result<(), String>,
    ) -> Result<ResidentDriverStep, String> {
        let mut adapter =
            |event: &ResidentTokenEvent| on_token(event).map_err(ferrule_common::Error::Execution);
        self.driver
            .step_with_model_expert_io(&mut adapter, self.expert_budget)
            .map_err(|error| error.to_string())
    }

    fn cancel_request(&mut self, request_id: RequestId) -> Result<InferenceCancelProgress, String> {
        match self.driver.cancel_request(request_id) {
            Ok(result) => Ok(InferenceCancelProgress::Complete(result)),
            Err(_) if self.driver.request_has_pending_model_progress(request_id) => {
                Ok(InferenceCancelProgress::WaitingForModelProgress)
            }
            Err(error) => Err(error.to_string()),
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
