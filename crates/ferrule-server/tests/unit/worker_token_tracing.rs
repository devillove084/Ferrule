use super::*;
use serde_json::{Map, Value, json};
use std::collections::VecDeque;

#[derive(Clone)]
struct Logs {
    events: Arc<Mutex<Vec<Value>>>,
    level: tracing::Level,
}

impl Logs {
    fn new(level: tracing::Level) -> Self {
        Self {
            events: Default::default(),
            level,
        }
    }

    fn named(&self, name: &str) -> Vec<Value> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter(|event| event["event"] == name)
            .cloned()
            .collect()
    }
}

impl tracing::Subscriber for Logs {
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        matches!(metadata.target(), "ferrule_token" | "ferrule_request")
            && *metadata.level() <= self.level
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(Map<String, Value>);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0
                    .insert(field.name().into(), json!(format!("{value:?}")));
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.0.insert(field.name().into(), json!(value));
            }
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                self.0.insert(field.name().into(), json!(value));
            }
            fn record_bool(&mut self, field: &tracing::field::Field, value: bool) {
                self.0.insert(field.name().into(), json!(value));
            }
        }
        let mut fields = Fields(Map::new());
        event.record(&mut fields);
        fields
            .0
            .insert("target".into(), json!(event.metadata().target()));
        fields
            .0
            .insert("level".into(), json!(event.metadata().level().as_str()));
        self.events.lock().unwrap().push(Value::Object(fields.0));
    }
}

#[derive(Clone, Copy)]
enum Mode {
    Finished,
    Cancelled,
    RuntimeFailed,
    Fatal,
    CancelFailure,
}

struct TraceEngine {
    hub: CompletionHub,
    requests: VecDeque<GenerateRequest>,
    pending: Option<GenerateRequest>,
    finished: Vec<SequenceState>,
    cancelled: Vec<SequenceState>,
    failed: Vec<SequenceState>,
    tokens: Vec<(u32, &'static str, usize)>,
    mode: Mode,
    cancellation: Arc<AtomicBool>,
    cancel_after_first: bool,
}

impl TraceEngine {
    fn state(&self, request: &GenerateRequest) -> SequenceState {
        let mut state = SequenceState::from_request(request, request.session_id.unwrap());
        state.generated = self.tokens.len();
        state
    }
}

impl InferenceEngine for TraceEngine {
    fn completion_hub(&self) -> CompletionHub {
        self.hub.clone()
    }
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        Vec::new()
    }
    fn has_pending_async_work(&self) -> bool {
        false
    }
    fn request_cleanup(&self, _: RequestId) -> InferenceRequestCleanup {
        InferenceRequestCleanup::TerminalQuiescent
    }
    fn encode(&self, _: &str) -> RuntimeResult<Vec<u32>> {
        Ok(vec![901, 902])
    }
    fn submit(&mut self, request: GenerateRequest) {
        self.requests.push_back(request);
    }
    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
    ) -> RuntimeResult<ResidentDriverStep> {
        let Some(request) = self.requests.pop_front() else {
            return Ok(ResidentDriverStep::Idle);
        };
        // These callbacks must never create an emission or an ordinal.
        for request_id in [None, Some(RequestId(999))] {
            on_token(&ResidentTokenEvent {
                request_id,
                session_id: SessionId(999),
                index: 99,
                token: 999,
                logit: Some(999.0),
                text: "unowned private text".into(),
            })?;
        }
        for (i, &(token, text, index)) in self.tokens.iter().enumerate() {
            on_token(&ResidentTokenEvent {
                request_id: Some(request.id),
                session_id: request.session_id.unwrap(),
                index,
                token,
                logit: Some(123.456),
                text: text.into(),
            })?;
            if i == 0 && self.cancel_after_first {
                self.cancellation.store(true, Ordering::Release);
            }
        }
        let mut state = self.state(&request);
        match self.mode {
            Mode::Finished => {
                state.finish_reason = Some(SequenceFinishReason::MaxTokens);
                self.finished.push(state);
            }
            Mode::RuntimeFailed => self.failed.push(state),
            Mode::Cancelled | Mode::CancelFailure => self.pending = Some(request),
            Mode::Fatal => {
                self.pending = Some(request);
                return Err(ferrule_runtime::Error::EngineUnavailable);
            }
        }
        Ok(ResidentDriverStep::Executed {
            action_kind: ferrule_runtime::ResidentActionKind::Decode,
            rows: 1,
            staged: self.tokens.len(),
            finished: 0,
        })
    }
    fn cancel_request(&mut self, request_id: RequestId) -> RuntimeResult<InferenceCancelProgress> {
        let request = self.pending.take().unwrap();
        let mut state = self.state(&request);
        if matches!(self.mode, Mode::CancelFailure) {
            // A failed cancellation may notify HTTP before a real failed state
            // is drained. Neither path may fabricate a second terminal send.
            self.failed.push(state);
            return Err(ferrule_runtime::Error::EngineUnavailable);
        }
        state.finish_reason = Some(SequenceFinishReason::Cancelled);
        self.cancelled.push(state);
        Ok(InferenceCancelProgress::Complete(
            CancelRequestResult::Active {
                request_id,
                session_id: request.session_id.unwrap(),
            },
        ))
    }
    fn drain_finished(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.finished)
    }
    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.cancelled)
    }
    fn drain_failed(&mut self) -> Vec<SequenceState> {
        std::mem::take(&mut self.failed)
    }
}

// Poll the actual production owner loop under a scoped subscriber: no global
// subscriber, scheduler sleeps, GPU, retokenization, or replacement emit loop.
fn run_case(
    logs: &Logs,
    mode: Mode,
    capacity: usize,
    closed: bool,
    cancel_after_first: bool,
    tokens: Vec<(u32, &'static str, usize)>,
    ids: &[u64],
) -> Vec<Vec<WorkerEvent>> {
    tracing::subscriber::with_default(logs.clone(), || {
        let control = Arc::new(ready_control());
        let cancellation = Arc::new(AtomicBool::new(false));
        let mut engine = TraceEngine {
            hub: CompletionHub::new(),
            requests: VecDeque::new(),
            pending: None,
            finished: vec![],
            cancelled: vec![],
            failed: vec![],
            tokens,
            mode,
            cancellation: cancellation.clone(),
            cancel_after_first,
        };
        let owner = InferenceCompletionOwner::attach(&mut engine);
        let (commands, receiver) = mpsc::channel(ids.len());
        let mut streams = vec![];
        let mut acceptances = vec![];
        for &id in ids {
            let (events, stream) = mpsc::channel(capacity);
            let (accepted, acceptance) = oneshot::channel();
            commands
                .try_send(WorkerCommand::Submit(SubmitCommand {
                    request_id: RequestId(id),
                    enqueued_at: Instant::now(),
                    request: WorkerRequest {
                        prompt: "private prompt canary".into(),
                        max_tokens: 8,
                        stop: vec![],
                        ignore_eos: false,
                    },
                    permit: test_permit(),
                    events,
                    cancellation: cancellation.clone(),
                    accepted,
                }))
                .unwrap();
            acceptances.push(acceptance);
            if !closed {
                streams.push(stream);
            }
        }
        let mut run = Box::pin(run_worker(
            engine,
            owner,
            receiver,
            WorkerConfig::default(),
            control.clone(),
        ));
        let mut run_result = None;
        for _ in 0..32 {
            match poll_once(run.as_mut()) {
                std::task::Poll::Pending => {}
                std::task::Poll::Ready(result) => {
                    run_result = Some(result);
                    break;
                }
            }
            if logs.named("request_worker_terminal").len() == ids.len() {
                break;
            }
            // INFO-only capture has no summaries; completed success fills the
            // reserved terminal slot and closes its sender instead.
            if logs.level == tracing::Level::INFO && streams.iter().all(mpsc::Receiver::is_closed) {
                break;
            }
        }
        for acceptance in &mut acceptances {
            assert!(matches!(acceptance.try_recv(), Ok(Ok(()))));
        }
        let results = streams
            .iter_mut()
            .map(|stream| {
                let mut events = vec![];
                while let Ok(event) = stream.try_recv() {
                    events.push(event);
                }
                events
            })
            .collect();
        if run_result.is_none() {
            control.begin_shutdown(ShutdownDeadline::now() + std::time::Duration::from_secs(1));
            for _ in 0..32 {
                if let std::task::Poll::Ready(result) = poll_once(run.as_mut()) {
                    run_result = Some(result);
                    break;
                }
            }
        }
        let result = run_result.expect("owner must complete CPU-only shutdown");
        assert_eq!(
            result.is_err(),
            matches!(mode, Mode::Fatal | Mode::CancelFailure)
        );
        results
    })
}

fn sample_tokens() -> Vec<(u32, &'static str, usize)> {
    // An empty text delta cannot be retokenized to recover its selecting ID.
    vec![(31415, "", 10), (27182, "private token canary", 13)]
}

#[tokio::test]
async fn actual_callback_traces_ids_and_contiguous_per_request_ordinals() {
    let logs = Logs::new(tracing::Level::TRACE);
    let results = run_case(
        &logs,
        Mode::Finished,
        4,
        false,
        false,
        sample_tokens(),
        &[71, 72],
    );
    let tokens = logs.named("request_worker_token_enqueued");
    assert_eq!(tokens.len(), 4);
    for (i, event) in tokens.iter().enumerate() {
        assert_eq!(event["request_id"], json!(71 + i / 2));
        assert_eq!(event["session_id"], event["request_id"]);
        assert_eq!(event["emission_ordinal"], json!(1 + i % 2));
        assert_eq!(event["actual_token_id"], json!([31415, 27182][i % 2]));
        assert_eq!(event["resident_token_index"], json!([10, 13][i % 2]));
        assert_eq!(event["target"], "ferrule_token");
        assert_eq!(event["level"], "TRACE");
        assert_eq!(event["delivery_boundary"], "worker_channel_enqueue");
        for key in ["text", "logit", "logprob", "prompt"] {
            assert!(event.get(key).is_none());
        }
    }
    let terminals = logs.named("request_worker_terminal");
    let enqueues = logs.named("request_worker_terminal_enqueue");
    assert_eq!(terminals.len(), 2);
    assert_eq!(enqueues.len(), 2);
    for ((events, terminal), enqueue) in results.iter().zip(&terminals).zip(&enqueues) {
        assert!(matches!(&events[0], WorkerEvent::Token { text } if text.is_empty()));
        assert!(
            matches!(&events[1], WorkerEvent::Token { text } if text == "private token canary")
        );
        let WorkerEvent::Finished { reason, usage } = &events[2] else {
            panic!("finished")
        };
        assert_eq!(*reason, SequenceFinishReason::MaxTokens);
        for event in [terminal, enqueue] {
            assert_eq!(event["status"], "finished");
            assert_eq!(event["finish_reason"], reason.as_str());
            assert_eq!(event["last_emission_ordinal"], 2);
            assert_eq!(event["usage_available"], true);
            assert_eq!(event["prompt_tokens"], json!(usage.prompt_tokens));
            assert_eq!(event["completion_tokens"], json!(usage.completion_tokens));
            assert_eq!(event["total_tokens"], json!(usage.total_tokens));
        }
        assert_eq!(terminal["tokens_reconcile"], "Some(true)");
        assert_eq!(enqueue["terminal_event_enqueued"], true);
    }
    let captured = format!("{:?}", logs.events.lock().unwrap());
    assert!(!captured.contains("private token canary"));
    assert!(!captured.contains("private prompt canary"));
    assert!(!captured.contains("unowned private text"));
}

#[tokio::test]
async fn default_info_and_request_debug_do_not_log_token_ids() {
    for level in [tracing::Level::INFO, tracing::Level::DEBUG] {
        let logs = Logs::new(level);
        let results = run_case(
            &logs,
            Mode::Finished,
            4,
            false,
            false,
            sample_tokens(),
            &[71],
        );
        assert_eq!(results[0].len(), 3);
        assert!(logs.named("request_worker_token_enqueued").is_empty());
        assert!(!format!("{:?}", logs.events.lock().unwrap()).contains("31415"));
        if level == tracing::Level::INFO {
            assert!(logs.events.lock().unwrap().is_empty());
        } else {
            assert_eq!(logs.named("request_worker_terminal").len(), 1);
        }
    }
}

#[tokio::test]
async fn cancelled_callbacks_and_backpressure_exclude_rejected_ids() {
    for (capacity, cancel_after_first) in [(4, true), (2, false)] {
        let logs = Logs::new(tracing::Level::TRACE);
        let results = run_case(
            &logs,
            Mode::Cancelled,
            capacity,
            false,
            cancel_after_first,
            sample_tokens(),
            &[71],
        );
        let tokens = logs.named("request_worker_token_enqueued");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0]["actual_token_id"], 31415);
        assert_eq!(tokens[0]["emission_ordinal"], 1);
        assert_eq!(results[0].len(), 2);
        assert!(matches!(results[0][1], WorkerEvent::Cancelled));
        let terminal = &logs.named("request_worker_terminal")[0];
        assert_eq!(terminal["status"], "cancelled");
        assert_eq!(terminal["last_emission_ordinal"], 1);
        assert_eq!(terminal["completion_tokens"], 2);
        assert_eq!(terminal["total_tokens"], 4);
        assert_eq!(terminal["tokens_reconcile"], "Some(false)");
        assert_eq!(terminal["cancellation_requested"], true);
        assert_eq!(terminal["cancellation_submitted"], true);
        let enqueue = &logs.named("request_worker_terminal_enqueue")[0];
        assert_eq!(enqueue["terminal_event_enqueued"], true);
        assert_eq!(enqueue["last_emission_ordinal"], 1);
        assert_eq!(enqueue["usage_available"], false);
        assert!(enqueue.get("completion_tokens").is_none());
    }
}

#[tokio::test]
async fn closed_channel_has_no_token_trace_and_false_terminal_enqueue() {
    let logs = Logs::new(tracing::Level::TRACE);
    run_case(
        &logs,
        Mode::Cancelled,
        4,
        true,
        false,
        sample_tokens(),
        &[71],
    );
    assert!(logs.named("request_worker_token_enqueued").is_empty());
    let terminal = &logs.named("request_worker_terminal")[0];
    assert_eq!(terminal["last_emission_ordinal"], 0);
    assert_eq!(terminal["completion_tokens"], 2);
    assert_eq!(terminal["tokens_reconcile"], "Some(false)");
    let enqueue = &logs.named("request_worker_terminal_enqueue")[0];
    assert_eq!(enqueue["terminal_event_enqueued"], false);
    assert_eq!(enqueue["last_emission_ordinal"], 0);
}

#[tokio::test]
async fn runtime_failure_has_known_usage_but_fatal_failure_does_not_invent_it() {
    for mode in [Mode::RuntimeFailed, Mode::Fatal, Mode::CancelFailure] {
        let logs = Logs::new(tracing::Level::TRACE);
        let results = run_case(
            &logs,
            mode,
            4,
            false,
            matches!(mode, Mode::CancelFailure),
            sample_tokens(),
            &[71],
        );
        let count = if matches!(mode, Mode::CancelFailure) {
            1
        } else {
            2
        };
        assert_eq!(logs.named("request_worker_token_enqueued").len(), count);
        assert_eq!(results[0].len(), count + 1);
        assert!(matches!(results[0][count], WorkerEvent::Failed { .. }));
        let terminals = logs.named("request_worker_terminal");
        let enqueues = logs.named("request_worker_terminal_enqueue");
        assert_eq!(terminals.len(), 1);
        assert_eq!(
            enqueues.len(),
            1,
            "unique terminal send even on cancel failure"
        );
        let terminal = &terminals[0];
        assert_eq!(terminal["status"], "failed");
        assert_eq!(terminal["last_emission_ordinal"], json!(count));
        assert_eq!(enqueues[0]["terminal_event_enqueued"], true);
        assert_eq!(enqueues[0]["usage_available"], false);
        assert_eq!(
            terminal["terminal_event_previously_attempted"],
            matches!(mode, Mode::CancelFailure)
        );
        let enqueue_reason = match mode {
            Mode::RuntimeFailed => "model_execution_failed",
            Mode::Fatal => "fatal_engine_error",
            Mode::CancelFailure => "cancellation_failed",
            _ => unreachable!(),
        };
        assert_eq!(enqueues[0]["finish_reason"], enqueue_reason);
        if matches!(mode, Mode::Fatal) {
            assert_eq!(terminal["finish_reason"], "fatal_engine_error");
            assert_eq!(terminal["usage_available"], false);
            assert!(terminal.get("completion_tokens").is_none());
            assert!(terminal.get("total_tokens").is_none());
            assert_eq!(terminal["tokens_reconcile"], "None");
        } else {
            assert_eq!(terminal["completion_tokens"], 2);
            assert_eq!(terminal["usage_available"], true);
        }
        if matches!(mode, Mode::CancelFailure) {
            assert_eq!(terminal["cancellation_failed"], true);
            assert_eq!(terminal["cancellation_submitted"], false);
            assert_eq!(terminal["tokens_reconcile"], "Some(false)");
        }
    }
}

#[tokio::test]
async fn zero_token_finish_has_zero_ordinal_and_numeric_zero_usage() {
    let logs = Logs::new(tracing::Level::TRACE);
    let results = run_case(&logs, Mode::Finished, 2, false, false, vec![], &[71]);
    assert_eq!(results[0].len(), 1);
    assert!(logs.named("request_worker_token_enqueued").is_empty());
    for name in ["request_worker_terminal", "request_worker_terminal_enqueue"] {
        let event = &logs.named(name)[0];
        assert_eq!(event["last_emission_ordinal"], 0);
        assert_eq!(event["usage_available"], true);
        assert_eq!(event["prompt_tokens"], 2);
        assert_eq!(event["completion_tokens"], 0);
        assert_eq!(event["total_tokens"], 2);
    }
}

#[tokio::test]
async fn real_http_completion_ids_join_traces_without_changing_wire_fields() {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    use tracing::instrument::WithSubscriber;

    let logs = Logs::new(tracing::Level::TRACE);
    let control = Arc::new(ready_control());
    let config = WorkerConfig::default();
    let (commands, receiver) = mpsc::channel(config.command_queue_capacity);
    let handle = ModelWorkerHandle {
        commands,
        control: control.clone(),
        next_request_id: Arc::new(AtomicU64::new(1)),
        config: config.clone(),
    };
    let mut engine = TraceEngine {
        hub: CompletionHub::new(),
        requests: VecDeque::new(),
        pending: None,
        finished: vec![],
        cancelled: vec![],
        failed: vec![],
        tokens: sample_tokens(),
        mode: Mode::Finished,
        cancellation: Arc::new(AtomicBool::new(false)),
        cancel_after_first: false,
    };
    let owner = InferenceCompletionOwner::attach(&mut engine);
    let app = crate::router(crate::ServerState::new(
        crate::ModelRegistration::new("test-model", ferrule_model::ChatTemplate::Plain),
        handle.clone(),
    ));
    let requests = async {
        for (i, (chat, stream)) in [(false, false), (true, false), (false, true), (true, true)]
            .into_iter()
            .enumerate()
        {
            let endpoint = if chat {
                "/v1/chat/completions"
            } else {
                "/v1/completions"
            };
            let mut input = json!({"model": "test-model", "max_tokens": 2, "stream": stream});
            if chat {
                input["messages"] = json!([{"role": "user", "content": "private prompt canary"}]);
            } else {
                input["prompt"] = json!("private prompt canary");
            }
            if stream {
                input["stream_options"] = json!({"include_usage": true});
            }
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(endpoint)
                        .header("content-type", "application/json")
                        .body(Body::from(input.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            let body = response.into_body().collect().await.unwrap().to_bytes();
            let text = std::str::from_utf8(&body).unwrap();
            let frames: Vec<Value> = if stream {
                assert!(text.contains("data: [DONE]"));
                text.lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|line| *line != "[DONE]")
                    .map(|line| serde_json::from_str(line).unwrap())
                    .collect()
            } else {
                vec![serde_json::from_slice(&body).unwrap()]
            };
            let id = (i + 1) as u64;
            let prefix = if chat {
                "chatcmpl-ferrule-"
            } else {
                "cmpl-ferrule-"
            };
            for frame in &frames {
                assert_eq!(frame["id"], json!(format!("{prefix}{id}")));
                for field in ["actual_token_id", "token_ids", "emission_ordinal"] {
                    assert!(frame.get(field).is_none());
                    for choice in frame["choices"].as_array().unwrap() {
                        assert!(choice.get(field).is_none());
                        assert!(choice["logprobs"].is_null());
                    }
                }
            }
            let finish_frames: Vec<_> = frames
                .iter()
                .filter(|frame| {
                    frame["choices"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|choice| choice["finish_reason"] == "length")
                })
                .collect();
            assert_eq!(finish_frames.len(), 1, "OpenAI projection remains length");
            let usage = &frames.last().unwrap()["usage"];
            assert_eq!(
                *usage,
                json!({"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4})
            );
            let tokens: Vec<_> = logs
                .named("request_worker_token_enqueued")
                .into_iter()
                .filter(|event| event["request_id"] == id)
                .collect();
            assert_eq!(tokens.len(), 2);
            assert_eq!(tokens[0]["emission_ordinal"], 1);
            assert_eq!(tokens[1]["emission_ordinal"], 2);
            for name in ["request_worker_terminal", "request_worker_terminal_enqueue"] {
                let events: Vec<_> = logs
                    .named(name)
                    .into_iter()
                    .filter(|event| event["request_id"] == id)
                    .collect();
                assert_eq!(events.len(), 1);
                assert_eq!(events[0]["session_id"], id);
                assert_eq!(
                    events[0]["last_emission_ordinal"],
                    usage["completion_tokens"]
                );
                for field in ["prompt_tokens", "completion_tokens", "total_tokens"] {
                    assert_eq!(events[0][field], usage[field]);
                }
            }
            let http_events: Vec<_> = logs
                .named("request_http_terminal")
                .into_iter()
                .filter(|event| event["request_id"] == id)
                .collect();
            assert_eq!(http_events.len(), 1);
            assert_eq!(http_events[0]["token_events"], 2);
            assert_eq!(http_events[0]["stream"], stream);
            assert_eq!(
                http_events[0]["finish_reason"],
                SequenceFinishReason::MaxTokens.as_str()
            );
        }
        handle.begin_shutdown();
    };
    let combined = async {
        let (result, ()) = tokio::join!(
            run_worker(engine, owner, receiver, config, control),
            requests
        );
        result.unwrap();
    };
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        combined.with_subscriber(logs.clone()),
    )
    .await
    .unwrap();
    let captured = format!("{:?}", logs.events.lock().unwrap());
    assert!(!captured.contains("private prompt canary"));
    assert!(!captured.contains("private token canary"));
}
