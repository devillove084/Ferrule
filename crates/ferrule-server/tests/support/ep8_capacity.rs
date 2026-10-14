//! Reconstructed acceptance contracts, not historical measurements or a soak.
//! In-process Router/SSE only; no TCP, process-EP or GPU memory-release claim.
use super::sse_contract::{Sample, required_u64};
use super::*;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Terminal {
    Finished,
    Cancelled,
    Failed,
}
#[derive(Debug, Default)]
struct Entry {
    tokens: usize,
    terminal: Option<Terminal>,
}
#[derive(Debug, Default)]
struct Ledger {
    requests: HashMap<u64, Entry>,
}
#[derive(Debug, Default, PartialEq, Eq)]
struct Totals {
    finished: usize,
    cancelled: usize,
    finished_disconnect_race: usize,
}

impl Ledger {
    fn admit(&mut self, id: u64) {
        assert!(
            self.requests.insert(id, Entry::default()).is_none(),
            "duplicate admission {id}"
        );
    }
    fn token(&mut self, id: u64) {
        let entry = self
            .requests
            .get_mut(&id)
            .expect("token for unknown request");
        assert!(entry.terminal.is_none(), "token after terminal");
        entry.tokens += 1;
    }
    fn terminal(&mut self, id: Option<RequestId>, terminal: Terminal) -> Result<(), String> {
        let id = id.ok_or("terminal without request ID")?.0;
        let entry = self
            .requests
            .get_mut(&id)
            .ok_or("terminal for unknown request")?;
        if entry.terminal.is_some() {
            return Err(format!("duplicate terminal for {id}"));
        }
        entry.terminal = Some(terminal);
        Ok(())
    }
    fn conserve(&self, clients: &[(u64, bool)]) -> Result<Totals, String> {
        let ids: HashSet<_> = clients.iter().map(|(id, _)| *id).collect();
        if ids.len() != clients.len() || ids.len() != self.requests.len() {
            return Err("client/admitted identities do not conserve".into());
        }
        let mut totals = Totals::default();
        for &(id, client_disconnected) in clients {
            let entry = self.requests.get(&id).ok_or("client ID was not admitted")?;
            if client_disconnected && entry.tokens == 0 {
                return Err("disconnect before real token".into());
            }
            match (entry.terminal, client_disconnected) {
                (Some(Terminal::Finished), false) => totals.finished += 1,
                (Some(Terminal::Finished), true) => totals.finished_disconnect_race += 1,
                (Some(Terminal::Cancelled), true) => totals.cancelled += 1,
                _ => {
                    return Err(format!(
                        "unexpected/missing actual terminal for {id}: {entry:?}"
                    ));
                }
            }
        }
        Ok(totals)
    }
}

#[derive(Default)]
struct CapacityProbe {
    ledger: Mutex<Ledger>,
    latest: Mutex<Value>,
    active_high_water: AtomicUsize,
    samples: AtomicUsize,
}
impl CapacityProbe {
    fn observe(&self, capacity: &Value) -> Result<(), String> {
        let active = required_u64(&capacity["scheduler"], "active_sequences")?;
        self.active_high_water
            .fetch_max(active as usize, Ordering::AcqRel);
        self.samples.fetch_add(1, Ordering::Release);
        Ok(())
    }
    fn require_active(&self, expected: usize) -> Result<(), String> {
        if self.samples.load(Ordering::Acquire) == 0
            || self.active_high_water.load(Ordering::Acquire) != expected
        {
            return Err(format!(
                "actual scheduler active high-water {}, expected {expected}",
                self.active_high_water.load(Ordering::Acquire)
            ));
        }
        Ok(())
    }
}

struct CapacityObserved {
    inner: BoxedSessionInferenceEngine,
    probe: Arc<CapacityProbe>,
}
impl CapacityObserved {
    fn sample(&self) {
        let capacity = serde_json::to_value(self.inner.capacity_snapshot()).unwrap();
        self.probe.observe(&capacity).unwrap();
        let admission = self
            .inner
            .admission_snapshot()
            .expect("resident admission contract");
        let kv = self
            .inner
            .observability_snapshot()
            .kv_cache
            .expect("resident KV contract");
        *self.probe.latest.lock().unwrap() = json!({
            "capacity":capacity,
            "waiting":admission.waiting_requests,
            "requests":admission.request_identities_held,
            "sessions":admission.session_identities_held,
            "retiring":kv.stats.retiring_pages,
        });
    }
    fn drain(&mut self, terminal: Terminal) -> Vec<SequenceState> {
        let states = match terminal {
            Terminal::Finished => self.inner.drain_finished(),
            Terminal::Cancelled => self.inner.drain_cancelled(),
            Terminal::Failed => self.inner.drain_failed(),
        };
        for state in &states {
            self.probe
                .ledger
                .lock()
                .unwrap()
                .terminal(state.request_id, terminal)
                .unwrap();
        }
        self.sample();
        states
    }
}
impl InferenceEngine for CapacityObserved {
    fn completion_hub(&self) -> CompletionHub {
        self.inner.completion_hub()
    }
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        self.inner.take_completion_reactors()
    }
    fn has_pending_async_work(&self) -> bool {
        self.inner.has_pending_async_work()
    }
    fn has_background_work(&self) -> bool {
        self.inner.has_background_work()
    }
    fn start_background_work(&mut self) -> RuntimeResult<()> {
        self.inner.start_background_work()
    }
    fn encode(&self, prompt: &str) -> RuntimeResult<Vec<u32>> {
        self.inner.encode(prompt)
    }
    fn submit(&mut self, request: GenerateRequest) {
        self.try_submit(request).unwrap();
    }
    fn try_submit(&mut self, request: GenerateRequest) -> RuntimeResult<()> {
        let id = request.id.0;
        self.inner.try_submit(request)?;
        self.probe.ledger.lock().unwrap().admit(id);
        self.sample();
        Ok(())
    }
    fn request_cleanup(&self, id: RequestId) -> ferrule_runtime::InferenceRequestCleanup {
        self.inner.request_cleanup(id)
    }
    fn admission_snapshot(&self) -> Option<ferrule_runtime::RuntimeAdmissionSnapshot> {
        self.inner.admission_snapshot()
    }
    fn set_admission_options(
        &mut self,
        options: ferrule_runtime::RuntimeAdmissionOptions,
    ) -> RuntimeResult<()> {
        self.inner.set_admission_options(options)
    }
    fn close_admission(&mut self) -> RuntimeResult<()> {
        self.inner.close_admission()
    }
    // InferenceCapacitySnapshot is not publicly nameable in current runtime.
    // Do NOT claim wrapper /admission capacity forwarding: the test reads the
    // real inner owner in sample(), not the trait's unsupported default.
    fn step(
        &mut self,
        emit: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
    ) -> RuntimeResult<ResidentDriverStep> {
        self.sample();
        let result = self.inner.step(&mut |event| {
            self.probe
                .ledger
                .lock()
                .unwrap()
                .token(event.request_id.expect("request token").0);
            emit(event)
        });
        self.sample();
        result
    }
    fn cancel_request(&mut self, id: RequestId) -> RuntimeResult<InferenceCancelProgress> {
        self.inner.cancel_request(id)
    }
    fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.drain(Terminal::Finished)
    }
    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        self.drain(Terminal::Cancelled)
    }
    fn drain_failed(&mut self) -> Vec<SequenceState> {
        self.drain(Terminal::Failed)
    }
    fn shutdown(&mut self) -> RuntimeResult<InferenceShutdownProgress> {
        let result = self.inner.shutdown()?;
        self.sample();
        Ok(result)
    }
}

fn require_clean(value: &Value) -> Result<(), String> {
    for key in ["waiting", "requests", "sessions", "retiring"] {
        if required_u64(value, key)? != 0 {
            return Err(format!("nonzero {key}: {value}"));
        }
    }
    let scheduler = &value["capacity"]["scheduler"];
    for key in ["active_sequences", "waiting_requests"] {
        if required_u64(scheduler, key)? != 0 {
            return Err(format!("nonzero {key}"));
        }
    }
    let kv = &value["capacity"]["kv"];
    if required_u64(kv, "logical_allocated_pages")? != 0
        || required_u64(kv, "logical_capacity_pages")? == 0
        || required_u64(kv, "logical_free_pages")? != required_u64(kv, "logical_capacity_pages")?
    {
        return Err(format!("KV conservation failed: {kv}"));
    }
    Ok(())
}

async fn clean(handle: &ferrule_server::ModelWorkerHandle, probe: &CapacityProbe) {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let server = handle.admission_snapshot();
            let latest = probe.latest.lock().unwrap().clone();
            if server.held_requests == 0
                && server.held_prompt_bytes == 0
                && require_clean(&latest).is_ok()
            {
                assert_eq!(server.available_requests, server.request_limit);
                assert_eq!(server.available_prompt_bytes, server.prompt_bytes_limit);
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("physical cleanup/admission deadline");
}

fn request(prompt: &str, output: usize) -> Request<Body> {
    post(
        "/v1/completions",
        json!({"model":"hybrid","prompt":prompt,"max_tokens":output,"ignore_eos":true,"stream":true,"stream_options":{"include_usage":true}}),
    )
}
async fn completed(app: axum::Router, prompt: &str, output: usize, prompt_tokens: usize) -> Sample {
    let response = app.oneshot(request(prompt, output)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/event-stream"
    );
    let mut body = response.into_body();
    let mut sample = Sample::default();
    while let Some(frame) = body.frame().await {
        if let Some(data) = frame.unwrap().data_ref() {
            sample.push(data).unwrap();
        }
    }
    sample.require_length(output, Some(prompt_tokens)).unwrap();
    sample
}
async fn disconnected(app: axum::Router, prompt: &str) -> u64 {
    let response = app.oneshot(request(prompt, 24)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let mut sample = Sample::default();
    while sample.tokens == 0 {
        let frame = body
            .frame()
            .await
            .expect("disconnect stream EOF before token")
            .unwrap();
        if let Some(data) = frame.data_ref() {
            sample.push(data).unwrap();
        }
    }
    assert!(
        sample.finish_reason.is_none(),
        "stream terminal before disconnect"
    );
    let id = sample.request_id().unwrap();
    drop(body);
    id // Client disconnect is an observation, NOT an actual Cancelled terminal.
}

#[test]
fn terminal_conservation_rejects_missing_duplicate_wrong_id_and_failed() {
    let mut ledger = Ledger::default();
    ledger.admit(1);
    assert!(ledger.conserve(&[(1, true)]).is_err());
    ledger.token(1);
    assert!(ledger.terminal(None, Terminal::Cancelled).is_err());
    assert!(
        ledger
            .terminal(Some(RequestId(2)), Terminal::Cancelled)
            .is_err()
    );
    ledger
        .terminal(Some(RequestId(1)), Terminal::Cancelled)
        .unwrap();
    assert!(
        ledger
            .terminal(Some(RequestId(1)), Terminal::Finished)
            .is_err()
    );
    assert!(ledger.conserve(&[(2, true)]).is_err());
    assert!(ledger.conserve(&[(1, true), (1, true)]).is_err());
    assert!(ledger.conserve(&[(1, false)]).is_err());
    assert_eq!(ledger.conserve(&[(1, true)]).unwrap().cancelled, 1);
    ledger.admit(2);
    ledger.token(2);
    ledger
        .terminal(Some(RequestId(2)), Terminal::Failed)
        .unwrap();
    assert!(ledger.conserve(&[(1, true), (2, true)]).is_err());
}

#[test]
fn active_high_water_uses_running_sequences_not_limit_or_waiting() {
    let probe = CapacityProbe::default();
    assert!(probe.require_active(0).is_err());
    probe.observe(&json!({"scheduler":{"active_sequences":2,"waiting_requests":6,"max_active_sequences":8}})).unwrap();
    assert!(probe.require_active(8).is_err());
    probe.observe(&json!({"scheduler":{"active_sequences":8,"waiting_requests":0,"max_active_sequences":8}})).unwrap();
    probe.require_active(8).unwrap();
    assert!(
        probe
            .observe(&json!({"scheduler":{"active_sequences":null}}))
            .is_err()
    );
}

#[test]
fn cleanup_conservation_rejects_null_missing_and_retiring_resources() {
    let value = json!({"waiting":0,"requests":0,"sessions":0,"retiring":0,"capacity":{
        "scheduler":{"active_sequences":0,"waiting_requests":0},
        "kv":{"logical_allocated_pages":0,"logical_free_pages":16,"logical_capacity_pages":16}}});
    require_clean(&value).unwrap();
    for pointer in [
        "/requests",
        "/retiring",
        "/capacity/scheduler/active_sequences",
        "/capacity/kv/logical_free_pages",
        "/capacity/kv/logical_capacity_pages",
    ] {
        let mut bad = value.clone();
        *bad.pointer_mut(pointer).unwrap() = Value::Null;
        assert!(require_clean(&bad).is_err(), "{pointer}");
    }
    let mut bad = value;
    bad["retiring"] = json!(1);
    assert!(require_clean(&bad).is_err());
    assert!(require_clean(&json!({})).is_err());
}

#[test]
fn real_cpu_wrapper_distinguishes_cancelled_from_finished_disconnect_race() {
    let probe = Arc::new(CapacityProbe::default());
    let mut engine = CapacityObserved {
        inner: cpu_engine(fixture::Fixture::new()).unwrap(),
        probe: probe.clone(),
    };
    for (id, output) in [(1, 1), (2, 24)] {
        engine
            .try_submit(GenerateRequest {
                id: RequestId(id),
                session_id: Some(SessionId(id)),
                prompt_tokens: engine.encode("t1 t3 t2").unwrap(),
                max_new_tokens: output,
                stop: vec![],
                ignore_eos: true,
            })
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut cancel_sent = false;
        loop {
            engine.step(&mut |_| Ok(())).unwrap();
            engine.drain_finished();
            engine.drain_cancelled();
            engine.drain_failed();
            let entry = &probe.ledger.lock().unwrap().requests[&id];
            let (tokens, terminal) = (entry.tokens, entry.terminal);
            if terminal.is_some() {
                break;
            }
            if id == 2 && tokens > 0 && !cancel_sent {
                engine.cancel_request(RequestId(id)).unwrap();
                cancel_sent = true;
            }
            assert!(Instant::now() < deadline);
        }
    }
    let totals = probe
        .ledger
        .lock()
        .unwrap()
        .conserve(&[(1, true), (2, true)])
        .unwrap();
    assert_eq!(
        totals,
        Totals {
            finished: 0,
            cancelled: 1,
            finished_disconnect_race: 1
        }
    );
    require_clean(&probe.latest.lock().unwrap()).unwrap();
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn real_cpu_http_strict_sse_terminal_and_cleanup_conservation() {
    let probe = Arc::new(CapacityProbe::default());
    let owner_probe = probe.clone();
    let worker = spawn_model_worker_with(
        move || {
            Ok::<_, ferrule_runtime::Error>(CapacityObserved {
                inner: cpu_engine(fixture::Fixture::new())?,
                probe: owner_probe,
            })
        },
        WorkerConfig {
            event_queue_capacity: 4,
            ..Default::default()
        },
    )
    .unwrap();
    let handle = worker.handle();
    let app = router(ServerState::new(
        ModelRegistration::new("hybrid", ChatTemplate::Plain),
        handle.clone(),
    ));
    tokio::time::timeout(DEADLINE, async {
        let sample = completed(app.clone(), "t1 t3 t2", 2, 3).await;
        assert_eq!(sample.tokens, 2);
        // The very helper used by GPU smoke must accept real CPU output too.
        let body = text(app.clone().oneshot(request("t1 t3 t2", 2)).await.unwrap()).await;
        require_ep8_usage("real-cpu", &body, 2).unwrap();
        let second = Sample::parse(&body).unwrap();
        clean(&handle, &probe).await;
        let wave = short_wave(app.clone(), "t1 t3 t2", 3, 2).await;
        clean(&handle, &probe).await;
        let cancelled = disconnected(app.clone(), "t1 t3 t2").await;
        let recovered = completed(app, "t1 t3 t2", 1, 3).await;
        clean(&handle, &probe).await;
        let mut clients = vec![
            (sample.request_id().unwrap(), false),
            (second.request_id().unwrap(), false),
            (cancelled, true),
            (recovered.request_id().unwrap(), false),
        ];
        clients.extend(
            wave.iter()
                .map(|sample| (sample.request_id().unwrap(), false)),
        );
        let totals = probe.ledger.lock().unwrap().conserve(&clients).unwrap();
        assert_eq!(totals.finished, 11);
        assert_eq!(totals.cancelled + totals.finished_disconnect_race, 1);
        probe.require_active(1).unwrap();
    })
    .await
    .unwrap();
    worker.shutdown().await.unwrap();
    assert!(handle.snapshot().first_fatal.is_none());
    assert!(!handle.snapshot().shutdown_report.quarantine);
}

async fn short_wave(
    app: axum::Router,
    prompt: &str,
    prompt_tokens: usize,
    output: usize,
) -> Vec<Sample> {
    let barrier = Arc::new(tokio::sync::Barrier::new(9));
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..8 {
        let app = app.clone();
        let prompt = prompt.to_owned();
        let barrier = barrier.clone();
        tasks.spawn(async move {
            barrier.wait().await;
            completed(app, &prompt, output, prompt_tokens).await
        });
    }
    barrier.wait().await;
    let mut samples = vec![];
    while let Some(result) = tasks.join_next().await {
        samples.push(result.unwrap());
    }
    assert_eq!(samples.len(), 8);
    samples
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires exclusive EP8 GPUs and FERRULE_NUMERIC_FP8_MODEL_DIR; 1k prompt/ctx2048, active8, two rounds; NOT a soak or historical result"]
async fn qwen35_35b_fp8_ep8_http_capacity_1k_active8_and_shutdown() {
    cfg!(feature = "cuda")
        .then_some(())
        .expect("enable --features cuda for the ignored EP8 acceptance test");
    use ferrule_composition::{BackendSelection, ModelFactoryOptions, ResidentModelPlanner};

    let path = std::path::PathBuf::from(
        std::env::var_os("FERRULE_NUMERIC_FP8_MODEL_DIR")
            .expect("provide the real full 35B FP8 checkpoint explicitly"),
    );
    assert!(path.join("config.json").is_file());
    let processes = std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ])
        .output()
        .expect("exclusive GPU process preflight");
    assert!(processes.status.success());
    assert!(
        processes.stdout.iter().all(u8::is_ascii_whitespace),
        "refusing occupied GPUs"
    );
    let tokenizer = TokenizerHandle::load(&path).unwrap();
    let mut prompt = String::from("Read these notes and continue the final sentence.\n");
    while tokenizer.encode(&prompt).unwrap().len() < 1024 {
        prompt.push_str(
            "The river flows through the valley and the village stands beside the bridge.\n",
        );
    }
    let long_tokens = tokenizer.encode(&prompt).unwrap().len();
    let short_tokens = tokenizer.encode("Hello").unwrap().len();
    assert!((1024..=1100).contains(&long_tokens));
    assert!(long_tokens + 32 <= 2048);
    let config = ferrule_model::AutoConfig::from_pretrained(&path).unwrap();
    let options = ModelFactoryOptions {
        max_layers: None,
        max_tensor_mebibytes: 1024,
        output_head_chunk_rows: 4096,
        expert_reader_max_tensor_mebibytes: 64,
        expert_cache: Default::default(),
        qwen35_moe_capacity: None,
        numeric_fp8_profile: None,
        qwen35_host_cache: Some(
            ferrule_model::transformer::host_experts::HostExpertCacheOptions {
                max_bytes: 32u64 << 30,
                ..Default::default()
            },
        ),
        moe_hotset_experts: 0,
        kv_cache_mebibytes: Some(1280),
        scheduler_config: ferrule_runtime::ResidentSchedulerConfig {
            max_active_sequences: 8,
            max_decode_batch: 8,
            decode_cohort_target: 8,
            max_batch_tokens: 16,
            ..Default::default()
        },
        driver_config: ferrule_runtime::ResidentTopKDriverConfig {
            ctx_size: 2048,
            enable_native_proposals: false,
            ..Default::default()
        },
    };
    let plan = ResidentModelPlanner
        .prepare_qwen35_expert_parallel(
            &config,
            BackendSelection::Cuda,
            None,
            options,
            ferrule_composition::model_factory::PipelineBuildOptions {
                parallelism: ferrule_common::ParallelismPlan {
                    expert_parallel: 8,
                    ..Default::default()
                },
                devices: Some((0..8).collect()),
                ..Default::default()
            },
        )
        .expect("physical-card planning must pass; do not enlarge budgets to pass this test");
    assert!(plan.qwen35_expert_placement().is_some());
    assert!(plan.pipeline_options().is_none());
    let probe = Arc::new(CapacityProbe::default());
    let owner_probe = probe.clone();
    let worker = spawn_model_worker_with(
        move || {
            Ok::<_, ferrule_runtime::Error>(CapacityObserved {
                inner: plan.build()?,
                probe: owner_probe,
            })
        },
        WorkerConfig {
            event_queue_capacity: 64,
            admission_timeout: Duration::from_secs(180),
            shutdown_timeout: Duration::from_secs(120),
            ..Default::default()
        },
    )
    .unwrap();
    let handle = worker.handle();
    let app = router(ServerState::new(
        ModelRegistration::new("hybrid", ChatTemplate::Qwen35),
        handle.clone(),
    ));
    let task_handle = handle.clone();
    // Isolate assertion failures so even a failed contract reaches shutdown.
    let workload = tokio::spawn(async move {
        tokio::time::timeout(Duration::from_secs(900), async {
            let mut clients = vec![];
            for round in 0..2 {
                let (long, short) = tokio::join!(
                    completed(app.clone(), &prompt, 32, long_tokens),
                    completed(app.clone(), "Hello", 32, short_tokens),
                );
                clients.extend([
                    (long.request_id().unwrap(), false),
                    (short.request_id().unwrap(), false),
                ]);
                clean(&task_handle, &probe).await;
                probe.active_high_water.store(0, Ordering::Release);
                probe.samples.store(0, Ordering::Release);
                for sample in short_wave(app.clone(), "Hello", short_tokens, 32).await {
                    clients.push((sample.request_id().unwrap(), false));
                }
                probe.require_active(8).unwrap(); // Not configured limit or HTTP held=8.
                clean(&task_handle, &probe).await;
                clients.push((disconnected(app.clone(), "Hello").await, true));
                for _ in 0..2 {
                    clients.push((
                        completed(app.clone(), "Hello", 32, short_tokens)
                            .await
                            .request_id()
                            .unwrap(),
                        false,
                    ));
                }
                clean(&task_handle, &probe).await;
                let totals = probe.ledger.lock().unwrap().conserve(&clients).unwrap();
                assert_eq!(totals.finished, (round + 1) * 12);
                assert_eq!(
                    totals.cancelled + totals.finished_disconnect_race,
                    round + 1
                );
                eprintln!(
                    "EP8 contract round={round} actual terminals={totals:?}; snapshot={}",
                    probe.latest.lock().unwrap()
                );
            }
        })
        .await
        .expect("bounded EP8 workload deadline");
    })
    .await;
    let shutdown = tokio::time::timeout(Duration::from_secs(125), worker.shutdown()).await;
    shutdown.expect("EP8 shutdown deadline").unwrap();
    workload.expect("EP8 acceptance contract failed");
    let snapshot = handle.snapshot();
    assert_eq!(snapshot.phase, ferrule_server::WorkerPhase::Stopped);
    assert!(snapshot.first_fatal.is_none());
    assert!(!snapshot.shutdown_report.quarantine);
    assert!(!snapshot.shutdown_report.custody_unknown);
    assert_eq!(snapshot.shutdown_report.remaining_operations, Some(0));
    assert!(snapshot.shutdown_report.remaining_requests.is_empty());
}
