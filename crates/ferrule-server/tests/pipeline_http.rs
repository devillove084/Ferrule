//! Real tiny checkpoint -> catalog/build plan -> pipeline -> worker -> HTTP/SSE.
//! Test wrappers observe/control owner boundaries; they never implement decoder
//! math, page allocation, pipeline dispatch or a substitute HTTP server.
//! Process tests build their matching child once, unless FERRULE_PIPELINE_CHILD is explicit.

#[cfg(unix)]
#[path = "../../ferrule-runtime/tests/support/build_process_child.rs"]
mod build_process_child;

use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use ferrule_common::{CompletionHub, ParallelismPlan};
use ferrule_model::{AutoConfig, ChatTemplate};
use ferrule_runtime::engine::model_factory::{PipelineBuildOptions, PipelineRankBackend};
use ferrule_runtime::engine::{BoxedSessionInferenceEngine, SessionInferenceEngine};
#[cfg(unix)]
use ferrule_runtime::parallel::process::ProcessLaunch;
use ferrule_runtime::{
    BackendSelection, CancelRequestResult, GenerateRequest, InferenceCancelProgress,
    InferenceCompletionReactor, InferenceEngine, InferenceShutdownProgress, ModelFactoryOptions,
    RequestId, ResidentDriverStep, ResidentModelBuildPlan, ResidentModelPlanner,
    ResidentSchedulerConfig, ResidentTokenEvent, ResidentTopKDriverConfig, Result as RuntimeResult,
    SequenceState, SessionId,
};
use ferrule_server::{
    ModelRegistration, ModelWorker, ServerState, WorkerConfig, router, spawn_model_worker_with,
};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

const CONTEXT: usize = 64;
const DEADLINE: Duration = Duration::from_secs(10);

struct Fixture(PathBuf);
impl Fixture {
    fn new(eos: u32) -> Self {
        Self::with_experts(eos, false)
    }
    fn with_experts(eos: u32, moe: bool) -> Self {
        Self::with_variant(eos, moe, false)
    }
    fn with_variant(eos: u32, moe: bool, tensor: bool) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrule-pipeline-http-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        let mut config = json!({
            "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3",
            "hidden_size":4, "intermediate_size":8, "num_hidden_layers":2,
            "num_attention_heads":if tensor { 4 } else { 2 },
            "num_key_value_heads":if tensor { 4 } else { 1 }, "head_dim":2,
            "rms_norm_eps":0.000001, "rope_theta":10000.0, "rope_scaling":null,
            "max_position_embeddings":CONTEXT, "vocab_size":8, "tie_word_embeddings":true,
            "attention_bias":false, "attention_dropout":0.0, "hidden_act":"silu",
            "torch_dtype":"bfloat16", "use_cache":true, "use_sliding_window":false,
            "sliding_window":null, "max_window_layers":2, "initializer_range":0.02,
            "bos_token_id":2, "eos_token_id":eos
        });
        if moe {
            config["architectures"] = json!(["Qwen3MoeForCausalLM"]);
            config["model_type"] = json!("qwen3_moe");
            config.as_object_mut().unwrap().extend(
                json!({
                    "moe_intermediate_size":8, "num_experts":2, "num_experts_per_tok":2,
                    "norm_topk_prob":true, "output_router_logits":false,
                    "router_aux_loss_coef":0.001, "mlp_only_layers":[], "decoder_sparse_step":1
                })
                .as_object()
                .unwrap()
                .clone(),
            );
        }
        std::fs::write(path.join("config.json"), config.to_string()).unwrap();
        std::fs::write(path.join("tokenizer.json"), json!({
            "version":"1.0", "truncation":null, "padding":null, "added_tokens":[],
            "normalizer":null, "pre_tokenizer":{"type":"WhitespaceSplit"},
            "post_processor":null, "decoder":null,
            "model":{"type":"WordLevel", "vocab":{"<unk>":0,"a":1,"hello":2,"b":3,"c":4,"d":5,"e":6,"<eos>":7},"unk_token":"<unk>"}
        }).to_string()).unwrap();
        let mut tensors = vec![
            ("model.embed_tokens.weight".to_owned(), vec![8, 4]),
            ("model.norm.weight".to_owned(), vec![4]),
        ];
        for layer in 0..2 {
            for (name, shape) in [
                ("input_layernorm.weight", vec![4]),
                ("post_attention_layernorm.weight", vec![4]),
                (
                    "self_attn.q_proj.weight",
                    vec![if tensor { 8 } else { 4 }, 4],
                ),
                (
                    "self_attn.k_proj.weight",
                    vec![if tensor { 8 } else { 2 }, 4],
                ),
                (
                    "self_attn.v_proj.weight",
                    vec![if tensor { 8 } else { 2 }, 4],
                ),
                (
                    "self_attn.o_proj.weight",
                    vec![4, if tensor { 8 } else { 4 }],
                ),
                ("self_attn.q_norm.weight", vec![2]),
                ("self_attn.k_norm.weight", vec![2]),
                ("mlp.gate_proj.weight", vec![8, 4]),
                ("mlp.up_proj.weight", vec![8, 4]),
                ("mlp.down_proj.weight", vec![4, 8]),
            ] {
                if !moe || !name.starts_with("mlp.") {
                    tensors.push((format!("model.layers.{layer}.{name}"), shape));
                }
            }
            if moe {
                tensors.push((format!("model.layers.{layer}.mlp.gate.weight"), vec![2, 4]));
                for expert in 0..2 {
                    for (name, shape) in [
                        ("gate_proj", vec![8, 4]),
                        ("up_proj", vec![8, 4]),
                        ("down_proj", vec![4, 8]),
                    ] {
                        tensors.push((
                            format!("model.layers.{layer}.mlp.experts.{expert}.{name}.weight"),
                            shape,
                        ));
                    }
                }
            }
        }
        let mut payload = Vec::new();
        let mut header = serde_json::Map::new();
        for (name, shape) in tensors {
            let start = payload.len();
            for index in 0..shape.iter().product::<usize>() {
                // Nonzero embedding and identity residuals make tied-head greedy
                // token 1 deterministic while running every real decoder layer.
                let value: f32 = if name == "model.embed_tokens.weight" {
                    if index / 4 == 1 { 0.5 } else { 0.125 }
                } else if shape.len() == 1 {
                    1.0
                } else if moe || tensor {
                    // Exercise nonzero attention/MLP/router/expert computation, not
                    // merely the identity residual. Both experts are selected.
                    ((index % 7) as f32 + 1.0) / 128.0
                } else {
                    0.0
                };
                payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
            header.insert(
                name,
                json!({"dtype":"BF16", "shape":shape, "data_offsets":[start,payload.len()]}),
            );
        }
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(payload);
        std::fs::write(path.join("model.safetensors"), file).unwrap();
        Self(path)
    }

    fn options(capacity: usize) -> ModelFactoryOptions {
        ModelFactoryOptions {
            max_layers: None,
            max_tensor_mebibytes: 1,
            output_head_chunk_rows: 8,
            expert_reader_max_tensor_mebibytes: 1,
            expert_cache: Default::default(),
            qwen35_moe_capacity: None,
            qwen35_host_cache: None,
            moe_hotset_experts: 0,
            kv_cache_mebibytes: None,
            scheduler_config: ResidentSchedulerConfig {
                max_active_sequences: capacity,
                max_batch_tokens: 4,
                prefill_chunk_size: 4,
                ..Default::default()
            },
            driver_config: ResidentTopKDriverConfig {
                ctx_size: CONTEXT,
                ..Default::default()
            },
        }
    }
    fn plan(&self, capacity: usize) -> ResidentModelBuildPlan {
        let config = AutoConfig::from_pretrained(&self.0).unwrap();
        ResidentModelPlanner::new()
            .prepare(
                &config,
                BackendSelection::Cpu,
                Some("plain"),
                Self::options(capacity),
            )
            .unwrap()
    }
    fn parallel_plan(
        &self,
        backend: BackendSelection,
        pp: usize,
        ep: usize,
        devices: Option<Vec<usize>>,
    ) -> ResidentModelBuildPlan {
        let config = AutoConfig::from_pretrained(&self.0).unwrap();
        ResidentModelPlanner::new()
            .prepare_pipeline(
                &config,
                backend,
                Some("plain"),
                Self::options(1),
                PipelineBuildOptions {
                    parallelism: ParallelismPlan {
                        pipeline_parallel: pp,
                        expert_parallel: ep,
                        ..Default::default()
                    },
                    devices,
                    ..Default::default()
                },
            )
            .unwrap()
    }
    #[cfg(feature = "cuda")]
    fn tensor_plan(&self, pp: usize, tp: usize) -> ResidentModelBuildPlan {
        let config = AutoConfig::from_pretrained(&self.0).unwrap();
        let mut options = Self::options(1);
        options.kv_cache_mebibytes = Some(1);
        ResidentModelPlanner::new()
            .prepare_pipeline(
                &config,
                BackendSelection::Cuda,
                Some("plain"),
                options,
                PipelineBuildOptions {
                    parallelism: ParallelismPlan {
                        pipeline_parallel: pp,
                        tensor_parallel: tp,
                        ..Default::default()
                    },
                    devices: Some((0..pp * tp).rev().collect()),
                    ..Default::default()
                },
            )
            .unwrap()
    }

    #[cfg(unix)]
    fn process_plan(
        &self,
        backend: BackendSelection,
        pp: usize,
        ep: usize,
        devices: Option<Vec<usize>>,
    ) -> ResidentModelBuildPlan {
        let config = AutoConfig::from_pretrained(&self.0).unwrap();
        static CHILD: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
        let executable = CHILD.get_or_init(|| {
            std::env::var_os("FERRULE_PIPELINE_CHILD")
                .map(PathBuf::from)
                .unwrap_or_else(|| {
                    build_process_child::build(
                        "ferrule-server",
                        "example",
                        "pipeline_process_child",
                    )
                })
        });
        let evidence = self.0.join("process-owners");
        std::fs::create_dir_all(&evidence).unwrap();
        let mut launch =
            ProcessLaunch::new(executable).env("FERRULE_PIPELINE_CHILD_EVIDENCE", evidence);
        if let Some(devices) = std::env::var_os("FERRULE_PIPELINE_CHILD_CUDA_VISIBLE_DEVICES") {
            launch = launch.env("CUDA_VISIBLE_DEVICES", devices);
        }
        ResidentModelPlanner::new()
            .prepare_pipeline(
                &config,
                backend,
                Some("plain"),
                Self::options(1),
                PipelineBuildOptions {
                    parallelism: ParallelismPlan {
                        pipeline_parallel: pp,
                        expert_parallel: ep,
                        ..Default::default()
                    },
                    rank_backend: PipelineRankBackend::Process,
                    process_launch: Some(launch),
                    devices,
                    rank_timeout: Duration::from_secs(30),
                    ..Default::default()
                },
            )
            .unwrap()
    }

    #[cfg(unix)]
    fn assert_process_owners_stopped(&self, pp: usize, ep: usize, cuda: bool) {
        let mut records = Vec::new();
        for entry in std::fs::read_dir(self.0.join("process-owners")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|ext| ext == "json") {
                assert!(
                    path.with_extension("stopped").is_file(),
                    "owner missed shutdown: {path:?}"
                );
                records
                    .push(serde_json::from_slice::<Value>(&std::fs::read(path).unwrap()).unwrap());
            }
        }
        assert_eq!(records.len(), pp * if ep > 1 { ep + 1 } else { 1 });
        let mut ranks = std::collections::BTreeSet::new();
        let mut pids = std::collections::BTreeSet::new();
        for record in records {
            let pid = record["pid"].as_u64().unwrap();
            assert_ne!(pid, u64::from(std::process::id()));
            assert!(pids.insert(pid));
            let rank = record["identity"]["rank"].as_u64().unwrap() as usize;
            assert!(ranks.insert(rank));
            let wrapper = &record["config"];
            let boot = wrapper.get("expert_boot").unwrap_or(wrapper);
            assert_eq!(
                boot["version"],
                ferrule_runtime::parallel::process::decoder::DECODER_WIRE_VERSION
            );
            if ep > 1 {
                let experts = &boot["experts"];
                assert_eq!(experts["source_scope"], "external_stage");
                assert_eq!(experts["source"], boot["rank"]);
                assert!(
                    !experts["members"]
                        .as_array()
                        .unwrap()
                        .contains(&boot["rank"])
                );
            }
            let precision = if cuda || ep > 1 {
                "f32"
            } else {
                "bf16_compatibility"
            };
            assert_eq!(boot["precision"], precision);
            let device = if rank < pp {
                &boot["device"]
            } else {
                let slot = (rank - pp) % ep;
                assert_eq!(wrapper["owner"], rank);
                &boot["experts"]["devices"][slot]
            };
            if cuda {
                assert_eq!(device["cuda"]["ordinal"], rank);
            } else {
                assert_eq!(device, "cpu");
            }
        }
    }

    fn pipeline_plan(&self, capacity: usize) -> ResidentModelBuildPlan {
        self.plan(capacity)
            .with_pipeline(PipelineBuildOptions {
                parallelism: ParallelismPlan {
                    pipeline_parallel: 2,
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[derive(Default)]
struct Probe {
    cancelled: AtomicUsize,
    prefills: AtomicUsize,
    shutdowns: AtomicUsize,
}
struct Gate {
    entered: Sender<()>,
    release: Receiver<()>,
}
struct ObservedEngine {
    inner: BoxedSessionInferenceEngine,
    probe: Arc<Probe>,
    gate: Option<Gate>,
    fail_after_prefill: bool,
}
impl InferenceEngine for ObservedEngine {
    fn completion_hub(&self) -> CompletionHub {
        self.inner.completion_hub()
    }
    fn take_completion_reactors(&mut self) -> Vec<InferenceCompletionReactor> {
        self.inner.take_completion_reactors()
    }
    fn has_pending_async_work(&self) -> bool {
        self.inner.has_pending_async_work()
    }
    fn encode(&self, text: &str) -> RuntimeResult<Vec<u32>> {
        self.inner.encode(text)
    }
    fn submit(&mut self, request: GenerateRequest) {
        self.inner.submit(request);
    }
    fn request_cleanup(&self, request: RequestId) -> ferrule_runtime::InferenceRequestCleanup {
        self.inner.request_cleanup(request)
    }
    fn try_submit(&mut self, request: GenerateRequest) -> RuntimeResult<()> {
        self.inner.try_submit(request)
    }
    fn step(
        &mut self,
        on_token: &mut dyn FnMut(&ResidentTokenEvent) -> RuntimeResult<()>,
    ) -> RuntimeResult<ResidentDriverStep> {
        if self.fail_after_prefill && self.probe.prefills.load(Ordering::Acquire) > 0 {
            return Err(ferrule_runtime::Error::Invariant {
                message: "injected serving failure after real pipeline prefill".into(),
            });
        }
        let mut emitted = false;
        let step = self.inner.step(&mut |event| {
            emitted = true;
            on_token(event)
        })?;
        if matches!(
            step,
            ResidentDriverStep::Executed {
                action_kind: ferrule_runtime::ResidentActionKind::Prefill,
                ..
            }
        ) {
            self.probe.prefills.fetch_add(1, Ordering::Release);
        }
        if emitted && let Some(gate) = self.gate.take() {
            gate.entered.send(()).unwrap();
            gate.release
                .recv_timeout(DEADLINE)
                .expect("test must release owner gate");
        }
        Ok(step)
    }
    fn cancel_request(&mut self, request: RequestId) -> RuntimeResult<InferenceCancelProgress> {
        let result = self.inner.cancel_request(request)?;
        if matches!(
            result,
            InferenceCancelProgress::Complete(
                CancelRequestResult::Active { .. } | CancelRequestResult::Waiting { .. }
            )
        ) {
            self.probe.cancelled.fetch_add(1, Ordering::Release);
        }
        Ok(result)
    }
    fn drain_finished(&mut self) -> Vec<SequenceState> {
        self.inner.drain_finished()
    }
    fn drain_cancelled(&mut self) -> Vec<SequenceState> {
        self.inner.drain_cancelled()
    }
    fn drain_failed(&mut self) -> Vec<SequenceState> {
        self.inner.drain_failed()
    }
    fn shutdown(&mut self) -> RuntimeResult<InferenceShutdownProgress> {
        let result = self.inner.shutdown()?;
        if result == InferenceShutdownProgress::Complete {
            let kv = self.inner.observability_snapshot().kv_cache.unwrap();
            assert_eq!(kv.stats.allocated_pages, 0);
            assert_eq!(kv.stats.retiring_pages, 0);
            self.probe.shutdowns.fetch_add(1, Ordering::Release);
        }
        Ok(result)
    }
}

fn start(
    fixture: &Fixture,
    events: usize,
    gate: Option<Gate>,
    fail: bool,
) -> (axum::Router, ModelWorker, Arc<Probe>) {
    let plan = fixture.pipeline_plan(1);
    assert_eq!(plan.backend_profile(), "cpu-pipeline-serial");
    start_plan(plan, events, gate, fail)
}
fn start_plan(
    plan: ResidentModelBuildPlan,
    events: usize,
    gate: Option<Gate>,
    fail: bool,
) -> (axum::Router, ModelWorker, Arc<Probe>) {
    let probe = Arc::new(Probe::default());
    let owner_probe = Arc::clone(&probe);
    let worker = spawn_model_worker_with(
        move || -> RuntimeResult<ObservedEngine> {
            Ok(ObservedEngine {
                inner: plan.build()?,
                probe: owner_probe,
                gate,
                fail_after_prefill: fail,
            })
        },
        WorkerConfig {
            event_queue_capacity: events,
            max_inflight_requests: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let app = router(ServerState::new(
        ModelRegistration::new("tiny-pipeline", ChatTemplate::Plain),
        worker.handle(),
    ));
    (app, worker, probe)
}
fn post(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}
fn completion(stream: bool, count: usize) -> Value {
    json!({"model":"tiny-pipeline", "prompt":"hello", "max_tokens":count, "stream":stream})
}
async fn text(response: axum::response::Response) -> String {
    let bytes = tokio::time::timeout(DEADLINE, response.into_body().collect())
        .await
        .unwrap()
        .unwrap()
        .to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}
async fn stop_worker(worker: ModelWorker, probe: &Probe) {
    tokio::time::timeout(DEADLINE, worker.shutdown())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(probe.shutdowns.load(Ordering::Acquire), 1);
}
async fn wait_credits_returned(handle: &ferrule_server::ModelWorkerHandle) {
    tokio::time::timeout(DEADLINE, async {
        while handle.admission_snapshot().held_requests != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let snapshot = handle.admission_snapshot();
    assert_eq!(snapshot.available_requests, 2);
    assert_eq!(snapshot.held_prompt_bytes, 0);
}

async fn wait_cancel(probe: &Probe) {
    tokio::time::timeout(DEADLINE, async {
        while probe.cancelled.load(Ordering::Acquire) == 0 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();
}
fn sse_values(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|line| *line != "[DONE]")
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[tokio::test]
async fn tiny_checkpoint_serves_tokenization_nonstream_and_both_sse_endpoints() {
    let fixture = Fixture::new(7);
    let (app, worker, probe) = start(&fixture, 64, None, false);
    let response = app
        .clone()
        .oneshot(post(
            "/v1/tokenize",
            json!({"model":"tiny-pipeline","prompt":"hello hello"}),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value: Value = serde_json::from_str(&text(response).await).unwrap();
    assert_eq!(value["data"][0]["tokens"], json!([2, 2]));
    let response = app
        .clone()
        .oneshot(post("/v1/completions", completion(false, 3)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value: Value = serde_json::from_str(&text(response).await).unwrap();
    assert_eq!(value["choices"][0]["text"], "a a a");
    assert_eq!(value["usage"]["completion_tokens"], 3);
    assert_eq!(value["usage"]["prompt_tokens"], 1);
    for chat in [false, true] {
        let mut request = if chat {
            json!({"model":"tiny-pipeline", "messages":[{"role":"user","content":"hello"}], "max_completion_tokens":3,"stream":true})
        } else {
            completion(true, 3)
        };
        request["stream_options"] = json!({"include_usage":true});
        let response = app
            .clone()
            .oneshot(post(
                if chat {
                    "/v1/chat/completions"
                } else {
                    "/v1/completions"
                },
                request,
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let body = text(response).await;
        assert_eq!(body.matches("data: [DONE]").count(), 1);
        let values = sse_values(&body);
        let content = values
            .iter()
            .filter_map(|v| {
                if chat {
                    v["choices"][0]["delta"]["content"].as_str()
                } else {
                    v["choices"][0]["text"].as_str()
                }
            })
            .collect::<String>();
        assert_eq!(content, "a a a");
        assert!(
            values
                .iter()
                .any(|v| v["choices"][0]["finish_reason"] == "length")
        );
        assert!(values.iter().any(|v| v["usage"]["completion_tokens"] == 3));
    }
    assert!(probe.prefills.load(Ordering::Acquire) >= 3);
    stop_worker(worker, &probe).await;
}

#[tokio::test]
async fn tiny_checkpoint_stop_is_withheld_across_tokens_and_flushed_at_length() {
    let fixture = Fixture::new(7);
    let (app, worker, probe) = start(&fixture, 64, None, false);
    for stream in [false, true] {
        let mut request = completion(stream, 8);
        request["stop"] = json!(["a a"]);
        if stream {
            request["stream_options"] = json!({"include_usage":true});
        }
        let response = app
            .clone()
            .oneshot(post("/v1/completions", request))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = text(response).await;
        if stream {
            let values = sse_values(&body);
            assert_eq!(
                values
                    .iter()
                    .filter_map(|v| v["choices"][0]["text"].as_str())
                    .collect::<String>(),
                ""
            );
            assert!(
                values
                    .iter()
                    .any(|v| v["choices"][0]["finish_reason"] == "stop")
            );
            assert!(values.iter().any(|v| v["usage"]["completion_tokens"] == 2));
        } else {
            let value: Value = serde_json::from_str(&body).unwrap();
            assert_eq!(value["choices"][0]["text"], "");
            assert_eq!(value["choices"][0]["finish_reason"], "stop");
            assert_eq!(value["usage"]["completion_tokens"], 2);
        }
    }
    let mut request = completion(false, 1);
    request["stop"] = json!("a a");
    let response = app.oneshot(post("/v1/completions", request)).await.unwrap();
    let value: Value = serde_json::from_str(&text(response).await).unwrap();
    assert_eq!(value["choices"][0]["text"], "a");
    assert_eq!(value["usage"]["completion_tokens"], 1);
    stop_worker(worker, &probe).await;
}

#[tokio::test]
async fn tiny_checkpoint_eos_and_ignore_eos_use_the_real_tokenizer() {
    let fixture = Fixture::new(1);
    for stop_at_eos in [false, true] {
        let mut options = Fixture::options(1);
        options.driver_config.stop_at_eos = stop_at_eos;
        let plan = ResidentModelPlanner::new()
            .prepare_pipeline(
                &AutoConfig::from_pretrained(&fixture.0).unwrap(),
                BackendSelection::Cpu,
                Some("plain"),
                options,
                PipelineBuildOptions::default(),
            )
            .unwrap();
        let (app, worker, probe) = start_plan(plan, 64, None, false);
        for ignore_eos in [false, true] {
            let stopped = stop_at_eos && !ignore_eos;
            let expected_text = if stopped { "" } else { "a a a" };
            let expected_tokens = if stopped { 0 } else { 3 };
            let reason = if stopped { "stop" } else { "length" };
            for stream in [false, true] {
                let mut request = completion(stream, 3);
                request["ignore_eos"] = json!(ignore_eos);
                if stream {
                    request["stream_options"] = json!({"include_usage":true});
                }
                let response = app
                    .clone()
                    .oneshot(post("/v1/completions", request))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK);
                let body = text(response).await;
                if stream {
                    let values = sse_values(&body);
                    assert_eq!(body.matches("data: [DONE]").count(), 1);
                    assert_eq!(
                        values
                            .iter()
                            .filter(|v| v["choices"][0]["finish_reason"].as_str().is_some())
                            .count(),
                        1
                    );
                    assert!(
                        values
                            .iter()
                            .any(|v| v["choices"][0]["finish_reason"] == reason)
                    );
                    assert!(
                        values
                            .iter()
                            .any(|v| v["usage"]["completion_tokens"] == expected_tokens)
                    );
                    assert_eq!(
                        values
                            .iter()
                            .filter_map(|v| v["choices"][0]["text"].as_str())
                            .collect::<String>(),
                        expected_text
                    );
                } else {
                    let value: Value = serde_json::from_str(&body).unwrap();
                    assert_eq!(value["choices"][0]["text"], expected_text);
                    assert_eq!(value["choices"][0]["finish_reason"], reason);
                    assert_eq!(value["usage"]["completion_tokens"], expected_tokens);
                }
            }
        }
        stop_worker(worker, &probe).await;
    }
}

#[test]
fn pr13_real_pipeline_receipt_is_terminal_consumed_and_generation_exact() {
    let fixture = Fixture::new(7);
    let mut engine = fixture.pipeline_plan(2).build().unwrap();
    let mut previous: Option<ferrule_runtime::RequestCleanupReceipt> = None;
    for round in 0..10 {
        let request = GenerateRequest {
            id: RequestId(1),
            session_id: Some(SessionId(1)),
            prompt_tokens: vec![2],
            max_new_tokens: 1,
            stop: vec![],
            ignore_eos: false,
        };
        engine.try_submit(request).unwrap();
        let ferrule_runtime::InferenceRequestCleanup::Tracked(receipt) =
            engine.request_cleanup(RequestId(1))
        else {
            panic!("pipeline must issue exact proof");
        };
        assert!(!receipt.is_released());
        if let Some(old) = previous.take() {
            assert!(old.is_released());
        }
        if round % 2 == 0 {
            engine.cancel_request(RequestId(1)).unwrap();
        } else {
            for _ in 0..4 {
                engine.step(&mut |_| Ok(())).unwrap();
            }
        }
        assert!(
            !receipt.is_released(),
            "even physical cleanup needs terminal consumption"
        );
        assert!(engine.take_request_terminal(RequestId(1)).is_some());
        assert!(receipt.is_released());
        assert!(engine.take_request_terminal(RequestId(1)).is_none());
        assert!(matches!(
            engine.request_cleanup(RequestId(1)),
            ferrule_runtime::InferenceRequestCleanup::Unavailable
        ));
        previous = Some(receipt);
    }
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
}

#[tokio::test]
async fn pr13_pipeline_and_direct_resident_http_reuse_two_credits_for_ten_rounds() {
    let fixture = Fixture::new(0);
    for pipeline in [false, true] {
        let plan = if pipeline {
            fixture.pipeline_plan(1)
        } else {
            fixture.plan(1)
        };
        let (app, worker, probe) = start_plan(plan, 64, None, false);
        let handle = worker.handle();
        for round in 0..10 {
            let response = tokio::time::timeout(
                DEADLINE,
                app.clone()
                    .oneshot(post("/v1/completions", completion(round % 2 == 0, 1))),
            )
            .await
            .unwrap()
            .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "round {round}");
            let body = text(response).await;
            if round % 2 == 0 {
                assert_eq!(body.matches("data: [DONE]").count(), 1);
            }
            tokio::time::timeout(DEADLINE, async {
                while handle.admission_snapshot().held_requests != 0 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap();
            let snapshot = handle.admission_snapshot();
            assert_eq!(snapshot.available_requests, 2);
            assert_eq!(snapshot.held_prompt_bytes, 0);
        }
        stop_worker(worker, &probe).await;
    }
}

#[tokio::test]
async fn tiny_checkpoint_invalid_admission_is_400_and_does_not_poison_worker() {
    let fixture = Fixture::new(7);
    let (app, worker, probe) = start(&fixture, 64, None, false);
    let mut request = completion(false, 1);
    request["prompt"] = json!("hello ".repeat(CONTEXT + 1));
    let response = app
        .clone()
        .oneshot(post("/v1/completions", request))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = text(response).await;
    assert!(body.contains("\"code\":\"invalid_request\""));
    assert!(!body.contains("context capacity"));
    let response = app
        .oneshot(post("/v1/completions", completion(false, 1)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(text(response).await.contains("\"completion_tokens\":1"));
    stop_worker(worker, &probe).await;
}

#[tokio::test]
async fn tiny_checkpoint_disconnect_cancels_and_releases_before_reuse() {
    let fixture = Fixture::new(7);
    let (entered_tx, entered) = std::sync::mpsc::channel();
    let (release, release_rx) = std::sync::mpsc::channel();
    let (app, worker, probe) = start(
        &fixture,
        64,
        Some(Gate {
            entered: entered_tx,
            release: release_rx,
        }),
        false,
    );
    let handle = worker.handle();
    let response = app
        .clone()
        .oneshot(post("/v1/completions", completion(true, 32)))
        .await
        .unwrap();
    let mut body = response.into_body();
    assert!(
        tokio::time::timeout(DEADLINE, body.frame())
            .await
            .unwrap()
            .unwrap()
            .is_ok()
    );
    entered.recv_timeout(DEADLINE).unwrap();
    drop(body);
    release.send(()).unwrap();
    wait_cancel(&probe).await;
    wait_credits_returned(&handle).await;
    assert_eq!(probe.cancelled.load(Ordering::Acquire), 1);
    let response = app
        .oneshot(post("/v1/completions", completion(false, 1)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(text(response).await.contains("\"completion_tokens\":1"));
    wait_credits_returned(&handle).await;
    assert_eq!(probe.cancelled.load(Ordering::Acquire), 1);
    stop_worker(worker, &probe).await;
}

#[tokio::test]
async fn tiny_checkpoint_capacity_is_429_without_accepting_a_second_session() {
    let fixture = Fixture::new(7);
    let (entered_tx, entered) = std::sync::mpsc::channel();
    let (release, release_rx) = std::sync::mpsc::channel();
    let (app, worker, probe) = start(
        &fixture,
        64,
        Some(Gate {
            entered: entered_tx,
            release: release_rx,
        }),
        false,
    );
    let first = app
        .clone()
        .oneshot(post("/v1/completions", completion(true, 32)))
        .await
        .unwrap();
    entered.recv_timeout(DEADLINE).unwrap();
    let second = app.oneshot(post("/v1/completions", completion(false, 1)));
    tokio::pin!(second);
    // Poll until the request has queued its worker command, while the first
    // request still owns the gated physical session. No timing-based race.
    std::future::poll_fn(|cx| {
        assert!(second.as_mut().poll(cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    release.send(()).unwrap();
    let response = tokio::time::timeout(DEADLINE, second)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = text(first).await;
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    assert_eq!(body.matches("\"finish_reason\":\"length\"").count(), 1);
    assert!(!body.contains("\"error\""));
    stop_worker(worker, &probe).await;
}

#[tokio::test]
async fn slow_stream_preserves_a_terminal_slot_instead_of_succeeding_with_truncated_text() {
    let fixture = Fixture::new(7);
    let (app, worker, probe) = start(&fixture, 2, None, false);
    let handle = worker.handle();
    let response = app
        .oneshot(post("/v1/completions", completion(true, 32)))
        .await
        .unwrap();
    wait_cancel(&probe).await;
    assert_eq!(handle.admission_snapshot().held_requests, 1);
    assert_eq!(handle.admission_snapshot().held_prompt_bytes, 0);
    let body = text(response).await;
    assert!(body.contains("request_cancelled"));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    assert!(!body.contains("\"finish_reason\":\"length\""));
    wait_credits_returned(&handle).await;
    assert_eq!(probe.cancelled.load(Ordering::Acquire), 1);
    stop_worker(worker, &probe).await;
}

#[tokio::test]
async fn execution_error_after_real_prefill_emits_sse_error_and_shutdown_drains_kv() {
    let fixture = Fixture::new(7);
    let (app, worker, probe) = start(&fixture, 64, None, true);
    let response = app
        .oneshot(post("/v1/completions", completion(true, 8)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = text(response).await;
    assert!(body.contains("\"code\":\"execution_failed\""));
    assert!(!body.contains("injected serving failure"));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    assert_eq!(probe.prefills.load(Ordering::Acquire), 1);
    let error = tokio::time::timeout(DEADLINE, worker.shutdown())
        .await
        .unwrap()
        .unwrap_err();
    assert!(error.to_string().contains("injected serving failure"));
    assert_eq!(probe.shutdowns.load(Ordering::Acquire), 1);
}

#[tokio::test]
async fn shutdown_cancels_a_live_pipeline_stream_and_retires_all_pages() {
    let fixture = Fixture::new(7);
    let (entered_tx, entered) = std::sync::mpsc::channel();
    let (release, release_rx) = std::sync::mpsc::channel();
    let (app, worker, probe) = start(
        &fixture,
        64,
        Some(Gate {
            entered: entered_tx,
            release: release_rx,
        }),
        false,
    );
    let response = app
        .oneshot(post("/v1/completions", completion(true, 32)))
        .await
        .unwrap();
    entered.recv_timeout(DEADLINE).unwrap();
    let shutdown = worker.shutdown();
    tokio::pin!(shutdown);
    std::future::poll_fn(|cx| {
        assert!(shutdown.as_mut().poll(cx).is_pending());
        std::task::Poll::Ready(())
    })
    .await;
    release.send(()).unwrap();
    tokio::time::timeout(DEADLINE, shutdown)
        .await
        .unwrap()
        .unwrap();
    let body = text(response).await;
    assert!(body.contains("request_cancelled"));
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    assert_eq!(probe.shutdowns.load(Ordering::Acquire), 1);
    assert_eq!(probe.cancelled.load(Ordering::Acquire), 1);
}

#[test]
fn rejected_and_duplicate_submissions_cannot_replace_an_existing_terminal() {
    use ferrule_runtime::{RequestTerminal, SequenceFinishReason};

    let fixture = Fixture::new(7);
    for cancel in [false, true] {
        for drain in [false, true] {
            let mut engine = fixture.pipeline_plan(1).build().unwrap();
            let request = GenerateRequest {
                id: RequestId(1),
                session_id: Some(SessionId(99)),
                prompt_tokens: vec![2],
                max_new_tokens: 2,
                stop: Vec::new(),
                ignore_eos: false,
            };
            engine.try_submit(request.clone()).unwrap();
            // Test both queued and active requests, including invalid duplicates
            // with a different session and a capacity rejection of another ID.
            for active in [false, true] {
                if active {
                    engine.step(&mut |_| Ok(())).unwrap();
                }
                for invalid in [false, true] {
                    let mut duplicate = request.clone();
                    if invalid {
                        duplicate.session_id = Some(SessionId(100));
                        duplicate.prompt_tokens.clear();
                    }
                    assert!(engine.try_submit(duplicate.clone()).is_err());
                    engine.submit(duplicate);
                    assert!(engine.take_request_terminal(request.id).is_none());
                    assert!(engine.drain_failed().is_empty());
                }
                let mut other = request.clone();
                other.id = RequestId(2);
                other.session_id = Some(SessionId(2));
                assert!(engine.try_submit(other.clone()).is_err());
                engine.submit(other.clone());
                assert!(engine.try_submit(other.clone()).is_err());
                engine.submit(other);
                let failed = engine.drain_failed();
                assert_eq!(failed.len(), 1);
                assert_eq!(failed[0].request_id, Some(RequestId(2)));
                assert!(engine.drain_failed().is_empty());
                assert!(engine.take_request_terminal(request.id).is_none());
            }
            let mut tokens = Vec::new();
            if cancel {
                engine.cancel_request(request.id).unwrap();
            } else {
                for _ in 0..4 {
                    engine
                        .step(&mut |event| {
                            tokens.push(event.token);
                            Ok(())
                        })
                        .unwrap();
                }
                assert_eq!(tokens, [1, 1]);
            }
            // A completed ID remains occupied until its original terminal is consumed.
            assert!(engine.try_submit(request.clone()).is_err());
            engine.submit(request.clone());
            assert!(engine.drain_failed().is_empty());
            let terminal = if drain {
                let mut sequences = if cancel {
                    engine.drain_cancelled()
                } else {
                    engine.drain_finished()
                };
                assert_eq!(sequences.len(), 1);
                sequences.remove(0)
            } else {
                match engine.take_request_terminal(request.id).unwrap() {
                    RequestTerminal::Cancelled(sequence) if cancel => sequence,
                    RequestTerminal::Finished(sequence) if !cancel => sequence,
                    terminal => panic!("wrong terminal: {terminal:?}"),
                }
            };
            assert_eq!(terminal.session_id, SessionId(99));
            assert_eq!(
                terminal.finish_reason,
                Some(if cancel {
                    SequenceFinishReason::Cancelled
                } else {
                    SequenceFinishReason::MaxTokens
                })
            );
            assert!(engine.take_request_terminal(request.id).is_none());
            assert!(engine.drain_finished().is_empty());
            assert!(engine.drain_cancelled().is_empty());
            assert!(engine.drain_failed().is_empty());
            assert_eq!(
                engine
                    .observability_snapshot()
                    .kv_cache
                    .unwrap()
                    .stats
                    .allocated_pages,
                0
            );
            engine.try_submit(request).unwrap();
            engine.shutdown().unwrap();
        }
    }
}

#[test]
fn pipeline_plan_rejects_expert_policy_overrides_before_build() {
    for moe in [false, true] {
        let fixture = Fixture::with_experts(7, moe);
        let config = AutoConfig::from_pretrained(&fixture.0).unwrap();
        let planner = ResidentModelPlanner::new();
        let mut cases = Vec::new();
        let mut hotset = Fixture::options(1);
        hotset.moe_hotset_experts = 1;
        cases.push((hotset, "moe_hotset_experts"));
        for field in 0..4 {
            let mut options = Fixture::options(1);
            match field {
                0 => options.expert_cache.host_entries = 0,
                1 => options.expert_cache.host_mebibytes = 1,
                2 => options.expert_cache.pinned_entries = 0,
                _ => options.expert_cache.pinned_mebibytes = 1,
            }
            cases.push((options, "expert_cache"));
        }
        for (options, name) in cases {
            let direct = planner.prepare_pipeline(
                &config,
                BackendSelection::Cpu,
                None,
                options.clone(),
                PipelineBuildOptions::default(),
            );
            let converted = planner
                .prepare(&config, BackendSelection::Cpu, None, options)
                .unwrap()
                .with_pipeline(PipelineBuildOptions::default());
            for result in [direct, converted] {
                assert!(
                    matches!(result, Err(ferrule_runtime::Error::InvalidRequest { ref message }) if message.contains(name))
                );
            }
        }
    }
}

#[test]
fn pipeline_plan_rejects_unintegrated_capabilities_and_runs_retained_turns() {
    let fixture = Fixture::new(7);
    for options in [
        PipelineBuildOptions {
            rank_backend: PipelineRankBackend::Process,
            ..Default::default()
        },
        PipelineBuildOptions {
            devices: Some(vec![0]),
            ..Default::default()
        },
        PipelineBuildOptions {
            rank_restarts: 1,
            ..Default::default()
        },
        PipelineBuildOptions {
            parallelism: ParallelismPlan {
                expert_parallel: 2,
                ..Default::default()
            },
            ..Default::default()
        },
        PipelineBuildOptions {
            parallelism: ParallelismPlan {
                pipeline_parallel: 3,
                ..Default::default()
            },
            ..Default::default()
        },
    ] {
        assert!(fixture.plan(1).with_pipeline(options).is_err());
    }
    let mut engine = fixture.pipeline_plan(1).build().unwrap();
    let session = SessionId(99);
    engine.retain_session(session).unwrap();
    for id in 1..=2 {
        engine
            .try_submit(GenerateRequest {
                id: RequestId(id),
                session_id: Some(session),
                prompt_tokens: vec![2],
                max_new_tokens: 2,
                stop: Vec::new(),
                ignore_eos: false,
            })
            .unwrap();
        let ferrule_runtime::InferenceRequestCleanup::Tracked(receipt) =
            engine.request_cleanup(RequestId(id))
        else {
            panic!("missing retained turn proof")
        };
        assert!(!receipt.is_released());
        let mut tokens = Vec::new();
        for _ in 0..10 {
            engine
                .step(&mut |event| {
                    tokens.push(event.token);
                    Ok(())
                })
                .unwrap();
            if engine.take_request_terminal(RequestId(id)).is_some() {
                break;
            }
        }
        assert!(receipt.is_released());
        assert_eq!(tokens, vec![1, 1]);
        assert_eq!(
            engine.retained_session_position(session),
            Some(id as usize * 3)
        );
    }
    engine.reset_session(session).unwrap();
    assert_eq!(engine.retained_session_position(session), Some(0));
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
    assert_eq!(
        engine
            .observability_snapshot()
            .kv_cache
            .unwrap()
            .stats
            .allocated_pages,
        0
    );
}

async fn check_parallel_http(plan: ResidentModelBuildPlan) -> (Value, String, String) {
    let (entered_tx, entered) = std::sync::mpsc::channel();
    let (release, release_rx) = std::sync::mpsc::channel();
    let (app, worker, probe) = start_plan(
        plan,
        64,
        Some(Gate {
            entered: entered_tx,
            release: release_rx,
        }),
        false,
    );
    for (name, value) in [
        ("temperature", json!(0.7)),
        ("top_p", json!(0.9)),
        ("top_k", json!(2)),
        ("n", json!(2)),
    ] {
        let mut request = completion(false, 1);
        request[name] = value;
        let response = app
            .clone()
            .oneshot(post("/v1/completions", request))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "accepted {name}"
        );
    }
    assert_eq!(probe.prefills.load(Ordering::Acquire), 0);
    let response = app
        .clone()
        .oneshot(post("/v1/completions", completion(true, 32)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    assert!(
        tokio::time::timeout(DEADLINE, body.frame())
            .await
            .unwrap()
            .unwrap()
            .is_ok()
    );
    entered.recv_timeout(DEADLINE).unwrap();
    drop(body);
    release.send(()).unwrap();
    wait_cancel(&probe).await;
    // The single resident slot is reusable only after all PP/EP work drains.
    let response = app
        .clone()
        .oneshot(post("/v1/completions", completion(false, 3)))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let value: Value = serde_json::from_str(&text(response).await).unwrap();
    assert_eq!(value["choices"][0]["text"], "a a a");
    assert_eq!(value["usage"]["completion_tokens"], 3);
    let response = app
        .clone()
        .oneshot(post(
            "/v1/chat/completions",
            json!({
                "model":"tiny-pipeline", "messages":[{"role":"user","content":"hello"}],
                "max_completion_tokens":3, "stream":true
            }),
        ))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = text(response).await;
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    let values = sse_values(&body);
    let output = values
        .iter()
        .filter_map(|v| v["choices"][0]["delta"]["content"].as_str())
        .collect::<String>();
    assert_eq!(output, "a a a");
    assert!(
        values
            .iter()
            .any(|v| v["choices"][0]["finish_reason"] == "length")
    );
    let mut request = completion(true, 3);
    request["stream_options"] = json!({"include_usage":true});
    let response = app.oneshot(post("/v1/completions", request)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = text(response).await;
    assert_eq!(body.matches("data: [DONE]").count(), 1);
    let events = sse_values(&body);
    let completion_text = events
        .iter()
        .filter_map(|v| v["choices"][0]["text"].as_str())
        .collect::<String>();
    assert_eq!(completion_text, "a a a");
    assert!(events.iter().any(|v| v["usage"]["completion_tokens"] == 3));
    assert!(
        events
            .iter()
            .any(|v| v["choices"][0]["finish_reason"] == "length")
    );
    stop_worker(worker, &probe).await;
    (
        json!({"choices": value["choices"], "usage": value["usage"]}),
        output,
        completion_text,
    )
}

#[tokio::test]
async fn cpu_moe_ep_serves_with_and_without_pp() {
    let fixture = Fixture::with_experts(7, true);
    for pp in [1, 2] {
        let plan = fixture.parallel_plan(BackendSelection::Cpu, pp, 2, None);
        assert_eq!(plan.backend_profile(), "cpu-pipeline-ep-f32-serial");
        check_parallel_http(plan).await;
    }
}

#[cfg(unix)]
#[tokio::test]
async fn cpu_process_pp_and_ep_serve_http_sse_and_reuse_after_cancel() {
    let dense = Fixture::new(7);
    check_parallel_http(dense.process_plan(BackendSelection::Cpu, 2, 1, None)).await;
    dense.assert_process_owners_stopped(2, 1, false);
    let moe = Fixture::with_experts(7, true);
    check_parallel_http(moe.process_plan(BackendSelection::Cpu, 2, 2, None)).await;
    moe.assert_process_owners_stopped(2, 2, false);
}

#[cfg(unix)]
#[test]
fn process_launch_failure_and_oversized_ipc_never_fall_back_to_threads() {
    let fixture = Fixture::new(7);
    let config = AutoConfig::from_pretrained(&fixture.0).unwrap();
    let process = PipelineBuildOptions {
        rank_backend: PipelineRankBackend::Process,
        process_launch: Some(ProcessLaunch::new(fixture.0.join("missing-child"))),
        ..Default::default()
    };
    let error = ResidentModelPlanner::new()
        .prepare_pipeline(
            &config,
            BackendSelection::Cpu,
            None,
            Fixture::options(1),
            process.clone(),
        )
        .unwrap()
        .build()
        .err()
        .unwrap();
    assert!(error.to_string().contains("spawn"), "{error}");
    let error = ResidentModelPlanner::new()
        .prepare_pipeline(
            &config,
            BackendSelection::Cpu,
            None,
            Fixture::options(200_000),
            process,
        )
        .unwrap()
        .build()
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("process IPC capacity"),
        "{error}"
    );
}

#[test]
fn pipeline_execution_dtype_controls_kv_budget_and_context_is_bounded() {
    let fixture = Fixture::with_experts(7, true);
    let config = AutoConfig::from_pretrained(&fixture.0).unwrap();
    let planner = ResidentModelPlanner::new();
    let pipeline = |ep| PipelineBuildOptions {
        parallelism: ParallelismPlan {
            pipeline_parallel: 2,
            expert_parallel: ep,
            ..Default::default()
        },
        ..Default::default()
    };
    for (ep, bytes) in [(1, 256), (2, 512)] {
        let mut options = Fixture::options(1);
        options.kv_cache_mebibytes = Some(1);
        let mut engine = planner
            .prepare_pipeline(
                &config,
                BackendSelection::Cpu,
                Some("plain"),
                options,
                pipeline(ep),
            )
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(
            engine.observability_snapshot().kv_cache.unwrap().page_bytes,
            Some(bytes)
        );
        engine.shutdown().unwrap();
    }
    // BF16's full contexts fit in 1 MiB; F32's do not. EP adds no KV replica.
    let mut options = Fixture::options(1024);
    options.kv_cache_mebibytes = Some(1);
    let mut engine = planner
        .prepare_pipeline(
            &config,
            BackendSelection::Cpu,
            Some("plain"),
            options.clone(),
            pipeline(1),
        )
        .unwrap()
        .build()
        .unwrap();
    engine.shutdown().unwrap();
    let error = planner
        .prepare_pipeline(
            &config,
            BackendSelection::Cpu,
            Some("plain"),
            options,
            pipeline(2),
        )
        .unwrap()
        .build()
        .err()
        .unwrap();
    assert!(error.to_string().contains("KV budget"));
    let mut options = Fixture::options(1);
    options.driver_config.ctx_size = CONTEXT + 1;
    let error = planner
        .prepare_pipeline(
            &config,
            BackendSelection::Cpu,
            Some("plain"),
            options,
            pipeline(2),
        )
        .unwrap()
        .build()
        .err()
        .unwrap();
    assert!(error.to_string().contains("position range"));
}

#[test]
fn pipeline_plan_validates_moe_degrees_and_capacities_before_build() {
    let fixture = Fixture::with_experts(7, true);
    let config = AutoConfig::from_pretrained(&fixture.0).unwrap();
    let planner = ResidentModelPlanner::new();
    for (pp, ep) in [(0, 1), (1, 0), (3, 1), (1, 3)] {
        assert!(
            planner
                .prepare_pipeline(
                    &config,
                    BackendSelection::Cpu,
                    None,
                    Fixture::options(1),
                    PipelineBuildOptions {
                        parallelism: ParallelismPlan {
                            pipeline_parallel: pp,
                            expert_parallel: ep,
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                )
                .is_err()
        );
    }
    let mut cases = Vec::new();
    let mut partial = Fixture::options(1);
    partial.max_layers = Some(1);
    cases.push(partial);
    let mut prefix = Fixture::options(1);
    prefix.scheduler_config.prefix_cache_capacity_pages = 1;
    cases.push(prefix);
    let mut zero_context = Fixture::options(1);
    zero_context.driver_config.ctx_size = 0;
    cases.push(zero_context);
    cases.push(Fixture::options(0));
    for options in cases {
        assert!(
            planner
                .prepare_pipeline(
                    &config,
                    BackendSelection::Cpu,
                    None,
                    options,
                    PipelineBuildOptions::default()
                )
                .is_err()
        );
    }
}

#[cfg(feature = "cuda")]
async fn check_tensor_http(pp: usize, tp: usize) {
    let fixture = Fixture::with_variant(7, false, true);
    let baseline = check_parallel_http(fixture.tensor_plan(1, 1)).await;
    // Accounting is global: 2 layers * 2 K/V * 4 heads * 2 dim * 16 tokens * 4 bytes.
    let mut engine = fixture.tensor_plan(pp, tp).build().unwrap();
    let kv = engine.observability_snapshot().kv_cache.unwrap();
    assert_eq!(kv.page_bytes, Some(2048));
    assert_eq!(kv.full_capacity_pages, 4);
    assert_eq!(kv.configured_pages, 4);
    assert_eq!(kv.configured_bytes, Some(8192));
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
    let result = check_parallel_http(fixture.tensor_plan(pp, tp)).await;
    assert_eq!(result, baseline, "PP{pp} TP{tp} differs from GPU PP1 TP1");
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires two visible CUDA GPUs; real dense checkpoint factory/HTTP/SSE"]
async fn cuda_dense_tensor_tp2_matches_tp1_http_sse() {
    check_tensor_http(1, 2).await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires four visible CUDA GPUs; real dense TP4 factory/HTTP/SSE"]
async fn cuda_dense_tensor_tp4_matches_tp1_http_sse() {
    check_tensor_http(1, 4).await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires four visible CUDA GPUs; real dense PP2TP2 factory/HTTP/SSE"]
async fn cuda_dense_tensor_pp2tp2_matches_tp1_http_sse() {
    check_tensor_http(2, 2).await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires six visible CUDA GPUs; runs real checkpoint PP/EP serving"]
async fn cuda_dense_and_moe_thread_pp_ep_serve_real_http() {
    for moe in [false, true] {
        let fixture = Fixture::with_experts(7, moe);
        let config = AutoConfig::from_pretrained(&fixture.0).unwrap();
        // The resident adapter must not accidentally acquire a CUDA claim.
        assert!(
            ResidentModelPlanner::new()
                .prepare(&config, BackendSelection::Cuda, None, Fixture::options(1))
                .is_err()
        );
        for pp in [1, 2] {
            let ep = if moe { 2 } else { 1 };
            let count = if moe { pp * 3 } else { pp };
            let devices = Some((0..count).rev().collect());
            let plan = fixture.parallel_plan(BackendSelection::Cuda, pp, ep, devices);
            assert_eq!(plan.backend(), ferrule_model::ModelExecutionBackend::Cuda);
            assert_eq!(plan.backend_profile(), "cuda-pipeline-f32-serial");
            check_parallel_http(plan).await;
        }
        if moe {
            check_parallel_http(fixture.parallel_plan(BackendSelection::Cuda, 2, 1, None)).await;
        }
        // Explicit colocation is supported, but every owner is still separate.
        let ep = if moe { 2 } else { 1 };
        check_parallel_http(fixture.parallel_plan(
            BackendSelection::Cuda,
            2,
            ep,
            Some(vec![0; if moe { 6 } else { 2 }]),
        ))
        .await;
    }
}

#[cfg(all(unix, feature = "cuda"))]
#[tokio::test]
#[ignore = "requires CUDA process children and six visible CUDA GPUs"]
async fn cuda_process_pp_ep_serve_real_http() {
    let dense = Fixture::new(7);
    check_parallel_http(dense.process_plan(BackendSelection::Cuda, 2, 1, Some(vec![0, 1]))).await;
    let moe = Fixture::with_experts(7, true);
    check_parallel_http(moe.process_plan(
        BackendSelection::Cuda,
        2,
        2,
        Some(vec![0, 1, 2, 3, 4, 5]),
    ))
    .await;
    moe.assert_process_owners_stopped(2, 2, true);
}
