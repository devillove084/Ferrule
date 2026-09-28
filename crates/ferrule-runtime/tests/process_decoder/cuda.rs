//! Opt-in real GPU process acceptance, using the CLI's production endpoint.
//! The matching CUDA CLI is built once; run this test target with
//! --features cuda cuda:: -- --ignored --test-threads=1 --nocapture.
//! Missing CLI, CUDA, GPUs or NAS files fail; there are no runtime skips.

use super::*;

#[path = "faults.rs"]
mod faults;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::c_void;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

const SOURCE: SessionId = SessionId(19);
const BRANCH: SessionId = SessionId(20);
const REPLAY: SessionId = SessionId(21);

/// Keep the driver loaded throughout the test so unloading cannot mask a cuInit.
/// cuCtxGetCurrent returns NOT_INITIALIZED (3), not success-with-null, until
/// cuInit has been called. This probe never calls cuInit/device_count itself.
struct NoParentCuda {
    library: *mut c_void,
    current: unsafe extern "C" fn(*mut *mut c_void) -> i32,
}
impl NoParentCuda {
    #[expect(
        unsafe_code,
        reason = "test-only driver query; no initialization or context creation"
    )]
    fn new() -> Self {
        unsafe {
            let library = libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            assert!(
                !library.is_null(),
                "libcuda.so.1 is required, not an optional test"
            );
            let symbol = libc::dlsym(library, c"cuCtxGetCurrent".as_ptr());
            assert!(!symbol.is_null(), "driver has no cuCtxGetCurrent");
            let probe = Self {
                library,
                current: std::mem::transmute::<
                    *mut c_void,
                    unsafe extern "C" fn(*mut *mut c_void) -> i32,
                >(symbol),
            };
            probe.check();
            probe
        }
    }
    #[expect(
        unsafe_code,
        reason = "borrow the loaded CUDA driver for a non-initializing query"
    )]
    fn check(&self) {
        let mut context = std::ptr::null_mut();
        let result = unsafe { (self.current)(&mut context) };
        assert_eq!(
            result, 3,
            "parent called cuInit (context={context:p}, result={result})"
        );
        assert!(context.is_null());
    }
}
impl Drop for NoParentCuda {
    #[expect(unsafe_code, reason = "release the test's dlopen reference")]
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.library);
        }
    }
}

fn smi(query: &str) -> String {
    let output = Command::new("nvidia-smi")
        .args([query, "--format=csv,noheader,nounits"])
        .output()
        .expect("nvidia-smi is required for independent PID/device evidence");
    assert!(
        output.status.success(),
        "nvidia-smi: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}
fn gpu_uuids(minimum: usize) -> BTreeMap<usize, String> {
    let devices: BTreeMap<_, _> = smi("--query-gpu=index,uuid")
        .lines()
        .map(|line| {
            let (index, uuid) = line.split_once(',').expect("GPU index,uuid");
            (
                index.trim().parse::<usize>().unwrap(),
                uuid.trim().to_owned(),
            )
        })
        .collect();
    assert!(
        devices.len() >= minimum,
        "requires {minimum} physical GPUs, found {}",
        devices.len()
    );
    devices
}
fn gpu_processes() -> BTreeMap<u32, String> {
    smi("--query-compute-apps=pid,gpu_uuid")
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (pid, uuid) = line.split_once(',').expect("GPU pid,uuid");
            (pid.trim().parse().unwrap(), uuid.trim().to_owned())
        })
        .collect()
}
fn gpu_options() -> ProcessOwnerConfig {
    ProcessOwnerConfig {
        // Three full Qwen3 rows alone exceed the default 4 MiB output limit.
        frame_limits: ProcessFrameLimits {
            max_frame_bytes: 32 * 1024 * 1024,
            max_config_bytes: 1024 * 1024,
            max_command_bytes: 1024 * 1024,
            max_output_bytes: 24 * 1024 * 1024,
            max_error_bytes: 4096,
        },
        startup_timeout: Duration::from_secs(120),
        command_timeout: Duration::from_secs(60),
        terminate_grace: Duration::from_millis(50),
        kill_grace: Duration::from_secs(3),
    }
}
fn gpu_launch() -> ProcessLaunch {
    gpu_launch_with_timeout(300000)
}
fn gpu_launch_with_timeout(timeout_ms: u64) -> ProcessLaunch {
    static CHILD: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    let executable = CHILD.get_or_init(|| {
        std::env::var_os("FERRULE_DECODER_CHILD")
            .map(PathBuf::from)
            .unwrap_or_else(|| build_process_child::build("ferrule-cli", "bin", "ferrule"))
    });
    assert!(
        executable.is_file(),
        "missing GPU CLI {}",
        executable.display()
    );
    ProcessLaunch::new(executable)
        .arg("__rank-worker")
        .arg("--max-frame-bytes")
        .arg(gpu_options().frame_limits.max_frame_bytes.to_string())
        .arg("--io-timeout-ms")
        .arg(timeout_ms.to_string())
}

fn boots(
    path: &Path,
    recipe: DecoderRecipeKind,
    layers: usize,
    degree: u32,
    cfg: PipelineConfig,
    cuda: bool,
    ep: bool,
) -> Vec<(ProcessIdentity, DecoderBoot)> {
    assert!(matches!(degree, 1 | 2));
    assert!(!ep || cuda);
    (0..degree)
        .map(|rank| {
            let start = rank as usize * layers / degree as usize;
            let end = (rank as usize + 1) * layers / degree as usize;
            let experts = ep.then(|| {
                let first = 10 + rank * 2;
                ExpertPlacementFrame {
                    source_scope: ferrule_common::topology::ExpertSourceScope::ExternalStage,
                    source: rank,
                    members: vec![first, first + 1],
                    entries: (start..end)
                        .flat_map(|layer| [(layer, 0, first), (layer, 1, first + 1)])
                        .collect(),
                    max_tokens: cfg.max_batch_tokens * 2,
                    max_bytes: 4096,
                    devices: (0..2)
                        .map(|slot| DecoderDevice::Cuda {
                            ordinal: 2 + rank as usize * 2 + slot,
                        })
                        .collect(),
                    timeout_ms: 30000,
                }
            });
            (
                identity(rank),
                DecoderBoot {
                    version: DECODER_WIRE_VERSION,
                    rank,
                    checkpoint: path.to_owned(),
                    recipe,
                    segment: SegmentFrame::encode(
                        &LayerSegmentPlan::new(layers, start..end, rank == 0, rank + 1 == degree)
                            .unwrap(),
                    ),
                    precision: DecoderPrecision::F32,
                    device: if cuda {
                        DecoderDevice::Cuda {
                            ordinal: rank as usize,
                        }
                    } else {
                        DecoderDevice::Cpu
                    },
                    kv: KvConfigFrame::encode_for_device(
                        cfg,
                        if cuda {
                            DecoderDevice::Cuda {
                                ordinal: rank as usize,
                            }
                        } else {
                            DecoderDevice::Cpu
                        },
                    )
                    .unwrap(),
                    experts,
                },
            )
        })
        .collect()
}

/// Synthetic dense Qwen3-schema checkpoint, all two layers and output present.
/// BF16 values are exact multiples of 1/64, so no test rounding dependency is needed.
fn dense_fixture() -> Fixture {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "ferrule-process-dense-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    let cfg = json!({
        "architectures":["Qwen3ForCausalLM"], "model_type":"qwen3",
        "hidden_size":8, "intermediate_size":12, "num_hidden_layers":2,
        "num_attention_heads":4, "num_key_value_heads":2, "head_dim":4,
        "rms_norm_eps":0.000001, "rope_theta":1000000.0, "rope_scaling":null,
        "max_position_embeddings":16, "vocab_size":8, "tie_word_embeddings":false,
        "attention_bias":false, "attention_dropout":0.0, "hidden_act":"silu",
        "torch_dtype":"bfloat16", "use_cache":true, "use_sliding_window":false,
        "sliding_window":null, "max_window_layers":2, "initializer_range":0.02,
        "bos_token_id":1, "eos_token_id":2
    });
    std::fs::write(path.join("config.json"), serde_json::to_vec(&cfg).unwrap()).unwrap();
    let mut shapes = vec![
        ("model.embed_tokens.weight".to_owned(), vec![8, 8]),
        ("model.norm.weight".to_owned(), vec![8]),
        ("lm_head.weight".to_owned(), vec![8, 8]),
    ];
    for layer in 0..2 {
        for (name, shape) in [
            ("input_layernorm.weight", vec![8]),
            ("post_attention_layernorm.weight", vec![8]),
            ("self_attn.q_proj.weight", vec![16, 8]),
            ("self_attn.k_proj.weight", vec![8, 8]),
            ("self_attn.v_proj.weight", vec![8, 8]),
            ("self_attn.o_proj.weight", vec![8, 16]),
            ("self_attn.q_norm.weight", vec![4]),
            ("self_attn.k_norm.weight", vec![4]),
            ("mlp.gate_proj.weight", vec![12, 8]),
            ("mlp.up_proj.weight", vec![12, 8]),
            ("mlp.down_proj.weight", vec![8, 12]),
        ] {
            shapes.push((format!("model.layers.{layer}.{name}"), shape));
        }
    }
    let mut header = serde_json::Map::new();
    let mut payload = Vec::new();
    for (tensor, (name, shape)) in shapes.into_iter().enumerate() {
        let start = payload.len();
        for index in 0..shape.iter().product::<usize>() {
            let value = if shape.len() == 1 {
                1.0 + (index % 2) as f32 / 8.0
            } else {
                ((index * 17 + tensor * 7 + index * index * 3) % 41) as f32 / 64.0 - 0.3125
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
    Fixture(path)
}

struct ProcessPipeline {
    executor: PipelineParallelExecutor,
    transport: SharedProcessPipelineTransport,
    initial: Vec<DecoderProcessStats>,
    nvml_pids: Vec<u32>,
}
impl ProcessPipeline {
    fn new(
        boots: Vec<(ProcessIdentity, DecoderBoot)>,
        cfg: PipelineConfig,
        probe: &NoParentCuda,
    ) -> Self {
        let start = Instant::now();
        let degree = boots.len();
        let before_gpu = gpu_processes();
        let plans = boots
            .iter()
            .map(|(_, boot)| boot.segment.decode().unwrap())
            .collect();
        let transport = ProcessPipelineTransport::spawn(gpu_launch(), boots.clone(), gpu_options())
            .expect("real GPU child Boot must succeed")
            .shared();
        let executor = PipelineParallelExecutor::new_with_external_expert_transport(
            topology(degree as u32),
            plans,
            cfg,
            transport.clone(),
        )
        .unwrap();
        let initial: Vec<_> = (0..degree)
            .map(|rank| transport.process_stats(rank as u32).unwrap())
            .collect();
        let mut pids = BTreeSet::new();
        for stats in &initial {
            assert!(pids.insert(stats.pid));
            assert_ne!(stats.pid, std::process::id());
            assert_eq!(
                (stats.sessions, stats.executions, stats.active_transactions),
                (0, 0, 0)
            );
            for expert in &stats.experts {
                assert!(pids.insert(expert.pid));
                assert_ne!(expert.pid, std::process::id());
                assert_eq!((expert.calls, expert.tokens), (0, 0));
            }
        }
        let mut nvml_pids = Vec::new();
        if matches!(boots[0].1.device, DecoderDevice::Cuda { .. }) {
            let devices = gpu_uuids(if boots[0].1.experts.is_some() {
                2 + degree * 2
            } else {
                degree
            });
            // NVML reports host PIDs, whereas child DTOs report namespace PIDs.
            // Require the exact new context/device multiset, not a best-effort
            // lookup which silently skips evidence inside containers.
            let active: BTreeMap<_, _> = gpu_processes()
                .into_iter()
                .filter(|(pid, _)| !before_gpu.contains_key(pid))
                .collect();
            let mut expected = Vec::new();
            for ((_, boot), stats) in boots.iter().zip(&initial) {
                let DecoderDevice::Cuda { ordinal } = boot.device else {
                    unreachable!()
                };
                expected.push(devices[&ordinal].clone());
                for expert in &stats.experts {
                    let DecoderDevice::Cuda { ordinal } = expert.device else {
                        panic!("CPU EP fallback")
                    };
                    expected.push(devices[&ordinal].clone());
                }
            }
            let mut actual = active.values().cloned().collect::<Vec<_>>();
            expected.sort();
            actual.sort();
            assert_eq!(
                actual, expected,
                "one real CUDA process/context per PP/EP owner on the requested GPU"
            );
            assert_eq!(active.len(), pids.len());
            eprintln!("NVML host_pids/devices={active:?}; namespace child_pids={pids:?}");
            nvml_pids = active.keys().copied().collect();
        }
        probe.check();
        eprintln!(
            "Boot PP{degree} pids={pids:?} in {:?}; parent cuCtxGetCurrent=NOT_INITIALIZED",
            start.elapsed()
        );
        Self {
            executor,
            transport,
            initial,
            nvml_pids,
        }
    }
    fn pids(&self) -> Vec<u32> {
        self.initial
            .iter()
            .flat_map(|s| std::iter::once(s.pid).chain(s.experts.iter().map(|e| e.pid)))
            .collect()
    }
    fn drained(&self) {
        assert!(!self.executor.is_quarantined());
        assert!(!self.transport.is_quarantined());
        assert_eq!(self.executor.outstanding(), 0);
        assert_eq!(self.executor.coordinator().in_use_credits(), 0);
        assert_eq!(self.executor.coordinator().retained_transaction_count(), 0);
        assert_eq!(self.executor.coordinator().retained_operation_count(), 0);
        assert_eq!(self.executor.page_manager().stats().retiring_pages, 0);
        for initial in &self.initial {
            let current = self.transport.process_stats(initial.rank).unwrap();
            assert_eq!(current.pid, initial.pid);
            assert_eq!(
                (current.active_transactions, current.expert_outstanding),
                (0, 0)
            );
            assert_eq!(
                current.experts.iter().map(|e| e.pid).collect::<Vec<_>>(),
                initial.experts.iter().map(|e| e.pid).collect::<Vec<_>>()
            );
        }
    }
    fn forward(
        &mut self,
        session: SessionId,
        tokens: &[u32],
        phase: ForwardPhase,
        probe: &NoParentCuda,
    ) -> DenseLogits {
        let start = Instant::now();
        let publications = self.executor.coordinator().publication_count();
        let output = self
            .executor
            .forward(session, tokens, phase)
            .expect("GPU child computation/commit failed");
        assert_eq!(
            self.executor.coordinator().publication_count(),
            publications + 1
        );
        assert_eq!(output.logits.rows(), tokens.len());
        assert!(output.logits.values().iter().all(|v| v.is_finite()));
        self.drained();
        probe.check();
        eprintln!(
            "PP{} {phase:?} tokens={tokens:?} {:?}",
            self.initial.len(),
            start.elapsed()
        );
        output.logits
    }
    fn finish(mut self, sessions: &[SessionId], probe: &NoParentCuda) {
        for &session in sessions {
            self.executor.release_session(session).unwrap();
        }
        self.drained();
        assert_eq!(self.executor.page_manager().allocated_pages(), 0);
        for initial in &self.initial {
            let stats = self.transport.process_stats(initial.rank).unwrap();
            assert_eq!(
                (
                    stats.sessions,
                    stats.resident_pages,
                    stats.active_transactions
                ),
                (0, 0, 0)
            );
            assert_eq!(stats.free_pages, stats.physical_pages);
        }
        self.executor.shutdown().unwrap();
        for pid in self.pids() {
            wait_gone(pid);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while gpu_processes()
            .keys()
            .any(|pid| self.nvml_pids.contains(pid))
        {
            assert!(
                Instant::now() < deadline,
                "GPU contexts remain after acknowledged shutdown/reap"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        probe.check();
    }
}
fn wait_gone(pid: u32) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while PathBuf::from(format!("/proc/{pid}")).exists() {
        assert!(
            Instant::now() < deadline,
            "child {pid} remains live or zombie after lifecycle teardown"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn close(label: &str, actual: &DenseLogits, expected: &DenseLogits, atol: f32, rtol: f32) {
    assert_eq!(
        (actual.rows(), actual.width()),
        (expected.rows(), expected.width())
    );
    let mut maximum = 0.0f32;
    for (index, (&a, &e)) in actual.values().iter().zip(expected.values()).enumerate() {
        let difference = (a - e).abs();
        assert!(
            a.is_finite() && e.is_finite() && difference <= atol + rtol * e.abs(),
            "{label}[{index}]: actual={a} expected={e} difference={difference}"
        );
        maximum = maximum.max(difference);
    }
    assert!(
        actual.values().iter().any(|v| v.abs() > 1e-4),
        "not zero/echo output"
    );
    eprintln!(
        "{label}: rows={} width={} max_abs={maximum:e}",
        actual.rows(),
        actual.width()
    );
}
fn trajectory(mut pipeline: ProcessPipeline, probe: &NoParentCuda) -> Vec<DenseLogits> {
    let mut output = vec![pipeline.forward(SOURCE, &[1, 2, 3], ForwardPhase::Prefill, probe)];
    pipeline.executor.fork_session(SOURCE, BRANCH).unwrap();
    let slot = pipeline.executor.session_slot(SOURCE).unwrap();
    let original = pipeline
        .executor
        .page_manager()
        .block_table(slot)
        .unwrap()
        .pages()
        .to_vec();
    assert_eq!(
        pipeline.executor.page_manager().page_refcount(original[1]),
        2
    );
    output.push(pipeline.forward(SOURCE, &[4], ForwardPhase::Decode, probe));
    assert_ne!(
        pipeline
            .executor
            .page_manager()
            .block_table(slot)
            .unwrap()
            .pages()[1],
        original[1],
        "source must COW its shared partial tail"
    );
    output.push(pipeline.forward(BRANCH, &[5], ForwardPhase::Decode, probe));
    output.push(pipeline.forward(SOURCE, &[6], ForwardPhase::Decode, probe));
    let replay = pipeline.forward(REPLAY, &[1, 2, 3, 4, 6], ForwardPhase::Prefill, probe);
    for (row, result) in [(3, &output[1]), (4, &output[3])] {
        let replay_row =
            DenseLogits::new(1, replay.width(), replay.row(row).unwrap().to_vec()).unwrap();
        close("causal replay/decode", &replay_row, result, 2e-5, 2e-4);
    }
    let before = pipeline.executor.coordinator().publication_count();
    let cancellation = AtomicBool::new(false);
    let error = pipeline
        .executor
        .forward_observed(
            SOURCE,
            &[7],
            ForwardPhase::Decode,
            &cancellation,
            |progress| {
                if progress.state == TransactionState::Preparing {
                    cancellation.store(true, Ordering::Release);
                }
            },
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("cancelled before decision"),
        "{error}"
    );
    assert_eq!(pipeline.executor.coordinator().publication_count(), before);
    assert_eq!(
        pipeline
            .executor
            .page_manager()
            .block_table(slot)
            .unwrap()
            .committed_tokens(),
        5
    );
    pipeline.drained();
    for initial in &pipeline.initial {
        let stats = pipeline.transport.process_stats(initial.rank).unwrap();
        assert_eq!(stats.executions, 6);
        for expert in stats.experts {
            assert_eq!(expert.owned_experts.len(), 1);
            assert_eq!(
                (expert.calls, expert.tokens),
                (6, 12),
                "top-k=2 must execute every route, including the aborted decode"
            );
            eprintln!(
                "EP owner={} pid={} device={:?} calls={} tokens={}",
                expert.owner, expert.pid, expert.device, expert.calls, expert.tokens
            );
        }
    }
    pipeline.finish(&[SOURCE, BRANCH, REPLAY], probe);
    output
}
fn synthetic(dense: bool, ep: bool) {
    let probe = NoParentCuda::new();
    gpu_uuids(if ep { 6 } else { 2 });
    let fixture = if dense {
        dense_fixture()
    } else {
        Fixture::new()
    };
    let recipe = if dense {
        DecoderRecipeKind::Qwen3Dense
    } else {
        DecoderRecipeKind::Synthetic
    };
    let cpu = trajectory(
        ProcessPipeline::new(
            boots(&fixture.0, recipe, 2, 1, config(), false, false),
            config(),
            &probe,
        ),
        &probe,
    );
    let pp1 = trajectory(
        ProcessPipeline::new(
            boots(&fixture.0, recipe, 2, 1, config(), true, false),
            config(),
            &probe,
        ),
        &probe,
    );
    let pp2 = trajectory(
        ProcessPipeline::new(
            boots(&fixture.0, recipe, 2, 2, config(), true, ep),
            config(),
            &probe,
        ),
        &probe,
    );
    for (index, ((actual, reference), cpu)) in pp2.iter().zip(&pp1).zip(&cpu).enumerate() {
        close(
            &format!("dense={dense} ep={ep} PP2/PP1 step={index}"),
            actual,
            reference,
            2e-5,
            2e-5,
        );
        close("GPU/independent CPU kernels", actual, cpu, 2e-5, 2e-4);
    }
    probe.check();
}

#[test]
#[ignore = "real GPU processes: build CUDA CLI; requires two GPUs, --test-threads=1"]
fn gpu_process_dense_pp1_pp2_full_checkpoint_prefill_decode_fork_cow() {
    synthetic(true, false);
}
#[test]
#[ignore = "real GPU processes: build CUDA CLI; requires two GPUs, --test-threads=1"]
fn gpu_process_moe_top2_pp1_pp2_full_checkpoint_prefill_decode_fork_cow() {
    synthetic(false, false);
}
#[test]
#[ignore = "real GPU processes: build CUDA CLI; requires six GPUs, --test-threads=1"]
fn gpu_process_pp2_ep2_independent_children_execute_every_route() {
    synthetic(false, true);
}

#[test]
#[ignore = "NAS Qwen3-0.6B, CUDA CLI, two GPUs and external 300s timeout; --test-threads=1"]
fn gpu_process_nas_qwen3_06b_all_28_layers_pp1_pp2_prefill_decode() {
    let probe = NoParentCuda::new();
    gpu_uuids(2);
    let path = std::env::var_os("FERRULE_QWEN3_06B_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/nas1/hf/Qwen3-0.6B"));
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path.join("config.json")).expect("NAS Qwen3 config required"),
    )
    .unwrap();
    assert_eq!(metadata["num_hidden_layers"], 28);
    assert_eq!(metadata["hidden_size"], 1024);
    assert_eq!(metadata["vocab_size"], 151936);
    let cfg = PipelineConfig {
        max_parameter_bytes: 1024 * 1024 * 1024,
        max_batch_tokens: 8,
        ..config()
    };
    eprintln!(
        "NAS={} full_layers=28 parameter_limit={} frames={:?}",
        path.display(),
        cfg.max_parameter_bytes,
        gpu_options().frame_limits
    );
    let mut reference = Vec::new();
    let mut token = 0;
    for degree in [1, 2] {
        let mut pipeline = ProcessPipeline::new(
            boots(
                &path,
                DecoderRecipeKind::Qwen3Dense,
                28,
                degree,
                cfg,
                true,
                false,
            ),
            cfg,
            &probe,
        );
        assert_eq!(
            pipeline
                .executor
                .stage_descriptions()
                .iter()
                .flat_map(|d| d.plan.layers())
                .collect::<Vec<_>>(),
            (0..28).collect::<Vec<_>>()
        );
        assert!(pipeline.executor.stage_descriptions().iter().all(|d| (
            d.hidden,
            d.vocabulary,
            d.kv_heads,
            d.head_dim
        ) == (1024, 151936, 8, 128)));
        let prefill = pipeline.forward(SOURCE, &[151643, 9707, 11], ForwardPhase::Prefill, &probe);
        if degree == 1 {
            token = argmax(prefill.row(2).unwrap());
        }
        let decode = pipeline.forward(SOURCE, &[token], ForwardPhase::Decode, &probe);
        if degree == 1 {
            reference.extend([prefill, decode]);
        } else {
            for (label, (actual, expected)) in ["NAS prefill", "NAS decode"]
                .into_iter()
                .zip([prefill, decode].iter().zip(&reference))
            {
                close(label, actual, expected, 2e-5, 2e-5);
                for row in 0..actual.rows() {
                    assert_eq!(
                        argmax(actual.row(row).unwrap()),
                        argmax(expected.row(row).unwrap())
                    );
                }
            }
        }
        assert_eq!(pipeline.executor.page_manager().stats().committed_tokens, 4);
        for initial in &pipeline.initial {
            assert_eq!(
                pipeline
                    .transport
                    .process_stats(initial.rank)
                    .unwrap()
                    .executions,
                2
            );
        }
        pipeline.finish(&[SOURCE], &probe);
    }
    eprintln!(
        "PASS NAS 28-layer BF16 weights/F32 execution GPU child PP1/PP2; decode_token={token}; parent not initialized"
    );
}
fn argmax(values: &[f32]) -> u32 {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0 as u32
}

#[test]
#[ignore = "CUDA CLI and four GPUs; idle PP/EP and rejected input must retain live owners"]
fn gpu_idle_and_oversize_keep_pp_and_bucketless_ep_ready_without_parent_cuda() {
    let probe = NoParentCuda::new();
    gpu_uuids(4);
    let fixture = Fixture::new();
    let mut boots = boots(
        &fixture.0,
        DecoderRecipeKind::Synthetic,
        2,
        1,
        config(),
        true,
        true,
    );
    boots[0].1.experts.as_mut().unwrap().timeout_ms = 2000;
    let mut options = gpu_options();
    options.command_timeout = Duration::from_secs(2);
    options.frame_limits.max_command_bytes = 1024;
    let boot = boots[0].1.clone();
    let transport = ProcessPipelineTransport::spawn(gpu_launch_with_timeout(2000), boots, options)
        .unwrap()
        .shared();
    let initial = transport.process_stats(0).unwrap();
    assert_eq!(initial.experts.len(), 2);
    assert!(
        initial
            .experts
            .iter()
            .all(|expert| (expert.calls, expert.tokens) == (0, 0))
    );
    probe.check();
    std::thread::sleep(Duration::from_secs(6));
    let after = transport.process_stats(0).unwrap();
    assert_eq!(initial.pid, after.pid);
    for (before, after) in initial.experts.iter().zip(&after.experts) {
        assert_eq!(before.pid, after.pid);
        assert_eq!((after.calls, after.tokens), (0, 0));
    }
    assert!(!transport.is_quarantined());
    probe.check();
    assert_local_oversize_keeps_prepared_custody(transport, &boot);
    wait_gone(initial.pid);
    for expert in initial.experts {
        wait_gone(expert.pid);
    }
    probe.check();
}
