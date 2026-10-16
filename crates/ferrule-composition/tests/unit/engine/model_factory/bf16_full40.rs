//! Independent BF16/profile full40 gate on the public factory's actual runner.
//! Private unit mount before type erasure; no replacement owner or forward loop.
use crate::model_factory::full40_strict::oracle::{
    BF16_ATOL, BF16_PROFILE, BF16_RTOL, Oracle, StrictReport, check_hash,
};
use crate::model_factory::*;
use ferrule_common::execution::*;
use ferrule_model::decoder::{CudaHybridSequenceState, GenericDecoderRunner, HybridCudaDecoder};
use ferrule_model::runner::MultiSessionRunner;
use ferrule_runtime::InferenceShutdownProgress;
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    io::{Seek, Write},
    path::PathBuf,
    rc::Rc,
};

use ferrule_model::transformer::{
    ExpertReadPipelineOptions, ExpertReadSchedule, ExpertTransferPolicy, NumericFp8Precision,
};

type Runner = GenericDecoderRunner<HybridCudaDecoder>;
use crate::model_factory::bf16_serving::expert_read_reporting::{
    self, PredictionReports, ReadQualityLog,
};
use ferrule_model::runner::ResidentModelRunner;
type ReadLog = Option<Rc<RefCell<ReadQualityLog>>>;

fn observe_operation<T>(
    runner: &mut Runner,
    reads: &ReadLog,
    label: &str,
    operation: &str,
    execute: impl FnOnce(&mut Runner) -> T,
) -> T {
    let Some(log) = reads else {
        return execute(runner);
    };
    let before = runner.physical_resource_snapshot();
    let index = log.borrow_mut().begin(label, operation, before.clone());
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(runner)));
    log.borrow_mut()
        .finish(index, &before, runner.physical_resource_snapshot());
    match result {
        Ok(value) => value,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

fn attach_read_reports(
    report: &mut Value,
    reads: &ReadLog,
    predictions: Option<&PredictionReports>,
) {
    let reads = reads.as_ref().map(|log| log.borrow());
    expert_read_reporting::attach(report, reads.as_deref(), predictions);
}

// The real source-header plan is 6,846,743,296 bytes INCLUDING the existing
// allocator margin. The former 6400 MiB cap rejected it before any GPU work.
// Round the TEST ceiling up to 7 GiB; do not shrink workspace/state/margin.
const FULL40_CAPACITY: Qwen35MoeCapacityLimits = Qwen35MoeCapacityLimits {
    max_experts: 8,
    max_bytes: 320 << 20,
    scratch_bytes: 0,
    workspace_bytes: 2 << 30,
    kv_bytes: 16 << 20,
    device_bytes: 7 << 30,
    allocator_margin_bytes: 512 << 20,
    max_positions: 128,
    max_sequences: 2,
    max_batch_tokens: 8,
};
const TILED_FULL40_CAPACITY: Qwen35MoeCapacityLimits = Qwen35MoeCapacityLimits {
    // Included in the 320 MiB cache ceiling; keep all geometry/state/margins.
    scratch_bytes: 8 << 20,
    ..FULL40_CAPACITY
};
// Closed profiles: changing a capacity requires a named maintained selector, not
// an environment override that silently changes the meaning of an old receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Full40TestConfig {
    Historical8,
    StorageMiddle64,
    StorageMiddle64Blocking,
    StorageMiddle64Lookahead,
    StorageMiddle64RetainedBlocking,
    StorageMiddle64RetainedLookahead,
}

impl Full40TestConfig {
    fn name(self) -> &'static str {
        match self {
            Self::Historical8 => "historical8",
            Self::StorageMiddle64 => "storage-middle64",
            Self::StorageMiddle64Blocking => "storage-middle64-blocking",
            Self::StorageMiddle64Lookahead => "storage-middle64-lookahead",
            Self::StorageMiddle64RetainedBlocking => "storage-middle64-retained-blocking",
            Self::StorageMiddle64RetainedLookahead => "storage-middle64-retained-lookahead",
        }
    }

    fn capacity(self, precision: NumericFp8Precision) -> Qwen35MoeCapacityLimits {
        let historical = match precision {
            NumericFp8Precision::Bf16ModelF32Accumulate => FULL40_CAPACITY,
            NumericFp8Precision::Bf16ModelTiledF32Accumulate
            | NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate
            | NumericFp8Precision::Bf16ModelMixedF32Accumulate => TILED_FULL40_CAPACITY,
            _ => panic!("full40 BF16 gate requires an explicit BF16-model profile"),
        };
        match self {
            Self::Historical8 => historical,
            Self::StorageMiddle64
            | Self::StorageMiddle64Blocking
            | Self::StorageMiddle64Lookahead
            | Self::StorageMiddle64RetainedBlocking
            | Self::StorageMiddle64RetainedLookahead => {
                assert_eq!(precision, NumericFp8Precision::Bf16ModelMixedF32Accumulate);
                Qwen35MoeCapacityLimits {
                    max_experts: 64,
                    ..TILED_FULL40_CAPACITY
                }
            }
        }
    }

    fn scheduler(self) -> ResidentSchedulerConfig {
        ResidentSchedulerConfig {
            max_active_sequences: 2,
            max_decode_batch: match self {
                Self::Historical8 => 1,
                Self::StorageMiddle64
                | Self::StorageMiddle64Blocking
                | Self::StorageMiddle64Lookahead
                | Self::StorageMiddle64RetainedBlocking
                | Self::StorageMiddle64RetainedLookahead => 2,
            },
            max_batch_tokens: 8,
            prefill_chunk_size: 8,
            prefix_cache_capacity_pages: 0,
            decode_cohort_target: 1,
            decode_cohort_max_deferrals: 0,
            prefill_reserve_tokens: 1,
            allow_mixed_batches: true,
        }
    }

    fn expert_read_pipeline(self) -> Option<ExpertReadPipelineOptions> {
        let schedule = match self {
            Self::Historical8 | Self::StorageMiddle64 => return None,
            Self::StorageMiddle64Blocking | Self::StorageMiddle64RetainedBlocking => {
                ExpertReadSchedule::Blocking
            }
            Self::StorageMiddle64Lookahead | Self::StorageMiddle64RetainedLookahead => {
                ExpertReadSchedule::Lookahead
            }
        };
        Some(ExpertReadPipelineOptions {
            schedule,
            max_host_bytes: 64 << 20,
            max_metadata_bytes: 8 << 20,
            worker_stack_bytes: 2 << 20,
        })
    }

    fn expert_transfer_policy(self) -> ExpertTransferPolicy {
        match self {
            Self::StorageMiddle64RetainedBlocking | Self::StorageMiddle64RetainedLookahead => {
                ExpertTransferPolicy::Retained {
                    policy_headroom_bytes: 16 << 20,
                    readiness_timeout: std::time::Duration::from_secs(30),
                }
            }
            _ => ExpertTransferPolicy::Synchronous,
        }
    }

    fn transfer_manifest(self) -> Option<Value> {
        match self.expert_transfer_policy() {
            ExpertTransferPolicy::Synchronous => None,
            ExpertTransferPolicy::Retained {
                policy_headroom_bytes,
                readiness_timeout,
            } => Some(json!({
                "mode":"retained", "policy_headroom_bytes":policy_headroom_bytes,
                "readiness_timeout_ms":readiness_timeout.as_millis(),
                "host_quota_unchanged":true, "physical_overhead_measured":false,
                "gpu_overlap_claim":false,
            })),
        }
    }

    fn manifest(self, precision: NumericFp8Precision) -> Value {
        let options = full40_options(self, precision);
        let mut manifest = json!({"name":self.name(),"capacity_and_options":full40_options_manifest(&options),
            "cache_bytes_include_scratch":true,"model_precision":"Bf16",
            "numeric_precision":format!("{precision:?}"),"storage_policy":"E4M3FnBf16Scales",
            "serving_model_profile":null,"topology":{"pp":1,"tp":1,"ep":1}});
        // Disabled receipts retain their historical manifest identity.
        if self.expert_read_pipeline().is_some() {
            manifest["expert_read_pipeline"] = expert_read_manifest(self.expert_read_pipeline());
        }
        if let Some(policy) = self.transfer_manifest() {
            manifest["expert_transfer_policy"] = policy;
        }
        manifest
    }

    fn assert_options(self, options: &ModelFactoryOptions, precision: NumericFp8Precision) {
        assert_eq!(options.qwen35_moe_capacity, Some(self.capacity(precision)));
        assert_eq!(options.scheduler_config, self.scheduler());
        assert_eq!(options.driver_config.ctx_size, 128);
        assert!(!options.driver_config.enable_native_proposals);
        assert!(!options.driver_config.stop_at_eos);
        assert_eq!(options.kv_cache_mebibytes, Some(16));
        assert_eq!(options.max_tensor_mebibytes, 1024);
        assert_eq!(options.output_head_chunk_rows, 4096);
        assert_eq!(options.expert_reader_max_tensor_mebibytes, 64);
        assert_eq!(options.expert_cache, ExpertCacheOptions::default());
        assert_eq!(options.moe_hotset_experts, 0);
        assert!(options.numeric_fp8_profile.is_none());
    }
}

fn expert_read_manifest(read: Option<ExpertReadPipelineOptions>) -> Value {
    read.map_or(Value::Null, |read| {
        json!({"schedule":format!("{:?}",read.schedule),"max_host_bytes":read.max_host_bytes,
            "max_metadata_bytes":read.max_metadata_bytes,"worker_stack_bytes":read.worker_stack_bytes,
            "metadata_and_stack_included_in_host_cap":true,"device_budget_separate":true})
    })
}

fn full40_options(
    test_config: Full40TestConfig,
    precision: NumericFp8Precision,
) -> ModelFactoryOptions {
    let mut options = crate::model_factory::tests::options();
    options.max_layers = None;
    options.max_tensor_mebibytes = 1024;
    options.output_head_chunk_rows = 4096;
    options.expert_reader_max_tensor_mebibytes = 64;
    options.expert_cache = Default::default();
    options.numeric_fp8_profile = None;
    options.qwen35_host_cache = None;
    options.qwen35_moe_capacity = Some(test_config.capacity(precision));
    options.moe_hotset_experts = 0;
    options.kv_cache_mebibytes = Some(16);
    options.driver_config.ctx_size = 128;
    options.driver_config.enable_native_proposals = false;
    options.driver_config.stop_at_eos = false;
    options.scheduler_config = test_config.scheduler();
    test_config.assert_options(&options, precision);
    options
}

fn full40_options_manifest(options: &ModelFactoryOptions) -> Value {
    let limits = options.qwen35_moe_capacity.unwrap();
    let scheduler = options.scheduler_config;
    json!({"max_experts":limits.max_experts,"max_bytes":limits.max_bytes,
        "scratch_bytes":limits.scratch_bytes,"workspace_bytes":limits.workspace_bytes,
        "kv_bytes":limits.kv_bytes,"device_bytes":limits.device_bytes,
        "allocator_margin_bytes":limits.allocator_margin_bytes,"max_positions":limits.max_positions,
        "max_sequences":limits.max_sequences,"max_batch_tokens":limits.max_batch_tokens,
        "scheduler":{"max_active_sequences":scheduler.max_active_sequences,
            "max_decode_batch":scheduler.max_decode_batch,"max_batch_tokens":scheduler.max_batch_tokens,
            "prefill_chunk_size":scheduler.prefill_chunk_size,
            "prefix_cache_capacity_pages":scheduler.prefix_cache_capacity_pages,
            "decode_cohort_target":scheduler.decode_cohort_target,
            "decode_cohort_max_deferrals":scheduler.decode_cohort_max_deferrals,
            "prefill_reserve_tokens":scheduler.prefill_reserve_tokens,"allow_mixed_batches":scheduler.allow_mixed_batches},
        "ctx_size":options.driver_config.ctx_size,"enable_native_proposals":options.driver_config.enable_native_proposals,
        "stop_at_eos":options.driver_config.stop_at_eos,"max_layers":options.max_layers,
        "kv_cache_mebibytes":options.kv_cache_mebibytes,"max_tensor_mebibytes":options.max_tensor_mebibytes,
        "output_head_chunk_rows":options.output_head_chunk_rows,"expert_reader_max_tensor_mebibytes":options.expert_reader_max_tensor_mebibytes,
        "expert_cache":format!("{:?}",options.expert_cache),"moe_hotset_experts":options.moe_hotset_experts,
        "qwen35_host_cache":format!("{:?}",options.qwen35_host_cache),
        "numeric_fp8_profile":format!("{:?}",options.numeric_fp8_profile)})
}

fn full40_effective_config(
    plan: &ResidentModelBuildPlan,
    test_config: Full40TestConfig,
    precision: NumericFp8Precision,
) -> Value {
    let requested = plan.requested_options();
    let effective = plan.effective_options();
    test_config.assert_options(requested, precision);
    test_config.assert_options(&effective, precision);
    assert_eq!(requested.max_layers, None);
    assert_eq!(effective.max_layers, Some(40));
    assert_eq!(plan.backend(), ModelExecutionBackend::Cuda);
    assert_eq!(plan.backend_profile(), precision.hybrid_backend_name());
    assert_eq!(
        plan.qwen35_moe_capacity_limits().unwrap(),
        Some(test_config.capacity(precision))
    );
    assert_eq!(
        qwen35::capacity_limits(&plan.request).unwrap(),
        test_config.capacity(precision)
    );
    assert_eq!(
        plan.request.execution.numeric_bf16_precision,
        Some(precision)
    );
    assert_eq!(
        plan.request.execution.kv_execution_policy,
        ferrule_runtime::engine::KvExecutionPolicy::Resident
    );
    assert!(plan.requested_parallel_options().is_none());
    assert!(requested.qwen35_host_cache.is_none());
    let host = plan.qwen35_host_cache_options().unwrap();
    assert_eq!(
        host.mode,
        ferrule_model::transformer::host_experts::ExpertPrewarmMode::Lazy
    );
    let requested_read = test_config.expert_read_pipeline();
    let effective_read = plan.expert_read_pipeline_options();
    assert_eq!(effective_read, requested_read);
    let memory = plan.numeric_bf16_memory_plan().unwrap().unwrap();
    assert_eq!(memory.expert_read_pipeline_options(), requested_read);
    assert_eq!(
        memory.runner_options().expert_read_pipeline_options(),
        requested_read
    );
    assert!(memory.resources().host_experts().is_none());
    let admitted_device_bytes = memory.total_device_bytes();
    let topology = plan.topology_placement();
    assert_eq!(
        [
            topology.logical.pipeline_parallel,
            topology.logical.tensor_parallel,
            topology.logical.expert_parallel,
            topology.logical.data_parallel,
            topology.logical.sequence_parallel,
            topology.logical.context_parallel
        ],
        [1; 6]
    );
    let mut result = json!({"assertions_passed":true,"requested":full40_options_manifest(requested),
        "effective":full40_options_manifest(&effective),"backend_profile":plan.backend_profile(),
        "requested_expert_read_pipeline":expert_read_manifest(requested_read),
        "effective_expert_read_pipeline":expert_read_manifest(effective_read),
        "admitted_device_bytes":admitted_device_bytes,
        "combined_host_and_device_cap_bytes":test_config.capacity(precision).device_bytes as u64 + requested_read.map_or(0, |read| read.max_host_bytes),
        "factory_entry":"ResidentModelPlanner::prepare_numeric_fp8_bf16_serving_with_precision",
        "source_lowering":{"model_precision":"Bf16","storage_policy":"E4M3FnBf16Scales",
            "numeric_precision":format!("{precision:?}"),"kv_execution_policy":"Resident",
            "expert_prewarm":format!("{:?}",host.mode),"topology":format!("{topology:?}"),
            "relation":"explicit numeric factory lowering; not a sealed CLI binary/source match"},
        "resolution_report":plan.resolution_report(),"scope":"metadata assertions, not physical admission or concurrent scheduler execution"});
    if let Some(policy) = test_config.transfer_manifest() {
        result["expert_transfer_policy"] = policy;
        result["expert_transfer_selection"] =
            json!("explicit existing-owner capability before first prediction");
    }
    result
}

#[test]
fn full40_frozen_profiles_preserve_historical8_and_storage_middle64() {
    for precision in [
        NumericFp8Precision::Bf16ModelF32Accumulate,
        NumericFp8Precision::Bf16ModelTiledF32Accumulate,
        NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
    ] {
        let config = Full40TestConfig::Historical8;
        let options = full40_options(config, precision);
        assert_eq!(options.qwen35_moe_capacity.unwrap().max_experts, 8);
        assert_eq!(options.scheduler_config.max_decode_batch, 1);
        assert_eq!(
            options.qwen35_moe_capacity.unwrap().scratch_bytes,
            if precision == NumericFp8Precision::Bf16ModelF32Accumulate {
                0
            } else {
                8 << 20
            }
        );
    }
    let config = Full40TestConfig::StorageMiddle64;
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    let manifest = config.manifest(precision);
    let values = &manifest["capacity_and_options"];
    for (name, expected) in [
        ("max_experts", 64),
        ("max_bytes", 335_544_320),
        ("scratch_bytes", 8_388_608),
        ("workspace_bytes", 2_147_483_648),
        ("kv_bytes", 16_777_216),
        ("device_bytes", 7_516_192_768),
        ("allocator_margin_bytes", 536_870_912),
        ("max_positions", 128),
        ("max_sequences", 2),
        ("max_batch_tokens", 8),
    ] {
        assert_eq!(values[name].as_u64(), Some(expected), "{name}");
    }
    assert_eq!(values["scheduler"]["max_decode_batch"], 2);
}

#[test]
fn full40_enabled_storage_middle64_schedules_freeze_same_host_and_device_budgets() {
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    let legacy = Full40TestConfig::StorageMiddle64;
    assert_eq!(Full40TestConfig::Historical8.expert_read_pipeline(), None);
    assert_eq!(legacy.expert_read_pipeline(), None);
    assert!(
        legacy
            .manifest(precision)
            .get("expert_read_pipeline")
            .is_none()
    );
    for (config, schedule) in [
        (
            Full40TestConfig::StorageMiddle64Blocking,
            ExpertReadSchedule::Blocking,
        ),
        (
            Full40TestConfig::StorageMiddle64Lookahead,
            ExpertReadSchedule::Lookahead,
        ),
    ] {
        let read = config.expert_read_pipeline().unwrap();
        read.validate().unwrap();
        assert_eq!(
            read,
            ExpertReadPipelineOptions {
                schedule,
                max_host_bytes: 64 << 20,
                max_metadata_bytes: 8 << 20,
                worker_stack_bytes: 2 << 20,
            }
        );
        assert_eq!(config.capacity(precision), legacy.capacity(precision));
        assert_eq!(config.scheduler(), legacy.scheduler());
        assert_eq!(
            config.manifest(precision)["capacity_and_options"],
            legacy.manifest(precision)["capacity_and_options"]
        );
        assert_eq!(
            config.manifest(precision)["expert_read_pipeline"],
            expert_read_manifest(Some(read))
        );
        assert_eq!(
            read.max_host_bytes - read.max_metadata_bytes - read.worker_stack_bytes as u64,
            54 << 20
        );
        assert_eq!(
            config.capacity(precision).device_bytes as u64 + read.max_host_bytes,
            7_583_301_632
        );
    }
}

#[test]
fn full40_storage_middle64_matches_frozen_case_middle_argv() {
    let case: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../papers/arxiv/experiments/storage_demand/case-middle.json"
    )))
    .unwrap();
    let argv = case["argv"].as_array().unwrap();
    let flag = |name: &str| {
        argv[argv.iter().position(|v| v == name).unwrap() + 1]
            .as_str()
            .unwrap()
    };
    let options = full40_options(
        Full40TestConfig::StorageMiddle64,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
    );
    let limits = options.qwen35_moe_capacity.unwrap();
    for (name, expected) in [
        ("--ctx-size", limits.max_positions),
        ("--max-active-sequences", limits.max_sequences),
        ("--max-batch-tokens", limits.max_batch_tokens),
        (
            "--prefill-chunk-size",
            options.scheduler_config.prefill_chunk_size,
        ),
        ("--kv-cache-mb", limits.kv_bytes >> 20),
        ("--max-tensor-mb", options.max_tensor_mebibytes as usize),
        ("--cuda-device-budget-mb", limits.device_bytes >> 20),
        ("--cuda-workspace-mb", limits.workspace_bytes >> 20),
        ("--cuda-expert-scratch-mb", limits.scratch_bytes >> 20),
        ("--cuda-expert-device-entries", limits.max_experts),
        ("--cuda-expert-device-bytes", limits.max_bytes),
    ] {
        assert_eq!(flag(name).parse::<usize>().unwrap(), expected, "{name}");
    }
    assert_eq!(flag("--model-profile"), "qwen35-mixed-bf16");
    assert_eq!(flag("--expert-prewarm"), "lazy");
    assert_eq!(flag("--kv-execution-policy"), "resident");
    assert_eq!(case["topology"]["tp"], 1);
    assert_eq!(case["topology"]["pp"], 1);
    assert_eq!(case["topology"]["ep"], 1);
    assert_eq!(case["cache_bytes_include_scratch"], true);
    assert_eq!(case["environment"]["CUBLAS_WORKSPACE_CONFIG"], ":0:0");
    assert_eq!(options.scheduler_config.max_decode_batch, 2);
}

#[test]
fn full40_frozen_profiles_reject_capacity_scheduler_and_precision_drift() {
    let config = Full40TestConfig::StorageMiddle64;
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    for drift in 0..4 {
        let mut options = full40_options(config, precision);
        match drift {
            0 => options.qwen35_moe_capacity.as_mut().unwrap().max_experts = 8,
            1 => options.scheduler_config.max_decode_batch = 1,
            2 => options.scheduler_config.prefix_cache_capacity_pages = 1,
            _ => options.driver_config.enable_native_proposals = true,
        }
        assert!(std::panic::catch_unwind(|| config.assert_options(&options, precision)).is_err());
    }
    assert!(
        std::panic::catch_unwind(|| config.capacity(NumericFp8Precision::Bf16ModelF32Accumulate))
            .is_err()
    );
}
fn assert_reference_next_token(oracle: &Oracle, call_index: usize, stage: &str) {
    let call = &oracle.report["calls"][call_index];
    assert_eq!(call["stage"], stage);
    let logits = oracle.values(&format!("{stage}.logits"));
    assert_eq!(
        logits.len(),
        call["input_token_ids"].as_array().unwrap().len() * 248320
    );
    assert_eq!(
        fresh_quality::top1(logits.rchunks_exact(248320).next().unwrap()),
        call["next_token_id"].as_u64().unwrap() as u32,
        "{stage}: reference logits must agree with frozen next-token ID"
    );
}

#[test]
#[ignore = "CPU reference artifacts only; GPUs hidden; independently pinned full40 and fresh reference next-token IDs"]
fn numeric_bf16_storage_middle64_reference_greedy_ids_match_frozen_logits() {
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    let reference = required_path("FERRULE_NUMERIC_BF16_REFERENCE_DIR");
    let pin = std::env::var("FERRULE_NUMERIC_BF16_REFERENCE_SHA256").unwrap();
    for case in ["hello", "capital"] {
        let oracle = Oracle::open_bf16(&reference, case, &pin);
        for (i, stage) in ["prefill", "decode.0"].into_iter().enumerate() {
            assert_reference_next_token(&oracle, i, stage);
        }
    }
    let fresh = required_path("FERRULE_BF16_FRESH_DIR").join("oracle");
    let pin = std::env::var("FERRULE_BF16_FRESH_SHA256").unwrap();
    for case in ["qa", "arithmetic", "chinese"] {
        let oracle = Oracle::open_bf16_fresh(&fresh, case, &pin);
        let expected: Vec<u32> =
            serde_json::from_value(oracle.report["generated_ids"].clone()).unwrap();
        assert_eq!(expected.len(), 9);
        for (step, token) in expected.into_iter().enumerate() {
            assert_reference_next_token(&oracle, step, &format!("step.{step:02}"));
            assert_eq!(oracle.report["calls"][step]["next_token_id"], token);
        }
    }
}

fn required_path(name: &str) -> PathBuf {
    let value = std::env::var_os(name).unwrap_or_else(|| panic!("set {name}; no fallback"));
    assert!(!value.is_empty());
    value.into()
}
fn typed(values: &[f32]) {
    assert!(
        values
            .iter()
            .all(|v| v.is_finite() && v.to_bits() & 0xffff == 0),
        "actual BF16 words required"
    );
}

const CAPTURE_PHASES_ENV: &str = "FERRULE_NUMERIC_BF16_CAPTURE_PHASES";
const DEFAULT_CAPTURE_PHASE: &str = "hello/prefill";
const VALID_CAPTURE_PHASES: [&str; 4] = [
    "hello/prefill",
    "hello/decode.0",
    "capital/prefill",
    "capital/decode.0",
];

fn parse_capture_phases(value: Option<&str>) -> std::result::Result<BTreeSet<String>, String> {
    let raw = value.unwrap_or(DEFAULT_CAPTURE_PHASE);
    let mut phases = BTreeSet::new();
    for item in raw.split(',').map(str::trim) {
        if !VALID_CAPTURE_PHASES.contains(&item) {
            return Err(format!(
                "{CAPTURE_PHASES_ENV} contains unknown phase {item:?}"
            ));
        }
        if !phases.insert(item.to_owned()) {
            return Err(format!("{CAPTURE_PHASES_ENV} repeats phase {item:?}"));
        }
    }
    if phases.is_empty() {
        return Err(format!(
            "{CAPTURE_PHASES_ENV} must select at least one phase"
        ));
    }
    Ok(phases)
}

fn selected_capture_phases() -> BTreeSet<String> {
    let value = match std::env::var(CAPTURE_PHASES_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("invalid {CAPTURE_PHASES_ENV}: {error}"),
    };
    parse_capture_phases(value.as_deref()).unwrap_or_else(|error| panic!("{error}"))
}

#[test]
fn capture_phase_selection_is_explicit_and_disjoint() {
    assert_eq!(
        parse_capture_phases(None).unwrap(),
        BTreeSet::from([DEFAULT_CAPTURE_PHASE.to_owned()])
    );
    assert_eq!(
        parse_capture_phases(Some("capital/prefill,hello/decode.0")).unwrap(),
        BTreeSet::from(["capital/prefill".to_owned(), "hello/decode.0".to_owned()])
    );
    assert!(parse_capture_phases(Some("hello/prefill,hello/prefill")).is_err());
    assert!(parse_capture_phases(Some("layer1/prefill")).is_err());
    assert!(parse_capture_phases(Some("")).is_err());
}

const CAPTURE_LAYERS_ENV: &str = "FERRULE_NUMERIC_BF16_CAPTURE_LAYERS";

fn parse_capture_layers(
    value: Option<&str>,
) -> std::result::Result<std::ops::Range<usize>, String> {
    let raw = value.unwrap_or("0:2");
    let (start, end) = raw
        .split_once(':')
        .ok_or_else(|| format!("{CAPTURE_LAYERS_ENV} requires start:end (exclusive end)"))?;
    let start: usize = start
        .parse()
        .map_err(|_| format!("invalid capture start: {start:?}"))?;
    let end: usize = end
        .parse()
        .map_err(|_| format!("invalid capture end: {end:?}"))?;
    if start >= end || end > 40 || end - start > 21 {
        return Err(format!(
            "{CAPTURE_LAYERS_ENV} requires 0 <= start < end <= 40, at most 21 layers"
        ));
    }
    Ok(start..end)
}

#[test]
fn capture_layer_range_is_bounded_and_preserves_default() {
    assert_eq!(parse_capture_layers(None).unwrap(), 0..2);
    assert_eq!(parse_capture_layers(Some("0:12")).unwrap(), 0..12);
    assert_eq!(parse_capture_layers(Some("0:21")).unwrap(), 0..21);
    assert_eq!(parse_capture_layers(Some("2:12")).unwrap(), 2..12);
    assert_eq!(parse_capture_layers(Some("28:40")).unwrap(), 28..40);
    for invalid in [
        "", "2", "-1:2", "2:2", "3:2", "0:22", "39:41", "0:2:3", "a:2",
    ] {
        assert!(parse_capture_layers(Some(invalid)).is_err(), "{invalid}");
    }
}

fn snapshot_layer_rows(
    words: &[u16],
    descriptor: KvPlaneDescriptor,
    page_tokens: usize,
    valid: usize,
) -> Vec<Vec<u16>> {
    assert_eq!(descriptor.element_type, KvElementType::Bf16);
    assert!(valid > 0 && valid <= page_tokens);
    assert_eq!(
        words.len(),
        descriptor.checked_page_elements(page_tokens).unwrap()
    );
    let layer_stride = page_tokens * descriptor.elements_per_token;
    (0..descriptor.layer_count)
        .map(|layer| {
            let start = layer * layer_stride;
            words[start..start + valid * descriptor.elements_per_token].to_vec()
        })
        .collect()
}

#[test]
fn full40_two_kv_planes_cover_ten_layers_and_partial_pages() {
    use ferrule_model::decoder::StandardGqaPlanes;
    let schema = StandardGqaPlanes::new(10, 2, 3, 16, 128, KvElementType::Bf16).unwrap();
    assert_eq!(schema.planes().len(), 2);
    let mut captured = vec![Vec::new(); 20];
    for (page, valid) in [(0, 16), (1, 3)] {
        for (plane, descriptor) in schema.planes().iter().enumerate() {
            let words: Vec<u16> = (0..10)
                .flat_map(|layer| {
                    (0..16).flat_map(move |row| {
                        (0..6).map(move |column| {
                            (plane * 16000 + layer * 1000 + (page * 16 + row) * 6 + column) as u16
                        })
                    })
                })
                .collect();
            let rows = snapshot_layer_rows(&words, *descriptor, 16, valid);
            assert_eq!(rows.len(), 10);
            for (layer, rows) in rows.into_iter().enumerate() {
                captured[layer * 2 + plane].extend(rows);
            }
        }
    }
    for (slot, rows) in captured.iter().enumerate() {
        let expected: Vec<u16> = (0..19 * 6)
            .map(|offset| ((slot % 2) * 16000 + (slot / 2) * 1000 + offset) as u16)
            .collect();
        assert_eq!(rows, &expected, "plane/layer {slot}");
    }
    assert!(
        std::panic::catch_unwind(|| snapshot_layer_rows(&[0; 6], schema.planes()[0], 16, 1))
            .is_err()
    );
}

#[allow(clippy::too_many_arguments)]
fn check_state(
    runner: &mut Runner,
    state: &CudaHybridSequenceState,
    pages: &[KvPageId],
    oracle: &Oracle,
    case: &str,
    stage: &str,
    report: &mut StrictReport,
    reads: &ReadLog,
) {
    assert!(
        state.kv_state().layers().is_empty(),
        "no F32 history mirror"
    );
    let mut gdn = 0;
    for (layer, value) in state.kv_state().bf16_layers().unwrap().iter().enumerate() {
        if let Some(value) = value {
            gdn += 1;
            assert_eq!(value.position(), state.core().position());
            let (conv, ssm) = observe_operation(
                runner,
                reads,
                &format!("{case}/{stage}/gdn_snapshot/{layer}"),
                "diagnostic_state_snapshot",
                |_| value.diagnostic_snapshot(),
            )
            .unwrap();
            typed(&conv);
            for (field, actual) in [("conv_states", conv), ("recurrent_states", ssm)] {
                let key = format!("{stage}.cache.{field}.{layer:02}");
                report.bf16_values(&actual, &oracle.values(&key), &format!("{case}/{key}"));
            }
        }
    }
    assert_eq!(gdn, 30);
    let descriptors = runner.backend().pool().planes();
    let page_tokens = runner.backend().pool().page_size();
    assert_eq!(descriptors.len(), 2);
    assert_eq!(descriptors[0].name, "standard_gqa.key");
    assert_eq!(descriptors[1].name, "standard_gqa.value");
    assert!(descriptors.iter().all(|d| d.layer_count == 10));
    assert_eq!(pages.len(), state.core().position().div_ceil(page_tokens));
    let mut planes = vec![Vec::<f32>::new(); 20];
    for (logical, &page) in pages.iter().enumerate() {
        runner.preempt_kv_pages(&[page]).unwrap();
        let snapshot = observe_operation(
            runner,
            reads,
            &format!("{case}/{stage}/kv_snapshot/{logical}"),
            "diagnostic_kv_snapshot",
            |runner| runner.backend().pool().typed_host_snapshot(page),
        )
        .unwrap();
        assert_eq!(snapshot.planes.len(), descriptors.len());
        let valid = (state.core().position() - logical * page_tokens).min(page_tokens);
        for (i, (plane, descriptor)) in snapshot.planes.iter().zip(&descriptors).enumerate() {
            assert_eq!(plane.element_type(), KvElementType::Bf16);
            for (layer, rows) in
                snapshot_layer_rows(plane.bf16_words().unwrap(), *descriptor, page_tokens, valid)
                    .into_iter()
                    .enumerate()
            {
                planes[layer * 2 + i]
                    .extend(rows.into_iter().map(|w| f32::from_bits(u32::from(w) << 16)));
            }
        }
        runner.restore_kv_pages(&[page]).unwrap();
    }
    for (slot, actual) in planes.iter_mut().enumerate() {
        let layer = (slot / 2) * 4 + 3;
        let field = if slot % 2 == 0 {
            "key_cache"
        } else {
            "value_cache"
        };
        let key = format!("{stage}.cache.{field}.{layer:02}");
        let shape: Vec<usize> =
            serde_json::from_value(oracle.header[&key]["shape"].clone()).unwrap();
        assert_eq!(shape[0], 1);
        assert_eq!(shape[2], state.core().position());
        let hf = oracle.values(&key);
        // HF [batch, heads, sequence, dim] -> paged [sequence, heads, dim].
        let mut expected = Vec::with_capacity(hf.len());
        for token in 0..shape[2] {
            for head in 0..shape[1] {
                let start = (head * shape[2] + token) * shape[3];
                expected.extend_from_slice(&hf[start..start + shape[3]]);
            }
        }
        assert_eq!(
            actual.len(),
            expected.len(),
            "all valid KV rows, no padding"
        );
        typed(actual);
        report.bf16_values(actual, &expected, &format!("{case}/{key}"));
    }
}

#[test]
#[ignore = "sole GPU owner only; real 35B factory plus independently pinned BF16/profile full40 reference required"]
fn numeric_bf16_35b_public_factory_full40_logits_routes_states_match_profile_reference() {
    full40(NumericFp8Precision::Bf16ModelF32Accumulate);
}

#[test]
#[ignore = "sole GPU owner only; explicit tiled BF16 public factory against unchanged pinned profile reference"]
fn numeric_bf16_tiled_35b_public_factory_full40_logits_routes_states_match_profile_reference() {
    full40(NumericFp8Precision::Bf16ModelTiledF32Accumulate);
}

#[test]
#[ignore = "sole GPU owner only; explicit sliced-K2 gate schedule plus existing tiled fallback against unchanged reference"]
fn numeric_bf16_sliced_k2_35b_public_factory_full40_logits_routes_states_match_profile_reference() {
    full40(NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate);
}

#[test]
#[ignore = "sole GPU owner; explicit mixed reduction using unchanged independent BF16 profile reference"]
fn numeric_bf16_mixed_35b_public_factory_full40_logits_routes_states_match_profile_reference() {
    full40(NumericFp8Precision::Bf16ModelMixedF32Accumulate);
}

#[test]
#[ignore = "sole GPU owner; frozen storage-middle64/maxdecode2 Mixed factory against unchanged independent full40 reference"]
fn numeric_bf16_mixed_storage_middle64_public_factory_full40_logits_routes_states_match_profile_reference()
 {
    full40_run(
        Full40TestConfig::StorageMiddle64,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
        None,
    );
}

#[test]
#[ignore = "sole GPU owner; enabled Blocking host reads; unchanged storage-middle64 full40 reference/tolerances"]
fn numeric_bf16_mixed_storage_middle64_blocking_public_factory_full40_logits_routes_states_match_profile_reference()
 {
    full40_run(
        Full40TestConfig::StorageMiddle64Blocking,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
        None,
    );
}

#[test]
#[ignore = "sole GPU owner; enabled Lookahead host reads; unchanged storage-middle64 full40 reference/tolerances"]
fn numeric_bf16_mixed_storage_middle64_lookahead_public_factory_full40_logits_routes_states_match_profile_reference()
 {
    full40_run(
        Full40TestConfig::StorageMiddle64Lookahead,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
        None,
    );
}

#[test]
#[ignore = "sole GPU owner; retained Blocking uploads, unchanged full40 reference and host64MiB quota"]
fn numeric_bf16_mixed_storage_middle64_retained_blocking_public_factory_full40_logits_routes_states_match_profile_reference()
 {
    full40_run(
        Full40TestConfig::StorageMiddle64RetainedBlocking,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
        None,
    );
}

#[test]
#[ignore = "sole GPU owner; retained Lookahead uploads, unchanged full40 reference and host64MiB quota"]
fn numeric_bf16_mixed_storage_middle64_retained_lookahead_public_factory_full40_logits_routes_states_match_profile_reference()
 {
    full40_run(
        Full40TestConfig::StorageMiddle64RetainedLookahead,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
        None,
    );
}

fn configure_full40_expert_transfer(runner: &mut Runner, config: Full40TestConfig) {
    assert_eq!(
        runner.expert_transfer_policy(),
        ExpertTransferPolicy::Synchronous
    );
    if config.transfer_manifest().is_some() {
        runner
            .set_expert_transfer_policy(config.expert_transfer_policy())
            .unwrap();
        assert!(
            runner
                .set_expert_transfer_policy(ExpertTransferPolicy::Synchronous)
                .is_err(),
            "an explicit retained profile must not escape to default-off execution"
        );
        assert!(
            runner
                .set_expert_transfer_policy(config.expert_transfer_policy())
                .is_err(),
            "duplicate policy configuration must not replace owner state"
        );
    }
    assert_eq!(
        runner.expert_transfer_policy(),
        config.expert_transfer_policy()
    );
}

fn assert_full40_expert_transfer_exercised(runner: &Runner, config: Full40TestConfig) {
    assert_eq!(
        runner.expert_transfer_policy(),
        config.expert_transfer_policy()
    );
    if let ExpertTransferPolicy::Retained {
        policy_headroom_bytes,
        ..
    } = config.expert_transfer_policy()
    {
        let read = config.expert_read_pipeline().unwrap();
        let observed = runner
            .forward_executor()
            .module()
            .expert_read_pipeline_observation();
        // Original Qwen3.5 source headers: three 512*2048 FP8 matrices,
        // each with 64 BF16 scales. This is logical storage, not physical RSS.
        let payload = 3 * (512 * 2048 + 64 * 2);
        let minimum = 3 * payload
            + std::mem::size_of::<
                ferrule_backend::cuda::operators::linear::CudaNumericFp8UploadTicket,
            >()
            + policy_headroom_bytes
            + read.max_metadata_bytes as usize
            + read.worker_stack_bytes;
        assert!(
            runner
                .forward_executor()
                .module()
                .expert_cache_stats()
                .unwrap()
                .uploads
                > 0
        );
        assert!(
            observed.stats.peak_host_charged_bytes >= minimum as u64,
            "no positive aggregate retained-upload reservation evidence"
        );
        assert!(observed.stats.peak_host_charged_bytes <= read.max_host_bytes);
        assert_eq!(observed.current_charge.pinned_upload_bytes, 0);
        assert_eq!(observed.current_charge.upload_ticket_metadata_bytes, 0);
        assert_eq!(observed.current_charge.upload_policy_headroom_bytes, 0);
        assert!(!observed.worker_unknown);
    }
}

fn prepare_full40(
    model: &std::path::Path,
    precision: NumericFp8Precision,
) -> ResidentModelBuildPlan {
    prepare_full40_with_config(model, precision, Full40TestConfig::Historical8)
}

fn prepare_full40_with_config(
    model: &std::path::Path,
    precision: NumericFp8Precision,
    test_config: Full40TestConfig,
) -> ResidentModelBuildPlan {
    let config = AutoConfig::from_pretrained(model).unwrap();
    let options = full40_options(test_config, precision);
    let plan = ResidentModelPlanner
        .prepare_numeric_fp8_bf16_serving_with_precision(
            &config,
            BackendSelection::Cuda,
            None,
            options,
            ServingModelOptions {
                expert_read_pipeline: test_config.expert_read_pipeline(),
                model_precision: Some(ModelPrecision::Bf16),
                max_tensor_mebibytes: Some(1024),
                ..Default::default()
            },
            NumericStoragePolicy::E4M3FnBf16Scales,
            precision,
        )
        .unwrap();
    full40_effective_config(&plan, test_config, precision);
    plan
}

fn full40(precision: NumericFp8Precision) {
    full40_run(Full40TestConfig::Historical8, precision, None);
}

#[test]
#[ignore = "sole GPU owner; one full-image Capital prefill/decode layer01 diagnostic capture, not acceptance"]
fn numeric_bf16_mixed_capital_layer01_capture() {
    assert_eq!(
        selected_capture_phases(),
        BTreeSet::from(["capital/prefill".to_owned(), "capital/decode.0".to_owned()])
    );
    full40_run(
        Full40TestConfig::Historical8,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
        Some("capital"),
    );
}

#[test]
#[ignore = "sole GPU owner; one full-image Hello prefill producer capture, not acceptance"]
fn numeric_bf16_mixed_hello_prefill_producer_capture() {
    assert_eq!(
        selected_capture_phases(),
        BTreeSet::from(["hello/prefill".to_owned()])
    );
    full40_run(
        Full40TestConfig::Historical8,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
        Some("hello"),
    );
}

fn full40_run(
    test_config: Full40TestConfig,
    precision: NumericFp8Precision,
    capture_case: Option<&str>,
) {
    let capture_only = capture_case.is_some();
    let reads = test_config
        .expert_read_pipeline()
        .map(|read| Rc::new(RefCell::new(ReadQualityLog::new(read, true))));
    let limits = test_config.capacity(precision);
    let config_manifest = test_config.manifest(precision);
    let execution_profile = precision.hybrid_backend_name();
    assert_eq!(
        std::env::var("FERRULE_NUMERIC_BF16_FACTORY_GPU").as_deref(),
        Ok("1")
    );
    let model = required_path("FERRULE_NUMERIC_BF16_MODEL");
    let reference = required_path("FERRULE_NUMERIC_BF16_REFERENCE_DIR");
    let pin =
        std::env::var("FERRULE_NUMERIC_BF16_REFERENCE_SHA256").expect("explicit manifest pin");
    let report_dir = required_path("FERRULE_NUMERIC_BF16_REPORT_DIR");
    // Reject incomplete/unknown/F32/native-unstable reference before opening a GPU.
    let oracles = [
        ("hello", Oracle::open_bf16(&reference, "hello", &pin)),
        ("capital", Oracle::open_bf16(&reference, "capital", &pin)),
    ]
    .into_iter()
    .filter(|(case, _)| capture_case.is_none_or(|selected| *case == selected))
    .collect::<Vec<_>>();
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(reference.join("manifest.json")).unwrap()).unwrap();
    check_hash(
        &model.join("config.json"),
        manifest["checkpoint"]["config_sha256"].as_str().unwrap(),
    );
    for name in ["tokenizer.json", "tokenizer_config.json"] {
        check_hash(
            &model.join(name),
            manifest["tokenizer_files"][name].as_str().unwrap(),
        );
    }
    for (case, oracle) in &oracles {
        let ids = if *case == "hello" {
            vec![9419]
        } else {
            vec![760, 6511, 314, 9338, 369]
        };
        assert_eq!(oracle.report["calls"][0]["input_token_ids"], json!(ids));
        assert_eq!(oracle.ids("prefill.input_ids"), ids);
        assert_eq!(
            oracle.ids("decode.0.input_ids"),
            vec![oracle.report["calls"][0]["next_token_id"].as_u64().unwrap() as usize]
        );
        for (i, stage) in ["prefill", "decode.0"].into_iter().enumerate() {
            let rows = if i == 0 { ids.len() } else { 1 };
            assert_eq!(
                oracle.header[format!("{stage}.logits")]["shape"],
                json!([1, rows, 248320])
            );
            assert_eq!(oracle.report["calls"][i]["stage"], stage);
            assert_reference_next_token(oracle, i, stage);
            assert_eq!(
                oracle.report["calls"][i]["input_token_ids"],
                json!(oracle.ids(&format!("{stage}.input_ids")))
            );
        }
    }
    std::fs::create_dir_all(&report_dir).unwrap();
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(report_dir.join("full40-bf16-profile.json"))
        .unwrap();
    let mut initial = json!({"schema":"ferrule.numeric-bf16-full40-profile-gate.v1",
        "status":"reference verified; factory preparation pending","passed":false,"complete":false,
        "profile":execution_profile,"reference_profile":BF16_PROFILE,"reference_manifest_sha256":pin,
        "test_config":config_manifest,"effective_config":null,"observer_mode":"diagnostic_routes_and_states",
        "ordinary_no_observer_claim":false,"concurrent_scheduler_claim":false,
        "greedy_tie_policy":"lower_token_id","next_token_checks":[]});
    attach_read_reports(&mut initial, &reads, None);
    serde_json::to_writer_pretty(&mut output, &initial).unwrap();
    output.sync_all().unwrap();
    let plan = prepare_full40_with_config(&model, precision, test_config);
    let effective_config = full40_effective_config(&plan, test_config, precision);
    let memory = plan.numeric_bf16_memory_plan().unwrap().unwrap();
    let budget = memory.model_budget();
    let estimate = memory.model_estimate();
    eprintln!(
        "BF16 full40 factory estimate={estimate:?} budget={budget:?} total={}",
        memory.total_device_bytes()
    );
    drop(memory);
    let report = Rc::new(RefCell::new(StrictReport::default()));
    let result = Rc::clone(&report);
    let token_checks = Rc::new(RefCell::new(Vec::<Value>::new()));
    let observed_tokens = Rc::clone(&token_checks);
    let progress_tokens = Rc::clone(&token_checks);
    let progress_config = config_manifest.clone();
    let progress_effective = effective_config.clone();
    let progress_file = Rc::new(RefCell::new(output.try_clone().unwrap()));
    let progress_report = Rc::clone(&report);
    let progress_pin = pin.clone();
    let progress_reads = reads.clone();
    let persist = Rc::new(move |status: &str| {
        let r = progress_report.borrow();
        let mut file = progress_file.borrow_mut();
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        file.set_len(0).unwrap();
        let mut data = json!({"schema":"ferrule.numeric-bf16-full40-profile-gate.v1",
            "status":status,"passed":false,"complete":false,"profile":execution_profile,"reference_profile":BF16_PROFILE,
            "reference_manifest_sha256":progress_pin,"logits_compared":r.logits,
            "test_config":progress_config,"effective_config":progress_effective,
            "observer_mode":"diagnostic_routes_and_states","ordinary_no_observer_claim":false,
            "concurrent_scheduler_claim":false,"greedy_tie_policy":"lower_token_id",
            "next_token_checks":*progress_tokens.borrow(),"failures":r.failures,"metrics":r.metrics});
        attach_read_reports(&mut data, &progress_reads, None);
        serde_json::to_writer_pretty(&mut *file, &data).unwrap();
        file.sync_all().unwrap();
    });
    persist("prepared; computation not started");
    let progress = Rc::clone(&persist);
    let diagnostic_path =
        std::env::var_os("FERRULE_NUMERIC_BF16_FIRST_TRACE_DIR").map(PathBuf::from);
    let capture_phases = selected_capture_phases();
    let layers_value = match std::env::var(CAPTURE_LAYERS_ENV) {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => panic!("invalid {CAPTURE_LAYERS_ENV}: {error}"),
    };
    let capture_layers = parse_capture_layers(layers_value.as_deref()).unwrap();
    assert!(
        diagnostic_path.is_some() || layers_value.is_none(),
        "capture layers require a trace directory"
    );
    assert!(
        diagnostic_path.is_some() || std::env::var_os(CAPTURE_PHASES_ENV).is_none(),
        "capture phases require FERRULE_NUMERIC_BF16_FIRST_TRACE_DIR"
    );
    let call_reads = reads.clone();
    crate::model_factory::bf16_serving::NUMERIC_BF16_TEST_RUNNER.with(|slot| {
        assert!(slot.borrow().is_none());
        *slot.borrow_mut() = Some(Box::new(move |runner| {
            configure_full40_expert_transfer(runner, test_config);
            assert_eq!(runner.forward_executor().module().expert_read_pipeline_options(), test_config.expert_read_pipeline());
            assert!(
                runner
                    .forward_executor()
                    .module()
                    .numeric_bf16_profile()
                    .is_some()
            );
            assert_eq!(runner.forward_executor().module().numeric_fp8_precision(), Some(precision));
            let routes = Rc::new(RefCell::new(BTreeMap::new()));
            let capture = Rc::clone(&routes);
            let observing = Rc::new(RefCell::new(String::new()));
            let enabled = Rc::clone(&observing);
            let state_capture_path = diagnostic_path.clone();
            let mut diagnostic = diagnostic_path.map(|path| {
                std::fs::create_dir(&path).unwrap();
                let mut streams = BTreeMap::new();
                for phase in &capture_phases {
                    let directory = path.join(phase);
                    std::fs::create_dir_all(&directory).unwrap();
                    let events = std::fs::OpenOptions::new().create_new(true).write(true)
                        .open(directory.join("events.jsonl")).unwrap();
                    streams.insert(phase.clone(), (directory, events, 0usize, 0usize));
                }
                streams
            });
            let event_layers = capture_layers.clone();
            runner
                .forward_executor()
                .module()
                .set_diagnostic_trace(move |event| {
                    let phase = enabled.borrow();
                    if event.layer.is_none_or(|layer| event_layers.contains(&layer))
                        && let Some((path, log, sequence, bytes)) = diagnostic.as_mut().and_then(|streams| streams.get_mut(phase.as_str())) {
                        let values = event.operators.download_f32_buffer(event.buffer)?;
                        *bytes += values.len()*4;
                        assert!(*bytes <= 64 << 20, "bounded-layer diagnostic cap");
                        let file = format!("{sequence:04}.f32");
                        std::fs::write(path.join(&file), values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
                        let tensor = |s: &ferrule_model::CheckpointTensorSlice| json!({"name":s.name,"path":s.path,"offset":s.offset,"bytes":s.bytes,"shape":s.shape,"dtype":format!("{:?}",s.dtype)});
                        writeln!(log, "{}", json!({"stage":phase.as_str(),"event":event.name,"layer":event.layer,"shape":[event.shape.rows(),event.shape.width()],
                            "file":file,"parameter":event.parameter.map(|p| p.path().to_string()),
                            "weight":event.parameter.map(|p| tensor(p.weight().slice())),
                            "scale":event.parameter.and_then(|p| p.scale()).map(|p| tensor(p.slice())),
                            "epsilon":event.norm.map(|n| n.epsilon()),"one_plus_weight":event.norm.map(|n| n.one_plus_weight()),
                            "ids":event.routes.map(|r| r.expert_ids()),"weights":event.routes.map(|r| r.weights())}))?;
                        log.sync_data()?;
                        *sequence += 1;
                    }
                    if let Some(routes) = event.routes {
                        assert!(
                            capture
                                .borrow_mut()
                                .insert(
                                    event.layer.unwrap(),
                                    (
                                        event.shape.rows(),
                                        routes.expert_ids().to_vec(),
                                        routes.weights().to_vec(),
                                        event.operators.download_f32_buffer(event.buffer)?
                                    )
                                )
                                .is_none()
                        );
                    }
                    Ok(())
                });
            let mut transaction = 0;
            for (case, oracle) in oracles {
                let mut states = vec![runner.create_sequence_state().unwrap()];
                let mut pages = vec![Vec::new()];
                for (i, stage) in ["prefill", "decode.0"].into_iter().enumerate() {
                    if capture_only && !capture_phases.contains(&format!("{case}/{stage}")) {
                        continue;
                    }
                    let tokens: Vec<u32> = serde_json::from_value(
                        oracle.report["calls"][i]["input_token_ids"].clone(),
                    )
                    .unwrap();
                    assert_eq!(
                        oracle.report["calls"][i]["position_start"],
                        states[0].core().position()
                    );
                    transaction += 1;
                    let capture_phase = format!("{case}/{stage}");
                    *observing.borrow_mut() = capture_phase.clone();
                    if capture_phases.contains(&capture_phase)
                        && let Some(path) = &state_capture_path {
                        for (layer, state) in states[0].kv_state().bf16_layers().unwrap().iter().enumerate().filter(|(layer, _)| capture_layers.contains(layer)) {
                            let Some(state) = state.as_ref() else { continue };
                            let (conv, ssm) = observe_operation(runner, &call_reads, &format!("{capture_phase}/before_gdn_snapshot/{layer}"), "diagnostic_state_snapshot", |_| state.diagnostic_snapshot()).unwrap();
                            for (field, values) in [("conv", conv), ("ssm", ssm)] {
                                std::fs::write(path.join(&capture_phase).join(format!("before-{layer}-{field}.f32")),
                                    values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
                            }
                        }
                    }
                    let logits = observe_operation(runner, &call_reads, &format!("{case}/{stage}"), "prediction", |runner| crate::model_factory::qwen35::full_logits_step(
                        runner,
                        &mut states,
                        &mut pages,
                        &tokens,
                        if i == 0 {
                            ForwardPhase::Prefill
                        } else {
                            ForwardPhase::Decode
                        },
                        transaction,
                    ));
                    assert_eq!(
                        oracle.report["calls"][i]["sequence_length_after"],
                        states[0].core().position()
                    );
                    assert_eq!(logits.len(), tokens.len());
                    assert!(logits.iter().all(|r| r.len() == 248320));
                    let next = fresh_quality::top1(logits.last().unwrap());
                    let expected_next = oracle.report["calls"][i]["next_token_id"].as_u64().unwrap() as u32;
                    observed_tokens.borrow_mut().push(json!({"case":case,"stage":stage,
                        "actual":next,"expected":expected_next,"equal":next==expected_next,
                        "observation":"actual runner final-row logits argmax; reference-token input history"}));
                    if next != expected_next {
                        result.borrow_mut().failures.push(format!("{case}/{stage}/next_token: actual={next}, expected={expected_next}"));
                    }
                    let actual: Vec<f32> = logits.into_iter().flatten().collect();
                    typed(&actual);
                    let mut report = result.borrow_mut();
                    report.logits += actual.len();
                    report.bf16_values(
                        &actual,
                        &oracle.values(&format!("{stage}.logits")),
                        &format!("{case}/{stage}/all_logits"),
                    );
                    let captured = std::mem::take(&mut *routes.borrow_mut());
                    assert_eq!(captured.len(), 40);
                    for (layer, (rows, ids, weights, raw)) in captured {
                        assert_eq!(rows, tokens.len());
                        assert_eq!(ids.len(), rows * 8);
                        let prefix = format!("{stage}.layers.{layer:02}");
                        report.route_ids(
                            &ids,
                            &oracle.ids(&format!("{prefix}.selected_expert_ids")),
                            &format!("{case}/{prefix}/ordered_routes"),
                        );
                        report.bf16_values(
                            &weights,
                            &oracle.values(&format!("{prefix}.routing_weights")),
                            &format!("{case}/{prefix}/weights"),
                        );
                        typed(&raw);
                        report.bf16_values(
                            &raw,
                            &oracle.values(&format!("{prefix}.raw_router_logits")),
                            &format!("{case}/{prefix}/raw_router_logits"),
                        );
                    }
                    drop(report);
                    progress(&format!("{case}/{stage}: logits/routes complete; state checks pending"));
                    let mut report = result.borrow_mut();
                    check_state(
                        runner,
                        &states[0],
                        &pages[0],
                        &oracle,
                        case,
                        stage,
                        &mut report,
                        &call_reads,
                    );
                    drop(report);
                    progress(&format!("{case}/{stage}: logits/routes/GDN/KV complete"));
                }
                runner.try_release_sequence_state(states.remove(0)).unwrap();
                runner.release_kv_pages(&pages[0]).unwrap();
            }
            runner.forward_executor().module().clear_diagnostic_trace();
            assert_full40_expert_transfer_exercised(runner, test_config);
        }));
    });
    let mut engine = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| plan.build())) {
        Ok(Ok(engine)) => engine,
        Ok(Err(error)) => {
            persist(&format!("build/forward error: {error}"));
            panic!("{error}");
        }
        Err(panic) => {
            persist("panicked; partial metrics only; NOT passed");
            std::panic::resume_unwind(panic);
        }
    };
    assert_eq!(engine.bound_layer_count(), Some(40));
    assert_eq!(engine.model_info().backend, execution_profile);
    let device = crate::model_factory::bf16_serving::NUMERIC_BF16_TEST_DEVICE
        .with(|slot| slot.borrow_mut().take())
        .unwrap();
    assert_eq!(device.live_state_bytes(), estimate.per_sequence_state_bytes);
    let metrics = device.operators().allocator_metrics();
    assert!(metrics.peak_reserved_bytes <= limits.device_bytes);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        if engine.shutdown().unwrap() == InferenceShutdownProgress::Complete {
            break;
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    drop(engine);
    assert_eq!(device.live_state_bytes(), 0);
    assert!(!device.needs_quarantine());
    let report = report.borrow();
    assert_eq!(
        report.logits,
        match capture_case {
            Some("hello") => 248_320,
            Some("capital") => 1_489_920,
            None => 1_986_560,
            _ => unreachable!(),
        }
    );
    let token_checks = token_checks.borrow();
    assert_eq!(
        token_checks.len(),
        if capture_case == Some("hello") {
            1
        } else if capture_only {
            2
        } else {
            4
        }
    );
    let passed = report.failures.is_empty()
        && token_checks.iter().all(|call| call["equal"] == true)
        && reads.as_ref().is_none_or(|log| log.borrow().passed());
    let mut summary = json!({"schema":"ferrule.numeric-bf16-full40-profile-gate.v1", "passed":passed && !capture_only,
        "diagnostic_capture_only":capture_only,"test_config":config_manifest,"effective_config":effective_config,
        "observer_mode":"diagnostic_routes_and_states","ordinary_no_observer_claim":false,
        "concurrent_scheduler_claim":false,"greedy_tie_policy":"lower_token_id","next_token_checks":*token_checks,
        "complete":true,"status":if capture_only {"diagnostic_capture_complete_not_acceptance"} else if passed {"passed"} else {"numerical_failure"},
        "profile":execution_profile,"reference_profile":BF16_PROFILE,"reference_manifest_sha256":pin,"router":manifest["router"],
        "logits_compared":report.logits,"atol":BF16_ATOL,"rtol":BF16_RTOL,
        "tolerance_source":"existing hybrid_bf16 tiny policy, NOT old F32 gate or calibrated full40 bound",
        "scope":if capture_only {"selected full-image phases and bounded-layer observer; numerical metrics retained without acceptance assertion"} else {"hello/capital full40 prefill and reference-token cached decode; all logits, ordered routes, weights, raw router logits, actual BF16 KV/conv and F32 SSM"},
        "failures":report.failures,"metrics":report.metrics,"actual_peak_reserved_bytes":metrics.peak_reserved_bytes,
        "device_budget_bytes":limits.device_bytes,"state_bytes_after_shutdown":0,"quarantine":false});
    attach_read_reports(&mut summary, &reads, None);
    if reads.is_some() {
        summary["diagnostic_denominators"] = json!({"prediction_calls":token_checks.len(),"logit_values_compared":report.logits,
            "ordered_route_ids_compared":report.metrics.iter().filter(|m|m["label"].as_str().is_some_and(|label|label.ends_with("ordered_routes"))).map(|m|m["elements"].as_u64().unwrap()).sum::<u64>(),
            "state_metric_elements_compared":report.metrics.iter().filter(|m|m["label"].as_str().is_some_and(|label|label.contains(".cache."))).map(|m|m["elements"].as_u64().unwrap()).sum::<u64>(),
            "ordinary_no_observer_prediction_calls":0});
    }

    output.seek(std::io::SeekFrom::Start(0)).unwrap();
    output.set_len(0).unwrap();
    writeln!(
        output,
        "{}",
        serde_json::to_string_pretty(&summary).unwrap()
    )
    .unwrap();
    output.sync_all().unwrap();
    if !capture_only {
        assert!(
            passed,
            "BF16 profile full40 mismatch; inspect immutable report (not a W8/F32 failure)"
        );
    }
}

#[test]
#[ignore = "real 35B source headers only; CUDA_VISIBLE_DEVICES must be empty; no CUDA owner or payload loads"]
fn numeric_bf16_35b_full40_metadata_budget_matches_source_headers() {
    metadata_budget(NumericFp8Precision::Bf16ModelF32Accumulate);
}

#[test]
#[ignore = "real 35B source headers only; GPUs hidden; tiled all-projection preflight and public factory freeze"]
fn numeric_bf16_tiled_35b_full40_metadata_budget_matches_source_headers() {
    metadata_budget(NumericFp8Precision::Bf16ModelTiledF32Accumulate);
}

#[test]
#[ignore = "35B headers only; explicit sliced-K2 schedule, one bounded workspace and unchanged 7GiB cap"]
fn numeric_bf16_sliced_k2_35b_full40_metadata_budget_matches_source_headers() {
    metadata_budget(NumericFp8Precision::Bf16ModelSlicedK2F32Accumulate);
}

#[test]
#[ignore = "35B headers only; mixed reduction max scratch, unchanged 7GiB cap and public factory"]
fn numeric_bf16_mixed_35b_full40_metadata_budget_matches_source_headers() {
    metadata_budget(NumericFp8Precision::Bf16ModelMixedF32Accumulate);
}

#[test]
#[ignore = "35B headers only; GPUs hidden; storage-middle64/maxdecode2 effective factory and source-lowering assertions"]
fn numeric_bf16_mixed_storage_middle64_full40_metadata_budget_matches_source_headers() {
    metadata_budget_with_config(
        Full40TestConfig::StorageMiddle64,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
    );
}

fn metadata_budget(precision: NumericFp8Precision) {
    metadata_budget_with_config(Full40TestConfig::Historical8, precision);
}

fn metadata_budget_with_config(test_config: Full40TestConfig, precision: NumericFp8Precision) {
    use ferrule_model::CheckpointDType;
    use ferrule_model::nn::ParameterResidency;
    use ferrule_model::transformer::Attention;
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    let model = required_path("FERRULE_NUMERIC_BF16_MODEL");
    let limits = test_config.capacity(precision);
    let plan = Qwen35NumericBf16MemoryPlan::open_hf_with_precision(
        &model,
        ModelPrecision::Bf16,
        NumericStoragePolicy::E4M3FnBf16Scales,
        limits,
        1 << 30,
        precision,
    )
    .unwrap();
    assert_eq!(plan.numeric_precision(), precision);
    assert_eq!(plan.backend_profile(), precision.hybrid_backend_name());
    let mut numeric_roles = BTreeMap::<String, usize>::new();
    let mut native_roles = BTreeMap::<String, usize>::new();
    let mut native_linear_temporary_bytes = 0;
    let mut largest_activation = 0;
    let mut largest_weight_tile = 0;
    let mut largest_scratch_grant = 0;
    let mut provider_reserve_bytes = 0;
    for p in plan.resources().state_dict().parameters() {
        if let Some(encoding) = p.numeric_fp8_encoding() {
            use ferrule_backend::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};
            let source = p.numeric_fp8_source(encoding).unwrap();
            let [n, k] = source.matrix_shape();
            for rows in [1, limits.max_batch_tokens] {
                let linear = precision
                    .linear_plan(
                        p.role(),
                        NumericFp8Layout {
                            n,
                            k,
                            row_origin: 0,
                            column_origin: 0,
                            scale_type: NumericFp8ScaleType::Bf16,
                        },
                        rows,
                        limits.scratch_bytes,
                    )
                    .unwrap();
                assert!(linear.workspace_requirements().bytes as usize <= limits.scratch_bytes);
                provider_reserve_bytes = provider_reserve_bytes.max(linear.provider_bytes());
                largest_activation = largest_activation.max(linear.activation_bytes());
                largest_weight_tile = largest_weight_tile.max(linear.weight_tile_bytes());
                largest_scratch_grant =
                    largest_scratch_grant.max(linear.workspace_allocation_grant_bytes().unwrap());
            }
            *numeric_roles.entry(format!("{:?}", p.role())).or_default() += 1;
        } else if let [n, k] = p.weight().slice().shape.as_slice()
            && !matches!(
                p.role(),
                ferrule_model::TensorRole::TokenEmbedding
                    | ferrule_model::TensorRole::LinearAttentionConv
            )
        {
            use ferrule_backend::cuda::operators::linear::{
                Bf16GemmLayout, bf16_gemm_workspace_requirements,
            };
            assert_eq!(p.weight().slice().dtype, CheckpointDType::Bf16);
            for rows in [1, limits.max_batch_tokens] {
                assert_eq!(
                    bf16_gemm_workspace_requirements(Bf16GemmLayout::contiguous(rows, *n, *k))
                        .unwrap()
                        .bytes,
                    0
                );
                // Same conservative BF16+F32 temporary charge as bf16_linear_rows.
                native_linear_temporary_bytes =
                    native_linear_temporary_bytes.max((n + k) * rows * 8);
            }
            *native_roles.entry(format!("{:?}", p.role())).or_default() += 1;
        }
    }
    assert_eq!(native_roles.get("RouterLogits"), Some(&40));
    assert_eq!(native_roles.get("OutputHead"), Some(&1));
    assert!(native_linear_temporary_bytes + limits.scratch_bytes <= limits.max_bytes);
    eprintln!(
        "BF16_NUMERIC_PROJECTION_PREFLIGHT {}",
        json!({"execution_profile":plan.backend_profile(),"numeric_roles":numeric_roles,"native_roles":native_roles,"native_linear_temporary_bytes":native_linear_temporary_bytes,"largest_activation_bytes":largest_activation,"largest_weight_tile_bytes":largest_weight_tile,"largest_scratch_grant_bytes":largest_scratch_grant,"provider_reserve_bytes":provider_reserve_bytes})
    );
    assert!(largest_scratch_grant <= limits.scratch_bytes);
    let public = prepare_full40_with_config(&model, precision, test_config);
    let effective_config = full40_effective_config(&public, test_config, precision);
    assert_eq!(public.backend_profile(), precision.hybrid_backend_name());
    assert_eq!(
        public
            .numeric_bf16_memory_plan()
            .unwrap()
            .unwrap()
            .model_budget(),
        plan.model_budget()
    );
    public
        .build_with(
            |request: ModelBuildRequest| -> ferrule_runtime::Result<()> {
                assert_eq!(
                    request.resolved.execution.numeric_bf16_precision,
                    Some(precision)
                );
                assert_eq!(request.resolved.scheduler_config, test_config.scheduler());
                assert_eq!(
                    qwen35::capacity_limits(&request.resolved).unwrap(),
                    test_config.capacity(precision)
                );
                assert_eq!(
                    request.resolved.execution.kv_execution_policy,
                    ferrule_runtime::engine::KvExecutionPolicy::Resident
                );
                Ok(())
            },
        )
        .unwrap();
    let estimate = plan.model_estimate();
    if precision == NumericFp8Precision::Bf16ModelMixedF32Accumulate {
        assert_eq!(
            provider_reserve_bytes,
            ferrule_backend::cuda_plan::NUMERIC_BF16_GEMV_PROVIDER_BYTES
        );
        let tiled = Qwen35NumericBf16MemoryPlan::open_hf_with_precision(
            &model,
            ModelPrecision::Bf16,
            NumericStoragePolicy::E4M3FnBf16Scales,
            limits,
            1 << 30,
            NumericFp8Precision::Bf16ModelTiledF32Accumulate,
        )
        .unwrap();
        let (projection, external) = ferrule_model::decoder::HybridMemoryPlan::bf16_gemmex_bytes(
            plan.resources(),
            plan.runner_options(),
        )
        .unwrap();
        let attention = ferrule_model::decoder::HybridMemoryPlan::bf16_attention_plan(
            plan.resources(),
            plan.runner_options(),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            estimate.workspace_bytes_upper_bound,
            tiled.model_estimate().workspace_bytes_upper_bound
                + ferrule_backend::cuda_allocation_grant_bytes(projection).unwrap()
                + external
                + attention.total_charge_bytes()
                + ferrule_model::decoder::HybridMemoryPlan::bf16_attention_metadata_bytes(
                    plan.resources(),
                    plan.runner_options()
                )
                .unwrap()
                + ferrule_backend::cuda_allocation_grant_bytes(
                    ferrule_model::decoder::HybridMemoryPlan::bf16_chunk_workspace_bytes(
                        plan.resources(),
                        plan.runner_options()
                    )
                    .unwrap(),
                )
                .unwrap()
        );
    }
    let budget = plan.model_budget();
    assert_eq!(plan.resources().spec().layers().len(), 40);
    let mut unique = std::collections::BTreeSet::new();
    let (mut raw_fp8_bytes, mut scale_bytes, mut native_bf16_bytes, mut cast_gdn_bytes) =
        (0usize, 0usize, 0usize, 0usize);
    for p in plan.resources().state_dict().parameters() {
        if matches!(p.residency(), ParameterResidency::Expert { .. })
            || !unique.insert(p.canonical_id())
        {
            continue;
        }
        let bytes = usize::try_from(p.weight().slice().bytes).unwrap();
        match p.weight().slice().dtype {
            CheckpointDType::F8E4M3 => raw_fp8_bytes += bytes,
            CheckpointDType::Bf16 => native_bf16_bytes += bytes,
            CheckpointDType::F32 => cast_gdn_bytes += bytes / 2,
            ref other => panic!("unexpected resident source dtype {other:?}"),
        }
        scale_bytes += usize::try_from(p.scale().map_or(0, |scale| scale.slice().bytes)).unwrap();
    }
    let rope_bytes: usize = plan
        .resources()
        .spec()
        .layers()
        .iter()
        .filter_map(|layer| {
            if let Attention::Gqa(g) = layer.attention() {
                Some(limits.max_positions * g.rotary().region().dimensions() * 2)
            } else {
                None
            }
        })
        .sum();
    let bounded_cache_bytes = limits.max_bytes - limits.scratch_bytes;
    assert_eq!(
        budget.weight_bytes,
        raw_fp8_bytes
            + scale_bytes
            + native_bf16_bytes
            + cast_gdn_bytes
            + rope_bytes
            + bounded_cache_bytes
    );
    assert_eq!(
        budget.workspace_bytes,
        limits.workspace_bytes + limits.scratch_bytes
    );
    assert!(estimate.workspace_bytes_upper_bound <= budget.workspace_bytes);
    let state_copies = 2 * limits.max_sequences + 1;
    assert_eq!(state_copies, 5);
    assert_eq!(
        budget.state_bytes,
        state_copies * estimate.per_sequence_state_bytes
    );
    assert_eq!(budget.kv_pages, 32);
    assert_eq!(estimate.kv_bytes, 10 << 20);
    let required_total = budget.weight_bytes
        + budget.workspace_bytes
        + budget.state_bytes
        + estimate.kv_bytes
        + limits.allocator_margin_bytes;
    assert_eq!(plan.total_device_bytes(), required_total);
    assert!(required_total <= limits.device_bytes);
    assert_eq!(
        limits.allocator_margin_bytes,
        Qwen35MoeCapacityLimits::default().allocator_margin_bytes
    );
    eprintln!(
        "BF16_FULL40_METADATA_ONLY {}",
        json!({"test_config":test_config.manifest(precision),"effective_config":effective_config,
            "observer_mode":"metadata_only_no_owner","ordinary_no_observer_claim":false,
            "raw_fp8_nonexpert_bytes":raw_fp8_bytes,"paired_scale_bytes":scale_bytes,
            "native_bf16_bytes":native_bf16_bytes,"cast_gdn_bf16_bytes":cast_gdn_bytes,
            "rope_bytes":rope_bytes,"bounded_cache_bytes":bounded_cache_bytes,
            "weight_requirement_bytes":budget.weight_bytes,"workspace_requirement_bytes":estimate.workspace_bytes_upper_bound,
            "workspace_grant_bytes":budget.workspace_bytes,"per_sequence_state_bytes":estimate.per_sequence_state_bytes,
            "state_copies":state_copies,"state_grant_bytes":budget.state_bytes,
            "kv_pages":budget.kv_pages,"kv_requirement_bytes":estimate.kv_bytes,
            "allocator_margin_bytes":limits.allocator_margin_bytes,"required_total_bytes":required_total,
            "device_cap_bytes":limits.device_bytes,"cap_headroom_bytes":limits.device_bytes-required_total,
            "layers":40,"context":limits.max_positions,"batch":limits.max_batch_tokens,"sequences":limits.max_sequences
        })
    );
    let old = Qwen35MoeCapacityLimits {
        device_bytes: 6400 << 20,
        ..limits
    };
    let error = Qwen35NumericBf16MemoryPlan::open_hf_with_precision(
        &model,
        ModelPrecision::Bf16,
        NumericStoragePolicy::E4M3FnBf16Scales,
        old,
        1 << 30,
        precision,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("image plus allocator margin exceeds device byte cap")
    );
    // The lifecycle's larger KV ceiling did not buy more configured pages:
    // context/sequence geometry, not the byte ceiling, determines this image.
    let lifecycle_limits = Qwen35MoeCapacityLimits {
        kv_bytes: 1 << 30,
        device_bytes: 16 << 30,
        ..limits
    };
    let lifecycle = Qwen35NumericBf16MemoryPlan::open_hf_with_precision(
        &model,
        ModelPrecision::Bf16,
        NumericStoragePolicy::E4M3FnBf16Scales,
        lifecycle_limits,
        1 << 30,
        precision,
    )
    .unwrap();
    assert_eq!(lifecycle.model_budget(), budget);
    assert_eq!(lifecycle.total_device_bytes(), required_total);
}

#[path = "bf16_boundary.rs"]
mod boundary;

#[path = "bf16_fresh_quality/mod.rs"]
mod fresh_quality;
