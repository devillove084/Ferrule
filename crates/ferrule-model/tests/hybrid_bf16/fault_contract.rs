//! Component-only evidence contracts. CPU tests exercise reporting/accounting,
//! never produce an observed CUDA outcome.
use super::model;
#[path = "../support/expert_read_worker_contract.rs"]
pub mod worker_contract;

use super::model::{
    ModelFamily, WeightSource,
    decoder::GenericDecoderOptions,
    execution::ExecutionPrecisionPolicy,
    transformer::{
        BoundDecoderResources, ExpertCacheLimits, ExpertCacheStats, ExpertReadHostCharge,
        ExpertReadPipelineOptions, ExpertReadSchedule, NumericFp8Precision,
    },
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs::File, io::Write, path::Path};

pub const REPORT_ENV: &str = "FERRULE_COMPONENT_FAULT_REPORT_DIR";
pub const CACHE_BYTES: usize = 64 << 10;
pub const HOST_BYTES: u64 = 2 << 20;
pub const HOST_METADATA_BYTES: u64 = 256 << 10;
pub const HOST_STACK_BYTES: usize = 1 << 20;
pub const PRECISION: NumericFp8Precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;

pub fn options(
    resources: &BoundDecoderResources,
    schedule: ExpertReadSchedule,
) -> GenericDecoderOptions {
    GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Unknown("component-mixed-fault-fixture".into()),
        WeightSource::Safetensors,
        2,
        32,
        8,
        1,
        1 << 20,
        ExecutionPrecisionPolicy::bf16_compatibility(),
    )
    .unwrap()
    .with_hybrid_cuda_numeric_fp8_bf16_precision(
        ExpertCacheLimits {
            max_experts: 1,
            max_bytes: CACHE_BYTES,
        },
        4096,
        PRECISION,
    )
    .unwrap()
    .with_expert_read_pipeline(ExpertReadPipelineOptions {
        schedule,
        max_host_bytes: HOST_BYTES,
        max_metadata_bytes: HOST_METADATA_BYTES,
        worker_stack_bytes: HOST_STACK_BYTES,
    })
    .unwrap()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultKind {
    Clean,
    Unknown,
}
impl FaultKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Clean => "mixed_clean",
            Self::Unknown => "mixed_unknown",
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct CacheLedger {
    pub scope: &'static str,
    pub budget: usize,
    pub free: usize,
    pub resident: usize,
    pub pending: usize,
    pub scratch: usize,
    pub charged: usize,
    pub unknown_subset: usize,
}
impl CacheLedger {
    pub fn checked(stats: ExpertCacheStats, budget: usize) -> Option<Self> {
        let charged = stats
            .resident_bytes
            .checked_add(stats.pending_upload_bytes)?
            .checked_add(stats.scratch_bytes)?;
        if stats.unknown_quarantine_bytes > charged {
            return None;
        }
        Some(Self {
            scope: "model_expert_cache_logical_bytes_not_runtime_broker_or_allocator_grants",
            budget,
            free: budget.checked_sub(charged)?,
            resident: stats.resident_bytes,
            pending: stats.pending_upload_bytes,
            scratch: stats.scratch_bytes,
            charged,
            unknown_subset: stats.unknown_quarantine_bytes,
        })
    }
}

pub fn host_ledger_valid(charge: ExpertReadHostCharge, budget: u64) -> bool {
    let payload = charge.packed_bytes.checked_add(charge.artifact_bytes);
    let charged = payload
        .and_then(|n| n.checked_add(charge.metadata_bytes))
        .and_then(|n| n.checked_add(charge.requested_stack_bytes));
    payload == Some(charge.payload_bytes)
        && charged == Some(charge.host_charged_bytes)
        && charge.host_charged_bytes <= budget
        && charge.unknown_packed_bytes <= charge.packed_bytes
        && charge.unknown_stack_bytes <= charge.requested_stack_bytes
        && charge.unknown_read_slots <= charge.reserved_read_slots
}

pub fn header(kind: FaultKind) -> Value {
    let lifecycle = json!({
        "host_lifecycle_contract_revision": 2,
        "host_charge_basis": "logical_host_reservation_not_physical_rss",
        "worker_counter_scope": "full_owner_lifetime_created_reuses_joined_not_submitted_reads",
        "host_stages": {
            "sequence_end": "wait_take_release_payload_keep_worker",
            "idle_payload_artifact_read_slots": 0,
            "idle_metadata_plus_requested_stack_bytes": HOST_METADATA_BYTES + HOST_STACK_BYTES as u64,
            "final_owner_drop": "zero_total_only_after_materializer_drop_and_observed_offworker_join",
            "device_Unknown": "retain_device_runner_and_states; healthy_host_worker_may_remain_idle_until_owner_drop_is_proved"
        }
    });
    let mut header = json!({
        "schema": "ferrule.component-pressure-fault.v1",
        "record": "header",
        "case": kind.name(),
        "gpu_execution": "unobserved",
        "experimental_report_directory_env": REPORT_ENV,
        "scope": "tiny_actual_Mixed_CudaHybridModule_execution_boundary_component",
        "production_cli_E5_integrated": false,
        "full_model_35B_fault_guarantee": false,
        "runtime_broker_five_bucket_equation": "unobserved_not_present_in_fixture",
        "config": {
            "precision": "Bf16ModelMixedF32Accumulate",
            "fixture": "existing_hybrid_bf16_numeric_profile",
            "seed": 3501635,
            "hidden": 16, "vocab": 32, "layers": ["GatedDeltaNet", "GQA"],
            "experts_per_layer": 4, "top_k": 3, "expert_intermediate": 16,
            "history_dtype": "BF16", "ssm_dtype": "F32", "kv_dtype": "BF16",
            "weights": "tiny_deterministic_numeric_FP8_with_BF16_scales_not_SOTA",
            "cache_entries": 1, "cache_bytes": CACHE_BYTES,
            "host_bytes": HOST_BYTES, "host_metadata_bytes": HOST_METADATA_BYTES,
            "host_worker_stack_bytes": HOST_STACK_BYTES, "numeric_scratch_reservation": 4096,
            "schedules": ["blocking", "lookahead"],
            "prime_tokens": [1], "fault_continuation_tokens": [2, 3],
            "fault_boundary": "layer_1_first_routed_down_output_before_consumer_proof",
            "trigger": "scheduled_test_error_after_real_submit_not_hardware_failure",
            "unknown_trigger": "discard_real_consumer_proof_after_wait_not_forced_hang",
            "observation": "passive_snapshots_physical_actions_logged_separately",
            "timing_or_overlap_claim": false,
            "input_privacy": "checked_in_known_seed_fixture_only_no_user_prompt_or_weights_export"
        }
    });
    header
        .as_object_mut()
        .unwrap()
        .extend(lifecycle.as_object().unwrap().clone());
    header
}

/// Append-only, create-new JSONL. An interrupted run has no seal. Constructing
/// a report or compiling an ignored selector never labels GPU work as observed.
pub struct Report {
    file: Option<File>,
    schedules: BTreeSet<&'static str>,
}
impl Report {
    pub fn new(kind: FaultKind, directory: Option<&Path>) -> std::io::Result<Self> {
        let file = directory
            .map(|dir| {
                File::options()
                    .write(true)
                    .create_new(true)
                    .open(dir.join(format!("{}.jsonl", kind.name())))
            })
            .transpose()?;
        let mut report = Self {
            file,
            schedules: BTreeSet::new(),
        };
        report.record(header(kind))?;
        Ok(report)
    }
    pub fn record(&mut self, value: Value) -> std::io::Result<()> {
        let line = serde_json::to_string(&value)?;
        if let Some(file) = &mut self.file {
            writeln!(file, "{line}")?;
            file.flush()?;
        }
        eprintln!("COMPONENT_FAULT {line}");
        Ok(())
    }
    pub fn observed_case(
        &mut self,
        schedule: &'static str,
        evidence: Value,
    ) -> std::io::Result<()> {
        if !matches!(schedule, "blocking" | "lookahead")
            || self.schedules.contains(schedule)
            || evidence["real_compute_launch_delta"].as_u64().unwrap_or(0) == 0
            || evidence["real_h2d_copy_delta"].as_u64().unwrap_or(0) == 0
            || evidence["real_read_byte_delta"].as_u64().unwrap_or(0) == 0
            || evidence["faults_observed"].as_u64() != Some(1)
            || evidence["committed_frontier_before"].as_u64() != Some(1)
            || evidence["committed_frontier_after"].as_u64() != Some(1)
        {
            return Err(std::io::Error::other(
                "missing or duplicate component execution evidence",
            ));
        }
        self.record(json!({"record": "case_outcome", "schedule": schedule,
            "gpu_execution": "observed", "status": "assertions_passed", "evidence": evidence}))?;
        self.schedules.insert(schedule);
        Ok(())
    }
    pub fn seal(mut self) -> std::io::Result<()> {
        if self.schedules.len() != 2 {
            return Err(std::io::Error::other(
                "both real execution schedules required before seal",
            ));
        }
        self.record(json!({"record": "seal", "scope": "component_only",
            "gpu_execution": "observed", "completed_cells": 2, "status": "assertions_passed"}))?;
        if let Some(file) = self.file {
            file.sync_all()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_gpu_fixture_options_admit_on_cpu_with_identical_bounded_mixed_plans() {
        use super::super::model::decoder::HybridMemoryPlan;
        let fixture = super::super::fixture::Fixture::numeric_profile();
        let blocking = options(&fixture.resources, ExpertReadSchedule::Blocking);
        let lookahead = options(&fixture.resources, ExpertReadSchedule::Lookahead);
        let plan = HybridMemoryPlan::for_resources(&fixture.resources, &blocking, 32).unwrap();
        assert_eq!(
            plan,
            HybridMemoryPlan::for_resources(&fixture.resources, &lookahead, 32).unwrap()
        );
        assert_eq!(plan.per_sequence_state_bytes, 512);
        assert!(
            HybridMemoryPlan::bf16_chunk_workspace_bytes(&fixture.resources, &blocking).unwrap()
                > 0
        );
        for (options, schedule) in [
            (blocking, ExpertReadSchedule::Blocking),
            (lookahead, ExpertReadSchedule::Lookahead),
        ] {
            assert_eq!(options.hybrid_cuda_numeric_fp8_precision(), Some(PRECISION));
            assert_eq!(
                options.expert_read_pipeline_options().unwrap().schedule,
                schedule
            );
        }
        assert_eq!(header(FaultKind::Clean)["gpu_execution"], "unobserved");
    }

    #[test]
    fn unknown_is_a_subset_not_a_fifth_cache_bucket() {
        let stats = ExpertCacheStats {
            resident_bytes: 7,
            pending_upload_bytes: 3,
            scratch_bytes: 5,
            unknown_quarantine_bytes: 15,
            quarantined: true,
            ..Default::default()
        };
        let ledger = CacheLedger::checked(stats, 20).unwrap();
        assert_eq!(ledger.budget, ledger.free + ledger.charged);
        assert_eq!(
            ledger.charged,
            ledger.resident + ledger.pending + ledger.scratch
        );
        assert_eq!(ledger.unknown_subset, ledger.charged);
        assert!(CacheLedger::checked(stats, 14).is_none());
        assert!(
            CacheLedger::checked(
                ExpertCacheStats {
                    unknown_quarantine_bytes: 16,
                    ..stats
                },
                20
            )
            .is_none()
        );
        assert!(
            CacheLedger::checked(
                ExpertCacheStats {
                    resident_bytes: usize::MAX,
                    ..stats
                },
                usize::MAX
            )
            .is_none()
        );
    }

    #[test]
    fn host_only_conservation_does_not_imply_device_release() {
        assert!(host_ledger_valid(
            ExpertReadHostCharge::default(),
            HOST_BYTES
        ));
        let charge = ExpertReadHostCharge {
            packed_bytes: 8,
            artifact_bytes: 8,
            payload_bytes: 16,
            metadata_bytes: 4,
            requested_stack_bytes: 64,
            host_charged_bytes: 84,
            ..Default::default()
        };
        assert!(host_ledger_valid(charge, 84));
        assert!(!host_ledger_valid(charge, 83));
        assert!(!host_ledger_valid(
            ExpertReadHostCharge {
                payload_bytes: 0,
                ..charge
            },
            84
        ));
    }

    #[test]
    fn retained_worker_stage_requires_exact_charge_and_full_owner_counts() {
        use super::model::transformer::{ExpertReadPipelineObservation, ExpertReadPipelineStats};
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = ExpertReadPipelineOptions {
                schedule,
                max_host_bytes: HOST_BYTES,
                max_metadata_bytes: HOST_METADATA_BYTES,
                worker_stack_bytes: HOST_STACK_BYTES,
            };
            let idle = ExpertReadPipelineObservation {
                enabled: true,
                requested_mode: Some(schedule),
                effective_mode: Some(schedule),
                limits: Some(read),
                status: "idle_retained",
                worker_alive: true,
                worker_read_observed: true,
                current_charge: ExpertReadHostCharge {
                    metadata_bytes: HOST_METADATA_BYTES,
                    requested_stack_bytes: HOST_STACK_BYTES as u64,
                    host_charged_bytes: 1280 << 10,
                    ..Default::default()
                },
                stats: ExpertReadPipelineStats {
                    sequences_started: 2,
                    sequences_finished: 2,
                    workers_created: 1,
                    worker_reuses: 1,
                    submitted_reads: 6,
                    completed_reads: 6,
                    verified_reads: 6,
                    worker_started_reads: 6,
                    io_started_reads: 6,
                    proved_read_bytes: 6 * 128,
                    peak_metadata_bytes: HOST_METADATA_BYTES,
                    peak_requested_stack_bytes: HOST_STACK_BYTES as u64,
                    peak_host_charged_bytes: 1280 << 10,
                    ..Default::default()
                },
                ..Default::default()
            };
            worker_contract::assert_idle_retained(&idle, read);
            for field in [
                "zero_total",
                "payload",
                "slot",
                "unknown",
                "join",
                "worker_per_read",
            ] {
                let mut bad = idle;
                match field {
                    "zero_total" => bad.current_charge = Default::default(),
                    "payload" => {
                        bad.current_charge.artifact_bytes = 1;
                        bad.current_charge.payload_bytes = 1;
                        bad.current_charge.host_charged_bytes += 1;
                    }
                    "slot" => bad.current_charge.reserved_read_slots = 1,
                    "unknown" => bad.worker_unknown = true,
                    "join" => bad.stats.workers_joined = 1,
                    "worker_per_read" => bad.stats.workers_created = 6,
                    _ => unreachable!(),
                }
                assert!(
                    std::panic::catch_unwind(|| worker_contract::assert_idle_retained(&bad, read))
                        .is_err(),
                    "{field}"
                );
            }
            assert_eq!(
                header(FaultKind::Unknown)["host_stages"]["idle_metadata_plus_requested_stack_bytes"],
                1280 << 10
            );
            assert_eq!(header(FaultKind::Unknown)["gpu_execution"], "unobserved");
        }
    }

    #[test]
    fn cpu_report_is_unobserved_and_cannot_seal_missing_gpu_evidence() {
        for kind in [FaultKind::Clean, FaultKind::Unknown] {
            assert_eq!(header(kind)["gpu_execution"], "unobserved");
            assert_eq!(header(kind)["production_cli_E5_integrated"], false);
            let mut report = Report::new(kind, None).unwrap();
            assert!(report.observed_case("blocking", json!({})).is_err());
            assert!(
                report
                    .observed_case("lookahead", json!({"real_compute_launch_delta": 1}))
                    .is_err()
            );
            assert!(report.seal().is_err());
        }
    }

    #[test]
    fn reporter_create_new_never_overwrites_or_seals_cpu_header() {
        let dir = std::env::temp_dir().join(format!("ferrule-fault-report-{}", std::process::id()));
        std::fs::create_dir(&dir).unwrap();
        let report = Report::new(FaultKind::Clean, Some(&dir)).unwrap();
        assert!(Report::new(FaultKind::Clean, Some(&dir)).is_err());
        assert!(report.seal().is_err());
        let text = std::fs::read_to_string(dir.join("mixed_clean.jsonl")).unwrap();
        let records: Vec<Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records, vec![header(FaultKind::Clean)]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
