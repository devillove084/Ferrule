//! Read-only quality receipts from the existing physical owner; never a worker driver.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]
use ferrule_model::resource_observability::{PhysicalResourceObservation, ResourceEvidence};
use ferrule_model::transformer::{ExpertReadPipelineObservation, ExpertReadPipelineOptions};
use serde_json::{Value, json};
use std::collections::BTreeMap;

use crate::model_factory::full40_strict::oracle::StrictReport;

pub(crate) fn observer_fields(diagnostic: bool) -> Value {
    json!({"observer_mode": if diagnostic {"diagnostic_routes_and_states"} else {"ordinary_no_diagnostic_observer"},
        "diagnostic_trace_installed":diagnostic,"routes_observed":diagnostic,"states_observed":diagnostic,
        "ordinary_no_observer_claim":!diagnostic,"read_only_resource_snapshots":true,
        "gpu_overlap_claim":false,"concurrent_scheduler_claim":false})
}

fn configuration_matches(
    value: &ExpertReadPipelineObservation,
    requested: ExpertReadPipelineOptions,
) -> bool {
    value.enabled
        && value.requested_mode == Some(requested.schedule)
        && value.limits == Some(requested)
        && value
            .effective_mode
            .is_none_or(|mode| mode == requested.schedule)
}

fn bounded(value: &ExpertReadPipelineObservation, requested: ExpertReadPipelineOptions) -> bool {
    let payload = requested.max_host_bytes
        - requested.max_metadata_bytes
        - requested.worker_stack_bytes as u64;
    let c = value.current_charge;
    let s = value.stats;
    c.packed_bytes <= payload / 2
        && c.artifact_bytes <= payload / 2
        && c.payload_bytes <= payload
        && c.metadata_bytes <= requested.max_metadata_bytes
        && c.requested_stack_bytes <= requested.worker_stack_bytes as u64
        && c.host_charged_bytes <= requested.max_host_bytes
        && c.reserved_read_slots <= 2
        && s.peak_packed_bytes <= payload / 2
        && s.peak_artifact_bytes <= payload / 2
        && s.peak_payload_bytes <= payload
        && s.peak_metadata_bytes <= requested.max_metadata_bytes
        && s.peak_requested_stack_bytes <= requested.worker_stack_bytes as u64
        && s.peak_host_charged_bytes <= requested.max_host_bytes
        && c.packed_bytes <= s.peak_packed_bytes
        && c.artifact_bytes <= s.peak_artifact_bytes
        && c.payload_bytes <= s.peak_payload_bytes
        && c.metadata_bytes <= s.peak_metadata_bytes
        && c.requested_stack_bytes <= s.peak_requested_stack_bytes
        && c.host_charged_bytes <= s.peak_host_charged_bytes
        && (s.workers_created == 0
            || (s.peak_metadata_bytes == requested.max_metadata_bytes
                && s.peak_requested_stack_bytes == requested.worker_stack_bytes as u64
                && s.peak_host_charged_bytes
                    >= requested.max_metadata_bytes + requested.worker_stack_bytes as u64))
}

// This is an endpoint contract, not a worker driver. Keep the producer's
// statuses explicit: newly introduced statuses must not silently qualify.
#[derive(Clone, Copy)]
enum SettledWorkerStatus {
    Configured,
    IdleRetained,
    Closed,
}
impl SettledWorkerStatus {
    fn parse(status: &str) -> Result<Self, &'static str> {
        match status {
            "configured" => Ok(Self::Configured),
            "idle_retained" => Ok(Self::IdleRetained),
            "closed" => Ok(Self::Closed),
            // The materializer getter sets worker_unknown even when every
            // unknown-byte gauge is zero after observation-lock poisoning.
            "unknown_observation_poisoned" => Err("observation_ledger_poisoned"),
            // Budget, join or admission-lock poison; retained bytes may be zero.
            "unknown_ledger_poisoned" => Err("worker_or_admission_ledger_poisoned"),
            // Admission quarantine, lost worker custody, unknown slots/stack.
            "quarantined" => Err("quarantined_custody"),
            "join_pending_at_scope_end" => Err("worker_join_pending"),
            "active" => Err("read_scope_still_active"),
            "closed_retained" => Err("payload_custody_still_retained"),
            "disabled" => Err("pipeline_disabled"),
            _ => Err("unsupported_worker_status"),
        }
    }
}

fn worker_snapshot(value: &ExpertReadPipelineObservation) -> Result<(), &'static str> {
    use ferrule_model::transformer::{ExpertReadHostCharge, ExpertReadPipelineStats};
    let status = SettledWorkerStatus::parse(value.status)?;
    // Also covers join failure / custody loss without an unknown byte or ticket.
    if value.worker_unknown {
        return Err("worker_unknown");
    }
    if value.worker_join_pending {
        return Err("worker_join_pending");
    }
    let s = value.stats;
    if s.workers_created.checked_sub(s.workers_joined) != Some(u64::from(value.worker_alive))
        || s.workers_created.checked_add(s.worker_reuses) != Some(s.sequences_started)
        || s.sequences_started != s.sequences_finished
        || s.join_failed_sequences != 0
    {
        return Err("inconsistent_worker_lifetime");
    }
    if value.current_read.is_some() || value.pending_read.is_some() {
        return Err("read_identity_still_retained");
    }
    if s.submitted_reads != s.completed_reads
        || s.verified_reads
            .checked_add(s.cancelled_reads)
            .and_then(|n| n.checked_add(s.failed_reads))
            != Some(s.completed_reads)
        || s.worker_started_reads > s.submitted_reads
        || s.io_started_reads > s.worker_started_reads
        || s.verified_reads > s.io_started_reads
        || value.worker_read_observed != (s.io_started_reads != 0 && s.proved_read_bytes != 0)
    {
        return Err("inconsistent_read_lifetime");
    }
    let expected_charge = match status {
        SettledWorkerStatus::Configured => {
            if value.effective_mode.is_some()
                || value.last_receipt.is_some()
                || s != ExpertReadPipelineStats::default()
            {
                return Err("configured_but_already_executed");
            }
            ExpertReadHostCharge::default()
        }
        SettledWorkerStatus::IdleRetained | SettledWorkerStatus::Closed => {
            if s.workers_created == 0 || value.effective_mode != value.requested_mode {
                return Err("missing_effective_worker_lifetime");
            }
            let idle = matches!(status, SettledWorkerStatus::IdleRetained);
            if value.worker_alive != idle {
                return Err("worker_status_liveness_mismatch");
            }
            if idle {
                let limits = value.limits.ok_or("missing_worker_limits")?;
                ExpertReadHostCharge {
                    metadata_bytes: limits.max_metadata_bytes,
                    requested_stack_bytes: limits.worker_stack_bytes as u64,
                    host_charged_bytes: limits
                        .max_metadata_bytes
                        .checked_add(limits.worker_stack_bytes as u64)
                        .ok_or("worker_charge_overflow")?,
                    ..Default::default()
                }
            } else {
                ExpertReadHostCharge::default()
            }
        }
    };
    if value.current_charge != expected_charge {
        return Err("worker_stage_charge_mismatch");
    }
    Ok(())
}

fn unambiguous(value: &ExpertReadPipelineObservation) -> bool {
    !value.worker_unknown
        && value.stats.unknown_reads == 0
        && value.stats.unknown_bytes == 0
        && value.stats.unobserved_terminal_reads == 0
        && value.stats.unobserved_reader_intervals == 0
        && value.current_charge.unknown_packed_bytes == 0
        && value.current_charge.unknown_stack_bytes == 0
        && value.current_charge.unknown_read_slots == 0
        && worker_snapshot(value).is_ok()
}

fn snapshot_json(value: &PhysicalResourceObservation) -> Value {
    serde_json::to_value(value).unwrap()
}

pub(crate) struct ReadQualityLog {
    requested: ExpertReadPipelineOptions,
    diagnostic: bool,
    pub(crate) calls: Vec<Value>,
}
impl ReadQualityLog {
    pub(crate) fn new(requested: ExpertReadPipelineOptions, diagnostic: bool) -> Self {
        requested.validate().unwrap();
        Self {
            requested,
            diagnostic,
            calls: Vec::new(),
        }
    }

    pub(crate) fn begin(
        &mut self,
        label: &str,
        operation: &str,
        before: PhysicalResourceObservation,
    ) -> usize {
        let index = self.calls.len();
        self.calls.push(
            json!({"label":label,"operation":operation,"requested":self.requested,
            "before":snapshot_json(&before),"after":null,"delta":null,"same_owner":null,
            "configuration_matches":null,"within_charged_limits":null,"unambiguous":null,
            "configuration_pair":{"before":null,"after":null},
            "positive_worker_read":null,"evidence_complete":false}),
        );
        index
    }

    pub(crate) fn finish(
        &mut self,
        index: usize,
        before: &PhysicalResourceObservation,
        after: PhysicalResourceObservation,
    ) {
        let same_owner = match (
            before.scope.value(),
            after.scope.value(),
            before.prepared_plan_id.value(),
            after.prepared_plan_id.value(),
            before.image_generation.value(),
            after.image_generation.value(),
        ) {
            (Some(a), Some(b), Some(p), Some(q), Some(i), Some(j)) => {
                Some(a == b && p == q && i == j)
            }
            _ => None,
        };
        let before_observed = before.expert_read_pipeline.value();
        let observed = after.expert_read_pipeline.value();
        let pair = before_observed.zip(observed);
        let delta = pair
            .filter(|_| same_owner == Some(true))
            .and_then(|(a, b)| b.stats.checked_delta(a.stats));
        let requested = self.requested;
        let record = &mut self.calls[index];
        record["after"] = snapshot_json(&after);
        record["same_owner"] = json!(same_owner);
        record["delta"] = json!(delta);
        record["delta_scope"] = json!(
            "owner counters only; peak fields remain lifetime gauges; reader intervals not certified per-ticket syscalls"
        );
        // Keep v1 aggregate flag names, but qualify the entire interval. Missing
        // endpoint evidence remains null; configuration_pair identifies which
        // endpoint failed without changing counter-delta or worker-read evidence.
        record["configuration_matches"] = json!(pair.map(|(a, b)| configuration_matches(
            a, requested
        ) && configuration_matches(
            b, requested
        )));
        record["within_charged_limits"] =
            json!(pair.map(|(a, b)| bounded(a, requested) && bounded(b, requested)));
        record["unambiguous"] = json!(pair.map(|(a, b)| unambiguous(a) && unambiguous(b)));
        let endpoint = |value: Option<&ExpertReadPipelineObservation>| {
            json!({
                "configuration_matches":value.map(|o|configuration_matches(o,requested)),
                "within_charged_limits":value.map(|o|bounded(o,requested)),
                "unambiguous":value.map(unambiguous),
                "worker_snapshot_valid":value.map(|o| worker_snapshot(o).is_ok()),
                "worker_snapshot_rejection":value.and_then(|o| worker_snapshot(o).err()),
            })
        };
        record["configuration_pair"] =
            json!({"before":endpoint(before_observed),"after":endpoint(observed)});
        record["positive_worker_read"] = json!(delta.map(|s| s.submitted_reads > 0
            && s.completed_reads > 0
            && s.verified_reads > 0
            && s.io_started_reads > 0
            && s.proved_read_bytes > 0
            && observed.is_some_and(
                |o| o.worker_read_observed && o.effective_mode == Some(requested.schedule)
            )));

        record["evidence_complete"] = json!(delta.is_some() && same_owner == Some(true));
    }

    pub(crate) fn passed(&self) -> bool {
        !self.calls.is_empty()
            && self.calls.iter().any(|r| r["positive_worker_read"] == true)
            && self.calls.iter().all(|r| {
                r["evidence_complete"] == true
                    && r["configuration_matches"] == true
                    && r["within_charged_limits"] == true
                    && r["unambiguous"] == true
            })
    }

    pub(crate) fn json(&self) -> Value {
        json!({"schema":"ferrule.expert-read-quality-evidence.v1","interval_validation_revision":3,
            "worker_snapshot_contract_revision":1,
            "worker_snapshot_contract":"typed current producer fields required; both observed endpoints; only configured (unexecuted), idle_retained (one unjoined live worker, exact metadata plus requested stack), or closed (joined, no charge); no payloads/read slots at endpoints; created/reuses/joins are monotonic full-owner-lifetime counters, not read counts; missing endpoint evidence is null",
            "worker_unknown_contract":"worker_unknown is independently disqualifying, including zero-byte join failure/custody loss; unknown_observation_poisoned means observation-lock poison; unknown_ledger_poisoned means budget/join/admission-lock poison; quarantined means admission quarantine, custody loss or unknown slots/stack; unsupported statuses fail closed",
            "cold_report_finish_contract":"finishing a report is not materializer Drop; an idle worker retains logical metadata/stack, not measured physical RSS",
            "interval_validation_contract":"aggregate configuration_matches, within_charged_limits and unambiguous require both observed endpoints; configuration_pair records each endpoint; missing evidence is null; initial effective_mode may be null",
            "requested":self.requested,
            "observation_scope":observer_fields(self.diagnostic),"passed":self.passed(),
            "positive_worker_read_calls":self.calls.iter().filter(|r| r["positive_worker_read"] == true).count(),
            "sampled_operations":self.calls.len(),"calls":self.calls,
            "no_poll_join_drain_or_cuda_sync":true,"charge_basis":"logical_host_reservation_not_physical_rss",
            "no_new_resource_authority_or_first_source_identity_inference":true})
    }
}

#[derive(Default)]
struct PredictionStream {
    calls: usize,
    rows: usize,
    values: usize,
    greedy_checks: usize,
    history_matched_calls: usize,
    report: StrictReport,
}

pub(crate) struct PredictionSample<'a> {
    pub(crate) rows: usize,
    pub(crate) actual: &'a [f32],
    pub(crate) expected: &'a [f32],
    pub(crate) actual_token: u32,
    pub(crate) expected_token: u32,
    pub(crate) history_matches: bool,
    pub(crate) label: &'a str,
}

pub(crate) struct PredictionReports {
    diagnostic: bool,
    streams: BTreeMap<&'static str, PredictionStream>,
}

impl PredictionReports {
    pub(crate) fn new(diagnostic: bool) -> Self {
        Self {
            diagnostic,
            streams: [
                "common_history",
                "free_feedback",
                "reset_replay",
                "fork_replay",
            ]
            .into_iter()
            .map(|name| (name, PredictionStream::default()))
            .collect(),
        }
    }
    pub(crate) fn record(&mut self, stream: &'static str, sample: PredictionSample<'_>) {
        let PredictionSample {
            rows,
            actual,
            expected,
            actual_token,
            expected_token,
            history_matches,
            label,
        } = sample;
        let value = self.streams.get_mut(stream).unwrap();
        value.calls += 1;
        value.rows += rows;
        value.greedy_checks += 1;
        if actual_token != expected_token {
            value.report.failures.push(format!(
                "{label}/greedy: {actual_token} != {expected_token}"
            ));
        }
        if history_matches {
            value.history_matched_calls += 1;
            value.values += actual.len();
            value.report.bf16_values(actual, expected, label);
        }
    }
    pub(crate) fn passed(&self, cases: usize) -> bool {
        self.streams.iter().all(|(name, s)| {
            let expected = match *name {
                "common_history" | "free_feedback" => cases * 9,
                "reset_replay" => 9,
                "fork_replay" => 1,
                _ => unreachable!(),
            };
            s.calls == expected
                && s.greedy_checks == expected
                && s.history_matched_calls == expected
                && s.values > 0
                && s.report.failures.is_empty()
        })
    }
    pub(crate) fn json(&self) -> Value {
        let streams: serde_json::Map<String, Value> = self.streams.iter().map(|(name, s)| ((*name).into(),
            json!({"prediction_calls":s.calls,"logit_rows":s.rows,"logit_values_compared":s.values,
                "greedy_checks":s.greedy_checks,"input_history_matched_reference_calls":s.history_matched_calls,
                "ordered_route_ids_compared": None::<usize>,
                "diagnostic_state_values_compared": None::<usize>,
                "route_and_state_metrics_in_separate_diagnostic_report":self.diagnostic,
                "metrics":s.report.metrics,"failures":s.report.failures}))).collect();
        json!({"schema":"ferrule.fresh-prediction-denominators.v1","observation_scope":observer_fields(self.diagnostic),
            "streams":streams,"fork_scope":"one same-history committed-frontier replay; not divergent KV COW proof",
            "no_cross_stream_metric_sums":true})
    }
}

pub(crate) fn attach(
    report: &mut Value,
    reads: Option<&ReadQualityLog>,
    predictions: Option<&PredictionReports>,
) {
    if let Some(reads) = reads {
        report["expert_read_quality"] = reads.json();
        report["observation_scope"] = observer_fields(reads.diagnostic);
    }
    if let Some(predictions) = predictions {
        report["prediction_denominators"] = predictions.json();
    }
}

pub(crate) struct PredictionFingerprints {
    pub(crate) calls: BTreeMap<String, Value>,
    expected: Option<Value>,
    failures: Vec<String>,
}
impl PredictionFingerprints {
    pub(crate) fn new(expected: Option<Value>) -> Self {
        Self {
            calls: BTreeMap::new(),
            expected,
            failures: Vec::new(),
        }
    }
    pub(crate) fn record(&mut self, label: &str, rows: usize, logits: &[f32], token: u32) {
        use std::io::Write;
        use std::process::{Command, Stdio};
        assert!(rows > 0 && !logits.is_empty() && logits.iter().all(|v| v.is_finite()));
        let mut child = Command::new("sha256sum")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        {
            let mut input = std::io::BufWriter::new(child.stdin.take().unwrap());
            for value in logits {
                input.write_all(&value.to_bits().to_le_bytes()).unwrap();
            }
            input.flush().unwrap();
        }
        let digest = child.wait_with_output().unwrap();
        assert!(digest.status.success());
        let digest = String::from_utf8(digest.stdout)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .to_owned();
        let value =
            json!({"rows":rows,"values":logits.len(),"next_token_id":token,"f32_le_sha256":digest});
        if let Some(expected) = &self.expected
            && expected["calls"][label] != value
        {
            self.failures.push(label.into());
        }
        assert!(self.calls.insert(label.into(), value).is_none());
    }
    pub(crate) fn passed(&self) -> bool {
        !self.calls.is_empty()
            && self.failures.is_empty()
            && self.expected.as_ref().is_none_or(|expected| {
                expected["calls"]
                    .as_object()
                    .is_some_and(|calls| calls.len() == self.calls.len())
            })
    }
    pub(crate) fn json(&self) -> Value {
        json!({"schema":"ferrule.fresh-actual-logit-fingerprints.v1","calls":self.calls,
            "paired_diagnostic_report_supplied":self.expected.is_some(),"paired_exact_logits_and_greedy_equal":self.expected.as_ref().map(|_|self.passed()),
            "basis":"SHA256 of nonempty actual host-exported F32 little-endian words plus shape and exact greedy ID; reference tolerance comparator remains separate",
            "failures":self.failures})
    }
}

pub(crate) fn assert_diagnostic_primary(
    prior: &Value,
    config: &Value,
    effective: &Value,
    pin: &str,
) {
    assert_eq!(prior["schema"], "ferrule.bf16-fresh-quality-result.v1");
    assert_eq!(prior["passed"], true);
    assert_eq!(prior["complete"], true);
    assert_eq!(prior["cold_owner_restart"], false);
    assert_eq!(&prior["test_config"], config);
    assert_eq!(&prior["effective_config"], effective);
    assert_eq!(prior["oracle_manifest_sha256"], pin);
    assert_eq!(prior["observer_mode"], "diagnostic_routes_and_states");
    assert_eq!(prior["ordinary_no_observer_claim"], false);
    assert_eq!(prior["expert_read_quality"]["passed"], true);
    assert!(
        prior["prediction_fingerprints"]["calls"]
            .as_object()
            .is_some_and(|calls| calls.len() == 64)
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrule_model::resource_observability::ResourceObservationScope;
    use ferrule_model::transformer::{ExpertReadPipelineStats, ExpertReadSchedule};
    use std::num::NonZeroU64;

    fn options(schedule: ExpertReadSchedule) -> ExpertReadPipelineOptions {
        ExpertReadPipelineOptions {
            schedule,
            max_host_bytes: 64 << 20,
            max_metadata_bytes: 8 << 20,
            worker_stack_bytes: 2 << 20,
        }
    }
    fn sample(read: ExpertReadPipelineOptions) -> PhysicalResourceObservation {
        PhysicalResourceObservation {
            scope: ResourceEvidence::Observed(ResourceObservationScope::fresh(
                NonZeroU64::new(1).unwrap(),
            )),
            prepared_plan_id: ResourceEvidence::Observed(1),
            image_generation: ResourceEvidence::Observed(1),
            expert_read_pipeline: ResourceEvidence::Observed(ExpertReadPipelineObservation {
                enabled: true,
                requested_mode: Some(read.schedule),
                limits: Some(read),
                status: "configured",
                ..Default::default()
            }),
            ..Default::default()
        }
    }
    #[test]
    fn read_quality_json_missing_unknown_and_counter_reset_are_null_not_zero() {
        let read = options(ExpertReadSchedule::Blocking);
        let before = sample(read);
        let mut after = before.clone();
        let mut log = ReadQualityLog::new(read, false);
        for evidence in [ResourceEvidence::Unobserved, ResourceEvidence::Unknown] {
            after.expert_read_pipeline = evidence;
            let i = log.begin("missing", "prediction", before.clone());
            log.finish(i, &before, after.clone());
            assert_eq!(log.calls[i]["delta"], Value::Null);
            assert_eq!(log.calls[i]["positive_worker_read"], Value::Null);
        }
        let mut higher = before.clone();
        higher.expert_read_pipeline = ResourceEvidence::Observed(ExpertReadPipelineObservation {
            stats: ExpertReadPipelineStats {
                submitted_reads: 1,
                ..Default::default()
            },
            ..*before.expert_read_pipeline.value().unwrap()
        });
        let i = log.begin("reset", "prediction", higher.clone());
        log.finish(i, &higher, before.clone());
        assert!(log.calls[i]["delta"].is_null());
        let i = log.begin("observed_zero", "snapshot", before.clone());
        log.finish(i, &before, before.clone());
        assert_eq!(log.calls[i]["delta"]["submitted_reads"], 0);
        assert_eq!(log.calls[i]["positive_worker_read"], false);
        assert!(!log.passed());
    }
    #[test]
    fn read_quality_both_modes_require_positive_same_owner_matching_bounds() {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = options(schedule);
            let before = sample(read);
            let mut after = positive_after(&before, schedule);
            let observed = *after.expert_read_pipeline.value().unwrap();
            let mut log = ReadQualityLog::new(read, true);
            let i = log.begin("call", "prediction", before.clone());
            log.finish(i, &before, after.clone());
            assert!(log.passed());
            for drift in 0..4 {
                let mut bad = observed;
                match drift {
                    0 => bad.current_charge.host_charged_bytes = read.max_host_bytes + 1,
                    1 => bad.stats.unknown_reads = 1,
                    2 => bad.limits.as_mut().unwrap().worker_stack_bytes += 1,
                    _ => {
                        bad.effective_mode = Some(if schedule == ExpertReadSchedule::Blocking {
                            ExpertReadSchedule::Lookahead
                        } else {
                            ExpertReadSchedule::Blocking
                        })
                    }
                }
                after.expert_read_pipeline = ResourceEvidence::Observed(bad);
                let mut log = ReadQualityLog::new(read, false);
                let i = log.begin("bad", "prediction", before.clone());
                log.finish(i, &before, after.clone());
                assert!(!log.passed());
            }
            after.scope = sample(read).scope;
            let mut log = ReadQualityLog::new(read, false);
            let i = log.begin("foreign", "prediction", before.clone());
            log.finish(i, &before, after);
            assert!(log.calls[i]["delta"].is_null());
            assert!(!log.passed());
        }
    }
    fn positive_after(
        before: &PhysicalResourceObservation,
        schedule: ExpertReadSchedule,
    ) -> PhysicalResourceObservation {
        use ferrule_model::transformer::ExpertReadHostCharge;
        let mut after = before.clone();
        let read = before.expert_read_pipeline.value().unwrap().limits.unwrap();
        let idle_charge = read.max_metadata_bytes + read.worker_stack_bytes as u64;
        after.expert_read_pipeline = ResourceEvidence::Observed(ExpertReadPipelineObservation {
            effective_mode: Some(schedule),
            status: "idle_retained",
            worker_read_observed: true,
            worker_alive: true,
            current_charge: ExpertReadHostCharge {
                metadata_bytes: read.max_metadata_bytes,
                requested_stack_bytes: read.worker_stack_bytes as u64,
                host_charged_bytes: idle_charge,
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
                peak_metadata_bytes: read.max_metadata_bytes,
                peak_requested_stack_bytes: read.worker_stack_bytes as u64,
                peak_host_charged_bytes: idle_charge + 256,
                peak_packed_bytes: 128,
                peak_artifact_bytes: 128,
                peak_payload_bytes: 256,
                ..Default::default()
            },
            ..*before.expert_read_pipeline.value().unwrap()
        });
        after
    }

    #[test]
    fn read_quality_interval_configuration_rejects_bad_before_with_matching_positive_after() {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = options(schedule);
            let before = sample(read);
            let after = positive_after(&before, schedule);
            let other = if schedule == ExpertReadSchedule::Blocking {
                ExpertReadSchedule::Lookahead
            } else {
                ExpertReadSchedule::Blocking
            };
            assert_eq!(
                before.expert_read_pipeline.value().unwrap().effective_mode,
                None
            );
            assert!(
                !before
                    .expert_read_pipeline
                    .value()
                    .unwrap()
                    .worker_read_observed
            );
            let mut valid = ReadQualityLog::new(read, false);
            let i = valid.begin("configured first read", "prediction", before.clone());
            valid.finish(i, &before, after.clone());
            assert!(valid.passed());
            for field in [
                "disabled",
                "requested_mode",
                "missing_limits",
                "host_limit",
                "metadata_limit",
                "stack_limit",
                "limits_schedule",
                "effective_mode",
            ] {
                let mut bad = before.clone();
                let mut observed = *before.expert_read_pipeline.value().unwrap();
                match field {
                    "disabled" => observed.enabled = false,
                    "requested_mode" => observed.requested_mode = Some(other),
                    "missing_limits" => observed.limits = None,
                    "host_limit" => observed.limits.as_mut().unwrap().max_host_bytes += 1,
                    "metadata_limit" => observed.limits.as_mut().unwrap().max_metadata_bytes += 1,
                    "stack_limit" => observed.limits.as_mut().unwrap().worker_stack_bytes += 1,
                    "limits_schedule" => observed.limits.as_mut().unwrap().schedule = other,
                    "effective_mode" => observed.effective_mode = Some(other),
                    _ => unreachable!(),
                }
                bad.expert_read_pipeline = ResourceEvidence::Observed(observed);
                let mut log = ReadQualityLog::new(read, false);
                let i = log.begin(field, "prediction", bad.clone());
                log.finish(i, &bad, after.clone());
                assert_eq!(log.calls[i]["same_owner"], true, "{schedule:?}/{field}");
                assert_eq!(
                    log.calls[i]["positive_worker_read"], true,
                    "{schedule:?}/{field}"
                );
                assert_eq!(
                    log.calls[i]["configuration_matches"], false,
                    "{schedule:?}/{field}"
                );
                assert!(!log.passed(), "{schedule:?}/{field}");
            }
        }
    }

    #[test]
    fn read_quality_interval_rejects_before_overbudget_or_unknown_with_positive_after() {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = options(schedule);
            let before = sample(read);
            let after = positive_after(&before, schedule);
            let payload =
                read.max_host_bytes - read.max_metadata_bytes - read.worker_stack_bytes as u64;
            for field in [
                "packed",
                "artifact",
                "payload",
                "metadata",
                "stack",
                "host",
                "slots",
                "peak_packed",
                "peak_artifact",
                "peak_payload",
                "peak_metadata",
                "peak_stack",
                "peak_host",
                "unknown_packed",
                "unknown_stack",
                "unknown_slots",
                "quarantined",
            ] {
                let mut bad = before.clone();
                let mut observed = *before.expert_read_pipeline.value().unwrap();
                let (flag, expected) = match field {
                    "packed" => {
                        observed.current_charge.packed_bytes = payload / 2 + 1;
                        ("within_charged_limits", false)
                    }
                    "artifact" => {
                        observed.current_charge.artifact_bytes = payload / 2 + 1;
                        ("within_charged_limits", false)
                    }
                    "payload" => {
                        observed.current_charge.payload_bytes = payload + 1;
                        ("within_charged_limits", false)
                    }
                    "metadata" => {
                        observed.current_charge.metadata_bytes = read.max_metadata_bytes + 1;
                        ("within_charged_limits", false)
                    }
                    "stack" => {
                        observed.current_charge.requested_stack_bytes =
                            read.worker_stack_bytes as u64 + 1;
                        ("within_charged_limits", false)
                    }
                    "host" => {
                        observed.current_charge.host_charged_bytes = read.max_host_bytes + 1;
                        ("within_charged_limits", false)
                    }
                    "slots" => {
                        observed.current_charge.reserved_read_slots = 3;
                        ("within_charged_limits", false)
                    }
                    "peak_packed" => {
                        observed.stats.peak_packed_bytes = payload / 2 + 1;
                        ("within_charged_limits", false)
                    }
                    "peak_artifact" => {
                        observed.stats.peak_artifact_bytes = payload / 2 + 1;
                        ("within_charged_limits", false)
                    }
                    "peak_payload" => {
                        observed.stats.peak_payload_bytes = payload + 1;
                        ("within_charged_limits", false)
                    }
                    "peak_metadata" => {
                        observed.stats.peak_metadata_bytes = read.max_metadata_bytes + 1;
                        ("within_charged_limits", false)
                    }
                    "peak_stack" => {
                        observed.stats.peak_requested_stack_bytes =
                            read.worker_stack_bytes as u64 + 1;
                        ("within_charged_limits", false)
                    }
                    "peak_host" => {
                        observed.stats.peak_host_charged_bytes = read.max_host_bytes + 1;
                        ("within_charged_limits", false)
                    }
                    "unknown_packed" => {
                        observed.current_charge.unknown_packed_bytes = 1;
                        ("unambiguous", false)
                    }
                    "unknown_stack" => {
                        observed.current_charge.unknown_stack_bytes = 1;
                        ("unambiguous", false)
                    }
                    "unknown_slots" => {
                        observed.current_charge.unknown_read_slots = 1;
                        ("unambiguous", false)
                    }
                    "quarantined" => {
                        observed.status = "quarantined";
                        ("unambiguous", false)
                    }
                    _ => unreachable!(),
                };
                bad.expert_read_pipeline = ResourceEvidence::Observed(observed);
                let mut log = ReadQualityLog::new(read, false);
                let i = log.begin(field, "prediction", bad.clone());
                log.finish(i, &bad, after.clone());
                assert_eq!(log.calls[i]["same_owner"], true, "{schedule:?}/{field}");
                assert_eq!(
                    log.calls[i]["positive_worker_read"], true,
                    "{schedule:?}/{field}"
                );
                assert_eq!(log.calls[i][flag], expected, "{schedule:?}/{field}");
                assert!(!log.passed(), "{schedule:?}/{field}");
            }
        }
    }

    #[test]
    fn read_quality_interval_missing_before_and_counter_reset_remain_unqualified() {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = options(schedule);
            let before = sample(read);
            let after = positive_after(&before, schedule);
            for evidence in [ResourceEvidence::Unobserved, ResourceEvidence::Unknown] {
                let mut bad = before.clone();
                bad.expert_read_pipeline = evidence;
                let mut log = ReadQualityLog::new(read, false);
                let i = log.begin("missing before", "prediction", bad.clone());
                log.finish(i, &bad, after.clone());
                for field in [
                    "delta",
                    "positive_worker_read",
                    "configuration_matches",
                    "within_charged_limits",
                    "unambiguous",
                ] {
                    assert!(log.calls[i][field].is_null(), "{schedule:?}/{field}");
                }
                assert!(!log.passed());
            }
            let mut reset = before.clone();
            reset.expert_read_pipeline =
                ResourceEvidence::Observed(ExpertReadPipelineObservation {
                    stats: ExpertReadPipelineStats {
                        submitted_reads: 7,
                        ..Default::default()
                    },
                    ..*before.expert_read_pipeline.value().unwrap()
                });
            let mut log = ReadQualityLog::new(read, false);
            let i = log.begin("counter reset", "prediction", reset.clone());
            log.finish(i, &reset, after);
            assert!(log.calls[i]["delta"].is_null());
            assert!(log.calls[i]["positive_worker_read"].is_null());
            assert!(!log.passed());
        }
    }

    #[test]
    fn read_quality_interval_report_keeps_v1_flags_and_documents_both_endpoints() {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = options(schedule);
            let before = sample(read);
            let after = positive_after(&before, schedule);
            let mut log = ReadQualityLog::new(read, false);
            let i = log.begin("configured", "prediction", before.clone());
            assert!(log.calls[i]["configuration_pair"]["before"].is_null());
            assert!(log.calls[i]["configuration_pair"]["after"].is_null());
            log.finish(i, &before, after.clone());
            let report = log.json();
            assert_eq!(report["schema"], "ferrule.expert-read-quality-evidence.v1");
            assert_eq!(report["interval_validation_revision"], 3);
            assert_eq!(report["worker_snapshot_contract_revision"], 1);
            assert_eq!(report["passed"], true);
            assert!(
                report["interval_validation_contract"]
                    .as_str()
                    .unwrap()
                    .contains("both observed endpoints")
            );
            for field in [
                "configuration_matches",
                "within_charged_limits",
                "unambiguous",
            ] {
                assert_eq!(report["calls"][i][field], true);
                assert_eq!(
                    report["calls"][i]["configuration_pair"]["before"][field],
                    true
                );
                assert_eq!(
                    report["calls"][i]["configuration_pair"]["after"][field],
                    true
                );
            }
            assert!(
                report["calls"][i]["before"]["expert_read_pipeline"]["value"]["effective_mode"]
                    .is_null()
            );
            let mut disabled = before.clone();
            disabled.expert_read_pipeline =
                ResourceEvidence::Observed(ExpertReadPipelineObservation {
                    enabled: false,
                    ..*before.expert_read_pipeline.value().unwrap()
                });
            let i = log.begin("disabled before", "prediction", disabled.clone());
            log.finish(i, &disabled, after.clone());
            assert_eq!(
                log.calls[i]["configuration_pair"]["before"]["configuration_matches"],
                false
            );
            assert_eq!(
                log.calls[i]["configuration_pair"]["after"]["configuration_matches"],
                true
            );
            assert_eq!(log.calls[i]["configuration_matches"], false);
            assert!(!log.passed());
            let mut missing = before;
            missing.expert_read_pipeline = ResourceEvidence::Unknown;
            let i = log.begin("unknown before", "prediction", missing.clone());
            log.finish(i, &missing, after);
            for field in [
                "configuration_matches",
                "within_charged_limits",
                "unambiguous",
            ] {
                assert!(log.calls[i][field].is_null());
                assert!(log.calls[i]["configuration_pair"]["before"][field].is_null());
                assert_eq!(log.calls[i]["configuration_pair"]["after"][field], true);
            }
        }
    }

    fn record_pair(
        log: &mut ReadQualityLog,
        before: &PhysicalResourceObservation,
        after: &PhysicalResourceObservation,
    ) -> usize {
        let i = log.begin("worker lifecycle", "snapshot", before.clone());
        log.finish(i, before, after.clone());
        i
    }

    #[test]
    fn read_quality_zero_byte_observation_poison_rejects_even_after_positive_reads() {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = options(schedule);
            let configured = sample(read);
            let idle = positive_after(&configured, schedule);
            let mut closed = idle.clone();
            let mut value = *idle.expert_read_pipeline.value().unwrap();
            value.status = "closed";
            value.worker_alive = false;
            value.stats.workers_joined = 1;
            value.current_charge = Default::default();
            closed.expert_read_pipeline = ResourceEvidence::Observed(value);
            assert!(unambiguous(&value));
            // Exact getter mapping, not an invented unknown byte: snapshot()
            // retains positive read counts, then the getter marks lock poison.
            value.status = "unknown_observation_poisoned";
            value.worker_unknown = true;
            let mut poisoned = closed.clone();
            poisoned.expert_read_pipeline = ResourceEvidence::Observed(value);
            assert_eq!(value.current_charge, Default::default());
            assert_eq!(value.stats.unknown_reads, 0);
            assert_eq!(value.stats.unknown_bytes, 0);
            assert!(value.worker_read_observed && value.stats.proved_read_bytes > 0);
            for before_poisoned in [false, true] {
                let mut log = ReadQualityLog::new(read, false);
                record_pair(&mut log, &configured, &idle);
                assert!(log.passed());
                let (before, after) = if before_poisoned {
                    (&poisoned, &closed)
                } else {
                    (&closed, &poisoned)
                };
                let i = record_pair(&mut log, before, after);
                assert_eq!(log.calls[i]["evidence_complete"], true);
                assert_eq!(log.calls[i]["delta"]["submitted_reads"], 0);
                assert_eq!(log.calls[i]["unambiguous"], false);
                let endpoint = if before_poisoned { "before" } else { "after" };
                assert_eq!(
                    log.calls[i]["configuration_pair"][endpoint]["worker_snapshot_rejection"],
                    "observation_ledger_poisoned"
                );
                assert!(!log.passed());
            }
        }
    }

    #[test]
    fn read_quality_idle_and_final_join_are_distinct_exact_charge_stages_both_budgets() {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            for read in [
                options(schedule),
                ExpertReadPipelineOptions {
                    schedule,
                    max_host_bytes: 2 << 20,
                    max_metadata_bytes: 256 << 10,
                    worker_stack_bytes: 1 << 20,
                },
            ] {
                let configured = sample(read);
                let idle = positive_after(&configured, schedule);
                let value = *idle.expert_read_pipeline.value().unwrap();
                assert_eq!(value.stats.submitted_reads, 6);
                assert_eq!(
                    (
                        value.stats.workers_created,
                        value.stats.worker_reuses,
                        value.stats.workers_joined
                    ),
                    (1, 1, 0)
                );
                assert_eq!(
                    value.current_charge.host_charged_bytes,
                    if read.max_host_bytes == 2 << 20 {
                        1280 << 10
                    } else {
                        10 << 20
                    }
                );
                let mut log = ReadQualityLog::new(read, false);
                record_pair(&mut log, &configured, &idle);
                record_pair(&mut log, &idle, &idle);
                assert!(log.passed(), "report finish does not drop its owner");
                let mut closed = idle.clone();
                closed.expert_read_pipeline =
                    ResourceEvidence::Observed(ExpertReadPipelineObservation {
                        status: "closed",
                        worker_alive: false,
                        current_charge: Default::default(),
                        stats: ExpertReadPipelineStats {
                            workers_joined: 1,
                            ..value.stats
                        },
                        ..value
                    });
                let i = record_pair(&mut log, &idle, &closed);
                assert_eq!(log.calls[i]["delta"]["workers_joined"], 1);
                assert_eq!(log.calls[i]["delta"]["workers_created"], 0);
                assert_eq!(log.calls[i]["delta"]["worker_reuses"], 0);
                assert!(log.passed());
                assert_eq!(
                    log.json()["charge_basis"],
                    "logical_host_reservation_not_physical_rss"
                );
            }
        }
    }

    #[test]
    fn read_quality_status_allowlist_and_typed_unknown_flags_fail_closed_at_both_ends() {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = options(schedule);
            let configured = sample(read);
            let idle = positive_after(&configured, schedule);
            for (status, unknown, pending, reason) in [
                (
                    "unknown_observation_poisoned",
                    true,
                    false,
                    "observation_ledger_poisoned",
                ),
                (
                    "unknown_ledger_poisoned",
                    true,
                    false,
                    "worker_or_admission_ledger_poisoned",
                ),
                ("quarantined", true, false, "quarantined_custody"),
                (
                    "join_pending_at_scope_end",
                    false,
                    true,
                    "worker_join_pending",
                ),
                ("active", false, false, "read_scope_still_active"),
                (
                    "closed_retained",
                    false,
                    false,
                    "payload_custody_still_retained",
                ),
                ("disabled", false, false, "pipeline_disabled"),
                ("future_status", false, false, "unsupported_worker_status"),
                (
                    "unknown_future_reason",
                    false,
                    false,
                    "unsupported_worker_status",
                ),
                ("idle_retained", true, false, "worker_unknown"),
                ("idle_retained", false, true, "worker_join_pending"),
            ] {
                let mut bad = idle.clone();
                bad.expert_read_pipeline =
                    ResourceEvidence::Observed(ExpertReadPipelineObservation {
                        status,
                        worker_unknown: unknown,
                        worker_join_pending: pending,
                        ..*idle.expert_read_pipeline.value().unwrap()
                    });
                for before_bad in [false, true] {
                    let mut log = ReadQualityLog::new(read, false);
                    record_pair(&mut log, &configured, &idle);
                    let (before, after) = if before_bad {
                        (&bad, &idle)
                    } else {
                        (&idle, &bad)
                    };
                    let i = record_pair(&mut log, before, after);
                    let endpoint = if before_bad { "before" } else { "after" };
                    assert_eq!(
                        log.calls[i]["configuration_pair"][endpoint]["worker_snapshot_rejection"],
                        reason
                    );
                    assert_eq!(log.calls[i]["unambiguous"], false);
                    assert!(!log.passed(), "{schedule:?}/{status}/{before_bad}");
                }
            }
        }
    }

    #[test]
    fn read_quality_impossible_worker_counts_and_payload_aliases_never_qualify() {
        use ferrule_model::transformer::ExpertReadIdentity;
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = options(schedule);
            let configured = sample(read);
            let idle = positive_after(&configured, schedule);
            for field in [
                "created",
                "joined",
                "alive",
                "reuse",
                "finished",
                "join_failed",
                "payload",
                "slot",
                "metadata",
                "stack",
                "total",
                "current",
                "pending",
                "configured_after_reads",
                "closed_alive",
                "read_started",
                "terminal_count",
            ] {
                let mut value = *idle.expert_read_pipeline.value().unwrap();
                let identity = Some(ExpertReadIdentity {
                    image_generation: 1,
                    layer: 0,
                    expert: 0,
                    operation: 1,
                });
                match field {
                    "created" => value.stats.workers_created = 2,
                    "joined" => value.stats.workers_joined = 2,
                    "alive" => value.worker_alive = false,
                    "reuse" => value.stats.worker_reuses = 6,
                    "finished" => value.stats.sequences_finished = 1,
                    "join_failed" => value.stats.join_failed_sequences = 1,
                    "payload" => {
                        value.current_charge.artifact_bytes = 128;
                        value.current_charge.payload_bytes = 128;
                        value.current_charge.host_charged_bytes += 128;
                    }
                    "slot" => value.current_charge.reserved_read_slots = 1,
                    "metadata" => {
                        value.current_charge.metadata_bytes -= 1;
                        value.current_charge.host_charged_bytes -= 1;
                    }
                    "stack" => {
                        value.current_charge.requested_stack_bytes -= 1;
                        value.current_charge.host_charged_bytes -= 1;
                    }
                    "total" => value.current_charge.host_charged_bytes -= 1,
                    "current" => value.current_read = identity,
                    "pending" => value.pending_read = identity,
                    "configured_after_reads" => value.status = "configured",
                    "closed_alive" => value.status = "closed",
                    "read_started" => value.stats.worker_started_reads = 0,
                    "terminal_count" => value.stats.completed_reads = 5,
                    _ => unreachable!(),
                }
                let mut bad = idle.clone();
                bad.expert_read_pipeline = ResourceEvidence::Observed(value);
                for before_bad in [false, true] {
                    let mut log = ReadQualityLog::new(read, false);
                    record_pair(&mut log, &configured, &idle);
                    let (before, after) = if before_bad {
                        (&bad, &idle)
                    } else {
                        (&idle, &bad)
                    };
                    let i = record_pair(&mut log, before, after);
                    assert_eq!(log.calls[i]["unambiguous"], false, "{schedule:?}/{field}");
                    assert!(!log.passed(), "{schedule:?}/{field}");
                }
            }
        }
    }

    #[test]
    fn read_quality_retained_idle_budget_failures_are_checked_at_both_endpoints() {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = options(schedule);
            let configured = sample(read);
            let idle = positive_after(&configured, schedule);
            let payload =
                read.max_host_bytes - read.max_metadata_bytes - read.worker_stack_bytes as u64;
            for field in [
                "peak_packed",
                "peak_artifact",
                "peak_payload",
                "peak_metadata",
                "peak_stack",
                "peak_host",
                "missing_peak_metadata",
                "missing_peak_stack",
                "missing_peak_host",
            ] {
                let mut value = *idle.expert_read_pipeline.value().unwrap();
                match field {
                    "peak_packed" => value.stats.peak_packed_bytes = payload / 2 + 1,
                    "peak_artifact" => value.stats.peak_artifact_bytes = payload / 2 + 1,
                    "peak_payload" => value.stats.peak_payload_bytes = payload + 1,
                    "peak_metadata" => {
                        value.stats.peak_metadata_bytes = read.max_metadata_bytes + 1
                    }
                    "peak_stack" => {
                        value.stats.peak_requested_stack_bytes = read.worker_stack_bytes as u64 + 1
                    }
                    "peak_host" => value.stats.peak_host_charged_bytes = read.max_host_bytes + 1,
                    "missing_peak_metadata" => value.stats.peak_metadata_bytes = 0,
                    "missing_peak_stack" => value.stats.peak_requested_stack_bytes = 0,
                    "missing_peak_host" => value.stats.peak_host_charged_bytes = 0,
                    _ => unreachable!(),
                }
                let mut bad = idle.clone();
                bad.expert_read_pipeline = ResourceEvidence::Observed(value);
                for before_bad in [false, true] {
                    let mut log = ReadQualityLog::new(read, false);
                    record_pair(&mut log, &configured, &idle);
                    let (before, after) = if before_bad {
                        (&bad, &idle)
                    } else {
                        (&idle, &bad)
                    };
                    let i = record_pair(&mut log, before, after);
                    let endpoint = if before_bad { "before" } else { "after" };
                    assert_eq!(log.calls[i]["unambiguous"], true, "{field}");
                    assert_eq!(
                        log.calls[i]["configuration_pair"][endpoint]["within_charged_limits"],
                        false
                    );
                    assert_eq!(log.calls[i]["within_charged_limits"], false);
                    assert!(!log.passed());
                }
            }
        }
    }

    #[test]
    fn read_quality_worker_counter_resets_and_missing_endpoints_are_never_defaulted() {
        for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
            let read = options(schedule);
            let configured = sample(read);
            let idle = positive_after(&configured, schedule);
            for counter in ["workers_created", "worker_reuses", "workers_joined"] {
                let mut value = *idle.expert_read_pipeline.value().unwrap();
                match counter {
                    "workers_created" => value.stats.workers_created += 1,
                    "worker_reuses" => value.stats.worker_reuses += 1,
                    "workers_joined" => value.stats.workers_joined += 1,
                    _ => unreachable!(),
                }
                let mut before = idle.clone();
                before.expert_read_pipeline = ResourceEvidence::Observed(value);
                let mut log = ReadQualityLog::new(read, false);
                let i = record_pair(&mut log, &before, &idle);
                assert!(log.calls[i]["delta"].is_null(), "{counter}");
                assert!(log.calls[i]["positive_worker_read"].is_null());
                assert!(!log.passed());
            }
            for missing in [ResourceEvidence::Unknown, ResourceEvidence::Unobserved] {
                let mut absent = idle.clone();
                absent.expert_read_pipeline = missing;
                for before_absent in [false, true] {
                    let mut log = ReadQualityLog::new(read, false);
                    let (before, after) = if before_absent {
                        (&absent, &idle)
                    } else {
                        (&idle, &absent)
                    };
                    let i = record_pair(&mut log, before, after);
                    let endpoint = if before_absent { "before" } else { "after" };
                    assert!(
                        log.calls[i]["configuration_pair"][endpoint]["worker_snapshot_valid"]
                            .is_null()
                    );
                    assert!(log.calls[i]["delta"].is_null());
                    assert!(!log.passed());
                }
            }
        }
    }

    #[test]
    fn read_quality_disabled_reports_omit_fields_and_ordinary_never_claims_routes() {
        let mut legacy = json!({"ordinary_no_observer_claim":false,"metrics":[]});
        let frozen = legacy.clone();
        attach(&mut legacy, None, None);
        assert_eq!(legacy, frozen);
        let ordinary = observer_fields(false);
        assert_eq!(ordinary["diagnostic_trace_installed"], false);
        assert_eq!(ordinary["routes_observed"], false);
        assert_eq!(ordinary["states_observed"], false);
        assert_eq!(ordinary["ordinary_no_observer_claim"], true);
        assert_eq!(ordinary["gpu_overlap_claim"], false);
        let reports = PredictionReports::new(false);
        assert!(
            reports.json()["streams"]["common_history"]["ordered_route_ids_compared"].is_null()
        );
        assert!(!reports.passed(3));
    }
    #[test]
    fn ordinary_fingerprints_require_matching_actual_nonempty_logits_and_shape() {
        let mut diagnostic = PredictionFingerprints::new(None);
        diagnostic.record("qa/common_history/step.00", 1, &[0., 1.], 1);
        let prior = diagnostic.json();
        let mut ordinary = PredictionFingerprints::new(Some(prior.clone()));
        ordinary.record("qa/common_history/step.00", 1, &[0., 1.], 1);
        assert!(ordinary.passed());
        assert_eq!(
            ordinary.json()["paired_exact_logits_and_greedy_equal"],
            true
        );
        for (values, token, rows) in [([0., 2.], 1, 1), ([0., 1.], 0, 1), ([0., 1.], 1, 2)] {
            let mut ordinary = PredictionFingerprints::new(Some(prior.clone()));
            ordinary.record("qa/common_history/step.00", rows, &values, token);
            assert!(!ordinary.passed());
        }
        let empty = PredictionFingerprints::new(Some(prior));
        assert!(!empty.passed());
    }

    #[test]
    fn prediction_sample_preserves_rows_logits_tokens_and_label() {
        let mut reports = PredictionReports::new(false);
        let before = reports.json();
        reports.record(
            "common_history",
            PredictionSample {
                rows: 2,
                actual: &[0., 1., 2., 3.],
                expected: &[0., 1., 2., 4.],
                actual_token: 2,
                expected_token: 3,
                history_matches: true,
                label: "multirow sample",
            },
        );
        let report = reports.json();
        let stream = &report["streams"]["common_history"];
        assert_eq!(stream["prediction_calls"], 1);
        assert_eq!(stream["logit_rows"], 2);
        assert_eq!(stream["logit_values_compared"], 4);
        assert_eq!(stream["greedy_checks"], 1);
        assert_eq!(stream["input_history_matched_reference_calls"], 1);
        assert_eq!(stream["metrics"].as_array().unwrap().len(), 1);
        assert_eq!(stream["metrics"][0]["label"], "multirow sample");
        assert_eq!(stream["metrics"][0]["elements"], 4);
        assert_eq!(stream["metrics"][0]["failures"], 1);
        assert_eq!(stream["metrics"][0]["first_failure"]["actual"], 3.0);
        assert_eq!(stream["metrics"][0]["first_failure"]["expected"], 4.0);
        assert_eq!(
            stream["failures"],
            json!(["multirow sample/greedy: 2 != 3", "multirow sample"])
        );
        for name in ["free_feedback", "reset_replay", "fork_replay"] {
            assert_eq!(report["streams"][name], before["streams"][name]);
        }
    }

    #[test]
    fn prediction_sample_divergent_history_keeps_greedy_check_without_logit_comparison() {
        let mut reports = PredictionReports::new(false);
        reports.record(
            "free_feedback",
            PredictionSample {
                rows: 2,
                actual: &[0., 1., 2., 3.],
                expected: &[],
                actual_token: 3,
                expected_token: 1,
                history_matches: false,
                label: "divergent history",
            },
        );
        let report = reports.json();
        let stream = &report["streams"]["free_feedback"];
        assert_eq!(stream["prediction_calls"], 1);
        assert_eq!(stream["logit_rows"], 2);
        assert_eq!(stream["logit_values_compared"], 0);
        assert_eq!(stream["greedy_checks"], 1);
        assert_eq!(stream["input_history_matched_reference_calls"], 0);
        assert_eq!(stream["metrics"], json!([]));
        assert_eq!(
            stream["failures"],
            json!(["divergent history/greedy: 3 != 1"])
        );
    }

    #[test]
    fn prediction_denominators_do_not_mix_or_admit_vacuous_passes() {
        let mut reports = PredictionReports::new(false);
        for stream in [
            "common_history",
            "free_feedback",
            "reset_replay",
            "fork_replay",
        ] {
            let count = if stream == "fork_replay" { 1 } else { 9 };
            for _ in 0..count {
                reports.record(
                    stream,
                    PredictionSample {
                        rows: 1,
                        actual: &[0., 1.],
                        expected: &[0., 1.],
                        actual_token: 1,
                        expected_token: 1,
                        history_matches: true,
                        label: "same pinned comparator",
                    },
                );
            }
        }
        assert!(reports.passed(1));
        let json = reports.json();
        assert_eq!(json["streams"]["fork_replay"]["logit_values_compared"], 2);
        assert_eq!(json["streams"]["reset_replay"]["logit_values_compared"], 18);
        reports.record(
            "free_feedback",
            PredictionSample {
                rows: 1,
                actual: &[0., 1.],
                expected: &[0., 1.],
                actual_token: 1,
                expected_token: 1,
                history_matches: false,
                label: "divergent history",
            },
        );
        assert!(!reports.passed(1));
    }
}
