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
}

fn unambiguous(value: &ExpertReadPipelineObservation) -> bool {
    value.stats.unknown_reads == 0
        && value.stats.unknown_bytes == 0
        && value.stats.unobserved_terminal_reads == 0
        && value.stats.unobserved_reader_intervals == 0
        && value.current_charge.unknown_packed_bytes == 0
        && value.current_charge.unknown_stack_bytes == 0
        && value.current_charge.unknown_read_slots == 0
        && value.status != "quarantined"
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
        let observed = after.expert_read_pipeline.value();
        let delta = before
            .expert_read_pipeline
            .value()
            .zip(observed)
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
        record["configuration_matches"] = json!(observed.map(|o| {
            o.enabled
                && o.requested_mode == Some(requested.schedule)
                && o.limits == Some(requested)
                && o.effective_mode
                    .is_none_or(|mode| mode == requested.schedule)
        }));
        record["within_charged_limits"] = json!(observed.map(|o| bounded(o, requested)));
        record["unambiguous"] = json!(observed.map(unambiguous));
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
        json!({"schema":"ferrule.expert-read-quality-evidence.v1","requested":self.requested,
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
    pub(crate) fn record(
        &mut self,
        stream: &'static str,
        rows: usize,
        actual: &[f32],
        expected: &[f32],
        actual_token: u32,
        expected_token: u32,
        history_matches: bool,
        label: &str,
    ) {
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
            let mut after = before.clone();
            let mut observed = *before.expert_read_pipeline.value().unwrap();
            observed.effective_mode = Some(schedule);
            observed.worker_read_observed = true;
            observed.stats = ExpertReadPipelineStats {
                submitted_reads: 1,
                completed_reads: 1,
                verified_reads: 1,
                io_started_reads: 1,
                proved_read_bytes: 128,
                ..Default::default()
            };
            after.expert_read_pipeline = ResourceEvidence::Observed(observed);
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
                    1,
                    &[0., 1.],
                    &[0., 1.],
                    1,
                    1,
                    true,
                    "same pinned comparator",
                );
            }
        }
        assert!(reports.passed(1));
        let json = reports.json();
        assert_eq!(json["streams"]["fork_replay"]["logit_values_compared"], 2);
        assert_eq!(json["streams"]["reset_replay"]["logit_values_compared"], 18);
        reports.record(
            "free_feedback",
            1,
            &[0., 1.],
            &[0., 1.],
            1,
            1,
            false,
            "divergent history",
        );
        assert!(!reports.passed(1));
    }
}
