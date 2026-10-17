//! CPU-only producer contract. Materializer observations are real; owner bindings
//! and nonzero/unknown custody injections are typed fixtures, not GPU evidence.
use super::*;
use ferrule_model::resource_observability::{
    PhysicalResourceObservation, ResourceEvidence, ResourceObservationScope,
};
use ferrule_model::transformer::{
    ExpertReadPipelineObservation, ExpertReadSchedule, ExpertTransferPolicy, StateDictMaterializer,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

#[derive(Clone, Default)]
struct Logs(Arc<Mutex<Vec<BTreeMap<String, Value>>>>);
impl tracing::Subscriber for Logs {
    fn register_callsite(
        &self,
        _: &'static tracing::Metadata<'static>,
    ) -> tracing::subscriber::Interest {
        tracing::subscriber::Interest::sometimes()
    }
    fn max_level_hint(&self) -> Option<tracing::metadata::LevelFilter> {
        Some(tracing::metadata::LevelFilter::INFO)
    }
    fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
        metadata.target() == "ferrule_performance"
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        struct Fields(BTreeMap<String, Value>);
        impl tracing::field::Visit for Fields {
            fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
                self.0
                    .insert(field.name().into(), format!("{value:?}").into());
            }
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                self.0.insert(field.name().into(), value.into());
            }
            fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
                self.0.insert(field.name().into(), value.into());
            }
        }
        let mut fields = Fields(BTreeMap::new());
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
}

fn materializer(schedule: ExpertReadSchedule) -> StateDictMaterializer {
    let mut materializer = StateDictMaterializer::new(1024).unwrap();
    materializer
        .configure_expert_read_pipeline(ExpertReadPipelineOptions {
            schedule,
            max_host_bytes: 64 << 20,
            max_metadata_bytes: 8 << 20,
            worker_stack_bytes: 2 << 20,
        })
        .unwrap();
    materializer
}

fn owner_fixture(observation: ExpertReadPipelineObservation) -> PhysicalResourceObservation {
    PhysicalResourceObservation {
        scope: ResourceEvidence::Observed(ResourceObservationScope {
            owner: NonZeroU64::new(17).unwrap(),
            generation: NonZeroU64::new(3).unwrap(),
        }),
        prepared_plan_id: ResourceEvidence::Observed(23),
        image_generation: ResourceEvidence::Observed(29),
        device_ordinal: ResourceEvidence::Observed(0),
        expert_read_pipeline: ResourceEvidence::Observed(observation),
        ..Default::default()
    }
}

fn emit(snapshot: &PhysicalResourceObservation) -> BTreeMap<String, Value> {
    let logs = Logs::default();
    tracing::subscriber::with_default(logs.clone(), || {
        log_initial_expert_read_endpoint(
            || snapshot.clone(),
            Some(ServingExpertTransferPolicy::GroupedSynchronous),
            ExpertTransferPolicy::GroupedSynchronous,
        )
        .unwrap();
    });
    let mut records = logs.0.lock().unwrap();
    assert_eq!(records.len(), 1, "one event, never one event per expert");
    records.remove(0)
}

fn payload(record: &BTreeMap<String, Value>) -> Value {
    serde_json::from_str(record["expert_read_pipeline_initial"].as_str().unwrap()).unwrap()
}

#[test]
fn initial_endpoint_serializes_actual_configured_unexecuted_materializer_observation() {
    for schedule in [ExpertReadSchedule::Blocking, ExpertReadSchedule::Lookahead] {
        let materializer = materializer(schedule);
        let observed = materializer.expert_read_pipeline_observation();
        assert_eq!(observed.status, "configured");
        assert!(observed.enabled);
        assert_eq!(observed.requested_mode, Some(schedule));
        assert_eq!(observed.effective_mode, None);
        assert!(!observed.worker_alive);
        let snapshot = owner_fixture(observed);
        let record = emit(&snapshot);
        assert_eq!(record["event"], "expert_read_initial_endpoint");
        assert_eq!(record["producer_schema_version"], 1);
        assert_eq!(
            record["source_contract"],
            "resident_model_runner.physical_resource_snapshot.expert_read_pipeline"
        );
        assert_eq!(
            record["endpoint"],
            "after_owner_configuration_before_prediction"
        );
        assert_eq!(record["process_id"], std::process::id());
        assert_eq!(
            record["requested_expert_transfer_policy"],
            "Some(GroupedSynchronous)"
        );
        assert_eq!(record["expert_transfer_policy"], "GroupedSynchronous");
        if schedule == ExpertReadSchedule::Blocking {
            eprintln!(
                "typed_initial_endpoint_fixture={}",
                serde_json::to_string(&record).unwrap()
            );
        }
        assert_eq!(
            payload(&record),
            serde_json::to_value(snapshot.expert_read_pipeline).unwrap()
        );
        assert_eq!(payload(&record)["state"], "observed");
        assert_eq!(payload(&record)["value"]["stats"]["submitted_reads"], 0);
        let binding: Value =
            serde_json::from_str(record["owner_binding"].as_str().unwrap()).unwrap();
        assert_eq!(
            binding,
            json!({
                "scope":{"state":"observed","value":{"owner":17,"generation":3}},
                "prepared_plan_id":{"state":"observed","value":23},
                "image_generation":{"state":"observed","value":29},
                "device_ordinal":{"state":"observed","value":0},
            })
        );
        assert_eq!(
            materializer.expert_read_pipeline_observation(),
            observed,
            "logging must not start a worker, clear counters, or change custody"
        );
    }
}

#[test]
fn initial_endpoint_retains_nonzero_counters_live_worker_and_all_upload_charges() {
    let mut observed =
        materializer(ExpertReadSchedule::Lookahead).expert_read_pipeline_observation();
    // Injected producer sample tests verbatim serialization, not a GPU/idle proof.
    observed.worker_alive = true;
    observed.stats.workers_created = 1;
    observed.stats.reader_interval_calls = 3;
    observed.stats.reader_interval_bytes = 4096;
    observed.stats.unobserved_reader_intervals = 2;
    observed.current_charge.upload_reserved_packed_bytes = 128;
    observed.current_charge.pinned_upload_bytes = 256;
    observed.current_charge.upload_ticket_metadata_bytes = 32;
    observed.current_charge.upload_policy_headroom_bytes = 64;
    observed.current_charge.unknown_upload_bytes = 512;
    let snapshot = owner_fixture(observed);
    let before = snapshot.clone();
    let record = emit(&snapshot);
    assert_eq!(
        payload(&record),
        serde_json::to_value(snapshot.expert_read_pipeline).unwrap()
    );
    let initial = payload(&record);
    assert_eq!(initial["value"]["stats"]["reader_interval_calls"], 3);
    assert_eq!(initial["value"]["worker_alive"], true);
    for (key, value) in [
        ("upload_reserved_packed_bytes", 128),
        ("pinned_upload_bytes", 256),
        ("upload_ticket_metadata_bytes", 32),
        ("upload_policy_headroom_bytes", 64),
        ("unknown_upload_bytes", 512),
    ] {
        assert_eq!(initial["value"]["current_charge"][key], value);
    }
    assert_eq!(snapshot, before);
}

#[test]
fn initial_endpoint_preserves_unknown_unobserved_and_poisoned_evidence() {
    for evidence in [ResourceEvidence::Unobserved, ResourceEvidence::Unknown] {
        let snapshot = PhysicalResourceObservation {
            expert_read_pipeline: evidence,
            ..Default::default()
        };
        let record = emit(&snapshot);
        assert_eq!(payload(&record), serde_json::to_value(evidence).unwrap());
        assert!(payload(&record).get("value").is_none());
        let binding: Value =
            serde_json::from_str(record["owner_binding"].as_str().unwrap()).unwrap();
        assert_eq!(binding["scope"], json!({"state":"unobserved"}));
    }
    let mut observed =
        materializer(ExpertReadSchedule::Blocking).expert_read_pipeline_observation();
    observed.status = "unknown_observation_poisoned";
    observed.worker_unknown = true;
    observed.stats.unknown_reads = 1;
    let mut snapshot = owner_fixture(observed);
    snapshot.image_generation = ResourceEvidence::Unknown;
    let record = emit(&snapshot);
    assert_eq!(
        payload(&record),
        serde_json::to_value(snapshot.expert_read_pipeline).unwrap()
    );
    assert_eq!(
        payload(&record)["value"]["status"],
        "unknown_observation_poisoned"
    );
    let binding: Value = serde_json::from_str(record["owner_binding"].as_str().unwrap()).unwrap();
    assert_eq!(binding["image_generation"], json!({"state":"unknown"}));
}

#[test]
fn initial_endpoint_factory_call_is_once_after_policy_and_before_callback_or_engine() {
    // CPU ordering contract for the real CUDA builder, without constructing an owner.
    let source = include_str!("../../../../src/model_factory/bf16_serving.rs");
    let builder = source
        .split_once("pub(super) fn build_numeric_bf16(")
        .unwrap()
        .1
        .split_once("// Metadata-only builds")
        .unwrap()
        .0;
    let policy = builder.find("runner.set_expert_transfer_policy(").unwrap();
    let initial = builder.find("log_initial_expert_read_endpoint(").unwrap();
    let callback = builder.find("NUMERIC_BF16_TEST_RUNNER.with(").unwrap();
    let compose = builder
        .find("ferrule_runtime::engine::compose_resident_engine(")
        .unwrap();
    assert!(policy < initial && initial < callback && callback < compose);
    assert_eq!(
        builder.matches("log_initial_expert_read_endpoint(").count(),
        1
    );
    assert!(builder[..initial].contains("if request.execution.expert_read_pipeline.is_some()"));
    assert!(builder[initial..callback].contains("|| runner.physical_resource_snapshot()"));
    assert!(builder[initial..callback].contains(")?;"));
    assert!(builder[initial..callback].contains("runner.expert_transfer_policy()"));
}

#[test]
fn initial_endpoint_preserves_omission_and_the_independent_runtime_policy() {
    let snapshot = owner_fixture(
        materializer(ExpertReadSchedule::Blocking).expert_read_pipeline_observation(),
    );
    for (requested, effective, expected_requested, expected_effective) in [
        (
            None,
            ExpertTransferPolicy::Synchronous,
            "None",
            "Synchronous",
        ),
        (
            Some(ServingExpertTransferPolicy::Synchronous),
            ExpertTransferPolicy::GroupedSynchronous,
            "Some(Synchronous)",
            "GroupedSynchronous",
        ),
    ] {
        let logs = Logs::default();
        tracing::subscriber::with_default(logs.clone(), || {
            log_initial_expert_read_endpoint(|| snapshot.clone(), requested, effective).unwrap();
        });
        let records = logs.0.lock().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0]["requested_expert_transfer_policy"],
            expected_requested
        );
        assert_eq!(records[0]["expert_transfer_policy"], expected_effective);
        assert_eq!(
            payload(&records[0]),
            serde_json::to_value(snapshot.expert_read_pipeline).unwrap()
        );
    }
}

#[test]
fn initial_endpoint_disabled_info_does_not_call_snapshot_getter() {
    use std::cell::Cell;
    let calls = Cell::new(0);
    let snapshot = owner_fixture(
        materializer(ExpertReadSchedule::Blocking).expert_read_pipeline_observation(),
    );
    let getter = || {
        calls.set(calls.get() + 1);
        snapshot.clone()
    };
    tracing::subscriber::with_default(tracing::subscriber::NoSubscriber::default(), || {
        log_initial_expert_read_endpoint(getter, None, ExpertTransferPolicy::Synchronous).unwrap();
    });
    assert_eq!(
        calls.get(),
        0,
        "disabled INFO must skip observation and JSON creation"
    );
    let logs = Logs::default();
    tracing::subscriber::with_default(logs.clone(), || {
        log_initial_expert_read_endpoint(getter, None, ExpertTransferPolicy::Synchronous).unwrap();
    });
    assert_eq!(
        calls.get(),
        1,
        "enabled INFO samples the existing owner once"
    );
    let records = logs.0.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(
        payload(&records[0]),
        serde_json::to_value(snapshot.expert_read_pipeline).unwrap()
    );
}
