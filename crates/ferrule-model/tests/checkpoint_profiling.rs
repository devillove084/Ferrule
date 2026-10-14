//! Host-only worker-span boundaries and per-submission dispatcher ownership.
//! A small tracing Subscriber avoids adding a model dependency on an SDK or
//! tracing-subscriber. These tests also run without nvtx-profile.
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::ThreadId;

use ferrule_common::SourceGeneration;
use ferrule_model::checkpoint::{
    CheckpointReadCounters, CheckpointReadExtent, CheckpointReadPipeline, CheckpointReadPlan,
    CheckpointSourceFileIdentity, CheckpointTensorReader, ReadOutcome, ReadPipelineLimits,
};
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing::{Event, Metadata, Subscriber};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    directory: PathBuf,
    path: PathBuf,
    plan: CheckpointReadPlan,
}

impl Fixture {
    fn new() -> Self {
        let directory = std::env::temp_dir().join(format!(
            "ferrule-checkpoint-profile-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("private-checkpoint.bin");
        std::fs::write(&path, (0..32u8).collect::<Vec<_>>()).unwrap();
        let snapshot = CheckpointSourceFileIdentity::capture(&path).unwrap();
        let plan = CheckpointReadPlan::new(
            [
                CheckpointReadExtent::new(path.clone(), 10, 4).unwrap(),
                CheckpointReadExtent::new(path.clone(), 1, 4).unwrap(),
            ],
            Arc::<[CheckpointSourceFileIdentity]>::from([snapshot]),
        )
        .unwrap();
        Self {
            directory,
            path,
            plan,
        }
    }

    fn pipeline(&self, reader: &CheckpointTensorReader) -> CheckpointReadPipeline {
        CheckpointReadPipeline::new(
            reader.clone(),
            ReadPipelineLimits {
                max_host_bytes: 16,
                max_read_slots: 2,
            },
        )
        .unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.directory).unwrap();
    }
}

#[derive(Clone, Default)]
struct Fields(BTreeMap<String, u64>);

impl Visit for Fields {
    fn record_u64(&mut self, field: &Field, value: u64) {
        self.0.insert(field.name().into(), value);
    }
    fn record_debug(&mut self, _: &Field, _: &dyn std::fmt::Debug) {
        panic!("worker span must contain only numeric fields");
    }
}

struct Transition {
    entered: bool,
    thread: ThreadId,
    thread_name: Option<String>,
    fields: Fields,
    counters: CheckpointReadCounters,
}

struct ProbeState {
    reader: CheckpointTensorReader,
    truncate_on_enter: Option<PathBuf>,
    next: AtomicU64,
    spans: Mutex<HashMap<u64, Fields>>,
    transitions: Mutex<Vec<Transition>>,
}

#[derive(Clone)]
struct Probe(Arc<ProbeState>);

impl Probe {
    fn new(reader: &CheckpointTensorReader, truncate_on_enter: Option<PathBuf>) -> Self {
        Self(Arc::new(ProbeState {
            reader: reader.clone(),
            truncate_on_enter,
            next: AtomicU64::new(1),
            spans: Mutex::new(HashMap::new()),
            transitions: Mutex::new(Vec::new()),
        }))
    }

    fn transition(&self, id: &Id, entered: bool) {
        let fields = self.0.spans.lock().unwrap()[&id.into_u64()].clone();
        self.0.transitions.lock().unwrap().push(Transition {
            entered,
            thread: std::thread::current().id(),
            thread_name: std::thread::current().name().map(str::to_owned),
            fields,
            counters: self.0.reader.read_counters(),
        });
        if entered && let Some(path) = &self.0.truncate_on_enter {
            // Inject a real pread failure after session opening, not a failed
            // open mislabeled as I/O. This mutation is test-only.
            std::fs::OpenOptions::new()
                .write(true)
                .open(path)
                .unwrap()
                .set_len(0)
                .unwrap();
        }
    }
}

impl Subscriber for Probe {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.is_span()
            && metadata.target() == "ferrule_profile"
            && metadata.name() == "read_subset"
    }

    fn new_span(&self, attributes: &Attributes<'_>) -> Id {
        let id = self.0.next.fetch_add(1, Ordering::Relaxed);
        let mut fields = Fields::default();
        attributes.record(&mut fields);
        self.0.spans.lock().unwrap().insert(id, fields);
        Id::from_u64(id)
    }

    fn record(&self, _: &Id, _: &Record<'_>) {}
    fn record_follows_from(&self, _: &Id, _: &Id) {}
    fn event(&self, _: &Event<'_>) {}
    fn enter(&self, id: &Id) {
        self.transition(id, true);
    }
    fn exit(&self, id: &Id) {
        self.transition(id, false);
    }
    fn try_close(&self, id: Id) -> bool {
        self.0.spans.lock().unwrap().remove(&id.into_u64());
        true
    }
}

#[test]
fn worker_range_encloses_pread_not_session_open_or_post_verification() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(16);
    let mut pipeline = fixture.pipeline(&reader); // Spawned outside the subscriber.
    let probe = Probe::new(&reader, None);
    let ticket = tracing::subscriber::with_default(probe.clone(), || {
        pipeline
            .submit(SourceGeneration::new(7), fixture.plan.clone())
            .unwrap()
    });
    pipeline.wait(&ticket).unwrap();
    let completion = pipeline.take(&ticket).unwrap();
    assert!(
        matches!(completion.outcome(), ReadOutcome::SourceVerified(bytes) if bytes == &[10, 11, 12, 13, 1, 2, 3, 4])
    );
    let transitions = probe.0.transitions.lock().unwrap();
    assert_eq!(transitions.len(), 2);
    let (enter, exit) = (&transitions[0], &transitions[1]);
    assert!(enter.entered);
    assert!(!exit.entered);
    assert_eq!(enter.thread, exit.thread);
    assert_ne!(enter.thread, std::thread::current().id());
    assert_eq!(enter.thread_name.as_deref(), Some("checkpoint-read"));
    let fields = BTreeMap::from([
        ("operation".into(), ticket.operation().get()),
        ("source_generation".into(), 7),
        ("requested_bytes".into(), 8),
    ]);
    assert_eq!(enter.fields.0, fields);
    assert_eq!(exit.fields.0, fields);
    assert_eq!(enter.counters.open_calls, 1);
    assert!(enter.counters.source_checks > 0);
    assert_eq!(enter.counters.read_calls, 0);
    assert_eq!(exit.counters.read_calls, 2);
    assert_eq!(exit.counters.bytes_read, 8);
    assert_eq!(enter.counters.source_checks, exit.counters.source_checks);
    assert_eq!(
        enter.counters.path_metadata_calls,
        exit.counters.path_metadata_calls
    );
    assert_eq!(
        enter.counters.handle_metadata_calls,
        exit.counters.handle_metadata_calls
    );
    assert!(reader.read_counters().source_checks > exit.counters.source_checks);
    assert!(completion.receipt().source_verified_at.is_some());
}

#[test]
fn each_submission_propagates_its_own_scoped_dispatcher() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(16);
    let mut pipeline = fixture.pipeline(&reader);
    let first = Probe::new(&reader, None);
    let second = Probe::new(&reader, None);
    for (probe, generation) in [(&first, 7), (&second, 8)] {
        let ticket = tracing::subscriber::with_default(probe.clone(), || {
            pipeline
                .submit(SourceGeneration::new(generation), fixture.plan.clone())
                .unwrap()
        });
        pipeline.wait(&ticket).unwrap();
        let completion = pipeline.take(&ticket).unwrap();
        assert!(matches!(
            completion.outcome(),
            ReadOutcome::SourceVerified(_)
        ));
        drop(completion);
        let transitions = probe.0.transitions.lock().unwrap();
        assert_eq!(transitions.len(), 2);
        assert_eq!(transitions[0].fields.0["source_generation"], generation);
        assert_eq!(
            transitions[0].fields.0["operation"],
            ticket.operation().get()
        );
    }
    let first = first.0.transitions.lock().unwrap();
    let second = second.0.transitions.lock().unwrap();
    assert_eq!(first[0].thread, second[0].thread);
    assert_ne!(
        first[0].fields.0["operation"],
        second[0].fields.0["operation"]
    );
    assert_eq!(pipeline.budget().reserved_bytes, 0);
    assert_eq!(pipeline.budget().reserved_slots, 0);
}

#[test]
fn failed_session_open_emits_no_read_range() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(16);
    let mut pipeline = fixture.pipeline(&reader);
    let probe = Probe::new(&reader, None);
    std::fs::remove_file(&fixture.path).unwrap();
    let ticket = tracing::subscriber::with_default(probe.clone(), || {
        pipeline
            .submit(SourceGeneration::new(7), fixture.plan.clone())
            .unwrap()
    });
    pipeline.wait(&ticket).unwrap();
    assert!(matches!(
        pipeline.take(&ticket).unwrap().outcome(),
        ReadOutcome::Failed(_)
    ));
    assert!(probe.0.transitions.lock().unwrap().is_empty());
    assert_eq!(reader.read_counters().read_calls, 0);
}

#[test]
fn actual_read_failure_still_exits_the_worker_range() {
    let fixture = Fixture::new();
    let reader = CheckpointTensorReader::new(16);
    let mut pipeline = fixture.pipeline(&reader);
    let probe = Probe::new(&reader, Some(fixture.path.clone()));
    let ticket = tracing::subscriber::with_default(probe.clone(), || {
        pipeline
            .submit(SourceGeneration::new(7), fixture.plan.clone())
            .unwrap()
    });
    pipeline.wait(&ticket).unwrap();
    let completion = pipeline.take(&ticket).unwrap();
    assert!(matches!(completion.outcome(), ReadOutcome::Failed(_)));
    assert!(completion.receipt().io_started_at.is_some());
    assert!(completion.receipt().io_finished_at.is_some());
    let transitions = probe.0.transitions.lock().unwrap();
    assert_eq!(transitions.len(), 2);
    assert!(transitions[0].entered);
    assert!(!transitions[1].entered);
    assert_eq!(transitions[0].counters.open_calls, 1);
    assert_eq!(transitions[0].counters.read_calls, 0);
    assert!(transitions[1].counters.read_calls > 0);
    assert_eq!(transitions[1].counters.bytes_read, 0);
}
