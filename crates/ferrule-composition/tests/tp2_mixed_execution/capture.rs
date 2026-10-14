//! Candidate serialization only. Independent mapping/comparison belongs to the paper owner.
use super::*;
use serde_json::{Value, json};
use std::{fs::OpenOptions, io::Write, path::PathBuf};

pub(super) struct CandidateCapture {
    root: PathBuf,
    config: ferrule_runtime::parallel::pipeline::MixedTraceConfig,
    expected_invocations: usize,
}
pub(super) fn write_new(path: impl AsRef<std::path::Path>, value: &Value) {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    serde_json::to_writer_pretty(&mut file, value).unwrap();
    file.write_all(b"\n").unwrap();
    file.sync_all().unwrap();
}
impl CandidateCapture {
    pub(super) fn new() -> Self {
        let root = PathBuf::from(
            std::env::var("FERRULE_MIXED_EVIDENCE_DIR").expect("fresh evidence directory"),
        )
        .canonicalize()
        .unwrap();
        let oracle = PathBuf::from(std::env::var("FERRULE_MIXED_ORACLE_DIR").unwrap())
            .canonicalize()
            .unwrap();
        assert!(
            !root.starts_with(&oracle),
            "candidate must not write into oracle"
        );
        let root = root.join("candidate");
        std::fs::create_dir(&root).unwrap();
        let config = ferrule_runtime::parallel::pipeline::MixedTraceConfig::new(
            &root,
            1 << 30,
            8 << 20,
            40000,
            4,
        )
        .unwrap();
        write_new(
            root.join("contract.json"),
            &json!({
                "schema":"ferrule.mixed-candidate.v2", "oracle_manifest_sha256":std::env::var("FERRULE_MIXED_ORACLE_SHA256").unwrap(),
                "owners":[0,1],"cases":["hello","capital"],"phases":["prefill","decode.0"],
                "storage":"f32le-native-host-view", "route_weight_bits":"u32 raw F32 bits in JSON",
                "identity":"owner plus invocation begin transaction/sessions/generations/tokens/positions",
                "publication":"only parent receipt with published=true authenticates a committed candidate",
                "max_tensor_bytes":8<<20,"max_bytes_per_owner":1usize<<30,"max_events_per_owner":40000,"max_invocations":4,
                "complete_required_inventory":false,
                "max_queue_records":config.queue_record_limit().unwrap(),
                "max_metadata_bytes_per_owner":config.max_metadata_bytes_per_owner,
                "queue_policy":"entire admitted capture; no writer-throughput assumption",
                "native_metadata":["native_dtype","shape","strides","sequence","parameter_partition","attention_partition"],
                "router_probabilities":"native F32 source sink, no runtime softmax recomputation",
                "committed_KV":"post-publication native BF16 full prefix chunks; require matching kv receipt and complete owner status",
                "state_scope":"GDN operator observations provisional; committed.kv is separate retained-source export",
                "quality":"independent comparator required; raw capture success is not full-boundary acceptance"
            }),
        );
        Self {
            root,
            config,
            expected_invocations: 4,
        }
    }
    pub(super) fn for_recipe(
        root: PathBuf,
        pin: &str,
        plan: &Value,
        epoch: usize,
        invocations: usize,
        max_events: usize,
        payload: usize,
    ) -> Self {
        std::fs::create_dir(&root).unwrap();
        let config = ferrule_runtime::parallel::pipeline::MixedTraceConfig::new(
            &root,
            payload,
            8 << 20,
            max_events,
            invocations,
        )
        .unwrap();
        write_new(
            root.join("contract.json"),
            &json!({
                "schema":"ferrule.mixed-lifecycle-owner-capture.v1",
                "oracle_manifest_sha256":pin,"plan":plan,"epoch":epoch,"owners":[0,1],
                "max_invocations":invocations,"max_events_per_owner":max_events,
                "max_bytes_per_owner":payload,"max_metadata_bytes_per_owner":config.max_metadata_bytes_per_owner,
                "queue_record_limit":config.queue_record_limit().unwrap(),
                "storage":"f32le-native-host-view","complete_required_inventory":false,
                "state_scope":"provisional GDN observations joined to parent publication; retained-source committed KV"
            }),
        );
        Self {
            root,
            config,
            expected_invocations: invocations,
        }
    }
    pub(super) fn recipe_receipt(
        &self,
        id: &str,
        phase: &str,
        session: SessionId,
        tokens: &[u32],
        result: &ferrule_common::Result<ferrule_runtime::parallel::pipeline::PipelineOutput>,
    ) {
        write_new(
            self.root.join(format!("{id}.receipt.json")),
            &json!({
                "case":id,"phase":phase,"session":session.0,"input_token_ids":tokens,
                "published":result.is_ok(),"transaction":result.as_ref().ok().map(|o|o.transaction.get()),
                "error":result.as_ref().err().map(|e|e.to_string())
            }),
        );
    }
    pub(super) fn recipe_kv_receipt(
        &self,
        id: &str,
        phase: &str,
        session: SessionId,
        transaction: ferrule_common::execution::ExecutionTransactionId,
        result: &ferrule_common::Result<()>,
    ) {
        write_new(
            self.root.join(format!("{id}.kv-receipt.json")),
            &json!({
                "schema":"ferrule.mixed-committed-kv-receipt.v1","case":id,"phase":phase,
                "session":session.0,"transaction":transaction.get(),"owners":[0,1],
                "export_completed":result.is_ok(),"writer_drain_required":true,
                "error":result.as_ref().err().map(|e|e.to_string())
            }),
        );
    }
    pub(super) fn config(&self) -> ferrule_runtime::parallel::pipeline::MixedTraceConfig {
        self.config.clone()
    }
    pub(super) fn receipt(
        &self,
        case: usize,
        phase: &str,
        session: SessionId,
        tokens: &[u32],
        result: &ferrule_common::Result<ferrule_runtime::parallel::pipeline::PipelineOutput>,
    ) {
        let id = ["hello", "capital"][case];
        write_new(
            self.root.join(format!("{id}.{phase}.receipt.json")),
            &json!({
                "case":id,"phase":phase,"session":session.0,"input_token_ids":tokens,
                "published":result.is_ok(),"transaction":result.as_ref().ok().map(|o|o.transaction.get()),
                "error":result.as_ref().err().map(|e|e.to_string())
            }),
        );
    }
    pub(super) fn kv_receipt(
        &self,
        case: usize,
        phase: &str,
        session: SessionId,
        transaction: ferrule_common::execution::ExecutionTransactionId,
        result: &ferrule_common::Result<()>,
    ) {
        let id = ["hello", "capital"][case];
        write_new(
            self.root.join(format!("{id}.{phase}.kv-receipt.json")),
            &json!({
                "schema":"ferrule.mixed-committed-kv-receipt.v1",
                "case":id,"phase":phase,"session":session.0,"transaction":transaction.get(),
                "owners":[0,1],"export_completed":result.is_ok(),
                "writer_drain_required":true,"error":result.as_ref().err().map(|e|e.to_string())
            }),
        );
    }
    pub(super) fn abort(&self) {
        self.config.abort_capture();
    }
    pub(super) fn finish(&self) -> bool {
        if self
            .config
            .wait_for_writers(std::time::Duration::from_secs(10))
            .is_err()
        {
            return false;
        }
        (0..2).all(|owner| {
            let path = self.root.join(format!("rank-{owner}/capture-status.json"));
            let status = std::fs::read(path)
                .ok()
                .and_then(|b| serde_json::from_slice::<Value>(&b).ok());
            status.is_some_and(|s| {
                s["complete_available_trace"] == true
                    && s["required_inventory_complete"] == false
                    && s["invocations"] == self.expected_invocations
            })
        })
    }
}

impl Drop for CandidateCapture {
    fn drop(&mut self) {
        self.config.abort_capture();
    }
}
