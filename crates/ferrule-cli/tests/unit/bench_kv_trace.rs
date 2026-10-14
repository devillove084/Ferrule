use super::*;
use ferrule_runtime::engine::{RuntimeKvTraceHeader, RuntimeKvTraceStatus, RuntimeKvTraceUsage};

#[test]
fn exporter_rejects_lost_history_even_with_empty_complete_final() {
    let root = std::env::temp_dir().join(format!(
        "ferrule-trace-export-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let config = RuntimeKvTraceConfig {
        run_scope: "export-test".into(),
        ..Default::default()
    };
    let final_batch = RuntimeKvTraceSnapshot {
        header: RuntimeKvTraceHeader {
            schema: "ferrule-runtime-kv-trace".into(),
            schema_version: 2,
            config: config.clone(),
            page_size: 2,
            prefix_capacity_pages: 0,
            namespace: [1; 5],
        },
        start_seq: 2,
        next_seq: 2,
        lifecycle_events: vec![],
        tokens: vec![],
        status: RuntimeKvTraceStatus::default(),
        complete: true,
        finalized: true,
        usage: RuntimeKvTraceUsage {
            events: 2,
            ..Default::default()
        },
    };
    let mut capture = Capture {
        config: Some(config),
        batches: vec![final_batch],
        ..Default::default()
    };
    let output = root.join("diagnostic.json");
    assert!(
        capture
            .export(&output, true)
            .unwrap_err()
            .to_string()
            .contains("diagnostic saved")
    );
    let diagnostic: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output).unwrap()).unwrap();
    assert_eq!(diagnostic["schema"], "ferrule-runtime-kv-trace-diagnostic");
    assert_eq!(diagnostic["complete"], false);
    assert_eq!(diagnostic["batches"][0]["start_seq"], 2);
    assert_eq!(diagnostic["batches"][0]["complete"], false);
    std::fs::remove_dir_all(root).unwrap();
}
