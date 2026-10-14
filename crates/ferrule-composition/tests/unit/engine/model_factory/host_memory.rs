use super::*;

// Counter parsing and cgroup ancestry contracts are tested by the canonical
// backend probe. Composition only owns the model-source error boundary.
#[test]
fn canonical_probe_failure_preserves_model_source_and_reason() {
    let error = map_probe_error(HostMemoryProbeError {
        reason: "counter/path probe detail".into(),
    });
    let ferrule_common::Error::ModelSource { source } = error else {
        panic!("probe failure must remain a model source error");
    };
    let source = source.downcast_ref::<HostExpertCacheError>().unwrap();
    assert!(matches!(
        source,
        HostExpertCacheError::MemoryProbe { reason } if reason == "counter/path probe detail"
    ));
}
