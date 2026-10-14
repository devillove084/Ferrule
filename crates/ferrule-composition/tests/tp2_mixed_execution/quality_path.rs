//! Test-only claim separation; no tensor observer or GPU readback is installed here.
use super::*;
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum QualityPath {
    DiagnosticFallback,
    UnobservedFused,
}
impl QualityPath {
    pub(super) fn from_capture(capture: bool) -> Self {
        if capture {
            Self::DiagnosticFallback
        } else {
            Self::UnobservedFused
        }
    }
    pub(super) fn captures(self) -> bool {
        self == Self::DiagnosticFallback
    }
    pub(super) fn contract(self, cache: Option<usize>) -> Value {
        json!({
            "schema":"ferrule.paper.routed-fusion-execution-path.v1",
            "path":if self.captures() {"diagnostic_fallback"} else {"unobserved_fused"},
            "numeric_observer_installed":self.captures(),
            "committed_tensor_export":self.captures(),
            "cache_grants_per_owner":cache,
            "quality_scope":if self.captures() {"original full boundary/state inventory; not fused tail"}
                else {"independent finite full-vocabulary logits and tokens; host publication/frontier checks; no internal tensor equality claim"},
            "device_cap_bytes":22u64<<30,
            "atol":0.00025,"rtol":0.012,
        })
    }
}
pub(super) fn frontier(pipeline: &PipelineParallelExecutor, session: SessionId) -> usize {
    pipeline
        .session_slot(session)
        .and_then(|slot| pipeline.page_manager().block_table(slot))
        .map_or(0, |table| table.committed_tokens())
}
fn publication_matches(
    before: usize,
    after: usize,
    start: usize,
    rows: usize,
    committed: usize,
) -> bool {
    rows > 0 && before.checked_add(1) == Some(after) && start.checked_add(rows) == Some(committed)
}
pub(super) fn publication(
    pipeline: &PipelineParallelExecutor,
    session: SessionId,
    before: usize,
    start: usize,
    rows: usize,
) -> Value {
    let after = pipeline.coordinator().publication_count();
    let committed = frontier(pipeline, session);
    assert!(publication_matches(before, after, start, rows, committed));
    assert!(!pipeline.is_quarantined());
    assert_eq!(pipeline.outstanding(), 0);
    json!({"schema":"ferrule.paper.unobserved-publication.v1", "session":session.0,
        "publications_before":before,"publications_after":after,"position_start":start,
        "committed_tokens":committed,"submitted_rows":rows,"outstanding":0,
        "quarantined":false,"tensor_readback":false})
}
#[test]
fn diagnostic_capture_never_claims_fused_quality() {
    let value = QualityPath::from_capture(true).contract(Some(3 << 30));
    assert_eq!(value["path"], "diagnostic_fallback");
    assert_eq!(value["numeric_observer_installed"], true);
    assert_eq!(value["committed_tensor_export"], true);
}
#[test]
fn unobserved_cache_contract_installs_no_tensor_observer() {
    let path = QualityPath::from_capture(false);
    assert!(!path.captures());
    let value = path.contract(Some(3 << 30));
    assert_eq!(value["path"], "unobserved_fused");
    assert_eq!(value["numeric_observer_installed"], false);
    assert_eq!(value["committed_tensor_export"], false);
    assert_eq!(value["cache_grants_per_owner"], 3u64 << 30);
}
#[test]
fn publication_requires_one_commit_and_exact_frontier() {
    assert!(publication_matches(9, 10, 5, 2, 7));
    for args in [
        (9, 9, 5, 2, 7),
        (9, 11, 5, 2, 7),
        (9, 10, 5, 2, 6),
        (9, 10, 5, 0, 5),
        (usize::MAX, 0, 5, 2, 7),
        (9, 10, usize::MAX, 1, 0),
    ] {
        assert!(!publication_matches(args.0, args.1, args.2, args.3, args.4));
    }
}
