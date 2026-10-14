//! Standalone integration probe for callers that already depend on serde_json.
//! Not part of common's module graph: common intentionally has no JSON dependency.
//! Compile with rustc --test and freshly built common / serde_json --extern paths.
use ferrule_common::kvset::*;

const TRACE: &str = include_str!("../../fixtures/kvset/kvset_trace_v1.json");
const CONFIG: &str = include_str!("../../fixtures/kvset/kvset_config_v1.json");

#[test]
fn json_fixtures_roundtrip_and_replay() {
    let trace: TraceInput = serde_json::from_str(TRACE).unwrap();
    let config: AnalyzerConfig = serde_json::from_str(CONFIG).unwrap();
    let report = analyze(&trace, config.clone()).unwrap();
    assert_eq!(
        report
            .capacity_points
            .iter()
            .map(|p| (p.capacity_pages, p.page_hits, p.contiguous_prefix_tokens))
            .collect::<Vec<_>>(),
        [(0, 0, 0), (1, 1, 0), (2, 3, 5), (3, 4, 6), (80, 4, 6)]
    );
    assert_eq!(
        serde_json::from_str::<TraceInput>(&serde_json::to_string(&trace).unwrap()).unwrap(),
        trace
    );
    assert_eq!(
        serde_json::from_str::<AnalyzerConfig>(&serde_json::to_string(&config).unwrap()).unwrap(),
        config
    );
    assert_eq!(
        serde_json::from_str::<AnalysisReport>(&serde_json::to_string(&report).unwrap()).unwrap(),
        report
    );
    let header_line = serde_json::to_string(&trace.header).unwrap();
    let mut analyzer = Analyzer::new(serde_json::from_str(&header_line).unwrap(), config).unwrap();
    for event in &trace.events {
        let line = serde_json::to_string(event).unwrap();
        assert!(!line.contains("\"Demand\""));
        analyzer
            .ingest(&serde_json::from_str::<TraceEvent>(&line).unwrap())
            .unwrap();
    }
    assert_eq!(analyzer.finish().unwrap(), report);
}

#[test]
fn strict_json_and_null_report_contract() {
    for bad in [
        r#"{"seq":0,"seq":1,"type":"demand","request":{"id":"r","generation":0},"namespace_id":"n","prompt_tokens":[],"reuse_limit":"all_prompt_tokens"}"#,
        r#"{"seq":0,"type":"demand","request":{"id":"r","generation":0},"namespace_id":"n","prompt_tokens":[],"reuse_limit":"all_prompt_tokens","unknown":0}"#,
        r#"{"seq":0,"type":"demand","request":{"id":"r","generation":0},"prompt_tokens":[],"reuse_limit":"all_prompt_tokens"}"#,
        r#"{"seq":0,"type":"demand","request":{"id":"r","generation":0},"namespace_id":"n","prompt_tokens":[1.0],"reuse_limit":"all_prompt_tokens"}"#,
    ] {
        assert!(serde_json::from_str::<TraceEvent>(bad).is_err());
    }
    assert!(serde_json::from_str::<Ratio>("0.95").is_err());
    let mut trace: TraceInput = serde_json::from_str(TRACE).unwrap();
    trace.events.clear();
    let report = analyze(&trace, AnalyzerConfig::default()).unwrap();
    let json = serde_json::to_value(&report).unwrap();
    assert!(json["capacity_points"][0]["page_reference_hit_rate"].is_null());
    assert_eq!(
        json["target_capacities"][0]["requirement"]["status"],
        "undefined"
    );
}
