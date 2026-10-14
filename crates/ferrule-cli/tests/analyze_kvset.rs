use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use ferrule_common::kvset::{AnalysisReport, AnalyzerConfig, TraceInput, analyze};
use ferrule_runtime::engine::{
    RuntimeKvTraceSnapshot, project_runtime_kv_trace, replay_runtime_prefix_lru,
};
use serde_json::{Value, json};

const TRACE: &str = include_str!("../../ferrule-common/tests/fixtures/kvset/kvset_trace_v1.json");
const CONFIG: &str = include_str!("../../ferrule-common/tests/fixtures/kvset/kvset_config_v1.json");

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "ferrule-analyze-kvset-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn write(&self, name: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, bytes).unwrap();
        path
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn invoke(args: &[&str], stdin: &[u8]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ferrule"))
        .arg("analyze-kvset")
        .args(args)
        .env("FERRULE_LOG", "trace")
        .env("FERRULE_LOG_FORMAT", "json")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut pipe = child.stdin.take().unwrap();
    let bytes = stdin.to_vec();
    let writer = std::thread::spawn(move || {
        // Early argument/resource rejection can close stdin before the producer finishes.
        let _ = pipe.write_all(&bytes);
    });
    let output = child.wait_with_output().unwrap();
    writer.join().unwrap();
    output
}
fn success(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn failure(output: Output, contains: &str) {
    assert!(
        !output.status.success(),
        "unexpected success: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(output.stdout.is_empty(), "partial report on stdout");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(contains),
        "expected {contains:?}, got {stderr}"
    );
}
fn run_json(input: &Value, extra: &[&str]) -> Output {
    let mut args = vec!["--trace", "-", "--json"];
    args.extend_from_slice(extra);
    invoke(&args, &serde_json::to_vec(input).unwrap())
}
fn fixture() -> Value {
    serde_json::from_str(TRACE).unwrap()
}
fn jsonl(value: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(&value["header"]).unwrap();
    for event in value["events"].as_array().unwrap() {
        bytes.push(b'\n');
        bytes.extend(serde_json::to_vec(event).unwrap());
    }
    bytes
}
fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn json_jsonl_roundtrip_matches_common_fixture_and_report_schema() {
    let temp = Temp::new();
    let config = temp.write("config.json", CONFIG);
    let trace_path = temp.write("trace.json", TRACE);
    let input: TraceInput = serde_json::from_str(TRACE).unwrap();
    let config_dto: AnalyzerConfig = serde_json::from_str(CONFIG).unwrap();
    let expected = analyze(&input, config_dto).unwrap();
    let expected_json = serde_json::to_value(&expected).unwrap();
    let json_report = success(invoke(
        &[
            "--trace",
            path(&trace_path),
            "--config",
            path(&config),
            "--json",
        ],
        &[],
    ));
    let mut reports = vec![json_report];
    for trailing_newline in [false, true] {
        let mut bytes = jsonl(&fixture());
        if trailing_newline {
            bytes.push(b'\n');
        }
        reports.push(success(invoke(
            &[
                "--trace",
                "-",
                "--format",
                "jsonl",
                "--config",
                path(&config),
                "--json",
            ],
            &bytes,
        )));
    }
    for report in reports {
        assert_eq!(report["schema"], "ferrule-kvset-report");
        assert_eq!(report["schema_version"], 1);
        assert_eq!(report["trace_schema_version"], 1);
        assert_eq!(report["trace_semantics_version"], 1);
        assert_eq!(report["page_size_tokens"], 2);
        assert_eq!(report["capacity_unit"], "equal_charge_logical_pages");
        assert_eq!(report["analysis"], expected_json);
        let roundtrip: AnalysisReport = serde_json::from_value(report["analysis"].clone()).unwrap();
        assert_eq!(roundtrip, expected);
        assert_eq!(report["config"], expected_json["config"]);
        assert!(report["runtime_actual_entry_lru"].is_null());
        assert!(report["replay"]["input_bytes"].as_u64().unwrap() > 0);
        assert!(report["replay"]["elapsed_ns"].is_number());
        assert!(
            report["replay"]["rss_scope"]
                .as_str()
                .unwrap()
                .contains("Process lifetime")
        );
    }
}

#[test]
fn exact_ratios_overrides_windows_and_config_stdin() {
    let temp = Temp::new();
    let trace = temp.write("trace.json", TRACE);
    let report = success(invoke(
        &[
            "--trace",
            path(&trace),
            "--config",
            "-",
            "--json",
            "--capacity",
            "0",
            "--capacity",
            "2",
            "--alpha",
            "2/2",
            "--coverage",
            "18446744073709551615/18446744073709551615",
            "--window-start",
            "2",
            "--window-end",
            "5",
            "--capacity-budget",
            "2",
            "--target-basis",
            "eligible-tokens",
            "--denominator",
            "all-demands",
        ],
        CONFIG.as_bytes(),
    ));
    let mut config: AnalyzerConfig = serde_json::from_str(CONFIG).unwrap();
    config.capacities = Some(vec![0, 2]);
    config.alphas = vec![ferrule_common::kvset::Ratio::new(1, 1).unwrap()];
    config.p_values = config.alphas.clone();
    config.capacity_budget = Some(2);
    config.target_basis = ferrule_common::kvset::TargetBasis::EligibleTokens;
    config.coverage_denominator = ferrule_common::kvset::CoverageDenominator::AllDemands;
    assert_eq!(
        report["analysis"],
        serde_json::to_value(analyze(&serde_json::from_str(TRACE).unwrap(), config).unwrap())
            .unwrap()
    );
    for bad in [
        "0.95",
        "1/0",
        "2/1",
        "18446744073709551616/18446744073709551616",
        "1/2/3",
        "+1/2",
        "1/",
    ] {
        failure(run_json(&fixture(), &["--alpha", bad]), "exact a/b");
        failure(run_json(&fixture(), &["--coverage", bad]), "exact a/b");
    }
    failure(
        run_json(&fixture(), &["--window-start", "1"]),
        "--window-end",
    );
    failure(
        run_json(&fixture(), &["--window-start", "5", "--window-end", "4"]),
        "inverted measurement window",
    );
    failure(
        invoke(&["--trace", "-", "--config", "-", "--json"], &[]),
        "cannot both read stdin",
    );
}

#[test]
fn capacity_zero_empty_trace_and_pending_eof_preserve_null_and_undefined() {
    let mut trace = fixture();
    trace["events"] = json!([]);
    let report = success(run_json(&trace, &["--capacity", "0"]));
    let point = &report["analysis"]["capacity_points"][0];
    assert_eq!(point["capacity_pages"], 0);
    assert_eq!(point["page_hits"], 0);
    for field in [
        "page_reference_hit_rate",
        "contiguous_prefix_token_hit_rate_theoretical",
        "contiguous_prefix_token_hit_rate_eligible",
        "contiguous_prefix_token_hit_rate_prompt",
    ] {
        assert!(point[field].is_null(), "{field}");
    }
    assert_eq!(
        report["analysis"]["target_capacities"][0]["requirement"],
        json!({"status":"undefined"})
    );
    assert!(
        report["analysis"]["request_target_coverage"][0]["points"][0]["coverage_opportunity"]
            .is_null()
    );
    trace["events"] = json!([fixture()["events"][0].clone()]);
    for format in ["json", "jsonl"] {
        let bytes = if format == "jsonl" {
            jsonl(&trace)
        } else {
            serde_json::to_vec(&trace).unwrap()
        };
        let report = success(invoke(
            &[
                "--trace",
                "-",
                "--format",
                format,
                "--json",
                "--capacity",
                "0",
            ],
            &bytes,
        ));
        assert_eq!(report["analysis"]["outcomes"]["abandoned_at_trace_end"], 1);
        assert_eq!(report["analysis"]["outcomes"]["pending"], 0);
        assert_eq!(
            report["analysis"]["target_capacities"][0]["requirement"]["status"],
            "undefined"
        );
    }
}

#[test]
fn unreachable_beyond_budget_and_deferred_sample_contracts() {
    let report = success(run_json(
        &fixture(),
        &[
            "--target-basis",
            "full-prompt",
            "--denominator",
            "all-demands",
            "--coverage",
            "1/1",
        ],
    ));
    assert_eq!(
        report["analysis"]["target_capacities"][0]["requirement"]["status"],
        "unreachable"
    );
    let report = success(run_json(
        &fixture(),
        &[
            "--window-start",
            "2",
            "--window-end",
            "5",
            "--capacity-budget",
            "2",
            "--coverage",
            "1/1",
        ],
    ));
    assert_eq!(
        report["analysis"]["target_capacities"][0]["requirement"],
        json!({"status":"beyond_budget","budget":2})
    );
    let mut trace = fixture();
    trace["header"]["recency_model"] = json!("deferred_admission_prefix_lru_v1");
    failure(
        run_json(&trace, &[]),
        "deferred requires explicit capacities",
    );
    failure(
        run_json(&trace, &["--capacity", "2", "--capacity-budget", "1"]),
        "deferred budget must be sampled",
    );
    let extra = ["--capacity", "2", "--coverage", "0/1"];
    let report = success(run_json(&trace, &extra));
    assert_eq!(
        report["analysis"]["target_capacities"][0]["requirement"],
        json!({"status":"exact","capacity":0})
    );
    let mut args = vec!["--trace", "-", "--format", "jsonl", "--json"];
    args.extend(extra);
    assert_eq!(
        success(invoke(&args, &jsonl(&trace)))["analysis"],
        report["analysis"]
    );
}

#[test]
fn strict_common_schema_rejects_unknown_duplicate_missing_and_malformed_fields() {
    let mut bad = Vec::new();
    let mut v = fixture();
    v["unknown"] = json!(1);
    bad.push(v);
    let mut v = fixture();
    v["header"]["unknown"] = json!(1);
    bad.push(v);
    let mut v = fixture();
    v["header"]["namespaces"][0]["namespace"]["secret-token-value"] = json!(1);
    bad.push(v);
    let mut v = fixture();
    v["events"][0]["request"]["unknown"] = json!(1);
    bad.push(v);
    let mut v = fixture();
    v["events"][0]["unknown"] = json!(1);
    bad.push(v);
    let mut v = fixture();
    v["events"][1]["pages"][0]["unknown"] = json!(1);
    bad.push(v);
    let mut v = fixture();
    v["events"][0]
        .as_object_mut()
        .unwrap()
        .remove("reuse_limit");
    bad.push(v);
    let mut v = fixture();
    v["events"][0]["prompt_tokens"][0] = json!(1.0);
    bad.push(v);
    let mut v = fixture();
    v["events"][0]["reuse_limit"] = json!("secret-token-value");
    bad.push(v);
    for value in bad {
        let output = run_json(&value, &[]);
        assert!(!String::from_utf8_lossy(&output.stderr).contains("secret-token-value"));
        failure(output, "JSON Data error");
    }
    for raw in [
        TRACE.replacen("\"seq\":0", "\"seq\":0,\"seq\":0", 1),
        TRACE.replacen("\"header\": {", "\"header\": {\"page_size\":2,", 1),
        format!("{TRACE} {{}}"),
        TRACE[..TRACE.len() - 3].to_owned(),
    ] {
        failure(invoke(&["--trace", "-", "--json"], raw.as_bytes()), "JSON");
    }
    for (field, value) in [
        ("schema_version", json!(99)),
        ("semantics_version", json!(99)),
        ("schema", json!("unknown")),
        ("page_size", json!(0)),
    ] {
        let mut v = fixture();
        v["header"][field] = value;
        failure(run_json(&v, &[]), "invalid KVSET trace");
    }
    let mut v = fixture();
    v["events"][1]["seq"] = json!(0);
    failure(run_json(&v, &[]), "seq must strictly increase");
}

#[test]
fn jsonl_rejects_blank_wrapped_bad_and_truncated_records_without_partial_report() {
    let good = jsonl(&fixture());
    for bytes in [
        Vec::new(),
        b"\n".to_vec(),
        format!("{}\n\n", String::from_utf8_lossy(&good)).into_bytes(),
        format!("{}\n{{", String::from_utf8_lossy(&good)).into_bytes(),
        serde_json::to_vec(&json!({"header":fixture()["header"]})).unwrap(),
    ] {
        let output = invoke(&["--trace", "-", "--format", "jsonl", "--json"], &bytes);
        failure(output, "JSONL");
    }
    let mut v = fixture();
    v["events"][0]["unknown"] = json!(true);
    failure(
        invoke(&["--trace", "-", "--format", "jsonl", "--json"], &jsonl(&v)),
        "JSON Data error",
    );
}

#[test]
fn input_and_line_byte_limits_include_newlines_and_accept_exact_boundary() {
    let trace = serde_json::to_vec(&fixture()).unwrap();
    let exact = trace.len().to_string();
    success(invoke(
        &["--trace", "-", "--json", "--max-input-bytes", &exact],
        &trace,
    ));
    failure(
        invoke(
            &[
                "--trace",
                "-",
                "--json",
                "--max-input-bytes",
                &(trace.len() - 1).to_string(),
            ],
            &trace,
        ),
        "input byte limit",
    );
    failure(
        invoke(
            &["--trace", "-", "--json", "--max-input-bytes", "0"],
            b"not json",
        ),
        "input byte limit",
    );
    let mut lines = jsonl(&fixture());
    lines.push(b'\n');
    let largest = lines
        .split_inclusive(|b| *b == b'\n')
        .map(<[u8]>::len)
        .max()
        .unwrap();
    let report = success(invoke(
        &[
            "--trace",
            "-",
            "--format",
            "jsonl",
            "--json",
            "--max-input-bytes",
            &lines.len().to_string(),
            "--max-line-bytes",
            &largest.to_string(),
        ],
        &lines,
    ));
    assert_eq!(report["replay"]["input_bytes"], lines.len());
    assert_eq!(report["replay"]["largest_input_record_bytes"], largest);
    failure(
        invoke(
            &[
                "--trace",
                "-",
                "--format",
                "jsonl",
                "--json",
                "--max-line-bytes",
                &(largest - 1).to_string(),
            ],
            &lines,
        ),
        "line byte limit",
    );
    failure(
        invoke(
            &[
                "--trace",
                "-",
                "--format",
                "jsonl",
                "--json",
                "--max-input-bytes",
                &(lines.len() - 1).to_string(),
            ],
            &lines,
        ),
        "input byte limit",
    );
    let huge = vec![b' '; 32_769];
    failure(
        invoke(
            &[
                "--trace",
                "-",
                "--format",
                "jsonl",
                "--json",
                "--max-line-bytes",
                "8192",
            ],
            &huge,
        ),
        "line byte limit",
    );
}

#[test]
fn config_byte_limits_and_strict_config_schema() {
    let temp = Temp::new();
    let trace = temp.write("trace.json", TRACE);
    success(invoke(
        &[
            "--trace",
            path(&trace),
            "--config",
            "-",
            "--json",
            "--max-config-bytes",
            &CONFIG.len().to_string(),
        ],
        CONFIG.as_bytes(),
    ));
    failure(
        invoke(
            &[
                "--trace",
                path(&trace),
                "--config",
                "-",
                "--json",
                "--max-config-bytes",
                &(CONFIG.len() - 1).to_string(),
            ],
            CONFIG.as_bytes(),
        ),
        "input byte limit",
    );
    for config in [
        r#"{"unknown":1}"#,
        r#"{"alphas":[],"alphas":[]}"#,
        r#"{"alphas":[{"numerator":1,"denominator":0}]}"#,
        r#"{"alphas":[{"numerator":1,"denominator":2,"unknown":0}]}"#,
        r#"{"capacity_budget":-1}"#,
    ] {
        let output = invoke(
            &["--trace", path(&trace), "--config", "-", "--json"],
            config.as_bytes(),
        );
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn analyzer_resource_limits_are_enforced_and_untrusted_limits_cannot_raise_ceiling() {
    for (field, limit) in [
        ("max_events", 0),
        ("max_publish_positions", 0),
        ("max_tokens_per_event", 1),
        ("max_histogram_entries", 0),
        ("max_report_points", 0),
        ("max_pending_tokens", 1),
        ("max_unique_keys", 0),
    ] {
        let mut trace = fixture();
        trace["header"]["limits"] = json!({field:limit});
        for format in ["json", "jsonl"] {
            let bytes = if format == "jsonl" {
                jsonl(&trace)
            } else {
                serde_json::to_vec(&trace).unwrap()
            };
            failure(
                invoke(&["--trace", "-", "--format", format, "--json"], &bytes),
                "resource limit",
            );
        }
    }
    let mut trace = fixture();
    trace["header"]["limits"] =
        json!({"max_total_capacity_entries":u64::MAX, "max_events":u64::MAX});
    let report = success(run_json(&trace, &[]));
    assert_eq!(report["effective_resource_limits"]["max_events"], 1_000_000);
    trace["header"]["recency_model"] = json!("deferred_admission_prefix_lru_v1");
    failure(
        run_json(&trace, &["--capacity", "1000001"]),
        "finite cache entries",
    );
    failure(
        run_json(
            &trace,
            &["--capacity", "18446744073709551615", "--capacity", "2"],
        ),
        "arithmetic overflow",
    );
}

fn runtime_fixture() -> Value {
    json!({
        "header": {"schema":"ferrule-runtime-kv-trace", "schema_version":2,
            "config":{"run_scope":"cli-test", "stable_content_namespace":null,
                "max_events":64,"max_token_bytes":1024,"max_observations":8},
            "page_size":2,"prefix_capacity_pages":0,"namespace":[1,1,0,1,1]},
        "start_seq":0,"next_seq":5,
        "lifecycle_events":[
            {"seq":0,"kind":{"type":"demand","observation":0,"request":1,"session":1,"tokens":{"start":0,"len":4}}},
            {"seq":1,"kind":{"type":"admission_published","observation":0,"actual_tokens":0,"actual_pages":0}},
            {"seq":2,"kind":{"type":"execution_input","observation":0,"reservation":1,"start":0,"tokens":{"start":0,"len":4}}},
            {"seq":3,"kind":{"type":"logical_kv_published","observation":0,"reservation":1,"committed_tokens":4}},
            {"seq":4,"kind":{"type":"terminal","observation":0,"status":"success"}}
        ],
        "tokens":[1,2,3,4], "status":{"overflow":false,"unsupported":false,"observation_missing":false},
        "complete":true,"finalized":true,"usage":{"events":5,"token_bytes":16,"observations":1}
    })
}
fn runtime_empty() -> Value {
    let mut v = runtime_fixture();
    v["next_seq"] = json!(0);
    v["lifecycle_events"] = json!([]);
    v["tokens"] = json!([]);
    v["usage"] = json!({"events":0,"token_bytes":0,"observations":0});
    v
}
fn runtime_report(value: &Value) -> Value {
    success(run_json(value, &["--format", "runtime-json"]))
}

#[test]
fn runtime_json_roundtrip_matches_official_projection_and_actual_replay_at_capacity_zero() {
    let value = runtime_fixture();
    let snapshot: RuntimeKvTraceSnapshot = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(&snapshot).unwrap(), value);
    let batches = &[snapshot];
    let projected = project_runtime_kv_trace(batches).unwrap();
    let expected = analyze(&projected, AnalyzerConfig::default()).unwrap();
    let actual = replay_runtime_prefix_lru(batches).unwrap();
    let report = runtime_report(&value);
    assert_eq!(report["analysis"], serde_json::to_value(expected).unwrap());
    let runtime = &report["runtime_actual_entry_lru"];
    assert_eq!(
        runtime["actual_entry_lru"],
        serde_json::to_value(actual).unwrap()
    );
    assert_eq!(runtime["source_schema_version"], 2);
    assert_eq!(runtime["observed_prefix_capacity_pages"], 0);
    assert_eq!(report["analysis"]["workload"]["demands"], 1);
    assert_eq!(report["analysis"]["usage"]["publish_positions"], 2);
    assert!(
        runtime["comparison_scope"]
            .as_str()
            .unwrap()
            .contains("not counterfactual")
    );
}

#[test]
fn runtime_actual_hits_remain_distinct_from_sampled_ideal_capacities() {
    let mut v = runtime_fixture();
    v["header"]["prefix_capacity_pages"] = json!(2);
    v["tokens"] = json!([1, 2, 3, 4, 1, 2, 8, 9]);
    let events = v["lifecycle_events"].as_array_mut().unwrap();
    events.insert(4, json!({"kind":{"type":"prefix_entry_published","entry":10,"tokens":{"start":0,"len":2},"charge_pages":1,"generation":1,"replaced":false}}));
    events.extend([
        json!({"kind":{"type":"demand","observation":1,"request":2,"session":2,"tokens":{"start":4,"len":4}}}),
        json!({"kind":{"type":"lookup","observation":1,"outcome":"hit","entry":10,"matched_tokens":2,"generation":2}}),
        json!({"kind":{"type":"admission_published","observation":1,"actual_tokens":2,"actual_pages":1}}),
        json!({"kind":{"type":"unpin_ack","entry":10}}),
        json!({"kind":{"type":"terminal","observation":1,"status":"success"}}),
    ]);
    for (seq, event) in events.iter_mut().enumerate() {
        event["seq"] = json!(seq);
    }
    let len = events.len();
    v["next_seq"] = json!(len);
    v["usage"] = json!({"events":len,"token_bytes":32,"observations":2});
    let report = success(run_json(
        &v,
        &["--format", "runtime-json", "--capacity", "0"],
    ));
    assert_eq!(
        report["runtime_actual_entry_lru"]["actual_entry_lru"]["lookup_hits"],
        1
    );
    assert_eq!(
        report["runtime_actual_entry_lru"]["actual_entry_lru"]["actual_reused_tokens"],
        2
    );
    assert_eq!(
        report["analysis"]["capacity_points"][0]["contiguous_prefix_tokens"],
        0
    );
    assert_eq!(report["analysis"]["actual"]["reused_prefix_tokens"], 2);
}

#[test]
fn runtime_empty_final_is_valid_but_missing_history_and_invalid_watermarks_fail() {
    let empty = runtime_empty();
    let report = runtime_report(&empty);
    assert_eq!(report["analysis"]["workload"]["demands"], 0);
    assert_eq!(
        report["analysis"]["target_capacities"][0]["requirement"]["status"],
        "undefined"
    );
    assert!(report["analysis"]["capacity_points"][0]["page_reference_hit_rate"].is_null());
    let mut lost = empty.clone();
    lost["start_seq"] = json!(5);
    lost["next_seq"] = json!(5);
    failure(
        run_json(&lost, &["--format", "runtime-json"]),
        "runtime trace projection rejected",
    );
    for (field, value) in [
        ("next_seq", json!(1)),
        ("start_seq", json!(1)),
        ("finalized", json!(false)),
        ("complete", json!(false)),
    ] {
        let mut bad = empty.clone();
        bad[field] = value;
        failure(
            run_json(&bad, &["--format", "runtime-json"]),
            "runtime trace projection rejected",
        );
    }
    let mut bad = runtime_fixture();
    bad["lifecycle_events"][1]["seq"] = json!(2);
    failure(
        run_json(&bad, &["--format", "runtime-json"]),
        "runtime trace projection rejected",
    );
}

#[test]
fn runtime_strict_schema_overflow_status_arena_and_frontier_are_fail_closed() {
    for field in ["overflow", "unsupported", "observation_missing"] {
        let mut value = runtime_fixture();
        value["status"][field] = json!(true);
        failure(
            run_json(&value, &["--format", "runtime-json"]),
            "runtime trace projection rejected",
        );
    }
    for field in ["start_seq", "next_seq"] {
        let mut value = runtime_fixture();
        value.as_object_mut().unwrap().remove(field);
        failure(
            run_json(&value, &["--format", "runtime-json"]),
            "JSON Data error",
        );
    }
    for pointer in [
        "",
        "/header",
        "/header/config",
        "/status",
        "/usage",
        "/lifecycle_events/0",
        "/lifecycle_events/0/kind",
        "/lifecycle_events/0/kind/tokens",
    ] {
        let mut value = runtime_fixture();
        value.pointer_mut(pointer).unwrap()["unknown"] = json!(true);
        failure(
            run_json(&value, &["--format", "runtime-json"]),
            "JSON Data error",
        );
    }
    let mut bad = runtime_fixture();
    bad["header"]["schema_version"] = json!(1);
    failure(
        run_json(&bad, &["--format", "runtime-json"]),
        "runtime trace projection rejected",
    );
    let mut bad = runtime_fixture();
    bad["lifecycle_events"][0]["kind"]["tokens"]["len"] = json!(u64::MAX);
    failure(
        run_json(&bad, &["--format", "runtime-json"]),
        "runtime trace projection rejected",
    );
    let mut bad = runtime_fixture();
    bad["lifecycle_events"][2]["kind"]["start"] = json!(3);
    failure(
        run_json(&bad, &["--format", "runtime-json"]),
        "execution does not continue proven frontier",
    );
    let mut bad = runtime_fixture();
    bad["lifecycle_events"][3]["kind"]["committed_tokens"] = json!(5);
    failure(
        run_json(&bad, &["--format", "runtime-json"]),
        "publish crosses unproven frontier",
    );
    let raw = serde_json::to_string(&runtime_fixture()).unwrap().replacen(
        "\"start_seq\":0",
        "\"start_seq\":0,\"start_seq\":0",
        1,
    );
    failure(
        invoke(
            &["--trace", "-", "--format", "runtime-json", "--json"],
            raw.as_bytes(),
        ),
        "JSON Data error",
    );
}

#[test]
fn runtime_projection_checks_cumulative_budgets_before_full_prefix_expansion() {
    let value = runtime_fixture();
    success(run_json(
        &value,
        &[
            "--format",
            "runtime-json",
            "--max-projected-events",
            "5",
            "--max-projected-tokens",
            "8",
            "--max-projected-pages",
            "2",
        ],
    ));
    // The arena has four tokens, but Demand + full Publish copies eight tokens.
    for (flag, limit) in [
        ("--max-projected-events", "4"),
        ("--max-projected-tokens", "7"),
        ("--max-projected-pages", "1"),
    ] {
        failure(
            run_json(&value, &["--format", "runtime-json", flag, limit]),
            "resource limit",
        );
    }
    failure(
        run_json(
            &value,
            &[
                "--format",
                "runtime-json",
                "--max-projected-tokens",
                "16000001",
            ],
        ),
        "may only lower",
    );
    failure(
        run_json(
            &value,
            &["--format", "runtime-json", "--max-input-bytes", "1"],
        ),
        "input byte limit",
    );
}

#[test]
fn both_public_runtime_entry_points_reject_invalid_entry_history_without_report() {
    let mut value = runtime_fixture();
    value["lifecycle_events"]
        .as_array_mut()
        .unwrap()
        .push(json!({"seq":5,"kind":{"type":"unpin_ack","entry":999}}));
    value["next_seq"] = json!(6);
    value["usage"]["events"] = json!(6);
    let snapshot: RuntimeKvTraceSnapshot = serde_json::from_value(value.clone()).unwrap();
    let batches = [snapshot];
    assert!(project_runtime_kv_trace(&batches).is_err());
    assert!(replay_runtime_prefix_lru(&batches).is_err());
    let temp = Temp::new();
    let output = temp.0.join("report.json");
    failure(
        run_json(
            &value,
            &["--format", "runtime-json", "--output", path(&output)],
        ),
        "runtime trace projection rejected",
    );
    assert!(!output.exists());
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn output_is_atomic_private_no_clobber_and_errors_leave_no_partial_file() {
    let temp = Temp::new();
    let output = temp.0.join("report.json");
    let result = run_json(&fixture(), &["--output", path(&output)]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
    let original = fs::read(&output).unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&original).unwrap()["schema"],
        "ferrule-kvset-report"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&output).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    failure(
        run_json(&fixture(), &["--output", path(&output)]),
        "without overwriting",
    );
    assert_eq!(fs::read(&output).unwrap(), original);
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 1);
    let bad_path = temp.0.join("bad.json");
    let mut bad = fixture();
    bad["events"][7]["seq"] = json!(0);
    failure(
        run_json(&bad, &["--output", path(&bad_path)]),
        "seq must strictly increase",
    );
    assert!(!bad_path.exists());
    let nonexistent = temp.0.join("missing/report.json");
    failure(
        run_json(&fixture(), &["--output", path(&nonexistent)]),
        "create report staging file",
    );
    success(run_json(&fixture(), &["--output", "-"]));
    failure(
        invoke(
            &["--trace", path(&temp.0.join("missing.json")), "--json"],
            &[],
        ),
        "open input",
    );
}

#[test]
fn help_and_text_mode_explain_formats_privacy_units_and_runtime_scope() {
    let help = invoke(&["--help"], &[]);
    assert!(help.status.success());
    let text = String::from_utf8(help.stdout).unwrap();
    for expected in [
        "jsonl",
        "runtime-json",
        "--max-input-bytes",
        "--max-line-bytes",
        "--max-config-bytes",
        "--max-projected-tokens",
        "--alpha",
        "--coverage",
        "sensitive",
        "stdin",
    ] {
        assert!(text.contains(expected), "missing {expected}: {text}");
    }
    let output = invoke(
        &["--trace", "-", "--format", "runtime-json"],
        &serde_json::to_vec(&runtime_fixture()).unwrap(),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8(output.stdout).unwrap();
    for expected in [
        "equal_charge_logical_pages",
        "Actual entry-LRU",
        "not counterfactual",
        "not analyzer heap",
        "page_reference_hit_rate",
        "undefined",
    ] {
        assert!(text.contains(expected), "missing {expected}: {text}");
    }
}

#[test]
fn repeated_runtime_full_prefixes_use_cumulative_not_arena_or_per_event_budgets() {
    let mut value = runtime_fixture();
    let mut events = value["lifecycle_events"].as_array().unwrap()[..4].to_vec();
    let mut projected_tokens = 8usize;
    let mut projected_pages = 2usize;
    for committed in 5usize..=36 {
        events.push(json!({"kind":{"type":"execution_input","observation":0,
            "reservation":committed,"start":committed-1,"tokens":{"start":committed-1,"len":1}}}));
        events.push(
            json!({"kind":{"type":"logical_kv_published","observation":0,
            "reservation":committed,"committed_tokens":committed}}),
        );
        projected_tokens += committed;
        projected_pages += committed.div_ceil(2);
    }
    events.push(json!({"kind":{"type":"terminal","observation":0,"status":"success"}}));
    for (seq, event) in events.iter_mut().enumerate() {
        event["seq"] = json!(seq);
    }
    value["header"]["config"]["max_events"] = json!(1000);
    value["tokens"] = json!((1..=36).collect::<Vec<u32>>());
    value["next_seq"] = json!(events.len());
    value["usage"] = json!({"events":events.len(),"token_bytes":144,"observations":1});
    value["lifecycle_events"] = json!(events);
    let report = success(run_json(
        &value,
        &[
            "--format",
            "runtime-json",
            "--max-projected-tokens",
            &projected_tokens.to_string(),
            "--max-projected-pages",
            &projected_pages.to_string(),
        ],
    ));
    assert_eq!(
        report["analysis"]["usage"]["publish_positions"],
        projected_pages
    );
    let temp = Temp::new();
    let output = temp.0.join("report.json");
    for (flag, limit) in [
        ("--max-projected-tokens", projected_tokens - 1),
        ("--max-projected-pages", projected_pages - 1),
    ] {
        failure(
            run_json(
                &value,
                &[
                    "--format",
                    "runtime-json",
                    flag,
                    &limit.to_string(),
                    "--output",
                    path(&output),
                ],
            ),
            "projection resource limit",
        );
        assert!(!output.exists());
    }
    assert_eq!(fs::read_dir(&temp.0).unwrap().count(), 0);
}

#[test]
fn file_input_for_each_format_matches_stdio_and_jsonl_accepts_crlf() {
    let temp = Temp::new();
    for (format, value) in [
        ("json", fixture()),
        ("jsonl", fixture()),
        ("runtime-json", runtime_fixture()),
    ] {
        let bytes = if format == "jsonl" {
            let text = String::from_utf8(jsonl(&value))
                .unwrap()
                .replace('\n', "\r\n");
            format!("{text}\r\n").into_bytes()
        } else {
            serde_json::to_vec(&value).unwrap()
        };
        let trace = temp.write(format, &bytes);
        let from_file = success(invoke(
            &["--trace", path(&trace), "--format", format, "--json"],
            &[],
        ));
        let from_stdin = success(invoke(
            &["--trace", "-", "--format", format, "--json"],
            &bytes,
        ));
        assert_eq!(from_file["analysis"], from_stdin["analysis"]);
        assert_eq!(
            from_file["runtime_actual_entry_lru"],
            from_stdin["runtime_actual_entry_lru"]
        );
        assert_eq!(from_file["replay"]["input_bytes"], bytes.len());
        assert_eq!(from_stdin["replay"]["input_bytes"], bytes.len());
    }
}
