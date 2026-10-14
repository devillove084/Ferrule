use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

use ferrule_common::kvset::{AnalyzerConfig, analyze};
use ferrule_runtime::engine::kv_trace::{RuntimeKvTraceKind as K, RuntimeKvTraceSnapshot};
use ferrule_runtime::engine::{project_runtime_kv_trace, replay_runtime_prefix_lru};
use serde_json::{Value, json};

#[path = "bench_kv_trace/bench_model.rs"]
mod bench_model;
use bench_model::Fixture;

const PROMPT: &str = "hello a b c d e a b c d e a b c d e a b c d e a b c d e a b";
fn path(p: &Path) -> &str {
    p.to_str().unwrap()
}
fn bench(f: &Fixture, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ferrule"))
        .args([
            "bench-interactive",
            path(&f.0),
            "--backend",
            "cpu",
            "--max-layers",
            "2",
            "--max-tokens",
            "3",
            "--chat-template",
            "chatml",
            "--prefill-chunk-size",
            "8",
            "--json",
        ])
        .args(extra)
        .output()
        .unwrap()
}
fn succeeded(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn failed(output: Output, text: &str) {
    assert!(
        !output.status.success(),
        "unexpected success: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(output.stdout.is_empty(), "partial success report");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains(text),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn analyze_file(p: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_ferrule"))
        .args([
            "analyze-kvset",
            "--format",
            "runtime-json",
            "--trace",
            path(p),
            "--json",
        ])
        .output()
        .unwrap()
}
fn batches(p: &Path) -> Vec<RuntimeKvTraceSnapshot> {
    let value: Value = serde_json::from_slice(&fs::read(p).unwrap()).unwrap();
    if value.is_array() {
        serde_json::from_value(value).unwrap()
    } else {
        vec![serde_json::from_value(value).unwrap()]
    }
}

#[test]
fn real_cpu_bench_exports_every_batch_and_roundtrips_through_official_analyzers() {
    for capacity in [0, 8] {
        let f = Fixture::new();
        let output = f.0.join("runtime.json");
        let result = bench(
            &f,
            &[
                "--independent-prompts",
                "--kv-trace",
                path(&output),
                "--prefix-cache-capacity-pages",
                &capacity.to_string(),
                "--warmup-tokens",
                "2",
                "--prompt",
                PROMPT,
                "--prompt",
                PROMPT,
            ],
        );
        assert!(
            String::from_utf8_lossy(&result.stderr)
                .contains("sensitive prompt/generated token IDs")
        );
        let report = succeeded(result);
        assert_eq!(report["independent_prompts"], true);
        assert_eq!(report["kv_trace"]["physical_shutdown_proof"], false);
        assert_eq!(report["kv_trace"]["complete"], true);
        assert_eq!(report["total_turns"], 2);
        assert_eq!(
            report["turns"][0]["prompt_token_ids"],
            report["turns"][1]["prompt_token_ids"]
        );
        let trace = batches(&output);
        assert_eq!(
            trace.len(),
            4,
            "warmup, each measured request, and final must all be retained"
        );
        assert_eq!(report["kv_trace"]["batches"], trace.len());
        let mut next = 0;
        let mut sessions = BTreeSet::new();
        let mut requests = BTreeSet::new();
        let mut frontiers = BTreeMap::new();
        let mut observations = BTreeMap::new();
        for (i, batch) in trace.iter().enumerate() {
            assert_eq!(batch.header.schema_version, 2);
            assert_eq!(batch.start_seq, next);
            assert_eq!(batch.finalized, i == trace.len() - 1);
            for event in &batch.lifecycle_events {
                assert_eq!(event.seq, next);
                next += 1;
                match event.kind {
                    K::Demand {
                        observation,
                        request,
                        session,
                        ..
                    } => {
                        assert!(sessions.insert(session));
                        requests.insert(request);
                        observations.insert(observation, request);
                    }
                    K::LogicalKvPublished {
                        observation,
                        committed_tokens,
                        ..
                    } => {
                        frontiers.insert(observation, committed_tokens);
                    }
                    _ => {}
                }
            }
            assert_eq!(batch.next_seq, next);
        }
        assert_eq!(
            requests,
            BTreeSet::from([0, 1, 2]),
            "warmup must be recorded"
        );
        assert!(trace.last().unwrap().complete);
        for (observation, frontier) in frontiers {
            let request = observations[&observation];
            if request > 0 {
                let turn = &report["turns"][(request - 1) as usize];
                assert_eq!(frontier as u64, turn["final_position"].as_u64().unwrap());
                assert!(
                    frontier > turn["prompt_tokens"].as_u64().unwrap() as usize,
                    "generated token commits must be traced too"
                );
            }
        }
        let projected = project_runtime_kv_trace(&trace).unwrap();
        let actual = replay_runtime_prefix_lru(&trace).unwrap();
        let cli = succeeded(analyze_file(&output));
        assert_eq!(
            cli["analysis"],
            serde_json::to_value(analyze(&projected, AnalyzerConfig::default()).unwrap()).unwrap()
        );
        assert_eq!(
            cli["runtime_actual_entry_lru"]["actual_entry_lru"],
            serde_json::to_value(&actual).unwrap()
        );
        if capacity == 0 {
            assert_eq!(actual.actual_reused_tokens, 0);
        } else {
            assert!(actual.lookup_hits > 0);
            assert!(actual.actual_reused_tokens > 0);
            assert!(
                trace
                    .last()
                    .unwrap()
                    .lifecycle_events
                    .iter()
                    .any(|event| matches!(event.kind, K::PrefixEntryRemoved { .. })),
                "shutdown removals belong in final batch"
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&output).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}

#[test]
fn real_cpu_trace_on_and_off_preserve_generation_and_steps() {
    let f = Fixture::new();
    let output = f.0.join("runtime.json");
    let off = succeeded(bench(
        &f,
        &[
            "--independent-prompts",
            "--prompt",
            PROMPT,
            "--prompt",
            PROMPT,
        ],
    ));
    assert!(off.get("kv_trace").is_none());
    assert!(!output.exists());
    let on = succeeded(bench(
        &f,
        &[
            "--independent-prompts",
            "--kv-trace",
            path(&output),
            "--prompt",
            PROMPT,
            "--prompt",
            PROMPT,
        ],
    ));
    for i in 0..2 {
        for field in [
            "prompt_token_ids",
            "generated_token_ids",
            "final_position",
            "finish_reason",
        ] {
            assert_eq!(off["turns"][i][field], on["turns"][i][field], "{field}");
        }
        let shape = |value: &Value| {
            value["turns"][i]["runtime_steps"]
                .as_array()
                .unwrap()
                .iter()
                .map(|step| {
                    (
                        step["action_kind"].clone(),
                        step["rows"].clone(),
                        step["finished"].clone(),
                    )
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(shape(&off), shape(&on));
    }
    let legacy = succeeded(bench(&f, &["--prompt", "hello", "--prompt", "hello"]));
    assert!(legacy.get("independent_prompts").is_none());
    assert!(legacy.get("kv_trace").is_none());
    assert!(
        legacy["turns"][0]["prompt_tokens"].as_u64() > legacy["turns"][1]["prompt_tokens"].as_u64(),
        "legacy second turn must retain continuation formatting"
    );
}

#[test]
fn real_cpu_empty_final_is_kept_and_missing_or_reordered_batches_fail_closed() {
    let f = Fixture::new();
    let output = f.0.join("runtime.json");
    succeeded(bench(
        &f,
        &[
            "--independent-prompts",
            "--kv-trace",
            path(&output),
            "--prefix-cache-capacity-pages",
            "0",
            "--prompt",
            PROMPT,
            "--prompt",
            PROMPT,
        ],
    ));
    let trace = batches(&output);
    let final_batch = trace.last().unwrap();
    assert!(final_batch.lifecycle_events.is_empty());
    assert!(final_batch.finalized && final_batch.complete);
    assert!(final_batch.start_seq > 0);
    let bad = f.0.join("bad.json");
    let mut reversed = trace.clone();
    reversed.swap(0, 1);
    for input in [
        trace[1..].to_vec(),
        trace[..trace.len() - 1].to_vec(),
        vec![final_batch.clone()],
        reversed,
    ] {
        fs::write(&bad, serde_json::to_vec(&input).unwrap()).unwrap();
        failed(analyze_file(&bad), "runtime trace projection rejected");
    }
    fs::write(&bad, serde_json::to_vec(final_batch).unwrap()).unwrap();
    failed(analyze_file(&bad), "runtime trace projection rejected");
}

#[test]
fn real_cpu_recorder_overflow_exports_private_diagnostic_and_no_success_report() {
    for flag in ["--kv-trace-max-events", "--kv-trace-max-token-bytes"] {
        let f = Fixture::new();
        let output = f.0.join("diagnostic.json");
        failed(
            bench(
                &f,
                &[
                    "--independent-prompts",
                    "--kv-trace",
                    path(&output),
                    flag,
                    "1",
                    "--prompt",
                    PROMPT,
                ],
            ),
            "diagnostic saved",
        );
        let diagnostic: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
        assert_eq!(diagnostic["schema"], "ferrule-runtime-kv-trace-diagnostic");
        assert_eq!(diagnostic["complete"], false);
        assert_eq!(diagnostic["physical_shutdown_proof"], false);
        assert!(
            diagnostic["batches"]
                .as_array()
                .unwrap()
                .iter()
                .all(|b| b["status"]["overflow"] == true && b["complete"] == false)
        );
        failed(analyze_file(&output), "JSON Data error");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&output).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}

#[test]
fn real_cpu_later_request_failure_preserves_earlier_batches_without_faking_complete() {
    let f = Fixture::new();
    let output = f.0.join("diagnostic.json");
    let too_long = "hello ".repeat(5000);
    failed(
        bench(
            &f,
            &[
                "--independent-prompts",
                "--kv-trace",
                path(&output),
                "--prompt",
                "hello",
                "--prompt",
                &too_long,
            ],
        ),
        "diagnostic saved",
    );
    let diagnostic: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    let trace: Vec<RuntimeKvTraceSnapshot> =
        serde_json::from_value(diagnostic["batches"].clone()).unwrap();
    assert!(trace.len() >= 2);
    assert!(
        trace[0]
            .lifecycle_events
            .iter()
            .any(|e| matches!(e.kind, K::Terminal { .. }))
    );
    assert!(!trace.last().unwrap().complete);
    assert!(project_runtime_kv_trace(&trace).is_err());
    failed(analyze_file(&output), "JSON Data error");
}

#[test]
fn collected_trace_budget_is_bounded_across_successful_drains() {
    let f = Fixture::new();
    let reference = f.0.join("reference.json");
    succeeded(bench(
        &f,
        &[
            "--independent-prompts",
            "--kv-trace",
            path(&reference),
            "--prompt",
            PROMPT,
        ],
    ));
    let first = batches(&reference)[0].lifecycle_events.len();
    let output = f.0.join("diagnostic.json");
    failed(
        bench(
            &f,
            &[
                "--independent-prompts",
                "--kv-trace",
                path(&output),
                "--kv-trace-max-events",
                &first.to_string(),
                "--prompt",
                PROMPT,
                "--prompt",
                PROMPT,
            ],
        ),
        "diagnostic saved",
    );
    let diagnostic: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    let trace: Vec<RuntimeKvTraceSnapshot> =
        serde_json::from_value(diagnostic["batches"].clone()).unwrap();
    assert!(
        trace
            .iter()
            .map(|b| b.lifecycle_events.len())
            .sum::<usize>()
            > first
    );
    assert!(
        trace.len() >= 3,
        "both drained batches and final diagnostic are retained"
    );
    assert!(trace.iter().all(|b| !b.status.overflow));
    assert!(!trace.last().unwrap().complete);
}

#[test]
fn flags_privacy_and_no_overwrite_are_enforced() {
    let f = Fixture::new();
    let output = f.0.join("trace.json");
    for flags in [
        vec!["--kv-trace", path(&output)],
        vec!["--independent-prompts", "--golden", "not-read.json"],
        vec!["--kv-trace-max-events", "1"],
        vec!["--kv-trace-max-token-bytes", "1"],
    ] {
        failed(bench(&f, &flags), "error:");
    }
    failed(
        bench(&f, &["--independent-prompts", "--kv-trace", "-"]),
        "not stdout",
    );
    fs::write(&output, b"do not overwrite").unwrap();
    failed(
        bench(&f, &["--independent-prompts", "--kv-trace", path(&output)]),
        "refusing to overwrite",
    );
    assert_eq!(fs::read(&output).unwrap(), b"do not overwrite");
    assert!(
        !fs::read_dir(&f.0).unwrap().any(|p| p
            .unwrap()
            .file_name()
            .to_string_lossy()
            .ends_with(".tmp"))
    );
    let help = bench(&f, &["--help"]);
    assert!(help.status.success());
    let text = String::from_utf8(help.stdout).unwrap();
    for expected in [
        "--independent-prompts",
        "--kv-trace",
        "--kv-trace-max-events",
        "--kv-trace-max-token-bytes",
        "--prefix-cache-capacity-pages",
        "sensitive",
    ] {
        assert!(text.contains(expected), "missing {expected}");
    }
}

#[test]
fn runtime_batch_input_remains_strict_and_empty_batch_array_is_not_an_empty_trace() {
    let f = Fixture::new();
    let output = f.0.join("runtime.json");
    succeeded(bench(
        &f,
        &[
            "--independent-prompts",
            "--kv-trace",
            path(&output),
            "--prompt",
            "hello",
        ],
    ));
    let mut value: Value = serde_json::from_slice(&fs::read(&output).unwrap()).unwrap();
    value[0]["unknown"] = json!(true);
    let bad = f.0.join("bad.json");
    fs::write(&bad, serde_json::to_vec(&value).unwrap()).unwrap();
    failed(analyze_file(&bad), "JSON Data error");
    fs::write(&bad, b"[]").unwrap();
    failed(analyze_file(&bad), "missing trace batches");
}
