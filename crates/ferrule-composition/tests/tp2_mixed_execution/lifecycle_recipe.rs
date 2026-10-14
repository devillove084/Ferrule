//! Bounded recipe driver only: every numerical operation uses the production executor.
use super::super::*;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const RAW_SCHEMA: &str = "ferrule.mixed-lifecycle-raw.v1";
const CANDIDATE_SCHEMA: &str = "ferrule.paper.tp2-mixed-lifecycle-candidate.v1";

fn fixture() -> Value {
    serde_json::from_str(include_str!("lifecycle_plans.json")).unwrap()
}
fn validate_plan(plan: &Value) -> std::result::Result<Vec<usize>, String> {
    // Admit only the exact released operations: greedy4, fork2, canary2,
    // recreate1 and cold-owner1. Names or decode counts alone cannot authorize
    // a modified plan.
    if !fixture()["plans"].as_array().unwrap().contains(plan) {
        return Err("expected exact frozen lifecycle release plan".into());
    }
    let mut calls = vec![0usize];
    for op in plan["operations"].as_array().unwrap() {
        match op["op"].as_str().unwrap() {
            "call" => *calls.last_mut().unwrap() += 1,
            "recreate_model" => calls.push(0),
            _ => {}
        }
    }
    if calls.iter().sum::<usize>() > 20 || calls.iter().any(|&n| n == 0 || n * 10_000 > 100_000) {
        return Err("recipe exceeds current bounded trace event/invocation admission".into());
    }
    Ok(calls)
}
fn recipe_config() -> PipelineConfig {
    PipelineConfig {
        session_capacity: 3,
        ..config()
    }
}
fn new_capture(
    root: &Path,
    pin: &str,
    plan: &Value,
    epoch: usize,
    calls: usize,
) -> capture::CandidateCapture {
    // The paper's 4 GiB total tensor allowance is shared equally by the TP owners.
    // Recreate epochs execute sequentially and get proportional, not duplicated, caps.
    let total_calls = validate_plan(plan).unwrap().iter().sum::<usize>();
    let payload = ((4usize << 30) / 2 / total_calls) * calls;
    capture::CandidateCapture::for_recipe(
        root.join(format!("epoch-{epoch}")),
        pin,
        plan,
        epoch,
        calls,
        calls * 10_000,
        payload,
    )
}

pub(super) fn run() {
    run_with_cache(None);
}
pub(super) fn run_with_cache(cache_grants: Option<usize>) {
    run_path(cache_grants, quality_path::QualityPath::DiagnosticFallback);
}
pub(super) fn run_unobserved_with_cache(cache_grants: Option<usize>) {
    run_path(cache_grants, quality_path::QualityPath::UnobservedFused);
}
fn run_path(cache_grants: Option<usize>, path: quality_path::QualityPath) {
    let reference_dir = PathBuf::from(
        std::env::var("FERRULE_MIXED_LIFECYCLE_ORACLE_DIR").expect("pinned lifecycle reference"),
    );
    let oracle_root = reference_dir
        .parent()
        .and_then(Path::parent)
        .expect("oracle runs directory");
    oracle::hash(
        &oracle_root.join("lifecycle-release.json"),
        "834c8a1ff0aa12d0e58e500e870e68950a2dbe4e3cb6902d47fe02fdcd459639",
    );
    let pin = std::env::var("FERRULE_MIXED_LIFECYCLE_ORACLE_SHA256").expect("manifest pin");
    let model = PathBuf::from(std::env::var("FERRULE_MIXED_MODEL").unwrap());
    let reference = oracle::Oracle::open_lifecycle(&reference_dir, &pin, &model);
    let release = oracle::json(&oracle_root.join("lifecycle-release.json"));
    assert!(
        release["runs"].as_array().unwrap().iter().any(|entry| {
            entry["manifest_sha256"] == pin
                && oracle_root
                    .join(entry["directory"].as_str().unwrap())
                    .canonicalize()
                    .unwrap()
                    == reference_dir.canonicalize().unwrap()
        }),
        "reference must be an exact released run"
    );
    let plan = reference.manifest()["suite"]["plan"].clone();
    let epoch_calls = validate_plan(&plan).unwrap();
    let operations = plan["operations"].as_array().unwrap();
    let calls: BTreeMap<String, oracle::Case> = operations
        .iter()
        .filter(|o| o["op"] == "call")
        .map(|o| {
            let id = o["id"].as_str().unwrap();
            (id.into(), reference.case(id))
        })
        .collect();
    assert_eq!(
        reference.manifest()["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        operations
            .iter()
            .filter(|o| o["op"] == "call")
            .map(|o| o["id"].as_str().unwrap())
            .collect::<Vec<_>>()
    );
    let reference_events = reference.manifest()["suite"]["events"].as_array().unwrap();
    assert_eq!(reference_events.len(), operations.len());
    for (op, event) in operations.iter().zip(reference_events) {
        for (key, value) in op.as_object().unwrap() {
            assert_eq!(&event[key], value);
        }
        if op["op"] == "call" {
            assert_eq!(
                calls[op["id"].as_str().unwrap()].call("call"),
                &event["call"]
            );
        }
    }
    // Validate every input and source artifact before opening either CUDA owner.
    for call in calls.values() {
        assert!(call.tokens("call").len() <= recipe_config().max_batch_tokens);
        assert!(
            call.call("call")["sequence_length_after"].as_u64().unwrap()
                <= recipe_config().max_positions as u64
        );
    }
    grant();
    let evidence = PathBuf::from(std::env::var("FERRULE_MIXED_EVIDENCE_DIR").unwrap())
        .canonicalize()
        .unwrap();
    assert!(!evidence.starts_with(reference_dir.canonicalize().unwrap()));
    capture::write_new(
        evidence.join("execution-path.json"),
        &path.contract(cache_grants),
    );
    let raw_schema = if path.captures() {
        RAW_SCHEMA
    } else {
        "ferrule.mixed-lifecycle-unobserved.v1"
    };
    let root = evidence.join(if path.captures() {
        "lifecycle-raw"
    } else {
        "lifecycle-unobserved"
    });
    std::fs::create_dir(&root).unwrap();
    capture::write_new(
        root.join("contract.json"),
        &json!({
            "schema":raw_schema,"target_candidate_schema":if path.captures() {Some(CANDIDATE_SCHEMA)} else {None},
            "execution_path":path.contract(cache_grants),
            "reference_manifest_sha256":pin,"plan":plan,"epoch_calls":epoch_calls,
            "checkpoint":reference.manifest()["checkpoint"],"protocol":reference.manifest()["protocol"],
            "state_scope":if path.captures() {"observed provisional GDN joined to publication; leased post-publication KV; no fabricated alias snapshots"}
                else {"host publication and exact committed frontiers only; no GDN/KV tensor readback or internal state equality claim"},
            "max_total_tensor_bytes":4u64<<30,"max_tensor_bytes":8<<20,
            "max_events_per_call_allowance":10000,"gpu_bytes_per_owner":22u64<<30,
            "pipeline_config":{"page_size":16,"max_pages":32,"max_positions":128,"max_batch_tokens":8,"session_capacity":3},
            "cold_owner":"one fresh supervised test process; repeat as a separate authorized launch, not a second in-process loop"
        }),
    );
    let mut epoch = 0;
    let mut capture = path
        .captures()
        .then(|| new_capture(&root, &pin, &plan, epoch, epoch_calls[epoch]));
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(&model).unwrap();
    let mut pipeline = Some(build_with_trace_and_cache(
        resources,
        recipe_config(),
        capture.as_ref().map(|c| c.config()),
        cache_grants,
    ));
    let mut events = Vec::new();
    // Preserve insertion order for the paper's untouched_sessions contract.
    let mut sessions: Vec<(String, SessionId, usize)> = Vec::new();
    let mut ids = BTreeMap::new();
    let mut next_id = 1;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        for op in operations {
            let mut event = op.clone();
            event["epoch"] = json!(epoch);
            let name = op["session"].as_str().unwrap_or("");
            match op["op"].as_str().unwrap() {
                "create" => {
                    assert!(!sessions.iter().any(|s| s.0 == name));
                    let id = *ids.entry(name.to_string()).or_insert_with(|| {
                        let id = SessionId(next_id);
                        next_id += 1;
                        id
                    });
                    sessions.push((name.into(), id, 0));
                    event["frontier"] = json!(0);
                    event["runtime_session"] = json!(id.0);
                }
                "fork" => {
                    let source = sessions
                        .iter()
                        .find(|s| s.0 == op["source"].as_str().unwrap())
                        .unwrap()
                        .clone();
                    let target = SessionId(next_id);
                    next_id += 1;
                    pipeline
                        .as_mut()
                        .unwrap()
                        .fork_session(source.1, target)
                        .unwrap();
                    sessions.push((name.into(), target, source.2));
                    event["frontier"] = json!(source.2);
                    event["runtime_session"] = json!(target.0);
                    event["runtime_source_session"] = json!(source.1.0);
                    // No post-fork aliased read is attempted. The converter binds
                    // fork state to the source's preceding actual captured call.
                    event["state_binding"] = json!(
                        "source preceding publication; validate branch before-state on its next call"
                    );
                }
                "drop" => {
                    let i = sessions.iter().position(|s| s.0 == name).unwrap();
                    pipeline
                        .as_mut()
                        .unwrap()
                        .release_session(sessions[i].1)
                        .unwrap();
                    sessions.remove(i);
                }
                "recreate_model" => {
                    assert!(sessions.is_empty());
                    pipeline.as_mut().unwrap().shutdown().unwrap();
                    assert!(capture.as_ref().is_none_or(|c| c.finish()));
                    assert_eq!(
                        pipeline.as_ref().unwrap().page_manager().allocated_pages(),
                        0
                    );
                    // Drop the entire prior worker/provider image, then use the same constructor.
                    drop(pipeline.take());
                    epoch += 1;
                    event["epoch"] = json!(epoch);
                    capture = path
                        .captures()
                        .then(|| new_capture(&root, &pin, &plan, epoch, epoch_calls[epoch]));
                    let (_, resources) =
                        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(&model)
                            .unwrap();
                    pipeline = Some(build_with_trace_and_cache(
                        resources,
                        recipe_config(),
                        capture.as_ref().map(|c| c.config()),
                        cache_grants,
                    ));
                    event["decoded_cache_cleared"] = json!(true);
                }
                "call" => {
                    let id = op["id"].as_str().unwrap();
                    let expected = &calls[id];
                    let tokens = expected.tokens("call");
                    if op["input"]["kind"] == "tokens" {
                        assert_eq!(json!(tokens), op["input"]["ids"]);
                    }
                    let i = sessions.iter().position(|s| s.0 == name).unwrap();
                    let (_, session, start) = sessions[i];
                    assert_eq!(expected.call("call")["position_start"], start);
                    let phase = if start == 0 {
                        ForwardPhase::Prefill
                    } else {
                        ForwardPhase::Decode
                    };
                    let phase_name = if start == 0 { "prefill" } else { "decode" };
                    let before = pipeline.as_ref().unwrap().coordinator().publication_count();
                    assert_eq!(
                        quality_path::frontier(pipeline.as_ref().unwrap(), session),
                        start
                    );
                    let result = pipeline.as_mut().unwrap().forward(session, &tokens, phase);
                    if let Some(capture) = &capture {
                        capture.recipe_receipt(id, phase_name, session, &tokens, &result);
                    }
                    let actual = result.unwrap();
                    event["publication"] = quality_path::publication(
                        pipeline.as_ref().unwrap(),
                        session,
                        before,
                        start,
                        tokens.len(),
                    );
                    for (_, other, frontier) in sessions.iter().filter(|s| s.0 != name) {
                        assert_eq!(
                            quality_path::frontier(pipeline.as_ref().unwrap(), *other),
                            *frontier
                        );
                    }
                    if let Some(capture) = &capture {
                        // Only the diagnostic path obtains committed tensor read authority.
                        let export = pipeline
                            .as_mut()
                            .unwrap()
                            .export_committed_trace(session, actual.transaction);
                        capture.recipe_kv_receipt(
                            id,
                            phase_name,
                            session,
                            actual.transaction,
                            &export,
                        );
                        export.unwrap();
                    }
                    expected.compare("call", actual.logits.values());
                    let next = actual.logits.values()[actual.logits.values().len() - 248320..]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                        .unwrap()
                        .0;
                    event["call"] = json!({"stage":"call","input_token_ids":tokens,"position_start":start,
                        "sequence_length_after":start+tokens.len(),"next_token_id":next});
                    event["untouched_sessions"] = json!(
                        sessions
                            .iter()
                            .filter(|s| s.0 != name)
                            .map(|s| &s.0)
                            .collect::<Vec<_>>()
                    );
                    event["runtime_session"] = json!(session.0);
                    event["transaction"] = json!(actual.transaction.get());
                    sessions[i].2 += tokens.len();
                }
                _ => unreachable!("validated recipe"),
            }
            assert!(sessions.len() <= 3);
            capture::write_new(
                root.join(format!("operation-{:02}.json", events.len())),
                &event,
            );
            events.push(event);
        }
        assert!(sessions.is_empty());
        assert_eq!(pipeline.as_ref().unwrap().outstanding(), 0);
    }));
    let cleanup = pipeline.as_mut().map_or(Ok(()), |p| p.shutdown());
    if (outcome.is_err() || cleanup.is_err())
        && let Some(capture) = &capture
    {
        capture.abort();
    }
    let complete = capture.as_ref().is_none_or(|c| c.finish());
    capture::write_new(
        root.join("result.json"),
        &json!({"schema":raw_schema,
        "status":if outcome.is_ok()&&cleanup.is_ok()&&complete {
            if path.captures() {"captured"} else {"executed_unobserved"}
        } else {"incomplete"},
        "events":events,"writer_complete":path.captures().then_some(complete),"shutdown_ok":cleanup.is_ok(),
        "numeric_observer_installed":path.captures(),"canonical_candidate_complete":false,
        "independent_quality":if path.captures() {"CPU lifecycle bridge and original full inventory comparator required"}
            else {"independent full-vocabulary logits and per-call tokens; host publication checks only"}}),
    );
    cleanup.unwrap();
    if let Err(p) = outcome {
        std::panic::resume_unwind(p);
    }
    assert!(complete);
}

#[test]
fn canonical_recipe_fixtures_budget_invocations_and_recreation_without_gpu() {
    let fixtures = fixture();
    oracle::hash(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../papers/arxiv/experiments/tp2_mixed/oracle/multistep.py"),
        fixtures["source_sha256"].as_str().unwrap(),
    );
    for (file, key) in [
        ("lifecycle-release.json", "release_sha256"),
        ("lifecycle-contract.json", "contract_sha256"),
    ] {
        oracle::hash(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../papers/arxiv/experiments/tp2_mixed/oracle")
                .join(file),
            fixtures[key].as_str().unwrap(),
        );
    }
    for plan in fixtures["plans"].as_array().unwrap() {
        let counts = validate_plan(plan).unwrap();
        assert_eq!(
            counts.iter().sum::<usize>(),
            match plan["scenario"].as_str().unwrap() {
                "greedy" => 10,
                "fork" => 6,
                "canary" => 5,
                "recreate" | "cold-owner" => 4,
                _ => unreachable!(),
            }
        );
        assert_eq!(
            counts.len(),
            if plan["scenario"] == "recreate" { 2 } else { 1 }
        );
        let d = tempfile::tempdir().unwrap();
        let capture = new_capture(d.path(), &"a".repeat(64), plan, 0, counts[0]);
        let c = capture.config();
        assert_eq!(c.max_invocations, counts[0]);
        assert_eq!(c.max_events_per_owner, counts[0] * 10000);
        assert!(c.host_reserve(2).unwrap() > c.max_bytes_per_owner as u64 * 2);
        let cfg = recipe_config();
        let pages = ParallelRankCapacity {
            page_size: cfg.page_size,
            max_positions: cfg.max_positions,
            max_rows: cfg.max_batch_tokens,
            sessions: cfg.session_capacity,
            logical_pages: cfg.max_pages,
        }
        .physical_pages()
        .unwrap();
        eprintln!(
            "RECIPE_BUDGET scenario={} calls={:?} physical_pages={} payload_per_owner={} events_per_owner={} trace_host={} execution_plus_trace_host={} device_target={}",
            plan["scenario"],
            counts,
            pages,
            c.max_bytes_per_owner,
            c.max_events_per_owner,
            c.host_reserve_with_pages(2, pages).unwrap(),
            (256u64 << 20) + c.host_reserve_with_pages(2, pages).unwrap(),
            22u64 << 30
        );
        let mut bad = plan.clone();
        bad["decode_steps"] = json!(8);
        assert!(validate_plan(&bad).is_err());
        let mut truncated = plan.clone();
        truncated["operations"].as_array_mut().unwrap().pop();
        assert!(validate_plan(&truncated).is_err());
        let mut relabelled = plan.clone();
        relabelled["scenario"] = json!("greedy");
        relabelled["decode_steps"] = json!(1);
        assert!(validate_plan(&relabelled).is_err());
        let mut widened = plan.clone();
        widened["limits"]["max_total_capture_bytes"] = json!(8u64 << 30);
        assert!(validate_plan(&widened).is_err());
        let mut reordered = plan.clone();
        reordered["operations"].as_array_mut().unwrap().swap(0, 1);
        assert!(validate_plan(&reordered).is_err());
    }
}

#[test]
#[ignore = "CPU headers only: requires FERRULE_MIXED_HEADER_DIR and hidden GPUs"]
fn full40_recipe_header_budget_without_cuda() {
    assert_eq!(std::env::var("CUDA_VISIBLE_DEVICES").as_deref(), Ok(""));
    let path = std::env::var("FERRULE_MIXED_HEADER_DIR").unwrap();
    let (_, resources) =
        ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(Path::new(&path)).unwrap();
    let plan = ParallelExecutionPlan::describe(resources.spec(), &placement(), contract(), vec![])
        .unwrap();
    let cfg = recipe_config();
    let capacity = ParallelRankCapacity {
        page_size: cfg.page_size,
        max_positions: cfg.max_positions,
        max_rows: cfg.max_batch_tokens,
        sessions: cfg.session_capacity,
        logical_pages: cfg.max_pages,
    };
    let inputs = ferrule_runtime::parallel::pipeline::MixedTensorExecutionPlan::new(
        resources.spec(),
        plan.topology(),
        plan.segments(),
        plan.layout(),
        cfg,
        HostCollectiveLimits {
            max_ranks: 2,
            max_elements_per_rank: cfg.max_batch_tokens * resources.spec().vocab_size().div_ceil(2),
            max_host_bytes: 64 << 20,
        },
    )
    .unwrap();
    let report = plan
        .resource_requirements_from_model(
            &resources,
            capacity,
            ParallelOwnerMargins {
                context_bytes: 64 << 20,
                allocator_bytes: 64 << 20,
            },
            &[inputs.collective().as_ref().clone()],
        )
        .unwrap();
    assert!(report.resource_complete());
    for owner in report.owners() {
        let total = owner.device_bytes().unwrap().unwrap() + (8 << 20);
        assert!(total <= 22u64 << 30);
        eprintln!(
            "FULL_RECIPE_DEVICE owner={} device={} state_copies={} weights={} state={} kv={} total_with_trace={} target={}",
            owner.owner.get(),
            owner.device,
            capacity.state_copies().unwrap(),
            owner.storage.weight_grants,
            owner.storage.state.total_grants,
            owner.storage.kv_grants,
            total,
            22u64 << 30
        );
    }
    eprintln!(
        "FULL_RECIPE_HOST_KNOWN {}",
        report.known_host_bytes().unwrap()
    );
}
