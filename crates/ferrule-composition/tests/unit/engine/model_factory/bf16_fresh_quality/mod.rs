//! Fresh quality gate reusing the maintained public-factory full40 runner.
use super::*;
use crate::model_factory::bf16_serving::expert_read_reporting::{
    PredictionFingerprints, PredictionSample,
};

mod retained_transfer;

pub(super) fn top1(values: &[f32]) -> u32 {
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    values
        .iter()
        .enumerate()
        .fold(0, |best, (i, v)| if *v > values[best] { i } else { best }) as u32
}
#[test]
fn fresh_greedy_ties_choose_lower_token_id() {
    assert_eq!(top1(&[1., 3., 3., 2.]), 1);
    assert_eq!(top1(&[-2., -1., -3.]), 1);
    assert_eq!(top1(&[-0., 0., -0.]), 0);
    assert!(std::panic::catch_unwind(|| top1(&[])).is_err());
    assert!(std::panic::catch_unwind(|| top1(&[f32::NAN])).is_err());
}

#[test]
#[ignore = "sole GPU owner; three genuinely fresh prompts, retained HF histories, full-vocab/state checks"]
fn mixed_fresh_quality_retained_generation() {
    run(false, Full40TestConfig::Historical8);
}
#[test]
#[ignore = "sole GPU owner; separate cold owner, first fresh prompt only; requires prior quality pass"]
fn mixed_fresh_quality_cold_owner_restart() {
    run(true, Full40TestConfig::Historical8);
}

#[test]
#[ignore = "sole GPU owner; frozen storage-middle64/maxdecode2; unchanged three-prompt independent fresh reference"]
fn mixed_storage_middle64_fresh_quality_retained_generation() {
    run(false, Full40TestConfig::StorageMiddle64);
}

#[test]
#[ignore = "sole GPU owner; separate cold storage-middle64 owner; requires primary pass with the same frozen config/reference"]
fn mixed_storage_middle64_fresh_quality_cold_owner_restart() {
    run(true, Full40TestConfig::StorageMiddle64);
}

#[test]
#[ignore = "sole GPU owner; enabled Blocking host reads; unchanged three-prompt storage-middle64 fresh reference"]
fn mixed_storage_middle64_blocking_fresh_quality_retained_generation() {
    run(false, Full40TestConfig::StorageMiddle64Blocking);
}

#[test]
#[ignore = "sole GPU owner; enabled Lookahead host reads; unchanged three-prompt storage-middle64 fresh reference"]
fn mixed_storage_middle64_lookahead_fresh_quality_retained_generation() {
    run(false, Full40TestConfig::StorageMiddle64Lookahead);
}

#[test]
#[ignore = "sole GPU owner; cold Blocking owner; requires primary pass with identical frozen config/reference"]
fn mixed_storage_middle64_blocking_fresh_quality_cold_owner_restart() {
    run(true, Full40TestConfig::StorageMiddle64Blocking);
}

#[test]
#[ignore = "sole GPU owner; cold Lookahead owner; requires primary pass with identical frozen config/reference"]
fn mixed_storage_middle64_lookahead_fresh_quality_cold_owner_restart() {
    run(true, Full40TestConfig::StorageMiddle64Lookahead);
}

#[test]
#[ignore = "sole GPU owner; Blocking ordinary no-diagnostic-observer equality; requires newly pinned matching diagnostic primary report"]
fn mixed_storage_middle64_blocking_fresh_quality_no_observer_equality() {
    run_with_observer(false, Full40TestConfig::StorageMiddle64Blocking, false);
}

#[test]
#[ignore = "sole GPU owner; Lookahead ordinary no-diagnostic-observer equality; requires newly pinned matching diagnostic primary report"]
fn mixed_storage_middle64_lookahead_fresh_quality_no_observer_equality() {
    run_with_observer(false, Full40TestConfig::StorageMiddle64Lookahead, false);
}

#[test]
#[ignore = "sole GPU owner; explicit retained blocking uploads; unchanged quality/reference/quota and primary policy binding"]
fn mixed_storage_middle64_retained_blocking_fresh_quality_retained_generation() {
    run(false, Full40TestConfig::StorageMiddle64RetainedBlocking);
}

#[test]
#[ignore = "sole GPU owner; explicit retained blocking uploads; unchanged quality/reference/quota and primary policy binding"]
fn mixed_storage_middle64_retained_blocking_fresh_quality_cold_owner_restart() {
    run(true, Full40TestConfig::StorageMiddle64RetainedBlocking);
}

#[test]
#[ignore = "sole GPU owner; explicit retained blocking uploads; unchanged quality/reference/quota and primary policy binding"]
fn mixed_storage_middle64_retained_blocking_fresh_quality_no_observer_equality() {
    run_with_observer(
        false,
        Full40TestConfig::StorageMiddle64RetainedBlocking,
        false,
    );
}

#[test]
#[ignore = "sole GPU owner; explicit retained lookahead uploads; unchanged quality/reference/quota and primary policy binding"]
fn mixed_storage_middle64_retained_lookahead_fresh_quality_retained_generation() {
    run(false, Full40TestConfig::StorageMiddle64RetainedLookahead);
}

#[test]
#[ignore = "sole GPU owner; explicit retained lookahead uploads; unchanged quality/reference/quota and primary policy binding"]
fn mixed_storage_middle64_retained_lookahead_fresh_quality_cold_owner_restart() {
    run(true, Full40TestConfig::StorageMiddle64RetainedLookahead);
}

#[test]
#[ignore = "sole GPU owner; explicit retained lookahead uploads; unchanged quality/reference/quota and primary policy binding"]
fn mixed_storage_middle64_retained_lookahead_fresh_quality_no_observer_equality() {
    run_with_observer(
        false,
        Full40TestConfig::StorageMiddle64RetainedLookahead,
        false,
    );
}

fn assert_cold_primary(
    prior: &Value,
    test_config: Full40TestConfig,
    precision: NumericFp8Precision,
    pin: &str,
) {
    assert_eq!(
        prior["passed"], true,
        "cold restart is only admitted after fresh quality pass"
    );
    // Historical receipts predate explicit config fields and remain historical.
    // They cannot admit a new storage-middle64 cold-owner trial.
    if test_config != Full40TestConfig::Historical8 {
        assert_eq!(prior["complete"], true);
        assert_eq!(prior["cold_owner_restart"], false);
        assert_eq!(prior["test_config"], test_config.manifest(precision));
        assert_eq!(prior["oracle_manifest_sha256"], pin);
        assert_eq!(prior["ordinary_no_observer_claim"], false);
        if let Some(read) = test_config.expert_read_pipeline() {
            assert_eq!(
                prior["expert_read_quality"]["passed"], true,
                "cold owner requires positive bounded worker evidence in the primary report"
            );
            assert_eq!(prior["expert_read_quality"]["requested"], json!(read));
        }
    }
}

#[test]
fn storage_middle64_cold_primary_requires_same_frozen_profile_and_reference() {
    let config = Full40TestConfig::StorageMiddle64;
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    let prior = json!({"passed":true,"complete":true,"cold_owner_restart":false,
        "test_config":config.manifest(precision),"oracle_manifest_sha256":"test-pin",
        "ordinary_no_observer_claim":false});
    assert_cold_primary(&prior, config, precision, "test-pin");
    for field in [
        "test_config",
        "oracle_manifest_sha256",
        "passed",
        "complete",
        "cold_owner_restart",
    ] {
        let mut bad = prior.clone();
        bad[field] = Value::Null;
        assert!(
            std::panic::catch_unwind(|| assert_cold_primary(&bad, config, precision, "test-pin"))
                .is_err(),
            "{field}"
        );
    }
}

#[test]
fn enabled_storage_middle64_cold_primary_rejects_schedule_and_host_budget_drift() {
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    for config in [
        Full40TestConfig::StorageMiddle64Blocking,
        Full40TestConfig::StorageMiddle64Lookahead,
    ] {
        let prior = json!({"passed":true,"complete":true,"cold_owner_restart":false,
            "test_config":config.manifest(precision),"oracle_manifest_sha256":"test-pin",
            "ordinary_no_observer_claim":false});
        let mut prior = prior;
        prior["expert_read_quality"] =
            json!({"passed":true,"requested":config.expert_read_pipeline().unwrap()});
        assert_cold_primary(&prior, config, precision, "test-pin");
        for field in ["passed", "requested"] {
            let mut bad = prior.clone();
            bad["expert_read_quality"][field] = Value::Null;
            assert!(
                std::panic::catch_unwind(|| assert_cold_primary(
                    &bad, config, precision, "test-pin"
                ))
                .is_err()
            );
        }
        for field in [
            "schedule",
            "max_host_bytes",
            "max_metadata_bytes",
            "worker_stack_bytes",
        ] {
            let mut bad = prior.clone();
            bad["test_config"]["expert_read_pipeline"][field] = Value::Null;
            assert!(
                std::panic::catch_unwind(|| assert_cold_primary(
                    &bad, config, precision, "test-pin"
                ))
                .is_err(),
                "{field}"
            );
        }
        let other = if config == Full40TestConfig::StorageMiddle64Blocking {
            Full40TestConfig::StorageMiddle64Lookahead
        } else {
            Full40TestConfig::StorageMiddle64Blocking
        };
        assert!(
            std::panic::catch_unwind(|| assert_cold_primary(&prior, other, precision, "test-pin"))
                .is_err()
        );
    }
}

#[test]
fn ordinary_equality_requires_identical_derived_profile_and_pinned_diagnostic_receipt() {
    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    for config in [
        Full40TestConfig::StorageMiddle64Blocking,
        Full40TestConfig::StorageMiddle64Lookahead,
    ] {
        let fixture = crate::model_factory::mixed_serving::fixture::Fixture::new();
        let plan = prepare_full40_with_config(fixture.dir.path(), precision, config);
        let effective = full40_effective_config(&plan, config, precision);
        let mut calls = serde_json::Map::new();
        for i in 0..64 {
            calls.insert(i.to_string(), json!({"sample":i}));
        }
        let prior = json!({"schema":"ferrule.bf16-fresh-quality-result.v1","passed":true,"complete":true,"cold_owner_restart":false,
            "test_config":config.manifest(precision),"effective_config":effective,"oracle_manifest_sha256":"same-pin",
            "observer_mode":"diagnostic_routes_and_states","ordinary_no_observer_claim":false,
            "expert_read_quality":{"passed":true},"prediction_fingerprints":{"calls":calls}});
        expert_read_reporting::assert_diagnostic_primary(
            &prior,
            &config.manifest(precision),
            &effective,
            "same-pin",
        );
        for field in [
            "test_config",
            "effective_config",
            "oracle_manifest_sha256",
            "expert_read_quality",
            "ordinary_no_observer_claim",
            "prediction_fingerprints",
        ] {
            let mut bad = prior.clone();
            bad[field] = Value::Null;
            assert!(
                std::panic::catch_unwind(|| expert_read_reporting::assert_diagnostic_primary(
                    &bad,
                    &config.manifest(precision),
                    &effective,
                    "same-pin"
                ))
                .is_err(),
                "{field}"
            );
        }
        assert_eq!(
            effective["requested_expert_read_pipeline"],
            effective["effective_expert_read_pipeline"]
        );
        assert_eq!(
            expert_read_reporting::observer_fields(false)["diagnostic_trace_installed"],
            false
        );
    }
}

fn write_report(path: &std::path::Path, file: &Option<Rc<RefCell<std::fs::File>>>, report: &Value) {
    let bytes = serde_json::to_vec_pretty(report).unwrap();
    if let Some(file) = file {
        let mut file = file.borrow_mut();
        file.seek(std::io::SeekFrom::Start(0)).unwrap();
        file.set_len(0).unwrap();
        file.write_all(&bytes).unwrap();
    } else {
        std::fs::write(path, bytes).unwrap();
    }
}

#[allow(clippy::too_many_arguments)]
fn record_prediction(
    reports: &Option<Rc<RefCell<PredictionReports>>>,
    fingerprints: &Option<Rc<RefCell<PredictionFingerprints>>>,
    oracle: Option<&Oracle>,
    stream: &'static str,
    case: &str,
    step: usize,
    logits: &[Vec<f32>],
    token: u32,
    expected: u32,
    history_matches: bool,
) {
    if let Some(reports) = reports {
        let oracle = oracle.expect("retained unchanged fresh reference");
        let actual: Vec<f32> = logits.iter().flatten().copied().collect();
        let label = format!("{case}/{stream}/step.{step:02}");
        reports.borrow_mut().record(
            stream,
            PredictionSample {
                rows: logits.len(),
                actual: &actual,
                expected: &oracle.values(&format!("step.{step:02}.logits")),
                actual_token: token,
                expected_token: expected,
                history_matches,
                label: &label,
            },
        );
        fingerprints
            .as_ref()
            .unwrap()
            .borrow_mut()
            .record(&label, logits.len(), &actual, token);
    }
}

fn run(cold: bool, test_config: Full40TestConfig) {
    run_with_observer(cold, test_config, true);
}

fn run_with_observer(cold: bool, test_config: Full40TestConfig, diagnostic: bool) {
    let reads = test_config
        .expert_read_pipeline()
        .map(|read| Rc::new(RefCell::new(ReadQualityLog::new(read, diagnostic))));
    let predictions = reads
        .as_ref()
        .map(|_| Rc::new(RefCell::new(PredictionReports::new(diagnostic))));
    if !diagnostic {
        assert!(reads.is_some() && !cold);
        for flag in [
            "FERRULE_BF16_FRESH_NARROW_DIR",
            "FERRULE_BF16_FRESH_NARROW_CASE",
            "FERRULE_NUMERIC_BF16_FIRST_TRACE_DIR",
        ] {
            assert!(
                std::env::var_os(flag).is_none(),
                "ordinary equality rejects diagnostic overrides: {flag}"
            );
        }
    }

    let precision = NumericFp8Precision::Bf16ModelMixedF32Accumulate;
    let config_manifest = test_config.manifest(precision);
    let limits = test_config.capacity(precision);
    if test_config != Full40TestConfig::Historical8 {
        assert_eq!(
            std::env::var("FERRULE_NUMERIC_BF16_FACTORY_GPU").as_deref(),
            Ok("1")
        );
    }
    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
    let directory = required_path("FERRULE_BF16_FRESH_DIR");
    let output = required_path("FERRULE_BF16_FRESH_REPORT");
    assert!(!output.exists());
    let pin = std::env::var("FERRULE_BF16_FRESH_SHA256").unwrap();
    let spec: Value =
        serde_json::from_slice(&std::fs::read(directory.join("INPUT.json")).unwrap()).unwrap();
    assert_eq!(spec["schema"], "ferrule.bf16-fresh-quality-input.v1");
    assert_eq!(spec["decode_steps"], 8);
    assert_eq!(spec["cases"].as_array().unwrap().len(), 3);
    let oracle_dir = directory.join("oracle");
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(oracle_dir.join("manifest.json")).unwrap()).unwrap();
    check_hash(
        &directory.join("INPUT.json"),
        manifest["input_sha256"].as_str().unwrap(),
    );
    if cold {
        let prior: Value = serde_json::from_slice(
            &std::fs::read(
                std::env::var_os("FERRULE_BF16_FRESH_PRIMARY_REPORT")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| directory.join("quality.json")),
            )
            .unwrap(),
        )
        .unwrap();
        assert_cold_primary(&prior, test_config, precision, &pin);
    }
    let narrow_case = std::env::var("FERRULE_BF16_FRESH_NARROW_CASE").ok();
    if let Some(case) = &narrow_case {
        assert!(["qa", "arithmetic", "chinese"].contains(&case.as_str()));
    }
    let cases: Vec<_> = spec["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| {
            narrow_case
                .as_ref()
                .is_none_or(|name| c["id"].as_str() == Some(name))
        })
        .take(if cold { 1 } else { 3 })
        .map(|c| {
            let id = c["id"].as_str().unwrap().to_owned();
            let oracle = Oracle::open_bf16_fresh(&oracle_dir, &id, &pin);
            let ids: Vec<u32> = serde_json::from_value(c["input_ids"].clone()).unwrap();
            assert!((1..=8).contains(&ids.len()));
            assert_eq!(oracle.report["calls"][0]["input_token_ids"], c["input_ids"]);
            let generated: Vec<u32> =
                serde_json::from_value(oracle.report["generated_ids"].clone()).unwrap();
            assert_eq!(generated.len(), 9);
            for (step, token) in generated.into_iter().enumerate() {
                assert_reference_next_token(&oracle, step, &format!("step.{step:02}"));
                assert_eq!(oracle.report["calls"][step]["next_token_id"], token);
            }
            (id, oracle, ids)
        })
        .collect();
    let plan = prepare_full40_with_config(
        &required_path("FERRULE_NUMERIC_BF16_MODEL"),
        precision,
        test_config,
    );
    let effective_config = full40_effective_config(&plan, test_config, precision);
    let observed_primary = if diagnostic {
        None
    } else {
        let path = required_path("FERRULE_BF16_FRESH_OBSERVED_REPORT");
        let report_pin = std::env::var("FERRULE_BF16_FRESH_OBSERVED_REPORT_SHA256")
            .expect("explicit diagnostic primary report pin");
        check_hash(&path, &report_pin);
        let prior: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        expert_read_reporting::assert_diagnostic_primary(
            &prior,
            &config_manifest,
            &effective_config,
            &pin,
        );
        Some((prior, report_pin))
    };
    let fingerprints = reads.as_ref().map(|_| {
        Rc::new(RefCell::new(PredictionFingerprints::new(
            observed_primary
                .as_ref()
                .map(|(prior, _)| prior["prediction_fingerprints"].clone()),
        )))
    });
    let new_report = reads.as_ref().map(|_| {
        Rc::new(RefCell::new(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&output)
                .unwrap(),
        ))
    });
    let memory = plan.numeric_bf16_memory_plan().unwrap().unwrap();
    assert!(memory.total_device_bytes() <= limits.device_bytes);
    let admitted = memory.total_device_bytes();
    drop(memory);
    let report = Rc::new(RefCell::new(StrictReport::default()));
    let summary = Rc::new(RefCell::new(Vec::<Value>::new()));
    let result = Rc::clone(&report);
    let records = Rc::clone(&summary);
    let progress_path = output.clone();
    let progress_config = config_manifest.clone();
    let progress_effective = effective_config.clone();
    let narrow = std::env::var_os("FERRULE_BF16_FRESH_NARROW_DIR").map(PathBuf::from);
    let narrow_mode = narrow.is_some();
    let narrow_step =
        std::env::var("FERRULE_BF16_FRESH_NARROW_STEP").map_or(1, |v| v.parse::<usize>().unwrap());
    let narrow_start = std::env::var("FERRULE_BF16_FRESH_NARROW_LAYER")
        .map_or(14, |v| v.parse::<usize>().unwrap());
    assert!((1..=8).contains(&narrow_step) && narrow_start <= 36);
    if let Some(path) = &narrow {
        std::fs::create_dir(path).unwrap();
    }

    let call_reads = reads.clone();
    let call_predictions = predictions.clone();
    let call_fingerprints = fingerprints.clone();
    let progress_file = new_report.clone();
    crate::model_factory::bf16_serving::NUMERIC_BF16_TEST_RUNNER.with(|slot| {
        assert!(slot.borrow().is_none());
        *slot.borrow_mut()=Some(Box::new(move |runner| {
            configure_full40_expert_transfer(runner, test_config);
            assert_eq!(runner.forward_executor().module().expert_read_pipeline_options(), test_config.expert_read_pipeline());
            assert_eq!(runner.forward_executor().module().numeric_fp8_precision(), Some(precision));
            assert!(runner.forward_executor().module().numeric_bf16_profile().is_some());
            let routes=Rc::new(RefCell::new(BTreeMap::new()));let capture=Rc::clone(&routes);
            let active=Rc::new(RefCell::new(false));let observe=Rc::clone(&active);
            let mut trace_index=0usize;
            let mut trace_log=narrow.as_ref().map(|path|std::fs::File::create(path.join("events.jsonl")).unwrap());
            if diagnostic { runner.forward_executor().module().set_diagnostic_trace(move|event| {
                if *observe.borrow() && event.layer.is_some_and(|l|(narrow_start..narrow_start+4).contains(&l))
                    && let Some(path)=&narrow {
                    let values=event.operators.download_f32_buffer(event.buffer)?;
                    let file=format!("{trace_index:04}.f32");trace_index+=1;
                    std::fs::write(path.join(&file),values.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())?;
                    writeln!(trace_log.as_mut().unwrap(),"{}",json!({"event":event.name,"layer":event.layer,"shape":[event.shape.rows(),event.shape.width()],"file":file,"weight":event.parameter.map(|p|p.weight().slice().name.clone()),"ids":event.routes.map(|r|r.expert_ids()),"weights":event.routes.map(|r|r.weights())}))?;
                }
                if let Some(r)=event.routes {
                    assert!(capture.borrow_mut().insert(event.layer.unwrap(),(r.expert_ids().to_vec(),r.weights().to_vec(),event.operators.download_f32_buffer(event.buffer)?)).is_none());
                }
                Ok(())
            }); } else { runner.forward_executor().module().clear_diagnostic_trace(); }
            let mut transaction=0;
            let mut retained_oracles = BTreeMap::new();
            for (case,oracle,prompt) in cases {
                let mut states=vec![runner.create_sequence_state().unwrap()];let mut pages=vec![Vec::new()];
                let mut generated=Vec::new();let mut top1_failures=Vec::new();let mut token_checks=Vec::new();
                for step in 0..9 {
                    let call=&oracle.report["calls"][step];let stage=format!("step.{step:02}");
                    let tokens:Vec<u32>=serde_json::from_value(call["input_token_ids"].clone()).unwrap();
                    assert_eq!(states[0].core().position(),call["position_start"].as_u64().unwrap()as usize);
                    *active.borrow_mut()=step==narrow_step;
                    transaction+=1;
                    let logits=observe_operation(runner,&call_reads,&format!("{case}/common_history/{stage}"),"prediction",|runner|crate::model_factory::qwen35::full_logits_step(runner,&mut states,&mut pages,&tokens,if step==0 {ForwardPhase::Prefill}else{ForwardPhase::Decode},transaction));
                    assert_eq!(logits.len(),tokens.len());assert!(logits.iter().all(|row|row.len()==248320));
                                        assert_eq!(states[0].core().position(),call["sequence_length_after"].as_u64().unwrap()as usize);
                                        let next=top1(logits.last().unwrap());let expected=call["next_token_id"].as_u64().unwrap()as u32;generated.push(next);
                                        token_checks.push(json!({"step":step,"actual":next,"expected":expected,"equal":next==expected}));
                    if next!=expected {top1_failures.push(json!({"step":step,"actual":next,"expected":expected}));}
                    let actual:Vec<_>=logits.into_iter().flatten().collect();typed(&actual);
                    let mut r=result.borrow_mut();r.logits+=actual.len();
                    r.bf16_values(&actual,&oracle.values(&format!("{stage}.logits")),&format!("{case}/{stage}/all_logits"));
                    if let Some(reports) = &call_predictions {
                        reports.borrow_mut().record(
                            "common_history",
                            PredictionSample {
                                rows: tokens.len(),
                                actual: &actual,
                                expected: &oracle.values(&format!("{stage}.logits")),
                                actual_token: next,
                                expected_token: expected,
                                history_matches: true,
                                label: &format!("{case}/common_history/{stage}"),
                            },
                        );
                    }
                    if let Some(fingerprints) = &call_fingerprints { fingerprints.borrow_mut().record(&format!("{case}/common_history/{stage}"),tokens.len(),&actual,next); }
                    if diagnostic {
                    let captured=std::mem::take(&mut *routes.borrow_mut());assert_eq!(captured.len(),40);
                    for (layer,(ids,weights,raw)) in captured {
                        let key=format!("{stage}.layers.{layer:02}");
                        r.route_ids(&ids,&oracle.ids(&format!("{key}.selected_expert_ids")),&format!("{case}/{key}/ordered_routes"));
                        r.bf16_values(&weights,&oracle.values(&format!("{key}.routing_weights")),&format!("{case}/{key}/weights"));
                        r.bf16_values(&raw,&oracle.values(&format!("{key}.raw_router_logits")),&format!("{case}/{key}/raw_router_logits"));
                    }
                    check_state(runner,&states[0],&pages[0],&oracle,&case,&stage,&mut r,&call_reads);
                    }
                    let failed=!r.failures.is_empty()||!top1_failures.is_empty();
                    let mut data=json!({"complete":false,"passed":false,"test_config":progress_config,"effective_config":progress_effective,"observer_mode":if diagnostic {"diagnostic_routes_and_states"}else{"ordinary_no_diagnostic_observer"},"ordinary_no_observer_claim":!diagnostic,"concurrent_scheduler_claim":false,"greedy_tie_policy":"lower_token_id","current_case":case,"current_step":step,"actual_next_token_id":next,"expected_next_token_id":expected,"common_history_generated_ids":generated,"next_token_checks":token_checks,"metrics":r.metrics,"failures":r.failures,"top1_failures":top1_failures});
                    attach_read_reports(&mut data,&call_reads,call_predictions.as_ref().map(|p|p.borrow()).as_deref());
                    if let Some(fingerprints)=&call_fingerprints {data["prediction_fingerprints"]=fingerprints.borrow().json();}
                    write_report(&progress_path,&progress_file,&data);drop(r);
                    if failed {break;}
                }
                let expected:Vec<u32>=serde_json::from_value(oracle.report["generated_ids"].clone()).unwrap();
                let common_pass=result.borrow().failures.is_empty()&&top1_failures.is_empty();
                records.borrow_mut().push(json!({"case":case,"prompt_ids":prompt,"common_history_generated_ids":generated,"next_token_checks":token_checks,"oracle_generated_ids":expected,"free_greedy_ids":[],"top1_failures":top1_failures,"free_greedy_status":"pending_all_teacherforced_cases","free_greedy_equal":null,"fork_snapshot_equal":null,"reset_reused_state":false}));
                runner.try_release_sequence_state(states.remove(0)).unwrap();runner.release_kv_pages(&pages[0]).unwrap();
                if !common_pass {break;}
                if call_predictions.is_some() { retained_oracles.insert(case,oracle); }
            }
            let all_common_pass=result.borrow().failures.is_empty()
                &&records.borrow().len()==if cold {1}else{3}
                &&records.borrow().iter().all(|c|c["top1_failures"].as_array().unwrap().is_empty());
            if all_common_pass && !narrow_mode {
                *active.borrow_mut()=false;
                let case_count=records.borrow().len();
                for index in 0..case_count {
                    let record=records.borrow()[index].clone();
                    let case=record["case"].as_str().unwrap();
                    let prompt:Vec<u32>=serde_json::from_value(record["prompt_ids"].clone()).unwrap();
                    let expected:Vec<u32>=serde_json::from_value(record["oracle_generated_ids"].clone()).unwrap();
                    let oracle=retained_oracles.get(case);
                    let mut states=vec![runner.create_sequence_state().unwrap()];let mut pages=vec![Vec::new()];
                    let mut free=Vec::new();let mut free_checks=Vec::new();let mut fed=prompt.clone();
                    for step in 0..9 {
                        transaction+=1;
                        let y=observe_operation(runner,&call_reads,&format!("{case}/free_feedback/step.{step:02}"),"prediction",|runner|crate::model_factory::qwen35::full_logits_step(runner,&mut states,&mut pages,&fed,if step==0 {ForwardPhase::Prefill}else{ForwardPhase::Decode},transaction));
                        assert_eq!(y.len(),fed.len());assert!(y.iter().all(|row|row.len()==248320));
                        let token=top1(y.last().unwrap());
                        record_prediction(&call_predictions,&call_fingerprints,oracle,"free_feedback",case,step,&y,token,expected[step],free==expected[..step]);
                        free.push(token);
                        free_checks.push(json!({"step":step,"actual":token,"expected":expected[step],"equal":token==expected[step]}));
                        fed=vec![token];routes.borrow_mut().clear();
                    }
                    records.borrow_mut()[index]["free_greedy_ids"]=json!(free);
                    records.borrow_mut()[index]["free_greedy_next_token_checks"]=json!(free_checks);
                    records.borrow_mut()[index]["free_greedy_equal"]=json!(free==expected);
                    records.borrow_mut()[index]["free_greedy_status"]=json!("completed_independent_feedback");
                    if index==0 &&free==expected {
                        observe_operation(runner,&call_reads,&format!("{case}/reset"),"reset_sequence",|runner|runner.reset_sequence_state(&mut states[0])).unwrap();
                        runner.release_kv_pages(&pages[0]).unwrap();pages[0].clear();assert_eq!(states[0].core().position(),0);
                        let mut repeated=Vec::new();let mut reset_checks=Vec::new();let mut fed=prompt;
                        for step in 0..9 {
                            transaction+=1;
                            let y=observe_operation(runner,&call_reads,&format!("{case}/reset_replay/step.{step:02}"),"prediction",|runner|crate::model_factory::qwen35::full_logits_step(runner,&mut states,&mut pages,&fed,if step==0 {ForwardPhase::Prefill}else{ForwardPhase::Decode},transaction));
                            assert_eq!(y.len(),fed.len());assert!(y.iter().all(|row|row.len()==248320));
                            let token=top1(y.last().unwrap());
                            record_prediction(&call_predictions,&call_fingerprints,oracle,"reset_replay",case,step,&y,token,expected[step],repeated==expected[..step]);
                            repeated.push(token);
                            reset_checks.push(json!({"step":step,"actual":token,"expected":expected[step],"equal":token==expected[step]}));
                            fed=vec![token];routes.borrow_mut().clear();
                            if step==0 {
                                let fork=observe_operation(runner,&call_reads,&format!("{case}/fork"),"fork_sequence",|runner|runner.fork_sequence_state_from(&states[0],states[0].core().position())).unwrap();
                                assert_eq!(fork.core().position(),states[0].core().position());
                                if diagnostic {
                                    let mut compared=0;
                                    for(layer,(a,b))in states[0].kv_state().bf16_layers().unwrap().iter().zip(fork.kv_state().bf16_layers().unwrap()).enumerate() {
                                        if let(Some(a),Some(b))=(a,b){
                                            let left=observe_operation(runner,&call_reads,&format!("{case}/fork/parent_snapshot/{layer}"),"diagnostic_state_snapshot",|_|a.diagnostic_snapshot()).unwrap();
                                            let right=observe_operation(runner,&call_reads,&format!("{case}/fork/child_snapshot/{layer}"),"diagnostic_state_snapshot",|_|b.diagnostic_snapshot()).unwrap();
                                            assert_eq!(left,right);compared+=1;
                                        }
                                    }
                                    assert_eq!(compared,30);
                                    records.borrow_mut()[index]["fork_snapshot_equal"]=json!(true);
                                }
                                if call_predictions.is_some() {
                                    let mut fork_states=vec![fork];let mut fork_pages=pages.clone();transaction+=1;
                                    // Same-history replay in the existing partial page, not a divergent COW test.
                                    let y=observe_operation(runner,&call_reads,&format!("{case}/fork_replay/step.01"),"prediction",|runner|crate::model_factory::qwen35::full_logits_step(runner,&mut fork_states,&mut fork_pages,&fed,ForwardPhase::Decode,transaction));
                                    assert_eq!(y.len(),1);assert_eq!(y[0].len(),248320);assert_eq!(fork_pages,pages);
                                    let next=top1(&y[0]);
                                    record_prediction(&call_predictions,&call_fingerprints,oracle,"fork_replay",case,1,&y,next,expected[1],token==expected[0]);
                                    records.borrow_mut()[index]["fork_replay_next_token_checks"]=json!([{ "actual":next,"expected":expected[1],"equal":next==expected[1]}]);
                                    records.borrow_mut()[index]["fork_replay_equal"]=json!(next==expected[1]);
                                    // The page remains owned by the parent; releasing it here would destroy its prefix.
                                    runner.try_release_sequence_state(fork_states.remove(0)).unwrap();routes.borrow_mut().clear();
                                } else { runner.try_release_sequence_state(fork).unwrap(); }
                            }
                        }
                        records.borrow_mut()[index]["reset_reused_state"]=json!(true);
                        records.borrow_mut()[index]["reset_generated_ids"]=json!(repeated);
                        records.borrow_mut()[index]["reset_next_token_checks"]=json!(reset_checks);
                        records.borrow_mut()[index]["reset_equal"]=json!(repeated==expected);
                    }
                    runner.try_release_sequence_state(states.remove(0)).unwrap();runner.release_kv_pages(&pages[0]).unwrap();
                    if free!=expected {break;}
                }
            }
            runner.forward_executor().module().clear_diagnostic_trace();
            assert_full40_expert_transfer_exercised(runner, test_config);
        }));
    });
    let mut initial = json!({"schema":"ferrule.bf16-fresh-quality-result.v1","complete":false,"passed":false,"status":"prepared; computation not started","test_config":config_manifest,"effective_config":effective_config,"cold_owner_restart":cold,"oracle_manifest_sha256":pin});
    if reads.is_some() {
        attach_read_reports(
            &mut initial,
            &reads,
            predictions.as_ref().map(|p| p.borrow()).as_deref(),
        );
        write_report(&output, &new_report, &initial);
    }
    let mut engine = if reads.is_none() {
        plan.build().unwrap()
    } else {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| plan.build().unwrap())) {
            Ok(engine) => engine,
            Err(panic) => {
                initial["status"] = json!("build/forward panic; partial evidence only");
                initial["cases"] = json!(*summary.borrow());
                initial["metrics"] = json!(report.borrow().metrics);
                initial["failures"] = json!(report.borrow().failures);
                if let Some(fingerprints) = &fingerprints {
                    initial["prediction_fingerprints"] = fingerprints.borrow().json();
                }
                attach_read_reports(
                    &mut initial,
                    &reads,
                    predictions.as_ref().map(|p| p.borrow()).as_deref(),
                );
                write_report(&output, &new_report, &initial);
                std::panic::resume_unwind(panic)
            }
        }
    };
    assert_eq!(engine.bound_layer_count(), Some(40));
    assert_eq!(engine.model_info().backend, precision.hybrid_backend_name());
    let device = crate::model_factory::bf16_serving::NUMERIC_BF16_TEST_DEVICE
        .with(|s| s.borrow_mut().take())
        .unwrap();
    let peak = device.operators().allocator_metrics().peak_reserved_bytes;
    assert!(peak <= limits.device_bytes);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while engine.shutdown().unwrap() != InferenceShutdownProgress::Complete {
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    drop(engine);
    assert_eq!(device.live_state_bytes(), 0);
    assert!(!device.needs_quarantine());
    let r = report.borrow();
    let records = summary.borrow();
    let passed = reads.as_ref().is_none_or(|log| log.borrow().passed())
        && predictions
            .as_ref()
            .is_none_or(|p| p.borrow().passed(if cold { 1 } else { 3 }))
        && fingerprints.as_ref().is_none_or(|f| f.borrow().passed())
        && r.failures.is_empty()
        && records.len() == if cold { 1 } else { 3 }
        && records.iter().all(|c| {
            c["free_greedy_equal"] == true
                && c["top1_failures"].as_array().unwrap().is_empty()
                && (!c["reset_reused_state"].as_bool().unwrap_or(false) || c["reset_equal"] == true)
        });
    let mut data = json!({"schema":"ferrule.bf16-fresh-quality-result.v1","complete":true,"passed":passed,"test_config":config_manifest,"effective_config":effective_config,"observer_mode":if diagnostic {"diagnostic_routes_and_states"}else{"ordinary_no_diagnostic_observer"},"ordinary_no_observer_claim":!diagnostic,"concurrent_scheduler_claim":false,"greedy_tie_policy":"lower_token_id","cold_owner_restart":cold,"cases":*records,"logits_compared":r.logits,"failures":r.failures,"metrics":r.metrics,"atol":BF16_ATOL,"rtol":BF16_RTOL,"oracle_manifest_sha256":pin,"device_budget_bytes":limits.device_bytes,"admitted_device_bytes":admitted,"peak_reserved_bytes":peak,"state_bytes_after_shutdown":0,"quarantine":false,"task_quality_scope":"three raw completions; no chat template, no broad benchmark or performance claim"});
    attach_read_reports(
        &mut data,
        &reads,
        predictions.as_ref().map(|p| p.borrow()).as_deref(),
    );
    if let Some(fingerprints) = &fingerprints {
        data["prediction_fingerprints"] = fingerprints.borrow().json();
    }
    if let Some((_, report_pin)) = &observed_primary {
        data["diagnostic_primary_report_sha256"] = json!(report_pin);
    }
    if reads.is_some() {
        data["diagnostic_denominators"] = json!({"ordered_route_ids_compared":if diagnostic {Some(r.metrics.iter().filter(|m|m["label"].as_str().is_some_and(|l|l.ends_with("ordered_routes"))).map(|m|m["elements"].as_u64().unwrap()).sum::<u64>())}else{None},
            "state_metric_elements_compared":if diagnostic {Some(r.metrics.iter().filter(|m|m["label"].as_str().is_some_and(|l|l.contains(".cache."))).map(|m|m["elements"].as_u64().unwrap()).sum::<u64>())}else{None},
            "fork_gdn_snapshot_pairs":records.iter().filter(|record|record["fork_snapshot_equal"]==true).count()*30,"fork_replay_calls":predictions.as_ref().unwrap().borrow().json()["streams"]["fork_replay"]["prediction_calls"]});
    }
    write_report(&output, &new_report, &data);
    assert!(
        passed,
        "fresh quality failed; durable first-step evidence retained"
    );
}
