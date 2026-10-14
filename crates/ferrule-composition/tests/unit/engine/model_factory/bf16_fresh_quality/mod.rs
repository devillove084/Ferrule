//! Fresh quality gate reusing the maintained public-factory full40 runner.
use super::*;

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
        assert_cold_primary(&prior, config, precision, "test-pin");
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

fn run(cold: bool, test_config: Full40TestConfig) {
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

    crate::model_factory::bf16_serving::NUMERIC_BF16_TEST_RUNNER.with(|slot| {
        assert!(slot.borrow().is_none());
        *slot.borrow_mut()=Some(Box::new(move |runner| {
            assert_eq!(runner.forward_executor().module().expert_read_pipeline_options(), test_config.expert_read_pipeline());
            assert_eq!(runner.forward_executor().module().numeric_fp8_precision(), Some(precision));
            assert!(runner.forward_executor().module().numeric_bf16_profile().is_some());
            let routes=Rc::new(RefCell::new(BTreeMap::new()));let capture=Rc::clone(&routes);
            let active=Rc::new(RefCell::new(false));let observe=Rc::clone(&active);
            let mut trace_index=0usize;
            let mut trace_log=narrow.as_ref().map(|path|std::fs::File::create(path.join("events.jsonl")).unwrap());
            runner.forward_executor().module().set_diagnostic_trace(move|event| {
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
            });
            let mut transaction=0;
            for (case,oracle,prompt) in cases {
                let mut states=vec![runner.create_sequence_state().unwrap()];let mut pages=vec![Vec::new()];
                let mut generated=Vec::new();let mut top1_failures=Vec::new();let mut token_checks=Vec::new();
                for step in 0..9 {
                    let call=&oracle.report["calls"][step];let stage=format!("step.{step:02}");
                    let tokens:Vec<u32>=serde_json::from_value(call["input_token_ids"].clone()).unwrap();
                    assert_eq!(states[0].core().position(),call["position_start"].as_u64().unwrap()as usize);
                    *active.borrow_mut()=step==narrow_step;
                    transaction+=1;
                    let logits=crate::model_factory::qwen35::full_logits_step(runner,&mut states,&mut pages,&tokens,if step==0 {ForwardPhase::Prefill}else{ForwardPhase::Decode},transaction);
                    assert_eq!(logits.len(),tokens.len());assert!(logits.iter().all(|row|row.len()==248320));
                                        assert_eq!(states[0].core().position(),call["sequence_length_after"].as_u64().unwrap()as usize);
                                        let next=top1(logits.last().unwrap());let expected=call["next_token_id"].as_u64().unwrap()as u32;generated.push(next);
                                        token_checks.push(json!({"step":step,"actual":next,"expected":expected,"equal":next==expected}));
                    if next!=expected {top1_failures.push(json!({"step":step,"actual":next,"expected":expected}));}
                    let actual:Vec<_>=logits.into_iter().flatten().collect();typed(&actual);
                    let mut r=result.borrow_mut();r.logits+=actual.len();
                    r.bf16_values(&actual,&oracle.values(&format!("{stage}.logits")),&format!("{case}/{stage}/all_logits"));
                    let captured=std::mem::take(&mut *routes.borrow_mut());assert_eq!(captured.len(),40);
                    for (layer,(ids,weights,raw)) in captured {
                        let key=format!("{stage}.layers.{layer:02}");
                        r.route_ids(&ids,&oracle.ids(&format!("{key}.selected_expert_ids")),&format!("{case}/{key}/ordered_routes"));
                        r.bf16_values(&weights,&oracle.values(&format!("{key}.routing_weights")),&format!("{case}/{key}/weights"));
                        r.bf16_values(&raw,&oracle.values(&format!("{key}.raw_router_logits")),&format!("{case}/{key}/raw_router_logits"));
                    }
                    check_state(runner,&states[0],&pages[0],&oracle,&case,&stage,&mut r);
                    let failed=!r.failures.is_empty()||!top1_failures.is_empty();
                    std::fs::write(&progress_path,serde_json::to_vec_pretty(&json!({"complete":false,"passed":false,"test_config":progress_config,"effective_config":progress_effective,"observer_mode":"diagnostic_routes_and_states","ordinary_no_observer_claim":false,"concurrent_scheduler_claim":false,"greedy_tie_policy":"lower_token_id","current_case":case,"current_step":step,"actual_next_token_id":next,"expected_next_token_id":expected,"common_history_generated_ids":generated,"next_token_checks":token_checks,"metrics":r.metrics,"failures":r.failures,"top1_failures":top1_failures})).unwrap()).unwrap();drop(r);
                    if failed {break;}
                }
                let expected:Vec<u32>=serde_json::from_value(oracle.report["generated_ids"].clone()).unwrap();
                let common_pass=result.borrow().failures.is_empty()&&top1_failures.is_empty();
                records.borrow_mut().push(json!({"case":case,"prompt_ids":prompt,"common_history_generated_ids":generated,"next_token_checks":token_checks,"oracle_generated_ids":expected,"free_greedy_ids":[],"top1_failures":top1_failures,"free_greedy_status":"pending_all_teacherforced_cases","free_greedy_equal":null,"fork_snapshot_equal":null,"reset_reused_state":false}));
                runner.try_release_sequence_state(states.remove(0)).unwrap();runner.release_kv_pages(&pages[0]).unwrap();
                if !common_pass {break;}
            }
            let all_common_pass=result.borrow().failures.is_empty()
                &&records.borrow().len()==if cold {1}else{3}
                &&records.borrow().iter().all(|c|c["top1_failures"].as_array().unwrap().is_empty());
            if all_common_pass && !narrow_mode {
                *active.borrow_mut()=false;
                let case_count=records.borrow().len();
                for index in 0..case_count {
                    let record=records.borrow()[index].clone();
                    let prompt:Vec<u32>=serde_json::from_value(record["prompt_ids"].clone()).unwrap();
                    let expected:Vec<u32>=serde_json::from_value(record["oracle_generated_ids"].clone()).unwrap();
                    let mut states=vec![runner.create_sequence_state().unwrap()];let mut pages=vec![Vec::new()];
                    let mut free=Vec::new();let mut free_checks=Vec::new();let mut fed=prompt.clone();
                    for step in 0..9 {
                        transaction+=1;
                        let y=crate::model_factory::qwen35::full_logits_step(runner,&mut states,&mut pages,&fed,if step==0 {ForwardPhase::Prefill}else{ForwardPhase::Decode},transaction);
                        let token=top1(y.last().unwrap());free.push(token);
                                                free_checks.push(json!({"step":step,"actual":token,"expected":expected[step],"equal":token==expected[step]}));
                                                fed=vec![token];routes.borrow_mut().clear();
                    }
                    records.borrow_mut()[index]["free_greedy_ids"]=json!(free);
                                        records.borrow_mut()[index]["free_greedy_next_token_checks"]=json!(free_checks);
                    records.borrow_mut()[index]["free_greedy_equal"]=json!(free==expected);
                    records.borrow_mut()[index]["free_greedy_status"]=json!("completed_independent_feedback");
                    if index==0 &&free==expected {
                        runner.reset_sequence_state(&mut states[0]).unwrap();runner.release_kv_pages(&pages[0]).unwrap();pages[0].clear();assert_eq!(states[0].core().position(),0);
                        let mut repeated=Vec::new();let mut reset_checks=Vec::new();let mut fed=prompt;
                        for step in 0..9 {
                            transaction+=1;
                            let y=crate::model_factory::qwen35::full_logits_step(runner,&mut states,&mut pages,&fed,if step==0 {ForwardPhase::Prefill}else{ForwardPhase::Decode},transaction);
                            let token=top1(y.last().unwrap());repeated.push(token);
                                                        reset_checks.push(json!({"step":step,"actual":token,"expected":expected[step],"equal":token==expected[step]}));
                                                        fed=vec![token];routes.borrow_mut().clear();
                            if step==0 {
                                let fork=runner.fork_sequence_state_from(&states[0],states[0].core().position()).unwrap();
                                assert_eq!(fork.core().position(),states[0].core().position());
                                for(a,b)in states[0].kv_state().bf16_layers().unwrap().iter().zip(fork.kv_state().bf16_layers().unwrap()) {
                                    if let(Some(a),Some(b))=(a,b){assert_eq!(a.diagnostic_snapshot().unwrap(),b.diagnostic_snapshot().unwrap());}
                                }
                                runner.try_release_sequence_state(fork).unwrap();records.borrow_mut()[index]["fork_snapshot_equal"]=json!(true);
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
        }));
    });
    let mut engine = plan.build().unwrap();
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
    let passed = r.failures.is_empty()
        && records.len() == if cold { 1 } else { 3 }
        && records.iter().all(|c| {
            c["free_greedy_equal"] == true
                && c["top1_failures"].as_array().unwrap().is_empty()
                && (!c["reset_reused_state"].as_bool().unwrap_or(false) || c["reset_equal"] == true)
        });
    let data = json!({"schema":"ferrule.bf16-fresh-quality-result.v1","complete":true,"passed":passed,"test_config":config_manifest,"effective_config":effective_config,"observer_mode":"diagnostic_routes_and_states","ordinary_no_observer_claim":false,"concurrent_scheduler_claim":false,"greedy_tie_policy":"lower_token_id","cold_owner_restart":cold,"cases":*records,"logits_compared":r.logits,"failures":r.failures,"metrics":r.metrics,"atol":BF16_ATOL,"rtol":BF16_RTOL,"oracle_manifest_sha256":pin,"device_budget_bytes":limits.device_bytes,"admitted_device_bytes":admitted,"peak_reserved_bytes":peak,"state_bytes_after_shutdown":0,"quarantine":false,"task_quality_scope":"three raw completions; no chat template, no broad benchmark or performance claim"});
    std::fs::write(output, serde_json::to_vec_pretty(&data).unwrap()).unwrap();
    assert!(
        passed,
        "fresh quality failed; durable first-step evidence retained"
    );
}
