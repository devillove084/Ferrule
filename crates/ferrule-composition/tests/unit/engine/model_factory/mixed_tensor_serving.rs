use super::*;
use crate::model_factory::mixed_serving::fixture;

fn options() -> ModelFactoryOptions {
    let mut options = crate::model_factory::tests::options();
    options.max_layers = None;
    options.output_head_chunk_rows = 4096;
    options.expert_reader_max_tensor_mebibytes = 64;
    options.driver_config.ctx_size = 128;
    options.scheduler_config.max_active_sequences = 2;
    options.scheduler_config.max_batch_tokens = 8;
    options.scheduler_config.prefill_chunk_size = 8;
    options.kv_cache_mebibytes = Some(16);
    options
}
fn serving() -> ServingModelOptions {
    ServingModelOptions {
        model_profile: Some(ServingModelProfile::Qwen35MixedBf16),
        parallel: Some(PipelineBuildOptions {
            parallelism: ParallelismPlan {
                tensor_parallel: 2,
                ..Default::default()
            },
            devices: Some(vec![3, 1]),
            ..Default::default()
        }),
        ..Default::default()
    }
}
fn lowered() -> SelectionRequest {
    let mut serving = serving();
    let numeric_model = Some(
        ServingModelProfile::Qwen35MixedBf16
            .lower(&mut serving)
            .unwrap(),
    );
    SelectionRequest {
        serving,
        numeric_model,
    }
}

#[test]
fn mixed_tp2_authorized_gate_covers_named_and_direct_precision() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_authorized_gate_covers_named_and_direct_precision"
        ),
        || {
            let request = lowered();
            validate_request(
                BackendSelection::Cuda,
                &options(),
                &request.serving,
                request.numeric_model.unwrap().storage,
            )
            .unwrap();
            serving()
                .validate_before_model_io(
                    BackendSelection::Cuda,
                    &options(),
                    &ServingOptionProvenance::default(),
                )
                .unwrap();
            let mut direct = request.serving.clone();
            direct.model_profile = None;
            bf16_serving::validate_numeric_request(
                BackendSelection::Cuda,
                &options(),
                &direct,
                NumericStoragePolicy::E4M3FnBf16Scales,
                NumericFp8Precision::Bf16ModelMixedF32Accumulate,
            )
            .unwrap();
            let strategy =
                bf16_serving::select_numeric(BackendSelection::Cuda, &request, &options()).unwrap();
            assert_eq!(
                strategy.execution.construction.kind(),
                SelectedImplementationKind::Pipeline
            );
        },
    );
}

#[test]
fn mixed_tp2_rejects_unvalidated_topologies_and_policies_without_cuda() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_rejects_unvalidated_topologies_and_policies_without_cuda"
        ),
        || {
            let mutations: &[fn(&mut PipelineBuildOptions)] = &[
                |p| p.parallelism.pipeline_parallel = 2,
                |p| p.parallelism.tensor_parallel = 4,
                |p| p.parallelism.expert_parallel = 2,
                |p| p.parallelism.data_parallel = 2,
                |p| p.parallelism.sequence_parallel = 2,
                |p| p.parallelism.context_parallel = 2,
                |p| p.rank_backend = PipelineRankBackend::Process,
                |p| p.devices = None,
                |p| p.devices = Some(vec![1, 1]),
                |p| p.devices = Some(vec![1]),
                |p| p.rank_restarts = 1,
            ];
            for mutate in mutations {
                let mut s = lowered().serving;
                mutate(s.parallel.as_mut().unwrap());
                assert!(
                    validate_request(
                        BackendSelection::Cuda,
                        &options(),
                        &s,
                        NumericStoragePolicy::E4M3FnBf16Scales
                    )
                    .is_err()
                );
            }
            for backend in [BackendSelection::Cpu, BackendSelection::Auto] {
                assert!(
                    validate_request(
                        backend,
                        &options(),
                        &lowered().serving,
                        NumericStoragePolicy::E4M3FnBf16Scales
                    )
                    .is_err()
                );
            }
            let mutations: &[fn(&mut ModelFactoryOptions)] = &[
                |o| o.numeric_fp8_profile = Some(NumericFp8Profile::F32),
                |o| o.numeric_fp8_profile = Some(NumericFp8Profile::W8A16Bf16),
                |o| o.driver_config.ctx_size = 129,
                |o| o.scheduler_config.max_batch_tokens = 9,
                |o| o.scheduler_config.max_active_sequences = 3,
                |o| o.scheduler_config.prefix_cache_capacity_pages = 1,
                |o| {
                    o.qwen35_moe_capacity = Some(Qwen35MoeCapacityLimits {
                        device_bytes: 23 << 30,
                        ..Default::default()
                    })
                },
                |o| {
                    o.qwen35_moe_capacity = Some(Qwen35MoeCapacityLimits {
                        max_experts: 1,
                        ..Default::default()
                    })
                },
                |o| o.qwen35_host_cache = Some(Default::default()),
            ];
            for mutate in mutations {
                let mut o = options();
                mutate(&mut o);
                assert!(
                    validate_request(
                        BackendSelection::Cuda,
                        &o,
                        &lowered().serving,
                        NumericStoragePolicy::E4M3FnBf16Scales
                    )
                    .is_err()
                );
            }
            for mode in [
                ferrule_model::transformer::host_experts::ExpertPrewarmMode::Full,
                ferrule_model::transformer::host_experts::ExpertPrewarmMode::Lazy,
            ] {
                let e = serving()
                    .validate_before_model_io(
                        BackendSelection::Cuda,
                        &options(),
                        &ServingOptionProvenance {
                            expert_prewarm: Some(mode),
                            ..Default::default()
                        },
                    )
                    .unwrap_err();
                assert!(e.to_string().contains("prewarm flags"));
            }
        },
    );
}

#[test]
fn mixed_tp2_public_catalog_uses_existing_pipeline_and_complete_claims() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_public_catalog_uses_existing_pipeline_and_complete_claims"
        ),
        || {
            let f = fixture::Fixture::new();
            let config = AutoConfig::from_pretrained(f.dir.path()).unwrap();
            let plan = ResidentModelPlanner
                .prepare_serving(&config, BackendSelection::Cuda, None, options(), serving())
                .unwrap();
            assert_eq!(
                plan.selected_implementation().kind,
                SelectedImplementationKind::Pipeline
            );
            let expected: fn(ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> =
                build_hybrid_pipeline;
            assert!(std::ptr::fn_addr_eq(
                plan.request.execution.construction.builder(),
                expected
            ));
            assert_eq!(plan.backend_profile(), PROFILE);
            assert_eq!(plan.topology_placement().physical.devices, Some(vec![3, 1]));
            assert_eq!(plan.numeric_fp8_profile(), None);
            assert!(plan.numeric_bf16_memory_plan().unwrap().is_none());
            assert!(plan.effective_options().qwen35_host_cache.is_none());
            assert!(plan.qwen35_host_cache_options().is_none());
            assert_eq!(
                plan.qwen35_moe_capacity_limits()
                    .unwrap()
                    .unwrap()
                    .device_bytes,
                22 << 30
            );
            let (_, resources) =
                ferrule_model::models::qwen35::Qwen35Adapter::bind_hf_metadata(f.dir.path())
                    .unwrap();
            let admission = Admission::new(&plan.request, &resources).unwrap();
            assert_eq!(
                admission.plan.contract().precision(),
                HybridRankPrecision::MixedBf16F32
            );
            assert_eq!(
                admission.plan.contract().experts(),
                ParallelExpertMode::LocalTensorShards
            );
            assert_eq!(admission.plan.external_owner_count(), 0);
            assert_eq!(
                admission.config.precision,
                ExecutionPrecisionPolicy::bf16_compatibility()
            );
            assert_eq!(
                admission.kv.configured_pages,
                admission.config.max_pages * 2
            );
            assert_eq!(
                admission.capacities,
                vec![(3, DEVICE_LIMIT), (1, DEVICE_LIMIT)]
            );
            assert!(
                admission
                    .cache_requirements
                    .iter()
                    .all(|row| row.required_device_bytes > 0
                        && row.required_device_bytes <= DEVICE_LIMIT)
            );
            assert_eq!(
                plan.qwen35_moe_capacity_limits()
                    .unwrap()
                    .unwrap()
                    .workspace_bytes,
                4 << 30
            );
            assert!(
                plan.resolution_report()
                    .contains("decoded BF16 cache=3 GiB")
            );
            for row in &admission.cache_requirements {
                assert!(row.cache_enabled);
                assert_eq!(row.cache_capacity_grants, 3 << 30);
                assert_eq!(row.pinned_grants, 1340 << 20);
                assert_eq!(row.routed_working_set_grants, 768 << 20);
                assert!(row.required_workspace_bytes > 3 << 30);
                assert!(row.required_workspace_bytes <= row.workspace_ceiling_bytes as u64);
                eprintln!(
                    "PUBLIC_CACHE_LEDGER {}",
                    serde_json::to_string(row).unwrap()
                );
            }
            for program in &admission.programs {
                assert_eq!(
                    program
                        .decoded_weight_cache_budget()
                        .unwrap()
                        .capacity_grants,
                    3 << 30
                );
            }
            assert!(matches!(
                plan.request.execution.conversion,
                PipelineConversion::Frozen(_)
            ));
            // The authorized gate permits the existing family binder; execution still
            // remains behind the existing owner admission and source claims.
            assert!(qwen35::binding::hybrid(&plan.request).is_ok());
            let mut tight = plan.request.clone();
            if let ResolvedFamilyOptions::Qwen35(qwen35::Qwen35ResolvedAdapter::Moe {
                options,
                ..
            }) = &mut tight.family_options
            {
                options.device_capacity.as_mut().unwrap().device_bytes = 1;
            }
            assert!(Admission::new(&tight, &resources).is_err());
        },
    );
}

#[test]
fn mixed_tp2_cache_defaults_and_explicit_workspace_rejection_are_profile_local() {
    let strict = capacity(None);
    assert_eq!(strict.workspace_bytes, 4 << 30);
    assert_eq!(strict.device_bytes, 22 << 30);
    assert_eq!(Qwen35MoeCapacityLimits::default().workspace_bytes, 1 << 30);
    for bytes in [0, 1 << 30, 3 << 30] {
        let limits = Qwen35MoeCapacityLimits {
            workspace_bytes: bytes,
            ..strict
        };
        let error = validate_capacity(limits).unwrap_err().to_string();
        assert!(error.contains("cache-on") && error.contains("4096"));
    }
    validate_capacity(strict).unwrap();
}

#[test]
fn mixed_tp2_prepared_once_accessors_and_consuming_build() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_prepared_once_accessors_and_consuming_build"
        ),
        || {
            ADMISSION_CONSTRUCTIONS.with(|n| n.set(0));
            let f = fixture::Fixture::new();
            let config = AutoConfig::from_pretrained(f.dir.path()).unwrap();
            let prepare_clock = std::time::Instant::now();
            let plan = ResidentModelPlanner
                .prepare_serving(&config, BackendSelection::Cuda, None, options(), serving())
                .unwrap();
            let prepare_elapsed = prepare_clock.elapsed();
            let preview_clock = std::time::Instant::now();
            let first = plan.mixed_tp2_cache_requirements().unwrap().unwrap();
            assert_eq!(first, plan.mixed_tp2_cache_requirements().unwrap().unwrap());
            assert_eq!(
                first,
                plan.serving_resource_requirements().unwrap().unwrap()
            );
            let preview_elapsed = preview_clock.elapsed();
            assert_eq!(ADMISSION_CONSTRUCTIONS.with(|n| n.get()), 1);
            let owner_clock = std::time::Instant::now();
            std::thread::spawn(move || {
                plan.build_with(|request| {
                    let original = prepared(&request.prepared_source_plan).unwrap();
                    let (resources, admission) = consume_prepared(&request)?;
                    assert!(Arc::ptr_eq(&original.admission, &admission));
                    for (a, b) in original
                        .resources
                        .state_dict()
                        .parameters()
                        .iter()
                        .zip(resources.state_dict().parameters())
                    {
                        assert!(a.shares_storage_with(b));
                    }
                    assert_eq!(admission.cache_requirements, first);
                    assert_eq!(admission.programs.len(), 2);
                    assert!(
                        admission
                            .programs
                            .iter()
                            .all(|p| p.decoded_weight_cache_budget().is_some())
                    );
                    assert_eq!(ADMISSION_CONSTRUCTIONS.with(|n| n.get()), 0);
                    Ok(())
                })
            })
            .join()
            .unwrap()
            .unwrap();
            assert_eq!(ADMISSION_CONSTRUCTIONS.with(|n| n.get()), 1);
            eprintln!(
                "MIXED_PREPARED_CPU_ONLY prepare_ms={} accessor2_ms={} owner_validate_mock_ms={}",
                prepare_elapsed.as_millis(),
                preview_elapsed.as_millis(),
                owner_clock.elapsed().as_millis()
            );
        },
    );
}

#[test]
fn mixed_tp2_prepared_ledger_matches_fresh_admission() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_prepared_ledger_matches_fresh_admission"
        ),
        || {
            let f = fixture::Fixture::new();
            let config = AutoConfig::from_pretrained(f.dir.path()).unwrap();
            let plan = ResidentModelPlanner
                .prepare_serving(&config, BackendSelection::Cuda, None, options(), serving())
                .unwrap();
            let p = prepared(&plan.prepared_source_plan).unwrap();
            let fresh = Admission::new(&plan.request, &p.resources).unwrap();
            assert_eq!(fresh.cache_requirements, p.admission.cache_requirements);
            assert_eq!(fresh.host_required, p.admission.host_required);
            assert_eq!(fresh.capacities, p.admission.capacities);
            assert_eq!(
                fresh
                    .config
                    .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))
                    .unwrap(),
                p.admission
                    .config
                    .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cuda))
                    .unwrap()
            );
            assert_eq!(fresh.kv, p.admission.kv);
        },
    );
}

#[test]
fn mixed_tp2_prepared_rejects_final_request_changes() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_prepared_rejects_final_request_changes"
        ),
        || {
            let f = fixture::Fixture::new();
            let config = AutoConfig::from_pretrained(f.dir.path()).unwrap();
            let mut plan = ResidentModelPlanner
                .prepare_serving(&config, BackendSelection::Cuda, None, options(), serving())
                .unwrap();
            let source = plan.prepared_source_plan.clone();
            let p = prepared(&source).unwrap();
            macro_rules! reject {
                ($change:expr) => {{
                    let mut request = plan.request.clone();
                    ($change)(&mut request);
                    assert!(p.validate(&request).is_err());
                    let frozen = std::mem::replace(&mut plan.request, request.clone());
                    assert!(plan.serving_resource_requirements().is_err());
                    assert!(plan.mixed_tp2_cache_requirements().is_err());
                    plan.request = frozen;
                    let build = ModelBuildRequest {
                        resolved: request,
                        prepared_source_plan: plan.prepared_source_plan.clone(),
                    };
                    assert!(consume_prepared(&build).is_err());
                }};
            }
            reject!(|r: &mut ResolvedModelRequest| r.model_path = f.dir.path().join("other"));
            reject!(|r: &mut ResolvedModelRequest| r.max_layers -= 1);
            reject!(|r: &mut ResolvedModelRequest| r.pipeline = None);
            reject!(
                |r: &mut ResolvedModelRequest| r.execution.serving_resource_requirements =
                    no_serving_resource_requirements
            );
            reject!(|r: &mut ResolvedModelRequest| r.max_tensor_bytes += 1);
            reject!(|r: &mut ResolvedModelRequest| r.kv_cache_bytes = Some(32 << 20));
            reject!(|r: &mut ResolvedModelRequest| r.driver_config.ctx_size += 1);
            reject!(|r: &mut ResolvedModelRequest| r.driver_config.stop_at_eos =
                !r.driver_config.stop_at_eos);
            reject!(
                |r: &mut ResolvedModelRequest| r.driver_config.proposal_confidence_threshold = 0.5
            );
            reject!(|r: &mut ResolvedModelRequest| r.scheduler_config.max_batch_tokens += 1);
            reject!(|r: &mut ResolvedModelRequest| r.scheduler_config.max_active_sequences += 1);
            reject!(|r: &mut ResolvedModelRequest| r
                .pipeline
                .as_mut()
                .unwrap()
                .devices
                .as_mut()
                .unwrap()
                .swap(0, 1));
            reject!(
                |r: &mut ResolvedModelRequest| r.pipeline.as_mut().unwrap().rank_timeout +=
                    std::time::Duration::from_secs(1)
            );
            reject!(|r: &mut ResolvedModelRequest| {
                let ResolvedFamilyOptions::Qwen35(qwen35::Qwen35ResolvedAdapter::Moe {
                    options,
                    ..
                }) = &mut r.family_options
                else {
                    panic!()
                };
                options.device_capacity.as_mut().unwrap().workspace_bytes += 1;
            });
            assert!(p.validate(&plan.request).is_ok());
        },
    );
}

#[test]
fn mixed_tp2_prepared_rejects_source_replacement_and_metadata_mutation() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_prepared_rejects_source_replacement_and_metadata_mutation"
        ),
        || {
            for metadata in [false, true] {
                let f = fixture::Fixture::new();
                let config = AutoConfig::from_pretrained(f.dir.path()).unwrap();
                let plan = ResidentModelPlanner
                    .prepare_serving(&config, BackendSelection::Cuda, None, options(), serving())
                    .unwrap();
                let path = f.dir.path().join(if metadata {
                    "config.json"
                } else {
                    "model.safetensors"
                });
                if metadata {
                    std::fs::write(&path, "{}").unwrap();
                } else {
                    // A same-path replacement must not be accepted as the old catalog.
                    let original = std::fs::metadata(&path).unwrap();
                    std::fs::rename(&path, path.with_extension("original")).unwrap();
                    let replacement = std::fs::File::create(&path).unwrap();
                    replacement.set_len(original.len()).unwrap();
                    replacement
                        .set_times(
                            std::fs::FileTimes::new().set_modified(original.modified().unwrap()),
                        )
                        .unwrap();
                    let replaced = replacement.metadata().unwrap();
                    assert_eq!(original.len(), replaced.len());
                    assert_eq!(original.modified().unwrap(), replaced.modified().unwrap());
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::MetadataExt;
                        assert_ne!(original.ino(), replaced.ino());
                    }
                }
                assert!(plan.serving_resource_requirements().is_err());
                assert!(plan.mixed_tp2_cache_requirements().is_err());
                assert!(
                    plan.build_with(|r| consume_prepared(&r).map(|_| ()))
                        .is_err()
                );
            }
        },
    );
}

#[test]
fn mixed_tp2_prepared_rejects_weight_or_scale_snapshot_mutation() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::mixed_tp2_prepared_rejects_weight_or_scale_snapshot_mutation"
        ),
        || {
            use std::io::{Read, Write};
            fn sparse(path: &std::path::Path, header: &serde_json::Map<String, serde_json::Value>) {
                let mut header = header.clone();
                let mut length = 0;
                for value in header.values_mut() {
                    let bytes = value["data_offsets"][1].as_u64().unwrap()
                        - value["data_offsets"][0].as_u64().unwrap();
                    value["data_offsets"] = serde_json::json!([length, length + bytes]);
                    length += bytes;
                }
                let mut bytes = serde_json::to_vec(&header).unwrap();
                while !bytes.len().is_multiple_of(8) {
                    bytes.push(b' ');
                }
                let mut file = std::fs::File::create(path).unwrap();
                file.write_all(&(bytes.len() as u64).to_le_bytes()).unwrap();
                file.write_all(&bytes).unwrap();
                file.set_len(8 + bytes.len() as u64 + length).unwrap();
            }
            for scale in [false, true] {
                let f = fixture::Fixture::new();
                let mut file = std::fs::File::open(f.dir.path().join("model.safetensors")).unwrap();
                let mut length = [0; 8];
                file.read_exact(&mut length).unwrap();
                let mut bytes = vec![0; u64::from_le_bytes(length) as usize];
                file.read_exact(&mut bytes).unwrap();
                drop(file);
                let header: serde_json::Map<String, serde_json::Value> =
                    serde_json::from_slice(&bytes).unwrap();
                let mut weights = serde_json::Map::new();
                let mut scales = serde_json::Map::new();
                let mut weight_map = serde_json::Map::new();
                for (name, value) in header {
                    let is_scale = name.contains("scale_inv");
                    weight_map.insert(
                        name.clone(),
                        serde_json::Value::String(
                            if is_scale {
                                "scales.safetensors"
                            } else {
                                "weights.safetensors"
                            }
                            .into(),
                        ),
                    );
                    if is_scale {
                        scales.insert(name, value);
                    } else {
                        weights.insert(name, value);
                    }
                }
                assert!(!scales.is_empty());
                sparse(&f.dir.path().join("weights.safetensors"), &weights);
                sparse(&f.dir.path().join("scales.safetensors"), &scales);
                std::fs::remove_file(f.dir.path().join("model.safetensors")).unwrap();
                std::fs::write(
                    f.dir.path().join("model.safetensors.index.json"),
                    serde_json::to_vec(
                        &serde_json::json!({"metadata": {}, "weight_map": weight_map}),
                    )
                    .unwrap(),
                )
                .unwrap();
                let config = AutoConfig::from_pretrained(f.dir.path()).unwrap();
                let plan = ResidentModelPlanner
                    .prepare_serving(&config, BackendSelection::Cuda, None, options(), serving())
                    .unwrap();
                let p = prepared(&plan.prepared_source_plan).unwrap();
                let pair = p
                    .resources
                    .state_dict()
                    .parameters()
                    .iter()
                    .find(|b| b.numeric_fp8_encoding().is_some())
                    .unwrap();
                assert_ne!(
                    pair.weight().source_identity(),
                    pair.scale().unwrap().source_identity()
                );
                let path = f.dir.path().join(if scale {
                    "scales.safetensors"
                } else {
                    "weights.safetensors"
                });
                let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
                file.set_len(file.metadata().unwrap().len() + 1).unwrap();
                assert!(plan.serving_resource_requirements().is_err());
                assert!(plan.mixed_tp2_cache_requirements().is_err());
                assert!(
                    plan.build_with(|r| consume_prepared(&r).map(|_| ()))
                        .is_err()
                );
            }
        },
    );
}
