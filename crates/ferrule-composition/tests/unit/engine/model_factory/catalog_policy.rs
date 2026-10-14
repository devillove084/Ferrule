use super::*;
use crate::model_factory::tests::{descriptor, options};
use ferrule_model::transformer::{NumericFp8Precision, host_experts::ExpertPrewarmMode};
use ferrule_runtime::engine::KvExecutionPolicy;

use crate::model_factory::mixed_serving::fixture;

fn bounded_options() -> ModelFactoryOptions {
    let mut options = options();
    options.scheduler_config.max_batch_tokens = 8;
    options.scheduler_config.max_active_sequences = 1;
    options.scheduler_config.max_decode_batch = 1;
    options
}

fn synthetic_entry() -> ModelImplementation {
    fn resolve(d: &ModelDescriptor, _: BackendSelection) -> Result<ResolvedModelBackend> {
        Ok(ResolvedModelBackend {
            family: d.spec.family.clone(),
            model_name: "synthetic-policy",
            backend: ModelExecutionBackend::Cpu,
            backend_profile: "synthetic",
            default_chat_template: ChatTemplate::Plain,
        })
    }
    fn build(request: ModelBuildRequest) -> Result<BoxedSessionInferenceEngine> {
        assert_eq!(
            request.resolved.model_path.to_string_lossy(),
            "missing-policy-checkpoint"
        );
        assert_eq!(
            request.resolved.execution.adjustment_reason,
            "synthetic normalization"
        );
        let code = match request.resolved.execution.numeric_bf16_precision {
            Some(NumericFp8Precision::Bf16ModelMixedF32Accumulate) => 904,
            Some(_) => 905,
            None if request.resolved.execution.kv_execution_policy
                != KvExecutionPolicy::Resident =>
            {
                903
            }
            None if request.resolved.execution.support.storage == StorageEncoding::NativeBf16 => {
                902
            }
            None => 901,
        };
        Err(Error::RequestCapacity { limit: code })
    }
    fn select(
        entry: &ModelImplementation,
        _: BackendSelection,
        request: &SelectionRequest,
        _: &ModelFactoryOptions,
    ) -> Result<ResolvedStrategy> {
        let bf16 = request.serving.model_precision == Some(ModelPrecision::Bf16);
        let mut execution = ResolvedExecution::new(
            ModelExecutionBackend::Cpu,
            if bf16 && request.numeric_model.is_none() {
                ResolvedConstruction::Pipeline(entry.build)
            } else {
                ResolvedConstruction::Resident(entry.build)
            },
            if bf16 {
                StorageEncoding::NativeBf16
            } else {
                StorageEncoding::ProfileDefined
            },
            if bf16 {
                ArithmeticPrecision::Bf16Compatibility
            } else {
                ArithmeticPrecision::F32
            },
            "synthetic constraints",
        );
        execution.numeric_bf16_precision = request.numeric_model.map(|n| n.precision);
        execution.kv_execution_policy = request.serving.kv_execution_policy;
        execution.adjustment_reason = "synthetic normalization";
        execution.conversion = PipelineConversion::Frozen("synthetic conversion policy");
        Ok(ResolvedStrategy {
            execution,
            backend: ModelExecutionBackend::Cpu,
            catalog_backend: BackendSelection::Cpu,
            resolve: entry.resolve,
            policy: &RESIDENT_OPTIONS,
            parallel: None,
        })
    }
    ModelImplementation {
        family: ModelFamily::Unknown("synthetic-policy".into()),
        resolve,
        build,
        select,
    }
}

#[test]
fn all_normalized_profile_paths_reach_the_same_catalog_and_frozen_builder() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::all_normalized_profile_paths_reach_the_same_catalog_and_frozen_builder"
        ),
        || {
            let entry = synthetic_entry();
            let mut d = descriptor(entry.family.clone(), "synthetic-policy", 2);
            d.path = "missing-policy-checkpoint".into();
            let config = AutoConfig::from_descriptor(d);
            let idle =
                KvExecutionPolicy::IdleReclaim(ferrule_model::decoder::IdleKvReclaimConfig {
                    staging_page_limit: 1,
                    device_bytes: 1 << 20,
                    host_bytes: 1 << 20,
                    pinned_bytes: 1 << 20,
                });
            for (request, expected) in [
                (SelectionRequest::default(), 901),
                (
                    SelectionRequest {
                        serving: ServingModelOptions {
                            model_precision: Some(ModelPrecision::Bf16),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    902,
                ),
                (
                    SelectionRequest {
                        serving: ServingModelOptions {
                            kv_execution_policy: KvExecutionPolicy::HostKvDecode(Default::default()),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    903,
                ),
                (
                    SelectionRequest {
                        serving: ServingModelOptions {
                            kv_execution_policy: idle,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    903,
                ),
                (
                    SelectionRequest {
                        serving: ServingModelOptions {
                            model_profile: Some(ServingModelProfile::Qwen35MixedBf16),
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    904,
                ),
                (
                    SelectionRequest {
                        serving: ServingModelOptions {
                            model_precision: Some(ModelPrecision::Bf16),
                            ..Default::default()
                        },
                        numeric_model: Some(NumericModelRequest {
                            storage: NumericStoragePolicy::E4M3FnBf16Scales,
                            precision: NumericFp8Precision::Bf16ModelF32Accumulate,
                        }),
                    },
                    905,
                ),
            ] {
                let input = bounded_options();
                let normalized = request.normalize(BackendSelection::Cuda, &input).unwrap();
                let plan = ResidentModelPlanner
                    .prepare_catalog_serving(
                        &entry,
                        &config,
                        BackendSelection::Cuda,
                        None,
                        input,
                        normalized,
                    )
                    .unwrap();
                assert!(
                    plan.validate_pipeline_conversion()
                        .unwrap_err()
                        .to_string()
                        .contains("synthetic conversion policy")
                );
                let report = plan.resolution_report();
                assert!(report.contains("synthetic-policy"));
                assert!(std::ptr::fn_addr_eq(
                    plan.request.execution.construction.builder(),
                    entry.build
                ));
                assert!(
                    matches!(plan.build().err().unwrap(), Error::RequestCapacity { limit } if limit == expected)
                );
            }
        },
    );
}

#[test]
fn pre_io_validation_and_normalization_reject_the_same_explicit_options() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::pre_io_validation_and_normalization_reject_the_same_explicit_options"
        ),
        || {
            let options = bounded_options();
            for change in 0..8 {
                let mut serving = ServingModelOptions {
                    model_profile: Some(ServingModelProfile::Qwen35MixedBf16),
                    ..Default::default()
                };
                match change {
                    0 => serving.model_precision = Some(ModelPrecision::Bf16),
                    1 => serving.parallel = Some(PipelineBuildOptions::default()),
                    2 => serving.expert_host_cache_entries = Some(0),
                    3 => serving.expert_pinned_cache_mebibytes = Some(0),
                    4 => serving.moe_hotset_experts = Some(0),
                    5 => serving.engine = ServingEngineSelection::Pipeline,
                    6 => {
                        serving.kv_execution_policy =
                            KvExecutionPolicy::HostKvDecode(Default::default())
                    }
                    _ => serving.expert_host_cache_mebibytes = Some(0),
                }
                let direct = serving
                    .validate_before_model_io(
                        BackendSelection::Cuda,
                        &options,
                        &ServingOptionProvenance::default(),
                    )
                    .unwrap_err();
                let normalized = SelectionRequest {
                    serving,
                    numeric_model: None,
                }
                .normalize(BackendSelection::Cuda, &options)
                .unwrap_err();
                assert_eq!(direct.to_string(), normalized.to_string());
            }
            // These are metadata-only calls with no descriptor/path or catalog entry.
            // Exact default values must not be erased as if the option were omitted.
            let serving = ServingModelOptions {
                model_profile: Some(ServingModelProfile::Qwen35MixedBf16),
                ..Default::default()
            };
            for mode in [None, Some(ExpertPrewarmMode::Lazy)] {
                serving
                    .validate_before_model_io(
                        BackendSelection::Cuda,
                        &options,
                        &ServingOptionProvenance {
                            expert_prewarm: mode,
                            expert_prewarm_workers: None,
                        },
                    )
                    .unwrap();
            }
            for provenance in [
                ServingOptionProvenance {
                    expert_prewarm: Some(ExpertPrewarmMode::Full),
                    expert_prewarm_workers: None,
                },
                ServingOptionProvenance {
                    expert_prewarm: Some(ExpertPrewarmMode::Lazy),
                    expert_prewarm_workers: Some(0),
                },
                ServingOptionProvenance {
                    expert_prewarm: None,
                    expert_prewarm_workers: Some(8),
                },
            ] {
                assert!(
                    serving
                        .validate_before_model_io(BackendSelection::Cuda, &options, &provenance)
                        .is_err()
                );
            }
        },
    );
}

#[test]
fn frozen_conversion_is_not_inferred_from_storage_or_arithmetic() {
    use execution_policy::*;
    let parallel = PipelineBuildOptions::default();
    let transport = parallel.validated_transport().unwrap();
    assert_eq!(transport.timeout_ms(), None);
    let topology = validate_topology(Default::default(), OwnerLayout::SingleOwner).unwrap();
    let policy = PlacementPolicy {
        selection: DeviceSelection::AutomaticOrExplicit,
        sharing: DeviceSharing::Distinct,
    };
    assert_eq!(
        validate_placement(topology, PlacementTarget::Device, None, policy)
            .unwrap()
            .policy(),
        policy
    );
    let entry = synthetic_entry();
    let mut d = descriptor(entry.family.clone(), "synthetic-policy", 2);
    d.path = "missing-policy-checkpoint".into();
    let config = AutoConfig::from_descriptor(d);
    let mut plan = ResidentModelPlanner
        .prepare_catalog_serving(
            &entry,
            &config,
            BackendSelection::Cpu,
            None,
            bounded_options(),
            SelectionRequest::default(),
        )
        .unwrap();
    assert_eq!(plan.support().arithmetic, ArithmeticPrecision::F32);
    assert!(plan.validate_pipeline_conversion().is_err());
    plan.request.execution.conversion = PipelineConversion::Allowed;
    plan.request.execution.support.storage = StorageEncoding::NativeBf16;
    plan.validate_pipeline_conversion().unwrap();
}

// Per-thread counters avoid interference between independently running tests.
// Panics make an accidentally reached hook fail immediately, not just by count.
thread_local! {
    static UNEXPECTED_HOOKS: std::cell::Cell<[usize; 6]> = const { std::cell::Cell::new([0; 6]) };
}
fn unexpected_hook(index: usize) -> ! {
    UNEXPECTED_HOOKS.with(|calls| {
        let mut counts = calls.get();
        counts[index] += 1;
        calls.set(counts);
    });
    panic!("invalid selection reached serving/resolver/adapt/attach/finish/build hook {index}")
}
const TRAP_OPTIONS: ModelOptionPolicy = ModelOptionPolicy {
    serving: |_, _, _, _| unexpected_hook(0),
    parallel: |_, _, _| unexpected_hook(3),
    adapt: |_, _, _| unexpected_hook(2),
    finish: |_| unexpected_hook(4),
};
fn trap_strategy(
    policy: Option<execution_policy::TopologyPolicy>,
    parallel: Option<PipelineBuildOptions>,
    kind: SelectedImplementationKind,
) -> ResolvedStrategy {
    let mut execution = ResolvedExecution::new(
        ModelExecutionBackend::Cuda,
        match kind {
            SelectedImplementationKind::Resident => {
                ResolvedConstruction::Resident(|_| unexpected_hook(5))
            }
            SelectedImplementationKind::Pipeline => {
                ResolvedConstruction::Pipeline(|_| unexpected_hook(5))
            }
        },
        StorageEncoding::ProfileDefined,
        ArithmeticPrecision::F32,
        "synthetic frozen capability",
    );
    execution.topology_policy = policy;
    ResolvedStrategy {
        execution,
        backend: ModelExecutionBackend::Cuda,
        catalog_backend: BackendSelection::Cuda,
        resolve: |_, _| unexpected_hook(1),
        policy: &TRAP_OPTIONS,
        parallel,
    }
}
fn thread_unsharded_policy() -> execution_policy::TopologyPolicy {
    use execution_policy::*;
    TopologyPolicy {
        layout: OwnerLayout::StageRoots,
        execution: ExecutionCapabilities {
            ranks: RankCapability::ThreadOnly,
            tensor: TensorCapability::Unsharded,
        },
        placement: PlacementPolicy {
            selection: DeviceSelection::AutomaticOrExplicit,
            sharing: DeviceSharing::Distinct,
        },
    }
}
fn assert_no_unexpected_hooks() {
    UNEXPECTED_HOOKS.with(|calls| assert_eq!(calls.get(), [0; 6]));
}
fn absent_config(family: ModelFamily, path: std::path::PathBuf) -> AutoConfig {
    let mut d = descriptor(family, "missing-policy-checkpoint", 40);
    d.path = path;
    AutoConfig::from_descriptor(d)
}

#[test]
fn compatibility_parallel_argument_cannot_replace_any_selected_field() {
    UNEXPECTED_HOOKS.with(|calls| calls.set([0; 6]));
    let directory = tempfile::tempdir().unwrap();
    let config = absent_config(
        ModelFamily::Unknown("synthetic-policy".into()),
        directory.path().join("missing"),
    );
    let selected = PipelineBuildOptions {
        devices: Some(vec![0]),
        ..Default::default()
    };
    let mut supplied = Vec::new();
    for axis in 0..6 {
        let mut request = selected.clone();
        *match axis {
            0 => &mut request.parallelism.pipeline_parallel,
            1 => &mut request.parallelism.tensor_parallel,
            2 => &mut request.parallelism.expert_parallel,
            3 => &mut request.parallelism.data_parallel,
            4 => &mut request.parallelism.sequence_parallel,
            _ => &mut request.parallelism.context_parallel,
        } = 2;
        supplied.push(request);
    }
    for devices in [None, Some(vec![]), Some(vec![1]), Some(vec![0, 1])] {
        supplied.push(PipelineBuildOptions {
            devices,
            ..selected.clone()
        });
    }
    supplied.push(PipelineBuildOptions {
        rank_backend: PipelineRankBackend::Process,
        ..selected.clone()
    });
    supplied.push(PipelineBuildOptions {
        rank_timeout: std::time::Duration::from_secs(31),
        ..selected.clone()
    });
    supplied.push(PipelineBuildOptions {
        rank_restarts: 1,
        ..selected.clone()
    });
    for parallel in supplied {
        let inline = ServingModelOptions {
            parallel: Some(parallel.clone()),
            ..Default::default()
        };
        let error = trap_strategy(
            Some(thread_unsharded_policy()),
            Some(selected.clone()),
            SelectedImplementationKind::Pipeline,
        )
        .resolve_options(&config.descriptor().spec.family, &inline, bounded_options())
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("parallel options differ from the frozen selected strategy"),
            "{error}"
        );
        let error = ResidentModelPlanner
            .prepare_strategy(
                &config,
                BackendSelection::Cuda,
                None,
                bounded_options(),
                Some(parallel),
                trap_strategy(
                    Some(thread_unsharded_policy()),
                    Some(selected.clone()),
                    SelectedImplementationKind::Pipeline,
                ),
            )
            .err()
            .expect("override must fail before any hook");
        assert!(
            error
                .to_string()
                .contains("parallel options differ from the frozen selected strategy"),
            "{error}"
        );
    }
    // An explicit default is not permission to create a request the selector omitted.
    assert!(
        ResidentModelPlanner
            .prepare_strategy(
                &config,
                BackendSelection::Cuda,
                None,
                bounded_options(),
                Some(PipelineBuildOptions::default()),
                trap_strategy(None, None, SelectedImplementationKind::Resident)
            )
            .err()
            .unwrap()
            .to_string()
            .contains("parallel options differ")
    );
    #[cfg(unix)]
    {
        use ferrule_runtime::parallel::process::ProcessLaunch;
        let launch = ProcessLaunch::new("missing-worker")
            .arg("--profile")
            .arg("selected")
            .env("PROFILE", "selected")
            .env("OTHER", "value");
        let selected = PipelineBuildOptions {
            rank_backend: PipelineRankBackend::Process,
            process_launch: Some(launch.clone()),
            ..Default::default()
        };
        let mut variants = vec![None];
        let mut changed = launch.clone();
        changed.executable = "different-worker".into();
        variants.push(Some(changed));
        let mut changed = launch.clone();
        changed.args[1] = "different-profile".into();
        variants.push(Some(changed));
        let mut changed = launch.clone();
        changed.args.reverse();
        variants.push(Some(changed));
        let mut changed = launch.clone();
        changed.environment[0].0 = "DIFFERENT_KEY".into();
        variants.push(Some(changed));
        let mut changed = launch.clone();
        changed.environment[0].1 = "different-profile".into();
        variants.push(Some(changed));
        let mut changed = launch.clone();
        changed.environment.reverse();
        variants.push(Some(changed));
        let mut changed = launch.clone();
        changed.environment.push(("EXTRA".into(), "value".into()));
        variants.push(Some(changed));
        for process_launch in variants {
            let error = ResidentModelPlanner
                .prepare_strategy(
                    &config,
                    BackendSelection::Cuda,
                    None,
                    bounded_options(),
                    Some(PipelineBuildOptions {
                        process_launch,
                        ..selected.clone()
                    }),
                    trap_strategy(
                        Some(thread_unsharded_policy()),
                        Some(selected.clone()),
                        SelectedImplementationKind::Resident,
                    ),
                )
                .err()
                .unwrap();
            assert!(
                error.to_string().contains("parallel options differ"),
                "{error}"
            );
        }
    }
    assert_no_unexpected_hooks();
}

#[test]
fn frozen_capabilities_reject_before_serving_resolver_and_admission_hooks() {
    use execution_policy::*;
    UNEXPECTED_HOOKS.with(|calls| calls.set([0; 6]));
    let directory = tempfile::tempdir().unwrap();
    let config = absent_config(
        ModelFamily::Unknown("synthetic-policy".into()),
        directory.path().join("missing"),
    );
    let policy = thread_unsharded_policy();
    let root_policy = TopologyPolicy {
        layout: OwnerLayout::RootWithExperts,
        ..policy
    };
    let cases = [
        (
            Some(policy),
            Some(PipelineBuildOptions {
                rank_backend: PipelineRankBackend::Process,
                ..Default::default()
            }),
            "thread ranks",
        ),
        (
            Some(policy),
            Some(PipelineBuildOptions {
                parallelism: ferrule_common::ParallelismPlan {
                    tensor_parallel: 2,
                    ..Default::default()
                },
                ..Default::default()
            }),
            "TP degrees",
        ),
        (
            Some(root_policy),
            Some(PipelineBuildOptions {
                rank_backend: PipelineRankBackend::Process,
                parallelism: ferrule_common::ParallelismPlan {
                    expert_parallel: 2,
                    ..Default::default()
                },
                devices: Some(vec![0, 1]),
                ..Default::default()
            }),
            "thread ranks",
        ),
        (
            Some(policy),
            Some(PipelineBuildOptions {
                rank_restarts: 1,
                ..Default::default()
            }),
            "restart",
        ),
        (
            Some(policy),
            Some(PipelineBuildOptions {
                rank_timeout: std::time::Duration::from_millis(1),
                ..Default::default()
            }),
            "rank timeout",
        ),
        (
            Some(policy),
            Some(PipelineBuildOptions {
                devices: Some(vec![]),
                ..Default::default()
            }),
            "CUDA ordinals",
        ),
        // Implicit local/default requests must also satisfy the frozen layout.
        (Some(root_policy), None, "owner layout"),
        (
            None,
            Some(PipelineBuildOptions::default()),
            "no topology policy",
        ),
    ];
    for (policy, parallel, expected) in cases {
        for kind in [
            SelectedImplementationKind::Pipeline,
            SelectedImplementationKind::Resident,
        ] {
            for compatibility in [None, parallel.clone()] {
                let error = ResidentModelPlanner
                    .prepare_strategy(
                        &config,
                        BackendSelection::Cuda,
                        None,
                        bounded_options(),
                        compatibility,
                        trap_strategy(policy, parallel.clone(), kind),
                    )
                    .err()
                    .expect("invalid frozen policy");
                assert!(error.to_string().contains(expected), "{error}");
            }
            let error = trap_strategy(policy, parallel.clone(), kind)
                .resolve_options(
                    &config.descriptor().spec.family,
                    &ServingModelOptions::default(),
                    bounded_options(),
                )
                .unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }
    assert_no_unexpected_hooks();
}

#[test]
fn attachment_and_report_refresh_cannot_bypass_frozen_execution_policy() {
    UNEXPECTED_HOOKS.with(|calls| calls.set([0; 6]));
    let entry = synthetic_entry();
    let directory = tempfile::tempdir().unwrap();
    let config = absent_config(entry.family.clone(), directory.path().join("missing"));
    for kind in [
        SelectedImplementationKind::Pipeline,
        SelectedImplementationKind::Resident,
    ] {
        let mut plan = ResidentModelPlanner
            .prepare_catalog_serving(
                &entry,
                &config,
                BackendSelection::Cpu,
                None,
                bounded_options(),
                SelectionRequest::default(),
            )
            .unwrap();
        let parallel = PipelineBuildOptions {
            rank_backend: PipelineRankBackend::Process,
            ..Default::default()
        };
        plan.backend = ModelExecutionBackend::Cuda;
        plan.request.execution.topology_policy = Some(thread_unsharded_policy());
        plan.request.execution.option_policy = &TRAP_OPTIONS;
        plan.requested_parallel = Some(parallel.clone());
        assert!(
            plan.refresh_facts()
                .unwrap_err()
                .to_string()
                .contains("thread ranks")
        );
        let error = trap_strategy(
            Some(thread_unsharded_policy()),
            Some(parallel.clone()),
            kind,
        )
        .attach(plan, parallel)
        .err()
        .unwrap();
        assert!(error.to_string().contains("thread ranks"));
    }
    assert_no_unexpected_hooks();
}

#[test]
fn public_planner_rejects_explicit_mixed_defaults_before_missing_checkpoint_io() {
    fixture::with_workspace(
        concat!(
            module_path!(),
            "::public_planner_rejects_explicit_mixed_defaults_before_missing_checkpoint_io"
        ),
        || {
            UNEXPECTED_HOOKS.with(|calls| calls.set([0; 6]));
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("missing-checkpoint");
            assert!(!path.exists());
            let config = absent_config(ModelFamily::Qwen35Moe, path.clone());
            for field in 0..8 {
                let mut serving = ServingModelOptions {
                    model_profile: Some(ServingModelProfile::Qwen35MixedBf16),
                    ..Default::default()
                };
                match field {
                    0 => serving.model_precision = Some(ModelPrecision::Bf16),
                    1 => serving.parallel = Some(PipelineBuildOptions::default()),
                    2 => serving.expert_host_cache_entries = Some(0),
                    3 => serving.expert_host_cache_mebibytes = Some(0),
                    4 => serving.expert_pinned_cache_entries = Some(0),
                    5 => serving.expert_pinned_cache_mebibytes = Some(0),
                    6 => serving.moe_hotset_experts = Some(0),
                    _ => serving.engine = ServingEngineSelection::Pipeline,
                }
                let expected = serving
                    .validate_before_model_io(
                        BackendSelection::Cuda,
                        &bounded_options(),
                        &ServingOptionProvenance::default(),
                    )
                    .unwrap_err()
                    .to_string();
                let error = ResidentModelPlanner
                    .prepare_serving(
                        &config,
                        BackendSelection::Cuda,
                        None,
                        bounded_options(),
                        serving.clone(),
                    )
                    .err()
                    .expect("public policy rejection");
                assert!(matches!(error, Error::InvalidRequest { .. }), "{error}");
                assert_eq!(error.to_string(), expected);
                // Exercise the normalization-to-catalog chain with every metadata hook
                // trapped, rather than comparing two pure validation functions only.
                let mut entry = synthetic_entry();
                entry.resolve = |_, _| unexpected_hook(1);
                entry.build = |_| unexpected_hook(5);
                entry.select = |_, _, _, _| unexpected_hook(0);
                let error = SelectionRequest {
                    serving,
                    numeric_model: None,
                }
                .normalize(BackendSelection::Cuda, &bounded_options())
                .and_then(|request| {
                    ResidentModelPlanner.prepare_catalog_serving(
                        &entry,
                        &config,
                        BackendSelection::Cuda,
                        None,
                        bounded_options(),
                        request,
                    )
                })
                .err()
                .unwrap();
                assert_eq!(error.to_string(), expected);
            }
            assert!(!path.exists());
            assert_no_unexpected_hooks();
        },
    );
}
