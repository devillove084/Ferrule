#[path = "../../../host_kv/runner_fixture.rs"]
mod host_kv_fixture;
#[path = "host_kv_composition.rs"]
mod host_kv_tests;
#[path = "../../../idle_kv_reclaim/factory.rs"]
mod idle_reclaim_tests;
#[path = "qwen35_process.rs"]
mod process_tests;
#[path = "qwen35_trace.rs"]
mod trace_tests;

use super::*;
use ferrule_runtime::{
    GenerateRequest, InferenceShutdownProgress, RequestId, ResidentDriverStep, SessionId,
};

// Only the CUDA-specific telemetry hook is substituted. The production
// decorator and real resident inference/cleanup implementations run unchanged.
impl MoeShutdownReport for BoxedSessionInferenceEngine {
    fn report_shutdown(&self) {
        assert_eq!(
            self.capacity_snapshot().kv.unwrap().logical_allocated_pages,
            0
        );
    }
}

fn reported_cpu_engine() -> ReportedMoeEngine<BoxedSessionInferenceEngine> {
    let fixture = fixture::Fixture::new();
    let descriptor = super::super::tests::descriptor(ModelFamily::Qwen35, "tiny-hybrid", 4);
    let mut options = super::super::tests::options();
    options.driver_config.ctx_size = 32;
    options.driver_config.enable_native_proposals = false;
    options.scheduler_config.max_batch_tokens = 4;
    options.scheduler_config.prefill_chunk_size = 2;
    let entry = resolve(&descriptor, BackendSelection::Cpu).unwrap();
    let request = configure_request(&descriptor, entry, options).unwrap();
    ReportedMoeEngine {
        model_label: "Qwen3.5-35B",
        engine: resident_f32_bound::build_resident_f32_bound(
            request,
            fixture.resources(),
            TokenizerHandle::load(&fixture.dir).unwrap(),
        )
        .unwrap(),
        host_experts: None,
        limits: Qwen35MoeCapacityLimits::default().into(),
        numeric_profile: NumericFp8Profile::F32,
        expert_placement: None,
        shutdown_attempted: false,
        close_reported: false,
    }
}

#[test]
fn reported_moe_cpu_forwards_admission_capacity_and_close_barrier() {
    use ferrule_runtime::engine::InferenceEngine;
    let mut engine = reported_cpu_engine();
    let limits = ferrule_runtime::RuntimeAdmissionOptions {
        max_waiting_requests: 1,
        max_request_identities: 2,
        max_session_identities: 2,
    };
    engine.set_admission_options(limits).unwrap();
    assert_eq!(engine.admission_snapshot().unwrap().limits, limits);
    assert_eq!(
        engine.admission_snapshot(),
        engine.inner().admission_snapshot()
    );
    let capacity = engine.capacity_snapshot();
    assert!(capacity.scheduler.is_some());
    assert!(capacity.kv.is_some());
    assert_eq!(capacity, engine.inner().capacity_snapshot());
    assert!(matches!(
        engine.set_admission_options(ferrule_runtime::RuntimeAdmissionOptions {
            max_waiting_requests: 0,
            ..limits
        }),
        Err(Error::Admission {
            source: ferrule_runtime::RuntimeAdmissionError::InvalidOptions
        })
    ));
    assert_eq!(engine.admission_snapshot().unwrap().limits, limits);
    let request = |id| GenerateRequest {
        id: RequestId(id),
        session_id: Some(SessionId(id)),
        prompt_tokens: vec![1, 3, 2],
        max_new_tokens: 2,
        stop: vec![],
        ignore_eos: true,
    };
    engine.try_submit(request(1)).unwrap();
    assert_eq!(engine.admission_snapshot().unwrap().waiting_requests, 1);
    assert_eq!(
        engine
            .capacity_snapshot()
            .scheduler
            .unwrap()
            .waiting_requests,
        1
    );
    assert!(matches!(
        engine.try_submit(request(1)),
        Err(Error::Admission {
            source: ferrule_runtime::RuntimeAdmissionError::DuplicateRequest { .. }
        })
    ));
    assert!(matches!(
        engine.try_submit(request(2)),
        Err(Error::Admission {
            source: ferrule_runtime::RuntimeAdmissionError::Capacity { .. }
        })
    ));
    let ferrule_runtime::InferenceRequestCleanup::Tracked(receipt) =
        engine.request_cleanup(RequestId(1))
    else {
        panic!("production wrapper lost the resident receipt")
    };
    engine.step(&mut |_| Ok(())).unwrap();
    assert_eq!(
        engine.capacity_snapshot(),
        engine.inner().capacity_snapshot()
    );
    assert!(
        engine
            .capacity_snapshot()
            .kv
            .unwrap()
            .logical_allocated_pages
            > 0
    );
    engine.close_admission().unwrap();
    assert!(engine.admission_snapshot().unwrap().closed);
    assert!(engine.inner().admission_snapshot().unwrap().closed);
    assert!(
        !receipt.is_released(),
        "close is not terminal or physical cleanup"
    );
    assert!(matches!(
        engine.try_submit(request(3)),
        Err(Error::Admission {
            source: ferrule_runtime::RuntimeAdmissionError::Closed
        })
    ));
    engine.cancel_request(RequestId(1)).unwrap();
    assert!(!receipt.is_released(), "terminal is still unconsumed");
    assert_eq!(engine.drain_cancelled().len(), 1);
    assert!(receipt.is_released());
    assert_eq!(
        engine.admission_snapshot().unwrap().request_identities_held,
        0
    );
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
    assert!(engine.shutdown_attempted && engine.close_reported);
}

#[test]
fn reported_moe_cpu_forwards_exact_receipts_across_retained_turns() {
    use ferrule_runtime::engine::{InferenceEngine, SessionInferenceEngine};
    let mut engine = reported_cpu_engine();
    let report = engine.expert_report().unwrap();
    assert!(report.contains("compressed E4M3FN + numeric BF16 scales; F32Tf32x3"));
    engine.expert_placement = Some(Qwen35ExpertPlacement::new(8, (0..8).collect()).unwrap().0);
    let ep_report = engine.expert_report().unwrap();
    assert!(ep_report.contains("CUDA thread EP8; root=0"));
    assert!(ep_report.contains("root-only KV; TP/PP/restarts unsupported; host=missing"));
    engine.retain_session(SessionId(7)).unwrap();
    let mut previous: Option<ferrule_runtime::RequestCleanupReceipt> = None;
    for round in 0..2 {
        engine
            .try_submit(GenerateRequest {
                id: RequestId(1),
                session_id: Some(SessionId(7)),
                prompt_tokens: engine.encode("t1 t3 t2").unwrap(),
                max_new_tokens: 1,
                stop: vec![],
                ignore_eos: true,
            })
            .unwrap();
        let ferrule_runtime::InferenceRequestCleanup::Tracked(receipt) =
            engine.request_cleanup(RequestId(1))
        else {
            panic!("production wrapper must forward the exact generation receipt")
        };
        let ferrule_runtime::InferenceRequestCleanup::Tracked(inner_receipt) =
            engine.inner().request_cleanup(RequestId(1))
        else {
            panic!("real resident must issue a receipt")
        };
        if let Some(old) = &previous {
            assert!(old.is_released());
        }
        let mut tokens = 0;
        for _ in 0..100 {
            engine
                .step(&mut |_| {
                    tokens += 1;
                    Ok(())
                })
                .unwrap();
            assert!(!receipt.is_released());
            assert!(!inner_receipt.is_released());
            let consumed = if round == 0 {
                !engine.drain_finished().is_empty()
            } else {
                engine.take_request_terminal(RequestId(1)).is_some()
            };
            if consumed {
                break;
            }
        }
        assert_eq!(tokens, 1);
        assert!(receipt.is_released());
        assert!(inner_receipt.is_released());
        assert!(matches!(
            engine.request_cleanup(RequestId(1)),
            ferrule_runtime::InferenceRequestCleanup::Unavailable
        ));
        assert_eq!(
            engine.admission_snapshot().unwrap().request_identities_held,
            0
        );
        assert_eq!(
            engine.admission_snapshot().unwrap().session_identities_held,
            1
        );
        assert_eq!(
            engine.retained_session_position(SessionId(7)),
            Some((round + 1) * 4)
        );
        assert!(
            engine
                .capacity_snapshot()
                .kv
                .unwrap()
                .logical_allocated_pages
                > 0
        );
        previous = Some(receipt);
    }
    engine.reset_session(SessionId(7)).unwrap();
    assert_eq!(engine.retained_session_position(SessionId(7)), Some(0));
    assert_eq!(
        engine
            .capacity_snapshot()
            .kv
            .unwrap()
            .logical_allocated_pages,
        0
    );
    assert!(engine.drain_failed().is_empty());
    assert_eq!(engine.expert_report().unwrap(), ep_report);
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
    assert!(engine.shutdown_attempted && engine.close_reported);
}

#[test]
fn family_options_legacy_dto_matrix_preserves_defaults_and_rejections() {
    use ferrule_model::transformer::host_experts::{ExpertPrewarmMode, HostExpertCacheOptions};
    for family in [
        ModelFamily::Qwen3,
        ModelFamily::QwenMoe,
        ModelFamily::DeepSeekV4,
        ModelFamily::Qwen35,
        ModelFamily::Qwen35Moe,
    ] {
        for capacity in [None, Some(Qwen35MoeCapacityLimits::default())] {
            for host in [
                None,
                Some(HostExpertCacheOptions::default()),
                Some(HostExpertCacheOptions {
                    mode: ExpertPrewarmMode::Lazy,
                    max_bytes: 0,
                    max_experts: 0,
                    ..Default::default()
                }),
            ] {
                let mut input = super::super::tests::options();
                input.qwen35_moe_capacity = capacity;
                input.qwen35_host_cache = host;
                let result = resolve_family_options(&family, &input);
                if family != ModelFamily::Qwen35Moe && (capacity.is_some() || host.is_some()) {
                    let Error::InvalidRequest { message } = result.unwrap_err() else {
                        panic!("legacy family rejection must remain InvalidRequest");
                    };
                    assert_eq!(
                        message,
                        if capacity.is_some() {
                            "CUDA expert device cache options require Qwen3.5 MoE FP8"
                        } else {
                            "expert prewarm options require Qwen3.5 MoE FP8"
                        }
                    );
                    continue;
                }
                match result.unwrap() {
                    ResolvedFamilyOptions::Qwen35(Qwen35ResolvedAdapter::Moe {
                        options,
                        expert_placement,
                    }) => {
                        assert_eq!(family, ModelFamily::Qwen35Moe);
                        assert_eq!(options.device_capacity, capacity);
                        assert_eq!(options.host_cache, host.unwrap_or_default());
                        assert!(expert_placement.is_none());
                    }
                    ResolvedFamilyOptions::Qwen35(Qwen35ResolvedAdapter::Dense) => {
                        assert_eq!(family, ModelFamily::Qwen35);
                    }
                    ResolvedFamilyOptions::Generic => {
                        assert!(!matches!(
                            family,
                            ModelFamily::Qwen35 | ModelFamily::Qwen35Moe
                        ));
                    }
                }
                assert_eq!(input.qwen35_moe_capacity, capacity);
                assert_eq!(input.qwen35_host_cache, host);
            }
        }
    }
}

#[test]
fn family_adapter_keeps_capacity_before_host_validation_and_typed_host_errors() {
    use ferrule_model::transformer::host_experts::{HostExpertCacheError, HostExpertCacheOptions};
    let mut input = super::super::tests::options();
    input.qwen35_moe_capacity = Some(Qwen35MoeCapacityLimits {
        max_experts: 0,
        ..Default::default()
    });
    input.qwen35_host_cache = Some(HostExpertCacheOptions {
        workers: 0,
        ..Default::default()
    });
    let error = resolve_family_options(&ModelFamily::Qwen35Moe, &input).unwrap_err();
    assert!(matches!(error, Error::InvalidRequest { ref message }
        if message == "35B requires nonzero bounded expert cache with room beyond scratch"));
    input.qwen35_moe_capacity = None;
    let Error::Backend {
        source: ferrule_common::Error::ModelSource { source },
    } = resolve_family_options(&ModelFamily::Qwen35Moe, &input).unwrap_err()
    else {
        panic!("host validation must preserve the typed model source");
    };
    assert!(source.downcast_ref::<HostExpertCacheError>().is_some());
}

#[test]
fn resolved_capacity_uses_current_common_inputs_without_a_second_budget() {
    let descriptor = super::super::tests::descriptor(ModelFamily::Qwen35Moe, "metadata", 40);
    let mut input = super::super::tests::options();
    input.max_tensor_mebibytes = 1024;
    input.driver_config.enable_native_proposals = false;
    let requested = Qwen35MoeCapacityLimits {
        max_positions: 99,
        max_sequences: 99,
        max_batch_tokens: 99,
        ..Default::default()
    };
    input.qwen35_moe_capacity = Some(requested);
    // Internal metadata seam also exercises the adapter in a CPU-only build.
    // This does not bypass the public resolver's CUDA feature check.
    let entry = ResolvedModelBackend {
        family: ModelFamily::Qwen35Moe,
        model_name: "qwen3.5-35b-a3b-fp8",
        backend: ModelExecutionBackend::Cuda,
        backend_profile: Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
        default_chat_template: ChatTemplate::Qwen35,
    };
    let mut request = configure_request(&descriptor, entry, input).unwrap();
    for (context, sequences, batch, kv) in [(8, 1, 4, None), (64, 3, 32, Some(2 << 20))] {
        request.driver_config.ctx_size = context;
        request.scheduler_config.max_active_sequences = sequences;
        request.scheduler_config.max_batch_tokens = batch;
        request.kv_cache_bytes = kv;
        assert_eq!(
            capacity_limits(&request).unwrap(),
            Qwen35MoeCapacityLimits {
                max_positions: context,
                max_sequences: sequences,
                max_batch_tokens: batch,
                kv_bytes: kv.map_or(requested.kv_bytes, |bytes| bytes as usize),
                ..requested
            }
        );
    }
    let family = request.family_options.qwen35_mut().unwrap();
    family.set_expert_placement(Qwen35ExpertPlacement::new(2, vec![3, 1]).unwrap());
    let compatibility = ModelBuildRequest {
        resolved: request,
        prepared_source_plan: None,
    };
    let cloned = compatibility.clone();
    assert_eq!(
        capacity_limits(&compatibility.resolved).unwrap(),
        capacity_limits(&cloned.resolved).unwrap()
    );
    let family = cloned.resolved.family_options.qwen35().unwrap();
    assert_eq!(family.moe().unwrap().device_capacity, Some(requested));
    assert_eq!(family.expert_placement().unwrap().expert_devices(), &[3, 1]);
}

fn assert_ep_error(error: Error) -> String {
    let Error::Backend {
        source: ferrule_common::Error::ModelSource { source },
    } = error
    else {
        panic!("expected typed Qwen3.5 EP capability error")
    };
    let unsupported = source
        .downcast_ref::<ferrule_model::transformer::UnsupportedOperator>()
        .unwrap();
    assert_eq!(unsupported.operator, "qwen35_moe_expert_parallel");
    unsupported.to_string()
}

#[test]
fn ep_placement_is_explicit_distinct_and_not_a_kv_mesh() {
    for degree in [2, 4, 8] {
        let devices = (0..degree).rev().collect::<Vec<_>>();
        let placement = Qwen35ExpertPlacement::new(degree, devices.clone()).unwrap();
        assert_eq!(placement.root_device(), degree - 1);
        assert_eq!(placement.expert_devices(), devices);
        assert_eq!(
            placement.validate_expert_shape(40, 256).unwrap(),
            10240 / degree
        );
        assert_eq!(placement.validate_expert_shape(2, 8).unwrap(), 16 / degree);
        for (layers, experts) in [(0, 256), (40, 0), (40, 3)] {
            assert_ep_error(
                placement
                    .validate_expert_shape(layers, experts)
                    .unwrap_err(),
            );
        }
    }
    for (degree, devices) in [
        (0, vec![]),
        (1, vec![0]),
        (3, vec![0, 1, 2]),
        (2, vec![0]),
        (2, vec![0, 1, 2]),
        (2, vec![1, 1]),
        (2, vec![0, usize::MAX]),
    ] {
        assert_ep_error(Qwen35ExpertPlacement::new(degree, devices).unwrap_err());
    }
    assert!(
        Qwen35ExpertPlacement::new(2, vec![0, 1])
            .unwrap()
            .validate_expert_shape(usize::MAX, 4)
            .is_err()
    );
}

#[test]
fn ep_rejects_unimplemented_topologies_without_cuda_or_payloads() {
    let valid = PipelineBuildOptions {
        parallelism: ferrule_common::ParallelismPlan {
            expert_parallel: 8,
            ..Default::default()
        },
        devices: Some((0..8).collect()),
        ..Default::default()
    };
    assert!(Qwen35ExpertPlacement::from_options(&valid, BackendSelection::Auto).is_ok());
    assert_ep_error(
        Qwen35ExpertPlacement::from_options(&valid, BackendSelection::Cpu).unwrap_err(),
    );
    for variant in 0..10 {
        let mut options = valid.clone();
        match variant {
            0 => options.parallelism.tensor_parallel = 2,
            1 => options.parallelism.pipeline_parallel = 2,
            2 => options.parallelism.data_parallel = 2,
            3 => options.parallelism.context_parallel = 2,
            4 => options.parallelism.sequence_parallel = 2,
            5 => options.rank_backend = PipelineRankBackend::Process,
            6 => options.rank_restarts = 1,
            7 => options.rank_timeout = std::time::Duration::from_secs(1),
            8 => options.devices = None,
            _ => options.parallelism.expert_parallel = 1,
        }
        assert_ep_error(
            Qwen35ExpertPlacement::from_options(&options, BackendSelection::Auto).unwrap_err(),
        );
    }
}

fn tiny_ep_plan() -> Qwen35ExpertPhysicalPlan {
    Qwen35ExpertPhysicalPlan::new(
        &Qwen35ExpertPlacement::new(2, vec![7, 3]).unwrap(),
        Qwen35ExpertRootMemory {
            weights_bytes: 60,
            state_bytes: 10,
            kv_bytes: 10,
            workspace_bytes: 5,
        },
        &[40, 41],
        5,
        10,
    )
    .unwrap()
}

#[test]
fn ep_physical_budget_rejects_individually_fitting_colocated_contexts() {
    let plan = tiny_ep_plan();
    assert_eq!(
        plan.cards()[0],
        Qwen35PhysicalCardMemory {
            ordinal: 7,
            root_weights_bytes: 60,
            expert_weights_bytes: 40,
            state_bytes: 10,
            kv_bytes: 10,
            workspace_bytes: 10,
            contexts: 2,
            allocator_margin_bytes: 20,
            required_bytes: 150,
        }
    );
    assert_eq!(plan.cards()[1].required_bytes, 56);
    assert_eq!(plan.cards()[1].contexts, 1);
    assert_eq!(plan.cards()[1].state_bytes, 0);
    assert_eq!(plan.cards()[1].kv_bytes, 0);
    // Root = 95 and expert0 = 55 both fit 100; their combined 150 does not.
    let mut capacities = vec![
        Qwen35PhysicalCardCapacity {
            ordinal: 3,
            free_bytes: 56,
            limit_bytes: 56,
        },
        Qwen35PhysicalCardCapacity {
            ordinal: 7,
            free_bytes: 100,
            limit_bytes: 150,
        },
    ];
    assert!(assert_ep_error(plan.check_capacity(&capacities).unwrap_err()).contains("ordinal 7"));
    capacities[1].free_bytes = 150;
    plan.check_capacity(&capacities).unwrap();
    capacities[1].limit_bytes = 149;
    assert!(plan.check_capacity(&capacities).is_err());
    capacities[1].limit_bytes = 150;
    capacities[0].free_bytes = 55;
    assert!(assert_ep_error(plan.check_capacity(&capacities).unwrap_err()).contains("ordinal 3"));
}

#[test]
fn ep_physical_budget_requires_complete_unique_samples_and_checked_sums() {
    let plan = tiny_ep_plan();
    let sample = Qwen35PhysicalCardCapacity {
        ordinal: 7,
        free_bytes: usize::MAX,
        limit_bytes: usize::MAX,
    };
    for samples in [
        vec![],
        vec![sample],
        vec![sample, sample],
        vec![
            sample,
            Qwen35PhysicalCardCapacity {
                ordinal: 99,
                ..sample
            },
        ],
    ] {
        assert_ep_error(plan.check_capacity(&samples).unwrap_err());
    }
    let placement = Qwen35ExpertPlacement::new(2, vec![0, 1]).unwrap();
    let root = Qwen35ExpertRootMemory {
        weights_bytes: 1,
        state_bytes: 1,
        kv_bytes: 1,
        workspace_bytes: 1,
    };
    for owners in [&[1usize][..], &[0, 1], &[usize::MAX, 1]] {
        assert!(Qwen35ExpertPhysicalPlan::new(&placement, root, owners, 1, 1).is_err());
    }
    assert!(Qwen35ExpertPhysicalPlan::new(&placement, root, &[1, 1], 1, usize::MAX).is_err());
}

#[test]
fn mock_live_capacity_reject_preserves_typed_primary_and_secondary_cleanup() {
    let plan = tiny_ep_plan();
    for cleanup_fails in [false, true] {
        // Metadata budgets remain valid, but live free memory has changed.
        let samples = plan
            .cards()
            .iter()
            .map(|card| Qwen35PhysicalCardCapacity {
                ordinal: card.ordinal,
                free_bytes: card.required_bytes - 1,
                limit_bytes: card.required_bytes,
            })
            .collect::<Vec<_>>();
        let mut cleanup_calls = 0;
        let held_owners = std::cell::Cell::new(1);
        let result = finish_owner_admission(plan.check_capacity(&samples), || {
            cleanup_calls += 1;
            if cleanup_fails {
                Err(ep_unsupported("mock owner shutdown unknown"))
            } else {
                held_owners.set(0);
                Ok(())
            }
        });
        assert_eq!(cleanup_calls, 1);
        assert_eq!(held_owners.get(), usize::from(cleanup_fails));
        let error = result.unwrap_err();
        if cleanup_fails {
            let Error::Cleanup {
                source, cleanup, ..
            } = error
            else {
                panic!("typed pair lost")
            };
            assert!(assert_ep_error(*source).contains("requires"));
            assert!(assert_ep_error(*cleanup).contains("shutdown unknown"));
        } else {
            assert!(assert_ep_error(error).contains("requires"));
        }
    }
    // Successful admission transfers ownership; it must not run rollback.
    assert_eq!(
        finish_owner_admission(Ok(7), || panic!("cleanup after success")).unwrap(),
        7
    );
}

#[test]
fn ep8_expert_estimates_charge_all_owned_weights_without_root_cache_ceiling() {
    let placement = Qwen35ExpertPlacement::new(8, (0..8).collect()).unwrap();
    let owned = placement.validate_expert_shape(40, 256).unwrap();
    assert_eq!(owned, 1280);
    let owner_weights = owned * 3_146_112;
    let root = Qwen35ExpertRootMemory {
        // Mock estimator, not an artifact admission or hardcoded runtime budget.
        weights_bytes: 5_578_000_000,
        state_bytes: 512 << 20,
        kv_bytes: 1 << 30,
        workspace_bytes: 1 << 30,
    };
    let plan =
        Qwen35ExpertPhysicalPlan::new(&placement, root, &[owner_weights; 8], 64 << 20, 512 << 20)
            .unwrap();
    assert_eq!(plan.cards()[0].root_weights_bytes, root.weights_bytes);
    assert_eq!(
        plan.cards()[1].required_bytes,
        owner_weights + (64 << 20) + (512 << 20)
    );
    assert_eq!(
        plan.cards()[0].required_bytes - plan.cards()[1].required_bytes,
        root.weights_bytes + root.state_bytes + root.kv_bytes + root.workspace_bytes + (512 << 20)
    );
    let capacities = plan
        .cards()
        .iter()
        .map(|card| Qwen35PhysicalCardCapacity {
            ordinal: card.ordinal,
            free_bytes: card.required_bytes,
            limit_bytes: card.required_bytes,
        })
        .collect::<Vec<_>>();
    plan.check_capacity(&capacities).unwrap();
}

#[test]
fn ep_root_admission_uses_published_external_api_without_a_local_cache_charge() {
    use ferrule_model::decoder::HybridMemoryPlan;
    #[cfg(feature = "cuda")]
    use ferrule_model::decoder::{HybridCudaDecoder, HybridCudaDevice, HybridCudaRoutedExperts};
    use ferrule_model::transformer::{ExpertCacheLimits, NumericFp8Precision};
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Qwen35Moe,
        WeightSource::Safetensors,
        16,
        32,
        2,
        2,
        1 << 20,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap();
    assert!(Qwen35ExpertRootMemory::for_resources(&resources, &options, 8).is_err());
    let options = options
        .with_hybrid_cuda_numeric_fp8_precision(
            ExpertCacheLimits {
                max_experts: 8,
                max_bytes: 4usize << 30,
            },
            64 << 20,
            NumericFp8Precision::F32Tf32x3,
        )
        .unwrap();
    let root = Qwen35ExpertRootMemory::for_resources(&resources, &options, 8).unwrap();
    let local = HybridMemoryPlan::for_resources(&resources, &options, 8).unwrap();
    let external =
        HybridMemoryPlan::for_resources_with_routed_experts(&resources, &options, 8).unwrap();
    assert_eq!(
        local.weight_bytes_upper_bound - root.weights_bytes,
        (4usize << 30) - (64 << 20)
    );
    assert_eq!(root.weights_bytes, external.weight_bytes_upper_bound);
    assert_eq!(root.workspace_bytes, external.workspace_bytes_upper_bound);
    assert_eq!(root.kv_bytes, external.kv_bytes);
    assert_eq!(root.state_bytes, external.per_sequence_state_bytes * 5);
    #[cfg(feature = "cuda")]
    let _constructor: fn(
        BoundDecoderResources,
        TokenizerHandle,
        GenericDecoderOptions,
        HybridCudaDevice,
        HybridCudaRoutedExperts,
    )
        -> ferrule_common::Result<GenericDecoderRunner<HybridCudaDecoder>> =
        GenericDecoderRunner::<HybridCudaDecoder>::hybrid_cuda_with_routed_experts;
}

#[cfg(feature = "cuda")]
#[test]
fn ep_construct_rejects_cpu_before_any_payload_or_worker_creation() {
    let descriptor =
        super::super::tests::descriptor(ModelFamily::Qwen35Moe, "absent-ep-fixture", 40);
    let entry = ResolvedModelBackend {
        family: ModelFamily::Qwen35Moe,
        model_name: "qwen35-ep-fixture",
        backend: ModelExecutionBackend::Cuda,
        backend_profile: Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
        default_chat_template: ChatTemplate::Qwen35,
    };
    let mut options = super::super::tests::options();
    options.max_tensor_mebibytes = 1024;
    options.driver_config.enable_native_proposals = false;
    let mut request = configure_request(&descriptor, entry, options).unwrap();
    assert!(
        request
            .family_options
            .qwen35()
            .unwrap()
            .expert_placement()
            .is_none(),
        "default stays single"
    );
    request
        .family_options
        .qwen35_mut()
        .unwrap()
        .set_expert_placement(Qwen35ExpertPlacement::new(2, vec![1, 0]).unwrap());
    request.execution.construction = ResolvedConstruction::Resident(build_experts);
    request.backend = ModelExecutionBackend::Cpu;
    let error = build_experts(ModelBuildRequest {
        resolved: request,
        prepared_source_plan: None,
    })
    .err()
    .expect("EP must not fall back to CPU");
    assert!(assert_ep_error(error).contains("CUDA resident root"));
}

fn run(engine: &mut BoxedSessionInferenceEngine, id: u64) -> Vec<u32> {
    engine
        .try_submit(GenerateRequest {
            id: RequestId(id),
            session_id: Some(SessionId(7)),
            prompt_tokens: vec![1, 3, 2],
            max_new_tokens: 2,
            stop: vec![],
            ignore_eos: false,
        })
        .unwrap();
    let mut tokens = vec![];
    for _ in 0..100 {
        let step = engine
            .step(&mut |event| {
                tokens.push(event.token);
                Ok(())
            })
            .unwrap();
        assert!(!matches!(step, ResidentDriverStep::Blocked));
        assert!(engine.drain_failed().is_empty());
        if !engine.drain_finished().is_empty() {
            return tokens;
        }
    }
    panic!("resident hybrid did not finish");
}

#[test]
#[ignore = "requires NAS Qwen3.5-0.8B headers; no payload or CUDA initialization"]
fn qwen35_real_header_budget_uses_compact_kv_and_all_recurrent_layers() {
    use ferrule_model::decoder::HybridMemoryPlan;
    let path = std::env::var_os("FERRULE_QWEN35_08B_DIR")
        .unwrap_or_else(|| "/mnt/nas1/hf/Qwen3.5-0.8B".into());
    let (_, resources) = Qwen35Adapter::bind_hf_metadata(std::path::Path::new(&path)).unwrap();
    let schema = HybridStateSchema::from_spec(resources.spec(), 24)
        .unwrap()
        .kv_planes(16, 1024)
        .unwrap();
    assert_eq!(schema.planes()[0].layer_count, 6);
    assert_eq!(physical_page_bytes(&schema).unwrap(), 393216);
    let options = GenericDecoderOptions::standard_cpu(
        resources.spec(),
        ModelFamily::Qwen35,
        WeightSource::Safetensors,
        16,
        1024,
        32,
        4,
        1 << 30,
        ExecutionPrecisionPolicy::f32(),
    )
    .unwrap();
    let estimate = HybridMemoryPlan::for_resources(&resources, &options, 512).unwrap();
    assert_eq!(estimate.per_sequence_state_bytes, 20_643_840);
    assert_eq!(estimate.kv_bytes, 192 * 1024 * 1024);
    assert_eq!(estimate.weight_bytes_upper_bound, 8_054_954_496);
    assert_eq!(estimate.workspace_bytes_upper_bound, 1_018_167_296);
    assert_eq!(
        (2 * options.capabilities().max_sequences + 1) * estimate.per_sequence_state_bytes,
        185_794_560
    );
}

#[cfg(feature = "cuda")]
#[test]
fn staged_moe_factory_rejects_unconsumed_options_before_loading() {
    let descriptor = super::super::tests::descriptor(
        ModelFamily::Qwen35Moe,
        "Qwen3_5MoeForConditionalGeneration",
        40,
    );
    let entry = ResolvedModelBackend {
        family: ModelFamily::Qwen35Moe,
        model_name: "qwen3.5-35b-a3b",
        backend: ModelExecutionBackend::Cuda,
        backend_profile: Qwen35MoeCapacityPlan::PLANNED_BACKEND_PROFILE,
        default_chat_template: ChatTemplate::Qwen35,
    };
    let mut options = super::super::tests::options();
    options.driver_config.enable_native_proposals = false;
    options.max_tensor_mebibytes = 1024;
    let base = configure_request(&descriptor, entry, options).unwrap();
    for case in 0..9 {
        let mut request = base.clone();
        let expected = match case {
            0 => {
                request.backend = ModelExecutionBackend::Cpu;
                "qwen35_moe_cpu"
            }
            1 => {
                request.pipeline = Some(PipelineBuildOptions::default());
                "TP/PP/EP/process"
            }
            2 => {
                request.moe_hotset_experts = 1;
                "does not consume"
            }
            3 => {
                request.expert_reader_max_tensor_bytes = 1;
                "does not consume"
            }
            4 => {
                request.output_head_chunk_rows = 1;
                "does not consume"
            }
            5 => {
                request.driver_config.enable_native_proposals = true;
                "speculation disabled"
            }
            6 => {
                request.scheduler_config.prefix_cache_capacity_pages = 1;
                "prefix cache"
            }
            7 => {
                request.scheduler_config.prefill_chunk_size = 0;
                "bounded token-serial"
            }
            _ => {
                request.expert_memory_policy = ExpertMemoryPolicy::new(
                    MemoryPoolLimits::new(1, 64 << 20),
                    MemoryPoolLimits::new(0, 0),
                );
                "does not consume"
            }
        };
        let error = build_moe(request).err().unwrap();
        assert!(error.to_string().contains(expected), "case {case}: {error}");
    }
}

#[test]
fn host_startup_passes_remaining_base_peak_without_turning_cap_into_reservation() {
    use ferrule_model::transformer::host_experts::{
        ExpertPrewarmMode, HostExpertCacheOptions, StandardHostStartupMemory,
    };
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let input = HostExpertCacheOptions::default();
    let options = host_startup_options(&resources, input, 32, 1 << 20).unwrap();
    let ledger = StandardHostStartupMemory::for_resources(&resources, 32, 1 << 20).unwrap();
    assert_eq!(options.max_bytes, input.max_bytes);
    assert_eq!(
        options.memory_budget.base_resident_bytes,
        ledger.resident_bytes().unwrap()
    );
    assert_eq!(
        options.memory_budget.base_temporary_bytes,
        ledger.temporary_bytes()
    );
    assert!(options.memory_budget.base_resident_bytes > 0);
    assert!(options.memory_budget.base_temporary_bytes > 0);
    assert_eq!(options.memory_budget.base_already_resident_bytes, 0);
    assert_eq!(options.memory_budget.concurrent_warm_bytes, 0);
    let lazy = HostExpertCacheOptions {
        mode: ExpertPrewarmMode::Lazy,
        ..input
    };
    assert_eq!(host_startup_options(&resources, lazy, 32, 1).unwrap(), lazy);
}

#[test]
fn long_context_kv_pressure_preserves_retained_owner_and_blocked_identity() {
    let context = 1024;
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources_with_context(context);
    let descriptor = super::super::tests::descriptor(ModelFamily::Qwen35, "tiny-hybrid", 4);
    let mut options = super::super::tests::options();
    options.driver_config.ctx_size = context;
    options.driver_config.enable_native_proposals = false;
    options.scheduler_config.max_batch_tokens = 16;
    options.scheduler_config.prefill_chunk_size = 16;
    let entry = resolve(&descriptor, BackendSelection::Cpu).unwrap();
    let mut request = configure_request(&descriptor, entry, options).unwrap();
    let schema = HybridStateSchema::from_spec(resources.spec(), 4)
        .unwrap()
        .kv_planes(16, context)
        .unwrap();
    request.kv_cache_bytes = Some(physical_page_bytes(&schema).unwrap());
    let mut engine = resident_f32_bound::build_resident_f32_bound(
        request,
        resources,
        TokenizerHandle::load(&fixture.dir).unwrap(),
    )
    .unwrap();
    engine.retain_session(SessionId(7)).unwrap();
    assert_eq!(run(&mut engine, 1).len(), 2);
    let retained = engine.capacity_snapshot().kv.unwrap();
    assert_eq!(retained.logical_allocated_pages, 1);
    assert_eq!(retained.logical_free_pages, Some(0));
    engine
        .try_submit(GenerateRequest {
            id: RequestId(2),
            session_id: Some(SessionId(8)),
            prompt_tokens: vec![1],
            max_new_tokens: context - 1,
            stop: vec![],
            ignore_eos: true,
        })
        .unwrap();
    assert_eq!(
        engine.admission_snapshot().unwrap().request_identities_held,
        1
    );
    assert_eq!(
        engine
            .step(&mut |_| panic!("KV-blocked work emitted output"))
            .unwrap(),
        ResidentDriverStep::Blocked
    );
    assert_eq!(
        engine
            .capacity_snapshot()
            .kv
            .unwrap()
            .logical_allocated_pages,
        1
    );
    assert_eq!(
        engine.admission_snapshot().unwrap().request_identities_held,
        1
    );
    assert!(engine.drain_failed().is_empty());
    assert!(engine.retained_session_position(SessionId(7)).unwrap() > 3);
    engine.cancel_request(RequestId(2)).unwrap();
    engine.drain_cancelled();
    assert_eq!(
        engine
            .capacity_snapshot()
            .kv
            .unwrap()
            .logical_allocated_pages,
        1,
        "cancel of blocked request cannot release retained owner's KV"
    );
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
}

#[test]
fn long_context_hybrid_engine_rejects_envelope_without_ownership_changes() {
    for context in [1024, 2048, 4096] {
        let fixture = fixture::Fixture::new();
        let descriptor = super::super::tests::descriptor(ModelFamily::Qwen35, "tiny-hybrid", 4);
        let mut options = super::super::tests::options();
        options.driver_config.ctx_size = context;
        options.driver_config.enable_native_proposals = false;
        options.scheduler_config.max_batch_tokens = 32;
        options.scheduler_config.prefill_chunk_size = 32;
        let entry = resolve(&descriptor, BackendSelection::Cpu).unwrap();
        let request = configure_request(&descriptor, entry, options).unwrap();
        let mut engine = resident_f32_bound::build_resident_f32_bound(
            request,
            fixture.resources_with_context(context),
            TokenizerHandle::load(&fixture.dir).unwrap(),
        )
        .unwrap();
        let make = |id, prompt, max_new_tokens| GenerateRequest {
            id: RequestId(id),
            session_id: Some(SessionId(7)),
            prompt_tokens: vec![1; prompt],
            max_new_tokens,
            stop: vec![],
            ignore_eos: true,
        };
        let snapshot = engine.capacity_snapshot();
        for (prompt, output) in [(context - 16, 17), (context + 1, 0), (1, usize::MAX)] {
            assert!(matches!(
                engine.try_submit(make(1, prompt, output)),
                Err(Error::Admission {
                    source: ferrule_runtime::RuntimeAdmissionError::InvalidPosition { .. }
                })
            ));
            assert_eq!(engine.capacity_snapshot(), snapshot);
            assert_eq!(
                engine.admission_snapshot().unwrap().request_identities_held,
                0
            );
            assert!(engine.drain_failed().is_empty());
        }
        engine.try_submit(make(1, context - 16, 16)).unwrap();
        assert_eq!(engine.admission_snapshot().unwrap().waiting_requests, 1);
        assert_eq!(
            engine.capacity_snapshot().kv,
            snapshot.kv,
            "acceptance acquires no KV pages"
        );
        engine.step(&mut |_| Ok(())).unwrap();
        let active_kv = engine.capacity_snapshot().kv;
        assert!(active_kv.unwrap().logical_allocated_pages > 0);
        let mut waiting = make(4, context - 16, 16);
        waiting.session_id = Some(SessionId(8));
        engine.try_submit(waiting).unwrap();
        assert_eq!(engine.admission_snapshot().unwrap().waiting_requests, 1);
        assert_eq!(
            engine.admission_snapshot().unwrap().request_identities_held,
            2
        );
        assert_eq!(engine.capacity_snapshot().kv, active_kv);
        engine.cancel_request(RequestId(4)).unwrap();
        assert_eq!(engine.drain_cancelled().len(), 1);
        assert_eq!(
            engine.capacity_snapshot().kv,
            active_kv,
            "waiting cancellation cannot release active KV"
        );
        assert_eq!(
            engine.admission_snapshot().unwrap().request_identities_held,
            1
        );
        engine.cancel_request(RequestId(1)).unwrap();
        assert_eq!(engine.drain_cancelled().len(), 1);
        engine.retain_session(SessionId(7)).unwrap();
        engine.try_submit(make(2, 1, 1)).unwrap();
        for _ in 0..8 {
            engine.step(&mut |_| Ok(())).unwrap();
            if !engine.drain_finished().is_empty() {
                break;
            }
        }
        let position = engine.retained_session_position(SessionId(7)).unwrap();
        assert_eq!(position, 2);
        let retained = engine.capacity_snapshot();
        assert!(matches!(
            engine.try_submit(make(3, context - position - 1, 2)),
            Err(Error::Admission {
                source: ferrule_runtime::RuntimeAdmissionError::InvalidPosition { .. }
            })
        ));
        assert_eq!(
            engine.capacity_snapshot(),
            retained,
            "invalid turn must not release retained KV"
        );
        assert_eq!(
            engine.retained_session_position(SessionId(7)),
            Some(position)
        );
        engine
            .try_submit(make(3, context - position - 1, 1))
            .unwrap();
        engine.cancel_request(RequestId(3)).unwrap();
        engine.drain_cancelled();
        assert_eq!(
            engine.shutdown().unwrap(),
            InferenceShutdownProgress::Complete
        );
    }
}

#[test]
fn hermetic_cpu_factory_composition_generates_cancels_resets_and_shuts_down() {
    let fixture = fixture::Fixture::new();
    let descriptor = super::super::tests::descriptor(ModelFamily::Qwen35, "tiny-hybrid", 4);
    let mut options = super::super::tests::options();
    options.driver_config.ctx_size = 32;
    options.driver_config.enable_native_proposals = false;
    options.scheduler_config.max_batch_tokens = 4;
    options.scheduler_config.prefill_chunk_size = 2;
    let entry = resolve(&descriptor, BackendSelection::Cpu).unwrap();
    let request = configure_request(&descriptor, entry, options).unwrap();
    let mut engine = resident_f32_bound::build_resident_f32_bound(
        request,
        fixture.resources(),
        TokenizerHandle::load(&fixture.dir).unwrap(),
    )
    .unwrap();
    engine.retain_session(SessionId(7)).unwrap();
    let first = run(&mut engine, 1);
    assert_eq!(first.len(), 2);
    assert_eq!(first[0], 0, "HF oracle greedy token after [1, 3, 2]");
    assert!(engine.retained_session_position(SessionId(7)).unwrap() > 3);
    engine.reset_session(SessionId(7)).unwrap();
    assert_eq!(engine.retained_session_position(SessionId(7)), Some(0));
    assert_eq!(run(&mut engine, 2), first);
    engine.reset_session(SessionId(7)).unwrap();
    engine
        .try_submit(GenerateRequest {
            id: RequestId(3),
            session_id: Some(SessionId(7)),
            prompt_tokens: vec![1, 3, 2],
            max_new_tokens: 16,
            stop: vec![],
            ignore_eos: true,
        })
        .unwrap();
    engine.step(&mut |_| Ok(())).unwrap();
    engine.cancel_request(RequestId(3)).unwrap();
    assert_eq!(engine.drain_cancelled().len(), 1);
    engine.reset_session(SessionId(7)).unwrap();
    assert_eq!(run(&mut engine, 4), first);
    assert_eq!(
        engine.shutdown().unwrap(),
        InferenceShutdownProgress::Complete
    );
}
