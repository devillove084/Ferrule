use super::super::admission::{admit_expert, expert_resident_bytes, seal_binding};
use super::*;
use ferrule_model::checkpoint::CheckpointDType;
use ferrule_model::nn::{ParameterDType, ParameterResidency, ParameterSpec};
use ferrule_model::transformer::{ExactNameMapper, NameMapping, StateDictBinder, StateDictSchema};

/// Rebind only expert metadata/payloads; model code and the original fixture are
/// untouched. Raw slices use the same public checkpoint binder as real loaders.
fn expert_storage(
    base: &BoundDecoderResources,
    dtype: CheckpointDType,
) -> (tempfile::TempDir, BoundDecoderResources) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("experts.bin");
    let mut payload = Vec::new();
    let mut slices = Vec::new();
    let mut schema = StateDictSchema::builder();
    let mut mapper = ExactNameMapper::new();
    for p in base.state_dict().parameters() {
        let mut spec = p.spec().clone();
        let mut weight = p.weight().slice().clone();
        if matches!(p.residency(), ParameterResidency::Expert { .. }) {
            let element_bytes = match dtype {
                CheckpointDType::F32 => 4,
                CheckpointDType::Bf16 => 2,
                CheckpointDType::F8E4M3 => 1,
                _ => unreachable!(),
            };
            let parameter_dtype = match dtype {
                CheckpointDType::F32 => ParameterDType::F32,
                CheckpointDType::Bf16 => ParameterDType::Bf16,
                CheckpointDType::F8E4M3 => ParameterDType::F8E4M3,
                _ => unreachable!(),
            };
            spec = ParameterSpec::new(
                p.id(),
                p.path().clone(),
                parameter_dtype,
                p.spec().shape().to_vec(),
                p.residency().clone(),
            )
            .unwrap();
            weight.path = path.clone();
            weight.offset = payload.len() as u64;
            weight.dtype = dtype.clone();
            weight.bytes = (weight.element_count().unwrap() * element_bytes) as u64;
            payload.resize(payload.len() + weight.bytes as usize, 0);
            if dtype == CheckpointDType::F8E4M3 {
                spec = spec
                    .with_required_scale(ParameterDType::F32, vec![1, 1])
                    .unwrap();
                let mut scale = weight.clone();
                scale.name = format!("{}.scale", weight.name);
                scale.offset = payload.len() as u64;
                scale.dtype = CheckpointDType::F32;
                scale.shape = vec![1, 1];
                scale.bytes = 4;
                payload.extend_from_slice(&1f32.to_le_bytes());
                mapper
                    .insert(&scale.name, NameMapping::scale(p.path().clone()))
                    .unwrap();
                slices.push(scale);
            }
        }
        schema.register_with_role(spec, p.role().clone()).unwrap();
        mapper
            .insert(&weight.name, NameMapping::weight(p.path().clone()))
            .unwrap();
        slices.push(weight);
    }
    std::fs::write(&path, payload).unwrap();
    let state = StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(slices)
        .unwrap();
    (
        dir,
        BoundDecoderResources::new(base.spec().clone(), Arc::new(state)).unwrap(),
    )
}

#[test]
fn native_f32_and_bf16_charge_both_actual_storages_before_expert_startup() {
    let fixture = Arc::new(fixture::Fixture::new());
    let base = fixture.resources();
    let group = layout(base.spec(), 2, 2)
        .expert_group(0, options().expert_limits)
        .unwrap();
    let f32_bytes = expert_resident_bytes(&base, &group, group.members[0]).unwrap();
    for dtype in [CheckpointDType::F32, CheckpointDType::Bf16] {
        let (_dir, resources) = expert_storage(&base, dtype.clone());
        let storage = resources.state_dict().parameters().iter().filter(|p| {
            matches!(p.residency(), ParameterResidency::Expert { layer, expert }
                if group.layers.contains(layer) && group.placement.owner(*layer, *expert) == Some(group.members[0]))
        }).map(|p| p.weight().slice().bytes as usize).sum::<usize>();
        let required = expert_resident_bytes(&resources, &group, group.members[0]).unwrap();
        assert_eq!(required, storage * 2);
        let materializer = ferrule_model::transformer::StateDictMaterializer::new(1 << 20).unwrap();
        let mut actual = 0;
        for p in resources.state_dict().parameters() {
            if let ParameterResidency::Expert { layer, expert } = p.residency()
                && group.layers.contains(layer)
                && group.placement.owner(*layer, *expert) == Some(group.members[0])
            {
                let prepared = ferrule_model::transformer::PreparedLinear::from_parameter(
                    materializer.expert_parameter(*layer, *expert, p).unwrap(),
                    p.role().clone(),
                )
                .unwrap();
                let original = &prepared.parameter().weight().unwrap().bytes;
                let native = &prepared.weight().unwrap().weight.bytes;
                assert_ne!(original.as_ptr(), native.as_ptr());
                actual += original.len() + native.len();
            }
        }
        assert_eq!(required, actual);
        assert_eq!(
            required,
            if dtype == CheckpointDType::F32 {
                f32_bytes
            } else {
                f32_bytes / 2
            }
        );
        assert!(admit_expert(&resources, &group, group.members[0], storage).is_err());
        assert!(admit_expert(&resources, &group, group.members[0], required - 1).is_err());
        admit_expert(&resources, &group, group.members[0], required).unwrap();
        // The formal factory must reject W <= limit < 2W before spawning EP.
        let starts = Arc::new(AtomicUsize::new(0));
        let count = starts.clone();
        let mut limits = options();
        limits.max_expert_weight_bytes = required - 1;
        let spec = resources.spec().clone();
        let result = PipelineParallelExecutor::new_hybrid_cpu_tensor(
            topology(2, 2),
            &spec,
            layout(&spec, 2, 2),
            config(),
            limits,
            move |_| {
                EXPERT_STARTUP.with_borrow_mut(|slot| {
                    *slot = Some(Arc::new(move |_| {
                        count.fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    }))
                });
                Ok(resources)
            },
        );
        let message = match result {
            Ok(_) => panic!("undercharged expert admitted"),
            Err(e) => e.to_string(),
        };
        assert!(message.contains("residency budget"), "{message}");
        assert_eq!(starts.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn cpu_numeric_fp8_is_rejected_by_factory_before_any_expert_materialization() {
    let fixture = fixture::Fixture::new();
    let base = fixture.resources();
    let (_dir, resources) = expert_storage(&base, CheckpointDType::F8E4M3);
    assert!(
        resources
            .state_dict()
            .parameters()
            .iter()
            .any(|p| p.numeric_fp8_encoding().is_some())
    );
    let spec = resources.spec().clone();
    let starts = Arc::new(AtomicUsize::new(0));
    let count = starts.clone();
    let result = PipelineParallelExecutor::new_hybrid_cpu_tensor(
        topology(2, 2),
        &spec,
        layout(&spec, 2, 2),
        config(),
        options(),
        move |_| {
            EXPERT_STARTUP.with_borrow_mut(|slot| {
                *slot = Some(Arc::new(move |_| {
                    count.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                }))
            });
            Ok(resources)
        },
    );
    let message = match result {
        Ok(_) => panic!("numeric FP8 CPU admitted"),
        Err(e) => e.to_string(),
    };
    assert!(
        message.contains("numeric FP8 is unsupported before materialization"),
        "{message}"
    );
    assert_eq!(starts.load(Ordering::Relaxed), 0);
}

#[test]
fn same_spec_different_shard_binding_fails_and_shuts_down_started_roots_and_ep() {
    let first = Arc::new(fixture::Fixture::new());
    let second = Arc::new(fixture::Fixture::new());
    let shard = second.dir.join("model.safetensors");
    let mut bytes = std::fs::read(&shard).unwrap();
    let last = bytes.len() - 4;
    bytes[last..].copy_from_slice(&9f32.to_le_bytes());
    std::fs::write(&shard, bytes).unwrap();
    let spec = first.resources().spec().clone();
    assert_eq!(second.resources().spec(), &spec);
    let owner_layout = layout(&spec, 2, 2);
    let probe = Arc::new(Probe::default());
    let observed = probe.clone();
    let result = build(
        topology(2, 2),
        &spec,
        owner_layout.clone(),
        config(),
        options(),
        move |rank| {
            Ok(if rank.global.get() == 3 {
                second.resources()
            } else {
                first.resources()
            })
        },
        move |rank, stage| {
            let index = rank.global.get() as usize;
            Ok(stage.map_program(|inner| Observed {
                inner,
                parallel: owner_layout.stages()[index / 2].clone(),
                lane: index % 2,
                probe: observed,
            }))
        },
    );
    let message = match result {
        Ok(_) => panic!("mixed checkpoint admitted"),
        Err(e) => e.to_string(),
    };
    assert!(
        message.contains("checkpoint binding differs from startup seal"),
        "{message}"
    );
    assert_eq!(probe.stopped.load(Ordering::Relaxed), 3);
    assert_eq!(probe.expert_shutdowns.load(Ordering::Relaxed), 1);
}

#[test]
fn checkpoint_seal_compares_parameter_mapping_not_just_current_source_set() {
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let seal = Mutex::new(None);
    seal_binding(&seal, &resources).unwrap();
    seal_binding(&seal, &fixture.resources()).unwrap();
    let mut mapper = ExactNameMapper::new();
    let mut slices = Vec::new();
    let gate = resources
        .experts()
        .require_shape(1, 0, ferrule_model::TensorRole::RoutedExpertGate, &[12, 8])
        .unwrap();
    let up = resources
        .experts()
        .require_shape(1, 0, ferrule_model::TensorRole::RoutedExpertUp, &[12, 8])
        .unwrap();
    for p in resources.state_dict().parameters() {
        let target = if p.id() == gate.id() {
            up.path()
        } else if p.id() == up.id() {
            gate.path()
        } else {
            p.path()
        };
        mapper
            .insert(
                &p.weight().slice().name,
                NameMapping::weight(target.clone()),
            )
            .unwrap();
        slices.push(p.weight().slice().clone());
    }
    let state = StateDictBinder::new(resources.state_dict().schema(), &mapper)
        .bind_slices(slices)
        .unwrap();
    let remapped = BoundDecoderResources::new(resources.spec().clone(), Arc::new(state)).unwrap();
    assert!(remapped.state_dict().validate_source_identities());
    assert!(seal_binding(&seal, &remapped).is_err());
}

#[test]
fn shared_local_expert_is_in_owner_budget_not_external_native_ledger() {
    let fixture = fixture::Fixture::new();
    let resources = fixture.resources();
    let layout = layout(resources.spec(), 2, 2);
    let plan = &layout.stages()[0];
    let segment = LayerSegmentPlan::new(4, 0..2, true, false).unwrap();
    let local = plan.rank_layout(0).unwrap();
    let weights = local.weight_budget(&resources, &segment).unwrap();
    let shared = resources
        .state_dict()
        .parameters()
        .iter()
        .filter(|p| {
            matches!(p.residency(), ParameterResidency::Layer { layer: 1 })
                && matches!(
                    p.role(),
                    ferrule_model::TensorRole::SharedExpertGate
                        | ferrule_model::TensorRole::SharedExpertUp
                        | ferrule_model::TensorRole::SharedExpertDown
                )
        })
        .map(|p| p.weight().slice().bytes as usize / 2)
        .sum::<usize>();
    assert!(shared > 0);
    let cfg = config();
    let total = weights.resident_bytes
        + local.state_bytes().unwrap() * (cfg.session_capacity + 1)
        + local
            .kv_planes(cfg.page_size, cfg.max_positions)
            .unwrap()
            .checked_page_bytes()
            .unwrap()
            * cfg
                .physical_pages(PhysicalKvLayout::hybrid(ModelExecutionBackend::Cpu))
                .unwrap();
    let mut limits = options();
    limits.max_owner_resident_bytes = total;
    super::super::admission::admit_owner(&resources, &segment, plan, 0, cfg, &limits).unwrap();
    limits.max_owner_resident_bytes -= shared;
    assert!(
        super::super::admission::admit_owner(&resources, &segment, plan, 0, cfg, &limits).is_err()
    );
}

#[test]
fn checkpoint_seal_includes_paired_scale_mapping_and_snapshot() {
    let fixture = fixture::Fixture::new();
    let (_dir, resources) = expert_storage(&fixture.resources(), CheckpointDType::F8E4M3);
    let seal = Mutex::new(None);
    seal_binding(&seal, &resources).unwrap();
    let scaled = resources
        .state_dict()
        .parameters()
        .iter()
        .filter(|p| p.scale().is_some())
        .collect::<Vec<_>>();
    let mut mapper = ExactNameMapper::new();
    let mut slices = Vec::new();
    for p in resources.state_dict().parameters() {
        mapper
            .insert(
                &p.weight().slice().name,
                NameMapping::weight(p.path().clone()),
            )
            .unwrap();
        slices.push(p.weight().slice().clone());
        if let Some(scale) = p.scale() {
            let target = if p.id() == scaled[0].id() {
                scaled[1].path()
            } else if p.id() == scaled[1].id() {
                scaled[0].path()
            } else {
                p.path()
            };
            mapper
                .insert(&scale.slice().name, NameMapping::scale(target.clone()))
                .unwrap();
            slices.push(scale.slice().clone());
        }
    }
    let state = StateDictBinder::new(resources.state_dict().schema(), &mapper)
        .bind_slices(slices.clone())
        .unwrap();
    let remapped = BoundDecoderResources::new(resources.spec().clone(), Arc::new(state)).unwrap();
    assert!(remapped.state_dict().validate_source_identities());
    assert!(seal_binding(&seal, &remapped).is_err());
    // Restore the original mapping so this rejection isolates snapshot equality.
    let mut mapper = ExactNameMapper::new();
    for p in resources.state_dict().parameters() {
        mapper
            .insert(
                &p.weight().slice().name,
                NameMapping::weight(p.path().clone()),
            )
            .unwrap();
        if let Some(scale) = p.scale() {
            mapper
                .insert(&scale.slice().name, NameMapping::scale(p.path().clone()))
                .unwrap();
        }
    }
    // Rebinding a freshly changed source must not self-certify a new snapshot.
    let source = &scaled[0].scale().unwrap().slice().path;
    let mut bytes = std::fs::read(source).unwrap();
    bytes.push(0);
    std::fs::write(source, bytes).unwrap();
    let state = StateDictBinder::new(resources.state_dict().schema(), &mapper)
        .bind_slices(slices)
        .unwrap();
    let fresh = BoundDecoderResources::new(resources.spec().clone(), Arc::new(state)).unwrap();
    assert!(fresh.state_dict().validate_source_identities());
    assert!(seal_binding(&seal, &fresh).is_err());
}
