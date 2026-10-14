use super::*;
fn bound(dir: &Path, name: &str) -> BoundStateDict {
    let mut schema = StateDictSchema::builder();
    let module = ModulePath::new("router").unwrap();
    schema
        .register_with_role(
            ParameterSpec::new(
                ParameterId::new(1),
                module.clone(),
                ParameterDType::Bf16,
                vec![256, 2048],
                ParameterResidency::layer(0),
            )
            .unwrap(),
            TensorRole::RouterLogits,
        )
        .unwrap();
    let mut mapper = ExactNameMapper::default();
    mapper.insert(name, NameMapping::weight(module)).unwrap();
    StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(vec![CheckpointTensorSlice {
            name: name.into(),
            path: dir.join(format!("{name}.weight.bf16")),
            role: TensorRole::RouterLogits,
            offset: 0,
            bytes: 256 * 2048 * 2,
            shape: vec![256, 2048],
            dtype: CheckpointDType::Bf16,
        }])
        .unwrap()
}
fn exercise(unknown: bool) {
    let dir = PathBuf::from(std::env::var_os("FERRULE_RESIDENT_ROUTER_FIXTURE").unwrap());
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let baseline = ops.allocator_metrics().live_requested_bytes;
    for (name, m) in [
        ("hello0", 1),
        ("hello3", 1),
        ("capital0", 5),
        ("capital1", 5),
    ] {
        let bound = bound(&dir, name);
        let plans = [MIXED
            .resident_bf16_gemmex_plan(
                &TensorRole::RouterLogits,
                256,
                2048,
                m,
                usize::MAX,
                usize::MAX,
            )
            .unwrap()
            .unwrap()];
        let mut owner = owner(&ops, &bound, &plans);
        let c = Case {
            name: name.into(),
            m,
            n: 256,
            k: 2048,
            role: TensorRole::RouterLogits,
            expert: None,
        };
        let x = input(&owner, &dir, &c);
        let parameter = bound.get_by_id(ParameterId::new(1)).unwrap();
        let linear = StateDictMaterializer::new(2 << 20)
            .unwrap()
            .prepared_linear(parameter, TensorRole::RouterLogits)
            .unwrap();
        // Preparation/upload occurs before counting; invocation is native BF16.
        owner
            .synchronous(|this| this.bindings.linear(&this.ops, &linear))
            .unwrap();
        let result = owner.synchronous(|this| {
            let before = this.ops.counters();
            let y = this.bf16_linear_rows(&linear, &x, None)?;
            let after = this.ops.counters();
            assert_eq!(after.kernel_launches - before.kernel_launches, 1);
            assert_eq!(after.host_to_device_copies, before.host_to_device_copies);
            assert_eq!(compare(this, &dir, &c, &y), (0, m * 256));
            if unknown {
                this.lose_next_consumer_proof = true;
                return Err(cuda_error("resident router custody injection"));
            }
            let policy = crate::transformer::MoeRouterSpec::new(
                256, 8, crate::transformer::RouterScoreFunction::Softmax,
                crate::transformer::RouterSelection::TopK, true, 1.,
            ).unwrap();
            let OperatorProgress::Ready(routes) = this.bf16_router(&y, &policy)? else {
                panic!("resident router unsupported");
            };
            let original: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join(format!("{name}.routes.json"))).unwrap()).unwrap();
            let ids: Vec<usize> = serde_json::from_value(original["ids"].clone()).unwrap();
            assert_eq!(routes.expert_ids(), ids);
            let expected = std::fs::read(dir.join(format!("{name}.hf.weights"))).unwrap();
            let expected = expected.as_chunks::<4>().0.iter().copied().map(f32::from_le_bytes).collect::<Vec<_>>();
            let bad = routes.weights().iter().zip(&expected).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
            let error = routes.weights().iter().zip(&expected).map(|(a,b)|(a-b).abs()).fold(0.,f32::max);
            eprintln!("RESIDENT_ROUTE case={name} ids_unequal=0 weights_unequal={bad} words={} max_abs={error}",expected.len());
            assert_eq!(bad, 0, "Mixed full-softmax route weights: {name}");

            Ok(())
        });
        if unknown {
            assert!(result.is_err() && owner.needs_quarantine());
            assert!(!owner.bf16_hold.is_empty());
            assert!(!owner.expert_artifacts.is_empty());
            assert!(
                owner
                    .synchronous(|this| this.bf16_linear_rows(&linear, &x, None))
                    .is_err()
            );
            let live = ops.allocator_metrics().live_requested_bytes;
            drop(x);
            drop(owner);
            ops.sync_stream().unwrap();
            ops.trim_device_allocator().unwrap();
            assert!(ops.allocator_metrics().live_requested_bytes >= live);
            eprintln!("RESIDENT_MODEL_UNKNOWN retained={live} reuse_blocked=true");
            return;
        }
        result.unwrap();
        assert!(owner.bf16_hold.is_empty());
        drop(x);
        drop(owner);
        ops.trim_device_allocator().unwrap();
        assert_eq!(ops.allocator_metrics().live_requested_bytes, baseline);
    }
}
#[test]
#[ignore = "sole GPU owner; Mixed real resident router model dispatch, M1/M5, full output exact"]
fn mixed_resident_router_frozen_production_dispatch() {
    exercise(false)
}
#[test]
#[ignore = "isolated sole GPU child; actual resident projection then Unknown operand/weight/pool custody"]
fn mixed_resident_router_unknown_custody() {
    exercise(true)
}
