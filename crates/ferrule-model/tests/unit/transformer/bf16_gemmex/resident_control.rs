use super::*;
pub(super) fn bound(dir: &Path, name: &str, role: TensorRole, n: usize) -> BoundStateDict {
    let mut schema = StateDictSchema::builder();
    let module = ModulePath::new("router").unwrap();
    schema
        .register_with_role(
            ParameterSpec::new(
                ParameterId::new(1),
                module.clone(),
                ParameterDType::Bf16,
                vec![n, 2048],
                ParameterResidency::layer(0),
            )
            .unwrap(),
            role.clone(),
        )
        .unwrap();
    let mut mapper = ExactNameMapper::default();
    mapper.insert(name, NameMapping::weight(module)).unwrap();
    StateDictBinder::new(&schema.build().unwrap(), &mapper)
        .bind_slices(vec![CheckpointTensorSlice {
            name: name.into(),
            path: dir.join(format!("{name}.weight.bf16")),
            role: role.clone(),
            offset: 0,
            bytes: (n * 2048 * 2) as u64,
            shape: vec![n, 2048],
            dtype: CheckpointDType::Bf16,
        }])
        .unwrap()
}
fn exercise(unknown: bool) {
    let dir = PathBuf::from(std::env::var_os("FERRULE_RESIDENT_CONTROL_FIXTURE").unwrap());
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let baseline = ops.allocator_metrics().live_requested_bytes;
    let text = std::fs::read_to_string(dir.join("cases.txt")).unwrap();
    for line in text.lines() {
        let parts: Vec<_> = line.split_whitespace().collect();
        let name = parts[0];
        let m = parts[1].parse().unwrap();
        let role = if name.ends_with('a') {
            TensorRole::LinearAttentionA
        } else {
            TensorRole::LinearAttentionBeta
        };
        let bound = bound(&dir, name, role.clone(), 32);
        let plans = [MIXED
            .resident_bf16_gemmex_plan(&role, 32, 2048, m, usize::MAX, usize::MAX)
            .unwrap()
            .unwrap()];
        let mut owner = owner(&ops, &bound, &plans);
        let c = Case {
            name: name.into(),
            m,
            n: 32,
            k: 2048,
            role: role.clone(),
            expert: None,
        };
        let x = input(&owner, &dir, &c);
        let parameter = bound.get_by_id(ParameterId::new(1)).unwrap();
        let linear = StateDictMaterializer::new(2 << 20)
            .unwrap()
            .prepared_linear(parameter, role.clone())
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
            assert_eq!(compare(this, &dir, &c, &y), (0, m * 32));
            if unknown {
                this.lose_next_consumer_proof = true;
                return Err(cuda_error("resident router custody injection"));
            }
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
#[ignore = "sole GPU owner; Mixed real resident A/B model dispatch, M1/M5, full output exact"]
fn mixed_resident_control_frozen_production_dispatch() {
    exercise(false)
}
#[test]
#[ignore = "isolated sole GPU child; actual resident projection then Unknown operand/weight/pool custody"]
fn mixed_resident_control_unknown_custody() {
    exercise(true)
}
