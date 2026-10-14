use super::*;
#[test]
#[ignore = "sole GPU owner; original HF shared output gate N1 M1/M5 through Mixed dispatch"]
fn mixed_resident_scalar_frozen_dispatch() {
    let dir = PathBuf::from(std::env::var_os("FERRULE_RESIDENT_SCALAR_FIXTURE").unwrap());
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let baseline = ops.allocator_metrics().live_requested_bytes;
    let text = std::fs::read_to_string(dir.join("actual-cases.txt")).unwrap();
    for line in text.lines() {
        let parts: Vec<_> = line.split_whitespace().collect();
        let name = parts[0];
        let m = parts[1].parse().unwrap();
        let role = TensorRole::SharedExpertOutputGate;
        let bound = super::resident_control::bound(&dir, name, role.clone(), 1);
        let plans = [MIXED
            .resident_bf16_gemmex_plan(&role, 1, 2048, m, usize::MAX, usize::MAX)
            .unwrap()
            .unwrap()];
        let mut owner = owner(&ops, &bound, &plans);
        let c = Case {
            name: name.into(),
            m,
            n: 1,
            k: 2048,
            role: role.clone(),
            expert: None,
        };
        let x = input(&owner, &dir, &c);
        let parameter = bound.get_by_id(ParameterId::new(1)).unwrap();
        let linear = StateDictMaterializer::new(2 << 20)
            .unwrap()
            .prepared_linear(parameter, role)
            .unwrap();
        owner
            .synchronous(|this| this.bindings.linear(&this.ops, &linear))
            .unwrap();
        owner
            .synchronous(|this| {
                let before = this.ops.counters();
                let y = this.bf16_linear_rows(&linear, &x, None)?;
                let after = this.ops.counters();
                assert_eq!(after.kernel_launches - before.kernel_launches, 1);
                assert_eq!(after.host_to_device_copies, before.host_to_device_copies);
                assert_eq!(compare(this, &dir, &c, &y), (0, m));
                Ok(())
            })
            .unwrap();
        drop(x);
        drop(owner);
        ops.trim_device_allocator().unwrap();
        assert_eq!(ops.allocator_metrics().live_requested_bytes, baseline);
    }
}
