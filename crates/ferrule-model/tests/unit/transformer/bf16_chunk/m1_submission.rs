//! Model-owned plan/submission regression. Run only under pinned GPU supervision.
use super::*;

#[test]
#[ignore = "sole GPU owner; original layer9 actual operands/expected; explicit FERRULE_BF16_M1_FIXTURE"]
fn mixed_m1_actual_prefill_and_cached_schedule_reuse_one_workspace() {
    let dir = std::path::PathBuf::from(std::env::var_os("FERRULE_BF16_M1_FIXTURE").unwrap());
    let words = |name: &str| {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        assert!(bytes.len().is_multiple_of(2));
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_le_bytes)
            .collect::<Vec<_>>()
    };
    let floats = |name: &str| {
        let bytes = std::fs::read(dir.join(name)).unwrap();
        assert!(bytes.len().is_multiple_of(4));
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect::<Vec<_>>()
    };
    let l = GatedDeltaNetLayout {
        rows: 1,
        key_heads: 16,
        value_heads: 32,
        key_dim: 128,
        value_dim: 128,
    };
    let expected = words("expected.bf16");
    let chronological = words("actual.bf16");
    assert_eq!(expected.len(), 4096);
    assert_eq!(chronological.len(), 4096);
    assert_eq!(
        expected
            .iter()
            .zip(&chronological)
            .filter(|(a, b)| a != b)
            .count(),
        2
    );
    let expected_state = floats("expected-state.f32");
    let initial = floats("state.f32");
    assert_eq!(expected_state.len(), 524288);
    assert_eq!(initial.len(), 524288);
    assert!(initial.iter().all(|v| v.to_bits() == 0));
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let baseline = ops.allocator_metrics().live_requested_bytes;
    let mut owner = owner(&ops);
    owner.prepare_bf16_chunk_workspace(17_850_368).unwrap();
    let pointer = owner
        .bf16_chunk_workspace
        .as_ref()
        .unwrap()
        .as_device_buffer()
        .cu_deviceptr();
    let qkv = ops.upload_bf16_words(&words("qkv.bf16")).unwrap();
    let a = ops.upload_bf16_words(&words("a.bf16")).unwrap();
    let b = ops.upload_bf16_words(&words("b.bf16")).unwrap();
    let al = ops.upload_bf16_words(&words("a_log.bf16")).unwrap();
    let dt = ops.upload_bf16_words(&words("dt_bias.bf16")).unwrap();
    // Position is varied deliberately on identical frozen operands to isolate
    // the numerical schedule. This is not a claimed real cached-decode fixture.
    for position in [0, 1, 0] {
        let mut state = ops.upload_f32_buffer(&initial).unwrap();
        let mut output = ops.zero_bf16_buffer(4096).unwrap();
        owner
            .synchronous(|this| {
                let plan = this.bf16_chunk_plan(l, position)?;
                assert_eq!(plan.is_some(), position == 0);
                for buffer in [&qkv, &a, &b, &al, &dt, &output] {
                    this.hold_bf16(buffer)?;
                }
                this.hold_expert_buffer(&state)?;
                let before = ops.counters();
                this.submit_bf16_delta(
                    Bf16GatedDeltaNetBuffers {
                        qkv: &qkv,
                        a: &a,
                        b: &b,
                        a_log: &al,
                        dt_bias: &dt,
                        state: &mut state,
                        output: &mut output,
                    },
                    l,
                    plan,
                )?;
                let after = ops.counters();
                assert_eq!(
                    after.kernel_launches - before.kernel_launches,
                    if position == 0 { 15 } else { 1 }
                );
                assert_eq!(
                    after.device_allocation_attempts,
                    before.device_allocation_attempts
                );
                Ok(())
            })
            .unwrap();
        let wanted = if position == 0 {
            &expected
        } else {
            &chronological
        };
        let actual = ops.download_bf16_buffer(&output).unwrap();
        assert!(
            actual
                .iter()
                .zip(wanted)
                .all(|(a, b)| a.to_bits() == u32::from(*b) << 16)
        );
        let actual_state = ops.download_f32_buffer(&state).unwrap();
        assert!(
            actual_state
                .iter()
                .zip(&expected_state)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert_eq!(
            owner
                .bf16_chunk_workspace
                .as_ref()
                .unwrap()
                .as_device_buffer()
                .cu_deviceptr(),
            pointer
        );
    }
    assert!(!owner.needs_quarantine());
    drop((owner, qkv, a, b, al, dt));
    ops.trim_device_allocator().unwrap();
    assert_eq!(ops.allocator_metrics().live_requested_bytes, baseline);
}
