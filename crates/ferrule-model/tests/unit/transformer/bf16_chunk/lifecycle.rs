use super::*;
use crate::transformer::{ExpertCacheLimits, NumericFp8Precision};
fn owner(ops: &Rc<CudaOperators>) -> CudaStandardDecoderOperators {
    CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
        Rc::clone(ops),
        &[],
        ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 1 << 20,
        }),
        4096,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
    )
    .unwrap()
}
fn layout() -> GatedDeltaNetLayout {
    GatedDeltaNetLayout {
        rows: 5,
        key_heads: 1,
        value_heads: 2,
        key_dim: 4,
        value_dim: 8,
    }
}
fn submit(owner: &mut CudaStandardDecoderOperators) -> Result<()> {
    let l = layout();
    let plan = owner.bf16_chunk_plan(l, 0)?;
    let expected_launches = plan.expect("multi-row chunk plan").launch_count();
    let qkv = owner.ops.zero_bf16_buffer(l.qkv_elements()?)?;
    let a = owner.ops.zero_bf16_buffer(l.gate_elements()?)?;
    let b = owner.ops.zero_bf16_buffer(l.gate_elements()?)?;
    let al = owner.ops.zero_bf16_buffer(l.value_heads)?;
    let dt = owner.ops.zero_bf16_buffer(l.value_heads)?;
    let mut state = owner.ops.zero_f32_buffer(l.state_elements()?)?;
    let mut output = owner.ops.zero_bf16_buffer(l.output_elements()?)?;
    for buffer in [&qkv, &a, &b, &al, &dt, &output] {
        owner.hold_bf16(buffer)?;
    }
    owner.hold_expert_buffer(&state)?;
    let before = owner.ops.counters();
    owner.submit_bf16_delta(
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
    let after = owner.ops.counters();
    assert_eq!(
        after.kernel_launches - before.kernel_launches,
        expected_launches
    );
    assert_eq!(
        after.device_allocation_attempts,
        before.device_allocation_attempts
    );
    eprintln!(
        "CHUNK_SUBMIT launches={} allocation_attempts={}",
        after.kernel_launches - before.kernel_launches,
        after.device_allocation_attempts - before.device_allocation_attempts
    );
    Ok(())
}
#[test]
#[ignore = "GPU owner only; real chunk submission with clean error and retained image scratch"]
fn chunk_clean_failure_retains_reusable_workspace_and_releases_operands() {
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let baseline = ops.allocator_metrics().live_requested_bytes;
    let mut owner = owner(&ops);
    let bytes = Bf16ChunkPlan::new(layout(), usize::MAX)
        .unwrap()
        .workspace_requirements()
        .bytes as usize;
    owner.prepare_bf16_chunk_workspace(bytes).unwrap();
    let pointer = owner
        .bf16_chunk_workspace
        .as_ref()
        .unwrap()
        .as_device_buffer()
        .cu_deviceptr();
    assert!(
        owner
            .synchronous::<()>(|this| {
                submit(this)?;
                Err(cuda_error("after chunk submission"))
            })
            .is_err()
    );
    assert!(!owner.needs_quarantine());
    assert!(owner.bf16_hold.is_empty() && owner.expert_hold.is_empty());
    owner.synchronous(submit).unwrap();
    assert_eq!(
        owner
            .bf16_chunk_workspace
            .as_ref()
            .unwrap()
            .as_device_buffer()
            .cu_deviceptr(),
        pointer
    );
    drop(owner);
    ops.trim_device_allocator().unwrap();
    let reclaimed = ops.allocator_metrics().live_requested_bytes;
    assert_eq!(reclaimed, baseline);
    eprintln!(
        "CHUNK_CLEAN scratch_bytes={bytes} pointer_reused=true baseline={baseline} after_drop_trim={reclaimed}"
    );
}
#[test]
#[ignore = "GPU owner only; lost completion proof retains scratch and operands after drop"]
fn chunk_unknown_retains_workspace_and_operands_and_blocks_reuse() {
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let baseline = ops.allocator_metrics().live_requested_bytes;
    let mut owner = owner(&ops);
    let bytes = Bf16ChunkPlan::new(layout(), usize::MAX)
        .unwrap()
        .workspace_requirements()
        .bytes as usize;
    owner.prepare_bf16_chunk_workspace(bytes).unwrap();
    owner.lose_next_consumer_proof = true;
    assert!(owner.synchronous(submit).is_err());
    assert!(owner.needs_quarantine());
    assert!(!owner.bf16_hold.is_empty() && !owner.expert_hold.is_empty());
    assert_eq!(owner.bf16_chunk_workspace_bytes(), bytes);
    assert!(owner.quiesce().is_err());
    let live = ops.allocator_metrics().live_requested_bytes;
    let l = layout();
    let state_output_bytes = l.state_elements().unwrap() * 4 + l.output_elements().unwrap() * 2;
    assert!(live >= baseline + bytes + state_output_bytes);
    drop(owner);
    ops.sync_stream().unwrap();
    ops.trim_device_allocator().unwrap();
    let retained = ops.allocator_metrics().live_requested_bytes;
    assert!(retained >= live);
    assert!(Rc::strong_count(&ops) >= 2);
    eprintln!(
        "CHUNK_UNKNOWN scratch_bytes={bytes} state_output_bytes={state_output_bytes} baseline={baseline} before_drop={live} after_drop_trim={retained} reuse_blocked=true owner_retained=true"
    );
}
#[test]
#[ignore = "GPU owner only; allocation failure before any GDN mutation is clean"]
fn chunk_allocation_failure_is_before_submission_and_retryable() {
    let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
    let baseline = ops.allocator_metrics().live_requested_bytes;
    let mut owner = owner(&ops);
    let bytes = Bf16ChunkPlan::new(layout(), usize::MAX)
        .unwrap()
        .workspace_requirements()
        .bytes as usize;
    ops.failpoints().arm_allocation();
    let before = ops.counters();
    assert!(owner.prepare_bf16_chunk_workspace(bytes).is_err());
    assert_eq!(ops.counters().kernel_launches, before.kernel_launches);
    assert!(!owner.needs_quarantine());
    assert_eq!(owner.bf16_chunk_workspace_bytes(), 0);
    owner.prepare_bf16_chunk_workspace(bytes).unwrap();
    assert_eq!(owner.bf16_chunk_workspace_bytes(), bytes);
    drop(owner);
    ops.trim_device_allocator().unwrap();
    let reclaimed = ops.allocator_metrics().live_requested_bytes;
    assert_eq!(reclaimed, baseline);
    eprintln!(
        "CHUNK_ALLOCATION scratch_bytes={bytes} failure_launches=0 retry=true baseline={baseline} after_drop_trim={reclaimed}"
    );
}

#[path = "m1_submission.rs"]
mod m1_submission;
