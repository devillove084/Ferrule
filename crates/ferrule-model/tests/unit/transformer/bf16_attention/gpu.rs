use super::*;
use crate::transformer::{ExpertCacheLimits, NumericFp8Precision};
use ferrule_backend::cuda::operators::attention::{PagedBf16CausalGqaLayout, PagedBf16GqaBuffers};
use ferrule_backend::cuda::operators::linear::CudaBf16Buffer;
use ferrule_backend::cuda_plan::bf16_attention::{Bf16AttentionGroup, Bf16AttentionShape};
use std::path::Path;
fn owner(ops: &Rc<CudaOperators>) -> CudaStandardDecoderOperators {
    let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
        ops.clone(),
        &[],
        ExpertCachePolicy::Bounded(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: 1 << 20,
        }),
        4096,
        NumericFp8Precision::Bf16ModelMixedF32Accumulate,
    )
    .unwrap();
    let shapes = (1..=128)
        .map(|tokens| Bf16AttentionShape { rows: 1, tokens })
        .collect::<Vec<_>>();
    owner
        .prepare_bf16_attention(Some(
            Bf16AttentionPlan::from_shapes(&shapes, 1, 1, 16, 2, 256, 64 << 20, 16 << 20).unwrap(),
        ))
        .unwrap();
    owner
}
fn f32s(p: impl AsRef<Path>) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect()
}
fn words(p: impl AsRef<Path>) -> Vec<u16> {
    std::fs::read(p)
        .unwrap()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|x| u16::from_le_bytes(*x))
        .collect()
}
struct Buffers {
    q: CudaF32Buffer,
    k: CudaBf16Buffer,
    v: CudaBf16Buffer,
    ak: CudaBf16Buffer,
    av: CudaBf16Buffer,
    slots: CudaI32Buffer,
    offsets: CudaI32Buffer,
    seq: CudaI32Buffer,
    pos: CudaI32Buffer,
    len: CudaI32Buffer,
    out: CudaF32Buffer,
    layout: PagedBf16CausalGqaLayout,
}
impl Buffers {
    fn new(op: &CudaOperators) -> Self {
        Self {
            q: op.zero_f32_buffer(4096).unwrap(),
            k: op.upload_bf16_words(&vec![0x3f80; 128 * 512]).unwrap(),
            v: op.upload_bf16_words(&vec![0x4000; 128 * 512]).unwrap(),
            ak: op.zero_bf16_buffer(512).unwrap(),
            av: op.zero_bf16_buffer(512).unwrap(),
            slots: op.upload_i32_buffer(&(0..8).collect::<Vec<_>>()).unwrap(),
            offsets: op.upload_i32_buffer(&[0, 8]).unwrap(),
            seq: op.upload_i32_buffer(&[0]).unwrap(),
            pos: op.upload_i32_buffer(&[0]).unwrap(),
            len: op.upload_i32_buffer(&[1]).unwrap(),
            out: op.zero_f32_buffer(4096).unwrap(),
            layout: PagedBf16CausalGqaLayout::packed(1, 1, 16, 2, 256, 16, 0, 1, 8, 0.0625)
                .unwrap(),
        }
    }
    fn submit(&mut self, owner: &mut CudaStandardDecoderOperators, tokens: u32) {
        let op = &owner.ops;
        self.pos
            .as_device_buffer()
            .copy_from_host(&op.stream_clone(), &[tokens as i32 - 1])
            .unwrap();
        self.len
            .as_device_buffer()
            .copy_from_host(&op.stream_clone(), &[tokens as i32])
            .unwrap();
        let mut a = owner.bf16_attention.as_ref().unwrap().borrow_mut();
        let view = a
            .provider
            .plan()
            .view(&[Bf16AttentionGroup {
                row_start: 0,
                rows: 1,
                sequence: 0,
                tokens,
            }])
            .unwrap();
        let PreparedAttention {
            provider, status, ..
        } = &mut *a;
        let before = op.allocator_metrics();
        let counts = op.counters();
        let sync = op.stream_wide_sync_attempts();
        provider
            .execute_view(
                op,
                PagedBf16GqaBuffers {
                    query: &self.q,
                    append_key: &self.ak,
                    append_value: &self.av,
                    key_cache: &mut self.k,
                    value_cache: &mut self.v,
                    block_slots: &self.slots,
                    block_offsets: &self.offsets,
                    row_sequence_ids: &self.seq,
                    row_positions: &self.pos,
                    row_kv_lens: &self.len,
                    output: &mut self.out,
                    status,
                },
                self.layout,
                &view,
                false,
            )
            .unwrap();
        assert_eq!(before, op.allocator_metrics());
        assert_eq!(
            counts.device_allocation_attempts,
            op.counters().device_allocation_attempts
        );
        assert_eq!(sync, op.stream_wide_sync_attempts());
    }
}
#[test]
#[ignore = "sole GPU; actual raw/gated HF4096 plus max-cap changed views reuse"]
fn mixed_attention_frozen_and_growing_views_exact() {
    let op = Rc::new(CudaOperators::new().unwrap());
    let baseline = op.allocator_metrics().live_requested_bytes;
    let mut owner = owner(&op);
    let mut b = Buffers::new(&op);
    let identity = Rc::as_ptr(owner.bf16_attention.as_ref().unwrap());
    for tokens in 1..=128 {
        let tx = ExecutionTransactionId::new(tokens as u64).unwrap();
        owner.begin_bf16_attention(tx).unwrap();
        b.submit(&mut owner, tokens);
        owner.finish_bf16_attention(tx, false).unwrap();
        // Uniform F32 probabilities are materialized as BF16 before PV.
        let probability = half::bf16::from_f32(1.0 / tokens as f32).to_f32();
        let expected = half::bf16::from_f32(2.0 * tokens as f32 * probability).to_f32();
        assert!(
            op.download_f32_buffer(&b.out)
                .unwrap()
                .iter()
                .all(|v| *v == expected),
            "KV={tokens}, expected={expected}"
        );
        assert_eq!(identity, Rc::as_ptr(owner.bf16_attention.as_ref().unwrap()));
    }
    let root = std::path::PathBuf::from(std::env::var_os("FERRULE_ATTENTION_FIXTURE").unwrap())
        .join("actual-read");
    b.q = op.upload_f32_buffer(&f32s(root.join("query.f32"))).unwrap();
    b.k = op
        .upload_bf16_words(&words(root.join("cache-key.bf16")))
        .unwrap();
    b.v = op
        .upload_bf16_words(&words(root.join("cache-value.bf16")))
        .unwrap();
    let slots = std::fs::read(root.join("slots.i32"))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| i32::from_le_bytes(*x))
        .collect::<Vec<_>>();
    b.slots = op.upload_i32_buffer(&slots).unwrap();
    b.offsets = op.upload_i32_buffer(&[0, 1]).unwrap();
    b.layout = PagedBf16CausalGqaLayout::packed(1, 1, 16, 2, 256, 16, 0, 1, 2, 0.0625).unwrap();
    let tx = ExecutionTransactionId::new(200).unwrap();
    owner.begin_bf16_attention(tx).unwrap();
    b.submit(&mut owner, 11);
    owner.finish_bf16_attention(tx, false).unwrap();
    let exact = |got: Vec<f32>, expected: Vec<f32>| {
        let unequal = got
            .iter()
            .zip(&expected)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(got.len(), expected.len());
        assert_eq!(unequal, 0);
        unequal
    };
    let raw = exact(
        op.download_f32_buffer(&b.out).unwrap(),
        f32s(root.join("hf-raw.f32")),
    );
    let shape = RowsShape::new(1, 4096).unwrap();
    let rows = super::super::bf16::round_rows(&op, shape, None, &b.out).unwrap();
    let gate = super::super::bf16::rows_bf16(
        shape,
        None,
        op.upload_bf16_words(&words(root.join("gate.bf16")))
            .unwrap(),
    )
    .unwrap();
    let OperatorProgress::Ready(gated) = owner.bf16_gate(rows, &gate, false).unwrap() else {
        panic!("pending")
    };
    let got = op
        .bf16_to_f32(gated.cuda().unwrap().bf16_buffer().unwrap())
        .unwrap();
    let gated_count = exact(
        op.download_f32_buffer(&got).unwrap(),
        f32s(root.join("hf-gated.f32")),
    );
    println!(
        "MIXED_ATTENTION raw={raw}/4096 gated={gated_count}/4096 growing_views=128 provider_allocations_per_execute=0"
    );
    drop((gate, gated, got, b, owner));
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
}
#[test]
#[ignore = "sole GPU; shared sticky status rejects publication, abort consumes then reset reusable"]
fn mixed_attention_sticky_status_abort_reset() {
    let op = Rc::new(CudaOperators::new().unwrap());
    let mut owner = owner(&op);
    let mut b = Buffers::new(&op);
    let tx = ExecutionTransactionId::new(1).unwrap();
    owner.begin_bf16_attention(tx).unwrap();
    assert!(owner.begin_bf16_attention(tx).is_err());
    b.q = op.upload_f32_buffer(&vec![1.0000001; 4096]).unwrap();
    b.submit(&mut owner, 1);
    b.q = op.zero_f32_buffer(4096).unwrap();
    b.submit(&mut owner, 1);
    assert!(owner.finish_bf16_attention(tx, false).is_err());
    assert!(!owner.needs_quarantine());
    owner.finish_bf16_attention(tx, true).unwrap();
    assert!(
        op.download_f32_buffer(&b.out)
            .unwrap()
            .iter()
            .all(|x| *x == 0.)
    );
    let tx = ExecutionTransactionId::new(2).unwrap();
    owner.begin_bf16_attention(tx).unwrap();
    b.submit(&mut owner, 2);
    owner.finish_bf16_attention(tx, false).unwrap();
    assert!(
        op.download_f32_buffer(&b.out)
            .unwrap()
            .iter()
            .all(|x| *x == 2.)
    );
}
#[test]
#[ignore = "sole GPU; injected lost model consumer proof retains actual attention allocation/status/operands"]
fn mixed_attention_unknown_retains_actual_custody() {
    let op = Rc::new(CudaOperators::new().unwrap());
    let mut owner = owner(&op);
    let mut b = Buffers::new(&op);
    let tx = ExecutionTransactionId::new(1).unwrap();
    owner.begin_bf16_attention(tx).unwrap();
    owner.lose_next_consumer_proof = true;
    assert!(
        owner
            .synchronous(|this| {
                this.hold_expert_buffer(&b.q)?;
                this.hold_expert_buffer(&b.out)?;
                this.hold_bf16(&b.k)?;
                this.hold_bf16(&b.v)?;
                b.submit(this, 1);
                Ok(())
            })
            .is_err()
    );
    assert!(owner.needs_quarantine());
    assert!(
        owner
            .begin_bf16_attention(ExecutionTransactionId::new(2).unwrap())
            .is_err()
    );
    assert!(owner.quiesce().is_err());
    let held = op.allocator_metrics().live_requested_bytes;
    drop(owner);
    drop(b);
    op.sync_stream().unwrap();
    op.trim_device_allocator().unwrap();
    assert!(op.allocator_metrics().live_requested_bytes >= (34 << 20) + 4096 * 8 + 128 * 512 * 4);
    println!(
        "MIXED_ATTENTION_UNKNOWN before_drop={held} retained={} reuse_blocked=true",
        op.allocator_metrics().live_requested_bytes
    );
}
