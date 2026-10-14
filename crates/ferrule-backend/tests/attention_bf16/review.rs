use super::*;
use crate::cuda::operator_owner::{CudaBf16Buffer, CudaF32Buffer};
use crate::cuda::providers::admission::{Ledger, ledger};
use crate::cuda::runtime::{GraphFault, OWNED_DOWNLOAD_RETAINED_BYTES};
fn g(m: u32, n: u32) -> Bf16AttentionGroup {
    Bf16AttentionGroup {
        row_start: 0,
        rows: m,
        sequence: 0,
        tokens: n,
    }
}
fn p() -> Bf16AttentionPlan {
    Bf16AttentionPlan::new(&[g(1, 8)], 4, 2, 8, 64 << 20, 16 << 20).unwrap()
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
    status: CudaI32Buffer,
    layout: PagedBf16CausalGqaLayout,
}
impl Buffers {
    fn new(op: &CudaOperators, m: usize, n: usize) -> Self {
        let pages = n.div_ceil(16);
        Self {
            q: op.zero_f32_buffer(m * 4 * 8).unwrap(),
            k: op
                .upload_bf16_words(&vec![0x3f80; pages * 16 * 2 * 8])
                .unwrap(),
            v: op
                .upload_bf16_words(&vec![0x4000; pages * 16 * 2 * 8])
                .unwrap(),
            ak: op.upload_bf16_words(&vec![0x3f80; m * 2 * 8]).unwrap(),
            av: op.upload_bf16_words(&vec![0x4000; m * 2 * 8]).unwrap(),
            slots: op
                .upload_i32_buffer(&(0..pages as i32).collect::<Vec<_>>())
                .unwrap(),
            offsets: op.upload_i32_buffer(&[0, pages as i32]).unwrap(),
            seq: op.upload_i32_buffer(&vec![0; m]).unwrap(),
            pos: op
                .upload_i32_buffer(&((n - m) as i32..n as i32).collect::<Vec<_>>())
                .unwrap(),
            len: op.upload_i32_buffer(&vec![n as i32; m]).unwrap(),
            out: op.upload_f32_buffer(&vec![19.; m * 4 * 8]).unwrap(),
            status: op.zero_i32_buffer(1).unwrap(),
            layout: PagedBf16CausalGqaLayout::packed(m, 1, 4, 2, 8, 16, 0, 1, pages, 1.).unwrap(),
        }
    }
    fn args(&mut self) -> PagedBf16GqaBuffers<'_> {
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
            status: &mut self.status,
        }
    }
}
#[test]
fn admission_family_keys_sum_charges_without_excluding_other_owners() {
    let mut l = Ledger::default();
    let c = Charge {
        storage_grant: 36 << 20,
        external: 16 << 20,
    };
    l.reserve((1, ProviderFamily::Attention), 0, c).unwrap();
    l.reserve((1, ProviderFamily::GemmEx), 0, c).unwrap();
    assert_eq!(l.devices[&0].external, 32 << 20);
    assert!(l.reserve((1, ProviderFamily::Attention), 0, c).is_err());
    l.reserve((2, ProviderFamily::Attention), 0, c).unwrap();
    l.reserve((3, ProviderFamily::Attention), 1, c).unwrap();
    l.release((1, ProviderFamily::Attention));
    assert!(l.owners.contains_key(&(1, ProviderFamily::GemmEx)));
    assert_eq!(l.devices[&1], c);
}
fn coexist(two_devices: bool) {
    let a = CudaOperators::new_on_device(0).unwrap();
    let b = CudaOperators::new_on_device(usize::from(two_devices)).unwrap();
    let pa = p();
    let x = a.prepare_bf16_attention(pa.clone()).unwrap();
    let y = b.prepare_bf16_attention(pa.clone()).unwrap();
    let gp = crate::cuda_plan::bf16_gemmex::Bf16GemmExPlan::new_resident(
        128,
        128,
        1,
        64 << 20,
        16 << 20,
    )
    .unwrap();
    let z = a
        .prepare_bf16_gemmex_views(&[gp], 64 << 20, 16 << 20)
        .unwrap();
    let ka = Arc::as_ptr(&a.stream_clone()) as usize;
    let kb = Arc::as_ptr(&b.stream_clone()) as usize;
    assert_eq!(
        ledger().owners[&(ka, ProviderFamily::Attention)].1.external,
        16 << 20
    );
    assert_eq!(
        ledger().owners[&(ka, ProviderFamily::GemmEx)].1.external,
        16 << 20
    );
    assert_eq!(
        ledger().owners[&(kb, ProviderFamily::Attention)].0,
        usize::from(two_devices)
    );
    assert!(a.prepare_bf16_attention(pa.clone()).is_err());
    assert!(
        a.prepare_bf16_attention(
            Bf16AttentionPlan::new(&[g(1, 8)], 4, 2, 8, 0, 0).unwrap_or_else(|_| pa.clone())
        )
        .is_err()
    );
    let before = ledger().devices[&0].external;
    drop(x);
    assert_eq!(ledger().devices[&0].external, before - (16 << 20));
    drop(z);
    drop(y);
    assert!(!ledger().owners.keys().any(|k| k.0 == ka || k.0 == kb));
    println!(
        "shared admission: coexist, same-owner two families, quota, exact charges, clean retirement; two_devices={two_devices}"
    );
}
#[test]
#[ignore = "owned GPU"]
fn review_shared_admission_same_device() {
    coexist(false)
}
#[test]
#[ignore = "owned two GPUs"]
fn review_shared_admission_two_devices() {
    coexist(true)
}
#[test]
#[ignore = "owned GPU; 128 decode views plus M5 reuse one owner"]
fn review_maxcap_changed_views_and_sticky_status() {
    let op = CudaOperators::new().unwrap();
    let mut cohorts = (1..=128).map(|n| vec![g(1, n)]).collect::<Vec<_>>();
    cohorts.push(vec![g(5, 128)]);
    let plan = Bf16AttentionPlan::with_views(&cohorts, 4, 2, 8, 64 << 20, 16 << 20).unwrap();
    let views = cohorts
        .iter()
        .map(|g| plan.view(g).unwrap())
        .collect::<Vec<_>>();
    let mut buffers = cohorts
        .iter()
        .map(|c| Buffers::new(&op, c[0].rows as usize, c[0].tokens as usize))
        .collect::<Vec<_>>();
    let mut owner = op.prepare_bf16_attention(plan.clone()).unwrap();
    let alloc = op.allocator_metrics();
    for _ in 0..2 {
        for (view, b) in views.iter().zip(buffers.iter_mut()) {
            owner.begin_invocation(&op, &mut b.status).unwrap();
            let l = b.layout;
            let before = op.allocator_metrics();
            op.reset_counters();
            owner.execute_view(&op, b.args(), l, view, false).unwrap();
            assert_eq!(op.allocator_metrics(), before);
            assert_eq!(op.counters().stream_wide_syncs, 0);
            assert_eq!(op.counters().device_allocation_attempts, 0);
            owner.finish_invocation(&op, &b.status).unwrap();
            let values = op.download_f32_buffer(&b.out).unwrap();
            assert!(
                values
                    .iter()
                    .all(|v| v.is_finite() && (*v - 2.).abs() < 0.016)
            );
        }
    }
    assert_eq!(op.allocator_metrics(), alloc);
    assert!(plan.view(&[g(1, 129)]).is_err());
    let bad = Bf16AttentionPlan::new(&[g(5, 127)], 4, 2, 8, 64 << 20, 16 << 20)
        .unwrap()
        .view(&[g(5, 127)])
        .unwrap();
    let b = &mut buffers[7];
    let layout = b.layout;
    op.reset_counters();
    assert!(
        owner
            .execute_view(&op, b.args(), layout, &bad, false)
            .is_err()
    );
    assert_eq!(op.counters(), Default::default());
    let mut badseq = op.upload_i32_buffer(&[-1]).unwrap();
    let original_out = op.download_f32_buffer(&b.out).unwrap();
    let original_k = op.download_bf16_buffer(&b.k).unwrap();
    std::mem::swap(&mut b.seq, &mut badseq);
    owner.begin_invocation(&op, &mut b.status).unwrap();
    owner
        .execute_view(&op, b.args(), layout, &views[7], true)
        .unwrap();
    assert!(owner.begin_invocation(&op, &mut b.status).is_err());
    std::mem::swap(&mut b.seq, &mut badseq);
    // A later valid layer cannot reset/erase the first error or write append/output.
    owner
        .execute_view(&op, b.args(), layout, &views[7], true)
        .unwrap();
    assert!(owner.finish_invocation(&op, &b.status).is_err());
    assert!(!owner.is_poisoned());
    assert_eq!(op.download_f32_buffer(&b.out).unwrap(), original_out);
    assert_eq!(op.download_bf16_buffer(&b.k).unwrap(), original_k);
    owner.begin_invocation(&op, &mut b.status).unwrap();
    owner
        .execute_view(&op, b.args(), layout, &views[7], false)
        .unwrap();
    owner.finish_invocation(&op, &b.status).unwrap();
    println!(
        "129 cohorts, 258 executes stable allocator; sticky error survives later layer; clear requires consumed status"
    );
}
fn download_fault(fault: GraphFault, output: bool) {
    let op = CudaOperators::new().unwrap();
    let mut b = Buffers::new(&op, 1, 8);
    let mut owner = op.prepare_bf16_attention(p()).unwrap();
    let l = b.layout;
    owner.begin_invocation(&op, &mut b.status).unwrap();
    owner.execute(&op, b.args(), l, false).unwrap();
    let before = op.allocator_metrics();
    let provider_bytes = owner.plan().storage_bytes();
    let retained = OWNED_DOWNLOAD_RETAINED_BYTES.with(|n| n.get());
    op.stream_clone().context().arm_graph_fault(fault);
    if output {
        assert!(op.download_f32_buffer(&b.out).is_err());
    } else {
        assert!(owner.finish_invocation(&op, &b.status).is_err());
    }
    assert!(owner.is_poisoned());
    assert!(op.stream_clone().context().graph_completion_unknown());
    assert!(owner.execute(&op, b.args(), l, false).is_err());
    assert!(owner.begin_invocation(&op, &mut b.status).is_err());
    let bytes = if output { 32 * 4 } else { 4 };
    assert_eq!(
        OWNED_DOWNLOAD_RETAINED_BYTES.with(|n| n.get()),
        retained + bytes
    );
    let key = (
        Arc::as_ptr(&op.stream_clone()) as usize,
        ProviderFamily::Attention,
    );
    drop(owner);
    drop(b);
    let after = op.allocator_metrics();
    assert!(
        after.live_requested_bytes >= provider_bytes + bytes,
        "provider and DMA source lease stay live"
    );
    assert!(
        after.live_granted_bytes + after.capture_retirement_bytes >= before.live_granted_bytes,
        "all other operands remain quarantined"
    );
    assert_eq!(
        after.driver_frees, before.driver_frees,
        "Unknown cannot free device storage"
    );
    assert_eq!(after.reserved_bytes, before.reserved_bytes);
    assert!(ledger().owners.contains_key(&key));
    println!(
        "owned download Unknown: host {bytes} bytes + device/provider retained; next submission blocked"
    );
}
#[test]
#[ignore = "isolated owned GPU"]
fn review_status_copy_unknown() {
    download_fault(GraphFault::DownloadCopy, false)
}
#[test]
#[ignore = "isolated owned GPU"]
fn review_status_fence_unknown() {
    download_fault(GraphFault::DownloadSynchronize, false)
}
#[test]
#[ignore = "isolated owned GPU"]
fn review_output_copy_unknown() {
    download_fault(GraphFault::DownloadCopy, true)
}
#[test]
#[ignore = "isolated owned GPU"]
fn review_output_fence_unknown() {
    download_fault(GraphFault::DownloadSynchronize, true)
}
#[test]
#[ignore = "owned GPU; preflight no DMA"]
fn review_download_preflight_is_not_unknown() {
    let op = CudaOperators::new().unwrap();
    let other = CudaOperators::new().unwrap();
    let y = op.upload_f32_buffer(&[1.; 4]).unwrap();
    let before = OWNED_DOWNLOAD_RETAINED_BYTES.with(|n| n.get());
    assert!(
        y.as_device_buffer()
            .to_host_vec(&other.stream_clone())
            .is_err()
    );
    assert!(
        y.as_device_buffer()
            .copy_to_host(&op.stream_clone(), &mut [0.; 3])
            .is_err()
    );
    assert!(!op.stream_clone().context().graph_completion_unknown());
    assert!(!other.stream_clone().context().graph_completion_unknown());
    assert_eq!(OWNED_DOWNLOAD_RETAINED_BYTES.with(|n| n.get()), before);
    assert_eq!(op.download_f32_buffer(&y).unwrap(), [1.; 4]);
}
#[test]
#[ignore = "owned GPU; legitimate unrelated allocations allowed"]
fn review_legitimate_invocation_allocations_allowed() {
    let op = CudaOperators::new().unwrap();
    let mut b = Buffers::new(&op, 1, 8);
    let mut owner = op.prepare_bf16_attention(p()).unwrap();
    let l = b.layout;
    owner.begin_invocation(&op, &mut b.status).unwrap();
    let extra = DeviceBuffer::<u8>::uninitialized_dedicated(&op.stream_clone(), 4 << 20).unwrap();
    owner.execute(&op, b.args(), l, false).unwrap();
    owner.finish_invocation(&op, &b.status).unwrap();
    assert!(!owner.is_poisoned());
    owner.begin_invocation(&op, &mut b.status).unwrap();
    owner.execute(&op, b.args(), l, false).unwrap();
    owner.finish_invocation(&op, &b.status).unwrap();
    drop(extra);
    println!(
        "legitimate 4MiB invocation allocation allowed; same storage/handle reused, no global-delta quarantine"
    );
}

#[test]
#[ignore = "owned GPU; malformed F32 query rejected before append writes"]
fn review_invalid_query_precedes_append() {
    let op = CudaOperators::new().unwrap();
    let mut b = Buffers::new(&op, 1, 8);
    b.q = op.upload_f32_buffer(&[1.0000001; 32]).unwrap();
    b.ak = op.upload_bf16_words(&[0x4080; 16]).unwrap();
    let mut owner = op.prepare_bf16_attention(p()).unwrap();
    let original = op.download_bf16_buffer(&b.k).unwrap();
    let l = b.layout;
    owner.begin_invocation(&op, &mut b.status).unwrap();
    owner.execute(&op, b.args(), l, true).unwrap();
    assert!(owner.finish_invocation(&op, &b.status).is_err());
    assert_eq!(op.download_bf16_buffer(&b.k).unwrap(), original);
    assert_eq!(op.download_f32_buffer(&b.out).unwrap(), [19.; 32]);
    assert!(!owner.is_poisoned());
}
