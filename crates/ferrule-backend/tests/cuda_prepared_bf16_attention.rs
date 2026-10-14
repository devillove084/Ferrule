#![cfg(feature = "cuda")]
//! Bounded backend gates only. Launch with CUBLAS_WORKSPACE_CONFIG=:0:0.
use ferrule_backend::cuda::operator_owner::{
    CudaBf16Buffer, CudaF32Buffer, CudaI32Buffer, CudaOperators,
};
use ferrule_backend::cuda::operators::attention::prepared_bf16::{
    Bf16AttentionGroup, Bf16AttentionPlan,
};
use ferrule_backend::cuda::operators::attention::{PagedBf16CausalGqaLayout, PagedBf16GqaBuffers};
use std::path::{Path, PathBuf};
fn path() -> PathBuf {
    PathBuf::from(
        std::env::var_os("FERRULE_ATTENTION_FIXTURE").expect("prepared fixture directory"),
    )
}
fn bytes<const N: usize>(path: impl AsRef<Path>) -> Vec<[u8; N]> {
    let b = std::fs::read(path).unwrap();
    let (chunks, tail) = b.as_chunks::<N>();
    assert!(tail.is_empty(), "fixture word extent");
    chunks.to_vec()
}
fn f32s(path: impl AsRef<Path>) -> Vec<f32> {
    bytes::<4>(path)
        .into_iter()
        .map(f32::from_le_bytes)
        .collect()
}
fn words(path: impl AsRef<Path>) -> Vec<u16> {
    bytes::<2>(path)
        .into_iter()
        .map(u16::from_le_bytes)
        .collect()
}
fn ints(path: impl AsRef<Path>) -> Vec<i32> {
    bytes::<4>(path)
        .into_iter()
        .map(i32::from_le_bytes)
        .collect()
}
struct Fixture {
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
    expected: Vec<f32>,
    groups: Vec<Bf16AttentionGroup>,
}
impl Fixture {
    fn one(op: &CudaOperators, d: &Path) -> Self {
        let shape = std::fs::read_to_string(d.join("shape.txt")).unwrap();
        let shape = shape
            .split_whitespace()
            .map(|n| n.parse::<usize>().unwrap())
            .collect::<Vec<_>>();
        let [rows, heads, kh, dim, n, pages]: [usize; 6] = shape.try_into().unwrap();
        Self {
            q: op.upload_f32_buffer(&f32s(d.join("query.f32"))).unwrap(),
            k: op
                .upload_bf16_words(&words(d.join("cache-key.bf16")))
                .unwrap(),
            v: op
                .upload_bf16_words(&words(d.join("cache-value.bf16")))
                .unwrap(),
            ak: op
                .upload_bf16_words(&words(d.join("append-key.bf16")))
                .unwrap(),
            av: op
                .upload_bf16_words(&words(d.join("append-value.bf16")))
                .unwrap(),
            slots: op.upload_i32_buffer(&ints(d.join("slots.i32"))).unwrap(),
            offsets: op.upload_i32_buffer(&[0, pages as i32]).unwrap(),
            seq: op.upload_i32_buffer(&vec![0; rows]).unwrap(),
            pos: op
                .upload_i32_buffer(&ints(d.join("positions.i32")))
                .unwrap(),
            len: op.upload_i32_buffer(&vec![n as i32; rows]).unwrap(),
            out: op.zero_f32_buffer(rows * heads * dim).unwrap(),
            status: op.zero_i32_buffer(1).unwrap(),
            layout: PagedBf16CausalGqaLayout::packed(
                rows,
                1,
                heads,
                kh,
                dim,
                16,
                0,
                1,
                pages + 1,
                (dim as f64).sqrt().recip() as f32,
            )
            .unwrap(),
            expected: f32s(d.join("hf-raw.f32")),
            groups: vec![Bf16AttentionGroup {
                row_start: 0,
                rows: rows as u32,
                sequence: 0,
                tokens: n as u32,
            }],
        }
    }
    fn buffers(&mut self) -> PagedBf16GqaBuffers<'_> {
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
    fn plan(&self) -> Bf16AttentionPlan {
        Bf16AttentionPlan::new(
            &self.groups,
            self.layout.q_heads,
            self.layout.kv_heads,
            self.layout.head_dim,
            128 << 20,
            16 << 20,
        )
        .unwrap()
    }
    fn exact(&self, op: &CudaOperators) {
        op.check_paged_bf16_transformer_status(&self.status)
            .unwrap();
        let got = op.download_f32_buffer(&self.out).unwrap();
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            self.expected
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
#[test]
#[ignore = "sole CUDA owner; 24 original HF full-output cases and hot resource contract"]
fn frozen_prepared_attention_full_outputs_and_resources() {
    let op = CudaOperators::new().unwrap();
    let root = path();
    let cases = std::fs::read_to_string(root.join("cases.txt")).unwrap();
    for line in cases.lines() {
        let mut entry = line.split_whitespace();
        let d = root.join(entry.next().unwrap());
        let append = entry.next().unwrap() == "1";
        let mut f = Fixture::one(&op, &d);
        let plan = f.plan();
        let count = plan.submission_count(append);
        let mut owner = op.prepare_bf16_attention(plan).unwrap();
        assert_eq!(owner.provider_version(), 120405);
        for _ in 0..2 {
            let layout = f.layout;
            owner.begin_invocation(&op, &mut f.status).unwrap();
            op.reset_counters();
            let allocation = op.allocator_metrics();
            let syncs = op.stream_wide_sync_attempts();
            owner.execute(&op, f.buffers(), layout, append).unwrap();
            assert_eq!(
                op.allocator_metrics(),
                allocation,
                "launch must not allocate/free"
            );
            assert_eq!(op.stream_wide_sync_attempts(), syncs);
            let counters = op.counters();
            assert_eq!(counters.compute_kernel_launches, count);
            assert_eq!(counters.host_to_device_bytes, 0);
            assert_eq!(counters.device_to_host_bytes, 0);
            println!(
                "resource case={} append={} submissions={} h2d={} d2h={} allocator_unchanged=true owner_sync_delta=0 physical_grant={} external_reserve={} observed_provider_growth={}",
                d.file_name().unwrap().to_string_lossy(),
                append,
                counters.compute_kernel_launches,
                counters.host_to_device_bytes,
                counters.device_to_host_bytes,
                owner.plan().storage_grant_bytes(),
                owner.plan().provider_bytes(),
                owner.provider_observed_growth_bytes()
            );
            owner.finish_invocation(&op, &f.status).unwrap();
            f.exact(&op);
        }
    }
}
#[test]
#[ignore = "sole CUDA owner; ragged groups, page split, private COW pages, aliases and owner/capture rejection"]
fn ragged_prepared_attention_and_prelaunch_rejections() {
    let op = CudaOperators::new().unwrap();
    let root = path();
    let mut f = Fixture::one(&op, &root.join("actual-read"));
    let d = root.join("synthetic-10-read");
    let mut q = f32s(root.join("actual-read/query.f32"));
    q.extend(f32s(d.join("query.f32")));
    let mut k = words(root.join("actual-read/cache-key.bf16"));
    let mut v = words(root.join("actual-read/cache-value.bf16"));
    let first_slots = k.len() / (16 * 2 * 256);
    k.extend(words(d.join("cache-key.bf16")));
    v.extend(words(d.join("cache-value.bf16")));
    let mut slots = ints(root.join("actual-read/slots.i32"));
    slots.extend(
        ints(d.join("slots.i32"))
            .iter()
            .map(|s| s + first_slots as i32),
    );
    let mut positions = vec![10];
    positions.extend(ints(d.join("positions.i32")));
    let mut ak = words(root.join("actual-read/append-key.bf16"));
    ak.extend(words(d.join("append-key.bf16")));
    let mut av = words(root.join("actual-read/append-value.bf16"));
    av.extend(words(d.join("append-value.bf16")));
    f.q = op.upload_f32_buffer(&q).unwrap();
    f.k = op.upload_bf16_words(&k).unwrap();
    f.v = op.upload_bf16_words(&v).unwrap();
    f.ak = op.upload_bf16_words(&ak).unwrap();
    f.av = op.upload_bf16_words(&av).unwrap();
    f.slots = op.upload_i32_buffer(&slots).unwrap();
    f.offsets = op.upload_i32_buffer(&[0, 1, 10]).unwrap();
    f.seq = op.upload_i32_buffer(&[0, 1, 1, 1, 1, 1]).unwrap();
    f.pos = op.upload_i32_buffer(&positions).unwrap();
    f.len = op
        .upload_i32_buffer(&[11, 129, 129, 129, 129, 129])
        .unwrap();
    f.out = op.zero_f32_buffer(6 * 4096).unwrap();
    f.layout = PagedBf16CausalGqaLayout::packed(
        6,
        2,
        16,
        2,
        256,
        16,
        0,
        1,
        k.len() / (16 * 2 * 256),
        0.0625,
    )
    .unwrap();
    f.expected.extend(f32s(d.join("hf-raw.f32")));
    f.groups.push(Bf16AttentionGroup {
        row_start: 1,
        rows: 5,
        sequence: 1,
        tokens: 129,
    });
    let plan = f.plan();
    let mut owner = op.prepare_bf16_attention(plan).unwrap();
    let layout = f.layout;
    let other = CudaOperators::new().unwrap();
    owner.begin_invocation(&op, &mut f.status).unwrap();
    owner.execute(&op, f.buffers(), layout, true).unwrap();
    f.exact(&op);
    // Exact stream identity, not merely the same visible device.
    op.reset_counters();
    other.reset_counters();
    assert!(owner.execute(&other, f.buffers(), layout, false).is_err());
    assert_eq!(other.counters(), Default::default());
    let original = std::mem::replace(&mut f.out, f.q.prefix(q.len()).unwrap());
    assert!(owner.execute(&op, f.buffers(), layout, false).is_err());
    assert_eq!(op.counters(), Default::default());
    f.out = original;
    let original = std::mem::replace(&mut f.status, op.zero_i32_buffer(0).unwrap());
    op.reset_counters();
    assert!(owner.execute(&op, f.buffers(), layout, false).is_err());
    assert_eq!(op.counters(), Default::default());
    f.status = original;
    let mut wrong = layout;
    wrong.head_dim = 255;
    assert!(owner.execute(&op, f.buffers(), wrong, false).is_err());
    let graph = op
        .capture_decode_graph(|| {
            assert!(owner.execute(&op, f.buffers(), layout, false).is_err());
            Ok(())
        })
        .unwrap();
    drop(graph);
    assert!(!owner.is_poisoned());
    owner.execute(&op, f.buffers(), layout, false).unwrap();
    f.exact(&op);
    owner.finish_invocation(&op, &f.status).unwrap();
    // Physical COW: substitute a copied private page, append only there, and prove
    // the protected source allocation remains unchanged. Transaction scheduling is
    // separately covered by cuda_bf16_kv_transactions (not claimed by this test).
    let original_k = op.download_bf16_buffer(&f.k).unwrap();
    let original_v = op.download_bf16_buffer(&f.v).unwrap();
    let mut poisoned_k = k.clone();
    let mut poisoned_v = v.clone();
    for group in &f.groups {
        let table_start = if group.sequence == 0 { 0 } else { 1 };
        for &position in
            &positions[group.row_start as usize..(group.row_start + group.rows) as usize]
        {
            let slot = slots[table_start + position as usize / 16] as usize;
            let begin = (slot * 16 + position as usize % 16) * 2 * 256;
            poisoned_k[begin..begin + 2 * 256].fill(0x7fc0);
            poisoned_v[begin..begin + 2 * 256].fill(0x7fc0);
        }
    }
    assert_ne!(poisoned_k, k);
    assert_ne!(poisoned_v, v);
    let mut private_k = op.upload_bf16_words(&poisoned_k).unwrap();
    let mut private_v = op.upload_bf16_words(&poisoned_v).unwrap();
    std::mem::swap(&mut private_k, &mut f.k);
    std::mem::swap(&mut private_v, &mut f.v);
    owner.begin_invocation(&op, &mut f.status).unwrap();
    owner.execute(&op, f.buffers(), layout, true).unwrap();
    f.exact(&op);
    owner.finish_invocation(&op, &f.status).unwrap();
    // Both private planes must be repaired by append; merely reading protected
    // original pages would match output but fail these complete plane checks.
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    assert_eq!(
        bits(op.download_bf16_buffer(&f.k).unwrap()),
        k.iter().map(|&v| u32::from(v) << 16).collect::<Vec<_>>()
    );
    assert_eq!(
        bits(op.download_bf16_buffer(&f.v).unwrap()),
        v.iter().map(|&v| u32::from(v) << 16).collect::<Vec<_>>()
    );
    println!(
        "ragged private-copy: 6 appended rows poisoned/repaired, private and protected full planes checked; exact HF output"
    );
    assert_eq!(
        op.download_bf16_buffer(&private_k)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        original_k.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        op.download_bf16_buffer(&private_v)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        original_v.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "sole GPU owner; independent HF actual/KV128/M5-KV129 through one maximum-capacity owner"]
fn frozen_maxcap_views_share_production_geometry() {
    let op = CudaOperators::new().unwrap();
    let root = path();
    let mut fixtures = ["actual-read", "synthetic-4-read", "synthetic-10-read"]
        .map(|name| Fixture::one(&op, &root.join(name)));
    let cohorts = fixtures
        .iter()
        .map(|f| f.groups.clone())
        .collect::<Vec<_>>();
    let plan = Bf16AttentionPlan::with_views(&cohorts, 16, 2, 256, 64 << 20, 16 << 20).unwrap();
    let views = cohorts
        .iter()
        .map(|g| plan.view(g).unwrap())
        .collect::<Vec<_>>();
    let mut owner = op.prepare_bf16_attention(plan).unwrap();
    let allocated = op.allocator_metrics();
    for _ in 0..2 {
        for (f, view) in fixtures.iter_mut().zip(&views) {
            let layout = f.layout;
            owner.begin_invocation(&op, &mut f.status).unwrap();
            op.reset_counters();
            owner
                .execute_view(&op, f.buffers(), layout, view, false)
                .unwrap();
            assert_eq!(op.counters().device_allocation_attempts, 0);
            assert_eq!(op.counters().stream_wide_syncs, 0);
            owner.finish_invocation(&op, &f.status).unwrap();
            f.exact(&op);
            assert_eq!(op.allocator_metrics(), allocated);
        }
    }
    println!(
        "same H16/D256 maximum owner: actual M1/KV11, M1/KV128 and M5/KV129 HF raw outputs exact twice; allocation unchanged"
    );
}

#[path = "attention_bf16/transaction.rs"]
mod transaction;

#[test]
#[ignore = "owned GPU; model batch8/context128/two-sequence capacity uses 996 warm shapes and one handle"]
fn model_capacity8_context128_sequence_shapes() {
    use ferrule_backend::cuda_plan::bf16_attention::Bf16AttentionShape;
    let op = CudaOperators::new().unwrap();
    let root = path();
    let mut fixtures = ["actual-read", "synthetic-4-read", "synthetic-5-read"]
        .map(|name| Fixture::one(&op, &root.join(name)));
    let shapes = (1..=8)
        .flat_map(|m| (m..=128).map(move |n| Bf16AttentionShape { rows: m, tokens: n }))
        .collect::<Vec<_>>();
    let plan =
        Bf16AttentionPlan::from_shapes(&shapes, 8, 2, 16, 2, 256, 64 << 20, 16 << 20).unwrap();
    assert_eq!(plan.warm_groups().len(), 996);
    assert_eq!(plan.storage_bytes(), 35_848_192);
    let views = fixtures
        .iter()
        .map(|f| plan.view(&f.groups).unwrap())
        .collect::<Vec<_>>();
    let mut owner = op.prepare_bf16_attention(plan).unwrap();
    let allocation = op.allocator_metrics();
    for (f, view) in fixtures.iter_mut().zip(&views) {
        let layout = f.layout;
        owner.begin_invocation(&op, &mut f.status).unwrap();
        owner
            .execute_view(&op, f.buffers(), layout, view, false)
            .unwrap();
        owner.finish_invocation(&op, &f.status).unwrap();
        f.exact(&op);
        assert_eq!(op.allocator_metrics(), allocation);
    }
    println!(
        "H16/D256 model capacity8/context128/maxseq2: 996 distinct shapes, one allocation/handle; actual/KV128 independent HF exact"
    );
}
