use super::*;
use crate::cuda::operators::linear::{CudaBf16Buffer, CudaOperators};
use crate::cuda_plan::bf16_gemmex::*;
use ferrule_common::numeric_fp8::{NumericFp8Layout, NumericFp8ScaleType};
use std::{io::Write, path::PathBuf};
fn plan(m: usize, n: usize, k: usize) -> Bf16GemmExPlan {
    Bf16GemmExPlan::new(
        NumericFp8Layout {
            n,
            k,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        },
        m,
        512 << 20,
        16 << 20,
    )
    .unwrap()
}
fn words(b: &[u8]) -> Vec<u16> {
    assert_eq!(b.len() % 2, 0);
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect()
}
fn bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
}

#[test]
#[ignore = "sole GPU owner, pinned fixture and fresh report, CUDA12.4/sm86"]
fn frozen_backend_gemmex() {
    let dir = PathBuf::from(std::env::var_os("FERRULE_GEMMEX_FIXTURE").unwrap());
    let report = std::env::var_os("FERRULE_GEMMEX_REPORT").unwrap();
    let repeats: usize = std::env::var("FERRULE_GEMMEX_REPEATS")
        .unwrap()
        .parse()
        .unwrap();
    assert!([1, 16].contains(&repeats));
    let mut report = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(report)
        .unwrap();
    let cases = std::fs::read_to_string(dir.join("cases.txt")).unwrap();
    let op = CudaOperators::new_on_device(0).unwrap();
    let mut violations = 0;
    for line in cases.lines() {
        let f = line.split_whitespace().collect::<Vec<_>>();
        assert_eq!(f.len(), 4);
        let (name, m, n, k) = (
            f[0],
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
        );
        let p = plan(m, n, k);
        let load = |suffix: &str| std::fs::read(dir.join(format!("{name}.{suffix}"))).unwrap();
        let w = op
            .upload_numeric_fp8_linear(p.layout(), &load("weight.fp8"), &load("scale.bf16"))
            .unwrap();
        let x = op.upload_bf16_words(&words(&load("input.bf16"))).unwrap();
        let expected = words(&load("hf.bf16"));
        assert_eq!(expected.len(), m * n);
        let mut y = op.zero_bf16_buffer(m * n).unwrap();
        let mut prepared = op.prepare_bf16_gemmex(p).unwrap();
        assert_eq!(prepared.provider_version(), 120405);
        assert!(prepared.provider_observed_growth_bytes() <= 16 << 20);
        let before = op.memory_info().unwrap().0;
        let alloc = op.allocator_metrics();
        let mut max_unequal = 0;
        let mut first = None;
        let mut drift = 0;
        let mut previous = None;
        for repeat in 0..repeats {
            op.reset_counters();
            prepared.execute(&op, &w, &x, &mut y).unwrap();
            assert_eq!(op.counters().device_allocation_attempts, 0);
            assert_eq!(op.counters().stream_wide_syncs, 0);
            let actual = bits(&op.download_bf16_buffer(&y).unwrap());
            let unequal = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            if let Some(i) = actual.iter().zip(&expected).position(|(a, b)| a != b) {
                first = Some((i, actual[i], expected[i]));
            }
            max_unequal = max_unequal.max(unequal);
            if let Some(ref prior) = previous {
                if prior != &actual {
                    drift += 1;
                }
            } else {
                previous = Some(actual);
            }
            if repeat == 0 {
                let storage = prepared.native.storage.as_ref().unwrap();
                let decoded = storage
                    .slice(m * k * 2, n * k * 2)
                    .unwrap()
                    .to_host_vec(&op.stream_clone())
                    .unwrap();
                assert_eq!(
                    decoded,
                    load("decoded.bf16"),
                    "decoded operands differ {name}"
                );
                let packed = storage
                    .slice(0, m * k * 2)
                    .unwrap()
                    .to_host_vec(&op.stream_clone())
                    .unwrap();
                assert_eq!(packed, load("input.bf16"), "packed input differs {name}");
            }
        }
        assert_eq!(drift, 0);
        assert!(
            op.memory_info().unwrap().0 >= before,
            "provider grew after warmup"
        );
        assert_eq!(
            op.allocator_metrics().live_granted_bytes,
            alloc.live_granted_bytes
        );
        let diff = first.map_or("null".into(), |(i, a, b)| {
            format!("{{\"index\":{i},\"actual\":\"{a:04x}\",\"hf\":\"{b:04x}\"}}")
        });
        writeln!(report,"{{\"case\":\"{name}\",\"m\":{m},\"n\":{n},\"k\":{k},\"repeats\":{repeats},\"unequal\":{max_unequal},\"drift\":{drift},\"first\":{diff},\"staging_bytes\":{},\"workspace_bytes\":{},\"provider_growth\":{}}}",p.staging_bytes(),p.cublas_workspace_bytes(),prepared.provider_observed_growth_bytes()).unwrap();
        report.sync_all().unwrap();
        violations += max_unequal;
    }
    // Preserve diagnostic replay of isolated projection halves, but gate the
    // original concatenated HF geometry explicitly. Never infer it from IDs.
    if repeats == 16 || std::env::var("FERRULE_GEMMEX_REQUIRE_EXACT").as_deref() == Ok("1") {
        assert_eq!(violations, 0, "frozen BF16 gate");
    }
}

#[test]
#[ignore = "sole GPU owner; actual graph, alias, owner and allocation lifecycle"]
fn backend_gemmex_lifecycle() {
    let op = CudaOperators::new_on_device(0).unwrap();
    for (m, n, k) in [
        (1, 128, 128),
        (2, 256, 128),
        (5, 128, 256),
        (16, 256, 256),
        (32, 128, 128),
    ] {
        let p = plan(m, n, k);
        let w = op
            .upload_numeric_fp8_linear(
                p.layout(),
                &vec![0x38; n * k],
                &[0x80, 0x3f].repeat(n / 128 * (k / 128)),
            )
            .unwrap();
        let x = op.upload_bf16_words(&vec![0x3f80; m * k]).unwrap();
        let mut y = op.zero_bf16_buffer(m * n).unwrap();
        let mut prepared = op.prepare_bf16_gemmex(p).unwrap();
        prepared.execute(&op, &w, &x, &mut y).unwrap();
        assert_eq!(op.download_bf16_buffer(&y).unwrap(), vec![k as f32; m * n]);
        let mut small = op.zero_bf16_buffer(1).unwrap();
        assert!(prepared.execute(&op, &w, &x, &mut small).is_err());
        if k == n {
            let mut alias =
                CudaBf16Buffer::from_device_buffer(x.as_device_buffer().slice(0, m * k).unwrap());
            assert!(prepared.execute(&op, &w, &x, &mut alias).is_err());
        }
        let raw_alias =
            CudaBf16Buffer::from_device_buffer(y.as_device_buffer().slice(1, m * n - 1).unwrap());
        assert!(
            prepared
                .native
                .args(
                    &op.stream_clone(),
                    &w,
                    x.as_device_buffer(),
                    raw_alias.as_device_buffer()
                )
                .is_err()
        );
        let stream = op.stream_clone();
        stream.begin_capture().unwrap();
        assert!(op.prepare_bf16_gemmex(p).is_err());
        assert!(prepared.execute(&op, &w, &x, &mut y).is_err());
        drop(stream.end_capture().unwrap());
        assert!(!prepared.is_poisoned());
        let foreign = CudaOperators::new_on_device(0).unwrap();
        assert!(prepared.execute(&foreign, &w, &x, &mut y).is_err());
        drop(foreign);
        let lease = Arc::downgrade(&prepared.native);
        op.reset_counters();
        let graph = prepared.capture(&op, &w, &x, &mut y).unwrap();
        assert_eq!(op.counters().device_allocation_attempts, 0);
        assert_eq!(op.counters().stream_wide_syncs, 0);
        graph.upload(&op).unwrap();
        drop(prepared);
        drop(w);
        drop(x);
        drop(y);
        assert!(
            lease.upgrade().is_some(),
            "graph must retain bound workspace and handle"
        );
        for _ in 0..3 {
            graph.launch(&op).unwrap();
            assert_eq!(graph.download(&op).unwrap(), vec![k as f32; m * n]);
        }
        drop(graph);
        assert!(
            lease.upgrade().is_none(),
            "last graph retirement must release provider"
        );
    }
}

#[test]
#[ignore = "isolated sole-GPU process; injected terminal graph failure retains bound storage"]
fn backend_gemmex_unknown_quarantine() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let p = plan(1, 128, 128);
    let w = op
        .upload_numeric_fp8_linear(p.layout(), &vec![0x38; 128 * 128], &[0x80, 0x3f])
        .unwrap();
    let x = op.upload_bf16_words(&vec![0x3f80; 128]).unwrap();
    let mut y = op.zero_bf16_buffer(128).unwrap();
    let mut prepared = op.prepare_bf16_gemmex(p).unwrap();
    let graph = prepared.capture(&op, &w, &x, &mut y).unwrap();
    graph.launch(&op).unwrap();
    assert_eq!(graph.download(&op).unwrap(), vec![128.0; 128]);
    let before = op.allocator_metrics().live_granted_bytes;
    op.stream_clone()
        .context()
        .arm_graph_fault(crate::cuda::runtime::GraphFault::Launch);
    assert!(graph.launch(&op).is_err());
    assert!(prepared.is_poisoned());
    assert!(prepared.execute(&op, &w, &x, &mut y).is_err());
    assert!(op.prepare_bf16_gemmex(p).is_err());
    drop(graph);
    drop(prepared);
    drop(w);
    drop(x);
    drop(y);
    assert!(op.stream_clone().context().graph_completion_unknown());
    assert!(
        op.allocator_metrics().live_granted_bytes >= before,
        "unknown storage must not retire"
    );
}

#[test]
fn backend_gemmex_preflight_without_device_calls() {
    let p = plan(1, 128, 128);
    let args = || NumericArgs {
        m: 1,
        n: 128,
        k: 128,
        padded_k: 128,
        tile_rows: 128,
        scale_cols: 1,
        row_origin: 0,
        column_origin: 0,
        scale_bytes: 2,
        lda: 128,
        ldd: 128,
        reserved: 0,
        activation: 0x1000,
        weight: 0x10000,
        scales: 0x20000,
        output: 0x30000,
        workspace: 0x100000,
        workspace_bytes: p.storage_bytes() as u64,
        stream: 0,
    };
    assert_eq!(unsafe { ferrule_bf16_gemmex_can_implement(&args()) }, 0);
    for which in 0..8 {
        let mut a = args();
        match which {
            0 => a.output = a.activation,
            1 => a.activation = a.workspace,
            2 => a.output += 2,
            3 => a.workspace = !255u64,
            4 => a.workspace_bytes -= 1,
            5 => a.row_origin = 1,
            6 => a.m = 33,
            _ => a.k = u32::MAX,
        }
        assert_eq!(unsafe { ferrule_bf16_gemmex_can_implement(&a) }, 1);
    }
    assert_eq!(unsafe { ferrule_bf16_gemmex_launch(&args(), 0) }, 1);
}

#[test]
#[ignore = "isolated diagnostic negative gate: CUDA12.4 binary with pinned cuBLAS12.9"]
fn backend_gemmex_rejects_unavailable_runtime() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let error = op
        .prepare_bf16_gemmex(plan(1, 128, 128))
        .err()
        .expect("wrong runtime must fail closed");
    assert!(error.to_string().contains("loaded 120901"), "{error}");
    op.record_compute_event().unwrap().synchronize().unwrap();
    assert!(!op.stream_clone().context().graph_completion_unknown());
}
