use super::*;
use crate::cuda::operators::linear::{Bf16GemmLayout, CudaBf16Buffer, CudaOperators};
use std::path::{Path, PathBuf};
fn words(path: impl AsRef<Path>) -> Vec<u16> {
    let data = std::fs::read(path).unwrap();
    assert!(data.len().is_multiple_of(2));
    data.as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(u16::from_le_bytes)
        .collect()
}
fn plan(m: usize, n: usize, k: usize) -> Bf16GemmExPlan {
    Bf16GemmExPlan::new_resident(n, k, m, usize::MAX, usize::MAX).unwrap()
}
#[test]
fn positive_resident_dimensions_and_element_alignment_are_checked() {
    for m in [1, 5, 32] {
        for n in [1, 3, 7, 8, 31] {
            for k in [1, 7, 31, 2048] {
                let p = plan(m, n, k);
                assert!(p.is_resident());
                assert_eq!(p.staging_bytes() % 256, 0);
                assert!(
                    Bf16GemmExPlan::new_resident(n, k, m, p.storage_grant_bytes() - 1, 16 << 20)
                        .is_err()
                );
            }
        }
    }
    for (n, k, m) in [
        (0, 2048, 1),
        (1, 0, 1),
        (1, 2048, 0),
        (1, 2048, 33),
        (usize::MAX, 1, 1),
        (1, usize::MAX, 1),
    ] {
        assert!(Bf16GemmExPlan::new_resident(n, k, m, usize::MAX, usize::MAX).is_err());
    }
    let mut a = ResidentArgs {
        m: 5,
        n: 1,
        k: 7,
        reserved: 0,
        input: 0x1002,
        weight: 0x2002,
        output: 0x3002,
        stream: 0,
    };
    assert_eq!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
    a.output += 1;
    assert_ne!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
    a.output = a.input + 2;
    assert_ne!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
    a.output = u64::MAX - 1;
    assert_ne!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
}
#[test]
#[ignore = "sole GPU owner; frozen HF scalar and public GemmEx positive tails, reusable pool"]
fn resident_scalar_and_tails_match_public_gemmex() {
    let dir = PathBuf::from(std::env::var_os("FERRULE_RESIDENT_SCALAR_FIXTURE").unwrap());
    let text = std::fs::read_to_string(dir.join("cases.txt")).unwrap();
    let cases: Vec<_> = text
        .lines()
        .map(|l| {
            let c: Vec<_> = l.split_whitespace().collect();
            (
                c[0],
                c[1].parse::<usize>().unwrap(),
                c[2].parse::<usize>().unwrap(),
                c[3].parse::<usize>().unwrap(),
            )
        })
        .collect();
    let mut plans = Vec::new();
    for &(_, m, n, k) in &cases {
        let p = plan(m, n, k);
        if !plans.contains(&p) {
            plans.push(p);
        }
    }
    let op = CudaOperators::new_on_device(0).unwrap();
    let baseline = op.allocator_metrics().live_requested_bytes;
    let bytes = plans.iter().map(|p| p.storage_grant_bytes()).max().unwrap();
    assert!(
        op.prepare_bf16_gemmex_views(&plans, bytes - 1, 16 << 20)
            .is_err()
    );
    let mut pool = op
        .prepare_bf16_gemmex_views(&plans, bytes, 16 << 20)
        .unwrap();
    let mut count = 0;
    for (name, m, n, k) in cases {
        let offset = usize::from(name.starts_with("tail-"));
        let mut x = vec![0; offset];
        x.extend(words(dir.join(format!("{name}.input.bf16"))));
        let mut w = vec![0; offset];
        w.extend(words(dir.join(format!("{name}.weight.bf16"))));
        let xb = op.upload_bf16_words(&x).unwrap();
        let wb = op.upload_bf16_words(&w).unwrap();
        let yb = op.zero_bf16_buffer(m * n + offset).unwrap();
        let x =
            CudaBf16Buffer::from_device_buffer(xb.as_device_buffer().slice(offset, m * k).unwrap());
        let w =
            CudaBf16Buffer::from_device_buffer(wb.as_device_buffer().slice(offset, n * k).unwrap());
        let mut y =
            CudaBf16Buffer::from_device_buffer(yb.as_device_buffer().slice(offset, m * n).unwrap());
        for (buffer, suffix) in [(&x, "input.bf16"), (&w, "weight.bf16")] {
            let uploaded = op
                .download_bf16_buffer(buffer)
                .unwrap()
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect::<Vec<_>>();
            assert_eq!(uploaded, words(dir.join(format!("{name}.{suffix}"))));
        }
        let before = op.counters();
        pool.execute_bf16_view(
            &op,
            plan(m, n, k),
            &w,
            &x,
            &mut y,
            Bf16GemmLayout::contiguous(m, n, k),
        )
        .unwrap();
        let after = op.counters();
        assert_eq!(after.kernel_launches - before.kernel_launches, 1);
        assert_eq!(
            after.device_allocation_attempts,
            before.device_allocation_attempts
        );
        assert_eq!(after.host_to_device_copies, before.host_to_device_copies);
        assert_eq!(after.stream_wide_syncs, before.stream_wide_syncs);
        let actual = op
            .download_bf16_buffer(&y)
            .unwrap()
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        assert_eq!(
            actual,
            words(dir.join(format!("{name}.direct.bf16"))),
            "{name}"
        );
        if dir.join(format!("{name}.hf.bf16")).exists() {
            assert_eq!(
                actual,
                words(dir.join(format!("{name}.hf.bf16"))),
                "original HF {name}"
            );
        }
        eprintln!(
            "SCALAR_GEMMEX case={name} m={m} n={n} k={k} words={} unequal=0 offset_bytes={}",
            actual.len(),
            offset * 2
        );
        count += actual.len();
    }
    let p = plan(5, 1, 2048);
    let layout = Bf16GemmLayout::contiguous(5, 1, 2048);
    let x = op.zero_bf16_buffer(5 * 2048).unwrap();
    let w = op.zero_bf16_buffer(2048).unwrap();
    let mut y = op.zero_bf16_buffer(5).unwrap();
    let mut short = op.zero_bf16_buffer(4).unwrap();
    let mut alias = CudaBf16Buffer::from_device_buffer(x.as_device_buffer().slice(0, 5).unwrap());
    let foreign = CudaOperators::new_on_device(0).unwrap();
    assert!(
        pool.execute_bf16_view(&foreign, p, &w, &x, &mut y, layout)
            .is_err()
    );
    assert!(
        pool.execute_bf16_view(&op, p, &w, &x, &mut alias, layout)
            .is_err()
    );
    assert!(
        pool.execute_bf16_view(&op, p, &w, &x, &mut short, layout)
            .is_err()
    );
    let mut strided = layout;
    strided.output_stride = 2;
    assert!(
        pool.execute_bf16_view(&op, p, &w, &x, &mut y, strided)
            .is_err()
    );
    assert!(
        op.capture_decode_graph(|| pool.execute_bf16_view(&op, p, &w, &x, &mut y, layout))
            .is_err()
    );
    pool.execute_bf16_view(&op, p, &w, &x, &mut y, layout)
        .unwrap();
    op.sync_stream().unwrap();
    drop((pool, x, w, y, short, alias));
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
    eprintln!("SCALAR_TOTAL words={count} unequal=0 reclaimed=true");
}
