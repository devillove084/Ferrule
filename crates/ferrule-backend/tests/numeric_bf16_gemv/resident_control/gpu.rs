use super::*;
use crate::cuda::operators::linear::{Bf16GemmLayout, CudaBf16Buffer, CudaOperators};
use crate::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};
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
fn resident(m: usize, n: usize, k: usize) -> Bf16GemmExPlan {
    Bf16GemmExPlan::new_resident(n, k, m, usize::MAX, usize::MAX).unwrap()
}
#[test]
fn resident_tail_admission_is_distinct_from_fp8_and_charges_grants() {
    for m in [1, 5, 32] {
        for n in [1, 3, 7, 8, 24, 32, 40, 128, 136] {
            let p = resident(m, n, 2048);
            assert!(p.is_resident());
            assert_eq!(p.staging_bytes() % 256, 0);
            assert!(
                Bf16GemmExPlan::new_resident(n, 2048, m, p.storage_grant_bytes() - 1, 16 << 20)
                    .is_err()
            );
            assert!(Bf16GemmExPlan::new_resident(n, 2048, m, usize::MAX, (16 << 20) - 1).is_err());
            let numeric = Bf16GemmExPlan::new(
                NumericFp8Layout {
                    n,
                    k: 2048,
                    row_origin: 0,
                    column_origin: 0,
                    scale_type: NumericFp8ScaleType::Bf16,
                },
                m,
                usize::MAX,
                usize::MAX,
            );
            assert_eq!(numeric.is_ok(), n.is_multiple_of(128));
            if let Ok(other) = numeric {
                assert_ne!(p, other);
                assert_eq!(p.storage_bytes(), other.storage_bytes());
            }
        }
    }
    for (m, n, k) in [
        (0, 32, 2048),
        (33, 32, 2048),
        (1, 0, 2048),
        (1, 32, 0),
        (1, 32, usize::MAX),
        (1, usize::MAX, 2048),
    ] {
        assert!(Bf16GemmExPlan::new_resident(n, k, m, usize::MAX, usize::MAX).is_err());
    }
    let numeric = Bf16GemmExPlan::new(
        NumericFp8Layout {
            n: 8192,
            k: 2048,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        },
        5,
        usize::MAX,
        usize::MAX,
    )
    .unwrap();
    assert_eq!(numeric.storage_grant_bytes(), 69_206_016);
    assert!(resident(32, 32, 2048).storage_bytes() < numeric.storage_bytes());
    let mut a = ResidentArgs {
        m: 1,
        n: 32,
        k: 2048,
        reserved: 0,
        input: 0x1000,
        weight: 0x10000,
        output: 0x40000,
        stream: 0,
    };
    assert_eq!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
    a.n = 0;
    assert_ne!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
    a.n = 40;
    assert_eq!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
    a.output = a.weight;
    assert_ne!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
}
#[test]
#[ignore = "sole GPU owner; frozen actual A/B, SAMEINPUT HF, one reusable pool"]
fn resident_control_frozen_all() {
    let dir = PathBuf::from(std::env::var_os("FERRULE_RESIDENT_CONTROL_FIXTURE").unwrap());
    let cases = std::fs::read_to_string(dir.join("cases.txt")).unwrap();
    let op = CudaOperators::new_on_device(0).unwrap();
    let baseline = op.allocator_metrics().live_requested_bytes;
    let plans = [resident(1, 32, 2048), resident(5, 32, 2048)];
    let mut pool = op
        .prepare_bf16_gemmex_views(&plans, plans[1].storage_grant_bytes(), 16 << 20)
        .unwrap();
    let mut total = 0;
    for line in cases.lines() {
        let c: Vec<_> = line.split_whitespace().collect();
        let name = c[0];
        let m = c[1].parse().unwrap();
        let n = c[2].parse().unwrap();
        let k = c[3].parse().unwrap();
        let x = op
            .upload_bf16_words(&words(dir.join(format!("{name}.input.bf16"))))
            .unwrap();
        let w = op
            .upload_bf16_words(&words(dir.join(format!("{name}.weight.bf16"))))
            .unwrap();
        let mut y = op.zero_bf16_buffer(m * n).unwrap();
        let before = op.counters();
        pool.execute_bf16_view(
            &op,
            resident(m, n, k),
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
        let expected = words(dir.join(format!("{name}.hf.bf16")));
        let bad = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        eprintln!(
            "RESIDENT_CONTROL case={name} words={} unequal={bad}",
            actual.len()
        );
        assert_eq!(actual, expected, "{name}");
        total += actual.len();
    }
    drop(pool);
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
    eprintln!("RESIDENT_CONTROL_TOTAL words={total} unequal=0 reclaimed=true");
}
#[test]
#[ignore = "sole GPU owner; aligned non128 tails, rejection, allocation-free shape reuse"]
fn resident_control_tails_and_rejections() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let baseline = op.allocator_metrics().live_requested_bytes;
    let plans: Vec<_> = [1, 5, 32]
        .into_iter()
        .flat_map(|m| {
            [8, 24, 32, 40, 136]
                .into_iter()
                .map(move |n| resident(m, n, 40))
        })
        .collect();
    let bytes = plans.iter().map(|p| p.storage_grant_bytes()).max().unwrap();
    assert!(
        op.prepare_bf16_gemmex_views(&plans, bytes - 1, 16 << 20)
            .is_err()
    );
    let mut pool = op
        .prepare_bf16_gemmex_views(&plans, bytes, 16 << 20)
        .unwrap();
    for p in plans {
        let (m, n, k) = (p.rows(), p.layout().n, p.layout().k);
        let x = op.upload_bf16_words(&vec![0x3f80; m * k]).unwrap();
        let w = op.upload_bf16_words(&vec![0x3f00; n * k]).unwrap();
        let mut y = op.zero_bf16_buffer(m * n).unwrap();
        let layout = Bf16GemmLayout::contiguous(m, n, k);
        let mut alias =
            CudaBf16Buffer::from_device_buffer(w.as_device_buffer().slice(0, m * n).unwrap());
        let before = op.counters();
        assert!(
            pool.execute_bf16_view(&foreign, p, &w, &x, &mut y, layout)
                .is_err()
        );
        assert!(
            pool.execute_bf16_view(&op, p, &w, &x, &mut alias, layout)
                .is_err()
        );
        let mut short = op.zero_bf16_buffer(m * n - 1).unwrap();
        assert!(
            pool.execute_bf16_view(&op, p, &w, &x, &mut short, layout)
                .is_err()
        );
        assert!(
            op.capture_decode_graph(|| pool.execute_bf16_view(&op, p, &w, &x, &mut y, layout))
                .is_err()
        );
        assert_eq!(op.counters().kernel_launches, before.kernel_launches);
        for _ in 0..2 {
            pool.execute_bf16_view(&op, p, &w, &x, &mut y, layout)
                .unwrap();
            assert_eq!(op.download_bf16_buffer(&y).unwrap(), vec![20.; m * n]);
        }
    }
    drop(pool);
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
}
