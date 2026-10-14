use super::*;
use crate::cuda::operators::linear::{Bf16GemmLayout, CudaBf16Buffer, CudaOperators};
use crate::cuda_plan::{NumericFp8Layout, NumericFp8ScaleType};
fn plan(m: usize) -> Bf16GemmExPlan {
    Bf16GemmExPlan::new(
        NumericFp8Layout {
            n: 256,
            k: 2048,
            row_origin: 0,
            column_origin: 0,
            scale_type: NumericFp8ScaleType::Bf16,
        },
        m,
        128 << 20,
        16 << 20,
    )
    .unwrap()
}
fn words(path: impl AsRef<std::path::Path>) -> Vec<u16> {
    let data = std::fs::read(path).unwrap();
    assert!(data.len().is_multiple_of(2));
    data.as_chunks::<2>()
        .0
        .iter()
        .copied()
        .map(u16::from_le_bytes)
        .collect()
}
#[test]
fn resident_native_preflight_rejects_alias_alignment_and_overflow_without_cuda() {
    assert_eq!(std::mem::size_of::<ResidentArgs>(), 48);
    let mut a = ResidentArgs {
        m: 1,
        n: 256,
        k: 2048,
        reserved: 0,
        input: 0x1000,
        weight: 0x100000,
        output: 0x400000,
        stream: 0,
    };
    assert_eq!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
    a.output = a.weight;
    assert_ne!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
    a.output = 0x400001;
    assert_ne!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
    a.output = !255u64;
    assert_ne!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
    a.output = 0x400000;
    a.n = 0;
    assert_ne!(unsafe { ferrule_bf16_gemmex_resident_can(&a) }, 0);
}
#[test]
#[ignore = "sole GPU owner; frozen BF16 router M1/M5 and packed/alias/owner/capture rejection"]
fn resident_router_frozen_and_rejection() {
    let dir =
        std::path::PathBuf::from(std::env::var_os("FERRULE_RESIDENT_ROUTER_FIXTURE").unwrap());
    let op = CudaOperators::new_on_device(0).unwrap();
    let baseline = op.allocator_metrics().live_requested_bytes;
    let plans = [plan(1), plan(5)];
    let mut provider = op
        .prepare_bf16_gemmex_views(&plans, plans[1].storage_grant_bytes(), 16 << 20)
        .unwrap();
    for (name, m) in [
        ("hello0", 1),
        ("hello3", 1),
        ("capital0", 5),
        ("capital1", 5),
    ] {
        let p = plan(m);
        let l = Bf16GemmLayout::contiguous(m, 256, 2048);
        let w = op
            .upload_bf16_words(&words(dir.join(format!("{name}.weight.bf16"))))
            .unwrap();
        let x = op
            .upload_bf16_words(&words(dir.join(format!("{name}.input.bf16"))))
            .unwrap();
        let mut y = op.zero_bf16_buffer(m * 256).unwrap();
        let before = op.counters();
        provider
            .execute_bf16_view(&op, p, &w, &x, &mut y, l)
            .unwrap();
        let after = op.counters();
        assert_eq!(after.kernel_launches - before.kernel_launches, 1);
        assert_eq!(
            after.device_allocation_attempts,
            before.device_allocation_attempts
        );
        assert_eq!(after.host_to_device_copies, before.host_to_device_copies);
        assert_eq!(after.stream_wide_syncs, before.stream_wide_syncs);
        let actual = op.download_bf16_buffer(&y).unwrap();
        let expected = words(dir.join(format!("{name}.hf.bf16")));
        assert_eq!(
            actual
                .iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
            expected
        );
        let legacy_layout = l;
        let mut old = op.zero_f32_buffer(m * 256).unwrap();
        op.bf16_linear_into(&w, &x, &mut old, legacy_layout)
            .unwrap();
        let old = op
            .f32_rows_to_bf16_rne(
                &old,
                crate::cuda::operators::linear::F32ToBf16RowsLayout::packed(1, m, 256).unwrap(),
            )
            .unwrap();
        let old = op.download_bf16_buffer(&old).unwrap();
        let old_bad = old
            .iter()
            .zip(&expected)
            .filter(|(a, b)| (a.to_bits() >> 16) as u16 != **b)
            .count();
        eprintln!(
            "RESIDENT_ROUTER case={name} m={m} words={} direct_unequal=0 legacy_unequal={old_bad} launches=1 copies=0",
            expected.len()
        );
        for bad in [
            Bf16GemmLayout {
                activation_stride: 2056,
                ..l
            },
            Bf16GemmLayout {
                weight_stride: 2056,
                ..l
            },
            Bf16GemmLayout {
                output_stride: 264,
                ..l
            },
        ] {
            assert!(
                provider
                    .execute_bf16_view(&op, p, &w, &x, &mut y, bad)
                    .is_err()
            );
        }
        let mut alias =
            CudaBf16Buffer::from_device_buffer(x.as_device_buffer().slice(0, m * 256).unwrap());
        assert!(
            provider
                .execute_bf16_view(&op, p, &w, &x, &mut alias, l)
                .is_err()
        );
        let foreign = CudaOperators::new_on_device(0).unwrap();
        assert!(
            provider
                .execute_bf16_view(&foreign, p, &w, &x, &mut y, l)
                .is_err()
        );
        let short =
            CudaBf16Buffer::from_device_buffer(x.as_device_buffer().slice(0, x.len() - 1).unwrap());
        assert!(
            provider
                .execute_bf16_view(&op, p, &w, &short, &mut y, l)
                .is_err()
        );
        let stream = op.stream_clone();
        stream.begin_capture().unwrap();
        assert!(
            provider
                .execute_bf16_view(&op, p, &w, &x, &mut y, l)
                .is_err()
        );
        drop(stream.end_capture().unwrap());
        assert!(!provider.is_poisoned());
    }
    // Reuse the same provider with bounded exactly representable synthetic data.
    for p in plans {
        let m = p.rows();
        let x = op.upload_bf16_words(&vec![0x3f80; m * 2048]).unwrap();
        let w = op.upload_bf16_words(&vec![0x3f80; 256 * 2048]).unwrap();
        let mut y = op.zero_bf16_buffer(m * 256).unwrap();
        provider
            .execute_bf16_view(
                &op,
                p,
                &w,
                &x,
                &mut y,
                Bf16GemmLayout::contiguous(m, 256, 2048),
            )
            .unwrap();
        assert_eq!(op.download_bf16_buffer(&y).unwrap(), vec![2048.; m * 256]);
    }
    drop(provider);
    op.trim_device_allocator().unwrap();
    assert_eq!(op.allocator_metrics().live_requested_bytes, baseline);
}
