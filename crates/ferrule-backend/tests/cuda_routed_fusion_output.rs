//! Additional backend contract only; not independent HF/model numerical evidence.
#![cfg(feature = "cuda")]
use ferrule_backend::cuda::operators::linear::{
    Bf16GemmExPlan, Bf16GemmLayout, CudaOperators, NumericFp8Layout, NumericFp8ScaleType,
};

#[test]
#[ignore = "sole GPU owner: real beta=0 packed down output overwrites NaN C and preserves neighbors"]
fn routed_down_nan_c_packed_offset_beta_zero_m1_m2_m5() {
    assert_eq!(
        std::env::var("CUBLAS_WORKSPACE_CONFIG").as_deref(),
        Ok(":0:0")
    );
    let op = CudaOperators::new_on_device(0).expect("required assigned CUDA device");
    let n = 256;
    let plans: Vec<_> = [1, 2, 5]
        .into_iter()
        .map(|rows| {
            Bf16GemmExPlan::new(
                NumericFp8Layout {
                    n,
                    k: n,
                    row_origin: 0,
                    column_origin: 0,
                    scale_type: NumericFp8ScaleType::Bf16,
                },
                rows,
                128 << 20,
                16 << 20,
            )
            .unwrap()
        })
        .collect();
    let mut owner = op
        .prepare_bf16_gemmex_views(&plans, plans[2].storage_grant_bytes(), 16 << 20)
        .unwrap();
    let weight = op
        .upload_bf16_words(
            &(0..n * n)
                .map(|i| if i / n == i % n { 0x3f80 } else { 0 })
                .collect::<Vec<_>>(),
        )
        .unwrap();
    for plan in plans {
        let rows = plan.rows();
        let expected: Vec<u16> = (0..rows * n)
            .map(|i| [0x3f80, 0xbf40, 0x3eab][i % 3])
            .collect();
        let input = op.upload_bf16_words(&expected).unwrap();
        let len = rows * n;
        let mut seed = vec![0x3f81; 3 * len];
        seed[len..2 * len].fill(0x7fc1);
        let root = op.upload_bf16_words(&seed).unwrap();
        drop(seed);
        let mut output = root.slice(len, len).unwrap();
        let before = op.counters();
        owner
            .execute_bf16_view(
                &op,
                plan,
                &weight,
                &input,
                &mut output,
                Bf16GemmLayout::contiguous(rows, n, n),
            )
            .unwrap();
        let after = op.counters();
        assert_eq!(
            after.compute_kernel_launches - before.compute_kernel_launches,
            1
        );
        assert_eq!(after.device_allocations, before.device_allocations);
        assert_eq!(after.stream_wide_syncs, before.stream_wide_syncs);
        let actual = op.download_bf16_buffer(&root).unwrap();
        assert!(actual.iter().all(|x| x.is_finite()));
        let words: Vec<_> = actual.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
        assert!(words[..len].iter().all(|w| *w == 0x3f81));
        assert_eq!(&words[len..2 * len], &expected);
        assert!(words[2 * len..].iter().all(|w| *w == 0x3f81));
        let unchanged: Vec<_> = op
            .download_bf16_buffer(&input)
            .unwrap()
            .iter()
            .map(|x| (x.to_bits() >> 16) as u16)
            .collect();
        assert_eq!(unchanged, expected);
        eprintln!(
            "PACKED_DOWN_NAN_C rows={rows} offset_words={len} beta_zero_overwrite=true canaries=true input_unchanged=true"
        );
    }
}
