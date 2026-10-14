use ferrule_backend::cuda_plan::bf16_attention::*;
fn group(m: u32, n: u32) -> Bf16AttentionGroup {
    Bf16AttentionGroup {
        row_start: 0,
        rows: m,
        sequence: 0,
        tokens: n,
    }
}
fn plan(m: u32, n: u32) -> Bf16AttentionPlan {
    Bf16AttentionPlan::new(&[group(m, n)], 16, 2, 256, usize::MAX, usize::MAX).unwrap()
}
#[test]
fn workspace_geometry_and_exact_budget() {
    for (m, n) in [(1, 11), (5, 5), (1, 128), (5, 129), (1024, 1024)] {
        let p = plan(m, n);
        let offsets = p.offsets();
        assert!(offsets.iter().all(|n| n % 256 == 0));
        assert!(offsets.windows(2).all(|n| n[0] < n[1]));
        assert_eq!(
            p.storage_bytes(),
            offsets[6] + BF16_ATTENTION_WORKSPACE_BYTES
        );
        assert!(
            Bf16AttentionPlan::new(
                &[group(m, n)],
                16,
                2,
                256,
                p.storage_grant_bytes(),
                p.provider_bytes()
            )
            .is_ok()
        );
        assert!(
            Bf16AttentionPlan::new(
                &[group(m, n)],
                16,
                2,
                256,
                p.storage_grant_bytes() - 1,
                p.provider_bytes()
            )
            .is_err()
        );
        assert!(
            Bf16AttentionPlan::new(
                &[group(m, n)],
                16,
                2,
                256,
                p.storage_grant_bytes(),
                p.provider_bytes() - 1
            )
            .is_err()
        );
    }
}
#[test]
fn rejects_bad_dimensions_overflow_and_groups() {
    for (h, kh, d) in [
        (0, 2, 256),
        (16, 0, 256),
        (15, 2, 256),
        (16, 2, 0),
        (usize::MAX, 1, 256),
        (16, 2, usize::MAX),
        (i32::MAX as usize, 1, i32::MAX as usize),
    ] {
        assert!(Bf16AttentionPlan::new(&[group(1, 11)], h, kh, d, usize::MAX, usize::MAX).is_err());
    }
    for (m, n) in [(0, 1), (1, 0), (2, 1), (1, 1025), (u32::MAX, u32::MAX)] {
        assert!(
            Bf16AttentionPlan::new(&[group(m, n)], 16, 2, 256, usize::MAX, usize::MAX).is_err()
        );
    }
    assert!(Bf16AttentionPlan::new(&[], 16, 2, 256, usize::MAX, usize::MAX).is_err());
    assert!(
        Bf16AttentionPlan::new(
            &[group(1, 11), group(1, 11)],
            16,
            2,
            256,
            usize::MAX,
            usize::MAX
        )
        .is_err()
    );
}
#[test]
fn ragged_reuses_maximum_storage_without_splitting_heads() {
    let groups = [
        group(5, 129),
        Bf16AttentionGroup {
            row_start: 5,
            rows: 1,
            sequence: 1,
            tokens: 11,
        },
    ];
    let p = Bf16AttentionPlan::new(&groups, 16, 2, 256, usize::MAX, usize::MAX).unwrap();
    assert_eq!(p.storage_bytes(), plan(5, 129).storage_bytes());
    assert_eq!(p.rows(), 6);
    assert_eq!(p.groups(), groups);
    assert_eq!(p.max_rows(), 5);
    assert_eq!(p.max_tokens(), 129);
    assert_eq!(p.submission_count(false), 14);
    assert_eq!(p.submission_count(true), 15);
    assert!(p.total_charge_bytes() < 64 << 20);
}

#[test]
fn bounded_decode_views_share_one_maximum_workspace() {
    let cohorts = (1..=128).map(|n| vec![group(1, n)]).collect::<Vec<_>>();
    let p = Bf16AttentionPlan::with_views(&cohorts, 16, 2, 256, 64 << 20, 16 << 20).unwrap();
    assert_eq!(p.storage_bytes(), plan(1, 128).storage_bytes());
    assert_eq!(p.warm_groups().len(), 128);
    for c in &cohorts {
        let v = p.view(c).unwrap();
        assert!(p.admits(&v));
        assert_eq!(v.rows(), 1);
    }
    assert!(p.view(&[group(1, 129)]).is_err());
    let wrong = plan(5, 128).view(&[group(5, 128)]).unwrap();
    assert!(!p.admits(&wrong));
    assert!(
        Bf16AttentionPlan::with_views(&[vec![group(1, 1025)]], 16, 2, 256, usize::MAX, usize::MAX)
            .is_err()
    );
}

#[test]
fn sequence_shapes_compose_without_cohort_enumeration() {
    let shapes = (1..=8)
        .flat_map(|m| (m..=128).map(move |n| Bf16AttentionShape { rows: m, tokens: n }))
        .collect::<Vec<_>>();
    let p = Bf16AttentionPlan::from_shapes(&shapes, 8, 2, 16, 2, 256, 64 << 20, 16 << 20).unwrap();
    assert_eq!(p.warm_groups().len(), 996);
    assert_eq!(
        (
            p.max_total_rows(),
            p.max_sequences(),
            p.max_rows(),
            p.max_tokens()
        ),
        (8, 2, 8, 128)
    );
    assert_eq!(p.storage_bytes(), 35_848_192);
    for m in 1..8 {
        for n in [8, 17, 128] {
            let g = [
                group(m, n),
                Bf16AttentionGroup {
                    row_start: m,
                    rows: 8 - m,
                    sequence: 1,
                    tokens: 128,
                },
            ];
            assert!(p.admits(&p.view(&g).unwrap()));
        }
    }
    assert!(
        p.view(&[
            group(8, 128),
            Bf16AttentionGroup {
                row_start: 8,
                rows: 1,
                sequence: 1,
                tokens: 128
            }
        ])
        .is_err()
    );
    assert!(
        p.view(&[
            group(1, 128),
            Bf16AttentionGroup {
                row_start: 1,
                rows: 1,
                sequence: 1,
                tokens: 128
            },
            Bf16AttentionGroup {
                row_start: 2,
                rows: 1,
                sequence: 2,
                tokens: 128
            }
        ])
        .is_err()
    );
    assert!(p.view(&[group(1, 129)]).is_err());
    assert!(
        Bf16AttentionPlan::from_shapes(
            &[Bf16AttentionShape {
                rows: 1,
                tokens: 1025
            }],
            8,
            2,
            16,
            2,
            256,
            usize::MAX,
            usize::MAX
        )
        .is_err()
    );
}
