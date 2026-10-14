use super::*;

#[test]
fn descriptor_query_preserves_exact_f32_activation_charges() {
    for hidden in [1, 8, 2048] {
        for intermediate in [1, 12, 4096] {
            for bias in [false, true] {
                let descriptor = SwiGlu::new(hidden, intermediate, bias).unwrap();
                let bias_width = if bias {
                    2 * intermediate + hidden + 3
                } else {
                    0
                };
                let per_row = (2 * hidden + 3 * intermediate + bias_width) * 4;
                for rows in [0, 1, 3, 32] {
                    let operation = descriptor.f32_scratch_bytes(rows).unwrap();
                    assert_eq!(operation, rows * per_row);
                    let plan = SwiGluScratchPlan::for_descriptor(&descriptor, rows).unwrap();
                    assert_eq!(operation, plan.total_bytes().unwrap());
                    assert_eq!(plan.numeric_reservation_bytes, 0);
                    assert_eq!(plan.route_bytes, 0);
                    assert_eq!(plan.table_bytes, 0);
                    assert_eq!(plan.total_with(128, 4096).unwrap(), operation + 128 + 4096);
                }
            }
        }
    }
}

#[test]
fn query_checks_geometry_rows_bytes_and_exact_overflow_boundary() {
    let descriptor = SwiGlu::new(8, 12, false).unwrap();
    let bytes_per_row = (2 * 8 + 3 * 12) * 4;
    let last_row_count = usize::MAX / bytes_per_row;
    assert_eq!(descriptor.f32_scratch_bytes(0).unwrap(), 0);
    assert_eq!(
        descriptor.f32_scratch_bytes(last_row_count).unwrap(),
        last_row_count * bytes_per_row
    );
    assert!(descriptor.f32_scratch_bytes(last_row_count + 1).is_err());
    assert!(descriptor.f32_scratch_bytes(usize::MAX).is_err());
    // Zero-row weight preparation still rejects an overflowing per-row shape.
    for (hidden, intermediate, bias) in [
        (1, usize::MAX, false),
        (usize::MAX, 1, false),
        (usize::MAX / 4, usize::MAX / 4, true),
    ] {
        let descriptor = SwiGlu::new(hidden, intermediate, bias).unwrap();
        assert!(descriptor.f32_scratch_bytes(0).is_err());
        assert!(descriptor.f32_scratch_bytes(1).is_err());
    }
    assert!(SwiGlu::new(0, 12, false).is_err());
    assert!(SwiGlu::new(8, 0, false).is_err());
}

#[test]
fn owner_bucket_mixed_groups_charge_peak_grants_not_sum_or_last_group() {
    use super::super::expert_cache::Cache;
    use crate::transformer::ExpertCacheLimits;

    let grant = |bytes| ferrule_backend::cuda_allocation_grant_bytes(bytes).unwrap();
    for groups in [[1, 17], [17, 1], [2051, 2052], [2052, 2051]] {
        let tokens = groups.iter().sum::<usize>();
        let descriptor = SwiGlu::new(8, 12, false).unwrap();
        let scratch = groups.map(|rows| {
            let plan = SwiGluScratchPlan::for_descriptor(&descriptor, rows).unwrap();
            let extra = plan.owner_bucket_extra_bytes(tokens, 8).unwrap();
            let bytes = plan.total_with(extra, 4096).unwrap();
            assert_eq!(
                bytes,
                grant(tokens * 8 * 4) + 2 * grant(rows * 8 * 4) + 3 * grant(rows * 12 * 4) + 4096
            );
            bytes
        });
        let peak = *scratch.iter().max().unwrap();
        let cap = 294 + peak;
        let rejected = Cache::<usize>::new(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: cap - 1,
        })
        .unwrap();
        let before = rejected.stats();
        assert!(scratch.iter().any(|&n| rejected.preflight(294, n).is_err()));
        assert_eq!(
            rejected.stats(),
            before,
            "preflight never evicts or reserves"
        );
        let mut cache = Cache::new(ExpertCacheLimits {
            max_experts: 1,
            max_bytes: cap,
        })
        .unwrap();
        for &bytes in &scratch {
            cache.preflight(294, bytes).unwrap();
        }
        cache.scratch(peak, |_| panic!("fresh cache")).unwrap();
        for (id, bytes) in scratch.into_iter().enumerate() {
            cache.scratch(bytes, |_| {}).unwrap();
            let lease = cache.acquire(id, 294, |_| {}).unwrap();
            assert_eq!(cache.stats().charged_bytes(), bytes + 294);
            cache.complete(&lease, true).unwrap();
        }
        assert_eq!(cache.stats().peak_bytes, cap);
        assert_eq!(cache.stats().uploads, 2);
        assert_eq!(cache.stats().evictions, 1);
    }
    // Rounded grants, total scratch and expert + scratch each check overflow.
    assert!(ferrule_backend::cuda_allocation_grant_bytes(usize::MAX).is_none());
    assert!(SwiGluScratchPlan::reserved_bytes(usize::MAX, 1, 0).is_err());
    assert!(SwiGluScratchPlan::reserved_bytes(usize::MAX - 1, 1, 1).is_err());
    let cache = Cache::<usize>::new(ExpertCacheLimits {
        max_experts: 1,
        max_bytes: usize::MAX,
    })
    .unwrap();
    assert!(cache.preflight(1, usize::MAX).is_err());
}

#[test]
fn owner_bucket_budget_counts_output_and_live_group_allocator_grants() {
    let grant = |n| ferrule_backend::cuda_allocation_grant_bytes(n).unwrap();
    let descriptor = SwiGlu::new(8, 12, false).unwrap();
    for (tokens, group) in [(6, 3), (4102, 2051), (100_000, 50_000)] {
        let plan = SwiGluScratchPlan::for_descriptor(&descriptor, group).unwrap();
        let extra = plan.owner_bucket_extra_bytes(tokens, 8).unwrap();
        let expected = grant(tokens * 8 * 4) + 2 * grant(group * 8 * 4) + 3 * grant(group * 12 * 4);
        assert_eq!(plan.total_with(extra, 4096).unwrap(), expected + 4096);
        assert!(extra >= tokens * 8 * 4);
        for (cap, ok) in [
            (294 + expected + 4096, true),
            (294 + expected + 4095, false),
        ] {
            let cache = super::super::expert_cache::Cache::<usize>::new(
                crate::transformer::ExpertCacheLimits {
                    max_experts: 1,
                    max_bytes: cap,
                },
            )
            .unwrap();
            assert_eq!(
                cache
                    .preflight(294, plan.total_with(extra, 4096).unwrap())
                    .is_ok(),
                ok
            );
        }
    }
    let plan = SwiGluScratchPlan::for_descriptor(&descriptor, 1).unwrap();
    assert!(plan.owner_bucket_extra_bytes(usize::MAX, 8).is_err());
    assert!(plan.owner_bucket_extra_bytes(usize::MAX / 4, 1).is_err());
    assert!(plan.owner_bucket_extra_bytes(1, 0).is_err());
}
