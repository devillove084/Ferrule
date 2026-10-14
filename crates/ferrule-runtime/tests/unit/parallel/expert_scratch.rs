use super::*;
use ferrule_model::transformer::FeedForward;

use super::numeric_test_fixture as fixture;

struct Image(std::path::PathBuf);
impl Image {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        Self(std::env::temp_dir().join(format!(
            "expert-scratch-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )))
    }
}
impl Drop for Image {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[test]
fn owned_prepared_peak_matches_descriptor_without_multiplying_by_expert_count() {
    for numeric in [false, true] {
        let image = Image::new();
        let resources = fixture::bind(&image.0, 4, numeric, true);
        let group = ExpertGroup {
            source_rank: ParallelRankId::new(7),
            members: vec![ParallelRankId::new(109), ParallelRankId::new(42)],
            layers: 0..resources.spec().layers().len(),
            placement: ExpertPlacement::new((0..resources.spec().layers().len()).flat_map(
                |layer| {
                    (0..4).map(move |expert| {
                        (
                            layer,
                            expert,
                            ParallelRankId::new(if expert % 2 == 0 { 109 } else { 42 }),
                        )
                    })
                },
            ))
            .unwrap(),
            limits: ExpertDispatchLimits {
                max_tokens: 12,
                max_bytes: 12 * 8 * 4,
            },
        };
        for (slot, owner) in group.members.iter().copied().enumerate() {
            let worker = ExpertRankWorker::prepare_cpu(
                &resources,
                ExpertRank {
                    slot: ParallelRankId::new(slot as u32),
                    owner,
                },
                &group,
                4096,
            )
            .unwrap();
            let mut peak = 0;
            let mut sum = 0;
            for (id, prepared) in &worker.experts.0 {
                let FeedForward::Moe(moe) = resources.spec().layers()[id.layer].feed_forward()
                else {
                    panic!()
                };
                for rows in [0, 1, group.limits.max_tokens] {
                    assert_eq!(
                        prepared.f32_scratch_bytes(rows).unwrap(),
                        moe.expert().f32_scratch_bytes(rows).unwrap()
                    );
                    assert_eq!(
                        prepared.f32_scratch_bytes(rows).unwrap(),
                        rows * (3 * 12 + 8 + 8) * 4
                    );
                }
                assert!(prepared.f32_scratch_bytes(usize::MAX).is_err());
                let bytes = prepared.f32_scratch_bytes(group.limits.max_tokens).unwrap();
                peak = peak.max(bytes);
                sum += bytes;
            }
            assert_eq!(peak, 12 * (3 * 12 + 8 + 8) * 4);
            assert!(sum > peak);
            if numeric {
                let payload = worker
                    .experts
                    .0
                    .values()
                    .flat_map(|expert| [expert.gate(), expert.up(), expert.down()])
                    .map(|linear| {
                        usize::try_from(linear.numeric_fp8().unwrap().storage_bytes()).unwrap()
                    })
                    .sum::<usize>();
                assert_eq!(payload, worker.experts.0.len() * 294);
                let workspace = 64 << 20;
                let exact = payload + workspace + peak;
                assert!(check_numeric_resident_budget(payload, peak, workspace, exact).is_ok());
                assert!(
                    check_numeric_resident_budget(payload, peak, workspace, exact - 1).is_err()
                );
            }
        }
        for (max_tokens, max_bytes) in [(0, 384), (12, 0)] {
            let mut invalid = group.clone();
            invalid.limits = ExpertDispatchLimits {
                max_tokens,
                max_bytes,
            };
            assert!(invalid.validate().is_err());
        }
    }
}

#[test]
fn resident_budget_charges_workspace_once_and_preserves_rejections() {
    let payload = 4 * 294;
    let scratch = 12 * (3 * 12 + 8 + 8) * 4;
    let workspace = 64 << 20;
    let exact = payload + workspace + scratch;
    assert!(check_numeric_resident_budget(payload, scratch, workspace, exact).is_ok());
    assert!(check_numeric_resident_budget(payload, scratch, workspace, exact - 1).is_err());
    assert!(check_numeric_resident_budget(payload, scratch, 0, usize::MAX).is_err());
    // Preserve both additions, including zero-row/empty payload behavior.
    assert!(check_numeric_resident_budget(0, 0, 1, 1).is_ok());
    for (payload, scratch, workspace) in
        [(usize::MAX, 0, 1), (0, usize::MAX, 1), (1, 1, usize::MAX)]
    {
        assert!(check_numeric_resident_budget(payload, scratch, workspace, usize::MAX).is_err());
    }
}
