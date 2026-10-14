//! CPU-only geometry, exact packing and simultaneous-grant regression gates.
#[path = "../src/transformer/standard/cuda/hybrid_rank/mixed_output.rs"]
mod output;
use half::bf16;
use output::RoutedOutputLayout;

#[test]
fn ragged_buckets_partition_one_root_without_overlap_or_padding() {
    for width in [1, 3, 256, 4096] {
        let layout = RoutedOutputLayout::new(11, width).unwrap();
        let mut words = vec![0u16; layout.elements()];
        let mut start = 0;
        for (expert, rows) in [(2, 5), (17, 1), (127, 3), (255, 2)] {
            let range = layout.bucket(start, rows).unwrap();
            assert_eq!(range.start, start * width);
            assert_eq!(range.len(), rows * width);
            assert!(words[range.clone()].iter().all(|word| *word == 0));
            words[range].fill(expert);
            start += rows;
        }
        assert_eq!(start * width, layout.elements());
        assert!(words.iter().all(|word| *word != 0));
        assert_eq!(words[5 * width], 17);
        assert_eq!(words[6 * width], 127);
    }
}

#[test]
fn invalid_extents_fail_before_a_view_or_allocation() {
    for (routes, width) in [(0, 1), (1, 0), (usize::MAX, 2), (1, usize::MAX)] {
        assert!(RoutedOutputLayout::new(routes, width).is_err());
    }
    let layout = RoutedOutputLayout::new(5, 3).unwrap();
    for (start, rows) in [(0, 0), (5, 1), (4, 2), (usize::MAX, 1), (1, usize::MAX)] {
        assert!(layout.bucket(start, rows).is_err());
    }
    assert_eq!(layout.bucket(4, 1).unwrap(), 12..15);
}

#[test]
fn removing_expand_and_repack_preserves_every_non_nan_bf16_word() {
    // NaN payload canonicalization is not a finite tensor exactness contract.
    for word in 0..=u16::MAX {
        let value = bf16::from_bits(word);
        if !value.is_nan() {
            assert_eq!(bf16::from_f32(value.to_f32()).to_bits(), word);
        }
    }
}

fn combine(values: &[u16], weights: &[f32], legacy_pack: bool) -> u16 {
    values
        .iter()
        .zip(weights)
        .fold(0u16, |sum, (&word, &weight)| {
            let value = bf16::from_bits(word);
            let value = if legacy_pack {
                bf16::from_f32(value.to_f32())
            } else {
                value
            };
            let product = bf16::from_f32(value.to_f32() * weight);
            bf16::from_f32(bf16::from_bits(sum).to_f32() + product.to_f32()).to_bits()
        })
}

#[test]
fn f32_route_weights_and_both_bf16_rounds_survive_pack_removal() {
    let weights = [0.100123f32, 0.300789, 0.599088];
    for word in 0..=u16::MAX {
        if !bf16::from_bits(word).is_finite() {
            continue;
        }
        let values = [word, 0x3f81, 0xbf80];
        assert_eq!(
            combine(&values, &weights, false),
            combine(&values, &weights, true)
        );
    }
    // Ascending original expert ID is significant, not router slot order.
    let ordered = [0x4380, 0x3f80, 0xc380]; // 256, 1, -256
    let reordered = [0x4380, 0xc380, 0x3f80];
    assert_ne!(
        combine(&ordered, &[1.; 3], false),
        combine(&reordered, &[1.; 3], false)
    );
    assert!((0..=u16::MAX).any(|word| {
        bf16::from_bits(word).is_finite()
            && combine(&[word], &[weights[0]], false)
                != combine(&[word], &[bf16::from_f32(weights[0]).to_f32()], false)
    }));
}

fn grant(bytes: usize) -> usize {
    ferrule_backend::cuda_allocation_grant_bytes(bytes).unwrap()
}

#[test]
fn simultaneous_bucket_activations_outputs_and_tail_never_increase_grants() {
    for width in [3, 256, 4096] {
        for inner in [128, 1024, 14336] {
            for buckets in [&[1usize][..], &[1, 1, 1, 1, 1, 1, 1, 1], &[5, 1, 3, 2]] {
                let routes: usize = buckets.iter().sum();
                let layout = RoutedOutputLayout::new(routes, width).unwrap();
                let packed_bf16 = grant(layout.elements() * 2);
                // All buckets survive until the same outer fence.
                let activations: usize = buckets
                    .iter()
                    .map(|&m| grant(m * width * 2) + 3 * grant(m * inner * 2))
                    .sum();
                let metadata = 2 * grant(routes * 4);
                let final_output = grant(5 * width * 2);
                // Pair GEMM scratch/tail and provider stay live, not dropped to
                // manufacture a peak saving. Decoded weight leases are unchanged.
                let tail_workspace = grant(5 * 2 * inner * 2) + (16 << 20);
                let common = activations + metadata + final_output + tail_workspace;
                let old_down_and_expanded: usize = buckets
                    .iter()
                    .map(|&m| grant(m * width * 2) + grant(m * width * 4))
                    .sum();
                let old_peak =
                    common + packed_bf16 + grant(routes * width * 4) + old_down_and_expanded;
                let new_peak = common + packed_bf16;
                let gate_up: usize = buckets.iter().map(|&m| 2 * grant(m * inner * 2)).sum();
                let fused_peak = new_peak - gate_up;
                assert!(fused_peak < new_peak);
                assert!(new_peak < old_peak);
                assert_eq!(
                    old_peak - new_peak,
                    grant(routes * width * 4) + old_down_and_expanded
                );
            }
        }
    }
}

#[test]
fn fusion_policy_requires_routed_native_pair_and_no_observer() {
    for capability in [false, true] {
        for observer in [false, true] {
            assert_eq!(
                output::fuse_pair_swiglu(capability, observer),
                capability && !observer
            );
        }
    }
}

#[test]
fn pack_nonfinite_contract_excludes_nan_payload_equality() {
    for word in [0x0000, 0x8000, 0x7f80, 0xff80] {
        assert_eq!(
            bf16::from_f32(bf16::from_bits(word).to_f32()).to_bits(),
            word
        );
    }
    // Native bf16_rne sets the quiet bit. Direct BF16 down-output packing is
    // a view, not a floating-point conversion; it leaves an sNaN word alone.
    for word in [0x7f81u16, 0x7fbf, 0xff81, 0xffbf] {
        let expanded = f32::from_bits(u32::from(word) << 16);
        let repacked = bf16::from_f32(expanded).to_bits();
        assert!(bf16::from_bits(repacked).is_nan());
        assert_ne!(repacked & 0x0040, 0);
        assert_ne!(repacked, word);
        assert!(bf16::from_bits(word).is_nan());
    }
}

#[path = "support/mixed_route_batch_frozen.rs"]
mod route_fixture;
use output::{RoutedBatchPlan, RoutedBindingSlot, RoutedBindingSlots};
use std::collections::BTreeMap;

#[test]
fn prepared_slots_keep_original_vec_indices_and_reject_foreign_generation() {
    let original = [
        Some((9, 0, 1)),
        Some((3, 0, 1)),
        Some((3, 1, 0)),
        None,
        Some((3, 1, 2)),
        None,
        Some((3, 0, 2)),
        Some((9, 0, 2)),
        Some((3, 0, 0)),
        Some((9, 0, 0)),
        Some((3, 1, 1)),
        None,
    ];
    let slots = RoutedBindingSlots::prepare(
        19,
        original.len(),
        &BTreeMap::from([(3, 2), (9, 1)]),
        original.iter().enumerate().map(|(slot, item)| {
            Ok(item.map(|(layer, expert, role)| RoutedBindingSlot {
                slot,
                layer,
                expert,
                role,
            }))
        }),
    )
    .unwrap();
    assert_eq!(slots.generation(), 19);
    for (layer, expert, expected) in [(3, 0, [8, 1, 6]), (3, 1, [2, 10, 4]), (9, 0, [9, 0, 7])] {
        assert_eq!(slots.get(19, layer, expert).unwrap(), expected);
        for (role, slot) in expected.into_iter().enumerate() {
            assert_eq!(original[slot], Some((layer, expert, role)));
        }
    }
    for (generation, layer, expert) in [(18, 3, 0), (20, 3, 0), (19, 8, 0), (19, 3, 2)] {
        assert!(slots.get(generation, layer, expert).is_err());
    }
}

#[test]
fn prepared_slots_reject_holes_duplicates_and_all_out_of_bounds() {
    let valid = (0..3)
        .map(|role| RoutedBindingSlot {
            slot: role,
            layer: 7,
            expert: 0,
            role,
        })
        .collect::<Vec<_>>();
    let mut cases = vec![valid[..2].to_vec()];
    let mut duplicate_role = valid.clone();
    duplicate_role.push(RoutedBindingSlot {
        slot: 3,
        ..valid[0]
    });
    cases.push(duplicate_role);
    for replacement in [
        RoutedBindingSlot {
            slot: 0,
            ..valid[1]
        },
        RoutedBindingSlot {
            slot: 4,
            ..valid[1]
        },
        RoutedBindingSlot {
            layer: 8,
            ..valid[1]
        },
        RoutedBindingSlot {
            expert: 1,
            ..valid[1]
        },
        RoutedBindingSlot {
            role: 3,
            ..valid[1]
        },
    ] {
        let mut case = valid.clone();
        case[1] = replacement;
        cases.push(case);
    }
    for bindings in cases {
        assert!(
            RoutedBindingSlots::prepare(
                1,
                4,
                &BTreeMap::from([(7, 1)]),
                bindings.into_iter().map(|b| Ok(Some(b))),
            )
            .is_err()
        );
    }
    assert!(RoutedBindingSlots::prepare(1, 0, &BTreeMap::from([(7, 0)]), []).is_err());
    assert!(RoutedBindingSlots::prepare(1, 0, &BTreeMap::from([(7, usize::MAX)]), []).is_err());
    let source_error = Err(ferrule_common::Error::Model {
        message: "foreign prepared source".into(),
    });
    assert!(RoutedBindingSlots::prepare(1, 3, &BTreeMap::from([(7, 1)]), [source_error]).is_err());
}

fn legacy_buckets(rows: usize, ids: &[usize], weights: &[f32]) -> BTreeMap<usize, Vec<(i32, u32)>> {
    // Independent pre-batch row-major bucket construction, not planner accessors.
    let mut buckets = BTreeMap::<usize, Vec<(i32, u32)>>::new();
    let per_row = ids.len() / rows;
    for row in 0..rows {
        for slot in 0..per_row {
            let index = row * per_row + slot;
            buckets
                .entry(ids[index])
                .or_default()
                .push((row as i32, weights[index].to_bits()));
        }
    }
    buckets
}

#[test]
fn batch_matches_legacy_m1_m5_64_routes_overlap_and_decode_shapes_and_bits() {
    for (rows, per_row) in [(1, 8), (1, 64), (5, 8), (8, 8), (1, 8)] {
        let ids = (0..rows)
            .flat_map(|row| (0..per_row).map(move |slot| (slot * 37 + row * 11) % 256))
            .collect::<Vec<_>>();
        let weights = (0..ids.len())
            .map(|i| {
                f32::from_bits([0x80000000, 0x00000000, 0x00000001, 0x3dcd0e56, 0x3f000001][i % 5])
            })
            .collect::<Vec<_>>();
        let old = legacy_buckets(rows, &ids, &weights);
        for width in [3, 256, 4096] {
            let batch = RoutedBatchPlan::new(rows, width, 256, &ids, &weights).unwrap();
            assert_eq!(batch.layout.elements(), ids.len() * width);
            assert_eq!(batch.row_ids.capacity(), ids.len());
            assert_eq!(batch.weights.capacity(), ids.len());
            assert_eq!(batch.groups.capacity(), old.len());
            let input = (0..rows * width).map(|i| i as u16).collect::<Vec<_>>();
            let batch_gather = batch
                .row_ids
                .iter()
                .flat_map(|&row| {
                    input[row as usize * width..(row as usize + 1) * width]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            let mut begin = 0;
            for (group, (&expert, bucket)) in batch.groups.iter().zip(&old) {
                assert_eq!(
                    (group.expert, group.begin, group.count),
                    (expert, begin, bucket.len())
                );
                let old_rows = bucket.iter().map(|&(row, _)| row).collect::<Vec<_>>();
                let old_bits = bucket.iter().map(|&(_, bits)| bits).collect::<Vec<_>>();
                let end = begin + bucket.len();
                assert_eq!(batch.row_ids[begin..end], old_rows);
                assert_eq!(
                    batch.weights[begin..end]
                        .iter()
                        .map(|w| w.to_bits())
                        .collect::<Vec<_>>(),
                    old_bits
                );
                let old_gather = old_rows
                    .iter()
                    .flat_map(|&row| {
                        input[row as usize * width..(row as usize + 1) * width]
                            .iter()
                            .copied()
                    })
                    .collect::<Vec<_>>();
                let range = batch.layout.bucket(group.begin, group.count).unwrap();
                assert_eq!(batch_gather[range.clone()], old_gather);
                for inner in [128, 1024] {
                    assert_eq!(
                        [group.count, 2 * inner, width],
                        [bucket.len(), 2 * inner, width]
                    );
                    assert_eq!([group.count, width, inner], [bucket.len(), width, inner]);
                }
                assert_eq!(range, begin * width..end * width);
                begin = end;
            }
            assert_eq!(begin, ids.len());
        }
    }
}

#[test]
fn independently_frozen_raw_bf16_gather_and_route_bits_match_without_conversion() {
    use route_fixture::*;
    let weights = WEIGHT_BITS.map(f32::from_bits);
    let batch = RoutedBatchPlan::new(ROWS, WIDTH, 10, &IDS, &weights).unwrap();
    assert_eq!(
        batch
            .groups
            .iter()
            .map(|g| (g.expert, g.begin, g.count))
            .collect::<Vec<_>>(),
        GROUPS
    );
    assert_eq!(batch.row_ids, ORDERED_ROWS);
    assert_eq!(
        batch
            .weights
            .iter()
            .map(|w| w.to_bits())
            .collect::<Vec<_>>(),
        ORDERED_WEIGHT_BITS
    );
    let gathered = batch
        .row_ids
        .iter()
        .flat_map(|&row| {
            INPUT[row as usize * WIDTH..(row as usize + 1) * WIDTH]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(gathered, GATHERED);
    let old = legacy_buckets(ROWS, &IDS, &weights);
    let old_gathered = old
        .values()
        .flatten()
        .flat_map(|&(row, _)| {
            INPUT[row as usize * WIDTH..(row as usize + 1) * WIDTH]
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(old_gathered, GATHERED);
}

#[test]
fn batch_validation_rejects_invalid_routes_before_device_work() {
    for (rows, width, experts, ids, weights) in [
        (0, 3, 2, vec![0], vec![1.]),
        (1, 0, 2, vec![0], vec![1.]),
        (1, 3, 0, vec![0], vec![1.]),
        (1, 3, 2, vec![], vec![]),
        (1, 3, 2, vec![0], vec![]),
        (2, 3, 2, vec![0, 1, 0], vec![1.; 3]),
        (1, 3, 2, vec![2], vec![1.]),
        (1, 3, 2, vec![0, 0], vec![0.5; 2]),
        (1, 3, 2, vec![0], vec![-0.1]),
        (1, 3, 2, vec![0], vec![f32::NAN]),
        (1, 3, 2, vec![0], vec![f32::INFINITY]),
        (i32::MAX as usize + 1, 3, 2, vec![0], vec![1.]),
    ] {
        assert!(RoutedBatchPlan::new(rows, width, experts, &ids, &weights).is_err());
    }
}

#[test]
fn batch_gather_equal_payload_uses_actual_single_rounded_grants() {
    for counts in [&[1usize][..], &[1; 64], &[5, 1, 3, 2], &[2; 32]] {
        let routes = counts.iter().sum::<usize>();
        for width in [3, 256, 4096] {
            let old_payload = counts.iter().map(|&m| m * width * 2).sum::<usize>();
            assert_eq!(routes * width * 2, old_payload);
            let old_gather = counts.iter().map(|&m| grant(m * width * 2)).sum::<usize>();
            let new_gather = grant(routes * width * 2);
            let old_meta = 2 * grant(routes * 4);
            let new_meta = grant(routes * 8);
            // Keep packed output, pair-tail, provider, decoded pins and all other
            // activations identical: no early release is used for this bound.
            assert!(new_meta <= old_meta);
            // Power-of-two size classes are not subadditive. The neutral ledger
            // must cover the real batch grant, not claim per-bucket grant savings.
            let route_envelope = grant(routes * width * 16) + counts.len() * 12 * 256 + new_meta;
            assert!(new_gather + new_meta <= route_envelope);
            if counts == [5, 1, 3, 2] && width == 4096 {
                assert_eq!(old_gather, 122880);
                assert_eq!(new_gather, 131072);
                assert_eq!(new_gather - old_gather, 8192);
            }
        }
    }
    assert_eq!(grant(64 * 8), 512);
    assert_eq!(32 * std::mem::size_of::<output::RoutedGroup>(), 768);
    assert_eq!(
        std::mem::size_of::<RoutedBindingSlots>(),
        4 * std::mem::size_of::<usize>()
    );
    assert_eq!(256 * std::mem::size_of::<[usize; 3]>(), 6144);
}

#[test]
fn prepared_source_and_runtime_index_contracts_remain_separate() {
    let code = include_str!("../src/transformer/standard/cuda/hybrid_rank/mixed.rs");
    let prepare = code
        .split("pub(super) fn prepare_mixed_rank_block_pending")
        .nth(1)
        .unwrap()
        .split("fn mixed_projection")
        .next()
        .unwrap();
    for guard in [
        "p.validate_binding(b, &plan.layout)?",
        "self.bindings.validate_metadata(b)?",
        "n.plan().validate_binding(b, &plan.layout)?",
        "b.weight().source_is_current()",
        "scale.source_is_current()",
        "RoutedBindingSlots::prepare",
    ] {
        assert!(prepare.contains(guard), "missing {guard}");
    }
    let runtime = code
        .split("pub fn mixed_rank_routed_local")
        .nth(1)
        .unwrap()
        .split("/// Replicated scalar gate")
        .next()
        .unwrap();
    assert!(!runtime.contains(".position("));
    assert!(!runtime.contains("source_is_current"));
    assert!(!runtime.contains("validate_metadata"));
    assert!(runtime.contains("image.routed_slots.generation()"));
    assert!(code.contains(".get(token.slot)"));
    assert!(code.contains(".get(tokens[2].slot)"));
}

#[test]
fn opaque_api_bounds_do_not_change_the_original_shape_contract_or_observer_choice() {
    for (rows, routes, supported) in [
        (1, 8, true),
        (5, 40, true),
        (8, 64, true),
        (1, 64, false),
        (9, 9, false),
        (32, 256, false),
        (0, 8, false),
        (1, 0, false),
    ] {
        assert_eq!(output::batch_metadata_supported(rows, routes), supported);
    }
    let code = include_str!("../src/transformer/standard/cuda/hybrid_rank/mixed.rs");
    let runtime = code
        .split("pub fn mixed_rank_routed_local")
        .nth(1)
        .unwrap()
        .split("/// Replicated scalar gate")
        .next()
        .unwrap();
    assert!(runtime.contains("gathered.slice(range.start, range.len())?"));
    assert_eq!(runtime.matches("gather_bf16_routes_into(").count(), 1);
    assert_eq!(runtime.matches("prepare_bf16_route_metadata(").count(), 1);
    assert_eq!(
        runtime
            .matches("weighted_combine_bf16_routes_into(")
            .count(),
        1
    );
    assert!(!runtime.contains("diagnostic.is_enabled"));
    let install = runtime.find("image.pending_routes.push(metadata)").unwrap();
    let prepare = runtime.find("prepare_bf16_route_metadata").unwrap();
    let hold = runtime.find("this.hold_bf16(&gathered)?").unwrap();
    let gather = runtime.find("gather_bf16_routes_into").unwrap();
    assert!(install < prepare && prepare < hold && hold < gather);
    assert!(runtime.contains("this.mixed_numeric_completion(false, result)?"));
    assert!(code.contains("this.mixed_decoded_enabled |= !plan.expert_counts.is_empty()"));
    let complete = code
        .split("fn complete_mixed_decoded")
        .nth(1)
        .unwrap()
        .split("/// Cache-only")
        .next()
        .unwrap();
    assert!(
        complete.find("cache.complete(succeeded)?").unwrap()
            < complete.find("image.pending_routes.clear()").unwrap()
    );
}
