use super::*;
use crate::cuda::ffi::core::*;

#[test]
fn native_bf16_guards_reject_invalid_descriptors_without_cuda() {
    // Every descriptor is invalid: no context acquisition/driver call is reached.
    let gate = DataArgs {
        kind: DATA_SIGMOID_GATE_BF16,
        count: 15,
        rows: 3,
        width: 5,
        input0: 2,
        input1: 4,
        output0: 6,
        ..Default::default()
    };
    for bad in [
        DataArgs { count: 0, ..gate },
        DataArgs { count: 14, ..gate },
        DataArgs {
            count: u32::MAX,
            ..gate
        },
        DataArgs { rows: 0, ..gate },
        DataArgs {
            rows: u32::MAX,
            ..gate
        },
        DataArgs { width: 0, ..gate },
        DataArgs {
            width: u32::MAX,
            ..gate
        },
        DataArgs { flags: 2, ..gate },
        DataArgs {
            reserved: 1,
            ..gate
        },
        DataArgs { input0: 0, ..gate },
        DataArgs { input1: 0, ..gate },
        DataArgs { output0: 0, ..gate },
        DataArgs { input0: 3, ..gate },
        DataArgs { input1: 3, ..gate },
        DataArgs { output0: 3, ..gate },
    ] {
        assert_ne!(unsafe { ferrule_core_data_launch(&bad) }, 0);
    }
    let combine = MoeArgs {
        kind: MOE_WEIGHTED_COMBINE_BF16,
        tokens: 3,
        hidden: 5,
        batch_columns: 4,
        input: 2,
        route_slots: 4,
        route_weights: 8,
        output: 6,
        ..Default::default()
    };
    for bad in [
        MoeArgs {
            tokens: 0,
            ..combine
        },
        MoeArgs {
            hidden: 0,
            ..combine
        },
        MoeArgs {
            batch_columns: 0,
            ..combine
        },
        MoeArgs {
            tokens: u32::MAX,
            ..combine
        },
        MoeArgs {
            hidden: u32::MAX,
            ..combine
        },
        MoeArgs {
            batch_columns: u32::MAX,
            ..combine
        },
        MoeArgs {
            input: 0,
            ..combine
        },
        MoeArgs {
            route_slots: 0,
            ..combine
        },
        MoeArgs {
            route_weights: 0,
            ..combine
        },
        MoeArgs {
            output: 0,
            ..combine
        },
        MoeArgs {
            input: 3,
            ..combine
        },
        MoeArgs {
            route_slots: 2,
            ..combine
        },
        MoeArgs {
            route_weights: 2,
            ..combine
        },
        MoeArgs {
            output: 3,
            ..combine
        },
    ] {
        assert_ne!(unsafe { ferrule_core_moe_launch(&bad) }, 0);
    }
}

#[test]
#[ignore = "requires sole-owner GPU assignment; real allocation slices and grant accounting"]
fn writable_aliases_rejected_and_read_aliases_keep_grants() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let root = op.upload_bf16_words(&[0x3f80; 128]).unwrap();
    let view = |offset, len| {
        CudaBf16Buffer::from_device_buffer(root.as_device_buffer().slice(offset, len).unwrap())
    };
    for layout in [
        Bf16GateLayout::Elementwise { rows: 3, width: 5 },
        Bf16GateLayout::RowBroadcast { rows: 3, width: 5 },
    ] {
        for offset in [0, 1, 14, 32, 33] {
            let input = view(0, 15);
            let gate = view(32, layout.gate_elements().unwrap());
            let mut output = view(offset, 15);
            op.reset_counters();
            assert!(
                op.sigmoid_gate_bf16_into(&input, &gate, &mut output, layout)
                    .is_err()
            );
            assert_eq!(op.counters(), Default::default());
            assert_eq!(op.download_bf16_buffer(&root).unwrap(), vec![1.0; 128]);
        }
    }
    let rows = op.upload_i32_buffer(&[0, 1, 2]).unwrap();
    let weights = op.upload_f32_buffer(&[0.3, 0.5, 0.7]).unwrap();
    for offset in [0, 1, 14] {
        let values = view(0, 15);
        let mut output = view(offset, 15);
        op.reset_counters();
        assert!(
            op.weighted_combine_bf16_into(&values, &rows, &weights, &mut output, 3, 5)
                .is_err()
        );
        assert_eq!(op.counters(), Default::default());
        assert_eq!(op.download_bf16_buffer(&root).unwrap(), vec![1.0; 128]);
    }
    let input = view(0, 15);
    let gate = view(0, 15);
    let mut output = view(15, 15); // adjacent, not overlapping
    let before = op.allocator_metrics();
    op.failpoints().arm_allocation();
    op.reset_counters();
    {
        let _scope = op.scoped_capture_safe();
        op.sigmoid_gate_bf16_into(
            &input,
            &gate,
            &mut output,
            Bf16GateLayout::Elementwise { rows: 3, width: 5 },
        )
        .unwrap();
        op.weighted_combine_bf16_into(&input, &rows, &weights, &mut output, 3, 5)
            .unwrap();
    }
    assert_eq!(op.counters().kernel_launches, 2);
    assert_eq!(op.counters().device_allocation_attempts, 0);
    let after = op.allocator_metrics();
    assert_eq!(before.granted_bytes, after.granted_bytes);
    assert_eq!(before.live_granted_bytes, after.live_granted_bytes);
    // Neither operation consumed the armed allocation failure.
    assert!(op.zero_bf16_buffer(1).is_err());
    op.download_bf16_buffer(&output).unwrap();
}

#[test]
fn legacy_and_plan_gate_paths_are_the_same_type() {
    let plan = crate::cuda_plan::Bf16GateLayout::RowBroadcast { rows: 3, width: 5 };
    let canonical: Bf16GateLayout = plan;
    let legacy: crate::cuda::context::standard_bf16::Bf16GateLayout = canonical;
    assert_eq!(legacy, plan);
}

#[test]
#[ignore = "sole GPU owner; ordinary offset BF16 norm slices and capture-safe resources"]
fn ordinary_offset_norm_typed_aliases_and_no_hidden_allocation() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let root = op.upload_bf16_words(&[0x3f80; 64]).unwrap();
    let view = |offset, len| {
        CudaBf16Buffer::from_device_buffer(root.as_device_buffer().slice(offset, len).unwrap())
    };
    let input = view(0, 15);
    let weight = view(32, 5);
    let layout = crate::cuda_plan::F32RowsLayout { rows: 3, width: 5 };
    for offset in [0, 1, 14, 32, 33] {
        let mut output = view(offset, 15);
        op.reset_counters();
        assert!(
            op.offset_rms_norm_bf16_into(&input, &weight, &mut output, layout, 1e-6)
                .is_err()
        );
        assert_eq!(op.counters(), Default::default());
        assert_eq!(op.download_bf16_buffer(&root).unwrap(), vec![1.0; 64]);
    }
    let weight = view(0, 5);
    let mut output = view(15, 15);
    op.failpoints().arm_allocation();
    op.reset_counters();
    {
        let _scope = op.scoped_capture_safe();
        op.offset_rms_norm_bf16_into(&input, &weight, &mut output, layout, 1e-6)
            .unwrap();
    }
    let counters = op.counters();
    assert_eq!(counters.kernel_launches, 1);
    assert_eq!(counters.device_allocation_attempts, 0);
    assert_eq!(counters.host_to_device_copies, 0);
    assert_eq!(counters.device_to_host_copies, 0);
    assert_eq!(counters.stream_wide_syncs, 0);
    assert!(op.zero_bf16_buffer(1).is_err());
    assert_eq!(op.download_bf16_buffer(&output).unwrap(), vec![2.0; 15]);
}

fn route_test_words(op: &CudaOperators, buffer: &CudaBf16Buffer) -> Vec<u16> {
    buffer
        .as_device_buffer()
        .to_host_vec(&op.stream_clone())
        .unwrap()
}
fn route_test_rne(value: f32) -> u16 {
    let bits = value.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

#[test]
#[ignore = "GPU requires main authorization; old E gathers vs one batch, exact bits/canaries"]
fn packed_routes_gpu_old_expert_gathers_and_combine_match_batch() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let cases = [
        (1, 37, vec![vec![0]; 8]),
        (1, 1, vec![vec![0]]),
        (5, 5, vec![vec![0, 1, 2, 3, 4], vec![4, 1, 3, 0, 2]]),
        (5, 257, vec![vec![0], vec![1], vec![2], vec![3], vec![4]]),
        (
            5,
            127,
            (0..8)
                .map(|expert| {
                    (0..5)
                        .map(|rank| ((4 + expert + 2 * rank) % 5) as i32)
                        .collect()
                })
                .collect(),
        ),
        // Top8 with entirely disjoint expert sets across all five input rows.
        (
            5,
            37,
            (0..40).map(|expert| vec![(expert / 8) as i32]).collect(),
        ),
        (
            8,
            129,
            (0..8)
                .map(|expert| {
                    (0..8)
                        .map(|rank| ((7 + expert + 3 * rank) % 8) as i32)
                        .collect()
                })
                .collect(),
        ),
    ];
    for (rows, width, experts) in cases {
        // Expert groups are already ascending expert index. Within each expert,
        // deliberately retain nonmonotonic row order; do not repartition GEMMs.
        let ids: Vec<i32> = experts.iter().flatten().copied().collect();
        let routes = ids.len();
        let weights: Vec<f32> = (0..routes)
            .map(|i| [0.33333334, 0.7, 0.125, -0.0, 0.99999994][i % 5])
            .collect();
        let input_words: Vec<u16> = (0..rows * width)
            .map(|i| {
                if i % 17 == 0 {
                    0x8000
                } else {
                    0x3c00 + (i % 1024) as u16
                }
            })
            .collect();
        let input = op.upload_bf16_words(&input_words).unwrap();
        let projections: Vec<u16> = (0..routes * width)
            .map(|i| {
                route_test_rne([16.0, 0.03125, -16.0, 0.5, -0.3, 1.0][(i / width + i % width) % 6])
            })
            .collect();
        let values = op.upload_bf16_words(&projections).unwrap();
        let mut old_output = op.zero_bf16_buffer(rows * width).unwrap();
        op.reset_counters();
        let old_gathers: Vec<_> = experts
            .iter()
            .map(|expert| {
                let ids: Vec<_> = expert.iter().map(|&row| row as u32).collect();
                op.embedding_bf16(&input, &ids, width).unwrap()
            })
            .collect();
        let old_ids = op.upload_i32_buffer(&ids).unwrap();
        let old_weights = op.upload_f32_buffer(&weights).unwrap();
        op.weighted_combine_bf16_into(
            &values,
            &old_ids,
            &old_weights,
            &mut old_output,
            rows,
            width,
        )
        .unwrap();
        let old_counts = op.counters();
        assert_eq!(old_counts.kernel_launches, experts.len() as u64 + 1);
        assert_eq!(old_counts.host_to_device_copies, experts.len() as u64 + 2);
        assert_eq!(old_counts.host_to_device_bytes, 12 * routes as u64);
        let old_gather_words: Vec<_> = old_gathers
            .iter()
            .flat_map(|gather| route_test_words(&op, gather))
            .collect();
        let old_combined = route_test_words(&op, &old_output);

        let canary = 0x4abc;
        // Caller installs roots/custody before submitting metadata initialization.
        let mut metadata = op
            .allocate_bf16_route_metadata(&ids, &weights, rows)
            .unwrap();
        let gathered_root = op
            .upload_bf16_words(&vec![canary; routes * width + 4])
            .unwrap();
        let combined_root = op
            .upload_bf16_words(&vec![canary; rows * width + 4])
            .unwrap();
        let mut gathered = gathered_root.slice(2, routes * width).unwrap();
        let mut combined = combined_root.slice(2, rows * width).unwrap();
        assert!(!metadata.is_prepared());
        op.reset_counters();
        assert!(
            op.gather_bf16_routes_into(&input, &metadata, &mut gathered, width)
                .is_err()
        );
        assert_eq!(op.counters(), Default::default());
        op.prepare_bf16_route_metadata(&mut metadata).unwrap();
        assert!(metadata.is_prepared());
        assert_eq!(metadata.row_ids().collect::<Vec<_>>(), ids);
        assert_eq!(
            metadata.weight_bits(),
            weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(op.prepare_bf16_route_metadata(&mut metadata).is_err());
        assert!(metadata.is_prepared());
        op.failpoints().arm_allocation();
        let before = op.allocator_metrics();
        {
            let _scope = op.scoped_capture_safe();
            op.gather_bf16_routes_into(&input, &metadata, &mut gathered, width)
                .unwrap();
            op.weighted_combine_bf16_routes_into(&values, &metadata, &mut combined, width)
                .unwrap();
        }
        let counts = op.counters();
        assert_eq!(counts.kernel_launches, 2);
        assert_eq!(counts.host_to_device_copies, 1);
        assert_eq!(counts.host_to_device_bytes, 8 * routes as u64);
        assert_eq!(counts.device_allocation_attempts, 0);
        assert_eq!(counts.stream_wide_syncs, 0);
        assert_eq!(
            before.live_granted_bytes,
            op.allocator_metrics().live_granted_bytes
        );
        assert!(op.zero_bf16_buffer(1).is_err()); // Hot calls did not consume failpoint.

        assert_eq!(route_test_words(&op, &gathered), old_gather_words);
        assert_eq!(route_test_words(&op, &combined), old_combined);
        let expected_gather: Vec<_> = ids
            .iter()
            .flat_map(|&row| {
                input_words[row as usize * width..(row as usize + 1) * width]
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(old_gather_words, expected_gather);
        let mut route_offset = 0;
        for (expert, old) in experts.iter().zip(&old_gathers) {
            let view = gathered
                .slice(route_offset * width, expert.len() * width)
                .unwrap();
            assert_eq!(view.len(), expert.len() * width); // Unchanged per-expert M.
            assert_eq!(route_test_words(&op, &view), route_test_words(&op, old));
            route_offset += expert.len();
        }
        let mut expected_combine = vec![0u16; rows * width];
        for (route, &row) in ids.iter().enumerate() {
            for column in 0..width {
                let index = row as usize * width + column;
                let projection = f32::from_bits((projections[route * width + column] as u32) << 16);
                let product = route_test_rne(projection * weights[route]);
                let prior = f32::from_bits((expected_combine[index] as u32) << 16);
                expected_combine[index] =
                    route_test_rne(prior + f32::from_bits((product as u32) << 16));
            }
        }
        assert_eq!(old_combined, expected_combine);
        for (root, len) in [
            (&gathered_root, routes * width),
            (&combined_root, rows * width),
        ] {
            let words = route_test_words(&op, root);
            assert_eq!(&words[..2], &[canary; 2]);
            assert_eq!(&words[len + 2..], &[canary; 2]);
        }
        let device_words = metadata
            .device
            .as_device_buffer()
            .to_host_vec(&op.stream_clone())
            .unwrap();
        assert_eq!(device_words, metadata.packed.words.as_ref());
        // Existing completion fence only; every metadata/source/root is still held.
        op.sync_stream().unwrap();
    }
}

#[test]
#[ignore = "GPU requires main authorization; real same-ordinal owners, aliases and H2D fault"]
fn packed_routes_gpu_reject_invalid_owners_aliases_and_after_h2d_fault() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let root = op.upload_bf16_words(&[0x3f80; 40]).unwrap();
    let input = root.slice(0, 10).unwrap();
    let mut metadata = op
        .allocate_bf16_route_metadata(&[1, 0], &[0.3, 0.7], 2)
        .unwrap();
    op.prepare_bf16_route_metadata(&mut metadata).unwrap();
    for offset in [0, 1, 9] {
        let mut output = root.slice(offset, 10).unwrap();
        op.reset_counters();
        assert!(
            op.gather_bf16_routes_into(&input, &metadata, &mut output, 5)
                .is_err()
        );
        assert!(
            op.weighted_combine_bf16_routes_into(&input, &metadata, &mut output, 5)
                .is_err()
        );
        assert_eq!(op.counters(), Default::default());
    }
    let mut output = root.slice(10, 10).unwrap();
    let foreign_input = foreign.upload_bf16_words(&[0x3f80; 10]).unwrap();
    let mut foreign_output = foreign.zero_bf16_buffer(10).unwrap();
    let mut foreign_metadata = foreign
        .allocate_bf16_route_metadata(&[1, 0], &[0.3, 0.7], 2)
        .unwrap();
    foreign
        .prepare_bf16_route_metadata(&mut foreign_metadata)
        .unwrap();
    for combine in [false, true] {
        let call = |input, metadata, output: &mut CudaBf16Buffer| {
            if combine {
                op.weighted_combine_bf16_routes_into(input, metadata, output, 5)
            } else {
                op.gather_bf16_routes_into(input, metadata, output, 5)
            }
        };
        op.reset_counters();
        assert!(call(&foreign_input, &metadata, &mut output).is_err());
        assert!(call(&input, &foreign_metadata, &mut output).is_err());
        assert!(call(&input, &metadata, &mut foreign_output).is_err());
        let short = root.slice(20, 9).unwrap();
        let mut long = root.slice(20, 11).unwrap();
        assert!(call(&short, &metadata, &mut output).is_err());
        assert!(call(&input, &metadata, &mut long).is_err());
        assert_eq!(op.counters(), Default::default());
    }
    assert!(
        op.prepare_bf16_route_metadata(&mut foreign_metadata)
            .is_err()
    );
    let mut failed = op
        .allocate_bf16_route_metadata(&[1, 0], &[0.3, 0.7], 2)
        .unwrap();
    // Inject after actual H2D, using the production publication state machine.
    assert!(
        failed
            .preparation
            .submit(
                || op.upload_bf16_route_words(&failed.packed.words, &failed.device),
                || Err(invalid("injected after-H2D completion Unknown")),
            )
            .is_err()
    );
    assert!(!failed.is_prepared());
    op.reset_counters();
    assert!(op.prepare_bf16_route_metadata(&mut failed).is_err());
    assert!(
        op.gather_bf16_routes_into(&input, &failed, &mut output, 5)
            .is_err()
    );
    assert!(
        op.weighted_combine_bf16_routes_into(&input, &failed, &mut output, 5)
            .is_err()
    );
    assert_eq!(op.counters(), Default::default());
    op.sync_stream().unwrap(); // Original fence; failed metadata was retained throughout.
}

#[test]
#[ignore = "GPU requires main authorization; finite F32 contract and uploaded bits"]
fn packed_routes_gpu_finite_bits_and_preflight_no_work() {
    let op = CudaOperators::new_on_device(0).unwrap();
    for weight in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        op.reset_counters();
        assert!(op.allocate_bf16_route_metadata(&[0], &[weight], 1).is_err());
        assert_eq!(op.counters(), Default::default());
    }
    for (ids, weights, rows) in [
        (vec![], vec![], 1),
        (vec![0], vec![], 1),
        (vec![-1], vec![1.0], 1),
        (vec![1], vec![1.0], 1),
        (vec![0; 9], vec![1.0; 9], 2),
        (vec![0; 65], vec![1.0; 65], 8),
        (vec![0], vec![1.0], 9),
    ] {
        op.reset_counters();
        assert!(
            op.allocate_bf16_route_metadata(&ids, &weights, rows)
                .is_err()
        );
        assert_eq!(op.counters(), Default::default());
    }
    let bits = [
        0, 0x80000000, 1, 0x80000001, 0x3eaaaaab, 0x7f7fffff, 0xff7fffff, 0x00800000,
    ];
    let mut metadata = op
        .allocate_bf16_route_metadata(&[0; 8], &bits.map(f32::from_bits), 1)
        .unwrap();
    op.reset_counters();
    op.prepare_bf16_route_metadata(&mut metadata).unwrap();
    assert_eq!(op.counters().host_to_device_copies, 1);
    assert_eq!(op.counters().host_to_device_bytes, 64);
    assert_eq!(op.counters().kernel_launches, 0);
    let words = metadata
        .device
        .as_device_buffer()
        .to_host_vec(&op.stream_clone())
        .unwrap();
    assert_eq!(words[..8], [0; 8]);
    assert_eq!(words[8..], bits);
    assert_eq!(metadata.weight_bits(), bits);
    op.sync_stream().unwrap(); // Metadata/source held through the original fence.
}
