use ferrule_common::ParallelRankId;
use ferrule_model::transformer::parallel::{
    TensorParallelLinearPartition as Partition, TensorParallelLinearPlan,
};

fn rank(value: usize) -> ParallelRankId {
    ParallelRankId::new(u32::try_from(value).unwrap())
}

fn matmul(weight: &[f32], input: &[f32], rows: usize, out: usize, width: usize) -> Vec<f32> {
    assert_eq!(weight.len(), out * width);
    assert_eq!(input.len(), rows * width);
    let mut result = vec![0.0; rows * out];
    for row in 0..rows {
        for output in 0..out {
            for feature in 0..width {
                result[row * out + output] +=
                    input[row * width + feature] * weight[output * width + feature];
            }
        }
    }
    result
}

fn fixture(out: usize, width: usize, rows: usize) -> (Vec<f32>, Vec<f32>) {
    let weight = (0..out * width)
        .map(|i| (i as i32 % 17 - 8) as f32 * 0.125)
        .collect();
    let input = (0..rows * width)
        .map(|i| (i as i32 % 11 - 5) as f32 * 0.25)
        .collect();
    (weight, input)
}

fn cpu_partials(
    plan: &TensorParallelLinearPlan,
    weight: &[f32],
    input: &[f32],
    rows: usize,
) -> Vec<Vec<f32>> {
    (0..plan.ranks())
        .map(|r| {
            let (local_weight, range) = plan.shard_weight(rank(r), weight).unwrap();
            assert_eq!(range, plan.rank_range(rank(r)).unwrap());
            let local_input = plan.shard_input(rank(r), input, rows).unwrap();
            let (out, width) = plan.local_shape(rank(r)).unwrap();
            matmul(&local_weight, &local_input, rows, out, width)
        })
        .collect()
}

// This is deliberately a test-only oracle. Production Row reduction belongs to
// runtime HostCollective and must not be duplicated by the model plan.
fn row_sum_oracle(partials: &[Vec<f32>]) -> Vec<f32> {
    let mut output = vec![0.0; partials[0].len()];
    for partial in partials {
        assert_eq!(partial.len(), output.len());
        for (target, value) in output.iter_mut().zip(partial) {
            *target += value;
        }
    }
    output
}

fn output_oracle(plan: &TensorParallelLinearPlan, partials: &[Vec<f32>], rows: usize) -> Vec<f32> {
    match plan.partition() {
        Partition::Column => {
            let gathered: Vec<f32> = partials
                .iter()
                .enumerate()
                .flat_map(|(r, partial)| plan.pack_column_output(rank(r), partial, rows).unwrap())
                .collect();
            plan.unpack_column_gather(&gathered, rows).unwrap()
        }
        Partition::Row => row_sum_oracle(partials),
    }
}

#[test]
fn both_partitions_match_full_linear_for_odd_sizes_and_multiple_rows() {
    for (out, width, rows) in [(5, 7, 3), (7, 5, 4), (3, 3, 1), (1, 1, 2), (9, 11, 5)] {
        let (weight, input) = fixture(out, width, rows);
        let expected = matmul(&weight, &input, rows, out, width);
        for partition in [Partition::Column, Partition::Row] {
            let dimension = if partition == Partition::Column {
                out
            } else {
                width
            };
            for ranks in 1..=dimension {
                let plan = TensorParallelLinearPlan::new(out, width, ranks, partition).unwrap();
                let partials = cpu_partials(&plan, &weight, &input, rows);
                assert_eq!(
                    output_oracle(&plan, &partials, rows),
                    expected,
                    "{partition:?}, out={out}, in={width}, rows={rows}, ranks={ranks}"
                );
            }
        }
    }
}

#[test]
fn balanced_ranges_are_contiguous_nonempty_and_cover_the_split_dimension() {
    for partition in [Partition::Column, Partition::Row] {
        let plan = TensorParallelLinearPlan::new(7, 7, 3, partition).unwrap();
        assert_eq!(plan.rank_range(rank(0)).unwrap(), 0..3);
        assert_eq!(plan.rank_range(rank(1)).unwrap(), 3..5);
        assert_eq!(plan.rank_range(rank(2)).unwrap(), 5..7);
        assert_eq!(plan.partition(), partition);
        assert_eq!(plan.out_features(), 7);
        assert_eq!(plan.in_features(), 7);
        for ranks in 1..=7 {
            let plan = TensorParallelLinearPlan::new(7, 7, ranks, partition).unwrap();
            let ranges: Vec<_> = (0..ranks)
                .map(|r| plan.rank_range(rank(r)).unwrap())
                .collect();
            assert_eq!(ranges.first().unwrap().start, 0);
            assert_eq!(ranges.last().unwrap().end, 7);
            for pair in ranges.windows(2) {
                assert_eq!(pair[0].end, pair[1].start);
                assert!(pair[0].len() >= pair[1].len());
                assert!(pair[0].len() - pair[1].len() <= 1);
            }
            assert!(ranges.iter().all(|range| !range.is_empty()));
        }
    }
}

#[test]
fn row_weight_and_input_are_sliced_inside_every_row() {
    let plan = TensorParallelLinearPlan::new(3, 5, 2, Partition::Row).unwrap();
    let weight: Vec<_> = (0..15).map(|i| i as f32).collect();
    let input: Vec<_> = (20..30).map(|i| i as f32).collect();
    assert_eq!(
        plan.shard_weight(rank(0), &weight).unwrap(),
        (vec![0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 10.0, 11.0, 12.0], 0..3)
    );
    assert_eq!(
        plan.shard_weight(rank(1), &weight).unwrap(),
        (vec![3.0, 4.0, 8.0, 9.0, 13.0, 14.0], 3..5)
    );
    assert_eq!(
        plan.shard_input(rank(0), &input, 2).unwrap(),
        vec![20.0, 21.0, 22.0, 25.0, 26.0, 27.0]
    );
    assert_eq!(
        plan.shard_input(rank(1), &input, 2).unwrap(),
        vec![23.0, 24.0, 28.0, 29.0]
    );
    assert_eq!(plan.local_shape(rank(1)).unwrap(), (3, 2));
}

#[test]
fn column_weight_is_contiguous_and_input_is_complete() {
    let plan = TensorParallelLinearPlan::new(5, 3, 2, Partition::Column).unwrap();
    let weight: Vec<_> = (0..15).map(|i| i as f32).collect();
    let input: Vec<_> = (20..26).map(|i| i as f32).collect();
    assert_eq!(
        plan.shard_weight(rank(0), &weight).unwrap(),
        (weight[..9].to_vec(), 0..3)
    );
    assert_eq!(
        plan.shard_weight(rank(1), &weight).unwrap(),
        (weight[9..].to_vec(), 3..5)
    );
    for r in 0..2 {
        assert_eq!(plan.shard_input(rank(r), &input, 2).unwrap(), input);
    }
    assert_eq!(plan.local_shape(rank(1)).unwrap(), (2, 3));
}

#[test]
fn column_gather_interleaves_rows_instead_of_flattening_rank_major() {
    let plan = TensorParallelLinearPlan::new(5, 2, 2, Partition::Column).unwrap();
    let partials = vec![
        vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0],
        vec![13.0, 14.0, 23.0, 24.0, 33.0, 34.0],
    ];
    assert_eq!(plan.column_gather_width().unwrap(), 3);
    assert_eq!(plan.column_gather_count(3).unwrap(), 9);
    let first = plan.pack_column_output(rank(0), &partials[0], 3).unwrap();
    assert_eq!(first, partials[0]);
    let mut second = plan.pack_column_output(rank(1), &partials[1], 3).unwrap();
    assert_eq!(
        second,
        vec![13.0, 14.0, 0.0, 23.0, 24.0, 0.0, 33.0, 34.0, 0.0]
    );
    // Padding must be discarded, not added or copied into the logical output.
    for index in [2, 5, 8] {
        second[index] = f32::NAN;
    }
    let gathered = [first, second].concat();
    let actual = plan.unpack_column_gather(&gathered, 3).unwrap();
    assert_eq!(
        actual,
        vec![
            10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0, 30.0, 31.0, 32.0, 33.0,
            34.0
        ]
    );
    assert_ne!(actual, partials.concat());
}

#[test]
fn row_reference_oracle_sums_partials() {
    assert_eq!(
        row_sum_oracle(&[
            vec![1.0, 2.0, 3.0, 4.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![-2.0, 1.0, -3.0, 2.0],
        ]),
        vec![9.0, 23.0, 30.0, 46.0]
    );
}

#[test]
fn reject_zero_dimensions_and_invalid_rank_counts() {
    for partition in [Partition::Column, Partition::Row] {
        for (out, width, ranks) in [(0, 3, 1), (3, 0, 1), (0, 0, 1), (3, 3, 0), (3, 3, 4)] {
            assert!(TensorParallelLinearPlan::new(out, width, ranks, partition).is_err());
        }
    }
    // Rank limits apply only to the split dimension, not the other dimension.
    assert!(TensorParallelLinearPlan::new(5, 1, 5, Partition::Column).is_ok());
    assert!(TensorParallelLinearPlan::new(1, 5, 5, Partition::Row).is_ok());
}

#[test]
fn reject_out_of_range_rank_and_wrong_full_lengths() {
    for partition in [Partition::Column, Partition::Row] {
        let plan = TensorParallelLinearPlan::new(5, 3, 2, partition).unwrap();
        for r in [rank(2), ParallelRankId::new(u32::MAX)] {
            assert!(plan.rank_range(r).is_err());
            assert!(plan.local_shape(r).is_err());
            assert!(plan.shard_weight(r, &[0.0; 15]).is_err());
            assert!(plan.shard_input(r, &[0.0; 6], 2).is_err());
        }
        for length in [0, 6, 14, 16] {
            assert!(plan.shard_weight(rank(0), &vec![0.0; length]).is_err());
        }
        for length in [0, 3, 5, 7] {
            assert!(plan.shard_input(rank(0), &vec![0.0; length], 2).is_err());
        }
        assert!(plan.shard_input(rank(0), &[], 0).is_err());
    }
}

#[test]
fn reject_element_byte_and_batch_overflows_without_allocating() {
    for partition in [Partition::Column, Partition::Row] {
        for (out, width) in [
            (usize::MAX, 2),
            (2, usize::MAX),
            (usize::MAX / 4 + 1, 1),
            (isize::MAX as usize / 4 + 1, 1),
        ] {
            assert!(TensorParallelLinearPlan::new(out, width, 1, partition).is_err());
        }
        let plan = TensorParallelLinearPlan::new(5, 3, 2, partition).unwrap();
        for rows in [usize::MAX, usize::MAX / 4, isize::MAX as usize / 4] {
            assert!(plan.shard_input(rank(0), &[], rows).is_err());
            if partition == Partition::Column {
                assert!(plan.column_gather_count(rows).is_err());
                assert!(plan.pack_column_output(rank(0), &[], rows).is_err());
                assert!(plan.unpack_column_gather(&[], rows).is_err());
            }
        }
    }
}

#[test]
fn column_layout_rejects_row_plans_zero_rows_invalid_ranks_and_lengths() {
    let row_plan = TensorParallelLinearPlan::new(5, 3, 2, Partition::Row).unwrap();
    assert!(row_plan.column_gather_width().is_err());
    assert!(row_plan.column_gather_count(2).is_err());
    assert!(row_plan.pack_column_output(rank(0), &[0.0; 10], 2).is_err());
    assert!(row_plan.unpack_column_gather(&[0.0; 20], 2).is_err());

    let plan = TensorParallelLinearPlan::new(5, 3, 2, Partition::Column).unwrap();
    assert!(plan.column_gather_count(0).is_err());
    assert!(plan.pack_column_output(rank(0), &[], 0).is_err());
    assert!(plan.unpack_column_gather(&[], 0).is_err());
    for r in [rank(2), ParallelRankId::new(u32::MAX)] {
        assert!(plan.pack_column_output(r, &[0.0; 6], 2).is_err());
    }
    for r in 0..plan.ranks() {
        let count = 2 * plan.local_shape(rank(r)).unwrap().0;
        for length in [0, count - 1, count + 1] {
            assert!(
                plan.pack_column_output(rank(r), &vec![0.0; length], 2)
                    .is_err()
            );
        }
    }
    // Two blocks of six elements are required, including padding. Even an
    // unpadded tensor with the right logical output length is not a gather.
    for length in [0, 6, 10, 11, 13, 18] {
        assert!(plan.unpack_column_gather(&vec![0.0; length], 2).is_err());
    }
}

#[test]
fn column_padding_overflow_is_checked_even_when_logical_output_fits() {
    let plan = TensorParallelLinearPlan::new(5, 1, 2, Partition::Column).unwrap();
    let max_elements = isize::MAX as usize / std::mem::size_of::<f32>();
    let rows = max_elements / 5;
    assert!(rows.checked_mul(5).unwrap() <= max_elements);
    assert!(rows.checked_mul(3).unwrap() <= max_elements);
    assert!(rows.checked_mul(6).unwrap() > max_elements);
    assert!(plan.column_gather_count(rows).is_err());
    assert!(plan.unpack_column_gather(&[], rows).is_err());
}

#[test]
fn column_gather_uses_equal_counts_for_every_rank() {
    for (out, ranks, rows) in [(7, 3, 4), (6, 3, 2), (5, 5, 3), (5, 1, 2)] {
        let plan = TensorParallelLinearPlan::new(out, 1, ranks, Partition::Column).unwrap();
        let width = out.div_ceil(ranks);
        assert_eq!(plan.column_gather_width().unwrap(), width);
        assert_eq!(plan.column_gather_count(rows).unwrap(), rows * width);
        let mut gathered = Vec::new();
        for r in 0..ranks {
            let range = plan.rank_range(rank(r)).unwrap();
            let local: Vec<_> = (0..rows)
                .flat_map(|row| range.clone().map(move |column| (row * out + column) as f32))
                .collect();
            let packed = plan.pack_column_output(rank(r), &local, rows).unwrap();
            assert_eq!(packed.len(), plan.column_gather_count(rows).unwrap());
            for padded_row in packed.chunks_exact(width) {
                assert!(padded_row[range.len()..].iter().all(|value| *value == 0.0));
            }
            gathered.extend(packed);
        }
        assert_eq!(
            plan.unpack_column_gather(&gathered, rows).unwrap(),
            (0..rows * out).map(|i| i as f32).collect::<Vec<_>>()
        );
    }
}

#[test]
fn one_plan_is_reusable_for_different_batches() {
    for partition in [Partition::Column, Partition::Row] {
        let plan = TensorParallelLinearPlan::new(5, 7, 3, partition).unwrap();
        let (weight, _) = fixture(5, 7, 1);
        for rows in [1, 4, 2, 1] {
            let (_, input) = fixture(5, 7, rows);
            let partials = cpu_partials(&plan, &weight, &input, rows);
            assert_eq!(
                output_oracle(&plan, &partials, rows),
                matmul(&weight, &input, rows, 5, 7)
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA and explicit FERRULE_TP_TEST_DEVICE ordinal"]
fn cuda_artifact_shards_reuse_weights_and_match_cpu() {
    use ferrule_backend::cuda::operators::linear::CudaOperators;
    use ferrule_model::transformer::parallel::CudaLinearShard;
    use std::rc::Rc;

    let ordinal: usize = std::env::var("FERRULE_TP_TEST_DEVICE")
        .expect("set FERRULE_TP_TEST_DEVICE explicitly")
        .parse()
        .unwrap();
    let ops = Rc::new(CudaOperators::new_on_device(ordinal).unwrap());
    for partition in [Partition::Column, Partition::Row] {
        let plan = TensorParallelLinearPlan::new(5, 7, 3, partition).unwrap();
        let (weight, _) = fixture(5, 7, 1);
        let shards: Vec<_> = (0..plan.ranks())
            .map(|r| CudaLinearShard::new(Rc::clone(&ops), plan.clone(), rank(r), &weight).unwrap())
            .collect();
        for rows in [1, 3, 2] {
            let (_, input) = fixture(5, 7, rows);
            let mut partials = Vec::new();
            for shard in &shards {
                assert_eq!(shard.plan(), &plan);
                let partial = shard.execute(&input, rows).unwrap();
                let local = plan.shard_input(shard.rank(), &input, rows).unwrap();
                assert_eq!(shard.execute_local(&local, rows).unwrap(), partial);
                assert!(shard.execute_local(&[], rows).is_err());
                assert!(shard.execute(&[], usize::MAX).is_err());
                partials.push(partial);
            }
            let expected = matmul(&weight, &input, rows, 5, 7);
            let actual = output_oracle(&plan, &partials, rows);
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "{a} != {b}");
            }
        }
    }
}

use ferrule_backend::cpu::{
    CpuExecutionPrecision, CpuOperatorProvider, HostRows, LinearRef, LinearWeight,
    ReferenceCpuProvider, RowsDType, RowsShape, SwiGluRef,
};
use ferrule_model::CheckpointDType;
use ferrule_model::transformer::parallel::{
    CpuSwiGluShard, TensorParallelCollective, TensorParallelStagePlan, TensorParallelSwiGluPlan,
    TensorParallelSwiGluWeights, TensorParallelWeightShard,
};

fn swiglu_reference(
    hidden: usize,
    intermediate: usize,
    weights: [&[f32]; 3],
    input: &[f32],
    rows: usize,
) -> Vec<f32> {
    fn linear(weight: &[f32], out_features: usize, in_features: usize) -> LinearRef<'_> {
        LinearRef {
            weight: LinearWeight::F32(weight),
            out_features,
            in_features,
            bias: None,
        }
    }
    ReferenceCpuProvider
        .swiglu(
            SwiGluRef {
                gate: linear(weights[0], intermediate, hidden),
                up: linear(weights[1], intermediate, hidden),
                down: linear(weights[2], hidden, intermediate),
                activation_limit: None,
            },
            &HostRows::new(
                RowsShape::new(rows, hidden).unwrap(),
                RowsDType::F32,
                None,
                input.to_vec(),
            )
            .unwrap(),
            None,
            CpuExecutionPrecision::F32,
        )
        .unwrap()
        .into_values()
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() <= 2e-5 * (1.0 + b.abs()), "{a} != {b}");
    }
}

#[test]
fn swiglu_two_rank_ragged_intermediate_matches_unsplit_provider() {
    let hidden = 3;
    let intermediate = 5;
    let plan = TensorParallelSwiGluPlan::new(hidden, intermediate, 2).unwrap();
    assert_eq!(plan.rank_range(rank(0)).unwrap(), 0..3);
    assert_eq!(plan.rank_range(rank(1)).unwrap(), 3..5);
    let (gate, _) = fixture(intermediate, hidden, 1);
    let up: Vec<_> = gate.iter().rev().map(|v| v * 0.5).collect();
    let (down, _) = fixture(hidden, intermediate, 1);
    let shards: Vec<_> = (0..2)
        .map(|r| {
            let weights = plan.shard_weights_f32(rank(r), &gate, &up, &down).unwrap();
            assert_eq!(weights.gate().plan(), weights.up().plan());
            assert_eq!(
                weights.gate().local_shape()[0],
                weights.down().local_shape()[1]
            );
            CpuSwiGluShard::from_local(weights).unwrap()
        })
        .collect();
    let stage = TensorParallelStagePlan::from(plan.clone());
    assert_eq!(stage.collective(), TensorParallelCollective::Sum);
    assert_eq!(
        (stage.in_features(), stage.out_features(), stage.ranks()),
        (3, 3, 2)
    );
    for rows in [1, 4, 2] {
        let (_, input) = fixture(intermediate, hidden, rows);
        let expected = swiglu_reference(hidden, intermediate, [&gate, &up, &down], &input, rows);
        let partials: Vec<_> = shards
            .iter()
            .map(|shard| {
                assert_eq!(shard.plan(), &plan);
                let output = shard.execute(&input, rows).unwrap();
                assert_eq!(output.len(), rows * hidden);
                assert_eq!(stage.collective_count(rows).unwrap(), output.len());
                stage
                    .pack_local_output(shard.rank(), &output, rows)
                    .unwrap()
            })
            .collect();
        let reduced = row_sum_oracle(&partials);
        assert_close(
            &stage.unpack_collective_output(&reduced, rows).unwrap(),
            &expected,
        );
    }
    assert!(shards[0].execute(&[], 1).is_err());
    assert!(shards[0].execute(&[], usize::MAX).is_err());
    assert!(stage.pack_local_output(rank(2), &[0.0; 3], 1).is_err());
    assert!(stage.unpack_collective_output(&[0.0; 2], 1).is_err());
    assert!(stage.collective_count(0).is_err());
}

#[test]
fn swiglu_and_local_weight_contracts_reject_incompatible_storage() {
    for (hidden, intermediate, ranks) in [
        (0, 5, 2),
        (3, 0, 2),
        (3, 5, 0),
        (3, 5, 6),
        (usize::MAX, 2, 1),
    ] {
        assert!(TensorParallelSwiGluPlan::new(hidden, intermediate, ranks).is_err());
    }
    let plan = TensorParallelSwiGluPlan::new(3, 4, 2).unwrap();
    let local = |p: &TensorParallelLinearPlan, r, dtype: CheckpointDType| {
        let (out, width) = p.local_shape(r).unwrap();
        let bytes = vec![0; out * width * dtype.element_size_bytes().unwrap()];
        TensorParallelWeightShard::from_local_bytes(p.clone(), r, dtype, bytes).unwrap()
    };
    // Both ranks have identical physical shapes; provenance/rank still matters.
    assert!(
        TensorParallelSwiGluWeights::new(
            plan.clone(),
            rank(0),
            local(plan.gate_up_plan(), rank(0), CheckpointDType::F32),
            local(plan.gate_up_plan(), rank(1), CheckpointDType::F32),
            local(plan.down_plan(), rank(0), CheckpointDType::F32),
        )
        .is_err()
    );
    assert!(
        TensorParallelSwiGluWeights::new(
            plan.clone(),
            rank(0),
            local(plan.gate_up_plan(), rank(0), CheckpointDType::F32),
            local(plan.gate_up_plan(), rank(0), CheckpointDType::Bf16),
            local(plan.down_plan(), rank(0), CheckpointDType::F32),
        )
        .is_err()
    );
    for dtype in [
        CheckpointDType::I8,
        CheckpointDType::F8E4M3,
        CheckpointDType::F8E8M0,
    ] {
        assert!(
            TensorParallelWeightShard::from_local_bytes(
                plan.gate_up_plan().clone(),
                rank(0),
                dtype,
                vec![0; 6],
            )
            .is_err()
        );
    }
    assert!(
        TensorParallelWeightShard::from_local_bytes(
            plan.gate_up_plan().clone(),
            rank(0),
            CheckpointDType::F32,
            vec![0; 23],
        )
        .is_err()
    );
    assert!(
        TensorParallelWeightShard::from_local_bytes(
            plan.gate_up_plan().clone(),
            rank(2),
            CheckpointDType::F32,
            vec![0; 24],
        )
        .is_err()
    );
}

#[test]
fn stage_plan_preserves_legacy_linear_collective_layouts() {
    for partition in [Partition::Column, Partition::Row] {
        let plan = TensorParallelLinearPlan::new(5, 7, 2, partition).unwrap();
        let stage = TensorParallelStagePlan::from(plan.clone());
        assert_eq!(stage.in_features(), 7);
        assert_eq!(stage.out_features(), 5);
        let (weight, input) = fixture(5, 7, 3);
        let partials = cpu_partials(&plan, &weight, &input, 3);
        let packed: Vec<_> = partials
            .iter()
            .enumerate()
            .map(|(r, p)| stage.pack_local_output(rank(r), p, 3).unwrap())
            .collect();
        assert!(
            packed
                .iter()
                .all(|p| p.len() == stage.collective_count(3).unwrap())
        );
        let result = match stage.collective() {
            TensorParallelCollective::AllGather => packed.into_iter().flatten().collect(),
            TensorParallelCollective::Sum => row_sum_oracle(&packed),
        };
        assert_eq!(
            stage.unpack_collective_output(&result, 3).unwrap(),
            output_oracle(&plan, &partials, 3)
        );
    }
}

#[test]
fn cuda_swiglu_alignment_preflight_is_native_rank_local_and_device_free() {
    for (hidden, intermediate) in [(32, 256), (1024, 3072), (3072, 24576)] {
        for degree in [2, 4, 8] {
            let plan = TensorParallelSwiGluPlan::new(hidden, intermediate, degree).unwrap();
            for r in 0..degree {
                assert!(plan.validate_cuda(rank(r), &CheckpointDType::Bf16).is_ok());
                assert!(plan.validate_cuda(rank(r), &CheckpointDType::F32).is_ok());
            }
        }
    }
    for (hidden, intermediate, degree) in [(8, 19, 2), (32, 19, 4), (32, 19, 8), (31, 256, 8)] {
        let plan = TensorParallelSwiGluPlan::new(hidden, intermediate, degree).unwrap();
        for r in 0..degree {
            let error = plan
                .validate_cuda(rank(r), &CheckpointDType::Bf16)
                .unwrap_err();
            let message = error.to_string();
            assert!(message.contains("K16"), "{message}");
            assert!(message.contains(&format!("rank={r}")), "{message}");
            assert!(message.contains("local_shape="), "{message}");
            assert!(
                message.contains("no physical padding or F32 fallback"),
                "{message}"
            );
            assert!(plan.validate_cuda(rank(r), &CheckpointDType::F32).is_ok());
        }
    }
    // A globally unaligned width can contain an aligned local rank; validating
    // just the full dimension, or just rank zero, is not a TP admission check.
    let ragged = TensorParallelSwiGluPlan::new(32, 33, 2).unwrap();
    assert!(
        ragged
            .validate_cuda(rank(0), &CheckpointDType::Bf16)
            .is_err()
    );
    assert!(
        ragged
            .validate_cuda(rank(1), &CheckpointDType::Bf16)
            .is_ok()
    );
    assert!(
        ragged
            .validate_cuda(rank(2), &CheckpointDType::F32)
            .is_err()
    );
    assert!(
        ragged
            .validate_cuda(rank(0), &CheckpointDType::F8E4M3)
            .is_err()
    );
    let overflow = TensorParallelSwiGluPlan::new(65536, 65536, 1).unwrap();
    assert!(
        overflow
            .validate_cuda(rank(0), &CheckpointDType::Bf16)
            .is_err()
    );
    // BF16 requires K16, not N16 (linear output dimensions may be ragged).
    let linear = TensorParallelLinearPlan::new(19, 32, 2, Partition::Column).unwrap();
    assert!(
        linear
            .validate_cuda(rank(0), &CheckpointDType::Bf16)
            .is_ok()
    );
}
