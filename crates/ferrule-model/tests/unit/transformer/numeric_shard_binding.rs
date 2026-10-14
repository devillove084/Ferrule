//! Binding-directory checks need no CUDA context, driver, or GPU.
use super::*;
use crate::TensorRole;
use crate::transformer::materialize::numeric_rank_tests::{Fixture, layout};
use crate::transformer::{NumericFp8RankPlan, PreparedDecoderGeneration, StateDictMaterializer};

#[test]
fn numeric_rank_directory_rejects_wrong_image_lane_profile_and_physical_budget_without_gpu() {
    let f = Fixture::new([1040, 259], TensorRole::LinearAttentionQkv, Some(0), false);
    let plan = NumericFp8RankPlan::new(&f.binding, layout(2, 1)).unwrap();
    let precision = NumericFp8Precision::W8A16Bf16F32Accumulate;
    let budget = plan.cuda_execution_budget(9, 0, precision).unwrap();
    let mut directory = Bindings::new(std::slice::from_ref(&f.binding)).unwrap();
    assert!(
        directory
            .configure_numeric_rank(layout(2, 1), budget.resident_grant_bytes)
            .is_err()
    );
    directory.numeric_fp8 = true;
    assert!(directory.configure_numeric_rank(layout(2, 1), 0).is_err());
    directory
        .configure_numeric_rank(layout(2, 1), budget.resident_grant_bytes)
        .unwrap();
    assert!(
        directory
            .preflight_numeric_rank(&plan, 9, 0, precision)
            .is_ok()
    );
    assert_eq!(directory.resident_bytes(), 0);
    assert!(
        directory
            .configure_numeric_rank(layout(2, 0), budget.resident_grant_bytes)
            .is_err()
    );
    assert!(
        directory
            .configure_numeric_rank(layout(2, 1), budget.resident_grant_bytes + 1)
            .is_err()
    );
    let other_lane = NumericFp8RankPlan::new(&f.binding, layout(2, 0)).unwrap();
    assert!(
        directory
            .preflight_numeric_rank(&other_lane, 9, 0, precision)
            .is_err()
    );
    let other = Fixture::new([1040, 259], TensorRole::LinearAttentionQkv, Some(0), false);
    let other_plan = NumericFp8RankPlan::new(&other.binding, layout(2, 1)).unwrap();
    assert!(
        directory
            .preflight_numeric_rank(&other_plan, 9, 0, precision)
            .is_err()
    );
    directory.numeric_rank.as_mut().unwrap().limit -= 1;
    assert!(
        directory
            .preflight_numeric_rank(&plan, 9, 0, precision)
            .is_err()
    );
    directory.numeric_rank.as_mut().unwrap().limit += 1;
    directory.numeric_rank.as_mut().unwrap().grant_bytes = 1;
    assert!(
        directory
            .preflight_numeric_rank(&plan, 9, 0, precision)
            .is_err()
    );
    directory.numeric_rank.as_mut().unwrap().grant_bytes = usize::MAX;
    assert!(
        directory
            .preflight_numeric_rank(&plan, 9, 0, precision)
            .is_err()
    );
}

#[test]
fn numeric_rank_no_legacy_full_upload_and_no_forged_generation_or_slot_without_gpu() {
    let f = Fixture::new([259, 259], TensorRole::DenseMlpDown, Some(0), false);
    let plan = NumericFp8RankPlan::new(&f.binding, layout(2, 1)).unwrap();
    let full = StateDictMaterializer::new(1 << 20)
        .unwrap()
        .parameter(&f.binding)
        .unwrap();
    let mut directory = Bindings::new(std::slice::from_ref(&f.binding)).unwrap();
    directory.numeric_fp8 = true;
    assert!(directory.validate(&full).is_ok());
    directory
        .configure_numeric_rank(layout(2, 1), 1 << 20)
        .unwrap();
    assert!(directory.validate(&full).is_err());
    for generation in [
        directory.generation,
        PreparedDecoderGeneration::take().unwrap(),
    ] {
        let token = CudaNumericFp8RankBinding {
            generation,
            slot: 0,
            plan: plan.clone(),
        };
        assert!(directory.numeric_rank_linear(&token).is_err());
    }
}

#[test]
#[ignore = "requires a real CUDA device; compile-only in no-GPU validation"]
fn numeric_rank_gpu_segments_are_resident_origin_correct_and_allocation_free() {
    use crate::transformer::{
        CudaRows, CudaStandardDecoderOperators, ExpertCachePolicy, Rows, RowsShape,
    };
    use std::rc::Rc;
    for precision in [
        NumericFp8Precision::F32Tf32x3,
        NumericFp8Precision::W8A16Bf16F32Accumulate,
        NumericFp8Precision::Bf16RneF32Accumulate,
    ] {
        let f = Fixture::new([1040, 259], TensorRole::LinearAttentionQkv, Some(0), false);
        let prepared = f.prepare(2, 1);
        let plan = prepared.plan().clone();
        let scratch = if precision == NumericFp8Precision::W8A16Bf16F32Accumulate {
            0
        } else {
            1 << 20
        };
        let budget = plan.cuda_execution_budget(9, scratch, precision).unwrap();
        let ops = Rc::new(CudaOperators::new_on_device(0).unwrap());
        let mut owner = CudaStandardDecoderOperators::new_numeric_fp8_with_precision(
            Rc::clone(&ops),
            std::slice::from_ref(&f.binding),
            ExpertCachePolicy::KeepAll,
            scratch,
            precision,
        )
        .unwrap();
        owner
            .configure_numeric_fp8_rank(layout(2, 1), budget.resident_grant_bytes)
            .unwrap();
        let before = ops.allocator_metrics().granted_bytes;
        let token = owner.prepare_numeric_fp8_rank_linear(&prepared, 9).unwrap();
        assert_eq!(
            ops.allocator_metrics().granted_bytes - before,
            budget.resident_grant_bytes as u64
        );
        assert_eq!(
            owner.numeric_fp8_rank_resident_grant_bytes(),
            budget.resident_grant_bytes
        );
        owner.prepare_numeric_fp8_rank_linear(&prepared, 9).unwrap();
        assert_eq!(
            ops.allocator_metrics().granted_bytes - before,
            budget.resident_grant_bytes as u64
        );
        let expected = prepared
            .segments()
            .iter()
            .map(|s| {
                s.artifact()
                    .decode_f32(1 << 22)
                    .unwrap()
                    .as_chunks::<259>()
                    .0
                    .iter()
                    .map(|row| row.iter().sum::<f32>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        drop(prepared);
        // Execution cannot depend on source freshness or reopen either file.
        std::fs::remove_file(&f.binding.weight().slice().path).unwrap();
        std::fs::remove_file(&f.binding.scale().unwrap().slice().path).unwrap();
        for rows in [9, 1, 2] {
            let input = Rows::Cuda(
                CudaRows::f32(
                    RowsShape::new(rows, 259).unwrap(),
                    None,
                    ops.upload_f32_buffer(&vec![1.0; rows * 259]).unwrap(),
                )
                .unwrap(),
            );
            let mut outputs = expected
                .iter()
                .map(|e| ops.zero_f32_buffer(rows * e.len()).unwrap())
                .collect::<Vec<_>>();
            let before = ops.counters();
            owner
                .numeric_fp8_rank_linear_into_segments(&token, &input, &mut outputs)
                .unwrap();
            let after = ops.counters();
            assert_eq!(
                after.device_allocation_attempts,
                before.device_allocation_attempts
            );
            assert_eq!(after.device_to_host_bytes, before.device_to_host_bytes);
            assert_eq!(after.host_to_device_bytes, before.host_to_device_bytes);
            for (output, expected) in outputs.iter().zip(&expected) {
                let actual = ops.download_f32_buffer(output).unwrap();
                for row in actual.chunks_exact(expected.len()) {
                    for (actual, expected) in row.iter().zip(expected) {
                        assert!((actual - expected).abs() < 1e-3, "{actual} != {expected}");
                    }
                }
            }
            let before = ops.counters();
            let alias = outputs[0].prefix(outputs[0].len()).unwrap();
            outputs[1] = alias;
            assert!(
                owner
                    .numeric_fp8_rank_linear_into_segments(&token, &input, &mut outputs)
                    .is_err()
            );
            assert_eq!(ops.counters().kernel_launches, before.kernel_launches);
        }
    }
}
