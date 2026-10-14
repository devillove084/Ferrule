//! Runtime regressions for the F32-prefix alias audit; compile-only without a GPU owner.
#![cfg(feature = "cuda")]
use ferrule_backend::cuda::operator_owner::{CudaF32Buffer, CudaOperators};
use ferrule_backend::cuda::operators::SplitHalfRopeLayout;
use ferrule_backend::cuda::operators::recurrent::QueryGateLayout;
use ferrule_common::Result;

fn rejected(op: &CudaOperators, label: &str, result: Result<()>) {
    let error = result.expect_err(label);
    assert!(
        error.to_string().contains("overlapping"),
        "{label}: {error}"
    );
    assert_eq!(
        op.counters(),
        Default::default(),
        "{label} must fail before GPU work"
    );
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (&a, &e) in actual.iter().zip(expected) {
        assert!(
            a.is_finite() && (a - e).abs() < 1e-5 * (1.0 + e.abs()),
            "{a} != {e}"
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU; broadcast weight and destructive aliases must fail prelaunch"]
fn broadcast_rms_weight_copy_scatter_and_multioutput_aliases_are_rejected() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let mut root = op.upload_f32_buffer(&[3.0; 16]).unwrap();
    let input = op.upload_f32_buffer(&[1.0; 16]).unwrap();
    let rows = op.upload_i32_buffer(&[0, 1]).unwrap();
    let short = root.prefix(8).unwrap();
    let full = root.prefix(16).unwrap();
    let mut out8 = root.prefix(8).unwrap();
    op.reset_counters();
    rejected(
        &op,
        "RMS broadcast weight",
        op.rms_norm_f32_into(&input, 2, &short, 1e-6, &mut root),
    );
    rejected(
        &op,
        "legacy RMS broadcast weight",
        op.rms_norm_rows_from_device_into(&input, 2, &short, 1e-6, &mut root),
    );
    rejected(
        &op,
        "overlapping DMA",
        op.copy_f32_range(&short, 0, &mut root, 1, 8),
    );
    rejected(
        &op,
        "overlapping slot copy",
        op.copy_f32_into_slot(&short, &mut root, 1),
    );
    rejected(
        &op,
        "concat clobbers second read",
        op.concat_f32_buffers_into(&input.prefix(8).unwrap(), &short, 1, 8, &mut root),
    );
    rejected(
        &op,
        "residual clobbers update",
        op.residual_add_f32_into(&input, &full, &mut root),
    );
    rejected(
        &op,
        "scatter permutation",
        op.scatter_add_f32_rows(&full, &rows, &mut root, 2, 8),
    );
    rejected(
        &op,
        "combine broadcast weights",
        op.weighted_combine_f32_into(&input, &rows, &root.prefix(2).unwrap(), &mut root, 2, 8),
    );
    rejected(
        &op,
        "route reducer",
        op.reduce_moe_route_outputs_ranked(&full, 1, 2, 8, &mut out8),
    );
    rejected(
        &op,
        "split writes",
        op.split_query_gate_f32_into(
            &input,
            &mut out8,
            &mut root.prefix(8).unwrap(),
            QueryGateLayout {
                rows: 1,
                heads: 2,
                head_dim: 4,
            },
        ),
    );
    rejected(
        &op,
        "grouped weight broadcast",
        op.grouped_matvec_f32_rows_from_device_into(&input, 2, &short, 8, 1, 1, &mut root),
    );
    assert_eq!(op.download_f32_buffer(&root).unwrap(), [3.0; 16]);
    let mut topk_indices = op.upload_i32_buffer(&[0; 4]).unwrap();
    let mut topk_values = root.prefix(4).unwrap();
    op.reset_counters();
    rejected(
        &op,
        "top-k row packing",
        op.topk_vocab_rows_from_device_into(&full, 2, 8, 2, &mut topk_indices, &mut topk_values),
    );
    rejected(
        &op,
        "RoPE table broadcast",
        op.split_half_rope_f32(
            &mut root,
            &short,
            &input.prefix(8).unwrap(),
            &[0, 1],
            SplitHalfRopeLayout {
                rows: 2,
                heads: 1,
                head_dim: 8,
                rope_dim: 8,
                table_positions: 2,
                restore_bf16_boundary: false,
            },
        ),
    );
    let bytes: Vec<u8> = [1.0f32; 64]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let linear = op.upload_f32_linear(&bytes, 8, 8).unwrap();
    op.reset_counters();
    rejected(
        &op,
        "direct projection",
        op.artifact_linear_rows_from_device_into(&linear, &full, 2, &mut root),
    );
    rejected(
        &op,
        "dual projection writes",
        op.artifact_linear_pair_matvec_into(
            &linear,
            &linear,
            &input.prefix(8).unwrap(),
            &mut out8,
            &mut root.prefix(8).unwrap(),
        ),
    );
    assert_eq!(op.download_f32_buffer(&root).unwrap(), [3.0; 16]);
}

#[test]
#[ignore = "requires CUDA GPU; exact inplace remains numerically compatible"]
fn supported_exact_inplace_and_disjoint_same_allocation_copies_remain_valid() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let values = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
    let input = op.upload_f32_buffer(&values).unwrap();
    let weight = op.upload_f32_buffer(&[1.0, 0.5, 0.25, 2.0]).unwrap();
    let mut separate = op.zero_f32_buffer(8).unwrap();
    let mut inplace = op.upload_f32_buffer(&values).unwrap();
    op.rms_norm_f32_into(&input, 2, &weight, 1e-6, &mut separate)
        .unwrap();
    op.rms_norm_f32_into(&inplace.prefix(8).unwrap(), 2, &weight, 1e-6, &mut inplace)
        .unwrap();
    close(
        &op.download_f32_buffer(&inplace).unwrap(),
        &op.download_f32_buffer(&separate).unwrap(),
    );
    for clamped in [false, true] {
        let mut output = op.upload_f32_buffer(&values).unwrap();
        let up = output.prefix(8).unwrap();
        if clamped {
            op.swiglu_f32_clamped_into(&input, &input, &mut separate, 3.0)
                .unwrap();
            op.swiglu_f32_clamped_into(&up, &up, &mut output, 3.0)
                .unwrap();
        } else {
            op.swiglu_f32_into(&input, &input, &mut separate).unwrap();
            op.swiglu_f32_into(&up, &up, &mut output).unwrap();
        }
        close(
            &op.download_f32_buffer(&output).unwrap(),
            &op.download_f32_buffer(&separate).unwrap(),
        );
    }
    // Single-row weight/output exact alias has no broadcast-reader race.
    let row = input.prefix(4).unwrap();
    let mut weight_output = op.upload_f32_buffer(&[1.0, 0.5, 0.25, 2.0]).unwrap();
    let mut expected = op.zero_f32_buffer(4).unwrap();
    op.rms_norm_f32_into(&row, 1, &weight, 1e-6, &mut expected)
        .unwrap();
    op.rms_norm_f32_into(
        &row,
        1,
        &weight_output.prefix(4).unwrap(),
        1e-6,
        &mut weight_output,
    )
    .unwrap();
    close(
        &op.download_f32_buffer(&weight_output).unwrap(),
        &op.download_f32_buffer(&expected).unwrap(),
    );
    let mut root = op.upload_f32_buffer(&[1.0; 16]).unwrap();
    let prefix = root.prefix(8).unwrap();
    op.copy_f32_range(&prefix, 0, &mut root, 8, 8).unwrap();
    op.copy_f32_into_slot(&prefix, &mut root, 8).unwrap();
    op.reset_counters();
    op.copy_f32_range(&prefix, 0, &mut root, 0, 8).unwrap();
    op.copy_f32_into_slot(&prefix, &mut root, 0).unwrap();
    assert_eq!(
        op.counters(),
        Default::default(),
        "exact self-copy is a no-op"
    );
    let full = root.prefix(16).unwrap();
    op.residual_add_f32_into(&full, &full, &mut root).unwrap();
    assert_eq!(op.download_f32_buffer(&root).unwrap(), [2.0; 16]);
    op.saxpy_into(2.0, &full, &mut root).unwrap();
    assert_eq!(op.download_f32_buffer(&root).unwrap(), [6.0; 16]);
}

#[test]
#[ignore = "requires CUDA GPU; live allocator grants match CPU metadata"]
fn small_roots_and_replacements_use_sixteen_256_byte_grants() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let before = op.allocator_metrics().live_granted_bytes;
    let allocate = |rows| -> Vec<CudaF32Buffer> {
        [10usize, 2, 2, 4, 10, 4, 4, 4]
            .into_iter()
            .map(|width| op.zero_f32_buffer(rows * width).unwrap())
            .collect()
    };
    let old = allocate(1);
    assert_eq!(op.allocator_metrics().live_granted_bytes - before, 2048);
    let new = allocate(2);
    assert_eq!(op.allocator_metrics().live_granted_bytes - before, 4096);
    op.sync_stream().unwrap();
    drop(old);
    drop(new);
}
