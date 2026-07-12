//! Numerical and long-context boundary regressions for DSV4 device top-k kernels.

use ferrule_cuda::context::CudaArtifactOperatorContext;
use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_guard() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .expect("CUDA DSV4 decode top-k test lock poisoned")
}

#[allow(clippy::too_many_arguments)]
fn paged_decode_rows_reference(
    query: &[f32],
    weights: &[f32],
    indexer_plane: &[f32],
    block_slots: &[i32],
    block_offsets: &[i32],
    row_sequence_ids: &[i32],
    positions: &[i32],
    window_lens: &[i32],
    compressed_lens: &[i32],
    window_size: usize,
    index_topk: usize,
    index_heads: usize,
    index_head_dim: usize,
    page_tokens: usize,
    weight_scale: f32,
) -> (Vec<i32>, Vec<i32>) {
    let rows = positions.len();
    let cols = window_size + index_topk;
    let mut logical = vec![-1; rows * cols];
    let mut selectors = vec![-1; rows * cols];

    for row in 0..rows {
        let output_base = row * cols;
        let position = usize::try_from(positions[row]).expect("non-negative position");
        let window_len = usize::try_from(window_lens[row]).expect("non-negative window len");
        for col in 0..window_size {
            if window_len <= window_size && window_len <= position + 1 && col < window_len {
                logical[output_base + col] = (position + 1 - window_len + col) as i32;
                selectors[output_base + col] = 0;
            }
        }

        let compressed_len =
            usize::try_from(compressed_lens[row]).expect("non-negative compressed len");
        let sequence = usize::try_from(row_sequence_ids[row]).expect("valid row sequence ID");
        let block_start = usize::try_from(block_offsets[sequence]).expect("valid block start");
        let block_end = usize::try_from(block_offsets[sequence + 1]).expect("valid block end");
        let mut candidates = Vec::with_capacity(compressed_len);
        for index in 0..compressed_len {
            let block_entry = block_start + index / page_tokens;
            if block_entry >= block_end {
                continue;
            }
            let physical_slot =
                usize::try_from(block_slots[block_entry]).expect("valid physical slot");
            let kv_base = (physical_slot * page_tokens + index % page_tokens) * index_head_dim;
            let mut score = 0.0f32;
            for head in 0..index_heads {
                let query_base = (row * index_heads + head) * index_head_dim;
                let mut dot = 0.0f32;
                for dim in 0..index_head_dim {
                    dot += query[query_base + dim] * indexer_plane[kv_base + dim];
                }
                score += dot.max(0.0) * weights[row * index_heads + head] * weight_scale;
            }
            if score.is_finite() {
                candidates.push((index, score));
            }
        }
        candidates.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_index.cmp(right_index))
        });
        for (slot, (index, _)) in candidates.into_iter().take(index_topk).enumerate() {
            logical[output_base + window_size + slot] = index as i32;
            selectors[output_base + window_size + slot] = 1;
        }
    }

    (logical, selectors)
}

#[test]
#[ignore = "requires a CUDA device"]
fn paged_decode_rows_matches_stable_cpu_reference() {
    const ROWS: usize = 3;
    const WINDOW_SIZE: usize = 4;
    const INDEX_TOPK: usize = 3;
    const INDEX_HEADS: usize = 1;
    const INDEX_HEAD_DIM: usize = 4;
    const PAGE_TOKENS: usize = 2;

    let query = [
        1.0, 1.0, 1.0, 0.0, // row 0
        0.0, 1.0, 0.0, 1.0, // row 1
        1.0, 0.0, 0.0, 0.0, // row 2 (no compressed candidates)
    ];
    let weights = [1.0, 1.0, 1.0];
    // Physical slots are deliberately not in sequence order. Both non-empty
    // rows cross a page boundary; row 1 has a partially occupied final page.
    let indexer_plane = [
        -1.0, 0.0, 0.0, 0.0, // slot 0: row 0, logical compressed 2
        0.0, 0.0, 1.0, 0.0, // slot 0: row 0, logical compressed 3
        1.0, 0.0, 0.0, 0.0, // slot 1: row 0, logical compressed 0
        0.0, 1.0, 0.0, 0.0, // slot 1: row 0, logical compressed 1
        0.0, -1.0, 0.0, 2.0, // slot 2: row 1, logical compressed 2
        0.0, 0.0, 0.0, 0.0, // slot 2 padding
        0.0, 0.0, 0.0, 1.0, // slot 3: row 1, logical compressed 0
        0.0, 2.0, 0.0, 0.0, // slot 3: row 1, logical compressed 1
    ];
    let block_slots = [1, 0, 3, 2];
    let block_offsets = [0, 2, 4, 4];
    let row_sequence_ids = [1, 0, 1];
    let positions = [5, 8, 0];
    let window_lens = [4, 3, 1];
    let compressed_lens = [4, 3, 0];
    let expected = paged_decode_rows_reference(
        &query,
        &weights,
        &indexer_plane,
        &block_slots,
        &block_offsets,
        &row_sequence_ids,
        &positions,
        &window_lens,
        &compressed_lens,
        WINDOW_SIZE,
        INDEX_TOPK,
        INDEX_HEADS,
        INDEX_HEAD_DIM,
        PAGE_TOKENS,
        1.0,
    );

    let _guard = cuda_test_guard();
    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let query = context
        .upload_f32_buffer(&query)
        .expect("upload row queries");
    let weights = context
        .upload_f32_buffer(&weights)
        .expect("upload row weights");
    let indexer_plane = context
        .upload_f32_buffer(&indexer_plane)
        .expect("upload paged indexer plane");
    let block_slots = context
        .upload_i32_buffer(&block_slots)
        .expect("upload flattened block slots");
    let block_offsets = context
        .upload_i32_buffer(&block_offsets)
        .expect("upload block offsets");
    let row_sequence_ids = context
        .upload_i32_buffer(&row_sequence_ids)
        .expect("upload row sequence IDs");
    let positions = context
        .upload_i32_buffer(&positions)
        .expect("upload positions");
    let window_lens = context
        .upload_i32_buffer(&window_lens)
        .expect("upload window lengths");
    let compressed_lens = context
        .upload_i32_buffer(&compressed_lens)
        .expect("upload compressed lengths");

    let (logical, selectors) = context
        .dsv4_decode_topk_indices_paged_indexer_rows_from_device(
            &query,
            &weights,
            &indexer_plane,
            &block_slots,
            &block_offsets,
            &row_sequence_ids,
            &positions,
            &window_lens,
            &compressed_lens,
            ROWS,
            WINDOW_SIZE,
            INDEX_TOPK,
            INDEX_HEADS,
            INDEX_HEAD_DIM,
            PAGE_TOKENS,
            0,
            1,
            1.0,
        )
        .expect("launch paged decode rows");
    context
        .sync_stream()
        .expect("synchronize paged decode rows");

    assert_eq!(
        context
            .download_i32_buffer(&logical)
            .expect("download logical indices"),
        expected.0
    );
    assert_eq!(
        context
            .download_i32_buffer(&selectors)
            .expect("download plane selectors"),
        expected.1
    );
}
