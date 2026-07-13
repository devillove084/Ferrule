//! Numerical and long-context boundary regressions for DSV4 device top-k kernels.

use ferrule_cuda::context::{
    CudaArtifactOperatorContext, validate_dsv4_router_hash_table, validate_dsv4_router_token_ids,
};
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

fn router_topk_reference(
    logits: &[f32],
    bias: &[f32],
    tokens: usize,
    experts: usize,
    top_k: usize,
    route_scale: f32,
) -> (Vec<i32>, Vec<f32>) {
    let mut indices = Vec::with_capacity(tokens * top_k);
    let mut weights = Vec::with_capacity(tokens * top_k);
    for row in logits.chunks_exact(experts).take(tokens) {
        let scores = row
            .iter()
            .map(|&logit| {
                let softplus = if logit > 20.0 {
                    logit
                } else if logit < -20.0 {
                    logit.exp()
                } else {
                    logit.exp().ln_1p()
                };
                softplus.sqrt()
            })
            .collect::<Vec<_>>();
        let mut ranked = (0..experts).collect::<Vec<_>>();
        ranked.sort_by(|&left, &right| {
            (scores[right] + bias[right])
                .total_cmp(&(scores[left] + bias[left]))
                .then_with(|| left.cmp(&right))
        });
        let selected = &ranked[..top_k];
        let sum = selected.iter().map(|&expert| scores[expert]).sum::<f32>();
        for &expert in selected {
            indices.push(expert as i32);
            weights.push(scores[expert] / sum * route_scale);
        }
    }
    (indices, weights)
}

#[test]
#[ignore = "requires a CUDA device"]
fn router_topk_writes_i32_ids_without_hidden_download_and_matches_cpu_routes() {
    const TOKENS: usize = 2;
    const EXPERTS: usize = 6;
    const TOP_K: usize = 3;
    const ROUTE_SCALE: f32 = 1.25;

    let logits = [
        -2.0, 0.5, 3.0, -0.25, 1.5, -4.0, // row 0
        2.0, -1.0, 0.0, 4.0, -3.0, 1.0, // row 1
    ];
    let bias = [0.0, 0.25, -0.5, 0.0, 0.75, -0.25];
    let expected = router_topk_reference(&logits, &bias, TOKENS, EXPERTS, TOP_K, ROUTE_SCALE);

    let _guard = cuda_test_guard();
    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let logits = context
        .upload_f32_buffer(&logits)
        .expect("upload router logits");
    let bias = context
        .upload_f32_buffer(&bias)
        .expect("upload router bias");
    let mut indices = context
        .zero_i32_buffer(TOKENS * TOP_K)
        .expect("allocate i32 router IDs");
    let mut weights = context
        .zero_f32_buffer(TOKENS * TOP_K)
        .expect("allocate router weights");

    context.reset_counters();
    context.enable_capture_safe();
    context
        .dsv4_router_topk_sqrt_softplus_rows_from_device_into(
            &logits,
            Some(&bias),
            TOKENS,
            EXPERTS,
            TOP_K,
            ROUTE_SCALE,
            &mut indices,
            &mut weights,
        )
        .expect("launch DSV4 router top-k");
    context.disable_capture_safe();
    let launch_counters = context.counters();
    assert_eq!(launch_counters.device_to_host_copies, 0);
    assert_eq!(launch_counters.device_to_host_bytes, 0);
    assert_eq!(launch_counters.stream_wide_syncs, 0);

    let actual_indices = context
        .download_i32_buffer(&indices)
        .expect("download i32 router IDs");
    let actual_weights = context
        .download_f32_buffer(&weights)
        .expect("download router weights");
    assert_eq!(actual_indices, expected.0);
    for (actual, expected) in actual_weights.iter().zip(&expected.1) {
        assert!(
            (actual - expected).abs() <= 1.0e-4,
            "router weight mismatch: actual={actual} expected={expected}"
        );
    }
}

#[test]
fn hash_router_host_validation_covers_token_rows_topk_duplicates_and_bounds() {
    assert_eq!(
        validate_dsv4_router_token_ids(&[2, 0], 3).expect("valid token rows"),
        vec![2, 0]
    );
    assert!(
        validate_dsv4_router_token_ids(&[3], 3)
            .unwrap_err()
            .to_string()
            .contains("batch row 0")
    );

    let valid = [0usize, 1, 2, 2, 0, 1];
    assert_eq!(
        validate_dsv4_router_hash_table(&valid, 2, 3, 3, 2).expect("valid hash table"),
        vec![0, 1, 2, 2, 0, 1]
    );
    assert!(
        validate_dsv4_router_hash_table(&valid, 2, 3, 3, 4)
            .unwrap_err()
            .to_string()
            .contains("top_k")
    );
    assert!(
        validate_dsv4_router_hash_table(&[1, 1, 0], 1, 3, 3, 2)
            .unwrap_err()
            .to_string()
            .contains("duplicate expert id 1")
    );
    assert!(
        validate_dsv4_router_hash_table(&[3, 1, 0], 1, 3, 3, 2)
            .unwrap_err()
            .to_string()
            .contains("exceeds expert count 3")
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn hash_router_uses_token_rows_matches_weights_and_wrapper_has_no_copies_or_sync() {
    const TOKENS: usize = 3;
    const EXPERTS: usize = 4;
    const TOP_K: usize = 2;
    const HASH_ROWS: usize = 5;
    const HASH_COLS: usize = 3;
    const ROUTE_SCALE: f32 = 1.5;

    let logits: [f32; TOKENS * EXPERTS] = [
        -2.0, 0.5, 3.0, -0.25, // token 0 uses hash row 3
        2.0, -1.0, 0.0, 4.0, // token 1 uses hash row 0
        1.0, 2.0, -3.0, 0.25, // token 2 also uses hash row 3
    ];
    let token_id_values = [3u32, 0, 3];
    let hash_table = [
        3usize, 1, 2, // row 0
        0, 2, 1, // row 1
        1, 3, 0, // row 2
        2, 0, 3, // row 3
        0, 1, 2, // row 4
    ];
    let expected_indices = [2i32, 0, 3, 1, 2, 0];
    let mut expected_weights = Vec::with_capacity(TOKENS * TOP_K);
    for (row, selected) in logits
        .chunks_exact(EXPERTS)
        .zip(expected_indices.chunks_exact(TOP_K))
    {
        let scores = selected
            .iter()
            .map(|&expert| {
                let logit = row[expert as usize];
                let softplus = if logit > 20.0 {
                    logit
                } else if logit < -20.0 {
                    logit.exp()
                } else {
                    logit.exp().ln_1p()
                };
                softplus.sqrt()
            })
            .collect::<Vec<_>>();
        let sum = scores.iter().sum::<f32>();
        expected_weights.extend(scores.into_iter().map(|score| score / sum * ROUTE_SCALE));
    }

    let _guard = cuda_test_guard();
    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let logits = context
        .upload_f32_buffer(&logits)
        .expect("upload router logits");
    let mut token_ids = context
        .dsv4_router_token_ids(&token_id_values, HASH_ROWS)
        .expect("upload router token ids");
    let hash_table = context
        .upload_dsv4_router_hash_table(&hash_table, HASH_ROWS, HASH_COLS, EXPERTS, TOP_K)
        .expect("upload router hash table");
    let mut indices = context
        .zero_i32_buffer(TOKENS * TOP_K)
        .expect("allocate router IDs");
    let mut weights = context
        .zero_f32_buffer(TOKENS * TOP_K)
        .expect("allocate router weights");

    context.reset_counters();
    context
        .update_dsv4_router_token_ids(&token_id_values, HASH_ROWS, &mut token_ids)
        .expect("reuse unchanged router token ids");
    context.enable_capture_safe();
    context
        .dsv4_router_hash_sqrt_softplus_rows_from_device_into(
            &logits,
            &token_ids,
            &hash_table,
            TOKENS,
            EXPERTS,
            TOP_K,
            ROUTE_SCALE,
            &mut indices,
            &mut weights,
        )
        .expect("launch DSV4 hash router");
    context.disable_capture_safe();
    let launch_counters = context.counters();
    assert_eq!(launch_counters.host_to_device_copies, 0);
    assert_eq!(launch_counters.host_to_device_bytes, 0);
    assert_eq!(launch_counters.device_to_host_copies, 0);
    assert_eq!(launch_counters.device_to_host_bytes, 0);
    assert_eq!(launch_counters.stream_wide_syncs, 0);
    assert_eq!(launch_counters.device_allocation_attempts, 0);
    assert_eq!(launch_counters.device_allocations, 0);

    assert_eq!(
        context
            .download_i32_buffer(&indices)
            .expect("download hash router IDs"),
        expected_indices
    );
    let actual_weights = context
        .download_f32_buffer(&weights)
        .expect("download hash router weights");
    for (actual, expected) in actual_weights.iter().zip(&expected_weights) {
        assert!(
            (actual - expected).abs() <= 1.0e-4,
            "hash router weight mismatch: actual={actual} expected={expected}"
        );
    }

    context.reset_counters();
    context
        .update_dsv4_router_token_ids(&[3, 0, 4], HASH_ROWS, &mut token_ids)
        .expect("overwrite changed router token ids");
    let overwrite_counters = context.counters();
    assert_eq!(overwrite_counters.host_to_device_copies, 1);
    assert_eq!(
        overwrite_counters.host_to_device_bytes,
        (TOKENS * std::mem::size_of::<i32>()) as u64
    );
    assert_eq!(overwrite_counters.device_to_host_copies, 0);
    assert_eq!(overwrite_counters.stream_wide_syncs, 0);
    assert_eq!(overwrite_counters.device_allocation_attempts, 0);
    assert_eq!(overwrite_counters.device_allocations, 0);
}
