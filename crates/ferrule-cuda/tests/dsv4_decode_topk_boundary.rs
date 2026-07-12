//! Numerical and long-context boundary regressions for DSV4 device top-k kernels.

use cuda_core::CudaContext;
use ferrule_cuda::context::CudaArtifactOperatorContext;
use std::sync::{Mutex, MutexGuard};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_guard() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .expect("CUDA DSV4 decode top-k test lock poisoned")
}

fn has_cuda() -> bool {
    CudaContext::new(0).is_ok()
}

fn expected_indices(position: usize, window_size: usize, extra_cols: usize) -> Vec<i32> {
    let slot = position % window_size;
    let mut expected = Vec::with_capacity(window_size + extra_cols);
    expected.extend(((slot + 1)..window_size).map(|index| index as i32));
    expected.extend((0..=slot).map(|index| index as i32));
    expected.extend((0..extra_cols).map(|index| (window_size + index) as i32));
    expected
}

#[test]
fn decode_topk_kernels_handle_dsv4_4096_boundary() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    const POSITION: usize = 4096;
    const WINDOW_SIZE: usize = 128;
    const EXTRA_COLS: usize = 512;
    const COMPRESSED_LEN: usize = 1024;
    const INDEX_HEADS: usize = 64;
    const INDEX_HEAD_DIM: usize = 128;
    const ROPE_DIM: usize = 64;

    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    // Zero scores make every compressed candidate tie. The kernels' stable
    // tie-break must therefore select compressed indices 0..512 in order.
    let query = context
        .upload_f32_buffer(&vec![0.0; INDEX_HEADS * INDEX_HEAD_DIM])
        .expect("upload index query");
    let weights = context
        .upload_f32_buffer(&vec![1.0; INDEX_HEADS])
        .expect("upload index weights");
    let indexer_kv = context
        .upload_f32_buffer(&vec![0.0; COMPRESSED_LEN * INDEX_HEAD_DIM])
        .expect("upload indexer KV");
    let expected = expected_indices(POSITION, WINDOW_SIZE, EXTRA_COLS);

    let scalar = context
        .dsv4_decode_topk_indices_from_device(
            Some(&query),
            Some(&weights),
            Some(&indexer_kv),
            POSITION,
            WINDOW_SIZE,
            WINDOW_SIZE,
            EXTRA_COLS,
            WINDOW_SIZE,
            COMPRESSED_LEN,
            INDEX_HEADS,
            INDEX_HEAD_DIM,
            1.0,
        )
        .expect("launch scalar decode top-k");
    context
        .sync_stream()
        .expect("synchronize scalar decode top-k");
    let scalar = context
        .download_i32_buffer(&scalar)
        .expect("download scalar decode top-k");
    assert_eq!(scalar, expected, "scalar decode top-k indices");

    let rope_rows = POSITION + 1;
    let rope_cols = ROPE_DIM / 2;
    let cos = context
        .upload_f32_buffer(&vec![1.0; rope_rows * rope_cols])
        .expect("upload cosine table");
    let sin = context
        .upload_f32_buffer(&vec![0.0; rope_rows * rope_cols])
        .expect("upload sine table");
    let fused = context
        .dsv4_decode_topk_indices_fused_index_query_from_device(
            &query,
            &weights,
            &indexer_kv,
            &cos,
            &sin,
            POSITION,
            WINDOW_SIZE,
            WINDOW_SIZE,
            EXTRA_COLS,
            WINDOW_SIZE,
            COMPRESSED_LEN,
            INDEX_HEADS,
            INDEX_HEAD_DIM,
            ROPE_DIM,
            1.0,
        )
        .expect("launch fused decode top-k");
    context
        .sync_stream()
        .expect("synchronize fused decode top-k");
    let fused = context
        .download_i32_buffer(&fused)
        .expect("download fused decode top-k");
    assert_eq!(fused, expected, "fused decode top-k indices");
}

#[test]
fn prefill_topk_matches_stable_cpu_reference() {
    let _guard = cuda_test_guard();
    if !has_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    const TOKENS: usize = 9;
    const WINDOW_SIZE: usize = 4;
    const EXTRA_COLS: usize = 3;
    const COMPRESS_RATIO: usize = 2;
    const COMPRESSED_LEN: usize = 4;
    const INDEX_HEADS: usize = 2;
    const INDEX_HEAD_DIM: usize = 4;
    const WEIGHT_SCALE: f32 = 0.125;

    let query = (0..TOKENS * INDEX_HEADS * INDEX_HEAD_DIM)
        .map(|index| ((index * 17 % 29) as f32 - 14.0) * 0.0625)
        .collect::<Vec<_>>();
    let weights = (0..TOKENS * INDEX_HEADS)
        .map(|index| if index % 3 == 0 { -0.75 } else { 1.25 })
        .collect::<Vec<_>>();
    let indexer_kv = (0..COMPRESSED_LEN * INDEX_HEAD_DIM)
        .map(|index| ((index * 11 % 23) as f32 - 11.0) * 0.125)
        .collect::<Vec<_>>();

    let mut expected = vec![-1i32; TOKENS * (WINDOW_SIZE + EXTRA_COLS)];
    for token in 0..TOKENS {
        let row = token * (WINDOW_SIZE + EXTRA_COLS);
        let first = (token + 1).saturating_sub(WINDOW_SIZE);
        for col in 0..WINDOW_SIZE {
            let index = first + col;
            if index <= token {
                expected[row + col] = index as i32;
            }
        }

        let visible = ((token + 1) / COMPRESS_RATIO).min(COMPRESSED_LEN);
        let mut candidates = (0..visible)
            .map(|index| {
                let mut score = 0.0f32;
                for head in 0..INDEX_HEADS {
                    let q_base = (token * INDEX_HEADS + head) * INDEX_HEAD_DIM;
                    let kv_base = index * INDEX_HEAD_DIM;
                    let mut dot = 0.0f32;
                    for dim in 0..INDEX_HEAD_DIM {
                        dot += query[q_base + dim] * indexer_kv[kv_base + dim];
                    }
                    score += dot.max(0.0) * weights[token * INDEX_HEADS + head] * WEIGHT_SCALE;
                }
                (index, score)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|(left_index, left_score), (right_index, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_index.cmp(right_index))
        });
        for (slot, (index, score)) in candidates.into_iter().take(EXTRA_COLS).enumerate() {
            if score.is_finite() {
                expected[row + WINDOW_SIZE + slot] = (TOKENS + index) as i32;
            }
        }
    }

    let context = CudaArtifactOperatorContext::new().expect("CUDA artifact context");
    let query = context
        .upload_f32_buffer(&query)
        .expect("upload prefill index query");
    let weights = context
        .upload_f32_buffer(&weights)
        .expect("upload prefill index weights");
    let indexer_kv = context
        .upload_f32_buffer(&indexer_kv)
        .expect("upload prefill indexer KV");
    let actual = context
        .dsv4_prefill_topk_indices_from_device(
            Some(&query),
            Some(&weights),
            Some(&indexer_kv),
            TOKENS,
            WINDOW_SIZE,
            WINDOW_SIZE,
            EXTRA_COLS,
            TOKENS,
            COMPRESS_RATIO,
            COMPRESSED_LEN,
            INDEX_HEADS,
            INDEX_HEAD_DIM,
            WEIGHT_SCALE,
        )
        .expect("launch prefill top-k");
    context.sync_stream().expect("synchronize prefill top-k");
    let actual = context
        .download_i32_buffer(&actual)
        .expect("download prefill top-k");

    assert_eq!(actual, expected, "prefill top-k indices");
}
