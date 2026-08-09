#![cfg(feature = "cuda")]

//! Provider-neutral semantic operator correctness tests on CUDA.

use std::sync::{Mutex, MutexGuard};

use ferrule_backend::cuda::operators::attention::{
    CudaOperators, PagedBf16CausalGqaLayout, PagedBf16GqaBuffers,
};
use ferrule_backend::cuda::operators::kv::CudaKvPagePool;
use ferrule_backend::cuda::operators::linear::{
    F32ToBf16RowsLayout, StridedBf16RowsLayout, StridedF32RowsLayout,
};
use ferrule_backend::cuda::operators::moe::{Bf16MoeRowsLayout, SelectedSoftmaxTopKLayout};
use ferrule_backend::cuda::operators::rope::SplitHalfRopeLayout;
use ferrule_common::execution::{KvBlockId, KvElementType, KvPageId, KvPlaneDescriptor};

static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

fn cuda_test_guard() -> MutexGuard<'static, ()> {
    CUDA_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn cuda_context() -> Option<CudaOperators> {
    match CudaOperators::new() {
        Ok(context) => Some(context),
        Err(error) => {
            eprintln!("skipping CUDA semantic operator test: {error}");
            None
        }
    }
}

fn bf16_word(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let bias = 0x7fff + ((bits >> 16) & 1);
    bits.wrapping_add(bias).wrapping_shr(16) as u16
}

fn bf16_value(word: u16) -> f32 {
    f32::from_bits(u32::from(word) << 16)
}

fn bf16_round(value: f32) -> f32 {
    bf16_value(bf16_word(value))
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            actual.is_finite() && error <= tolerance.max(expected.abs() * tolerance),
            "{label}[{index}] actual={actual} expected={expected} error={error}"
        );
    }
}

#[test]
fn split_half_indexed_rope_matches_cpu_reference() {
    let _guard = cuda_test_guard();
    const ROWS: usize = 3;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 8;
    const ROPE_DIM: usize = 6;
    const POSITIONS: usize = 5;
    let layout = SplitHalfRopeLayout {
        rows: ROWS,
        heads: HEADS,
        head_dim: HEAD_DIM,
        rope_dim: ROPE_DIM,
        table_positions: POSITIONS,
        restore_bf16_boundary: true,
    };
    let positions = [4, 0, 2];
    let cosine = (0..POSITIONS * (ROPE_DIM / 2))
        .map(|index| ((index as f32 + 1.0) * 0.17).cos())
        .collect::<Vec<_>>();
    let sine = (0..POSITIONS * (ROPE_DIM / 2))
        .map(|index| ((index as f32 + 1.0) * 0.17).sin())
        .collect::<Vec<_>>();
    let input = (0..ROWS * HEADS * HEAD_DIM)
        .map(|index| index as f32 * 0.03125 - 0.75)
        .collect::<Vec<_>>();
    let mut expected = input.clone();
    let half = ROPE_DIM / 2;
    for (row, &position) in positions.iter().enumerate() {
        for head in 0..HEADS {
            let base = (row * HEADS + head) * HEAD_DIM;
            for pair in 0..half {
                let table = position * half + pair;
                let first = expected[base + pair];
                let second = expected[base + half + pair];
                expected[base + pair] = bf16_round(first * cosine[table] - second * sine[table]);
                expected[base + half + pair] =
                    bf16_round(first * sine[table] + second * cosine[table]);
            }
        }
    }

    let Some(context) = cuda_context() else {
        return;
    };
    let mut values = context
        .upload_f32_buffer(&input)
        .expect("upload RoPE input");
    let cosine = context
        .upload_f32_buffer(&cosine)
        .expect("upload RoPE cosine");
    let sine = context.upload_f32_buffer(&sine).expect("upload RoPE sine");
    let positions = context
        .upload_i32_buffer(&positions.map(|value| value as i32))
        .expect("upload RoPE positions");
    context
        .apply_split_half_rope(&mut values, &cosine, &sine, &positions, layout, false)
        .expect("launch split-half RoPE");
    let actual = context
        .download_f32_buffer(&values)
        .expect("download RoPE output");
    assert_close(&actual, &expected, 0.0, "split-half RoPE");
    for row in 0..ROWS {
        for head in 0..HEADS {
            let untouched = (row * HEADS + head) * HEAD_DIM + ROPE_DIM;
            assert_eq!(actual[untouched], input[untouched]);
            assert_eq!(actual[untouched + 1], input[untouched + 1]);
        }
    }
}

#[test]
fn f32_to_bf16_rne_rows_matches_exact_words_and_strides() {
    let _guard = cuda_test_guard();
    let Some(context) = cuda_context() else {
        return;
    };
    let layout = F32ToBf16RowsLayout {
        input: StridedF32RowsLayout {
            rows: 2,
            heads: 2,
            dimensions: 3,
            row_stride_bytes: 48,
            head_stride_bytes: 20,
        },
        output: StridedBf16RowsLayout {
            rows: 2,
            heads: 2,
            dimensions: 3,
            row_stride_bytes: 20,
            head_stride_bytes: 8,
        },
    };
    layout.validate().expect("strided conversion layout");

    let logical = [
        f32::from_bits(0x3f80_8000),
        f32::from_bits(0x3f81_8000),
        -0.0,
        f32::from_bits(0x7f80_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x0000_8000),
        f32::from_bits(0x0001_8000),
        1.5,
        -2.25,
        0.333_251_95,
        -0.333_251_95,
    ];
    let input_indices = [0, 1, 2, 5, 6, 7, 12, 13, 14, 17, 18, 19];
    let output_indices = [0, 1, 2, 4, 5, 6, 10, 11, 12, 14, 15, 16];
    let mut input = vec![f32::from_bits(0x7fc0_1234); 20];
    for (&index, &value) in input_indices.iter().zip(&logical) {
        input[index] = value;
    }

    let input = context
        .upload_f32_buffer(&input)
        .expect("upload strided F32 rows");
    let mut output = context
        .zero_bf16_buffer(layout.output.required_bytes().unwrap() / 2)
        .expect("allocate strided BF16 rows");
    context
        .f32_rows_to_bf16_rne_into(&input, &mut output, layout)
        .expect("convert F32 rows to BF16");
    let actual = output
        .as_device_buffer()
        .to_host_vec(&context.stream_clone())
        .expect("download raw BF16 words");
    let mut expected = vec![0u16; actual.len()];
    for ((&index, &value), &source_index) in output_indices.iter().zip(&logical).zip(&input_indices)
    {
        assert_eq!(
            input.as_device_buffer().len(),
            20,
            "source index {source_index}"
        );
        expected[index] = bf16_word(value);
    }
    assert_eq!(actual, expected);
    assert_eq!(actual[0], 0x3f80, "even lower BF16 tie rounds down");
    assert_eq!(actual[1], 0x3f82, "odd lower BF16 tie rounds up");
    assert_eq!(actual[4], 0x7fc0, "NaN payload remains non-zero");
}

#[test]
fn typed_kv_pool_borrows_distinct_planes_and_lowers_blocks() {
    let _guard = cuda_test_guard();
    let Some(context) = cuda_context() else {
        return;
    };
    let planes = [
        KvPlaneDescriptor::new("key", 2, 1, KvElementType::Bf16),
        KvPlaneDescriptor::new("value", 3, 1, KvElementType::Bf16),
    ];
    let mut pool = CudaKvPagePool::new(&context, &planes, 2, 3).expect("allocate typed KV pool");
    assert_eq!(pool.planes(), &planes);
    assert!(pool.bf16_plane_pair_mut(0, 0).is_err());
    {
        let (value, key) = pool
            .bf16_plane_pair_mut(1, 0)
            .expect("borrow reversed distinct BF16 planes");
        assert_ne!(
            value.as_device_buffer().cu_deviceptr(),
            key.as_device_buffer().cu_deviceptr()
        );
        let stream = context.stream_clone();
        value
            .as_device_buffer()
            .copy_from_host(&stream, &vec![0x4000; value.len()])
            .expect("write value plane");
        key.as_device_buffer()
            .copy_from_host(&stream, &vec![0x3f80; key.len()])
            .expect("write key plane");
        assert_eq!(
            value
                .as_device_buffer()
                .to_host_vec(&stream)
                .expect("download value plane"),
            vec![0x4000; value.len()]
        );
        assert_eq!(
            key.as_device_buffer()
                .to_host_vec(&stream)
                .expect("download key plane"),
            vec![0x3f80; key.len()]
        );
    }

    let first = KvPageId(41);
    let second = KvPageId(9);
    pool.ensure(&context, first).expect("map first page");
    pool.ensure(&context, second).expect("map second page");
    let expected = [
        i32::try_from(pool.physical_slot(second).unwrap()).unwrap(),
        i32::try_from(pool.physical_slot(first).unwrap()).unwrap(),
        i32::try_from(pool.physical_slot(second).unwrap()).unwrap(),
    ];
    assert_eq!(
        pool.physical_slots_for_pages(&[second, first, second])
            .unwrap(),
        expected
    );
    assert_eq!(
        pool.physical_slots_for_block_ids(&[
            KvBlockId::new(second.0),
            KvBlockId::new(first.0),
            KvBlockId::new(second.0),
        ])
        .unwrap(),
        expected
    );
    assert!(pool.physical_slots_for_pages(&[KvPageId(99)]).is_err());
}

#[test]
fn paged_gqa_wrapper_reports_device_metadata_status() {
    let _guard = cuda_test_guard();
    let Some(context) = cuda_context() else {
        return;
    };
    let layout = PagedBf16CausalGqaLayout::packed(1, 1, 2, 1, 2, 2, 0, 1, 1, 0.5)
        .expect("paged GQA error layout");
    let query = context.upload_f32_buffer(&[1.0; 4]).unwrap();
    let append_key = context.upload_bf16_words(&[0x3f80; 2]).unwrap();
    let append_value = context.upload_bf16_words(&[0x4000; 2]).unwrap();
    let mut key_cache = context.zero_bf16_buffer(4).unwrap();
    let mut value_cache = context.zero_bf16_buffer(4).unwrap();
    let block_slots = context.upload_i32_buffer(&[-1]).unwrap();
    let block_offsets = context.upload_i32_buffer(&[0, 1]).unwrap();
    let sequence_ids = context.upload_i32_buffer(&[0]).unwrap();
    let positions = context.upload_i32_buffer(&[0]).unwrap();
    let kv_lens = context.upload_i32_buffer(&[0]).unwrap();
    let mut output = context.zero_f32_buffer(4).unwrap();
    let mut status = context.zero_i32_buffer(1).unwrap();

    let error = context
        .append_and_attend_paged_bf16(
            PagedBf16GqaBuffers {
                query: &query,
                append_key: &append_key,
                append_value: &append_value,
                key_cache: &mut key_cache,
                value_cache: &mut value_cache,
                block_slots: &block_slots,
                block_offsets: &block_offsets,
                row_sequence_ids: &sequence_ids,
                row_positions: &positions,
                row_kv_lens: &kv_lens,
                output: &mut output,
                status: &mut status,
            },
            layout,
        )
        .expect_err("invalid physical slot must fail after launch");
    assert!(
        error.to_string().contains("page-table metadata"),
        "unexpected checked GQA error: {error}"
    );
    assert_eq!(context.download_i32_buffer(&status).unwrap(), [1]);
}

#[test]
fn stable_selected_softmax_topk_uses_expert_id_ties() {
    let _guard = cuda_test_guard();
    const ROWS: usize = 3;
    const EXPERTS: usize = 7;
    const TOP_K: usize = 4;
    let layout = SelectedSoftmaxTopKLayout {
        rows: ROWS,
        experts: EXPERTS,
        top_k: TOP_K,
        output_scale: 1.25,
    };
    let logits: [f32; ROWS * EXPERTS] = [
        2.0, 2.0, 2.0, 1.0, -3.0, 0.5, 2.0, // ties select 0, 1, 2, 6
        1000.0, 999.0, 998.0, 997.0, -1000.0, 4.0, 3.0, // stability
        -5.0, -1.0, -1.0, -1.0, -2.0, -1.0, -3.0, // ties select 1, 2, 3, 5
    ];
    let mut expected_ids = Vec::with_capacity(ROWS * TOP_K);
    let mut expected_weights = Vec::with_capacity(ROWS * TOP_K);
    for row in logits.chunks_exact(EXPERTS) {
        let mut selected = row.iter().copied().enumerate().collect::<Vec<_>>();
        selected.sort_by(|(first_id, first), (second_id, second)| {
            second
                .total_cmp(first)
                .then_with(|| first_id.cmp(second_id))
        });
        selected.truncate(TOP_K);
        let maximum = selected[0].1;
        let denominator = selected
            .iter()
            .map(|(_, value)| (*value - maximum).exp())
            .sum::<f32>();
        for (expert, value) in selected {
            expected_ids.push(expert as i32);
            expected_weights.push((value - maximum).exp() / denominator * layout.output_scale);
        }
    }
    assert_eq!(&expected_ids[..TOP_K], &[0, 1, 2, 6]);
    assert_eq!(&expected_ids[2 * TOP_K..], &[1, 2, 3, 5]);

    let Some(context) = cuda_context() else {
        return;
    };
    let logits = context
        .upload_f32_buffer(&logits)
        .expect("upload router logits");
    let mut ids = context
        .zero_i32_buffer(layout.output_elements().expect("router outputs"))
        .expect("allocate router IDs");
    let mut weights = context
        .zero_f32_buffer(layout.output_elements().expect("router outputs"))
        .expect("allocate router weights");
    context
        .route_selected_softmax_topk(&logits, &mut ids, &mut weights, layout)
        .expect("launch selected-softmax router");
    assert_eq!(
        context
            .download_i32_buffer(&ids)
            .expect("download router IDs"),
        expected_ids
    );
    let actual_weights = context
        .download_f32_buffer(&weights)
        .expect("download router weights");
    assert_close(
        &actual_weights,
        &expected_weights,
        2e-6,
        "selected-softmax weights",
    );
    for row in actual_weights.chunks_exact(TOP_K) {
        assert!((row.iter().sum::<f32>() - layout.output_scale).abs() <= 2e-6);
    }
}

fn customize_value_plane(layout: &mut PagedBf16CausalGqaLayout, physical_slots: usize) {
    layout.value_cache.head_stride_bytes += 2;
    layout.value_cache.token_stride_bytes = layout.kv_heads * layout.value_cache.head_stride_bytes;
    layout.value_cache.layer_stride_bytes =
        layout.page_tokens * layout.value_cache.token_stride_bytes + 8;
    layout.value_cache.slot_stride_bytes =
        layout.layer_count * layout.value_cache.layer_stride_bytes + 16;
    layout.value_cache.capacity_bytes = physical_slots * layout.value_cache.slot_stride_bytes;
    layout.validate().expect("custom independent value plane");
}

fn append_reference(
    layout: PagedBf16CausalGqaLayout,
    key_cache: &mut [u16],
    value_cache: &mut [u16],
    append_key: &[u16],
    append_value: &[u16],
    block_slots: &[i32],
    block_offsets: &[i32],
    row_sequence_ids: &[i32],
    row_positions: &[i32],
) {
    for row in 0..layout.rows {
        let sequence = row_sequence_ids[row] as usize;
        let position = row_positions[row] as usize;
        for head in 0..layout.kv_heads {
            for dimension in 0..layout.head_dim {
                let input = (row * layout.kv_heads + head) * layout.head_dim + dimension;
                let key_offset = layout
                    .resolve_cache_byte_offset(
                        false,
                        sequence,
                        position,
                        head,
                        dimension,
                        block_slots,
                        block_offsets,
                    )
                    .expect("key reference address")
                    / 2;
                let value_offset = layout
                    .resolve_cache_byte_offset(
                        true,
                        sequence,
                        position,
                        head,
                        dimension,
                        block_slots,
                        block_offsets,
                    )
                    .expect("value reference address")
                    / 2;
                key_cache[key_offset] = append_key[input];
                value_cache[value_offset] = append_value[input];
            }
        }
    }
}

fn combined_gqa_reference(
    layout: PagedBf16CausalGqaLayout,
    query: &[f32],
    key_cache: &[u16],
    value_cache: &[u16],
    block_slots: &[i32],
    block_offsets: &[i32],
    row_sequence_ids: &[i32],
    row_positions: &[i32],
    row_kv_lens: &[i32],
) -> Vec<f32> {
    let mut output = vec![0.0; layout.rows * layout.q_heads * layout.head_dim];
    let queries_per_kv_head = layout.q_heads / layout.kv_heads;
    for row in 0..layout.rows {
        let sequence = row_sequence_ids[row] as usize;
        let position = row_positions[row] as usize;
        let visible = (row_kv_lens[row] as usize)
            .max(position + 1)
            .min(position + 1);
        for query_head in 0..layout.q_heads {
            let kv_head = query_head / queries_per_kv_head;
            let query_base = (row * layout.q_heads + query_head) * layout.head_dim;
            let mut scores = Vec::with_capacity(visible);
            for token in 0..visible {
                let mut score = 0.0f32;
                for dimension in 0..layout.head_dim {
                    let key_offset = layout
                        .resolve_cache_byte_offset(
                            false,
                            sequence,
                            token,
                            kv_head,
                            dimension,
                            block_slots,
                            block_offsets,
                        )
                        .expect("attention key address")
                        / 2;
                    score += query[query_base + dimension] * bf16_value(key_cache[key_offset]);
                }
                scores.push(score * layout.softmax_scale);
            }
            for dimension in 0..layout.head_dim {
                let mut maximum = f32::NEG_INFINITY;
                let mut denominator = 0.0f32;
                let mut accumulator = 0.0f32;
                for (token, &score) in scores.iter().enumerate() {
                    let value_offset = layout
                        .resolve_cache_byte_offset(
                            true,
                            sequence,
                            token,
                            kv_head,
                            dimension,
                            block_slots,
                            block_offsets,
                        )
                        .expect("attention value address")
                        / 2;
                    let value = bf16_value(value_cache[value_offset]);
                    if denominator == 0.0 || score > maximum {
                        let rescale = if denominator == 0.0 {
                            0.0
                        } else {
                            (maximum - score).exp()
                        };
                        accumulator = accumulator * rescale + value;
                        denominator = denominator * rescale + 1.0;
                        maximum = score;
                    } else {
                        let weight = if score == maximum {
                            1.0
                        } else {
                            (score - maximum).exp()
                        };
                        accumulator += weight * value;
                        denominator += weight;
                    }
                }
                output[query_base + dimension] = accumulator / denominator;
            }
        }
    }
    output
}

fn packed_query(rows: usize, q_heads: usize, head_dim: usize, seed: usize) -> Vec<f32> {
    (0..rows * q_heads * head_dim)
        .map(|index| {
            let centered = ((index + seed * 7) % 19) as f32 - 9.0;
            centered * 0.0625
        })
        .collect()
}

fn packed_kv(rows: usize, kv_heads: usize, head_dim: usize, seed: usize) -> (Vec<u16>, Vec<u16>) {
    let elements = rows * kv_heads * head_dim;
    let key = (0..elements)
        .map(|index| {
            let centered = ((index * 3 + seed * 5) % 17) as f32 - 8.0;
            bf16_word(centered * 0.125)
        })
        .collect();
    let value = (0..elements)
        .map(|index| {
            let centered = ((index * 5 + seed * 11) % 23) as f32 - 11.0;
            bf16_word(centered * 0.25 + 2.0)
        })
        .collect();
    (key, value)
}

#[test]
fn paged_bf16_gqa_matches_ragged_prefill_and_decode_reference() {
    let _guard = cuda_test_guard();
    if cuda_context().is_none() {
        eprintln!("skipping paged BF16 GQA: no CUDA device");
        return;
    }
    const SEQUENCES: usize = 2;
    const Q_HEADS: usize = 4;
    const KV_HEADS: usize = 2;
    const HEAD_DIM: usize = 3;
    const PAGE_TOKENS: usize = 2;
    const LAYER_INDEX: usize = 1;
    const LAYER_COUNT: usize = 2;
    const PHYSICAL_SLOTS: usize = 6;
    let block_slots = [4, 1, 3, 0, 5];
    let block_offsets = [0, 2, 5];

    let mut prefill_layout = PagedBf16CausalGqaLayout::packed(
        5,
        SEQUENCES,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        PAGE_TOKENS,
        LAYER_INDEX,
        LAYER_COUNT,
        PHYSICAL_SLOTS,
        (HEAD_DIM as f32).sqrt().recip(),
    )
    .expect("prefill GQA layout");
    customize_value_plane(&mut prefill_layout, PHYSICAL_SLOTS);
    assert_ne!(
        prefill_layout.key_cache.slot_stride_bytes,
        prefill_layout.value_cache.slot_stride_bytes
    );
    let prefill_sequence_ids = [0, 0, 1, 1, 1];
    let prefill_positions = [0, 1, 0, 1, 2];
    let prefill_kv_lens = [0, 0, 0, 0, 0];
    let prefill_query = packed_query(prefill_layout.rows, Q_HEADS, HEAD_DIM, 1);
    let (prefill_key, prefill_value) = packed_kv(prefill_layout.rows, KV_HEADS, HEAD_DIM, 2);
    let mut key_reference = vec![0u16; prefill_layout.key_cache.capacity_bytes / 2];
    let mut value_reference = vec![0u16; prefill_layout.value_cache.capacity_bytes / 2];
    append_reference(
        prefill_layout,
        &mut key_reference,
        &mut value_reference,
        &prefill_key,
        &prefill_value,
        &block_slots,
        &block_offsets,
        &prefill_sequence_ids,
        &prefill_positions,
    );
    let expected_prefill = combined_gqa_reference(
        prefill_layout,
        &prefill_query,
        &key_reference,
        &value_reference,
        &block_slots,
        &block_offsets,
        &prefill_sequence_ids,
        &prefill_positions,
        &prefill_kv_lens,
    );

    let context = CudaOperators::new().expect("CUDA artifact context");
    let block_slots_device = context
        .upload_i32_buffer(&block_slots)
        .expect("upload block slots");
    let block_offsets_device = context
        .upload_i32_buffer(&block_offsets)
        .expect("upload block offsets");
    let mut key_cache = context
        .zero_bf16_buffer(key_reference.len())
        .expect("allocate key cache");
    let mut value_cache = context
        .zero_bf16_buffer(value_reference.len())
        .expect("allocate value cache");
    let query = context
        .upload_f32_buffer(&prefill_query)
        .expect("upload prefill query");
    let append_key = context
        .upload_bf16_words(&prefill_key)
        .expect("upload prefill keys");
    let append_value = context
        .upload_bf16_words(&prefill_value)
        .expect("upload prefill values");
    let sequence_ids = context
        .upload_i32_buffer(&prefill_sequence_ids)
        .expect("upload prefill sequence IDs");
    let positions = context
        .upload_i32_buffer(&prefill_positions)
        .expect("upload prefill positions");
    let kv_lens = context
        .upload_i32_buffer(&prefill_kv_lens)
        .expect("upload prefill KV lengths");
    let mut output = context
        .zero_f32_buffer(prefill_layout.output.required_bytes().unwrap() / 4)
        .expect("allocate prefill output");
    let mut status = context.zero_i32_buffer(1).expect("allocate GQA status");
    context
        .append_and_attend_paged_bf16(
            PagedBf16GqaBuffers {
                query: &query,
                append_key: &append_key,
                append_value: &append_value,
                key_cache: &mut key_cache,
                value_cache: &mut value_cache,
                block_slots: &block_slots_device,
                block_offsets: &block_offsets_device,
                row_sequence_ids: &sequence_ids,
                row_positions: &positions,
                row_kv_lens: &kv_lens,
                output: &mut output,
                status: &mut status,
            },
            prefill_layout,
        )
        .expect("launch ragged prefill GQA");
    assert_eq!(
        context
            .download_i32_buffer(&status)
            .expect("download prefill status"),
        [0]
    );
    let actual_prefill = context
        .download_f32_buffer(&output)
        .expect("download prefill output");
    assert_close(
        &actual_prefill,
        &expected_prefill,
        2e-5,
        "ragged prefill GQA",
    );
    assert!(actual_prefill.iter().any(|value| value.abs() > 0.5));

    let mut decode_layout = PagedBf16CausalGqaLayout::packed(
        2,
        SEQUENCES,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        PAGE_TOKENS,
        LAYER_INDEX,
        LAYER_COUNT,
        PHYSICAL_SLOTS,
        (HEAD_DIM as f32).sqrt().recip(),
    )
    .expect("decode GQA layout");
    decode_layout.key_cache = prefill_layout.key_cache;
    decode_layout.value_cache = prefill_layout.value_cache;
    decode_layout.validate().expect("decode cache reuse layout");
    let decode_sequence_ids = [0, 1];
    let decode_positions = [2, 3];
    let decode_kv_lens = [2, 3];
    let decode_query = packed_query(decode_layout.rows, Q_HEADS, HEAD_DIM, 7);
    let (decode_key, decode_value) = packed_kv(decode_layout.rows, KV_HEADS, HEAD_DIM, 9);
    append_reference(
        decode_layout,
        &mut key_reference,
        &mut value_reference,
        &decode_key,
        &decode_value,
        &block_slots,
        &block_offsets,
        &decode_sequence_ids,
        &decode_positions,
    );
    let expected_decode = combined_gqa_reference(
        decode_layout,
        &decode_query,
        &key_reference,
        &value_reference,
        &block_slots,
        &block_offsets,
        &decode_sequence_ids,
        &decode_positions,
        &decode_kv_lens,
    );
    let query = context
        .upload_f32_buffer(&decode_query)
        .expect("upload decode query");
    let append_key = context
        .upload_bf16_words(&decode_key)
        .expect("upload decode keys");
    let append_value = context
        .upload_bf16_words(&decode_value)
        .expect("upload decode values");
    let sequence_ids = context
        .upload_i32_buffer(&decode_sequence_ids)
        .expect("upload decode sequence IDs");
    let positions = context
        .upload_i32_buffer(&decode_positions)
        .expect("upload decode positions");
    let kv_lens = context
        .upload_i32_buffer(&decode_kv_lens)
        .expect("upload decode KV lengths");
    let mut output = context
        .zero_f32_buffer(decode_layout.output.required_bytes().unwrap() / 4)
        .expect("allocate decode output");
    context
        .append_and_attend_paged_bf16(
            PagedBf16GqaBuffers {
                query: &query,
                append_key: &append_key,
                append_value: &append_value,
                key_cache: &mut key_cache,
                value_cache: &mut value_cache,
                block_slots: &block_slots_device,
                block_offsets: &block_offsets_device,
                row_sequence_ids: &sequence_ids,
                row_positions: &positions,
                row_kv_lens: &kv_lens,
                output: &mut output,
                status: &mut status,
            },
            decode_layout,
        )
        .expect("launch paged decode GQA");
    assert_eq!(
        context
            .download_i32_buffer(&status)
            .expect("download decode status"),
        [0]
    );
    let actual_decode = context
        .download_f32_buffer(&output)
        .expect("download decode output");
    assert_close(&actual_decode, &expected_decode, 2e-5, "paged decode GQA");

    let actual_key_cache = context
        .download_bf16_buffer(&key_cache)
        .expect("download key cache");
    let actual_value_cache = context
        .download_bf16_buffer(&value_cache)
        .expect("download value cache");
    assert_eq!(
        actual_key_cache,
        key_reference
            .iter()
            .copied()
            .map(bf16_value)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        actual_value_cache,
        value_reference
            .iter()
            .copied()
            .map(bf16_value)
            .collect::<Vec<_>>()
    );
}

#[test]
fn bf16_moe_gather_and_weighted_scatter_match_cpu_reference() {
    let _guard = cuda_test_guard();
    let layout = Bf16MoeRowsLayout {
        source_rows: 4,
        route_rows: 6,
        output_rows: 4,
        row_width: 7,
    };
    let source = (0..layout.source_elements().unwrap())
        .map(|index| bf16_word(index as f32 * 0.2 - 1.75))
        .collect::<Vec<_>>();
    let route_rows = [2, 0, 2, 3, -1, 1];
    let route_weights = [0.5, 1.0, -0.25, 0.75, 9.0, -1.5];
    let mut expected_gather = vec![0.0; layout.route_elements().unwrap()];
    for (route, &row) in route_rows.iter().enumerate() {
        let Ok(row) = usize::try_from(row) else {
            continue;
        };
        if row >= layout.source_rows {
            continue;
        }
        for column in 0..layout.row_width {
            expected_gather[route * layout.row_width + column] =
                bf16_value(source[row * layout.row_width + column]);
        }
    }
    let initial = (0..layout.output_elements().unwrap())
        .map(|index| index as f32 * 0.01)
        .collect::<Vec<_>>();
    let mut expected_output = initial.clone();
    for output_row in 0..layout.output_rows {
        for column in 0..layout.row_width {
            let output = output_row * layout.row_width + column;
            for route in 0..layout.route_rows {
                if route_rows[route] == output_row as i32 {
                    expected_output[output] +=
                        bf16_round(expected_gather[route * layout.row_width + column])
                            * route_weights[route];
                }
            }
        }
    }

    let Some(context) = cuda_context() else {
        return;
    };
    let source = context
        .upload_bf16_words(&source)
        .expect("upload BF16 MoE rows");
    let route_rows = context
        .upload_i32_buffer(&route_rows)
        .expect("upload MoE route rows");
    let route_weights_device = context
        .upload_f32_buffer(&route_weights)
        .expect("upload MoE route weights");
    let gathered = context
        .gather_moe_rows(&source, &route_rows, layout)
        .expect("gather BF16 MoE rows");
    let actual_gather = context
        .download_f32_buffer(&gathered)
        .expect("download gathered MoE rows");
    assert_eq!(actual_gather, expected_gather);

    let mut output = context
        .upload_f32_buffer(&initial)
        .expect("upload MoE output base");
    context
        .weighted_scatter_add_moe_rows(
            &gathered,
            &route_rows,
            &route_weights_device,
            &mut output,
            layout,
        )
        .expect("weighted scatter-add MoE rows");
    let actual_output = context
        .download_f32_buffer(&output)
        .expect("download scattered MoE rows");
    assert_close(
        &actual_output,
        &expected_output,
        1e-6,
        "BF16 MoE weighted scatter",
    );
}
