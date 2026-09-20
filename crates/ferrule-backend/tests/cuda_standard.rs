#![cfg(feature = "cuda")]

use ferrule_backend::cuda::context::CudaOperators;
use ferrule_backend::cuda::standard::{
    PagedF32GqaBuffers, PagedF32GqaLayout, PagedF32GqaMetadata, SelectedSoftmaxTopKLayout,
    SplitHalfRopeLayout,
};

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "element {i}: actual={a}, expected={e}"
        );
    }
}

fn layout(rows: usize) -> PagedF32GqaLayout {
    PagedF32GqaLayout {
        rows,
        sequences: 2,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 3,
        page_tokens: 2,
        layer_index: 1,
        layer_count: 3,
        physical_slots: 5,
        softmax_scale: 1.0 / 3.0f32.sqrt(),
    }
}

#[test]
fn standard_shapes_and_metadata_reject_overflow_holes_and_aliases() {
    let l = layout(5);
    let metadata = PagedF32GqaMetadata {
        block_slots: &[3, 1, 4, 0],
        block_offsets: &[0, 2, 4],
        row_sequence_ids: &[0, 1, 0, 0, 1],
        row_positions: &[0, 0, 1, 2, 1],
        committed_lengths: &[0, 0],
    };
    assert_eq!(metadata.validate(l).unwrap(), [1, 1, 2, 3, 2]);
    assert!(PagedF32GqaLayout { q_heads: 3, ..l }.validate().is_err());
    assert!(PagedF32GqaLayout { kv_heads: 0, ..l }.validate().is_err());
    assert!(
        PagedF32GqaLayout {
            head_dim: usize::MAX,
            ..l
        }
        .validate()
        .is_err()
    );
    assert!(
        PagedF32GqaLayout {
            layer_index: 3,
            ..l
        }
        .validate()
        .is_err()
    );
    assert!(
        PagedF32GqaLayout {
            softmax_scale: f32::NAN,
            ..l
        }
        .validate()
        .is_err()
    );
    for bad in [
        PagedF32GqaMetadata {
            row_positions: &[0, 0, 2, 3, 1],
            ..metadata
        },
        PagedF32GqaMetadata {
            block_slots: &[3, 1, 3, 0],
            ..metadata
        },
        PagedF32GqaMetadata {
            block_slots: &[3, 1, 999, 0],
            ..metadata
        },
        PagedF32GqaMetadata {
            block_offsets: &[0, 3, 2],
            ..metadata
        },
        PagedF32GqaMetadata {
            row_sequence_ids: &[0, -1, 0, 0, 1],
            ..metadata
        },
        PagedF32GqaMetadata {
            committed_lengths: &[i32::MAX, 0],
            ..metadata
        },
    ] {
        assert!(bad.validate(l).is_err(), "{bad:?}");
    }
    let shared = PagedF32GqaMetadata {
        block_slots: &[3, 1, 3, 0],
        block_offsets: &[0, 2, 4],
        row_sequence_ids: &[0, 1],
        row_positions: &[2, 2],
        committed_lengths: &[2, 2],
    };
    assert!(shared.validate(layout(2)).is_ok());
    // Sequence zero would overwrite sequence one's initialized prefix cell.
    assert!(
        PagedF32GqaMetadata {
            block_slots: &[3, 1, 1, 0],
            ..shared
        }
        .validate(layout(2))
        .is_err()
    );
}

#[test]
#[ignore = "requires one native CUDA GPU; clamped F32 SwiGLU CPU reference and owner checks"]
fn standard_f32_clamped_swiglu_matches_cpu_and_checks_contract() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let gates = [
        -30.0f32, -17.0, -8.0, -2.5, -0.37, -0.0, 0.0, 0.19, 1.5, 2.5, 7.3, 80.0,
    ];
    let ups = [
        10000.0f32, 4.0, -4.0, 3.75, -0.51, 1.0, 4.13, -0.71, -5.0, -2.5, 0.73, 10000.0,
    ];
    // Cross the native 256-thread block boundary with a partially filled block.
    let gates: Vec<_> = gates.into_iter().cycle().take(257).collect();
    let ups: Vec<_> = ups.into_iter().cycle().take(257).collect();
    let g = op.upload_f32_buffer(&gates).unwrap();
    let u = op.upload_f32_buffer(&ups).unwrap();
    let mut output = op.zero_f32_buffer(gates.len()).unwrap();
    for limit in [0.0f32, 0.75, 2.5, 18.0] {
        op.swiglu_f32_clamped_into(&g, &u, &mut output, limit)
            .unwrap();
        let expected: Vec<f32> = gates
            .iter()
            .zip(&ups)
            .map(|(&g, &u)| {
                let gate = f64::from(g.min(limit));
                let up = f64::from(u.clamp(-limit, limit));
                (gate / (1.0 + (-gate).exp()) * up) as f32
            })
            .collect();
        let actual = op.download_f32_buffer(&output).unwrap();
        close(&actual, &expected, 3e-7);
        if limit > 0.0 {
            // The gate has no lower clamp and standard SiLU has no -16 cutoff.
            assert!(actual[0] < 0.0 && actual[1] < 0.0);
            assert!((actual[0] - expected[0]).abs() <= expected[0].abs() * 2e-6);
            assert!(actual.iter().any(|v| v.to_bits() & 0xffff != 0));
        }
    }
    close(&op.download_f32_buffer(&g).unwrap(), &gates, 0.0);
    close(&op.download_f32_buffer(&u).unwrap(), &ups, 0.0);
    // The existing unclamped entry must remain unclamped after a clipped launch.
    op.swiglu_f32_into(&g, &u, &mut output).unwrap();
    let expected: Vec<f32> = gates
        .iter()
        .zip(&ups)
        .map(|(&g, &u)| {
            let gate = f64::from(g);
            (gate / (1.0 + (-gate).exp()) * f64::from(u)) as f32
        })
        .collect();
    let before = op.download_f32_buffer(&output).unwrap();
    close(&before, &expected, 3e-7);
    for limit in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            op.swiglu_f32_clamped_into(&g, &u, &mut output, limit)
                .is_err()
        );
    }
    let short = op.zero_f32_buffer(1).unwrap();
    let mut short_output = op.upload_f32_buffer(&[73.0]).unwrap();
    assert!(
        op.swiglu_f32_clamped_into(&short, &u, &mut output, 2.5)
            .is_err()
    );
    assert!(
        op.swiglu_f32_clamped_into(&g, &short, &mut output, 2.5)
            .is_err()
    );
    assert!(
        op.swiglu_f32_clamped_into(&g, &u, &mut short_output, 2.5)
            .is_err()
    );
    let empty = op.zero_f32_buffer(0).unwrap();
    let mut empty_output = op.zero_f32_buffer(0).unwrap();
    assert!(
        op.swiglu_f32_clamped_into(&empty, &empty, &mut empty_output, 2.5)
            .is_err()
    );
    let other = foreign.upload_f32_buffer(&gates).unwrap();
    let mut other_output = foreign.upload_f32_buffer(&gates).unwrap();
    assert!(
        op.swiglu_f32_clamped_into(&other, &u, &mut output, 2.5)
            .is_err()
    );
    assert!(
        op.swiglu_f32_clamped_into(&g, &other, &mut output, 2.5)
            .is_err()
    );
    assert!(
        op.swiglu_f32_clamped_into(&g, &u, &mut other_output, 2.5)
            .is_err()
    );
    close(&op.download_f32_buffer(&output).unwrap(), &before, 0.0);
    close(
        &op.download_f32_buffer(&short_output).unwrap(),
        &[73.0],
        0.0,
    );
    close(
        &foreign.download_f32_buffer(&other_output).unwrap(),
        &gates,
        0.0,
    );
}

#[test]
#[ignore = "requires one native CUDA GPU; CPU-reference standard F32 math"]
fn standard_f32_norm_rope_swiglu_residual_embedding_and_linear() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let rows = 3;
    let width = 257;
    let x: Vec<f32> = (0..rows * width)
        .map(|i| ((i * 17 % 131) as f32 - 65.0) / 37.13)
        .collect();
    let w: Vec<f32> = (0..width)
        .map(|i| 0.731 + (i % 19) as f32 * 0.0137)
        .collect();
    let xd = op.upload_f32_buffer(&x).unwrap();
    let wd = op.upload_f32_buffer(&w).unwrap();
    let mut y = op.zero_f32_buffer(x.len()).unwrap();
    op.rms_norm_f32_into(&xd, rows, &wd, 1e-5, &mut y).unwrap();
    let mut expected = Vec::new();
    for row in x.chunks_exact(width) {
        let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-5).sqrt();
        expected.extend(row.iter().zip(&w).map(|(&v, &w)| v * inv * w));
    }
    let norm = op.download_f32_buffer(&y).unwrap();
    close(&norm, &expected, 3e-6);
    assert!(norm.iter().any(|v| v.to_bits() & 0xffff != 0));
    let legacy = op.rms_norm_rows_from_device(&xd, rows, &wd, 1e-5).unwrap();
    assert!(
        op.download_f32_buffer(&legacy)
            .unwrap()
            .iter()
            .all(|v| v.to_bits() & 0xffff == 0)
    );

    let rope = SplitHalfRopeLayout {
        rows: 3,
        heads: 2,
        head_dim: 7,
        rope_dim: 6,
        table_positions: 8,
        restore_bf16_boundary: false,
    };
    let input: Vec<f32> = (0..rope.value_elements().unwrap())
        .map(|i| (i as f32 - 11.0) * 0.0371)
        .collect();
    let cos: Vec<f32> = (0..rope.table_elements().unwrap())
        .map(|i| (i as f32 * 0.173).cos())
        .collect();
    let sin: Vec<f32> = (0..rope.table_elements().unwrap())
        .map(|i| (i as f32 * 0.173).sin())
        .collect();
    let positions = [5, 0, 2];
    let mut rotated = input.clone();
    for (row, &position) in positions.iter().enumerate() {
        for head in 0..2 {
            for pair in 0..3 {
                let base = (row * 2 + head) * 7;
                let table = position as usize * 3 + pair;
                let a = input[base + pair];
                let b = input[base + 3 + pair];
                rotated[base + pair] = a * cos[table] - b * sin[table];
                rotated[base + 3 + pair] = a * sin[table] + b * cos[table];
            }
        }
    }
    let mut values = op.upload_f32_buffer(&input).unwrap();
    let cosine = op.upload_f32_buffer(&cos).unwrap();
    let sine = op.upload_f32_buffer(&sin).unwrap();
    op.split_half_rope_f32(&mut values, &cosine, &sine, &positions, rope)
        .unwrap();
    close(&op.download_f32_buffer(&values).unwrap(), &rotated, 2e-6);
    assert!(
        op.split_half_rope_f32(&mut values, &cosine, &sine, &[8, 0, 2], rope)
            .is_err()
    );

    let gates: [f32; 7] = [-30.0, -17.0, -1.13, 0.0, 0.37, 2.15, 19.0];
    let ups = [1.0, 10000.0, -0.75, 3.17, 0.51, 1.23, -0.73];
    let g = op.upload_f32_buffer(&gates).unwrap();
    let u = op.upload_f32_buffer(&ups).unwrap();
    let mut activation = op.zero_f32_buffer(gates.len()).unwrap();
    op.swiglu_f32_into(&g, &u, &mut activation).unwrap();
    let expected: Vec<_> = gates
        .iter()
        .zip(ups)
        .map(|(&g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    close(
        &op.download_f32_buffer(&activation).unwrap(),
        &expected,
        2e-6,
    );
    let sum: Vec<_> = gates.iter().zip(ups).map(|(&g, u)| g + u).collect();
    op.residual_add_f32_into(&g, &u, &mut activation).unwrap();
    close(&op.download_f32_buffer(&activation).unwrap(), &sum, 0.0);
    let mut inplace = op.upload_f32_buffer(&gates).unwrap();
    op.residual_add_f32_in_place(&u, &mut inplace).unwrap();
    close(&op.download_f32_buffer(&inplace).unwrap(), &sum, 0.0);

    let embedding = op
        .upload_f32_buffer(&[0.17, 0.23, 0.31, 0.47, 0.59, 0.61])
        .unwrap();
    let gathered = op.embedding_f32(&embedding, &[2, 0, 2], 2).unwrap();
    close(
        &op.download_f32_buffer(&gathered).unwrap(),
        &[0.59, 0.61, 0.17, 0.23, 0.59, 0.61],
        0.0,
    );
    assert!(op.embedding_f32(&embedding, &[3], 2).is_err());
    let indices = op.upload_i32_buffer(&[-1, 3, i32::MAX, 1]).unwrap();
    let bounded = op.gather_f32_rows(&embedding, &indices, 4, 2).unwrap();
    close(
        &op.download_f32_buffer(&bounded).unwrap(),
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.31, 0.47],
        0.0,
    );
    let weight: [f32; 6] = [0.13, -0.79, 1.23, 0.47, 0.71, -1.19];
    let bytes: Vec<_> = weight.iter().flat_map(|v| v.to_le_bytes()).collect();
    let handle = op.upload_f32_linear(&bytes, 3, 2).unwrap();
    let mut linear = op.zero_f32_buffer(9).unwrap();
    op.linear_f32_into(&handle, &gathered, 3, &mut linear)
        .unwrap();
    let mut expected = Vec::new();
    for row in [[0.59, 0.61], [0.17, 0.23], [0.59, 0.61]] {
        expected.extend(
            weight
                .as_chunks::<2>()
                .0
                .iter()
                .map(|w| row[0] * w[0] + row[1] * w[1]),
        );
    }
    // Inputs may be dropped immediately after enqueue; allocator retirement holds them.
    drop(handle);
    drop(gathered);
    let _churn = op.zero_f32_buffer(8192).unwrap();
    close(&op.download_f32_buffer(&linear).unwrap(), &expected, 2e-6);
}

fn cache_index(
    l: PagedF32GqaLayout,
    slots: &[i32],
    offsets: &[i32],
    seq: usize,
    pos: usize,
    h: usize,
    d: usize,
) -> usize {
    let slot = slots[offsets[seq] as usize + pos / l.page_tokens] as usize;
    ((((slot * l.layer_count + l.layer_index) * l.page_tokens + pos % l.page_tokens) * l.kv_heads
        + h)
        * l.head_dim)
        + d
}

fn gqa_reference(
    l: PagedF32GqaLayout,
    m: PagedF32GqaMetadata<'_>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    kc: &mut [f32],
    vc: &mut [f32],
) -> Vec<f32> {
    // Deliberately sequential append-and-attend oracle: production appends all first.
    let mut output = vec![0.0; q.len()];
    for row in 0..l.rows {
        let seq = m.row_sequence_ids[row] as usize;
        let pos = m.row_positions[row] as usize;
        for h in 0..l.kv_heads {
            for d in 0..l.head_dim {
                let index = cache_index(l, m.block_slots, m.block_offsets, seq, pos, h, d);
                kc[index] = k[(row * l.kv_heads + h) * l.head_dim + d];
                vc[index] = v[(row * l.kv_heads + h) * l.head_dim + d];
            }
        }
        for h in 0..l.q_heads {
            let kh = h / (l.q_heads / l.kv_heads);
            let scores: Vec<f32> = (0..=pos)
                .map(|t| {
                    (0..l.head_dim)
                        .map(|d| {
                            q[(row * l.q_heads + h) * l.head_dim + d]
                                * kc[cache_index(l, m.block_slots, m.block_offsets, seq, t, kh, d)]
                        })
                        .sum::<f32>()
                        * l.softmax_scale
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
            let denom: f32 = weights.iter().sum();
            for d in 0..l.head_dim {
                output[(row * l.q_heads + h) * l.head_dim + d] = weights
                    .iter()
                    .enumerate()
                    .map(|(t, w)| {
                        w * vc[cache_index(l, m.block_slots, m.block_offsets, seq, t, kh, d)]
                    })
                    .sum::<f32>()
                    / denom;
            }
        }
    }
    output
}

#[test]
#[ignore = "requires one native CUDA GPU; ragged causal prefill and decode CPU reference"]
fn standard_f32_paged_gqa_prefill_and_decode() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let mut kc = vec![-777.25; layout(5).cache_elements().unwrap()];
    let mut vc = kc.clone();
    let mut kd = op.upload_f32_buffer(&kc).unwrap();
    let mut vd = op.upload_f32_buffer(&vc).unwrap();
    for (seq, pos, committed) in [
        (vec![0, 1, 0, 0, 1], vec![0, 0, 1, 2, 1], vec![0, 0]),
        (vec![1, 0], vec![2, 3], vec![3, 2]),
    ] {
        let l = layout(seq.len());
        let m = PagedF32GqaMetadata {
            block_slots: &[3, 1, 4, 0],
            block_offsets: &[0, 2, 4],
            row_sequence_ids: &seq,
            row_positions: &pos,
            committed_lengths: &committed,
        };
        let q: Vec<_> = (0..l.query_elements().unwrap())
            .map(|i| (i as f32 * 0.171).sin() * 2.7)
            .collect();
        let k: Vec<_> = (0..l.append_elements().unwrap())
            .map(|i| (i as f32 * 0.271).cos() * 1.3)
            .collect();
        let v: Vec<_> = (0..k.len()).map(|i| i as f32 * 0.137 - 2.71).collect();
        let expected = gqa_reference(l, m, &q, &k, &v, &mut kc, &mut vc);
        let qd = op.upload_f32_buffer(&q).unwrap();
        let ak = op.upload_f32_buffer(&k).unwrap();
        let av = op.upload_f32_buffer(&v).unwrap();
        let mut output = op.zero_f32_buffer(q.len()).unwrap();
        op.append_and_attend_paged_f32(
            PagedF32GqaBuffers {
                query: &qd,
                append_key: &ak,
                append_value: &av,
                key_cache: &mut kd,
                value_cache: &mut vd,
                output: &mut output,
            },
            m,
            l,
        )
        .unwrap();
        close(&op.download_f32_buffer(&output).unwrap(), &expected, 3e-6);
        // Exact F32 cache copies, untouched layers/pages retain the sentinel.
        close(&op.download_f32_buffer(&kd).unwrap(), &kc, 0.0);
        close(&op.download_f32_buffer(&vd).unwrap(), &vc, 0.0);
        // Output projection hidden=5 differs from q_heads*head_dim=12.
        let weight = vec![0.123f32; 5 * 12];
        let bytes: Vec<_> = weight.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weight = op.upload_f32_linear(&bytes, 5, 12).unwrap();
        let mut hidden = op.zero_f32_buffer(l.rows * 5).unwrap();
        op.linear_f32_into(&weight, &output, l.rows, &mut hidden)
            .unwrap();
        let expected: Vec<_> = expected
            .as_chunks::<12>()
            .0
            .iter()
            .flat_map(|row| [row.iter().sum::<f32>() * 0.123; 5])
            .collect();
        close(&op.download_f32_buffer(&hidden).unwrap(), &expected, 4e-6);
    }
}

#[test]
#[ignore = "requires one native CUDA GPU; deterministic router and ordered F32 combine"]
fn standard_router_combine_and_owner_rejection() {
    let op = CudaOperators::new_on_device(0).unwrap();
    let foreign = CudaOperators::new_on_device(0).unwrap();
    let logits = op.upload_f32_buffer(&[1.0, 2.0, 2.0, 0.0, 2.0]).unwrap();
    let mut ids = op.zero_i32_buffer(3).unwrap();
    let mut weights = op.zero_f32_buffer(3).unwrap();
    let routing = SelectedSoftmaxTopKLayout {
        rows: 1,
        experts: 5,
        top_k: 3,
        output_scale: 1.0,
    };
    op.router_softmax_topk_f32_into(&logits, &mut ids, &mut weights, routing)
        .unwrap();
    assert_eq!(op.download_i32_buffer(&ids).unwrap(), [1, 2, 4]);
    close(
        &op.download_f32_buffer(&weights).unwrap(),
        &[1.0 / 3.0; 3],
        1e-7,
    );
    let values = op
        .upload_f32_buffer(&[1e20, 0.13, 3.0, 0.71, -1e20, -0.27, 1.0, 0.19])
        .unwrap();
    let routes = op.upload_i32_buffer(&[0, 1, 0, 0]).unwrap();
    let weights = op.upload_f32_buffer(&[1.0; 4]).unwrap();
    let mut output = op.zero_f32_buffer(4).unwrap();
    for _ in 0..4 {
        op.weighted_combine_f32_into(&values, &routes, &weights, &mut output, 2, 2)
            .unwrap();
        close(
            &op.download_f32_buffer(&output).unwrap(),
            &[1.0, (0.13f32 - 0.27) + 0.19, 3.0, 0.71],
            0.0,
        );
    }
    let a = op.upload_f32_buffer(&[1.0; 4]).unwrap();
    let b = foreign.upload_f32_buffer(&[1.0; 4]).unwrap();
    let before = op.download_f32_buffer(&output).unwrap();
    assert!(op.swiglu_f32_into(&b, &a, &mut output).is_err());
    assert!(op.swiglu_f32_into(&a, &b, &mut output).is_err());
    assert!(op.rms_norm_f32_into(&b, 1, &a, 1e-5, &mut output).is_err());
    assert!(op.rms_norm_f32_into(&a, 1, &b, 1e-5, &mut output).is_err());
    assert!(op.residual_add_f32_into(&a, &b, &mut output).is_err());
    assert!(op.residual_add_f32_in_place(&b, &mut output).is_err());
    assert!(op.embedding_f32(&b, &[0], 4).is_err());
    assert!(op.gather_f32_rows(&b, &routes, 4, 2).is_err());
    let other_routes = foreign.upload_i32_buffer(&[0, 1, 0, 0]).unwrap();
    assert!(op.gather_f32_rows(&a, &other_routes, 4, 2).is_err());
    assert!(
        op.weighted_combine_f32_into(&values, &other_routes, &weights, &mut output, 2, 2)
            .is_err()
    );
    let foreign_weights = foreign.upload_f32_buffer(&[1.0; 4]).unwrap();
    assert!(
        op.weighted_combine_f32_into(&values, &routes, &foreign_weights, &mut output, 2, 2)
            .is_err()
    );
    close(&op.download_f32_buffer(&output).unwrap(), &before, 0.0);
    let mut foreign_output = foreign.zero_f32_buffer(4).unwrap();
    assert!(op.swiglu_f32_into(&a, &a, &mut foreign_output).is_err());
    assert!(
        op.residual_add_f32_into(&a, &a, &mut foreign_output)
            .is_err()
    );
    assert!(
        op.rms_norm_f32_into(&a, 1, &a, 1e-5, &mut foreign_output)
            .is_err()
    );
    let weight_bytes: Vec<_> = [1.0f32; 16].iter().flat_map(|v| v.to_le_bytes()).collect();
    let wrong_weight = foreign.upload_f32_linear(&weight_bytes, 4, 4).unwrap();
    assert!(
        op.linear_f32_into(&wrong_weight, &a, 1, &mut output)
            .is_err()
    );
    let rope = SplitHalfRopeLayout {
        rows: 1,
        heads: 1,
        head_dim: 4,
        rope_dim: 4,
        table_positions: 2,
        restore_bf16_boundary: false,
    };
    assert!(
        op.split_half_rope_f32(&mut output, &a, &b, &[0], rope)
            .is_err()
    );
    assert!(
        op.split_half_rope_f32(&mut foreign_output, &a, &a, &[0], rope)
            .is_err()
    );
    let foreign_logits = foreign.upload_f32_buffer(&[1.0; 5]).unwrap();
    let mut router_weights = op.zero_f32_buffer(3).unwrap();
    assert!(
        op.router_softmax_topk_f32_into(&foreign_logits, &mut ids, &mut router_weights, routing)
            .is_err()
    );
    let mut foreign_ids = foreign.zero_i32_buffer(3).unwrap();
    assert!(
        op.router_softmax_topk_f32_into(&logits, &mut foreign_ids, &mut router_weights, routing)
            .is_err()
    );
    // The selected-softmax implementation is not subject to legacy k<=64.
    let wide_logits = op.upload_f32_buffer(&[0.5; 70]).unwrap();
    let mut wide_ids = op.zero_i32_buffer(65).unwrap();
    let mut wide_weights = op.zero_f32_buffer(65).unwrap();
    op.router_softmax_topk_f32_into(
        &wide_logits,
        &mut wide_ids,
        &mut wide_weights,
        SelectedSoftmaxTopKLayout {
            rows: 1,
            experts: 70,
            top_k: 65,
            output_scale: 1.0,
        },
    )
    .unwrap();
    assert_eq!(
        op.download_i32_buffer(&wide_ids).unwrap(),
        (0..65).collect::<Vec<i32>>()
    );
    close(
        &op.download_f32_buffer(&wide_weights).unwrap(),
        &[1.0 / 65.0; 65],
        1e-7,
    );

    let l = layout(1);
    let m = PagedF32GqaMetadata {
        block_slots: &[3, 1, 4, 0],
        block_offsets: &[0, 2, 4],
        row_sequence_ids: &[0],
        row_positions: &[0],
        committed_lengths: &[0, 0],
    };
    let q = op.zero_f32_buffer(l.query_elements().unwrap()).unwrap();
    let append = op.zero_f32_buffer(l.append_elements().unwrap()).unwrap();
    let wrong_append = foreign.zero_f32_buffer(append.len()).unwrap();
    let mut cache = op.zero_f32_buffer(l.cache_elements().unwrap()).unwrap();
    let mut other_cache = op.zero_f32_buffer(cache.len()).unwrap();
    let mut wrong_cache = foreign.zero_f32_buffer(cache.len()).unwrap();
    let mut out = op.zero_f32_buffer(q.len()).unwrap();
    assert!(
        op.append_and_attend_paged_f32(
            PagedF32GqaBuffers {
                query: &q,
                append_key: &wrong_append,
                append_value: &append,
                key_cache: &mut cache,
                value_cache: &mut other_cache,
                output: &mut out
            },
            m,
            l
        )
        .is_err()
    );
    assert!(
        op.append_and_attend_paged_f32(
            PagedF32GqaBuffers {
                query: &q,
                append_key: &append,
                append_value: &append,
                key_cache: &mut cache,
                value_cache: &mut wrong_cache,
                output: &mut out
            },
            m,
            l
        )
        .is_err()
    );
    close(
        &op.download_f32_buffer(&cache).unwrap(),
        &vec![0.0; cache.len()],
        0.0,
    );
}
