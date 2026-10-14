//! Reconstructed delivery-contract tests. Dense f64 oracle is deliberately
//! independent of online FP32 production state and partition merge.
use ferrule_backend::cpu::{
    StreamingF32GqaConfig as Config, StreamingF32GqaOwnedKv as OwnedKv,
    StreamingF32GqaPartial as Partial, StreamingF32GqaProtection as Protection,
    StreamingF32GqaRequest as Request, streaming_f32_gqa_partial as partial,
};

fn config(dim: usize) -> Config {
    Config {
        rows: 3,
        q_heads: 4,
        kv_heads: 2,
        head_dim: dim,
        block_tokens: 7,
        softmax_scale: 1.0 / (dim as f32).sqrt(),
        sliding_window: None,
    }
}
fn data(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 17 + seed * 7) % 53) as f32 - 26.0) / 19.0)
        .collect()
}
fn oracle(r: Request<'_>) -> Vec<f32> {
    let c = r.config;
    let n = r.key.len() / (c.kv_heads * c.head_dim);
    let mut out = vec![0.0; r.query.len()];
    for row in 0..c.rows {
        for head in 0..c.q_heads {
            let pos = r.query_start + row as u64;
            let q = (row * c.q_heads + head) * c.head_dim;
            let kh = head / (c.q_heads / c.kv_heads);
            let visible: Vec<_> = (0..n)
                .filter(|t| {
                    let kp = r.kv_start + *t as u64;
                    kp <= pos && c.sliding_window.is_none_or(|w| pos - kp < w)
                })
                .collect();
            if visible.is_empty() {
                continue;
            }
            let scores: Vec<f64> = visible
                .iter()
                .map(|t| {
                    let k = (*t * c.kv_heads + kh) * c.head_dim;
                    (0..c.head_dim)
                        .map(|d| f64::from(r.query[q + d]) * f64::from(r.key[k + d]))
                        .sum::<f64>()
                        * f64::from(c.softmax_scale)
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for d in 0..c.head_dim {
                out[q + d] = (visible
                    .iter()
                    .zip(&weights)
                    .map(|(t, w)| w * f64::from(r.value[(*t * c.kv_heads + kh) * c.head_dim + d]))
                    .sum::<f64>()
                    / sum) as f32;
            }
        }
    }
    out
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (a - b).abs() <= 6e-5 * (1.0 + b.abs()),
            "{i}: {a} != {b}"
        );
    }
}
fn fixture(c: Config, tokens: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        data(c.query_elements().unwrap(), 1),
        data(c.kv_elements(tokens).unwrap(), 2),
        data(c.kv_elements(tokens).unwrap(), 3),
    )
}

#[test]
fn cpu_gqa_partitions_match_dense_f64_oracle() {
    for dim in [1, 5, 257] {
        for window in [None, Some(1), Some(9)] {
            let c = Config {
                sliding_window: window,
                ..config(dim)
            };
            let (q, k, v) = fixture(c, 37);
            let r = Request {
                config: c,
                query: &q,
                key: &k,
                value: &v,
                query_start: 129,
                kv_start: 100,
            };
            close(&partial(r).unwrap().output(), &oracle(r));
            for block in [1, 7, 16, 37, 64] {
                let mut merged = Partial::empty(c, r.query_start).unwrap();
                for begin in (0..37).step_by(block) {
                    merged
                        .merge(
                            &partial(r.token_range(begin, (begin + block).min(37)).unwrap())
                                .unwrap(),
                        )
                        .unwrap();
                }
                close(&merged.output(), &oracle(r));
            }
        }
    }
}
#[test]
fn cpu_long_offsets_tail_and_bounded_state() {
    let c = Config {
        rows: 2,
        q_heads: 2,
        kv_heads: 1,
        block_tokens: 2051,
        ..config(3)
    };
    let (q, k, v) = fixture(c, 4103);
    let base = i32::MAX as u64 + 1000;
    let r = Request {
        config: c,
        query: &q,
        key: &k,
        value: &v,
        query_start: base + 4101,
        kv_start: base,
    };
    let whole = partial(r).unwrap();
    let mut split = partial(r.token_range(0, 2051).unwrap()).unwrap();
    split
        .merge(&partial(r.token_range(2051, 4103).unwrap()).unwrap())
        .unwrap();
    close(&whole.output(), &oracle(r));
    close(&split.output(), &oracle(r));
    assert_eq!(
        whole.storage_bytes(),
        c.rows * c.q_heads * (c.head_dim + 2) * 4
    );
    assert_eq!(
        whole.storage_bytes(),
        partial(r.token_range(0, 1).unwrap())
            .unwrap()
            .storage_bytes()
    );
}
#[test]
fn cpu_empty_all_masked_and_both_merge_identities() {
    let c = config(5);
    let (q, k, v) = fixture(c, 13);
    let r = Request {
        config: c,
        query: &q,
        key: &k,
        value: &v,
        query_start: 20,
        kv_start: 10,
    };
    let p = partial(r).unwrap();
    let empty = partial(r.token_range(0, 0).unwrap()).unwrap();
    assert_eq!(empty.output(), vec![0.0; q.len()]);
    let mut left = empty.clone();
    left.merge(&p).unwrap();
    assert_eq!(left.state(), p.state());
    let mut right = p.clone();
    right.merge(&empty).unwrap();
    assert_eq!(right.state(), p.state());
    let masked = partial(Request {
        query_start: 0,
        ..r
    })
    .unwrap();
    assert!(masked.output().iter().all(|v| *v == 0.0));
    for state in masked.state().chunks(c.head_dim + 2) {
        assert_eq!(state[0], f32::NEG_INFINITY);
        assert_eq!(state[1], 0.0);
    }
    let old = partial(Request {
        config: Config {
            sliding_window: Some(1),
            ..c
        },
        query_start: 1000,
        ..r
    })
    .unwrap();
    assert!(old.output().iter().all(|v| *v == 0.0));
}
#[test]
fn cpu_extreme_finite_logits_stable_merge_not_average() {
    let c = Config {
        rows: 1,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        block_tokens: 1,
        softmax_scale: 1.0,
        sliding_window: None,
    };
    let q = [1.0];
    let k = [-3e38, 3e38, 3e38];
    let v = [100.0, 2.0, 4.0];
    let r = Request {
        config: c,
        query: &q,
        key: &k,
        value: &v,
        query_start: 2,
        kv_start: 0,
    };
    let mut p = partial(r.token_range(0, 1).unwrap()).unwrap();
    p.merge(&partial(r.token_range(1, 3).unwrap()).unwrap())
        .unwrap();
    assert_eq!(p.output(), vec![3.0]);
    assert_eq!(partial(r).unwrap().output(), vec![3.0]);
}
#[test]
fn cpu_nonfinite_visible_inputs_rejected_but_masks_are_not_errors() {
    let c = Config {
        rows: 1,
        q_heads: 1,
        kv_heads: 1,
        ..config(1)
    };
    let q = [1.0];
    let good = [1.0];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let bad = [bad];
        let r = Request {
            config: c,
            query: &q,
            key: &bad,
            value: &good,
            query_start: 0,
            kv_start: 0,
        };
        assert!(partial(r).is_err());
        assert!(
            partial(Request {
                key: &good,
                value: &bad,
                ..r
            })
            .is_err()
        );
        assert!(
            partial(Request {
                query: &bad,
                key: &good,
                ..r
            })
            .is_err()
        );
        assert_eq!(
            partial(Request { kv_start: 1, ..r }).unwrap().output(),
            vec![0.0]
        );
    }
    assert!(
        partial(Request {
            config: c,
            query: &[f32::MAX],
            key: &[2.0],
            value: &good,
            query_start: 0,
            kv_start: 0
        })
        .is_err()
    );
}
#[test]
fn cpu_validation_overflow_and_merge_mismatch() {
    let c = config(3);
    for bad in [
        Config { q_heads: 3, ..c },
        Config { head_dim: 0, ..c },
        Config {
            rows: usize::MAX,
            ..c
        },
        Config {
            softmax_scale: f32::NAN,
            ..c
        },
        Config {
            block_tokens: 0,
            ..c
        },
        Config {
            sliding_window: Some(0),
            ..c
        },
    ] {
        assert!(bad.validate().is_err());
    }
    let (q, k, v) = fixture(c, 2);
    let r = Request {
        config: c,
        query: &q,
        key: &k,
        value: &v,
        query_start: 0,
        kv_start: 0,
    };
    assert!(partial(Request { key: &k[..1], ..r }).is_err());
    assert!(
        partial(Request {
            query_start: u64::MAX,
            ..r
        })
        .is_err()
    );
    assert!(
        partial(Request {
            kv_start: u64::MAX,
            ..r
        })
        .is_err()
    );
    assert!(r.token_range(1, 3).is_err());
    let mut p = partial(r).unwrap();
    let before = p.state().to_vec();
    assert!(p.merge(&Partial::empty(c, 1).unwrap()).is_err());
    assert_eq!(p.state(), before);
}
#[test]
fn owned_protection_binds_addresses_lengths_and_rejects_overlap() {
    let k = vec![1.0, 2.0];
    let v = vec![3.0, 4.0];
    let p = OwnedKv::new(&k, &v);
    p.validate(&k, &v).unwrap();
    assert!(p.validate(&k.clone(), &v).is_err());
    assert!(p.validate(&k[..1], &v[..1]).is_err());
    assert!(OwnedKv::new(&k, &k).validate(&k, &k).is_err());
    OwnedKv::new(&Vec::new(), &Vec::new())
        .validate(&[], &[])
        .unwrap();
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use ferrule_backend::cuda::operator_owner::CudaOperators;
    use ferrule_backend::cuda::{
        CudaStreamingF32GqaRequest as DeviceRequest, CudaStreamingF32GqaWorkspace as Workspace,
    };
    use ferrule_common::{Error, Result};

    #[test]
    #[ignore = "requires an explicitly selected idle CUDA device"]
    fn real_cuda_fixed_split_matches_oracle_without_device_growth() {
        let ops = CudaOperators::new_on_device(0).expect("CUDA required");
        for (dim, tokens, block) in [(5, 37, 7), (257, 19, 7), (3, 4103, 2051)] {
            let c = Config {
                rows: 2,
                block_tokens: block,
                ..config(dim)
            };
            let (q, k, v) = fixture(c, tokens);
            let base = i32::MAX as u64 + 100;
            let r = Request {
                config: c,
                query: &q,
                key: &k,
                value: &v,
                query_start: base + tokens as u64 - 2,
                kv_start: base,
            };
            let protection = OwnedKv::new(&k, &v);
            let budget = Workspace::required_bytes(c).unwrap();
            assert!(ops.streaming_f32_gqa_workspace(c, budget - 1).is_err());
            let mut w = ops.streaming_f32_gqa_workspace(c, budget).unwrap();
            let allocations = ops.allocator_metrics().allocation_requests;
            for split in [0, 1, tokens / 2, tokens] {
                let result = ops
                    .streaming_f32_gqa_fixed_split(r, split, &protection, &mut w)
                    .unwrap();
                close(&result.output, &oracle(r));
                assert_eq!(result.stats.gpu_blocks, (tokens - split).div_ceil(block));
                assert_eq!(
                    result.stats.kv_h2d_bytes,
                    (tokens - split) * c.kv_heads * c.head_dim * 8
                );
                assert_eq!(ops.allocator_metrics().allocation_requests, allocations);
                assert_eq!(w.allocated_bytes(), budget);
            }
        }
    }
    #[test]
    #[ignore = "requires an explicitly selected idle CUDA device"]
    fn real_cuda_masks_empty_history_extreme_logits_and_device_output() {
        let ops = CudaOperators::new_on_device(0).expect("CUDA required");
        for window in [None, Some(1), Some(9)] {
            let c = Config {
                sliding_window: window,
                ..config(5)
            };
            let (q, k, v) = fixture(c, 23);
            let query = ops.upload_f32_buffer(&q).unwrap();
            let mut output = ops.zero_f32_buffer(q.len()).unwrap();
            let protection = OwnedKv::new(&k, &v);
            let mut w = ops
                .streaming_f32_gqa_workspace(c, Workspace::required_bytes(c).unwrap())
                .unwrap();
            for pos in [0, 20, 1000] {
                let r = Request {
                    config: c,
                    query: &q,
                    key: &k,
                    value: &v,
                    query_start: pos,
                    kv_start: 10,
                };
                ops.streaming_f32_gqa_fixed_split_into(
                    DeviceRequest {
                        config: c,
                        query: &query,
                        key: &k,
                        value: &v,
                        query_start: pos,
                        kv_start: 10,
                    },
                    5,
                    &protection,
                    &mut output,
                    &mut w,
                )
                .unwrap();
                close(&ops.download_f32_buffer(&output).unwrap(), &oracle(r));
            }
            let empty = Vec::new();
            let r = Request {
                config: c,
                query: &q,
                key: &empty,
                value: &empty,
                query_start: 0,
                kv_start: 0,
            };
            assert_eq!(
                ops.streaming_f32_gqa_fixed_split(r, 0, &OwnedKv::new(&empty, &empty), &mut w)
                    .unwrap()
                    .output,
                vec![0.0; q.len()]
            );
        }
        let c = Config {
            rows: 1,
            q_heads: 1,
            kv_heads: 1,
            block_tokens: 1,
            softmax_scale: 1.0,
            ..config(1)
        };
        let k = vec![-3e38, 3e38, 3e38];
        let v = vec![100.0, 2.0, 4.0];
        let mut w = ops
            .streaming_f32_gqa_workspace(c, Workspace::required_bytes(c).unwrap())
            .unwrap();
        let r = Request {
            config: c,
            query: &[1.0],
            key: &k,
            value: &v,
            query_start: 2,
            kv_start: 0,
        };
        assert_eq!(
            ops.streaming_f32_gqa_fixed_split(r, 1, &OwnedKv::new(&k, &v), &mut w)
                .unwrap()
                .output,
            vec![3.0]
        );
    }
    struct Deny;
    impl Protection for Deny {
        fn validate(&self, _: &[f32], _: &[f32]) -> Result<()> {
            Err(Error::Internal {
                message: "lease rejected".into(),
            })
        }
    }
    #[test]
    #[ignore = "requires an explicitly selected idle CUDA device"]
    fn real_cuda_preflight_rejection_not_poison_numeric_errors_poison() {
        let ops = CudaOperators::new_on_device(0).expect("CUDA required");
        let foreign = CudaOperators::new_on_device(0).expect("second owner required");
        let c = config(3);
        let (q, k, v) = fixture(c, 7);
        let r = Request {
            config: c,
            query: &q,
            key: &k,
            value: &v,
            query_start: 10,
            kv_start: 0,
        };
        let p = OwnedKv::new(&k, &v);
        let mut w = ops
            .streaming_f32_gqa_workspace(c, Workspace::required_bytes(c).unwrap())
            .unwrap();
        let before = ops.counters();
        assert!(
            ops.streaming_f32_gqa_fixed_split(r, 0, &Deny, &mut w)
                .is_err()
        );
        assert!(
            foreign
                .streaming_f32_gqa_fixed_split(r, 0, &p, &mut w)
                .is_err()
        );
        assert!(ops.streaming_f32_gqa_fixed_split(r, 8, &p, &mut w).is_err());
        assert!(!w.is_poisoned());
        assert_eq!(before, ops.counters());
        let qbad = vec![f32::NAN; q.len()];
        for split in [0, 7] {
            let mut w = ops
                .streaming_f32_gqa_workspace(c, Workspace::required_bytes(c).unwrap())
                .unwrap();
            assert!(
                ops.streaming_f32_gqa_fixed_split(Request { query: &qbad, ..r }, split, &p, &mut w)
                    .is_err()
            );
            assert!(w.is_poisoned());
            assert!(ops.streaming_f32_gqa_fixed_split(r, 0, &p, &mut w).is_err());
        }
    }
}
