//! Mathematical contract only: not a DMA/decoder integration fixture.
use ferrule_backend::cpu::{
    StreamingF32GqaConfig, StreamingF32GqaRequest, streaming_f32_gqa_partial,
};

#[test]
fn append_tail_needs_unnormalized_partials_not_normalized_prefix_rows() {
    let request = StreamingF32GqaRequest {
        config: StreamingF32GqaConfig {
            rows: 1,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 1,
            block_tokens: 1,
            softmax_scale: 1.0,
            sliding_window: None,
        },
        query: &[1.0],
        key: &[0.0, 2.0],
        value: &[0.0, 10.0],
        query_start: 1,
        kv_start: 0,
    };
    let prefix = streaming_f32_gqa_partial(request.token_range(0, 1).unwrap()).unwrap();
    let tail = streaming_f32_gqa_partial(request.token_range(1, 2).unwrap()).unwrap();
    let expected = streaming_f32_gqa_partial(request).unwrap().output()[0];
    let normalized_average = (prefix.output()[0] + tail.output()[0]) / 2.0;
    assert!((normalized_average - expected).abs() > 3.0);
    let mut composed = prefix;
    composed.merge(&tail).unwrap();
    assert!((composed.output()[0] - expected).abs() < 1e-6);
    // A current-token-only window must discard the old prefix at composition.
    let windowed = StreamingF32GqaRequest {
        config: StreamingF32GqaConfig {
            sliding_window: Some(1),
            ..request.config
        },
        ..request
    };
    let mut prefix = streaming_f32_gqa_partial(windowed.token_range(0, 1).unwrap()).unwrap();
    prefix
        .merge(&streaming_f32_gqa_partial(windowed.token_range(1, 2).unwrap()).unwrap())
        .unwrap();
    assert_eq!(prefix.output(), vec![10.0]);
}
