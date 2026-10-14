use super::*;

#[test]
fn budget_includes_size_classes_and_checks_overflow() {
    let c = StreamingF32GqaConfig {
        rows: 1,
        q_heads: 2,
        kv_heads: 1,
        head_dim: 3,
        block_tokens: 7,
        softmax_scale: 1.0,
        sliding_window: None,
    };
    assert_eq!(
        CudaStreamingF32GqaWorkspace::required_bytes(c).unwrap(),
        512
    );
    assert!(
        CudaStreamingF32GqaWorkspace::required_bytes(StreamingF32GqaConfig {
            block_tokens: usize::MAX,
            ..c
        })
        .is_err()
    );
    assert_eq!(granted(257).unwrap(), 2048);
    assert_eq!(granted((1 << 18) + 1).unwrap(), 2 << 20);
}

#[test]
#[ignore = "requires an explicitly selected idle CUDA device"]
fn real_cuda_alias_rejection_and_transfer_poison() {
    let ops = CudaOperators::new_on_device(0).expect("CUDA required");
    let c = StreamingF32GqaConfig {
        rows: 1,
        q_heads: 1,
        kv_heads: 1,
        head_dim: 1,
        block_tokens: 2,
        softmax_scale: 1.0,
        sliding_window: None,
    };
    let key = vec![1.0];
    let value = vec![2.0];
    let protection = StreamingF32GqaOwnedKv::new(&key, &value);
    let mut w = ops.streaming_f32_gqa_workspace(c, 512).unwrap();
    let query = ops.upload_f32_buffer(&[1.0]).unwrap();
    let mut alias = CudaF32Buffer::from_device_buffer(query.buffer.slice(0, 1).unwrap());
    let request = || CudaStreamingF32GqaRequest {
        config: c,
        query: &query,
        key: &key,
        value: &value,
        query_start: 0,
        kv_start: 0,
    };
    assert!(
        ops.streaming_f32_gqa_fixed_split_into(request(), 0, &protection, &mut alias, &mut w)
            .is_err()
    );
    assert!(!w.is_poisoned());
    let mut output = ops.zero_f32_buffer(1).unwrap();
    ops.failpoints.arm_stream_sync();
    assert!(
        ops.streaming_f32_gqa_fixed_split_into(request(), 0, &protection, &mut output, &mut w)
            .is_err()
    );
    assert!(w.is_poisoned());
}
