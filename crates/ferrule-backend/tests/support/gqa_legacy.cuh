// Frozen pre-cooperative GPU oracle: do not share dot/softmax implementation.
template <typename T = uint16_t>
__global__ void gqa_legacy_oracle_kernel(FerruleCoreTransformerArgs args) {
  const uint64_t index =
      static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t values =
      static_cast<uint64_t>(args.rows) * args.q_heads * args.head_dim;
  if (index >= values) {
    return;
  }
  const uint32_t dimension = static_cast<uint32_t>(index % args.head_dim);
  const uint64_t head_row = index / args.head_dim;
  const uint32_t q_head = static_cast<uint32_t>(head_row % args.q_heads);
  const uint32_t row = static_cast<uint32_t>(head_row / args.q_heads);
  const uint32_t queries_per_kv_head = args.q_heads / args.kv_heads;
  const uint32_t kv_head = q_head / queries_per_kv_head;
  uint32_t sequence;
  uint32_t position;
  uint32_t kv_len;
  if (!transformer_row_metadata(args, row, &sequence, &position, &kv_len)) {
    return;
  }
  const uint32_t effective_kv_len =
      (args.kind == FERRULE_CORE_TRANSFORMER_PAGED_BF16_APPEND_CAUSAL_GQA ||
       args.kind == FERRULE_CORE_TRANSFORMER_PAGED_F32_APPEND_CAUSAL_GQA)
          ? max(kv_len, position + 1)
          : kv_len;
  const uint32_t visible = min(effective_kv_len, position + 1);
  const uint64_t query_head =
      static_cast<uint64_t>(row) * args.query_row_stride_bytes +
      static_cast<uint64_t>(q_head) * args.query_head_stride_bytes;

  float maximum = -CUDART_INF_F;
  float denominator = 0.0f;
  float accumulator = 0.0f;
  for (uint32_t token = 0; token < visible; ++token) {
    float score = 0.0f;
    for (uint32_t dot_dimension = 0; dot_dimension < args.head_dim;
         ++dot_dimension) {
      uint64_t key_offset;
      if (!transformer_cache_offset(args, sequence, token, kv_head,
                                    dot_dimension, false, &key_offset, sizeof(T))) {
        return;
      }
      const float query = const_pointer<float>(
          args.query_f32 + query_head +
          static_cast<uint64_t>(dot_dimension) * sizeof(float))[0];
      const float key = transformer_value(
          const_pointer<T>(args.key_cache_bf16 + key_offset)[0]);
      score += query * key;
    }
    score *= args.softmax_scale;
    uint64_t value_offset;
    if (!transformer_cache_offset(args, sequence, token, kv_head, dimension,
                                  true, &value_offset, sizeof(T))) {
      return;
    }
    const float value = transformer_value(
        const_pointer<T>(args.value_cache_bf16 + value_offset)[0]);
    if (denominator == 0.0f || score > maximum) {
      const float rescale = denominator == 0.0f ? 0.0f : expf(maximum - score);
      accumulator = accumulator * rescale + value;
      denominator = denominator * rescale + 1.0f;
      maximum = score;
    } else {
      const float weight = score == maximum ? 1.0f : expf(score - maximum);
      accumulator += weight * value;
      denominator += weight;
    }
  }
  const uint64_t output_offset =
      static_cast<uint64_t>(row) * args.output_row_stride_bytes +
      static_cast<uint64_t>(q_head) * args.output_head_stride_bytes +
      static_cast<uint64_t>(dimension) * sizeof(float);
  pointer<float>(args.output_f32 + output_offset)[0] =
      denominator > 0.0f ? accumulator / denominator : 0.0f;
}

