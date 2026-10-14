#pragma once
#include "core/abi.h"
struct FerruleBf16AttentionGroup { uint32_t row_start, rows, sequence, tokens; };
struct FerruleBf16AttentionPlan {
  uint32_t heads, kv_heads, dim, max_rows, max_tokens, reserved;
  uint64_t offsets[7], bytes;
};
static_assert(sizeof(FerruleBf16AttentionPlan)==88);
