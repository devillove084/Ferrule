#ifndef FERRULE_CORE_RECURRENT_BF16_CHUNK_ABI_H_
#define FERRULE_CORE_RECURRENT_BF16_CHUNK_ABI_H_
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Opt-in multi-row chunk algorithm. No allocations or implicit state reset.
 * BF16 QKV/gate inputs, F32 [Hv,Dk,Dv] S, BF16 output, F32 workspace.
 * All writes (including the entire supplied workspace) must be disjoint from
 * all other operands. On partial submission, retain workspace AND state until
 * completion is proved; a nonzero return is not rollback evidence. */
typedef struct FerruleCoreBf16ChunkArgs {
  uint32_t rows, key_heads, value_heads, key_dim, value_dim, reserved;
  uint64_t qkv_bf16, a_bf16, b_bf16, a_log_bf16, dt_bias_bf16;
  uint64_t state_f32, output_bf16, workspace_f32, workspace_elements, stream;
} FerruleCoreBf16ChunkArgs;
/* Legacy multi-row entry rejects rows=1. Descriptor reserved remains zero. */
int32_t ferrule_core_bf16_chunk_launch(const FerruleCoreBf16ChunkArgs *args);
/* Explicit prefill arithmetic: rows>=1, full workspace, no recurrent delegation.
 * Same checked descriptor, disjointness and partial-submission lifetime rules. */
int32_t ferrule_core_bf16_chunk_prefill_launch(const FerruleCoreBf16ChunkArgs *args);
#ifdef __cplusplus
}
static_assert(sizeof(FerruleCoreBf16ChunkArgs) == 104);
static_assert(offsetof(FerruleCoreBf16ChunkArgs, qkv_bf16) == 24);
static_assert(offsetof(FerruleCoreBf16ChunkArgs, workspace_f32) == 80);
static_assert(offsetof(FerruleCoreBf16ChunkArgs, stream) == 96);
#endif
#endif
