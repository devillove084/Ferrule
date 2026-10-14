#ifndef FERRULE_CORE_RECURRENT_BF16_ABI_H_
#define FERRULE_CORE_RECURRENT_BF16_ABI_H_
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* HF eager boundaries. All BF16 pointers address uint16_t storage words.
 * Packed rows and channel-major raw history; no implicit initialization. */
typedef struct FerruleCoreBf16ConvArgs {
  uint32_t rows, channels, kernel_size, reserved;
  uint64_t input_bf16, weight_bf16, bias_bf16, history_bf16, output_bf16, stream;
} FerruleCoreBf16ConvArgs;

/* QKV=[Q[Hk,Dk], K[Hk,Dk], V[Hv,Dv]], a/b=[rows,Hv].
 * Native HF-loaded A_log/dt_bias are BF16; gate computation promotes to F32.
 * State is genuinely F32 [Hv,Dk,Dv], never a BF16 allocation/view. */
typedef struct FerruleCoreBf16DeltaArgs {
  uint32_t rows, key_heads, value_heads, key_dim, value_dim, reserved;
  uint64_t qkv_bf16, a_bf16, b_bf16, a_log_bf16, dt_bias_bf16;
  uint64_t state_f32, output_bf16, stream;
} FerruleCoreBf16DeltaArgs;

typedef struct FerruleCoreBf16GatedNormArgs {
  uint32_t rows, width;
  float epsilon;
  uint32_t reserved;
  uint64_t input_bf16, weight_bf16, gate_bf16, output_bf16, stream;
} FerruleCoreBf16GatedNormArgs;

int32_t ferrule_core_bf16_conv_launch(const FerruleCoreBf16ConvArgs *args);
int32_t ferrule_core_bf16_delta_launch(const FerruleCoreBf16DeltaArgs *args);
int32_t ferrule_core_bf16_gated_norm_launch(const FerruleCoreBf16GatedNormArgs *args);
#ifdef __cplusplus
}
static_assert(sizeof(FerruleCoreBf16ConvArgs) == 64);
static_assert(offsetof(FerruleCoreBf16ConvArgs, input_bf16) == 16);
static_assert(offsetof(FerruleCoreBf16ConvArgs, stream) == 56);
static_assert(sizeof(FerruleCoreBf16DeltaArgs) == 88);
static_assert(offsetof(FerruleCoreBf16DeltaArgs, qkv_bf16) == 24);
static_assert(offsetof(FerruleCoreBf16DeltaArgs, state_f32) == 64);
static_assert(offsetof(FerruleCoreBf16DeltaArgs, stream) == 80);
static_assert(sizeof(FerruleCoreBf16GatedNormArgs) == 56);
static_assert(offsetof(FerruleCoreBf16GatedNormArgs, input_bf16) == 16);
static_assert(offsetof(FerruleCoreBf16GatedNormArgs, stream) == 48);
#endif
#endif
