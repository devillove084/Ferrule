W8A16 is an explicit numeric-storage arithmetic choice, not a change to model
execution boundaries or the existing F32 production profile. Configuration is
available even in builds without CUDA:

```no_run
use ferrule_common::Result;
use ferrule_model::decoder::GenericDecoderOptions;
use ferrule_model::transformer::{ExpertCacheLimits, NumericFp8Precision};

fn select_w8a16(options: GenericDecoderOptions) -> Result<GenericDecoderOptions> {
    options.with_hybrid_cuda_numeric_fp8_precision(
        ExpertCacheLimits { max_experts: 1, max_bytes: 1 << 20 },
        0, // No global numeric scratch; operation buffers still consume budget.
        NumericFp8Precision::W8A16Bf16F32Accumulate,
    )
}
```
