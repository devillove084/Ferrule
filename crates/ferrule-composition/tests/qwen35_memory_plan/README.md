# Static hybrid memory contracts

`qwen35_memory_plan` binds sparse checkpoint headers only. It does not read
weight payloads, create CUDA contexts, probe devices, or execute kernels.
The same exact state/KV/weight/workspace and physical-card ledger assertions run
in default and CUDA builds. CUDA builds additionally compare the existing model
CUDA estimate with the pure `HybridMemoryPlan`.

The slice covers F32 model boundaries with dense weights or explicit numeric
FP8 operand profiles. Native BF16 image admission remains on its existing CUDA
path. A successful plan is neither a reservation nor proof of GPU executability:
physical owner memory/capability checks and allocation still run during prepare.

Related model tests:

- `decoder::hybrid_memory_plan::tests`: routed scratch, numeric ABI selection,
  exact workspace/allocation grants, recurrent ABI geometry and overflow.
- `segment_inventory`: independent host/device ledgers, canonical/tied views,
  segment state/KV projections, startup temporaries and overflow.
- `hybrid_segment`: endpoint/range-aware metadata planning without CUDA.

Run the same contracts without CUDA:

```sh
CUDA_VISIBLE_DEVICES='' cargo test -p ferrule-composition --no-default-features --test qwen35_memory_plan
CUDA_VISIBLE_DEVICES='' cargo test -p ferrule-model --no-default-features --lib decoder::hybrid_memory_plan::tests
CUDA_VISIBLE_DEVICES='' cargo test -p ferrule-model --no-default-features --lib segment_inventory
```

With CUDA 12.4 `nvcc` on PATH and its libraries on the linker/loader search paths:

```sh
CUDA_VISIBLE_DEVICES='' FERRULE_CUDA_ARCH=sm_80 cargo test -p ferrule-composition --features cuda --test qwen35_memory_plan
CUDA_VISIBLE_DEVICES='' FERRULE_CUDA_ARCH=sm_80 cargo test -p ferrule-model --features cuda --lib decoder::hybrid_memory_plan::tests
CUDA_VISIBLE_DEVICES='' FERRULE_CUDA_ARCH=sm_80 cargo test -p ferrule-model --features cuda --lib segment_inventory
```

These are metadata tests, not GPU numerical validation or serving readiness.
