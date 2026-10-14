# Pipeline physical KV capacity contract

`pipeline_capacity` is a CPU/metadata-only integration target. It requires no
checkpoint, CUDA context or GPU, including when compiled with `--features cuda`.

| Adapter | Custody | Physical slots | Slot-count limit |
| --- | --- | --- | --- |
| Standard CPU | `InPlace` | L | `u32::MAX` |
| Hybrid CPU (including reference TP) | `Shadowed` | 2L | `u32::MAX` |
| Standard/hybrid CUDA | `Shadowed` | 2L | `i32::MAX + 1` |

Custody and index representation are independent axes of `PhysicalKvLayout`.
`standard`/`hybrid` reuse `ModelExecutionBackend` to select existing adapter
contracts, not to initialize a device. Physical slots never widen logical
`PipelineConfig::max_pages` admission. Quarantine is charged inside the same
physical pool; this metadata is not a page or transaction authority.

The old `physical_pages(bool)` and composition planner's `cuda: bool` are gone.
All callers supply an explicit layout. Hybrid wire encoding now takes the actual
`DecoderDevice`, because CPU and CUDA have different index limits despite both
using 2L. Wire fields/version are unchanged; missing/mismatched physical capacity
still fails closed. Frame/budget metadata and the new layout are available
without the CUDA feature.

## Coverage

- CPU standard L versus CPU hybrid 2L; independent custody/index combinations.
- CUDA shapes without CUDA initialization (also compiled in the no-CUDA lane).
- Exact unsigned/signed limits, zero capacity, multiplication and logical extent overflow.
- CPU hybrid above the CUDA signed bound, including description/wire validation.
- Standard/hybrid CPU/CUDA wire round trips; wrong/missing physical slot counts.
- Real CPU page pools with mismatched L/2L descriptions, including startup cleanup.
- `compile.md`: old bool API must not compile; metadata example must compile/run.

## Validation

```sh
cargo check -p ferrule-runtime --all-targets
cargo check -p ferrule-runtime --all-targets --features cpu-reference
cargo test -p ferrule-runtime --test pipeline_capacity --test pipeline_program --test hybrid_pipeline_state
cargo test -p ferrule-composition --test hybrid_pipeline
cargo test -p ferrule-runtime --features cpu-reference --test pipeline_capacity --test pipeline_program --test hybrid_pipeline_state
cargo test -p ferrule-composition --features cpu-reference --test hybrid_pipeline
cargo test -p ferrule-runtime --lib engine::resident_composition::
cargo test -p ferrule-runtime --lib parallel::pipeline --no-default-features
cargo test -p ferrule-runtime --doc PhysicalKvLayout --no-default-features
cargo test -p ferrule-runtime --test process_decoder capacity_wire_roundtrip_and_old_boot_rejection_precede_factory
```

On this validation host, use the existing CUDA 12.4 environment (NVCC 12.4.99,
explicit `sm_86`), overriding its GPU selection. Rustdoc needs the native search
paths separately from `RUSTFLAGS`:

```sh
. target/validation/qwen35-35b-current/toolchain/env-cuda124.sh
export CUDA_VISIBLE_DEVICES=
export RUSTDOCFLAGS='-L native=/home/admin/luhuanbing.lhb/Ferrule/target/validation/qwen35-35b-current/toolchain/link -L native=/home/admin/wxw403883/anaconda3/envs/tao/lib'
cargo check -p ferrule-runtime --all-targets --features cuda
cargo test -p ferrule-runtime --features cuda --test pipeline_capacity
cargo test -p ferrule-runtime --features cuda --doc PhysicalKvLayout
```

Do not run GPU-provider tests in this lane. Each command was bounded by 900000 ms.
