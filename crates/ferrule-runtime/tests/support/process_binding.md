# Process binding and routed expert semantic boundary

## Application composition

Process wire versions, discriminants, recipe tags and the numeric profile wire
names/defaults are unchanged. No global registry or dynamic plugin was added.

`DecoderEndpoint::initialize(boot, child_launch, bindings)` now requires an
explicit `DecoderBindings` pair of metadata binder functions. The production
CLI hidden worker installs
`ferrule_composition::BUILTIN_PROCESS_BINDINGS` through
`initialize_builtin_process_endpoint`.
That composition adapter, not parallel or runtime, selects Qwen3 recipes and checks the strict Qwen3.5
0.8B dense / 35B-A3B numeric FP8 profiles. It rejects CPU execution without the
`cpu-reference` feature; synthetic execution is not installed in default builds.
The test-only process executable explicitly installs its synthetic binder.

Other API migrations:

- `DecoderBoot::load_resources()` was removed. Application code invokes its
  configured binder; `DecoderBoot::validate_resources(&resources)` checks the
  rebound geometry/layout/EP assignment without selecting a model family.
- `DecoderChild::initialize_builtin(boot, launch, bind)` and
  `NumericExpertChild::initialize(boot, bind)` require explicit metadata binders.
  The existing lower-level `DecoderChild::initialize(boot, factory)` remains.
- `DecoderChild::cpu_factory(boot, resources)` consumes already-bound metadata.
  `hybrid_factory(boot, resources)` and numeric `initialize_bound(boot, resources)`
  remain available. They do not infer family acceptance from wire tags.

Callbacks run after protocol/envelope validation. Numeric startup checks the
CUDA feature before calling the binder, then checks exact rebound catalog/source
identity and geometry before warmup/device allocation. Decoder startup validates
rebound depth and hybrid geometry before payload preparation or nested EP spawn.
Binder functions must only bind metadata, not create physical execution resources.

Parallel retains recipe tag/schema consistency checks for wire compatibility.
`ModelFamily::Unknown` execution labels and the serialized `Qwen*` recipe tags
are not concrete family imports. Real Qwen imports now live in the composition adapter and other existing
model-factory family implementations, not parallel.
`tests/parallel_architecture.rs` uses only `rg` to guard concrete family imports,
concrete family types/labels and model-factory back-edges under `src/parallel`;
it is a text guard, not an AST dependency proof.

## Local and remote expert lowering

`ValidatedExpertCatalog::for_decoder_range` supports dense/mixed segments and
validates routed shapes/bias/roles across the complete selected range before any
payload reads. EP groups use `new`, which additionally requires a routed layer.
`ExpertSemantics` owns only cloned bindings and the global id/activation policy.

Standard local FFNs retain these validated semantic records. Their lazy provider
calls the same `ExpertSemantics::prepare` used by EP `prepare_selected`, only for
requested experts. CUDA metadata hits remain payload-free; materializer/host-cache
freshness and cache policy are unchanged. Shared FFN activation limits stay with
the shared descriptor, and no second full expert materialization is performed.

## CPU verification

```sh
cargo test -p ferrule-runtime --test parallel_architecture
cargo test -p ferrule-runtime --test process_decoder
cargo test -p ferrule-runtime --features cpu-reference --test process_decoder
cargo test -p ferrule-runtime --features cpu-reference --lib semantic_tests
cargo test -p ferrule-runtime --features cpu-reference --lib application_
cargo test -p ferrule-runtime --features cpu-reference --lib injected_expert_binding
cargo test -p ferrule-runtime --features cpu-reference --lib parallel::expert
cargo test -p ferrule-model --features cpu-reference --lib shared_gate_tests
cargo test -p ferrule-cli --test rank_worker_process
```

CUDA validation is compile/clippy only with CUDA 12.4, `FERRULE_CUDA_ARCH=sm_86`
and `CUDA_VISIBLE_DEVICES=''`; it is not GPU execution evidence.
