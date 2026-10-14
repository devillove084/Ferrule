# Automation maintenance

## Current CLI entry points

Use `dsv4-serve` for public HTTP serving and `dsv4-runtime-driver-bench` /
`bench-interactive` for interactive measurements. Retired generation/smoke
recipes are not aliases for these commands or their output schemas. `chat` and
`dsv4-chat` select `--backend cuda`; additional CLI options follow the model,
without the removed positional quantization argument or `-q` option.

## GPU selection membership without executing tests

`just architecture` runs the source-boundary guard and
`scripts/tests/test_automation_contracts.py`. GPU CI lists compiled tests with
devices hidden, then runs `check_nextest_membership.py` before the selected lane.
A nonempty lane is insufficient: its required branch must have compiled members
and Nextest must select them.

The checker accepts `--nextest-json` or Cargo CUDA
`--no-run --message-format=json` output via `--cargo-artifacts`. The latter
invokes only `--list --format terse` on the recorded library binary. Both modes
require `--profile gpu-smoke` or `--profile gpu-nightly`; neither builds nor runs
a test body. Regenerate lists after source changes: an archived binary proves
only its own membership, not that current source compiled.

## CPU script tests

The runtime `scripts/test_vllm_qwen35.py` keeps its public name; its unit tests
are `scripts/tests/test_vllm_validation.py`. Whole-directory discovery collects
without that module-name collision. Oracle modules do not initialize HF during
collection; real CPU numeric tests initialize their pinned dependencies in
`setUpClass`. Pure CUDA-exporter contract tests use mocks and need no NumPy/HF.

Ordinary contract tests (no oracle numerical classes), in one command:

```sh
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void \
  /opt/conda310/bin/python -B -m unittest discover -s scripts/tests -p 'test_[!q]*.py'
```

The complete suite needs two environments: native BF16 uses Transformers 5.5.3,
while the legacy FP8 numerical adapters use 5.2.0. Both require the existing
Torch 2.13.0+cu129. Run these two disjoint lanes; neither weakens pins nor skips
selected tests. The first includes ordinary contracts, mocked exporter contracts
and real native-BF16 tests; the second runs the two real legacy numeric classes.

```sh
env -u PYTHONPATH CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  /opt/conda310/bin/python -B -m unittest discover -s scripts/tests -p 'test_*.py' \
  -k 'test_[!q]*' -k '*.ContractTests.*' -k '*.NativeBf16Reference.*'

CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=target/validation/w8a16-full40-independent-20261002/python-overlay \
  target/vendor/vllm028-py312/bin/python -B -m unittest discover -s scripts/tests \
  -p 'test_qwen35_fp8*reference.py' -k '*.NumericTests.*' -k '*.PrecisionAdapterTests.*'
```

The overlay is read-only. Unfiltered discovery in either environment deliberately
reports incompatible oracle setup as an error, not a skip or a fabricated pass.
Keep lane selectors aligned when adding oracle classes. These are CPU tests of
mocks and tiny fixtures, not GPU execution or production reference regeneration.

## Prepared attention jobs

Attention preparers freeze jobs through `scripts/run_pinned_jobs.py`, using
`test_vllm_qwen35.run_process` and `_owned_subreaper.py` for authenticated process
custody. Source, executable/interpreter, workflow and resolved-library pins are
checked before telemetry/launch and again before each job. Each job has a frozen
explicit environment and bounded timeout. BF16 arithmetic overrides and the
eight-card whole-host quiescence requirement remain part of this workflow.

CPU-only pin check, substituting literal owner-selected input paths:

```text
python -B scripts/run_pinned_jobs.py --jobs PREPARED/jobs.json --libraries LIBRARY_RECEIPT.json
```

Without `--sole-gpu-owner-granted`, this mode neither queries GPUs nor starts
children. Execution additionally requires a new `--output` directory. Timeout,
nonzero status or unknown custody stops the sequence even if cleanup succeeds.
Release requires verified descendant drain and three idle telemetry observations.
After workflow changes, prepare fresh manifests; never update old pins or
historical receipts in place. Resolved-library receipts and oracle inputs remain
read-only dependencies.

The [network-linear validator](validate_network_localhost.md) has concurrent
fault-injection scenarios and explicitly documented custody limits; it is not
silently replaced by the sequential pinned-job runner.
