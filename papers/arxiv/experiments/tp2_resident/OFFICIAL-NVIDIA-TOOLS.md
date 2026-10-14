# Official NVIDIA tool closeout (CPU-readable entry points)

This note records the tool path and commands already verified in this project.
It is documentation, not a launch authorization. The closeout after the one
full-model run used CPU-only reads/tests and performed no new GPU collection.

## Nsight Systems installation

Official standalone NVIDIA package:

- Source: `https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64/NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb`
- Package SHA256: `d2484ad0faf6831b11fa0bf73c54232d9ea8beafb50414019e6ba299c4ed5718`
- CLI: `target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys`
- CLI version: `2025.3.1.90-253135822126v0`
- CLI SHA256: `525f35a10de49b396e91a3831ae40e98a8631d619b7c8bbb4c54469b8c08a72e`
- Matching importer: `target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/host-linux-x64/QdstrmImporter`
- Importer SHA256: `23e4185b7b20d00c27dc2e521c1048a75abe04f04dbbd88e411b6dfba2cc8053`
- Bundled CUPTI: `target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/libcupti.so.12.9`
- CUPTI SHA256: `19f0f06e0c5579113840169d5f6d6d4222d4a4fc2d66403cbe3cad2d91db1ace`

Source/package/dependency details: [installation](runs/nsys-official-micro-20261009-002/installation.json),
[dependency hashes](runs/nsys-official-micro-20261009-002/dependency-sha256.json),
and [tool inventory](runs/nsys-official-micro-20261009-002/tool-inventory-sha256.json).
The NVIDIA-channel [conda dry-run](runs/nsys-official-micro-20261009-001/research/conda-dry-run.json)
returned `PackagesNotFoundError` for `nsight-systems=2025.3.1` (not a claim that
the channel has no other Nsight-named packages). `/opt/conda310` remained conda
23.1.0 / Python 3.10.9, with no transaction. The official standalone `.deb` was
extracted under `target/vendor/`, without system installation or CUDA upgrade.
Provenance is NVIDIA HTTPS plus repository-index SHA256; the package signature
was not independently verified. This CPU closeout did not reinstall or execute
the vendor tools.

## Historical verified micro command — do not rerun

Existing NVIDIA CUDA 12.4 sample
`/home/admin/wxw403883/anaconda3/envs/tao/demo_suite/vectorAdd`, not a custom timing
harness. This consumed command is evidence, not authorization; any future GPU
collection requires a fresh explicit grant and run directory:

```sh
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_resident/nsys_micro.py run \
  papers/arxiv/experiments/tp2_resident/runs/nsys-official-micro-20261009-002 \
  --sole-gpu-owner
```

[Micro acceptance](runs/nsys-official-micro-20261009-002/acceptance.json) records
two physical GPU UUIDs, real kernels, H2D/D2H memcpy, correlated runtime APIs,
and zero fatal diagnostics.

## Existing model profile entry point

The model profile uses the existing `paired.py` collector-first supervised
serving lifecycle, not a second server harness. The immutable qualified serving
binary is the staged `4d40c34738d36a1c78f62c0e400dd732da40e34c316ce4bd3c356dc4d29b3173`.
The consumed run and command receipts are under
[the consumed run](runs/tp2-official-full-profile-20261010-001/execution.json)
and its [launch receipt](runs/tp2-official-full-profile-20261010-001/launch.json).

The 2025.3 compatibility layer uses only flags verified in the bundled
[saved `nsys profile --help`](runs/nsys-official-micro-20261009-001/research/profile-help.txt):

```text
--trace=cuda,nvtx,osrt,cublas
--start-later=true
--sample=none
--cpuctxsw=none
--gpu-metrics-devices=none
--gpuctxsw=false
--trace-fork-before-exec=false
--kill=none
--wait=primary
--stop-on-exit=true
--duration=240
--cuda-flush-interval=0
--force-overwrite=false
--output=<run>/runs/P1/session/nsight/nsys
```

The 2025.3 path deliberately does not use 2026-only `cuda-sw`,
`--cuda-trace-scope`, `--cuda-trace-all-apis`, or `--gpu-video-devices`.
The serving environment retains `CUBLAS_WORKSPACE_CONFIG=:0:0`, TP2 thread
PP1 EP1, context 128, batch tokens 8, two active sequences, and the two ordered
physical UUIDs.

## CPU-only audit entry point

Run from the repository root. The existing entry point delegates to the v2
audit; there is no third audit or new harness. The command that produced the
already-present output has this form (do not rerun to replace it):

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=none \
/home/admin/wxw403883/anaconda3/bin/python3.12 -S -B \
  papers/arxiv/experiments/tp2_resident/profile_trace_audit.py \
  papers/arxiv/experiments/tp2_resident/runs/tp2-official-full-profile-20261010-001 \
  --output-name profile-audit-v2.json
```

The output defaults to `profile-audit-v2.json`. Existing output, the historical
name `profile-audit.json`, and path components are rejected before audit work.
Read [the existing v2 result](runs/tp2-official-full-profile-20261010-001/profile-audit-v2.json)
instead of regenerating it. [Initial CPU closeout](runs/tp2-official-full-profile-20261010-001/cpu-closeout-v2/closeout-manifest.json)
and [recovery completion manifest](runs/tp2-official-full-profile-20261010-001/cpu-closeout-v2/completion-manifest.json)
are separate from the historical execution/hash receipts.

## Audit rule and current result

The original `profile-audit.json` remains unchanged. The revised CPU-only audit
is `profile-audit-v2.json`; it performs both directions:

```text
activity -> unique successful API
successful launch/copy/memset API -> unique activity
```

It also checks full-export PID/correlation mapping, request-window containment,
and same-thread runtime overlap. Request6 is marked
`observed_correlation_closed`, but `full_trace_accepted=false`: a pair of
simultaneously missing API/activity records cannot be discovered from the export.
The generic `libAnalysis` TraceSummaryVerifier warning is retained. No explicit
CUPTI dropped/overflow message was found, so the cause is unknown—not attributed
to capture boundary or buffer overflow.

No prefill/decode boundary, `cublasGemmEx` API range, or CPU collective callsite
was invented. Kernel names such as `bf16_gemmex::split_pair` remain kernel names,
not GemmEx API measurements.

## W8 candidate constraint

The real-operand W8 routed-down candidate failed the frozen exact gate at two
rank1 BF16 words (`16382/16384` versus native `16384/16384`). It remains
default-off. See the sole [model test result](../../../../crates/ferrule-model/tests/mixed_routed_down/RESULT.md).
No model, kernel, expected value, threshold, or default was changed by this
closeout.
