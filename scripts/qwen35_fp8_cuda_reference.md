# Independent native HF Qwen3.5 CUDA reference

`scripts/qwen35_fp8_cuda_reference.py` exports the independent full-model
Transformers 5.2.0 Qwen3.5 MoE CUDA reference. It imports neither Ferrule nor the
CPU oracle, CUTLASS, production reduction/address helpers or vLLM. It installs
nothing and does not modify vendor code. Native HF source is SHA256-pinned.

This F32-boundary reference is distinct from the native-BF16 model-boundary
oracle in `qwen35_fp8_bf16_reference.py`. That exporter also imports this file's
checkpoint/provenance, placement, RoPE and capture utilities; preserve those
interfaces independently of this CLI.

## Environment and safe discovery

Required existing environment: Torch `2.13.0+cu129` (CUDA build `12.9`),
Transformers `5.2.0`, and installed safetensors, tokenizers and NumPy. Native
Torch GDN/conv fallbacks must be present and accelerated replacements absent.
Other versions fail closed; do not install packages or select an environment
post hoc to reduce numerical errors.

`--help` parses arguments without importing Torch or initializing CUDA:

```sh
/opt/conda310/bin/python -B scripts/qwen35_fp8_cuda_reference.py --help
```

The existing pinned 5.2.0 environment can run the CPU-only contract tests:

```sh
CUDA_VISIBLE_DEVICES='' NVIDIA_VISIBLE_DEVICES=void \
PYTHONPATH=target/validation/w8a16-full40-independent-20261002/python-overlay \
  target/vendor/vllm028-py312/bin/python -B scripts/tests/test_qwen35_fp8_cuda_reference.py
```

The overlay is a read-only dependency, not a script to execute or update.
Static preflight requires a fresh `--output` directory and validates dependencies
and local model metadata. Without `--execute-cuda`, no model loading, CUDA probe,
memory admission or forward is attempted. Its manifest remains incomplete and
its exit code is **2**, not successful reference generation.

## Full-model execution contract

Only an explicitly authorized GPU owner may use `--execute-cuda`, with an
owner-selected local `--model`, fresh `--output`, and assigned visible logical
`--devices`. `--devices cuda:0,cuda:1` distributes contiguous layer groups over
those devices. There is no NCCL, distributed worker, Accelerate or CPU-compute
offload. Placement hooks move tensors, not cache objects or mathematical formulas.

`--case` selects `both` (default), `hello` or `capital`. A completed single-case
export covers only that case; consumers requiring both must validate the manifest
case inventory. Full40 means all 40 layers for each exported case, not arbitrary
prompt/task coverage.

Admission, loading, forward and capture failures propagate to an incomplete
manifest and nonzero worker status. There is no conditional-layer replay,
module-I/O tracing mode, CPU substitution, second environment or success fallback.
The worker deadline is bounded by `--timeout` (default/max 900 seconds). Timeout
kills its process group and marks the manifest incomplete; this exporter does
not claim authenticated descendant custody or whole-host GPU quiescence.

## Arithmetic and model loading

This is **not** `from_pretrained(dtype=bfloat16)` or native HF dynamic-FP8
quantization. A meta text model is bound directly to local checkpoint operands;
vision and MTP modules are not instantiated.

- E4M3FN weights and numeric BF16 multiplicative scales on a
  `[ceil(rows/128), ceil(columns/128)]` grid, not reciprocal or UE8M0 scales.
- Weight decode-times-scale in F32, then BF16 round-to-nearest-even and F32
  representation. Quantized linear inputs also round through BF16.
- F32 `torch.nn.functional.linear` accumulation **and output**, without BF16
  output rounding. An operand tensor tag preserves native expert forwards;
  there is no global `F.linear` monkey patch.
- Nonquantized weights, norm/router/shared gate, residuals, attention, KV and
  GDN convolution/recurrent states remain F32. RoPE frequencies must equal
  fresh native F32 construction; blanket BF16 conversion is not allowed.
- Matmul/cuDNN TF32 are disabled; matmul precision is `highest`, deterministic
  algorithms are required, and cuDNN benchmarking is off. Native eager
  attention/experts and Torch GDN/conv are retained.

A byte-bounded per-device operand LRU (default 256 MiB, maximum 1024 MiB) streams
selected matrices and experts. Embeddings and other non-linear parameters are
resident F32; large LM-head operands bypass the LRU. Temporary tensors, allocator
reservations and snapshots are additional memory. Admission requires 8 GiB plus
the operand cache free on each selected device: a refusal threshold, **not a
measured peak-memory guarantee**. There is no hard GPU/host RSS limit.

## Outputs and provenance

Schema: `ferrule.qwen35-fp8-cuda-reference.v1`. Successful full40 prefill plus one
cached decode for every selected case sets:

```json
{
  "status": "complete",
  "reference_kind": "independent_cuda_full_model",
  "full_reference_complete": true
}
```

Other executions remain incomplete. The manifest records exact dependencies,
script/native-source hashes, checkpoint/config/tokenizer provenance, device
admission/placement, precision contract, diagnostics and artifact hashes.
`source-payloads.json` hashes actually read tensors and records shard size/mtime;
unused experts, vision and MTP are not certified. Failed runs may retain complete
individual artifacts; their presence never proves full-reference completion.

Fixed token IDs, without chat templating or retokenization:

- Hello: `[9419]`.
- Capital: `[760,6511,314,9338,369]`.

Each `<case>.json` and `<case>.safetensors` records 654 tensors: every vocabulary
logit, embedding/final norm, all layers' hidden/attention/MoE outputs, shared-gate
logits, selected expert IDs/weights, GDN conv/recurrent states, attention KV, and
input IDs/positions/masks. Floating boundaries are F32; IDs are I64. `decode.0`
consumes the CUDA prefill prediction and retains the actual prefill cache.
Complete key coverage, shapes and dtypes are checked before publishing a case;
snapshots clone before mutation.

Reference generation is not a Ferrule comparison or task-quality PASS. Numerical
consumers must independently check the pinned arithmetic, payloads, case coverage,
trajectories and strict state/logit criteria. Neither operator agreement nor
argmax agreement certifies full-model quality. Historical reference and failure
receipts remain unchanged; new source or workflow pins require fresh outputs.
