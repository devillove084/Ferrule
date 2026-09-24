#!/usr/bin/env python3
"""Offline, CPU-only Qwen3.5 numerical reference using native Transformers.

Run with /opt/conda310/bin/python -B scripts/qwen35_reference.py --tiny-fixture.
Only writes target/validation/qwen35-reference-*/. No production dependencies,
custom model forward, downloads, GPU allocations, or resident server. See the
emitted manifest.json and README.md for the versioned Rust-consumer contract.
"""

import argparse
import copy
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
NAS = Path("/mnt/nas1/hf/Qwen3.5-0.8B")
SCHEMA = "ferrule.qwen35-reference.v1"
CACHE_FIELDS = ("conv_states", "recurrent_states", "key_cache", "value_cache")
ATOL, RTOL = 2e-4, 2e-4


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [json_safe(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare(actual, reference, atol=ATOL, rtol=RTOL):
    require(actual.shape == reference.shape, "Comparison shape mismatch")
    a, r = actual.double(), reference.double()
    diff = (a - r).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(r).all())
    require(finite, "Non-finite reference/comparison tensor")
    allowed = atol + rtol * r.abs()
    return {
        "shape": list(r.shape), "atol": atol, "rtol": rtol,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rmse": diff.square().mean().sqrt().item(),
        "max_relative_floor_1e-8": (diff / r.abs().clamp_min(1e-8)).max().item(),
        "max_tolerance_ratio": (diff / allowed).max().item(),
        "failed_elements": int((diff > allowed).sum()),
        "allclose": bool((diff <= allowed).all()),
    }


class Capture:
    """Observe actual module outputs; never recompute or replace a layer."""

    def __init__(self, model):
        self.values = {}
        self.handles = []
        modules = [("embedding", model.model.embed_tokens),
                   ("final_norm", model.model.norm)]
        modules += [(f"layers.{i:02d}.hidden", layer)
                    for i, layer in enumerate(model.model.layers)]
        for name, module in modules:
            def hook(_module, _args, output, name=name):
                self.values[name] = output.detach().clone().contiguous()
            self.handles.append(module.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


def check_backend(model, q):
    require(all(p.device.type == "cpu" and p.dtype == torch.float32
                for p in model.parameters()), "Expected exclusively CPU/F32 parameters")
    require(model.config._attn_implementation == "eager", "Expected eager attention")
    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            attn = layer.linear_attn
            require(attn.causal_conv1d_fn is None, "External conv kernel is not allowed")
            require(attn.causal_conv1d_update is q.torch_causal_conv1d_update,
                    "Expected native torch conv update")
            require(attn.chunk_gated_delta_rule is q.torch_chunk_gated_delta_rule,
                    "Expected native torch chunk rule, not FLA")
            require(attn.recurrent_gated_delta_rule is q.torch_recurrent_gated_delta_rule,
                    "Expected native torch recurrent rule, not FLA")
    require(model.lm_head.weight is model.model.embed_tokens.weight, "Expected tied head")


def run_case(model, case_id, tokens, prompt, decode_text, output_dir, save_states,
             chunk_replay=False, expected_next=None):
    tensors, calls, checks, snapshots = {}, [], {}, {}
    capture = Capture(model)

    def put(key, tensor):
        require(key not in tensors, f"Duplicate tensor key: {key}")
        value = tensor.detach().cpu().contiguous().clone()
        if value.is_floating_point():
            require(bool(torch.isfinite(value).all()), f"Non-finite tensor: {key}")
        tensors[key] = value

    def snapshot(stage, cache):
        entries = []
        for field in CACHE_FIELDS:
            for i, value in enumerate(getattr(cache, field)):
                if value is not None:
                    key = f"{stage}.cache.{field}.{i:02d}"
                    put(key, value)
                    entries.append(key)
        snapshots[stage] = {"sequence_length": cache.get_seq_length(), "keys": entries}

    def forward(stage, ids, cache=None, cache_source=None):
        start = 0 if cache is None else cache.get_seq_length()
        ids = torch.tensor([ids], dtype=torch.int64)
        positions = torch.arange(start, start + ids.shape[1], dtype=torch.int64)
        mask = torch.ones((1, start + ids.shape[1]), dtype=torch.int64)
        capture.values.clear()
        with torch.inference_mode():
            result = model(input_ids=ids, attention_mask=mask,
                           position_ids=positions[None, :], cache_position=positions,
                           past_key_values=cache, use_cache=True,
                           output_hidden_states=True, return_dict=True, logits_to_keep=0)
        # HF hidden_states[-1] is ALREADY normalized, unlike the final layer hook.
        hidden = result.hidden_states
        require(len(hidden) == len(model.model.layers) + 1, "Unexpected hidden-state count")
        require(torch.equal(hidden[0], capture.values["embedding"]), "Embedding hook mismatch")
        for i in range(len(model.model.layers) - 1):
            require(torch.equal(hidden[i + 1], capture.values[f"layers.{i:02d}.hidden"]),
                    f"Layer {i} hook mismatch")
        require(torch.equal(hidden[-1], capture.values["final_norm"]), "Final norm mismatch")
        put(f"{stage}.input_ids", ids)
        put(f"{stage}.position_ids", positions[None, :])
        put(f"{stage}.cache_position", positions)
        put(f"{stage}.attention_mask", mask)
        for name, value in capture.values.items():
            put(f"{stage}.{name}", value)
        put(f"{stage}.logits", result.logits)
        put(f"{stage}.last_logits", result.logits[:, -1, :])
        next_id = int(result.logits[0, -1].argmax())
        put(f"{stage}.next_token_id", torch.tensor([next_id], dtype=torch.int64))
        calls.append({"stage": stage, "input_token_ids": ids[0].tolist(),
                      "cache_source": cache_source, "position_start": start,
                      "sequence_length_after": result.past_key_values.get_seq_length(),
                      "next_token_id": next_id, "next_token_text": decode_text([next_id])})
        return result.past_key_values, next_id

    def compare_stage(actual, reference, reference_slice=None):
        names = ["logits", "embedding", "final_norm"]
        names += [f"layers.{i:02d}.hidden" for i in range(len(model.model.layers))]
        metrics = {}
        for name in names:
            ref = tensors[f"{reference}.{name}"]
            if reference_slice is not None:
                ref = ref[:, reference_slice, :]
            metrics[name] = compare(tensors[f"{actual}.{name}"], ref)
        checks[f"{actual}_vs_{reference}"] = metrics

    try:
        cache, next_id = forward("prefill", tokens)
        if expected_next is not None:
            require(decode_text([next_id]) == expected_next,
                    f"{case_id}: expected {expected_next!r}, got {decode_text([next_id])!r}")
        # Snapshot BEFORE further calls: the native cache mutates in place.
        prefix_cache = copy.deepcopy(cache) if chunk_replay else None
        if save_states:
            snapshot("prefill", cache)
        generated = [next_id]
        history = list(tokens)
        for step in range(2):
            stage = f"decode.{step}"
            fed_token = next_id
            history.append(fed_token)
            source = "prefill" if step == 0 else "decode.0"
            cache, next_id = forward(stage, [fed_token], cache, source)
            generated.append(next_id)
        if save_states:
            snapshot("final", cache)

        if chunk_replay:
            # Full-prefix recomputation and cached decode must represent the same history.
            replay_cache, _ = forward("full_replay", history)
            compare_stage("decode.1", "full_replay", slice(-1, None))
            if save_states:
                snapshot("full_replay", replay_cache)
                metrics = {}
                for key in snapshots["final"]["keys"]:
                    suffix = key.removeprefix("final.")
                    metrics[suffix] = compare(tensors[key], tensors[f"full_replay.{suffix}"])
                checks["final_cache_vs_full_replay"] = metrics
            del replay_cache

            # Exact replay from an independent snapshot, not an already-mutated cache.
            replay_cache = prefix_cache
            for step in range(2):
                stage = f"cache_replay.{step}"
                source = "prefill(snapshot)" if step == 0 else "cache_replay.0"
                replay_cache, _ = forward(stage, [generated[step]], replay_cache, source)
                compare_stage(stage, f"decode.{step}")
                require(torch.equal(tensors[f"{stage}.logits"], tensors[f"decode.{step}.logits"]),
                        "Restored-cache replay is not bitwise identical")
            del replay_cache, prefix_cache

            # TF 5.2 resets linear state for cached inputs longer than one token.
            # A multi-token FIRST chunk followed by one-token chunks is supported.
            split = max(1, len(tokens) // 2)
            sizes = [split] + [1] * (len(tokens) - split)
            chunk_cache, offset, previous = None, 0, None
            for i, size in enumerate(sizes):
                stage = f"chunk.{i}"
                chunk_cache, _ = forward(stage, tokens[offset:offset + size], chunk_cache, previous)
                compare_stage(stage, "prefill", slice(offset, offset + size))
                offset += size
                previous = stage
            if save_states:
                snapshot("chunk_final", chunk_cache)
                checks["chunk_cache_vs_prefill"] = {
                    key.removeprefix("chunk_final."): compare(
                        tensors[key], tensors["prefill." + key.removeprefix("chunk_final.")])
                    for key in snapshots["chunk_final"]["keys"]
                }
            del chunk_cache

        for group, metrics in checks.items():
            for name, metric in metrics.items():
                require(metric["allclose"], f"Reference self-check failed: {group}/{name}: {metric}")
        filename = f"{case_id}.safetensors"
        save_file(tensors, str(output_dir / filename), metadata={"schema": SCHEMA, "case": case_id})
        report = {
            "id": case_id, "prompt": prompt, "request_token_ids": tokens,
            "add_special_tokens": False, "chat_template": False,
            "expected_first_token_text": expected_next,
            "greedy_prediction_ids": generated, "greedy_prediction_text": decode_text(generated),
            "decode_input_token_ids": generated[:2],
            "calls": calls, "cache_snapshots": snapshots, "comparisons": checks,
            "file": filename, "sha256": sha256(output_dir / filename),
            "tensors": {key: {"dtype": "F32" if t.dtype == torch.float32 else "I64",
                              "shape": list(t.shape)} for key, t in sorted(tensors.items())},
        }
        write_json(output_dir / f"{case_id}.json", report)
        print(f"{case_id}: tokens={tokens}, predictions={generated} {decode_text(generated)!r}, "
              f"{len(tensors)} tensors", flush=True)
        return {"id": case_id, "report": f"{case_id}.json", "file": filename,
                "sha256": report["sha256"], "self_checks_passed": True}
    finally:
        capture.close()


def tiny_fixture(config, output_dir, seed, q):
    text = config.text_config.to_dict()
    text.update(vocab_size=64, hidden_size=64, intermediate_size=96,
                num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2,
                linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4,
                layer_types=["linear_attention"] * 3 + ["full_attention"],
                max_position_embeddings=128, mtp_num_hidden_layers=0, dtype="float32",
                eos_token_id=2, tie_word_embeddings=True,
                rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                                 "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                                 "mrope_section": [1, 1, 2]})
    torch.manual_seed(seed)
    tiny_config = q.Qwen3_5TextConfig(**text)
    tiny_config._attn_implementation = "eager"
    model = q.Qwen3_5ForCausalLM(tiny_config).float().eval()
    # NAS layout, not standalone CausalLM's model.layers.* layout. No vision/MTP tensors.
    weights = {"model.language_model." + key.removeprefix("model."): value.contiguous().clone()
               for key, value in model.state_dict().items() if key.startswith("model.")}
    directory = output_dir / "tiny-checkpoint"
    directory.mkdir()
    nested = config.to_dict()
    nested.update(text_config=tiny_config.to_dict(), dtype="float32",
                  tie_word_embeddings=True)
    write_json(directory / "config.json", nested)
    save_file(weights, str(directory / "model.safetensors"), metadata={"format": "pt"})
    # Test the same native NAS-prefix conversion consumers rely on.
    reloaded, info = AutoModelForCausalLM.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
        device_map={"": "cpu"}, attn_implementation="eager", output_loading_info=True)
    require(not any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")),
            f"Tiny checkpoint reload mismatch: {info}")
    require(all(torch.equal(v, reloaded.state_dict()[k]) for k, v in model.state_dict().items()),
            "Tiny checkpoint prefix conversion changed weights")
    check_backend(reloaded, q)
    descriptor = {
        "seed": seed, "purpose": "random text-only unit-test fixture, not a language oracle",
        "config": "tiny-checkpoint/config.json", "weights": "tiny-checkpoint/model.safetensors",
        "sha256": sha256(directory / "model.safetensors"),
        "config_sha256": sha256(directory / "config.json"),
        "nested_model_type": "qwen3_5", "text_model_type": "qwen3_5_text",
        "weight_prefix": "model.language_model.", "tie_word_embeddings": True,
        "absent_weights": ["lm_head.weight (tied)", "model.visual.*", "mtp.*"],
        "vision_config": "Copied from NAS for nested config compatibility; no vision weights or tests",
        "reload": "AutoModelForCausalLM local reload and every tensor equality verified",
        "initialization": "torch.manual_seed(seed); native Qwen3_5ForCausalLM F32 initialization",
    }
    return reloaded.eval(), descriptor


README = """# Qwen3.5 reference v1

Native Transformers only; CPU F32; eager attention; no FLA/causal-conv1d kernels.
BF16 checkpoint tensors are converted directly to F32 on load; original F32
checkpoint tensors retain their precision. No BF16 forward or autocast.
A stdlib-only supervisor enforces the wall-clock timeout, kills the worker process
group on expiry, and leaves status=failed. Existing output directories are refused.
Only status=complete artifacts may be consumed. run.log records native warnings.

## Rust consumer contract

Read manifest.json, then each cases[].report. Tensor data is in the corresponding
safetensors file. Every report enumerates exact tensor keys, shapes, and F32/I64
dtypes. Safetensors stores little-endian, contiguous row-major data, no transpose.
Check file SHA256 before reading. Reject status != complete or unknown schema.
Config, original NAS file hashes, tokenizer hashes, source hash and versions are
recorded. Request IDs are authoritative; do not tokenize again in numerical tests.
Use case calls[] order/cache_source/position_start to replay the same execution.

Each call has input_ids [1,T], position_ids [1,T], cache_position [T], and
attention_mask [1,past+T] (all ones). Text-only positions are expanded to 3 axes
inside HF. No padding, chat template, sampling, temperature, or special tokens.
Call tensors:
- embedding [1,T,H]: actual embedding output.
- layers.NN.hidden [1,T,H]: actual decoder-layer output AFTER its residual/MLP,
  BEFORE the model's final norm, zero-based layer index. Captured by hooks.
- final_norm [1,T,H]: actual final norm output, equals HF hidden_states[-1].
  Do NOT normalize it again. The final layer hook still contains the raw output.
- logits [1,T,V]: every vocabulary value at every input position (not top-k).
- last_logits [1,V]: full last row of logits.
- next_token_id [1]: greedy argmax (first index in a tie).

A prefill predicts g0; decode.0 consumes g0 and predicts g1; decode.1 consumes g1
and predicts g2. Thus prefill + TWO decode forwards yields THREE predictions.
The final cache contains request + g0 + g1, NOT g2. Reports record both prediction
IDs and the two IDs actually fed back. Hello is a plain completion, not chat.

Layer outputs are hidden vectors, not vocabulary logits. They are the correct
single-layer numerical oracle given that layer's input and cache. There is no
invented per-layer LM head or extra norm operation.

Optional cache snapshots use stage.cache.FIELD.NN:
- conv_states: [B,2*Hk*Dk+Hv*Dv,Kconv], projected pre-conv Q/K/V history;
  channels concatenate Q,K,V, time axis oldest to newest, left-zero-padded.
- recurrent_states: [B,Hv,Dk,Dv], native gated-delta state (key axis before value).
- key_cache/value_cache: [B,Hkv,total_tokens,Dhead]; K after QK norm + RoPE,
  V as stored by native HF. Only full-attention layers have KV entries.
Missing cache keys mean None/not applicable, not an empty tensor or zero state.
Snapshots are cloned BEFORE mutation; sequence_length is in the JSON report.

## Chunk and replay semantics

The capital case also includes:
1. full_replay: no cache, recompute request + g0 + g1; compare final row and each
   final-row hidden against decode.1, and final cache if exported.
2. cache_replay.0/.1: deep-copy the prefill cache and replay exactly g0/g1;
   logits must be bitwise identical to decode.0/.1.
3. chunk.N: multi-token FIRST chunk, then single-token chunks of the original
   request; concatenate outputs by position to compare to one-shot prefill.

IMPORTANT: local TF 5.2 GatedDeltaNet uses previous recurrent/conv state only if
cached seq_len == 1. Later multi-token chunks reset linear state (initial_state
None). This artifact does NOT certify multi-token cached continuation; do not
patch or reinterpret that native forward as an oracle for that operation.

## Exact error calculation

First require identical shape, finite values, and exact integer/token equality.
For floats cast BOTH actual and reference to F64 before metrics. Reference r is
the artifact, a is the Rust result. d=abs(a-r); pass iff EVERY element satisfies
 d <= 0.0002 + 0.0002*abs(r)
Report max(d), mean(d), sqrt(mean((a-r)^2)),
max(d/max(abs(r),1e-8)), failed element count, and
max(d/(0.0002+0.0002*abs(r))). No transposes, broadcasting, relative-only criterion,
or top-k-only checks. Stored self-comparisons use these same formulas. Tolerances
are for this F32 native reference, not a guarantee for BF16/CUDA implementations.
The cache snapshot replay additionally requires exact logits equality.

Tiny fixture (when requested) is random, fixed-seed, F32, 4 text layers. Its JSON
retains NAS's nested qwen3_5/text_config shape; safetensors keys have NAS's
model.language_model.* prefix and omit the tied lm_head. Vision config is present
but vision and MTP weights are absent: load with AutoModelForCausalLM, NOT the
multimodal class. Its own outputs and seed/config/hash descriptor are in manifest.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=NAS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300, help="Wall-clock seconds, includes loading/hashing")
    parser.add_argument("--seed", type=int, default=3508)
    parser.add_argument("--cache-states", action="store_true", help="Export conv/recurrent/KV snapshots")
    parser.add_argument("--tiny-fixture", action="store_true", help="Also export fixed-seed random text checkpoint and oracle")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(args.threads > 0 and args.timeout > 0, "threads and timeout must be positive")
    require(args.model.is_dir(), "--model must be an existing local checkpoint directory")
    validation = ROOT / "target" / "validation"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or validation / f"qwen35-reference-{stamp}").resolve()
    require(output.parent == validation.resolve() and output.name.startswith("qwen35-reference-"),
            "Output must be target/validation/qwen35-reference-* (fresh directory)")
    require(validation.is_dir(), "target/validation must already exist")
    os.environ.update(CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                      PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads))
    if not args.worker:
        # A separate supervisor enforces the deadline even inside blocked native code.
        output.mkdir(exist_ok=False)
        command = [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:],
                   "--worker", "--output", str(output)]
        with (output / "run.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                status = child.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                path = output / "manifest.json"
                report = json.loads(path.read_text()) if path.exists() else {"schema": SCHEMA}
                report.update(status="failed", error=f"{type(error).__name__}: worker terminated")
                write_json(path, report)
                raise
        print((output / "run.log").read_text(), end="", flush=True)
        raise SystemExit(status if status >= 0 else 1)
    require(output.is_dir(), "Internal worker requires a supervisor-created output directory")
    started = time.monotonic()

    def timeout(_signum, _frame):
        raise TimeoutError(f"Oracle exceeded {args.timeout}s")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.timeout)
    arguments = {key: value for key, value in vars(args).items() if key != "worker"}
    manifest = {"schema": SCHEMA, "status": "incomplete", "cases": [], "arguments": arguments}
    manifest["arguments"] = {k: str(v) if isinstance(v, Path) else v
                             for k, v in manifest["arguments"].items()}
    try:
        global torch, save_file, AutoModelForCausalLM
        import torch
        from safetensors.torch import save_file
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        from transformers.utils import logging
        logging.disable_progress_bar()
        require(q.FusedRMSNormGated is None and q.causal_conv1d_fn is None
                and q.chunk_gated_delta_rule is None and q.fused_recurrent_gated_delta_rule is None,
                "Run in a no-FLA/no-causal-conv1d environment; do not substitute accelerated kernels")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        config = AutoConfig.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        require(config.model_type == "qwen3_5", "Expected nested Qwen3.5 config")
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        model, info = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
            device_map={"": "cpu"}, attn_implementation="eager", output_loading_info=True)
        require(not any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")),
                f"Invalid text checkpoint loading: {info}")
        model.eval()
        check_backend(model, q)
        manifest.update(
            versions={p: importlib.metadata.version(p) for p in
                      ("torch", "transformers", "safetensors", "accelerate", "tokenizers", "huggingface-hub")},
            python=sys.version, executable=sys.executable, platform=platform.platform(),
            script_sha256=sha256(__file__), implementation_sha256=sha256(q.__file__),
            implementation_file=q.__file__, model_directory=str(args.model.resolve()),
            model_class=type(model).__name__, loading_info=info,
            execution={"device": "cpu", "dtype": "F32", "weight_conversion": "BF16 -> F32 at load; F32 retained",
                       "attention": "eager", "linear_attention": "native torch fallback",
                       "threads": args.threads, "interop_threads": 1, "seed": args.seed,
                       "deterministic_algorithms": True, "cuda_initialized": torch.cuda.is_initialized()},
            comparison={"atol": ATOL, "rtol": RTOL, "metric_dtype": "F64",
                        "rule": "abs(actual-reference) <= atol + rtol*abs(reference), every element"},
            checkpoint_config=config.to_dict(), effective_text_config=model.config.to_dict())
        index = args.model / "model.safetensors.index.json"
        names = {"config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"}
        if index.exists():
            names.add(index.name)
            names.update(json.loads(index.read_text())["weight_map"].values())
        else:
            names.add("model.safetensors")
        print("Hashing source checkpoint/tokenizer files...", flush=True)
        manifest["source_files"] = {
            name: {"bytes": (args.model / name).stat().st_size, "sha256": sha256(args.model / name)}
            for name in sorted(names) if (args.model / name).is_file()}
        (output / "README.md").write_text(README)
        write_json(output / "manifest.json", manifest)
        decode = lambda ids: tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        for name, prompt, chunk, expected in [
            ("capital", "The capital of France is", True, " Paris"),
            ("hello", "Hello", False, None),
        ]:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            manifest["cases"].append(run_case(model, name, tokens, prompt, decode, output,
                                              args.cache_states, chunk, expected))
        del model
        if args.tiny_fixture:
            tiny, descriptor = tiny_fixture(config, output, args.seed, q)
            manifest["tiny_fixture"] = descriptor
            manifest["cases"].append(run_case(tiny, "tiny", [4, 7, 9, 12, 15], None,
                                              lambda ids: str(ids), output, args.cache_states, True))
            del tiny
        require(not torch.cuda.is_initialized(), "CUDA was unexpectedly initialized")
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started)
        write_json(output / "manifest.json", manifest)
        print(f"PASS: {output / 'manifest.json'}", flush=True)
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}",
                        elapsed_seconds=time.monotonic() - started)
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
