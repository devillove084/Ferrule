#!/usr/bin/env python3
"""
DSV4 Synthetic Fixture Generator for Ferrule offload/streaming testing.

Generates a structurally valid DeepSeek-V4 safetensors artifact with zero-valued
mock tensor data. The model dimensions are scaled so the total artifact size
targets VRAM × 2 (default), forcing Ferrule's expert streaming / residency paths
even on single-GPU machines.

Usage:
  python scripts/dsv4_synthetic_fixture.py [--target-bytes N] [--output DIR]

Requirements: Python 3.10+, no extra packages needed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

# ── dtype helpers ──────────────────────────────────────────────────────────

DTYPE_BYTES: dict[str, int] = {
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I8": 1,
    "I32": 4,
    "I64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "BOOL": 1,
}

FP8_BLOCK = 128        # FP8 E4M3: block dim for both rows and cols
FP4_SCALE_BLOCK = 32   # FP4 E2M1: each F8_E8M0 scale covers 32 FP4 elements (row-wise)


def dtype_size(dtype: str) -> int:
    return DTYPE_BYTES.get(dtype, 1)


def tensor_bytes(shape: list[int], dtype: str) -> int:
    n = 1
    for d in shape:
        n *= d
    return n * dtype_size(dtype)


# ── VRAM detection ─────────────────────────────────────────────────────────

def probe_nvidia_vram_bytes() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        mb = float(out.strip().split("\n")[0].strip())
        return int(mb * 1024 * 1024)
    except Exception:
        pass
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return int(info.total)
    except Exception:
        pass
    return None


# ── tensor enumeration ─────────────────────────────────────────────────────

def _fp8_scale(rows: int, cols: int) -> list[int]:
    """FP8 E4M3 scale tensor shape: [ceil(rows/128), ceil(cols/128)]."""
    return [
        (rows + FP8_BLOCK - 1) // FP8_BLOCK,
        (cols + FP8_BLOCK - 1) // FP8_BLOCK,
    ]


def _fp4_weight(out_features: int, in_features: int) -> list[int]:
    """FP4 E2M1 weight stored in I8: 2 FP4 values per byte → cols = in_features // 2."""
    return [out_features, in_features // 2]


def _fp4_scale(out_features: int, in_features: int) -> list[int]:
    """FP4 E2M1 scale: row-wise, [out_features, in_features // 32]."""
    return [out_features, in_features // FP4_SCALE_BLOCK]


def iter_tensor_specs(config: dict[str, Any]) -> Iterator[tuple[str, str, list[int]]]:
    """Yield (name, dtype, shape) for every tensor in the model."""
    hidden = config["hidden_size"]
    vocab = config.get("vocab_size", 129280)
    num_layers = config.get("num_hidden_layers", 43)
    num_heads = config.get("num_attention_heads", 64)
    head_dim = config.get("head_dim", 512)
    q_lora = config.get("q_lora_rank", 1024)
    o_lora = config.get("o_lora_rank", 1024)
    moe_inter = config.get("moe_intermediate_size", 2048)
    n_experts = config.get("n_routed_experts", 256)
    num_hash = config.get("num_hash_layers", 3)
    hc_mult = config.get("hc_mult", 4)
    hc_hidden = hc_mult * hidden
    mix_hc = (2 + hc_mult) * hc_mult
    o_groups = config.get("o_groups", 8)
    n_experts_per_tok = config.get("num_experts_per_tok", 6)
    num_mtp = config.get("num_nextn_predict_layers", 1)
    markov_rank = config.get("dspark_markov_rank", 256)

    T = tuple[str, str, list[int]]

    def t(name: str, dtype: str, shape: list[int]) -> T:
        return (name, dtype, shape)

    # ── Top-level ──
    yield t("embed.weight", "BF16", [vocab, hidden])
    yield t("norm.weight", "BF16", [hidden])
    yield t("head.weight", "BF16", [vocab, hidden])
    yield t("hc_head_base", "F32", [hc_mult])
    yield t("hc_head_fn", "F32", [hc_mult, hc_hidden])
    yield t("hc_head_scale", "F32", [1])

    q_out = num_heads * head_dim           # q_full_dim
    output_latent = o_groups * o_lora       # output_latent_dim
    group_input = q_out // o_groups         # output_group_input_dim
    kv_dim = head_dim
    wo_a_cols = output_latent               # keep backward compat name

    # ── Main transformer layers ──
    for layer in range(num_layers):
        pf = f"layers.{layer}"

        # Norms
        yield t(f"{pf}.attn_norm.weight", "BF16", [hidden])
        yield t(f"{pf}.ffn_norm.weight", "BF16", [hidden])

        # Hyper-connections
        yield t(f"{pf}.hc_attn_base", "F32", [mix_hc])
        yield t(f"{pf}.hc_attn_fn", "F32", [mix_hc, hc_hidden])
        yield t(f"{pf}.hc_attn_scale", "F32", [3])
        yield t(f"{pf}.hc_ffn_base", "F32", [mix_hc])
        yield t(f"{pf}.hc_ffn_fn", "F32", [mix_hc, hc_hidden])
        yield t(f"{pf}.hc_ffn_scale", "F32", [3])

        # MLA attention (all FP8 except norms + attn_sink)
        yield t(f"{pf}.attn.attn_sink", "F32", [num_heads])
        yield t(f"{pf}.attn.wq_a.weight", "F8_E4M3", [q_lora, hidden])
        yield t(f"{pf}.attn.wq_a.scale", "F8_E8M0", _fp8_scale(q_lora, hidden))
        yield t(f"{pf}.attn.wq_b.weight", "F8_E4M3", [q_out, q_lora])
        yield t(f"{pf}.attn.wq_b.scale", "F8_E8M0", _fp8_scale(q_out, q_lora))
        yield t(f"{pf}.attn.q_norm.weight", "BF16", [q_lora])
        yield t(f"{pf}.attn.wo_a.weight", "F8_E4M3", [output_latent, group_input])
        yield t(f"{pf}.attn.wo_a.scale", "F8_E8M0", _fp8_scale(output_latent, group_input))
        yield t(f"{pf}.attn.wkv.weight", "F8_E4M3", [kv_dim, hidden])
        yield t(f"{pf}.attn.wkv.scale", "F8_E8M0", _fp8_scale(kv_dim, hidden))
        yield t(f"{pf}.attn.kv_norm.weight", "BF16", [kv_dim])
        yield t(f"{pf}.attn.wo_b.weight", "F8_E4M3", [hidden, output_latent])
        yield t(f"{pf}.attn.wo_b.scale", "F8_E8M0", _fp8_scale(hidden, output_latent))

        # Router
        yield t(f"{pf}.ffn.gate.weight", "BF16", [n_experts, hidden])
        if layer < num_hash:
            yield t(f"{pf}.ffn.gate.tid2eid", "I64", [vocab, n_experts_per_tok])
        else:
            yield t(f"{pf}.ffn.gate.bias", "F32", [n_experts])

        # Shared experts (FP8)
        yield t(f"{pf}.ffn.shared_experts.w1.weight", "F8_E4M3", [moe_inter, hidden])
        yield t(f"{pf}.ffn.shared_experts.w1.scale", "F8_E8M0", _fp8_scale(moe_inter, hidden))
        yield t(f"{pf}.ffn.shared_experts.w3.weight", "F8_E4M3", [moe_inter, hidden])
        yield t(f"{pf}.ffn.shared_experts.w3.scale", "F8_E8M0", _fp8_scale(moe_inter, hidden))
        yield t(f"{pf}.ffn.shared_experts.w2.weight", "F8_E4M3", [hidden, moe_inter])
        yield t(f"{pf}.ffn.shared_experts.w2.scale", "F8_E8M0", _fp8_scale(hidden, moe_inter))

        # Routed experts (FP4 packed in I8: 2 FP4 per I8 byte, 32-el scale blocks)
        for expert in range(n_experts):
            epf = f"{pf}.ffn.experts.{expert}"
            yield t(f"{epf}.w1.weight", "I8", _fp4_weight(moe_inter, hidden))
            yield t(f"{epf}.w1.scale", "F8_E8M0", _fp4_scale(moe_inter, hidden))
            yield t(f"{epf}.w3.weight", "I8", _fp4_weight(moe_inter, hidden))
            yield t(f"{epf}.w3.scale", "F8_E8M0", _fp4_scale(moe_inter, hidden))
            yield t(f"{epf}.w2.weight", "I8", _fp4_weight(hidden, moe_inter))
            yield t(f"{epf}.w2.scale", "F8_E8M0", _fp4_scale(hidden, moe_inter))

    # ── MTP speculative layers ──
    for m in range(num_mtp):
        pf = f"mtp.{m}"

        yield t(f"{pf}.attn_norm.weight", "BF16", [hidden])
        yield t(f"{pf}.ffn_norm.weight", "BF16", [hidden])

        yield t(f"{pf}.hc_attn_base", "F32", [mix_hc])
        yield t(f"{pf}.hc_attn_fn", "F32", [mix_hc, hc_hidden])
        yield t(f"{pf}.hc_attn_scale", "F32", [3])
        yield t(f"{pf}.hc_ffn_base", "F32", [mix_hc])
        yield t(f"{pf}.hc_ffn_fn", "F32", [mix_hc, hc_hidden])
        yield t(f"{pf}.hc_ffn_scale", "F32", [3])

        # MLA attention (FP8)
        yield t(f"{pf}.attn.attn_sink", "F32", [num_heads])
        yield t(f"{pf}.attn.wq_a.weight", "F8_E4M3", [q_lora, hidden])
        yield t(f"{pf}.attn.wq_a.scale", "F8_E8M0", _fp8_scale(q_lora, hidden))
        yield t(f"{pf}.attn.wq_b.weight", "F8_E4M3", [q_out, q_lora])
        yield t(f"{pf}.attn.wq_b.scale", "F8_E8M0", _fp8_scale(q_out, q_lora))
        yield t(f"{pf}.attn.q_norm.weight", "BF16", [q_lora])
        yield t(f"{pf}.attn.wo_a.weight", "F8_E4M3", [output_latent, group_input])
        yield t(f"{pf}.attn.wo_a.scale", "F8_E8M0", _fp8_scale(output_latent, group_input))
        yield t(f"{pf}.attn.wkv.weight", "F8_E4M3", [kv_dim, hidden])
        yield t(f"{pf}.attn.wkv.scale", "F8_E8M0", _fp8_scale(kv_dim, hidden))
        yield t(f"{pf}.attn.kv_norm.weight", "BF16", [kv_dim])
        yield t(f"{pf}.attn.wo_b.weight", "F8_E4M3", [hidden, output_latent])
        yield t(f"{pf}.attn.wo_b.scale", "F8_E8M0", _fp8_scale(hidden, output_latent))

        # Router
        yield t(f"{pf}.ffn.gate.weight", "BF16", [n_experts, hidden])
        yield t(f"{pf}.ffn.gate.bias", "F32", [n_experts])

        # Shared experts (FP8)
        yield t(f"{pf}.ffn.shared_experts.w1.weight", "F8_E4M3", [moe_inter, hidden])
        yield t(f"{pf}.ffn.shared_experts.w1.scale", "F8_E8M0", _fp8_scale(moe_inter, hidden))
        yield t(f"{pf}.ffn.shared_experts.w3.weight", "F8_E4M3", [moe_inter, hidden])
        yield t(f"{pf}.ffn.shared_experts.w3.scale", "F8_E8M0", _fp8_scale(moe_inter, hidden))
        yield t(f"{pf}.ffn.shared_experts.w2.weight", "F8_E4M3", [hidden, moe_inter])
        yield t(f"{pf}.ffn.shared_experts.w2.scale", "F8_E8M0", _fp8_scale(hidden, moe_inter))

        # Routed experts (FP4/I8)
        for expert in range(n_experts):
            epf = f"{pf}.ffn.experts.{expert}"
            yield t(f"{epf}.w1.weight", "I8", _fp4_weight(moe_inter, hidden))
            yield t(f"{epf}.w1.scale", "F8_E8M0", _fp4_scale(moe_inter, hidden))
            yield t(f"{epf}.w3.weight", "I8", _fp4_weight(moe_inter, hidden))
            yield t(f"{epf}.w3.scale", "F8_E8M0", _fp4_scale(moe_inter, hidden))
            yield t(f"{epf}.w2.weight", "I8", _fp4_weight(hidden, moe_inter))
            yield t(f"{epf}.w2.scale", "F8_E8M0", _fp4_scale(hidden, moe_inter))

        # MTP-specific projections
        yield t(f"{pf}.main_proj.weight", "BF16", [hidden, hidden])
        yield t(f"{pf}.main_proj.scale", "F32", [1])
        yield t(f"{pf}.main_norm.weight", "BF16", [hidden])

    # MTP output head (only last MTP layer)
    if num_mtp > 0:
        last = num_mtp - 1
        yield t(f"mtp.{last}.hc_head_base", "F32", [hc_mult])
        yield t(f"mtp.{last}.hc_head_fn", "F32", [hc_mult, hc_hidden])
        yield t(f"mtp.{last}.hc_head_scale", "F32", [1])
        yield t(f"mtp.{last}.norm.weight", "BF16", [hidden])
        yield t(f"mtp.{last}.markov_head.markov_w1.weight", "BF16", [vocab, markov_rank])
        yield t(f"mtp.{last}.markov_head.markov_w2.weight", "BF16", [vocab, markov_rank])
        yield t(f"mtp.{last}.confidence_head.proj.weight", "BF16", [1, hidden + markov_rank])


def total_bytes_for_config(config: dict[str, Any]) -> int:
    return sum(
        tensor_bytes(shape, dtype) for _, dtype, shape in iter_tensor_specs(config)
    )


# ── config scaling ─────────────────────────────────────────────────────────

def _align_up(val: int, align: int) -> int:
    """Round val up to the nearest multiple of align."""
    return ((val + align - 1) // align) * align


def _nearest_divisor(n: int, target: int) -> int:
    """Return divisor of n closest to target (for o_groups | num_heads)."""
    best, best_dist = 1, n
    for d in range(1, n + 1):
        if n % d == 0:
            dist = abs(d - target)
            if dist < best_dist:
                best, best_dist = d, dist
    return best


def scale_config(config: dict[str, Any], factor: float) -> dict[str, Any]:
    """Scale all integer dimensions; preserve architecture identity."""
    cfg = copy.deepcopy(config)

    def s(val: int) -> int:
        return max(1, int(round(val * factor)))

    cfg["hidden_size"] = s(cfg["hidden_size"])
    cfg["moe_intermediate_size"] = s(cfg.get("moe_intermediate_size", 2048))
    cfg["q_lora_rank"] = s(cfg.get("q_lora_rank", 1024))
    cfg["o_lora_rank"] = s(cfg.get("o_lora_rank", 1024))
    cfg["head_dim"] = s(cfg.get("head_dim", 512))
    # Align to LCM of FP4 scale block (32) and FP8 block (128) = 128
    cfg["hidden_size"] = _align_up(cfg["hidden_size"], 128)
    cfg["moe_intermediate_size"] = _align_up(cfg["moe_intermediate_size"], 128)
    cfg["q_lora_rank"] = _align_up(cfg["q_lora_rank"], 128)
    cfg["o_lora_rank"] = _align_up(cfg["o_lora_rank"], 128)
    cfg["head_dim"] = _align_up(cfg["head_dim"], 128)
    cfg["qk_rope_head_dim"] = s(cfg.get("qk_rope_head_dim", 64))
    # QAT FP8 quant uses block_size=64 on non_rope = head_dim - rope_dim
    # Ensure non_rope is divisible by 64
    cfg["qk_rope_head_dim"] = _align_up(cfg["qk_rope_head_dim"], 64)
    if cfg["qk_rope_head_dim"] > cfg["head_dim"]:
        cfg["qk_rope_head_dim"] = (cfg["head_dim"] // 64) * 64
    cfg["num_attention_heads"] = max(1, s(cfg.get("num_attention_heads", 64)))
    raw_groups = max(1, s(cfg.get("o_groups", 8)))
    cfg["o_groups"] = _nearest_divisor(cfg["num_attention_heads"], raw_groups)
    cfg["index_head_dim"] = s(cfg.get("index_head_dim", 128))
    cfg["index_n_heads"] = max(1, s(cfg.get("index_n_heads", 64)))
    cfg["index_topk"] = s(cfg.get("index_topk", 512))
    cfg["sliding_window"] = s(cfg.get("sliding_window", 128))
    cfg["vocab_size"] = s(cfg.get("vocab_size", 129280))
    cfg["n_routed_experts"] = max(2, s(cfg.get("n_routed_experts", 256)))
    cfg["num_experts_per_tok"] = min(
        cfg.get("num_experts_per_tok", 6), cfg["n_routed_experts"]
    )
    if "dspark_markov_rank" in cfg:
        cfg["dspark_markov_rank"] = s(cfg["dspark_markov_rank"])

    # Drop MTP if model shrinks below 30%
    if factor < 0.3:
        cfg["num_nextn_predict_layers"] = 0
    else:
        cfg["num_nextn_predict_layers"] = min(cfg.get("num_nextn_predict_layers", 1), 3)

    # At very small factors, reduce layer count
    if factor < 0.1:
        cfg["num_hidden_layers"] = max(2, s(cfg.get("num_hidden_layers", 43)))
        cfg["num_hash_layers"] = min(cfg["num_hash_layers"], cfg["num_hidden_layers"])

    # Trim compress_ratios — synthetic fixture doesn't have compressor/indexer
    # auxiliary tensors, so set all ratios to 0 (standard attention only)
    cfg["compress_ratios"] = [0] * cfg["num_hidden_layers"]

    if "dspark_target_layer_ids" in cfg:
        cfg["dspark_target_layer_ids"] = [
            lid
            for lid in cfg["dspark_target_layer_ids"]
            if lid < cfg["num_hidden_layers"]
        ]

    return cfg


def find_scale_factor(
    config: dict[str, Any],
    target_bytes: int,
    *,
    tol: float = 0.02,
    max_iter: int = 40,
) -> float:
    """Binary search for scale factor hitting target_bytes."""
    lo, hi = 0.02, 1.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        est = total_bytes_for_config(scale_config(config, mid))
        if abs(est - target_bytes) / target_bytes < tol:
            return mid
        if est > target_bytes:
            hi = mid
        else:
            lo = mid
    return lo


# ── safetensors writer ─────────────────────────────────────────────────────

def write_safetensors_shard(
    path: Path,
    tensors: list[tuple[str, str, list[int]]],
) -> int:
    """Write a safetensors shard with zero-filled data. Returns file bytes."""
    header: dict[str, Any] = {}
    data_offset = 0

    for name, dtype, shape in tensors:
        nbytes = tensor_bytes(shape, dtype)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [data_offset, data_offset + nbytes],
        }
        data_offset += nbytes

    header_json = json.dumps(header, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")

    # 8-byte alignment padding
    padded_len = (8 + len(header_bytes) + 7) & ~7
    pad = b"\x00" * (padded_len - 8 - len(header_bytes))

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(pad)

        # Write data in 16 MB zero chunks
        chunk = 16 * 1024 * 1024
        remaining = data_offset
        zero_chunk = b"\x00" * chunk
        while remaining >= chunk:
            f.write(zero_chunk)
            remaining -= chunk
        if remaining > 0:
            f.write(b"\x00" * remaining)

    return padded_len + data_offset


def distribute_to_shards(
    tensors: list[tuple[str, str, list[int]]],
    max_shard_bytes: int,
) -> list[list[tuple[str, str, list[int]]]]:
    """Pack tensors into shards ≤ max_shard_bytes each."""
    shards: list[list[tuple[str, str, list[int]]]] = [[]]
    cur = 0
    for t in tensors:
        tb = tensor_bytes(t[2], t[1])
        if shards[-1] and cur + tb > max_shard_bytes:
            shards.append([])
            cur = 0
        shards[-1].append(t)
        cur += tb
    return shards


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="models/DeepSeek-V4-Flash-DSpark",
        help="Path to official DSV4 HF dir (for config.json, tokenizer)",
    )
    parser.add_argument(
        "--output", "-o", default="models/DeepSeek-V4-Flash-DSpark-synthetic",
        help="Output directory",
    )
    parser.add_argument(
        "--target-bytes", type=int, default=None,
        help="Target total model bytes (default: VRAM × 2)",
    )
    parser.add_argument(
        "--scale", type=float, default=None,
        help="Explicit scale factor (overrides --target-bytes)",
    )
    parser.add_argument(
        "--shard-bytes", type=int, default=4 * 1024 * 1024 * 1024,
        help="Max bytes per safetensors shard (default 4 GiB)",
    )
    parser.add_argument(
        "--no-mtp", action="store_true",
        help="Omit MTP speculative layers",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without writing files",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    if not (source / "config.json").exists():
        print(f"Error: {source}/config.json not found", file=sys.stderr)
        sys.exit(1)

    with open(source / "config.json") as f:
        source_config = json.load(f)

    # ── Determine scale ──
    orig_total = total_bytes_for_config(source_config)
    print(f"Original full-scale estimate: {orig_total / 1e9:.2f} GB")

    if args.scale is not None:
        scale = args.scale
        target_bytes = int(orig_total * scale)
    elif args.target_bytes is not None:
        target_bytes = args.target_bytes
        scale = find_scale_factor(source_config, target_bytes)
    else:
        vram = probe_nvidia_vram_bytes()
        if vram is None:
            vram = 24 * 1024 * 1024 * 1024
            print("Warning: VRAM detection failed, assuming 24 GB", file=sys.stderr)
        target_bytes = vram * 2
        print(f"VRAM: {vram / 1e9:.1f} GB → target: {target_bytes / 1e9:.2f} GB")
        scale = find_scale_factor(source_config, target_bytes)

    if args.no_mtp:
        source_config["num_nextn_predict_layers"] = 0

    config = scale_config(source_config, scale)
    est = total_bytes_for_config(config)
    print(f"Scale factor: {scale:.3f} → estimated {est / 1e9:.2f} GB")

    # ── Generate tensors ──
    tensors = list(iter_tensor_specs(config))
    total = sum(tensor_bytes(shape, dtype) for _, dtype, shape in tensors)
    print(
        f"Tensors: {len(tensors)}, data bytes: {total / 1e9:.2f} GB\n"
        f"Config: hidden={config['hidden_size']}, layers={config['num_hidden_layers']}, "
        f"experts={config['n_routed_experts']}, heads={config['num_attention_heads']}, "
        f"head_dim={config['head_dim']}, vocab={config['vocab_size']}, "
        f"q_lora={config['q_lora_rank']}, moe_inter={config['moe_intermediate_size']}"
    )

    # ── Distribute ──
    shards = distribute_to_shards(tensors, args.shard_bytes)
    print(f"Shards: {len(shards)}")

    if args.dry_run:
        for i, shard in enumerate(shards[:3]):
            st = sum(tensor_bytes(t[2], t[1]) for t in shard)
            print(f"  shard {i}: {len(shard)} tensors, {st / 1e6:.1f} MB")
            for t in shard[:8]:
                print(f"    {t[0]:64s} {t[1]:10s} {t[2]}")
        if len(shards) > 3:
            print(f"  ... {len(shards) - 3} more shards")
        print("Dry run complete (no files written).")
        return

    # ── Write output ──
    output.mkdir(parents=True, exist_ok=True)

    clean_config = {k: v for k, v in config.items() if not k.startswith("_")}
    with open(output / "config.json", "w") as f:
        json.dump(clean_config, f, indent=2)
    print("Wrote config.json")

    for name in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        src = source / name
        if src.exists():
            shutil.copy2(src, output / name)

    for sub in ["encoding", "inference"]:
        src_dir = source / sub
        if src_dir.is_dir():
            dst_dir = output / sub
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
    print("Copied tokenizer + encoding + inference")

    # ── Write shards ──
    weight_map: dict[str, str] = {}
    total_file_size = 0

    for i, shard_tensors in enumerate(shards):
        shard_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        shard_path = output / shard_name
        st = sum(tensor_bytes(t[2], t[1]) for t in shard_tensors)
        print(
            f"[{i + 1}/{len(shards)}] {shard_name} "
            f"({len(shard_tensors)} tensors, {st / 1e6:.1f} MB)...",
            end=" ", flush=True,
        )
        size = write_safetensors_shard(shard_path, shard_tensors)
        total_file_size += size
        print(f"ok ({size / 1e6:.1f} MB on disk)")
        for t in shard_tensors:
            weight_map[t[0]] = shard_name

    # ── Write index ──
    index = {
        "metadata": {"total_size": total},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with open(output / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote model.safetensors.index.json ({len(weight_map)} entries)")

    # ── Done ──
    print(f"\n✓ Synthetic fixture ready at: {output.resolve()}")
    print(f"  On-disk:  {total_file_size / 1e9:.2f} GB")
    print(f"  Data:     {total / 1e9:.2f} GB")
    print(f"  Tensors:  {len(weight_map)}")
    print(f"\nTest with:")
    print(f"  cargo run -p ferrule-cli --release -- deepseek-v4-probe {output} --max-layers 0")
    print(f"  cargo run -p ferrule-cli --release -- deepseek-v4-generate {output} --backend cpu --max-layers 1 --max-tokens 1")


if __name__ == "__main__":
    main()
