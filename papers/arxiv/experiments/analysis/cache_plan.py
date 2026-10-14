"""Header-only cache counterfactual arithmetic; NOT CUDA/live factory admission.

Reads config/index and bounded safetensors JSON headers, never tensor payloads.
Prints proposed flag diffs, never edits a runner configuration or launches a CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from pathlib import Path

MIB = 1 << 20
GIB = 1 << 30
EXPERT = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(weight|weight_scale_inv)$")


def expert_headers(model: str | Path) -> dict:
    """Verify all routed experts in the checkpoint using metadata only."""
    root = Path(model).resolve()
    config_bytes = (root / "config.json").read_bytes()
    index_bytes = (root / "model.safetensors.index.json").read_bytes()
    config = json.loads(config_bytes)["text_config"]
    index = json.loads(index_bytes)["weight_map"]
    headers = {}
    header_hashes = {}
    experts = defaultdict(dict)
    header_bytes = 0
    for name, shard in index.items():
        match = EXPERT.fullmatch(name)
        if not match:
            continue
        layer, expert = map(int, match.group(1, 2))
        role, kind = match.group(3, 4)
        if shard not in headers:
            path = (root / shard).resolve()
            if path.parent != root:
                raise ValueError("checkpoint shard must be a direct model child")
            with path.open("rb", buffering=0) as stream:
                prefix = stream.read(8)
                if len(prefix) != 8:
                    raise ValueError("truncated safetensors prefix")
                size, = struct.unpack("<Q", prefix)
                if not 0 < size <= 64 * MIB:
                    raise ValueError("safetensors header exceeds 64 MiB bound")
                data = stream.read(size)
                if len(data) != size:
                    raise ValueError("truncated safetensors header")
            headers[shard] = json.loads(data)
            header_hashes[shard] = hashlib.sha256(prefix + data).hexdigest()
            header_bytes += len(prefix) + len(data)
        item = headers[shard][name]
        hidden, intermediate = config["hidden_size"], config["moe_intermediate_size"]
        expected = [hidden, intermediate] if role == "down" else [intermediate, hidden]
        if kind == "weight_scale_inv":
            expected = [(n + 127) // 128 for n in expected]
        dtype = "F8_E4M3" if kind == "weight" else "BF16"
        if item["shape"] != expected or item["dtype"] != dtype:
            raise ValueError(f"unexpected numeric expert metadata: {name}")
        start, end = item["data_offsets"]
        size = math.prod(expected) * (1 if kind == "weight" else 2)
        if type(start) is not int or type(end) is not int or start < 0 or end - start != size:
            raise ValueError(f"invalid expert extent: {name}")
        experts[layer, expert][role, kind] = size
    layers, count = config["num_hidden_layers"], config["num_experts"]
    expected_ids = {(layer, expert) for layer in range(layers) for expert in range(count)}
    expected_parts = {(role, kind) for role in ("gate", "up", "down")
                      for kind in ("weight", "weight_scale_inv")}
    if set(experts) != expected_ids or any(set(parts) != expected_parts for parts in experts.values()):
        raise ValueError("incomplete routed expert header directory")
    sizes = {sum(parts.values()) for parts in experts.values()}
    if len(sizes) != 1:
        raise ValueError("counterfactual requires uniform expert payload sizes")
    return {"model": str(root), "layers": layers, "experts_per_layer": count,
            "top_k": config["num_experts_per_tok"], "expert_count": len(experts),
            "expert_payload_bytes": sizes.pop(), "header_bytes_read": header_bytes,
            "payload_bytes_read": 0, "header_sha256": header_hashes,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest()}


def cache_capacity(*, entries, cache_bytes, scratch_bytes, expert_bytes):
    if any(type(v) is not int or v <= 0 for v in (entries, cache_bytes, expert_bytes)):
        raise ValueError("positive integer cache capacities required")
    if type(scratch_bytes) is not int or not 0 <= scratch_bytes < cache_bytes:
        raise ValueError("cache must exceed reserved scratch")
    byte_bound = (cache_bytes - scratch_bytes) // expert_bytes
    return {"entries": entries, "cache_bytes": cache_bytes, "reserved_scratch_bytes": scratch_bytes,
            "byte_only_expert_upper_bound": byte_bound,
            "resident_expert_upper_bound": min(entries, byte_bound),
            "bound_excludes_operation_scratch_and_pending_consumers": True}


def counterfactuals(metadata, baseline_required_device_bytes):
    """Fixed geometry/scratch/workspace; affine TOTAL bound, not workspace validation.

    hybrid_memory_plan.rs charges cache_bytes - scratch in weight_bytes. The
    serving planner separately charges the fixed workspace grant. The workspace
    *demand* also changes with weight slack and still needs factory validation.
    """
    if type(baseline_required_device_bytes) is not int or baseline_required_device_bytes <= 0:
        raise ValueError("positive baseline header-admission total required")
    specs = [("cache320-slots8", 8, 320 * MIB, 7168),
             ("cache320-slots256", 256, 320 * MIB, 7168),
             ("cache2048-slots512-G10", 512, 2 * GIB, 10240)]
    cases = []
    for name, entries, cache, budget_mb in specs:
        capacity = cache_capacity(entries=entries, cache_bytes=cache, scratch_bytes=8 * MIB,
                                  expert_bytes=metadata["expert_payload_bytes"])
        delta = cache - 320 * MIB
        required = baseline_required_device_bytes + delta
        cases.append({"name": name, **capacity,
                      "flag_diff_from_M7": {k: str(v) for k, v in {
                          "--cuda-expert-device-entries": entries,
                          "--cuda-expert-device-bytes": cache,
                          "--cuda-device-budget-mb": budget_mb,
                      }.items() if str(v) != {
                          "--cuda-expert-device-entries": "8",
                          "--cuda-expert-device-bytes": str(320 * MIB),
                          "--cuda-device-budget-mb": "7168",
                      }[k]},
                      "derived_required_device_bytes": required,
                      "configured_device_budget_bytes": budget_mb * MIB,
                      "derived_total_headroom_bytes": budget_mb * MIB - required,
                      "workspace_demand_extra_weight_slack_upper_bound_bytes": (delta + 9) // 10,
                      "factory_metadata_admission": "not_run",
                      "live_device_admission": "unknown",
                      "quality": "unverified_for_candidate"})
    return {"schema": "ferrule.paper.cache-counterfactuals.v1", "metadata": metadata,
            "single_row_layer_expert_visits": metadata["layers"] * metadata["top_k"],
            "two_row_unique_layer_expert_upper_bound": metadata["layers"] * min(
                metadata["experts_per_layer"], 2 * metadata["top_k"]),
            "full_catalog_layer_experts": metadata["layers"] * metadata["experts_per_layer"],
            "requested_generation_tokens": 8, "cases": cases,
            "launch_authorized": False, "protocol_modified": False,
            "constraints": [
                "Same Mixed CUDA profile, full 40 layers, resident KV and lazy experts; no CPU fallback.",
                "First pair varies ONLY the entry cap at fixed 320 MiB and 7 GiB admission ceiling.",
                "256 is an entry ceiling, not 256 realizable residents: bytes still cap capacity.",
                "2 GiB is a larger-budget counterfactual, not same-budget acceleration evidence.",
                "For a G10-budget paired control, explicitly use 10240 MiB admission in BOTH arms.",
                "Keep scratch=8 MiB, workspace=2048 MiB, context=128, batch/chunk=8, sequences=2, KV=16 MiB.",
                "Derived totals include the baseline margin; not CUDA physical-memory reservations.",
                "Revalidate public factory metadata/workspace, binary/source pins and same-budget quality before launch.",
                "No routing reuse, cache miss count, proof count, or speedup is inferred from these bounds.",
            ]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--baseline-required-device-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(counterfactuals(expert_headers(args.model), args.baseline_required_device_bytes),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
