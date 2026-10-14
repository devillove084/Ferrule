"""CPU byte-only preparation; never invent a missing HF expert input."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
CPU = HERE / "runs/routed-down-m1-cpu-01"
CUDA = HERE / "runs/routed-down-m1-cuda-01"
REF = HERE / "runs/full40-20261009-03"
CPU_PIN = "6fe6b0c5b8d886586fef422be52f61151fc72345b4077eae33987e8fa325c65f"
CUDA_PIN = "4f3081e16cc496121817bd49e4732cb5b49a030ca9853f616751c17d23ba1704"
REF_PIN = "fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d"
STATUS = "blocked_missing_actual_normalized_expert_input"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, raw):
    with path.open("xb") as f:
        f.write(raw)


def write_json(path, value):
    write(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def record(path):
    return dict(file=path.relative_to(HERE).as_posix(), bytes=path.stat().st_size, sha256=sha(path))


def check_sums(directory, name, pin):
    require(sha(directory / name) == pin, "Checksum-list pin differs: " + str(directory))
    seen = set()
    total = 0
    for line in (directory / name).read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (directory / relative).resolve()
        require(path.is_relative_to(directory.resolve()) and relative not in seen, "Invalid checksum path")
        total += path.stat().st_size
        require(total <= 192 * 1024**2, "Source bundle byte cap")
        require(sha(path) == expected, "Frozen bytes differ: " + str(path))
        seen.add(relative)
    return seen


def header(path):
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        require(size <= 8 * 1024**2, "Safetensors header cap")
        value = json.loads(f.read(size))
    return value, 8 + size


def tensor_bytes(path, key):
    values, base = header(path)
    item = values[key]
    low, high = item["data_offsets"]
    require(0 <= low <= high and base + high <= path.stat().st_size, "Tensor bounds")
    with path.open("rb") as f:
        f.seek(base + low)
        raw = f.read(high - low)
    return raw, item


def local_rows(raw, rank, rows, columns, width):
    require(rank in (0, 1) and rows % 2 == 0, "Rank or row geometry")
    require(len(raw) == rows * columns * width, "Original extent")
    size = rows // 2 * columns * width
    return raw[rank * size:(rank + 1) * size]


def finite_bf16(raw, positive=False):
    require(len(raw) % 2 == 0, "BF16 extent")
    for (word,) in struct.iter_unpack("<H", raw):
        require(word & 0x7f80 != 0x7f80, "Nonfinite BF16")
        if positive:
            require(0 < word < 0x7f80, "Nonpositive BF16 scale")


def payload(cpu, name, rank, scale):
    source = cpu["original_payloads"][name]
    raw = (CPU / source["file"]).read_bytes()
    rows, columns, width = (4, 16, 2) if scale else (512, 2048, 1)
    require(source["original_header"]["shape"] == [rows, columns], "Original shape")
    require(source["original_header"]["dtype"] == ("BF16" if scale else "F8_E4M3"), "Original dtype")
    require(len(raw) == source["bytes"] and digest(raw) == source["sha256"], "Original payload pin")
    require(source["original_used_payload"]["sha256"] == digest(raw), "Historical used-payload pin")
    result = local_rows(raw, rank, rows, columns, width)
    if scale:
        finite_bf16(result, positive=True)
    else:
        require(all(b & 0x7f != 0x7f for b in result), "FP8 NaN")
    return result, dict(tensor=name, original=source,
        source_file=record(CPU / source["file"]),
        source_byte_range=[rank * len(result), (rank + 1) * len(result)],
        global_rows=[rank * (rows // 2), (rank + 1) * (rows // 2)],
        global_columns=[0, columns])


def m5_inventory(reference):
    case = next(c for c in reference["cases"] if c["id"] == "capital")
    path = REF / case["file"]
    require(path.stat().st_size == case["bytes"] and sha(path) == case["sha256"], "Capital archive pin")
    layers = []
    for layer in range(40):
        key = "prefill.rank.0.layers.%02d.selected_expert_ids" % layer
        raw, item = tensor_bytes(path, key)
        other, other_item = tensor_bytes(path, key.replace("rank.0.", "rank.1."))
        require(item["shape"] == other_item["shape"] == [5, 8] and item["dtype"] == other_item["dtype"] == "I64", "M5 route geometry")
        require(raw == other, "Rank routes differ")
        ids = struct.unpack("<40q", raw)
        rows = [list(ids[i:i + 8]) for i in range(0, 40, 8)]
        require(all(len(set(row)) == 8 for row in rows), "Duplicate expert route")
        counts = Counter(ids)
        layers.append(dict(layer=layer, selected_expert_ids=rows,
            selected_ids_sha256=digest(raw), tensor_key_rank0=key,
            active_expert_count=len(counts), all_five_experts=sorted(e for e, count in counts.items() if count == 5)))
    selected = next(row for row in layers if row["all_five_experts"])
    ids, item = tensor_bytes(path, "prefill.input_ids")
    require(item["shape"] == [1, 5] and item["dtype"] == "I64", "Actual prefill geometry")
    return dict(schema="ferrule.paper.pair-swiglu.m5-evidence.v1", status="pending_authorized_capture",
        source=record(path), input_token_ids=list(struct.unpack("<5q", ids)), route_inventory=layers,
        selection="First layer, then minimum expert ID actually routed by all five Capital prefill tokens",
        selected_layer=selected["layer"], selected_expert=selected["all_five_experts"][0],
        missing=["actual normalized BF16 [5,2048] input", "actual HF pre-down activation BF16 [5,256] per rank",
                 "CPU-frozen original layer-9 norm/router and all 24 active expert gate/up/down payloads for full control"],
        full_control_calls=dict(cuda_tasks=1, physical_devices=1, logical_ranks=2, norm=1, router=2,
            native_experts_forward=2, packed_gate_up_linear=2 * selected["active_expert_count"],
            down_linear=2 * selected["active_expert_count"]),
        candidate_executed=False, cuda_executed=False)


def extract(output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("NVIDIA_VISIBLE_DEVICES") in ("void", "none"), "Hide GPUs before invocation")
    require("torch" not in sys.modules, "Stdlib-only byte extraction required")
    output = output.resolve()
    require(output.parent == HERE / "runs" and not output.exists(), "Fresh immediate paper runs child required")
    require(sha(CUDA / "manifest.json") == CUDA_PIN and sha(REF / "manifest.json") == REF_PIN, "Oracle release pin")
    check_sums(CPU, "SHA256SUMS.cpu-extract", CPU_PIN)
    cpu, cuda, reference = load(CPU / "manifest.json"), load(CUDA / "manifest.json"), load(REF / "manifest.json")
    require(cuda["status"] == "complete" and cuda["selected_rank_rows"] == 8, "Incomplete original CUDA oracle")
    require(cuda["bundle_sha256"] == CPU_PIN and cpu["reference_manifest_sha256"] == REF_PIN, "Source chain")
    require(len(cuda["controls"]) == 40 and all(c["exact"] and c["actual_sha256"] == c["expected_sha256"] for c in cuda["controls"]), "Historical exact controls")
    require(sha(CUDA / "numerical-authentication.json") == cuda["numerical_authentication_sha256"], "Numerical authentication pin")
    require(sha(CUDA / "worker-launch.json") == cuda["launch_sha256"], "Historical launch pin")
    m5 = m5_inventory(reference)
    output.mkdir()
    write(output / "extractor-source.py", Path(__file__).read_bytes())
    write_json(output / "m5-evidence.json", m5)
    fixtures = []
    for entry, replay in zip(cpu["fixtures"], cpu["gpu_replays"]):
        name = entry["directory"]
        require(replay["fixture"] == name, "Replay identity")
        original = next(f for f in cuda["fixtures"] if f["directory"] == name)
        check_sums(CUDA / name, "SHA256SUMS", original["sha256"])
        old = load(CUDA / name / "manifest.json")
        require((old["rows"], old["n"], old["local_k"], old["layer"], old["expert"]) == (1, 2048, 256, entry["layer"], entry["expert"]), "Down geometry or identity")
        archive = CPU / name / "observed.safetensors"
        inventory, _ = header(archive)
        require(len(inventory) == 13 and not any(any(s in k for s in ("norm", "expert_input", "swiglu")) for k in inventory), "Inventory changed; reassess missing input")
        folder = output / name
        folder.mkdir()
        manifest = dict(schema="ferrule.paper.pair-swiglu.preparation.v1", status=STATUS,
            fixture_ready=False, rows=1, n=256, k=2048, packed_n=512, tp=2,
            case=entry["case"], stage=entry["stage"], layer=entry["layer"], expert=entry["expert"],
            source_down_manifest=record(CUDA / name / "manifest.json"), source_down_sums=record(CUDA / name / "SHA256SUMS"),
            source_observed_archive=record(archive), observed_tensor_inventory=inventory,
            normalized_input=dict(status="missing_not_captured", file=None, sha256=None, dtype="BF16", shape=[1, 2048],
                upstream_input_key=replay["upstream_input_key"], attention_key="decode.0.layers.%02d.attention" % entry["layer"],
                explanation="HF CUDA norm(state + attention) existed only transiently; residual inputs are not normalized expert inputs"),
            provenance=old["provenance"],
            oracle={key: old[key] for key in ("native_source", "workflow_pins", "versions", "launch_sha256", "numerical_authentication_sha256", "source_bundle_sha256", "execution", "source_audit")},
            ranks=[])
        for rank in (0, 1):
            files = {}
            for part in ("gate", "up"):
                base = "model.language_model.layers.%d.mlp.experts.%d.%s_proj.weight" % (entry["layer"], entry["expert"], part)
                for scale in (False, True):
                    raw, provenance = payload(cpu, base + ("_scale_inv" if scale else ""), rank, scale)
                    filename = "rank%d.%s.%s" % (rank, part, "scale.bf16" if scale else "weight.fp8")
                    write(folder / filename, raw)
                    files[part + ("_scale" if scale else "_weight")] = dict(file=filename, bytes=len(raw), sha256=digest(raw),
                        dtype="BF16" if scale else "F8_E4M3", shape=[2, 16] if scale else [256, 2048], source=provenance)
            source = CUDA / name / ("rank%d.input.bf16" % rank)
            raw = source.read_bytes()
            require(len(raw) == 512, "HF activation extent")
            finite_bf16(raw)
            filename = "rank%d.hf.activation.bf16" % rank
            write(folder / filename, raw)
            files["expected_activation"] = dict(file=filename, bytes=512, sha256=digest(raw), dtype="BF16", shape=[1, 256],
                source=record(source), transform="identity bytes; original HF pre-down input, not down output or weighted aggregate")
            manifest["ranks"].append(dict(rank=rank, global_intermediate_rows=[rank * 256, (rank + 1) * 256],
                global_scale_rows=[rank * 2, (rank + 1) * 2], files=files))
        write_json(folder / "manifest.json", manifest)
        sums = "".join(sha(p) + "  " + p.name + "\n" for p in sorted(folder.iterdir()))
        write(folder / "SHA256SUMS", sums.encode())
        fixtures.append(dict(directory=name, fixture_ready=False, manifest_sha256=sha(folder / "manifest.json"), sha256sums_sha256=sha(folder / "SHA256SUMS")))
    require(len(fixtures) == 4, "Four-case contract")
    check_sums(CPU, "SHA256SUMS.cpu-extract", CPU_PIN)
    require(sha(CUDA / "manifest.json") == CUDA_PIN and sha(REF / "manifest.json") == REF_PIN, "Old reference changed")
    release = dict(schema="ferrule.paper.pair-swiglu.cpu-preparation.v1", status=STATUS, fixture_ready=False,
        complete_rank_fixtures=0, prepared_weight_activation_rank_pairs=8, fixtures=fixtures,
        source_pins=[record(p) for p in (CPU / "SHA256SUMS.cpu-extract", CPU / "manifest.json", CUDA / "manifest.json",
            CUDA / "numerical-authentication.json", CUDA / "worker-launch.json", REF / "manifest.json")],
        generation_identity=dict(run=output.name, extractor_sha256=sha(output / "extractor-source.py"),
            python=sys.version, gpu_visibility={k: os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")},
            method="stdlib raw-byte verification, row slicing and HF activation identity copy only", cuda_executed=False,
            candidate_executed=False, cpu_numerical_expected_computed=False),
        acceptance="Reject fixture_ready=false. Future gate compares every BF16 word to independent HF activation; shared helper equivalence alone is insufficient.",
        m5_evidence=dict(file="m5-evidence.json", sha256=sha(output / "m5-evidence.json")))
    write_json(output / "manifest.json", release)
    sums = "".join(sha(p) + "  " + p.relative_to(output).as_posix() + "\n" for p in sorted(output.rglob("*")) if p.is_file())
    write(output / "SHA256SUMS", sums.encode())
    print(json.dumps(dict(directory=str(output), status=STATUS, manifest_sha256=sha(output / "manifest.json"), sha256sums_sha256=sha(output / "SHA256SUMS"), fixtures=fixtures), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    extract(parser.parse_args().output)
