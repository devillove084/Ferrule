#!/opt/conda310/bin/python
"""External Ferrule reference/performance tool, never called by Ferrule.

This does not replace production HostCollective. With --ferrule, run the supplied
binary's bench-parallel command first, then Python NCCL, using one JSON fixture.
No shell, generated harness, Rust build, or concurrent backend execution is used.
Without --ferrule this remains a standalone benchmark. Launch directly with
Python, not torchrun; one spawned process owns each selected logical CUDA device.

Weights [out, in] and inputs [rows, in] are row-major float32 without bias.
Balanced ragged shards match TensorParallelLinearPlan. DP has no tensor
collectives or distributed process group. TP column pads before all_gather and
unpacks by rank; TP row uses all_reduce(SUM), without intermediate host staging.
Two separate measurement passes report host-to-host wall time and device-resident
CUDA-event time. Each pass has its own --warmup and --iterations rounds.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

SEED = 0
ATOL = 1e-5
RTOL = 1e-5


def positive_int(value: str) -> int:
    """Parse a strictly positive integer."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative_int(value: str) -> int:
    """Parse a nonnegative integer."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_float(value: str) -> float:
    """Parse a finite, strictly positive duration."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def parse_args() -> argparse.Namespace:
    """Keep --help independent of torch imports and CUDA availability."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("dp", "tp-column", "tp-row"), required=True)
    parser.add_argument("--world-size", type=positive_int, required=True)
    parser.add_argument("--rows", type=positive_int, default=128, help="rows per batch, independently per DP rank")
    parser.add_argument("--in-features", type=positive_int, default=1024)
    parser.add_argument("--out-features", type=positive_int, default=1024)
    parser.add_argument("--warmup", type=nonnegative_int, default=10, help="warmup rounds per timing scope")
    parser.add_argument("--iterations", type=positive_int, default=100, help="measured rounds per timing scope")
    parser.add_argument("--devices", help="comma-separated logical CUDA ordinals; respects CUDA_VISIBLE_DEVICES")
    parser.add_argument("--output", default="-", metavar="JSON", help="JSON path (parents created), or '-' for stdout")
    parser.add_argument("--ferrule", type=Path, metavar="BINARY", help="optional production binary with bench-parallel support")
    parser.add_argument("--timeout-seconds", type=positive_float, default=300.0,
                        help="deadline for each backend, including setup; also bounds barriers (default: 300)")
    parser.add_argument("--emit-values", action="store_true", help="include final row-major F32 outputs (automatic with --ferrule)")
    args = parser.parse_args()
    if args.world_size > 2**32 - 1:
        parser.error("--world-size must fit u32")
    dimension = args.out_features if args.mode == "tp-column" else args.in_features
    if args.mode != "dp" and args.world_size > dimension:
        parser.error("TP world size must not exceed the split dimension")
    try:
        args.devices = (list(range(args.world_size)) if args.devices is None else
                        [int(part) for part in args.devices.split(",")])
    except ValueError:
        parser.error("--devices must be comma-separated integer ordinals")
    if (len(args.devices) != args.world_size or len(set(args.devices)) != args.world_size
            or any(device < 0 for device in args.devices)):
        parser.error("--devices must contain exactly world-size distinct nonnegative ordinals")
    return args


def shape_of(args: argparse.Namespace) -> dict[str, int]:
    return {"rows": args.rows, "in_features": args.in_features, "out_features": args.out_features}


def balanced_ranges(dimension: int, world_size: int) -> list[tuple[int, int]]:
    """Assign one extra feature to each of the first remainder ranks."""
    base, remainder = divmod(dimension, world_size)
    return [(rank * base + min(rank, remainder),
             rank * base + min(rank, remainder) + base + (rank < remainder))
            for rank in range(world_size)]


def finite_number(value: Any, label: str) -> float:
    """Reject booleans, strings, NaN and infinity in numeric JSON fields."""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")
    return float(value)


def strict_json(text: str) -> Any:
    """Reject nonstandard constants and ambiguous duplicate object keys."""
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Publish only complete, finite JSON via a same-filesystem rename."""
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def make_fixture(args: argparse.Namespace) -> dict[str, Any]:
    """Convert to torch CPU F32 before JSON serialization, once for both backends."""
    import torch

    weight = ((torch.arange(args.out_features * args.in_features, dtype=torch.int64)
               % 17 - 8).to(torch.float32) * 0.125)
    inputs = ((torch.arange(args.rows * args.in_features, dtype=torch.int64)
               % 11 - 5).to(torch.float32) * 0.25)
    return {**shape_of(args), "weight": weight.tolist(), "input": inputs.tolist()}


def f32_bytes(values: Any, count: int, label: str) -> bytes:
    """Validate finite JSON numbers and encode IEEE F32 without importing torch."""
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{label}: expected {count} flattened float32 values")
    encoded = bytearray()
    for index, value in enumerate(values):
        number = finite_number(value, f"{label}[{index}]")
        try:
            encoded.extend(struct.pack("<f", number))
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{label}[{index}]: value overflows float32") from error
    return bytes(encoded)


def tp_consistency(outputs: list[dict[str, Any]], args: argparse.Namespace,
                   label: str) -> dict[str, Any]:
    """Require exact F32 bytes across TP ranks, independently of oracle tolerance."""
    if args.mode == "dp":
        return {"applicable": False, "reason": "DP batches are independent"}
    if len(outputs) != args.world_size:
        raise ValueError(f"{label}: TP output rank count mismatch")
    count = args.rows * args.out_features
    encoded: dict[int, bytes] = {}
    for output in outputs:
        rank = output.get("rank")
        if type(rank) is not int or rank not in range(args.world_size) or rank in encoded:
            raise ValueError(f"{label}: invalid/duplicate TP output rank")
        encoded[rank] = f32_bytes(output.get("values"), count, f"{label} rank {rank}")
    reference = encoded[0]
    for rank in range(1, args.world_size):
        if encoded[rank] != reference:
            index = next(index for index in range(count)
                         if encoded[rank][index * 4:(index + 1) * 4] != reference[index * 4:(index + 1) * 4])
            raise ValueError(f"{label}: TP F32 exact mismatch: rank {rank} vs rank 0, flat index {index}")
    return {"applicable": True, "passed": True, "comparison": "IEEE float32 little-endian bytes (including signed zero)",
            "reference_rank": 0, "ranks_checked": args.world_size, "elements_per_rank": count}


def f32_values(values: Any, count: int, label: str) -> Tensor:
    """Decode a validated JSON F32 array into a CPU tensor."""
    import torch

    f32_bytes(values, count, label)
    return torch.tensor(values, dtype=torch.float32, device="cpu")


def read_fixture(path: Path, args: argparse.Namespace) -> tuple[Tensor, Tensor, Tensor]:
    """Load the actual shared fixture; never regenerate values inside a rank."""
    import torch

    fixture = strict_json(path.read_text(encoding="utf-8"))
    for key, expected in shape_of(args).items():
        if type(fixture.get(key)) is not int or fixture[key] != expected:
            raise ValueError(f"fixture {key} mismatch")
    weight = f32_values(fixture["weight"], args.out_features * args.in_features,
                        "fixture.weight").reshape(args.out_features, args.in_features)
    inputs = f32_values(fixture["input"], args.rows * args.in_features,
                        "fixture.input").reshape(args.rows, args.in_features)
    reference = inputs.to(torch.float64) @ weight.to(torch.float64).T
    if not torch.isfinite(reference).all().item():
        raise ValueError("non-finite CPU float64 oracle")
    return weight, inputs, reference


def compare_tensors(actual: Tensor, reference: Tensor, label: str) -> dict[str, Any]:
    """Check every element, not checksums; all verification is outside timing."""
    import torch

    actual = actual.to(device="cpu", dtype=torch.float64)
    reference = reference.to(device="cpu", dtype=torch.float64)
    if actual.shape != reference.shape:
        raise ValueError(f"{label}: output shape mismatch")
    if not torch.isfinite(actual).all().item() or not torch.isfinite(reference).all().item():
        raise ValueError(f"{label}: non-finite output/reference")
    absolute = (actual - reference).abs()
    failed = absolute > ATOL + RTOL * reference.abs()
    result = {"passed": not failed.any().item(), "elements": actual.numel(),
              "max_absolute_error": absolute.max().item(),
              "max_relative_error": (absolute / reference.abs().clamp_min(ATOL)).max().item(),
              "reference_checksum": reference.sum().item(), "output_checksum": actual.sum().item()}
    if not result["passed"]:
        index = failed.flatten().nonzero()[0].item()
        raise ValueError(f"{label}: mismatch at flat index {index}: "
                         f"actual={actual.flatten()[index].item()}, reference={reference.flatten()[index].item()}; {result}")
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_summary(samples: list[float], scope: str) -> dict[str, Any]:
    """Python percentiles use linear interpolation at (n - 1) * p."""
    if not samples or any(finite_number(value, "timing") <= 0 for value in samples):
        raise ValueError("timing samples must be positive")
    return {"scope": scope, "samples_seconds": samples,
            "mean_seconds": statistics.fmean(samples), "p50_seconds": percentile(samples, 0.5),
            "p95_seconds": percentile(samples, 0.95), "max_seconds": max(samples),
            "total_seconds": sum(samples)}


def validate_ferrule(report: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Validate the agreed bench-parallel v1 protocol before GPU ranks start.

    The protocol's values are implicitly F32; it has no required dtype field.
    If a producer also supplies dtype metadata, contradictory metadata is fatal.
    """
    if not isinstance(report, dict):
        raise ValueError("Ferrule report must be a JSON object")
    expected = {"schema_version": 1, "implementation": "ferrule", "transport": "host-staged",
                "mode": args.mode, "ranks": args.world_size,
                "warmup": args.warmup, "iterations": args.iterations}
    for key, value in expected.items():
        if type(report.get(key)) is not type(value) or report[key] != value:
            raise ValueError(f"Ferrule {key} mismatch: expected {value!r}, got {report.get(key)!r}")
    shape = report.get("shape")
    if not isinstance(shape, dict):
        raise ValueError("Ferrule shape must be an object")
    for key, value in shape_of(args).items():
        if type(shape.get(key)) is not int or shape[key] != value:
            raise ValueError(f"Ferrule shape.{key} mismatch")
    for metadata in (report, shape):
        for key in ("dtype", "input_dtype", "weight_dtype", "output_dtype"):
            if key in metadata and metadata[key] not in ("f32", "float32"):
                raise ValueError(f"Ferrule {key} mismatch: expected float32")
    if "devices" in report and report["devices"] != args.devices:
        raise ValueError("Ferrule devices mismatch")
    timing = report.get("timing")
    if not isinstance(timing, dict) or timing.get("scope") != "host_to_host":
        raise ValueError("Ferrule timing.scope must be host_to_host")
    samples = timing.get("samples_seconds")
    if not isinstance(samples, list) or len(samples) != args.iterations:
        raise ValueError("Ferrule timing sample count mismatch")
    timing_summary(samples, "host_to_host")
    for key in ("mean_seconds", "p50_seconds", "p95_seconds", "total_seconds"):
        if finite_number(timing.get(key), f"Ferrule timing.{key}") <= 0:
            raise ValueError(f"Ferrule timing.{key} must be positive")
    # Ferrule owns its percentile convention; do not silently recompute its p50/p95.
    if not math.isclose(timing["total_seconds"], sum(samples), rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError("Ferrule total_seconds inconsistent with samples")
    if not math.isclose(timing["mean_seconds"], statistics.fmean(samples), rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError("Ferrule mean_seconds inconsistent with samples")
    outputs = report.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != args.world_size:
        raise ValueError("Ferrule output rank count mismatch")
    seen: set[int] = set()
    for output in outputs:
        if not isinstance(output, dict) or type(output.get("rank")) is not int:
            raise ValueError("Ferrule outputs require integer rank IDs")
        rank = output["rank"]
        if rank in seen or rank not in range(args.world_size):
            raise ValueError("Ferrule duplicate/out-of-range output rank")
        seen.add(rank)
        if "dtype" in output and output["dtype"] not in ("f32", "float32"):
            raise ValueError("Ferrule output dtype mismatch")
        f32_bytes(output.get("values"), args.rows * args.out_features, f"Ferrule rank {rank}")
    report["tp_consistency"] = tp_consistency(outputs, args, "Ferrule host-to-host")
    return report


def run_ferrule(args: argparse.Namespace, fixture: Path) -> dict[str, Any]:
    """Run the real CLI synchronously; subprocess.run kills/reaps on timeout."""
    binary = args.ferrule.expanduser().resolve(strict=True)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"Ferrule binary is not executable: {binary}")
    command = [str(binary), "bench-parallel", "--mode", args.mode,
               "--ranks", str(args.world_size), "--fixture", str(fixture),
               "--warmup", str(args.warmup), "--iterations", str(args.iterations),
               "--json", "--emit-values", "--devices", ",".join(map(str, args.devices))]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True,
                                   timeout=args.timeout_seconds, shell=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Ferrule exceeded {args.timeout_seconds}s; child killed and reaped") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"Ferrule exited {error.returncode}: {error.stderr}") from error
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="" if completed.stderr.endswith("\n") else "\n")
    return validate_ferrule(strict_json(completed.stdout), args)


def make_operation(args: argparse.Namespace, rank: int, device: Any,
                   weight: Tensor, inputs: Tensor) -> tuple[Callable[[], None], Callable[[], Tensor], Callable[[], Tensor]]:
    """Prepare resident weights/buffers; return upload, compute, download steps.

    The host-to-host upload starts from the full CPU input, so row input slicing
    and contiguous packing are included in each wall-timed round. Weights and
    preallocated (pageable) host/device output buffers are outside timing.
    """
    import torch
    import torch.distributed as dist

    dimension = args.out_features if args.mode == "tp-column" else args.in_features
    ranges = balanced_ranges(dimension, args.world_size) if args.mode != "dp" else []
    first, last = ranges[rank] if ranges else (0, args.in_features)
    if args.mode == "tp-row":
        local_weight = weight[:, first:last].contiguous().to(device)
        local_input = torch.empty((args.rows, last - first), dtype=torch.float32, device=device)
    else:
        local_weight = (weight[first:last] if args.mode == "tp-column" else weight).contiguous().to(device)
        local_input = torch.empty_like(inputs, device=device)
    output = torch.empty((args.rows, args.out_features), dtype=torch.float32, device=device)
    host_output = torch.empty(output.shape, dtype=torch.float32, device="cpu")
    if args.mode == "tp-column":
        width = (args.out_features + args.world_size - 1) // args.world_size
        local_output = torch.empty((args.rows, last - first), dtype=torch.float32, device=device)
        packed = torch.empty((args.rows, width), dtype=torch.float32, device=device)
        gathered = [torch.empty_like(packed) for _ in range(args.world_size)]

    def upload() -> None:
        source = inputs[:, first:last].contiguous() if args.mode == "tp-row" else inputs
        local_input.copy_(source, non_blocking=False)

    def compute() -> Tensor:
        if args.mode == "tp-column":
            torch.mm(local_input, local_weight.T, out=local_output)
            packed.zero_()
            packed[:, :last - first].copy_(local_output)
            dist.all_gather(gathered, packed)
            for block, (start, end) in zip(gathered, ranges):
                output[:, start:end].copy_(block[:, :end - start])
        else:
            torch.mm(local_input, local_weight.T, out=output)
            if args.mode == "tp-row":
                dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

    def download() -> Tensor:
        host_output.copy_(output, non_blocking=False)
        torch.cuda.synchronize(device)
        return host_output

    return upload, compute, download


def measure(args: argparse.Namespace, device: Any, align: Callable[[], None],
            operation: tuple[Callable[[], None], Callable[[], Tensor], Callable[[], Tensor]],
            scope: str) -> tuple[list[float], Tensor]:
    """Use separate passes so CUDA-event instrumentation is not in wall timing."""
    import torch

    upload, compute, download = operation
    resident = scope == "device_resident"
    start = torch.cuda.Event(enable_timing=True) if resident else None
    end = torch.cuda.Event(enable_timing=True) if resident else None
    if resident:
        upload()
        start.record()
        end.record()
    align()
    samples: list[float] = []
    for iteration in range(args.warmup + args.iterations):
        align()  # Includes a device synchronize, outside either timing scope.
        if resident:
            start.record()
            output = compute()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) / 1000.0
        else:
            before = time.perf_counter()
            upload()
            compute()
            output = download()
            elapsed = time.perf_counter() - before
        if not torch.isfinite(output).all().item():
            raise ValueError(f"{scope}: non-finite output in round {iteration}")
        if iteration >= args.warmup:
            samples.append(elapsed)
    # Clone on CPU before the next pass can reuse any buffers; not timed.
    return samples, download().clone()


def worker(rank: int, args: argparse.Namespace, directory: str, barrier: Any) -> None:
    """Own one CUDA device; report through atomic files, never tensor IPC."""
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.cuda.set_device(args.devices[rank])
    torch.cuda.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", args.devices[rank])
    root = Path(directory)
    try:
        if args.mode != "dp":
            dist.init_process_group("nccl", init_method=(root / "nccl_store").as_uri(),
                                    rank=rank, world_size=args.world_size,
                                    timeout=timedelta(seconds=args.timeout_seconds))

        def align() -> None:
            if args.mode == "dp":
                barrier.wait(timeout=args.timeout_seconds)
            else:
                dist.barrier(device_ids=[args.devices[rank]])
            torch.cuda.synchronize(device)

        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            weight, inputs, reference = read_fixture(root / "fixture.json", args)
            operation = make_operation(args, rank, device, weight, inputs)
            wall, host_output = measure(args, device, align, operation, "host_to_host")
            verification = compare_tensors(host_output, reference, f"Python rank {rank} host-to-host oracle")
            events, resident_output = measure(args, device, align, operation, "device_resident")
            resident_verification = compare_tensors(resident_output, reference, f"Python rank {rank} resident oracle")
        nccl_version = torch.cuda.nccl.version()
        report = {"rank": rank, "device": args.devices[rank],
                  "device_name": torch.cuda.get_device_name(device),
                  "timing": timing_summary(wall, "host_to_host"),
                  "device_resident_timing": timing_summary(events, "device_resident"),
                  "verification": verification, "device_resident_verification": resident_verification,
                  "versions": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                               "nccl": list(nccl_version) if isinstance(nccl_version, tuple) else nccl_version}}
        if args.mode != "dp" or args.ferrule is not None or args.emit_values:
            report["values"] = host_output.flatten().tolist()
        if args.mode != "dp":
            report["resident_values"] = resident_output.flatten().tolist()
        atomic_json(root / f"rank-{rank}.json", report)
        barrier.wait(timeout=args.timeout_seconds)
        if rank == 0:
            reports = [strict_json((root / f"rank-{index}.json").read_text(encoding="utf-8"))
                       for index in range(args.world_size)]
            atomic_json(root / "python.json", build_report(args, reports))
        barrier.wait(timeout=args.timeout_seconds)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def throughput(args: argparse.Namespace, timing: dict[str, Any]) -> dict[str, Any]:
    """A DP round contains world-size batches, not world-size * rows requests."""
    batches = args.world_size if args.mode == "dp" else 1
    return {"scope": timing["scope"], "batches_per_round": batches, "rows_per_batch": args.rows,
            "rows_per_round": batches * args.rows, "total_batches": batches * args.iterations,
            "total_rows": batches * args.rows * args.iterations,
            "batches_per_second": batches * args.iterations / timing["total_seconds"],
            "rows_per_second": batches * args.rows * args.iterations / timing["total_seconds"]}


def build_report(args: argparse.Namespace, ranks: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-iteration rank maxima, excluding parent/coordinator IPC."""
    if [entry["rank"] for entry in ranks] != list(range(args.world_size)):
        raise ValueError("incomplete/out-of-order Python rank reports")
    timings: dict[str, Any] = {}
    for key, scope in (("timing", "host_to_host"), ("device_resident_timing", "device_resident")):
        if any(len(entry[key]["samples_seconds"]) != args.iterations for entry in ranks):
            raise ValueError("Python timing sample count mismatch")
        samples = [max(entry[key]["samples_seconds"][i] for entry in ranks) for i in range(args.iterations)]
        timings[key] = timing_summary(samples, scope)
        timings[key]["aggregation"] = "per-iteration maximum across rank-local times; then statistics"
    timings["timing"].update({
        "clock": "time.perf_counter wall seconds",
        "includes": ["full host input -> local input packing/H2D", "local matmul",
                     "TP NCCL and column pack/unpack (none for DP)", "full output D2H and device synchronize"],
        "excludes": ["weights setup", "preallocation", "fixture/oracle", "barriers", "verification",
                     "rank/parent IPC", "report serialization", "process/group startup and teardown"],
        "host_memory": "preallocated pageable CPU input/output; row input packing allocated per round",
    })
    timings["device_resident_timing"].update({
        "clock": "CUDA events on current stream, end event synchronized",
        "includes": ["local matmul", "TP NCCL and column pack/unpack (none for DP)"],
        "excludes": ["H2D/D2H", "setup", "barriers", "oracle/verification", "report IPC/serialization"],
        "separate_measurement_pass": True,
    })
    dimension = args.out_features if args.mode == "tp-column" else args.in_features
    outputs = [{"rank": rank["rank"], "values": rank.pop("values")} for rank in ranks if "values" in rank]
    consistency = tp_consistency(outputs, args, "Python host-to-host")
    if args.mode != "dp":
        resident_outputs = [{"rank": rank["rank"], "values": rank.pop("resident_values")}
                            for rank in ranks if "resident_values" in rank]
        consistency["device_resident"] = tp_consistency(resident_outputs, args, "Python device-resident")
    report = {
        "schema_version": 1, "implementation": "python-torch-nccl", "transport": "nccl",
        "transport_used": args.mode != "dp", "mode": args.mode, "world_size": args.world_size,
        "report_rank": 0, "devices": args.devices, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "role": "external reference/perf tool; not called by Ferrule; not production HostCollective",
        "shape": {**shape_of(args), "dtype": "float32", "weight_layout": "row-major [out, in]",
                  "input_layout": "row-major [rows, in]", "bias": False},
        "partition": {"kind": "replicated" if args.mode == "dp" else "balanced-ragged",
                      "axis": {"dp": None, "tp-column": "out_features", "tp-row": "in_features"}[args.mode],
                      "ranges": balanced_ranges(dimension, args.world_size) if args.mode != "dp" else None,
                      "range_convention": "[start, end); remainder goes to lowest ranks",
                      "column_padded_width": ((args.out_features + args.world_size - 1) // args.world_size
                                              if args.mode == "tp-column" else None),
                      "collective": {"dp": "none", "tp-column": "all_gather", "tp-row": "all_reduce_sum"}[args.mode]},
        "warmup": args.warmup, "iterations": args.iterations, "round_counts_apply_per_timing_scope": True,
        "fixture": {"seed": SEED, "random": False, "source": "same serialized CPU float32 fixture on all ranks/backends",
                    "weight": "(flat_index % 17 - 8) * 0.125", "input": "(flat_index % 11 - 5) * 0.25",
                    "tf32": False, "autocast": False},
        **timings, "throughput": throughput(args, timings["timing"]),
        "device_resident_throughput": throughput(args, timings["device_resident_timing"]),
        "percentile_method": "Python: linear interpolation at (n - 1) * p",
        "verification": {"passed": all(rank["verification"]["passed"] for rank in ranks),
                         "reference": "CPU float64 full input @ weight.T", "atol": ATOL, "rtol": RTOL,
                         "relative_error_denominator": "max(abs(reference), atol)",
                         "max_absolute_error": max(rank["verification"]["max_absolute_error"] for rank in ranks),
                         "max_relative_error": max(rank["verification"]["max_relative_error"] for rank in ranks),
                         "reference_checksum": ranks[0]["verification"]["reference_checksum"],
                         "checksum_definition": "diagnostic float64 sum only; not used for correctness"},
        "ranks": ranks, "versions": ranks[0]["versions"], "tp_consistency": consistency,
    }
    if args.ferrule is not None or args.emit_values:
        if len(outputs) != args.world_size:
            raise ValueError("missing Python output values")
        report["outputs"] = outputs
    return report


def add_comparison(report: dict[str, Any], ferrule: dict[str, Any],
                   args: argparse.Namespace, reference: Tensor) -> None:
    """Compare both backends with the CPU oracle and with each other, by rank."""
    rust_by_rank = {output["rank"]: output["values"] for output in ferrule["outputs"]}
    comparisons = []
    for output in report["outputs"]:
        rank = output["rank"]
        actual = f32_values(output["values"], args.rows * args.out_features, f"Python rank {rank}")
        rust = f32_values(rust_by_rank[rank], args.rows * args.out_features, f"Ferrule rank {rank}")
        oracle = compare_tensors(rust, reference.flatten(), f"Ferrule rank {rank} CPU oracle")
        pair = compare_tensors(actual, rust, f"Python vs Ferrule rank {rank}")
        comparisons.append({"rank": rank, **pair, "ferrule_cpu_oracle": oracle})
    report["ferrule"] = ferrule
    report["comparison"] = {
        "passed": True, "execution_order": ["ferrule", "python-torch-nccl"],
        "same_fixture": True, "atol": ATOL, "rtol": RTOL, "ranks": comparisons,
        "tp_consistency": {"python": report["tp_consistency"], "ferrule": ferrule["tp_consistency"]},
        "max_absolute_error": max(entry["max_absolute_error"] for entry in comparisons),
        "max_relative_error": max(entry["max_relative_error"] for entry in comparisons),
        "timing": {"scope": "host_to_host", "python": report["timing"], "ferrule": ferrule["timing"],
                   "speedup_computed": False,
                   "caveat": "Same host endpoints, different transports. Python uses per-round max rank wall time, "
                             "excluding parent/coordinator IPC; Ferrule coordinator costs are defined by the binary. "
                             "Side-by-side measurements, not an asserted equivalent end-to-end speedup. "
                             "Device-resident CUDA-event times must not be compared to Ferrule host-to-host times."},
        "throughput": {"python": report["throughput"], "ferrule": throughput(args, ferrule["timing"])},
    }


def run_python(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    """Bound the entire Python backend, terminating/reaping ranks on any failure."""
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    barrier = context.Barrier(args.world_size)
    processes: list[Any] = []
    deadline = time.monotonic() + args.timeout_seconds
    try:
        for rank in range(args.world_size):
            process = context.Process(target=worker, args=(rank, args, str(root), barrier))
            process.start()
            processes.append(process)
        while True:
            for rank, process in enumerate(processes):
                if process.exitcode not in (None, 0):
                    raise RuntimeError(f"Python rank {rank} exited {process.exitcode}; see stderr")
            if all(process.exitcode == 0 for process in processes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Python backend exceeded {args.timeout_seconds}s")
            time.sleep(0.05)
        return strict_json((root / "python.json").read_text(encoding="utf-8"))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            if not process.is_alive():
                process.close()


def main() -> int:
    """Run optional Rust first, then Python, and publish one finite JSON report."""
    args = parse_args()
    try:
        if "RANK" in os.environ or "LOCAL_RANK" in os.environ:
            raise ValueError("launch directly with Python, not torchrun")
        import torch
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_nccl_available():
            raise RuntimeError("requires torch.distributed with NCCL support")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        count = torch.cuda.device_count()
        if args.world_size > count or any(device >= count for device in args.devices):
            raise ValueError(f"requested devices {args.devices}, but only {count} CUDA devices are visible")
        output = None if args.output == "-" else Path(args.output).expanduser().resolve()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(prefix="ferrule-nccl-", dir=output.parent if output else None) as directory:
            root = Path(directory).resolve()
            fixture = root / "fixture.json"
            atomic_json(fixture, make_fixture(args))
            ferrule = run_ferrule(args, fixture) if args.ferrule is not None else None
            # No Python CUDA worker exists until the Rust subprocess has exited.
            report = run_python(args, root)
            if ferrule is not None:
                _, _, reference = read_fixture(fixture, args)
                add_comparison(report, ferrule, args, reference)
            if output is None:
                print(json.dumps(report, allow_nan=False))
            else:
                atomic_json(output, report)
        return 0
    except KeyboardInterrupt:
        print("bench_parallel_nccl: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"bench_parallel_nccl: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
