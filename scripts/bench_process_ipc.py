#!/usr/bin/env python3
"""PR26 CPU baseline runner; prebuilt test + real process_rank_child, no Cargo/CI changes.

Like bench_parallel_nccl.py, run supplied executables with fixed inputs, warmup,
iterations and bounded subprocesses. This measures transport/fixture costs, not
production-model throughput. All six off/on experiments share a 100s deadline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PINNED = {
    "FERRULE_NO_CUDA": "1", "CUDA_VISIBLE_DEVICES": "", "FERRULE_LOG": "off",
    "RUST_LOG": "off", "RAYON_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false", "RUST_TEST_THREADS": "1",
}
TESTS = {
    "echo": "cpu_echo_baseline",
    "activation": "cpu_activation_baseline",
    "codec": "cpu_activation_codec_baseline",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate_counters(records: list[dict], report: dict, name: str) -> None:
    if name == "codec":
        assert not records, "isolated codec experiment must not claim IPC counters"
        return
    parent = report["pid"]
    children = ([report["result"]["child_pid"]] if name == "echo"
                else report["result"]["child_pids"])
    assert {parent, *children} <= {r["pid"] for r in records}, "missing parent/child counters"
    for record in records:
        assert record["schema"] == "ferrule.ipc-timing.v1"
        phases = record["phases"]
        for counter in phases.values():
            assert counter["calls"] == counter["completed"], "failed phase in successful baseline"
        assert phases["copy"]["completed_bytes"] == phases["encode"]["completed_bytes"]
        assert phases["copy"]["wall_ns"] <= phases["encode"]["wall_ns"]
        # Poll wait is contained in frame wall, not an additional latency bucket.
        assert phases["pipe_wait"]["wall_ns"] <= (
            phases["frame_read"]["wall_ns"] + phases["frame_write"]["wall_ns"])
    if name == "echo":
        # One Boot/Ready, 4*(4 warmup+32 measured) Execute/Complete, Shutdown/ACK.
        assert len(records) == 2
        expected = 2 + 4 * (4 + 32)
        for record in records:
            assert record["phases"]["frame_write"]["completed"] == expected
            assert record["phases"]["frame_read"]["completed"] == expected
        by_pid = {record["pid"]: record["phases"] for record in records}
        for sender, receiver in [(parent, children[0]), (children[0], parent)]:
            assert by_pid[sender]["frame_write"]["completed_bytes"] == by_pid[receiver]["frame_read"]["completed_bytes"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    if not __debug__:
        parser.error("Python -O disables baseline validation; run without optimization")
    parser.add_argument("--test-binary", type=Path, required=True, help="Cargo's process_ipc_measurement executable")
    parser.add_argument("--child", type=Path, required=True, help="same instrumented build process_rank_child example")
    parser.add_argument("--default-test-binary", type=Path, help="optional normal-build comparison (no compile-time instrumentation)")
    parser.add_argument("--default-child", type=Path, help="matching normal-build child")
    parser.add_argument("--output", type=Path, default=ROOT / "target/validation/pr26" / str(time.time_ns()),
                        help="new artifact directory under target/; existing directories refused")
    args = parser.parse_args()
    test, child, output = args.test_binary.resolve(), args.child.resolve(), args.output.resolve()
    if bool(args.default_test_binary) != bool(args.default_child):
        parser.error("default test and child must be supplied together")
    binaries = [test, child]
    if args.default_test_binary:
        binaries += [args.default_test_binary.resolve(), args.default_child.resolve()]
    for binary in binaries:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f"not an executable: {binary}")
    if ROOT / "target" not in output.parents:
        parser.error("artifacts must be under the ignored repository target/ directory")
    output.mkdir(parents=True, exist_ok=False)
    env = {key: value for key, value in os.environ.items() if not key.startswith("FERRULE_")}
    env.update(PINNED)
    env["FERRULE_PR26_CHILD"] = str(child)
    metadata = {
        "schema": "ferrule.pr26.run.v1", "status": "UNVERIFIED",
        "scope": "CPU synthetic fixture/pipe baseline; not production throughput",
        "timeout_seconds": 100, "environment": PINNED, "platform": platform.platform(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cpu_isolated": False, "warmup": 4, "iterations": 32,
        "test_binary": str(test), "test_sha256": sha256(test),
        "child_binary": str(child), "child_sha256": sha256(child),
        "default_binaries": {str(b): sha256(b) for b in binaries[2:]},
        "order": (["default"] if args.default_test_binary else []) + ["off", "on"],
        "revision": subprocess.check_output(["git", "--no-pager", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_status": subprocess.check_output(["git", "--no-pager", "--no-optional-locks", "status", "--short"], cwd=ROOT, text=True),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "download": {"status": "not_applicable_cpu", "wall_ns": None},
        "non_additive": [
            "pipe_wait is contained in frame_read/frame_write; not pure owner execution",
            "parent receive wall overlaps child execution/encoding/write",
            "execute_call includes codec/pipe/child work and is inside forward_wall",
            "IPC counters include boot, warmup, correctness oracle, release and shutdown",
            "codec/copy experiment is separate, not part of real roundtrip",
            "encode includes bounded output-validation reserialization, not just sends",
            "copy is LimitedBuffer append/allocation inside encode; never add both",
            "instrumented copy clocks every serializer chunk: on/off reports perturbation, not optimization gains",
        ],
        "source_sha256": {path: sha256(ROOT / path) for path in [
            "Cargo.lock", "crates/ferrule-runtime/src/parallel/process/ipc.rs",
            "crates/ferrule-runtime/src/parallel/process/decoder_wire.rs",
            "crates/ferrule-runtime/tests/process_ipc_measurement.rs",
            "crates/ferrule-runtime/tests/support/pr26_process_fixture.rs",
            "crates/ferrule-runtime/tests/support/process_rank_child.rs",
        ]},
        "runs": [],
    }
    manifest = output / "manifest.json"
    write_json(manifest, metadata)
    deadline = time.monotonic() + 100
    try:
        for mode in metadata["order"]:
            selected_test = args.default_test_binary.resolve() if mode == "default" else test
            env["FERRULE_PR26_CHILD"] = str(args.default_child.resolve() if mode == "default" else child)
            directory = output / mode
            directory.mkdir()
            env["FERRULE_PR26_ARTIFACT_DIR"] = str(directory)
            for name, test_name in TESTS.items():
                timing = directory / f"{name}-ipc.jsonl"
                if mode in ["default", "on"]:
                    # A sink on a normal binary must still produce no records.
                    env["FERRULE_PROCESS_IPC_TIMING"] = str(timing)
                else:
                    env.pop("FERRULE_PROCESS_IPC_TIMING", None)
                command = [str(selected_test), "--exact", test_name, "--ignored", "--nocapture", "--test-threads=1"]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("100s experiment deadline")
                start = time.monotonic()
                with (directory / f"{name}.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=remaining)
                report = json.loads((directory / f"{name}.json").read_text())
                assert (report["warmup"], report["iterations"]) == (4, 32)
                assert report["instrumentation_compiled"] == (mode != "default"), "wrong build instrumentation option"
                assert not report["cuda_feature"], "CPU baseline requires CPU-only build"
                records = [json.loads(line) for line in timing.read_text().splitlines()] if timing.exists() else []
                if mode == "on":
                    validate_counters(records, report, name)
                else:
                    assert not records, "default timing must not emit counters"
                write_json(directory / f"{name}-ipc.json", records)
                metadata["runs"].append({"mode": mode, "name": name, "command": command,
                                         "elapsed_seconds": time.monotonic()-start})
        metadata["status"] = "VERIFIED_CPU_FIXTURE_ONLY"
    except (subprocess.SubprocessError, TimeoutError, AssertionError, OSError, ValueError) as error:
        metadata["status"] = "FAILED_OR_TIMED_OUT"
        metadata["error"] = repr(error)
        write_json(manifest, metadata)
        print(f"FAILED: {error}; evidence: {output}")
        return 1
    write_json(manifest, metadata)
    print(f"VERIFIED CPU fixture only: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
