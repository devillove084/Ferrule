#!/usr/bin/env python3
"""PR26 CPU baseline runner; prebuilt test + real process_rank_child, no Cargo/CI changes.

Like bench_parallel_nccl.py, run supplied executables with fixed inputs, warmup,
iterations and bounded subprocesses. This measures transport/fixture costs, not
production-model throughput. All six off/on experiments share a 100s deadline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sqlite3
import sys
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
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


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
        assert record["schema"] in ("ferrule.ipc-timing.v1", "ferrule.ipc-timing.v2")
        phases = record["phases"]
        for counter in phases.values():
            assert counter["calls"] == counter["completed"], "failed phase in successful baseline"
        assert phases["copy"]["completed_bytes"] == phases["encode"]["completed_bytes"]
        copy_ns = phases["copy"]["wall_ns"]
        if copy_ns is None:
            assert record["schema"] == "ferrule.ipc-timing.v2"
        else:
            assert type(copy_ns) is int and 0 <= copy_ns <= phases["encode"]["wall_ns"]
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


PHASES = ('encode', 'payload_decode', 'frame_decode', 'frame_write', 'frame_read', 'pipe_wait', 'copy')
MAX_PROFILE_BYTES = 16 * 1024 * 1024


def bounded_text(path: Path) -> str:
    with path.open('rb') as stream:
        data = stream.read(MAX_PROFILE_BYTES + 1)
    if len(data) > MAX_PROFILE_BYTES:
        raise ValueError(f'Artifact exceeds 16 MiB parser budget: {path}')
    return data.decode('utf-8')


def summarize_ipc(records: list[dict]) -> dict:
    """Lifetime per-thread counters, not disjoint request costs or CUDA copies."""
    totals = {p: dict(calls=0, completed=0, completed_bytes=0, wall_ns=0) for p in PHASES}
    identities = set()
    for record in records:
        if record.get('schema') not in ('ferrule.ipc-timing.v1', 'ferrule.ipc-timing.v2'):
            raise ValueError('Unknown IPC schema')
        pid, thread = record.get('pid'), record.get('thread')
        if type(pid) is not int or pid <= 0 or not isinstance(thread, str) or not thread:
            raise ValueError('Missing PID/thread identity')
        if (pid, thread) in identities:
            raise ValueError('Duplicate PID/thread; possible appended/reused sink')
        identities.add((pid, thread))
        phases = record.get('phases', {})
        if set(phases) != set(PHASES):
            raise ValueError('Incomplete or unknown IPC phases')
        for phase, counter in phases.items():
            if set(counter) != set(totals[phase]):
                raise ValueError('Incomplete or unknown IPC counter fields')
            for field, value in counter.items():
                if (record['schema'] == 'ferrule.ipc-timing.v2' and phase == 'copy'
                        and field == 'wall_ns' and value is None):
                    continue
                if type(value) is not int or value < 0:
                    raise ValueError('IPC counters require nonnegative integers (v2 copy ns may be null)')
            if counter['completed'] > counter['calls']:
                raise ValueError('Completed exceeds calls')
            for field, value in counter.items():
                previous = totals[phase][field]
                totals[phase][field] = None if previous is None or value is None else previous + value
        if phases['copy']['wall_ns'] is not None and phases['copy']['wall_ns'] > phases['encode']['wall_ns']:
            raise ValueError('copy must be contained in encode')
        if phases['pipe_wait']['wall_ns'] > phases['frame_read']['wall_ns'] + phases['frame_write']['wall_ns']:
            raise ValueError('pipe_wait must be contained in frame read/write')
    for counter in totals.values():
        counter['inclusive_thread_wall_seconds'] = (
            None if counter['wall_ns'] is None else counter['wall_ns'] / 1_000_000_000)
        counter['incomplete_calls'] = counter['calls'] - counter['completed']
        if not records:
            counter.update(dict.fromkeys(counter))
    return dict(status='observed_lifetime_counters' if records else 'missing_not_zero',
                pids=sorted({pid for pid, _ in identities}), threads=len(identities), phases=totals,
                scope='whole process/thread lifetime including boot, warmup, measurement and shutdown',
                rank_coverage='unverified; PID count is not proof all EP owners emitted counters',
                request_attribution=None, kernel_fraction=None,
                non_additive=['copy is serializer buffer append/allocation inside encode, NOT D2H',
                    'pipe_wait is inside frame_read/frame_write; parent receive overlaps child work',
                    'thread/process totals overlap; never sum phases or divide them by request wall',
                    'aggregate copy counts are frame-local; copy wall_ns is null unless detailed mode is explicit'])


def parse_nsys_summary(text: str) -> list[dict]:
    """Only explicit ns-unit stats CSVs; ignore Time (%) (not wall utilization)."""
    reader = csv.DictReader(text.splitlines())
    headers = reader.fieldnames or []
    if 'Total Time (ns)' not in headers or not any(k in headers for k in ('Name', 'Operation')):
        raise ValueError('Expected nsys stats CSV with Total Time (ns) and Name/Operation; no unit inference')
    rows = []
    for index, row in enumerate(reader):
        if index >= 10000:
            raise ValueError('nsys CSV exceeds 10000-row budget')
        value = row['Total Time (ns)']
        if value is None or not value.isascii() or not value.isdigit():
            raise ValueError('Expected nonnegative integer Total Time (ns)')
        name = row.get('Name') or row.get('Operation')
        if not name:
            raise ValueError('Missing kernel/API/operation name')
        count = row.get('Instances', row.get('Num Calls'))
        if count is not None and (not count.isascii() or not count.isdigit()):
            raise ValueError('Invalid nsys instance/call count')
        rows.append(dict(name=name, calls=int(count) if count is not None else None,
                         duration_sum_ns=int(value), duration_sum_seconds=int(value)/1_000_000_000,
                         wall_fraction=None))
    return rows


def read_profile_ipc(path: Path) -> dict:
    """Keep valid records after interrupted writes, but never call them complete."""
    records, errors, seen = [], [], set()
    if path.exists():
        lines = bounded_text(path).splitlines()
        if len(lines) > 10000:
            raise ValueError('IPC sink exceeds 10000-record budget')
        for number, line in enumerate(lines, 1):
            try:
                record = json.loads(line)
                summarize_ipc([record])
                identity = (record['pid'], record['thread'])
                if identity in seen:
                    raise ValueError('Duplicate PID/thread; possible appended/reused sink')
                seen.add(identity)
                records.append(record)
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                errors.append(dict(line=number, error=str(exc)))
    result = summarize_ipc(records)
    result['parse_errors'] = errors
    if errors:
        result['status'] = 'partial_invalid_not_zero'
    result['nonzero_transport_observed'] = any(
        record['phases'][phase]['completed_bytes'] > 0 and record['phases'][phase]['wall_ns'] > 0
        for record in records for phase in ('frame_read', 'frame_write'))
    return result


def sqlite_gpu_coverage(path: Path) -> dict:
    """Read-only bounded coverage evidence, not request attribution/utilization."""
    deadline = time.monotonic() + 20
    result = dict(status='missing_not_zero', tables={}, request_alignment='unverified',
                  pid_mapping='globalPid retained in nsys encoding; map to server/ranks before attribution')
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as connection:
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY'):
            if name not in tables:
                continue
            columns = {row[1] for row in connection.execute(f'PRAGMA table_info({name})')}
            if not {'start', 'end', 'deviceId', 'globalPid'} <= columns:
                raise ValueError(f'Unsupported nsys SQLite columns: {name}')
            rows = connection.execute(f'SELECT deviceId, globalPid, COUNT(*), MIN(start), MAX(end), '
                f'SUM(end-start) FROM {name} WHERE end > start GROUP BY deviceId, globalPid LIMIT 10001').fetchall()
            if len(rows) > 10000:
                raise ValueError('nsys SQLite exceeds PID/device group budget')
            result['tables'][name] = [dict(device_id=d, global_pid=p, events=n,
                first_start_ns=a, last_end_ns=b, duration_sum_ns=total, wall_fraction=None)
                for d, p, n, a, b, total in rows]
    kernels = result['tables'].get('CUPTI_ACTIVITY_KIND_KERNEL', [])
    result['kernel_devices'] = sorted({row['device_id'] for row in kernels})
    result['kernel_processes'] = sorted({row['global_pid'] for row in kernels})
    if kernels:
        result['status'] = 'gpu_events_observed_not_request_scoped'
    return result


def summarize_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description='Offline profiling artifact summary; no processes/GPU')
    parser.add_argument('--matrix', required=True, type=Path)
    parser.add_argument('--slot', type=int, default=0)
    parser.add_argument('--nsys-csv', action='append', type=Path, default=[])
    parser.add_argument('--nsys-sqlite', type=Path, help='Offline nsys SQLite export for PID/device event coverage')
    parser.add_argument('--require-artifacts', action='store_true',
                        help='Exit 1 after saving summary unless HTTP, IPC, export and GPU event smoke checks pass')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        matrix = json.loads(bounded_text(args.matrix))
        if matrix.get('schema') != 'ferrule.http_matrix.v1' or not matrix['manifest'].get('profiling'):
            raise ValueError('Expected profiling matrix report')
        if args.slot < 0:
            raise ValueError('Invalid slot')
        run = matrix['runs'][args.slot]
        directory = (args.matrix.parent / f'run-{run["index"]:05d}').resolve()
        sink = directory / 'ipc-timing.jsonl'
        ipc = read_profile_ipc(sink)
        rep = directory / 'nsys.nsys-rep'
        life_path = directory / 'lifecycle.json'
        lifecycle = json.loads(bounded_text(life_path)) if life_path.exists() else {}
        summary = dict(schema='ferrule.profiling-summary.v1', matrix_sha256=sha256(args.matrix),
            manifest_sha256=matrix['manifest_sha256'], pins=matrix['manifest']['profiling']['pins'],
            slot=run['index'], work_status=run['status'],
            custody=run.get('supervisor', {}).get('custody', {'status': 'unknown'}),
            ipc_sha256=sha256(sink) if sink.exists() else None, ipc=ipc,
            primary_work_status=lifecycle.get('work_status', run['status']),
            primary_error=lifecycle.get('error'), shutdown=lifecycle.get('phases', {}).get('shutdown'),
            nsys_report=dict(path=str(rep), bytes=rep.stat().st_size if rep.is_file() else 0),
            capture=matrix['manifest']['profiling'].get('capture', 'legacy_delayed_window'),
            nsys=[], nsys_coverage='unverified_summary_only' if args.nsys_csv else 'missing_not_zero',
            metrics_missing=['per-layer/operator CUDA event ranges',
                'PID/rank/device mapping for EP vs root numeric kernels',
                'measured-request to trace-clock alignment (whole-lifecycle stats include startup/warmup)',
                'per-device/stream interval unions and CUDA API correlation for host gaps',
                'GDN direct/general rows, GQA visible tokens, expert-group M histogram',
                'operator semantic ranges are unavailable without matching runtime instrumentation'],
            interpretation='nsys stats are duration sums, not wall shares; do not add overlapping kernels/APIs/IPC',
            gpu_quiescence='unverified; subreaper ECHILD is not GPU telemetry')
        for path in args.nsys_csv:
            rows = parse_nsys_summary(bounded_text(path))
            summary['nsys'].append(dict(path=str(path.resolve()), sha256=sha256(path), rows=rows))
        summary['gpu_coverage'] = (sqlite_gpu_coverage(args.nsys_sqlite) if args.nsys_sqlite
                                   else dict(status='missing_not_zero'))
        if args.nsys_sqlite:
            summary['gpu_coverage'].update(path=str(args.nsys_sqlite.resolve()),
                                           sha256=sha256(args.nsys_sqlite))
        summary['artifact_smoke_pass'] = (run['status'] == 'complete'
            and summary['custody'].get('status') == 'proven_drained'
            and ipc['status'] == 'observed_lifetime_counters' and ipc['nonzero_transport_observed']
            and summary['nsys_report']['bytes'] > 0
            and summary['gpu_coverage']['status'] == 'gpu_events_observed_not_request_scoped')
        summary['full_rank_request_coverage'] = 'unverified; smoke pass is not complete EP8/request coverage'
        with args.output.open('x') as stream:
            json.dump(summary, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, KeyError, TypeError, IndexError, sqlite3.Error) as exc:
        parser.error(str(exc))
    return 1 if args.require_artifacts and not summary['artifact_smoke_pass'] else 0


def main() -> int:
    if sys.argv[1:2] == ['summarize']:
        return summarize_main(sys.argv[2:])
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
            "crates/ferrule-runtime/tests/support/process_pipeline_fixture.rs",
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
