#!/usr/bin/env python3
"""Foreground nsys target; cooperative direct-child shutdown, never descendant cleanup.

The matrix worker writes profile-stop even after HTTP failure. Keep nsys alive
until the real server has had a chance to flush thread-exit IPC counters. Only
the outer verified subreaper may clean up remaining descendants. No retries.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def checkpoint(path, report):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_target(directory, command, lifetime_seconds, shutdown_seconds, *,
               natural_completion=False, on_spawn=None):
    if type(natural_completion) is not bool:
        raise ValueError("natural_completion must be an explicit boolean")
    started = time.monotonic()
    report = dict(schema='ferrule.profile-server.v1', started_monotonic=started,
                  started_unix_ns=time.time_ns(), returncode=None, stop_reason=None,
                  environment={key: os.environ.get(key) for key in (
                      'FERRULE_PROCESS_IPC_TIMING', 'CUDA_VISIBLE_DEVICES',
                      'LD_LIBRARY_PATH', 'LD_PRELOAD')},
                  instrumentation='compile_time_unverified_until_sink',
                  descendant_custody='outer_subreaper_required')
    path = directory / 'profile-server.json'
    stop_signals = []
    previous = {sig: signal.signal(sig, lambda number, frame: stop_signals.append(number))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        # Inherit nsys injection and output descriptors without an intervening
        # env -i / LD_LIBRARY_PATH assignment that could discard its libraries.
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        report['pid'] = child.pid
        checkpoint(path, report)
        if on_spawn is not None:
            on_spawn(child)  # Observation only; the maintained wrapper owns this child.
        deadline = started + lifetime_seconds
        while child.poll() is None:
            if stop_signals or (directory / 'profile-stop').exists():
                report['stop_reason'] = 'signal' if stop_signals else 'matrix_worker_stop_marker'
                break
            if time.monotonic() >= deadline:
                report['stop_reason'] = 'capture_lifetime_timeout'
                break
            time.sleep(.1)
        if child.poll() is None:
            report['shutdown_started_monotonic'] = time.monotonic()
            checkpoint(path, report)
            child.terminate()  # only our unreaped direct child, never a numeric PID/group
            try:
                child.wait(timeout=shutdown_seconds)
            except subprocess.TimeoutExpired:
                report['shutdown_timeout'] = True
                # Exit to allow nsys partial export; the outer subreaper owns
                # this still-live child. Do not race its descendant cleanup.
        if natural_completion and report['stop_reason'] is None:
            report['stop_reason'] = 'natural_completion'
        report['returncode'] = child.returncode
        report['ended_monotonic'] = time.monotonic()
        report['ended_unix_ns'] = time.time_ns()
        checkpoint(path, report)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    expected = 'natural_completion' if natural_completion else 'matrix_worker_stop_marker'
    return 0 if report['stop_reason'] == expected and report['returncode'] == 0 else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--lifetime-seconds', required=True, type=float)
    parser.add_argument('--shutdown-seconds', required=True, type=float)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if not args.command or args.command[0] != '--' or len(args.command) < 2:
        parser.error('requires -- followed by the direct server command')
    if not (math.isfinite(args.lifetime_seconds) and 0 < args.lifetime_seconds <= 1560
            and math.isfinite(args.shutdown_seconds) and 0 < args.shutdown_seconds <= 45):
        parser.error('invalid bounded capture/shutdown budget')
    return run_target(args.directory, args.command[1:], args.lifetime_seconds, args.shutdown_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
