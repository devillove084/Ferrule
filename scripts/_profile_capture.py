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
import secrets
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


def live_child_identity(child):
    if child.returncode is not None or os.waitid(
            os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
        raise RuntimeError("Planned stop requires an unreaped live direct child")
    fields = Path(f"/proc/{child.pid}/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in ("Z", "X") or int(fields[1]) != os.getpid():
        raise RuntimeError("Planned stop child is terminal or has a foreign parent")
    return dict(pid=child.pid, birth_ticks=int(fields[19]), ppid=int(fields[1]))


def run_target(directory, command, lifetime_seconds, shutdown_seconds, *,
               natural_completion=False, on_spawn=None, on_terminal=None, on_stop=None,
               absolute_deadline=None):
    """Run the direct child; an optional terminal hook fences its final reap.

    on_terminal(child, waitid_event, absolute_deadline, active) is cooperative:
    it must check active(), honour the deadline, and return only after observer
    acknowledgement. It must not poll/wait/reap/signal the child. Hook failure
    is non-success and leaves failure drain to the outer maintained owner.

    on_stop(child, stop_event, absolute_deadline, active) is an opt-in planned
    HTTP stop fence, incompatible with natural completion/on_terminal. It must
    bind the live child to its job/plan, validate completed requests and role-map
    evidence, stop/join the observer and reject sticky errors before returning
    exactly {"stop_event": stop_event, "observer_stopped": True}. Persist that
    evidence in the caller; this wrapper validates only the fresh handshake and
    live direct-child custody. The callback must not poll/wait/reap/signal the
    child. active() revalidates custody, cancellation and the absolute budget.
    ACK plus shutdown share shutdown_seconds within the original lifetime.
    absolute_deadline optionally caps that lifetime with the caller-provided
    monotonic deadline; validation/startup/ACK/shutdown never refresh this cap.
    It must be a finite positive int/float (not bool), checked again immediately
    before spawning. Omission preserves existing relative-budget behavior.
    """
    if type(natural_completion) is not bool:
        raise ValueError("natural_completion must be an explicit boolean")
    if on_terminal is not None and (not natural_completion or not callable(on_terminal)):
        raise ValueError("on_terminal requires natural completion and a callable")
    if on_terminal is not None and not (math.isfinite(lifetime_seconds) and 0 < lifetime_seconds <= 1560):
        raise ValueError("Observed capture requires a finite bounded lifetime")
    if on_stop is not None:
        if natural_completion or on_terminal is not None or not callable(on_stop):
            raise ValueError("on_stop requires server mode and a callable, without on_terminal")
        if not (math.isfinite(lifetime_seconds) and 0 < lifetime_seconds <= 1560
                and math.isfinite(shutdown_seconds) and 0 < shutdown_seconds <= 45):
            raise ValueError("Planned stop requires finite bounded lifetime and shutdown")
        if (directory / "profile-stop").exists():
            raise ValueError("Planned stop marker predates child launch")
    if absolute_deadline is not None:
        if type(absolute_deadline) not in (int, float):
            raise ValueError("absolute_deadline must be a finite positive monotonic int/float")
        try:
            absolute_deadline = float(absolute_deadline)
        except OverflowError as error:
            raise ValueError("absolute_deadline is outside the finite clock range") from error
        if not math.isfinite(absolute_deadline) or absolute_deadline <= 0:
            raise ValueError("absolute_deadline must be a finite positive monotonic int/float")
        if not (math.isfinite(lifetime_seconds) and 0 < lifetime_seconds <= 1560
                and math.isfinite(shutdown_seconds) and 0 < shutdown_seconds <= 45):
            raise ValueError("Absolute capture requires finite bounded lifetime and shutdown")
    started = time.monotonic()
    deadline = started + lifetime_seconds
    if absolute_deadline is not None:
        deadline = min(deadline, absolute_deadline)
    report = dict(schema='ferrule.profile-server.v1', started_monotonic=started,
                  started_unix_ns=time.time_ns(), returncode=None, stop_reason=None,
                  environment={key: os.environ.get(key) for key in (
                      'FERRULE_PROCESS_IPC_TIMING', 'CUDA_VISIBLE_DEVICES',
                      'LD_LIBRARY_PATH', 'LD_PRELOAD')},
                  instrumentation='compile_time_unverified_until_sink',
                  descendant_custody='outer_subreaper_required')
    if absolute_deadline is not None:
        report.update(absolute_deadline=absolute_deadline, lifetime_deadline=deadline)
    path = directory / 'profile-server.json'
    stop_signals = []
    previous = {sig: signal.signal(sig, lambda number, frame: stop_signals.append(number))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        # Inherit nsys injection and output descriptors without an intervening
        # env -i / LD_LIBRARY_PATH assignment that could discard its libraries.
        if absolute_deadline is not None and time.monotonic() >= deadline:
            report.update(stop_reason="absolute_deadline_expired", ended_monotonic=time.monotonic(),
                          ended_unix_ns=time.time_ns())
            checkpoint(path, report)
            raise TimeoutError("Capture absolute deadline expired before spawn")
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL)
        report['pid'] = child.pid
        checkpoint(path, report)
        if on_spawn is not None and on_stop is None:
            on_spawn(child)  # Observation only; the maintained wrapper owns this child.
        if on_terminal is not None:
            def active():
                if stop_signals or (directory / 'profile-stop').exists():
                    report['stop_reason'] = 'signal' if stop_signals else 'matrix_worker_stop_marker'
                    raise RuntimeError('Observed capture cancelled before terminal acknowledgement')
                if time.monotonic() >= deadline:
                    report['stop_reason'] = 'capture_lifetime_timeout'
                    raise TimeoutError('Observed capture lifetime includes startup and terminal acknowledgement')

            # poll()/wait() would discard the birth anchor while a sample may
            # still be in flight. The hook must honour deadline/active(); it
            # observes only, and must never wait/reap or signal this child.
            try:
                while True:
                    active()
                    terminal = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                    observed = time.monotonic()
                    if terminal is not None:
                        break
                    # No waitable event is not liveness: the group leader
                    # may already be a zombie while other threads remain.
                    time.sleep(.05)
                if terminal.si_pid != child.pid or terminal.si_code not in (os.CLD_EXITED, os.CLD_KILLED, os.CLD_DUMPED):
                    raise RuntimeError('Unexpected direct-child terminal waitid identity/status')
                event = dict(pid=child.pid, code=terminal.si_code, status=terminal.si_status,
                             started_monotonic=started,
                             observed_monotonic=observed)
                active()
                on_terminal(child, event, deadline, active)
                active()
                held = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                if held != terminal or child.returncode is not None:
                    raise RuntimeError('Terminal hook lost unreaped direct-child custody')
                code = child.wait(timeout=deadline - time.monotonic())
                expected_code = terminal.si_status if terminal.si_code == os.CLD_EXITED else -terminal.si_status
                if code != expected_code:
                    raise RuntimeError('Authoritative child wait differs from terminal evidence')
                active()
            except BaseException as error:
                report['stop_reason'] = report['stop_reason'] or 'terminal_observer_failure'
                report['error'] = type(error).__name__ + ': ' + str(error)
                report['ended_monotonic'] = time.monotonic()
                report['ended_unix_ns'] = time.time_ns()
                checkpoint(path, report)
                # No success receipt or alternate reaper. Failure drain remains
                # the outer maintained subreaper responsibility.
                raise
        elif on_stop is not None:
            report["planned_stop_validated"] = False
            try:
                root = live_child_identity(child)
                report["root_identity"] = root
                if on_spawn is not None:
                    on_spawn(child)

                def budget_active():
                    if stop_signals:
                        report["stop_reason"] = "signal"
                        raise RuntimeError("Planned stop cancelled before completion")
                    if time.monotonic() >= deadline:
                        report["stop_reason"] = "planned_stop_timeout"
                        raise TimeoutError("Planned stop budget includes observer ACK and shutdown")

                def active():
                    budget_active()
                    if live_child_identity(child) != root:
                        raise RuntimeError("Planned stop child PID/birth/parent identity changed")
                    budget_active()

                while True:
                    active()
                    if (directory / "profile-stop").exists():
                        break
                    time.sleep(.05)
                requested = time.monotonic()
                deadline = min(deadline, requested + shutdown_seconds)
                event = dict(root, stop_id=secrets.token_hex(16),
                             stop_reason="matrix_worker_stop_marker",
                             requested_monotonic=requested, absolute_deadline=deadline)
                report.update(stop_reason="planned_stop_requested", stop_event=event)
                checkpoint(path, report)
                active()
                ack = on_stop(child, dict(event), deadline, active)
                active()
                if (not isinstance(ack, dict) or set(ack) != {"stop_event", "observer_stopped"}
                        or ack["stop_event"] != event or ack["observer_stopped"] is not True):
                    raise RuntimeError("Missing/stale planned-stop observer acknowledgement")
                report.update(stop_ack=ack, observer_acknowledged_monotonic=time.monotonic())
                checkpoint(path, report)
                active()
                report["shutdown_started_monotonic"] = time.monotonic()
                child.terminate()  # Only this held direct child, after observer acknowledgement.
                # POSIX Popen.terminate polls first. With this sole waiter, a
                # cached status here means it saw a terminal child before kill.
                if child.returncode is not None:
                    report["stop_reason"] = "pre_signal_terminal"
                    raise RuntimeError("Child became terminal before SIGTERM; planned stop not established")
                try:
                    code = child.wait(timeout=deadline - time.monotonic())
                except subprocess.TimeoutExpired:
                    report["shutdown_timeout"] = True
                    raise
                budget_active()
                report.update(stop_reason="planned_stop",
                              planned_stop_validated=code in (0, -signal.SIGTERM))
            except BaseException as error:
                if report["stop_reason"] not in ("signal", "planned_stop_timeout", "pre_signal_terminal"):
                    report["stop_reason"] = "stop_observer_failure"
                report.update(error=type(error).__name__ + ": " + str(error),
                              returncode=child.returncode, ended_monotonic=time.monotonic(),
                              ended_unix_ns=time.time_ns())
                checkpoint(path, report)
                # Do not turn a missing/unsafe ACK into a stop; the outer owner drains.
                raise
        else:
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
                if absolute_deadline is not None and time.monotonic() >= deadline:
                    report["shutdown_timeout"] = True
                else:
                    child.terminate()  # only our unreaped direct child, never a numeric PID/group
                    timeout = shutdown_seconds if absolute_deadline is None else max(
                        0, min(shutdown_seconds, deadline - time.monotonic()))
                    try:
                        child.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        report["shutdown_timeout"] = True
                        # Exit to allow nsys partial export; the outer subreaper owns
                        # this still-live child. Do not race its descendant cleanup.
        if absolute_deadline is not None and time.monotonic() >= deadline:
            report["stop_reason"] = "absolute_deadline_expired"
            if on_stop is not None:
                report["planned_stop_validated"] = False
        if natural_completion and report['stop_reason'] is None:
            report['stop_reason'] = 'natural_completion'
        report['returncode'] = child.returncode
        report['ended_monotonic'] = time.monotonic()
        report['ended_unix_ns'] = time.time_ns()
        checkpoint(path, report)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if on_stop is not None:
        return 0 if (report.get("planned_stop_validated") is True
                     and report["stop_reason"] == "planned_stop"
                     and report["returncode"] in (0, -signal.SIGTERM)) else 1
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
