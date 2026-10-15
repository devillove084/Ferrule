"""Pressure-only wall/CPU spans and composable absolute caps; no validation cache."""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
import threading
import time

# Absent spans remain null, including every historical helper duration. These
# are inclusive wall intervals: nested spans and observer threads must not sum.
PHASES = (
    'worker_contract', 'checkpoint_metadata', 'runtime_metadata', 'source_options',
    'startup', 'request_phase', 'bookkeeping', 'shutdown', 'worker_finalization',
    'pre_admission_validation', 'pre_dispatch_validation', 'post_http_validation',
    'workflow_pins', 'candidate_pins', 'receipt_hashes', 'source_custody_checks',
    'library_maps', 'nfs', 'pid_io', 'http', 'token_join', 'source_counters',
    'pre_observer', 'post_observer', 'observer_pss', 'observer_diagnostics',
    'observer_finish',
)


@contextmanager
def deadline_cap(deadline):
    """Nest only by taking the earlier absolute cutoff, never restarting a cap.

    The shared HTTP helper owns a request alarm. Its private pressure copy uses
    this seam so that a phase alarm also covers callbacks/metadata/serialization.
    Restoring a parent timer subtracts elapsed time (including exception cleanup).
    """
    now = time.monotonic()
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError('finite absolute pressure deadline required')
    if deadline <= now:
        raise TimeoutError('absolute pressure phase deadline before admission')
    previous = signal.getsignal(signal.SIGALRM)
    old, interval = signal.getitimer(signal.ITIMER_REAL)
    if interval:
        raise ValueError('periodic alarm is not a pressure deadline')
    parent = now + old if old else None
    effective = min(deadline, parent) if parent is not None else deadline
    def expired(number, frame):
        raise TimeoutError('absolute pressure phase deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.000001, effective - time.monotonic()))
    try:
        yield
        if time.monotonic() >= effective:
            raise TimeoutError('absolute pressure phase deadline after callback')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if parent is not None:
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, parent - time.monotonic()))


class Timings:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.Lock()
        self.local = threading.local()
        self.rows = []
        self.created = False
        self.closed = False

    def emit(self, row):
        with self.lock:
            if self.closed:
                return
            # Do not serialize arguments, environment, paths, payloads or errors.
            with (self.directory / 'phase-timings.jsonl').open('a' if self.created else 'x') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            self.created = True
            self.rows.append(row)

    @contextmanager
    def span(self, name):
        if name not in PHASES:
            raise ValueError('unknown pressure timing phase')
        stack = getattr(self.local, 'stack', [])
        started, cpu = time.monotonic(), time.thread_time()
        row = dict(phase=name, parent=stack[-1] if stack else None,
                   thread='worker' if threading.current_thread() is threading.main_thread() else 'observer',
                   started_monotonic=started, ended_monotonic=None, wall_seconds=None, thread_cpu_seconds=None,
                   status='started')
        self.emit(row)
        self.local.stack = stack + [name]
        status = 'complete'
        try:
            yield
        except BaseException:
            status = 'interrupted'
            raise
        finally:
            ended = time.monotonic()
            self.emit(dict(row, status=status, ended_monotonic=ended, wall_seconds=ended - started,
                           thread_cpu_seconds=time.thread_time() - cpu))
            self.local.stack = stack

    def call(self, name, function, *args, **kwargs):
        with self.span(name):
            return function(*args, **kwargs)

    def finish(self):
        with self.lock:
            if not self.created or self.closed:
                return
            self.closed = True
            rows = list(self.rows)
        totals = {name: None for name in PHASES}
        for row in rows:
            if row['wall_seconds'] is not None:
                totals[row['phase']] = (totals[row['phase']] or 0) + row['wall_seconds']
        value = dict(schema='ferrule.storage-demand.pressure-phase-timings.v2',
                     inclusive_wall_seconds=totals, historical_helper_durations_seconds=None,
                     additive_across_names_or_threads=False, phase_events=len(rows),
                     missing_means='unobserved, not zero',
                     open_spans=sum(r['status'] == 'started' for r in rows) -
                                sum(r['status'] != 'started' for r in rows))
        with (self.directory / 'phase-timings.json').open('x') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
