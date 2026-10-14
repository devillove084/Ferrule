"""TP1 pilot adapters over maintained capture/HTTP/observer APIs, not a supervisor."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import signal
import subprocess
import sys
import time

from .evidence_v2 import (
    ct, finite, integer, require, UUID_RE, validate_schedule, censoring_summary,
    request_budget_seconds,
)
from .closure_v2 import parse_resource_shutdown_summaries, evidence
from .mountstats_v2 import capture_mountstats, mountstats_delta
from .trace_v2 import parse_token_trace, join_token_trace_http, logical_counter_window

sys.path.insert(0, str(ct.ROOT / 'scripts'))
sys.path.insert(0, str(ct.HERE / 'tp2_resident'))
import bench_ferrule_vllm as http
import observer
import runner

LOG_LIMIT = 64 * 1024 * 1024


def stamp():
    return dict(monotonic=time.monotonic(), unix_ns=time.time_ns())


@contextmanager
def absolute_alarm(deadline):
    """Admission/I/O deadline; cleanup runs after restoring the original handler."""
    require(finite(deadline) and deadline > time.monotonic(), 'absolute deadline expired')
    previous = signal.getsignal(signal.SIGALRM)
    old = signal.getitimer(signal.ITIMER_REAL)
    require(old == (0.0, 0.0), 'nested/sliding alarm is not permitted')
    def expired(number, frame):
        raise TimeoutError('absolute pilot phase deadline')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, deadline - time.monotonic())
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def log_lines(path, start=0, end=None):
    size = Path(path).stat().st_size
    end = size if end is None else end
    require(0 <= start <= end <= size <= LOG_LIMIT, 'server log window missing/oversized/reset')
    with Path(path).open('rb') as stream:
        stream.seek(start)
        raw = stream.read(end - start)
    require(not raw or raw.endswith(b'\n'), 'server JSON window is not fully flushed')
    return raw.decode('utf-8').splitlines()


def process_io(root):
    """Authenticated Linux PID accounting, explicitly NOT physical NFS traffic."""
    observer.verify_root(root)
    begin = stamp()
    raw = Path(f'/proc/{root["pid"]}/io').read_text()
    values = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(':')
        require(separator and value.strip().isdigit(), 'malformed process IO counter')
        values[key] = int(value)
    observer.verify_root(root)
    return dict(root_identity=dict(root), acquisition_begin=begin, acquisition_end=stamp(),
                raw=raw, counters=values, physical_nfs_bytes=None,
                scope='authenticated PID /proc/io; not NFS network or NAS backend bytes')


def library_maps(root, pins):
    observer.verify_root(root)
    raw = Path(f'/proc/{root["pid"]}/maps').read_text()
    paths = sorted({line.split()[-1] for line in raw.splitlines()
                    if '.so' in line and line.split()[-1].startswith('/')})
    require(paths and '(deleted)' not in raw, 'loaded library mappings missing/deleted')
    for path in paths:
        actual = str(Path(path).resolve())
        require(actual in pins and ct.sha(actual) == pins[actual], 'unbound loaded library: ' + actual)
    observer.verify_root(root)
    return dict(root_identity=dict(root), captured=stamp(), raw_maps=raw,
                libraries={str(Path(path).resolve()): pins[str(Path(path).resolve())] for path in paths})


def owner_summary(lines, root, uuid):
    """Map only actual scopes from the single owned server log; no invented owners."""
    parsed = parse_resource_shutdown_summaries(lines)
    scopes = []
    for row in parsed['attempts']:
        summary = row['summary']
        runtime = summary.get('runtime', {}).get('scope')
        state, physical = evidence(summary.get('physical', {}).get('scope'))
        valid = isinstance(runtime, dict) and isinstance(physical, dict)
        valid = valid and state in ('observed', 'verified') and all(
            integer(scope.get(key), 1) for scope in (runtime, physical) for key in ('owner', 'generation'))
        if not valid:
            parsed['final'].update(can_continue=False)
            parsed.setdefault('mapping_errors', []).append('actual runtime/physical scope unavailable')
            continue
        scopes.append(dict(runtime=runtime, physical=physical))
    identity_bound = isinstance(root, dict) and all(integer(root.get(key), 1) for key in ('pid', 'birth_ticks', 'ppid'))
    identity_bound = identity_bound and isinstance(uuid, str) and bool(UUID_RE.fullmatch(uuid))
    mapped = bool(identity_bound and scopes) and len(scopes) == len(parsed['attempts'])
    stable = mapped and all(scope == scopes[0] for scope in scopes)
    if stable:
        parsed = parse_resource_shutdown_summaries(lines, mapped_owner=True)
    elif mapped:
        parsed['final'].update(custody_state='unknown', can_continue=False)
        parsed['mapping_errors'] = ['owner/generation changed between actual shutdown attempts']
    parsed['source_binding'] = dict(server_identity=root, physical_uuid=uuid,
        original_scopes=scopes[0] if scopes else None,
        scope_source='first actual shutdown attempt in exclusively captured owner log; not a Ready snapshot',
        stable_actual_scopes=bool(stable), resource_owner_is_pid=False)
    return parsed


class CapturedServer:
    """Observer facade; only the maintained capture Popen may signal its child."""
    def __init__(self, capture, directory, root):
        self.capture, self.directory, self.root = capture, Path(directory), root
        self.pid = root['pid']

    def poll(self):
        if self.capture.poll() is not None:
            return self.capture.returncode
        observer.verify_root(self.root)
        return None

    def terminate(self):
        observer.verify_root(self.root)
        stop_capture(self.directory)


def stop_capture(directory):
    marker = Path(directory) / 'profile-stop'
    if not marker.exists():
        with marker.open('xb'):
            pass


def launch_capture(directory, spec, binary_sha256, lifetime, shutdown):
    """Unique immutable staged pathname: run_target closes inherited exec FDs."""
    runner.require_free_port(spec)
    with ct.pinned_executable(spec['argv'][0], binary_sha256):
        command = ct.isolated_argv(spec['argv'], spec['environment'])
        argv = [sys.executable, '-S', '-B', str(ct.ROOT / 'scripts/_profile_capture.py'),
                '--directory', str(directory), '--lifetime-seconds', str(lifetime),
                '--shutdown-seconds', str(shutdown), '--', *command]
        with (Path(directory) / 'server.log').open('xb') as log:
            return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, cwd=ct.ROOT)


def capture_root(capture, directory, deadline, authenticated):
    target = observer.proc_identity(capture.pid)
    require(target['ppid'] == authenticated['worker']['pid'], 'capture is not authenticated worker child')
    while time.monotonic() < deadline:
        require(capture.poll() is None, 'capture exited before identifying server')
        observer.verify_root(target)
        try:
            report = ct.load(Path(directory) / 'profile-server.json')
            root = observer.proc_identity(report['pid'])
            require(root['ppid'] == target['pid'], 'server is not capture direct child')
            observer.verify_root(root)
            return target, root, report
        except (FileNotFoundError, KeyError):
            time.sleep(.01)
    raise TimeoutError('absolute startup deadline before server identity')


def one_request(directory, planned, spec, deadline):
    """Reuse the maintained bounded HTTP client, with no extra warmup or retries."""
    remaining = deadline - time.monotonic()
    require(remaining > 0, 'request deadline before HTTP admission')
    with absolute_alarm(deadline):
        code = http.main(['--engine', 'ferrule', '--url', spec['url'], '--model', spec['model'],
                   '--prompt', planned['prompt'], '--input-tokens', str(planned['input_tokens']),
                   '--max-tokens', str(planned['max_tokens']), '--seed', '1',
                   '--concurrency', '1', '--requests', '1', '--warmup', '0', '--repeat', '1',
                   '--timeout', str(remaining), '--window-kind', 'fixed-work',
                   '--measurement-seconds', str(remaining), '--output', str(Path(directory) / 'http.json')])
    report = ct.load(Path(directory) / 'http.json')
    rows = [row for run in report.get('runs', []) for row in run.get('requests', [])]
    require(len(rows) == 1, 'issued HTTP request lacks one terminal receipt')
    return dict(rows[0], adapter_http_report_complete=code == 0 and report.get('status') == 'complete'
                and report.get('finalized') is True and not report.get('errors'),
                adapter_http_report_errors=report.get('errors'))


def request_gate(row, actual, trace, counters, nfs):
    text = actual.get('text')
    text_ok = isinstance(text, str) and bool(text) and (
        row['reference'] is None or text == row['reference']['reference_text'])
    usage = actual.get('usage') or {}
    usage_ok = all(type(usage.get(key)) is int for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    usage_ok = usage_ok and usage['prompt_tokens'] == row['input_tokens'] and usage['completion_tokens'] == row['max_tokens']
    usage_ok = usage_ok and usage['total_tokens'] == row['input_tokens'] + row['max_tokens']
    terminals = trace.get('worker_terminal_enqueue', [])
    terminal_usage = len(terminals) == 1 and terminals[0].get('usage_available') is True
    terminal_usage = terminal_usage and terminals[0].get('delivery_boundary') == 'worker_channel_enqueue'
    terminal_usage = terminal_usage and all(terminals[0].get(key) == usage.get(key)
                                           for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    return bool(actual.get('status') == 'success' and actual.get('adapter_http_report_complete') is True
                and actual.get('finish_reason') in ('length', 'stop')
                and text_ok and usage_ok and terminal_usage and trace['metric_eligible'] and counters['valid']
                and nfs['status'] == 'shared_client_delta')


def serial_requests(directory, schedule, spec, limits, target_deadline, outer_deadline,
                    pilot_deadline, check, sample, io_sample):
    """Journal every admitted slot before HTTP; unknown/lost results are censored."""
    validate_schedule(schedule)
    started = time.monotonic()
    observations, issued, seen_ids = [], [], set()
    log = Path(directory) / 'server.log'
    stop_reason = None
    for planned in schedule:
        try:
            check()
            budget = request_budget_seconds(limits, time.monotonic(), started,
                                            target_deadline, outer_deadline, pilot_deadline)
            require(budget > 0, 'aggregate/target/outer/pilot admission deadline')
            folder = Path(directory) / planned['name']
            folder.mkdir(exist_ok=False)
            pre = sample(planned['name'] + '-pre')
            before_io = io_sample()
            begin = stamp()
            offset = log.stat().st_size
            require(offset <= LOG_LIMIT, 'oversized server log')
            # Recheck after filesystem/proc sampling; never reset the phase deadline.
            check()
            budget = request_budget_seconds(limits, time.monotonic(), started,
                                            target_deadline, outer_deadline, pilot_deadline)
            require(budget > 0, 'deadline during request preparation; unissued')
            deadline = min(time.monotonic() + budget, started + limits['all_requests_seconds'],
                           target_deadline, outer_deadline, pilot_deadline)
            journal = dict(planned, issued=True, status='censored', censor_reason='dispatch without terminal',
                           admission=stamp(), absolute_deadline=deadline, log_start_byte=offset)
            ct.write(folder / 'issued.json', journal)
            issued.append(journal)
            actual = one_request(folder, planned, spec, deadline)
            ct.write(folder / 'http-terminal.json', actual)
            end = stamp()
            after_io = io_sample()
            post = sample(planned['name'] + '-post')
            check()
            window_end = log.stat().st_size
            lines = log_lines(log, offset, window_end)
            trace = join_token_trace_http(parse_token_trace(lines), [dict(planned, actual=actual)], lines)[0]
            duplicate = trace['actual_worker_request_id'] in seen_ids
            if trace['actual_worker_request_id'] is not None:
                seen_ids.add(trace['actual_worker_request_id'])
            counters = logical_counter_window(lines)
            nfs = mountstats_delta(pre, post)
            eligible = not duplicate and end['monotonic'] <= deadline and request_gate(planned, actual, trace, counters, nfs)
            status = 'success' if actual.get('status') == 'success' else (
                'censored' if actual.get('status') in ('timeout', 'client_error', 'transport_error') else 'failed')
            observed = dict(planned, issued=True, status=status, actual=actual, trace=trace,
                            duplicate_lifecycle_request_id=duplicate, quality_passed=eligible,
                            metric_eligible=eligible, logical_counters=counters, nfs=nfs,
                            process_io=dict(before=before_io, after=after_io),
                            log_start_byte=offset, log_end_byte=window_end,
                            window_begin=begin, window_end=end, absolute_deadline=deadline)
            if end['monotonic'] > deadline:
                observed.update(status='censored', right_censored=True, censor_reason='absolute request deadline')
            ct.write(folder / 'observed.json', observed)
            observations.append(observed)
            require(eligible, 'HTTP/text/usage/actual-ID/terminal/counter/mount/deadline gate: ' + planned['name'])
        except (Exception, KeyboardInterrupt) as error:
            stop_reason = type(error).__name__ + ': ' + str(error)
            if issued and issued[-1]['name'] == planned['name'] and not any(r['name'] == planned['name'] for r in observations):
                observations.append(dict(issued[-1], status='censored', right_censored=True,
                                         metric_eligible=False, censor_reason=stop_reason))
            break
    journal_schedule = [dict(row, issued=any(i['name'] == row['name'] for i in issued)) for row in schedule]
    result = dict(started_monotonic=started, absolute_aggregate_deadline=started + limits['all_requests_seconds'],
                  observations=observations, summary=censoring_summary(observations, journal_schedule),
                  stop_reason=stop_reason, can_continue=stop_reason is None and len(observations) == 12)
    ct.write(Path(directory) / 'requests-final.json', result)
    return result


def nfs_sampler(directory, reader, root, checkpoint_paths):
    """Worker reader survives server shutdown; seal its namespace to the server."""
    namespace = None
    def sample(label):
        nonlocal namespace
        try:
            value = capture_mountstats(directory, label, reader, observer.verify_root, checkpoint_paths)
            if namespace is None:
                namespace = value['mount_namespace']
            require(value['mount_namespace'] == namespace, 'reader mount namespace changed')
            if root is not None:
                observer.verify_root(root)
                require(str(Path(f'/proc/{root["pid"]}/ns/mnt').readlink()) == namespace,
                        'server/reader mount namespace mismatch')
            return value
        except Exception as error:
            failure = dict(label=label, acquisition_status='unavailable', error=str(error),
                           captured=stamp(), physical_nas_backend_bytes=None)
            path = Path(directory) / (label + '-mountstats-error.json')
            if not path.exists():
                ct.write(path, failure)
            raise
    return sample


def lifecycle_assessment(observed, supervisor, post_idle):
    reasons = []
    if not ct.drained(supervisor):
        reasons.append('unknown authenticated descendant custody')
    if not ct.successful_process(supervisor):
        reasons.append('worker failed/timed out')
    if observed.get('status') != 'complete':
        reasons.append(observed.get('error') or 'incomplete lifecycle')
    if not observed.get('closure', {}).get('final', {}).get('can_continue'):
        reasons.append('owner close missing/pending/unknown')
    if post_idle.get('status') != 'verified' or post_idle.get('all_eight_idle') is not True:
        reasons.append('independent post-idle not established')
    return dict(can_continue=not reasons, stop_reasons=reasons,
                resource_contract_complete='unverified', universal_final_release_verified=False)


def recover_requests(directory, schedule):
    """After lost worker output, admitted records never become unissued."""
    rows = []
    for planned in schedule:
        folder = Path(directory) / planned['name']
        if (folder / 'observed.json').is_file():
            rows.append(ct.load(folder / 'observed.json'))
        elif (folder / 'issued.json').is_file():
            rows.append(dict(ct.load(folder / 'issued.json'), status='censored', metric_eligible=False,
                             censor_reason='worker lost before terminal artifact'))
    return censoring_summary(rows, schedule)
