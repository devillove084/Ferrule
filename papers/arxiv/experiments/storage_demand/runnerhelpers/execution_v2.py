"""TP1 pilot adapters over maintained capture/HTTP/observer APIs, not a supervisor."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import os
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


def reference_context_for_layout(layout):
    from .http_expectations import HISTORICAL_LENGTH, BOUND_HTTP_REFERENCE
    contract = layout.get('http_expectation_contract', HISTORICAL_LENGTH)
    binding = layout.get('http_reference_binding')
    if contract == HISTORICAL_LENGTH:
        require(binding is None, 'unexpected HTTP reference binding')
        return None
    require(contract == BOUND_HTTP_REFERENCE and isinstance(binding, dict), 'unknown HTTP reference contract')
    path = Path(binding['source'])
    require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
        and path.stat().st_mode & 0o222 == 0 and ct.sha(path) == binding['source_sha256'],
        'HTTP reference source/hash changed')
    return dict(source=str(path), source_sha256=ct.sha(path), document=ct.load(path), reviewed_binding=binding)


def transfer_runtime_maps(directory, label, root, parent, spec):
    from . import library_maps_v1 as maps
    binding = spec['runtime_binding']
    runtime = ct.load(binding['manifest'])
    observer.verify_root(root)
    command = Path(f'/proc/{root["pid"]}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    environment = dict(item.split('=', 1) for item in
        Path(f'/proc/{root["pid"]}/environ').read_text().split('\0') if '=' in item)
    require(command == spec['argv'] and all(environment.get(k) == v for k, v in spec['environment'].items())
        and environment.get('FERRULE_PROFILE_NVTX') == '0', 'actual CLI argv/environment changed')
    result = maps.library_maps(root, runtime['pins'], directory=Path(directory) / (label + '-library-evidence'),
        parent=parent, executable=binding['binary'], executable_sha256=binding['binary_sha256'],
        runtime=runtime, role='cli', identity=observer.proc_identity)
    observer.verify_root(root)
    result['actual_argv'] = command
    result['actual_environment'] = {k: environment[k] for k in spec['environment']}
    return result


def read_initial_endpoint(lines, options, producer_contract, *, server_identity=None, producer_binding=None,
                          policy=None, ready_lines=None):
    """Parse the actual typed producer once, before Ready and any prediction work."""
    from .trace_v2 import fields, json_events
    from paired_read_v1 import quality
    events = list(json_events(lines))
    rows = [(number, event) for number, event in events
        if fields(event).get('event') == quality.INITIAL_ENDPOINT_PRODUCER['event']]
    require(len(rows) == 1, 'initial read endpoint unobserved/duplicate; cannot invent a zero baseline')
    number, event = rows[0]
    f = fields(event)
    require(isinstance(producer_binding, dict) and set(producer_binding) == {'contract', 'source', 'source_sha256'}
        and ct.digest(producer_binding['contract']) == ct.digest(quality.INITIAL_ENDPOINT_PRODUCER)
        and producer_binding['source'] == quality.INITIAL_ENDPOINT_SOURCE
        and isinstance(producer_binding['source_sha256'], str) and len(producer_binding['source_sha256']) == 64
        and all(c in '0123456789abcdef' for c in producer_binding['source_sha256']),
        'initial endpoint requires explicit compiled producer binding')
    require(isinstance(server_identity, dict) and all(integer(server_identity.get(k), 1) for k in ('pid', 'birth_ticks', 'ppid'))
        and type(f.get('process_id')) is int and f['process_id'] == server_identity['pid'],
        'initial endpoint captured process identity differs')
    require(all(f.get(k) == v for k, v in quality.INITIAL_ENDPOINT_PRODUCER.items() if k not in ('target', 'level'))
        and type(f.get('producer_schema_version')) is int
        and event.get('target') == 'ferrule_performance' and event.get('level') == 'INFO',
        'initial endpoint producer schema/source contract differs')
    expected = {'synchronous': 'Synchronous', 'grouped-synchronous': 'GroupedSynchronous'}.get(policy)
    require(expected is not None and f.get('requested_expert_transfer_policy') == 'Some(' + expected + ')'
        and f.get('expert_transfer_policy') == expected, 'initial endpoint requested/getter policy differs')
    require(isinstance(ready_lines, list) and lines[:len(ready_lines)] == ready_lines and number <= len(ready_lines),
        'initial endpoint not captured before observed HTTP Ready')
    listening = [n for n, row in events if fields(row).get('message') == 'Ferrule OpenAI server listening']
    require(len(listening) == 1 and number < listening[0] <= len(ready_lines), 'initial endpoint must precede actual listening/Ready')
    require(not any(fields(row).get('event') in ('mixed_decoder_invocation', 'request_http_admitted',
        'request_worker_token_enqueued', 'request_worker_terminal_enqueue') for _, row in events),
        'initial endpoint captured after prediction/request activity')
    require(isinstance(f.get('expert_read_pipeline_initial'), str) and isinstance(f.get('owner_binding'), str),
        'initial endpoint and owner binding must be JSON strings')
    envelope, owner = json.loads(f['expert_read_pipeline_initial']), json.loads(f['owner_binding'])
    require(isinstance(envelope, dict) and set(envelope) == {'state', 'value'} and envelope['state'] == 'observed',
        'initial endpoint resource evidence unobserved/Unknown')
    require(isinstance(owner, dict) and set(owner) == {'scope', 'prepared_plan_id', 'image_generation', 'device_ordinal'}
        and all(isinstance(v, dict) and set(v) == {'state', 'value'} and v['state'] == 'observed' for v in owner.values()),
        'initial endpoint owner evidence unobserved/Unknown')
    scope = owner['scope']['value']
    require(isinstance(scope, dict) and set(scope) == {'owner', 'generation'} and all(integer(v, 1) for v in scope.values())
        and all(integer(owner[k]['value'], 1) for k in ('prepared_plan_id', 'image_generation'))
        and type(owner['device_ordinal']['value']) is int and owner['device_ordinal']['value'] == 0,
        'initial endpoint owner/image/device binding invalid')
    value = envelope['value']
    quality.read_observation_v3(value, options, stage='configured', producer_contract=producer_contract)
    return dict(schema='ferrule.storage-demand.initial-read-evidence.v1', read_endpoint=value,
        owner_binding=owner, server_identity=dict(server_identity), producer_binding=producer_binding,
        raw_event=event, log_line=number, before_ready=True,
        invocation_owner_identity='unobserved_in_invocation_records; no invented owner join')


def _identity_key(value):
    return {key: value[key] for key in ('pid', 'birth_ticks', 'ppid')}


def process_argv(pid):
    return Path(f'/proc/{pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0')


def authenticate_capture_parent(payload, launch_hash):
    """Recheck the maintained live worker/helper chain, not caller-authored custody."""
    import launch_guard
    directory = Path(payload['directory'])
    launch_path = directory / 'worker-launch.json'
    require(directory.is_absolute() and directory.resolve() == directory
        and launch_path.is_file() and not launch_path.is_symlink() and launch_path.stat().st_mode & 0o222 == 0
        and ct.sha(launch_path) == launch_hash and ct.load(launch_path) == payload
        and payload.get('worker_token') and os.environ.get('FERRULE_PAPER_WORKER_TOKEN') == payload['worker_token']
        and payload.get('worker_source_sha256') == ct.sha(ct.HERE / 'runner.py')
        and payload.get('manifest') == dict(protocol=ct.SCHEMA + 'worker.v2'), 'capture worker seal/token/source mismatch')
    parent = launch_guard.identity(os.getppid())
    helper = launch_guard.identity(parent['ppid'])
    coordinator = launch_guard.identity(helper['ppid'])
    custody = ct.load(directory / 'custody.json')
    helper_argv = process_argv(helper['pid'])
    require(len(helper_argv) == 7 and helper_argv[:3] == [sys.executable, '-B', str(ct.ROOT / 'scripts/_owned_subreaper.py')]
        and helper_argv[-1] == custody.get('launch_id'), 'capture parent is not an authenticated maintained-supervisor worker')
    from paired_read_v1.run_v3 import ENTRY
    require(process_argv(parent['pid']) == [sys.executable, '-S', '-B', str(ENTRY), '--_worker', launch_hash,
        '--layout', payload['execution_path']]
        and ct.sha(Path(f'/proc/{parent["pid"]}/exe')) == ct.sha(sys.executable)
        and ct.sha(Path(f'/proc/{helper["pid"]}/exe')) == ct.sha(sys.executable), 'capture parent worker CLI/source interpreter differs')
    require(custody.get('containment_verified') is True and custody.get('method') == 'linux-subreaper-waitid'
        and custody.get('root_session_ready') is True and custody.get('root_identity') == parent
        and custody.get('helper_identity') == helper and custody.get('errors') == [] and custody.get('reporting_errors') == []
        and custody.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        and coordinator == payload['coordinator'] and parent['pid'] == parent['pgrp'] == parent['session']
        and os.getpgrp() == os.getsid(0) == parent['pid'], 'capture authenticated containment/ancestry changed')
    require(any(row.get('identity') == parent and row.get('role') == 'root' and row.get('signals') == []
        and row.get('reaped') is False for row in custody.get('entries', [])), 'capture worker custody is terminal/signalled/unknown')
    for item in (parent, helper, coordinator):
        observer.verify_root(item)
        require(Path(f'/proc/{item["pid"]}/ns/pid').readlink() == Path('/proc/self/ns/pid').readlink(),
            'capture ancestor PID namespace differs')
    return parent


def http_capture_main(directory, launch_hash, target_deadline):
    """Reauthorize the actual entry before the maintained wrapper may spawn."""
    import _profile_capture as capture_api
    from paired_read_v1 import run_v3 as transfer
    directory = Path(directory)
    payload = ct.load(directory / 'worker-launch.json')
    require(payload.get('directory') == str(directory), 'capture directory crossbinding')
    parent = authenticate_capture_parent(payload, launch_hash)
    approved = transfer.validate_http_capture_entry(directory, payload, launch_hash, target_deadline, parent)
    spec = payload['spec']
    command = ct.isolated_argv(spec['argv'], spec['environment'])
    def stopped(child, event, deadline, active):
        active()
        ct.write(directory / 'http-stop-request.json', dict(schema='ferrule.storage-demand.http-stop-request.v1',
            stop_event=event, launch_sha256=launch_hash, worker_identity=parent))
        request_path = directory / 'http-stop-request.json'
        request_path.chmod(0o400)
        ack_path = directory / 'http-stop-ack.json'
        while True:
            active()
            require(not ack_path.is_symlink(), 'linked planned stop acknowledgement')
            if ack_path.is_file() and ack_path.stat().st_mode & 0o222 == 0:
                break
            time.sleep(.01)
        active()
        ack = ct.load(ack_path)
        require(not ack_path.is_symlink() and ack_path.stat().st_mode & 0o222 == 0
            and ack.get('schema') == 'ferrule.storage-demand.http-stop-ack.v1'
            and ack['stop_event'] == event and ack['launch_sha256'] == launch_hash
            and ack['request_sha256'] == ct.sha(request_path) and ack['observer_stopped'] is True
            and ack['qualification'] is True, 'planned stop acknowledgement mismatched/unqualified')
        required = {str((directory / name).resolve()) for name in
            ('resources-final.json', 'stop-library-maps.json', 'requests-final.json')}
        require(set(ack['evidence_pins']) == required, 'planned stop evidence incomplete/foreign')
        for name, pin in ack['evidence_pins'].items():
            require(Path(name).resolve() == Path(name) and not Path(name).is_symlink(), 'foreign planned stop evidence')
            ct.check_pins({name: pin})
        resources = ct.load(directory / 'resources-final.json')
        require(resources['status'] == 'observed_within_targets' and resources['errors'] == [], 'observer errors before planned stop')
        return dict(stop_event=event, observer_stopped=True)
    transfer.validate_transfer_inputs(spec['input_contract'])
    with ct.pinned_executable(spec['argv'][0], spec['frozen_cli_sha256']):
        # Input and ELF verification consume the original cap; neither grants a fresh lifetime.
        authenticate_capture_parent(payload, launch_hash)
        if spec.get('static_preparation') is not None:
            require(time.monotonic() < target_deadline, 'absolute target deadline expired during capture input/ELF verification')
            static = transfer.validate_static_preparation(spec['static_preparation'], payload['execution_path'], deadline=target_deadline)
            require(approved == static['bound_candidate'], 'capture prepared candidate changed')
        else:
            ct.check_pins(approved['candidate']['artifact_pins'])
        ct.check_pins(payload['adapter_pins'])
        ct.check_pins({payload['execution_path']: payload['execution_sha256']})
        for uuid, lock in zip(payload['eight_uuids'], payload['all_uuid_locks']):
            ct.validate_owner_lock(lock, uuid, payload['coordinator']['pid'])
        remaining = target_deadline - time.monotonic()
        require(remaining > 0, 'absolute target deadline expired during capture input/ELF verification')
        return capture_api.run_target(directory, command, remaining, payload['limits']['shutdown_seconds'],
            on_stop=stopped, absolute_deadline=target_deadline)


def launch_http_capture(directory, payload, target_deadline, *, target_started):
    import launch_guard
    require(finite(target_deadline) and finite(target_started) and time.monotonic() < target_deadline,
        'absolute HTTP target launch deadline expired/invalid')
    runner.require_free_port(payload['spec'])
    launch_hash = ct.sha(Path(directory) / 'worker-launch.json')
    entry = payload['spec']['http_capture_entry']
    require(entry in payload['adapter_pins'], 'capture entry unpinned')
    context = dict(schema='ferrule.storage-demand.http-capture-launch.v1', launch_sha256=launch_hash,
        worker_identity=launch_guard.identity(os.getpid()), target_started=target_started, target_deadline=target_deadline)
    path = Path(directory) / 'http-capture-launch.json'
    ct.write(path, context)
    path.chmod(0o400)
    argv = [sys.executable, '-S', '-B', entry, '--_http_capture', str(directory), launch_hash, repr(target_deadline)]
    require(time.monotonic() < target_deadline, 'absolute HTTP target deadline before capture spawn')
    with (Path(directory) / 'server.log').open('xb') as log:
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=ct.ROOT)


def acknowledge_http_stop(directory, payload, root, parent, monitor, requests):
    """Observe while the owned model is alive; ACK only after a clean observer join."""
    directory = Path(directory)
    stop_capture(directory)
    request_path = directory / 'http-stop-request.json'
    deadline = min(time.monotonic() + payload['limits']['shutdown_seconds'], payload['outer_deadline'], payload['pilot_deadline'])
    while True:
        require(time.monotonic() < deadline, 'planned stop request/observer handoff deadline')
        observer.verify_root(root); observer.verify_root(parent)
        require(not request_path.is_symlink(), 'linked planned stop request')
        if request_path.is_file() and request_path.stat().st_mode & 0o222 == 0:
            break
        time.sleep(.01)
    event_request = ct.load(request_path)
    launch_hash = ct.sha(directory / 'worker-launch.json')
    event = event_request['stop_event']
    require(_identity_key(event) == _identity_key(root) and event_request['launch_sha256'] == launch_hash
        and _identity_key(event_request['worker_identity']) == _identity_key(observer.proc_identity(os.getpid()))
        and event_request.get('schema') == 'ferrule.storage-demand.http-stop-request.v1'
        and type(event['absolute_deadline']) is float and finite(event['absolute_deadline'])
        and finite(event['requested_monotonic'])
        and event['absolute_deadline'] <= min(payload['outer_deadline'], payload['pilot_deadline'])
        and event['absolute_deadline'] <= event['requested_monotonic'] + payload['limits']['shutdown_seconds'],
        'planned stop request identity/deadline mismatch')
    with absolute_alarm(min(deadline, event['absolute_deadline'])):
        require(requests.get('can_continue') is True and len(requests.get('observations', [])) == 12
            and all(r.get('metric_eligible') is True for r in requests['observations']), 'incomplete HTTP work cannot ACK planned success')
        terminal_maps = transfer_runtime_maps(directory, 'stop', root, parent, payload['spec'])
        ct.write(directory / 'stop-library-maps.json', terminal_maps)
        resources = monitor.finish()
        ct.write(directory / 'resources-final.json', resources)
        require(not monitor.thread.is_alive() and resources['status'] == 'observed_within_targets'
            and resources['errors'] == [] and not monitor.failed.is_set(), 'observer failed/not joined before owned stop')
        observer.verify_root(root); observer.verify_root(parent)
        require(time.monotonic() < event['absolute_deadline'], 'observer ACK exceeded original stop deadline')
        evidence = ct.inventory([directory / 'resources-final.json', directory / 'stop-library-maps.json',
            directory / 'requests-final.json'])
        ack = dict(schema='ferrule.storage-demand.http-stop-ack.v1', stop_event=event,
            launch_sha256=launch_hash, request_sha256=ct.sha(request_path), observer_stopped=True,
            qualification=True, evidence_pins=evidence, acknowledged_monotonic=time.monotonic())
        ct.write(directory / 'http-stop-ack.json', ack)
        (directory / 'http-stop-ack.json').chmod(0o400)
    return resources


def validate_http_stop(directory, report, root, *, target_deadline=None):
    directory = Path(directory)
    request = ct.load(directory / 'http-stop-request.json')
    ack = ct.load(directory / 'http-stop-ack.json')
    event = report.get('stop_event', {})
    if target_deadline is not None:
        require(finite(target_deadline) and report.get('absolute_deadline') == target_deadline
            and finite(report.get('lifetime_deadline')) and report['lifetime_deadline'] <= target_deadline
            and finite(event.get('absolute_deadline')) and event['absolute_deadline'] <= target_deadline,
            'owned HTTP stop refreshed/mismatched original target deadline')
    times = [report.get('started_monotonic'), event.get('requested_monotonic'), ack.get('acknowledged_monotonic'),
        report.get('observer_acknowledged_monotonic'), report.get('shutdown_started_monotonic'),
        report.get('ended_monotonic'), event.get('absolute_deadline')]
    require(set(event) == {'pid', 'birth_ticks', 'ppid', 'stop_id', 'stop_reason',
            'requested_monotonic', 'absolute_deadline'}
        and all(integer(event.get(k), 1) for k in ('pid', 'birth_ticks', 'ppid'))
        and isinstance(event.get('stop_id'), str) and len(event['stop_id']) == 32
        and all(c in '0123456789abcdef' for c in event['stop_id'])
        and event.get('stop_reason') == 'matrix_worker_stop_marker'
        and all(finite(t) and t > 0 for t in times) and times == sorted(times)
        and set(report.get('stop_ack', {})) == {'stop_event', 'observer_stopped'},
        'owned planned HTTP stop event/time contract invalid')
    require(type(report.get('returncode')) is int and report['returncode'] in (0, -signal.SIGTERM)
        and report.get('planned_stop_validated') is True and report.get('stop_reason') == 'planned_stop'
        and not report.get('error') and not report.get('shutdown_timeout')
        and report.get('pid') == root['pid']
        and _identity_key(report['root_identity']) == _identity_key(root)
        and _identity_key(report['stop_event']) == _identity_key(root)
        and report['stop_event'] == request['stop_event'] == ack['stop_event'] == report['stop_ack']['stop_event']
        and report['stop_ack']['observer_stopped'] is True and ack['qualification'] is True
        and ack['request_sha256'] == ct.sha(directory / 'http-stop-request.json')
        and ack['launch_sha256'] == request['launch_sha256'] == ct.sha(directory / 'worker-launch.json')
        and request['stop_event']['requested_monotonic'] <= ack['acknowledged_monotonic']
            <= report['observer_acknowledged_monotonic'] <= report['shutdown_started_monotonic']
            <= report['ended_monotonic'] <= request['stop_event']['absolute_deadline'], 'owned planned HTTP stop evidence invalid')
    required = {str((directory / name).resolve()) for name in
        ('resources-final.json', 'stop-library-maps.json', 'requests-final.json')}
    require(required == set(ack['evidence_pins']) and all(Path(name).resolve() == Path(name)
        and not Path(name).is_symlink() for name in ack['evidence_pins']), 'planned stop evidence incomplete/foreign')
    require(ack.get('schema') == 'ferrule.storage-demand.http-stop-ack.v1'
        and request.get('schema') == 'ferrule.storage-demand.http-stop-request.v1'
        and all(not (directory / name).is_symlink() and (directory / name).stat().st_mode & 0o222 == 0
            for name in ('http-stop-ack.json', 'http-stop-request.json')), 'planned stop receipts not sealed/typed')
    ct.check_pins(ack['evidence_pins'])
    resources = ct.load(directory / 'resources-final.json')
    require(resources.get('status') == 'observed_within_targets' and resources.get('errors') == [],
        'observer status unobserved/sticky errors')
    return dict(schema='ferrule.storage-demand.owned-http-stop-result.v1', disposition='expected_planned_stop',
        actual_returncode=report['returncode'], signalled_by='held_direct_child_Popen_after_observer_ack',
        observer_ack_sha256=ct.sha(directory / 'http-stop-ack.json'), physical_release_verified=False)


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


def request_gate(row, actual, trace, counters, nfs, *, expectation_contract=None, reference_context=None):
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
    if expectation_contract is not None:
        from .http_expectations import http_output_matches
        text_ok = usage_ok = http_output_matches(row, actual, trace, contract=expectation_contract,
            reference_context=reference_context)
    return bool(actual.get('status') == 'success' and actual.get('adapter_http_report_complete') is True
                and actual.get('finish_reason') in ('length', 'stop')
                and text_ok and usage_ok and terminal_usage and trace['metric_eligible'] and counters['valid']
                and nfs['status'] == 'shared_client_delta')


def serial_requests(directory, schedule, spec, limits, target_deadline, outer_deadline,
                    pilot_deadline, check, sample, io_sample, *, server_identity=None, ready_log_end=None):
    """Journal every admitted slot before HTTP; unknown/lost results are censored."""
    validate_schedule(schedule)
    started = time.monotonic()
    observations, issued, seen_ids = [], [], set()
    log = Path(directory) / 'server.log'
    stop_reason = None
    transfer_case = spec.get('expert_transfer_case')
    startup = previous_read = initial_read = initial_evidence = None
    expectation_contract = spec.get('http_expectation_contract')
    reference_context = spec.get('http_reference_context')
    if transfer_case is not None:
        from .trace_v2 import transfer_startup
        startup = transfer_startup(log_lines(log), transfer_case['transfer_policy'], spec['frozen_cli_sha256'])
        require(ready_log_end is not None, 'captured Ready log boundary required')
        initial_evidence = read_initial_endpoint(log_lines(log), transfer_case['read_options'], spec['upload_producer_contract'],
            server_identity=server_identity, producer_binding=spec['initial_read_producer_binding'],
            policy=transfer_case['transfer_policy'], ready_lines=log_lines(log, end=ready_log_end))
        initial_read = initial_evidence['read_endpoint']
        previous_read = initial_read
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
            trace_planned = planned
            if expectation_contract is not None:
                from .http_expectations import normalize_http_expectation
                expected = normalize_http_expectation(planned, contract=expectation_contract, reference_context=reference_context)
                trace_planned = dict(planned, reference=None if expected['reference_ids'] is None else dict(
                    reference_ids=expected['reference_ids'], reference_text=expected['reference_text']))
            trace = join_token_trace_http(parse_token_trace(lines), [dict(trace_planned, actual=actual)], lines)[0]
            duplicate = trace['actual_worker_request_id'] in seen_ids
            if trace['actual_worker_request_id'] is not None:
                seen_ids.add(trace['actual_worker_request_id'])
            counters = logical_counter_window(lines)
            if transfer_case is not None:
                from .trace_v2 import transfer_counter_window
                counters = transfer_counter_window(lines, transfer_case['read_options'], previous_read=previous_read,
                    producer_contract=spec['upload_producer_contract'])
                require(counters.get('read_continuity_valid') is True,
                    'read boundary missing/invalid: ' + str(counters.get('health_error')))
                previous_read = counters['final_read_endpoint']
            nfs = mountstats_delta(pre, post)
            eligible = not duplicate and end['monotonic'] <= deadline and request_gate(planned, actual, trace, counters, nfs,
                expectation_contract=expectation_contract, reference_context=reference_context)
            if transfer_case is not None:
                eligible = eligible and all(
                    integer(counters['deltas'].get(k), 1) for k in ('expert_uploads', 'h2d_bytes'))
            status = 'success' if actual.get('status') == 'success' else (
                'censored' if actual.get('status') in ('timeout', 'client_error', 'transport_error') else 'failed')
            observed = dict(planned, issued=True, status=status, actual=actual, trace=trace,
                            duplicate_lifecycle_request_id=duplicate, quality_passed=eligible,
                            metric_eligible=eligible, logical_counters=counters, nfs=nfs,
                            process_io=dict(before=before_io, after=after_io),
                            log_start_byte=offset, log_end_byte=window_end,
                            window_begin=begin, window_end=end, admission=journal['admission'], absolute_deadline=deadline)
            if transfer_case is not None:
                from .trace_v2 import fields, json_events
                admitted = [fields(event) for _, event in json_events(lines) if
                    fields(event).get('event') == 'request_http_admitted' and
                    fields(event).get('request_id') == trace['actual_worker_request_id']]
                observed['server_http_admission_us'] = admitted[0].get('http_admission_us') if len(admitted) == 1 else None
                eligible = eligible and integer(observed['server_http_admission_us'])
                observed.update(metric_eligible=eligible, quality_passed=eligible)
                from paired_read_v1.analysis import request_metrics
                eligible = eligible and request_metrics(planned, observed, contract=expectation_contract,
                    reference_context=reference_context)['reference_qualified']
                observed.update(metric_eligible=eligible, quality_passed=eligible)
            if end['monotonic'] > deadline:
                observed.update(status='censored', right_censored=True, censor_reason='absolute request deadline')
            ct.write(folder / 'observed.json', observed)
            observations.append(observed)
            require(eligible, 'HTTP/text/usage/actual-ID/terminal/counter/mount/deadline gate: ' + planned['name'])
        except (Exception, KeyboardInterrupt) as error:
            stop_reason = type(error).__name__ + ': ' + str(error)
            if issued and issued[-1]['name'] == planned['name'] and not any(r['name'] == planned['name'] for r in observations):
                terminal = folder / 'http-terminal.json'
                actual = ct.load(terminal) if terminal.is_file() else {}
                observations.append(dict(issued[-1], status='censored', right_censored=True,
                    actual=actual, metric_eligible=False, censor_reason=stop_reason))
            break
    journal_schedule = [dict(row, issued=any(i['name'] == row['name'] for i in issued)) for row in schedule]
    result = dict(started_monotonic=started, absolute_aggregate_deadline=started + limits['all_requests_seconds'],
                  observations=observations, summary=censoring_summary(observations, journal_schedule),
                  stop_reason=stop_reason, can_continue=stop_reason is None and len(observations) == 12)
    if transfer_case is not None:
        result['expert_transfers'] = dict(startup=startup, requested_case=transfer_case)
        result['initial_read_endpoint'] = initial_read
        result['initial_read_evidence'] = initial_evidence
        result['ready_log_end_byte'] = ready_log_end
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


def natural_descendant_drain(supervisor):
    """ECHILD permits cleanup; qualification additionally forbids forced drain."""
    custody = supervisor.get("custody", {})
    entries = custody.get("entries")
    return (ct.successful_process(supervisor) and type(supervisor.get("returncode")) is int
        and custody.get("errors") == [] and custody.get("reporting_errors") == []
        and isinstance(entries, list) and bool(entries)
        and all(isinstance(row, dict) and row.get("signals") == []
            and row.get("reaped") is True and type(row.get("exit_code")) is int
            and row["exit_code"] == 0 for row in entries)
        and ("natural_drain" not in custody or
            custody["natural_drain"].get("outcome") == "echild_without_signals"))


def lifecycle_assessment(observed, supervisor, post_idle, *, transfer_required=False):
    reasons = []
    if not ct.drained(supervisor):
        reasons.append('unknown authenticated descendant custody')
    if not ct.successful_process(supervisor):
        reasons.append('worker failed/timed out')
    if transfer_required or observed.get('transfer_controls_revision') == 2:
        if observed.get('transfer_controls_revision') != 2:
            reasons.append('explicit transfer lifecycle contract missing/changed')
        custody = supervisor.get('custody', {})
        root = custody.get('root_identity') or {}
        if not (custody.get('root_session_ready') is True
                and integer(root.get('pid'), 1) and root.get('pid') == root.get('pgrp') == root.get('session')):
            reasons.append('post-setsid authenticated worker custody missing')
        if not natural_descendant_drain(supervisor):
            reasons.append('forced/signalled/unknown descendant cleanup is not a qualified HTTP lifecycle')
        stop, capture = observed.get('planned_http_stop', {}), observed.get('capture', {})
        if not (stop.get('schema') == 'ferrule.storage-demand.owned-http-stop-result.v1'
                and stop.get('disposition') == 'expected_planned_stop'
                and type(stop.get('actual_returncode')) is int and stop['actual_returncode'] in (0, -signal.SIGTERM)
                and stop['actual_returncode'] == capture.get('returncode')
                and capture.get('planned_stop_validated') is True and capture.get('stop_reason') == 'planned_stop'
                and not capture.get('error') and not capture.get('shutdown_timeout')):
            reasons.append('typed owned planned-stop evidence missing/invalid')
    if observed.get('status') != 'complete':
        reasons.append(observed.get('error') or 'incomplete lifecycle')
    if not observed.get('closure', {}).get('final', {}).get('can_continue'):
        reasons.append('owner close missing/pending/unknown')
    if post_idle.get('status') != 'verified' or post_idle.get('all_eight_idle') is not True:
        reasons.append('independent post-idle not established')
    return dict(can_continue=not reasons, stop_reasons=reasons,
                resource_contract_complete='unverified', universal_final_release_verified=False)


def recover_observations(directory, schedule):
    """Preserve admitted slots even when a worker left partial JSON artifacts."""
    rows = []
    for planned in schedule:
        folder = Path(directory) / planned['name']
        errors = []
        def load(name):
            path = folder / name
            if not path.is_file():
                return None
            try:
                value = ct.load(path)
                require(isinstance(value, dict), 'request artifact is not an object')
                if name != 'http-terminal.json':
                    require(value.get('name') == planned['name'], 'request artifact identity differs')
                return value
            except (OSError, ValueError, TypeError) as error:
                errors.append(name + ': ' + type(error).__name__ + ': ' + str(error))
                return None
        observed = load('observed.json')
        if observed is not None:
            rows.append(observed)
            continue
        issued = load('issued.json')
        if issued is not None or (folder / 'issued.json').exists() or (folder / 'observed.json').exists():
            actual = load('http-terminal.json') or {}
            row = dict(issued or planned, issued=True, status='censored', metric_eligible=False,
                actual=actual, censor_reason='worker lost before valid observation artifact')
            if errors:
                row['recovery_errors'] = errors
            rows.append(row)
    return rows


def recover_requests(directory, schedule):
    """After lost worker output, admitted records never become unissued."""
    return censoring_summary(recover_observations(directory, schedule), schedule)
