"""Collector-first hooks; reuse the sealed supervisor, observer and model wrapper."""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import secrets
import time
from types import SimpleNamespace

import pair_contract as contract
import instrumentation_scope as scope

baseline, shared = contract.baseline, contract.shared
ct, guard, observer = baseline.ct, baseline.guard, baseline.observer
SCHEMA = 'ferrule.paper.collector-handoff.v1'


def readonly(path, value):
    temporary = path.with_name(path.name+'.pending')
    ct.write(temporary, value)
    temporary.chmod(0o400)
    try:
        os.link(temporary, path)  # atomic publication, exclusive/no overwrite
    finally:
        temporary.unlink()
    return ct.sha(path)


def sealed(path, digest):
    ct.require(path.is_file() and not path.is_symlink() and path.stat().st_mode & 0o222 == 0,
               'Missing read-only handoff receipt')
    ct.require(ct.sha(path) == digest, 'Handoff receipt changed')
    return ct.load(path)


def probe_grant(value, pair):
    scope.grant_scope(value.get('gpu_uuids', []), shared.UUIDS, value)
    ct.require(value.get('model_gpu_uuids') == list(shared.UUIDS), 'Model UUID order differs')
    ct.require(value.get('run_id') == pair.name
        and value.get('prepared_sha256') == ct.sha(pair/'prepared.json'),
        'Probe grant must bind this fresh prepared run')
    return value


def validate_locks(locks, coordinator):
    ct.require(len(locks) == 8 and list(locks) == sorted(locks), 'Eight sorted probe locks required')
    for uuid, lock in locks.items():
        ct.validate_owner_lock(lock, uuid, coordinator)


def descendant(expected, ancestor):
    """Kernel ancestry, never command-name matching or numeric PID signals."""
    observer.verify_root(expected)
    observer.verify_root(ancestor)
    current, seen = expected, set()
    while current['pid'] not in seen:
        if observer.root_key(current) == observer.root_key(ancestor):
            observer.verify_root(expected)
            return
        seen.add(current['pid'])
        ct.require(current['ppid'] > 1, 'Target outside authenticated collector subtree')
        current = observer.proc_identity(current['ppid'])
    raise ValueError('Invalid collector ancestry')


def live_auth(manifest):
    auth = manifest['authenticated']
    for key in ('worker', 'helper', 'coordinator'):
        ct.require(guard.identity(auth[key]['pid']) == auth[key], 'Live custody identity changed: '+key)
    case = Path(manifest['directory'])
    journal = ct.load(case/'custody.json')
    ct.require(journal.get('containment_verified') is True
        and journal.get('method') == 'linux-subreaper-waitid'
        and journal.get('launch_id') == auth['launch_id']
        and journal.get('helper_identity') == auth['helper']
        and observer.root_key(journal['root_identity']) == observer.root_key(auth['worker']),
        'Live authenticated custody journal differs')
    launch = sealed(case/'worker-launch.json', manifest['worker_launch_sha256'])
    ct.require(launch['coordinator'] == auth['coordinator']
        and launch['owner_locks'] == manifest['locks'], 'Worker launch binding differs')
    validate_locks(manifest['locks'], auth['coordinator']['pid'])
    ct.require(ct.sha(case/'baseline.json') == launch['baseline_sha256'], 'Original B0 changed')
    pair = case.parent.parent.parent
    grant = probe_grant(ct.load(pair/'probe-grant.json'), pair)
    ct.require(grant == manifest['grant'] and list(manifest['locks']) == grant['gpu_uuids'],
               'Probe grant/lock scope differs')
    return auth


def target_manifest(case, command):
    payload = ct.load(case/'worker-launch.json')
    auth = guard.authenticate(payload, ct.sha(case/'worker-launch.json'))
    pair = case.parent.parent.parent
    grant = probe_grant(ct.load(pair/'probe-grant.json'), pair)
    manifest = dict(schema=SCHEMA, directory=str(case), authenticated=auth,
        locks=payload['owner_locks'], grant=grant, command=command,
        command_sha256=ct.digest(command), binary_sha256=ct.sha(command[0]),
        worker_launch_sha256=ct.sha(case/'worker-launch.json'),
        nonce=secrets.token_hex(32), created_monotonic=time.monotonic())
    live_auth(manifest)
    return readonly(case/'collector-manifest.json', manifest)


def sample(case, note, phase):
    """Same bounded NVML query/retry as the maintained observer; missing != zero."""
    with (case/'collector-samples.jsonl').open('a') as stream:
        def emit(row):
            import json
            stream.write(json.dumps(dict(row, phase=phase), allow_nan=False)+'\n')
            stream.flush()
        try:
            row = observer.all_gpu_sample(shared.UUIDS[0], note, emit)
            emit(dict(kind='sample', gpu=row))
            return row
        except BaseException as error:
            emit(dict(kind='sample_failed', error=str(error), gpu=None, monotonic=time.monotonic()))
            raise


def state_from_manifest(manifest):
    case = Path(manifest['directory'])
    grant = manifest['grant']
    state = scope.BaselineScope(grant['gpu_uuids'], shared.UUIDS, grant)
    state.raw_idle(ct.load(case/'prelaunch.json')['observations'][-1])
    return state


def load_wrapper():
    path = ct.ROOT/'scripts/_profile_capture.py'
    spec = importlib.util.spec_from_file_location('_existing_profile_capture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class CollectorMonitor(observer.Monitor):
    """Only gate setup and delta policy differ; collect/host/PSS loop is inherited."""
    def start(self):
        self.handoff_error = None
        try:
            case = self.directory
            manifest = ct.load(case/'collector-manifest.json')
            self.manifest = manifest
            self.state = state_from_manifest(manifest)
            self.collector = dict(self.root, launch_id=self.authenticated['launch_id'])
            self.state.collector_started(self.collector, self.started)
            deadline = self.started + scope.PROBE_SECONDS
            while not (case/'collector-target-ready.json').exists():
                live_auth(manifest)
                observer.verify_root(self.root)
                ct.require(time.monotonic() < deadline, 'Collector target handshake timeout')
                time.sleep(.05)
            self.target = ct.load(case/'collector-target-ready.json')
            self.check_target()
            rows = []
            for i in range(3):
                self.check_target()
                rows.append(sample(case, self.note, 'collector_only_B1'))
                ct.require(time.monotonic() <= deadline, 'Collector B1 deadline exceeded')
                # Validate each device before waiting; do not hide an anomalous card.
                devices = scope.devices(rows[-1], self.state.inventory)
                ct.require(all(d['gpu_bytes'] <= scope.PROBE_CEILING_BYTES for d in devices.values()),
                           'Instrumentation ceiling exceeded')
                if i < 2:
                    time.sleep(2)
            frozen = self.state.freeze(rows, self.collector)
            self.check_target()
            digest = readonly(case/'collector-baseline.json', dict(frozen, samples=rows))
            self.state.admit_model(shared.UUIDS, self.collector)
            self.baseline = self.state.anchor
            readonly(case/'model-release.json', dict(schema=SCHEMA,
                manifest_sha256=ct.sha(case/'collector-manifest.json'),
                baseline_sha256=digest, target=self.target,
                collector=self.root, nonce=manifest['nonce'],
                launch_id=self.authenticated['launch_id'], model_gpu_uuids=list(shared.UUIDS)))
            super().start()
        except BaseException as error:
            self.handoff_error = str(error)
            ct.write(self.directory/'collector-handoff-failure.json', dict(error=str(error)))
            raise

    def check_target(self):
        live_auth(self.manifest)
        ct.require(self.target['manifest_sha256'] == ct.sha(self.directory/'collector-manifest.json')
            and self.target['nonce'] == self.manifest['nonce'], 'Target handshake replay/mismatch')
        descendant(self.target['identity'], self.root)

    def gpu_delta_check(self, gpu):
        # Return violations so the inherited collector persists the offending row
        # BEFORE its normal fatal/direct-child shutdown path runs.
        delta = observer.device_deltas(gpu, self.baseline, self.uuids)
        try:
            live_auth(self.manifest)
            observer.verify_root(self.root)
            self.state.check_model_sample(gpu, self.collector)
        except (ValueError, OSError) as error:
            delta['violations'].append(str(error))
        return delta

    def finish(self):
        if self.thread.ident is None:
            return dict(status='blocked', errors=[self.handoff_error or 'Collector not started'],
                        phase_memory=None, gpu_peak_bytes=None, tree_pss_peak_bytes=None)
        return super().finish()


def shutdown(server, case, timeout):
    """Model stop -> authenticated live target -> B1 -> target release -> export.

    Monitor.finish already stopped sampling. This bounded hook deliberately
    samples cleanup itself, while retaining the SAME supervisor's custody.
    """
    started = time.monotonic()
    deadline = started + timeout
    (case/'profile-stop').touch(exist_ok=False)
    failure = None
    try:
        manifest = ct.load(case/'collector-manifest.json')
        if (case/'model-release.json').exists():
            while not (case/'collector-model-ended.json').exists():
                live_auth(manifest)
                ct.require(server.poll() is None and time.monotonic() < deadline-12,
                           'Model cleanup handshake timeout/root exit')
                time.sleep(.1)
            ended = ct.load(case/'collector-model-ended.json')
            gate = ct.load(case/'model-release.json')
            ct.require(ended['target'] == gate['target'] and ended['manifest_sha256'] == ct.sha(case/'collector-manifest.json'),
                       'Model exit receipt binding differs')
            descendant(gate['target']['identity'], gate['collector'])
            wrapper = ct.load(case/'profile-server.json')
            ct.require(wrapper.get('returncode') == 0 and ended['wrapper_sha256'] == ct.sha(case/'profile-server.json'),
                       'Direct model child not successfully reaped')
            frozen = sealed(case/'collector-baseline.json', gate['baseline_sha256'])
            samples = []
            while len(samples) < 2:
                live_auth(manifest)
                descendant(gate['target']['identity'], gate['collector'])
                ct.require(time.monotonic() < deadline-30, 'B1 return deadline exceeded')
                row = sample(case, manifest['grant']['note'], 'after_model_B1')
                returned = scope.returned_to_instrumentation(row, frozen['instrumentation'], shared.UUIDS)
                samples = samples+[row] if returned else []
                if len(samples) < 2:
                    time.sleep(.25)
            readonly(case/'collector-after-model.json', dict(status='b1_returned',
                samples=samples, target=gate['target'], wrapper_sha256=ended['wrapper_sha256']))
    except BaseException as error:
        failure = type(error).__name__+': '+str(error)
        ct.write(case/'collector-cleanup-failure.json', dict(error=failure))
    finally:
        # Cooperative exit only; never signal a PID discovered by NVML.
        readonly(case/'collector-release.json', dict(status='release_for_drain', failure=failure,
            manifest_sha256=ct.sha(case/'collector-manifest.json')))
    try:
        server.wait(timeout=max(.1, deadline-time.monotonic()))
        result = dict(timed_out=False, returncode=server.returncode,
                      status='collector_exited_descendants_require_custodian', seconds=time.monotonic()-started)
    except baseline.subprocess.TimeoutExpired:
        result = dict(timed_out=True, status='export_timeout_requires_custodian', seconds=time.monotonic()-started)
    if failure:
        raise ValueError(failure)
    return result


def bind(module, pair, note):
    """Private hooks only. Original run/worker/requests remain byte-identical."""
    case = pair/'runs/P1/session'
    def grant():
        return probe_grant(ct.load(pair/'probe-grant.json'), pair)

    @contextmanager
    def locks(selected):
        g = grant()
        ct.require(tuple(selected) == shared.UUIDS, 'Model scope changed')
        try:
            with scope.probe_locks(g['gpu_uuids'], selected, g, ct.exclusive_owner_lock) as held:
                readonly(case/'probe-locks.json', dict(probe_gpu_uuids=g['gpu_uuids'],
                    model_gpu_uuids=list(selected), locks=held))
                yield held
        finally:
            # Unlocking on error is mandatory; release success is a separate fact.
            result = ct.load(case.parent/'execution.json') if (case.parent/'execution.json').exists() else {}
            ct.write(case.parent/'probe-lock-release.json', dict(local_contexts_closed=True,
                probe_gpu_uuids=g['gpu_uuids'], model_gpu_uuids=list(selected),
                authenticated_drained=result.get('authenticated_drained'),
                original_b0_verified=result.get('after', {}).get('status') == 'verified'))

    def idle(config, selected, owner_note, anchor=None):
        g = grant()  # reject BEFORE any GPU query
        result = observer.idle(config, selected, owner_note, anchor)
        try:
            for row in result.get('observations', []):
                rows = {d['uuid']:d for d in row['devices']}
                ct.require(set(rows) == set(g['gpu_uuids']), 'Probe inventory differs')
                ct.require(all(rows[u]['index'] == i for i,u in enumerate(selected)), 'Model UUID/index mismatch')
            if result['status'] == 'verified':
                rows = result['baseline']
                ct.require(all(d['gpu_bytes'] == 0 for d in rows.values()),
                           'Instrumentation custody requires original all-eight B0=0')
        except ValueError as error:
            result.update(status='blocked', error=str(error))
        return result

    module.shared = SimpleNamespace(**{**vars(module.shared), 'owner_locks':locks, 'validate_locks':validate_locks})
    module.observer = SimpleNamespace(**{**vars(observer), 'Monitor':CollectorMonitor, 'idle':idle})
    module.http.shutdown_owned_group = lambda server, timeout: shutdown(server, case, timeout)
    original_ready = module.http.wait_ready
    def ready(specification, startup_guard, timeout):
        deadline = ct.load(case/'collector-manifest.json')['created_monotonic'] + timeout
        remaining = deadline-time.monotonic()
        ct.require(remaining > 0, 'Startup deadline includes collector B1')
        return original_ready(specification, startup_guard, remaining)
    module.http.wait_ready = ready
    return module
