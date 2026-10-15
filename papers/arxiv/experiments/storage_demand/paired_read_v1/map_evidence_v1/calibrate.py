#!/usr/bin/env python3
"""One explicitly authorized Ready-only calibration, never paired-study data."""
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1.final_adapter_v3 import binding
from runnerhelpers import execution_v2 as ex, library_maps_v1 as maps
from runnerhelpers.evidence_v2 import ct, require
from newversionedprotocol import run_pilot as pilot

ENTRY = Path(__file__).resolve()
OUT = HERE / 'calibration-map-ready-20261015-02'
OLD = HERE / 'artifacts-v3-final-20261015-01'
QUALIFICATION = binding.evidence.ROOT
CLI_SHA = '7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01'
NOTE = 'Exactly one parent-authorized Ready-only map calibration; no inference, paired data, retries or foreign signals'
LIMITS = dict(startup_seconds=120, total_seconds=240, shutdown_seconds=45)


def prepare():
    ct.require_cpu_python()
    os.umask(0o077)
    OUT.mkdir(mode=0o700, exist_ok=False)
    review = binding.review(QUALIFICATION)
    require(len(review['quality_review']['gates']) == 9, 'all nine quality gates required')
    maps.private_write(OUT / 'qualification-review.json', review)
    binary = QUALIFICATION / 'staged-final/cli'
    runtime = ct.load(QUALIFICATION / 'runtime-library-pins-final.json')
    policy = maps.role_policy(binary, CLI_SHA, runtime, 'cli')
    maps.private_write(OUT / 'role-policy.json', policy)
    old_launch = ct.load(OLD / 'P1-blocking/worker-launch.json')
    spec = old_launch['spec']
    case = binding.original.case_for_mode('blocking')
    uuids = review['uuid_inventory']
    require(spec['argv'] == [str(binary), *case['argv_after_binary']], 'literal V3 Blocking argv drift')
    environment = dict(ex.runner.cpu_environment(), **case['environment'])
    environment.update(CUDA_VISIBLE_DEVICES=uuids[0], NVIDIA_VISIBLE_DEVICES=uuids[0],
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    require(spec['environment'] == environment and old_launch['eight_uuids'] == uuids
            and old_launch['uuid'] == uuids[0], 'same V3 environment/UUID cohort required')
    preservation = ct.inventory(p for p in OLD.rglob('*') if p.is_file())
    sources = [HERE / n for n in binding.SOURCE_FILES]
    inputs = ct.inventory([*sources, OUT / 'qualification-review.json', OUT / 'role-policy.json',
                           OLD / 'P1-blocking/worker-launch.json', QUALIFICATION / 'evidence-manifest.json'])
    value = dict(schema='ferrule.runtime-map-ready-calibration.v1', spec=spec, binary_sha256=CLI_SHA,
                 eight_uuids=uuids, uuid=uuids[0], limits=LIMITS, input_pins=inputs,
                 historical_artifact_pins=preservation, quality_gates=9,
                 required_pin_count=len(policy['required']), paired_data=False, trial_replacement=False,
                 inference_requests=0, retries=0, profiling_captures=0, cache_flushes=0)
    maps.private_write(OUT / 'prepared.json', value)
    return dict(status='cpu_prepared_only', quality_gates=9, required_pins=len(policy['required']),
                gpu_queries=0, server_launches=0)


@contextmanager
def readiness_log(directory, spec):
    """Record maintained model-list availability GETs, never bodies/auth headers."""
    build = ex.http.urllib.request.build_opener
    expected = spec['url'].replace('/v1/completions', '/v1/models')
    path = Path(directory) / 'readiness-get.jsonl'
    with path.open('x') as log:
        def factory(*args, **kwargs):
            opener = build(*args, **kwargs)
            original = opener.open
            def opened(url, *a, **kw):
                require(url == expected and not kw.get('data'), 'only the maintained non-generative readiness GET allowed')
                row = dict(method='GET', url=expected, started=maps.stamp(),
                           kind='model-list availability, not inference or a conditioning probe')
                try:
                    response = original(url, *a, **kw)
                    row['status'] = response.status
                    return response
                except Exception as error:
                    row['error_type'] = type(error).__name__
                    raise
                finally:
                    row['ended'] = maps.stamp()
                    log.write(json.dumps(row) + '\n')
                    log.flush()
            opener.open = opened
            return opener
        ex.http.urllib.request.build_opener = factory
        try:
            yield
        finally:
            ex.http.urllib.request.build_opener = build


def worker(launch_hash):
    ct.require_cpu_python()
    os.umask(0o077)
    payload = json.load(sys.stdin)
    authenticated = pilot.launch_guard.authenticate(payload, launch_hash)
    directory = Path(payload['directory'])
    require(directory == OUT / 'owned-ready' and payload['limits'] == LIMITS,
            'wrong calibration directory/limits')
    require(os.getpid() == os.getpgrp(), 'fresh worker session missing')
    ct.check_pins(payload['input_pins'])
    for uuid, lock in zip(payload['eight_uuids'], payload['all_uuid_locks']):
        ct.validate_owner_lock(lock, uuid, authenticated['coordinator']['pid'])
    require(len(payload['all_uuid_locks']) == 8, 'all-eight locks required')
    prepared = ct.load(OUT / 'prepared.json')
    require(payload['spec'] == prepared['spec'] and payload['binary_sha256'] == CLI_SHA,
            'unbound calibration executable/spec')
    result = dict(status='failed', ready=False, inference_requests=0, paired_data=False,
                  trial_replacement=False, limits=LIMITS, started=maps.stamp())
    capture = monitor = root = parent = None
    try:
        launch = time.monotonic()
        target_deadline = payload['deadline'] - 70
        require(launch + LIMITS['startup_seconds'] < target_deadline, 'insufficient startup/cleanup reserve')
        capture = ex.launch_capture(directory, payload['spec'], CLI_SHA,
                                    target_deadline - launch, LIMITS['shutdown_seconds'])
        startup_deadline = launch + LIMITS['startup_seconds']
        parent, root, report = ex.capture_root(capture, directory, startup_deadline, authenticated)
        server = ex.CapturedServer(capture, directory, root)
        bridge = dict(authenticated, worker=pilot.launch_guard.identity(parent['pid']))
        require(bridge['worker']['birth_ticks'] == parent['birth_ticks'], 'capture birth changed')
        config = dict(ct.BASE, limits=dict(ct.BASE['limits'], case_seconds=240))
        monitor = pilot.PilotMonitor(directory, server, config, bridge, [payload['uuid']],
            NOTE, payload['baseline'], environment_keys=('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'))
        monitor.start()
        with readiness_log(directory, payload['spec']):
            ready = ex.http.wait_ready(payload['spec'], ex.runner.StartupGuard(server, monitor),
                                       startup_deadline - time.monotonic())
        require(ready <= startup_deadline, 'absolute startup deadline exceeded')
        monitor.mark_ready(ready)
        result.update(ready=True, startup_seconds=ready - launch, ready_at=maps.stamp(),
                      server_identity=root, capture_identity=parent)
        with ex.absolute_alarm(target_deadline):
            snapshot = maps.capture(directory / 'ready-library-evidence', root, parent,
                                    CLI_SHA, ex.observer.proc_identity)
            result['snapshot'] = str(directory / 'ready-library-evidence/snapshot.json')
            require(not snapshot['errors'], 'raw map acquisition failed')
            environment = dict(item.split('=', 1) for item in
                Path(f'/proc/{root["pid"]}/environ').read_text().split('\0') if '=' in item)
            require(environment.get('FERRULE_PROFILE_NVTX') == '0'
                    and all(environment.get(k) == payload['uuid'] for k in
                            ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES')), 'actual safe environment contract differs')
            result['safe_environment_verified'] = True
            require(not monitor.failed.is_set(), 'owned observer/custody failed')
            result['status'] = 'captured_ready_only'
    except (Exception, KeyboardInterrupt) as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        if root is not None and not result.get('snapshot'):
            result['failure_map_snapshot'] = maps.capture(directory / 'failed-library-evidence',
                root, parent, CLI_SHA, ex.observer.proc_identity)
    finally:
        if capture is not None:
            try:
                shutdown_deadline = time.monotonic() + LIMITS['shutdown_seconds']
                ex.stop_capture(directory)
                if monitor is not None:
                    result['resources'] = monitor.finish()
                capture.wait(timeout=max(0, shutdown_deadline - time.monotonic()))
                result['capture'] = ct.load(directory / 'profile-server.json')
                require(result['capture']['returncode'] == 0
                        and not result['capture'].get('shutdown_timeout')
                        and time.monotonic() <= shutdown_deadline, 'owned server shutdown failed/timed out')
            except Exception as error:
                result['cleanup_error'] = type(error).__name__ + ': ' + str(error)
        result['ended'] = maps.stamp()
        maps.private_write(directory / 'ready-only.json', result)
    return 0 if result['status'] == 'captured_ready_only' and not result.get('cleanup_error') else 1


def run(granted):
    require(granted is True, 'explicit one-calibration sole GPU owner grant required')
    ct.require_cpu_python()
    os.umask(0o077)
    prepared = ct.load(OUT / 'prepared.json')
    ct.check_pins(prepared['input_pins'])
    ct.check_pins(prepared['historical_artifact_pins'])
    ct.check_pins(ct.load(OUT / 'role-policy.json')['pins'])
    require(ct.sha(prepared['spec']['argv'][0]) == CLI_SHA, 'immutable CLI changed')
    pilot.validate_inventory(prepared['eight_uuids'], prepared['uuid'])
    pilot.telemetry.authorize(True, prepared['uuid'], NOTE, 'device_exclusive_local_custody')
    started = time.monotonic()
    deadline = started + LIMITS['total_seconds']
    # Exclusive marker denies re-entry, including after failed preflight or a lost coordinator.
    maps.private_write(OUT / 'launch-attempt.json', dict(started=maps.stamp(), deadline=deadline,
                       maximum_server_launches=1, limits=LIMITS, authorization=NOTE))
    result = dict(status='failed', paired_data=False, trial_replacement=False, inference_requests=0,
                  retries=0, profiling_captures=0, cache_flushes=0, limits=LIMITS)
    supervision = {}
    try:
        with ExitStack() as stack:
            locks = [stack.enter_context(ct.exclusive_owner_lock(uuid)) for uuid in prepared['eight_uuids']]
            for uuid, lock in zip(prepared['eight_uuids'], locks):
                ct.validate_owner_lock(lock, uuid, os.getpid())
            maps.private_write(OUT / 'locks-held.json', locks)
            pilot.idle(OUT, 'pre-idle', prepared['eight_uuids'], deadline)
            baseline = pilot.baseline(prepared['eight_uuids'], prepared['uuid'], NOTE)
            directory = OUT / 'owned-ready'
            directory.mkdir(mode=0o700, exist_ok=False)
            payload = dict(prepared, directory=str(directory), all_uuid_locks=locks,
                           baseline=baseline, deadline=deadline)
            payload, launch_hash = pilot.launch_guard.seal(payload, directory)
            environment = dict(ex.runner.cpu_environment(), FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
            argv = ct.isolated_argv([sys.executable, '-S', '-B', str(ENTRY), '--_worker', launch_hash], environment)
            supervision = pilot.supervisor.run_process(argv, deadline - time.monotonic() - 25, payload)
            maps.private_write(OUT / 'supervisor.json', supervision)
            result['authenticated_echild'] = ct.drained(supervision)
            post = pilot.idle(OUT, 'post-idle', prepared['eight_uuids'], deadline, custody=supervision)
            result['all_eight_idle'] = post['status'] == 'verified'
            result['recorded_owners_absent'] = pilot.idle_window.no_recorded_owner(supervision)
        maps.private_write(OUT / 'locks-released.json', dict(context_exited=True))
        with ExitStack() as stack:
            reprobed = [stack.enter_context(ct.exclusive_owner_lock(uuid)) for uuid in prepared['eight_uuids']]
            maps.private_write(OUT / 'lock-reprobe.json', dict(all_eight_nonblocking_reacquired=len(reprobed) == 8))
        result['all_eight_locks_released_reprobed'] = True
        observed = ct.load(OUT / 'owned-ready/ready-only.json')
        result.update(ready=observed['ready'], startup_seconds=observed.get('startup_seconds'),
                      server_identity=observed.get('server_identity'), capture_identity=observed.get('capture_identity'))
        if observed.get('snapshot'):
            snapshot = ct.load(observed['snapshot'])
            policy = ct.load(OUT / 'role-policy.json')
            fixed = maps.validate_snapshot(snapshot, policy)
            legacy = maps.legacy_assess(Path(snapshot['raw_maps_path']).read_text(), policy['pins'])
            comparison = dict(raw_maps_sha256=snapshot['raw_maps_sha256'], original=legacy, fixed=fixed,
                              same_snapshot=True, offline_after_shutdown=True)
            maps.private_write(OUT / 'offline-comparison.json', comparison)
            result.update(original_status=legacy['status'], fixed_status=fixed['status'],
                          required_pin_check=fixed['required_pin_check'],
                          required_pin_count=len(policy['required']), observed_library_count=fixed['observed_library_count'],
                          ignored_deleted_nonlibraries=fixed.get('ignored_deleted_nonlibraries', []),
                          missing_required_paths=fixed['missing_required_paths'], deleted_library_paths=fixed['deleted_library_paths'])
        ct.check_pins(prepared['historical_artifact_pins'])
        ct.check_pins(prepared['input_pins'])
        result['old_v3_artifacts_unchanged'] = True
        require(time.monotonic() <= deadline, '240-second total calibration deadline exceeded')
        require(ct.successful_process(supervision) and ct.drained(supervision)
                and observed['status'] == 'captured_ready_only' and result.get('fixed_status') == 'passed',
                'calibration not established; no second launch permitted')
        result['status'] = 'ready_only_calibration_passed_not_inference_quality'
    except (Exception, KeyboardInterrupt) as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        result['wall_seconds'] = time.monotonic() - started
        result['physical_resource_release'] = 'unobserved; ECHILD/idle is not a physical-release proof'
        maps.private_write(OUT / 'FINAL.json', result)
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == '--_worker':
        return worker(argv[1])
    require(argv in (['prepare'], ['run', '--sole-gpu-owner-granted']), 'use prepare or explicitly authorized run')
    result = prepare() if argv == ['prepare'] else run(True)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] != 'failed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
