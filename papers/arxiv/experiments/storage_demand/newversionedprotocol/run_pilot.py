#!/usr/bin/env python3
"""Narrow repeated-QA TP1 pilot. plan/check/seal-execution are CPU-only; run is gated."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STORAGE = HERE.parent
sys.path.insert(0, str(STORAGE))
from newversionedprotocol import protocol
from runnerhelpers import execution_v2 as ex
from runnerhelpers.evidence_v2 import ct, require, UUID_RE, validate_schedule
from runnerhelpers.closure_v2 import parse_resource_shutdown_summaries
from runnerhelpers.mountstats_v2 import mountstats_delta

import launch_guard
import telemetry
import gpu_idle_window_v2 as idle_window
import test_vllm_qwen35 as supervisor

ENTRY = Path(__file__).resolve()
DEFAULT_EXECUTION = HERE / 'execution-prepared-20261014-01.json'
EXECUTION_SCHEMA = 'ferrule.storage-demand.execution-prepared.v2'


def execution_paths(cpu_prepared=protocol.PREPARED_PATH):
    paths = [ct.ROOT / p for p in protocol.IMPORTS]
    paths += [ct.ROOT / 'scripts/gpu_idle_window_v2.py', ct.ROOT / 'scripts/run_pinned_jobs.py',
              Path(sys.executable), Path('/usr/bin/env'), Path('/usr/bin/nvidia-smi'),
              protocol.PREPARED_PATH, Path(cpu_prepared), protocol.CASE_PATH, protocol.PROTOCOL_PATH,
              HERE / 'requests-v2.json', HERE / 'new-baseline-build-requirements.json', HERE / 'EXECUTION.md',
              *protocol.REVIEW_DOCS]
    paths += list((STORAGE / 'runnerhelpers').glob('*.py')) + list(HERE.glob('*.py'))
    paths += list((STORAGE / 'tests').glob('*.py'))
    return paths


def plan(cpu_prepared=protocol.PREPARED_PATH):
    preparation_error = None
    try:
        protocol.validate_prepared(cpu_prepared)
    except ValueError as error:
        preparation_error = str(error)
    status = ('cpu_plan_workflow_unsealed_candidate_unbound_not_authorized' if preparation_error
              else 'cpu_prepared_candidate_unbound_not_authorized')
    return dict(schema=EXECUTION_SCHEMA, status=status, preparation_error=preparation_error,
        protocol_sha256=ct.sha(protocol.PROTOCOL_PATH), case_sha256=ct.sha(protocol.CASE_PATH),
        schedule_sha256=ct.sha(HERE / 'requests-v2.json'),
        historical_prepared_sha256=ct.sha(protocol.PREPARED_PATH),
        cpu_prepared_path=str(Path(cpu_prepared).resolve()), cpu_prepared_sha256=ct.sha(cpu_prepared),
        trial_order=protocol.load_protocol()['trial_order'], limits=protocol.load_protocol()['limits'],
        starts=5, scheduled_requests=60, scheduled_measured_qa=25, retries=0,
        worker_entry=str(ENTRY), request_ceiling=12, candidate_binary=None,
        gpu_authorized=False, gpu_queries=0, gpu_locks=0, gpu_runs=0,
        artifact_pins=ct.inventory(execution_paths(cpu_prepared)),
        prerequisites=['Owner-supplied new final-candidate receipt, validated current build/full40/fresh/cold/source relation',
                       'Explicit renewed sole-owner grant, literal selected physical UUID and exact ordered eight-UUID inventory',
                       'Pinned runtime libraries covering actual Ready/after-request loaded mappings, passive readable NFS mountstats',
                       'New exclusive artifact directory; authenticated worker, normal capture shutdown, ECHILD and post all-eight idle'],
        interpretation='CPU execution seal is not candidate binding, GPU authorization, measurement or scientific acceptance')


def validate_execution(path):
    value = ct.load(path)
    require(value.get('schema') == EXECUTION_SCHEMA and value.get('status') == 'cpu_prepared_candidate_unbound_not_authorized',
            'execution preparation missing/unknown')
    require(value.get('digest') == ct.digest({k: v for k, v in value.items() if k != 'digest'}), 'execution seal digest changed')
    cpu_prepared = value.get('cpu_prepared_path', str(protocol.PREPARED_PATH))
    protocol.validate_prepared(cpu_prepared)
    if 'cpu_prepared_path' in value:
        require(value.get('cpu_prepared_sha256') == ct.sha(cpu_prepared), 'CPU preparation selection changed')
    ct.check_pins(value['artifact_pins'])
    require(value['artifact_pins'] == ct.inventory(execution_paths(cpu_prepared)), 'execution workflow inventory changed; new seal required')
    require(value['protocol_sha256'] == ct.sha(protocol.PROTOCOL_PATH)
            and value['case_sha256'] == ct.sha(protocol.CASE_PATH)
            and value['schedule_sha256'] == ct.sha(HERE / 'requests-v2.json'), 'fixed protocol/case/schedule changed')
    require(value['trial_order'] == ['L1', 'L2', 'L3', 'L4', 'L5'] and value['limits'] == protocol.load_protocol()['limits'],
            'trial/deadline drift')
    return value


def validate_candidate(path):
    """Never bind a candidate here; re-read the supplied seal and all actual receipts."""
    value = ct.load(path)
    require(value.get('sealed_final_candidate') is True
            and value.get('status') == 'quality_passed_cpu_prepared_not_authorized', 'candidate is unbound/quality pending')
    require(value.get('digest') == ct.digest({k: v for k, v in value.items() if k != 'digest'}), 'candidate seal digest changed')
    expected = protocol.prepare_candidate(value.get('candidate', {}))
    require({k: v for k, v in value.items() if k != 'digest'} == expected, 'candidate build/quality/case binding changed')
    return value


def validate_inventory(uuids, selected):
    require(isinstance(uuids, list) and len(uuids) == len(set(uuids)) == 8
            and all(UUID_RE.fullmatch(uuid or '') for uuid in uuids), 'exact ordered eight physical UUIDs required')
    require(UUID_RE.fullmatch(selected or '') and selected in uuids, 'selected physical UUID missing')


def spec_for(candidate, uuid):
    case = ct.load(protocol.CASE_PATH)
    environment = dict(ex.runner.cpu_environment(), **case['environment'])
    environment.update(CUDA_VISIBLE_DEVICES=uuid, NVIDIA_VISIBLE_DEVICES=uuid,
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    argv = [str(ct.absolute(candidate['candidate']['binary']).resolve()), *case['argv_after_binary']]
    flags = dict(zip(argv[3::2], argv[4::2]))
    require(flags.get('--host') == '127.0.0.1' and flags.get('--port') == '18369', 'unreviewed endpoint')
    return dict(engine='ferrule', argv=argv, environment=environment,
                model=flags['--served-model-name'], url='http://127.0.0.1:18369/v1/completions')


def remaining(deadline, seconds, label):
    require(time.monotonic() + seconds <= deadline, 'insufficient absolute reserve: ' + label)


def idle(directory, label, uuids, deadline, custody=None):
    remaining(deadline, idle_window.SECONDS, label)
    result = idle_window.observe(uuids, custody=custody)
    ct.write(Path(directory) / (label + '.json'), result)
    require(result.get('status') == 'verified' and result.get('all_eight_idle') is True, 'all-eight idle failed: ' + label)
    require(time.monotonic() <= deadline, 'idle exceeded absolute phase deadline')
    return result


def baseline(uuids, selected, note):
    sample = ex.observer.all_gpu_sample(selected, note)
    devices = sample['devices']
    require([d['uuid'] for d in devices] == uuids and [d['index'] for d in devices] == list(range(8)),
            'ordered physical UUID inventory changed')
    require(not sample['apps'] and all(d['gpu_bytes'] == 0 and d['utilization'] == 0 for d in devices),
            'all-eight baseline not idle')
    require(next(d['total_bytes'] for d in devices if d['uuid'] == selected)
            == ct.BASE['resources']['gpu_expected_total_bytes'], 'selected physical capacity changed')
    return {d['uuid']: dict(d) for d in devices}


class PilotMonitor(ex.observer.Monitor):
    def gpu_delta_check(self, gpu):
        value = super().gpu_delta_check(gpu)
        require(not any(app['uuid'] not in self.uuids for app in gpu['apps']),
                'foreign nonselected GPU application; no foreign signal permitted')
        return value


def worker_contract(payload):
    require(payload.get('schema') == 'ferrule.storage-demand.pilot-worker.v2', 'unregistered TP1 worker')
    keys = ('schema', 'directory', 'trial', 'trial_order', 'request_ceiling', 'schedule', 'limits',
            'execution_path', 'execution_sha256', 'candidate_path', 'candidate_sha256',
            'uuid', 'eight_uuids', 'owner_lock', 'owner_note', 'granted', 'spec', 'baseline',
            'outer_deadline', 'pilot_deadline', 'adapter_pins')
    require(payload.get('contract_sha256') == ct.digest({k: payload[k] for k in keys}), 'sealed TP1 worker contract changed')
    require(payload['trial_order'] == ['L1', 'L2', 'L3', 'L4', 'L5'] and payload['trial'] in payload['trial_order']
            and payload['request_ceiling'] == 12, 'unregistered lifecycle/order/ceiling')
    validate_schedule(payload['schedule'])
    require(payload['schedule'] == ct.load(HERE / 'requests-v2.json')['schedule'], 'entire request order/content changed')
    require(payload['limits'] == protocol.load_protocol()['limits'], 'worker absolute limits changed')
    validate_inventory(payload['eight_uuids'], payload['uuid'])
    require(payload['granted'] is True, 'renewed sole-owner authorization missing')
    telemetry.authorize(True, payload['uuid'], payload['owner_note'], 'device_exclusive_local_custody')
    for key in ('execution', 'candidate'):
        require(ct.sha(payload[key + '_path']) == payload[key + '_sha256'], key + ' receipt changed')
    execution = validate_execution(payload['execution_path'])
    ct.check_pins(payload['adapter_pins'])
    require(payload['adapter_pins'] == execution['artifact_pins'], 'worker/helper registration not current')
    candidate = validate_candidate(payload['candidate_path'])
    require(payload['spec'] == spec_for(candidate, payload['uuid']), 'candidate-bound argv/environment changed')
    ct.validate_owner_lock(payload['owner_lock'], payload['uuid'], payload['coordinator']['pid'])
    return candidate


def worker(launch_hash):
    ct.require_cpu_python()
    payload = json.load(sys.stdin)
    authenticated = launch_guard.authenticate(payload, launch_hash)
    candidate = worker_contract(payload)
    directory = Path(payload['directory']).resolve()
    require(directory.parent.parent == HERE and directory.name == payload['trial'], 'new pilot artifact scope required')
    require(os.getpid() == os.getpgrp(), 'worker fresh session missing')
    limits = payload['limits']
    started = time.monotonic()
    result = dict(status='failed', trial=payload['trial'], started_monotonic=started,
                  authenticated_worker=authenticated, physical_nas_backend_bytes=None)
    capture = monitor = server = root = None
    shutdown_sample = None
    reader = dict(authenticated['worker'])
    paths = [row['path'] for row in ct.load(STORAGE / 'checkpoint-metadata.json')['shards']]
    sample_after = ex.nfs_sampler(directory, reader, None, paths)
    try:
        idle(directory, 'worker-pre-idle', payload['eight_uuids'],
             min(started + limits['worker_preidle_seconds'], payload['outer_deadline'], payload['pilot_deadline']))
        pre = sample_after('prelaunch')
        launch = time.monotonic()
        target_deadline = min(launch + limits['target_lifetime_seconds'],
                              payload['outer_deadline'] - limits['shutdown_seconds'] - limits['worker_finalization_seconds'],
                              payload['pilot_deadline'] - limits['shutdown_seconds'] - limits['worker_finalization_seconds'])
        remaining(target_deadline, limits['startup_seconds'] + limits['all_requests_seconds'], 'target admission')
        capture = ex.launch_capture(directory, payload['spec'], candidate['candidate']['binary_sha256'],
                                    target_deadline - launch, limits['shutdown_seconds'])
        startup_deadline = min(launch + limits['startup_seconds'], target_deadline)
        target, root, target_report = ex.capture_root(capture, directory, startup_deadline, authenticated)
        server = ex.CapturedServer(capture, directory, root)
        # The observed capture root is an authenticated worker's direct child;
        # its maintained run_target Popen owns exactly this server grandchild.
        bridge = dict(authenticated, worker=launch_guard.identity(target['pid']))
        require(bridge['worker']['birth_ticks'] == target['birth_ticks'], 'capture birth changed')
        config = copy.deepcopy(ct.BASE)
        config['limits']['case_seconds'] = limits['target_lifetime_seconds']
        monitor = PilotMonitor(directory, server, config, bridge, [payload['uuid']],
                               payload['owner_note'], payload['baseline'],
                               environment_keys=('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'FERRULE_LOG', 'FERRULE_LOG_FORMAT'))
        monitor.start()
        ready = ex.http.wait_ready(payload['spec'], ex.runner.StartupGuard(server, monitor),
                                   startup_deadline - time.monotonic())
        require(ready <= startup_deadline, 'absolute startup deadline exceeded')
        monitor.mark_ready(ready)
        result.update(server_identity=root, capture_identity=target, physical_uuid=payload['uuid'],
                      startup_seconds=ready - launch, ready=ex.stamp(), target_deadline=target_deadline)
        environment = dict(item.split('=', 1) for item in Path(f'/proc/{root["pid"]}/environ').read_text().split('\0') if '=' in item)
        require(all(environment.get(key) == payload['uuid'] for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES')),
                'actual server UUID visibility unbound')
        require(ct.sha(Path(f'/proc/{root["pid"]}/exe')) == candidate['candidate']['binary_sha256'], 'actual server executable changed')
        sample = ex.nfs_sampler(directory, reader, root, paths)
        ready_nfs = sample('ready')
        result['startup_nfs'] = mountstats_delta(pre, ready_nfs)
        require(result['startup_nfs']['status'] == 'shared_client_delta', 'startup mount sample/reset')
        result['startup_logical_counters'] = ex.logical_counter_window(ex.log_lines(directory / 'server.log'))
        runtime = ct.load(candidate['candidate']['runtime_pin_manifest'])['pins']
        maps = ex.library_maps(root, runtime)
        ct.write(directory / 'ready-library-maps.json', maps)
        phase_pre = sample('requests-pre')
        def check():
            require(time.monotonic() < target_deadline, 'target/outer/pilot deadline')
            ex.observer.verify_root(root)
            ex.observer.verify_root(target)
            require(not monitor.failed.is_set(), 'observer/custody failure: ' + str(monitor.errors))
            live = parse_resource_shutdown_summaries(ex.log_lines(directory / 'server.log'))
            require(not live['sticky_unknown'], 'explicit unknown physical custody')
            require(not live['errors'], 'resource summary serialization/parse failure')
            require(not live['attempts'], 'owner shutdown already started; no new admission')
            ct.validate_owner_lock(payload['owner_lock'], payload['uuid'], payload['coordinator']['pid'])
            ct.check_pins(payload['adapter_pins'])
            ct.check_pins(candidate['candidate']['artifact_pins'])
            for key in ('execution', 'candidate'):
                require(ct.sha(payload[key + '_path']) == payload[key + '_sha256'], key + ' receipt changed')
        result['requests'] = ex.serial_requests(directory, payload['schedule'], payload['spec'], limits,
            target_deadline, payload['outer_deadline'], payload['pilot_deadline'], check, sample,
            lambda: ex.process_io(root))
        phase_post = sample('requests-post')
        result['requests_nfs'] = mountstats_delta(phase_pre, phase_post)
        after_maps = ex.library_maps(root, runtime)
        ct.write(directory / 'after-requests-library-maps.json', after_maps)
        require(after_maps['libraries'] == maps['libraries'], 'loaded libraries changed')
        require(result['requests']['can_continue'] and result['requests_nfs']['status'] == 'shared_client_delta',
                result['requests']['stop_reason'] or 'request phase NFS reset')
        check()
        result['status'] = 'requests_complete'
    except (Exception, KeyboardInterrupt) as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if capture is not None:
            try:
                try:
                    if server is not None:
                        shutdown_sample = sample_after('shutdown-pre')
                except Exception as error:
                    result['shutdown_sample_error'] = str(error)
                if monitor is not None:
                    result['resources'] = monitor.finish()
                ex.stop_capture(directory)
                capture.wait(timeout=limits['shutdown_seconds'] + 2)
                result['capture'] = ct.load(directory / 'profile-server.json')
                try:
                    post = sample_after('shutdown-post')
                except Exception as error:
                    post = None
                    result['shutdown_post_sample_error'] = str(error)
                result['shutdown_nfs'] = mountstats_delta(shutdown_sample, post)
                lines = ex.log_lines(directory / 'server.log')
                result['closure'] = ex.owner_summary(lines, root, payload['uuid'])
                result['server_log_sha256'] = ct.sha(directory / 'server.log')
                result['whole_log_token_join'] = ex.join_token_trace_http(ex.parse_token_trace(lines),
                    result.get('requests', {}).get('observations', []), lines)
                require(result['capture'].get('pid') == (root or {}).get('pid')
                        and result['capture'].get('returncode') == 0 and capture.returncode == 0
                        and not result['capture'].get('shutdown_timeout'), 'normal capture/server shutdown missing')
                require(result['closure']['final']['can_continue'], 'owner close missing/pending/unknown')
                require(result['shutdown_nfs']['status'] == 'shared_client_delta', 'shutdown mount sample/reset')
                require(result.get('resources', {}).get('status') == 'observed_within_targets', 'resource observer failed')
                require(len(result['whole_log_token_join']) == 12
                        and all(row['metric_eligible'] for row in result['whole_log_token_join']), 'final drained token trace reconciliation failed')
                if result['status'] == 'requests_complete':
                    result['status'] = 'complete'
            except (Exception, KeyboardInterrupt) as error:
                result['cleanup_error'] = type(error).__name__ + ': ' + str(error)
                result['status'] = 'cleanup_unverified'
                # A live capture/server remains solely in maintained subreaper custody.
                if monitor is not None and monitor.thread.is_alive():
                    monitor.stop.set()
                ex.stop_capture(directory)
        result['ended'] = ex.stamp()
        if result['ended']['monotonic'] > min(payload['outer_deadline'], payload['pilot_deadline']):
            result.update(status='deadline_overrun', overrun=True)
        ct.write(directory / 'lifecycle.json', result)
    return 0 if result['status'] == 'complete' else 1


def supervise(payload, seconds):
    payload, launch_hash = launch_guard.seal(payload, payload['directory'])
    environment = dict(ex.runner.cpu_environment(), FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
    command = [sys.executable, '-S', '-B', str(ENTRY), '--_worker', launch_hash]
    if payload.get('spec', {}).get('expert_transfer_case') is not None:
        command += ['--layout', payload['execution_path']]
    argv = ct.isolated_argv(command, environment)
    try:
        return supervisor.run_process(argv, seconds, payload)
    except BaseException as error:
        result = getattr(error, 'supervisor_result', {})
        result.setdefault('error', type(error).__name__ + ': ' + str(error))
        return result


def run(args):
    # All read-only qualification checks precede the output reservation, lock and any query.
    execution = validate_execution(args.execution_prepared)
    candidate = validate_candidate(args.candidate)
    validate_inventory(args.eight_uuids, args.gpu_uuid)
    require(args.sole_gpu_owner_granted is True, 'renewed explicit sole-owner grant required')
    telemetry.authorize(True, args.gpu_uuid, args.owner_note, 'device_exclusive_local_custody')
    require(args.output.parent.resolve() == HERE and re.fullmatch(r'artifacts-[A-Za-z0-9_-]{1,64}', args.output.name),
            'new artifacts-NAME directory directly in newversionedprotocol required')
    directory = args.output.resolve()
    directory.mkdir(exist_ok=False)
    limits = execution['limits']
    schedule = ct.load(HERE / 'requests-v2.json')['schedule']
    started = time.monotonic()
    pilot_deadline = started + limits['pilot_seconds']
    report = dict(schema='ferrule.storage-demand.actual-pilot.v2', status='incomplete',
                  started=ex.stamp(), pilot_deadline=pilot_deadline, lifecycles=[],
                  scheduled_starts=5, scheduled_requests=60, scheduled_measured_qa=25,
                  candidate_sha256=ct.sha(args.candidate), execution_sha256=ct.sha(args.execution_prepared),
                  physical_uuid=args.gpu_uuid, eight_uuids=args.eight_uuids, owner_note=args.owner_note,
                  physical_nas_backend_bytes=None, retries=0)
    ct.write(directory / 'registration.json', dict(report, schedule=schedule, limits=limits,
                                                  candidate_path=str(args.candidate.resolve()),
                                                  execution_path=str(args.execution_prepared.resolve())))
    try:
        with ct.exclusive_owner_lock(args.gpu_uuid) as lock:
            ct.validate_owner_lock(lock, args.gpu_uuid, os.getpid())
            for trial in execution['trial_order']:
                remaining(pilot_deadline, limits['outer_lifecycle_seconds'], 'entire next lifecycle; no shortened/replacement trial')
                outer = min(time.monotonic() + limits['outer_lifecycle_seconds'], pilot_deadline)
                folder = directory / trial
                folder.mkdir(exist_ok=False)
                idle(folder, 'coordinator-pre-idle', args.eight_uuids,
                     min(time.monotonic() + limits['coordinator_preidle_seconds'], outer))
                validate_execution(args.execution_prepared)
                candidate = validate_candidate(args.candidate)
                anchor = baseline(args.eight_uuids, args.gpu_uuid, args.owner_note)
                ct.write(folder / 'baseline.json', anchor)
                payload = dict(schema='ferrule.storage-demand.pilot-worker.v2', directory=str(folder),
                    trial=trial, trial_order=execution['trial_order'], request_ceiling=12, schedule=schedule, limits=limits,
                    execution_path=str(args.execution_prepared.resolve()), execution_sha256=ct.sha(args.execution_prepared),
                    candidate_path=str(args.candidate.resolve()), candidate_sha256=ct.sha(args.candidate),
                    uuid=args.gpu_uuid, eight_uuids=args.eight_uuids, owner_lock=lock, owner_note=args.owner_note,
                    granted=True, spec=spec_for(candidate, args.gpu_uuid), baseline=anchor,
                    outer_deadline=outer, pilot_deadline=pilot_deadline, adapter_pins=execution['artifact_pins'])
                payload['contract_sha256'] = ct.digest(payload)
                ct.write(folder / 'launch.json', payload)
                remaining(outer, limits['supervisor_seconds'] + limits['supervisor_call_overhead_seconds']
                          + limits['coordinator_postidle_seconds'] + limits['coordinator_finalization_seconds'], 'supervisor and cleanup')
                receipt = supervise(payload, limits['supervisor_seconds'])
                ct.write(folder / 'supervisor.json', receipt)
                post = dict(status='blocked', all_eight_idle=False, reason='unknown custody; no stabilization')
                if ct.drained(receipt):
                    try:
                        post = idle(folder, 'coordinator-post-idle', args.eight_uuids,
                            min(time.monotonic() + limits['coordinator_postidle_seconds'], outer, pilot_deadline), receipt)
                    except Exception as error:
                        post = dict(status='blocked', all_eight_idle=False, error=str(error))
                observed = ct.load(folder / 'lifecycle.json') if (folder / 'lifecycle.json').is_file() else {}
                assessment = ex.lifecycle_assessment(observed, receipt, post)
                summary = ex.recover_requests(folder, schedule)
                summary['measured_qa']['request_only_eligible_success'] = summary['measured_qa']['eligible_success']
                if not assessment['can_continue']:
                    summary['measured_qa']['eligible_success'] = 0
                row = dict(trial=trial, summary=summary, assessment=assessment,
                           supervisor=receipt, post_idle=post, lifecycle=observed)
                report['lifecycles'].append(row)
                ct.write(folder / 'assessment.json', row)
                require(time.monotonic() <= outer, 'outer lifecycle deadline overrun; STOP')
                require(assessment['can_continue'], '; '.join(assessment['stop_reasons']))
            report['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        report['stop_reason'] = type(error).__name__ + ': ' + str(error)
    finally:
        recorded = {row['trial'] for row in report['lifecycles']}
        for trial in execution['trial_order']:
            if trial not in recorded:
                summary = ex.recover_requests(directory / trial, schedule)
                summary['measured_qa']['request_only_eligible_success'] = summary['measured_qa']['eligible_success']
                summary['measured_qa']['eligible_success'] = 0
                report['lifecycles'].append(dict(trial=trial, summary=summary, status='unissued_or_incomplete',
                                                stop_reason=report.get('stop_reason')))
        report['lifecycles'].sort(key=lambda row: execution['trial_order'].index(row['trial']))
        report['denominators'] = {key: sum(row['summary'][key] for row in report['lifecycles'])
                                 for key in ('scheduled', 'issued', 'terminal', 'successful', 'failed', 'censored', 'unissued')}
        report['measured_qa'] = {key: sum(row['summary']['measured_qa'][key] for row in report['lifecycles'])
                                for key in ('scheduled', 'success', 'eligible_success')}
        report.update(ended=ex.stamp(), resource_contract_complete='unverified',
                      universal_final_release_verified=False, scientific_acceptance=False)
        report['artifact_pins'] = ct.inventory(p for p in directory.rglob('*') if p.is_file())
        ct.write(directory / 'pilot-final.json', report)
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ct.require_cpu_python()
    if argv and argv[0] == '--_worker':
        require(len(argv) == 2, 'sealed worker hash required')
        return worker(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'check', 'seal-execution', 'run'), nargs='?', default='plan')
    parser.add_argument('--execution-prepared', type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--cpu-prepared', type=Path, default=protocol.PREPARED_PATH)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--eight-uuids', nargs=8)
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    parser.add_argument('--owner-note')
    args = parser.parse_args(argv)
    if args.mode == 'plan':
        value = plan(args.cpu_prepared)
    elif args.mode == 'seal-execution':
        require(args.output is not None and args.output.parent.resolve() == HERE, 'exclusive new seal inside newversionedprotocol required')
        value = plan(args.cpu_prepared)
        require(value['preparation_error'] is None, 'CPU workflow seal stale; reviewed new protocol seal required first')
        value['digest'] = ct.digest(value)
        ct.write(args.output, value)
    elif args.mode == 'check':
        value = validate_execution(args.execution_prepared)
        if args.candidate:
            validate_candidate(args.candidate)
    else:
        require(args.candidate is not None and args.output is not None, 'run requires owner-supplied bound candidate and exclusive output')
        value = run(args)
    print(json.dumps(dict(status=value['status'], candidate_bound=False if args.candidate is None else args.mode != 'plan',
                          preparation_error=value.get('preparation_error'),
                          gpu_runs=0 if args.mode != 'run' else 'see pilot-final.json'), sort_keys=True))
    return 0 if args.mode != 'run' or value['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
