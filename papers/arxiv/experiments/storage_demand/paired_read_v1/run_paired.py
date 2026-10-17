#!/usr/bin/env python3
"""Narrow ten-lifecycle coordinator over the maintained TP1 worker; no new process harness."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import protocol, analysis, quality
from runnerhelpers.evidence_v2 import ct, require, integer
from runnerhelpers.closure_v2 import evidence
from runnerhelpers import execution_v2 as ex
from newversionedprotocol import run_pilot as pilot

ENTRY = Path(__file__).resolve()
CONTRACT_KEYS = ('schema', 'directory', 'trial', 'trial_order', 'mode', 'request_ceiling', 'schedule',
    'limits', 'execution_path', 'execution_sha256', 'candidate_path', 'candidate_sha256',
    'uuid', 'eight_uuids', 'owner_lock', 'owner_note', 'granted', 'spec', 'baseline',
    'outer_deadline', 'pilot_deadline', 'adapter_pins')


def spec_for(candidate, uuid, mode):
    case = protocol.case_for_mode(mode)
    environment = dict(ex.runner.cpu_environment(), **case['environment'])
    environment.update(CUDA_VISIBLE_DEVICES=uuid, NVIDIA_VISIBLE_DEVICES=uuid,
        CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    argv = [str(ct.absolute(candidate['candidate']['binary']).resolve()), *case['argv_after_binary']]
    spec = dict(engine='ferrule', argv=argv, environment=environment,
        model='qwen35-storage-demand-paper', url='http://127.0.0.1:18369/v1/completions')
    if 'transfer_policy' in case:
        spec['expert_transfer_case'] = {k: case[k] for k in ('read_schedule', 'transfer_policy', 'read_options')}
        spec['frozen_cli_sha256'] = candidate['candidate']['binary_sha256']
    return spec


def worker_contract(payload):
    require(payload.get('schema') == 'ferrule.storage-demand.paired-read-worker.v1', 'unknown paired worker')
    require(all(k in payload for k in CONTRACT_KEYS) and payload.get('contract_sha256')
            == ct.digest({k: payload[k] for k in CONTRACT_KEYS}), 'sealed paired worker contract changed')
    require(payload['trial_order'] == protocol.trial_order() and payload['trial'] in payload['trial_order']
            and payload['trial'].endswith('-' + payload['mode']) and payload['request_ceiling'] == 12,
            'paired trial/order/mode drift')
    require(payload['schedule'] == ct.load(HERE / 'requests.json')['schedule'], 'entire frozen schedule changed')
    require(payload['limits'] == ct.load(protocol.PROTOCOL_PATH)['limits'], 'absolute limits changed')
    pilot.validate_inventory(payload['eight_uuids'], payload['uuid'])
    require(payload['granted'] is True, 'renewed owner grant required')
    for key in ('execution', 'candidate'):
        require(ct.sha(payload[key + '_path']) == payload[key + '_sha256'], key + ' receipt changed')
    execution = protocol.validate_prepared(payload['execution_path'])
    ct.check_pins(payload['adapter_pins'])
    require(payload['adapter_pins'] == execution['artifact_pins'], 'paired workflow registration stale')
    candidate = protocol.validate_candidate(payload['candidate_path'])
    require(payload['spec'] == spec_for(candidate, payload['uuid'], payload['mode']), 'candidate-bound mode/spec changed')
    pilot.telemetry.authorize(True, payload['uuid'], payload['owner_note'], 'device_exclusive_local_custody')
    ct.validate_owner_lock(payload['owner_lock'], payload['uuid'], payload['coordinator']['pid'])
    return candidate


@contextmanager
def maintained_worker_adapter():
    """Process-local entry/contract specialization, never edits the baseline module or data."""
    names = dict(HERE=HERE, ENTRY=ENTRY, worker_contract=worker_contract)
    originals = {name: getattr(pilot, name) for name in names}
    mapped = ex.library_maps
    def timing_maps(root, pins):
        ex.observer.verify_root(root)
        raw = Path(f'/proc/{root["pid"]}/environ').read_text()
        environment = dict(item.split('=', 1) for item in raw.split('\0') if '=' in item)
        require(environment.get('FERRULE_PROFILE_NVTX') == '0', 'actual timing NVTX must be OFF')
        return mapped(root, pins)
    try:
        for name, value in names.items():
            setattr(pilot, name, value)
        ex.library_maps = timing_maps
        yield
    finally:
        ex.library_maps = mapped
        for name, value in originals.items():
            setattr(pilot, name, value)


def read_ledger_assessment(observed, mode):
    try:
        require(mode in protocol.MODES, 'disabled/unknown read mode')
        attempts = observed.get('closure', {}).get('attempts', [])
        require(attempts, 'actual final owner ledger missing')
        state, read = evidence(attempts[-1]['summary']['physical']['expert_read_pipeline'])
        require(state in ('observed', 'verified') and isinstance(read, dict), 'read ledger unobserved/unknown')
        options = protocol.case_for_mode(mode)['read_options']
        quality.read_observation(read, options, effective_required=True, final=True)
        return dict(can_continue=True, final_read_ledger=read, logical_host_reservation_only=True)
    except (ValueError, KeyError, TypeError) as error:
        return dict(can_continue=False, stop_reason=str(error), final_read_ledger=None)


def worker(launch_hash):
    raw = sys.stdin.read()
    payload = json.loads(raw)
    previous = sys.stdin
    try:
        sys.stdin = io.StringIO(raw)
        with maintained_worker_adapter():
            code = pilot.worker(launch_hash)
    finally:
        sys.stdin = previous
    folder = Path(payload['directory'])
    observed = ct.load(folder / 'lifecycle.json')
    assessment = read_ledger_assessment(observed, payload['mode'])
    ct.write(folder / 'read-ledger-assessment.json', assessment)
    return 0 if code == 0 and assessment['can_continue'] else 1


def supervise(payload, seconds):
    with maintained_worker_adapter():
        return pilot.supervise(payload, seconds)


def can_admit(deadline, lifecycles=1, now=None):
    require(lifecycles in (1, 2), 'unregistered admission size')
    now = time.monotonic() if now is None else now
    limits = ct.load(protocol.PROTOCOL_PATH)['limits']
    return now + lifecycles * limits['outer_lifecycle_seconds'] + limits['study_cleanup_reserve_seconds'] <= deadline


def run(args):
    execution = protocol.validate_prepared(args.prepared)
    candidate = protocol.validate_candidate(args.candidate)
    pilot.validate_inventory(args.eight_uuids, args.gpu_uuid)
    require(args.sole_gpu_owner_granted is True, 'renewed explicit sole-owner grant required')
    pilot.telemetry.authorize(True, args.gpu_uuid, args.owner_note, 'device_exclusive_local_custody')
    require(args.output.parent.resolve() == HERE and re.fullmatch(r'artifacts-[A-Za-z0-9_-]{1,64}', args.output.name),
            'new artifacts-NAME directly in paired_read_v1 required')
    directory = args.output.resolve()
    directory.mkdir(exist_ok=False)
    limits, schedule = execution['limits'], ct.load(HERE / 'requests.json')['schedule']
    deadline = time.monotonic() + limits['pilot_seconds']
    report = dict(schema='ferrule.storage-demand.paired-read-actual.v1', status='incomplete',
        started=ex.stamp(), pilot_deadline=deadline, order=protocol.expected_order(), lifecycles=[],
        scheduled_blocks=5, scheduled_starts=10, scheduled_requests=120, scheduled_measured_qa=50,
        candidate_sha256=ct.sha(args.candidate), execution_sha256=ct.sha(args.prepared),
        physical_uuid=args.gpu_uuid, eight_uuids=args.eight_uuids, retries=0, replacements=0,
        physical_nas_backend_bytes=None, profile_lifecycles=0)
    ct.write(directory / 'registration.json', dict(report, schedule=schedule, limits=limits,
        candidate_path=str(args.candidate.resolve()), execution_path=str(args.prepared.resolve())))
    try:
        with ct.exclusive_owner_lock(args.gpu_uuid) as lock:
            ct.validate_owner_lock(lock, args.gpu_uuid, os.getpid())
            for trial in execution['trial_order']:
                mode = trial.split('-', 1)[1]
                require(can_admit(deadline), 'insufficient full lifecycle plus cleanup reserve; no shortened/replacement trial')
                outer = min(time.monotonic() + limits['outer_lifecycle_seconds'], deadline - limits['study_cleanup_reserve_seconds'])
                folder = directory / trial
                folder.mkdir(exist_ok=False)
                pilot.idle(folder, 'coordinator-pre-idle', args.eight_uuids,
                           min(time.monotonic() + limits['coordinator_preidle_seconds'], outer))
                protocol.validate_prepared(args.prepared)
                candidate = protocol.validate_candidate(args.candidate)
                anchor = pilot.baseline(args.eight_uuids, args.gpu_uuid, args.owner_note)
                ct.write(folder / 'baseline.json', anchor)
                payload = dict(schema='ferrule.storage-demand.paired-read-worker.v1', directory=str(folder),
                    trial=trial, trial_order=execution['trial_order'], mode=mode, request_ceiling=12,
                    schedule=schedule, limits=limits, execution_path=str(args.prepared.resolve()),
                    execution_sha256=ct.sha(args.prepared), candidate_path=str(args.candidate.resolve()),
                    candidate_sha256=ct.sha(args.candidate), uuid=args.gpu_uuid, eight_uuids=args.eight_uuids,
                    owner_lock=lock, owner_note=args.owner_note, granted=True,
                    spec=spec_for(candidate, args.gpu_uuid, mode), baseline=anchor,
                    outer_deadline=outer, pilot_deadline=deadline - limits['study_cleanup_reserve_seconds'],
                    adapter_pins=execution['artifact_pins'])
                payload['contract_sha256'] = ct.digest({k: payload[k] for k in CONTRACT_KEYS})
                ct.write(folder / 'launch.json', payload)
                pilot.remaining(outer, limits['supervisor_seconds'] + limits['supervisor_call_overhead_seconds']
                    + limits['coordinator_postidle_seconds'] + limits['coordinator_finalization_seconds'], 'supervisor and cleanup')
                receipt = supervise(payload, limits['supervisor_seconds'])
                ct.write(folder / 'supervisor.json', receipt)
                post = dict(status='blocked', all_eight_idle=False, reason='unknown custody; no stabilization')
                if ct.drained(receipt):
                    try:
                        post = pilot.idle(folder, 'coordinator-post-idle', args.eight_uuids,
                            min(time.monotonic() + limits['coordinator_postidle_seconds'], outer), receipt)
                    except Exception as error:
                        post = dict(status='blocked', all_eight_idle=False, error=str(error))
                observed = ct.load(folder / 'lifecycle.json') if (folder / 'lifecycle.json').is_file() else {}
                assessment = ex.lifecycle_assessment(observed, receipt, post)
                ledger = read_ledger_assessment(observed, mode)
                if not ledger['can_continue']:
                    assessment['can_continue'] = False
                    assessment['stop_reasons'].append(ledger['stop_reason'])
                summary = ex.recover_requests(folder, schedule)
                summary['measured_qa']['request_only_eligible_success'] = summary['measured_qa']['eligible_success']
                if not assessment['can_continue']:
                    summary['measured_qa']['eligible_success'] = 0
                row = dict(trial=trial, block=trial.split('-')[0], mode=mode, summary=summary,
                    assessment=assessment, read_ledger=ledger, supervisor=receipt, post_idle=post, lifecycle=observed)
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
                report['lifecycles'].append(dict(trial=trial, block=trial.split('-')[0], mode=trial.split('-', 1)[1],
                    summary=summary, status='unissued_or_incomplete', stop_reason=report.get('stop_reason')))
        report['lifecycles'].sort(key=lambda row: execution['trial_order'].index(row['trial']))
        report['analysis'] = analysis.summarize(report)
        report.update(ended=ex.stamp(), resource_contract_complete='unverified',
                      universal_final_release_verified=False, scientific_acceptance=False)
        report['artifact_pins'] = ct.inventory(p for p in directory.rglob('*') if p.is_file())
        ct.write(directory / 'paired-final.json', report)
    return report


def main(argv=None):
    ct.require_cpu_python()
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--_worker':
        require(len(argv) == 2, 'sealed worker launch hash required')
        return worker(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'check', 'run'), nargs='?', default='plan')
    parser.add_argument('--prepared', type=Path, default=protocol.PREPARED_PATH)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--eight-uuids', nargs=8)
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    parser.add_argument('--owner-note')
    args = parser.parse_args(argv)
    if args.mode == 'plan':
        value = protocol.plan()
    elif args.mode == 'check':
        value = protocol.validate_prepared(args.prepared)
        if args.candidate:
            protocol.validate_candidate(args.candidate)
    else:
        require(args.candidate is not None and args.output is not None, 'actual bound candidate and exclusive output required')
        value = run(args)
    print(json.dumps(dict(status=value['status'], gpu_runs=0 if args.mode != 'run' else 'see paired-final.json')))
    return 0 if args.mode != 'run' or value['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
