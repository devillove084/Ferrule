#!/usr/bin/env python3
"""Execute one preregistered arm per finite command using maintained lifecycle custody.

The exclusive attempt journal denies replay, replacement and resume after a lost
or failed attempt. One persisted monotonic deadline covers the entire study.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import current_binding_v2 as binding, protocol as prereg, run_paired as planned, analysis, quality
from runnerhelpers.evidence_v2 import ct, require
from runnerhelpers.closure_v2 import evidence
from runnerhelpers import execution_v2 as ex
from newversionedprotocol import run_pilot as pilot

ENTRY = Path(__file__).resolve()
NOTE = 'User renewed sole GPU owner authorization for actual paired_read_v1 five paired blocks; no probes/retries/profiling or foreign signals'
ORIGINAL_WORKER_CONTRACT = planned.worker_contract


def ledger(observed, mode):
    try:
        attempts = observed.get('closure', {}).get('attempts', [])
        require(attempts, 'actual owner shutdown summary omitted; read/resource contract unknown')
        state, read = evidence(attempts[-1]['summary']['physical']['expert_read_pipeline'])
        require(state in ('observed', 'verified') and isinstance(read, dict), 'actual read ledger missing/unknown')
        quality.read_observation(read, binding.raw_options(mode), effective_required=True, final=True)
        return dict(can_continue=True, final_read_ledger=read, logical_host_reservation_only=True,
                    physical_release='unobserved; no universal release claim')
    except (ValueError, KeyError, TypeError) as error:
        return dict(can_continue=False, stop_reason=str(error), final_read_ledger=None, resource_contract='unknown')


def contract(payload):
    candidate = ORIGINAL_WORKER_CONTRACT(payload)
    locks = payload.get('all_uuid_locks', [])
    require(len(locks) == 8, 'all eight authenticated local UUID locks required')
    for uuid, lock in zip(payload['eight_uuids'], locks):
        ct.validate_owner_lock(lock, uuid, payload['coordinator']['pid'])
    require(locks[payload['eight_uuids'].index(payload['uuid'])] == payload['owner_lock'], 'selected lock differs')
    return candidate


@contextmanager
def actual_adapter():
    changes = ((prereg, 'validate_prepared', binding.validate_execution),
               (prereg, 'validate_candidate', binding.validate_binding),
               (planned, 'ENTRY', ENTRY), (planned, 'worker_contract', contract),
               (planned, 'read_ledger_assessment', ledger))
    originals = [(module, name, getattr(module, name)) for module, name, value in changes]
    try:
        for module, name, value in changes:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)


@contextmanager
def coordinator_lock(directory):
    fd = os.open(Path(directory) / 'study-coordinator.lock', os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def registration(directory):
    value = ct.load(Path(directory) / 'registration.json')
    require(value.get('schema') == 'ferrule.storage-demand.actual-paired-registration.v2'
            and value.get('digest') == ct.digest({k: v for k, v in value.items() if k != 'digest'}), 'registration changed')
    require(value['boot_id'] == boot_id() and value['trial_order'] == prereg.trial_order()
            and value['order'] == prereg.expected_order() and value['limits'] == ct.load(prereg.PROTOCOL_PATH)['limits'],
            'boot/order/absolute limits changed; no clock reset')
    require(value['binding_sha256'] == ct.sha(value['binding_path'])
            and value['execution_sha256'] == ct.sha(value['execution_path']), 'candidate/workflow registration changed')
    binding.validate_execution(Path(value['execution_path']))
    binding.validate_binding(Path(value['binding_path']))
    require(value['owner_authorization'] is True and value['selected_uuid'] == value['eight_uuids'][0], 'owner/physical UUID changed')
    pilot.validate_inventory(value['eight_uuids'], value['selected_uuid'])
    return value


def next_trial(directory, registered):
    directory = Path(directory)
    require(not (directory / 'study-stop.json').exists() and not (directory / 'paired-final.json').exists(),
            'study permanently stopped/finalized; no resume or replacement')
    for trial in registered['trial_order']:
        folder = directory / trial
        if not folder.exists():
            return trial
        require((folder / 'attempt.json').is_file() and (folder / 'assessment.json').is_file(),
                'prior attempt lost/incomplete custody; permanent STOP, never replay')
        row = ct.load(folder / 'assessment.json')
        require(row['trial'] == trial and row['assessment']['can_continue'] is True,
                'prior attempt failed/unknown; no resume or replacement')
    raise ValueError('all ten planned arms already issued; no additional lifecycle')


def start(directory, granted):
    require(granted is True, 'explicit renewed sole-GPU-owner authorization required')
    execution = binding.validate_execution()
    candidate = binding.validate_binding()
    require(directory.parent.resolve() == HERE and re.fullmatch(r'artifacts-[A-Za-z0-9_-]{1,64}', directory.name),
            'new exclusive artifacts-NAME in paired_read_v1 required')
    directory.mkdir(exist_ok=False)
    now = time.monotonic()
    value = dict(schema='ferrule.storage-demand.actual-paired-registration.v2', started=ex.stamp(),
        pilot_deadline=now + 7200, boot_id=boot_id(), order=prereg.expected_order(), trial_order=prereg.trial_order(),
        limits=execution['limits'], schedule=ct.load(HERE / 'requests.json')['schedule'],
        binding_path=str(binding.BINDING_PATH), binding_sha256=ct.sha(binding.BINDING_PATH),
        execution_path=str(binding.EXECUTION_PATH), execution_sha256=ct.sha(binding.EXECUTION_PATH),
        selected_uuid=candidate['uuid_inventory'][0], eight_uuids=candidate['uuid_inventory'],
        owner_authorization=True, owner_note=NOTE, scheduled_blocks=5, scheduled_starts=10,
        scheduled_requests=120, scheduled_measured_qa=50, retries=0, replacements=0, profiling_captures=0,
        first_arm_is_first_planned_arm=True, extra_pilot_lifecycles=0, physical_nas_backend_bytes=None)
    ct.write(directory / 'registration.json', dict(value, digest=ct.digest(value)))
    return dict(status='registered', directory=str(directory), first_arm=prereg.trial_order()[0], deadline=value['pilot_deadline'])


def stop(directory, reason):
    path = Path(directory) / 'study-stop.json'
    if not path.exists():
        ct.write(path, dict(status='permanent_stop', reason=reason, recorded=ex.stamp(),
                           retries_permitted=False, remaining_slots='retain unissued/censored denominators'))


def execute_next(directory):
    directory = directory.resolve()
    with coordinator_lock(directory):
        registered = registration(directory)
        trial = next_trial(directory, registered)
        limits, schedule = registered['limits'], registered['schedule']
        require(planned.can_admit(registered['pilot_deadline']), 'absolute 7200s envelope lacks full next-arm plus reserve')
        folder = directory / trial
        folder.mkdir(exist_ok=False)
        outer = min(time.monotonic() + 650, registered['pilot_deadline'] - 100)
        ct.write(folder / 'attempt.json', dict(trial=trial, admission=ex.stamp(), outer_deadline=outer,
            pilot_deadline=registered['pilot_deadline'], registration_sha256=ct.sha(directory / 'registration.json'),
            coordinator=pilot.launch_guard.identity(os.getpid()), replay_permitted=False))
        receipt, observed, post, locks = {}, {}, dict(status='blocked', all_eight_idle=False), []
        mode = trial.split('-', 1)[1]
        try:
            pilot.telemetry.authorize(True, registered['selected_uuid'], NOTE, 'device_exclusive_local_custody')
            with ExitStack() as stack:
                for uuid in registered['eight_uuids']:
                    locks.append(stack.enter_context(ct.exclusive_owner_lock(uuid)))
                for uuid, lock in zip(registered['eight_uuids'], locks):
                    ct.validate_owner_lock(lock, uuid, os.getpid())
                ct.write(folder / 'locks-held.json', dict(locks=locks, coordinator=pilot.launch_guard.identity(os.getpid()),
                    all_eight_local=True, global_reservation=False))
                pilot.idle(folder, 'coordinator-pre-idle', registered['eight_uuids'], min(time.monotonic() + 20, outer))
                candidate = binding.validate_binding(Path(registered['binding_path']))
                execution = binding.validate_execution(Path(registered['execution_path']))
                anchor = pilot.baseline(registered['eight_uuids'], registered['selected_uuid'], NOTE)
                ct.write(folder / 'baseline.json', anchor)
                payload = dict(schema='ferrule.storage-demand.paired-read-worker.v1', directory=str(folder),
                    trial=trial, trial_order=registered['trial_order'], mode=mode, request_ceiling=12,
                    schedule=schedule, limits=limits, execution_path=registered['execution_path'],
                    execution_sha256=registered['execution_sha256'], candidate_path=registered['binding_path'],
                    candidate_sha256=registered['binding_sha256'], uuid=registered['selected_uuid'],
                    eight_uuids=registered['eight_uuids'], owner_lock=locks[0], owner_note=NOTE, granted=True,
                    spec=planned.spec_for(candidate, registered['selected_uuid'], mode), baseline=anchor,
                    outer_deadline=outer, pilot_deadline=registered['pilot_deadline'] - 100,
                    adapter_pins=execution['artifact_pins'], all_uuid_locks=locks)
                payload['contract_sha256'] = ct.digest({k: payload[k] for k in planned.CONTRACT_KEYS})
                ct.write(folder / 'launch.json', payload)
                pilot.remaining(outer, 530 + 28 + 20 + 32, 'supervisor and finite cleanup')
                with actual_adapter():
                    receipt = planned.supervise(payload, 530)
                ct.write(folder / 'supervisor.json', receipt)
                if ct.drained(receipt) and pilot.idle_window.no_recorded_owner(receipt):
                    post = pilot.idle(folder, 'coordinator-post-idle', registered['eight_uuids'],
                                      min(time.monotonic() + 20, outer), receipt)
                observed = ct.load(folder / 'lifecycle.json') if (folder / 'lifecycle.json').is_file() else {}
                assessment = ex.lifecycle_assessment(observed, receipt, post)
                read = ledger(observed, mode)
                if not read['can_continue']:
                    assessment['can_continue'] = False
                    assessment['stop_reasons'].append(read['stop_reason'])
                require(time.monotonic() <= outer, 'outer lifecycle deadline exceeded; permanent STOP')
                summary = ex.recover_requests(folder, schedule)
                summary['measured_qa']['request_only_eligible_success'] = summary['measured_qa']['eligible_success']
                if not assessment['can_continue']:
                    summary['measured_qa']['eligible_success'] = 0
                row = dict(trial=trial, block=trial.split('-')[0], mode=mode, summary=summary,
                    assessment=assessment, read_ledger=read, supervisor=receipt, post_idle=post, lifecycle=observed,
                    actual_http_token_ids_source='worker enqueue actual IDs joined to actual HTTP request ID, not retokenization',
                    outer_deadline=outer, coordinator_ended=ex.stamp())
                ct.write(folder / 'assessment.json', row)
                require(assessment['can_continue'], '; '.join(assessment['stop_reasons']))
            ct.write(folder / 'locks-released.json', dict(context_exited=True, all_eight_local=True))
            with ExitStack() as stack:
                reprobed = [stack.enter_context(ct.exclusive_owner_lock(uuid)) for uuid in registered['eight_uuids']]
                for uuid, lock in zip(registered['eight_uuids'], reprobed):
                    ct.validate_owner_lock(lock, uuid, os.getpid())
                ct.write(folder / 'locks-reprobed.json', dict(all_eight_nonblocking_reacquired=True,
                    context_exits_after_receipt=True, local_only=True))
            print('ARM ' + trial + ' complete: 12/12 requests, 5/5 measured QA; normal shutdown, ECHILD, all-eight idle', flush=True)
            return row
        except (Exception, KeyboardInterrupt) as error:
            reason = type(error).__name__ + ': ' + str(error)
            stop(directory, reason)
            if not (folder / 'assessment.json').exists():
                observed = ct.load(folder / 'lifecycle.json') if (folder / 'lifecycle.json').is_file() else observed
                summary = ex.recover_requests(folder, schedule)
                summary['measured_qa']['request_only_eligible_success'] = summary['measured_qa']['eligible_success']
                summary['measured_qa']['eligible_success'] = 0
                ct.write(folder / 'assessment.json', dict(trial=trial, block=trial.split('-')[0], mode=mode,
                    summary=summary, assessment=dict(can_continue=False, stop_reasons=[reason]), read_ledger=ledger(observed, mode),
                    supervisor=receipt, post_idle=post, lifecycle=observed, coordinator_error=reason))
            if not (folder / 'locks-released.json').exists():
                ct.write(folder / 'locks-released.json', dict(context_exited=True, all_eight_local=True,
                    physical_owner_release='unverified', descendant_custody_known=ct.drained(receipt)))
            print('ARM ' + trial + ' STOP: ' + reason, flush=True)
            return ct.load(folder / 'assessment.json')


def finalize(directory):
    directory = directory.resolve()
    with coordinator_lock(directory):
        registered = registration(directory)
        require(not (directory / 'paired-final.json').exists(), 'final receipt exclusive; never replace')
        rows = []
        schedule = registered['schedule']
        for trial in registered['trial_order']:
            folder = directory / trial
            if (folder / 'assessment.json').is_file():
                rows.append(ct.load(folder / 'assessment.json'))
            else:
                summary = ex.recover_requests(folder, schedule)
                summary['measured_qa']['request_only_eligible_success'] = summary['measured_qa']['eligible_success']
                summary['measured_qa']['eligible_success'] = 0
                rows.append(dict(trial=trial, block=trial.split('-')[0], mode=trial.split('-', 1)[1], summary=summary,
                    status='unissued_or_incomplete', stop_reason='prior STOP/incomplete attempt/finite envelope; no replacement'))
        complete = all(row.get('assessment', {}).get('can_continue') is True for row in rows)
        if (directory / 'study-stop.json').exists():
            complete = False
        require(complete or (directory / 'study-stop.json').exists() or time.monotonic() >= registered['pilot_deadline'] - 100,
                'do not finalize early without fixed STOP/deadline; planned arms remain')
        report = dict(schema='ferrule.storage-demand.paired-read-actual.v2', status='complete' if complete else 'stopped',
            order=registered['order'], lifecycles=rows, scheduled_blocks=5, scheduled_starts=10,
            scheduled_requests=120, scheduled_measured_qa=50, started=registered['started'], ended=ex.stamp(),
            pilot_deadline=registered['pilot_deadline'], candidate_sha256=registered['binding_sha256'],
            execution_sha256=registered['execution_sha256'], binding_path=registered['binding_path'],
            physical_uuid=registered['selected_uuid'], eight_uuids=registered['eight_uuids'],
            retries=0, replacements=0, extra_pilot_lifecycles=0, profile_lifecycles=0,
            physical_nas_backend_bytes=None, universal_final_release_verified=False,
            resource_contract_complete='unverified', scientific_acceptance=False)
        report['analysis'] = analysis.summarize(report)
        report['actual_completed_lifecycles'] = sum(row.get('assessment', {}).get('can_continue') is True for row in rows)
        report['actual_issued_lifecycles'] = sum((directory / trial / 'launch.json').is_file() for trial in registered['trial_order'])
        report['pipeline'] = [{key: row.get(key) for key in ('trial', 'mode', 'read_ledger')} for row in rows]
        report['http_quality_evidence'] = []
        for row in rows:
            requests = row.get('lifecycle', {}).get('requests', {}).get('observations', [])
            report['http_quality_evidence'].append(dict(trial=row['trial'], requests=[dict(
                name=request['name'], status=request.get('status'), metric_eligible=request.get('metric_eligible'),
                measured=request.get('measured'), actual=request.get('actual'), actual_token_id_join=request.get('trace'),
                nfs=request.get('nfs'), logical_counters=request.get('logical_counters')) for request in requests]))
        attempted = [row for row in rows if row.get('supervisor')]
        if attempted:
            last = attempted[-1]
            report['final_authenticated_echild'] = ct.drained(last['supervisor'])
            report['final_recorded_owners_absent'] = pilot.idle_window.no_recorded_owner(last['supervisor'])
            report['final_all_eight_idle'] = last.get('post_idle', {}).get('status') == 'verified'
            report['final_idle_path'] = str(directory / last['trial'] / 'coordinator-post-idle.json')
            report['final_lock_release_path'] = str(directory / last['trial'] / 'locks-released.json')
            report['final_lock_reprobe_path'] = str(directory / last['trial'] / 'locks-reprobed.json')
        if (directory / 'study-stop.json').is_file():
            report['stop'] = ct.load(directory / 'study-stop.json')
        report['artifact_pins'] = ct.inventory(path for path in directory.rglob('*') if path.is_file()
                                             and path.name != 'study-coordinator.lock')
        ct.write(directory / 'paired-final.json', report)
        return dict(status=report['status'], actual_completed_lifecycles=report['actual_completed_lifecycles'],
            actual_issued_lifecycles=report['actual_issued_lifecycles'], analysis=report['analysis'],
            final_authenticated_echild=report.get('final_authenticated_echild'), final_all_eight_idle=report.get('final_all_eight_idle'))


def main(argv=None):
    import signal
    def cancelled(signum, frame):
        raise KeyboardInterrupt('owned coordinator/worker cancellation')
    signal.signal(signal.SIGTERM, cancelled)
    ct.require_cpu_python()
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--_worker':
        require(len(argv) == 2, 'authenticated worker launch hash required')
        with actual_adapter():
            return planned.worker(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('start', 'next', 'finalize'))
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'start':
        value = start(args.directory.resolve(), args.sole_gpu_owner_granted)
    elif args.mode == 'next':
        value = execute_next(args.directory)
        print(json.dumps(dict(trial=value['trial'], can_continue=value['assessment']['can_continue'],
            reasons=value['assessment']['stop_reasons'], measured_qa=value['summary']['measured_qa'])), flush=True)
        return 0 if value['assessment']['can_continue'] else 1
    else:
        value = finalize(args.directory)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0 if value['status'] in ('registered', 'complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
