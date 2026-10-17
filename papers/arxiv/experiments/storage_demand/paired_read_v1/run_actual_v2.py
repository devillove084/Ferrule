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
    role = payload['spec'].get('role_admission')
    if role is not None:
        admission = getattr(binding, 'role_admission', None)
        require(admission is not None and admission.reference() == role
            and ct.load(Path(payload['directory']) / 'attempt.json')['role_admission'] == role,
            'worker role admission/attempt binding changed')
        admission.check(payload['execution_path'])
    prepared = payload['spec'].get('static_preparation')
    if prepared is not None:
        folder = Path(payload['directory'])
        require(Path(prepared['path']).parent == folder.parent
            and ct.sha(prepared['path']) == prepared['sha256']
            and ct.load(prepared['path'])['trial'] == payload['trial']
            and ct.load(folder / 'attempt.json')['static_preparation'] == prepared,
            'worker static preparation/attempt binding changed')
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


def admission_refused(directory):
    # Presence is terminal even for a partial/unknown-trial refusal journal.
    return any(path.name == 'static-refusal.json'
        or (path.name.startswith('static-refusal-') and path.name.endswith('.json'))
        for path in Path(directory).iterdir())


def require_admission_open(directory):
    directory = Path(directory)
    require(not admission_refused(directory) and not (directory / 'study-stop.json').exists()
        and not (directory / 'paired-final.json').exists(),
        'study permanently stopped/finalized/admission refused; no resume or replacement')


def next_trial(directory, registered):
    directory = Path(directory)
    require_admission_open(directory)
    for trial in registered['trial_order']:
        folder = directory / trial
        if not folder.exists():
            return trial
        complete_attempt = (folder / 'attempt.json').is_file() and (folder / 'assessment.json').is_file()
        if 'study_layout' in registered and not complete_attempt:
            stop(directory, 'prior attempt lost/incomplete custody; permanent STOP, never replay')
        require(complete_attempt, 'prior attempt lost/incomplete custody; permanent STOP, never replay')
        row = effective_assessment(folder, ct.load(folder / 'assessment.json'),
            require_completion='study_layout' in registered)
        qualified = row['trial'] == trial and row['assessment']['can_continue'] is True
        if 'study_layout' in registered and not qualified:
            stop(directory, 'prior attempt failed/unknown; no resume or replacement')
        require(qualified, 'prior attempt failed/unknown; no resume or replacement')
    raise ValueError('all planned arms already issued; no additional lifecycle')


def coordinator_terminal_binding(folder):
    folder = Path(folder)
    return dict(schema='ferrule.storage-demand.coordinator-terminal.v1', trial=folder.name,
        attempt_sha256=ct.sha(folder / 'attempt.json'), assessment_sha256=ct.sha(folder / 'assessment.json'),
        locks_held_sha256=ct.sha(folder / 'locks-held.json'))


def coordinator_failure(folder, reason):
    folder = Path(folder)
    failure = folder / 'coordinator-failure.json'
    if not failure.exists():
        ct.write(failure, dict(schema='ferrule.storage-demand.coordinator-failure.v1',
            trial=folder.name, reason=reason, recorded=ex.stamp(),
            assessment_sha256=ct.sha(folder / 'assessment.json'), qualification=False))
    stop(folder.parent, reason)
    return ct.load(failure)


def coordinator_completion(folder):
    """Read-only validation: terminal receipts cannot be replayed across arms."""
    folder = Path(folder)
    try:
        binding = coordinator_terminal_binding(folder)
        release, reprobe = (ct.load(folder / name) for name in ('locks-released.json', 'locks-reprobed.json'))
        return (not (folder / 'coordinator-failure.json').exists()
            and all(release.get(k) == v and reprobe.get(k) == v for k, v in binding.items())
            and release.get('phase') == 'release' and release.get('context_exited') is True
            and release.get('all_eight_local') is True and reprobe.get('phase') == 'reprobe'
            and reprobe.get('all_eight_nonblocking_reacquired') is True and reprobe.get('local_only') is True
            and reprobe.get('context_exited') is True)
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def effective_assessment(folder, row, *, require_completion=False):
    """Append-only coordinator failures supersede qualification, not evidence."""
    folder = Path(folder)
    failure = folder / 'coordinator-failure.json'
    if not failure.exists():
        if not require_completion or coordinator_completion(folder):
            return row
        terminal = coordinator_failure(folder, 'coordinator terminal release/reprobe evidence missing/invalid; permanent STOP')
    else:
        terminal = ct.load(failure)
    require(terminal.get('schema') == 'ferrule.storage-demand.coordinator-failure.v1'
        and terminal.get('trial') == folder.name and terminal.get('qualification') is False
        and terminal.get('assessment_sha256') == ct.sha(folder / 'assessment.json'),
        'coordinator failure/assessment binding changed')
    import copy
    result = copy.deepcopy(row)
    result['assessment']['can_continue'] = False
    result['assessment'].setdefault('stop_reasons', []).append(terminal['reason'])
    result['summary']['measured_qa']['eligible_success'] = 0
    result['coordinator_failure'] = terminal
    return result


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


def admission_refusal(directory, error, *, trial=None, started=None, deadline=None, source_proof):
    """Best-effort independent journals; neither may replace the primary error."""
    reason = type(error).__name__ + ': ' + str(error)
    try:
        if admission_refused(directory):
            return  # Preserve the first refusal and its primary cause, including on retry.
    except OSError as journal_error:
        error.add_note('refusal barrier inspection failed: ' + str(journal_error))
    if not isinstance(trial, str) or re.fullmatch(r'[A-Za-z0-9_-]{1,96}', trial) is None:
        trial = None
    operations = (
        ('STOP', lambda: stop(directory, reason)),
        ('refusal', lambda: ct.write(Path(directory) / (
            'static-refusal-' + trial + '.json' if trial is not None else 'static-refusal.json'), dict(
                schema='ferrule.storage-demand.http-static-refusal.v1', trial=trial,
                started=started, deadline=deadline, recorded=ex.stamp(), reason=reason,
                worker_state='never_spawned', supervisor_call_entered=False, gpu_locks_acquired=False,
                source_proof=source_proof, authenticated_echild=False, physical_owner_release='unverified'))),
    )
    for label, operation in operations:
        try:
            operation()
        except (Exception, KeyboardInterrupt) as journal_error:
            error.add_note(label + ' journal failure: ' + type(journal_error).__name__ + ': ' + str(journal_error))


@contextmanager
def admission_entry(directory, admission):
    entered = False
    try:
        require_admission_open(directory)
        with admission(directory) if admission else ExitStack():
            entered = True
            yield
    except (Exception, KeyboardInterrupt) as error:
        if admission is not None and not entered:
            # Covers context factory/__enter__ as well as preparation itself.
            admission_refusal(directory, error, source_proof='admission context did not yield to execute_next')
        raise


def execute_next(directory, *, admission=None):
    require(not getattr(binding, 'finalization_only', False), 'closeout cannot admit another lifecycle')
    directory = directory.resolve()
    with coordinator_lock(directory), admission_entry(directory, admission):
        admission_deadline = getattr(binding, 'admission_deadline', None)
        try:
            with ex.absolute_alarm(admission_deadline) if admission_deadline is not None else ExitStack():
                registered = registration(directory)
                trial = next_trial(directory, registered)
                limits, schedule = registered['limits'], registered['schedule']
                require(planned.can_admit(registered['pilot_deadline']), 'absolute 7200s envelope lacks full next-arm plus reserve')
                if admission_deadline is not None:
                    binding.validate_binding(Path(registered['binding_path']))
                    binding.validate_execution(Path(registered['execution_path']))
                    require(time.monotonic() <= admission_deadline, 'static admission deadline exceeded after final setup')
                    require(planned.can_admit(registered['pilot_deadline']), 'static preparation exhausted next-arm admission')
                prepared = getattr(binding, 'static_preparation', None)
                if prepared is not None:
                    require(Path(prepared['path']).parent == directory
                        and ct.load(prepared['path'])['trial'] == trial, 'static preparation belongs to another arm')
                folder = directory / trial
                folder.mkdir(exist_ok=False)
        except (Exception, KeyboardInterrupt) as error:
            if admission_deadline is not None:
                admission_refusal(directory, error, trial=locals().get('trial'),
                    deadline=admission_deadline,
                    source_proof='execute_next failed before active clock/owner locks')
            raise
        active_started = ex.stamp()
        outer = min(active_started['monotonic'] + 650, registered['pilot_deadline'] - 100)
        role_admission = getattr(binding, 'role_admission', None)
        if role_admission is not None:
            role_admission.activate(active_started, outer)
        ct.write(folder / 'attempt.json', dict(trial=trial, admission=active_started, outer_deadline=outer,
            **({'role_admission': role_admission.reference()} if role_admission is not None else {}),
            static_preparation=getattr(binding, 'static_preparation', None),
            static_preparation_seconds=(active_started['monotonic'] - binding.static_preparation_started
                if getattr(binding, 'static_preparation', None) or role_admission is not None else None),
            pilot_deadline=registered['pilot_deadline'], registration_sha256=ct.sha(directory / 'registration.json'),
            coordinator=pilot.launch_guard.identity(os.getpid()), replay_permitted=False))
        receipt, observed, post, locks = {}, {}, dict(status='blocked', all_eight_idle=False), []
        mode = trial.split('-', 1)[1]
        supervisor_entered = locks_authenticated = post_attempted = False
        cleanup_errors = []

        def post_idle():
            nonlocal post, post_attempted
            post_attempted = True
            if not supervisor_entered:
                try:
                    ct.write(folder / 'worker-refusal.json', dict(
                        schema='ferrule.storage-demand.never-spawned-refusal.v1', trial=trial,
                        attempt_sha256=ct.sha(folder / 'attempt.json'),
                        locks_held_sha256=ct.sha(folder / 'locks-held.json') if (folder / 'locks-held.json').exists() else None,
                        coordinator_source_sha256=ct.sha(Path(__file__).resolve()),
                        coordinator=pilot.launch_guard.identity(os.getpid()),
                        worker_state='never_spawned', supervisor_call_entered=False,
                        source_proof='execute_next: planned.supervise not entered',
                        authenticated_echild=False, physical_owner_release='unverified'))
                except (Exception, KeyboardInterrupt) as error:
                    cleanup_errors.append('refusal receipt: ' + type(error).__name__ + ': ' + str(error))
            if not locks_authenticated:
                return
            try:
                if not supervisor_entered:
                    custody = None  # Strict idle, never inferred ECHILD/stabilization.
                elif ct.drained(receipt) and pilot.idle_window.no_recorded_owner(receipt):
                    custody = receipt
                else:
                    return
                post = pilot.idle(folder, 'coordinator-post-idle', registered['eight_uuids'],
                                  min(time.monotonic() + 20, outer), custody)
            except (Exception, KeyboardInterrupt) as error:
                reason = type(error).__name__ + ': ' + str(error)
                post = dict(status='blocked', all_eight_idle=False, error=reason)
                cleanup_errors.append('post-idle: ' + reason)
            try:
                if not (folder / 'coordinator-post-idle.json').exists():
                    ct.write(folder / 'coordinator-post-idle.json', post)
            except (Exception, KeyboardInterrupt) as error:
                cleanup_errors.append('post-idle receipt: ' + type(error).__name__ + ': ' + str(error))

        try:
            pilot.telemetry.authorize(True, registered['selected_uuid'], NOTE, 'device_exclusive_local_custody')
            with ExitStack() as stack:
                for uuid in registered['eight_uuids']:
                    locks.append(stack.enter_context(ct.exclusive_owner_lock(uuid)))
                for uuid, lock in zip(registered['eight_uuids'], locks):
                    ct.validate_owner_lock(lock, uuid, os.getpid())
                locks_authenticated = True
                stack.callback(lambda: None if post_attempted else post_idle())
                ct.write(folder / 'locks-held.json', dict(locks=locks, coordinator=pilot.launch_guard.identity(os.getpid()),
                    all_eight_local=True, global_reservation=False))
                with ExitStack() as worker_adapter:
                    with ex.absolute_alarm(outer - (530 + 28 + 20 + 32)):
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
                        worker_adapter.enter_context(actual_adapter())
                        if role_admission is not None:
                            role_admission.before_spawn()
                    pilot.remaining(outer, 530 + 28 + 20 + 32, 'supervisor and finite cleanup')
                    supervisor_entered = True
                    receipt = planned.supervise(payload, 530)
                ct.write(folder / 'supervisor.json', receipt)
                post_idle()
                observed = ct.load(folder / 'lifecycle.json') if (folder / 'lifecycle.json').is_file() else {}
                if 'study_layout' in registered:
                    observed.setdefault('requests', {})['observations'] = ex.recover_observations(folder, schedule)
                assessment = ex.lifecycle_assessment(observed, receipt, post, transfer_required='study_layout' in registered)
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
                if ('study_layout' in registered and registered['study_layout'].get('purpose') != 'public_cli_acceptance'
                        and assessment['can_continue']):
                    block = next(b for b in registered['order'] if b['block'] == row['block'])
                    paired = {trial: row}
                    for pair_mode in block['modes']:
                        pair_trial = block['block'] + '-' + pair_mode
                        prior = directory / pair_trial / 'assessment.json'
                        if pair_trial != trial and prior.is_file():
                            paired[pair_trial] = ct.load(prior)
                    if len(paired) == 2:
                        paired_metrics = analysis.transfer_pair_metrics(block, paired, schedule,
                            contract=registered['study_layout']['http_expectation_contract'],
                            reference_context=ex.reference_context_for_layout(registered['study_layout']),
                            producer_contract=registered['study_layout']['upload_producer_contract'])
                        row['paired_quality'] = {k: paired_metrics[k] for k in
                            ('qualified', 'crosspolicy_exact_output', 'crosspolicy_equal_work')}
                        if not paired_metrics['qualified']:
                            assessment['can_continue'] = False
                            assessment['stop_reasons'].append('paired HTTP output/work/metric mismatch; permanent STOP')
                            summary['measured_qa']['eligible_success'] = 0
                ct.write(folder / 'assessment.json', row)
                require(assessment['can_continue'], '; '.join(assessment['stop_reasons']))
        except (Exception, KeyboardInterrupt) as error:
            reason = type(error).__name__ + ': ' + str(error)
            if not post_attempted:
                post_idle()
            stop(directory, reason)
            if (folder / 'assessment.json').exists():
                coordinator_failure(folder, reason)
            else:
                observed = ct.load(folder / 'lifecycle.json') if (folder / 'lifecycle.json').is_file() else observed
                if 'study_layout' in registered:
                    observed.setdefault('requests', {})['observations'] = ex.recover_observations(folder, schedule)
                summary = ex.recover_requests(folder, schedule)
                summary['measured_qa']['request_only_eligible_success'] = summary['measured_qa']['eligible_success']
                summary['measured_qa']['eligible_success'] = 0
                ct.write(folder / 'assessment.json', dict(trial=trial, block=trial.split('-')[0], mode=mode,
                    summary=summary, assessment=dict(can_continue=False, stop_reasons=[reason]), read_ledger=ledger(observed, mode),
                    supervisor=receipt, post_idle=post, lifecycle=observed, coordinator_error=reason))
        finally:
            # Both paths leave the lock context before bounded nonblocking reprobe.
            terminal = {}
            try:
                terminal = coordinator_terminal_binding(folder) if locks_authenticated and 'study_layout' in registered else {}
                ct.write(folder / 'locks-released.json', dict(terminal, phase='release', context_exited=True,
                    all_eight_local=len(locks) == 8, acquired_local_locks=len(locks),
                    physical_owner_release='unverified', descendant_custody_known=ct.drained(receipt)))
            except (Exception, KeyboardInterrupt) as error:
                cleanup_errors.append('release receipt: ' + type(error).__name__ + ': ' + str(error))
            if locks_authenticated:
                try:
                    with ex.absolute_alarm(min(time.monotonic() + 32, outer)):
                        with ExitStack() as stack:
                            reprobed = [stack.enter_context(ct.exclusive_owner_lock(uuid)) for uuid in registered['eight_uuids']]
                            for uuid, lock in zip(registered['eight_uuids'], reprobed):
                                ct.validate_owner_lock(lock, uuid, os.getpid())
                        ct.write(folder / 'locks-reprobed.json', dict(terminal, phase='reprobe',
                            all_eight_nonblocking_reacquired=True, context_exited=True, local_only=True))
                except (Exception, KeyboardInterrupt) as error:
                    cleanup_errors.append('reprobe: ' + type(error).__name__ + ': ' + str(error))
            if cleanup_errors:
                ct.write(folder / 'coordinator-cleanup-errors.json', dict(
                    schema='ferrule.storage-demand.coordinator-cleanup-errors.v1', trial=trial,
                    errors=cleanup_errors, assessment_sha256=ct.sha(folder / 'assessment.json'),
                    qualification=False, recorded=ex.stamp()))
                coordinator_failure(folder, '; '.join(cleanup_errors))
        row = effective_assessment(folder, ct.load(folder / 'assessment.json'),
            require_completion='study_layout' in registered)
        if row['assessment']['can_continue']:
            print('ARM ' + trial + ' complete: 12/12 requests, 5/5 measured QA; normal owned shutdown, ECHILD, all-eight idle', flush=True)
        else:
            print('ARM ' + trial + ' STOP: ' + '; '.join(row['assessment']['stop_reasons']), flush=True)
        return row


def never_spawned(folder):
    """Positive arm-bound source/control-flow evidence, never file absence."""
    folder = Path(folder)
    try:
        refusal = ct.load(folder / 'worker-refusal.json')
        attempt = ct.load(folder / 'attempt.json')
        registered = ct.load(folder.parent / 'registration.json')
        source = str(Path(__file__).resolve())
        source_pin = registered.get('study_layout', {}).get('artifact_pins', {}).get(source)
        if source_pin is None:
            source_pin = ct.sha(source)
        return (refusal.get('schema') == 'ferrule.storage-demand.never-spawned-refusal.v1'
            and refusal.get('trial') == folder.name and refusal.get('worker_state') == 'never_spawned'
            and refusal.get('supervisor_call_entered') is False
            and refusal.get('source_proof') == 'execute_next: planned.supervise not entered'
            and refusal.get('coordinator') == attempt['coordinator']
            and refusal.get('coordinator_source_sha256') == source_pin
            and refusal.get('attempt_sha256') == ct.sha(folder / 'attempt.json')
            and refusal.get('authenticated_echild') is False)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def finalize(directory):
    directory = directory.resolve()
    with coordinator_lock(directory):
        registered = registration(directory)
        require(not (directory / 'paired-final.json').exists(), 'final receipt exclusive; never replace')
        rows = []
        schedule = registered['schedule']
        for trial in registered['trial_order']:
            folder = directory / trial
            recovery_error = None
            if (folder / 'assessment.json').is_file():
                try:
                    original = ct.load(folder / 'assessment.json')
                    require(original['trial'] == trial and isinstance(original['assessment']['can_continue'], bool),
                        'arm assessment identity/type differs')
                    rows.append(effective_assessment(folder, original, require_completion='study_layout' in registered))
                    continue
                except (OSError, ValueError, TypeError, KeyError) as error:
                    if 'study_layout' not in registered:
                        raise
                    recovery_error = type(error).__name__ + ': ' + str(error)
                    stop(directory, 'invalid/partial arm assessment; permanent STOP')
                    if not (folder / 'coordinator-failure.json').exists():
                        coordinator_failure(folder, recovery_error)
            if 'study_layout' in registered and folder.exists():
                stop(directory, 'interrupted/invalid arm evidence; permanent STOP')
            summary = ex.recover_requests(folder, schedule)
            summary['measured_qa']['request_only_eligible_success'] = summary['measured_qa']['eligible_success']
            summary['measured_qa']['eligible_success'] = 0
            rows.append(dict(trial=trial, block=trial.split('-')[0], mode=trial.split('-', 1)[1], summary=summary,
                status='unissued_or_incomplete', stop_reason='prior STOP/incomplete attempt/finite envelope; no replacement'))
            if 'study_layout' in registered:
                rows[-1]['lifecycle'] = dict(requests=dict(observations=ex.recover_observations(folder, schedule)))
            if recovery_error is not None:
                rows[-1]['assessment_recovery_error'] = recovery_error
        complete = all(row.get('assessment', {}).get('can_continue') is True for row in rows)
        if (directory / 'study-stop.json').exists():
            complete = False
        require(complete or (directory / 'study-stop.json').exists() or time.monotonic() >= registered['pilot_deadline'] - 100,
                'do not finalize early without fixed STOP/deadline; planned arms remain')
        report = dict(schema='ferrule.storage-demand.paired-read-actual.v2', status='complete' if complete else 'stopped',
            order=registered['order'], lifecycles=rows, scheduled_blocks=registered.get('scheduled_blocks', 5), scheduled_starts=len(registered['trial_order']),
            scheduled_requests=len(rows) * 12, scheduled_measured_qa=len(rows) * 5, started=registered['started'], ended=ex.stamp(),
            pilot_deadline=registered['pilot_deadline'], candidate_sha256=registered['binding_sha256'],
            execution_sha256=registered['execution_sha256'], binding_path=registered['binding_path'],
            physical_uuid=registered['selected_uuid'], eight_uuids=registered['eight_uuids'],
            retries=0, replacements=0, extra_pilot_lifecycles=0, profile_lifecycles=0,
            physical_nas_backend_bytes=None, universal_final_release_verified=False,
            resource_contract_complete='unverified', scientific_acceptance=False)
        if 'study_layout' in registered:
            report['study_layout'] = dict(registered['study_layout'])

        acceptance = registered.get('study_layout', {}).get('purpose') == 'public_cli_acceptance'
        if acceptance:
            report['analysis'] = dict(quality_only=True, scheduled_cells=4, scheduled_requests=48,
                completed_cells=sum(row.get('assessment', {}).get('can_continue') is True for row in rows),
                performance_claim=False, filesystem_cold_claim=False)
        else:
            analysis_report = report
            if 'study_layout' in registered:
                analysis_report = dict(report, study_layout=dict(report['study_layout'],
                    http_reference_context=ex.reference_context_for_layout(registered['study_layout']),
                            producer_contract=registered['study_layout']['upload_producer_contract']))
            report['analysis'] = analysis.summarize(analysis_report)
        if not acceptance and 'study_layout' in registered and not all(b['qualified'] for b in report['analysis']['blocks']):
            report['status'] = 'stopped'
        report['actual_completed_lifecycles'] = sum(row.get('assessment', {}).get('can_continue') is True for row in rows)
        report['actual_issued_lifecycles'] = sum((directory / trial / 'launch.json').is_file() and not never_spawned(directory / trial)
            for trial in registered['trial_order'])
        report['pipeline'] = [{key: row.get(key) for key in ('trial', 'mode', 'read_ledger')} for row in rows]
        report['http_quality_evidence'] = []
        for row in rows:
            requests = row.get('lifecycle', {}).get('requests', {}).get('observations', [])
            report['http_quality_evidence'].append(dict(trial=row['trial'], requests=[dict(
                name=request['name'], status=request.get('status'), metric_eligible=request.get('metric_eligible'),
                measured=request.get('measured'), actual=request.get('actual'), actual_token_id_join=request.get('trace'),
                nfs=request.get('nfs'), logical_counters=request.get('logical_counters')) for request in requests]))
        attempted = [row for row in rows if row.get('supervisor') or never_spawned(directory / row['trial'])]
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
