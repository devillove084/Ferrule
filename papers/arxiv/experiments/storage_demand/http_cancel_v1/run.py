#!/usr/bin/env python3
"""One separately authorized HTTP cancellation/recovery lifecycle, no retries or Git writes."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import textwrap
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STORAGE = HERE.parent
sys.path.insert(0, str(STORAGE / 'pressure_fault_v1'))
import subset_adapter as maintained
from phase_timing import deadline_cap
import client

ex, ct, pilot, trace = maintained.ex, maintained.ct, maintained.pilot, maintained.trace
require = maintained.require
ENTRY = Path(__file__).resolve()
PRIVATE = HERE / 'private-case-20261015-01'
PREPARED = PRIVATE / 'prepared.json'
SCHEMA = 'ferrule.storage-demand.http-cancel.v1'
TRIAL = 'cancel-recovery-lookahead'
NOTE = 'Explicit parent GO: one normal HTTP cancel then one new QA8 recovery; no other GPU work, retries, profiler, cache conditioning or foreign signals'
COMMIT = '86c58a017d7c4b53490f536a3b1a133f2f7848cc'
BINARY_SHA = '7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01'
LIMITS = dict(startup_seconds=60, request_seconds=60, all_requests_seconds=165,
              cancellation_wait_seconds=45, worker_bookkeeping_seconds=0,
              target_lifetime_seconds=225, shutdown_seconds=20, worker_preidle_seconds=10,
              worker_finalization_seconds=4, supervisor_seconds=249,
              supervisor_call_overhead_seconds=11, outer_lifecycle_seconds=270)
OFFSETS = dict(startup=60, requests=225, shutdown=245, worker=249, supervisor=260, postidle=265, outer=270)


def schedule():
    qa = copy.deepcopy(maintained.schedule()[1])
    require(qa['reference']['reference_ids'] == [25358, 13, 198, 760, 6511, 314, 6124, 369], 'QA8 oracle drift')
    rows = []
    for index, name in enumerate(('cancel-qa8', 'recovery-qa8')):
        rows.append(dict(copy.deepcopy(qa), index=index, name=name, phase=name, population=name,
                         measured=False, controlled_cancel=index == 0))
    return rows


def validate_schedule(rows):
    require(rows == schedule() and len(rows) == 2, 'exact cancel QA8 then fresh recovery QA8 only')


def spec_for(candidate):
    case = ct.load(STORAGE / 'paired_read_v1/cases-v3.json')['lookahead']
    require(case['environment']['FERRULE_PROFILE_NVTX'] == '0', 'NVTX must remain disabled')
    require(case['topology'] == dict(pp=1, tp=1, ep=1, rank_backend='thread', owners=1, visible_ordinals=[0]), 'TP1 only')
    uuid = candidate['uuid_inventory'][0]
    environment = dict(ex.runner.cpu_environment(), **case['environment'])
    environment.update(CUDA_VISIBLE_DEVICES=uuid, NVIDIA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return dict(engine='ferrule', argv=[candidate['candidate']['binary'], *case['argv_after_binary']],
                environment=environment, model='qwen35-storage-demand-paper', url='http://127.0.0.1:18369/v1/completions')


def source_qualification(candidate):
    source = ct.load(maintained.qualified.QUALIFICATION / 'source-freeze.json')
    root = Path(source['frozen_source'])
    checks = {
        'crates/ferrule-server/src/http.rs': ['subscription: EventSubscription', 'impl Drop for OpenAiEventStream', '"client_disconnect"'],
        'crates/ferrule-server/src/worker.rs': ['impl Drop for EventSubscription', 'self.cancellation.store(true, Ordering::Release);',
            'match engine.cancel_request(request_id)', 'engine.drain_cancelled()', 'cancellation_submitted = request.cancellation_submitted',
            'event = "request_worker_token_enqueued"', 'event = "request_worker_terminal"'],
        'crates/ferrule-runtime/src/engine/driver.rs': ['pub fn cancel_request(', 'self.request_transaction_abort(transaction, request_id)',
            'self.register_transaction_cancellation(transaction, request_id, session_id)?;'],
        'crates/ferrule-model/src/checkpoint/read_session.rs': ['let result = self.file.read_at(bytes, offset);', 'self.validate_boundary()?;'],
    }
    pins = {}
    for name, needles in checks.items():
        path = root / name
        maintained.immutable(path)
        require(ct.sha(path) == source['pins_relative_to_frozen_source'][name]
                and all(needle in path.read_text() for needle in needles), 'archived cancellation source differs: ' + name)
        pins[str(path)] = ct.sha(path)
    return dict(pins=pins, compiled_commit=candidate['candidate_identity']['commit'], live_HEAD_consumed=False,
                ownership='OpenAiEventStream owns EventSubscription; subscription drop sets cancellation and signals worker',
                terminal='runtime drain_cancelled plus request_worker_terminal cancelled/submitted; HTTP close alone insufficient',
                synchronous_read='cooperative only; cannot interrupt a synchronous file read',
                per_request_transaction_ledger=None, per_request_pending_operations=None, epoch=None,
                source_is_provenance_not_live_transaction_observation=True)


def protocol(candidate):
    return dict(schema=SCHEMA + '.protocol', candidate_identity=candidate['candidate_identity'],
        schedule=schedule(), spec=spec_for(candidate), limits=LIMITS, absolute_offsets=OFFSETS,
        lifecycles=1, maximum_generative_requests=2, initial_smoke_requests=0, retries=0,
        cancellation_trigger='first actual nonempty SSE content event; immediately shutdown TCP and close response/connection',
        terminal_gate='matching HTTP ID/session, worker cancelled + cancellation_submitted true + no failed/restore state; closed-channel terminal enqueue and client_disconnect',
        terminal_wait='passive complete JSON log lines, <=45s, no inference polling',
        failure_policy='stop first failure; natural max_tokens race is not cancellation success; recovery unissued unless cancelled accepted',
        cancelled_outcome='controlled right censor; no QA exact PASS, no required client usage or DONE, worker enqueues not consumer IDs',
        recovery='new request/session same resident process; exact frozen eight IDs/text/usage/length/DONE, not same-request resume',
        budgets='60 startup + 120 HTTP + 45 cancellation/control/validation + 45 cleanup =270 <=300 seconds; shared non-sliding cutoffs',
        resource_scope='serial log-window invocation/cache observations and final owner closure, not per-request integrated ledger',
        missing_ledger_values=None, Unknown_proved=False, integrated_fault_ledger_proved=False,
        cache_conditioning=False, profiler=False, production_changes=False, public_fault_flag=False,
        commit_policy='parent owns Git index; this agent makes no Git writes')


def seal(path, value):
    ct.write(path, value)
    Path(path).chmod(0o400)


def prepare():
    require(not PRIVATE.exists(), 'one private generation only; never resume/reprepare')
    candidate = maintained.validate_candidate()
    require(candidate['candidate_identity']['commit'] == COMMIT and candidate['candidate']['binary_sha256'] == BINARY_SHA,
            'wrong qualified CLI86')
    build = maintained.review_build(candidate)
    provenance = source_qualification(candidate)
    inputs = maintained.validate_execution()['artifact_pins']
    inputs.update(ct.inventory([ENTRY, HERE / 'client.py', HERE / 'test_http_cancel.py', HERE / 'PLAN.md']))
    inputs.update(provenance['pins'])
    candidate['candidate']['artifact_pins'].update(inputs)
    plan = protocol(candidate)
    PRIVATE.mkdir(mode=0o700)
    seal(HERE / 'plan.json', plan)
    seal(PRIVATE / 'candidate-review.json', build)
    seal(PRIVATE / 'source-qualification.json', provenance)
    value = dict(schema=SCHEMA + '.prepared', candidate=candidate['candidate'], candidate_record=candidate,
                 artifact_pins=inputs, protocol=plan, plan_sha256=ct.sha(HERE / 'plan.json'),
                 build_review_sha256=ct.sha(PRIVATE / 'candidate-review.json'),
                 source_qualification_sha256=ct.sha(PRIVATE / 'source-qualification.json'), gpu_runs=0)
    seal(PREPARED, value)
    return dict(status='frozen_cpu_prepared', prepared_sha256=ct.sha(PREPARED), plan_sha256=value['plan_sha256'], gpu_runs=0)


def prepared():
    maintained.immutable(PREPARED)
    value = ct.load(PREPARED)
    require(value['schema'] == SCHEMA + '.prepared' and value['protocol'] == protocol(value['candidate_record']), 'frozen protocol changed')
    require(value['plan_sha256'] == ct.sha(HERE / 'plan.json')
            and value['build_review_sha256'] == ct.sha(PRIVATE / 'candidate-review.json')
            and value['source_qualification_sha256'] == ct.sha(PRIVATE / 'source-qualification.json'), 'frozen receipt changed')
    ct.check_pins(value['artifact_pins'])
    ct.check_pins(value['candidate']['artifact_pins'])
    maintained.roles.check_source_model()
    return value


def caps(payload):
    origin = payload['outer_deadline'] - OFFSETS['outer']
    return {key: origin + value for key, value in OFFSETS.items()}


def worker_contract(payload):
    with deadline_cap(caps(payload)['startup']):
        require(payload['schema'] == SCHEMA + '.worker' and payload['contract_sha256']
                == ct.digest({key: payload[key] for key in maintained.CONTRACT_KEYS}), 'worker sealed contract changed')
        value = prepared()
        require(payload['trial_order'] == [TRIAL] and payload['trial'] == TRIAL and payload['mode'] == 'lookahead'
                and payload['request_ceiling'] == 2 and payload['schedule'] == schedule()
                and payload['limits'] == LIMITS and payload['granted'] is True and payload['owner_note'] == NOTE,
                'worker protocol/authorization drift')
        candidate = value['candidate_record']
        require(payload['spec'] == spec_for(candidate) and payload['adapter_pins'] == value['artifact_pins'], 'worker spec/pins differ')
        for key in ('execution', 'candidate'):
            require(payload[key + '_path'] == str(PREPARED) and payload[key + '_sha256'] == ct.sha(PREPARED), 'worker preparation seal changed')
        require(payload['eight_uuids'] == candidate['uuid_inventory'] and payload['uuid'] == candidate['uuid_inventory'][0], 'UUID inventory changed')
        require(len(payload['all_uuid_locks']) == 8, 'all-eight local locks required')
        for uuid, lock in zip(payload['eight_uuids'], payload['all_uuid_locks']):
            ct.validate_owner_lock(lock, uuid, payload['coordinator']['pid'])
        require(payload['owner_lock'] == payload['all_uuid_locks'][0], 'selected lock differs')
        pilot.telemetry.authorize(True, payload['uuid'], NOTE, 'device_exclusive_local_custody')
        return candidate


def summary(rows, schedule=None):
    schedule = globals()['schedule']() if schedule is None else schedule
    return dict(scheduled=len(schedule), issued=len(rows), unissued=len(schedule) - len(rows),
                controlled_censors=sum(r.get('actual', {}).get('status') == 'controlled_client_disconnect' for r in rows),
                accepted_cancellations=sum(r.get('trace', {}).get('cancellation_accepted') is True for r in rows),
                recovery_exact_pass=sum(not r.get('controlled_cancel') and r.get('quality_passed') is True for r in rows),
                cancelled_quality_pass=None)


def one_request(directory, planned, spec, deadline):
    with deadline_cap(deadline):
        actual = client.stream(spec['url'], planned['payload'], cancel=planned['controlled_cancel'], deadline=deadline)
    # Persist the close before waiting, so a missing runtime terminal cannot become unissued.
    ct.write(Path(directory) / 'http-stream.json', actual)
    if planned['controlled_cancel']:
        wait_deadline = min(actual['closed_at'] + LIMITS['cancellation_wait_seconds'], deadline)
        with deadline_cap(deadline):
            evidence = client.wait_cancel(Path(directory).parent / 'server.log', actual,
                                         planned['reference']['reference_ids'], deadline=wait_deadline)
        ct.write(Path(directory) / 'cancel-terminal.json', evidence)
        actual['cancellation_evidence'] = evidence
    return actual


def join(trace_rows, requests, lines=()):
    results = []
    for row in requests:
        if row.get('controlled_cancel'):
            result = client.cancellation_evidence(lines, row.get('actual', {}), row['reference']['reference_ids'])
        else:
            result = trace.join_token_trace_http(trace_rows, [row], lines)[0]
            result['protocol_accepted'] = result['metric_eligible']
        results.append(result)
    if len(requests) == 2:
        ids = [row['actual_worker_request_id'] for row in results]
        require(ids == [1, 2], 'fresh distinct HTTP request/session identities required, no hidden request')
        require([r['request_id'] for r in trace_rows] == [ids[0]] * results[0]['enqueued_token_count'] + [ids[1]] * 8,
                'extra/missing/reordered worker tokens')
        events = [trace.fields(value) for _, value in trace.json_events(lines)]
        for name in maintained.REQUEST_EVENTS:
            rows = [row for row in events if row.get('event') == name]
            require([r.get('request_id') for r in rows] == ids and all(r.get('session_id') == r.get('request_id') for r in rows),
                    'extra/missing/reordered worker/HTTP event: ' + name)
    return results


def request_gate(row, actual, bound, counters, nfs):
    if row['controlled_cancel']:
        return bool(bound['cancellation_accepted'] and actual.get('cancellation_evidence', {}).get('cancellation_accepted')
                    and row['reference']['reference_text'].startswith(actual.get('text', ''))
                    and counters['valid'] and nfs['status'] == 'shared_client_delta')
    return actual.get('finish_reason') == 'length' and ex.request_gate(row, actual, bound, counters, nfs)


def serial_requests(*args):
    function = maintained._specializations(ex.serial_requests, (
        ('len(observations) == 12', 'len(observations) == 2'),
        ('quality_passed=eligible,', "quality_passed=None if planned['controlled_cancel'] else eligible, protocol_accepted=eligible,"),
        ('metric_eligible=eligible, logical_counters=counters', "metric_eligible=eligible and not planned['controlled_cancel'], logical_counters=counters"),
        ("status = 'success' if actual.get('status') == 'success' else (", "status = 'controlled_censor' if actual.get('status') == 'controlled_client_disconnect' else 'success' if actual.get('status') == 'success' else ("),
        ("'HTTP/text/usage/actual-ID/terminal/counter/mount/deadline gate: '", "'cancel-or-recovery/actual-ID/terminal/counter/mount/deadline gate: '"),
    ), validate_schedule=validate_schedule, one_request=one_request, join_token_trace_http=join,
        request_gate=request_gate, censoring_summary=summary)
    return function(*args)


def source_options(lines):
    function = maintained._specializations(maintained.source_options, (
        ('schedule=Blocking', 'schedule=Lookahead'),
    ))
    return function('middle', lines)


def worker(launch_hash):
    raw = sys.stdin.read()
    payload = json.loads(raw)
    deadlines = caps(payload)
    def contract(value):
        return worker_contract(value)
    with maintained.mapped.mapped_worker(payload):
        original_maps = ex.library_maps
        def library_maps(root, pins):
            environment = dict(item.split('=', 1) for item in Path(f'/proc/{root["pid"]}/environ').read_text().split('\0') if '=' in item)
            require(environment.get('FERRULE_PROFILE_NVTX') == '0', 'actual NVTX must be off')
            return original_maps(root, pins)
        def requests(directory, *args):
            ct.write(Path(directory) / 'source-options.json', source_options(ex.log_lines(Path(directory) / 'server.log')))
            return serial_requests(directory, *args)
        facade = SimpleNamespace(**dict(vars(ex), library_maps=library_maps, serial_requests=requests, join_token_trace_http=join))
        source = inspect.getsource(pilot.worker)
        target_start = source.index('        target_deadline = min(')
        target_end = source.index('        capture = ex.launch_capture', target_start)
        request_start = source.index('        environment = dict(')
        request_end = source.index('    except (Exception, KeyboardInterrupt)', request_start)
        block = source[request_start:request_end]
        block = block.replace('            ct.check_pins(payload[\'adapter_pins\'])',
                              "            maintained.roles.check_source_model()\n            ct.check_pins(payload['adapter_pins'])")
        final_check = "                require(len(result['whole_log_token_join']) == 12\n                        and all(row['metric_eligible'] for row in result['whole_log_token_join']), 'final drained token trace reconciliation failed')"
        replacements = (
            (source[target_start:target_end], "        target_deadline = deadlines['requests']\n"),
            ("startup_deadline = min(launch + limits['startup_seconds'], target_deadline)", "startup_deadline = deadlines['startup']"),
            (source[request_start:request_end], "        with deadline_cap(deadlines['requests']):\n" + textwrap.indent(block, '    ')),
            ("capture.wait(timeout=limits['shutdown_seconds'] + 2)", "capture.wait(timeout=max(.001, min(limits['shutdown_seconds'] + 2, deadlines['shutdown'] - time.monotonic())))"),
            (final_check, "                if result['status'] == 'requests_complete':\n                    require(len(result['whole_log_token_join']) == 2 and all(row['protocol_accepted'] for row in result['whole_log_token_join']), 'final cancellation/recovery reconciliation failed')"),
        )
        function = maintained._specializations(pilot.worker, replacements, HERE=HERE, ex=facade,
            worker_contract=contract, idle=maintained.idle, deadline_cap=deadline_cap, deadlines=deadlines, maintained=maintained)
        previous = sys.stdin
        try:
            import io
            sys.stdin = io.StringIO(raw)
            return function(launch_hash)
        finally:
            sys.stdin = previous


def supervise(payload):
    deadline = caps(payload)['worker']
    def bounded(argv, seconds, sealed):
        with deadline_cap(deadline):
            return pilot.supervisor.run_process(argv, seconds, sealed)
    function = maintained._specialize(pilot.supervise, 'str(ENTRY)', 'str(ENTRY)', ENTRY=ENTRY,
                                     supervisor=SimpleNamespace(run_process=bounded))
    return function(payload, max(.001, deadline - time.monotonic()))


def execute(granted):
    require(granted is True, 'explicit parent sole-owner GO required')
    require(not (PRIVATE / 'GO.json').exists(), 'one attempt only, no retry/resume')
    started = ex.stamp()
    outer = started['monotonic'] + OFFSETS['outer']
    receipt, lifecycle, post = {}, {}, dict(status='unobserved', all_eight_idle=False)
    result = dict(schema=SCHEMA + '.result', status='stopped', started=started, outer_deadline=outer,
                  scheduled_lifecycles=1, scheduled_requests=2, retries=0, unknown_proved=False,
                  integrated_request_ledger=None, per_request_pending_operations=None, per_request_KV_snapshot=None,
                  physical_resource_release=None, logical_scope='normal production HTTP cancellation only')
    folder = PRIVATE / TRIAL
    folder.mkdir(mode=0o700)
    seal(PRIVATE / 'GO.json', dict(parent_explicit_go=True, sole_gpu_owner=True, no_other_gpu_jobs=True,
         fault_owner_handoff='user confirms all prior fault work cleaned', note=NOTE, prepared_sha256=ct.sha(PREPARED),
         started=started, outer_deadline=outer, source_committed_by_this_agent=False))
    try:
        with deadline_cap(outer):
            with deadline_cap(started['monotonic'] + OFFSETS['startup']):
                value = prepared()
            candidate = value['candidate_record']
            uuids = candidate['uuid_inventory']
            pilot.telemetry.authorize(True, uuids[0], NOTE, 'device_exclusive_local_custody')
            with ExitStack() as stack:
                locks = [stack.enter_context(ct.exclusive_owner_lock(uuid)) for uuid in uuids]
                for uuid, lock in zip(uuids, locks):
                    ct.validate_owner_lock(lock, uuid, os.getpid())
                ct.write(folder / 'locks-held.json', dict(locks=locks, all_eight_local=True, global_reservation=False))
                maintained.idle(folder, 'coordinator-pre-idle', uuids, min(time.monotonic() + 10, started['monotonic'] + 60))
                anchor = pilot.baseline(uuids, uuids[0], NOTE)
                ct.write(folder / 'baseline.json', anchor)
                payload = dict(schema=SCHEMA + '.worker', directory=str(folder), trial=TRIAL, trial_order=[TRIAL],
                    mode='lookahead', request_ceiling=2, schedule=schedule(), limits=LIMITS,
                    execution_path=str(PREPARED), execution_sha256=ct.sha(PREPARED), candidate_path=str(PREPARED),
                    candidate_sha256=ct.sha(PREPARED), uuid=uuids[0], eight_uuids=uuids, owner_lock=locks[0],
                    owner_note=NOTE, granted=True, spec=value['protocol']['spec'], baseline=anchor,
                    outer_deadline=outer, pilot_deadline=outer, adapter_pins=value['artifact_pins'], all_uuid_locks=locks)
                payload['contract_sha256'] = ct.digest({key: payload[key] for key in maintained.CONTRACT_KEYS})
                ct.write(folder / 'launch.json', payload)
                receipt = supervise(payload)
                ct.write(folder / 'supervisor.json', receipt)
                if ct.drained(receipt) and pilot.idle_window.no_recorded_owner(receipt):
                    post = maintained.idle(folder, 'coordinator-post-idle', uuids, min(time.monotonic() + 5, outer - 5), receipt)
                if (folder / 'lifecycle.json').is_file():
                    lifecycle = ct.load(folder / 'lifecycle.json')
                result['assessment'] = ex.lifecycle_assessment(lifecycle, receipt, post)
            ct.write(folder / 'locks-released.json', dict(context_exited=True, all_eight_local=True))
            with ExitStack() as stack:
                reprobed = [stack.enter_context(ct.exclusive_owner_lock(uuid)) for uuid in uuids]
                for uuid, lock in zip(uuids, reprobed):
                    ct.validate_owner_lock(lock, uuid, os.getpid())
                ct.write(folder / 'locks-reprobed.json', dict(all_eight_nonblocking_reacquired=True, context_exits_after_receipt=True))
            result['all_eight_locks_released_and_reprobed'] = True
            result['status'] = 'complete' if result['assessment']['can_continue'] else 'stopped'
    except (Exception, KeyboardInterrupt) as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        result.update(ended=ex.stamp(), final_authenticated_echild=ct.drained(receipt),
                      final_recorded_owners_absent=pilot.idle_window.no_recorded_owner(receipt) if receipt else None,
                      final_all_eight_idle=post.get('all_eight_idle') is True,
                      lifecycle=lifecycle, supervisor=receipt, post_idle=post)
        result['within_270_seconds'] = result['ended']['monotonic'] <= outer
        result['raw_pins'] = ct.inventory(path for path in PRIVATE.rglob('*') if path.is_file())
        ct.write(PRIVATE / 'result.json', result)
    return dict(status=result['status'], error=result.get('error'), assessment=result.get('assessment'),
                final_authenticated_echild=result['final_authenticated_echild'], final_all_eight_idle=result['final_all_eight_idle'],
                duration_seconds=result['ended']['monotonic'] - started['monotonic'])


def main():
    os.umask(0o077)
    ct.require_cpu_python()
    def cancelled(number, frame):
        raise KeyboardInterrupt('owned coordinator/worker cancellation')
    signal.signal(signal.SIGTERM, cancelled)
    if len(sys.argv) == 3 and sys.argv[1] == '--_worker':
        return worker(sys.argv[2])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    args = parser.parse_args()
    value = prepare() if args.mode == 'prepare' else execute(args.sole_gpu_owner_granted)
    print(json.dumps(value, indent=2), flush=True)
    return 0 if value['status'] in ('frozen_cpu_prepared', 'complete') else 1


if __name__ == '__main__':
    raise SystemExit(main())
