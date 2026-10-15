#!/usr/bin/env python3
"""Separate one-pair TP1 official profile. Nothing launches without parent GO.

Uses the maintained authenticated collector, direct-child capture wrapper,
subreaper, lock and HTTP client. No ordinary paired protocol is modified.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from profile_overlap_v1 import protocol as p, tp1_collector as tc, analyze, scope_contract, role_context
from paired_read_v1 import quality
from runnerhelpers import execution_v2 as ex, library_maps_v1 as maps
from runnerhelpers.closure_v2 import evidence
from runnerhelpers.trace_v2 import json_events, fields, parse_token_trace, join_token_trace_http

ct, require, live = p.ct, p.require, tc.live


def authorization(directory, path, digest):
    prepared = p.prepared(directory)
    p.require_launch(prepared)
    grant = live.sealed(Path(path), digest)
    require(grant.get('schema') == p.ID + '.parent-go' and grant.get('decision') == 'GO'
            and grant.get('prepared_sha256') == ct.sha(directory / 'prepared.json')
            and grant.get('directory') == str(directory) and grant.get('pair_ceiling') == 1
            and grant.get('order') == p.ORDER and grant.get('ordinary_timing_closed') is True
            and grant.get('sole_gpu_owner') is True and grant.get('instrumentation_all_eight') is True
            and grant.get('pressure_cleanup_complete') is True
            and grant.get('selected_topology') == p.TOPOLOGY
            and grant.get('warning_decision') == scope_contract.policy('process-tree'),
            'separate explicit parent GO after ordinary timing is required')
    now = time.time_ns()
    require(grant.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            and type(grant.get('issued_unix_ns')) is int and type(grant.get('expires_unix_ns')) is int
            and grant['issued_unix_ns'] <= now < grant['expires_unix_ns'] <= grant['issued_unix_ns'] + 900_000_000_000,
            'stale/expired parent authorization')
    closeout = grant.get('ordinary_closeout', {})
    require(closeout == prepared['ordinary_closeout'] and closeout.get('path')
            and closeout.get('sha256') == ct.sha(closeout['path']),
            'parent must pin ordinary closeout (failure allowed; never inherit its acceptance)')
    pressure = grant.get('pressure_closeout', {})
    require(pressure.get('path') and pressure.get('sha256') == ct.sha(pressure['path']),
            'fresh parent GO must pin completed pressure cleanup')
    return prepared, grant


def config():
    value = copy.deepcopy(ct.BASE)
    value['limits']['case_seconds'] = 300
    value['limits']['idle_seconds'] = 20
    return value


def snapshot(case, label, root, target, prepared):
    directory = case / (label + '-library-evidence')
    raw = maps.capture(directory, root, target, p.BINARY_SHA, live.observer.proc_identity)
    policy = ct.load(Path(prepared['directory']) / 'runtime-policy.json')
    maps.private_write(directory / 'policy.json', policy)
    result = maps.validate_snapshot(raw, policy)
    maps.private_write(directory / 'validation.json', result)
    require(result['status'] == 'passed', 'actual CLI profile maps rejected; raw private evidence preserved')
    return result


def check_environment(root):
    live.observer.verify_root(root)
    value = dict(s.split('=', 1) for s in Path(f'/proc/{root["pid"]}/environ').read_text().split('\0') if '=' in s)
    require(value.get('FERRULE_PROFILE_NVTX') == '1'
            and all(value.get(k) == tc.UUID for k in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES')),
            'actual model NVTX/TP1 visibility differs')
    ordinary_env = p.binding.case_for_mode('blocking')['environment']
    require(value.get('FERRULE_LOG') == ordinary_env['FERRULE_LOG'] and value.get('FERRULE_LOG_FORMAT') == 'json'
            and 'FERRULE_IO_TRACE' not in value, 'actual typed stats or filename trace environment differs')
    require(ct.sha(f'/proc/{root["pid"]}/exe') == p.BINARY_SHA, 'actual model executable is not qualified CLI')
    live.observer.verify_root(root)
    return {k: value[k] for k in ('FERRULE_PROFILE_NVTX', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES')}


def qa_result(case, actual, lines):
    planned = p.qa()
    tokens = parse_token_trace(lines)
    joined = join_token_trace_http(tokens, [dict(planned, actual=actual)], lines)[0]
    usage = actual.get('usage') or {}
    require(actual.get('adapter_http_report_complete') is True and actual.get('status') == 'success'
            and actual.get('text') == planned['reference']['reference_text']
            and usage == dict(prompt_tokens=5, completion_tokens=8, total_tokens=13)
            and actual.get('finish_reason') in ('length', 'stop') and joined['metric_eligible']
            and joined['exact_token_pass'] is True and len(tokens) == 8,
            'one QA actual generation IDs/text/usage/terminals failed before profile acceptance')
    events = [event for _, event in json_events(lines)]
    admitted = [event for event in events if fields(event).get('event') == 'request_admitted']
    terminals = [event for event in events if fields(event).get('event') == 'request_worker_terminal']
    rid = joined['actual_worker_request_id']
    require(len(admitted) == len(terminals) == 1
            and all(fields(event).get('request_id') == rid for event in admitted + terminals),
            'extra/unbound generation request or missing actual QA clock markers')
    begin, end = (analyze.epoch_ns(row['timestamp']) for row in (admitted[0], terminals[0]))
    require(begin < end, 'actual request clock order invalid')
    return dict(status='passed', actual=actual, trace=joined, request_begin_unix_ns=begin,
                request_end_unix_ns=end, request_markers=[admitted[0], terminals[0]],
                profiling=True, ordinary_performance=False)


def read_ledger(lines, root, mode):
    closure = ex.owner_summary(lines, root, tc.UUID)
    require(closure['final']['can_continue'], 'actual owner shutdown is incomplete/unknown')
    state, read = evidence(closure['attempts'][-1]['summary']['physical']['expert_read_pipeline'])
    require(state in ('observed', 'verified'), 'final actual read ledger unavailable')
    quality.read_observation(read, p.binding.raw_options(mode), effective_required=True, final=True)
    stats = read['stats']
    require(all(stats.get(k) == 0 for k in ('failed_reads', 'cancelled_reads', 'unknown_reads')),
            'failed/cancelled reads: one-request profile coverage is not closed')
    invocations = [fields(event) for _, event in json_events(lines)
                   if fields(event).get('event') in ('mixed_decoder_invocation', 'expert_read_decoder_invocation')]
    require(invocations and all(r.get('status') == 'completed'
            and r.get('expert_read_counter_reset_detected') is False for r in invocations),
            'current decoder read counters missing/reset')
    deltas = [json.loads(r['expert_read_pipeline_delta']) for r in invocations]
    for name in ('io_started_reads', 'proved_read_bytes'):
        require(sum(r[name] for r in deltas) == stats[name] > 0, 'trace/final read count reconciliation failed: ' + name)
    image_state, generation = evidence(closure['attempts'][-1]['summary']['physical']['image_generation'])
    require(image_state in ('observed', 'verified') and type(generation) is int and generation > 0
            and read['last_receipt']['identity']['image_generation'] == generation, 'actual owner/source generation absent or mismatched')
    require(all(type(stats.get(k)) is int and stats[k] == stats['io_started_reads'] > 0 for k in
                ('submitted_reads', 'worker_started_reads', 'completed_reads', 'verified_reads'))
            and stats.get('unobserved_terminal_reads') == 0, 'typed read coverage not positively balanced')
    return dict(closure=closure, read=read, model_source_generation=generation, invocation_count=len(invocations),
                logical_reservations_only=True, full_memory_bound=False)


def export(case):
    argv = p.export_argv(case)
    ct.write(case / 'export-command.json', dict(argv=argv, timeout_seconds=60))
    with (case / 'export.stdout').open('xb') as out, (case / 'export.stderr').open('xb') as err:
        result = subprocess.run(argv, env=ex.runner.cpu_environment(), stdin=subprocess.DEVNULL,
                                stdout=out, stderr=err, timeout=60)
    ct.write(case / 'export-result.json', dict(returncode=result.returncode))
    require(result.returncode == 0 and (case / 'nsys.sqlite').is_file(), 'official full SQLite export failed')


def worker(digest):
    ct.require_cpu_python()
    payload = json.load(sys.stdin)
    auth = live.guard.authenticate(payload, digest)
    case = Path(payload['directory'])
    pair = case.parent.parent.parent
    prepared = p.prepared(pair)
    p.require_launch(prepared)
    require(payload['mode'] in p.ORDER and case == pair / 'runs' / payload['mode'] / 'session'
            and payload['prepared_sha256'] == ct.sha(pair / 'prepared.json'), 'unbound profile worker')
    live.validate_locks(payload['owner_locks'], auth['coordinator']['pid'])
    require(ct.sha(case / 'baseline.json') == payload['baseline_sha256'], 'original B0 changed')
    before = live.observer.idle(config(), (tc.UUID,), p.NOTE, payload['baseline'])
    ct.write(case / 'prelaunch.json', before)
    require(before['status'] == 'verified' and all(d['gpu_bytes'] == 0 for d in before['baseline'].values()),
            'original all-eight B0 required before collector')
    spec = prepared['specs'][payload['mode']]
    ex.runner.require_free_port(spec)
    target_digest = live.target_manifest(case, spec['argv'])
    env = dict(spec['environment'], TMPDIR=str(case / 'nsys-tmp'),
               FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
    argv = p.profile_argv(case, target_digest)
    ct.write(case / 'collector-command.json', dict(argv=argv, environment=env))
    outcome = dict(status='failed', mode=payload['mode'], start=ex.stamp(), ordinary_performance=False,
                   authenticated_worker=auth)
    server = monitor = root = None
    launched = time.monotonic()
    try:
        with (case / 'server.log').open('xb') as out:
            server = subprocess.Popen(ct.isolated_argv(argv, env), stdin=subprocess.DEVNULL,
                                      stdout=out, stderr=subprocess.STDOUT, cwd=ct.ROOT)
        monitor = live.CollectorMonitor(case, server, config(), auth, (tc.UUID,), p.NOTE, payload['baseline'])
        monitor.start()
        startup_deadline = launched + 120
        target, root = tc.model_identity(server, case, startup_deadline, auth)
        outcome.update(server_identity=root, target_identity=target, collector_identity=monitor.root)
        facade = ex.CapturedServer(server, case, root)
        require(time.monotonic() < startup_deadline, 'startup budget consumed by collector handshake')
        ready = ex.http.wait_ready(spec, ex.runner.StartupGuard(facade, monitor), startup_deadline - time.monotonic())
        require(ready <= startup_deadline, 'fixed startup 120s exceeded')
        monitor.mark_ready(ready)
        outcome['ready'] = ex.stamp()
        outcome['startup_seconds'] = ready - launched
        outcome['actual_environment'] = check_environment(root)
        outcome['ready_maps'] = snapshot(case, 'ready', root, target, prepared)
        outcome['process_role_receipts'] = {'ready': role_context.capture(case, 'ready', outcome, prepared)}
        live.live_auth(monitor.manifest)
        require(not monitor.failed.is_set(), 'observer failed before single QA admission')
        folder = case / 'canary-qa'
        folder.mkdir(mode=0o700)
        deadline = min(time.monotonic() + p.LIMITS['request_seconds'], launched + 145)
        require(deadline > time.monotonic(), 'no request budget before finite trace end')
        ct.write(folder / 'issued.json', dict(p.qa(), admitted=ex.stamp(), deadline=deadline))
        actual = ex.one_request(folder, p.qa(), spec, deadline)
        ct.write(folder / 'http-terminal.json', actual)
        qa = qa_result(case, actual, ex.log_lines(case / 'server.log'))
        ct.write(folder / 'quality.json', qa)
        outcome['quality'] = qa
        outcome['after_maps'] = snapshot(case, 'after-requests', root, target, prepared)
        outcome['process_role_receipts']['after-requests'] = role_context.capture(case, 'after-requests', outcome, prepared)
        require(not monitor.failed.is_set() and time.monotonic() < launched + 150, 'capture ended before QA/bookkeeping')
        outcome['status'] = 'qa_complete'
    except (Exception, KeyboardInterrupt) as error:
        outcome['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        if monitor is not None:
            try:
                outcome['monitor'] = monitor.finish()
            except (Exception, KeyboardInterrupt) as error:
                outcome['monitor'] = dict(errors=[type(error).__name__ + ': ' + str(error)])
        if server is not None:
            try:
                # Maintained stop marker -> direct model child TERM -> natural
                # collector exit/export. Never nsys kill or a discovered PID signal.
                outcome['shutdown'] = live.shutdown(server, case, 100)
            except (Exception, KeyboardInterrupt) as error:
                outcome['cleanup_error'] = type(error).__name__ + ': ' + str(error)
        outcome['ended'] = ex.stamp()
        ct.write(case / 'worker-capture.json', outcome)
    try:
        require(outcome['status'] == 'qa_complete' and not outcome.get('cleanup_error')
                and outcome['shutdown']['returncode'] == 0 and not outcome['shutdown']['timed_out']
                and not outcome['monitor']['errors']
                                and outcome['monitor'].get('memory_components', {}).get('gpu_samples', 0) > 0,
                'collector/observer/QA did not close successfully')
        ledger = read_ledger(ex.log_lines(case / 'server.log'), root, payload['mode'])
        ct.write(case / 'read-ledger.json', ledger)
        export(case)
        report = audit_capture(case, outcome, qa, ledger)
        ct.write(case / 'worker-profile-audit.json', report)
        require(scope_contract.sql_eligible(report), 'scoped official trace audit failed; stop before next arm')
        return 0
    except (Exception, KeyboardInterrupt) as error:
        failure = dict(error=type(error).__name__ + ': ' + str(error))
        # A failed QA/map gate can still leave an official report after natural
        # cleanup. Export it once for diagnostics, never for full acceptance.
        if ((case / 'nsys.nsys-rep').is_file() and not (case / 'export-command.json').exists()
                and outcome.get('shutdown', {}).get('returncode') == 0
                and not outcome.get('cleanup_error')):
            try:
                export(case)
                import sqlite3
                context = dict(model_pid=outcome['server_identity']['pid'],
                    model_birth_ticks=outcome['server_identity']['birth_ticks'],
                    wrapper_pids=[outcome['target_identity']['pid'], outcome['collector_identity']['pid']])
                with sqlite3.connect((case / 'nsys.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
                    ct.write(case / 'partial-profile-audit.json', analyze.raw_summary(db, context))
            except (Exception, KeyboardInterrupt) as diagnostic_error:
                failure['diagnostic_error'] = type(diagnostic_error).__name__ + ': ' + str(diagnostic_error)
        ct.write(case / 'worker-acceptance-failure.json', failure)
        return 1


def run(directory, go, go_sha, sole):
    require(sole is True, 'renewed sole-owner grant required; preparation is CPU-only')
    prepared, grant = authorization(directory, go, go_sha)
    inventory = sorted(prepared['candidate']['uuid_inventory'])
    require(not (directory / 'attempt.json').exists(), 'one pair already attempted; no replay')
    probe = dict(scope=tc.SCOPE, granted=True, note=p.NOTE, gpu_uuids=inventory,
                 model_gpu_uuids=[tc.UUID], run_id=directory.name,
                 prepared_sha256=ct.sha(directory / 'prepared.json'))
    live.readonly(directory / 'parent-go.json', grant)
    live.readonly(directory / 'probe-grant.json', probe)
    ct.write(directory / 'attempt.json', dict(order=p.ORDER, started=ex.stamp(), parent_go_sha256=go_sha))
    result = dict(status='failed', arms=[], ordinary_performance=False, profile_cost='whole collector lifecycle',
                  started=ex.stamp(), original_idle_verified=False, authenticated_drained=False,
                  scoped_overlap_accepted=False, full_trace_accepted=False)
    with tc.scope.probe_locks(inventory, (tc.UUID,), probe, ct.exclusive_owner_lock) as locks:
        # This separately authorized fixed generation has its own one-shot claim.
        # Never overwrite the rejected predecessor's global capture claim.
        ct.write(directory / 'capture-claimed.json', dict(directory=str(directory), parent_go_sha256=go_sha,
                                                        predecessor=str(p.PRIOR)))
        try:
            for mode in p.ORDER:
                case = directory / 'runs' / mode / 'session'
                (case / 'nsys-tmp').mkdir(mode=0o700, parents=True)
                before = live.observer.idle(config(), (tc.UUID,), p.NOTE)
                ct.write(case / 'before.json', before)
                require(before['status'] == 'verified' and all(d['gpu_bytes'] == 0 for d in before['baseline'].values()),
                        'original all-eight idle not verified')
                ct.write(case / 'baseline.json', before['baseline'])
                payload = dict(directory=str(case), mode=mode, owner_locks=locks, baseline=before['baseline'],
                    baseline_sha256=ct.sha(case / 'baseline.json'), prepared_sha256=ct.sha(directory / 'prepared.json'))
                payload, digest = live.guard.seal(payload, case)
                env = dict(ex.runner.cpu_environment(), FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
                argv = ct.isolated_argv([p.PYTHON, '-S', '-B', str(HERE / 'run.py'), 'worker', '--digest', digest], env)
                try:
                    receipt = live.baseline.base.supervisor.run_process(argv, 300, payload)
                except BaseException as error:
                    receipt = getattr(error, 'supervisor_result', {})
                    receipt.setdefault('error', str(error))
                ct.write(case / 'supervisor.json', receipt)
                import gpu_idle_window_v2 as idle_v2
                idle = idle_v2.observe(prepared['candidate']['uuid_inventory'], custody=receipt)
                ct.write(case / 'final-idle-v2.json', idle)
                v2_verified = (idle['status'] == 'verified' and idle['all_eight_idle'] is True
                    and idle['authenticated_echild'] is True and idle['recorded_owners_absent'] is True)
                after = (live.observer.idle(config(), (tc.UUID,), p.NOTE, before['baseline']) if v2_verified
                         else dict(status='blocked', baseline=before['baseline'], observations=[],
                                   error='V2 post-ECHILD idle failed; no further device query'))
                ct.write(case / 'after.json', after)
                arm = dict(mode=mode, authenticated_drained=ct.drained(receipt),
                    supervisor_success=ct.successful_process(receipt),
                    original_idle_verified=v2_verified and after['status'] == 'verified'
                        and all(d['gpu_bytes'] == 0 for d in after['baseline'].values()),
                    b1_returned=(case / 'collector-after-model.json').is_file())
                result['arms'].append(arm)
                require(all(arm[k] is True for k in ('authenticated_drained', 'supervisor_success',
                                                     'original_idle_verified', 'b1_returned')), 'arm failed; no replacement')
            result.update(status='captured_requires_offline_acceptance', original_idle_verified=True, authenticated_drained=True)
        except (Exception, KeyboardInterrupt) as error:
            result['error'] = type(error).__name__ + ': ' + str(error)
        finally:
            result['authenticated_drained'] = bool(result['arms']) and all(a['authenticated_drained'] for a in result['arms'])
            result['original_idle_verified'] = bool(result['arms']) and all(a['original_idle_verified'] for a in result['arms'])
            result['full_trace_accepted'] = False
            result['unissued_arms'] = p.ORDER[len(result['arms']):]
            result['ended'] = ex.stamp()
            ct.write(directory / 'capture-result.json', result)
    ct.write(directory / 'lock-release.json', dict(local_lock_context_exited=True, gpu_uuids=inventory,
        physical_release='unverified; ECHILD and original idle are process/device observations, not universal release'))
    return result


def collector_log_issues(path, lines):
    issues = []
    for number, line in enumerate(lines, 1):
        try:
            event = json.loads(line)
        except ValueError:
            event = None
        if isinstance(event, dict):
            bad = event.get('level') in ('WARN', 'WARNING', 'ERROR', 'FATAL')
        else:
            bad = re.search(r'^\s*(?:WARNING|WARN|ERROR|FATAL)\b|Not all CUDA events|StringStorage|NumTpcs|TargetProfilingFailed', line, re.I)
        if scope_contract.LOSS.search(line):
            # Nullable JSON error fields are not errors; explicit loss messages are.
            message = fields(event).get('message', '') if isinstance(event, dict) else line
            bad = bad or bool(scope_contract.LOSS.search(message))
        if bad:
            issues.append(dict(path=str(path), line=number, text=line))
    return issues


def audit_capture(case, worker, qa, ledger):
    """Preserve warnings and counts even when strict schema/attribution rejects.

    The worker calls this after export, before the coordinator can admit the next
    arm. It is evidence analysis, not a replacement capture or a PASS override.
    """
    prepared = p.prepared(case.parent.parent.parent)
    context = dict(scope_contract=prepared['overlap_scope'], model_source_generation=ledger['model_source_generation'],
        model_pid=worker['server_identity']['pid'], model_birth_ticks=worker['server_identity']['birth_ticks'],
        uuid=tc.UUID,
        request_begin_unix_ns=qa['request_begin_unix_ns'], request_end_unix_ns=qa['request_end_unix_ns'],
        read_stats=ledger['read']['stats'], last_receipt=ledger['read'].get('last_receipt'),
        cupti_path=str(p.TOOL / 'target-linux-x64/libcupti.so.12.9'))
    log_issues = []
    for name in ('server.log', 'export.stdout', 'export.stderr'):
        log_issues.extend(collector_log_issues(case / name, ex.log_lines(case / name)))
    import sqlite3
    with sqlite3.connect((case / 'nsys.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        diagnostics = analyze.rows(db, 'DIAGNOSTIC_EVENT') if 'DIAGNOSTIC_EVENT' in tables else None
        counts = {name: db.execute('SELECT count(*) FROM ' + name).fetchone()[0]
                  for name in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_RUNTIME',
                               'CUPTI_ACTIVITY_KIND_DRIVER', 'NVTX_EVENTS', 'OSRT_API') if name in tables}
    try:
        with sqlite3.connect((case / 'nsys.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            context.update(role_context.bind(case, worker, prepared, analyze.rows(db, 'PROCESSES')))
        report = analyze.analyze(case / 'nsys.sqlite', context)
    except (ValueError, KeyError, sqlite3.Error) as error:
        report = dict(status='rejected', full_trace_accepted=False, scoped_overlap_accepted=False,
                      scoped_sql_evidence_consistent=False, scoped_issues=[str(error)], counts=counts, diagnostics=diagnostics,
                      issues=[type(error).__name__ + ': ' + str(error)], warnings=None,
                      attributed_kernel_count=None, correlation_counts=None, dropped_event_count=None,
                      read_envelope_kernel_intersection_ns=None, pread_kernel_intersection_ns=None)
        try:
            with sqlite3.connect((case / 'nsys.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
                partial = analyze.raw_summary(db, context)
            report.update(warnings=partial['warnings'], correlation_counts=partial['correlation_counts'],
                          diagnostic_partial=partial, osrt_counter_coverage=partial['osrt_counter_coverage'])
        except (ValueError, KeyError, sqlite3.Error) as partial_error:
            report['diagnostic_partial_error'] = type(partial_error).__name__ + ': ' + str(partial_error)
    return dict(report, log_issues=log_issues, ordinary_performance=False,
                private_report_path=str(case / 'nsys.nsys-rep'), private_sqlite_path=str(case / 'nsys.sqlite'),
                report_sha256=ct.sha(case / 'nsys.nsys-rep'), sqlite_sha256=ct.sha(case / 'nsys.sqlite'))


def capture_evidence(case, worker, supervisor):
    """Offline replay of the maintained collector's identity/baseline handoff."""
    manifest_path = case / 'collector-manifest.json'
    manifest = live.sealed(manifest_path, ct.sha(manifest_path))
    launch = live.sealed(case / 'worker-launch.json', manifest['worker_launch_sha256'])
    require(manifest['authenticated'] == worker['authenticated_worker']
            and manifest['command_sha256'] == ct.digest(manifest['command'])
            and manifest['binary_sha256'] == p.BINARY_SHA
            and launch['owner_locks'] == manifest['locks'], 'offline authenticated manifest binding differs')
    target = live.sealed(case / 'collector-target-ready.json', ct.sha(case / 'collector-target-ready.json'))
    gate = live.sealed(case / 'model-release.json', ct.sha(case / 'model-release.json'))
    require(target['identity'] == worker['target_identity'] and gate['target'] == target
            and target['manifest_sha256'] == gate['manifest_sha256'] == ct.sha(manifest_path)
            and target['nonce'] == gate['nonce'] == manifest['nonce']
            and gate['launch_id'] == manifest['authenticated']['launch_id']
            and gate['collector'] == worker['collector_identity']
            and gate['model_gpu_uuids'] == [tc.UUID]
            and worker['server_identity']['ppid'] == worker['target_identity']['pid'],
            'offline collector/inner target/model identity differs')
    ended = live.sealed(case / 'collector-model-ended.json', ct.sha(case / 'collector-model-ended.json'))
    wrapper = ct.load(case / 'profile-server.json')
    require(ended['target'] == target and ended['manifest_sha256'] == ct.sha(manifest_path)
            and ended['result'] == 0 and ended['wrapper_sha256'] == ct.sha(case / 'profile-server.json')
            and wrapper['pid'] == worker['server_identity']['pid'] and wrapper['returncode'] == 0
            and wrapper['stop_reason'] == 'matrix_worker_stop_marker', 'model shutdown receipt not closed')
    frozen = live.sealed(case / 'collector-baseline.json', gate['baseline_sha256'])
    state = live.state_from_manifest(manifest)
    require(frozen['samples'][-1]['monotonic'] - manifest['created_monotonic'] <= 120,
            'collector B1 exceeded fixed startup budget')
    state.collector_started(frozen['collector_identity'], frozen['samples'][0]['monotonic'])
    state.freeze(frozen['samples'], frozen['collector_identity'])
    require(frozen['collector_identity'] == dict(gate['collector'], launch_id=gate['launch_id']), 'B1 collector identity differs')
    state.admit_model((tc.UUID,), frozen['collector_identity'])
    after_model = live.sealed(case / 'collector-after-model.json', ct.sha(case / 'collector-after-model.json'))
    returned = (len(after_model['samples']) >= 2 and all(
        tc.scope.returned_to_instrumentation(row, state.anchor, (tc.UUID,)) for row in after_model['samples']))
    state.after_model(ct.drained(supervisor), returned)
    after = ct.load(case / 'after.json')
    state.finish(ct.drained(supervisor), after['observations'][-1])
    release = live.sealed(case / 'collector-release.json', ct.sha(case / 'collector-release.json'))
    require(release['failure'] is None and release['manifest_sha256'] == ct.sha(manifest_path), 'collector drain release failed')


def verify_final_idle(receipt, inventory):
    """Replay V2 raw samples offline, not its saved PASS booleans alone."""
    import gpu_idle_window_v2 as idle
    require(receipt['schema'] == idle.SCHEMA and receipt['status'] == 'verified'
            and receipt['all_eight_idle'] is True and receipt['authenticated_echild'] is True
            and receipt['recorded_owners_absent'] is True and receipt['mode'] == 'post_authenticated_echild'
            and receipt['deadline_seconds'] == idle.SECONDS and receipt['required_consecutive'] == idle.CONSECUTIVE
            and 0 <= receipt['elapsed_seconds'] < idle.SECONDS, 'V2 final idle receipt missing or failed')
    samples = receipt['samples']
    require(len(samples) >= idle.CONSECUTIVE and receipt['accepted_sample_indices'] ==
            list(range(len(samples) - idle.CONSECUTIVE, len(samples))), 'final idle consecutive coverage differs')
    consecutive = 0
    previous_end = None
    for sample in samples:
        state = idle.classify(sample, inventory)
        require(state in ('idle', 'utilization_only_transient') and sample['classification'] == state,
                'raw final idle memory/apps/inventory invalid')
        begin, end = sample['monotonic_started'], sample['monotonic_finished']
        require(begin <= end and (previous_end is None or begin >= previous_end), 'idle clock coverage differs')
        previous_end = end
        consecutive = consecutive + 1 if state == 'idle' else 0
        require(sample['consecutive_idle'] == consecutive, 'idle raw consecutive count differs')
    require(consecutive == idle.CONSECUTIVE and receipt['consecutive_idle'] == consecutive
            and samples[-1]['monotonic_finished'] - samples[0]['monotonic_started'] < idle.SECONDS,
            'final idle window incomplete or expired')


def accept(directory):
    prepared = p.prepared(directory)
    result = ct.load(directory / 'capture-result.json')
    require(result['status'] == 'captured_requires_offline_acceptance'
            and [r['mode'] for r in result['arms']] == p.ORDER
            and result['authenticated_drained'] and result['original_idle_verified'], 'pair capture/release not complete')
    arms = []
    for mode in p.ORDER:
        case = directory / 'runs' / mode / 'session'
        worker = ct.load(case / 'worker-capture.json')
        supervisor = ct.load(case / 'supervisor.json')
        require(ct.drained(supervisor) and ct.successful_process(supervisor),
                'actual authenticated subreaper not drained')
        capture_evidence(case, worker, supervisor)
        require(worker['actual_environment'] == dict(FERRULE_PROFILE_NVTX='1', CUDA_VISIBLE_DEVICES=tc.UUID,
                                                     NVIDIA_VISIBLE_DEVICES=tc.UUID), 'actual profile environment changed')
        require(worker['status'] == 'qa_complete' and worker['shutdown']['returncode'] == 0
                and not worker['shutdown']['timed_out'] and not worker.get('cleanup_error')
                and not worker['monitor']['errors'] and (case / 'collector-after-model.json').exists(), 'worker capture rejected')
        spec = prepared['specs'][mode]
        command = ct.load(case / 'collector-command.json')
        require(command['argv'] == p.profile_argv(case, ct.sha(case / 'collector-manifest.json'))
                and command['environment']['FERRULE_PROFILE_NVTX'] == '1'
                and ct.load(case / 'collector-manifest.json')['command'] == spec['argv'], 'capture command/spec drift')
        require(ct.load(case / 'export-command.json')['argv'] == p.export_argv(case)
                and ct.load(case / 'export-result.json')['returncode'] == 0, 'full export command missing')
        # Recheck raw inner-child snapshots, not only saved PASS summaries.
        policy = ct.load(directory / 'runtime-policy.json')
        for name in ('ready', 'after-requests'):
            folder = case / (name + '-library-evidence')
            raw = ct.load(folder / 'snapshot.json')
            require(raw['root_identity'] == worker['server_identity']
                    and raw['parent_identity'] == worker['target_identity']
                    and ct.load(folder / 'policy.json') == policy
                    and maps.validate_snapshot(raw, policy)['status'] == 'passed', 'saved actual CLI maps no longer validate')
        lines = ex.log_lines(case / 'server.log')
        actual = ct.load(case / 'canary-qa/http-terminal.json')
        qa = qa_result(case, actual, lines)
        ledger = read_ledger(lines, worker['server_identity'], mode)
        report = audit_capture(case, worker, qa, ledger)
        report.update(mode=mode, private_case=str(case),
                      private_map_paths=[str(case / (name + '-library-evidence/maps.raw')) for name in ('ready', 'after-requests')],
                      sqlite_sha256=ct.sha(case / 'nsys.sqlite'), report_sha256=ct.sha(case / 'nsys.nsys-rep'))
        verify_final_idle(ct.load(case / 'final-idle-v2.json'), prepared['candidate']['uuid_inventory'])
        gates = dict(one_frozen_qa_http_token_join=True, authenticated_model_pid_birth_source=True,
            ready_and_after_raw_maps=True, dynamic_owner_pair_closed=True, authenticated_echild=True,
            original_all_eight_stable_idle=True, collector_export_closed=True)
        report.update(scope_contract.acceptance(report, gates), lifecycle_gates=gates)
        ct.write(case / 'overlap-analysis.json', report)
        require(report['scoped_overlap_accepted'], 'scoped SQL/collector/coverage rejected; preserved analysis')
        arms.append(dict(mode=mode, qa_ids=qa['trace']['actual_generated_ids'], analysis=report))
    require(arms[0]['qa_ids'] == arms[1]['qa_ids'], 'paired generated token IDs differ')
    final = dict(schema=p.ID + '.acceptance', status='scoped_overlap_accepted', arms=arms, ordinary_performance=False,
        captures=2, request_count=2, scoped_overlap_accepted=True,
        full_trace_accepted=all(a['analysis']['full_trace_accepted'] for a in arms),
        dropped_event_count=None, simultaneous_unreported_loss_excluded=False,
        universal_trace_completeness_established=False, full_memory_bound=False, profile_cost=result,
        interpretation='One profiled QA per arm, not an ordinary performance estimate or physical-NAS bound')
    ct.write(directory / 'acceptance.json', final)
    return dict(status=final['status'], scoped_overlap_accepted=True, full_trace_accepted=final['full_trace_accepted'],
                directory=str(directory), captures=2, ordinary_performance=False)


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'prepare', 'check', 'run', 'accept', 'worker', 'target'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--digest')
    parser.add_argument('--parent-go', type=Path)
    parser.add_argument('--parent-go-sha256')
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'target':
        require(args.directory is not None and args.digest, 'authenticated target arguments required')
        p.require_launch(p.prepared(args.directory.parent.parent.parent))
        manifest = live.sealed(args.directory / 'collector-manifest.json', args.digest)
        return tc.capture.run(args.directory, args.digest, manifest['command'])
    ct.require_cpu_python()
    if args.mode in ('worker', 'run'):
        def cancelled(signum, frame):
            raise KeyboardInterrupt('owned profile coordinator/worker cancellation')
        signal.signal(signal.SIGTERM, cancelled)
    if args.mode == 'worker':
        return worker(args.digest)
    if args.mode == 'plan':
        value = p.plan()
    else:
        require(args.directory is not None, '--directory required')
        directory = p.directory_check(args.directory)
        if args.mode == 'prepare':
            value = p.prepare(directory)
        elif args.mode == 'check':
            p.prepared(directory)
            value = dict(status='cpu_checked_parent_GO_required', gpu_queries=0, captures_issued=0)
        elif args.mode == 'accept':
            value = accept(directory)
        else:
            require(args.parent_go is not None and args.parent_go_sha256, 'explicit parent GO path and digest required')
            value = run(directory, args.parent_go, args.parent_go_sha256, args.sole_gpu_owner_granted)
    print(json.dumps(value, indent=2), flush=True)
    return 1 if value.get('status') == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
