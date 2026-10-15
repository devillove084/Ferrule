#!/usr/bin/env python3
"""Four-slot pressure specialization of the maintained finite-arm coordinator.

CPU check never launches. Start requires a separate explicit GO and profiler
handoff; next issues at most one point, with no retry or replacement authority.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import inspect
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from types import SimpleNamespace

import prepare as p
from evidence import NAMES, POINT_IDS, counter_totals, denominator_summary, integer

sys.dont_write_bytecode = True
sys.path.insert(0, str(p.HERE.parent))
from runnerhelpers import execution_v2 as ex, trace_v2 as trace
from runnerhelpers.evidence_v2 import ct, require
from runnerhelpers.build_receipts_v1 import immutable
from runnerhelpers.closure_v2 import evidence as observed_value
from paired_read_v1 import current_binding_v3 as qualified, run_paired as paired
from paired_read_v1 import run_actual_v2 as coordinator, quality
from paired_read_v1.final_adapter_v3 import binding as prior_binding, evidence as prior_evidence
from paired_read_v1.map_evidence_v1 import adapter as mapped
from newversionedprotocol import run_pilot as pilot

BINDING_PATH = p.HERE / 'bindings.json'
ENTRY = Path(__file__).resolve()
REGISTRATION_SCHEMA = 'ferrule.storage-demand.pressure-registration.v1'
TRIALS = [point + '-blocking' for point in POINT_IDS]
CONTRACT_KEYS = paired.CONTRACT_KEYS
IDLE_OBSERVER = pilot.idle_window.observe
WORKFLOW_TOOLS = ('/home/admin/wxw403883/anaconda3/bin/python3.12', '/usr/bin/env', '/usr/bin/nvidia-smi')
SCHEMA = 'ferrule.storage-demand.pressure-worker.v1'
NOTE = 'Separate parent GO and profiling-owner handoff required for three blocking pressure points; no retries or profiling'
REQUEST_EVENTS = ('request_admitted', 'request_http_admitted', 'request_worker_terminal',
                  'request_worker_terminal_enqueue', 'request_http_terminal')


def design():
    return p.validate_design(p.load(p.HERE / 'plan.json'))


def schedule():
    return p.frozen_requests(design())


def validate_schedule(rows, ceiling=4):
    require(type(ceiling) is int and ceiling == 4, 'pressure request ceiling must be exactly four')
    require(type(rows) is list and len(rows) == 4 and ct.digest(rows) == ct.digest(schedule()),
            'exact frozen Hello2 and three canonical eight-ID requests required')
    return dict(status='checked', count=4, ceiling=4, planned_completion_tokens=26)


def case_for_point(point):
    require(point in POINT_IDS, 'unknown pressure point; no interpolation/adaptation')
    value = design()
    base = p.load(p.ROOT / value['pressure']['case_source'])['blocking']
    requested = next(r for r in p.cases(value) if r['id'] == point)
    return dict(id=point, argv_after_binary=requested['requested_argv_after_binary'],
                environment=copy.deepcopy(base['environment']), read_options=copy.deepcopy(base['read_options']),
                expected_scheduler=copy.deepcopy(base['expected_scheduler']),
                requested_entries=requested['requested_entries'], requested_cache_bytes=requested['requested_cache_bytes'],
                full_numerical_per_point='unverified', source_preflight=requested['static_header_preflight'])


def seal_design():
    """Deterministic contents for the owned input seal, never a launch/quality PASS."""
    value = design()
    return dict(schema='ferrule.storage-demand.pressure-budget-variant.v1',
        plan_sha256=p.sha(p.HERE / 'plan.json'), request_ceiling=4, trial_order=TRIALS,
        schedule_sha256=ct.digest(schedule()),
        case_sha256={point: ct.digest(case_for_point(point)) for point in POINT_IDS},
        limits=value['pressure']['limits'], prior_numerical_scope='middle64_320MiB_only',
        new_point_full_numerical_status='unverified', gpu_authorized=False, launch_entry=True)


def validate_execution(path=BINDING_PATH):
    require(Path(path) == BINDING_PATH, 'only the owned pressure seal is accepted')
    value = p.check_bindings()
    require(value.get('budget_variant') == seal_design(), 'pressure budget variant seal stale')
    pins = {str(p.ROOT / name): pin for name, pin in value['files'].items()}
    tools = value.get('workflow_tools', {})
    require(set(tools) == set(WORKFLOW_TOOLS), 'exact maintained CPU/launch/telemetry tools required')
    ct.check_pins(tools)
    pins.update(tools)
    pins[str(BINDING_PATH)] = p.sha(BINDING_PATH)
    return dict(artifact_pins=pins, limits=design()['pressure']['limits'], trial_order=list(TRIALS),
                schedule=schedule(), request_ceiling=4, launch_entry=True, gpu_authorized=False)


def validate_candidate(path=BINDING_PATH):
    """Review the compiled 86 cut and its prior middle evidence, not current HEAD.

The historical seal is integrity evidence only. Its middle-case acceptance is
not relabeled as a new-point numerical gate; no source tree/receipt is recaptured.
"""
    validate_execution(path)
    value = design()
    recorded = p.load(p.ROOT / value['candidate']['binding'])
    require(recorded.get('digest') == ct.digest({k: v for k, v in recorded.items() if k != 'digest'}),
            'prior qualified binding digest changed')
    identity = qualified.candidate_identity(qualified.QUALIFICATION)
    require(recorded['candidate_identity'] == identity
            and identity['commit'] == value['candidate']['commit']
            and identity['cli_binary_sha256'] == value['candidate']['cli_binary_sha256']
            and identity['source_digest'] == value['candidate']['source_digest'], 'wrong compiled candidate')
    record = ct.load(recorded['review_path'])
    require(ct.sha(recorded['review_path']) == recorded['review_sha256']
            and record['candidate_identity'] == identity, 'prior raw review relation changed')
    ct.check_pins(recorded['candidate']['artifact_pins'])
    candidate = {key: recorded['candidate'][key] for key in ('binary', 'binary_sha256', 'runtime_pin_manifest')}
    require(candidate == ct.load(BINDING_PATH).get('candidate')
            and Path(candidate['binary']) == p.ROOT / value['candidate']['binary'], 'pressure CLI/runtime identity differs')
    immutable(candidate['binary'])
    require(ct.sha(candidate['binary']) == identity['cli_binary_sha256'], 'compiled CLI bytes changed')
    runtime = ct.load(candidate['runtime_pin_manifest'])
    require(runtime['pins'] and runtime['per_role']['cli'], 'runtime reference inventory empty')
    ct.check_pins(runtime['pins'])
    manifest = prior_evidence.load('evidence-manifest.json')
    inventory, source = prior_evidence.source_inventory(identity, manifest)
    prior_binding.disposition(inventory, manifest)
    qualification = prior_evidence.load('qualification-plan-v2.json')
    final = prior_evidence.load('FINAL.json')
    qualified.validate_gate_plan(qualification['gates'])
    require(final['status'] == 'passed' and final['all_nine_qualified'] is True
            and final['actual_gpu_tests_issued'] == 9 and final['not_issued_gates'] == []
            and final['source_digest'] == identity['source_digest']
            and final['qualification_plan_sha256'] == ct.sha(qualified.QUALIFICATION / 'qualification-plan-v2.json'),
            'prior nine-gate qualification missing/incomplete')
    references = prior_evidence.load('reference-pins.json')
    require(references['reference_files'] and references['model_metadata'], 'empty reference/source inventory')
    reviewer = qualified.reviewer(qualified.QUALIFICATION, identity)
    actual_review = prior_evidence.gate_review(reviewer, qualification, source, runtime)
    require(actual_review == record['quality_review'] and len(actual_review['gates']) == 9,
            'prior middle precision/reference/gate review differs')
    # Per-request rechecks retain prior raw producer pins and the new exact argv seal.
    pins = dict(recorded['candidate']['artifact_pins'], **runtime['pins'],
                **references['reference_files'], **references['model_metadata'])
    pins.update(validate_execution()['artifact_pins'])
    candidate['artifact_pins'] = pins
    return dict(schema='ferrule.storage-demand.pressure-scoped-candidate.v1', candidate=candidate,
        candidate_identity=identity, uuid_inventory=recorded['uuid_inventory'],
        prior_middle_numerical_gates=9, prior_middle_numerical_only=True,
        point_full_numerical_verified=False, visible_pressure_quality='not_run',
        gpu_authorized_by_this_receipt=False, source_preflight_new_caps='not_executed')


def spec_for(candidate, uuid, point):
    pilot.validate_inventory(candidate['uuid_inventory'], uuid)
    require(uuid == candidate['uuid_inventory'][0], 'only the frozen first physical UUID is enabled')
    case = case_for_point(point)
    environment = dict(ex.runner.cpu_environment(), **case['environment'])
    environment.update(CUDA_VISIBLE_DEVICES=uuid, NVIDIA_VISIBLE_DEVICES=uuid,
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return dict(engine='ferrule', argv=[candidate['candidate']['binary'], *case['argv_after_binary']],
                environment=environment, model='qwen35-storage-demand-paper',
                url='http://127.0.0.1:18369/v1/completions')


def worker_contract(payload):
    require(payload.get('schema') == SCHEMA, 'unknown pressure worker schema')
    require(all(k in payload for k in CONTRACT_KEYS) and payload.get('contract_sha256')
            == ct.digest({k: payload[k] for k in CONTRACT_KEYS}), 'sealed pressure worker contract changed')
    require(payload['trial_order'] == TRIALS and payload['trial'] in TRIALS
            and payload['mode'] == 'blocking' and type(payload['request_ceiling']) is int
            and payload['request_ceiling'] == 4, 'pressure point/order/mode/ceiling drift')
    validate_schedule(payload['schedule'], payload['request_ceiling'])
    require(payload['limits'] == design()['pressure']['limits'], 'pressure absolute limits changed')
    pilot.validate_inventory(payload['eight_uuids'], payload['uuid'])
    require(payload['granted'] is True and payload['owner_note'] == NOTE, 'separate renewed owner grant required')
    for key in ('candidate', 'execution'):
        require(Path(payload[key + '_path']) == BINDING_PATH
                and ct.sha(BINDING_PATH) == payload[key + '_sha256'], 'pressure seal path/hash changed')
    execution = validate_execution(payload['execution_path'])
    ct.check_pins(payload['adapter_pins'])
    require(payload['adapter_pins'] == execution['artifact_pins'], 'pressure workflow registration stale')
    candidate = validate_candidate(payload['candidate_path'])
    require(payload['eight_uuids'] == candidate['uuid_inventory']
            and payload['spec'] == spec_for(candidate, payload['uuid'], payload['trial'].rsplit('-', 1)[0]),
            'pressure candidate-bound point/spec changed')
    pilot.telemetry.authorize(True, payload['uuid'], payload['owner_note'], 'device_exclusive_local_custody')
    locks = payload.get('all_uuid_locks', [])
    require(len(locks) == 8, 'all eight authenticated local UUID locks required')
    for uuid, lock in zip(payload['eight_uuids'], locks):
        ct.validate_owner_lock(lock, uuid, payload['coordinator']['pid'])
    require(locks[payload['eight_uuids'].index(payload['uuid'])] == payload['owner_lock'], 'selected lock differs')
    return candidate


def _specialize(function, before, after, **bindings):
    return _specializations(function, ((before, after),), **bindings)


def _specializations(function, replacements, **bindings):
    """Exact expressions in a pinned private globals copy; never shared edits."""
    path = Path(inspect.getsourcefile(function)).resolve()
    pins = p.load(BINDING_PATH)['files']
    name = str(path.relative_to(p.ROOT))
    require(name in pins and p.sha(path) == pins[name], 'maintained specialization source changed')
    source = inspect.getsource(function)
    for before, after in replacements:
        require(source.count(before) == 1, 'maintained expression changed: ' + before)
        source = source.replace(before, after)
    namespace = dict(function.__globals__, **bindings)
    exec(compile(source, str(path) + ':pressure-four-slot', 'exec'), namespace)
    return namespace[function.__name__]


def idle(directory, label, uuids, deadline, custody=None):
    """Same three-sample/identity policy, shortened to the frozen absolute envelope."""
    now = pilot.time.monotonic()
    require(ex.finite(deadline) and now < deadline, 'pressure idle absolute deadline expired')
    absolute = min(deadline, now + pilot.idle_window.SECONDS)
    observe = _specialize(IDLE_OBSERVER, 'deadline = started + SECONDS', 'deadline = absolute_deadline',
                          absolute_deadline=absolute, SECONDS=absolute - now)
    result = observe(uuids, custody=custody)
    if isinstance(result.get('error'), str):
        result['error'] = result['error'].replace('Fixed 15-second', 'Pressure absolute')
    ct.write(Path(directory) / (label + '.json'), result)
    require(result.get('status') == 'verified' and result.get('all_eight_idle') is True,
            'pressure all-eight idle failed: ' + label)
    require(pilot.time.monotonic() <= deadline, 'pressure idle exceeded absolute envelope')
    return result


def serial_requests(*args, **kwargs):
    private = _specialize(ex.serial_requests, 'len(observations) == 12', 'len(observations) == 4',
                          validate_schedule=validate_schedule)
    return private(*args, **kwargs)


def whole_log_join(trace_rows, requests, server_lines=()):
    """The maintained actual-ID join, plus exact whole-lifecycle coverage/order."""
    expected = schedule()
    require(len(requests) == 4 and [r.get('name') for r in requests] == list(NAMES),
            'whole-log join requires all four requests in frozen order')
    for row, frozen in zip(requests, expected):
        require(all(row.get(key) == value for key, value in frozen.items() if key not in ('status', 'result'))
                and row.get('issued') is True and row.get('status') == 'success', 'whole-log request slot/content drift')
    joined = trace.join_token_trace_http(trace_rows, requests, server_lines)
    ids = [row['actual_worker_request_id'] for row in joined]
    require(len(joined) == len(set(ids)) == 4 and all(row['metric_eligible'] for row in joined),
            'four actual HTTP/worker token joins required')
    require([row['request_id'] for row in trace_rows]
            == [request_id for request_id, row in zip(ids, expected) for _ in range(row['max_tokens'])],
            'extra/missing/reordered whole-log tokens')
    events = [trace.fields(event) for _, event in trace.json_events(server_lines)]
    for event in REQUEST_EVENTS:
        rows = [row for row in events if row.get('event') == event]
        require([row.get('request_id') for row in rows] == ids
                and all(row.get('session_id') == row.get('request_id') for row in rows),
                'extra/missing/reordered whole-log event: ' + event)
    for row, joined_row in zip(requests, joined):
        require(ex.request_gate(row, row.get('actual', {}), joined_row,
                                row.get('logical_counters', {}), row.get('nfs', {})),
                'whole-log HTTP/text/usage/source-window gate failed')
    return joined


def _rust_fields(text):
    pairs = re.findall(r'(\w+): (true|false|\d+)(?:,|\s*\})', text + '}')
    require(pairs and len(dict(pairs)) == len(pairs), 'missing/duplicate effective source fields')
    return {key: value == 'true' if value in ('true', 'false') else int(value) for key, value in pairs}


def source_options(point, lines):
    """Check real startup reporting, not a fabricated all-cap CPU lowering receipt."""
    case = case_for_point(point)
    reports = [trace.fields(event).get('report') for _, event in trace.json_events(lines)
               if trace.fields(event).get('message') == 'resident model build entering owner-local admission']
    require(len(reports) == 1 and isinstance(reports[0], str), 'one actual source/owner admission report required')
    report = reports[0]
    require('profile=cuda-hybrid-numeric-fp8-model-bf16-mixed-f32-ssm;' in report
            and 'max_layers: descriptor default -> 40 ' in report
            and 'NumericFp8E4m3fnBf16Scales' in report and 'context=128;' in report,
            'full40 Mixed numeric-FP8 source profile differs')
    capacities = re.findall(r'Qwen35MoeCapacityLimits \{([^}]+)\}', report)
    require(len(capacities) == 2, 'requested/effective source cache lowering absent')
    fixed = design()['pressure']['fixed']
    expected = dict(max_experts=case['requested_entries'], max_bytes=case['requested_cache_bytes'],
        scratch_bytes=fixed['scratch_bytes'], workspace_bytes=fixed['workspace_bytes'], kv_bytes=fixed['kv_bytes'],
        device_bytes=fixed['device_bytes'], allocator_margin_bytes=fixed['allocator_margin_bytes'],
        max_positions=128, max_sequences=2, max_batch_tokens=8)
    require(_rust_fields(capacities[-1]) == expected, 'actual source effective cache/options differ')
    schedulers = re.findall(r'ResidentSchedulerConfig \{([^}]+)\}', report)
    require(len(schedulers) == 2 and all(_rust_fields(s) == case['expected_scheduler'] for s in schedulers),
            'actual source scheduler lowering differs')
    host = 'expert read host staging: schedule=Blocking host_cap_bytes=67108864 metadata_bytes=8388608 worker_stack_bytes=2097152;'
    require(sum(line.startswith(host) for line in lines) == 1, 'actual source read staging options absent/different')
    return dict(status='actual_source_options_observed', point=point, effective_metadata_options=expected,
                effective_admitted_entries=None, effective_admitted_cache_bytes=None,
                physical_admission=None, full_numerical_per_point='unverified')


def point_assessment(point, lifecycle, lines):
    """Only complete four-slot evidence can qualify a point; no empty all() PASS."""
    try:
        source = source_options(point, lines)
        rows = lifecycle.get('requests', {}).get('observations', [])
        joined = whole_log_join(trace.parse_token_trace(lines), rows, lines)
        require(lifecycle.get('status') == 'complete', 'incomplete pressure lifecycle')
        attempts = lifecycle.get('closure', {}).get('attempts', [])
        require(attempts and lifecycle['closure']['final']['can_continue'] is True, 'owner closure missing/unknown')
        physical = attempts[-1]['summary']['physical']
        state, read = observed_value(physical.get('expert_read_pipeline'))
        require(state in ('observed', 'verified') and isinstance(read, dict), 'actual source/read ledger absent')
        options = dict(case_for_point(point)['read_options'], schedule='blocking')
        quality.read_observation(read, options, effective_required=True, final=True)
        invocations = []
        for row in rows:
            for invocation in row['logical_counters']['rows']:
                fields = invocation['fields']
                pipeline = json.loads(fields['expert_read_pipeline'])
                quality.read_observation(pipeline, options, effective_required=True)
                require(fields.get('expert_read_counter_reset_detected') is False, 'source/read counter reset')
                invocations.append(fields)
        totals = counter_totals(invocations)
        require(all(integer(totals[k]) and totals[k] > 0 for k in
                    ('checkpoint_read_calls', 'checkpoint_read_bytes', 'checkpoint_source_checks')),
                'point requires positive actual source-verified reads')
        state, cache = observed_value(physical.get('expert_cache'))
        require(state in ('observed', 'verified') and isinstance(cache, dict), 'actual cache accounting absent')
        case = case_for_point(point)
        require(all(integer(cache.get(k)) for k in ('resident_entries', 'charged_bytes', 'peak_bytes', 'scratch_bytes',
                                                   'pending_upload_bytes', 'unknown_quarantine_bytes'))
                and cache['resident_entries'] <= case['requested_entries']
                and cache['charged_bytes'] <= cache['peak_bytes'] <= case['requested_cache_bytes']
                and cache['scratch_bytes'] == 8 << 20 and cache['pending_upload_bytes'] == 0
                and cache['unknown_quarantine_bytes'] == 0 and cache.get('quarantined') is False,
                'cache charge/entry limits or custody failed')
        source['runtime_admission'] = 'Ready_and_four_requests_completed'
        return dict(can_continue=True, source=source, whole_log_token_join=joined,
                    visible_quality='three_exact_eight_ids_and_Hello2_HTTP_usage',
                    full_numerical_per_point='unverified', invocation_counter_totals=totals,
                    final_owner_cache=cache, final_owner_read=read,
                    owner_peaks_and_cumulative_stats_summed=False, physical_release='unobserved')
    except (ValueError, KeyError, TypeError, IndexError) as error:
        return dict(can_continue=False, stop_reason=str(error), full_numerical_per_point='unverified',
                    invocation_counter_totals=counter_totals([]))


@contextmanager
def private_worker(payload):
    """Yield the maintained authenticated worker with only private count/scope hooks.

No call or launch occurs on context entry. The command entry requires the
separate immutable GO; maintained launch authentication remains mandatory.
"""
    point = payload['trial'].rsplit('-', 1)[0]
    require(payload['trial'] in TRIALS, 'unknown worker point')
    with mapped.mapped_worker(payload):
        original_maps = ex.library_maps
        def timing_maps(root, pins):
            ex.observer.verify_root(root)
            environment = dict(item.split('=', 1) for item in
                Path(f'/proc/{root["pid"]}/environ').read_text().split('\0') if '=' in item)
            require(environment.get('FERRULE_PROFILE_NVTX') == '0', 'actual timing NVTX must be OFF')
            return original_maps(root, pins)
        def point_requests(directory, *args, **kwargs):
            source = source_options(point, ex.log_lines(Path(directory) / 'server.log'))
            ct.write(Path(directory) / 'source-options.json', source)
            return serial_requests(directory, *args, **kwargs)
        facade = SimpleNamespace(**dict(vars(ex), serial_requests=point_requests,
                                        library_maps=timing_maps, join_token_trace_http=whole_log_join))
        worker = _specialize(pilot.worker, "len(result['whole_log_token_join']) == 12",
                             "len(result['whole_log_token_join']) == 4",
                             ex=facade, HERE=p.HERE, worker_contract=worker_contract, idle=idle)
        yield worker


def worker(launch_hash):
    """Parent-entry hook only; maintained authentication runs before any launch."""
    raw = sys.stdin.read()
    payload = json.loads(raw)
    previous = sys.stdin
    try:
        sys.stdin = io.StringIO(raw)
        with private_worker(payload) as maintained_worker:
            code = maintained_worker(launch_hash)
    finally:
        sys.stdin = previous
    folder = Path(payload['directory'])
    lifecycle = ct.load(folder / 'lifecycle.json')
    log = folder / 'server.log'
    lines = ex.log_lines(log) if log.is_file() else []
    assessment = point_assessment(payload['trial'].rsplit('-', 1)[0], lifecycle, lines)
    ct.write(folder / 'point-assessment.json', assessment)
    return 0 if code == 0 and assessment['can_continue'] else 1


def lifecycle_assessment(point, lifecycle, lines, supervisor, post_idle):
    """A point-only result cannot replace the maintained custody/idle assessment."""
    result = ex.lifecycle_assessment(lifecycle, supervisor, post_idle)
    pressure = point_assessment(point, lifecycle, lines)
    if not pressure['can_continue']:
        result['can_continue'] = False
        result['stop_reasons'].append(pressure['stop_reason'])
    result['pressure'] = pressure
    return result


def next_point(directory, registered):
    require(registered['trial_order'] == TRIALS and registered['limits'] == design()['pressure']['limits'],
            'pressure coordinator order/limits changed')
    return coordinator.next_trial(directory, registered)


def review_build(candidate):
    """Recheck the exact preserved build, not current HEAD or a new source cut."""
    identity = candidate['candidate_identity']
    manifest = prior_evidence.load('evidence-manifest.json')
    _, source = prior_evidence.source_inventory(identity, manifest)
    artifacts = {role: prior_evidence.load('artifact-final-' + role + '.json') for role in prior_evidence.ROLES}
    runtime = ct.load(candidate['candidate']['runtime_pin_manifest'])
    build = prior_evidence.build_review(qualified.reviewer(qualified.QUALIFICATION, identity), source, artifacts, runtime)
    recorded = ct.load(ct.load(p.ROOT / design()['candidate']['binding'])['review_path'])
    require(build == recorded['build_review'], 'precise preserved build review changed')
    policy = mapped.maps.role_policy(candidate['candidate']['binary'],
        candidate['candidate']['binary_sha256'], runtime, 'cli')
    require(len(policy['required']) == 12, 'CLI must require all twelve role/dependency libraries')
    return dict(build_review=build, runtime_role_policy=policy, candidate_identity=identity,
                prior_middle_numerical_gates=9, new_point_full_numerical_verified=False)


def sealed_write(path, value):
    ct.write(path, dict(value, digest=ct.digest(value)))
    Path(path).chmod(0o400)


def unseal(path, schema):
    immutable(path)
    value = ct.load(path)
    require(value.get('schema') == schema and value.get('digest') == ct.digest(
        {k: v for k, v in value.items() if k != 'digest'}), 'pressure receipt seal changed')
    return value


def directory_check(directory):
    directory = Path(directory)
    require(directory.parent.resolve() == p.HERE and not directory.is_symlink()
            and re.fullmatch(r'artifacts-go-[A-Za-z0-9_-]{1,60}', directory.name),
            'new private artifacts-go-NAME directly in pressure_fault_v1 required')
    return directory.resolve()


def committed_sources():
    def git(*args):
        return subprocess.check_output(['git', '--no-pager', *args], cwd=p.ROOT, timeout=15)
    commit = git('rev-parse', 'HEAD').decode().strip()
    require(commit == git('rev-parse', '@{upstream}').decode().strip(), 'source commit must already be pushed')
    require(not git('diff', '--cached', '--name-only').strip(), 'index must be empty before GPU ownership')
    names = (p.HERE / 'STAGE_FILES.txt').read_text().splitlines()
    require(len(names) == len(set(names)) == 10, 'exact ten-file source whitelist required')
    for name in names:
        require(git('show', commit + ':' + name) == (p.ROOT / name).read_bytes(), 'uncommitted pressure source: ' + name)
    return dict(commit=commit, source_files={name: ct.sha(p.ROOT / name) for name in names})


def start(directory, granted, profiling_paused):
    require(granted is True and profiling_paused is True, 'explicit GO and profiling-owner paused handoff required')
    directory = directory_check(directory)
    require(not any(p.HERE.glob('artifacts-go-*')), 'a pressure GO already exists; no replacement study')
    execution, candidate = validate_execution(), validate_candidate()
    build = review_build(candidate)
    committed = committed_sources()
    directory.mkdir(mode=0o700, exist_ok=False)
    authorization = dict(schema='ferrule.storage-demand.pressure-go.v1', id=directory.name,
        recorded=ex.stamp(), parent_explicit_go=True, profiling_owner_paused=True,
        active_profiling_captures=0, owner_note=NOTE, sources=committed,
        limits=execution['limits'], trial_order=TRIALS, request_ceiling=4,
        failure_policy=design()['pressure']['failure_policy'],
        scope=design()['pressure']['quality']['scope_decision'], no_other_gpu_work=True,
        retries=0, replacements=0, profiling_captures=0)
    sealed_write(directory / 'GO.json', authorization)
    sealed_write(directory / 'candidate-review.json', dict(schema='ferrule.storage-demand.pressure-build-review.v1', **build))
    cases = dict(schema='ferrule.storage-demand.pressure-case-registration.v1',
        candidate_identity=candidate['candidate_identity'], case_variants=[dict(
            **case_for_point(point), spec=spec_for(candidate, candidate['uuid_inventory'][0], point),
            candidate_case_variant=True, inherits_full_numerical_qualification=False) for point in POINT_IDS])
    sealed_write(directory / 'cases.json', cases)
    now = pilot.time.monotonic()
    value = dict(schema=REGISTRATION_SCHEMA, protocol_id=directory.name, started=ex.stamp(),
        pilot_deadline=now + execution['limits']['phase_seconds'], boot_id=coordinator.boot_id(),
        trial_order=TRIALS, order=list(POINT_IDS), limits=execution['limits'], schedule=schedule(),
        binding_path=str(BINDING_PATH), binding_sha256=ct.sha(BINDING_PATH),
        execution_path=str(BINDING_PATH), execution_sha256=ct.sha(BINDING_PATH),
        go_sha256=ct.sha(directory / 'GO.json'), cases_sha256=ct.sha(directory / 'cases.json'),
        build_review_sha256=ct.sha(directory / 'candidate-review.json'),
        selected_uuid=candidate['uuid_inventory'][0], eight_uuids=candidate['uuid_inventory'],
        owner_authorization=True, owner_note=NOTE, request_ceiling=4,
        scheduled_starts=3, scheduled_requests=12, scheduled_measured_qa=0,
        retries=0, replacements=0, profiling_captures=0, historical_resume_permitted=False)
    sealed_write(directory / 'registration.json', value)
    return dict(status='registered', directory=str(directory), first_point=TRIALS[0], deadline=value['pilot_deadline'])


def registration(directory):
    directory = directory_check(directory)
    value = unseal(directory / 'registration.json', REGISTRATION_SCHEMA)
    execution = validate_execution()
    require(value['boot_id'] == coordinator.boot_id() and value['trial_order'] == TRIALS
            and value['order'] == list(POINT_IDS) and value['limits'] == execution['limits']
            and value['schedule'] == schedule() and value['request_ceiling'] == 4, 'boot/order/budget/schedule drift')
    for key in ('binding', 'execution'):
        require(value[key + '_path'] == str(BINDING_PATH) and value[key + '_sha256'] == ct.sha(BINDING_PATH),
                'pressure candidate/workflow changed')
    for key, name in (('go', 'GO.json'), ('cases', 'cases.json'), ('build_review', 'candidate-review.json')):
        immutable(directory / name)
        require(value[key + '_sha256'] == ct.sha(directory / name), 'immutable pressure registration input changed')
    go = unseal(directory / 'GO.json', 'ferrule.storage-demand.pressure-go.v1')
    require(go['parent_explicit_go'] is True and go['profiling_owner_paused'] is True
            and go['owner_note'] == value['owner_note'] == NOTE and value['owner_authorization'] is True,
            'separate pressure owner handoff missing')
    prior = ct.load(p.ROOT / design()['candidate']['binding'])
    require(value['eight_uuids'] == prior['uuid_inventory'] and value['selected_uuid'] == value['eight_uuids'][0],
            'registered GPU inventory changed')
    pilot.validate_inventory(value['eight_uuids'], value['selected_uuid'])
    return value


def supervise(payload, seconds):
    return _specialize(pilot.supervise, 'str(ENTRY)', 'str(ENTRY)', ENTRY=ENTRY)(payload, seconds)


def can_admit(deadline):
    return p.can_admit(deadline - pilot.time.monotonic())


def ledger(observed, mode):
    require(mode == 'blocking', 'pressure is blocking only')
    binding = SimpleNamespace(raw_options=lambda ignored: dict(case_for_point('middle')['read_options'], schedule=mode))
    return _specialize(coordinator.ledger, 'binding.raw_options(mode)', 'binding.raw_options(mode)',
                       binding=binding)(observed, mode)


def maintained_next():
    """Specialize only entry, counts, scope, and absolute budgets; no new launch loop."""
    replacements = (
        ('absolute 7200s envelope lacks full next-arm plus reserve', 'absolute 900s pressure envelope lacks full point plus reserve'),
        ("outer = min(time.monotonic() + 650, registered['pilot_deadline'] - 100)",
         "outer = min(time.monotonic() + limits['outer_lifecycle_seconds'], registered['pilot_deadline'] - limits['study_cleanup_reserve_seconds'])"),
        ("with ExitStack() as stack:\n                for uuid", "with ex.absolute_alarm(outer), ExitStack() as stack:\n                for uuid"),
        ("pilot.idle(folder, 'coordinator-pre-idle', registered['eight_uuids'], min(time.monotonic() + 20, outer))",
         "pilot.idle(folder, 'coordinator-pre-idle', registered['eight_uuids'], min(time.monotonic() + limits['coordinator_preidle_seconds'], outer))"),
        ("schema='ferrule.storage-demand.paired-read-worker.v1'", 'schema=SCHEMA'),
        ('request_ceiling=12', 'request_ceiling=4'),
        ("spec=planned.spec_for(candidate, registered['selected_uuid'], mode)",
         "spec=planned.spec_for(candidate, registered['selected_uuid'], trial.rsplit('-', 1)[0])"),
        ("pilot_deadline=registered['pilot_deadline'] - 100", "pilot_deadline=registered['pilot_deadline'] - limits['study_cleanup_reserve_seconds']"),
        ("pilot.remaining(outer, 530 + 28 + 20 + 32, 'supervisor and finite cleanup')",
         "pilot.remaining(outer, sum(limits[k] for k in ('supervisor_seconds', 'supervisor_call_overhead_seconds', 'coordinator_postidle_seconds', 'coordinator_finalization_seconds')), 'supervisor and finite cleanup')"),
        ('receipt = planned.supervise(payload, 530)', "receipt = planned.supervise(payload, limits['supervisor_seconds'])"),
        ('assessment = ex.lifecycle_assessment(observed, receipt, post)',
         "assessment = lifecycle_assessment(trial.rsplit('-', 1)[0], observed, ex.log_lines(folder / 'server.log') if (folder / 'server.log').is_file() else [], receipt, post)"),
        ('complete: 12/12 requests, 5/5 measured QA;', 'complete: 4/4 requests, three eight-ID canaries plus Hello2;'),
    )
    binding = SimpleNamespace(validate_execution=validate_execution, validate_binding=validate_candidate)
    planned = SimpleNamespace(can_admit=can_admit, supervise=supervise, spec_for=spec_for, CONTRACT_KEYS=CONTRACT_KEYS)
    private_pilot = SimpleNamespace(**dict(vars(pilot), idle=idle))
    return _specializations(coordinator.execute_next, replacements, registration=registration, next_trial=next_point,
        binding=binding, planned=planned, pilot=private_pilot, actual_adapter=nullcontext, NOTE=NOTE,
        SCHEMA=SCHEMA, ledger=ledger, lifecycle_assessment=lifecycle_assessment)


def execute_next(directory):
    try:
        return maintained_next()(directory_check(directory))
    except (Exception, KeyboardInterrupt) as error:
        coordinator.stop(directory, type(error).__name__ + ': ' + str(error))
        raise


def finalize(directory):
    directory = directory_check(directory)
    with coordinator.coordinator_lock(directory):
        registered = registration(directory)
        rows, records = [], {}
        for point, trial in zip(POINT_IDS, TRIALS):
            folder = directory / trial
            row = ct.load(folder / 'assessment.json') if (folder / 'assessment.json').is_file() else dict(
                trial=trial, status='unissued' if not folder.exists() else 'incomplete',
                summary=ex.recover_requests(folder, registered['schedule']))
            rows.append(row)
            for slot in registered['schedule']:
                request = folder / slot['name']
                if (request / 'observed.json').is_file():
                    records[point, slot['name']] = ct.load(request / 'observed.json')
                elif (request / 'issued.json').is_file():
                    records[point, slot['name']] = dict(issued=True, status='censored')
        complete = all(row.get('assessment', {}).get('can_continue') is True for row in rows)
        require(complete or (directory / 'study-stop.json').exists(), 'remaining points must not be silently omitted')
        complete = complete and not (directory / 'study-stop.json').exists()
        report = dict(schema='ferrule.storage-demand.pressure-final.v1', status='complete' if complete else 'stopped',
            registration_sha256=ct.sha(directory / 'registration.json'), trial_order=TRIALS, lifecycles=rows,
            denominators=denominator_summary(records), ended=ex.stamp(),
            issued_lifecycles=sum((directory / t / 'launch.json').is_file() for t in TRIALS),
            completed_lifecycles=sum(r.get('assessment', {}).get('can_continue') is True for r in rows),
            actual_server_starts=sum((directory / t / 'profile-server.json').is_file() for t in TRIALS),
            comparison_eligible=complete, full_numerical_per_point='unverified',
            prior_numerical_scope='middle64_320MiB_only', repetitions_per_point=1,
            retries=0, replacements=0, profiling_captures=0,
            physical_nas_backend_bytes=None, universal_final_release_verified=False,
            claim_limits=design()['claim_limits'])
        sealed_write(directory / 'pressure-final.json', report)
        return dict(status=report['status'], issued_lifecycles=report['issued_lifecycles'],
            completed_lifecycles=report['completed_lifecycles'], denominators=report['denominators'],
            directory=str(directory))


def main(argv=None):
    ct.require_cpu_python()
    os.umask(0o077)
    def cancelled(signum, frame):
        raise KeyboardInterrupt('owned pressure coordinator/worker cancellation')
    signal.signal(signal.SIGTERM, cancelled)
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--_worker':
        require(len(argv) == 2, 'authenticated pressure worker launch hash required')
        return worker(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'start', 'next', 'finalize'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    parser.add_argument('--profiling-owner-paused', action='store_true')
    args = parser.parse_args(argv)
    if args.mode == 'check':
        candidate = validate_candidate()
        build = review_build(candidate)
        maintained_next()
        _specialize(ex.serial_requests, 'len(observations) == 12', 'len(observations) == 4', validate_schedule=validate_schedule)
        result = dict(status='cpu_checked_separate_GO_required', request_ceiling=4, order=list(POINT_IDS),
            prior_middle_numerical_gates=candidate['prior_middle_numerical_gates'],
            required_cli_libraries=len(build['runtime_role_policy']['required']),
            all_cap_full_numerical_verified=False, all_cap_actual_header_lowering_executed=False,
            launch_entry=True, gpu_queries=0, gpu_locks=0, gpu_runs=0)
    else:
        require(args.directory is not None, 'private pressure directory required')
        if args.mode == 'start':
            result = start(args.directory, args.sole_gpu_owner_granted, args.profiling_owner_paused)
        elif args.mode == 'next':
            row = execute_next(args.directory)
            result = dict(trial=row['trial'], can_continue=row['assessment']['can_continue'], summary=row['summary'])
        else:
            result = finalize(args.directory)
    print(json.dumps(result, indent=2))
    return 1 if result.get('can_continue') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
