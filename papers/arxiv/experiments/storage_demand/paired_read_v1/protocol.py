#!/usr/bin/env python3
"""CPU-only preregistration and read-only candidate binding; never launches a workload."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
STORAGE = HERE.parent
sys.path.insert(0, str(STORAGE))
from runnerhelpers.evidence_v2 import ct, integer, require, validate_schedule
from runnerhelpers.quality_reports_v1 import validate_quality_reports, token_checks, validate_producer_final
from runnerhelpers.build_receipts_v1 import pinned_path
from paired_read_v1 import build_adapter, quality
from newversionedprotocol import protocol as baseline

PROTOCOL_PATH = HERE / 'protocol.json'
CASE_PATH = HERE / 'cases.json'
PREPARED_PATH = HERE / 'cpu-sealed.json'
MODES = ('blocking', 'lookahead')
LABELS = ('full40', 'fresh', 'ordinary', 'cold')
DESIGN_FILES = ('__init__.py', 'protocol.py', 'protocol.json', 'cases.json', 'requests.json',
                'oracles.json', 'order.json', 'source-cut.json', 'build_adapter.py',
                'run_paired.py', 'analysis.py', 'profile_audit.py', 'quality.py', 'README.md',
                'test_paired_read_v1.py', 'test_quality_v1.py')


def expected_order(seed=20261015):
    require(seed == 20261015, 'unregistered order seed')
    blocks = [f'P{i}' for i in range(1, 6)]
    first = set(sorted(blocks, key=lambda b: hashlib.sha256(f'{seed}:{b}'.encode()).hexdigest())[:3])
    return [dict(block=b, modes=list(MODES if b in first else reversed(MODES))) for b in blocks]


def trial_order():
    return [row['block'] + '-' + mode for row in expected_order() for mode in row['modes']]


def case_for_mode(mode):
    require(mode in MODES, 'disabled/unknown expert read mode')
    return copy.deepcopy(ct.load(CASE_PATH)[mode])


def validate_cases(cases):
    require(set(cases) == set(MODES), 'both enabled modes required')
    historical = ct.load(baseline.CASE_PATH)
    common_argv = historical['argv_after_binary'] + [
        '--expert-read-host-mb', '64', '--expert-read-metadata-mb', '8',
        '--expert-read-worker-stack-mb', '2', '--expert-read-schedule', 'blocking']
    common_env = dict(historical['environment'], FERRULE_PROFILE_NVTX='0')
    limits = ct.load(PROTOCOL_PATH)['limits']
    for mode in MODES:
        case = cases[mode]
        argv = list(common_argv)
        argv[-1] = mode
        read = dict(schedule=mode.title(), max_host_bytes=64 << 20,
                    max_metadata_bytes=8 << 20, worker_stack_bytes=2 << 20)
        quality = copy.deepcopy(historical['quality_test_config'])
        quality.update(name='storage-middle64-' + mode,
                       expert_read_pipeline=dict(read, metadata_and_stack_included_in_host_cap=True,
                                                 device_budget_separate=True))
        require(case.get('schema') == 'ferrule.storage-demand.paired-read-case.v1'
                and case.get('mode') == mode, 'case schema/mode drift')
        require(case.get('argv_after_binary') == argv and case.get('environment') == common_env,
                'source pair/budgets/argv/environment must differ only by mode')
        require(case.get('model') == historical['model'] and case.get('topology') == historical['topology']
                and case.get('read_options') == read and case.get('quality_test_config') == quality,
                'exact paired factory configuration drift')
        require(case.get('limits') == limits and case.get('request_ceiling') == 12
                and case.get('cache_bytes_include_scratch') is True
                and case.get('pagecache_policy') == 'uncontrolled; no cache dropping', 'case policy drift')
    return cases


def validate_design():
    p = ct.load(PROTOCOL_PATH)
    require(p.get('schema') == 'ferrule.storage-demand.paired-read-preregistration.v1'
            and p.get('status') == 'preregistered_cpu_only_candidate_unbound_not_authorized'
            and p.get('candidate_binary') is None, 'unknown/prerecorded candidate state')
    require((p.get('block_count'), p.get('fresh_server_lifecycles'), p.get('scheduled_requests'),
             p.get('scheduled_measured_qa'), p.get('retry_count'), p.get('replacement_count'))
            == (5, 10, 120, 50, 0, 0), 'fixed paired denominators changed')
    require(p.get('order_seed') == 20261015 and ct.load(HERE / 'order.json')
            == dict(seed=20261015, blocks=expected_order()), 'seeded counterbalanced order changed')
    limits = p['limits']
    require(all(integer(x, 1) for x in limits.values()), 'finite integer deadlines required')
    require(limits == dict(startup_seconds=120, request_seconds=60, all_requests_seconds=300,
        worker_bookkeeping_seconds=40, target_lifetime_seconds=460, shutdown_seconds=30,
        worker_preidle_seconds=20, worker_finalization_seconds=20, supervisor_seconds=530,
        supervisor_call_overhead_seconds=28, coordinator_preidle_seconds=20,
        coordinator_postidle_seconds=20, coordinator_finalization_seconds=32,
        outer_lifecycle_seconds=650, pilot_seconds=7200, study_cleanup_reserve_seconds=100),
        'absolute runtime envelope changed')
    require(10 * limits['outer_lifecycle_seconds'] + limits['study_cleanup_reserve_seconds']
            <= limits['pilot_seconds'], 'ten fresh lifecycles and cleanup must fit two hours')
    validate_cases(ct.load(CASE_PATH))
    schedule = ct.load(HERE / 'requests.json')['schedule']
    validate_schedule(schedule)
    require(schedule == ct.load(baseline.HERE / 'requests-v2.json')['schedule'], 'frozen request sequence drift')
    qa = ct.load(HERE / 'oracles.json')['qa']
    require(qa['prompt'] == 'The capital of Japan is' and qa['input_ids'] == [760, 6511, 314, 6124, 369]
            and qa['reference_ids'] == [25358, 13, 198, 760, 6511, 314, 6124, 369]
            and qa['reference_text'] == ' Tokyo.\nThe capital of Japan is', 'exact QA input/text/ID oracle changed')
    source = ct.load(HERE / 'source-cut.json')
    require(source['production_source_digest'] == ct.digest(source['production_source_pins']), 'source seal digest drift')
    require(source['declared_commit'] == p['source_cut'] == '033519c', 'source cut identity drift')
    ct.check_pins(source['production_source_pins'])
    require(source['production_source_pins'] == ct.source_pins(), 'source inventory changed')
    return p


def workflow_pins():
    paths = [HERE / name for name in DESIGN_FILES]
    paths += [ct.ROOT / name for name in baseline.IMPORTS]
    paths += [baseline.CASE_PATH, baseline.HERE / 'requests-v2.json',
              baseline.HERE / 'run_pilot.py', baseline.HERE / 'protocol.py', STORAGE / 'checkpoint-metadata.json',
              ct.ROOT / 'scripts/gpu_idle_window_v2.py', Path(sys.executable).resolve(), Path('/usr/bin/env')]
    paths += list((STORAGE / 'runnerhelpers').glob('*.py'))
    paths += [STORAGE / 'tests' / name for name in ('test_execution_v2.py', 'test_protocol_v2.py', 'test_build_receipts_v1.py')]
    return ct.inventory(paths)


def plan():
    p = validate_design()
    return dict(schema='ferrule.storage-demand.paired-read-cpu-prepared.v1',
        status='cpu_prepared_candidate_unbound_not_authorized', protocol_sha256=ct.sha(PROTOCOL_PATH),
        cases_sha256=ct.sha(CASE_PATH), order=expected_order(), trial_order=trial_order(), limits=p['limits'],
        scheduled_blocks=5, scheduled_starts=10, scheduled_requests=120, scheduled_measured_qa=50,
        artifact_pins=workflow_pins(), source_digest=ct.load(HERE / 'source-cut.json')['production_source_digest'],
        candidate_binary=None, sealed_final_candidate=False, gpu_authorized=False,
        gpu_queries=0, gpu_locks=0, gpu_runs=0, tensor_payload_bytes_read=0,
        quality_pending=['actual same-cut constructor, each-mode full40/fresh/ordinary/cold with producer custody',
                         'actual single CLI cuda,nvtx-profile build/help/runtime source and native dependency bindings',
                         'renewed sole-owner grant, literal ordered eight UUIDs and physical selected UUID'],
        profile_launch_entry=False, interpretation='CPU seal is preregistration, not candidate PASS or GPU authorization')


def unseal(path, schema):
    value = ct.load(path)
    require(value.get('schema') == schema and value.get('digest')
            == ct.digest({k: v for k, v in value.items() if k != 'digest'}), 'unknown/stale seal')
    return value


def validate_prepared(path=PREPARED_PATH):
    value = unseal(path, 'ferrule.storage-demand.paired-read-cpu-prepared.v1')
    require({k: v for k, v in value.items() if k != 'digest'} == plan(), 'CPU workflow/source seal stale')
    ct.check_pins(value['artifact_pins'])
    return value


def validate_read_quality(report, case):
    q = report.get('expert_read_quality', {})
    require(q.get('schema') == 'ferrule.expert-read-quality-evidence.v1'
            and q.get('interval_validation_revision') == 2 and q.get('passed') is True
            and q.get('requested') == case['read_options'], 'disabled/unknown/mismatched read pipeline qualification')
    calls = q.get('calls')
    require(isinstance(calls, list) and calls and q.get('sampled_operations') == len(calls), 'read interval denominator missing')
    for row in calls:
        require(all(row.get(k) is True for k in ('same_owner', 'configuration_matches',
                    'within_charged_limits', 'unambiguous', 'evidence_complete')), 'read quality endpoint/custody failure')
        require(row.get('requested') == case['read_options'], 'per-operation read mode/budget drift')
        for endpoint in ('before', 'after'):
            flags = row.get('configuration_pair', {}).get(endpoint, {})
            require(all(flags.get(k) is True for k in ('configuration_matches', 'within_charged_limits', 'unambiguous')),
                    'read interval endpoint missing/disabled/unknown')
        quality.interval_endpoints(row, case['read_options'])
    positive = sum(row.get('positive_worker_read') is True for row in calls)
    require(positive > 0 and q.get('positive_worker_read_calls') == positive, 'configured mode alone is not positive read evidence')
    effective = report.get('effective_config', {})
    expected = case['quality_test_config']['expert_read_pipeline']
    require(effective.get('assertions_passed') is True and effective.get('requested_expert_read_pipeline') == expected
            and effective.get('effective_expert_read_pipeline') == expected, 'factory/runner read lowering mismatch')


def validate_ordinary(report, diagnostic, diagnostic_hash, case):
    require(report.get('schema') == 'ferrule.bf16-fresh-quality-result.v1'
            and report.get('passed') is True and report.get('complete') is True
            and report.get('test_config') == case['quality_test_config']
            and report.get('effective_config') == diagnostic['effective_config'], 'ordinary equality failed/config changed')
    require(report.get('observer_mode') == 'ordinary_no_diagnostic_observer'
            and report.get('ordinary_no_observer_claim') is True and report.get('cold_owner_restart') is False
            and report.get('diagnostic_primary_report_sha256') == diagnostic_hash
            and report.get('oracle_manifest_sha256') == diagnostic['oracle_manifest_sha256'], 'ordinary primary relation missing')
    require(report.get('failures') == [] and report.get('quarantine') is False
            and type(report.get('state_bytes_after_shutdown')) is int and report['state_bytes_after_shutdown'] == 0
            and report.get('atol') == 0.00025 and report.get('rtol') == 0.012, 'ordinary numerical/closure failure')
    fp, prior = report.get('prediction_fingerprints', {}), diagnostic.get('prediction_fingerprints', {})
    require(fp.get('schema') == 'ferrule.fresh-actual-logit-fingerprints.v1'
            and fp.get('paired_exact_logits_and_greedy_equal') is True and fp.get('failures') == []
            and isinstance(fp.get('calls'), dict) and len(fp['calls']) == 64
            and fp['calls'] == prior.get('calls'), 'ordinary exact logit/greedy equality missing')
    for row in fp['calls'].values():
        require(integer(row.get('rows'), 1) and integer(row.get('values'), 1)
                and integer(row.get('next_token_id')) and isinstance(row.get('f32_le_sha256'), str)
                and len(row['f32_le_sha256']) == 64, 'ordinary fingerprint shape/hash missing')
    require([r.get('case') for r in report.get('cases', [])] == ['qa', 'arithmetic', 'chinese'], 'ordinary case coverage')
    for actual, expected in zip(report['cases'], diagnostic['cases']):
        require(actual.get('oracle_generated_ids') == actual.get('common_history_generated_ids')
                == actual.get('free_greedy_ids') == expected['oracle_generated_ids']
                and actual.get('free_greedy_equal') is True and actual.get('top1_failures') == [], 'ordinary actual token equality')
        token_checks(actual.get('next_token_checks'), expected['oracle_generated_ids'])
        token_checks(actual.get('free_greedy_next_token_checks'), expected['oracle_generated_ids'])
    prediction = report.get('prediction_denominators', {})
    require(prediction.get('schema') == 'ferrule.fresh-prediction-denominators.v1', 'ordinary prediction denominator missing')
    streams = prediction.get('streams', {})
    expected_counts = dict(common_history=27, free_feedback=27, reset_replay=9, fork_replay=1)
    require(set(streams) == set(expected_counts), 'ordinary prediction stream coverage missing')
    for stream, count in expected_counts.items():
        row = streams[stream]
        require(all(row.get(k) == count for k in ('prediction_calls', 'greedy_checks', 'input_history_matched_reference_calls'))
                and integer(row.get('logit_rows'), 1) and integer(row.get('logit_values_compared'), 1)
                and row.get('failures') == [], 'ordinary prediction denominator/failure')
    validate_read_quality(report, case)


def validate_gate_custody(candidate, record, label, facts):
    """Reconcile actual selected producer, source, descendant custody and durable final."""
    def load(key):
        return ct.load(pinned_path(candidate, record.get(key), 'quality/' + label + '/' + key))
    report_path = str(pinned_path(candidate, record.get('report'), label + '/report'))
    executable = str(pinned_path(candidate, record.get('executable'), label + '/executable'))
    custody, launch, manifest, final = (load(k) for k in ('supervisor', 'authenticated_launch', 'source_manifest', 'producer_final'))
    require(ct.successful_process(custody), 'quality producer failed/unknown descendant custody: ' + label)
    sources = manifest.get('original_source_pins', manifest)
    frozen = ct.load(HERE / 'source-cut.json')['production_source_pins']
    require(sources == facts['sources'] and all(sources.get(k) == h for k, h in frozen.items()), 'quality-to-CLI same source cut mismatch')
    identity = launch.get('identity', {})
    require(all(isinstance(identity.get(role), dict) and all(integer(identity[role].get(k), 1)
            for k in ('pid', 'birth_ticks', 'ppid')) for role in ('helper', 'worker')), 'producer PID/birth/parent missing')
    argv = launch.get('argv', [])
    require(argv and argv[0] == executable and launch.get('binary_sha256') == ct.sha(executable)
            and launch.get('exec_preserves_authenticated_root_pid_birth') is True
            and launch.get('all_eight_uuid_locks_validated') is True
            and identity.get('launch_id') == custody.get('handshake', {}).get('launch_id')
            and identity.get('helper') == custody.get('handshake', {}).get('helper_identity')
            and identity.get('worker') == custody.get('custody', {}).get('root_identity'), 'actual quality launch/custody relation changed')
    if label in ('full40', 'fresh', 'cold'):
        validate_producer_final(final, label, report_path, ct.sha(report_path), executable,
                                ct.sha(executable), facts['source_digest'], argv)
    else:
        require(final.get('schema') == 'ferrule.storage-demand.paired-read-quality-producer.v1'
                and final.get('wrapper_accepted') is True and final.get('label') == label
                and final.get('report') == report_path and final.get('report_sha256') == ct.sha(report_path)
                and final.get('executable') == executable and final.get('executable_sha256') == ct.sha(executable)
                and final.get('production_source_digest') == facts['source_digest']
                and final.get('argv') == argv and final.get('supervisor_sha256') == ct.sha(record['supervisor']),
                'constructor/ordinary durable final producer binding missing')
    return ct.load(report_path)


def prepare_candidate(candidate):
    validate_design()
    require(candidate.get('schema') == 'ferrule.storage-demand.paired-read-candidate-input.v1'
            and candidate.get('status') == 'quality_passed_cpu_bind_requested'
            and candidate.get('protocol_sha256') == ct.sha(PROTOCOL_PATH)
            and candidate.get('cases_sha256') == ct.sha(CASE_PATH), 'unknown/pending/stale candidate input')
    ct.check_pins(candidate.get('artifact_pins'))
    build = ct.load(pinned_path(candidate, candidate.get('build_receipt'), 'build_receipt'))
    facts = build_adapter.review_cli_build(candidate, build, case_for_mode('blocking'))
    frozen = ct.load(HERE / 'source-cut.json')['production_source_pins']
    require(all(facts['sources'].get(p) == h for p, h in frozen.items()), 'CLI source cut changed')
    tokenizer = candidate.get('tokenizer_files', {})
    expected_tokenizer = ct.load(HERE / 'oracles.json')['tokenizer']['files']
    require(set(tokenizer) == set(expected_tokenizer), 'pinned actual tokenizer copies required')
    for name, expected_hash in expected_tokenizer.items():
        path = pinned_path(candidate, tokenizer[name], 'tokenizer/' + name)
        require(ct.sha(path) == expected_hash, 'actual tokenizer oracle mismatch')
    runtime = ct.load(pinned_path(candidate, candidate.get('runtime_pin_manifest'), 'runtime_pin_manifest'))
    ct.check_pins(runtime.get('pins'))
    require(runtime.get('unknown_nccl_enabled') is False
            and all(runtime['pins'].get(p) == h for p, h in facts['runtime_pins'].items()), 'runtime/native library binding changed')
    constructor = validate_gate_custody(candidate, candidate.get('constructor', {}), 'constructor', facts)
    quality.constructor(constructor)
    require(set(candidate.get('quality', {})) == set(MODES), 'both mode quality gates required')
    for mode in MODES:
        case = case_for_mode(mode)
        records = candidate['quality'][mode]
        require(set(records) == set(LABELS), 'same-cut full40/fresh/ordinary/cold coverage required')
        reports = {label: validate_gate_custody(candidate, records[label], label, facts) for label in LABELS}
        validate_quality_reports({label: reports[label] for label in ('full40', 'fresh', 'cold')}, case)
        for label in ('full40', 'fresh', 'cold'):
            validate_read_quality(reports[label], case)
            ref = pinned_path(candidate, records[label].get('reference_manifest'), label + '/reference')
            key = 'reference_manifest_sha256' if label == 'full40' else 'oracle_manifest_sha256'
            require(ct.sha(ref) == reports[label][key], 'independent numerical reference pin changed')
        reference_pins = ct.load(HERE / 'oracles.json')['reference_pins']
        frozen_oracle_hash = next(h for path, h in reference_pins.items() if path.endswith('/oracle/manifest.json'))
        require(reports['fresh']['oracle_manifest_sha256'] == reports['cold']['oracle_manifest_sha256'] == frozen_oracle_hash, 'frozen independent reference changed')
        validate_ordinary(reports['ordinary'], reports['fresh'], ct.sha(records['fresh']['report']), case)
        refs = {r['id']: r for r in ct.load(HERE / 'oracles.json')['references']}
        for row in reports['fresh']['cases']:
            require(row['oracle_generated_ids'][:8] == refs[row['case']]['reference_ids'], 'serving generation8 oracle differs')
    return dict(schema='ferrule.storage-demand.paired-read-final-candidate.v1',
        status='quality_passed_cpu_prepared_not_authorized', sealed_final_candidate=True,
        candidate=candidate, protocol_sha256=ct.sha(PROTOCOL_PATH), cases_sha256=ct.sha(CASE_PATH),
        build_adapter=facts['adapter'], build_evidence_coverage=facts['coverage'],
        gpu_authorized=False, gpu_queries=0, gpu_locks=0, gpu_runs=0)


def validate_candidate(path):
    value = unseal(path, 'ferrule.storage-demand.paired-read-final-candidate.v1')
    require(value.get('sealed_final_candidate') is True
            and value.get('status') == 'quality_passed_cpu_prepared_not_authorized', 'candidate pending/unbound')
    require({k: v for k, v in value.items() if k != 'digest'} == prepare_candidate(value.get('candidate', {})),
            'candidate raw build/quality/source receipts changed')
    return value


def seal_to(path, value):
    require(path.parent.resolve() == HERE and not path.exists(), 'new exclusive output in paired_read_v1 required')
    ct.write(path, dict(value, digest=ct.digest(value)))


def main(argv=None):
    ct.require_cpu_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'seal-cpu', 'check', 'bind-candidate'))
    parser.add_argument('--prepared', type=Path, default=PREPARED_PATH)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'plan':
        value = plan()
    elif args.mode == 'seal-cpu':
        require(args.output is not None, 'exclusive output required')
        value = plan()
        seal_to(args.output, value)
    elif args.mode == 'check':
        value = validate_prepared(args.prepared)
        if args.candidate:
            validate_candidate(args.candidate)
    else:
        require(args.candidate is not None and args.output is not None, 'real owner-supplied candidate input/output required')
        value = prepare_candidate(ct.load(args.candidate))
        seal_to(args.output, value)
    print(json.dumps(dict(status=value['status'], gpu_queries=0, gpu_locks=0, gpu_runs=0), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
