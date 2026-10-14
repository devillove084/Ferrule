#!/usr/bin/env python3
"""Prospective corrective V3 CPU plan. No bind, run, GPU, HTTP, build or help entry."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from runnerhelpers.evidence_v2 import ct, require, validate_schedule
from paired_read_v1 import protocol as historical

ID = 'storage-middle64-paired-read-corrective-20261015-v3'
PROTOCOL_PATH = HERE / 'protocol-v3.json'
CASES_PATH = HERE / 'cases-v3.json'
PREPARED_PATH = HERE / 'cpu-sealed-v3.json'
REJECTION_PATH = HERE / 'v2-configuration-rejection-v3.json'
FILES = ('plan_v3.py', 'test_plan_v3.py', 'PLAN_V3.md', 'protocol-v3.json', 'cases-v3.json',
         'v2-configuration-rejection-v3.json')
MODES = ('blocking', 'lookahead')
CLI_COHORT_DEFAULTS = dict(decode_cohort_target=4, decode_cohort_max_deferrals=3)
SCHEDULER = dict(prefill_chunk_size=8, max_active_sequences=2, max_decode_batch=2,
    decode_cohort_target=1, decode_cohort_max_deferrals=0, max_batch_tokens=8,
    prefill_reserve_tokens=1, allow_mixed_batches=True, prefix_cache_capacity_pages=0)
COHORT_FLAGS = ['--decode-cohort-target', '1', '--decode-cohort-max-deferrals', '0']


class ConfigurationMismatch(ValueError):
    def __init__(self, differences):
        self.differences = differences
        super().__init__('CLI/qualified configuration mismatch: ' + ', '.join(sorted(differences)))


def flags(argv):
    require(isinstance(argv, list) and len(argv) >= 2 and argv[0] == 'serve', 'literal serve argv required')
    require((len(argv) - 2) % 2 == 0, 'flag/value pairs required')
    result = {}
    for name, value in zip(argv[2::2], argv[3::2]):
        require(isinstance(name, str) and name.startswith('--') and name not in result
                and isinstance(value, str), 'unknown/duplicate/missing CLI flag/value')
        result[name] = value
    return result


def cli_scheduler(argv):
    """Model the reviewed CLI lowering only; this is not a live factory receipt."""
    f = flags(argv)
    for name in ('--prefill-chunk-size', '--max-active-sequences', '--max-batch-tokens'):
        require(name in f, 'explicit bounded scheduler flag required: ' + name)
    def number(name, default=None):
        text = f.get(name, None if default is None else str(default))
        require(isinstance(text, str) and text.isdecimal(), 'noninteger scheduler flag: ' + name)
        return int(text)
    active = number('--max-active-sequences')
    # The remaining three fields are the maintained ResidentSchedulerConfig defaults.
    return dict(prefill_chunk_size=number('--prefill-chunk-size'), max_active_sequences=active,
        max_decode_batch=active, decode_cohort_target=number('--decode-cohort-target', 4),
        decode_cohort_max_deferrals=number('--decode-cohort-max-deferrals', 3),
        max_batch_tokens=number('--max-batch-tokens'), prefill_reserve_tokens=1,
        allow_mixed_batches=True, prefix_cache_capacity_pages=0)


def compare_scheduler(actual, expected=SCHEDULER, label='scheduler'):
    require(isinstance(actual, dict), label + ' missing')
    differences = {}
    for key in set(actual) | set(expected):
        a, e = actual.get(key), expected.get(key)
        if key not in actual or key not in expected or type(a) is not type(e) or a != e:
            differences[label + '.' + key] = dict(actual=a, expected=e)
    if differences:
        raise ConfigurationMismatch(differences)
    return dict(configuration_equal=True, fields_compared=sorted(expected), candidate_qualified=False)


def expected_case(mode):
    require(mode in MODES, 'disabled/unknown read mode')
    case = copy.deepcopy(ct.load(HERE / 'cases.json')[mode])
    require(all(flag not in flags(case['argv_after_binary']) for flag in COHORT_FLAGS[::2]),
            'historical argv unexpectedly changed')
    case.update(schema='ferrule.storage-demand.paired-read-case.v3', protocol_id=ID,
                case_id=ID + '-' + mode, candidate_binary=None, candidate_source_cut=None,
                actual_lifecycles=0, expected_scheduler=copy.deepcopy(SCHEDULER),
                configuration_scope='Declared CLI lowering, not a factory/build/GPU admission receipt')
    case['argv_after_binary'] += COHORT_FLAGS
    return case


def validate_cases(cases):
    require(isinstance(cases, dict) and set(cases) == set(MODES), 'exactly both enabled arms required')
    for mode in MODES:
        case = cases[mode]
        f = flags(case['argv_after_binary'])
        require(f.get('--decode-cohort-target') == '1' and f.get('--decode-cohort-max-deferrals') == '0',
                'V3 requires explicit cohort 1/0, not public CLI defaults 4/3')
        compare_scheduler(cli_scheduler(case['argv_after_binary']), label=mode + '.cli')
        compare_scheduler(case.get('expected_scheduler'), label=mode + '.declared')
        compare_scheduler(case.get('quality_test_config', {}).get('capacity_and_options', {}).get('scheduler'),
                          label=mode + '.quality_test_config')
        require(case == expected_case(mode), 'V3 case/config/budgets drift; correction is cohort flags only')
    b, l = (cases[mode]['argv_after_binary'] for mode in MODES)
    differences = [(i, x, y) for i, (x, y) in enumerate(zip(b, l)) if x != y]
    require(len(b) == len(l) and differences == [(b.index('--expert-read-schedule') + 1, 'blocking', 'lookahead')],
            'paired CLI must differ only by read-mode value')
    return cases


def compare_report_configuration(case, report):
    """Check all nine scheduler fields at every layer, without conferring quality PASS."""
    compare_scheduler(cli_scheduler(case['argv_after_binary']), label='cli')
    config = report.get('test_config', {})
    compare_scheduler(config.get('capacity_and_options', {}).get('scheduler'), label='test_config')
    require(config == case['quality_test_config'], 'complete qualified test_config differs')
    effective = report.get('effective_config', {})
    require(effective.get('assertions_passed') is True, 'factory assertions missing')
    for endpoint in ('requested', 'effective'):
        compare_scheduler(effective.get(endpoint, {}).get('scheduler'), label='factory.' + endpoint)
    read = case['quality_test_config']['expert_read_pipeline']
    require(effective.get('requested_expert_read_pipeline') == read
            and effective.get('effective_expert_read_pipeline') == read, 'factory read mode/64 MiB budget differs')
    return dict(scope='CPU configuration comparison only', configuration_equal=True,
        scheduler_fields_compared=sorted(SCHEDULER), candidate_qualified=False,
        v3_quality_pass=False, source_cut_qualified=False, gpu_authorized=False)


def require_v3_identity(receipt):
    """Namespace guard for a future reviewed binder, not an implementation of that binder."""
    require(receipt.get('schema') == 'ferrule.storage-demand.paired-read-candidate-input.v3'
            and receipt.get('protocol_id') == ID and receipt.get('protocol_sha256') == ct.sha(PROTOCOL_PATH)
            and receipt.get('cases_sha256') == ct.sha(CASES_PATH),
            'wrong protocol/case identity; old V2 PASS cannot qualify V3')
    return dict(identity_matches=True, candidate_qualified=False,
                reason='Fresh immutable fixed-source CLI/build/help/native/runtime and nine GPU gates remain mandatory')


def validate_design():
    p = ct.load(PROTOCOL_PATH)
    require(p.get('schema') == 'ferrule.storage-demand.paired-read-preregistration.v3' and p.get('id') == ID
            and p.get('status') == 'prospective_cpu_only_candidate_unbound_not_authorized'
            and p.get('candidate_binary') is None and p.get('candidate_source_cut') is None
            and p.get('actual_lifecycles') == p.get('gpu_queries') == p.get('gpu_runs') == 0,
            'unknown/bound/non-prospective V3 state')
    parent = ct.load(HERE / 'protocol.json')
    inherited = ('block_count', 'fresh_server_lifecycles', 'order_seed', 'order_algorithm', 'modes',
        'option_difference', 'request_phases', 'requests_per_lifecycle', 'scheduled_requests',
        'scheduled_measured_qa', 'generation_tokens', 'retry_count', 'replacement_count', 'limits',
        'primary_estimand', 'goodput', 'interval', 'failure_policy', 'quality_tolerances', 'custody', 'claim_limits')
    require(all(p.get(k) == parent[k] for k in inherited), 'original n=5 design/order/budget/estimand drift')
    require(p.get('order') == historical.expected_order()
            and p.get('corrective_flags') == COHORT_FLAGS and p.get('expected_scheduler') == SCHEDULER
            and p.get('public_cli_defaults') == CLI_COHORT_DEFAULTS
            and p.get('profiling_captures') == 0 and p.get('launch_entry') is False,
            'corrective flags/order/default scope drift')
    preservation = p['historical_preservation']
    require(ct.digest(preservation['artifact_pins']) == preservation['inventory_digest'], 'preservation inventory digest changed')
    ct.check_pins(preservation['artifact_pins'])
    require(p['cases_sha256'] == ct.sha(CASES_PATH), 'V3 case binding changed')
    validate_cases(ct.load(CASES_PATH))
    for name in ('requests.json', 'oracles.json', 'order.json'):
        require(p['reference_bindings'][name] == ct.sha(HERE / name), 'original request/oracle/order binding changed')
    validate_schedule(ct.load(HERE / 'requests.json')['schedule'])
    prior = ct.load(HERE / 'artifacts-current-20261015-01/paired-final.json')
    require(prior['status'] == 'stopped' and prior['actual_issued_lifecycles'] == 1
            and prior['actual_completed_lifecycles'] == 0 and prior['analysis']['denominators']['unissued'] == 120
            and prior['analysis']['measured_qa']['scheduled'] == 50, 'historical stopped denominators changed')
    require(p['historical_study']['replacement_or_resume'] is False
            and p['historical_study']['timing_outcomes_observed'] is False, 'not a corrective independent study')
    return p


def plan():
    p = validate_design()
    return dict(schema='ferrule.storage-demand.paired-read-cpu-plan.v3', protocol_id=ID,
        status=p['status'], protocol_sha256=ct.sha(PROTOCOL_PATH), cases_sha256=ct.sha(CASES_PATH),
        order=p['order'], trial_order=historical.trial_order(), limits=p['limits'],
        scheduled_blocks=5, scheduled_starts=10, scheduled_requests=120, scheduled_measured_qa=50,
        actual_lifecycles=0, candidate_binary=None, candidate_source_cut=None, sealed_final_candidate=False,
        gpu_authorized=False, gpu_queries=0, gpu_locks=0, gpu_runs=0, server_runs=0, nas_queries=0,
        launch_entry=False, bind_entry=False, profiling_captures=0,
        arms={mode: dict(argv_after_binary=case['argv_after_binary'], environment=case['environment'],
                        scheduler=case['expected_scheduler'], read_options=case['read_options'])
              for mode, case in ct.load(CASES_PATH).items()},
        prerequisites=p['candidate_prerequisites'], historical_study=p['historical_study'],
        artifact_pins=ct.inventory([HERE / name for name in FILES]))


def validate_prepared(path=PREPARED_PATH):
    value = historical.unseal(path, 'ferrule.storage-demand.paired-read-cpu-plan.v3')
    require({k: v for k, v in value.items() if k != 'digest'} == plan(), 'prospective V3 CPU seal stale')
    return value


def audit_v2():
    from paired_read_v1 import current_binding_v2 as v2
    v2.validate_binding()
    # Integrity remains valid; semantic compatibility is a separate append-only finding.
    for path, schema in ((historical.PREPARED_PATH, 'ferrule.storage-demand.paired-read-cpu-prepared.v1'),
                         (v2.EXECUTION_PATH, 'ferrule.storage-demand.paired-read-reviewed-execution.v2')):
        value = historical.unseal(path, schema)
        ct.check_pins(value['artifact_pins'])
    errors = {}
    reports = {}
    for mode, number in (('blocking', '04'), ('lookahead', '05')):
        path = v2.QUALIFICATION / 'gates' / (number + '-' + mode + '-fresh') / 'quality.json'
        reports[mode] = dict(path=str(path), sha256=ct.sha(path))
        try:
            compare_report_configuration(ct.load(HERE / 'cases.json')[mode], ct.load(path))
        except ConfigurationMismatch as error:
            errors[mode] = error.differences
    require(set(errors) == set(MODES), 'expected V2 cohort mismatch absent; do not silently approve')
    return dict(schema='ferrule.storage-demand.v2-configuration-rejection.v3',
        status='expected_semantic_rejection_historical_integrity_preserved',
        v2_binding_digest_valid=True, v2_execution_digest_and_workflow_pins_valid=True,
        v2_cases_sha256=ct.sha(HERE / 'cases.json'), v2_execution_sha256=ct.sha(v2.EXECUTION_PATH),
        v2_binding_sha256=ct.sha(v2.BINDING_PATH), reports=reports, differences=errors,
        public_cli_defaults=CLI_COHORT_DEFAULTS, qualified_scheduler=SCHEDULER,
        v3_candidate_qualified=False, v2_failure_reclassified=False, old_files_changed=False,
        gpu_queries=0, gpu_runs=0, server_runs=0,
        interpretation='V2 immutable receipts remain valid history; their implicit CLI cohort4/3 does not match qualified1/0 and cannot pass V3')


def main(argv=None):
    ct.require_cpu_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'seal-cpu', 'check', 'audit-v2-config', 'compare-report'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--arm', choices=MODES)
    args = parser.parse_args(argv)
    if args.mode == 'plan':
        value = plan()
    elif args.mode == 'seal-cpu':
        require(args.output is not None and args.output.resolve() == PREPARED_PATH, 'exclusive cpu-sealed-v3.json required')
        historical.seal_to(args.output, plan())
        value = validate_prepared()
    elif args.mode == 'check':
        value = validate_prepared()
    elif args.mode == 'audit-v2-config':
        value = audit_v2()
        if args.output:
            require(args.output.resolve() == REJECTION_PATH, 'new append-only V3 rejection receipt required')
            ct.write(args.output, value)
    else:
        require(args.report is not None and args.arm is not None, 'local report and arm required; no producer launched')
        validate_design()
        value = compare_report_configuration(ct.load(CASES_PATH)[args.arm], ct.load(args.report))
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
