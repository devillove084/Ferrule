"""Validate actual full40/fresh Rust report fields without inventing custody."""
import re

from .evidence_v2 import integer, require

DENOMINATORS = dict(full40=(804, 1986560), fresh=(5427, 9187840), cold=(1809, 3228160))


def token_checks(rows, expected=None, count=None):
    require(isinstance(rows, list) and len(rows) == (len(expected) if expected is not None else count),
            'actual next-token denominator missing')
    for index, row in enumerate(rows):
        require(row.get('equal') is True and row.get('actual') is not None
                and row.get('actual') == row.get('expected'), 'actual next-token equality failed/unknown')
        if expected is not None:
            require(row.get('step') == index and row['actual'] == expected[index], 'actual token history/index mismatch')


def validate_quality_reports(reports, case):
    require(set(reports) == set(DENOMINATORS), 'all three current quality gates required')
    for label, report in reports.items():
        schema = ('ferrule.numeric-bf16-full40-profile-gate.v1' if label == 'full40'
                  else 'ferrule.bf16-fresh-quality-result.v1')
        require(report.get('schema') == schema and report.get('passed') is True
                and report.get('complete') is True, 'quality failed/pending/unknown: ' + label)
        require(report.get('test_config') == case['quality_test_config'], 'same StorageMiddle64 config missing: ' + label)
        require(report.get('effective_config', {}).get('assertions_passed') is True, 'effective factory assertions missing: ' + label)
        require(report.get('failures') == [] and report.get('quarantine') is False
                and type(report.get('state_bytes_after_shutdown')) is int and report['state_bytes_after_shutdown'] == 0,
                'quality failures/quarantine/shutdown missing: ' + label)
        require(report.get('ordinary_no_observer_claim') is False and report.get('concurrent_scheduler_claim') is False
                and report.get('observer_mode') == 'diagnostic_routes_and_states'
                and report.get('greedy_tie_policy') == 'lower_token_id', 'unknown quality claim scope: ' + label)
        require(report.get('atol') == 0.00025 and report.get('rtol') == 0.012, 'quality tolerance changed: ' + label)
        metrics, logits = DENOMINATORS[label]
        rows = report.get('metrics')
        require(isinstance(rows, list) and len(rows) == metrics and report.get('logits_compared') == logits
                and all(type(r.get('failures')) is int and r['failures'] == 0
                        and type(r.get('nonfinite', 0)) is int and r.get('nonfinite', 0) == 0 for r in rows),
                'numerical denominator/failures/nonfinite missing: ' + label)
        peak = report.get('actual_peak_reserved_bytes' if label == 'full40' else 'peak_reserved_bytes')
        require(integer(peak) and integer(report.get('device_budget_bytes'), 1)
                and peak <= report['device_budget_bytes'], 'quality device budget missing/exceeded')
        if label == 'full40':
            require(report.get('diagnostic_capture_only') is False and report.get('status') == 'passed', 'partial full40 is not PASS')
            token_checks(report.get('next_token_checks'), count=4)
            continue
        require(report.get('cold_owner_restart') is (label == 'cold'), 'fresh/cold owner order mismatch')
        cases = report.get('cases')
        names = ['qa'] if label == 'cold' else ['qa', 'arithmetic', 'chinese']
        require(isinstance(cases, list) and [r.get('case') for r in cases] == names, 'fresh/cold case coverage')
        for row in cases:
            expected = row.get('oracle_generated_ids')
            require(isinstance(expected, list) and len(expected) == 9 and all(integer(i) for i in expected)
                    and row.get('common_history_generated_ids') == row.get('free_greedy_ids') == expected
                    and row.get('free_greedy_equal') is True and row.get('top1_failures') == []
                    and row.get('free_greedy_status') == 'completed_independent_feedback', 'actual fresh generated IDs failed/unknown')
            for key in ('next_token_checks', 'free_greedy_next_token_checks'):
                token_checks(row.get(key), expected)
            if row['case'] == 'qa':
                require(row.get('reset_reused_state') is True and row.get('reset_equal') is True
                        and row.get('fork_snapshot_equal') is True and row.get('reset_generated_ids') == expected,
                        'QA reset/fork preservation missing')
                token_checks(row.get('reset_next_token_checks'), expected)
    require(reports['fresh'].get('oracle_manifest_sha256') == reports['cold'].get('oracle_manifest_sha256')
            and re.fullmatch(r'[0-9a-f]{64}', reports['fresh'].get('oracle_manifest_sha256') or ''), 'fresh/cold reference mismatch')
    require(reports['cold']['cases'][0]['oracle_generated_ids'] == reports['fresh']['cases'][0]['oracle_generated_ids'],
            'cold primary QA history mismatch')
    require(re.fullmatch(r'[0-9a-f]{64}', reports['full40'].get('reference_manifest_sha256') or ''), 'full40 reference pin missing')
    return dict(status='pass_same_StorageMiddle64', performance_acceptance=False)


PRODUCER_SCHEMAS = dict(
    full40='ferrule.storage-middle64.current-full40-one-shot.v2',
    fresh='ferrule.storage-middle64.current-retained-fresh-one-shot.v3',
    cold='ferrule.storage-middle64.current-cold-owner-one-shot.v4')


def validate_producer_final(final, label, report_path, report_sha256, executable, executable_sha256, source_digest, argv):
    """Successful libtest/subreaper custody does not repair a rejected wrapper."""
    require(final.get('schema') == PRODUCER_SCHEMAS[label] and final.get('wrapper_accepted') is True,
            'quality producer final wrapper failed/unknown: ' + label)
    process = final.get('single_test' if label == 'full40' else 'process', {})
    require(type(process.get('returncode')) is int and process['returncode'] == 0
            and process.get('timed_out') is False
            and process.get('invocations_this_authorization' if label == 'full40' else 'invocations') == 1
            and len(argv) >= 2 and argv[1] == process.get('selector'), 'quality producer selected process relation missing: ' + label)
    if label == 'full40':
        require(process.get('report') == report_path and process.get('report_sha256') == report_sha256,
                'quality producer final report relation missing: ' + label)
        binary, digest = final.get('binary', {}), final.get('source', {}).get('pins_digest')
        path = binary.get('path')
    else:
        report = final.get('quality_report', {})
        require(report.get('path') == report_path and report.get('sha256') == report_sha256
                and report.get('complete') is True and report.get('passed') is True,
                'quality producer final report relation missing: ' + label)
        binary = final.get('artifact' if label == 'fresh' else 'binary', {})
        digest = binary.get('source_pins_digest') if label == 'fresh' else final.get('source', {}).get('source_pins_digest')
        path = binary.get('binary' if label == 'fresh' else 'path')
    require(path == executable and binary.get('sha256') == executable_sha256 and digest == source_digest,
            'quality producer final binary/source relation missing: ' + label)
