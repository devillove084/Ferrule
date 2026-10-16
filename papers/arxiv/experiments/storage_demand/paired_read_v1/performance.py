"""Same-cut fixed-pair contracts; no process, GPU, clock-tuning or retry engine."""
from decimal import Decimal
import re

from runnerhelpers.evidence_v2 import ct, integer, require
import quality
import retained_evidence

COMMIT = '57e46295d3cf5725605e076eb5543f6124bd9245'
BINARY_SHA = '2c15bbde1aa377630045068439a68adf260c3c836a8de69b120784deb0b067bd'
SOURCE_SHA = '81a399a5cb004e9dd2a22a9303fc4c609cebcf0856654ec29866f50fa2f9e441'
VERSION = '2025.3.1.90'
PREFIX = 'model_factory::dense_bf16_tests::full40::fresh_quality::mixed_storage_middle64_'
ORDER = (('synchronous', 'retained'), ('retained', 'synchronous'),
         ('retained', 'synchronous'), ('synchronous', 'retained'))
READ_WORK = ('submitted_reads', 'completed_reads', 'verified_reads', 'io_started_reads',
             'proved_read_bytes', 'reader_interval_calls', 'reader_interval_bytes',
             'demanded_reads', 'speculative_reads', 'prefetched_reads_taken', 'resident_skips')
CACHE_WORK = ('uploads', 'evictions', 'hits')


def fixed_jobs():
    rows = []
    def add(phase, schedule, policy, repetition=None):
        name = f'{len(rows)+1:02d}-{phase}-{schedule}-{policy}'
        if repetition is not None:
            name += f'-r{repetition}'
        suffix = 'retained_generation' if phase == 'primary' else 'no_observer_equality'
        selector = PREFIX + ('retained_' if policy == 'retained' else '') + schedule + '_fresh_quality_' + suffix
        rows.append(dict(id=name, phase=phase, schedule=schedule, transfer_mode=policy,
                         kind='fresh' if phase == 'primary' else 'ordinary', role='composition',
                         executable=str(ct.ROOT / 'target/validation/retained-upload-57e4629-gpu-20261016-01/staged/composition'),
                         binary_sha256=BINARY_SHA,
                         repetition=repetition, selector=selector, seconds=1200))
    for schedule, modes in (('blocking', ORDER[0]), ('lookahead', ORDER[1])):
        for policy in modes:
            add('primary', schedule, policy)
    for repetition, modes in enumerate(ORDER, 1):
        for schedule in (('blocking', 'lookahead') if repetition % 2 else ('lookahead', 'blocking')):
            for policy in modes:
                add('latency', schedule, policy, repetition)
    for schedule, modes in (('blocking', ORDER[0]), ('lookahead', ORDER[1])):
        for policy in modes:
            add('profile', schedule, policy)
    return rows


def profile_jobs():
    return fixed_jobs()[20:]


def profile_only(plan):
    return plan.get('schema') == 'ferrule.retained-upload.profile-only-plan.v1'


def validate_plan(plan):
    require(plan['schema'] in ('ferrule.retained-upload.performance-plan.v1', 'ferrule.retained-upload.profile-only-plan.v1')
            and plan['commit'] == COMMIT and plan['binary_sha256'] == BINARY_SHA
            and plan['source_archive_sha256'] == SOURCE_SHA, 'Wrong immutable same-cut source/binary')
    require(ct.digest(plan['jobs']) == ct.digest(profile_jobs() if profile_only(plan) else fixed_jobs()),
            'Fixed jobs, selectors, order or deadlines changed')
    require(plan['retries'] == plan['extra_warmups'] == plan['calibration'] == 0
            and plan['stop_on_first_failure'] is True, 'No retries/warmups/calibration permitted')
    require(plan['atol'] == .00025 and plan['rtol'] == .012
            and plan['host_options'] == dict(max_host_bytes=64 << 20,
                max_metadata_bytes=8 << 20, worker_stack_bytes=2 << 20), 'Tolerance/host cap changed')
    require(plan['nsys_version'] == VERSION and plan['nsys'].endswith(
        '/nsight-systems-cli/2025.3.1/target-linux-x64/nsys'), 'Official pinned Nsight path/version required')
    require(len(plan['uuid_inventory']) == len(set(plan['uuid_inventory'])) == 8
            and plan['uuid_inventory'][0] == 'GPU-b980f22b-58a3-fc88-21c1-ae86053d8535',
            'All-eight exact UUID inventory required')
    require(plan['measurement'] == ('official_nsight_process_wide_counts_and_host_durations_not_unprofiled_latency'
            if profile_only(plan) else 'whole_test_libtest_footer_only_not_token_TTFT_or_request_latency'),
            'No per-token/per-request latency claim on immutable57')
    if profile_only(plan):
        parent = ct.ROOT / 'target/validation/retained-performance-57e4629-20261016-01'
        require(plan['completed_performance_source'] == dict(directory=str(parent),
            plan_sha256='875957578ca9479e553d5017b04bff4c8b3dd44bff270c86211c6b129109b991',
            final_sha256='9e080fa6149832d2d2b3c6962048a2ecc9dc4364ac5cacdd520a887d393b52e2'),
            'Profile-only source must bind consumed same-cut20; never resume/retry it')
        require(plan['pins'].get(str(parent / 'performance-plan.json')) == plan['completed_performance_source']['plan_sha256']
            and plan['pins'].get(str(parent / 'FINAL.json')) == plan['completed_performance_source']['final_sha256'],
            'Consumed source plan/final must remain content-pinned')
        bindings = plan['diagnostic_primaries']
        require(isinstance(bindings, list) and len(bindings) == 4
            and ct.digest([b['job'] for b in bindings]) == ct.digest(fixed_jobs()[:4]),
            'Exact four completed policy/schedule diagnostic primaries required')
        for binding in bindings:
            directory = parent / 'gates' / binding['job']['id']
            for field, filename in (('report', 'quality.json'), ('gate_result', 'gate-result.json')):
                require(binding[field] == str(directory / filename)
                    and re.fullmatch(r'[0-9a-f]{64}', binding[field + '_sha256'] or '')
                    and plan['pins'].get(binding[field]) == binding[field + '_sha256'],
                    'Completed primary path/hash/policy crossbinding differs')
    return plan


def job(plan, value):
    wanted = next((row for row in plan['jobs'] if row['id'] == value.get('id')), None)
    require(wanted is not None and ct.digest(value) == ct.digest(wanted), 'Job selector/policy/scope changed')
    return wanted


def policy(report, mode, schedule, capacity):
    conf, effective = report['test_config'], report['effective_config']
    retained = mode == 'retained'
    require(mode in ('synchronous', 'retained') and schedule in ('blocking', 'lookahead'), 'Unknown policy/schedule')
    require(conf['name'] == 'storage-middle64-' + ('retained-' if retained else '') + schedule
            and conf['numeric_precision'] == 'Bf16ModelMixedF32Accumulate'
            and conf['capacity_and_options'] == capacity
            and conf['expert_read_pipeline'] == dict(schedule=schedule.capitalize(), max_host_bytes=64 << 20,
                max_metadata_bytes=8 << 20, worker_stack_bytes=2 << 20,
                metadata_and_stack_included_in_host_cap=True, device_budget_separate=True), 'Policy/config/schedule differs')
    if retained:
        retained_evidence.validate(report, schedule)
    else:
        require(all('expert_transfer_policy' not in item and 'expert_transfer_selection' not in item
                    for item in (conf, effective)), 'Default synchronous receipt contains retained policy')
        quality.read_quality_v3(report, schedule)
        retained_evidence.validated_cache_pairs(report)
        for row in report['expert_read_quality']['calls']:
            for endpoint in ('before', 'after'):
                charge = row[endpoint]['expert_read_pipeline']['value']['current_charge']
                require(all(type(charge.get(k, 0)) is int and charge.get(k, 0) == 0
                            for k in retained_evidence.UPLOAD_FIELDS), 'Synchronous upload charge/Unknown')


def primary_key(row):
    return row['transfer_mode'], row['schedule']


def validate_primary(accepted, report, row, plan, digest):
    require(accepted['status'] == 'passed' and accepted['report_sha256'] == digest
            and accepted['commit'] == plan['commit'] and accepted['binary_sha256'] == plan['binary_sha256']
                        and accepted['source_digest'] == plan['source_digest']
            and accepted['job'] in ([b['job'] for b in plan['diagnostic_primaries']] if profile_only(plan) else plan['jobs'])
            and accepted['job']['phase'] == 'primary'
            and primary_key(accepted['job']) == primary_key(row), 'Cross-policy/schedule/source or failed primary')
    require(report['complete'] is True and report['passed'] is True
            and report['cold_owner_restart'] is False
            and report['observer_mode'] == 'diagnostic_routes_and_states'
            and report['oracle_manifest_sha256'] == plan['fresh_reference_sha256']
            and len(report['prediction_fingerprints']['calls']) == 64, 'Incomplete diagnostic primary')
    policy(report, row['transfer_mode'], row['schedule'], plan['capacity_and_options'])


def work_vector(report):
    rows = []
    pairs = retained_evidence.validated_cache_pairs(report)
    for call, (before, after) in zip(report['expert_read_quality']['calls'], pairs):
        if call['operation'] != 'prediction':
            continue
        reads = {key: call['delta'][key] for key in READ_WORK}
        require(all(integer(value) for value in reads.values()), 'Work counter missing/negative')
        caches = {key: after[key] - before[key] for key in CACHE_WORK}
        require(all(integer(value) for value in caches.values()), 'Cache work counter reset')
        rows.append(dict(label=call['label'], operation=call['operation'], reads=reads, cache=caches))
    require(len(rows) == 64 and len({row['label'] for row in rows}) == 64, 'Exact64 ordered prediction work rows required')
    return rows


def check_vector(report):
    keys = ('prediction_calls', 'greedy_checks', 'input_history_matched_reference_calls',
            'logit_values_compared', 'failures')
    return {name: {key: stream[key] for key in keys}
            for name, stream in report['prediction_denominators']['streams'].items()}


def compare_work(left, right):
    a, b = work_vector(left), work_vector(right)
    require(a == b, 'Ordered work/read/upload/check arrays differ; do not compare timings')
    checks_a, checks_b = check_vector(left), check_vector(right)
    require(checks_a == checks_b, 'Prediction check arrays differ')
    return dict(equal=True, left=a, right=b, prediction_checks_left=checks_a,
                prediction_checks_right=checks_b,
                fingerprints_equal=left['prediction_fingerprints']['calls'] == right['prediction_fingerprints']['calls'])


def libtest_elapsed(text):
    matches = re.findall(r'^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; [0-9]+ filtered out; finished in ([0-9]+\.[0-9]+)s$', text, re.M)
    require(len(matches) == 1, 'Exactly one passing single-test libtest timing footer required')
    value = Decimal(matches[0])
    require(value.is_finite() and value > 0, 'Nonpositive whole-test elapsed time')
    return dict(printed_seconds=matches[0], seconds=float(value),
                printed_resolution_seconds=str(Decimal(1).scaleb(value.as_tuple().exponent)),
                scope='whole test including IO/construction/checks/teardown; not TTFT/token/request latency')


def payload_argv(plan, row):
    job(plan, row)
    return [plan['executable'], row['selector'], '--exact', '--ignored', '--nocapture', '--test-threads=1']


def profile_argv(plan, row, directory, target):
    job(plan, row)
    require(row['phase'] == 'profile', 'Profiler cannot wrap an unprofiled pair/primary')
    return [plan['nsys'], 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
            '--trace-fork-before-exec=false', '--kill=none', '--wait=primary', '--stop-on-exit=true',
            '--delay=0', '--duration=1080', '--cuda-flush-interval=1000', '--cuda-event-trace=false',
            '--cuda-memory-usage=false', '--cuda-um-cpu-page-faults=false', '--cuda-um-gpu-page-faults=false',
            '--stats=false', '--force-overwrite=false', '--output=' + str(directory / 'nsys'), *target]


def validate_role(record, wanted, binary_sha, command):
    require(set(record) >= {'before', 'after', 'executable_sha256', 'command'}, 'Missing live model role')
    for point in ('before', 'after'):
        actual = record[point]
        require(all(type(wanted.get(k)) is int and wanted[k] > 0 and actual.get(k) == wanted[k]
                    for k in ('pid', 'birth_ticks', 'ppid')) and actual.get('state') in ('R', 'S', 'D', 'I', 'T', 't'),
                'Model PID/birth/parent/liveness differs')
    require(record['executable_sha256'] == binary_sha and record['command'] == command,
            'Model executable/selector differs')


def pair_summary(results):
    measured = [r for r in results if r['job']['phase'] == 'latency']
    require(len(measured) == 16 and all(r['status'] == 'passed' for r in measured), 'Exactly16 successful unprofiled trials required')
    pairs = []
    for schedule in ('blocking', 'lookahead'):
        for repetition in range(1, 5):
            rows = [r for r in measured if r['job']['schedule'] == schedule and r['job']['repetition'] == repetition]
            require(len(rows) == 2 and {r['job']['transfer_mode'] for r in rows} == {'synchronous', 'retained'}, 'Missing/duplicate pair')
            values = {r['job']['transfer_mode']: r['whole_test_elapsed']['seconds'] for r in rows}
            pairs.append(dict(schedule=schedule, repetition=repetition, whole_test_seconds=values,
                              retained_over_synchronous=values['retained'] / values['synchronous']))
    from statistics import median
    by_schedule = {schedule: dict(pairs=4, median_retained_over_synchronous=median(ratios),
            min_retained_over_synchronous=min(ratios), max_retained_over_synchronous=max(ratios))
        for schedule in ('blocking', 'lookahead')
        for ratios in [[row['retained_over_synchronous'] for row in pairs if row['schedule'] == schedule]]}
    return dict(pairs=pairs, by_schedule=by_schedule, significance_claim=False, sample_size_warning='Only4 pairs per schedule; no p95/p99/significance',
                scope='whole-test elapsed only; profiling and diagnostic primaries excluded')
