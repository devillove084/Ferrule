"""Recompute the paired study; export only allowlisted public plotting data."""
import json
from derive import (ROOT, BASE, RAW, PLAN, ANALYSIS, MODES, ORDER, TRIALS, NAMES,
                    METRICS, MIB, GIB, require, digest, input_paths, paired_summary)
from receipts import arm_data

SOURCES = ('derive.py', 'receipts.py', 'study.py', 'plot_paired_read.py', 'render.py')


def source_hashes(root=ROOT):
    return {ANALYSIS + name: digest((root / ANALYSIS / name).read_bytes()) for name in SOURCES}


def validate_identity(audit, binding, review):
    keys = ('commit', 'source_digest', 'source_archive_sha256', 'cli_binary_sha256', 'source_members')
    identity = {k: audit['source_identity'][k] for k in keys}
    require(all(identity == {k: row['candidate_identity'][k] for k in keys} for row in (binding, review)),
            'actual source identity differs; historical identity cannot substitute')
    require(identity['source_members'] == review['source_members']
            == review['source_disposition']['compiled_archive_members_unchanged'] == 1484,
            'actual source archive member count differs')
    return identity


def build_data(sources, manifest, manifest_hash):
    require(set(sources) == set(manifest['files']) == set(input_paths()), 'input copy set differs')
    require(all(digest(content) == manifest['files'][path] for path, content in sources.items()), 'input copy hash mismatch')
    load = lambda path: json.loads(sources[path])
    final = load(RAW + 'paired-final.json')
    audit = load(PLAN + 'poststudy-audit.json')
    prereg = load(PLAN + 'preregistration.json')
    binding = load(PLAN + 'actual-binding-v3.json')
    review = load(PLAN + 'actual-build-review-v3.json')
    identity = validate_identity(audit, binding, review)
    require(audit['paired_final_sha256'] == digest(sources[RAW + 'paired-final.json']), 'poststudy final hash differs')
    require(final['order'] == prereg['order'] == load(BASE + 'order.json')['blocks'] == ORDER,
            'frozen block order differs')
    require(prereg['trial_order'] == [r['trial'] for r in final['lifecycles']] == TRIALS, 'actual arm order differs')
    require(final['status'] == 'complete' and final['actual_completed_lifecycles'] == 10
            and final['actual_issued_lifecycles'] == 10, 'failed study cannot produce complete-case plots')
    require(audit['completed_lifecycles'] == 10 and audit['failed_lifecycles'] == audit['unissued_lifecycles'] == 0,
            'audit lifecycle denominator differs')
    require(all(final[k] == 0 for k in ('retries', 'replacements', 'extra_pilot_lifecycles', 'profile_lifecycles')),
            'unplanned replacement, retry, probe or profiling')
    require(final['historical_resume_permitted'] is False and prereg['pooling'] is False,
            'historical pooling forbidden')
    require(final['scientific_acceptance'] is False and audit['scientific_acceptance'] is False,
            'descriptive study cannot become scientific acceptance')
    for key, filename in (('protocol_sha256', 'protocol-v3.json'), ('cases_sha256', 'cases-v3.json')):
        require(final[key] == prereg[key] == binding[key] == digest(sources[BASE + filename]), 'protocol/case hash differs')
    # Validate the selected raw inputs against the study's pre-existing pins, without
    # publishing its private map paths or walking unrelated historical artifacts.
    checked_pins = set()
    for receipt in (final, audit, prereg):
        for raw_path, sha in receipt['artifact_pins'].items():
            if BASE in raw_path:
                path = raw_path[raw_path.index(BASE):]
                if path in sources:
                    require(digest(sources[path]) == sha, 'original receipt pin differs: ' + path)
                    checked_pins.add(path)
    require(all(path in checked_pins for path in input_paths() if path.startswith(RAW)), 'unsealed raw plot input')
    histories = []
    require(audit['historical_attempts'] == prereg['historical_attempts'], 'historical attempt accounting differs')
    for record in audit['historical_attempts']:
        require(record['status'] == 'stopped' and record['pooled'] is False and record['resumed'] is False,
                'historical failed attempt cannot be pooled')
        histories.append({k: record[k] for k in ('sha256', 'status', 'issued_lifecycles', 'completed_lifecycles',
                                                'unissued_requests', 'scheduled_measured_qa', 'pooled', 'resumed')})
    require(len(histories) == 2, 'historical failed-study accounting differs')
    frozen = load(BASE + 'oracles.json')
    oracles = {r['id']: r for r in frozen['references']}
    schedule = load(BASE + 'requests.json')['schedule']
    require([r['name'] for r in schedule] == NAMES, 'twelve frozen slots required')
    audit_arms = {r['trial']: r for r in audit['arms']}
    audit_requests = {(r['trial'], r['name']): r for r in audit['requests']}
    require(len(audit_arms) == 10 and len(audit_requests) == len(audit['requests']) == 120, 'duplicate/missing audited slots')
    arms = []
    cases = load(BASE + 'cases-v3.json')
    for arm in final['lifecycles']:
        trial, mode = arm['trial'], arm['mode']
        arms.append(arm_data(trial, mode, arm, audit_arms[trial], audit_requests, schedule,
                             oracles, sources, RAW + trial + '/', cases))
    result = paired_summary({r['trial']: r['metrics'] for r in arms})
    require(result['blocks'] == final['analysis']['blocks'] == audit['analysis']['blocks'], 'frozen paired estimator differs')
    for key in METRICS:
        computed, frozen_metric = result['aggregate'][key], final['analysis']['aggregate'][key]
        require(computed['ratio'] == frozen_metric['ratio']
                and computed['descriptive_range'] == frozen_metric['interval_ratio']
                and computed['draws'] == frozen_metric['draws']
                and computed['nearest_ranks_1based'] == frozen_metric['ranks'], 'exact paired bootstrap differs')
    denominators = final['analysis']['denominators']
    require(denominators == dict(scheduled=120, issued=120, terminal=120, successful=120,
                                 failed=0, censored=0, unissued=0), 'failed/censored/unissued request denominator differs')
    coverage = dict(requests=sum(len(r['requests']) for r in arms), measured_qa=sum(
        q['measured'] for r in arms for q in r['requests']),
        actual_worker_generated_tokens=sum(q['completion_tokens'] for r in arms for q in r['requests']),
        exact_eight_token_oracles=sum(q['exact_token_oracle'] is True for r in arms for q in r['requests']),
        length_only_two_token_smoke=sum(q['exact_token_oracle'] is None for r in arms for q in r['requests']),
        pipeline_invocation_jsons=sum(len(r['timeline']) for r in arms))
    require(coverage == dict(requests=120, measured_qa=50, actual_worker_generated_tokens=900,
                            exact_eight_token_oracles=110, length_only_two_token_smoke=10,
                            pipeline_invocation_jsons=1020), 'actual quality/timeline coverage differs')
    require(all(audit['coverage'][k] == v for k, v in coverage.items()), 'poststudy quality coverage differs')
    require(audit['coverage']['startup_shutdown_pid_io_deltas'] == 0, 'startup/shutdown PID IO coverage differs')
    for arm in arms:
        require(arm['logical_counters']['checkpoint_read_bytes'] == 128811263616
                and arm['logical_counters']['h2d_bytes'] == 128811927312, 'equal-work read/H2D totals differ')
        require(all(arm['pipeline_stats'][k] == 40943 for k in ('verified_reads', 'completed_reads', 'demanded_reads'))
                and arm['pipeline_stats']['proved_read_bytes'] == arm['logical_counters']['checkpoint_read_bytes']
                and arm['pipeline_stats']['reader_interval_bytes'] == arm['logical_counters']['checkpoint_read_bytes'],
                'equal verified read accounting differs')
        require(arm['host_logical_peak_bytes'] == 16777984 and arm['sampled_device_max_bytes'] == 4125 * MIB,
                'observed logical/device gauges differ')
        require(all(arm['pipeline_stats'][k] == 0 for k in ('wasted_reads', 'wasted_read_bytes',
                    'unproved_wasted_reads', 'unknown_reads', 'unknown_bytes', 'failed_reads',
                    'cancelled_reads', 'unobserved_reader_intervals')), 'zero-waste/known-accounting evidence differs')
        require(arm['nfs_phase_client_serverreadbytes']['requests_nfs'] == (314825320 if arm['trial'] == 'P1-blocking' else 0),
                'shared NFS client condition differs')
    by_trial = {r['trial']: r for r in arms}
    changes = []
    for block in ORDER:
        b, l = (by_trial[block['block'] + '-' + m] for m in MODES)
        changes.append(dict(block=block['block'], execution_order=block['modes'],
            wait_change_seconds=l['median_request_demand_wait_seconds'] - b['median_request_demand_wait_seconds'],
            e2e_change_seconds=l['metrics']['request_e2e_seconds'] - b['metrics']['request_e2e_seconds'],
            ttft_change_seconds=l['metrics']['ttft_seconds'] - b['metrics']['ttft_seconds']))
    return dict(schema='ferrule.paired-read.plot-data.v1', provenance=dict(
        input_manifest_path=ANALYSIS + 'inputs.json', input_manifest_sha256=manifest_hash,
        input_count=len(sources), checked_original_pin_count=len(checked_pins),
        source_hashes=source_hashes(), actual_candidate_identity=identity,
        time_policy='No analysis timestamp; only bound recorded measurements',
        privacy_policy='Allowlisted fields only; no environment, credentials, process maps, mount paths, or raw copies'),
        design=dict(independent_unit='paired block', n=5, measured_requests_per_arm=5,
                    total_measured_requests=50, order=ORDER, retries=0, replacements=0,
                    historical_studies_pooled=False, instrumented=True,
                    startup_scope='recorded launch-to-ready probe; not lifecycle elapsed time or a request',
                    scientific_acceptance=False),
        denominators=denominators, coverage=coverage, paired=result, arms=arms, wait_latency_changes=changes,
        excluded_historical_attempts=histories,
        units=dict(bytes_per_MiB=MIB, bytes_per_GiB=GIB, nanoseconds_per_second=10**9),
        unknowns=dict(startup_pid_io_delta=None, shutdown_pid_io_delta=None, startup_logical_read_bytes=None,
                      physical_nas_backend_bytes=None, owned_process_gpu_peak_bytes=None,
                      universal_physical_release=None, continuous_device_peak_bytes=None,
                      causal_critical_path_savings_seconds=None),
        interpretation=dict(bootstrap='Exact 3125 paired-block resamples; nearest ranks 79 and 3047; descriptive, not coverage-guaranteed',
            wait='Sum per-invocation host demand_wait_ns within each measured request, then median of five requests per arm. Association, not GPU-overlap or critical-path causality proof.',
            accounting='Logical reader and facade H2D request-period totals per lifecycle; equal between modes, not equal to each other. Not physical backend or PCIe traffic.',
            memory='Owner-lifetime logical host reservation maximum and device-wide sampled maximum have separate scopes. Neither is RSS; do not add PSS or sum lifetime peaks.',
            cache='Shared NFS client counters; fresh server is not a cold cache. P1 blocking has nonzero serverreadbytes, other arms record zero.',
            quality='900 worker-enqueued token IDs joined to HTTP IDs; 110 exact eight-token oracles and 10 length-only two-token smokes. SSE events are not token boundaries.',
            limitations='One sealed instrumented exploratory n=5 study; no significance, power, repetition-based confidence, SOTA, physical-release or backend-I/O claim.'))
