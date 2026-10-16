"""Raw read-ledger and constructor checks; reported PASS flags never repair missing evidence."""
from runnerhelpers.evidence_v2 import integer, require
from runnerhelpers.closure_v2 import evidence


def read_observation(value, options, effective_required=False, final=False):
    require(isinstance(value, dict) and value.get('enabled') is True
            and value.get('requested_mode') == options['schedule'] and value.get('limits') == options,
            'raw read options disabled/mismatched')
    require(value.get('effective_mode') == options['schedule'] or
            (not effective_required and value.get('effective_mode') is None), 'raw effective read mode mismatch')
    require(value.get('status') not in ('quarantined', 'unknown_ledger_poisoned', 'disabled')
            and isinstance(value.get('status'), str), 'raw read status unknown/disabled')
    if final:
        require(value['status'] in ('closed', 'closed_retained'), 'raw worker read close pending/unknown')
    stats, charge = value.get('stats', {}), value.get('current_charge', {})
    for key in ('unknown_reads', 'unknown_bytes', 'unobserved_terminal_reads', 'unobserved_reader_intervals', 'join_failed_sequences'):
        require(type(stats.get(key)) is int and stats[key] == 0, 'raw read custody unknown: ' + key)
    for key in ('unknown_packed_bytes', 'unknown_stack_bytes', 'unknown_read_slots'):
        require(type(charge.get(key)) is int and charge[key] == 0, 'raw read charge unknown: ' + key)
    payload = options['max_host_bytes'] - options['max_metadata_bytes'] - options['worker_stack_bytes']
    for current, peak, cap in (
        ('packed_bytes', 'peak_packed_bytes', payload // 2),
        ('artifact_bytes', 'peak_artifact_bytes', payload // 2),
        ('payload_bytes', 'peak_payload_bytes', payload),
        ('metadata_bytes', 'peak_metadata_bytes', options['max_metadata_bytes']),
        ('requested_stack_bytes', 'peak_requested_stack_bytes', options['worker_stack_bytes']),
        ('host_charged_bytes', 'peak_host_charged_bytes', options['max_host_bytes'])):
        require(integer(charge.get(current)) and integer(stats.get(peak))
                and charge[current] <= stats[peak] <= cap, 'raw read budget missing/exceeded: ' + current)
    require(integer(charge.get('reserved_read_slots')) and charge['reserved_read_slots'] <= 2, 'raw read slot budget')
    if effective_required:
        require(value.get('worker_read_observed') is True and all(integer(stats.get(key), 1) for key in
            ('submitted_reads', 'completed_reads', 'verified_reads', 'io_started_reads', 'proved_read_bytes')),
            'raw positive worker-read evidence missing')
    return value


def interval_endpoints(row, options):
    endpoints = []
    for name in ('before', 'after'):
        endpoint = row.get(name, {})
        observed = []
        for key in ('scope', 'prepared_plan_id', 'image_generation', 'expert_read_pipeline'):
            state, value = evidence(endpoint.get(key))
            require(state in ('observed', 'verified') and value is not None, 'raw interval identity/read endpoint unavailable')
            observed.append(value)
        require(isinstance(observed[0], dict) and all(integer(observed[0].get(k), 1) for k in ('owner', 'generation')),
                'raw endpoint owner identity missing')
        read_observation(observed[3], options)
        endpoints.append(observed)
    require(endpoints[0][:3] == endpoints[1][:3], 'raw owner/source identity changed during read interval')
    delta = row.get('delta', {})
    require(isinstance(delta, dict) and delta, 'raw read counter delta missing')
    for key in delta:
        if key.startswith('peak_'):
            continue
        before = endpoints[0][3]['stats'].get(key)
        after = endpoints[1][3]['stats'].get(key)
        require(integer(before) and integer(after) and after >= before and delta[key] == after - before,
                'raw read counter reset/delta mismatch')
    if row.get('positive_worker_read') is True:
        require(all(integer(delta.get(k), 1) for k in
            ('submitted_reads', 'completed_reads', 'verified_reads', 'io_started_reads', 'proved_read_bytes')),
            'raw positive read interval denominator missing')
        read_observation(endpoints[1][3], options, effective_required=True)


def constructor(report):
    require(report.get('schema') == 'ferrule.expert-read-constructor-quality.v1'
            and report.get('complete') is True and report.get('passed') is True
            and report.get('expected_cells') == 4 and report.get('total_logit_values_checked') == 128
            and report.get('diagnostic_trace_installed') is False
            and report.get('ordinary_no_observer_claim') is True, 'actual constructor gate missing/partial')
    cells = report.get('cells', [])
    require(len(cells) == 4, 'constructor four-cell denominator missing')
    keys = set()
    for cell in cells:
        options = cell.get('requested', {})
        mode = options.get('schedule')
        require(mode in ('Blocking', 'Lookahead') and options == dict(schedule=mode,
            max_host_bytes=2 << 20, max_metadata_bytes=256 << 10, worker_stack_bytes=1 << 20), 'tiny constructor budget changed')
        keys.add((cell.get('model_boundary'), mode))
        require(all(cell.get(k) is True for k in ('passed', 'same_owner', 'charged_limits_satisfied', 'positive_worker_read'))
                and cell.get('logit_values_checked') == 32 and type(cell.get('state_bytes_after_shutdown')) is int and cell['state_bytes_after_shutdown'] == 0,
                'constructor actual cell evidence missing/failed')
        read_observation(cell.get('observed'), options, effective_required=True)
    require(keys == {(boundary, mode) for boundary in ('Bf16ModelF32Accumulate', 'W8A16Bf16F32Accumulate')
                     for mode in ('Blocking', 'Lookahead')}, 'constructor missing/duplicate mode/model cell')


# Explicitly versioned: historical v1/v2 receipts keep their original contract.
V3_COUNTERS = tuple("""
sequences_started sequences_finished workers_created worker_reuses workers_joined
join_pending_sequences join_failed_sequences submitted_reads speculative_reads
rejected_submissions demanded_reads completed_reads verified_reads cancelled_reads
failed_reads unknown_reads unobserved_terminal_reads worker_started_reads
io_started_reads prefetched_reads_taken resident_skips demand_failures
materialization_failures wasted_reads wasted_read_bytes unproved_wasted_reads
proved_read_bytes reader_interval_calls reader_interval_bytes
wasted_reader_interval_calls wasted_reader_interval_bytes unobserved_reader_intervals
demand_wait_calls demand_wait_ns io_duration_ns unknown_bytes
""".split())
V3_PEAKS = tuple('peak_' + key for key in (
    'packed_bytes', 'artifact_bytes', 'metadata_bytes', 'requested_stack_bytes',
    'payload_bytes', 'host_charged_bytes'))
V3_CHARGES = tuple("""
packed_bytes artifact_bytes metadata_bytes requested_stack_bytes payload_bytes
host_charged_bytes unknown_packed_bytes unknown_stack_bytes reserved_read_slots
unknown_read_slots
""".split())
V3_SCOPES = dict(
    timing_origin='expert_read_sequence_start_host_monotonic_ns_not_cuda',
    reader_counter_scope='shared_checkpoint_reader_submit_through_take_not_ticket_certified',
    charge_basis='logical_host_reservation_not_physical_rss',
    stats_scope='owner_lifetime_owner_observed_transitions_peaks_not_invocation_deltas')
CONSTRUCTOR_DROP_UNOBSERVED = 'unobserved_public_constructor_api_has_no_retained_worker_witness'


def _v3_integer(value):
    return integer(value) and value <= (1 << 64) - 1


def read_observation_v3(value, options, *, stage=None):
    """Validate a copied settled endpoint; never drive/join/refund an owner.

    ``closed`` is structural evidence only. Call final_owner_join_v3 with the
    retained pre-drop observation to qualify a real final-owner join witness.
    """
    require(isinstance(options, dict) and set(options) == {
        'schedule', 'max_host_bytes', 'max_metadata_bytes', 'worker_stack_bytes'}
        and options['schedule'] in ('blocking', 'lookahead')
        and all(_v3_integer(options[k]) and options[k] > 0 for k in
                ('max_host_bytes', 'max_metadata_bytes', 'worker_stack_bytes')),
        'v3 missing/invalid typed options')
    require(options['max_metadata_bytes'] + options['worker_stack_bytes'] <= options['max_host_bytes'],
            'v3 metadata/stack exceeds total host cap')
    require(isinstance(value, dict) and value.get('enabled') is True
        and value.get('limits') == options and value.get('requested_mode') == options['schedule']
        and all(value.get(k) == v for k, v in V3_SCOPES.items()), 'v3 raw options/scope mismatch')
    status = value.get('status')
    require(status in ('configured', 'idle_retained', 'closed')
            and (stage is None or status == stage), 'v3 unsupported/incorrect worker stage')
    require(value.get('worker_unknown') is False and value.get('worker_join_pending') is False
            and type(value.get('worker_alive')) is bool
            and type(value.get('worker_read_observed')) is bool, 'v3 worker Unknown/pending/liveness missing')
    require('current_read' in value and value['current_read'] is None
            and 'pending_read' in value and value['pending_read'] is None and 'last_receipt' in value,
            'v3 current/pending read custody retained or missing')
    stats, charge = value.get('stats'), value.get('current_charge')
    require(isinstance(stats, dict) and set(stats) == set(V3_COUNTERS + V3_PEAKS)
            and all(_v3_integer(v) for v in stats.values()), 'v3 typed full-owner stats missing/invalid')
    require(isinstance(charge, dict) and set(charge) == set(V3_CHARGES)
            and all(_v3_integer(v) for v in charge.values()), 'v3 typed host charge missing/invalid')
    require(all(stats[k] == 0 for k in ('unknown_reads', 'unknown_bytes', 'unobserved_terminal_reads',
        'unobserved_reader_intervals', 'join_failed_sequences', 'join_pending_sequences',
        'failed_reads', 'demand_failures', 'materialization_failures', 'unproved_wasted_reads')),
        'v3 sticky unknown/pending/failed lifetime')
    require(stats['sequences_started'] == stats['sequences_finished']
            == stats['workers_created'] + stats['worker_reuses']
            and stats['workers_created'] - stats['workers_joined'] == int(value['worker_alive']),
            'v3 inconsistent worker creation/reuse/join lifetime')
    require(stats['submitted_reads'] == stats['completed_reads']
            == stats['verified_reads'] + stats['cancelled_reads'] + stats['failed_reads']
            and stats['verified_reads'] <= stats['io_started_reads'] <= stats['worker_started_reads'] <= stats['submitted_reads']
            and value['worker_read_observed'] is (stats['io_started_reads'] > 0 and stats['proved_read_bytes'] > 0),
            'v3 inconsistent read lifetime')
    expected = dict.fromkeys(V3_CHARGES, 0)
    if status == 'configured':
        require(value.get('effective_mode') is None and 'effective_mode' in value
                and 'last_receipt' in value and value['last_receipt'] is None
                and not any(stats.values()), 'v3 configured endpoint already executed')
    else:
        require(stats['workers_created'] == 1 and value.get('effective_mode') == options['schedule'],
                'v3 compatible scopes must retain exactly one worker')
        require(value['worker_alive'] is (status == 'idle_retained'), 'v3 worker status/liveness mismatch')
        if status == 'idle_retained':
            expected.update(metadata_bytes=options['max_metadata_bytes'],
                requested_stack_bytes=options['worker_stack_bytes'],
                host_charged_bytes=options['max_metadata_bytes'] + options['worker_stack_bytes'])
        require(stats['peak_metadata_bytes'] == options['max_metadata_bytes']
                and stats['peak_requested_stack_bytes'] == options['worker_stack_bytes']
                and stats['peak_host_charged_bytes'] >= options['max_metadata_bytes'] + options['worker_stack_bytes'],
                'v3 lifetime metadata/stack peak missing')
    require(charge == expected, 'v3 exact stage charge/payload/slots/artifacts mismatch')
    read_observation(value, options)
    return value


def _v3_endpoint(physical, options):
    require(isinstance(physical, dict) and physical.get('release') == {'state': 'unobserved'},
            'v3 physical release must remain unobserved')
    identity = []
    for key in ('scope', 'prepared_plan_id', 'image_generation', 'expert_read_pipeline'):
        state, value = evidence(physical.get(key))
        require(state == 'observed' and value is not None, 'v3 raw identity/read endpoint unavailable')
        identity.append(value)
    require(isinstance(identity[0], dict) and set(identity[0]) == {'owner', 'generation'}
            and all(_v3_integer(v) and v > 0 for v in identity[0].values())
            and all(_v3_integer(v) and v > 0 for v in identity[1:3]), 'v3 invalid raw owner/source identity')
    read = read_observation_v3(identity[3], options)
    require(read['status'] in ('configured', 'idle_retained'),
            'v3 sampled live materializer cannot claim final join')
    return identity[:3], read


def interval_endpoints_v3(row, options):
    require(isinstance(row, dict) and row.get('requested') == options
            and all(row.get(k) is True for k in ('same_owner', 'configuration_matches',
                'within_charged_limits', 'unambiguous', 'evidence_complete')),
            'v3 interval flags/options incomplete')
    pair = row.get('configuration_pair')
    require(isinstance(pair, dict), 'v3 endpoint qualification pair missing')
    for name in ('before', 'after'):
        flags = pair.get(name, {})
        require(isinstance(flags, dict), 'v3 endpoint qualification flags missing')
        require(all(flags.get(k) is True for k in ('configuration_matches', 'within_charged_limits',
                'unambiguous', 'worker_snapshot_valid'))
                and 'worker_snapshot_rejection' in flags and flags['worker_snapshot_rejection'] is None,
                'v3 endpoint worker qualification missing/rejected')
    return _v3_interval(row, options)


def _v3_interval(row, options):
    endpoints = [_v3_endpoint(row.get(name), options) for name in ('before', 'after')]
    require(endpoints[0][0] == endpoints[1][0], 'v3 raw owner/source identity changed')
    before, after = (item[1]['stats'] for item in endpoints)
    delta = row.get('delta')
    require(isinstance(delta, dict) and set(delta) == set(V3_COUNTERS + V3_PEAKS)
            and all(_v3_integer(v) for v in delta.values()), 'v3 complete typed counter delta missing')
    for key in V3_COUNTERS:
        require(after[key] >= before[key] and delta[key] == after[key] - before[key],
                'v3 full-owner counter reset/delta mismatch: ' + key)
    for key in V3_PEAKS:
        require(after[key] >= before[key] and delta[key] == after[key],
                'v3 lifetime peak reset/delta mismatch: ' + key)
    positive = all(delta[k] > 0 for k in
        ('submitted_reads', 'completed_reads', 'verified_reads', 'io_started_reads', 'proved_read_bytes'))
    positive = positive and endpoints[1][1]['worker_read_observed']
    require(type(row.get('positive_worker_read')) is bool and row['positive_worker_read'] is positive,
            'v3 positive read flag/counter mismatch')
    if positive:
        require(row.get('operation') == 'prediction', 'v3 positive read outside prediction')
    return endpoints


def read_quality_v3(report, mode):
    """Corrected-cut contract; revision 2 cannot be accepted as v3 evidence."""
    options = dict(schedule=mode, max_host_bytes=64 << 20,
                   max_metadata_bytes=8 << 20, worker_stack_bytes=2 << 20)
    require(isinstance(report, dict), 'v3 report must be an object')
    q = report.get('expert_read_quality')
    require(isinstance(q, dict), 'v3 raw quality report missing')
    require(q.get('schema') == 'ferrule.expert-read-quality-evidence.v1'
            and type(q.get('interval_validation_revision')) is int and q['interval_validation_revision'] == 3
            and type(q.get('worker_snapshot_contract_revision')) is int and q['worker_snapshot_contract_revision'] == 1
            and q.get('requested') == options and q.get('passed') is True
            and q.get('charge_basis') == V3_SCOPES['charge_basis'], 'v3 report revision/config missing or stale')
    calls = q.get('calls')
    require(isinstance(calls, list) and calls and type(q.get('sampled_operations')) is int
            and q['sampled_operations'] == len(calls), 'v3 sampled operation denominator mismatch')
    previous = None
    for row in calls:
        endpoints = interval_endpoints_v3(row, options)
        # Adjacent samples can enclose unsampled work, but not a new owner,
        # a restarted worker, counter reset, or premature final-owner join.
        if previous is not None:
            identity, observation = endpoints[0]
            require(identity == previous[0] and all(observation['stats'][k] >= previous[1]['stats'][k]
                    for k in V3_COUNTERS + V3_PEAKS), 'v3 sampled owner/counters reset between intervals')
        previous = endpoints[1]
    require(previous[1]['status'] == 'idle_retained' and previous[1]['stats']['worker_reuses'] > 0,
            'v3 compatible real-model scopes must demonstrate worker reuse')
    positive = sum(row['positive_worker_read'] for row in calls)
    require(positive > 0 and type(q.get('positive_worker_read_calls')) is int
            and q['positive_worker_read_calls'] == positive, 'v3 positive read denominator mismatch')


def final_owner_join_v3(idle, witness, options):
    """Only explicit passive post-final-owner-drop evidence proves the host join."""
    read_observation_v3(idle, options, stage='idle_retained')
    read_observation_v3(witness, options, stage='closed')
    expected = dict(idle['stats'], workers_joined=1)
    require(witness['stats'] == expected and all(witness[k] == idle[k] for k in
        ('enabled', 'requested_mode', 'effective_mode', 'limits', 'last_receipt', *V3_SCOPES)),
        'v3 final-owner witness changed more than the real final join')


def constructor_v3(report):
    require(report.get('schema') == 'ferrule.expert-read-constructor-quality.v1'
            and type(report.get('host_lifecycle_contract_revision')) is int
            and report['host_lifecycle_contract_revision'] == 2
            and report.get('host_charge_basis') == V3_SCOPES['charge_basis']
            and report.get('worker_counter_scope') == 'full_owner_lifetime_created_reuses_joined_not_submitted_reads'
            and report.get('complete') is True and report.get('passed') is True
            and report.get('expected_cells') == 4 and report.get('total_logit_values_checked') == 128
            and report.get('diagnostic_trace_installed') is False
            and report.get('ordinary_no_observer_claim') is True,
            'v3 constructor lifecycle contract/denominator missing')
    cells = report.get('cells', [])
    require(isinstance(cells, list) and len(cells) == 4 and
            {(c.get('model_boundary'), c.get('requested', {}).get('schedule')) for c in cells}
            == {(b, m) for b in ('W8A16Bf16F32Accumulate', 'Bf16ModelF32Accumulate')
                for m in ('blocking', 'lookahead')}, 'v3 constructor boundary/schedule cells differ')
    for cell in cells:
        options = cell['requested']
        require(options == dict(schedule=options['schedule'], max_host_bytes=2 << 20,
                    max_metadata_bytes=256 << 10, worker_stack_bytes=1 << 20)
                and all(cell.get(k) is True for k in
                    ('passed', 'same_owner', 'positive_worker_read', 'charged_limits_satisfied'))
                and cell.get('logit_values_checked') == 32 and cell.get('quarantine') is False
                and type(cell.get('state_bytes_after_shutdown')) is int and cell['state_bytes_after_shutdown'] == 0,
                'v3 constructor numeric/budget/cleanup drift')
        read_observation_v3(cell.get('configured'), options, stage='configured')
        observed = read_observation_v3(cell.get('observed'), options, stage='idle_retained')
        require(observed['stats']['submitted_reads'] == observed['stats']['verified_reads'] == 6
                and observed['stats']['sequences_started'] == 2 and observed['stats']['worker_reuses'] == 1
                and observed['stats']['cancelled_reads'] == 0
                and observed['stats']['reader_interval_calls'] > 0
                and observed['stats']['reader_interval_bytes'] > 0,
                'v3 constructor read/worker reuse denominator differs')
        # Tiny constructor has its unchanged 1.25 MiB fixture charge, not the
        # real-model 10 MiB reservation. Neither is a physical RSS observation.
        row = dict(before=cell.get('before_prediction'), after=cell.get('after_prediction'),
            delta=cell.get('delta'), positive_worker_read=True, operation='prediction')
        endpoints = _v3_interval(row, options)
        require(endpoints[0][1] == cell['configured'] and endpoints[1][1] == observed,
                'v3 constructor copied observation differs from raw endpoint')
        require(cell.get('host_after_prediction_stage') == 'payloads_released_worker_idle_retained'
                and cell.get('host_after_shutdown_before_owner_drop') == observed,
                'v3 constructor shutdown must retain idle charge until final owner drop')
        require('host_after_final_owner_drop' in cell and cell['host_after_final_owner_drop'] is None
                and cell.get('host_final_join_witness') == CONSTRUCTOR_DROP_UNOBSERVED,
                'v3 public constructor postDrop must be null/unobserved, not a fabricated join')
