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
