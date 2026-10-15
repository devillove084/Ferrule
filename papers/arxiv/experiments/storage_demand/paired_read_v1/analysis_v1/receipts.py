"""Checked, deliberately allowlisted projections of private receipts."""
import json
import statistics
from derive import (COUNTERS, MIB, PHASE_SNAPSHOTS, integer, number, require,
                    log_rows, gauge, digest)


def nfs_delta(before, after):
    require(before['acquisition_status'] == after['acquisition_status'] == 'captured', 'missing NFS window')
    b, a = before['selected_mount'], after['selected_mount']
    require(not b['errors'] and not a['errors'], 'NFS parse errors')
    require(all(b[k] == a[k] for k in ('device', 'mountpoint', 'fstype', 'statvers')), 'NFS identity differs')
    require(set(b['counters']) == set(a['counters']), 'NFS counter coverage differs')
    result = {k: a['counters'][k] - v for k, v in b['counters'].items()}
    require(all(integer(v) for v in result.values()), 'NFS counter reset or missing value')
    return result


def check_tokens(request, rows, oracle):
    actual, trace = request['actual'], request['trace']
    n = request['max_tokens']
    require(request['status'] == 'success' and request['metric_eligible'] is True
            and request['quality_passed'] is True and actual['status'] == 'success', 'failed request retained; no plot')
    require(trace['trace_valid'] is True and trace['terminal_reconciled'] is True
            and trace['binding_status'] == 'bound' and not trace['duplicate_http_binding'], 'invalid token join')
    fields = [r['fields'] for r in rows]
    emitted = [r for r in fields if r.get('event') == 'request_worker_token_enqueued']
    ids = [r['actual_token_id'] for r in emitted]
    ordinals = [r['emission_ordinal'] for r in emitted]
    require(ordinals == trace['enqueue_ordinals'] == list(range(1, n + 1)), 'token enqueue ordinals differ')
    require(ids == trace['actual_generated_ids'] and len(ids) == n and all(integer(i) for i in ids), 'actual token IDs differ')
    worker = trace['actual_worker_request_id']
    require(all(r['request_id'] == worker for r in emitted), 'mixed worker IDs in serial window')
    http_ids = set()
    content_times = []
    usage = None
    for event in actual['raw_sse_events']:
        if event['data'] == '[DONE]':
            continue
        obj = json.loads(event['data'])
        if 'id' in obj:
            http_ids.add(obj['id'])
        if obj.get('usage'):
            usage = obj['usage']
        if any(choice.get('text') for choice in obj.get('choices', [])):
            content_times.append(event['at'])
    require(sorted(http_ids) == trace['actual_http_completion_ids'] == [f'cmpl-ferrule-{worker}'], 'HTTP/worker ID join differs')
    require(content_times == actual['content_event_times'] and content_times, 'content timing differs')
    require(actual['usage'] == usage and usage['completion_tokens'] == n
            and usage['total_tokens'] == usage['prompt_tokens'] + n, 'actual token usage differs')
    require(usage['prompt_tokens'] == request['input_tokens'], 'prompt usage differs')
    for event, key in (('request_worker_terminal_enqueue', 'worker_terminal_enqueue'),
                       ('request_http_terminal', 'http_response_terminal')):
        terminal = [r for r in fields if r.get('event') == event]
        require(terminal == trace[key] and len(terminal) == 1 and terminal[0]['request_id'] == worker
                and terminal[0]['status'] == 'finished', 'terminal token/HTTP reconciliation differs')
    require(trace['worker_terminal_enqueue'][0]['last_emission_ordinal'] == n
            and trace['http_response_terminal'][0]['token_events'] == n, 'terminal token count differs')
    if oracle is None:
        require(n == 2 and trace['exact_token_pass'] is None and request['reference'] is None,
                'smoke usage is not an exact-token oracle')
    else:
        require(n == 8 and request['reference'] == oracle and ids == oracle['reference_ids']
                and trace['exact_token_pass'] is True and actual['text'] == oracle['reference_text'], 'frozen token oracle differs')
    ttft = content_times[0] - actual['sent_at']
    e2e = actual['ended_at'] - actual['sent_at']
    require(number(ttft) and 0 < ttft <= e2e and ttft == actual['ttft_seconds']
            and e2e == actual['request_e2e_seconds'], 'HTTP latency equation differs')
    return dict(actual_generated_ids=ids, enqueue_ordinals=ordinals, worker_request_id=worker,
                exact_token_oracle=None if oracle is None else True, completion_tokens=n,
                ttft_seconds=ttft, request_e2e_seconds=e2e)


def add(rows):
    require(rows and all(set(r) == set(rows[0]) for r in rows), 'counter keys differ')
    require(all(integer(v) for r in rows for v in r.values()), 'unknown additive counter')
    return {k: sum(r[k] for r in rows) for k in rows[0]}


def arm_data(trial, mode, arm, audit_arm, audit_requests, schedule, oracles, sources, prefix, cases):
    load = lambda path: json.loads(sources[prefix + path])
    life = load('lifecycle.json')
    requests = load('requests-final.json')
    read_ledger = load('read-ledger-assessment.json')
    require(life == arm['lifecycle'] and requests == life['requests'], 'lifecycle/request input copy differs')
    require(read_ledger == arm['read_ledger'] and read_ledger['can_continue'] is True
            and arm['assessment']['can_continue'] is True, 'failed or divergent arm retained; no aggregate')
    request_summary = requests['summary']
    arm_summary = {**arm['summary'], 'measured_qa': {k: v for k, v in arm['summary']['measured_qa'].items() if k != 'request_only_eligible_success'}}
    require(request_summary == arm_summary and arm['summary']['measured_qa']['request_only_eligible_success'] == 5, 'arm denominator copy differs')
    require([q['name'] for q in requests['observations']] == [q['name'] for q in schedule], 'request schedule differs')
    require(len(requests['observations']) == 12, 'all twelve request slots required')
    ledger = read_ledger['final_read_ledger']
    log = sources[prefix + 'server.log']
    require(digest(log) == life['server_log_sha256'], 'server timeline hash differs')
    whole = log_rows(log)
    all_invocations = [r for r in whole if r['fields'].get('event') == 'mixed_decoder_invocation']
    records, timeline, used_lines = [], [], []
    previous_stats = None
    previous_end = 0
    for request, slot in zip(requests['observations'], schedule):
        name = request['name']
        require(all(request[k] == v for k, v in slot.items() if k not in ('status', 'result')), 'frozen request slot differs')
        require(request == load(name + '/observed.json'), 'observed request input copy differs')
        actual = request['actual']
        require(actual == load(name + '/http-terminal.json'), 'terminal request input copy differs')
        http = load(name + '/http.json')['runs'][0]['requests'][0]
        require({k: v for k, v in actual.items() if k not in ('adapter_http_report_complete', 'adapter_http_report_errors')} == http,
                'raw HTTP request input copy differs')
        issued = load(name + '/issued.json')
        require(all(issued[k] == slot[k] for k in slot if k not in ('status', 'result'))
                and issued['log_start_byte'] == request['log_start_byte'], 'issued request slot differs')
        start, end = request['log_start_byte'], request['log_end_byte']
        require(integer(start) and integer(end) and previous_end <= start < end <= len(log), 'overlapping/invalid log windows')
        require((start == 0 or log[start - 1:start] == b'\n') and log[end - 1:end] == b'\n', 'partial log line boundary')
        window = log_rows(log[start:end])
        selected = [r for r in window if r['fields'].get('event') == 'mixed_decoder_invocation']
        require(selected == request['logical_counters']['rows'] and request['logical_counters']['valid'] is True,
                'raw invocation timeline/input copy differs')
        oracle = oracles.get(request['reference_id']) if request['reference_id'] is not None else None
        require(oracle is not None or name == 'hello2', 'missing frozen oracle')
        token = check_tokens(request, window, oracle)
        counter = {k: sum(r['fields'][k] for r in selected) for k in COUNTERS}
        require(counter == request['logical_counters']['deltas'], 'logical byte/counter sum differs')
        nonpeaks, peaks = [], []
        line_base = log[:start].count(b'\n')
        for row in selected:
            f = row['fields']
            require(f['status'] == 'completed' and f['quarantined'] is False and f['counters_available'] is True
                    and f['counter_reset_detected'] is False and f['expert_read_counter_reset_detected'] is False,
                    'invalid/reset raw invocation')
            pipeline, delta = json.loads(f['expert_read_pipeline']), json.loads(f['expert_read_pipeline_delta'])
            require(pipeline['requested_mode'] == pipeline['effective_mode'] == mode
                    and pipeline['charge_basis'] == 'logical_host_reservation_not_physical_rss', 'read mode or charge scope differs')
            require(pipeline['limits']['max_host_bytes'] == 64 * MIB and pipeline['status'] == 'closed', 'read bound/closure differs')
            require(all(v == 0 for v in pipeline['current_charge'].values()), 'nonzero invocation close charge')
            stats = pipeline['stats']
            diff = {k: v for k, v in delta.items() if not k.startswith('peak_')}
            require(all(integer(v) for v in stats.values()) and all(integer(v) for v in diff.values()), 'unknown pipeline quantity')
            prior = previous_stats or {k: 0 for k in stats}
            require(all(stats[k] - prior[k] == v for k, v in diff.items()), 'cumulative/delta read counters differ')
            require(all(stats[k] >= prior[k] for k in stats if k.startswith('peak_')), 'owner peak decreased')
            require(stats['peak_host_charged_bytes'] <= pipeline['limits']['max_host_bytes'], 'logical host limit exceeded')
            previous_stats = stats
            nonpeaks.append(diff)
            peaks.append(stats['peak_host_charged_bytes'])
            source_line = line_base + row['line']
            used_lines.append(source_line)
            timeline.append(dict(ordinal=len(timeline) + 1, request=name, source_log_line=source_line,
                                 transaction=f['transaction'], rows=f['rows'], prefill_rows=f['prefill_rows'],
                                 decode_rows=f['decode_rows'], demand_wait_ns=diff['demand_wait_ns'],
                                 checkpoint_read_bytes=f['checkpoint_read_bytes'], h2d_bytes=f['h2d_bytes'],
                                 host_logical_peak_bytes=stats['peak_host_charged_bytes']))
        nfs = nfs_delta(load(name + '-pre-mountstats.json'), load(name + '-post-mountstats.json'))
        require(nfs == request['nfs']['deltas'] and request['nfs']['status'] == 'shared_client_delta', 'request NFS delta differs')
        io = request['process_io']
        require(all(io[side]['root_identity'][k] == life['server_identity'][k]
                    for side in ('before', 'after') for k in ('pid', 'birth_ticks', 'ppid')), 'PID birth binding differs')
        pid_io = {k: io['after']['counters'][k] - v for k, v in io['before']['counters'].items()}
        require(all(integer(v) for v in pid_io.values()), 'PID IO counter reset')
        record = dict(name=name, index=request['index'], phase=request['phase'], measured=request['measured'],
                      **token, decoder_invocations=len(selected), logical_counters=counter,
                      pipeline_nonpeak_deltas=add(nonpeaks), host_logical_peak_bytes=gauge(peaks)['maximum'],
                      log_byte_window=[start, end], source_log_lines=[line_base + r['line'] for r in selected],
                      nfs_client_serverreadbytes=nfs['serverreadbytes'], pid_io_delta=pid_io)
        audit = audit_requests[(trial, name)]
        for key in ('actual_generated_ids', 'enqueue_ordinals', 'ttft_seconds', 'request_e2e_seconds',
                    'decoder_invocations', 'logical_counters', 'pipeline_nonpeak_deltas', 'pid_io_delta'):
            require(record[key] == audit[key], 'poststudy request audit differs: ' + key)
        require(record['host_logical_peak_bytes'] == audit['owner_peak_host_charge'], 'poststudy peak audit differs')
        records.append(record)
        previous_end = end
    require(used_lines == [r['line'] for r in all_invocations] and len(timeline) == 102, 'incomplete runtime timeline')
    require(sum(r['completion_tokens'] for r in records) == 90, 'actual per-lifecycle token count differs')
    require(previous_stats == ledger['stats'] == audit_arm['pipeline_stats'], 'lifetime pipeline ledger differs')
    totals = add([r['pipeline_nonpeak_deltas'] for r in records])
    require(all(ledger['stats'][k] == v for k, v in totals.items()), 'request/lifetime read accounting differs')
    logical = add([r['logical_counters'] for r in records])
    require(logical == audit_arm['logical_counters'], 'per-lifecycle logical accounting differs')
    require(ledger['current_charge'] == audit_arm['final_logical_charge']
            and all(v == 0 for v in ledger['current_charge'].values()), 'final logical charge differs')
    measured = [r for r in records if r['measured']]
    require(len(measured) == 5, 'five measured QA per arm required')
    metrics = {k: statistics.median(r[k] for r in measured) for k in ('request_e2e_seconds', 'ttft_seconds')}
    denominator = sum(r['request_e2e_seconds'] for r in measured)
    metrics.update(qualified_goodput_tokens_per_second=sum(r['completion_tokens'] for r in measured) / denominator,
                   measured_slots=5, quality_qualified_tokens=40, client_e2e_denominator_seconds=denominator)
    require(metrics == audit_arm['metrics'], 'poststudy arm metrics differ')
    samples = [json.loads(line) for line in sources[prefix + 'samples.jsonl'].splitlines()]
    samples = [r for r in samples if r['kind'] == 'sample']
    selected_uuid = life['physical_uuid']
    gpu = []
    for sample in samples:
        values = [d['gpu_bytes'] for d in sample['gpu']['devices'] if d['uuid'] == selected_uuid]
        require(len(values) == 1, 'device sample coverage differs')
        gpu.append(values[0])
    maximum = gauge(gpu)['maximum']
    require(maximum == life['resources']['gpu_peak_bytes'][selected_uuid] == audit_arm['peak_device_sample_bytes'],
            'sampled device maximum differs')
    require(len(samples) == audit_arm['memory_sample_count'], 'memory sample count differs')
    require(life['resources']['owned_gpu_attribution'] == 'unknown'
            and life['resources']['exact_peaks'] is False, 'sample scope differs')
    argv = cases[mode]['argv_after_binary']
    device_limit = int(argv[argv.index('--cuda-device-budget-mb') + 1]) * MIB
    require(device_limit == 7 * 2**30 and maximum <= device_limit, 'configured device target differs')
    nfs_phases = {}
    for phase, (before, after) in PHASE_SNAPSHOTS.items():
        delta = nfs_delta(load(before + '-mountstats.json'), load(after + '-mountstats.json'))
        require(delta == life[phase]['deltas'] == audit_arm['nfs_phase_client_deltas'][phase], 'phase NFS delta differs')
        nfs_phases[phase] = delta['serverreadbytes']
    request_nfs_sum = sum(r['nfs_client_serverreadbytes'] for r in records)
    nfs_between_windows = nfs_phases['requests_nfs'] - request_nfs_sum
    require(integer(nfs_between_windows), 'NFS windows exceed enclosing phase')
    require(life['startup_seconds'] == audit_arm['startup_seconds'] and number(life['startup_seconds']), 'startup probe copy differs')
    require(life['startup_logical_counters']['deltas']['checkpoint_read_bytes'] is None, 'unknown startup bytes changed')
    return dict(trial=trial, mode=mode, metrics=metrics, requests=records, timeline=timeline,
                median_request_demand_wait_seconds=statistics.median(r['pipeline_nonpeak_deltas']['demand_wait_ns'] / 1e9 for r in measured),
                measured_demand_wait_seconds=sum(r['pipeline_nonpeak_deltas']['demand_wait_ns'] for r in measured) / 1e9,
                startup_probe_seconds=life['startup_seconds'], logical_counters=logical,
                pipeline_stats=ledger['stats'], final_logical_charge=ledger['current_charge'],
                host_logical_peak_bytes=gauge([r['host_logical_peak_bytes'] for r in records])['maximum'],
                host_logical_limit_bytes=ledger['limits']['max_host_bytes'],
                sampled_device_max_bytes=maximum, configured_device_target_bytes=device_limit,
                device_sample_count=len(samples), nfs_phase_client_serverreadbytes=nfs_phases,
                nfs_request_window_sum_bytes=request_nfs_sum, nfs_between_request_windows_bytes=nfs_between_windows,
                startup_pid_io_delta=None, shutdown_pid_io_delta=None, startup_logical_read_bytes=None,
                physical_nas_backend_bytes=None, owned_process_gpu_peak_bytes=None,
                universal_physical_release=None)
