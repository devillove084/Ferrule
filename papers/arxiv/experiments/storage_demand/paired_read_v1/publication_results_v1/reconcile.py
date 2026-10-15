"""Independent CPU checks of HTTP/token, invocation, NFS and PID evidence."""
from __future__ import annotations

import math

from common import CLI_SHA256, decode, pick, require, sha

LOGICAL = ('checkpoint_read_calls checkpoint_read_bytes checkpoint_source_checks checkpoint_path_metadata_calls '
           'checkpoint_handle_metadata_calls h2d_copies h2d_bytes artifact_uploads artifact_upload_bytes '
           'expert_hits expert_uploads expert_evictions').split()
NFS = ('normalreadbytes normalwritebytes directreadbytes directwritebytes serverreadbytes serverwritebytes '
       'readpages writepages').split() + ['read_rpc_' + str(i) for i in range(8)]
IO = 'rchar wchar syscr syscw read_bytes write_bytes cancelled_write_bytes'.split()


def integer(value, lower=0):
    return type(value) is int and value >= lower


def close(actual, expected, label):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)


def same_birth(left, right):
    require(all(integer(left[k], 1) and left[k] == right[k] for k in ('pid', 'birth_ticks', 'ppid')),
            'PID/birth/parent changed')


def summed(rows):
    require(rows and all(set(r) == set(rows[0]) for r in rows), 'counter key sets differ')
    require(all(integer(v) for row in rows for v in row.values()), 'missing/null/negative counter')
    return {key: sum(row[key] for row in rows) for key in rows[0]}


def delta(before, after, keys):
    require(set(before) == set(after) == set(keys), 'counter coverage keys differ')
    require(all(integer(before[k]) and integer(after[k]) and after[k] >= before[k] for k in keys),
            'missing/null/negative/reset counter')
    return {k: after[k] - before[k] for k in keys}


def read_ledger(value, mode):
    limits = dict(schedule=mode, max_host_bytes=64 << 20, max_metadata_bytes=8 << 20, worker_stack_bytes=2 << 20)
    require(value['enabled'] is True and value['requested_mode'] == value['effective_mode'] == mode
            and value['limits'] == limits and value['worker_read_observed'] is True, 'read mode/limits/effectiveness')
    require(value['status'] in ('closed', 'closed_retained') and value['current_read'] is None
            and value['pending_read'] is None, 'read ledger not closed')
    stats, charge = value['stats'], value['current_charge']
    require(all(integer(v) for v in stats.values()) and all(integer(v) for v in charge.values()), 'null/invalid ledger counter')
    for key in ('unknown_reads', 'unknown_bytes', 'unobserved_terminal_reads', 'unobserved_reader_intervals',
                'join_failed_sequences', 'join_pending_sequences'):
        require(stats[key] == 0, 'unknown read custody')
    for key in ('unknown_packed_bytes', 'unknown_stack_bytes', 'unknown_read_slots'):
        require(charge[key] == 0, 'unknown host charge')
    for current, cap in (('packed_bytes', 27 << 20), ('artifact_bytes', 27 << 20), ('payload_bytes', 54 << 20),
                         ('metadata_bytes', 8 << 20), ('requested_stack_bytes', 2 << 20), ('host_charged_bytes', 64 << 20)):
        require(0 <= charge[current] <= stats['peak_' + current] <= cap, 'read host bound violated: ' + current)
    require(charge['reserved_read_slots'] <= 2, 'read slot bound')
    require(charge['payload_bytes'] == charge['packed_bytes'] + charge['artifact_bytes']
            and charge['host_charged_bytes'] == charge['payload_bytes'] + charge['metadata_bytes'] + charge['requested_stack_bytes'],
            'current host charge accounting')
    for key in ('submitted_reads', 'completed_reads', 'verified_reads', 'io_started_reads', 'proved_read_bytes'):
        require(stats[key] > 0, 'no actual positive worker read')
    require(stats['sequences_started'] == stats['sequences_finished'], 'sequence closure mismatch')
    receipt = value['last_receipt']
    require(receipt['outcome'] == 'source_verified' and receipt['requested_bytes'] == receipt['io_bytes'] > 0,
            'last receipt verification mismatch')
    times = [receipt[k] for k in ('submitted_ns', 'worker_started_ns', 'io_started_ns', 'io_finished_ns',
                                 'source_verified_ns', 'completed_ns', 'taken_ns')]
    require(all(integer(t) for t in times) and times == sorted(times), 'last receipt timing disorder')
    return stats


def invocations(request, events, mode, previous, transaction):
    selected = [e for e in events if request['log_start_byte'] <= e['source_byte_start']
                and e['source_byte_end'] <= request['log_end_byte']
                and e['record']['fields']['event'] == 'mixed_decoder_invocation']
    rows = request['logical_counters']['rows']
    require(len(rows) == len(selected) == request['max_tokens'] + 1, 'missing/duplicate decoder invocation')
    require(request['logical_counters']['valid'] is True, 'logical counter validity')
    deltas, raw_fields = [], []
    for row, event in zip(rows, selected):
        fields = event['record']['fields']
        require(row['raw'] == event['record'] and row['fields'] == fields, 'log/window invocation differs from observed receipt')
        transaction += 1
        require(fields['transaction'] == transaction and fields['status'] == 'completed'
                and fields['quarantined'] is False and fields['counters_available'] is True
                and fields['counter_reset_detected'] is False and fields['expert_read_counter_reset_detected'] is False,
                'invocation failed/quarantined/reset/out of order')
        ledger = decode(fields['expert_read_pipeline'])
        stats = read_ledger(ledger, mode)
        changes = decode(fields['expert_read_pipeline_delta'])
        require(set(stats) == set(changes), 'pipeline delta key coverage')
        baseline = previous if previous is not None else {k: 0 for k in stats}
        for key, value in changes.items():
            require(integer(value), 'null pipeline delta')
            if key.startswith('peak_'):
                require(stats[key] == value and stats[key] >= baseline[key], 'peak is not cumulative/high-water')
            else:
                require(stats[key] >= baseline[key] and stats[key] - baseline[key] == value,
                        'pipeline counter reset/nonpeak delta does not reconcile')
        require(fields['rows'] == fields['prefill_rows'] + fields['decode_rows'], 'decoder row accounting')
        deltas.append({k: v for k, v in changes.items() if not k.startswith('peak_')})
        previous = stats
        raw_fields.append(fields)
    logical = summed([{k: row[k] for k in LOGICAL} for row in raw_fields])
    require(logical == request['logical_counters']['deltas'], 'logical counter sum differs')
    require(request['logical_counters']['cache_miss_count'] is None and request['logical_counters']['physical_nas_bytes'] is None,
            'unsupported cache-miss/backend inference')
    pipeline = summed(deltas)
    require(pipeline['reader_interval_bytes'] == logical['checkpoint_read_bytes']
            and pipeline['reader_interval_calls'] == logical['checkpoint_read_calls']
            and pipeline['proved_read_bytes'] == logical['artifact_upload_bytes'], 'logical/read/source cross-ledger mismatch')
    return dict(decoder_invocations=len(rows), rows=sum(r['rows'] for r in raw_fields),
                prefill_rows=sum(r['prefill_rows'] for r in raw_fields), decode_rows=sum(r['decode_rows'] for r in raw_fields),
                logical_counters=logical, pipeline_nonpeak_deltas=pipeline,
                owner_peak_host_charge=max(decode(r['expert_read_pipeline'])['stats']['peak_host_charged_bytes'] for r in raw_fields)), previous, transaction


def client_tokens(request, schedule, terminal, http, issued, events, oracles):
    for key in ('index', 'name', 'phase', 'population', 'measured', 'prompt', 'input_tokens', 'max_tokens', 'reference_id', 'reference', 'payload'):
        require(request[key] == schedule[key] == issued[key], 'frozen request schedule/oracle changed: ' + key)
    require(issued['issued'] is True and issued['status'] == 'censored', 'dispatch journal missing; not a final failure')
    actual, trace = request['actual'], request['trace']
    require(actual == terminal, 'terminal copy differs from observation')
    require(len(http['runs']) == 1 and len(http['runs'][0]['requests']) == 1 and http['warmups'] == [], 'extra HTTP requests')
    client = http['runs'][0]['requests'][0]
    require({k: v for k, v in actual.items() if not k.startswith('adapter_')} == client, 'HTTP report/terminal mismatch')
    require(http['finalized'] is True and http['errors'] == [] and http['aggregate']['success'] == 1, 'HTTP report incomplete')
    require(request['status'] == actual['status'] == 'success' and request['issued'] is True
            and request['quality_passed'] is True and request['metric_eligible'] is True
            and request['duplicate_lifecycle_request_id'] is False, 'unqualified request')
    require(actual['http_status'] == 200 and actual['content_type'] == 'text/event-stream'
            and actual['right_censored'] is False and actual['error'] is None and actual['after_deadline_events'] == []
            and actual['adapter_http_report_complete'] is True and actual['adapter_http_report_errors'] == [], 'HTTP terminal missing/error')
    require(actual['request_payload'] == request['payload'] and actual['prompt_sha256'] == sha(request['prompt'].encode()), 'client prompt changed')
    n, count = request['input_tokens'], request['max_tokens']
    usage = dict(prompt_tokens=n, completion_tokens=count, total_tokens=n + count)
    require(actual['usage'] == usage and actual['actual_input_tokens'] == n and actual['actual_output_tokens'] == count
            and actual['finish_reason'] == 'length', 'usage/length mismatch')
    contents, content_times, ids, usages, finishes, done = [], [], set(), [], [], []
    times = []
    for index, event in enumerate(actual['raw_sse_events']):
        times.append(event['at'])
        if event['data'] == '[DONE]':
            done.append((index, event['at']))
            continue
        item = decode(event['data'])
        ids.add(item['id'])
        if 'usage' in item:
            usages.append((event['at'], item['usage']))
        for choice in item['choices']:
            require(choice['index'] == 0, 'unexpected completion choice')
            if choice['text']:
                contents.append(choice['text'])
                content_times.append(event['at'])
            if choice['finish_reason'] is not None:
                finishes.append((event['at'], choice['finish_reason']))
    require(times == sorted(times) and times and actual['sent_at'] <= times[0] <= times[-1] <= actual['ended_at'] <= actual['deadline_at'], 'client timing ordering/deadline')
    require(done == [(len(times) - 1, actual['done_at'])] and usages == [(actual['usage_at'], usage)]
            and finishes == [(actual['finish_at'], 'length')], 'SSE DONE/usage/finish reconciliation')
    require(''.join(contents) == actual['text'] and content_times == actual['content_event_times'], 'client content reconstruction')
    require(actual['finish_at'] <= actual['usage_at'] <= actual['done_at'], 'SSE terminal ordering')
    close(actual['ttft_seconds'], content_times[0] - actual['sent_at'], 'TTFT is not first nonempty SSE content minus send')
    close(actual['request_e2e_seconds'], actual['ended_at'] - actual['sent_at'], 'E2E is not terminal minus send')
    require(0 < actual['ttft_seconds'] <= actual['request_e2e_seconds'], 'invalid request time')
    request_id = request['index'] + 1
    require(ids == {'cmpl-ferrule-' + str(request_id)} == set(trace['actual_http_completion_ids'])
            and trace['actual_worker_request_id'] == request_id, 'HTTP/worker ID join failed')
    records = [e for e in events if e['record']['fields'].get('request_id') == request_id]
    require(all(request['log_start_byte'] <= e['source_byte_start'] and e['source_byte_end'] <= request['log_end_byte'] for e in records),
            'worker record outside serial request log window')
    by_event = {}
    for e in records:
        f = e['record']['fields']
        by_event.setdefault(f['event'], []).append(f)
    require(set(by_event) == {'request_admitted', 'request_http_admitted', 'request_worker_token_enqueued',
                             'request_worker_terminal', 'request_worker_terminal_enqueue', 'request_http_terminal'}, 'request event coverage')
    for key in set(by_event) - {'request_worker_token_enqueued'}:
        require(len(by_event[key]) == 1, 'duplicate request terminal/admission')
    admitted = by_event['request_admitted'][0]
    require(admitted['prompt_tokens'] == n and admitted['max_new_tokens'] == count, 'worker admission differs')
    enqueued = by_event['request_worker_token_enqueued']
    ordinals = [f['emission_ordinal'] for f in enqueued]
    tokens = [f['actual_token_id'] for f in enqueued]
    require(ordinals == trace['enqueue_ordinals'] == list(range(1, count + 1)), 'wrong/duplicate/missing enqueue ordinals')
    require(tokens == trace['actual_generated_ids'] and all(integer(t) for t in tokens), 'actual token IDs differ')
    require(all(f['delivery_boundary'] == 'worker_channel_enqueue' and f['session_id'] == request_id for f in enqueued), 'enqueue boundary/session mismatch')
    worker = by_event['request_worker_terminal_enqueue'][0]
    worker_done = by_event['request_worker_terminal'][0]
    response = by_event['request_http_terminal'][0]
    require(trace['worker_terminal_enqueue'] == [worker] and trace['http_response_terminal'] == [response], 'raw terminal trace mismatch')
    require(worker['terminal_event_enqueued'] is True and worker['usage_available'] is True
            and worker['last_emission_ordinal'] == count and pick(worker, 'prompt_tokens completion_tokens total_tokens') == usage,
            'terminal enqueue count/usage')
    require(worker_done['emitted_token_events'] == worker_done['last_emission_ordinal'] == count
            and worker_done['tokens_reconcile'] == 'Some(true)' and worker_done['cancellation_requested'] is False,
            'worker terminal incomplete/cancelled')
    require(all(f['status'] == 'finished' and f['finish_reason'] == 'max_tokens' for f in (worker, worker_done, response))
            and response['token_events'] == count, 'worker/HTTP terminal mismatch')
    require(trace['binding_status'] == 'bound' and trace['terminal_reconciled'] is True and trace['trace_valid'] is True
            and trace['duplicate_http_binding'] is False and trace['opaque_http_id'] is False, 'unbound/duplicate trace')
    require(trace['client_consumption']['content_events_are_token_boundaries'] is False
            and actual['output_token_ids'] is None and actual['token_itl_seconds'] is None,
            'SSE events incorrectly treated as exact token boundaries')
    if request['reference_id'] is None:
        require(count == 2 and trace['exact_token_pass'] is None, 'smoke has no exact frozen oracle')
    else:
        reference = oracles[request['reference_id']]
        require(tokens == reference['reference_ids'] and actual['text'] == reference['reference_text']
                and trace['exact_token_pass'] is True, 'actual frozen token/text oracle mismatch')
    return tokens


def pid_io(value, root, actual):
    before, after = value['before'], value['after']
    for snapshot in (before, after):
        same_birth(snapshot['root_identity'], root)
        parsed = {key: int(number) for key, number in (line.split(': ') for line in snapshot['raw'].splitlines())}
        require(parsed == snapshot['counters'], 'proc IO raw/counter mismatch')
        require(snapshot['physical_nfs_bytes'] is None, 'proc IO is not NFS bytes')
        require(snapshot['acquisition_begin']['monotonic'] <= snapshot['acquisition_end']['monotonic'], 'PID snapshot ordering')
    require(before['acquisition_end']['monotonic'] <= actual['sent_at'] <= actual['ended_at'] <= after['acquisition_begin']['monotonic'],
            'PID IO does not bracket request')
    return delta(before['counters'], after['counters'], IO)


def nfs_delta(before, after, reported):
    for snapshot in (before, after):
        require(snapshot['acquisition_status'] == 'captured' and snapshot['shared_client_traffic'] is True
                and snapshot['owned_server_physical_bytes'] is None and snapshot['physical_nas_backend_bytes'] is None,
                'NFS observation unavailable/misattributed')
        require(snapshot['selected_mount']['errors'] == [], 'NFS parse errors')
    require(before['mount_identity_alias'] == after['mount_identity_alias'], 'NFS mount identity changed')
    require(before['selected_mount']['fstype'] == after['selected_mount']['fstype'] == 'nfs4'
            and before['selected_mount']['statvers'] == after['selected_mount']['statvers'], 'NFS layout changed')
    require(before['selected_mount']['age_seconds'] <= after['selected_mount']['age_seconds']
            and before['acquisition_end']['monotonic'] <= after['acquisition_begin']['monotonic'], 'NFS counter reset/time disorder')
    changes = delta(before['selected_mount']['counters'], after['selected_mount']['counters'], NFS)
    require(reported['status'] == 'shared_client_delta' and reported['deltas'] == changes
            and reported['reasons'] == [] and reported['shared_client_traffic'] is True
            and reported['physical_nas_backend_bytes'] is None and reported['owned_server_physical_bytes'] is None, 'NFS delta/reported scope mismatch')
    require(reported['before_window'] == before['acquisition_begin'] and reported['after_window'] == after['acquisition_end'], 'NFS timing window changed')
    return changes


def map_evidence(value, trial, phase, root, runtime):
    s, p, v = value['snapshot'], value['policy'], value['validation']
    require(v['status'] == 'passed' and v['required_pin_check'] is True and v['errors'] == []
            and v['missing_required_paths'] == [] and v['deleted_library_paths'] == [], 'library map validation failed')
    require(len(p['required']) == len(set(p['required'])) == v['observed_library_count'] == 12
            and p['pins'] == v['libraries'] == runtime and set(p['required']) == set(runtime), 'required library 12/12 hash coverage')
    require(s['executable_sha256'] == s['expected_executable_sha256'] == p['executable_sha256'] == CLI_SHA256, 'mapped CLI changed')
    same_birth(s['root_identity'], root)
    same_birth(s['before'], s['after'])
    same_birth(s['after'], s['after_executable'])
    same_birth(s['parent_before'], s['parent_after'])
    same_birth(v['root_identity'], root)
    require(s['raw_maps_sha256'] == v['raw_maps_sha256'] == value['original_raw_map']['original_sha256'], 'raw map original hash pointer differs')
    require(s['acquisition_begin']['monotonic'] <= s['raw_saved']['monotonic'] <= s['acquisition_end']['monotonic'] <= v['captured']['monotonic'],
            'map not saved before validation')
    require(v['live_mapped_memory_sha256'] is None and v['overlay_backing_identity'] == 'unobserved', 'unobserved map identity upgraded')
    return dict(trial=trial, phase=phase, status='passed', required=12, found=12,
                raw_maps_sha256=s['raw_maps_sha256'], pid_birth_bound=True,
                ignored_deleted_nonlibraries=v['ignored_deleted_nonlibraries_count'],
                overlay_backing_identity='unobserved', mapped_memory_hash='unobserved')
