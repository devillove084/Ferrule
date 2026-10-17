"""Actual token enqueue/HTTP ID binding, independent of client content consumption."""
from __future__ import annotations

import json
import re

from .evidence_v2 import finite, integer, require

COMPLETION = re.compile(r'^cmpl-ferrule-(0|[1-9][0-9]*)$')


def fields(event):
    if not isinstance(event, dict):
        return {}
    return event['fields'] if isinstance(event.get('fields'), dict) else event


def json_events(lines):
    for number, raw in enumerate(lines, 1):
        try:
            event = json.loads(raw)
        except (TypeError, ValueError, UnicodeError):
            continue
        if isinstance(event, dict):
            yield number, event


def parse_token_trace(lines):
    rows = []
    for number, event in json_events(lines):
        f = fields(event)
        if f.get('event') != 'request_worker_token_enqueued':
            continue
        require(integer(f.get('request_id'), 1) and integer(f.get('session_id'), 1),
                'token trace requires actual request/session IDs')
        require(integer(f.get('emission_ordinal'), 1) and integer(f.get('actual_token_id')),
                'invalid actual token or enqueue ordinal')
        require(f.get('delivery_boundary') == 'worker_channel_enqueue', 'wrong enqueue boundary')
        rows.append(dict(line=number, raw=event, **f))
    return rows


def http_consumption(request):
    actual = request.get('actual', request)
    ids = []
    content = []
    errors = []
    for index, event in enumerate(actual.get('raw_sse_events', [])):
        data = event.get('data')
        if data == '[DONE]':
            continue
        try:
            value = json.loads(data)
        except (TypeError, ValueError):
            errors.append('invalid SSE JSON')
            continue
        if not isinstance(value, dict):
            errors.append('nonobject SSE JSON')
            continue
        if isinstance(value.get('id'), str) and value['id']:
            ids.append(value['id'])
        else:
            errors.append('missing HTTP completion ID')
        choices = value.get('choices')
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            choice = choices[0]
            text = choice.get('text')
            if isinstance(choice.get('delta'), dict):
                text = choice['delta'].get('content')
            if isinstance(text, str) and text:
                content.append(dict(content_event_ordinal=len(content) + 1, sse_event_index=index,
                                    at=event.get('at'), content_bytes=len(text.encode())))
    return dict(completion_ids=sorted(set(ids)), errors=errors, content_events=content,
                content_events_are_token_boundaries=False,
                client_status=actual.get('status'), done_at=actual.get('done_at'),
                usage=actual.get('usage'))


def join_token_trace_http(trace_rows, requests, server_lines=()):
    """No suffix guessing for opaque IDs; retain them with an unbound status."""
    result = []
    used_ids = set()
    terminals = [fields(event) for _, event in json_events(server_lines) if fields(event).get('event') in ('request_worker_terminal_enqueue', 'request_http_terminal')]
    for request in requests:
        actual = request.get('actual', request)
        client = http_consumption(request)
        ids = client['completion_ids']
        match = COMPLETION.fullmatch(ids[0]) if len(ids) == 1 else None
        request_id = int(match.group(1)) if match else None
        rows = [row for row in trace_rows if row.get('request_id') == request_id] if match else []
        ordinals = [row.get('emission_ordinal') for row in rows]
        count = actual.get('actual_output_tokens')
        duplicate = request_id in used_ids if match else False
        if match:
            used_ids.add(request_id)
        valid = bool(rows) and not duplicate and not client['errors'] and len(ids) == 1
        valid = valid and all(row.get('session_id') == request_id for row in rows)
        valid = valid and ordinals == list(range(1, len(rows) + 1))
        valid = valid and integer(count, 1) and len(rows) == count
        valid = valid and actual.get('status') == 'success' and finite(actual.get('done_at'))
        worker_terminal = [event for event in terminals if event.get('event') == 'request_worker_terminal_enqueue' and event.get('request_id') == request_id]
        http_terminal = [event for event in terminals if event.get('event') == 'request_http_terminal' and event.get('request_id') == request_id]
        terminal_reconciled = len(worker_terminal) == len(http_terminal) == 1 and worker_terminal[0].get('terminal_event_enqueued') is True and worker_terminal[0].get('status') == http_terminal[0].get('status') == 'finished' and worker_terminal[0].get('last_emission_ordinal') == http_terminal[0].get('token_events') == count
        ids_actual = [row['actual_token_id'] for row in rows]
        reference = request.get('reference')
        reference_ids = reference.get('reference_ids') if isinstance(reference, dict) else None
        exact = (ids_actual == reference_ids) if valid and isinstance(reference_ids, list) else None
        result.append(dict(
            name=request.get('name'), actual_http_completion_ids=ids,
            actual_worker_request_id=request_id,
            binding_status='bound' if valid else 'unbound',
            opaque_http_id=bool(ids) and match is None,
            duplicate_http_binding=duplicate,
            enqueue_ordinals=ordinals, actual_generated_ids=ids_actual if rows else None,
            trace_valid=bool(valid), exact_token_pass=exact, client_consumption=client,
            worker_terminal_enqueue=worker_terminal, http_response_terminal=http_terminal,
            terminal_reconciled=terminal_reconciled, metric_eligible=bool(valid and terminal_reconciled and (reference is None or exact is True)),
            boundary='successful worker enqueue; neither HTTP consumption nor content/token alignment',
        ))
    return result


def logical_counter_window(lines):
    """Serialized log-byte-window deltas, never transaction-to-request identity."""
    from .evidence_v2 import COUNTERS
    rows = [dict(line=number, raw=event, fields=fields(event))
            for number, event in json_events(lines)
            if fields(event).get('event') == 'mixed_decoder_invocation']
    valid = bool(rows) and all(
        row['fields'].get('counters_available') is True
        and row['fields'].get('counter_reset_detected') is False
        and row['fields'].get('status') == 'completed'
        and row['fields'].get('device') == 0
        and row['fields'].get('quarantined') is False
        for row in rows)
    deltas = {key: sum(row['fields'][key] for row in rows)
              if valid and all(integer(row['fields'].get(key)) for row in rows) else None
              for key in COUNTERS}
    return dict(rows=rows, valid=valid and all(v is not None for v in deltas.values()),
                deltas=deltas, cache_miss_count=None, physical_nas_bytes=None,
                binding='serial request log-byte window, not transaction or worker ID',
                scope='logical checkpoint reader and facade transfer, not physical traffic')


TRANSFER_WORK = ('rows', 'sequences', 'prefill_rows', 'decode_rows',
    'expert_uploads', 'expert_hits', 'expert_evictions', 'h2d_copies', 'h2d_bytes',
    'artifact_uploads', 'artifact_upload_bytes', 'checkpoint_read_bytes')


def transfer_counter_window(lines, options, *, previous_read=None, initial_read=None, producer_contract=None):
    """Validate raw health; absent boundary evidence stays explicitly unreconciled.

    Supply initial_read once, then previous_read for later serial windows. Without
    either, valid describes local health only, NOT owner-lifetime continuity.
    """
    from paired_read_v1 import quality
    from .evidence_v2 import COUNTERS
    value = logical_counter_window(lines)
    value.update(read_continuity_valid=False, read_continuity='unobserved_initial_baseline',
                 final_read_endpoint=None)
    try:
        require(value['valid'], 'missing/invalid invocation counters')
        require(previous_read is None or initial_read is None, 'ambiguous read boundary baseline')
        previous = initial_read if initial_read is not None else previous_read
        bound = previous is not None
        if bound:
            quality.read_observation_v3(previous, options, producer_contract=producer_contract)
            require(previous['status'] in ('configured', 'idle_retained'), 'read baseline is not live/settled')
        for row in value['rows']:
            f = row['fields']
            require(all(quality._v3_integer(f.get(k)) for k in COUNTERS + TRANSFER_WORK),
                    'missing/overflowed typed work counter')
            require(f.get('expert_cache_counters_overflowed') is False
                and f.get('expert_read_counter_reset_detected') is False,
                'cache overflow/read counter reset or missing health')
            for key in ('expert_pending_upload_bytes', 'expert_unknown_quarantine_bytes',
                        'facade_stream_sync_failures'):
                require(type(f.get(key)) is int and f[key] == 0, 'pending/Unknown/failure: ' + key)
            require(all(integer(f.get(k)) for k in TRANSFER_WORK + ('expert_resident_entries',
                'expert_resident_bytes', 'expert_scratch_bytes', 'expert_owner_peak_bytes')),
                'missing typed work/cache gauge')
            require(f['expert_resident_entries'] <= 64 and f['expert_scratch_bytes'] <= 8 << 20
                and f['expert_resident_bytes'] + f['expert_scratch_bytes']
                    <= f['expert_owner_peak_bytes'] <= 320 << 20, 'cache budget/conservation')
            read = json.loads(f['expert_read_pipeline'])
            quality.read_observation_v3(read, options, stage='idle_retained', producer_contract=producer_contract)
            delta = json.loads(f['expert_read_pipeline_delta'])
            require(isinstance(delta, dict) and set(delta) == set(quality.V3_COUNTERS + quality.V3_PEAKS)
                and all(quality._v3_integer(v) for v in delta.values()), 'missing typed read delta')
            for key in quality.V3_COUNTERS:
                require(delta[key] <= read['stats'][key] and (previous is None or
                    read['stats'][key] - previous['stats'][key] == delta[key]),
                    'read lifetime reset/delta mismatch: ' + key)
            require(all(delta[k] == read['stats'][k] for k in quality.V3_PEAKS), 'read peak delta mismatch')
            if previous is not None:
                require(all(read['stats'][k] >= previous['stats'][k] for k in quality.V3_PEAKS),
                        'read lifetime peak reset')
            previous = read
            row['read_endpoint'] = read
            row['read_delta'] = delta
        value.update(read_continuity_valid=bound,
            read_continuity='reconciled' if bound else 'unobserved_initial_baseline',
            final_read_endpoint=previous, transfer_health_valid=True, health_error=None,
            grouped_backend_call_count=None,
            dispatch_evidence='positive cache uploads plus reviewed frozen-source policy dispatch; no direct grouped-call counter')
    except (ValueError, KeyError, TypeError) as error:
        value.update(valid=False, transfer_health_valid=False, health_error=str(error),
                     read_continuity_valid=False, read_continuity='invalid',
                     final_read_endpoint=None, grouped_backend_call_count=None)
    return value


def transfer_read_continuity(windows, options, *, initial_read, producer_contract=None):
    """Revalidate ordered raw windows from an observed baseline; never heal a bad window.

    A gap with unaccounted inter-request work fails delta equality. No synthetic
    zero baseline or final healthy endpoint can replace missing/failed evidence.
    """
    from paired_read_v1 import quality
    checked = []
    try:
        require(initial_read is not None, 'initial read baseline unobserved')
        quality.read_observation_v3(initial_read, options, producer_contract=producer_contract)
        require(initial_read['status'] in ('configured', 'idle_retained'), 'initial read baseline is not live/settled')
        require(isinstance(windows, (list, tuple)) and windows, 'read invocation windows missing')
        previous = initial_read
        for index, window in enumerate(windows):
            require(isinstance(window, dict) and window.get('valid') is True
                and window.get('transfer_health_valid') is True, 'prior read/cache window invalid or unknown')
            lines = [json.dumps(dict(fields=row['fields'])) for row in window['rows']]
            current = transfer_counter_window(lines, options, previous_read=previous, producer_contract=producer_contract)
            checked.append(current)
            require(current['valid'] and current['read_continuity_valid'],
                    'read boundary/window ' + str(index) + ': ' + str(current.get('health_error')))
            previous = current['final_read_endpoint']
        return dict(can_continue=True, read_continuity='reconciled', final_read_endpoint=previous,
                    checked_windows=checked, stop_reason=None)
    except (ValueError, KeyError, TypeError) as error:
        return dict(can_continue=False, read_continuity='unobserved_or_invalid', final_read_endpoint=None,
                    checked_windows=checked, stop_reason=str(error))


def transfer_startup(lines, policy, cli_sha256):
    expected = {'synchronous': 'Synchronous', 'grouped-synchronous': 'GroupedSynchronous'}[policy]
    rows = [fields(event) for _, event in json_events(lines) if fields(event).get('message')
        == 'configured resident expert transfers before engine composition']
    require(len(rows) == 1 and rows[0].get('requested_expert_transfer_policy') == 'Some(' + expected + ')'
        and rows[0].get('expert_transfer_policy') == expected, 'startup requested/observed transfer policy mismatch')
    require(re.fullmatch(r'[0-9a-f]{64}', cli_sha256 or ''), 'frozen CLI hash missing')
    return dict(requested=policy, effective=policy, observed_getter=rows[0], frozen_cli_sha256=cli_sha256,
                backend_grouped_dispatch_proven_by_getter=False)
