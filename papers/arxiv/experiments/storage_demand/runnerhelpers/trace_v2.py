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
