"""Bounded stdlib SSE client and actual-worker cancellation evidence, no inference polling."""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import re
import socket
import time
from urllib.parse import urlsplit

ID = re.compile(r"cmpl-ferrule-([1-9][0-9]*)\Z")
LOG_LIMIT = 64 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(lines):
    for line in lines:
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            yield value.get("fields", value)


def complete_lines(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(LOG_LIMIT + 1)
    require(len(raw) <= LOG_LIMIT, "oversized raw worker log")
    # An in-progress write is not a terminal observation. Re-read on the next finite poll.
    return raw[:raw.rfind(b"\n") + 1].decode("utf-8").splitlines()


def stream(url, payload, *, cancel, deadline):
    """Exactly one POST. Close the TCP connection immediately after nonempty content."""
    endpoint = urlsplit(url)
    require(endpoint.scheme == "http" and endpoint.hostname == "127.0.0.1"
            and endpoint.path == "/v1/completions" and not endpoint.query,
            "only the frozen local completion endpoint is supported")
    require(payload.get("stream") is True, "SSE required")
    started = time.monotonic()
    require(deadline > started, "HTTP deadline before dispatch")
    connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=deadline - started)
    response = wire = None
    result = dict(status="incomplete", started_at=started, raw_sse_events=[], text="", usage=None,
                  finish_reason=None, done_at=None, actual_output_tokens=None,
                  client_token_ids=None, content_events_are_token_boundaries=False,
                  content_event_count=0, completion_id=None, adapter_http_report_complete=False,
                  closed_after_content=False, tcp_shutdown=False, closed_at=None)
    data_lines = []
    try:
        connection.request("POST", endpoint.path, body=json.dumps(payload).encode(),
                           headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        wire = connection.sock
        response = connection.getresponse()
        result["http_status"] = response.status
        require(response.status == 200 and response.getheader("Content-Type", "").startswith("text/event-stream"),
                "HTTP status/content type rejected")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("absolute HTTP deadline")
            wire.settimeout(remaining)
            line = response.readline(1 << 20)
            require(line and len(line) < (1 << 20), "SSE EOF/oversized line before terminal")
            line = line.decode("utf-8").rstrip("\r\n")
            if line:
                if line.startswith("data:"):
                    data_lines.append(line[5:].removeprefix(" "))
                continue
            if not data_lines:
                continue
            data, data_lines = "\n".join(data_lines), []
            at = time.monotonic()
            result["raw_sse_events"].append(dict(data=data, at=at))
            if data == "[DONE]":
                result["done_at"] = at
                require(not cancel, "natural completion before controlled content disconnect")
                require(result["finish_reason"] == "length" and isinstance(result["usage"], dict),
                        "missing exact terminal/usage")
                result.update(status="success", adapter_http_report_complete=True,
                              actual_output_tokens=result["usage"].get("completion_tokens"))
                break
            value = json.loads(data)
            require(isinstance(value, dict) and "error" not in value, "SSE error/nonobject")
            completion_id = value.get("id")
            require(isinstance(completion_id, str) and ID.fullmatch(completion_id), "missing/opaque completion identity")
            require(result["completion_id"] in (None, completion_id), "completion identity changed within stream")
            result["completion_id"] = completion_id
            if value.get("usage") is not None:
                require(result["usage"] is None, "duplicate usage event")
                result["usage"] = value["usage"]
            choices = value.get("choices")
            require(isinstance(choices, list) and len(choices) <= 1, "unexpected SSE choices")
            if not choices:
                continue
            choice = choices[0]
            text = choice.get("text", "")
            require(isinstance(text, str), "non-string completion text")
            if choice.get("finish_reason") is not None:
                require(result["finish_reason"] is None, "duplicate finish event")
                result["finish_reason"] = choice["finish_reason"]
            if not text:
                continue
            result["text"] += text
            result["content_event_count"] += 1
            if cancel:
                # Do not wait for validation, observers, disk writes or another SSE event.
                wire.shutdown(socket.SHUT_RDWR)
                result.update(tcp_shutdown=True, closed_after_content=True,
                              closed_at=time.monotonic(), status="controlled_client_disconnect")
                response.close()
                connection.close()
                break
        result["ended_at"] = time.monotonic()
        return result
    finally:
        if response is not None:
            response.close()
        connection.close()


def cancellation_evidence(lines, actual, expected_ids):
    completion_id = actual.get("completion_id")
    match = ID.fullmatch(completion_id or "")
    require(match is not None, "cancel completion identity absent")
    request_id = int(match.group(1))
    events = [row for row in fields(lines) if row.get("request_id") == request_id]
    selected = lambda name: [row for row in events if row.get("event") == name]
    tokens = selected("request_worker_token_enqueued")
    terminal = selected("request_worker_terminal")
    enqueue = selected("request_worker_terminal_enqueue")
    http = selected("request_http_terminal")
    ids = [row.get("actual_token_id") for row in tokens]
    ordinals = [row.get("emission_ordinal") for row in tokens]
    identity = all(row.get("session_id") == request_id for row in events)
    prefix = bool(ids) and len(ids) <= len(expected_ids) and ids == expected_ids[:len(ids)]
    ordinal_ok = ordinals == list(range(1, len(tokens) + 1))
    boundary = all(row.get("delivery_boundary") == "worker_channel_enqueue" for row in tokens)
    submitted = len(terminal) == 1 and terminal[0].get("status") == "cancelled" and all(
        terminal[0].get(key) is expected for key, expected in (
            ("cancellation_requested", True), ("cancellation_submitted", True),
            ("cancellation_failed", False), ("requires_restore_or_shutdown", False)))
    worker = submitted and terminal[0].get("last_emission_ordinal") == len(tokens)
    delivery = (len(enqueue) == 1 and enqueue[0].get("status") == "cancelled"
                and enqueue[0].get("terminal_event_enqueued") is False
                and enqueue[0].get("usage_available") is False
                and enqueue[0].get("last_emission_ordinal") == len(tokens))
    disconnect = (len(http) == 1 and http[0].get("status") == "cancelled"
                  and http[0].get("finish_reason") == "client_disconnect")
    client = (actual.get("status") == "controlled_client_disconnect"
              and actual.get("closed_after_content") is True and actual.get("tcp_shutdown") is True
              and actual.get("content_event_count") == 1 and actual.get("usage") is None
              and actual.get("finish_reason") is None and actual.get("done_at") is None)
    admissions = all(len(selected(name)) == 1 for name in ("request_admitted", "request_http_admitted"))
    accepted = bool(identity and prefix and ordinal_ok and boundary and worker and delivery and disconnect and client and admissions)
    return dict(name="cancel-qa8", actual_http_completion_ids=[completion_id], actual_worker_request_id=request_id,
                actual_generated_ids=ids or None, enqueue_ordinals=ordinals, enqueued_token_count=len(tokens),
                worker_terminal=terminal, worker_terminal_enqueue=enqueue, http_response_terminal=http,
                cancellation_accepted=accepted, protocol_accepted=accepted, metric_eligible=False,
                exact_token_pass=None, prefix_matches_frozen=prefix, client_content_events=actual.get("content_event_count"),
                boundary="actual successful worker enqueue, not client token IDs or content/token alignment",
                quality_status="not_applicable_controlled_censor", terminal_reconciled=accepted,
                natural_finish_race=any(row.get("status") == "finished" for row in terminal))


def wait_cancel(log, actual, expected_ids, *, deadline):
    """Only passive raw-log observation; no HTTP, inference or runtime progress calls."""
    polls = 0
    started = time.monotonic()
    while time.monotonic() < deadline:
        polls += 1
        evidence = cancellation_evidence(complete_lines(log), actual, expected_ids)
        if evidence["worker_terminal"]:
            if evidence["natural_finish_race"] or evidence["worker_terminal"][0].get("status") != "cancelled":
                evidence.update(wait_seconds=time.monotonic() - started, polls=polls, wait_status="non_cancel_terminal")
                return evidence
            if evidence["worker_terminal_enqueue"] and evidence["http_response_terminal"]:
                evidence.update(wait_seconds=time.monotonic() - started, polls=polls, wait_status="terminal_observed")
                return evidence
        time.sleep(min(.02, max(0, deadline - time.monotonic())))
    evidence = cancellation_evidence(complete_lines(log), actual, expected_ids)
    evidence.update(wait_seconds=time.monotonic() - started, polls=polls, wait_status="deadline")
    evidence["cancellation_accepted"] = evidence["protocol_accepted"] = False
    return evidence
