"""Read-only, standard-library analysis of one Ferrule server log/process.

No CUDA imports, subprocesses, runner integration, or writes. Unknown quantities
remain None (JSON null), including aggregates with incomplete coverage.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

DELTA_FIELDS = (
    "facade_kernel_launches", "h2d_copies", "h2d_bytes", "d2h_copies", "d2h_bytes",
    "artifact_uploads", "artifact_upload_bytes", "allocation_attempts", "allocation_bytes",
    "facade_stream_syncs", "facade_stream_sync_failures", "expert_hits", "expert_uploads",
    "expert_evictions", "checkpoint_read_calls", "checkpoint_read_bytes",
    "checkpoint_source_checks", "checkpoint_path_metadata_calls", "checkpoint_handle_metadata_calls",
)
PROOF_PREFIXES = (
    "consumer_event_record_sync", "semantic_compute_sync", "semantic_upload_sync",
)
PROOF_FIELDS = tuple(f"{prefix}_{suffix}" for prefix in PROOF_PREFIXES
                     for suffix in ("calls", "failures", "host_us"))
GAUGE_FIELDS = (
    "expert_resident_entries", "expert_resident_bytes", "expert_pending_upload_bytes",
    "expert_scratch_bytes", "expert_owner_peak_bytes", "expert_unknown_quarantine_bytes",
)
INVOCATION_FIELDS = ("host_invocation_us", *DELTA_FIELDS, *PROOF_FIELDS, "allocation_host_us")
WORKER_FIELDS = (
    "worker_request_us", "emitted_token_events", "worker_first_emission_elapsed_us",
    "worker_last_emission_elapsed_us", "worker_emission_interval_count",
    "worker_emission_interval_sum_us", "worker_emission_interval_min_us",
    "worker_emission_interval_max_us",
)
REQUEST_EVENTS = {
    "request_admitted", "request_http_admitted", "request_worker_terminal", "request_http_terminal",
}
EXPECTED_SCOPE = "decoder_validate_through_finish_or_failure_not_device_kernel_or_publication"


def uint(value):
    """Logger counters are unsigned integers, not booleans or numeric strings."""
    return value if type(value) is int and 0 <= value <= 2**64 - 1 else None


def finite(value):
    if type(value) not in (int, float):
        return None
    try:
        return value if math.isfinite(value) and value >= 0 else None
    except OverflowError:
        return None


def text(value):
    return value if isinstance(value, str) else None


def boolean(value):
    return value if type(value) is bool else None


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return finite(numerator / denominator)


def aggregate(values):
    """Never silently sum a partially observed quantity or call an empty set zero."""
    known = [v for v in values if v is not None]
    complete = bool(values) and len(known) == len(values)
    observed_sum = finite(sum(known)) if known else None
    return {"observations": len(values), "known": len(known),
            "unknown": len(values) - len(known),
            "total": observed_sum if complete else None,
            "observed_sum": observed_sum}


def phase(fields):
    rows, prefill, decode = (uint(fields.get(k)) for k in ("rows", "prefill_rows", "decode_rows"))
    if rows is None or rows == 0 or prefill is None or decode is None or prefill + decode != rows:
        return "unknown"
    return "mixed" if prefill and decode else "prefill" if prefill else "decode"


def invocation(fields, line):
    counters_ok = fields.get("counters_available") is True and fields.get("counter_reset_detected") is False
    scope_ok = fields.get("timing_scope") == EXPECTED_SCOPE
    values = {k: uint(fields.get(k)) for k in INVOCATION_FIELDS}
    if not counters_ok:
        values.update(dict.fromkeys(DELTA_FIELDS))
    # The logger has no allocation duration measurement, regardless of input.
    values["allocation_host_us"] = None
    row_count = uint(fields.get("rows"))
    proof_us = [values[f"{p}_host_us"] for p in PROOF_PREFIXES]
    proof_total = sum(proof_us) if all(v is not None for v in proof_us) else None
    return {
        "line": line, "transaction": uint(fields.get("transaction")),
        "device": uint(fields.get("device")), "status": text(fields.get("status")),
        "quarantined": fields.get("quarantined") if type(fields.get("quarantined")) is bool else None,
        "phase": phase(fields), "rows": row_count, "sequences": uint(fields.get("sequences")),
        "prefill_rows": uint(fields.get("prefill_rows")), "decode_rows": uint(fields.get("decode_rows")),
        "scope_verified": scope_ok, "counter_deltas_valid": counters_ok,
        "values": values, "gauges": {k: uint(fields.get(k)) for k in GAUGE_FIELDS},
        "per_row": {k: ratio(v, row_count) if phase(fields) != "unknown" else None
                    for k, v in values.items()},
        "observed_proof_host_us": proof_total,
        "observed_proof_host_fraction": ratio(proof_total, values["host_invocation_us"]),
        "allocation_host_us": None,
    }


def worker(fields, line):
    values = {k: uint(fields.get(k)) for k in WORKER_FIELDS}
    enabled = boolean(fields.get("emission_timing_enabled"))
    if not enabled:
        for name in WORKER_FIELDS[2:]:
            values[name] = None
    n = values["emitted_token_events"]
    count = values["worker_emission_interval_count"]
    first, last = (values[f"worker_{part}_emission_elapsed_us"] for part in ("first", "last"))
    total = values["worker_emission_interval_sum_us"]
    valid = enabled is True and n is not None and count == max(0, n - 1)
    if n == 0:
        valid &= first is None and last is None and total == 0
    elif n is not None:
        valid &= first is not None and last is not None and last >= first
        if first is not None and last is not None and total is not None:
            # Durations are rounded only on export, so endpoint subtraction may
            # differ from the exported exact-duration sum by one microsecond.
            valid &= abs((last - first) - total) <= 1
        else:
            valid = False
    low, high = (values[f"worker_emission_interval_{part}_us"] for part in ("min", "max"))
    if count is not None and count > 0:
        valid &= low is not None and high is not None and low <= high
    else:
        valid &= low is None and high is None
    return {
        "line": line, "request_id": uint(fields.get("request_id")),
        "session_id": uint(fields.get("session_id")), "status": text(fields.get("status")),
        "finish_reason": text(fields.get("finish_reason")), "values": values,
        "emission_timing_enabled": enabled, "intervals_consistent": bool(valid) if enabled else None,
        "worker_observed_mean_interval_us": ratio(total, count) if valid else None,
        "client_exact_token_tpot_seconds": None,
        **{k: fields.get(k) if type(fields.get(k)) is bool else None for k in (
            "cancellation_requested", "cancellation_submitted", "cancellation_failed",
            "requires_restore_or_shutdown")},
    }


def summarize_invocations(records):
    metrics = {}
    rows = aggregate([r["rows"] if r["phase"] != "unknown" else None for r in records])
    for field in INVOCATION_FIELDS:
        summary = aggregate([r["values"][field] for r in records])
        summary["per_row"] = ratio(summary["total"], rows["total"])
        metrics[field] = summary
    gauges = {}
    for field in GAUGE_FIELDS:
        values = [r["gauges"][field] for r in records]
        known = [v for v in values if v is not None]
        gauges[field] = {
            "observations": len(values), "known": len(known), "unknown": len(values) - len(known),
            "max": max(known) if known and len(known) == len(values) else None,
            "last": values[-1] if values else None,
            "observed_max": max(known) if known else None,
        }
    return {"invocations": len(records), "rows": rows, "metrics": metrics, "gauges": gauges,
            "status_counts": dict(Counter(str(r["status"]) for r in records))}


def analyze_lines(lines: Iterable[str], *, source: str = "unspecified") -> dict:
    """Analyze one ordered log from one server process; never join cost to requests.

    Mixed batch costs appear once. Phase groups are mutually exclusive: a mixed
    invocation is not also charged to prefill and decode. First prefill means
    first *observed* pure-prefill invocation in this file, not per-request TTFT.
    """
    invocations, workers, admissions = [], [], []
    diagnostics = Counter()
    seen = set()
    identities = set()
    ambiguous = False
    for number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except (ValueError, TypeError):
            diagnostics["malformed_json_lines" if line.lstrip().startswith("{") else "non_json_lines"] += 1
            continue
        if not isinstance(record, dict) or not isinstance(record.get("fields"), dict):
            diagnostics["non_event_json_lines"] += 1
            continue
        fields, target = record["fields"], record.get("target")
        event = fields.get("event")
        if not isinstance(event, str):
            diagnostics["other_events"] += 1
            continue
        if not ((target == "ferrule_performance" and event == "mixed_decoder_invocation")
                or (target == "ferrule_request" and event in REQUEST_EVENTS)):
            diagnostics["other_events"] += 1
            continue
        # Exact logger-record repetition is not an extra invocation/request.
        fingerprint = json.dumps(record, sort_keys=True, allow_nan=True)
        if fingerprint in seen:
            diagnostics["duplicate_records"] += 1
            continue
        seen.add(fingerprint)
        if target == "ferrule_performance":
            item = invocation(fields, number)
            identity = (item["device"], item["transaction"])
            if None not in identity:
                if identity in identities:
                    ambiguous = True
                    diagnostics["conflicting_invocation_identities"] += 1
                identities.add(identity)
            invocations.append(item)
        elif event == "request_worker_terminal":
            workers.append(worker(fields, number))
        else:
            admissions.append({"line": number, "event": event, "request_id": uint(fields.get("request_id")),
                               "status": text(fields.get("status")),
                               **{k: uint(fields.get(k)) for k in (
                                   "worker_queue_us", "tokenization_us", "worker_admission_us",
                                   "http_admission_us", "http_response_enqueue_us", "token_events")}})
    first_prefill = next((r for r in invocations if r["phase"] == "prefill"), None)
    return {
        "schema": "ferrule.paper.performance-analysis.v1", "source": source,
        "instrumentation": "observed" if invocations or workers else "not_observed",
        "diagnostics": dict(diagnostics), "aggregate_identity_valid": not ambiguous,
        "invocations": invocations, "worker_requests": workers, "request_events": admissions,
        "invocation_summary": None if ambiguous else summarize_invocations(invocations),
        "phases": None if ambiguous else {
            name: summarize_invocations([r for r in invocations if r["phase"] == name])
            for name in ("prefill", "decode", "mixed", "unknown")},
        "completed_invocation_summary": None if ambiguous else summarize_invocations(
            [r for r in invocations if r["status"] == "completed" and r["quarantined"] is False]),
        "first_observed_prefill": first_prefill,
        "request_cost_attribution": None,
        "shared_cost_scope": "one transaction/batch invocation; no request-level allocation",
        "caveats": [
            "One source must contain one server lifetime; IDs are not global across restarts.",
            "Invocation costs are shared batch totals, never copied to each request.",
            "Counters are facade deltas, not native kernel counts; read bytes are not physical disk bytes.",
            "Proof timers are host wall time around existing calls, not device-kernel timings.",
            "Allocation counters/bytes are available; allocation duration is not measured.",
            "Cache owner peak is a cumulative gauge, not an invocation peak or summed byte cost.",
            "Worker successful channel-emission intervals differ from client-visible SSE TPOT proxies.",
            "Missing counters, NaN, malformed values and reset deltas remain unknown; observed_sum is partial.",
            "Log absence does not prove zero work, disabled instrumentation, or complete capture.",
        ],
    }


def analyze_log(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8", errors="replace") as stream:
        return analyze_lines(stream, source=str(path))


def analyze_http(report: dict) -> dict:
    """Separate maintained HTTP-client measurements; excludes warmups and aggregates."""
    requests = []
    for run_index, run in enumerate(report.get("runs", [])):
        for request_index, request in enumerate(run.get("requests", [])):
            requests.append({
                "run_index": run_index, "request_index": request_index, "status": text(request.get("status")),
                "actual_input_tokens": uint(request.get("actual_input_tokens")),
                "actual_output_tokens": uint(request.get("actual_output_tokens")),
                **{k: finite(request.get(k)) for k in (
                    "ttft_seconds", "request_e2e_seconds", "mean_tpot_usage_proxy_seconds",
                    "exact_token_tpot_seconds")},
            })
    return {"scope": "client_SSE_not_worker_emission", "requests": requests,
            "joined_to_worker_requests": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--http-report", type=Path)
    args = parser.parse_args(argv)
    result = analyze_log(args.server_log)
    if args.http_report:
        result["client"] = analyze_http(json.loads(args.http_report.read_text()))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
