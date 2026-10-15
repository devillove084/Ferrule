#!/usr/bin/env python3
"""Verify public pressure evidence offline, using only the Python standard library."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
POINTS = ("small", "middle", "big")
NAMES = ("hello2", "canary-qa", "canary-arithmetic", "canary-chinese")
CAPS = ((8, 64 << 20), (64, 320 << 20), (128, 640 << 20))
COUNTERS = ("checkpoint_read_bytes", "checkpoint_read_calls", "checkpoint_source_checks",
            "checkpoint_path_metadata_calls", "checkpoint_handle_metadata_calls", "expert_hits",
            "expert_uploads", "expert_evictions", "artifact_uploads", "artifact_upload_bytes",
            "h2d_copies", "h2d_bytes")
REQUEST_EVENTS = ("request_admitted", "request_http_admitted", "request_worker_terminal",
                  "request_worker_terminal_enqueue", "request_http_terminal")
NULL_COUNTERS = ("source_extents_read", "expert_cache_misses", "cache_hit_rate",
                 "working_set_unique_expert_ids", "physical_nas_backend_bytes")
DENOMINATORS = dict(scheduled=12, issued=12, terminal=12, successful=12,
                    failed=0, censored=0, unissued=0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(type(actual) in (int, float) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), message)


def one(rows, kind):
    found = [row["data"] for row in rows if row["kind"] == kind]
    require(len(found) == 1, "expected one " + kind)
    return found[0]


def values(rows, kind):
    return [row["data"] for row in rows if row["kind"] == kind]


def integer(value):
    return type(value) is int and value >= 0


def spans(rows):
    totals = {}
    for row in rows:
        value = totals.setdefault(row["phase"], dict(count=0, inclusive_wall_seconds=0,
                                                    inclusive_thread_cpu_seconds=0))
        value["count"] += 1
        value["inclusive_wall_seconds"] += row["wall_seconds"]
        value["inclusive_thread_cpu_seconds"] += row["thread_cpu_seconds"]
    return totals


def completed_spans(rows):
    starts = {}
    finished = []
    for row in rows:
        key = row["phase"], row["thread"], row["started_monotonic"]
        if row["status"] == "started":
            require(key not in starts, "duplicate timing start")
            starts[key] = row
        else:
            require(row["status"] == "complete" and key in starts, "incomplete or unpaired timing")
            begin = starts.pop(key)
            require(row["parent"] == begin["parent"], "timing parent changed")
            close(row["wall_seconds"], row["ended_monotonic"] - row["started_monotonic"], "timing wall")
            require(row["wall_seconds"] >= 0 and row["thread_cpu_seconds"] >= 0, "negative timing")
            finished.append(row)
    require(not starts, "open timing spans")
    return finished


def check_idle(value):
    require(value["status"] == "verified" and value["all_eight_idle"] is True, "idle gate")
    require(value["required_consecutive"] == value["consecutive_idle"] == 3, "three idle samples")
    selected = value["accepted_sample_indices"]
    require(len(selected) == 3 and selected == list(range(selected[0], selected[0] + 3)), "consecutive idle indices")
    for index in selected:
        sample = value["samples"][index]
        require(sample["classification"] == "idle" and sample["no_compute_apps"], "idle sample classification")
        require([row["index"] for row in sample["devices"]] == list(range(8)), "all eight idle devices")
        require(all(row["memory_mib"] == row["utilization_percent"] == 0 for row in sample["devices"]), "nonidle device")
    require(value["physical_owner_release"] == "unverified", "idle must not prove release")


def validate_request(row, frozen, events):
    require(row["name"] == frozen["name"] and row["issued"] is True and row["status"] == "success", "request slot")
    for key in ("prompt", "input_tokens", "max_tokens", "reference", "payload"):
        require(row[key] == frozen[key], "frozen request differs: " + key)
    raw = row["actual"]
    require(row["http_report_status"] == "complete" and raw["adapter_http_report_complete"] is True
            and not raw["adapter_http_report_errors"], "HTTP report completion")
    require(raw["status"] == "success" and raw["http_status"] == 200 and raw["error"] is None
            and raw["right_censored"] is False and not raw["after_deadline_events"], "HTTP completion")
    require(raw["request_payload"] == frozen["payload"], "HTTP payload differs")
    require(raw["prompt_sha256"] == hashlib.sha256(frozen["prompt"].encode()).hexdigest(), "prompt digest")
    require(raw["finish_reason"] == "length" and raw["length_matched_success"] is True, "HTTP length/usage")
    require(raw["sent_at"] < raw["finish_at"] <= raw["usage_at"] <= raw["done_at"] <= raw["ended_at"]
            <= raw["deadline_at"], "HTTP timing order/deadline")
    close(raw["request_e2e_seconds"], raw["ended_at"] - raw["sent_at"], "HTTP E2E")
    sse = raw["raw_sse_events"]
    require(sse and sse[-1]["data"] == "[DONE]" and sum(x["data"] == "[DONE]" for x in sse) == 1, "SSE DONE")
    decoded = [json.loads(event["data"]) for event in sse[:-1]]
    ids = {event["id"] for event in decoded}
    require(len(ids) == 1, "one actual completion ID")
    http_id = next(iter(ids))
    match = re.fullmatch(r"cmpl-ferrule-(\d+)", http_id)
    require(match is not None, "HTTP completion ID is not an observed worker binding")
    worker_id = int(match.group(1))
    worker = [event for event in events if event.get("request_id") == worker_id]
    require(worker and all(event.get("session_id") == worker_id for event in worker), "worker/session binding")
    enqueues = [event for event in worker if event.get("event") == "request_worker_token_enqueued"]
    require([event["emission_ordinal"] for event in enqueues] == list(range(1, frozen["max_tokens"] + 1)), "enqueue order/count")
    require(all(event["delivery_boundary"] == "worker_channel_enqueue" for event in enqueues), "token boundary")
    generated = [event["actual_token_id"] for event in enqueues]
    require(all(integer(token) for token in generated), "typed actual token IDs")
    text = "".join(choice.get("text", "") for event in decoded for choice in event["choices"])
    require(text == raw["text"], "SSE text differs")
    usages = [event["usage"] for event in decoded if "usage" in event]
    expected_usage = dict(prompt_tokens=frozen["input_tokens"], completion_tokens=frozen["max_tokens"],
                          total_tokens=frozen["input_tokens"] + frozen["max_tokens"])
    require(usages == [raw["usage"]] == [expected_usage], "SSE usage reconciliation")
    content_times = [event["at"] for event, decoded_event in zip(sse, decoded)
                     if any(choice.get("text") for choice in decoded_event["choices"])]
    require(content_times == raw["content_event_times"], "content event times")
    close(raw["ttft_seconds"], content_times[0] - raw["sent_at"], "client TTFT")
    for kind in REQUEST_EVENTS:
        matched = [event for event in worker if event.get("event") == kind]
        require(len(matched) == 1, "one worker gate: " + kind)
        gate = matched[0]
        if kind == "request_admitted":
            require(gate["prompt_tokens"] == frozen["input_tokens"] and gate["max_new_tokens"] == frozen["max_tokens"], "admission lengths")
        if kind in ("request_worker_terminal", "request_worker_terminal_enqueue"):
            require(all(gate[key] == expected_usage[key] for key in expected_usage), "worker terminal usage")
            require(gate["status"] == "finished" and gate["usage_available"] is True, "worker terminal status")
        if kind == "request_worker_terminal_enqueue":
            require(gate["terminal_event_enqueued"] is True, "terminal enqueue")
        if kind == "request_worker_terminal":
            require(gate["tokens_reconcile"] is True and gate["generated_tokens"] == frozen["max_tokens"], "worker token reconciliation")
        if kind == "request_http_terminal":
            require(gate["status"] == "finished" and gate["token_events"] == frozen["max_tokens"], "server HTTP terminal")
    exact = None
    if frozen["reference"] is not None:
        exact = generated == frozen["reference"]["reference_ids"]
        require(exact and text == frozen["reference"]["reference_text"], "exact eight-ID/text oracle")
    trace = row["trace"]
    require(trace["actual_worker_request_id"] == worker_id and trace["actual_http_completion_ids"] == [http_id]
            and trace["actual_generated_ids"] == generated and trace["exact_token_pass"] is exact,
            "reported token join differs from actual events")
    require(row["quality_passed"] is True and row["metric_eligible"] is True
            and trace["terminal_reconciled"] is True and trace["trace_valid"] is True,
            "visible quality eligibility")
    return dict(name=row["name"], status="success", worker_id=worker_id, http_ids=[http_id], generated_ids=generated,
                exact_token_pass=exact, usage=expected_usage, http_e2e_seconds=raw["request_e2e_seconds"],
                client_ttft_seconds=raw["ttft_seconds"], quality_passed=True, metric_eligible=True)


def derive(records):
    require(records, "empty publication")
    for record in records:
        require(record["schema"] == "ferrule.pressure-public-record.v1", "record schema")
        require(record["point"] in (None, *POINTS), "unregistered point")
        require(re.fullmatch(r"[0-9a-f]{64}", record["source"]["sha256"]), "source digest type")
    study = one(records, "study")
    require(study["status"] == "complete" and study["comparison_eligible"] is True, "complete study required")
    require(study["denominators"] == DENOMINATORS and study["trial_order"] == [p + "-blocking" for p in POINTS], "fixed denominator/order")
    require(study["issued_lifecycles"] == study["completed_lifecycles"] == study["actual_server_starts"] == 3, "lifecycle denominator")
    require(study["repetitions_per_point"] == 1 and study["retries"] == study["replacements"] == study["profiling_captures"] == 0, "no replicates/retries/profiling")
    require(study["historical_resume_permitted"] is False and study["full_numerical_per_point"] == "unverified"
            and study["new_point_full_numerical_verified"] is False, "study quality/scope")
    require(study["source_commit"] == "f3472c10656d255f783c1d8035fd7167bb439e4b", "source role binding commit")
    require(study["candidate_identity"]["cli_binary_sha256"] == "7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01", "candidate identity")
    require(all(row["live_checkout_consumed"] is False for row in study["compiled_source_relations"]), "compiled roles are frozen")
    require(sum(row["live_matches_compiled"] is False for row in study["compiled_source_relations"]) == 2, "preserve both live/frozen discrepancies")
    require([row["name"] for row in study["schedule"]] == list(NAMES), "frozen four-prompt schedule")
    points = []
    for point, (entries, budget) in zip(POINTS, CAPS):
        rows = [record for record in records if record["point"] == point]
        config = one(rows, "configuration")
        require((config["requested_entries"], config["requested_cache_bytes"]) == (entries, budget), "coupled cache settings")
        options = one(rows, "source_options")
        require(options["effective_admitted_entries"] is options["effective_admitted_cache_bytes"] is options["physical_admission"] is None, "physical admission remains unreported")
        effective = options["effective_metadata_options"]
        expected = dict(max_experts=entries, max_bytes=budget, scratch_bytes=8 << 20,
                        workspace_bytes=2 << 30, kv_bytes=16 << 20, device_bytes=7 << 30,
                        allocator_margin_bytes=512 << 20, max_positions=128, max_sequences=2, max_batch_tokens=8)
        require(effective == expected, "actual source options")
        geometry = study["prespecified_source_geometry"]
        payload = geometry["max_whole_expert_payload_bytes"]
        require(integer(payload) and payload > 0, "header-derived maximum expert payload")
        byte_bound = (budget - effective["scratch_bytes"]) // payload
        arithmetic = dict(max_whole_expert_payload_bytes=payload, byte_only_payload_bound=byte_bound,
                          joint_positive_payload_bound=min(entries, byte_bound),
                          model_expert_id_count=geometry["routed_layers"] * geometry["experts_per_layer"],
                          interpretation="Logical header-derived arithmetic, not physical admission or working-set occupancy")
        events = [event["fields"] for event in values(rows, "server_event")]
        reports = [e["report"] for e in events if e.get("message") == "resident model build entering owner-local admission"]
        require(len(reports) == 1 and "max_layers: descriptor default -> 40 " in reports[0]
                and "NumericFp8E4m3fnBf16Scales" in reports[0], "actual full40 startup report")
        lowered = re.findall(r"Qwen35MoeCapacityLimits \{([^}]+)\}", reports[0])
        require(len(lowered) == 2, "source lowering reports")
        require({key: int(value) for key, value in re.findall(r"(\w+): (\d+)", lowered[-1])} == expected, "source lowering actual caps")
        require(one(rows, "host_staging_report").startswith("expert read host staging: schedule=Blocking host_cap_bytes=67108864 metadata_bytes=8388608 worker_stack_bytes=2097152;"), "blocking host options")
        invocations = [event for event in events if event.get("event") == "mixed_decoder_invocation"]
        require(len(invocations) == 30 and [v["transaction"] for v in invocations] == list(range(1, 31)), "30 ordered invocations")
        for invocation in invocations:
            require(invocation["status"] == "completed" and invocation["quarantined"] is False
                    and invocation["counters_available"] is True and invocation["counter_reset_detected"] is False
                    and invocation["expert_read_counter_reset_detected"] is False, "invocation status/counters")
            require(all(integer(invocation[key]) for key in COUNTERS), "typed invocation counters")
            read = invocation["expert_read_pipeline"]
            require(read["effective_mode"] == "blocking" and read["stats_scope"].startswith("owner_lifetime"), "read mode/scope")
            require(invocation["expert_resident_entries"] <= entries, "invocation entry bound")
            charged = sum(invocation[key] for key in ("expert_resident_bytes", "expert_pending_upload_bytes", "expert_scratch_bytes"))
            require(charged <= invocation["expert_owner_peak_bytes"] <= budget, "logical cache charge bound")
        totals = {key: sum(invocation[key] for invocation in invocations) for key in COUNTERS}
        totals.update({key: None for key in NULL_COUNTERS})
        shutdowns = [event["summary_json"] for event in events if event.get("event") == "resource_shutdown_summary"]
        require(len(shutdowns) == 1 and shutdowns[0]["outcome"] == "complete", "owner close observation")
        physical = shutdowns[0]["physical"]
        cache, read = (physical[key]["value"] for key in ("expert_cache", "expert_read_pipeline"))
        require(cache["charged_bytes"] == cache["resident_bytes"] + cache["pending_upload_bytes"] + cache["scratch_bytes"], "charge partitions")
        require(cache["resident_entries"] == cache["uploads"] - cache["evictions"] == entries, "installs minus evictions")
        require(cache["uploads"] == totals["expert_uploads"] and cache["evictions"] == totals["expert_evictions"]
                and cache["hits"] == totals["expert_hits"] == 0, "cache counter reconciliation")
        require(cache["charged_bytes"] <= cache["peak_bytes"] <= budget and cache["scratch_bytes"] == 8 << 20
                and cache["pending_upload_bytes"] == cache["unknown_quarantine_bytes"] == 0
                and cache["quarantined"] is False, "final cache bounds")
        require(cache["peak_bytes"] == max(v["expert_owner_peak_bytes"] for v in invocations), "owner peak is max, never sum")
        require(read == invocations[-1]["expert_read_pipeline"], "owner cumulative read snapshot")
        require(read["status"] == "closed" and read["current_charge"]["host_charged_bytes"] == 0, "final logical read charge")
        require(read["stats"]["peak_host_charged_bytes"] == 16777984 < read["limits"]["max_host_bytes"] == 64 << 20, "host logical bound")
        for key, counter in (("verified_reads", "expert_uploads"), ("reader_interval_bytes", "checkpoint_read_bytes"),
                             ("reader_interval_calls", "checkpoint_read_calls")):
            require(read["stats"][key] == totals[counter] == sum(v["expert_read_pipeline_delta"][key] for v in invocations), "read delta/cumulative reconciliation: " + key)
        require(physical["release"]["state"] == physical["allocator"]["state"] == "unobserved", "physical release unobserved")
        requests = values(rows, "request")
        require([row["name"] for row in requests] == list(NAMES), "four serial request slots")
        joined = [validate_request(row, frozen, events) for row, frozen in zip(requests, study["schedule"])]
        ids = [row["worker_id"] for row in joined]
        require(len(set(ids)) == 4, "distinct worker requests")
        for kind in REQUEST_EVENTS:
            require([e["request_id"] for e in events if e.get("event") == kind] == ids, "whole-log request coverage: " + kind)
        require([e["request_id"] for e in events if e.get("event") == "request_worker_token_enqueued"]
                == [row["worker_id"] for row in joined for _ in row["generated_ids"]], "whole-log token coverage")
        timings = completed_spans(values(rows, "phase_timing"))
        begins = [r for r in timings if r["phase"] == "pre_admission_validation"]
        ends = [r for r in timings if r["phase"] == "post_http_validation"]
        require(len(begins) == len(ends) == 4, "four helper windows")
        for row, raw, begin, end in zip(joined, requests, begins, ends):
            helpers = [span for span in timings if span["thread"] == "worker"
                       and span["started_monotonic"] >= begin["started_monotonic"]
                       and span["ended_monotonic"] <= end["ended_monotonic"]]
            helper_totals = spans(helpers)
            require(helper_totals["candidate_pins"]["count"] == 3 and helper_totals["http"]["count"] == 1, "three validation pin checks per request")
            http = next(span for span in helpers if span["phase"] == "http")
            require(http["started_monotonic"] <= raw["actual"]["sent_at"] < raw["actual"]["ended_at"] <= http["ended_monotonic"], "HTTP helper/client nesting")
            for pin in [span for span in helpers if span["phase"] == "candidate_pins"]:
                require(pin["ended_monotonic"] <= http["started_monotonic"] or pin["started_monotonic"] >= http["ended_monotonic"], "pin checks must be outside HTTP E2E")
            row.update(helper_window=dict(start=begin["started_monotonic"], end=end["ended_monotonic"]), helper_timings=helper_totals)
        samples = values(rows, "memory_sample")
        require(len(samples) == 36 and all(s["host"]["status"] == "complete" for s in samples), "36 complete memory samples")
        bounds = {}
        for label, component, field in (("device_gpu", "gpu", "gpu_bytes"), ("tree_rss", "host", "rss_bytes"), ("tree_pss", "host", "pss_bytes")):
            measured = [s[component][field] for s in samples]
            require(all(integer(v) for v in measured), "missing memory sample is not zero")
            bounds[label] = dict(sample_count=len(measured), missing_samples=0, observed_min_bytes=min(measured),
                                 sampled_max_bytes=max(measured), exact_lifetime_peak_bytes=None, continuous_coverage=False,
                                 interpretation="sampled maximum is a lower bound on the lifetime peak, not a proven upper bound")
        nfs = {row["phase"]: row for row in values(rows, "nfs_snapshot")}
        def delta(before, after):
            a, b = nfs[before], nfs[after]
            require(not a["errors"] and not b["errors"] and a["counters"].keys() == b["counters"].keys(), "NFS counter coverage")
            values_ = {key: b["counters"][key] - a["counters"][key] for key in a["counters"]}
            require(all(integer(v) for v in values_.values()), "NFS reset")
            return values_
        lifecycle = one(rows, "lifecycle")
        require(lifecycle["status"] == "complete" and lifecycle["summary"]["successful"] == 4, "complete point")
        require(delta("requests-pre", "requests-post") == lifecycle["requests_nfs"]["deltas"], "NFS request-phase projection")
        for row in requests:
            require(delta(row["name"] + "-pre", row["name"] + "-post") == row["nfs"]["deltas"], "NFS request projection")
        require(lifecycle["requests_nfs"]["shared_client_traffic"] is True
                and lifecycle["requests_nfs"]["physical_nas_backend_bytes"] is None, "NFS is not backend bytes")
        for kind in ("coordinator-pre-idle", "worker-pre-idle", "coordinator-post-idle"):
            check_idle(one(rows, kind))
        post = one(rows, "coordinator-post-idle")
        custody = one(rows, "custody")
        require(custody["returncode"] == 0 and custody["timed_out"] is False and not custody["cleanup_errors"]
                and custody["custody"]["status"] == "proven_drained" and custody["custody"]["echild"] is True
                and custody["custody"]["containment_verified"] is True and not custody["custody"]["errors"]
                and post["authenticated_echild"] is post["recorded_owners_absent"] is True, "authenticated drain and absent owners")
        require(one(rows, "locks-released")["all_eight_local"] is True
                and one(rows, "locks-reprobed")["all_eight_nonblocking_reacquired"] is True, "local lock release/reprobe")
        libraries = values(rows, "library_validation")
        require(len(libraries) == 2 and {row["phase"] for row in libraries} == {"ready", "after-requests"}, "two library checks")
        require(all(row["status"] == "passed" and row["observed_library_count"] == len(row["libraries_by_basename"]) == 12
                    and not row["errors"] and not row["missing_required_paths"] and not row["deleted_library_paths"]
                    and row["live_mapped_memory_sha256"] is None for row in libraries), "library mapping checks")
        elapsed = lifecycle["coordinator_ended"]["monotonic"] - lifecycle["admission"]["monotonic"]
        require(elapsed <= study["limits"]["outer_lifecycle_seconds"] == 240, "point cap")
        reference = one(rows, "scalar_reference")
        require(reference["requests"] == joined, "derived request scalars differ")
        require(reference["invocation_counter_totals"] == totals and reference["final_owner_cache"] == cache
                and reference["final_owner_read"] == read and reference["memory_sample_bounds"] == bounds, "derived point scalars differ")
        close(elapsed, reference["point_elapsed_seconds"], "point elapsed")
        points.append(dict(point=point, requested_entries=entries, requested_cache_bytes=budget,
                           source_options=options, payload_arithmetic=arithmetic,
                           invocation_count=len(invocations), invocation_counter_totals=totals,
                           final_owner_cache=cache, final_owner_read=read, memory_sample_bounds=bounds,
                           requests=joined, point_elapsed_seconds=elapsed, launch_to_ready_seconds=lifecycle["startup_seconds"],
                           phase_timings=spans(timings), nfs_request_phase=lifecycle["requests_nfs"],
                           owned_gpu_attribution=lifecycle["owned_gpu_attribution"],
                           full_numerical_per_point="unverified", physical_owner_release="unobserved"))
    handoff = one(records, "handoff")
    require(handoff["status"] == "complete_idle_handoff" and handoff["all_eight_idle"] is True
            and handoff["all_eight_locks_reacquired_then_released"] is True, "final handoff")
    require(handoff["signals_by_handoff"] == handoff["extra_requests"] == handoff["extra_server_starts"] == handoff["profiling_captures"] == 0, "handoff does not run extra work")
    check_idle(one(records, "handoff_idle"))
    require(one(records, "handoff_locks") == dict(all_eight_reacquired=True, lock_count=8), "eight final lock probes")
    elapsed = study["ended"]["monotonic"] - study["started"]["monotonic"]
    require(elapsed <= study["limits"]["phase_seconds"] == 900, "study cap")
    tokens = sum(len(row["generated_ids"]) for point in points for row in point["requests"])
    canaries = sum(row["exact_token_pass"] is True for point in points for row in point["requests"])
    require(tokens == 78 and canaries == 9, "quality denominator")
    return dict(schema="ferrule.pressure-public-summary.v1", status="verified", denominators=DENOMINATORS,
                generated_tokens=tokens, oracle_covered_tokens=72, exact_eight_id_canaries=canaries,
                study_elapsed_seconds=elapsed, study=study, points=points, historical_results_pooled=False,
                historical_helper_durations_seconds=None, physical_nas_backend_bytes=None,
                cache_hit_rate=None, actual_working_set_unique_ids=None,
                claim_scope="n=1 fixed-order coupled entry/byte-budget sensitivity, blocking only; not isolated LRU, causal speedup or physical frontier")


def load_bundle(directory=HERE):
    manifest = json.loads((directory / "manifest.json").read_text())
    source_bytes = (directory / "source-manifest.json").read_bytes()
    require(hashlib.sha256(source_bytes).hexdigest() == manifest["source_manifest_sha256"]
            == "c570b05aa5c6ffc63c588c7e78441728cf1c1db9119e83a1a9a3c813efb7a283", "source manifest hash")
    pins = json.loads(source_bytes)["pins_relative_to_directory"]
    require(len(pins) == manifest["source_files_verified"] == 244, "244-file source proof inventory")
    for name, metadata in manifest["files"].items():
        path = (directory / name).resolve()
        require(path.is_relative_to(directory.resolve()), "bundle path escape")
        data = path.read_bytes()
        require(len(data) == metadata["bytes"] and hashlib.sha256(data).hexdigest() == metadata["sha256"], "bundle digest: " + name)
    data = gzip.decompress((directory / "records.jsonl.gz").read_bytes())
    require(len(data) == manifest["records"]["uncompressed_bytes"]
            and hashlib.sha256(data).hexdigest() == manifest["records"]["uncompressed_sha256"], "uncompressed record digest")
    records = [json.loads(line) for line in data.splitlines()]
    require(len(records) == manifest["records"]["count"], "record count")
    for row in records:
        require(pins[row["source"]["path"]] == row["source"]["sha256"], "record source binding")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=HERE)
    parser.add_argument("--check-source", type=Path, help="Optional private source rehash; never executes the candidate")
    args = parser.parse_args()
    records = load_bundle(args.bundle)
    derived = derive(records)
    require(derived == json.loads((args.bundle / "summary.json").read_text()), "published summary differs from raw derivation")
    if args.check_source:
        pins = json.loads((args.bundle / "source-manifest.json").read_text())["pins_relative_to_directory"]
        for name, digest in pins.items():
            path = (args.check_source / name).resolve()
            require(path.is_relative_to(args.check_source.resolve()), "source path escape")
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "private source changed: " + name)
        manifest = json.loads((args.bundle / "manifest.json").read_text())
        for name, digest in manifest["supplemental_source_sha256"].items():
            path = (args.check_source.parent / name).resolve()
            require(path.is_relative_to(args.check_source.parent.resolve()), "supplemental source path escape")
            require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "supplemental source changed: " + name)
    print(json.dumps(dict(status="verified", lifecycles=3, requests=12, exact_eight_id_canaries=9,
                          generated_tokens=78, invocations=90, memory_samples=108,
                          record_kinds=dict(Counter(row["kind"] for row in records)),
                          original_source_files_rehashed=244 if args.check_source else None), indent=2))


if __name__ == "__main__":
    main()
