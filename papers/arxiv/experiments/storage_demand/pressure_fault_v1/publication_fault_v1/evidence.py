"""CPU-only, fail-closed reader for the frozen actual Mixed fault cut.

No CUDA imports, subprocesses, environment capture, checkpoint reads, or writes.
This is a cut-specific evidence checker, not a generic telemetry validator.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = ("mixed_clean", "mixed_unknown")
SCHEDULES = ("blocking", "lookahead")
SELECTORS = (
    "transformer::forward::tests::cuda::mixed_fault::"
    "mixed_bf16_clean_fault_after_real_expert_submit_both_schedules",
    "transformer::forward::tests::cuda::mixed_fault::"
    "mixed_bf16_unknown_after_real_expert_submit_both_schedules",
)
JOBS = ("01-clean", "02-unknown")
HANDOFF_SHA = "8f653a69707d057160118f536e77aa061209eb0d3914222223d7228d2be5abda"
COMPONENT_COMMIT = "79d684f5c79fc6c3961161111ae6efd1a37ea8ea"
ARCHIVE_COMMIT = "bd0673579286e614079ccfe5d972be19f860e9e8"
ARCHIVE_SHA = "95423c59156abc391eb0a3d80c3c1b2c368bddb5366931456d0bebf1fdd18983"
BINARY_SHA = "e531be47a5113cc53665fdc6576cf389b56c29cb7e16f275efec52b0f79b6903"
SOURCE_DIGEST = "c5f447cd814c8b672c4a6afc380f34a3633478946ee44a68b4143e5c8d238097"
LEDGER_KEYS = ("budget", "free", "resident", "pending", "scratch", "charged", "unknown_subset")
CACHE_MAP = dict(zip(
    ("resident", "pending", "scratch", "charged", "unknown_subset"),
    ("resident_bytes", "pending_upload_bytes", "scratch_bytes", "charged_bytes", "unknown_quarantine_bytes"),
))
HOLD_KEYS = ("artifact_holds", "bf16_holds", "f32_holds", "route_holds")
CLEAN_CONTENT = "GDN_and_KV_bitwise_unchanged"
UNKNOWN_CONTENT = "unobserved_post_Unknown_no_device_read_or_release_authority"


class EvidenceError(ValueError):
    """Messages name the failed check, never echo untrusted input values."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def equal(actual, expected, message):
    # JSON equality distinguishes bool from int, including nested fields.
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False), message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def no_duplicates(pairs):
    d = {}
    for k, v in pairs:
        require(k not in d, "duplicate JSON key")
        d[k] = v
    return d


def loads(text):
    try:
        return json.loads(text, object_pairs_hook=no_duplicates,
                          parse_constant=lambda _: require(False, "nonfinite JSON number"))
    except (json.JSONDecodeError, UnicodeError) as e:
        raise EvidenceError("invalid JSON encoding") from e


def load(path):
    return loads(Path(path).read_text(encoding="utf-8"))


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def get(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def pick(value, keys):
    return {key: get(value, key) for key in keys}


def safe_report(value, contract=None):
    """Closed vocabulary: only reviewed field names, enums and quantities.

    Unknown names and text are rejected rather than redacted into apparent
    evidence. The contract is checked in; it is never learned from new input.
    """
    contract = contract or load(HERE / "report-contract.json")
    keys, strings = set(contract["safe_keys"]), set(contract["safe_strings"])

    def visit(v):
        if isinstance(v, dict):
            require(set(v) <= keys, "unapproved report field name")
            for item in v.values():
                visit(item)
        elif isinstance(v, list):
            for item in v:
                visit(item)
        elif isinstance(v, str):
            require(v in strings, "unapproved report text")
        else:
            require(v is None or type(v) is bool or
                    (type(v) is int and 0 <= v < 2**64), "invalid report quantity")
    visit(value)


def read_rows(path):
    data = Path(path).read_bytes()
    require(len(data) <= 1024 * 1024, "oversized report")
    require(data.endswith(b"\n"), "truncated JSONL")
    rows = [loads(line) for line in data.splitlines()]
    safe_report(rows)
    return rows


def signature(row):
    return [row.get("record"), row.get("action"), row.get("label")]


def ledger_check(ledger, cache):
    require(all(type(ledger.get(k)) is int and ledger[k] >= 0 for k in LEDGER_KEYS),
            "missing or invalid ledger quantity")
    equal(ledger["budget"], 65536, "cache budget drift")
    equal(ledger["resident"] + ledger["pending"] + ledger["scratch"],
          ledger["charged"], "cache charged equation")
    equal(ledger["free"] + ledger["charged"], ledger["budget"], "cache free equation")
    require(ledger["unknown_subset"] <= ledger["charged"], "Unknown is a subset, not an extra bucket")
    for k, ck in CACHE_MAP.items():
        equal(ledger[k], cache.get(ck), "cache observation/ledger mismatch")


def snapshot_check(s, schedule, stage, unknown):
    ordinal = SCHEDULES.index(schedule)
    generation, topology = 1 + ordinal * 2, 2 + ordinal * 2
    equal(s["transaction_under_test"], 2, "wrong transaction")
    equal(s["committed"], dict(frontier=1, gdn_position=1, generation=0,
                              state_bytes=512, topology=topology), "committed frontier/generation drift")
    equal(s["image"]["image_generation"], generation, "image generation drift")
    final_clean = stage == "after_cleanup" and not unknown
    active = 0 if stage == "before" or final_clean else 1
    equal(s["kv"], dict(active_transactions=active, committed_page_slot=0,
                        free_slots=31 if active == 0 else 29), "KV custody/refund drift")
    equal(s["runner"], dict(active_transactions=active, commits=1,
                            aborts=1 if final_clean else 0, completed_batches=1), "runner custody drift")
    equal(s["live_state_bytes_including_default_committed_working"],
          1024 if active == 0 else 1536, "working state returned or lost")
    faulted = stage in ("after_failure", "after_cleanup")
    quarantined = unknown and faulted
    for v in (s["device_quarantined"], s["image"]["quarantined"], s["image"]["cache"]["quarantined"]):
        equal(v, quarantined, "quarantine drift")
    ledger_check(s["cache_ledger"], s["image"]["cache"])
    expected = ((65536, 58618, 0, 774, 6144, 6918, 6918) if quarantined else
                (65536, 61440, 0, 0, 4096, 4096, 0) if faulted else
                (65536, 60666, 774, 0, 4096, 4870, 0))
    equal([s["cache_ledger"][k] for k in LEDGER_KEYS], list(expected), "cache charge/refund drift")
    equal(s["image"]["chunk_workspace_bytes"], 318720, "separate recurrent workspace drift")
    equal([s["image"][k] for k in HOLD_KEYS], [3, 7, 3, 1] if quarantined else [0]*4,
          "transient owner holds drift")
    equal(s["image"]["admitted_expert"], [generation, 1, 0] if quarantined else None,
          "admission custody drift")
    host = s["host"]
    equal(host["status"], "closed", "host worker not closed")
    equal(host["effective_mode"], schedule, "host schedule mismatch")
    require(host["worker_read_observed"] is True, "actual host read absent")
    for v in host["current_charge"].values():
        equal(v, 0, "host charge not discharged")
    for k in ("unknown_bytes", "unknown_reads", "join_failed_sequences", "join_pending_sequences"):
        equal(host["stats"][k], 0, "host Unknown or join failure")
    equal(host["current_read"], None, "host current read remains")
    equal(host["pending_read"], None, "host pending read remains")
    physical = s["physical_owner"]
    equal(physical["expert_cache"], {"state": "observed", "value": s["image"]["cache"]},
          "physical cache view mismatch")
    equal(physical["expert_read_pipeline"], {"state": "observed", "value": host}, "physical host view mismatch")
    for k in ("release", "quiescence"):
        equal(physical[k], {"state": "unknown" if quarantined else "unobserved"},
              "invented physical release/quiescence proof")
    equal(physical["close"], {"state": "unknown"} if quarantined else
          {"state": "observed", "value": "open"}, "close observation drift")
    equal(physical["allocator"], {"state": "unobserved"}, "passive allocator proof invented")
    equal(physical["image_generation"], {"state": "observed", "value": generation}, "physical generation drift")
    equal(s["state_contents"], "unobserved_by_passive_sampler", "invented poststate contents")


def project_snapshot(s):
    """Allowlisted snapshot projection. Missing observations stay null, never 0."""
    return {
        "committed": pick(get(s, "committed"), ("frontier", "generation", "topology", "gdn_position", "state_bytes")),
        "cache": pick(get(s, "cache_ledger"), LEDGER_KEYS),
        "live_state_bytes": get(s, "live_state_bytes_including_default_committed_working"),
        "runner": pick(get(s, "runner"), ("active_transactions", "commits", "aborts", "completed_batches")),
        "kv": pick(get(s, "kv"), ("active_transactions", "free_slots", "committed_page_slot")),
        "host": {"status": get(s, "host", "status"),
                 "charge_bytes": get(s, "host", "current_charge", "host_charged_bytes"),
                 "unknown_bytes": get(s, "host", "stats", "unknown_bytes"),
                 "unknown_reads": get(s, "host", "stats", "unknown_reads")},
        "image_generation": get(s, "image", "image_generation"),
        "holds": pick(get(s, "image"), HOLD_KEYS),
        "device_quarantined": get(s, "device_quarantined"),
        "physical_flags": {k: {"state": get(s, "physical_owner", k, "state"),
                                "value": get(s, "physical_owner", k, "value")}
                           for k in ("allocator", "close", "quiescence", "release")},
        "recurrent_workspace_bytes_separate": get(s, "image", "chunk_workspace_bytes"),
        "workspace_overlap_bytes": None,
        "aggregate_physical_retained_bytes": None,
        "exact_kv_reference_counts": None,
        "state_content_bytes": None,
    }


def validate_rows(rows, case):
    contract = load(HERE / "report-contract.json")
    safe_report(rows, contract)
    require(case in CASES, "unsupported case")
    unknown = case == "mixed_unknown"
    equal(len(rows), 40, "record count/duplicate injection/incomplete cut")
    header = dict(contract["header"], case=case)
    equal(rows[0], header, "header/config/scope drift")
    equal(rows[-1], contract["seal"], "seal count/status drift")
    cells = []
    for ordinal, schedule in enumerate(SCHEDULES):
        start = 1 + ordinal * 19
        body = rows[start:start+19]
        equal([signature(r) for r in body], contract["sequence"][case], "ordered actions/injection count drift")
        for r in body:
            if "schedule" in r:
                equal(r["schedule"], schedule, "cell order/schedule drift")
        equal(body[0]["gpu_execution"], "running_not_yet_proved", "body entry drift")
        outcome = body[-1]
        equal(pick(outcome, ("status", "gpu_execution")),
              {"status": "assertions_passed", "gpu_execution": "observed"}, "body did not pass")
        snapshots = [(start+i+1, r) for i, r in enumerate(body) if r["record"] == "passive_snapshot"]
        equal(len(snapshots), 4, "snapshot count")
        stages = ("before", "prepared", "after_failure", "after_cleanup")
        by_stage = {}
        for stage, (line, r) in zip(stages, snapshots):
            snapshot_check(r["value"], schedule, stage, unknown)
            by_stage[stage] = r["value"]
        before, prepared, failed, after = (by_stage[s] for s in stages)
        boundary = body[8]
        equal(boundary["faults_observed"], 1, "injection count")
        equal(boundary["transaction"], 2, "fault transaction drift")
        equal(boundary["error_kind"], "CompletionUnknown" if unknown else "known_operation_failure", "wrong error kind")
        equal(boundary["physical_proof_before_test_discards_receipt"], {
            "compute_event_completed": True, "upload_sync_completed": True,
            "deliberately_discarded": unknown}, "not a successful actual wait before synthetic discard")
        submission = boundary["submission"]
        equal(pick(submission, ("layer", "expert", "bf16_elements", "input_compute_launches",
                                "output_compute_launches", "input_h2d_copies", "output_h2d_copies")),
              dict(layer=1, expert=0, bf16_elements=16, input_compute_launches=372,
                   output_compute_launches=374, input_h2d_copies=158, output_h2d_copies=158), "routed Down submission drift")
        receipt = submission["reads"]["last_receipt"]
        equal(pick(receipt, ("outcome", "io_bytes")), dict(outcome="source_verified", io_bytes=774), "routed read not verified")
        equal(pick(receipt["identity"], ("layer", "expert", "image_generation")),
              dict(layer=1, expert=0, image_generation=before["image"]["image_generation"]), "read/source generation drift")
        require(type(receipt["taken_ns"]) is int, "read take unobserved")
        equal(boundary["owner"]["image_generation"], before["image"]["image_generation"], "boundary image generation drift")
        equal([boundary["owner"]["cache"][k] for k in CACHE_MAP.values()],
              [0, 774, 6144, 6918, 0], "preproof charge drift")
        equal(boundary["owner"]["quarantined"], False, "preproof quarantine drift")
        host_receipt = body[10]
        equal(host_receipt["device_release_authority"], False, "host cleanup cannot release CUDA custody")
        equal(host_receipt["host_charge_after"], failed["host"]["current_charge"], "host cleanup mismatch")
        expected_reads = 5 + ordinal
        for k in ("submitted_delta", "completed_delta"):
            equal(host_receipt[k], expected_reads, "host cleanup count drift")
        equal(host_receipt["cancelled_delta"], ordinal, "host cancellation count drift")
        evidence = outcome["evidence"]
        equal(evidence["after_cleanup"], after, "outcome and snapshot mismatch")
        for k, v in dict(real_compute_launch_delta=183, real_h2d_copy_delta=50,
                         artifact_upload_delta=15, artifact_upload_byte_delta=3870,
                         real_read_byte_delta=3870+774*ordinal, faults_observed=1,
                         committed_frontier_before=1, committed_frontier_after=1).items():
            equal(evidence[k], v, "CUDA/read count or frontier drift")
        delta = evidence["read_statistics_delta"]
        for k, v in dict(submitted_reads=expected_reads, completed_reads=expected_reads,
                         verified_reads=5, cancelled_reads=ordinal, wasted_reads=ordinal,
                         proved_read_bytes=3870+774*ordinal, wasted_read_bytes=774*ordinal,
                         unknown_reads=0, unknown_bytes=0).items():
            equal(delta[k], v, "read statistics count drift")
        for k, value in delta.items():
            if not k.startswith("peak_"):
                equal(value, failed["host"]["stats"][k] - before["host"]["stats"][k], "read delta not backed by snapshots")
        equal(evidence["cuda_counter_scope"], "whole_tx2_owner_not_per_read_ticket_attribution", "counter scope drift")
        equal(evidence["committed_state_content_check"], UNKNOWN_CONTENT if unknown else CLEAN_CONTENT,
              "falsified post-Unknown state equality")
        equal(evidence["result"], "typed_Unknown_sticky_host_clean_device_custody_retained" if unknown else
              "clean_abort_transients_released_retry_succeeded", "outcome claim drift")
        release = body[-2]
        if unknown:
            equal(failed["image"], after["image"], "Unknown owner changed after independent waits")
            equal(failed["kv"], after["kv"], "Unknown KV returned")
            equal(release, dict(record="unknown_retention", action="retain_runner_states_device_until_process_exit",
                                forced_allocator_zero=False, owner_release="unobserved_not_authorized"), "Unknown refund/release invented")
        else:
            equal(pick(release, ("allocator_requested_baseline", "allocator_requested_after_drop_trim", "scope")),
                  dict(allocator_requested_baseline=0, allocator_requested_after_drop_trim=0,
                       scope="this_fixture_owner_not_device_wide_free_memory"), "clean local final allocator release drift")
        cells.append({
            "case": case, "schedule": schedule, "status": "passed_component_only",
            "source": {"report": f"evidence/{case}.jsonl", "case_lines": [start+1, start+19],
                       "boundary_line": start+9, "outcome_line": start+19,
                       "snapshot_lines": dict(zip(stages, [x[0] for x in snapshots]))},
            "snapshots": {k: project_snapshot(v) for k, v in by_stage.items()},
            "boundary": {"error_kind": boundary["error_kind"], "faults_observed": boundary["faults_observed"],
                         "physical_proof_before_discard": boundary["physical_proof_before_test_discards_receipt"],
                         "routed_down": pick(submission, ("layer", "expert", "bf16_elements", "input_compute_launches", "output_compute_launches")),
                         "cache_charged_bytes_before_proof": boundary["owner"]["cache"]["charged_bytes"],
                         "host_charge_bytes_before_cleanup": submission["reads"]["current_charge"]["host_charged_bytes"]},
            "tx2_counts": pick(evidence, ("real_compute_launch_delta", "real_h2d_copy_delta", "artifact_upload_delta",
                                           "artifact_upload_byte_delta", "real_read_byte_delta", "cuda_counter_scope")),
            "host_cleanup": pick(host_receipt, ("submitted_delta", "completed_delta", "cancelled_delta", "device_release_authority")),
            "host_read_statistics": pick(delta, ("submitted_reads", "completed_reads", "verified_reads", "cancelled_reads",
                                                     "wasted_reads", "proved_read_bytes", "wasted_read_bytes", "unknown_reads", "unknown_bytes")),
            "committed_content": {"equality": None if unknown else True,
                                  "evidence_kind": "unobserved" if unknown else "serialized_assertion_result_not_raw_content",
                                  "basis": UNKNOWN_CONTENT if unknown else CLEAN_CONTENT,
                                  "raw_contents": None},
            "assertion_backed": {
                "basis": "pinned_fault_cuda_source_actions_and_passed_outcome_not_individual_API_receipts",
                "source": "reference/fault_cuda.rs",
                "source_lines": [494, 581] if unknown else [424, 493],
                "independent_compute_waits_succeeded": 2 if unknown else None,
                "independent_upload_waits_succeeded": 2 if unknown else None,
                "abort_attempts": 2 if unknown else 1,
                "abort_completion": "CompletionUnknown" if unknown else "Completed",
                "independent_wait_rounds_succeeded": 2 if unknown else None,
                "standalone_wait_receipts": None, "individual_error_payloads": None,
                "rejected_operations": ["reset_sequence_state", "reset_session", "create_sequence_state",
                                        "release_kv_pages_1_2_3", "publish_tx2", "shutdown"] if unknown else None,
                "healthy_in_test_tx3": None if unknown else "passed_then_aborted_not_selector_retry",
            },
            "local_final_release": None if unknown else pick(release, ("allocator_requested_baseline", "allocator_requested_after_drop_trim", "scope")),
            "unknown_retention": pick(release, ("forced_allocator_zero", "owner_release")) if unknown else None,
        })
    return cells


def validate_streams(rows, stdout, stderr, selector):
    pattern = (r"\s*running 1 test\ntest " + re.escape(selector) + r" \.\.\. ok\n\n"
               r"test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; 902 filtered out; "
               r"finished in [0-9.]+s\s*")
    require(re.fullmatch(pattern, stdout) is not None, "producer stdout selector/count/unsafe text")
    lines = stderr.splitlines()
    require(len(lines) == len(rows) and all(x.startswith("COMPONENT_FAULT ") for x in lines),
            "stderr contains extra or missing producer output")
    equal([loads(x[len("COMPONENT_FAULT "):]) for x in lines], rows, "producer stderr/JSONL mismatch")


def build_model(root=HERE):
    cells = []
    config = None
    for case, selector in zip(CASES, SELECTORS):
        rows = read_rows(root / "evidence" / f"{case}.jsonl")
        cells.extend(validate_rows(rows, case))
        validate_streams(rows, (root / "evidence" / f"{case}.stdout").read_text(),
                         (root / "evidence" / f"{case}.stderr").read_text(), selector)
        config = rows[0]["config"]
    return {
        "schema": "ferrule.model-mixed-fault-postfailure.v1",
        "scope": "tiny_actual_Mixed_CudaHybridModule_component_only",
        "evidence_kinds": {"snapshots": "serialized_field_measurements_not_content_reads",
                           "boundary": "serialized_test_hook_observation",
                           "assertion_backed": "pinned_test_source_plus_actions_plus_PASS",
                           "missing_individual_API_receipts": None},
        "config": config, "cells": cells,
        "denominator": {"selectors": 2, "cells": 4, "passed_cells": len(cells), "selector_retries": 0},
        "math": {"cache_budget_bytes": 65536, "charged_equation": "resident + pending + scratch",
                 "budget_equation": "free + charged", "unknown_relation": "subset_of_charged_not_additive",
                 "runtime_broker_five_bucket_equation": None, "physical_nonoverlapping_total_bytes": None},
        "claims_not_established": ["physical_event_loss_or_hardware_failure", "post_Unknown_state_content_equality",
                                   "Unknown_in_process_release", "full_numerical_quality_reference",
                                   "35B_fault_guarantee", "production_CLI_integrated_E5", "timing_or_overlap"],
    }
