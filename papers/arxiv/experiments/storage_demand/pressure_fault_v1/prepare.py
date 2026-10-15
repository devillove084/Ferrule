#!/usr/bin/env python3
"""Offline preregistration checks/derivation only. No launch, bind, or run mode."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from evidence import NAMES, can_admit, counter_totals, denominator_summary, positive_payload_entries, require

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
MIB = 1 << 20
POINTS = (("small", 8, 64 * MIB), ("middle", 64, 320 * MIB), ("big", 128, 640 * MIB))
PREPARATION = HERE / "plan_v2"
PLAN_PATH = PREPARATION / "plan.json"
from binding_roles import BINDING_PATH
LIMITS = dict(startup_seconds=60, request_seconds=15, all_requests_seconds=120,
              worker_bookkeeping_seconds=15, target_lifetime_seconds=195,
              shutdown_seconds=20, worker_preidle_seconds=10, worker_finalization_seconds=4,
              supervisor_seconds=219, supervisor_call_overhead_seconds=11,
              coordinator_preidle_seconds=10, coordinator_postidle_seconds=5,
              coordinator_finalization_seconds=5, outer_lifecycle_seconds=240,
              phase_seconds=900, study_cleanup_reserve_seconds=180)


def budget_proof(limits):
    """Offsets share ONE point origin; pre-idles/reviews are inside startup."""
    require(limits == LIMITS, "v2 absolute phase caps changed")
    startup = limits["startup_seconds"]
    requests = startup + limits["all_requests_seconds"]
    target = requests + limits["worker_bookkeeping_seconds"]
    shutdown = target + limits["shutdown_seconds"]
    worker = shutdown + limits["worker_finalization_seconds"]
    supervisor = worker + limits["supervisor_call_overhead_seconds"]
    postidle = supervisor + limits["coordinator_postidle_seconds"]
    point = postidle + limits["coordinator_finalization_seconds"]
    require(target == limits["target_lifetime_seconds"] and worker == limits["supervisor_seconds"],
            "target/worker phases do not compose")
    require(point == limits["outer_lifecycle_seconds"] == 240, "point phases do not compose")
    require(3 * point + limits["study_cleanup_reserve_seconds"] == limits["phase_seconds"] == 900,
            "study phases do not compose")
    require(limits["request_seconds"] == 15 and 4 * limits["request_seconds"] < limits["all_requests_seconds"],
            "HTTP caps must leave observation/validation budget")
    return dict(startup=startup, requests=requests, bookkeeping=target, shutdown=shutdown,
                worker_finalization=worker, supervisor_cleanup=supervisor, postidle=postidle, point=point)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def check_bindings():
    from binding_roles import check_bindings as check_roles
    return check_roles(BINDING_PATH)


def validate_design(value):
    require(value["status"] == "cpu_prepared_execution_blocked_not_authorized"
            and value["launch_entry"] is False, "plan must not authorize execution")
    require(all(value[k] == 0 for k in ("gpu_queries", "gpu_locks", "gpu_runs")), "CPU-only plan")
    p = value["pressure"]
    require([(r["id"], r["entries"], r["cache_bytes"]) for r in p["points"]] == list(POINTS),
            "prospective three-point design drift")
    require(p["order"] == [r[0] for r in POINTS] and p["request_names"] == list(NAMES), "fixed order drift")
    require(p["schedule"] == "blocking" and p["repetitions_per_point"] == 1, "blocking-only n=1 design")
    require(all(p[k] == 0 for k in ("retry_count", "replacement_count", "extra_qa_requests"))
            and not p["adaptive_orders"] and not p["reuse_middle_baseline_samples"], "no extra data/adaptation")
    require(p["limits"] == LIMITS, "absolute budget changed")
    require(value["schema"] == "ferrule.storage-demand.pressure-fault-plan.v2"
            and value["id"] == "pressure-fault-observation-budget-20261015-v2", "new study identity required")
    require(p["phase_deadline_offsets"] == budget_proof(LIMITS), "phase deadline composition drift")
    require(value["historical_resume_permitted"] is False, "historical study cannot resume")
    require(p["quality"]["full_numerical_per_point"] == "unverified"
            and p["quality"]["arbitrary_cap_numeric_factory_gate_available"] is False,
            "cannot transfer middle numerical PASS")
    return value


def frozen_requests(value):
    p = value["pressure"]
    rows = load(ROOT / p["request_source"])["schedule"][:4]
    oracles = load(ROOT / p["oracle_source"])["rows"]
    require([r["name"] for r in rows] == list(NAMES), "canonical initial request order differs")
    require([r["reference"] for r in rows[1:]] == oracles, "canonical three-case token oracle differs")
    require(rows[0]["max_tokens"] == 2 and rows[0]["reference"] is None, "Hello2 smoke differs")
    for row in rows:
        payload = row["payload"]
        require(payload["temperature"] == 0 and payload["seed"] == 1 and payload["n"] == 1
                and payload["stream"] is True and payload["stream_options"] == {"include_usage": True},
                "frozen greedy serial controls differ")
    require(all(r["max_tokens"] == len(r["reference"]["reference_ids"]) == 8 for r in rows[1:]),
            "all canaries require all eight visible generated IDs")
    return rows


def cases(value):
    """Produce requested argv, deliberately NOT candidate-accepted case receipts."""
    p = value["pressure"]
    base = load(ROOT / p["case_source"])["blocking"]
    original = load(ROOT / p["candidate_points_source"])["cache_candidates"]
    require(all(original[name] == [entries, size] for name, entries, size in POINTS), "original points drift")
    require(p["vary_only"] == ["--cuda-expert-device-entries", "--cuda-expert-device-bytes"],
            "only the two coupled cache knobs may vary")
    argv = base["argv_after_binary"]
    require(argv[:2] == ["serve", base["model"]] and (len(argv) - 2) % 2 == 0
            and len(set(argv[2::2])) == len(argv[2::2]), "literal unique CLI flag/value pairs required")
    flags = dict(zip(argv[2::2], argv[3::2]))
    expected = {"--cuda-device-budget-mb": "7168", "--expert-read-host-mb": "64",
                "--expert-read-metadata-mb": "8", "--expert-read-worker-stack-mb": "2",
                "--cuda-expert-scratch-mb": "8", "--kv-cache-mb": "16", "--cuda-workspace-mb": "2048",
                "--ctx-size": "128", "--max-active-sequences": "2", "--max-batch-tokens": "8",
                "--prefill-chunk-size": "8", "--decode-cohort-target": "1",
                "--decode-cohort-max-deferrals": "0", "--expert-read-schedule": "blocking",
                "--model-profile": "qwen35-mixed-bf16"}
    require(all(flags.get(k) == v for k, v in expected.items()), "fixed CLI options differ")
    require(base["environment"]["FERRULE_PROFILE_NVTX"] == "0", "profiling must be off")
    geometry = p["source_geometry"]
    payload = geometry["max_whole_expert_payload_bytes"]
    total = geometry["routed_layers"] * geometry["experts_per_layer"]
    rows = []
    for name, entries, size in POINTS:
        requested = list(argv)
        for flag, number in zip(p["vary_only"], (entries, size)):
            requested[requested.index(flag) + 1] = str(number)
        effective = positive_payload_entries(entries, size, p["fixed"]["scratch_bytes"], payload)
        rows.append(dict(id=name, requested_argv_after_binary=requested,
                         requested_entries=entries, requested_cache_bytes=size,
                         scratch_bytes=p["fixed"]["scratch_bytes"], expert_payload_bytes=payload,
                         byte_only_payload_entries=(size - p["fixed"]["scratch_bytes"]) // payload,
                         logical_positive_payload_entry_bound=effective,
                         model_expert_id_count=total, logical_fraction_of_model_ids=effective / total,
                         max_source_extents_per_whole_expert=3 * geometry["max_projection_extents"],
                         effective_requested_read_options=copy.deepcopy(base["read_options"]),
                         effective_admitted_entries=None, effective_admitted_cache_bytes=None,
                         static_header_preflight=static_header_preflight(value, requested),
                         physical_admission=None, point_full_numerical_verified=False,
                         actual_working_set_unique_expert_ids=None, measurements=counter_totals([])))
    return rows


def static_header_preflight(value, argv):
    """Arithmetic on a pinned existing actual-header receipt, not new CLI lowering."""
    pressure = value["pressure"]
    text = (ROOT / pressure["source_geometry"]["evidence"]).read_text()
    rows = [json.loads(line.split("=", 1)[1]) for line in text.splitlines()
            if line.startswith("source_admission=")]
    prior = [row for row in rows if row.get("schedule") == "blocking"]
    require(len(prior) == 1 and prior[0]["status"] == "PASS"
            and prior[0]["cache_entries"] == 64
            and prior[0]["cache_including_scratch_bytes"] == 320 * MIB,
            "existing actual-header middle receipt missing/different")
    prior = prior[0]
    recorded = [json.loads(line.split("=", 1)[1])[1:] for line in text.splitlines()
                if line.startswith("v3_serve_argv=")]
    require(len(recorded) == 2, "actual-header literal argv evidence absent")
    original = recorded[0]
    allowed = {original.index(flag) + 1 for flag in pressure["vary_only"]}
    require(len(argv) == len(original)
            and {i for i, pair in enumerate(zip(argv, original)) if pair[0] != pair[1]} <= allowed,
            "new source options differ beyond the two cache caps")
    fixed = pressure["fixed"]
    require(prior["effective_cli_max_metadata_bytes"] == fixed["metadata_bytes"]
            and prior["host_cap_bytes"] == fixed["host_read_bytes"]
            and prior["worker_stack_bytes"] == fixed["worker_stack_bytes"]
            and prior["max_whole_expert_payload_bytes"] == pressure["source_geometry"]["max_whole_expert_payload_bytes"],
            "actual-header source/fixed limits changed")
    metadata_margin = fixed["metadata_bytes"] - prior["required_composition_metadata_bytes"]
    payload_cap = fixed["host_read_bytes"] - fixed["metadata_bytes"] - fixed["worker_stack_bytes"]
    required = 2 * prior["max_whole_expert_payload_bytes"]
    require(metadata_margin >= 0 and payload_cap >= required, "static source metadata/payload bound rejected")
    return dict(status="static_existing_header_arithmetic_checked", metadata_margin_bytes=metadata_margin,
                double_buffer_payload_cap_bytes=payload_cap, double_buffer_required_bytes=required,
                actual_cli_lowering_executed_for_point=False, actual_header_reopened=False,
                source_runtime_admission="pending_real_server_startup", physical_capacity_admitted=None)


def prepared():
    binding = check_bindings()
    value = validate_design(load(PLAN_PATH))
    candidate = value["candidate"]
    recorded = load(ROOT / candidate["binding"])
    require(recorded["candidate_identity"]["commit"] == candidate["commit"]
            and recorded["source_digest"] == candidate["source_digest"]
            and recorded["cli_binary_sha256"] == candidate["cli_binary_sha256"], "wrong qualified candidate")
    require(binding["files"][candidate["qualification_final"]] == candidate["qualification_final_sha256"],
            "nine-gate final not bound")
    requests = frozen_requests(value)
    require(load(PREPARATION / "requests.json") == requests, "sealed four-request copy differs")
    require(load(PREPARATION / "cases.json") == cases(value), "sealed prospective cases differ")
    geometry_path = ROOT / value["pressure"]["source_geometry"]["evidence"]
    geometry = [json.loads(line.split("source_geometry=", 1)[1]) for line in geometry_path.read_text().splitlines()
                if "source_geometry=" in line]
    require(len(geometry) == 1, "one actual-header source geometry required")
    for key, actual in geometry[0].items():
        if key in value["pressure"]["source_geometry"]:
            require(actual == value["pressure"]["source_geometry"][key], "source geometry differs: " + key)
    return dict(schema="ferrule.storage-demand.pressure-fault-prepared.v2",
                status=value["status"], launch_entry=False, gpu_runs=0, gpu_queries=0, gpu_locks=0,
                candidate=candidate, pressure_points=cases(value), requests=requests,
                denominators=denominator_summary({}), preregistered_denominators=value["pressure"]["denominators"],
                limits=LIMITS, blockers=value["blockers"], faults=value["faults"], claim_limits=value["claim_limits"],
                pressure_comparison=None, inputs_sha256=sha(BINDING_PATH))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "plan"))
    args = parser.parse_args()
    result = prepared()
    if args.mode == "check":
        result = dict(status="cpu_plan_checked_execution_blocked", gpu_queries=0, gpu_runs=0,
                      planned_requests=result["denominators"]["scheduled"],
                      logical_entry_bounds=[r["logical_positive_payload_entry_bound"] for r in result["pressure_points"]],
                      launch_entry=False)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
