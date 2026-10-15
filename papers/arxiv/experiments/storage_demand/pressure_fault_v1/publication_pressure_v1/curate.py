#!/usr/bin/env python3
"""Read-only, CPU-only curation of the sealed pressure run; never imports a runner."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

import verify

HERE = Path(__file__).resolve().parent
RUN = "artifacts-go-v2-20261015-02"
MANIFEST_SHA256 = "c570b05aa5c6ffc63c588c7e78441728cf1c1db9119e83a1a9a3c813efb7a283"


def pick(value, keys):
    return {key: value[key] for key in keys.split()}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


def typed_fields(fields):
    fields = dict(fields)
    for key in ("expert_read_pipeline", "expert_read_pipeline_delta", "summary_json"):
        if key in fields:
            fields[key] = json.loads(fields[key])
    for key in ("generated_tokens", "tokens_reconcile"):
        if key in fields:
            value = fields[key]
            verify.require(value == "None" or re.fullmatch(r"Some\((\d+|true|false)\)", value),
                           "unrecognized Rust Option field")
            fields[key] = None if value == "None" else json.loads(value[5:-1])
    return fields


def idle_projection(value):
    result = pick(value, "status all_eight_idle required_consecutive accepted_sample_indices "
                  "consecutive_idle physical_owner_release mode")
    result.update({key: value[key] for key in ("authenticated_echild", "recorded_owners_absent")
                   if key in value})
    result["samples"] = []
    for sample in value["samples"]:
        row = pick(sample, "monotonic_started monotonic_finished classification consecutive_idle")
        row["no_compute_apps"] = not sample["apps"].strip()
        row["devices"] = []
        for line in sample["gpu"].splitlines():
            index, uuid, memory, utilization = [part.strip() for part in line.split(",")]
            verify.require(memory.endswith(" MiB") and utilization.endswith(" %"), "idle units")
            row["devices"].append(dict(index=int(index), memory_mib=int(memory[:-4]),
                                       utilization_percent=int(utilization[:-2])))
        result["samples"].append(row)
    return result


def curate(source):
    source = source.resolve()
    verify.require(source.name == RUN, "only the completed, role-bound run is eligible")
    manifest_bytes = (source / "evidence-manifest.json").read_bytes()
    verify.require(sha(manifest_bytes) == MANIFEST_SHA256, "source manifest changed")
    manifest = json.loads(manifest_bytes)
    pins = manifest["pins_relative_to_directory"]
    verify.require(len(pins) == 244, "expected the sealed 244-file inventory")
    for name, digest in pins.items():
        path = (source / name).resolve()
        verify.require(path.is_relative_to(source), "manifest path outside source")
        verify.require(sha(path.read_bytes()) == digest, "source hash mismatch: " + name)
    verify.require((HERE / "source-manifest.json").read_bytes() == manifest_bytes,
                   "public source inventory must be the unchanged original")
    records = []
    supplemental = {}

    def load(name):
        verify.require(name in pins, "unmanifested source: " + name)
        return json.loads((source / name).read_bytes())

    def emit(kind, data, name, point=None, line=None):
        record = dict(schema="ferrule.pressure-public-record.v1", kind=kind, point=point,
                      source=dict(path=name, sha256=pins[name]), data=data)
        if line is not None:
            record["source"]["line"] = line
        records.append(record)

    def external(name):
        path = source.parent / name
        data = path.read_bytes()
        supplemental[name] = sha(data)
        return json.loads(data)

    final = load("pressure-final.json")
    registration = load("registration.json")
    review = load("candidate-review.json")
    roles = load("input-roles.json")
    scalars = load("point-scalars.json")
    cases = load("cases.json")
    study = pick(final, "status study_design_id limits trial_order denominators ended "
                 "issued_lifecycles completed_lifecycles actual_server_starts comparison_eligible "
                 "full_numerical_per_point prior_numerical_scope repetitions_per_point retries "
                 "replacements profiling_captures physical_nas_backend_bytes "
                 "universal_final_release_verified historical_resume_permitted claim_limits")
    plan = external("plan_v2/plan.json")
    verify.require(supplemental["plan_v2/plan.json"] == roles["predecessor_pins"][
        "papers/arxiv/experiments/storage_demand/pressure_fault_v1/plan_v2/plan.json"], "prespecified plan pin")
    study["fixed_options"] = plan["pressure"]["fixed"]
    study["prespecified_source_geometry"] = plan["pressure"]["source_geometry"]
    study["plan_source_sha256"] = supplemental["plan_v2/plan.json"]
    study["started"] = registration["started"]
    study["schedule"] = registration["schedule"]
    study["source_commit"] = scalars["source_commit"]
    study["candidate_identity"] = {key: val for key, val in review["candidate_identity"].items()
                                    if key != "qualification_directory"}
    study["compiled_source_relations"] = [
        {key: val for key, val in row.items() if key != "consumed_path"}
        for row in roles["compiled_source_relations"]]
    study["prior_middle_numerical_gates"] = review["prior_middle_numerical_gates"]
    study["new_point_full_numerical_verified"] = review["new_point_full_numerical_verified"]
    study["role_binding_sha256"] = roles["binding_sha256"]
    study["role_root_sha256"] = roles["role_root_sha256"]
    study["additional_source_files"] = {name: pins[name] for name in (
        "registration.json", "candidate-review.json", "input-roles.json", "point-scalars.json")}
    emit("study", study, "pressure-final.json")

    for point, case, scalar in zip(verify.POINTS, cases["case_variants"], scalars["points"]):
        prefix = point + "-blocking/"
        verify.require(case["id"] == scalar["point"] == point, "fixed point order")
        config = pick(case, "requested_entries requested_cache_bytes read_options expected_scheduler "
                      "source_preflight full_numerical_per_point")
        emit("configuration", config, "cases.json", point)
        emit("source_options", load(prefix + "source-options.json"), prefix + "source-options.json", point)
        for line_no, line in enumerate((source / prefix / "server.log").read_text().splitlines(), 1):
            if line.startswith("{"):
                row = json.loads(line)
                row["fields"] = typed_fields(row["fields"])
                # The listening URL is a local deployment coordinate, not evidence used here.
                if row["fields"].get("message") == "Ferrule OpenAI server listening":
                    continue
                emit("server_event", row, prefix + "server.log", point, line_no)
            elif line.startswith("expert read host staging:"):
                emit("host_staging_report", line, prefix + "server.log", point, line_no)
        lifecycle = load(prefix + "lifecycle.json")
        assessment = load(prefix + "assessment.json")
        attempt = load(prefix + "attempt.json")
        lifecycle_public = pick(lifecycle, "status trial started_monotonic ready startup_seconds "
                                "ended requests_nfs startup_nfs shutdown_nfs")
        lifecycle_public.update(admission=attempt["admission"],
                                coordinator_ended=assessment["coordinator_ended"],
                                summary=assessment["summary"], closure=lifecycle["closure"]["final"],
                                owned_gpu_attribution=lifecycle["resources"]["owned_gpu_attribution"])
        emit("lifecycle", lifecycle_public, prefix + "lifecycle.json", point)
        supervisor = load(prefix + "supervisor.json")
        custody = pick(supervisor, "returncode timed_out wall_seconds cleanup_errors")
        custody["custody"] = pick(supervisor["custody"], "status method containment_verified "
                                  "errors reporting_errors echild")
        custody["entries"] = [pick(row, "role reaped exit_code signals") for row in supervisor["custody"]["entries"]]
        emit("custody", custody, prefix + "supervisor.json", point)
        for name in ("coordinator-pre-idle", "worker-pre-idle", "coordinator-post-idle"):
            emit(name, idle_projection(load(prefix + name + ".json")), prefix + name + ".json", point)
        for name in ("locks-released", "locks-reprobed"):
            emit(name, load(prefix + name + ".json"), prefix + name + ".json", point)
        for label in ("ready", "after-requests"):
            name = prefix + label + "-library-evidence/validation.json"
            raw = load(name)
            value = pick(raw, "status required_pin_check errors missing_required_paths "
                         "deleted_library_paths observed_library_count hash_scope live_mapped_memory_sha256 "
                         "ignored_deleted_nonlibraries")
            value["libraries_by_basename"] = {Path(path).name: digest for path, digest in raw["libraries"].items()}
            value["phase"] = label
            emit("library_validation", value, name, point)
        for name in verify.NAMES:
            path = prefix + name + "/observed.json"
            raw = load(path)
            value = {key: val for key, val in raw.items() if key not in ("logical_counters", "process_io")}
            terminal = load(prefix + name + "/http-terminal.json")
            verify.require(value["actual"] == terminal, "terminal receipt differs from observed HTTP")
            report = load(prefix + name + "/http.json")
            verify.require(report["status"] == "complete", "HTTP report is not complete")
            value["http_report_status"] = report["status"]
            value["terminal_source_sha256"] = pins[prefix + name + "/http-terminal.json"]
            emit("request", value, path, point)
        for label in ("samples", "phase-timings"):
            name = prefix + label + ".jsonl"
            for line_no, line in enumerate((source / name).read_text().splitlines(), 1):
                value = json.loads(line)
                if label == "samples":
                    if value.get("kind") != "sample":
                        continue
                    gpu, host = value["gpu"], value["host"]
                    value = dict(monotonic=value["monotonic"], sample_end_monotonic=value["sample_end_monotonic"],
                                 gpu=pick(gpu, "index total_bytes gpu_bytes utilization"),
                                 host=pick(host, "status rss_bytes pss_bytes"),
                                 ownership=pick(gpu["ownership"], "status owned_gpu_attribution "
                                                "performance_basis absence_of_foreign_interference_proven"))
                emit("memory_sample" if label == "samples" else "phase_timing", value, name, point, line_no)
        for label in ("prelaunch", "ready", "requests-pre", "requests-post", "shutdown-pre", "shutdown-post",
                      *(name + suffix for name in verify.NAMES for suffix in ("-pre", "-post"))):
            name = prefix + label + "-mountstats.json"
            raw = load(name)
            mounts = [row for row in raw["mounts"] if row["fstype"].startswith("nfs")]
            verify.require(len(mounts) == 1, "one observed shared NFS client mount required")
            value = dict(phase=label, captured_monotonic=raw["captured_monotonic"],
                         mount_label="shared-checkpoint-client", **pick(mounts[0], "fstype counters errors"))
            emit("nfs_snapshot", value, name, point)
        reference = pick(scalar, "point_elapsed_seconds launch_to_ready_seconds invocation_counter_totals "
                         "final_owner_cache final_owner_read memory_sample_bounds nfs_request_phase")
        reference["requests"] = [pick(row, "name status metric_eligible quality_passed http_e2e_seconds "
                                      "client_ttft_seconds usage generated_ids worker_id http_ids exact_token_pass "
                                      "helper_window helper_timings") for row in scalar["requests"]]
        emit("scalar_reference", reference, "point-scalars.json", point)

    emit("handoff", load("final-handoff.json"), "final-handoff.json")
    emit("handoff_idle", idle_projection(load("final-handoff-idle.json")), "final-handoff-idle.json")
    locks = load("final-handoff-locks.json")
    emit("handoff_locks", dict(all_eight_reacquired=locks["all_eight_reacquired"],
                               lock_count=len(locks["locks"])), "final-handoff-locks.json")
    # Historical evidence is never included in the successful run's measurements.
    history = dict(schema="ferrule.pressure-public-history.v1", pooled=False,
                   historical_helper_durations_seconds=None,
                   dialogue_only_no_run="Not execution evidence; no measured zero or fabricated receipt")
    history["censored_study"] = dict(
        study="artifacts-go-20261015-01",
        stop=external("artifacts-go-20261015-01/study-stop.json"),
        analysis=external("plan_v2/root-analysis.json"))
    blocked = external("artifacts-go-v2-20261015-01/blocked-start.json")
    history["blocked_start_study"] = dict(
        study="artifacts-go-v2-20261015-01",
        stop=external("artifacts-go-v2-20261015-01/study-stop.json"),
        observation=pick(blocked, "denominators comparison_eligible error failure_path server_starts "
                         "registration_created candidate_validation_reached full_numerical_per_point"))
    history["supplemental_source_sha256"] = {name: digest for name, digest in supplemental.items()
                                              if name != "plan_v2/plan.json"}
    summary = verify.derive(records)
    verify.require(summary["study_elapsed_seconds"] == scalars["study_elapsed_to_final_seconds"],
                   "study elapsed differs from sealed scalars")
    raw = b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode() for row in records)
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as stream:
        stream.write(raw)
    outputs = {"records.jsonl.gz": buffer.getvalue(), "summary.json": json_bytes(summary),
               "history.json": json_bytes(history)}
    for name, data in outputs.items():
        (HERE / name).write_bytes(data)
    inventory = dict(schema="ferrule.pressure-public-manifest.v1", study=RUN,
                     source_manifest_sha256=MANIFEST_SHA256, source_files_verified=244,
                     supplemental_source_sha256=supplemental,
                     verification_scope="All source hashes checked during curation; public projections are not original bytes or independent attestation",
                     files={name: dict(sha256=sha(data), bytes=len(data)) for name, data in outputs.items()},
                     records=dict(count=len(records), uncompressed_bytes=len(raw), uncompressed_sha256=sha(raw)))
    (HERE / "manifest.json").write_bytes(json_bytes(inventory))
    print(json.dumps(dict(status="curated", source_files_verified=244, records=len(records),
                          uncompressed_bytes=len(raw), gzip_bytes=len(buffer.getvalue()),
                          requests=summary["denominators"]["successful"], tokens=summary["generated_tokens"]), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE.parent / RUN)
    args = parser.parse_args()
    curate(args.source)


if __name__ == "__main__":
    main()
