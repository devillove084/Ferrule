"""Verify the curated qualification using Python's standard library only.

No subprocess, runtime/model imports, network, CUDA calls, locks, or environment
capture. This checks historical receipts, not current machine state or serving.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

COMMIT = "033519c3def098ce417e4b6ff0f141a9f30bddb1"
SOURCE = "a4dceb084e21b80f6b45ff7ac980c6fa80ac8f17e902fd28c65f3d5bc7bb0efa"
SOURCE_REL = "bde8b14396855df4cf832cb89d01250f2b4e162ef7ed9b3abc93d2c4df975fa7"
SOURCE_ARCHIVE = "ac072e0d1bac7efcc85b7f26ce68723f412b0e22b5ee868104b1f6073dd2a82f"
SEAL = "0b6311f25951d4bb89724cf53caa57d47475dab6c46b70025d37f451c30b7ceb"
PLAN = "f83ab450578b574c25fa48d4039f66f6df1a91725d586e1c029b7f0b1f7a066d"
CANDIDATES = {
    "cli": "de5159951fe724e97e35c0fc86367642d33735994328eee1a5e604cf9d2b037b",
    "constructor": "cb9b8c998449b8f73a6a6919be19bcd8e1d743711a1c06d36e5332af57db4a23",
    "composition": "8810c3418f4b39b4c9203003b91e53a6106a833624db4d13a516fdfd63453d6a",
}
REFERENCES = {
    "full40": "351495921233e0158eb2a4507c415d173c895a6086d5905b7a72d82de71849f9",
    "fresh": "102c798baeb5200bc8ed8386eabed6146870b3b6da2a90fc6010677041352c7e",
}
GATES = ("01-constructor", "02-blocking-full40", "03-lookahead-full40",
         "04-blocking-fresh", "05-lookahead-fresh", "06-blocking-ordinary",
         "07-lookahead-ordinary", "08-blocking-cold", "09-lookahead-cold")
# metrics, main logit values, fingerprints, positive prediction calls, samples,
# last sampled owner-lifetime verified reads, last sampled lookahead takes.
COUNTS = {
    "constructor": (0, 128, 0, 4, 4, 6, None),
    "full40": (804, 1986560, 0, 4, 128, 2089, 1929),
    "fresh": (5427, 9187840, 64, 64, 963, 25116, 22556),
    "ordinary": (27, 9187840, 64, 64, 66, 25116, 22556),
    "cold": (1809, 3228160, 28, 28, 369, 11354, 10234),
}
CHARGE = "logical_host_reservation_not_physical_rss"
STATS = "owner_lifetime_owner_observed_transitions_peaks_not_invocation_deltas"
READER = "shared_checkpoint_reader_submit_through_take_not_ticket_certified"
RUNTIME_ENV = set((
    "PATH HOME CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES CUBLAS_WORKSPACE_CONFIG "
    "LD_LIBRARY_PATH FERRULE_PROFILE_NVTX FERRULE_NUMERIC_BF16_FACTORY_GPU "
    "FERRULE_NUMERIC_BF16_MODEL FERRULE_NUMERIC_BF16_REFERENCE_DIR "
    "FERRULE_NUMERIC_BF16_REFERENCE_SHA256 FERRULE_NUMERIC_BF16_REPORT_DIR "
    "FERRULE_BF16_FRESH_DIR FERRULE_BF16_FRESH_SHA256 FERRULE_BF16_FRESH_REPORT "
    "FERRULE_BF16_FRESH_PRIMARY_REPORT FERRULE_BF16_FRESH_OBSERVED_REPORT "
    "FERRULE_BF16_FRESH_OBSERVED_REPORT_SHA256 FERRULE_EXPERT_READ_CONSTRUCTOR_REPORT"
).split())
BUILD_ENV = set((
    "CARGO_TARGET_DIR CARGO_HOME RUSTUP_HOME RUSTUP_TOOLCHAIN RUSTUP_AUTO_INSTALL "
    "CARGO_BUILD_JOBS CARGO_INCREMENTAL CUBLAS_WORKSPACE_CONFIG CUDA_HOME NVCC "
    "FERRULE_CUDA_ARCH FERRULE_CUDA_TEST_ORACLE FERRULE_CUTLASS_DIR "
    "CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES LD_LIBRARY_PATH LIBRARY_PATH PATH "
    "PYTHONDONTWRITEBYTECODE HOME CC CXX CUDAHOSTCXX RUSTC RUSTDOC "
    "FERRULE_NVTX_INCLUDE_DIR FERRULE_PROFILE_NVTX TMPDIR"
).split())


class InvalidEvidence(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise InvalidEvidence(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def parse_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid(value):
        raise InvalidEvidence("Nonfinite JSON: " + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


class Archive:
    def __init__(self, root):
        self.root = Path(root).resolve()
        require(not (self.root / "manifest.json").is_symlink(), "Symlink manifest")
        self.manifest = parse_json((self.root / "manifest.json").read_bytes())
        self.entries = {}
        for entry in self.manifest["entries"]:
            path = PurePosixPath(entry["path"])
            require(not path.is_absolute() and ".." not in path.parts
                    and path.parts[0] == "receiptarchive", "Unsafe archive path")
            name = str(path.relative_to("receiptarchive"))
            if entry["encoding"] == "gzip":
                require(name.endswith(".gz"), "Gzip name mismatch")
                name = name[:-3]
            require(name not in self.entries, "Duplicate archive record")
            self.entries[name] = entry
        self.cache = {}

    def raw(self, name):
        require(name in self.entries, "Missing report/receipt: " + name)
        entry = self.entries[name]
        path = self.root / entry["path"]
        require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
                "Missing or symlink record: " + name)
        stored = path.read_bytes()
        require(len(stored) == entry["stored_bytes"] and sha(stored) == entry["stored_sha256"],
                "Stale stored hash: " + name)
        require(entry["encoding"] in ("identity", "gzip"), "Unknown encoding")
        content = gzip.decompress(stored) if entry["encoding"] == "gzip" else stored
        require(len(content) == entry["content_bytes"] and sha(content) == entry["content_sha256"],
                "Stale content hash: " + name)
        return content

    def load(self, name):
        if name not in self.cache:
            self.cache[name] = parse_json(self.raw(name))
        return self.cache[name]

    def content_sha(self, name):
        return sha(self.raw(name))

    def integrity(self):
        m = self.manifest
        require(m["schema"] == "ferrule.storage-demand.qualification-curation.v1", "Wrong manifest schema")
        require(m["original_evidence_manifest_sha256"] == SEAL
                and m["original_sealed_files"] == 1772
                and m["original_files_including_seal"] == 1773, "Wrong original seal/count")
        require(m["source_commit"] == COMMIT and m["source_digest"] == SOURCE
                and m["source_archive_sha256"] == SOURCE_ARCHIVE, "Stale source/cut")
        require(m["candidate_sha256"] == CANDIDATES and tuple(m["gates"]) == GATES,
                "Wrong candidates/nine-gate inventory")
        require(m["physical_release"] == "unobserved" and m["cross_stream_sums"] is False,
                "Invalid release or pooled denominator claim")
        require(m["qualification_scope"] == "same-source CLI candidate quality, not same-binary CLI serving qualification",
                "Invalid CLI serving qualification claim")
        require(self.content_sha("evidence-manifest.json") == SEAL, "Original seal changed")
        seal = self.load("evidence-manifest.json")["pins_relative_to_directory"]
        require(len(seal) == 1772 and len(self.entries) == 240, "Missing/extra evidence inventory")
        refs = self.load("reference-pins.json")["reference_files"]
        actual = {str(p.relative_to(self.root)) for p in (self.root / "receiptarchive").rglob("*") if p.is_file()}
        require(actual == {e["path"] for e in self.entries.values()}, "Missing/unlisted archive file")
        for name, entry in self.entries.items():
            content = self.raw(name)
            original = entry["original_relpath"]
            prefix = m["original_root"] + "/"
            if original.startswith(prefix):
                rel = original[len(prefix):]
                expected = SEAL if rel == "evidence-manifest.json" else seal.get(rel)
            else:
                expected = next((h for p,h in refs.items() if p.endswith("/" + original)), None)
            require(expected is not None and entry["original_sha256"] == expected,
                    "Unsealed/stale copy input: " + name)
            if entry["transform"] == "identity":
                require(sha(content) == expected and len(content) == entry["original_bytes"],
                        "Copy differs from original: " + name)
            else:
                require(entry["transform"] == "remove_worker_token_only"
                        and name.endswith("/worker-launch.public.json")
                        and "worker_token" not in parse_json(content), "Invalid credential projection")
        fields = ("gate_id", "status", "exact_test_passed", "constructor_cells", "constructor_finite_logit_values",
                  "main_report_metrics", "main_report_logit_values", "actual_fingerprint_calls", "positive_worker_read_calls",
                  "sampled_read_operations", "report_sha256", "authenticated_echild", "recorded_pid_birth_owners_absent", "natural_exit")
        require(m["gate_checks"] == [{k:r[k] for k in fields} for r in self.load("receipt-audit.json")["gates"]],
                "Manifest nine-gate counts/checks differ from sealed audit")
        external = m["external_artifacts"]
        candidates = {r["role"]:r for r in external if r["role"] in CANDIDATES}
        require(set(candidates) == set(CANDIDATES) and len([r for r in external if r["role"] in CANDIDATES]) == 3,
                "External candidate inventory changed")
        plan = self.load("qualification-plan-v2.json")
        for role, record in candidates.items():
            artifact = plan["artifacts"][role]
            require(record["sha256"] == CANDIDATES[role] and record["path"] == artifact["executable"]
                    and record["bytes"] == artifact["bytes"] and record["archived"] is False,
                    "External candidate path/hash changed")
        sources = [r for r in external if r["role"] == "committed_source_archive"]
        require(len(sources) == 1 and sources[0]["sha256"] == SOURCE_ARCHIVE and sources[0]["archived"] is False,
                "External source archive hash changed")
        payloads = {r["path"]:r["sha256"] for r in external if r["role"] == "reference_tensor_payload"}
        require(payloads == {p:h for p,h in refs.items() if not p.endswith(".json")}
                and all(r["archived"] is False for r in external), "External tensor-reference pins changed")
        require(sum(e["stored_bytes"] for e in self.entries.values()) < 20 * 1024 * 1024,
                "Archive exceeds textual-receipt size budget")


def check_source(source, members):
    require(source["commit"] == members["commit"] == COMMIT, "Stale source commit")
    require(source["archive_sha256"] == members["archive_sha256"] == SOURCE_ARCHIVE, "Stale source archive")
    require(source["file_count"] == members["file_count"] == 1484, "Source member count")
    relative = source["pins_relative_to_frozen_source"]
    require(len(relative) == 1484 and {p:v["sha256"] for p,v in members["members"].items()} == relative,
            "Source/member pins differ")
    require(digest(relative) == source["source_pins_digest_relative"] == SOURCE_REL
            and digest(source["original_source_pins"]) == source["source_pins_digest"] == SOURCE,
            "Stale source-pin digest")
    require(source["immutable_readonly_inputs"] is True and source["worktree_edits_excluded"] is True
            and members["exact_equality_to_frozen_source_pins"] is True, "Source freeze not exact/read-only")


def check_candidates(plan, artifacts, environment):
    require(plan["commit"] == COMMIT and plan["source_digest"] == SOURCE, "Plan source changed")
    require(set(artifacts) == set(CANDIDATES), "Candidate role set")
    require(set(environment) == BUILD_ENV and environment["FERRULE_CUDA_ARCH"] == "sm_86"
            and environment["FERRULE_PROFILE_NVTX"] == "0"
            and environment["RUSTUP_TOOLCHAIN"] == "nightly-2026-08-01", "Build environment changed/unapproved")
    for role, expected in CANDIDATES.items():
        a = artifacts[role]
        require(a == plan["artifacts"][role] and a["sha256"] == expected
                and a["source_digest"] == SOURCE, "Stale candidate: " + role)
        require(a["nvtx_compiled"] is True and a["qualification_nvtx_runtime_enabled"] is False
                and a["build_adapter_v1_accepted"] is False, "Candidate/legacy adapter scope")
        require(all("nvtx-profile" in c["features"] for c in a["common_artifacts"]), "Common NVTX feature missing")
    require(len(set(CANDIDATES.values())) == 3, "Different executables conflated")


def check_counts(final, plan, audit):
    require(tuple(g["id"] for g in plan["gates"]) == GATES
            and tuple(g["gate_id"] for g in final["results"]) == GATES
            and tuple(g["gate_id"] for g in audit["gates"]) == GATES, "Nine exact-once gates missing/duplicated/pooled")
    require(final["status"] == audit["status"] == "passed" and final["scheduled_gates"] == 9
            and final["actual_gpu_tests_issued"] == 9 and final["all_nine_qualified"] is True
            and final["not_issued_gates"] == [], "Nine-gate final PASS incomplete")
    require(all(final[k] == 0 for k in ("retries", "conditioning_probes", "warmups", "profiling_captures")),
            "Unexpected retry/probe/warmup/profile")
    require(final["commit"] == audit["commit"] == COMMIT
            and final["source_digest"] == SOURCE and final["source_archive_sha256"] == SOURCE_ARCHIVE
            and final["qualification_plan_sha256"] == audit["qualification_plan_sha256"] == PLAN
            and final["artifact_sha256"] == audit["artifact_sha256"] == CANDIDATES, "Final source/candidate/plan changed")
    require(final["binary_math_proof"] is False and audit["no_cross_stream_denominator_sums"] is True,
            "Invalid binary equivalence/denominator claim")
    require(final["physical_release"].startswith("unobserved")
            and audit["physical_release"].startswith("Unobserved"), "Physical release not unknown")
    for row, gate in zip(final["results"], plan["gates"]):
        require(row["status"] == "passed" and row["dispatched"] == row["expected_exact_tests"] == 1
                and row["actual_test_issued"] is True and row["retries"] == 0,
                "Gate not issued exactly once: " + gate["id"])
        require(row["selector"] == gate["selector"] and row["binary_sha256"] == gate["binary_sha256"]
                == CANDIDATES[gate["role"]] and row["source_digest"] == SOURCE, "Gate source/candidate changed")
        require(all(row[k] is True for k in ("report_present", "report_complete", "report_passed", "numerical_pass",
                                             "authenticated_drained", "echild", "recorded_owners_absent", "natural_exit"))
                and row["returncode"] == 0 and row["timed_out"] is False and row["failures"] == []
                and row["failure_classification"] is None, "Failed/incomplete gate")
        require(row["physical_release"].startswith("unobserved"), "Gate physical release not unknown")


def check_test(stdout, listed, selector):
    require(listed["returncode"] == 0 and listed["stderr"] == ""
            and [s.strip() for s in listed["stdout"].splitlines() if s.strip()]
            == [selector + ": test", "1 test, 0 benchmarks"], "Not exactly one selected test")
    require(len(re.findall(r"^test result:", stdout, re.M)) == 1
            and re.search(r"^test result: ok\. 1 passed; 0 failed; 0 ignored; 0 measured; \d+ filtered out;", stdout, re.M)
            and re.search(r"^test " + re.escape(selector) + r" \.\.\. ok$", stdout, re.M),
            "Skipped/ignored/failed test or false PASS")


def check_custody(supervisor, launch, worker, custody):
    require(supervisor["returncode"] == 0 and supervisor["timed_out"] is False
            and supervisor["cleanup_errors"] == [] and supervisor["descendant_custody"] == "proven_drained"
            and supervisor["stdout_truncated"] is False and supervisor["stderr_truncated"] is False,
            "Supervisor completion not natural/complete")
    c = supervisor["custody"]
    require(c == custody and c["status"] == "proven_drained" and c["method"] == "linux-subreaper-waitid"
            and c["containment_verified"] is True and c["echild"] is True
            and c["errors"] == c["reporting_errors"] == [], "Authenticated ECHILD/custody missing")
    identity = launch["identity"]
    roots = [e for e in c["entries"] if e["role"] == "root"]
    require(len(roots) == 1 and roots[0]["identity"] == c["root_identity"], "Custody root differs")
    require(all(identity["worker"][k] == c["root_identity"][k] for k in ("pid", "birth_ticks", "ppid"))
            and identity["worker"]["pgrp"] == identity["worker"]["session"] == identity["worker"]["pid"],
            "PID/birth/parent/fresh worker session differs")
    require(identity["launch_id"] == c["launch_id"] == supervisor["handshake"]["launch_id"]
            and identity["helper"] == c["helper_identity"] == supervisor["handshake"]["helper_identity"]
            and identity["coordinator"] == worker["coordinator"], "Launch authentication identity differs")
    require(all(e["signals"] == [] and e["reaped"] is True and e["exit_code"] == 0 for e in c["entries"]),
            "Signaled/unreaped/failed owner")
    require(launch["exec_preserves_authenticated_root_pid_birth"] is True
            and launch["all_eight_uuid_locks_validated"] is True, "Launch/root/lock validation missing")


def check_idle(observation, inventory, strict_pre=False):
    require(observation["schema"] == "ferrule.gpu-idle-window.v2" and observation["status"] == "verified"
            and observation["required_consecutive"] == 3 and observation["deadline_seconds"] == 15
            and observation["physical_owner_release"] == "unverified", "Idle protocol/release changed")
    samples = observation["samples"]
    indices = observation["accepted_sample_indices"]
    require(len(indices) == 3 and indices == list(range(indices[0], indices[0] + 3))
            and indices[-1] == len(samples) - 1, "Missing consecutive final idle window")
    if strict_pre:
        require(len(samples) == 3, "Preflight conditioning is not allowed")
    for i, sample in enumerate(samples):
        require(sample["classification"] in ("idle", "utilization_only_transient") and sample["apps"] == "",
                "Compute app/memory/inventory idle failure")
        lines = [line.split(", ") for line in sample["gpu"].strip().splitlines()]
        require(len(lines) == 8 and [line[1] for line in lines] == inventory
                and [line[0] for line in lines] == [str(n) for n in range(8)]
                and all(line[2] == "0 MiB" for line in lines), "All-eight inventory/memory mismatch")
        if i in indices:
            require(sample["classification"] == "idle" and all(line[3] == "0 %" for line in lines),
                    "Accepted idle utilization not zero")
    if observation["mode"] == "post_authenticated_echild":
        require(observation["authenticated_echild"] is True and observation["recorded_owners_absent"] is True,
                "Post-idle ECHILD/owner absence unknown")


def check_pipeline(o, limits, effective=False):
    require(o["enabled"] is True and o["limits"] == limits and o["requested_mode"] == limits["schedule"]
            and o["effective_mode"] in (None, limits["schedule"]), "Pipeline requested/effective mode changed")
    if effective:
        require(o["effective_mode"] == limits["schedule"] and o["worker_read_observed"] is True,
                "Effective mode/positive worker read unobserved")
    require(o["charge_basis"] == CHARGE and o["stats_scope"] == STATS and o["reader_counter_scope"] == READER,
            "Logical charge/RSS or counter scope changed")
    payload = limits["max_host_bytes"] - limits["max_metadata_bytes"] - limits["worker_stack_bytes"]
    require(payload >= 0, "Invalid logical host cap")
    bounds = {"packed_bytes": payload // 2, "artifact_bytes": payload // 2, "payload_bytes": payload,
              "metadata_bytes": limits["max_metadata_bytes"], "requested_stack_bytes": limits["worker_stack_bytes"],
              "host_charged_bytes": limits["max_host_bytes"]}
    for key, bound in bounds.items():
        require(type(o["current_charge"][key]) is int and 0 <= o["current_charge"][key] <= bound
                and type(o["stats"]["peak_" + key]) is int and 0 <= o["stats"]["peak_" + key] <= bound,
                "Logical host charge exceeds bounded component: " + key)
    require(o["current_charge"]["reserved_read_slots"] <= 2
            and all(o["current_charge"][k] == 0 for k in ("unknown_packed_bytes", "unknown_stack_bytes", "unknown_read_slots"))
            and all(o["stats"][k] == 0 for k in ("unknown_reads", "unknown_bytes", "unobserved_terminal_reads",
                                                 "unobserved_reader_intervals", "failed_reads", "demand_failures")),
            "Ambiguous/failed read charge")
    if effective:
        r = o["last_receipt"]
        require(r is not None and r["outcome"] == "source_verified" and r["io_bytes"] == r["requested_bytes"] > 0,
                "Positive numerical source read not proved")
        times = [r[k] for k in ("submitted_ns", "worker_started_ns", "io_started_ns", "io_finished_ns",
                                "source_verified_ns", "completed_ns", "taken_ns")]
        require(all(type(t) is int for t in times) and times == sorted(times), "Worker/IO/source timestamps unknown")


def check_metrics(metrics, count):
    require(len(metrics) == count and all(m["failures"] == 0 and m.get("nonfinite", 0) == 0 for m in metrics),
            "Numerical metric count/failures")
    for metric in metrics:
        for key in ("max_abs", "rms", "max_tolerance_ratio"):
            if key in metric:
                require(math.isfinite(metric[key]) and metric[key] >= 0, "Nonfinite metric")
        if "max_tolerance_ratio" in metric:
            require(metric["max_tolerance_ratio"] <= 1, "Tolerance exceeded")


def check_report(report, gate):
    kind = gate["kind"]
    metrics, logits, fingerprints, positive, samples, reads, takes = COUNTS[kind]
    require(report["passed"] is True and report["complete"] is True, "Report incomplete/failed")
    if kind == "constructor":
        require(report["expected_cells"] == len(report["cells"]) == 4
                and report["total_logit_values_checked"] == 128
                and report["route_checks"] is None and report["state_oracle_checks"] is None,
                "Constructor cells/logits/scope changed")
        cells = report["cells"]
        require(len({(c["model_boundary"], c["requested"]["schedule"]) for c in cells}) == 4,
                "Duplicate constructor boundary/schedule cell")
        for c in cells:
            require(c["passed"] is True and c["positive_worker_read"] is True
                    and c["logit_values_checked"] == 32 and c["delta"]["verified_reads"] == 6,
                    "Tiny constructor positive read/finite logit evidence missing")
            check_pipeline(c["configured"], c["requested"])
            check_pipeline(c["observed"], c["requested"], effective=True)
            for endpoint in ("before_prediction", "after_prediction"):
                require(c[endpoint]["release"]["state"] == "unobserved", "Constructor release claim")
        return {"gate": gate["id"], "constructor_cells": 4, "finite_logit_values": 128}
    require(report["failures"] == [] and report["quarantine"] is False and report["state_bytes_after_shutdown"] == 0,
            "Report failures/quarantine/logical shutdown")
    require(report["atol"] == 0.00025 and report["rtol"] == 0.012
            and report["greedy_tie_policy"] == "lower_token_id", "Tolerance/greedy policy changed")
    check_metrics(report["metrics"], metrics)
    require(report["logits_compared"] == logits, "Main logit denominator changed")
    ordinary = kind == "ordinary"
    scope = report["observation_scope"]
    require(scope["diagnostic_trace_installed"] is (not ordinary)
            and scope["ordinary_no_observer_claim"] is ordinary
            and scope["gpu_overlap_claim"] is False and scope["concurrent_scheduler_claim"] is False,
            "Observer/concurrency/overlap claim changed")
    config = report["test_config"]
    require(config["numeric_precision"] == "Bf16ModelMixedF32Accumulate"
            and config["storage_policy"] == "E4M3FnBf16Scales" and config["topology"] == {"tp": 1, "pp": 1, "ep": 1},
            "Numerical source profile/topology changed")
    capacities = config["capacity_and_options"]
    for key, value in {"max_bytes": 335544320, "max_experts": 64, "scratch_bytes": 8388608,
                       "device_bytes": 7516192768, "workspace_bytes": 2147483648, "kv_bytes": 16777216,
                       "ctx_size": 128, "max_batch_tokens": 8, "max_sequences": 2}.items():
        require(capacities[key] == value, "Capacity changed: " + key)
    require(capacities["scheduler"]["prefill_chunk_size"] == 8
            and capacities["scheduler"]["decode_cohort_target"] == 1
            and capacities["scheduler"]["decode_cohort_max_deferrals"] == 0, "Scheduler config changed")
    effective = report["effective_config"]
    require(effective["assertions_passed"] is True and effective["effective"]["max_layers"] == 40,
            "Effective factory config missing")
    peak = report.get("actual_peak_reserved_bytes", report.get("peak_reserved_bytes"))
    require(peak == 4106223616 <= report["device_budget_bytes"] == 7516192768, "Allocator observation/budget changed")
    q = report["expert_read_quality"]
    limits = {"max_host_bytes": 67108864, "max_metadata_bytes": 8388608,
              "worker_stack_bytes": 2097152, "schedule": gate["schedule"]}
    require(q["passed"] is True and q["charge_basis"] == CHARGE and q["requested"] == limits
            and q["positive_worker_read_calls"] == positive and q["sampled_operations"] == len(q["calls"]) == samples,
            "Read call/sample/mode counts changed")
    require(sum(c["positive_worker_read"] is True for c in q["calls"]) == positive,
            "Positive worker read count differs from raw calls")
    owners = {}
    for c in q["calls"]:
        require(all(c[k] is True for k in ("configuration_matches", "within_charged_limits", "unambiguous",
                                         "same_owner", "evidence_complete")), "Read interval unknown/invalid")
        require(c["before"]["scope"] == c["after"]["scope"], "Different owners in read interval")
        for endpoint in ("before", "after"):
            p = c[endpoint]
            require(p["release"]["state"] == "unobserved", "Physical release must remain unobserved")
            require(p["expert_read_pipeline"]["state"] == "observed", "Pipeline endpoint unobserved")
            check_pipeline(p["expert_read_pipeline"]["value"], limits)
            owners[digest(p["scope"])] = p["expert_read_pipeline"]["value"]
        if c["positive_worker_read"]:
            require(c["operation"] == "prediction" and all(c["delta"][k] > 0 for k in
                    ("submitted_reads", "completed_reads", "verified_reads", "io_started_reads", "proved_read_bytes")),
                    "Positive worker read lacks prediction/source counter evidence")
    require(len(owners) == 1, "Unexpected sampled owner set")
    last = next(iter(owners.values()))
    check_pipeline(last, limits, effective=True)
    stats = last["stats"]
    require(stats["verified_reads"] == stats["worker_started_reads"] == reads
            and stats["prefetched_reads_taken"] == stats["speculative_reads"] == (takes if gate["schedule"] == "lookahead" else 0)
            and stats["peak_host_charged_bytes"] == 16777984, "Last sampled owner-lifetime counters changed")
    if kind == "full40":
        require(report["reference_manifest_sha256"] == REFERENCES["full40"] and report["diagnostic_capture_only"] is False,
                "Full40 reference/partial report changed")
        require(report["diagnostic_denominators"] == {"logit_values_compared": 1986560,
                "ordered_route_ids_compared": 2560, "ordinary_no_observer_prediction_calls": 0,
                "prediction_calls": 4, "state_metric_elements_compared": 66990080}, "Full40 diagnostic counts changed")
        require(len(report["next_token_checks"]) == 4 and all(c["equal"] is True and c["actual"] == c["expected"]
                for c in report["next_token_checks"]), "Full40 greedy IDs changed")
    else:
        require(report["oracle_manifest_sha256"] == REFERENCES["fresh"] and report["cold_owner_restart"] is (kind == "cold"),
                "Fresh/cold reference or owner mode changed")
        fp = report["prediction_fingerprints"]
        require(len(fp["calls"]) == fingerprints and fp["failures"] == [], "64/28 fingerprint coverage changed")
        for row in fp["calls"].values():
            require(re.fullmatch(r"[0-9a-f]{64}", row["f32_le_sha256"]) and type(row["rows"]) is int
                    and row["rows"] > 0 and row["values"] == row["rows"] * 248320
                    and type(row["next_token_id"]) is int and 0 <= row["next_token_id"] < 248320,
                    "Actual fingerprint words/shape/greedy ID missing")
        pd = report["prediction_denominators"]
        require(pd["no_cross_stream_metric_sums"] is True and set(pd["streams"]) ==
                {"common_history", "free_feedback", "reset_replay", "fork_replay"}, "Pooled/missing semantic streams")
        for name, stream in pd["streams"].items():
            calls, values = ((1, 248320) if name == "fork_replay" else (9, 3228160)
                             if name == "reset_replay" or kind == "cold" else (27, 9187840))
            require(stream["prediction_calls"] == stream["greedy_checks"] == stream["input_history_matched_reference_calls"] == calls
                    and stream["logit_values_compared"] == values and stream["failures"] == [], "Per-stream denominator changed: " + name)
            require(sum(m["elements"] for m in stream["metrics"]) == values, "Per-stream raw metrics do not cover values")
            check_metrics(stream["metrics"], calls)
        require(sum(label.count("/common_history/") for label in fp["calls"]) == pd["streams"]["common_history"]["prediction_calls"],
                "Fingerprint stream labels incomplete")
    return dict(gate=gate["id"], main_metrics=metrics, main_logit_values=logits, fingerprints=fingerprints,
                positive_prediction_calls=positive, sampled_operations=samples,
                last_sampled_owner_lifetime_verified_reads=reads,
                last_sampled_owner_lifetime_lookahead_takes=stats["prefetched_reads_taken"],
                logical_host_peak_bytes=16777984, allocator_peak_reserved_bytes=peak,
                physical_release="unobserved", counter_scope=STATS)


def check_primary(report, primary, worker, primary_hash, primary_path, kind):
    require(worker["environment"]["FERRULE_BF16_FRESH_PRIMARY_REPORT"] == primary_path
            and report["test_config"] == primary["test_config"]
            and report["effective_config"] == primary["effective_config"]
            and report["oracle_manifest_sha256"] == primary["oracle_manifest_sha256"],
            "Current same-schedule primary config/oracle binding changed")
    if kind == "ordinary":
        require(report["diagnostic_primary_report_sha256"] == primary_hash
                == worker["environment"]["FERRULE_BF16_FRESH_OBSERVED_REPORT_SHA256"]
                and worker["environment"]["FERRULE_BF16_FRESH_OBSERVED_REPORT"] == primary_path,
                "Stale current primary SHA/path")
        require(report["prediction_fingerprints"]["paired_exact_logits_and_greedy_equal"] is True
                and report["prediction_fingerprints"]["calls"] == primary["prediction_fingerprints"]["calls"],
                "Ordinary/diagnostic actual words/shape/greedy fingerprints differ")


def check_log_metrics(stderr, report, kind):
    expected_counts = {"constructor": 0, "full40": 644, "fresh": 4411, "ordinary": 91, "cold": 1477}
    expected = {m["label"]: m["elements"] for m in report.get("metrics", [])}
    for stream in report.get("prediction_denominators", {}).get("streams", {}).values():
        expected.update({m["label"]: m["elements"] for m in stream["metrics"]})
    lines = [line for line in stderr.splitlines() if ": n=" in line]
    require(len(lines) == expected_counts[kind], "Exact numerical stderr coverage changed")
    for line in lines:
        match = re.fullmatch(r"(.+): n=(\d+) max_abs=\S+ rms=\S+ max_tolerance_ratio=\S+ nonfinite=0 failures=0 first=None", line)
        require(match is not None and expected.get(match[1]) == int(match[2]),
                "Numerical stderr count/failure/label differs from typed report")
    # Ordered-route integer comparisons live in the typed report; not every
    # report metric prints an n= log line, so stderr is not the full denominator.


def check_final_cleanup(final, idle, inventory, held, released, probe):
    require(all(final[k] is True for k in ("final_authenticated_echild", "final_recorded_owners_absent",
                "all_eight_lock_contexts_exited", "all_eight_local_locks_released_and_reprobed"))
            and final["final_idle_status"] == "verified" and final["final_idle_samples"] == 3,
            "Final ECHILD/idle/lock cleanup incomplete")
    check_idle(idle, inventory)
    require(held["all_eight"] is True and held["local_only"] is True and held["global_reservation"] is False
            and released["context_exited"] is True and probe["all_eight_nonblocking_reacquired"] is True
            and probe["context_exits_after_receipt"] is True, "Final local UUID lock release/probe unknown")
    for receipt in (held, released, probe):
        require(len(receipt["locks"]) == 8 and [Path(r["path"]).stem for r in receipt["locks"]] == inventory
                and all(r["global_reservation"] is False and r["mode"] == "local_nonblocking_flock" for r in receipt["locks"]),
                "All-eight local lock inventory/scope changed")


def verify(root):
    a = Archive(root)
    a.integrity()
    plan, final, audit = (a.load(n) for n in ("qualification-plan-v2.json", "FINAL.json", "receipt-audit.json"))
    require(a.content_sha("qualification-plan-v2.json") == PLAN, "Plan seal changed")
    check_source(a.load("source-freeze.json"), a.load("archive-exact-members.json"))
    check_candidates(plan, {r:a.load("artifact-" + r + ".json") for r in CANDIDATES}, a.load("build-environment.json"))
    check_counts(final, plan, audit)
    require(a.content_sha("FINAL.json") == audit["final_receipt_sha256"], "Audit/final seal differs")
    refs = a.load("reference-pins.json")
    require(refs["bf16_atol"] == 0.00025 and refs["bf16_rtol"] == 0.012 and refs["route_policy"] == "lower_expert_id",
            "Reference tolerance/router policy changed")
    for family, expected in REFERENCES.items():
        require(a.content_sha("references/" + family + "-manifest.json") == expected, "Reference manifest changed")
    rows = []
    runtime_pins = a.load("runtime-library-pins.json")["pins"]
    for gate, final_row, audit_row in zip(plan["gates"], final["results"], audit["gates"]):
        prefix = "gates/" + gate["id"] + "/"
        result = a.load(prefix + "gate-result.json")
        require(result == final_row, "Producer gate-result/final differs")
        report_name = prefix + ("numeric-report/full40-bf16-profile.json" if gate["kind"] == "full40" else "quality.json")
        report = a.load(report_name)
        report_sha = a.content_sha(report_name)
        require(report_sha == result["report_sha256"] == audit_row["report_sha256"], "Producer/report/audit hashes differ")
        rows.append(check_report(report, gate))
        check_log_metrics(a.raw(prefix + "test.stderr").decode(), report, gate["kind"])
        check_test(a.raw(prefix + "test.stdout").decode(), a.load(prefix + "exact-list-receipt.json"), gate["selector"])
        launch = a.load(prefix + "authenticated-launch.json")
        worker = a.load(prefix + "worker-launch.public.json")
        require(worker["gate"] == gate and worker["qualification_plan_sha256"] == PLAN
                and worker["gate_id"] == launch["gate_id"] == gate["id"]
                and launch["executable_sha256"] == gate["binary_sha256"], "Worker/launch seal changed")
        require(launch["argv"] == [gate["executable"], gate["selector"], "--exact", "--ignored", "--nocapture", "--test-threads=1"],
                "Exact test invocation changed")
        env = a.load(prefix + "runtime-environment.json")
        require(env == launch["environment"] == worker["environment"] and set(env) <= RUNTIME_ENV
                and env["FERRULE_PROFILE_NVTX"] == "0", "Runtime environment changed/unapproved/secret")
        check_custody(a.load(prefix + "supervisor.json"), launch, worker, a.load(prefix + "custody.json"))
        for name in ("pre-idle", "authenticated-pre-cuda-idle", "post-idle"):
            check_idle(a.load(prefix + name + ".json"), plan["uuid_inventory"], name != "post-idle")
        mapped = set()
        for sample in a.load(prefix + "runtime-mapping-observations.json")["samples"]:
            require(all(sample["identity"][k] == launch["identity"]["worker"][k] for k in ("pid", "birth_ticks")),
                    "Runtime mapping PID/birth owner unknown")
            for line in sample["maps"].splitlines():
                parts = line.split(None, 5)
                if len(parts) == 6 and parts[-1].startswith("/") and ".so" in parts[-1] and not parts[-1].endswith(" (deleted)"):
                    mapped.add(parts[-1])
        require(mapped == set(audit_row["actual_runtime_pins"])
                and all(runtime_pins.get(p) == h for p,h in audit_row["actual_runtime_pins"].items())
                and any("libcudart.so.12.4.99" in p for p in mapped) and any("libcuda.so." in p for p in mapped),
                "Actual mapped runtime/linked pin bundle differs")
        if gate["kind"] in ("ordinary", "cold"):
            primary_name = "gates/" + ("04-blocking-fresh" if gate["schedule"] == "blocking" else "05-lookahead-fresh") + "/quality.json"
            primary_path = audit_row["current_primary_report"]
            primary_hash = a.content_sha(primary_name)
            require(primary_hash == audit_row["current_primary_report_sha256"], "Audit current primary hash changed")
            check_primary(report, a.load(primary_name), worker, primary_hash, primary_path, gate["kind"])
    check_final_cleanup(final, a.load("final-idle.json"), plan["uuid_inventory"],
                        a.load("uuid-locks-held.json"), a.load("final-lock-release.json"), a.load("final-lock-probe.json"))
    return dict(status="passed", cpu_only=True, new_gpu_launches=0, archived_records=len(a.entries),
                archived_bytes=sum(e["stored_bytes"] for e in a.entries.values()), source_commit=COMMIT,
                factory_gates_passed=9, cli_serving_qualified=False, physical_release="unobserved",
                external_payload_bytes_checked=False, worker_projection_original_hash_reconstructed=False,
                cross_stream_sums=False, gates=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.archive), indent=2, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
        print("OFFLINE CHECK FAILED: " + str(error), file=sys.stderr)
        sys.exit(1)
