"""Read-only source verification and explicitly allowlisted CPU curation.

Never imports the original launcher, executes the test binary, queries NVML,
acquires locks, or captures environment/process data. Hashes are not signatures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import tarfile
from pathlib import Path, PurePosixPath

from evidence import (
    ARCHIVE_COMMIT, ARCHIVE_SHA, BINARY_SHA, CASES, COMPONENT_COMMIT,
    EvidenceError, HANDOFF_SHA, HERE, JOBS, SELECTORS, SOURCE_DIGEST,
    build_model, encode, equal, load, pick, read_rows, require, sha256,
    validate_rows, validate_streams,
)

CUT_NAME = "mixed-component-fault-current-20261015-01"
MANIFEST_SHA = "8488f5d8c1de3ec59b7c699ced46a4f1902334dc17fe2ea37efd116778bff36e"
ARCHIVE_PATHS = [
    "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "README.md", ".cargo", "crates", "scripts",
    "papers/arxiv/experiments/contract.py", "papers/arxiv/experiments/launch_guard.py",
    "papers/arxiv/experiments/runner.py", "papers/arxiv/experiments/m7.json",
    "papers/arxiv/experiments/m7-device-exclusive.json",
]
REFERENCES = {
    "fault_cuda.rs": ("crates/ferrule-model/tests/hybrid_bf16/fault_cuda.rs", "actual_component_test_and_assertions"),
    "fault_contract.rs": ("crates/ferrule-model/tests/hybrid_bf16/fault_contract.rs", "report_config_and_logical_ledger"),
    "fault_hook.rs": ("crates/ferrule-model/tests/hybrid_bf16/fault_hook.rs", "cfg_test_one_shot_actual_routed_down_seam"),
    "fixture.rs": ("crates/ferrule-model/tests/hybrid_bf16/fixture.rs", "known_seed_fixture_builder_not_quality_reference"),
    "residency.rs": ("crates/ferrule-model/src/transformer/standard/cuda/residency.rs", "actual_wait_before_cfg_test_proof_discard"),
    "hybrid.rs": ("crates/ferrule-model/src/transformer/standard/cuda/hybrid.rs", "cfg_test_hook_inclusion"),
    "forward.rs": ("crates/ferrule-model/src/transformer/forward.rs", "cfg_test_module_inclusion"),
    "forward_tests.rs": ("crates/ferrule-model/src/transformer/forward/tests.rs", "cuda_feature_test_inclusion"),
    "cuda_tests.rs": ("crates/ferrule-model/src/transformer/forward/cuda_tests.rs", "actual_selector_test_inclusion"),
}
SELECTOR_KEYS = (
    "job", "selector", "status", "tests_run", "passed", "failed", "ignored", "filtered_out",
    "returncode", "timed_out", "stable_pid_birth_parent_authenticated", "setsid_session_separately_verified",
    "authenticated_echild", "recorded_owners_absent_at_audit", "signals",
)
SUPERVISOR_KEYS = (
    "returncode", "timed_out", "wall_seconds", "cleanup_errors", "cleanup_scope", "descendant_custody",
    "gpu_quiescence", "stdout_truncated", "stderr_truncated",
)
CUSTODY_KEYS = ("status", "method", "containment_verified", "errors", "reporting_errors", "echild")
FINAL_KEYS = (
    "status", "gpu_tests_dispatched", "gpu_tests_issued", "gpu_proof", "retries", "profiling_runs", "cli_runs",
    "physical_cuda_release", "final_idle_verified", "authenticated_echild", "recorded_owners_absent", "unissued_selectors",
)
IDLE_KEYS = (
    "status", "all_eight_idle", "required_consecutive", "zero_memory_mib", "zero_utilization_percent",
    "no_compute_apps_required", "physical_owner_release", "mode", "authenticated_echild",
    "recorded_owners_absent", "accepted_sample_indices", "consecutive_idle",
)


def relative_name(name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "unsafe relative input name")
    return path


def pin_check(path, expected):
    require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "invalid SHA256 pin")
    require(sha256(path) == expected, "original input SHA256 mismatch")


def verify_source(root):
    """Hashes original inputs without writing them; returns only safe quantities.

    Original absolute paths are used transiently for external toolchain/library
    hashes, never marshalled into the portable bundle or an error message.
    """
    root = Path(root).resolve()
    pin_check(root / "actualreports/final-handoff.json", HANDOFF_SHA)
    pin_check(root / "evidence-manifest.json", MANIFEST_SHA)
    handoff = load(root / "actualreports/final-handoff.json")
    pin_check(root / "gpu-plan.json", handoff["source"]["plan_sha256"])
    # The original plan and manifest are pre-GPU records; do not rewrite False
    # or the historical empty-report claim to retrofit them into GPU evidence.
    plan = load(root / "gpu-plan.json")
    equal(plan["gpu_execution_observed"], False, "original pre-execution plan changed")
    equal(plan["gpu_authorized"], False, "original preparation authority changed")
    original_root = Path(load(root / "source-freeze.json")["frozen_source"]).parent

    def resolve_original(name):
        p = Path(name)
        if p.is_absolute():
            try:
                return root / p.relative_to(original_root)
            except ValueError:
                return p
        relative_name(name)
        return root / p

    count = 0
    for mapping in (plan["pins"], load(root / "evidence-manifest.json")["pins"]):
        for name, digest in mapping.items():
            pin_check(resolve_original(name), digest)
            count += 1
    for name, digest in handoff["artifact_sha256"].items():
        rel = PurePosixPath(name).relative_to(PurePosixPath("target/validation") / CUT_NAME)
        relative_name(str(rel))
        pin_check(root / str(rel), digest)
    frozen = load(root / "source-freeze.json")
    equal(pick(frozen, ("commit", "component_source_commit", "file_count", "archive_sha256", "source_digest")),
          dict(commit=ARCHIVE_COMMIT, component_source_commit=COMPONENT_COMMIT, file_count=1487,
               archive_sha256=ARCHIVE_SHA, source_digest=SOURCE_DIGEST), "source/archive identity drift")
    equal(frozen["archive_argv"], ["git", "--no-pager", "archive", "--format=tar", ARCHIVE_COMMIT, "--"] + ARCHIVE_PATHS,
          "original archive selection drift")
    pins = frozen["pins_relative_to_frozen_source"]
    equal(len(pins), 1487, "frozen source denominator")
    for name, digest in pins.items():
        relative_name(name)
        pin_check(root / "frozen-source" / name, digest)
    pin_check(root / "source.tar", ARCHIVE_SHA)
    # Inspect archive members in memory; never extract paths to disk.
    tar_pins = {}
    with tarfile.open(root / "source.tar", "r:") as archive:
        for member in archive:
            relative_name(member.name.rstrip("/"))
            if member.isdir():
                continue
            require(member.isfile() and member.name not in tar_pins, "unexpected/duplicate archive member")
            tar_pins[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    equal(tar_pins, pins, "archive/frozen inventory mismatch")
    external_count = 0
    for role in load(root / "external-input-pins.json").values():
        for name, digest in role["pins"].items():
            pin_check(resolve_original(name), digest)
            external_count += 1
    dependencies = load(root / "compiled-dependency-pins.json")["pins"]
    for name, digest in dependencies.items():
        pin_check(resolve_original(name), digest)
    native = load(root / "native-archive-pins.json")["immutable_copies"]
    for name, value in native.items():
        pin_check(resolve_original(name), value["sha256"])
    libraries = load(root / "runtime-library-pins.json")["resolved_libraries"]
    for item in libraries:
        pin_check(resolve_original(item["realpath"]), item["sha256"])
    pin_check(root / "staged/model-component-cuda-nvtx", BINARY_SHA)
    artifact = load(root / "artifact-model-component-cuda-nvtx.json")
    equal(artifact["source_digest"], SOURCE_DIGEST, "executable/source relationship drift")
    equal(artifact["sha256"], BINARY_SHA, "executable identity drift")
    equal(handoff["source"]["component_commit"], COMPONENT_COMMIT, "handoff source identity drift")
    equal(handoff["source"]["archive_commit"], ARCHIVE_COMMIT, "handoff archive identity drift")
    return {
        "source_files_verified": len(pins), "tar_members_verified": len(tar_pins),
        "original_plan_and_manifest_pins_verified": count,
        "handoff_artifact_pins_verified": len(handoff["artifact_sha256"]),
        "external_dependency_pins_verified": external_count,
        "compiled_dependency_pins_verified": len(dependencies), "native_archives_verified": len(native),
        "runtime_libraries_verified": len(libraries),
        "component_archive_selected_git_paths_identical": None,
        "gpu_queries": 0, "gpu_launches": 0, "source_writes": 0,
    }


def project_idle(d):
    """Drop UUIDs and process identities; preserve sample times and idle quantities."""
    out = pick(d, IDLE_KEYS)
    samples = []
    for sample in d.get("samples", []):
        devices = []
        for row in csv.reader(sample["gpu"].splitlines()):
            require(len(row) == 4, "unexpected NVML CSV columns")
            index, _, memory, util = (s.strip() for s in row)
            require(re.fullmatch(r"[0-7]", index) is not None, "invalid NVML ordinal")
            require(re.fullmatch(r"[0-9]+ MiB", memory) is not None, "unobserved NVML memory")
            require(re.fullmatch(r"[0-9]+ %", util) is not None, "unobserved NVML utilization")
            devices.append({"device": f"device_{index}", "memory_mib": int(memory.split()[0]),
                            "utilization_percent": int(util.split()[0])})
        require(not sample["apps"].strip(), "nonempty compute process sample")
        require(type(sample["recorded_unix_ns"]) is int, "missing input timestamp")
        require(sample["classification"] in ("idle", "utilization_only_transient"), "unexpected idle classification")
        samples.append({"recorded_unix_ns": sample["recorded_unix_ns"], "classification": sample["classification"],
                        "devices": devices, "compute_apps": 0})
    out["samples"] = samples
    return out


def project_custody(root):
    handoff = load(root / "actualreports/final-handoff.json")
    final = load(root / "gpu-run/FINAL.json")
    selectors = []
    for selector, job, case in zip(handoff["selectors"], JOBS, CASES):
        supervisor = load(root / "gpu-run" / job / "supervisor.json")
        custody = load(root / "gpu-run" / job / "custody.json")
        equal(supervisor["custody"], custody, "supervisor custody mismatch")
        # The authenticated adapter dup2s the test streams to pinned files.
        equal([supervisor["stdout"], supervisor["stderr"]], ["", ""], "unexpected supervisor stream")
        equal(selector["test_harness_stdout"], (root / "gpu-run" / job / "test.stdout").read_text(), "handoff stdout mismatch")
        selectors.append({"audit": pick(selector, SELECTOR_KEYS),
                          "supervisor": pick(supervisor, SUPERVISOR_KEYS),
                          "custody": pick(custody, CUSTODY_KEYS),
                          "stdout": f"evidence/{case}.stdout", "stderr": f"evidence/{case}.stderr"})
    idle_names = ["gpu-run/pre-lock-idle.json"]
    for job in JOBS:
        idle_names.extend(f"gpu-run/{job}/{name}.json" for name in
                          ("pre-idle", "authenticated-pre-cuda-idle", "post-idle"))
    idle_names.append("gpu-run/final-idle.json")
    locks = load(root / "gpu-run/uuid-locks-held.json")
    out = {
        "schema": "ferrule.mixed-fault-safe-custody.v1",
        "projection": "allowlisted_quantities_and_names_private_identity_environment_maps_omitted",
        "audit_recorded_unix_ns": handoff["recorded_unix_ns"],
        "denominator": handoff["denominator"], "selectors": selectors,
        "final": pick(final, FINAL_KEYS),
        "idle": {name: project_idle(load(root / name)) for name in idle_names},
        "locks": {"local_lock_count": len(locks["locks"]), "local_only": locks["local_only"],
                  "global_reservation": locks["global_reservation"],
                  **pick(load(root / "gpu-run/lock-release-probe.json"), ("all_eight_nonblocking_reacquired",)),
                  **pick(load(root / "gpu-run/lock-contexts-exited.json"), ("local_locks_released", "global_release_claim"))},
        "physical_owner_release_certificate": None,
        "physical_owner_release_certificate_observed": False,
    }
    validate_custody(out)
    return out


def validate_custody(d):
    equal(d["schema"], "ferrule.mixed-fault-safe-custody.v1", "custody schema drift")
    expected_idle = {"gpu-run/pre-lock-idle.json", "gpu-run/final-idle.json"}
    expected_idle.update(f"gpu-run/{job}/{stage}.json" for job in JOBS
                         for stage in ("pre-idle", "authenticated-pre-cuda-idle", "post-idle"))
    equal(sorted(d["idle"]), sorted(expected_idle), "idle receipt identity drift")
    equal(d["denominator"], dict(planned_selectors=2, issued_selectors=2, passed_selectors=2,
                                planned_cells=4, body_entered_cells=4, passed_cells=4, unissued_selectors=[]), "selector/cell denominator drift")
    equal(len(d["selectors"]), 2, "selector count drift")
    for s, selector, job in zip(d["selectors"], SELECTORS, JOBS):
        equal(s["audit"], dict(job=job, selector=selector, status="passed", tests_run=1, passed=1,
                               failed=0, ignored=0, filtered_out=902, returncode=0, timed_out=False,
                               stable_pid_birth_parent_authenticated=True, setsid_session_separately_verified=True,
                               authenticated_echild=True, recorded_owners_absent_at_audit=True, signals=0), "authenticated selector result drift")
        sup = s["supervisor"]
        equal(pick(sup, [k for k in SUPERVISOR_KEYS if k != "wall_seconds"]),
              dict(returncode=0, timed_out=False, cleanup_errors=[], cleanup_scope="isolated_subreaper_direct_children",
                   descendant_custody="proven_drained", gpu_quiescence="unverified",
                   stdout_truncated=False, stderr_truncated=False), "supervisor drain is not GPU quiescence")
        require(type(sup["wall_seconds"]) in (int, float) and sup["wall_seconds"] > 0, "invalid supervisor duration")
        equal(s["custody"], dict(status="proven_drained", method="linux-subreaper-waitid", containment_verified=True,
                                 errors=[], reporting_errors=[], echild=True), "ECHILD custody drift")
    equal(d["final"], dict(status="passed", gpu_tests_dispatched=2, gpu_tests_issued=2, gpu_proof=True,
                            retries=0, profiling_runs=0, cli_runs=0,
                            physical_cuda_release="unobserved; ECHILD/NVML idle are not a CUDA release certificate",
                            final_idle_verified=True, authenticated_echild=True, recorded_owners_absent=True,
                            unissued_selectors=[]), "runner final/physical release claim drift")
    equal(d["locks"], dict(local_lock_count=8, local_only=True, global_reservation=False,
                            all_eight_nonblocking_reacquired=True, local_locks_released=True,
                            global_release_claim=False), "local lock scope drift")
    equal(d["physical_owner_release_certificate"], None, "physical certificate invented")
    equal(d["physical_owner_release_certificate_observed"], False, "physical proof invented")
    equal(len(d["idle"]), 8, "idle receipt count drift")
    for name, idle in d["idle"].items():
        equal(pick(idle, ("status", "all_eight_idle", "required_consecutive", "zero_memory_mib", "zero_utilization_percent",
                          "no_compute_apps_required", "physical_owner_release", "consecutive_idle")),
              dict(status="verified", all_eight_idle=True, required_consecutive=3, zero_memory_mib=0,
                   zero_utilization_percent=0, no_compute_apps_required=True,
                   physical_owner_release="unverified", consecutive_idle=3), "NVML idle is not a physical release proof")
        transient = name.endswith("post-idle.json")
        equal(len(idle["samples"]), 4 if transient else 3, "idle sample count")
        equal(idle["accepted_sample_indices"], [1, 2, 3] if transient else [0, 1, 2], "idle acceptance window")
        timestamps = []
        for index, sample in enumerate(idle["samples"]):
            equal([x["device"] for x in sample["devices"]], [f"device_{i}" for i in range(8)], "device denominator")
            equal([x["memory_mib"] for x in sample["devices"]], [0]*8, "nonzero idle memory")
            util = [x["utilization_percent"] for x in sample["devices"]]
            require(all(type(x) is int and 0 <= x <= 100 for x in util), "invalid utilization")
            if transient and index == 0:
                equal(sample["classification"], "utilization_only_transient", "censored transient")
                require(any(util), "falsified transient")
            else:
                equal(sample["classification"], "idle", "invalid accepted classification")
                equal(util, [0]*8, "nonzero accepted utilization")
            equal(sample["compute_apps"], 0, "compute app custody drift")
            require(type(sample["recorded_unix_ns"]) is int, "missing original timestamp")
            timestamps.append(sample["recorded_unix_ns"])
        require(all(a < b for a, b in zip(timestamps, timestamps[1:])), "idle time order drift")
        if name.endswith("post-idle.json") or name == "gpu-run/final-idle.json":
            equal(pick(idle, ("mode", "authenticated_echild", "recorded_owners_absent")),
                  dict(mode="post_authenticated_echild", authenticated_echild=True, recorded_owners_absent=True), "post-exit idle provenance drift")


def reference_index(root):
    frozen = load(root / "source-freeze.json")
    return {name: {"original_path": original, "role": role,
                   "sha256": frozen["pins_relative_to_frozen_source"][original]}
            for name, (original, role) in REFERENCES.items()}


def source_projection(root, verification):
    frozen = load(root / "source-freeze.json")
    handoff = load(root / "actualreports/final-handoff.json")
    prefix = PurePosixPath("target/validation") / CUT_NAME
    inputs = {str(PurePosixPath(k).relative_to(prefix)): v
              for k, v in handoff["artifact_sha256"].items()}
    for name in ("actualreports/final-handoff.json", "source-freeze.json", "gpu-plan.json",
                 "evidence-manifest.json", "runtime-library-pins.json", "toolchain-versions.json",
                 "external-input-pins.json", "compiled-dependency-pins.json", "native-archive-pins.json",
                 "compiled-cuda-target.json", "HANDOFF.md"):
        inputs[name] = sha256(root / name)
    tools = load(root / "toolchain-versions.json")
    return {
        "schema": "ferrule.mixed-fault-public-source.v1", "cut": CUT_NAME,
        "identity": dict(component_commit=COMPONENT_COMMIT, archive_commit=ARCHIVE_COMMIT,
                         archive_sha256=ARCHIVE_SHA, source_digest=SOURCE_DIGEST,
                         executable_sha256=BINARY_SHA, final_handoff_sha256=HANDOFF_SHA,
                         source_files=1487),
        "original_inputs_sha256": inputs,
        "frozen_source_sha256": frozen["pins_relative_to_frozen_source"],
        "reference": reference_index(root), "curation_verification": verification,
        "component_archive_relationship": {
            "selected_paths_identical": handoff["source"]["selected_paths_identical_to_component_commit"],
            "basis": "pinned_actual_handoff_and_source_inventory_not_new_Git_audit"},
        "toolchain": {"version_stdout": {k: tools[k]["stdout"] for k in ("cargo", "rustc", "nvcc", "gcc")},
                      "cutlass_commit": tools["cutlass_commit"]["stdout"].strip(),
                      "linked_architectures": load(root / "compiled-cuda-target.json")["architectures"]},
        "runtime_library_pins": [dict(name=Path(x["realpath"]).name, sha256=x["sha256"])
                                 for x in load(root / "runtime-library-pins.json")["resolved_libraries"]],
        "native_archive_pins": [dict(name=Path(k).name, sha256=v["sha256"])
                                for k, v in sorted(load(root / "native-archive-pins.json")["immutable_copies"].items())],
        "external_pin_counts": {k: len(v["pins"]) for k, v in load(root / "external-input-pins.json").items()},
        "privacy": {"projection": "allowlist_not_blanket_raw_copy",
                    "omitted": ["absolute_paths", "environment", "worker_token", "authkey",
                                "process_identities", "GPU_UUIDs", "raw_library_maps",
                                "executable", "shared_libraries", "native_archives", "source_tar", "model_weights"]},
        "trust": "Hashes detect drift, not authenticity; redacted authentication conclusions are auditor assertions, not independently replayable handshakes.",
    }


def curate(root, destination):
    root, destination = root.resolve(), destination.resolve()
    require(not destination.is_relative_to(root) and not root.is_relative_to(destination), "source/output trees must be disjoint")
    verification = verify_source(root)
    payload = {}
    for case, job, selector in zip(CASES, JOBS, SELECTORS):
        rows = read_rows(root / "gpu-reports" / f"{case}.jsonl")
        validate_rows(rows, case)
        stdout = (root / "gpu-run" / job / "test.stdout").read_text()
        stderr = (root / "gpu-run" / job / "test.stderr").read_text()
        validate_streams(rows, stdout, stderr, selector)
        for suffix, source in (("jsonl", root / "gpu-reports" / f"{case}.jsonl"),
                               ("stdout", root / "gpu-run" / job / "test.stdout"),
                               ("stderr", root / "gpu-run" / job / "test.stderr")):
            payload[f"evidence/{case}.{suffix}"] = source.read_bytes()
    for name, (original, _) in REFERENCES.items():
        payload[f"reference/{name}"] = (root / "frozen-source" / original).read_bytes()
    payload["custody.json"] = encode(project_custody(root))
    payload["source-manifest.json"] = encode(source_projection(root, verification))
    payload["history.json"] = encode({
        "schema": "ferrule.mixed-fault-public-history.v1",
        "preparation": {"original": "HANDOFF.md", "gpu_execution": False,
                        "note": "The immutable preparation document is not the later execution verdict."},
        "actual": {"original": "actualreports/final-handoff.json", "sha256": HANDOFF_SHA,
                   "gpu_execution": True, "selectors_passed": 2, "cells_passed": 4,
                   "scope": "component_only_not_production_CLI_E5"},
        "preserved_limitations": [
            "Long temporary paths exceeded the fixed metadata quota; actual execution pins TMPDIR=/tmp.",
            "The initial CPU audit compared pre/post setsid whole identities; later audit uses stable PID/birth/parent and separately checks session.",
            "Unknown is synthetic owner-proof discard after successful real waits, not hardware proof loss.",
            "No new model, production CLI run, integrated broker, GPU probe, or rerun is part of publication."],
    })
    destination.mkdir(parents=True, exist_ok=True)
    # Only regenerate the explicitly owned publication payload, never originals.
    for name, data in payload.items():
        path = destination / name
        require(not path.is_symlink(), "refuse symlink output")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (destination / "postfailure.json").write_bytes(encode(build_model(destination)))
    payload["postfailure.json"] = (destination / "postfailure.json").read_bytes()
    # The checker/tooling is versioned source, not a generated input or its own trust root.
    (destination / "manifest.json").write_bytes(encode({
        "schema": "ferrule.mixed-fault-public-payload.v1", "final_handoff_sha256": HANDOFF_SHA,
        "files": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                  for name, data in sorted(payload.items())},
        "report_contract_sha256": sha256(HERE / "report-contract.json"),
        "scope": "curated_payload_only_not_self_signed_authentication",
    }))
    return verification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="original frozen validation cut; read only")
    parser.add_argument("--output", type=Path, default=HERE, help="owned publication output only")
    parser.add_argument("--verify-source-only", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_source(args.source) if args.verify_source_only else curate(args.source, args.output)
    except (EvidenceError, OSError, KeyError, TypeError, ValueError) as error:
        print("FAIL: source verification or curation rejected (private values omitted)")
        raise SystemExit(1) from error
    print(encode({"status": "PASS", **result}).decode(), end="")


if __name__ == "__main__":
    main()
