#!/usr/bin/env python3
"""Offline stdlib checker: pinned evidence plus semantic invariants, no GPU access."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import struct

from archive import CUT_NAME, REFERENCES, relative_name, validate_custody
from evidence import (
    ARCHIVE_COMMIT, ARCHIVE_SHA, BINARY_SHA, CASES, COMPONENT_COMMIT,
    EvidenceError, HANDOFF_SHA, HERE, JOBS, SOURCE_DIGEST, build_model,
    encode, equal, load, require, sha256,
)

REPORT_PINS = {
    "mixed_clean": "c842088ddfaed935db1102d726e03be937a135bd86b7541107f2b188d946f8ff",
    "mixed_unknown": "4b1f37cc5a05884d012211a10e6e28b0464a4c60bf7f96d8a53e66d1597089bc",
}
LIMIT = 10 * 1024 * 1024


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_source(source, root=HERE):
    equal(source["schema"], "ferrule.mixed-fault-public-source.v1", "source schema")
    equal(source["cut"], CUT_NAME, "source cut drift")
    equal(source["identity"], dict(component_commit=COMPONENT_COMMIT, archive_commit=ARCHIVE_COMMIT,
                                  archive_sha256=ARCHIVE_SHA, source_digest=SOURCE_DIGEST,
                                  executable_sha256=BINARY_SHA, final_handoff_sha256=HANDOFF_SHA,
                                  source_files=1487), "source identity drift")
    inventory = source["frozen_source_sha256"]
    equal(len(inventory), 1487, "source inventory denominator drift")
    equal(digest_json(inventory), SOURCE_DIGEST, "original source inventory digest drift")
    for name, digest in inventory.items():
        relative_name(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "invalid source digest")
    equal(sorted(source["reference"]), sorted(REFERENCES), "source reference selection drift")
    for name, (original, role) in REFERENCES.items():
        equal(source["reference"][name], dict(original_path=original, role=role, sha256=inventory[original]),
              "reference/source link drift")
        equal(sha256(root / "reference" / name), inventory[original], "pinned test source changed")
    inputs = source["original_inputs_sha256"]
    for name, digest in inputs.items():
        relative_name(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "invalid original input pin")
    equal(inputs["actualreports/final-handoff.json"], HANDOFF_SHA, "actual handoff pin drift")
    for case, job in zip(CASES, JOBS):
        equal(inputs[f"gpu-reports/{case}.jsonl"], REPORT_PINS[case], "actual report source pin drift")
        for suffix in ("jsonl", "stdout", "stderr"):
            original = f"gpu-reports/{case}.jsonl" if suffix == "jsonl" else f"gpu-run/{job}/test.{suffix}"
            equal(sha256(root / "evidence" / f"{case}.{suffix}"), inputs[original], "producer/input bytes drift")
    equal(source["component_archive_relationship"], {
        "selected_paths_identical": True, "basis": "pinned_actual_handoff_and_source_inventory_not_new_Git_audit"},
        "component/archive relationship overstated")
    equal(source["curation_verification"], dict(
        source_files_verified=1487, tar_members_verified=1487, original_plan_and_manifest_pins_verified=158,
        handoff_artifact_pins_verified=33, external_dependency_pins_verified=13268,
        compiled_dependency_pins_verified=111, native_archives_verified=4, runtime_libraries_verified=12,
        component_archive_selected_git_paths_identical=None, gpu_queries=0, gpu_launches=0, source_writes=0),
        "curation read-only verification drift")
    equal(source["toolchain"]["linked_architectures"], ["sm_86"], "compiled architecture drift")
    equal(source["toolchain"]["cutlass_commit"], "e05f953a5b3d38adc240df2ff928e0421c2abba3", "CUTLASS pin drift")
    equal(len(source["runtime_library_pins"]), 12, "runtime library count drift")
    equal(len(source["native_archive_pins"]), 4, "native archive count drift")
    for item in source["runtime_library_pins"] + source["native_archive_pins"]:
        equal(sorted(item), ["name", "sha256"], "private library fields")
        require(Path(item["name"]).name == item["name"], "private library path")
        require(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is not None, "invalid library pin")


def figure_data(model):
    return {"schema": "ferrule.fault-custody-figure-data.v1", "scope": model["scope"],
            "denominator": model["denominator"],
            "cells": [{k: c[k] for k in ("case", "schedule", "snapshots", "boundary", "assertion_backed",
                                        "committed_content", "host_read_statistics", "local_final_release")}
                      for c in model["cells"]]}


def validate_figure(record, model, root=HERE, figures=None):
    equal(record["schema"], "ferrule.fault-custody-figure.v1", "figure schema")
    expected = figure_data(model)
    equal(record["data"], expected, "figure data is not derived from checked cells")
    equal(record["data_sha256"], hashlib.sha256(encode(expected)).hexdigest(), "figure data hash drift")
    equal(record["publication_manifest_sha256"], sha256(root / "manifest.json"), "figure publication pin drift")
    layout = record["layout"]
    equal(layout["overlapping_text_pairs"], 0, "figure text collision")
    equal(layout["all_text_inside_canvas"], True, "figure text clipped")
    equal(layout["png_pixels"], [2052, 1728], "figure canvas drift")
    equal(len(layout["labels"]), layout["text_count"], "figure label count")
    for label in layout["labels"]:
        x0, y0, x1, y1 = label["bounds"]
        require(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, "label outside canvas")
    width, height = layout["png_pixels"]
    for a, b in itertools.combinations(layout["labels"], 2):
        a, b = a["bounds"], b["bounds"]
        require(not (min(a[2], b[2])-max(a[0], b[0]) > 1/width and
                     min(a[3], b[3])-max(a[1], b[1]) > 1/height), "figure measured text overlap")
    equal(sorted(record["outputs"]), ["fault-custody-contract.pdf", "fault-custody-contract.png"], "figure output selection")
    if figures is not None:
        figures = Path(figures)
        equal(sha256(figures / "fault-custody-contract.py"), record["source_sha256"], "renderer source drift")
        for name, pin in record["outputs"].items():
            data = (figures / name).read_bytes()
            equal(len(data), pin["bytes"], "figure size drift")
            equal(hashlib.sha256(data).hexdigest(), pin["sha256"], "figure bytes drift")
            if name.endswith(".png"):
                require(data.startswith(b"\x89PNG\r\n\x1a\n"), "invalid PNG header")
                equal(list(struct.unpack(">II", data[16:24])), layout["png_pixels"], "PNG dimensions")
            else:
                require(data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF"), "invalid PDF envelope")


def check(root=HERE, figures=None, include_figure=True):
    root = Path(root).resolve()
    manifest = load(root / "manifest.json")
    equal(manifest["schema"], "ferrule.mixed-fault-public-payload.v1", "payload schema")
    equal(manifest["final_handoff_sha256"], HANDOFF_SHA, "payload source identity drift")
    expected = {f"evidence/{case}.{suffix}" for case in CASES for suffix in ("jsonl", "stdout", "stderr")}
    expected |= {f"reference/{name}" for name in REFERENCES}
    expected |= {"custody.json", "source-manifest.json", "history.json", "postfailure.json"}
    equal(sorted(manifest["files"]), sorted(expected), "missing/unapproved payload")
    total = 0
    for name, pin in manifest["files"].items():
        relative_name(name)
        path = root / name
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "nonlocal payload")
        require(path.is_file(), "missing payload")
        equal(path.stat().st_size, pin["bytes"], "payload size drift")
        equal(sha256(path), pin["sha256"], "payload hash drift")
        total += path.stat().st_size
    require(total < LIMIT, "publication exceeds 10 MiB")
    equal(sha256(root / "report-contract.json"), manifest["report_contract_sha256"], "report vocabulary drift")
    validate_source(load(root / "source-manifest.json"), root)
    custody = load(root / "custody.json")
    validate_custody(custody)
    history = load(root / "history.json")
    equal(history["preparation"]["gpu_execution"], False, "preparation promoted to execution")
    equal(history["actual"], {"original": "actualreports/final-handoff.json", "sha256": HANDOFF_SHA,
                              "gpu_execution": True, "selectors_passed": 2, "cells_passed": 4,
                              "scope": "component_only_not_production_CLI_E5"}, "actual/history drift")
    model = build_model(root)
    equal(load(root / "postfailure.json"), model, "derived metrics/assertions do not match producer evidence")
    if include_figure:
        validate_figure(load(root / "figure.json"), model, root, figures)
    return {"status": "PASS", "selectors": 2, "cells": 4, "passed_cells": 4,
            "payload_files": len(expected), "payload_bytes": total, "source_files_pinned": 1487,
            "portable_source_references": len(REFERENCES),
            "post_Unknown_contents": "UNOBSERVED", "physical_owner_release_certificate": None,
            "gpu_queries": 0, "gpu_launches": 0, "private_inputs_required": False,
            "figure_data_checked": include_figure, "figure_files_checked": figures is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--figures", type=Path, help="optional companion directory; verify source/PDF/PNG without rendering")
    args = parser.parse_args()
    try:
        result = check(args.root, args.figures)
    except (EvidenceError, OSError, KeyError, TypeError, ValueError):
        print("FAIL: public evidence integrity or semantic invariant rejected")
        raise SystemExit(1)
    print(encode(result).decode(), end="")


if __name__ == "__main__":
    main()
