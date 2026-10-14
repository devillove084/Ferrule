"""CPU-only closeout verification; never launches a model or queries a device."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paired
import operator_retirement_ab as ab
import allocator_v3_release as release


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    with (OUT/name).open("x") as stream:
        json.dump(value, stream, indent=2); stream.write("\n")


def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "none"
    start = time.time()
    run = HERE/"runs/allocator-v3-serving-smoke-20261011-01"
    prepared = paired.validate_pair(run)
    assert prepared["purpose"] == "serving_smoke" and prepared["order"] == ["S1", "S2"]
    for slot in prepared["order"]:
        m = paired.validate_slot(run/"runs"/slot)
        assert m["executable_sha256"] == release.BINARY_SHA
    assert not any(p.name in ("launch.json", "execution.json", "serving-smoke.json") for p in run.rglob("*"))
    before = ab.frozen_audit()
    review = ab.reviewed_build(ab.REVIEWED_BUILD)
    assert review["after_binary_sha256"] == release.BINARY_SHA
    files = ab.pre.checked_document(dict(path=str(release.RELEASE/"handoff-file-pins.json"), sha256=release.FILE_PINS_SHA))
    assert len(files) == 96
    for name, digest in files.items(): assert sha(release.RELEASE/name) == digest
    assert sha(release.RELEASE/"handoff.json") == release.HANDOFF_SHA
    assert sha(release.RELEASE/"ferrule-allocator-v3") == release.BINARY_SHA
    restoration = json.loads((OUT/"restoration.json").read_text())
    assert sha(HERE/"pre_serving.py") == restoration["sha256"]
    workflows = {str(p):sha(p) for p in sorted(HERE.glob("*.py"))}
    write("source-pins-final.json", workflows)
    entry = json.loads((OUT/"source-pins-on-entry.json").read_text())
    changes = {name:dict(before=digest, after=workflows[str((paired.ct.ROOT/name).resolve())])
        for name,digest in entry.items() if digest != workflows[str((paired.ct.ROOT/name).resolve())]}
    write("paper-source-changes.json", changes)
    tests = []
    for name, count in (("full-resident-sealed",315), ("paper-contracts",42), ("paper-analysis",24)):
        receipt = json.loads((OUT/(name+".json")).read_text())
        log = OUT/(name+".log")
        text = log.read_text()
        assert receipt["exit_code"] == 0 and sha(log) == receipt["log_sha256"]
        assert re.search(r"Ran " + str(count) + r" tests", text) and text.rstrip().endswith("OK")
        tests.append(dict(name=name, tests=count, receipt=str(OUT/(name+".json")), **receipt))
    staged = Path(prepared["admission"]["engines"]["ferrule"]["binary"]["path"])
    assert sha(staged) == release.BINARY_SHA
    frozen = dict(before_binary_sha256=before["binary_sha256"], current_tree_qualified=False,
        before_evidence_pins=before["evidence_pins"], release_96_file_pins=files,
        review_manifest=ab.REVIEWED_BUILD, pre_serving_restoration=restoration)
    write("preserved-evidence-verification.json", frozen)
    write("prepared-verification.json", dict(status="CPU_verified_not_executed", directory=str(run),
        prepared_sha256=sha(run/"prepared.json"), slots=prepared["order"], source_pins=prepared["source_pins"],
        binary_sha256=sha(staged), admission_path=prepared["admission_path"],
        admission_sha256=prepared["admission_sha256"], authorized=False, launch_present=False, execution_present=False))
    status = subprocess.check_output(["git", "--no-optional-locks", "status", "--short"], cwd=paired.ct.ROOT)
    (OUT/"git-status-final.txt").write_bytes(status)
    write("cpu-closeout.json", dict(schema="ferrule.paper.operator-retirement-cpu-closeout.v1",
        status="CPU_green_sealed_GPU_not_authorized", timestamp_unix=time.time(), elapsed_seconds=time.time()-start,
        verification_command=[sys.executable,"-S","-B",str(Path(__file__).resolve())],
        cwd=str(Path.cwd()), environment={k:os.environ[k] for k in ("CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES")},
        verification_checks_passed=True, tests_passed=sum(t["tests"] for t in tests), tests=tests,
        suite_scope="All tp2_resident test*.py plus paper contracts and analysis; no Rust/GPU or torch oracle suites",
        prepared=str(run/"prepared.json"), release_handoff=review["release_handoff"], binary_sha256=release.BINARY_SHA,
        source_pins=workflows, production_sources=prepared["admission"]["engines"]["ferrule"]["source_pins"],
        commands=[json.loads((OUT/(name+".json")).read_text()) for name in ("bootstrap","prepare-smoke")],
        gpu_queried=False, gpu_locks_acquired=False, service_started=False, profiler_started=False,
        next_phase=dict(authorized=False, action="Fresh explicit owner grant required before same-binary S1/S2",
            run_argv_without_authorization=[sys.executable,"-S","-B",str(HERE/"paired.py"),"run","--id",run.name,"--purpose","serving_smoke"],
            warning="No grant flags/owner note issued here. Entry remains fail-closed until a new explicit grant."),
        limitations=["Serving smoke has not run; no GPU, throughput, fairmatch or exact-speedup acceptance.",
            "BEFORE metrics stay historical. Boundary pending zero is not a peak; unknown/pending peaks remain null.",
            "Formal info-only 9ef/7e1c traces still require separate authorization and actual same-artifact evidence."]))
    print("CPU seal checks passed; 381 tests; S1/S2 prepared, no GPU authorization")


if __name__ == "__main__": main()
