#!/usr/bin/env python3
"""New retrospective audit of frozen -04; never modifies or upgrades its verdict.

CPU-only. Both old sealed files and the corrected source are pinned in a new
private output. Historical prepared() is deliberately not called with new code.
"""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from profile_overlap_v1 import analyze, protocol as p, role_context, run, scope_contract


def audit(directory):
    directory = p.directory_check(directory)
    old = p.HERE / "artifacts-go-20261015-04"
    case = old / "runs/blocking/session"
    p.require(not directory.exists() and directory != p.HERE / p.GENERATION, "new offline directory required")
    before = p.ct.inventory(sorted(f for f in old.rglob("*") if f.is_file()))
    prepared = p.ct.load(old / "prepared.json")
    handoff = p.ct.load(old / "final-handoff.json")
    p.ct.check_pins(handoff["evidence_pins"])
    p.require(handoff["prepared_sha256"] == p.ct.sha(old / "prepared.json")
              and prepared["generation"] == old.name and prepared["directory"] == str(old)
              and prepared["binding_sha256"] == p.ct.sha(old / "candidate-binding.json")
              and prepared["candidate"] == p.ct.load(old / "candidate-binding.json")
              and prepared["runtime_policy_sha256"] == p.ct.sha(old / "runtime-policy.json")
              and prepared["source_qualification_sha256"] == p.ct.sha(old / "nvtx-source-qualification.json"),
              "frozen -04 source/binding seals changed")
    old_verdict = p.ct.load(case / "worker-profile-audit.json")
    p.require(handoff["status"] == "rejected_stop_lookahead_unissued"
              and old_verdict["status"] == "rejected" and not old_verdict["scoped_overlap_accepted"],
              "old frozen failure must remain in denominator")
    p.ct.check_pins(prepared["candidate"]["candidate"]["artifact_pins"])
    p.ct.check_pins(prepared["runtime"]["pins"])
    source = p.source_qualification(prepared["candidate"])
    worker = p.ct.load(case / "worker-capture.json")
    supervisor = p.ct.load(case / "supervisor.json")
    p.require(p.ct.drained(supervisor) and supervisor["returncode"] == 1
              and not supervisor["timed_out"] and not supervisor["cleanup_errors"],
              "expected historical audit-failed but authenticated/drained worker")
    p.require(worker["status"] == "qa_complete" and worker["shutdown"]["returncode"] == 0
              and not worker["shutdown"]["timed_out"] and not worker.get("cleanup_error")
              and not worker["monitor"]["errors"], "historical capture/QA cleanup failed")
    run.capture_evidence(case, worker, supervisor)
    for phase in ("ready", "after-requests"):
        folder = case / (phase + "-library-evidence")
        raw = p.ct.load(folder / "snapshot.json")
        policy = p.ct.load(folder / "policy.json")
        p.require(policy == p.ct.load(old / "runtime-policy.json")
                  and raw["root_identity"] == worker["server_identity"]
                  and raw["parent_identity"] == worker["target_identity"]
                  and p.maps.validate_snapshot(raw, policy)["status"] == "passed", "raw historical CLI maps failed")
    p.require(p.ct.load(case / "collector-command.json")["argv"] ==
              p.profile_argv(case, p.ct.sha(case / "collector-manifest.json"))
              and p.ct.load(case / "export-command.json")["argv"] == p.export_argv(case)
              and p.ct.load(case / "export-result.json")["returncode"] == 0, "historical capture/export command changed")
    lines = run.ex.log_lines(case / "server.log")
    qa = run.qa_result(case, p.ct.load(case / "canary-qa/http-terminal.json"), lines)
    ledger = run.read_ledger(lines, worker["server_identity"], "blocking")
    p.require(ledger == p.ct.load(case / "read-ledger.json"), "raw read ledger replay differs")
    context = dict(scope_contract=prepared["overlap_scope"], read_source_boundary=source["read_source_boundary"],
        model_source_generation=ledger["model_source_generation"], model_pid=worker["server_identity"]["pid"],
        model_birth_ticks=worker["server_identity"]["birth_ticks"], uuid=run.tc.UUID,
        request_begin_unix_ns=qa["request_begin_unix_ns"], request_end_unix_ns=qa["request_end_unix_ns"],
        read_stats=ledger["read"]["stats"], last_receipt=ledger["read"]["last_receipt"],
        cupti_path=str(p.TOOL / "target-linux-x64/libcupti.so.12.9"))
    with sqlite3.connect((case / "nsys.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        context.update(role_context.bind(case, worker, prepared, analyze.rows(db, "PROCESSES"), generation=old.name))
        schema = db.execute("SELECT sql FROM sqlite_master WHERE name=?", ("OSRT_API",)).fetchone()[0]
        report = analyze.audit_sqlite(db, context)
    report["log_issues"] = [issue for name in ("server.log", "export.stdout", "export.stderr")
        for issue in run.collector_log_issues(case / name, run.ex.log_lines(case / name))]
    run.verify_final_idle(p.ct.load(case / "final-idle-v2.json"), prepared["candidate"]["uuid_inventory"])
    gates = dict.fromkeys(("one_frozen_qa_http_token_join", "authenticated_model_pid_birth_source",
        "ready_and_after_raw_maps", "dynamic_owner_pair_closed", "authenticated_echild",
        "original_all_eight_stable_idle", "collector_export_closed"), True)
    predicate = scope_contract.acceptance(report, gates)
    p.ct.check_pins(before)
    directory.mkdir(mode=0o700)
    result = dict(schema=p.ID + ".offline-audit.v2", status="retrospective_corrected_predicate_evaluated",
        retrospective=True, historical_prepared_upgraded=False, historical_acceptance_transferred=False,
        scoped_overlap_accepted=False, full_trace_accepted=False,
        corrected_predicate=predicate, lifecycle_gates=gates,
        original_status=handoff["status"], original_sql_issues=old_verdict["scoped_issues"],
        original_capture_count=1, original_accepted_arms=0, original_lookahead_unissued=True,
        original_prepared_sha256=p.ct.sha(old / "prepared.json"), source_qualification=source,
        analyzer_sha256=p.ct.sha(p.HERE / "analyze.py"), audit_sha256=p.ct.sha(Path(__file__)),
        official_schema_sql=schema, analysis=report, quality=qa["trace"],
        original_files_unchanged=True, original_pins=before, gpu_queries=0, captures_issued=0)
    p.ct.write(directory / "offlineaudit_v2.json", result)
    return dict(directory=str(directory), status=result["status"], corrected_predicate=predicate,
                old_status=result["original_status"], historical_prepared_upgraded=False)


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().directory), indent=2))
