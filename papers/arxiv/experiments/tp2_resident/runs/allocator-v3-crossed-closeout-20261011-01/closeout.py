"""Offline verification and reporting only; no GPU query/launch or mutation of raw runs."""
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paired as p
import operator_retirement_ab as ab
import allocator_v3_release as release


def link(path):
    return dict(path=str(path), sha256=p.ct.sha(path))


def write(name, value):
    with (OUT/name).open("x") as stream:
        json.dump(value, stream, indent=2); stream.write("\n")


def idle(path):
    value = p.ct.load(path)
    assert value["status"] == "verified" and len(value["baseline"]) == 8
    assert all(d["gpu_bytes"] == 0 and d["utilization"] == 0 for d in value["baseline"].values())
    assert all(not r["apps"] for r in value["observations"])
    return link(path)


def main():
    p.ct.require_cpu_python()
    source_pins = p.ct.load(HERE/"plans/operator-retirement-v3-cpu-closeout-20261011-01/source-pins-final.json")
    p.ct.check_pins(source_pins)
    roots = [HERE/"runs/allocator-v3-serving-smoke-20261011-01"] + [
        HERE/("runs/allocator-v3-crossed-c"+str(c)+"-20261011-01") for c in (1,2)]
    lifecycles, measurements, builds = [], [], {}
    for run in roots:
        prepared = p.validate_pair(run)
        execution = p.ct.load(run/"execution.json")
        is_smoke = prepared["purpose"] == "serving_smoke"
        assert execution["status"] == ("complete_smoke" if is_smoke else "complete_descriptive_only")
        if not is_smoke:
            summary = p.ct.load(run/"operator-retirement-summary.json")
            p.ct.check_pins(summary["artifact_pins"])
            assert summary["order"] == list(ab.slots(summary["cycle"]))
        for slot in prepared["order"]:
            cell = run/"runs"/slot; session = cell/"session"
            assessed = p.pre_serving.assess_lifecycle(cell) if is_smoke else p.assess_slot(cell,"ferrule")
            m = p.slot_metadata(prepared["admission"], slot)
            p.ct.validate_staged_binary(m["binary"]["path"], m["binary"]["sha256"])
            supervisor = p.ct.load(session/"supervisor.json")
            assert p.ct.successful_process(supervisor) and p.ct.drained(supervisor)
            assert supervisor["custody"]["echild"] is True and not supervisor["cleanup_errors"]
            assert supervisor["returncode"] == 0 and supervisor["timed_out"] is False
            released = p.ct.load(cell/"lock-release.json")
            assert released["released"] is True and released["all_gpu_idle"] is True
            case = p.ct.load(session/"case.json")
            assert case["status"] == "complete" and case["resources"]["status"] == "observed_within_targets"
            assert not case["resources"]["errors"]
            before_idle, after_idle = idle(session/"before.json"), idle(session/"after.json")
            log = (session/"server.log").read_text()
            snapshots = release.d.snapshot_rows(log, profiling=False)
            allocator = ab.allocator_rows(log, m["binary"]["sha256"])
            assert allocator == assessed["runtime"]["allocator_retirement"]
            assert "mixed_rank_performance" not in log
            if not is_smoke: assert "request_worker_terminal" not in log
            samples = [json.loads(line) for line in (session/"samples.jsonl").read_text().splitlines()]
            diagnostics = [d for r in samples for d in r.get("process_diagnostics",[])
                if d["identity"]["pid"] == case["server_identity"]["pid"]]
            assert diagnostics
            visible = ",".join(p.shared.UUIDS)
            expected_log = "info,ferrule_request=debug" if is_smoke else "info"
            assert all(d["environment"]["CUDA_VISIBLE_DEVICES"] == visible
                and d["environment"]["NVIDIA_VISIBLE_DEVICES"] == visible
                and d["environment"]["FERRULE_LOG"] == expected_log for d in diagnostics)
            gpu_rows = [r["gpu"] for r in samples if r.get("gpu")]
            assert gpu_rows and all(not r["baseline_delta_check"]["violations"] for r in gpu_rows)
            assert all({u for u,d in r["baseline_delta_check"]["per_uuid"].items() if d["selected"]}
                == set(p.shared.UUIDS) for r in gpu_rows)
            counters = []
            for owner in (0,1):
                ready = next(r["metrics"] for r in allocator["rows"] if r["owner"]==owner and r["stage"]=="ready")
                shutdown = next(r["metrics"] for r in allocator["rows"] if r["owner"]==owner and r["stage"]=="shutdown")
                keys = ("retired_blocks","retirement_batches","retirement_created_fences")
                observed = all(k in ready and k in shutdown for k in keys)
                counters.append(dict(owner=owner, available=observed,
                    ready={k:ready[k] for k in keys} if observed else None,
                    shutdown={k:shutdown[k] for k in keys} if observed else None,
                    boundary_delta={k:shutdown[k]-ready[k] for k in keys} if observed else None))
            phases = {}
            for path in sorted(session.glob("*/http.json")):
                http = p.ct.load(path)
                requests = [r for b in http.get("runs",[]) for r in b.get("requests",[])]
                phases[path.parent.name] = dict(receipt=link(path), measured_requests=[
                    {k:r.get(k) for k in ("index","status","actual_input_tokens","actual_output_tokens","ttft_seconds","request_e2e_seconds")}
                    for r in requests])
            row = dict(run=str(run),slot=slot,kind="smoke" if is_smoke else "crossed_AB",
                launch_id=assessed["launch_id"],server_identity=case["server_identity"],binary=m["binary"],
                startup_seconds=case["startup_seconds"],spec=m["spec"],actual_log=expected_log,
                selected_uuids=list(p.shared.UUIDS),sampled_process_environment_matches_selected_observer=True,
                supervisor=link(session/"supervisor.json"),custody=supervisor["custody"],
                supervisor_gpu_quiescence=supervisor["gpu_quiescence"],
                observer_pre_idle=before_idle,observer_post_idle=after_idle,lock_release=link(cell/"lock-release.json"),
                resources=case["resources"],allocator=allocator,allocator_counters=counters,
                allocator_counter_scope="Ready/shutdown cumulative host ledger; includes nonmeasured requests, not isolated QA8 or total CUDA API events",
                snapshots=snapshots,http_phases=phases)
            lifecycles.append(row)
            builds[m["binary"]["sha256"]] = dict(binary=m["binary"],build_receipt=m["build_receipt"],
                source_pins=m["source_pins"],runtime_pins=m["runtime_pins"],test_quality=m["test_quality"],
                build_evidence_pins=m["build_evidence_pins"],historical_source_scope=slot=="BEFORE")
            if not is_smoke:
                http = p.ct.load(session/"qa8-V1/http.json")
                requests = [http["first_after_ready"],*http["warmups"],*http["runs"][0]["requests"]]
                assert len(http["warmups"])==2 and len(http["runs"][0]["requests"])==5 and len(requests)==8
                assert all(r["status"]=="success" and (r["actual_input_tokens"],r["actual_output_tokens"])==(5,8) for r in requests)
                summary_row = next(r for r in summary["rows"] if r["slot"]==slot)
                assert summary_row["allocator_retirement"] == allocator
                s = http["runs"][0]["summary"]; d = s["distributions"]
                measurements.append(dict(cycle=summary["cycle"],order=summary["order"],slot=slot,
                    launch_id=row["launch_id"],startup_seconds=row["startup_seconds"],
                    ttft_seconds=d["success_ttft_seconds"],request_e2e_seconds=d["success_request_e2e_seconds"],
                    successful_output_tokens_per_second=s["successful_output_tokens_per_second"],
                    measured_window_seconds=s["wall_seconds"],rate_denominator=s["denominator"],
                    unmeasured_qa8=3,measured_qa8=5,actual_input_tokens_each=5,actual_output_tokens_each=8,
                    measured_output_tokens=40,summary=summary_row,allocator_counters=counters,
                    http_receipt=link(session/"qa8-V1/http.json")))
    assert len(lifecycles)==6 and len({r["launch_id"] for r in lifecycles})==6
    assert len(measurements)==4
    pre, post = idle(OUT/"preflight-eight-idle.json"), idle(OUT/"final-eight-idle.json")
    assert p.ct.load(OUT/"final-lock-release.json")["all_eight_probe_locks_released"] is True
    comparisons = []
    for cycle in (1,2):
        before = next(r for r in measurements if r["cycle"]==cycle and r["slot"]=="BEFORE")
        after = next(r for r in measurements if r["cycle"]==cycle and r["slot"]=="AFTER")
        comparisons.append(dict(cycle=cycle,observed_rate_change_percent=100*(after["successful_output_tokens_per_second"]/before["successful_output_tokens_per_second"]-1),
            ttft_mean_change_percent=100*(after["ttft_seconds"]["mean"]/before["ttft_seconds"]["mean"]-1),
            e2e_mean_change_percent=100*(after["request_e2e_seconds"]["mean"]/before["request_e2e_seconds"]["mean"]-1)))
    admission = HERE/"admission/allocator-v3-release-20261011-01"
    promoted = p.ct.load(admission/"paired-admission.json")["engines"]["ferrule"]
    p.pre_serving.validate_smoke(promoted)
    write("measurements.json",dict(rows=measurements,comparisons=comparisons))
    write("lifecycle-audit.json",dict(rows=lifecycles,all_echild=True,all_selected_locks_released=True,
        all_observer_pre_post_eight_idle=True,supervisor_itself_does_not_prove_gpu_idle=True))
    write("build-source-scope.json",dict(builds=builds,workflow_source_pins_unchanged=source_pins,
        fixed_review=ab.REVIEWED_BUILD,source_delta=ab.reviewed_build(ab.REVIEWED_BUILD)["source_delta"]))
    raw_files = [f for run in roots for f in run.rglob("*") if f.is_file()]
    raw_files += [f for f in admission.rglob("*") if f.is_file()]
    raw = p.ct.inventory(raw_files); p.ct.check_pins(raw)
    write("raw-file-pins.json",raw)
    write("owner-closeout.json",dict(schema="ferrule.paper.allocator-v3-crossed-closeout.v1",
        status="completed_descriptive_only",timestamp_unix=time.time(),authorization=link(OUT/"authorization.json"),
        smoke=link(roots[0]/"serving-smoke.json"),promoted_admission=link(admission/"paired-admission.json"),
        cycles=[link(r/"operator-retirement-summary.json") for r in roots[1:]],
        measurements=link(OUT/"measurements.json"),lifecycle_audit=link(OUT/"lifecycle-audit.json"),
        build_scope=link(OUT/"build-source-scope.json"),raw_pins=link(OUT/"raw-file-pins.json"),
        pre_eight_idle=pre,post_eight_idle=post,final_lock_release=link(OUT/"final-lock-release.json"),
        six_authenticated_lifecycles_drained_echild=True,all_locks_released=True,foreign_kill=False,
        production_or_paper_source_changes=False,profile_executed=False,new_cpu_fix_required=False,
        geometry=ab.contract(1),actual_qa8_totals=dict(unmeasured=12,measured=20,measured_output_tokens=160),
        comparisons=comparisons,exact_speedup=None,fairmatch=False,isolated_retirement_effect=False,
        limitations=["Two crossed cycles are descriptive, not statistical significance or exclusive retirement causality.",
            "Sampled GPU peaks are not exact or continuous; NVML host PID to local process mapping remains unknown.",
            "Local ECHILD custody plus separately observed eight-GPU idle does not prove absence of foreign interference during execution.",
            "Ready/shutdown pending is zero; pending/Unknown high-watermarks remain null, never zero.",
            "BEFORE has no new retirement counters; AFTER fences count complete successful event sets, not all CUDA calls.",
            "Rates use actual HTTP fixed-work denominator and usage tokens, not exact visible token IDs or pure decode time.",
            "Smoke alone uses request DEBUG for cancellation proof; all A/B logs info, rankperf off. No official profile this round.",
            "Generic profile-admission.json emitted by existing promote is not an authorized info-only profile launch; use separately reviewed retirement_profile policy next phase."]))
    print(json.dumps(dict(status="completed_descriptive_only",lifecycles=len(lifecycles),measurements=len(measurements),raw_files=len(raw),comparisons=comparisons)))


if __name__ == "__main__": main()
