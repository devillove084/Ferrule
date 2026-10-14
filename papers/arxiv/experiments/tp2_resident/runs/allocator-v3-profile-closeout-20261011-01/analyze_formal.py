from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import sys
HERE=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import paired as p
from profile_trace_audit import kernel_group, interval_union
from profile_trace_audit_v2 import process_id
DIMS=("gridX","gridY","gridZ","blockX","blockY","blockZ")

def coverage(intervals,start,end):
    assert end>start
    intervals=list(intervals)
    assert all(b>=a for a,b in intervals)
    merged=interval_union((max(a,start),min(b,end)) for a,b in intervals if min(b,end)>max(a,start))
    total=sum(b-a for a,b in merged)
    gaps=[]; cursor=start
    for a,b in merged:
        if a>cursor: gaps.append(a-cursor)
        cursor=b
    if cursor<end: gaps.append(end-cursor)
    return dict(window_ns=end-start,observed_union_ns=total,no_exported_interval_ns=end-start-total,
        no_exported_fraction=(end-start-total)/(end-start),gap_count=len(gaps),largest_gap_ns=max(gaps,default=0))

def add(table,key,ns):
    row=table.setdefault(key,dict(count=0,duration_ns=0))
    row["count"]+=1; row["duration_ns"]+=ns

def write(name,value):
    with (OUT/name).open("x") as stream: json.dump(value,stream,indent=2);stream.write("\n")

def analyze(label):
    run=HERE/("runs/allocator-v3-profile-"+label+"-20261011-01")
    a=p.ct.load(run/"retirement-profile-audit.json"); p.ct.check_pins(a["artifact_pins"])
    assert a["bidirectional_correlation"]["measured"]["observed_correlation_closed"]
    w=a["actual_request_windows"]["measured"]; start,end=w["start_ns"],w["end_ns"]; pid=a["model_pid"]
    case=run/"runs/P1/session"
    db=sqlite3.connect((case/"nsight/nsys.sqlite").as_uri()+"?mode=ro",uri=True); db.row_factory=sqlite3.Row
    names=dict(db.execute("SELECT id,value FROM StringIds"))
    maps={(r["pid"],r["cudaId"]):r["gpuId"] for r in db.execute("SELECT * FROM TARGET_INFO_CUDA_DEVICE")}
    gpus={r["id"]:"GPU-"+r["uuid"].removeprefix("GPU-") for r in db.execute("SELECT id,uuid FROM TARGET_INFO_GPU")}
    groups={k:dict(count=0,duration_ns=0) for k in ("decode_weights","pack_bf16_activation","gemm_gemv_named","misc")}
    original={}; kernels={}; intervals=defaultdict(list); signature=Counter(); per_uuid=defaultdict(dict)
    for kind,table in (("kernel","CUPTI_ACTIVITY_KIND_KERNEL"),("memcpy","CUPTI_ACTIVITY_KIND_MEMCPY"),("memset","CUPTI_ACTIVITY_KIND_MEMSET")):
        for r in db.execute("SELECT * FROM "+table+" WHERE start<? AND end>?",(end,start)):
            if process_id(r["globalPid"])!=pid: continue
            uuid=gpus[maps[pid,r["deviceId"]]]; assert uuid in p.shared.UUIDS
            assert start<=r["start"]<=r["end"]<=end
            intervals[uuid].append((r["start"],r["end"]))
            if kind!="kernel": continue
            name=names[r["demangledName"]]; ns=r["end"]-r["start"]; old=kernel_group(name)
            group="gemm_gemv_named" if old=="matrix_math_named_kernels_not_GemmEx_API" else old if old in groups else "misc"
            add(groups,group,ns);add(original,old,ns);add(kernels,name,ns);add(per_uuid[uuid],group,ns)
            signature[(uuid,name,*(r[k] for k in DIMS))]+=1
    assert sum(v["count"] for v in groups.values())==a["bidirectional_correlation"]["measured"]["counts"]["kernel_observed"]
    api={}; threads=defaultdict(list); launches=defaultdict(list); table_rows={}; kinds=Counter()
    tables={r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME","CUPTI_ACTIVITY_KIND_DRIVER"):
        if table not in tables: table_rows[table]=None;continue
        table_rows[table]=0
        for r in db.execute("SELECT * FROM "+table+" WHERE start>=? AND end<=?",(start,end)):
            if process_id(r["globalTid"])!=pid:continue
            name=names[r["nameId"]]; table_rows[table]+=1
            add(api,name,r["end"]-r["start"]); dest=api[name].setdefault("returns",{}); ret=str(r["returnValue"])
            dest[ret]=dest.get(ret,0)+1
            kinds["driver_named" if name.startswith("cu") and not name.startswith("cuda") else "runtime_named"]+=1
            threads[r["globalTid"]].append((r["start"],r["end"]))
            if "LaunchKernel" in name and r["returnValue"]==0:launches[r["globalTid"]].append((r["start"],r["end"]))
    host=[]
    for tid,rs in sorted(launches.items()):
        host.append(dict(globalTid=tid,launches=len(rs),cuda_api_coverage=coverage(threads[tid],start,end),
            launch_interval_coverage=coverage(rs,start,end),
            interpretation="Uncovered time on a launch-emitting thread is not CPU idle: includes non-CUDA work, waiting, scheduling, logging and missing trace."))
    db.close()
    shapes=[dict(uuid=s[0],name=s[1],launch_dims=list(s[2:]),count=n) for s,n in sorted(signature.items())]
    http=p.ct.load(case/"profile/measured.json")
    output=dict(label=label,binary_sha256=a["binary_sha256"],window=w,model_pid=pid,
        http={k:http[k] for k in ("text","actual_input_tokens","actual_output_tokens","start_clock","end_clock","ttft_seconds","request_e2e_seconds")},
        request_scope="Profile-only whole HTTP request including prefill; not benchmark or pure decode tokens",
        groups=groups,baseline_compatible_groups=original,per_uuid_groups=dict(per_uuid),kernels=kernels,
        top_kernel_counts=sorted(kernels.items(),key=lambda r:-r[1]["count"])[:20],
        top_kernel_durations=sorted(kernels.items(),key=lambda r:-r[1]["duration_ns"])[:20],
        launch_signatures=shapes,launch_signature_digest=p.ct.digest(shapes),
        launches_per_output_token=sum(v["count"] for v in groups.values())/http["actual_output_tokens"],
        launch_normalization_caveat="Includes 5-token prefill plus request work; not kernels per pure decode step",
        all_api_by_name=api,api_table_model_rows=table_rows,api_layers_by_name=dict(kinds),
        api_storage_note="2025.3 exports cu* driver and cuda* runtime calls together in RUNTIME; absent DRIVER table is null, not zero driver calls",
        per_uuid_observed_activity_coverage={u:coverage(rs,start,end) for u,rs in intervals.items()},
        both_uuid_observed_activity_union=coverage([v for rs in intervals.values() for v in rs],start,end),
        gap_caveat="Complement of exported kernel/copy/set intervals only. Not proven GPU idle, CPU cost, CPU collective wait, or wall gain; trace can lose paired rows.",
        launch_thread_exported_intervals=host,host_idle_ns=None,cpu_collective_wait_ns=None,
        formal_closure=a["bidirectional_correlation"]["measured"],
        warmup_closure=dict(status=a["bidirectional_correlation"]["warmup"]["status"],
            issue_counts=dict(Counter((r["direction"]+":"+r["reason"]+":"+r["kind"]) for r in a["bidirectional_correlation"]["warmup"]["issues"]))),
        full_trace_accepted=False,diagnostics=a["diagnostics"],api_event_sync_launch_counts=a["api_counts"]["measured"],
        copies=a["per_uuid_activity"]["measured"],official_audit=dict(path=str(run/"retirement-profile-audit.json"),sha256=p.ct.sha(run/"retirement-profile-audit.json")))
    write(label+"-formal-analysis.json",output)
    return output

def main():
    assert coverage([(1,4),(2,5)],0,10)["observed_union_ns"]==4
    assert coverage([(-5,4),(8,20)],0,10)["observed_union_ns"]==6
    assert coverage([(1,2),(2,3)],0,4)["gap_count"]==2
    assert coverage([],0,10)["no_exported_interval_ns"]==10
    assert coverage([(0,10)],0,10)["no_exported_interval_ns"]==0
    assert coverage([(3,3)],0,10)["observed_union_ns"]==0
    write("offline-math-tests.json",dict(passed=6,scope="Synthetic interval union/clipping/gap checks, CPU only"))
    before,after=analyze("before"),analyze("after")
    assert before["launch_signatures"]==after["launch_signatures"]
    assert all(before["http"][k]==after["http"][k] for k in ("text","actual_input_tokens","actual_output_tokens"))
    baseline=[]
    for name in ("tp2-official-full-profile-20261010-001","tp2-decoded-cache-official-profile-20261010-001"):
        root=HERE/"runs"/name; path=root/"profile-audit-v2.json"; audit=p.ct.load(path)
        p.ct.check_pins({str(root/k):v for k,v in audit["raw_sha256"].items()})
        m=audit["requests"]["measured"]; counts=m["bidirectional_correlation"]["counts"]
        baseline.append(dict(run=name,report=dict(path=str(path),sha256=p.ct.sha(path)),
            status=audit["status"],full_trace_accepted=audit["full_trace_accepted"],
            formal_observed_closed=m["bidirectional_correlation"]["observed_correlation_closed"],
            kernel_launches=counts["kernel_observed"],launches_per_8_output_tokens=counts["kernel_observed"]/8,
            groups=m["kernel_categories"],
            comparison_scope="Historical public profile; source/cache geometry/logging and server-marker window differ. Descriptive kernel counts, no allocator-only causal or wall timing comparison."))
    comparison=dict(before=before["official_audit"],after=after["official_audit"],
        actual_text_tokens_match=True,kernel_name_launch_shape_count_per_uuid_exact_match=True,
        kernel_launch_signatures_digest=after["launch_signature_digest"],pure_tensor_shapes_from_trace=None,
        tensor_shape_caveat="Launch dimensions are observed; tensor arguments/pointers and pure prefill/decode phase identities are not exported",
        historical_public_profiles=baseline,full_trace_accepted=False,benchmark=False)
    write("formal-comparison.json",comparison)
    print(json.dumps(dict(status="formal_observed_closed_both_full_trace_not_accepted",launch_shapes_match=True,
        before_groups=before["groups"],after_groups=after["groups"],
        before_gap=before["both_uuid_observed_activity_union"],after_gap=after["both_uuid_observed_activity_union"])))

if __name__=="__main__":main()
