import json
import math
from pathlib import Path
import struct
import sys
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
HERE=OUT.parents[1]
sys.path.insert(0,str(HERE))
import routed_down_contract as rc
import routed_down_launch as launcher
PIN="807e5b5643ecce603088052e39fbee36ce15dca4a3ba319b51be120e2922d40c"
require=rc.require
require(rc.sha(OUT/"manifest.json")==PIN,"Release pin differs")
m=json.loads((OUT/"manifest.json").read_text())
require(m["status"]=="complete" and m["fixture_ready"] and len(m["fixtures"])==10,"Incomplete release")
require(len(m["controls"])==66 and all(c["exact"] and c["actual_sha256"]==c["expected_sha256"] for c in m["controls"]),"Controls differ")
require(m["expert_forward_calls"]==10 and m["gate_up_calls"]==m["down_calls"]==112,"Call inventory")
bundle=rc.pair_bundle()
for path,pin in m["workflow_pins"].items():
    require(rc.sha(path)==pin,"Workflow source differs")
for path in (OUT/"source").iterdir():
    require(rc.sha(path)==m["workflow_pins"][str(HERE/path.name)],"Frozen source snapshot differs")
for filename,key in (("supervisor.json","custody_sha256"),("all-before.json","all_before_sha256"),("all-after.json","all_after_sha256"),
                     ("worker-launch.json","launch_sha256"),("numerical-authentication.json","numerical_authentication_sha256")):
    require(rc.sha(OUT/filename)==m[key],"Generation identity differs: "+filename)
records=[]
for fixture in m["fixtures"]:
    folder=OUT/fixture["directory"]
    rc.check_sums(folder,"SHA256SUMS",fixture["sha256"],rc.PAIR_FILES)
    child=json.loads((folder/"manifest.json").read_text())
    rows,rank=child["rows"],child["rank"]
    require(child["fixture_ready"] and (child["n"],child["k"],child["packed_n"])==(256,2048,512),"Pair geometry")
    for filename,spec in child["files"].items():
        raw=(folder/filename).read_bytes()
        require(len(raw)==spec["bytes"]==math.prod(spec["shape"])*(1 if spec["dtype"]=="F8_E4M3" else 2),"Raw extent")
        require(rc.sha(folder/filename)==spec["sha256"],"Raw file hash")
        if spec["dtype"]=="F8_E4M3":
            require(all(b & 0x7f != 0x7f for b in raw),"FP8 NaN")
        else:
            words=[v[0] for v in struct.iter_unpack("<H",raw)]
            require(all(w & 0x7f80 != 0x7f80 for w in words),"Nonfinite BF16")
            if ".scale." in filename:require(all(0<w<0x7f80 for w in words),"Nonpositive BF16 scale")
        if filename.startswith(("gate.","up.")):
            source=spec["source"];entry=bundle["original_payloads"][source["tensor"]]
            path=Path(entry.get("source_directory",rc.PAIR_BUNDLE))/entry["file"]
            require(rc.sha(path)==entry["sha256"]==entry["original_used_payload"]["sha256"],"Original byte pin")
            original=path.read_bytes();half=len(original)//2
            require(raw==original[rank*half:(rank+1)*half],"Original N-axis partition differs")
    normalized=(folder/"normalized_hidden.bf16").read_bytes()
    order=child["gate_input_token_indices"]
    require(order==([0] if rows==1 else [1,2,4,3,0]),"Native expert row order")
    require((folder/"gateupinput.bf16").read_bytes()==b"".join(normalized[i*4096:(i+1)*4096] for i in order),"Actual argument differs from captured normalized rows")
    if rows==1:
        old=rc.OLD_CUDA/fixture["directory"].rsplit("-rank",1)[0]/("rank%d.input.bf16" % rank)
        require((folder/"hf.activation.bf16").read_bytes()==old.read_bytes(),"Immutable M1 expected changed")
    records.append(dict(directory=fixture["directory"],sha256=fixture["sha256"],rows=rows,rank=rank,finite=True,original_payload_exact=True,token_order=order))
supervisor=json.loads((OUT/"supervisor.json").read_text())
require(launcher.l.ct.successful_process(supervisor),"Custody not proven drained")
for filename in ("all-before.json","all-after.json"):
    samples=json.loads((OUT/filename).read_text())
    require(len(samples)==8 and all(not x["apps"] and x["gpu_bytes"]==x["utilization"]==0 for x in samples),"Pre/post idle differs")
auth=json.loads((OUT/"numerical-authentication.json").read_text())
absent=[]
for role in ("child","worker","helper","coordinator"):
    identity=auth[role]
    try:live=launcher.l.guard.identity(identity["pid"])
    except FileNotFoundError:live=None
    require(live is None or live["birth_ticks"]!=identity["birth_ticks"],"Owned process still alive: "+role)
    absent.append(dict(role=role,identity=identity,owned_birth_absent=True))
uuid=m["selected_device"]["cuda_visible"]
require(uuid=="GPU-b980f22b-58a3-fc88-21c1-ae86053d8535" and m["selected_device"]["physical_index"]==0,"Granted device differs")
with launcher.l.ct.exclusive_owner_lock(uuid) as lock:
    final_idle=launcher.all_idle("Post-capture CPU validation and final GPU release; no CUDA numerical work")
    lock_proof={k:v for k,v in lock.items() if k!="fd"}
report=dict(status="pass_cpu_validation_gpu_released",manifest_sha256=PIN,
    validator_source_sha256=rc.sha(Path(__file__)),fixtures=records,exact_controls=66,
    actual_native_expert_calls=10,actual_packed_gate_up_calls=112,actual_down_calls=112,
    source_pins_verified=True,old_preparations_unchanged=True,fixture_ready=True,
    owned_births_absent=absent,shared_lock_available_then_released=lock_proof,
    final_idle=final_idle,backend_candidate_executed=False,
    claim="Independent HF CUDA fixtures ready; backend numeric equivalence is a separate validator gate")
path=OUT.parent/(OUT.name+".validation.json")
with path.open("x") as f:json.dump(report,f,indent=2);f.write("\n")
print(json.dumps(dict(status=report["status"],validation=str(path),validation_sha256=rc.sha(path),fixtures=len(records),controls=66,final_idle_devices=len(final_idle)),indent=2))
