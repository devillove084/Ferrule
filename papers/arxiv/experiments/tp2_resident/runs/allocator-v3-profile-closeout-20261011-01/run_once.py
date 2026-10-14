"""One existing paired profile invocation with durable phase/PID receipt; no retry loop."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import hashlib

OUT = Path(__file__).resolve().parent
PAPER = OUT.parents[1]
label = sys.argv[1]
assert label in ("before", "after")
run = PAPER/("runs/allocator-v3-profile-"+label+"-20261011-01")
assert not (run/"launch.json").exists() and not (run/"execution.json").exists()
authorization = json.loads((OUT/"renewed-authorization.json").read_text())
assert authorization["authorized"] is True
argv = [sys.executable,"-S","-B",str(PAPER/"paired.py"),"run","--id",run.name,
        "--purpose","profiling","--sole-gpu-owner-granted","--owner-note",authorization["note"],
        "--probe-grant",str(run/"authorized-probe-grant.json")]
receipt = dict(argv=argv,cwd=os.getcwd(),started_unix=time.time(),wrapper_pid=os.getpid(),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    environment={k:os.environ[k] for k in ("CUDA_VISIBLE_DEVICES","NVIDIA_VISIBLE_DEVICES")},
    toolphase="before_popen",process_started=False,automatic_retry=False,
    authenticated_supervisor_seconds=1600,capture_seconds=240,outer_wait_seconds=2600)
start = OUT/(label+"-command-start.json")
with start.open("x") as f: json.dump(receipt,f,indent=2)
log = OUT/(label+"-command.log")
with log.open("x") as stream:
    process = subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT)
    receipt.update(toolphase="paired_process_started",process_started=True,child_pid=process.pid)
    receipt["child_proc_stat"] = Path("/proc",str(process.pid),"stat").read_text()
    with (OUT/(label+"-process-started.json")).open("x") as f: json.dump(receipt,f,indent=2)
    try:
        code = process.wait(timeout=2600)
        receipt.update(toolphase="paired_process_exited",exit_code=code)
    except BaseException as error:
        receipt.update(toolphase="outer_wait_interrupted",error=repr(error),child_poll=process.poll(),
            followup="Inspect existing task and authenticated receipts. Do not start another profile or retry.")
        with (OUT/(label+"-command-interrupted.json")).open("x") as f: json.dump(receipt,f,indent=2)
        raise
receipt.update(elapsed_seconds=time.time()-receipt["started_unix"],log=str(log),
    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
with (OUT/(label+"-command.json")).open("x") as f: json.dump(receipt,f,indent=2)
print(log.read_text())
raise SystemExit(code)
