#!/usr/bin/env python3
"""Small-task adapter using the existing paper auth/lock/subreaper, not full40."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
import launch as l
import routed_down_contract as rc


def validate(p):
    l.ct.require_cpu_python()
    rc.bounds(p)
    l.telemetry.authorize(p['sole_gpu_owner_granted'],p['gpu_uuid'],p['owner_note'],'device_exclusive_local_custody')
    rc.require(p['execute_cuda'] is True and p['pins']==l.pins() and p['protocol']==l.protocol(), 'Source/protocol/task differs')
    rc.require(p['python']=='/opt/conda310/bin/python' and rc.sha(p['python'])==p['python_sha256'], 'Numerical Python differs')


def all_idle(note):
    result=subprocess.run(['/usr/bin/nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=10)
    uuids=result.stdout.splitlines()
    rc.require(len(uuids)==8 and len(set(uuids))==8,'Expected eight devices')
    return [l.idle(uuid.strip(),note) for uuid in uuids]


def worker(pin):
    p=json.load(sys.stdin)
    auth=l.guard.authenticate(p,pin)
    validate(p)
    l.ct.validate_owner_lock(p['owner_lock'],p['gpu_uuid'],p['coordinator']['pid'])
    rc.require(os.getpid()==os.getpgrp(),'Worker session differs')
    l.ct.write(Path(p['directory'])/'authentication.json',auth)
    env=dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=p['gpu_uuid'],NVIDIA_VISIBLE_DEVICES=p['gpu_uuid'],HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        HF_HUB_DISABLE_TELEMETRY='1',FERRULE_PAPER_WORKER_TOKEN=p['worker_token'],TOKENIZERS_PARALLELISM='false',
        ATEN_CPU_CAPABILITY='avx512',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
    code='import json,sys;sys.path.insert(0,sys.argv[1]);import routed_down_replay;routed_down_replay.export(json.load(sys.stdin),sys.argv[2])'
    return subprocess.run([p['python'],'-B','-c',code,str(l.HERE),pin],input=json.dumps(p),text=True,env=env,cwd=l.ROOT,timeout=120).returncode


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--execute-cuda',action='store_true')
    ap.add_argument('--sole-gpu-owner-granted',action='store_true')
    ap.add_argument('--gpu-uuid')
    ap.add_argument('--owner-note',default='')
    ap.add_argument('--_worker',help=argparse.SUPPRESS)
    a=ap.parse_args(argv);l.ct.require_cpu_python()
    if a._worker:return worker(a._worker)
    if not a.execute_cuda:
        print(json.dumps(dict(status='cpu_only_not_executed',timeout=120,capture_mib=16,cache_mib=64,task='routed-down-m1',gpu_queries=False)))
        return 0
    l.telemetry.authorize(a.sole_gpu_owner_granted,a.gpu_uuid,a.owner_note,'device_exclusive_local_custody')
    rc.require(a.output is not None,'Fresh output required')
    out=a.output.resolve()
    p=dict(task='routed-down-m1',directory=str(out),execute_cuda=True,sole_gpu_owner_granted=a.sole_gpu_owner_granted,
        gpu_uuid=a.gpu_uuid,owner_note=a.owner_note,python='/opt/conda310/bin/python',python_sha256=rc.sha('/opt/conda310/bin/python'),
        timeout=120,capture_mib=16,cache_mib=64,bundle_sha256=rc.BUNDLE_PIN,reference_sha256=rc.REFERENCE_PIN,pins=l.pins(),protocol=l.protocol())
    validate(p);rc.require(not out.exists(),'Refusing existing output');rc.bundle()
    with l.ct.exclusive_owner_lock(a.gpu_uuid) as lock:
        before=all_idle(a.owner_note);rc.require(a.gpu_uuid in [s['uuid'] for s in before],'Granted device missing')
        out.mkdir();l.ct.write(out/'all-before.json',before)
        p['owner_lock']=lock;p,pin=l.guard.seal(p,out)
        env=dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',PYTHONDONTWRITEBYTECODE='1',FERRULE_PAPER_WORKER_TOKEN=p['worker_token'])
        argv=l.ct.isolated_argv([sys.executable,'-S','-B',str(Path(__file__).resolve()),'--_worker',pin],env)
        final=dict(schema='ferrule.paper.routed-down-fixture-release.v1',status='incomplete',model_fixture_ready=False)
        try:
            result=l.supervisor.run_process(argv,120,p);l.ct.write(out/'supervisor.json',result)
            if l.ct.drained(result):l.ct.write(out/'all-after.json',all_idle(a.owner_note))
            rc.require(l.ct.successful_process(result),'Replay failed/timed out/custody unknown')
            final=l.ct.load(out/'export.json')
            rc.require(final['status']=='pending_custody' and len(final['fixtures'])==4 and final['selected_rank_rows']==8 and final['down_calls']==64 and final['expert_forward_calls']==8,'Missing complete replay')
            for row in final['fixtures']:
                folder=out/row['directory'];rc.check_sums(folder,'SHA256SUMS',row['sha256'],rc.FILES,16*1024**2)
            rc.require(final['workflow_pins']==l.pins(),'Source changed')
            rc.bundle()
            final.update(status='complete',model_fixture_ready=True,custody_sha256=rc.sha(out/'supervisor.json'),
                         all_before_sha256=rc.sha(out/'all-before.json'),all_after_sha256=rc.sha(out/'all-after.json'),launch_sha256=pin)
        except BaseException as e:
            final.update(status='incomplete',model_fixture_ready=False,error=str(e));raise
        finally:
            l.ct.write(out/'manifest.json',final)
            with (out/'manifest.sha256').open('x') as f:f.write(rc.sha(out/'manifest.json')+'\n')
    print(json.dumps(dict(status=final['status'],manifest_sha256=rc.sha(out/'manifest.json'),fixtures=final['fixtures'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
