"""Execute only the bounded, hash-pinned GEMV contract plans after sole-GPU grant."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def verify(prepared):
    for path,pin in prepared['pins'].items():
        if sha(path)!=pin: raise ValueError('Changed pinned file: '+path)

def contract(calls,cases):
    if any(c.get('api')!='cublasGemmEx' for c in calls): raise ValueError('Unexpected public API')
    expected={(c['n'],1,c['k']) for c in cases}
    bf=[c for c in calls if c['ctype']==14]
    if not bf or { (c['m'],c['n'],c['k']) for c in bf } != expected:
        raise ValueError('Native BF16 GEMM dimensions not observed')
    for c in bf:
        wanted=dict(transa=1,transb=0,atype=14,btype=14,ctype=14,
                    compute=68,algo=99,math=16,atomics=0,pointer_mode=0,
                    alpha=1,beta=0,workspace_known=True,workspace_bytes=32<<20,status=0)
        for key,value in wanted.items():
            if c[key]!=value: raise ValueError('Native call differs: '+key)
        if c['lda']!=c['k'] or c['ldb']!=c['k'] or c['ldc']!=c['m']:
            raise ValueError('Native leading dimensions differ')
        if any(c[k]!=0 for k in ['a_mod256','b_mod256','c_mod256']):
            raise ValueError('Native operand alignment differs')
    return dict(calls=len(bf),matched=True)

def run(directory,label,execute_cuda,sole_owner):
    if not execute_cuda or not sole_owner:
        raise ValueError('Explicit --execute-cuda --sole-owner-permission required')
    if os.environ.get('LD_PRELOAD'):
        raise ValueError('Unexpected inherited LD_PRELOAD')
    prepared=json.loads((directory/'PREPARED.json').read_text());verify(prepared)
    plan=prepared['plans'][label];receipt=directory/(label+'-supervisor.json')
    if receipt.exists():raise ValueError('Fresh receipt required')
    for dependency in plan.get('after',[]):
        d=json.loads((directory/(dependency+'-supervisor.json')).read_text())
        if not d['passed']:raise ValueError('Prerequisite failed: '+dependency)
    if label.startswith('native-'):
        contract([json.loads(l) for l in (directory/'hf-calls.jsonl').read_text().splitlines()],prepared['cases'])
        plain=json.loads((directory/'hf-plain/report.json').read_text())
        logged=json.loads((directory/'hf-logged/report.json').read_text())
        if [c['output_sha256'] for c in plain['cases']] != [c['output_sha256'] for c in logged['cases']]:
            raise ValueError('Call observation changed native outputs')
    os.environ.update(plan['environment'])
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID';os.environ['NVIDIA_VISIBLE_DEVICES']='all'
    def query(command):return subprocess.check_output(command,text=True,timeout=15)
    def gpu():
        g=query(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader'])
        p=query(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader'])
        return dict(time_ns=time.time_ns(),gpus=g,processes=p,idle=len(g.splitlines())==8 and not p.strip() and all(x.split(',')[-2].strip()=='0 MiB' and x.split(',')[-1].strip()=='0 %' for x in g.splitlines()))
    before=gpu()
    if not before['idle']:raise ValueError('Other owner/device activity: refuse launch')
    if label.startswith('native-'):
        # cuBLAS12.4-built host-only probe, run in separate processes with exact
        # pinned12.4 or12.9 libraries. No CUDA device code from Ferrule is loaded.
        (directory/label).mkdir(exist_ok=False)
        link=query(['ldd',plan['command'][0]])
        if 'not found' in link or plan['library_directory']+'/libcublas.so.12' not in link:
            raise ValueError('Wrong loaded native libraries')
    else:link=None
    if label=='hf-logged':
        with (directory/'hf-calls.jsonl').open('x'): pass
    sys.path.insert(0,str(ROOT/'scripts'))
    from test_vllm_qwen35 import run_process
    result=run_process(plan['command'],plan['timeout'])
    # Do not inject the test observer into quiescence tools.
    os.environ.pop('LD_PRELOAD',None)
    after=[]
    for _ in range(3):time.sleep(1);after.append(gpu())
    h=result.get('handshake',{});c=result.get('custody',{})
    custody=(h.get('containment_verified') is True and c.get('containment_verified') is True and c.get('status')=='proven_drained' and c.get('echild') is True and h.get('launch_id')==c.get('launch_id') and h.get('helper_identity')==c.get('helper_identity'))
    unchanged=True
    try:verify(prepared)
    except ValueError:unchanged=False
    passed=(result.get('returncode')==0 and custody and unchanged and not result.get('error') and not result.get('timed_out') and not result.get('cleanup_errors') and not result.get('stdout_truncated') and all(x['idle'] for x in after))
    record=dict(label=label,command=plan['command'],environment=plan['environment'],prepared_sha256=sha(directory/'PREPARED.json'),ldd=link,supervisor=result,custody_verified=custody,pins_unchanged=unchanged,gpu_before=before,gpu_after=after,passed=passed)
    with receipt.open('x') as f:json.dump(record,f,indent=2)
    (directory/(label+'.stdout')).write_text(result.get('stdout',''))
    (directory/(label+'.stderr')).write_text(result.get('stderr',''))
    print(json.dumps(dict(label=label,passed=passed,returncode=result.get('returncode'),custody_verified=custody,idle_after=all(x['idle'] for x in after)),indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('label',choices=['hf-plain','hf-logged','native-129','native-124']);p.add_argument('--execute-cuda',action='store_true');p.add_argument('--sole-owner-permission',action='store_true');a=p.parse_args();run(a.directory.resolve(),a.label,a.execute_cuda,a.sole_owner_permission)
