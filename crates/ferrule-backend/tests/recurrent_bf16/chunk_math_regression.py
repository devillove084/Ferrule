"""Bounded post-fix continuation replay against retained independent HF files.

CPU prepare does not create references. GPU run requires explicit ownership and
uses the existing authenticated supervisor. Optional timing follows ALL exact gates;
no model execution.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import chunk_continuation as continuation
from performance import ROOT, HERE, CUDA_LIB, execute, sha, write_json

CAPTURE=ROOT/'target/validation/chunk-continuation-performance-gpu-20261006'
ORIGINAL=ROOT/'target/validation/chunk-continuation-cpu-20261006/prepared'
HANDOFF_SHA='27f3ec44237fbea9af4cf79c030cac7aedd5b38b16197d63c1b861ba39093c27'
EXPECTED_SHA='2d4cdcf4c6dba60cea08620c40c02fcdc05ef674a8e9cce6b0ffdd240e9e6c1d'


def retained_pins(prepared):
    assert sha(CAPTURE/'BACKEND-HANDOFF.json')==HANDOFF_SHA
    assert sha(CAPTURE/'expected/manifest.json')==EXPECTED_SHA
    handoff=json.loads((CAPTURE/'BACKEND-HANDOFF.json').read_text())
    assert sha(ORIGINAL/'prepared.json')==handoff['prepared_manifest_sha256']
    oracle=json.loads((CAPTURE/'expected/manifest.json').read_text())
    assert not oracle['native_results_used']
    assert oracle['prepared_sha256']==sha(ORIGINAL/'prepared.json')
    for name,pin in oracle['artifacts'].items():assert sha(CAPTURE/'expected'/name)==pin,name
    old=json.loads((ORIGINAL/'prepared.json').read_text())
    new=json.loads((prepared/'prepared.json').read_text())
    assert old['cases']==new['cases']
    checked={}
    # Same complete binary input payloads, including carried-call and reset seeds.
    for name,pin in old['artifact_pins'].items():
        if name.endswith(('.bf16','.f32')):
            assert sha(ORIGINAL/name)==pin and sha(prepared/name)==pin,name
            checked[name]=pin
    for name,meta in handoff['cases'].items():
        assert sha(CAPTURE/'stage-handoff'/name/'hf-observed-stages.safetensors')==meta['observed_hf_stages_sha256']
    return dict(original_prepared_sha256=sha(ORIGINAL/'prepared.json'),new_prepared_sha256=sha(prepared/'prepared.json'),
        expected_manifest_sha256=EXPECTED_SHA,identical_payloads=checked,references_regenerated=False)


def prepare(out):
    os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')
    continuation.prepare(out)
    proof=retained_pins(out)
    proof['status']='CPU_PREPARED_GPU_NOT_RUN'
    proof['driver_sha256']=sha(__file__)
    proof['bounds_seconds']=dict(each_native_case=120,each_HF_matrix_profile=120)
    write_json(out/'retained-reference-proof.json',proof)
    print('Retained independent HF expectations verified; GPU NOT run')


def profile_matrix(prepared,out,case):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('FERRULE_CHUNK_MATH_PROFILE_WORKER')!='1':
        raise RuntimeError('internal GPU child; use owned run')
    retained_pins(prepared)
    import torch
    from safetensors import safe_open
    sys.path.insert(0,str(ROOT/'scripts'))
    import qwen35_fp8_bf16_reference as ref
    pins=ref.initialize()
    source=CAPTURE/'stage-handoff'/case/'hf-observed-stages.safetensors'
    with safe_open(str(source),framework='pt',device='cpu') as f:
        a=f.get_tensor('k_beta').cuda();k=f.get_tensor('knorm').cuda();expected=f.get_tensor('matmul0')
    b=k.transpose(-1,-2)
    with torch.inference_mode():
        result=a@b
        torch.cuda.synchronize()
        assert torch.equal(result.cpu().view(torch.int32),expected.view(torch.int32))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            result=a@b
        torch.cuda.synchronize()
        assert torch.equal(result.cpu().view(torch.int32),expected.view(torch.int32))
    prof.export_chrome_trace(str(out/'matrix-profile.json'))
    events=[dict(name=e.name,device_type=str(e.device_type),shapes=e.input_shapes) for e in prof.events()]
    write_json(out/'matrix-contract.json',dict(case=case,pins=pins,stage_sha256=sha(source),output_exact=True,
        lhs=dict(shape=list(a.shape),stride=list(a.stride()),dtype=str(a.dtype)),
        rhs=dict(shape=list(b.shape),stride=list(b.stride()),dtype=str(b.dtype)),
        logical_contract='F32 matmul M64/N64/K512; alpha1/beta0; no transpose materialization',
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,matmul_precision=torch.get_float32_matmul_precision(),
        deterministic=torch.are_deterministic_algorithms_enabled(),workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),events=events,
        limitation='Profile identifies actual kernel; BLAS API algorithm/handle details may require an additional API trace before provider changes'))
    print('Original HF matrix-only replay and profile exact: '+case)


def run(prepared,evidence,execute_cuda,performance=False):
    if not execute_cuda or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('LD_PRELOAD'):
        raise RuntimeError('requires --execute-cuda, sole GPU ownership and CUDA_VISIBLE_DEVICES=0')
    manifest=continuation.verify(prepared)
    proof=retained_pins(prepared)
    original_proof=json.loads((prepared/'retained-reference-proof.json').read_text())
    assert original_proof['driver_sha256']==sha(__file__)
    evidence.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'scripts'))
    import test_vllm_qwen35 as supervisor
    def snapshot():
        return {k:execute(['nvidia-smi',q,'--format=csv,noheader'],dict(os.environ),10) for k,q in [
            ('gpu','--query-gpu=index,uuid,memory.used,utilization.gpu'),('apps','--query-compute-apps=pid,gpu_uuid,used_gpu_memory')]}
    def idle(s):
        return s['gpu']['returncode']==s['apps']['returncode']==0 and not s['apps']['stdout'].strip() and len(s['gpu']['stdout'].splitlines())==8 and all(l.split(',')[-2].strip()=='0 MiB' and l.split(',')[-1].strip()=='0 %' for l in s['gpu']['stdout'].splitlines())
    def custody(r):
        h,c=r.get('handshake',{}),r.get('custody',{})
        return h.get('containment_verified') and c.get('containment_verified') and c.get('status')=='proven_drained' and c.get('echild') and h.get('launch_id')==c.get('launch_id') and h.get('helper_identity')==c.get('helper_identity') and not r.get('cleanup_errors') and not r.get('error')
    before=snapshot();write_json(evidence/'before.json',before)
    if not idle(before):raise RuntimeError('all GPUs must be idle/released')
    receipts=[]
    try:
        libs=json.loads((ROOT/'target/validation/bf16-layer01-supported-retry-20261005/environment-preflight.json').read_text())['resolved_libraries']
        for lib in libs:assert sha(lib['realpath'])==lib['sha256']
        write_json(evidence/'launch.json',dict(reference_proof=proof,authenticated_libraries=libs))
        def owned(name,cmd,hf=False):
            os.environ['LD_LIBRARY_PATH']='' if hf else CUDA_LIB+':/usr/lib64'
            os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',FERRULE_CHUNK_MATH_PROFILE_WORKER='1')
            r=supervisor.run_process(list(map(str,cmd)),120);receipts.append(r);write_json(evidence/(name+'-supervisor.json'),r)
            if r['returncode'] or r['timed_out'] or not custody(r):raise RuntimeError(name+' failed; see receipt')
        for case in manifest['cases']:
            d=evidence/case['name'];d.mkdir()
            owned(case['name'],continuation.native_command(prepared,d,case))
        # Capture actual producer kernels on exact operands, not guessed CPU trees.
        for case in ['near-512-511','limit-512-512']:
            d=evidence/('profile-'+case);d.mkdir()
            owned('profile-'+case,['/opt/conda310/bin/python','-B',Path(__file__).resolve(),'profile',prepared,'--evidence',d,'--case',case],True)
        import numpy as np
        def load(path):
            return (np.fromfile(path,dtype='<u2').astype(np.uint32)<<16).view(np.float32) if path.suffix=='.bf16' else np.fromfile(path,dtype='<f4')
        comparison={}
        for case in manifest['cases']:
            name=case['name'];metrics={}
            for prefix in [*['carry-'+str(i) for i in range(len(case['rows']))],'reset']:
                for field in ['output','state']:
                    metrics[prefix+'-'+field]=continuation.metric(load(evidence/name/(prefix+'-'+field+('.bf16' if field=='output' else '.f32'))),load(CAPTURE/'expected'/name/(prefix+'-'+field+'.f32')))
            comparison[name]=metrics
        exact=all(v['unequal']==0 and v['nonfinite']==0 for c in comparison.values() for v in c.values())
        write_json(evidence/'comparison.json',dict(cases=comparison,exact=exact,tolerances=False,references_regenerated=False))
        continuation.verify(prepared);retained_pins(prepared)
        if not exact:raise RuntimeError('Exact continuation gate failed; no performance/default switch; inspect comparison and HF kernel profiles')
        if performance:
            import statistics
            timings={}
            for case in [c for c in manifest['cases'] if c.get('performance')]:
                name=case['name'];d=evidence/('perf-'+name);d.mkdir()
                owned('perf-'+name,continuation.native_command(prepared,d,case,True))
                receipt=json.loads((evidence/('perf-'+name+'-supervisor.json')).read_text())
                samples=[json.loads(line) for line in receipt['stdout'].splitlines() if line.startswith('{')]
                times={v:[r['microseconds'] for r in samples if r.get('event')=='timing' and r['variant']==v] for v in ['scan','chunk']}
                assert all(len(v)==20 for v in times.values()),'Incomplete matched timing'
                errors={v:{field:continuation.metric(load(d/(v+'-'+field+('.bf16' if field=='output' else '.f32'))),load(CAPTURE/'expected'/name/('carry-0-'+field+'.f32'))) for field in ['output','state']} for v in ['scan','chunk']}
                assert all(not v['unequal'] and not v['nonfinite'] for v in errors['chunk'].values()),'Timed chunk failed independent exact guard'
                timings[name]=dict(rows=case['rows'][0],numerical_errors=errors,raw_samples=samples,
                    median_us={k:statistics.median(v) for k,v in times.items()},p95_us={k:float(np.percentile(v,95)) for k,v in times.items()},
                    interpretation='Matched CUDA events; different scan/chunk numerical profiles, not equal-quality speedup. No throughput projection.')
            write_json(evidence/'performance.json',timings)
        continuation.verify(prepared);retained_pins(prepared)
    finally:
        after=[]
        for _ in range(3):time.sleep(1);after.append(snapshot())
        drained=bool(receipts) and all(custody(r) for r in receipts)
        release=drained and all(idle(s) for s in after)
        write_json(evidence/'release.json',dict(drained=drained,samples=after,release='released' if release else 'unverified'))
        if not release:raise RuntimeError('drain/quiescence unverified')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','profile']);p.add_argument('prepared',type=Path)
    p.add_argument('--evidence',type=Path);p.add_argument('--execute-cuda',action='store_true');p.add_argument('--performance',action='store_true');p.add_argument('--case',choices=['near-512-511','limit-512-512'])
    args=p.parse_args();prepared=args.prepared.resolve()
    if args.mode=='prepare':prepare(prepared)
    else:
        if args.evidence is None:p.error('fresh --evidence required')
        if args.mode=='profile':
            if args.case is None:p.error('profile needs --case')
            profile_matrix(prepared,args.evidence.resolve(),args.case)
        else:run(prepared,args.evidence.resolve(),args.execute_cuda,args.performance)
