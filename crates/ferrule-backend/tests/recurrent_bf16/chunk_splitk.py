"""Independent matrix-only HF/native reduction evidence. No model or provider."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from performance import ROOT,HERE,NATIVE,NVCC,TOOLCHAIN,CUDA_LIB,execute,sha,write_json


def prepare(out):
    import numpy as np
    from safetensors import safe_open
    out.mkdir(parents=True,exist_ok=False)
    cases=[];shapes={};retained_pins={}
    for case in ['near-512-511','limit-512-512']:
        d=out/case;d.mkdir();cases.append(case);shapes[case]=[64,1,1]
        source=ROOT/'target/validation/chunk-continuation-performance-gpu-20261006/stage-handoff'/case/'hf-observed-stages.safetensors'
        retained_pins[str(source)]=sha(source)
        with safe_open(str(source),framework='numpy') as f:
            a=f.get_tensor('k_beta').reshape(64,512);b=f.get_tensor('knorm').reshape(64,512);expected=f.get_tensor('matmul0').reshape(64,64)
        a.tofile(d/'lhs.f32');b.tofile(d/'rhs-transposed.f32');expected.tofile(d/'retained.f32')
    rng=np.random.default_rng(20261006)
    for kind in ['random-signed','random-positive','dynamic','cancelling']:
        d=out/kind;d.mkdir();cases.append(kind);shapes[kind]=[64,1,1]
        a=rng.normal(size=(64,512)).astype(np.float32);b=rng.normal(size=(64,512)).astype(np.float32)
        if kind=='random-positive':a=np.abs(a);b=np.abs(b)
        if kind=='dynamic':a*=np.exp2(rng.integers(-8,8,size=a.shape)).astype(np.float32);b*=np.exp2(rng.integers(-8,8,size=b.shape)).astype(np.float32)
        if kind=='cancelling':a[:,1::2]=-a[:,::2];b[:,1::2]=b[:,::2]+np.float32(0.00001)
        a.tofile(d/'lhs.f32');b.tofile(d/'rhs-transposed.f32')
    for case in ['near-512-511','limit-512-512']:
        n=511 if case=='near-512-511' else 512
        for j in [4,5]:
            name=case+'-state'+str(j);d=out/name;d.mkdir();cases.append(name);shapes[name]=[n,0,1]
            src=ROOT/'target/validation/gdn-chunk-splitk-20261006/state-matrices'/(case+'-'+str(j))
            for source,dest in [('lhs.f32','lhs.f32'),('rhs.f32','rhs-transposed.f32'),('expected.f32','retained.f32')]:
                retained_pins[str(src/source)]=sha(src/source)
                (d/dest).write_bytes((src/source).read_bytes())
    for n in [511,512]:
        name='random-state-'+str(n);d=out/name;d.mkdir();cases.append(name);shapes[name]=[n,0,1]
        rng.normal(size=(64,512)).astype(np.float32).tofile(d/'lhs.f32')
        rng.normal(size=(512,n)).astype(np.float32).tofile(d/'rhs-transposed.f32')
    for batches in [2,3]:
        name='batched-'+str(batches);d=out/name;d.mkdir();cases.append(name);shapes[name]=[64,1,batches]
        rng.normal(size=(batches,64,512)).astype(np.float32).tofile(d/'lhs.f32')
        rng.normal(size=(batches,64,512)).astype(np.float32).tofile(d/'rhs-transposed.f32')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=CUDA_LIB+':/usr/lib64')
    result=execute([NVCC,'-std=c++17','-O3','-arch=sm_86','--cudadevrt=none','--cudart=shared','-I',NATIVE,HERE/'chunk_splitk.cu','-L',TOOLCHAIN/'link','-o',out/'candidate'],env,120)
    write_json(out/'build.json',result);assert result['returncode']==0,result['stderr']
    write_json(out/'prepared.json',dict(cases=cases,shapes=shapes,retained_pins=retained_pins,source_pins={str(p):sha(p) for p in [Path(__file__),HERE/'chunk_splitk.cu',NATIVE/'core/recurrent_bf16_chunk.cuh',NATIVE/'core/recurrent_bf16_chunk_abi.h',NATIVE/'core/device.cuh',NATIVE/'core/validation.cuh']},artifact_pins={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))


def hf(out):
    import torch
    sys.path.insert(0,str(ROOT/'scripts'));import qwen35_fp8_bf16_reference as ref
    pins=ref.initialize();reports={}
    with torch.inference_mode():
        for case in json.loads((out/'prepared.json').read_text())['cases']:
            d=out/case
            n,tb,batches=json.loads((out/'prepared.json').read_text())['shapes'][case]
            def load(name,shape):return torch.frombuffer(bytearray((d/name).read_bytes()),dtype=torch.float32).reshape(shape).clone().cuda()
            a=load('lhs.f32',(1,batches,1,64,512))
            b=load('rhs-transposed.f32',(1,batches,1,n,512) if tb else (1,batches,1,512,n))
            if tb:b=b.transpose(-1,-2)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:y=a@b
            value=y.cpu().contiguous().numpy().tobytes();(d/'hf.f32').write_bytes(value)
            if (d/'retained.f32').exists():assert value==(d/'retained.f32').read_bytes()
            reports[case]=[e.name for e in prof.events() if 'CUDA' in str(e.device_type)]
    write_json(out/'hf.json',dict(pins=pins,kernels=reports))


def run(out,evidence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and not os.environ.get('LD_PRELOAD')
    m=json.loads((out/'prepared.json').read_text())
    for p,h in m['source_pins'].items():assert sha(p)==h
    for p,h in m['retained_pins'].items():assert sha(p)==h
    for p,h in m['artifact_pins'].items():assert sha(out/p)==h
    evidence.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'scripts'));import test_vllm_qwen35 as supervisor
    def snapshot():return {k:execute(['nvidia-smi',q,'--format=csv,noheader'],dict(os.environ),10) for k,q in [('gpu','--query-gpu=index,uuid,memory.used,utilization.gpu'),('apps','--query-compute-apps=pid,gpu_uuid,used_gpu_memory')]}
    def idle(s):return s['gpu']['returncode']==s['apps']['returncode']==0 and not s['apps']['stdout'].strip() and len(s['gpu']['stdout'].splitlines())==8 and all(l.split(',')[-2].strip()=='0 MiB' and l.split(',')[-1].strip()=='0 %' for l in s['gpu']['stdout'].splitlines())
    def custody(r):return r['custody'].get('containment_verified') and r['custody'].get('echild') and r['descendant_custody']=='proven_drained' and not r['cleanup_errors'] and not r.get('error')
    before=snapshot();write_json(evidence/'before.json',before);assert idle(before)
    receipts=[]
    try:
        libs=json.loads((ROOT/'target/validation/bf16-layer01-supported-retry-20261005/environment-preflight.json').read_text())['resolved_libraries']
        for lib in libs:assert sha(lib['realpath'])==lib['sha256']
        write_json(evidence/'launch.json',dict(libraries=libs,prepared_sha256=sha(out/'prepared.json')))
        cmds=[('hf',['/opt/conda310/bin/python','-B',str(Path(__file__).resolve()),'hf',str(out)])]+[(c,[str(out/'candidate'),str(out/c),*map(str,m['shapes'][c])]) for c in m['cases']]
        for name,cmd in cmds:
            os.environ['LD_LIBRARY_PATH']='' if name=='hf' else CUDA_LIB+':/usr/lib64';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            r=supervisor.run_process(cmd,120);receipts.append(r);write_json(evidence/(name+'-supervisor.json'),r)
            assert not r['returncode'] and not r['timed_out'] and custody(r),r['stderr']
        import numpy as np
        result={}
        for c in m['cases']:
            a=np.fromfile(out/c/'candidate.f32',np.float32);b=np.fromfile(out/c/'hf.f32',np.float32)
            result[c]=dict(elements=len(a),unequal=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))),max_abs=float(np.max(np.abs(a-b))))
        write_json(evidence/'comparison.json',result);print(json.dumps(result,indent=2))
        assert all(v['unequal']==0 for v in result.values()), 'Exact matrix gate failed'
    finally:
        after=[]
        for _ in range(3):time.sleep(1);after.append(snapshot())
        released=bool(receipts) and all(custody(r) for r in receipts) and all(idle(s) for s in after)
        write_json(evidence/'release.json',dict(after=after,release='released' if released else 'unverified'));assert released


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','hf']);p.add_argument('directory',type=Path);p.add_argument('--evidence',type=Path);a=p.parse_args();out=a.directory.resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='hf':hf(out)
    else:
        if a.evidence is None:p.error('--evidence required')
        run(out,a.evidence.resolve())
