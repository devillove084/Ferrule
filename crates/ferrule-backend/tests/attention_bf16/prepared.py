"""CPU-only preparation; explicit grant required by run-after-grant.sh.

Uses frozen independent HF stage files without rewriting them. Full production
native path is built with diagnostic-only stage hooks. No model execution.
"""
import json
import os
import shutil
import sys
from pathlib import Path
import numpy as np
from suite import HERE, ROOT, NATIVE, NVCC, TOOLCHAIN, execute, sha, write_json, write_jobs
SOURCE=ROOT/'target/validation/bf16-attention-20261007/candidate-suite-v2'

def prepare(out, rust_build):
    os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')
    out.mkdir(parents=True,exist_ok=False)
    cases=json.loads((SOURCE/'cases.json').read_text())
    entries=[]
    for meta in cases:
        source=SOURCE/meta['name']
        for append in [0,1]:
            entry=dict(meta,name=meta['name']+('-append' if append else '-read'),append=append)
            entries.append(entry);d=out/entry['name'];d.mkdir()
            (d/'shape.txt').write_text(' '.join(str(meta[k]) for k in ['rows','heads','kv_heads','dim','context','pages'])+'\n')
            for p in source.iterdir():
                if p.suffix in ['.bf16','.f32','.i32','.json'] and not p.name.startswith('candidate-'):
                    shutil.copyfile(p,d/p.name)
            kh,n,dim=[meta[k] for k in ['kv_heads','context','dim']]
            for name in ['key','value']:
                values=np.fromfile(source/(name+'.bf16'),np.uint16).reshape(kh,n,dim)
                values[:,meta['positions'],:].transpose(1,0,2).copy().tofile(d/('append-'+name+'.bf16'))
    build=execute([NVCC,'-std=c++17','-O3','-arch=sm_86','--cudadevrt=none','--cudart=shared',
        '--Werror=all-warnings','-Xptxas=-v','-I',NATIVE,HERE/'prepared.cu','-L',TOOLCHAIN/'link','-lcublas','-o',out/'native'],dict(os.environ),120)
    write_json(out/'build.json',build);assert build['returncode']==0,build['stderr']
    cpu=execute([out/'native','--cpu-contract'],dict(os.environ),30)
    write_json(out/'native-cpu.json',cpu);assert cpu['returncode']==0,cpu
    print(cpu['stdout'].strip())
    write_json(out/'cases.json',entries)
    (out/'cases.txt').write_text(''.join(c['name']+' '+str(c['append'])+'\n' for c in entries))
    pins={str(p):sha(p) for p in [HERE/'prepared.py',HERE/'prepared.cu',HERE/'suite.py',HERE/'../ordinary_norm/buffers.cuh',
        NATIVE/'core/attention_bf16.cuh',NATIVE/'core/attention_bf16_abi.h',NATIVE/'core/attention_ops.cuh',NATIVE/'core/device.cuh',
        NATIVE/'core/validation.cuh',NATIVE/'core/abi.h',NATIVE/'core/tensor_ops.cuh']}
    pins.update({str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    jobs=[dict(name=c['name'],argv=[str(out/'native'),str(out/c['name']),*[str(c[k]) for k in ['rows','heads','kv_heads','dim','context','pages','append']]],seconds=60,pins=pins,env=dict(CUBLAS_WORKSPACE_CONFIG=':0:0')) for c in entries]
    jobs.append(dict(name='compare',argv=['/opt/conda310/bin/python','-B',str(HERE/'prepared.py'),'compare',str(out)],seconds=30,pins=pins,env=dict(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')))
    # CPU-built Rust binary exercises the public typed operator without stage
    # hooks. Preserve immutable per-job environment and existing custody wrapper.
    artifacts=[json.loads(line) for line in rust_build.read_text().splitlines() if line.startswith('{')]
    binary=next(Path(a['executable']) for a in artifacts if a.get('target',{}).get('name')=='cuda_prepared_bf16_attention' and a.get('executable'))
    rust_sources=[ROOT/'crates/ferrule-backend/src/bf16_attention_plan.rs',
        ROOT/'crates/ferrule-backend/src/cuda/operators/attention/prepared_bf16.rs',
        ROOT/'crates/ferrule-backend/tests/cuda_prepared_bf16_attention.rs',
        HERE/'owner.rs',HERE/'review.rs',HERE/'run-after-grant.sh',
        ROOT/'crates/ferrule-backend/src/cuda/runtime.rs',
        ROOT/'crates/ferrule-backend/src/cuda/providers/admission.rs',
        ROOT/'crates/ferrule-backend/src/cuda/providers/bf16_gemmex.rs',
        ROOT/'crates/ferrule-backend/src/cuda/operators/linear/bf16_gemmex.rs',
        ROOT/'crates/ferrule-backend/src/cuda/operators/kv/transaction/bf16.rs',
        HERE/'transaction.rs']
    for source in rust_sources:pins[str(source)]=sha(source)
    pinned_binary=out/'typed-tests';shutil.copyfile(binary,pinned_binary);pinned_binary.chmod(0o755)
    pins[str(pinned_binary)]=sha(pinned_binary)
    for test in ['frozen_prepared_attention_full_outputs_and_resources','ragged_prepared_attention_and_prelaunch_rejections']:
        jobs.insert(-1,dict(name=test,argv=[str(pinned_binary),'--ignored','--exact',test,'--test-threads=1','--nocapture'],seconds=120,pins=pins,
            env=dict(CUDA_VISIBLE_DEVICES='0',NVIDIA_VISIBLE_DEVICES='all',CUBLAS_WORKSPACE_CONFIG=':0:0',FERRULE_ATTENTION_FIXTURE=str(out))))
    write_jobs(out,jobs)
    write_json(out/'PREPARED.json',dict(gpu_executed=False,cases=len(entries),source=str(SOURCE),stage_oracle='unchanged independently captured HF eager tensors',pins=pins))
    print('Prepared',len(entries),'full-native cases; GPU NOT executed')

def compare(out):
    results={}
    def expanded(path):return (np.fromfile(path,np.uint16).astype(np.uint32)<<16).view(np.float32)
    def diff(a,b):
        assert a.shape==b.shape
        ix=np.flatnonzero(a.view(np.uint32)!=b.view(np.uint32))
        return dict(unequal=len(ix),elements=a.size,first=ix[:8].tolist(),max_abs=float(np.max(np.abs(a-b))))
    for c in json.loads((out/'cases.json').read_text()):
        d=out/c['name'];stages={};extent=c['rows']*c['heads']*c['context']
        for stage,ref in [('qk','qk'),('prob','prob-bf16'),('pv','pv')]:
            stages[stage]=diff(expanded(d/('native-'+stage+'.bf16')),np.fromfile(d/('hf-'+ref+'.f32'),np.float32))
        for repeat in range(2):
            trace=np.fromfile(d/('native-softmax-'+str(repeat)+'.f32'),np.float32).reshape(3,extent)
            for i,stage in enumerate(['scaled','masked','prob-f32']):
                stages[stage+'-'+str(repeat)]=diff(trace[i],np.fromfile(d/('hf-'+stage+'.f32'),np.float32))
            for stage in ['raw','gated']:
                stages[stage+'-'+str(repeat)]=diff(expanded(d/('native-'+stage+'-'+str(repeat)+'.bf16')),np.fromfile(d/('hf-'+stage+'.f32'),np.float32))
        results[c['name']]=stages
    write_json(out/'comparison.json',results)
    failures={n:{s:v['unequal'] for s,v in stages.items() if v['unequal']} for n,stages in results.items()}
    print(json.dumps(failures,indent=2));assert not any(failures.values()),'Full stage mismatch; never accepted from matrix-only evidence'
    print(len(results),'cases, all 13 stage/repeat checks exact')
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare(Path(sys.argv[2]).resolve(),Path(sys.argv[3]).resolve())
    elif sys.argv[1]=='compare':compare(Path(sys.argv[2]).resolve())
    else:raise ValueError('prepare or compare')
