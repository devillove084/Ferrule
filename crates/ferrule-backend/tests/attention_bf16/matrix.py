"""Isolated matrix contract diagnostics; no production/reference modifications."""
import json
import os
import sys
from pathlib import Path
import numpy as np
from suite import HERE, ROOT, NATIVE, NVCC, TOOLCHAIN, execute, sha, write_json, write_jobs

def bf16(x):
    bits=np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((bits+np.uint32(32767)+((bits>>16)&1))>>16).astype(np.uint16)

def load_bf16(path,shape):
    return (np.fromfile(path,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(shape)

def cpu_diagnostics(src,meta):
    r,h,kh,d,n=[meta[k] for k in ['rows','heads','kv_heads','dim','context']]
    q=load_bf16(src/'query.bf16',(h,r,d));k=load_bf16(src/'key.bf16',(kh,n,d));v=load_bf16(src/'value.bf16',(kh,n,d))
    p=np.fromfile(src/'hf-prob-bf16.f32',np.float32).reshape(h,r,n)
    result={}
    for op,a,b in [('qk',q,k),('pv',p,v.transpose(0,2,1))]:
        b=np.repeat(b,h//kh,axis=0)
        # Diagnostic only: double products isolate F32 accumulation. This is not
        # a CPU fallback, substitute oracle, or a proposed high-precision fix.
        products=a[:,:,None,:].astype(np.float64)*b[:,None,:,:].astype(np.float64)
        expected=np.fromfile(src/('hf-'+op+'.f32'),np.float32).reshape(products.shape[:-1])
        exact=products.sum(-1);serial=np.zeros(products.shape[:-1],np.float32)
        for i in range(products.shape[-1]):serial=(serial.astype(np.float64)+products[...,i]).astype(np.float32)
        ix=np.argwhere(bf16(serial)!=bf16(expected))
        result[op]=dict(elements=expected.size,serial_unequal=len(ix),
            double_diagnostic_unequal=int(np.count_nonzero(bf16(exact)!=bf16(expected))),
            serial_differences=[dict(index=i.tolist(),serial_f32=float(serial[tuple(i)]),
                double_sum=float(exact[tuple(i)]),hf_bf16=float(expected[tuple(i)])) for i in ix[:16]])
        result[op]['strided_halving_tree_diagnostics']={}
        for parts in [2,4,8,16,32,64,128]:
            acc=np.zeros((*products.shape[:-1],parts),np.float32)
            for j in range(products.shape[-1]):acc[...,j%parts]=(acc[...,j%parts].astype(np.float64)+products[...,j]).astype(np.float32)
            stride=parts//2
            while stride:
                acc[...,:stride]=acc[...,:stride]+acc[...,stride:2*stride];stride//=2
            result[op]['strided_halving_tree_diagnostics'][str(parts)]=int(np.count_nonzero(bf16(acc[...,0])!=bf16(expected)))
    return result

def prepare(src,out):
    os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')
    out.mkdir(parents=True,exist_ok=False)
    cases=json.loads((src/'cases.json').read_text());diagnostics={}
    for meta in cases:
        name=meta['name'];inp=src/name;case=out/name;case.mkdir()
        r,h,kh,d,n=[meta[k] for k in ['rows','heads','kv_heads','dim','context']]
        for key in ['key','value']:
            t=np.fromfile(inp/(key+'.bf16'),np.uint16).reshape(kh,n,d)
            np.repeat(t,h//kh,axis=0).tofile(case/('repeat-'+key+'.bf16'))
        p=np.fromfile(inp/'hf-prob-bf16.f32',np.float32)
        assert np.all((p.view(np.uint32)&65535)==0),'Not exact expanded BF16 probabilities'
        (p.view(np.uint32)>>16).astype(np.uint16).tofile(case/'prob.bf16')
        diagnostics[name]=cpu_diagnostics(inp,meta)
    write_json(out/'cpu-math.json',diagnostics)
    build=execute([NVCC,'-std=c++17','-O3','-arch=sm_86','--cudadevrt=none','--cudart=shared','-Xptxas=-v',
        HERE/'matrix.cu','-L',TOOLCHAIN/'link','-lcublas','-o',out/'native'],dict(os.environ),120)
    write_json(out/'build.json',build);assert build['returncode']==0,build['stderr']
    pins={str(p):sha(p) for p in [HERE/'matrix.py',HERE/'matrix.cu',HERE/'suite.py',HERE/'../ordinary_norm/buffers.cuh']}
    pins.update({str(p):sha(p) for p in src.rglob('*') if p.is_file()})
    pins.update({str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    jobs=[dict(name=c['name'],argv=[str(out/'native'),str(src/c['name']),str(out/c['name']),*[str(c[k]) for k in ['rows','heads','dim','context']]],seconds=60,pins=pins) for c in cases]
    jobs.append(dict(name='compare',argv=['/opt/conda310/bin/python','-B',str(HERE/'matrix.py'),'compare',str(src),str(out)],seconds=30,pins=pins,env=dict(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')))
    write_jobs(out,jobs);print('Prepared',len(cases),'isolated QK/PV diagnostics; no GPU execution')

def compare(src,out):
    result={}
    for c in json.loads((src/'cases.json').read_text()):
        name=c['name'];result[name]={}
        for call in ['batched-gemm','per-head-gemm']+(['tst-gemv'] if c['rows']==1 else []):
            result[name][call]={}
            for op in ['qk','pv']:
                ref=bf16(np.fromfile(src/name/('hf-'+op+'.f32'),np.float32))
                actual=[np.fromfile(out/name/(call+'-'+str(i)+'-'+op+'.bf16'),np.uint16) for i in range(2)]
                ix=np.flatnonzero(actual[0]!=ref)
                result[name][call][op]=dict(unequal=len(ix),elements=len(ref),first=ix[:8].tolist(),repeat_drift=int(np.count_nonzero(actual[0]!=actual[1])))
    write_json(out/'comparison.json',result)
    for name,c in result.items():print(name,{mode:{op:v['unequal'] for op,v in ops.items()} for mode,ops in c.items()})
    print('Diagnostic completion only: nonzero counts remain failures of exact compatibility')
if __name__=='__main__':{'prepare':prepare,'compare':compare}[sys.argv[1]](Path(sys.argv[2]).resolve(),Path(sys.argv[3]).resolve())
