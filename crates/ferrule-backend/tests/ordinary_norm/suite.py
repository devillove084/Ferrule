"""Fixed ordinary BF16 offset-norm inputs, original HF stages and native gates."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'crates/ferrule-backend/tests/recurrent_bf16'))
from performance import NATIVE,NVCC,TOOLCHAIN,execute,sha,write_json
import oracle
FROZEN=ROOT/'target/validation/bf16-resident-control-20261007/frozen-first-norm'


def prepare(out):
    os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')
    import torch
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    source=(NATIVE/'core/bf16_boundary.cuh').read_text()
    start=source.index('__global__ void bf16_offset_norm_tree(')
    end=source.index('\n}',start)+2
    observed=source[start:end].replace('bf16_offset_norm_tree(','bf16_offset_norm_observed(').replace('float mean_factor)','float mean_factor, float *trace)')
    a='  const float inverse = rsqrtf(added);';assert observed.count(a)==1
    observed=observed.replace(a,a+'\n  if(lane==0){trace[row]=variance;trace[a.rows+row]=added;trace[2*a.rows+row]=inverse;}')
    a='    const float value = __fmul_rn(normalized, affine);';assert observed.count(a)==1
    observed=observed.replace(a,a+'\n    const uint64_t n=uint64_t(a.rows)*a.width;\n    trace[3*a.rows+base+d]=normalized;trace[3*a.rows+n+base+d]=affine;trace[3*a.rows+2*n+base+d]=value;')
    (out/'norm_observed.cuh').write_text('namespace ferrule::cuda::core {\n'+observed+'\n}\n')
    cases=[]
    def save(p,t):p.write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
    def case(name,x,w,epsilon=1e-6,nonfinite=False,timed=False):
        d=out/name;d.mkdir();save(d/'input.bf16',x.bfloat16());save(d/'weight.bf16',w.bfloat16())
        rows,width=x.shape;cases.append(dict(name=name,rows=rows,width=width,epsilon=epsilon,nonfinite=nonfinite,timed=timed))
    m=json.loads((FROZEN/'manifest.json').read_text());pins={}
    for name,pin in m['pins'].items():assert sha(FROZEN/name)==pin;pins[str(FROZEN/name)]=pin
    def bf(p):return torch.frombuffer(bytearray(p.read_bytes()),dtype=torch.bfloat16).clone()
    case('actual',bf(FROZEN/'input.bf16').reshape(1,2048),bf(FROZEN/'weight.bf16'),timed=True)
    shutil.copyfile(FROZEN/'expected.bf16',out/'actual/expected.bf16')
    shutil.copyfile(FROZEN/'actual.bf16',out/'actual/retained-old.bf16')
    gen=torch.Generator().manual_seed(2026100708)
    widths=[1,2,3,7,16,17,31,32,33,63,64,65,127,128,129,131,255,256,257,511,512,513,1023,1024,1025,2048,2049,4095,4096,4097,8192]
    for width in widths:
        for rows in [1,3,17]:
            x=torch.randn(rows,width,generator=gen)*.3;w=torch.randn(width,generator=gen)*.4-.2
            case('w'+str(width)+'-r'+str(rows),x,w,epsilon=[1e-12,1e-6,.01][[1,3,17].index(rows)])
    case('zeros-default',torch.zeros(3,129),torch.zeros(129))
    case('ones-default',torch.ones(3,128),torch.zeros(128))
    case('ones-weight-one',torch.ones(3,2048),torch.ones(2048))
    case('cancel-affine',torch.randn(3,131,generator=gen),torch.full((131,),-1.0))
    case('small-offset',torch.randn(3,127,generator=gen),torch.full((127,),1/512))
    case('tiny-epsilon',torch.randn(3,129,generator=gen)*1e-9,torch.zeros(129),epsilon=1e-12)
    case('large-epsilon',torch.ones(3,65),torch.zeros(65),epsilon=1e10)
    case('signed-zero',torch.full((3,33),-0.0),torch.full((33,),-2.0))
    x=torch.ones(3,129);x[0,0]=float('nan');x[1,0]=float('inf');x[2,0]=-float('inf')
    case('nonfinite-input',x,torch.zeros(129),nonfinite=True)
    w=torch.zeros(129);w[0]=float('nan');w[1]=float('inf');w[2]=-float('inf')
    case('nonfinite-weight',torch.ones(3,129),w,nonfinite=True)
    case('overflow-square',torch.full((3,128),1e30),torch.zeros(128),nonfinite=True)
    write_json(out/'cases.json',cases)
    (out/'cases.tsv').write_text(''.join(f"{c['name']} {c['rows']} {c['width']} {c['epsilon']:.9g} {int(c['timed'])}\n" for c in cases))
    build=execute([NVCC,'-std=c++17','-O3','-arch=sm_86','--cudadevrt=none','--cudart=shared','-Xptxas=-v','-I',NATIVE,'-I',out,HERE/'suite.cu','-L',TOOLCHAIN/'link','-o',out/'native'],dict(os.environ),120)
    write_json(out/'build.json',build);assert build['returncode']==0,build['stderr']
    c=execute([out/'native','--contracts'],dict(os.environ),15);write_json(out/'contracts.json',c);assert c['returncode']==0,c
    files=[Path(__file__),HERE/'suite.cu',HERE/'buffers.cuh',HERE/'oracle.py']+[NATIVE/'core'/n for n in ['bf16_boundary.cuh','bf16_rms_reduction.cuh','tensor_ops.cuh','abi.h','device.cuh','validation.cuh']]
    pins.update({str(p):sha(p) for p in files});artifacts={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'prepared.json',dict(cases=len(cases),source_pins=pins,artifact_pins=artifacts))
    pins.update({str(out/p):v for p,v in artifacts.items()})
    jobs=[dict(name='hf',argv=['/opt/conda310/bin/python','-B',str(Path(__file__).resolve()),'hf',str(out)],seconds=180,hf=True,pins=pins),dict(name='native',argv=[str(out/'native'),str(out)],seconds=120,pins=pins),dict(name='compare',argv=['/opt/conda310/bin/python','-B',str(Path(__file__).resolve()),'compare',str(out)],seconds=60,env=dict(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void'),pins=pins)]
    write_json(out/'jobs.json',jobs);print('Prepared',len(cases),'ordinary norm cases')


def hf(out):
    for c in json.loads((out/'cases.json').read_text()):
        if c['width']<=4096:oracle.observe(out/c['name'],c['rows'],c['width'],c['epsilon'])


def compare(out):
    import numpy as np
    def bf(p):return (np.fromfile(p,np.uint16).astype(np.uint32)<<16).view(np.float32)
    def metric(a,b,nonfinite=False):
        a=a.reshape(-1);b=b.reshape(-1);assert a.shape==b.shape
        equal=a.view(np.uint32)==b.view(np.uint32)
        if nonfinite:equal|=np.isnan(a)&np.isnan(b)
        ix=np.flatnonzero(~equal)
        return dict(elements=len(a),unequal=len(ix),first=ix[:5].tolist())
    report={}
    for c in json.loads((out/'cases.json').read_text()):
        d=out/c['name'];r,w=c['rows'],c['width'];n=r*w;v={}
        if w>4096:v['fallback_unchanged']=metric(bf(d/'native.bf16'),bf(d/'old.bf16'))
        else:
            trace=np.fromfile(d/'native-stages.f32',np.float32)
            for name,start,count in [('variance',0,r),('epsilon_added',r,r),('inverse',2*r,r),('normalized',3*r,n),('affine',3*r+n,n),('weighted',3*r+2*n,n)]:
                expected=np.fromfile(d/('hf-'+name+'.f32'),np.float32)
                if name=='affine':expected=np.tile(expected,r)
                v[name]=metric(trace[start:start+count],expected,c['nonfinite'])
            v['output']=metric(bf(d/'native.bf16'),bf(d/'hf.bf16'),c['nonfinite'])
        report[c['name']]=v
    actual=out/'actual';assert (actual/'old.bf16').read_bytes()==(actual/'retained-old.bf16').read_bytes()
    assert (actual/'native.bf16').read_bytes()==(actual/'expected.bf16').read_bytes()
    write_json(out/'comparison.json',report)
    failures={name:v for name,v in report.items() if any(m['unequal'] for m in v.values())}
    print(json.dumps(failures,indent=2));print('ordinary cases',len(report),'failures',len(failures));assert not failures


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','hf','compare']);p.add_argument('directory',type=Path);a=p.parse_args();out=a.directory.resolve()
    {'prepare':prepare,'hf':hf,'compare':compare}[a.mode](out)
