"""Bounded independent eager attention fixtures. CPU preparation, owned GPU jobs."""
import json,os,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'crates/ferrule-backend/tests/recurrent_bf16'))
from performance import sha,execute,NVCC,NATIVE,TOOLCHAIN,write_json
import oracle
FROZEN=ROOT/'target/validation/bf16-resident-scalar-20261007/frozen-attention-boundary'

def write_jobs(out, jobs):
    sys.path.insert(0, str(ROOT/'scripts'))
    from run_pinned_jobs import pin_jobs, validate
    library_receipt = ROOT/'target/validation/bf16-layer01-supported-retry-20261005/environment-preflight.json'
    pin_jobs(jobs, library_receipt, extra_paths=(HERE/'run-after-grant.sh', Path('/opt/conda310/bin/python')))
    validate(jobs, library_receipt)
    write_json(out/'jobs.json', jobs)

def prepare(out):
    os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void')
    import torch
    import numpy as np
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False);cases=[]
    def add(name,q,k,v,g,positions):
        d=out/name;d.mkdir();r,h,dim=q.shape;kh=k.shape[1];n=k.shape[0];pages=(n+15)//16
        meta=dict(name=name,rows=r,heads=h,kv_heads=kh,dim=dim,context=n,positions=positions,pages=pages)
        for key,t in [('query',q.transpose(0,1)),('key',k.transpose(0,1)),('value',v.transpose(0,1)),('gate',g)]:
            (d/(key+'.bf16')).write_bytes(t.bfloat16().contiguous().view(torch.uint8).numpy().tobytes())
        q.bfloat16().float().numpy().tofile(d/'query.f32')
        slots=list(range(pages,0,-1))
        for key,t in [('key',k),('value',v)]:
            cache=torch.full((pages+1,16,kh,dim),float('nan'),dtype=torch.bfloat16)
            for i in range(n):cache[slots[i//16],i%16]=t[i]
            (d/('cache-'+key+'.bf16')).write_bytes(cache.contiguous().view(torch.uint8).numpy().tobytes())
        for key,items in [('slots',slots),('offsets',[0,pages]),('sequences',[0]*r),('positions',positions),('lens',[n]*r)]:np.array(items,np.int32).tofile(d/(key+'.i32'))
        write_json(d/'case.json',meta);cases.append(meta)
    def load(name,shape):return torch.frombuffer(bytearray((FROZEN/(name+'.bf16')).read_bytes()),dtype=torch.bfloat16).reshape(shape).clone()
    frozen=json.loads((FROZEN/'manifest.json').read_text())
    for name,m in frozen['tensors'].items():assert sha(FROZEN/(name+'.bf16'))==m['sha256']
    add('actual',load('query',(16,1,256)).transpose(0,1),load('key',(2,11,256)).transpose(0,1),load('value',(2,11,256)).transpose(0,1),load('gate',(1,4096)),[10])
    for name in ['expected_ungated','expected_gated','actual_gated']:shutil.copyfile(FROZEN/(name+'.bf16'),out/'actual'/(name+'.bf16'))
    gen=torch.Generator().manual_seed(2026100715)
    shapes=[(1,16,2,256,1),(5,16,2,256,5),(1,16,2,256,11),(5,16,2,256,11),(1,16,2,256,128),(5,16,2,256,128),(1,4,2,1,5),(5,6,2,3,5),(1,8,2,64,33),(5,8,2,128,33),(5,16,2,256,129)]
    for i,(r,h,kh,dim,n) in enumerate(shapes):
        q=(torch.randn(r,h,dim,generator=gen)*.7).bfloat16();k=(torch.randn(n,kh,dim,generator=gen)*.7).bfloat16();v=(torch.randn(n,kh,dim,generator=gen)*.3).bfloat16();g=(torch.randn(r,h*dim,generator=gen)*2).bfloat16()
        positions=list(range(n-r,n)) if i%2==0 else list(range(r))
        add('synthetic-'+str(i),q,k,v,g,positions)
    build=execute([NVCC,'-std=c++17','-O3','-arch=sm_86','--cudadevrt=none','--cudart=shared','-Xptxas=-v','-I',NATIVE,HERE/'suite.cu','-L',TOOLCHAIN/'link','-o',out/'native'],dict(os.environ),120)
    write_json(out/'build.json',build);assert build['returncode']==0,build['stderr']
    sources=[HERE/'suite.py',HERE/'suite.cu',HERE/'candidate.cuh',HERE/'oracle.py',NATIVE/'core/attention_ops.cuh',NATIVE/'core/tensor_ops.cuh']
    pins={str(p):sha(p) for p in sources};pins.update({str(p):sha(p) for p in out.rglob('*') if p.is_file()})
    write_json(out/'cases.json',cases);jobs=[dict(name='hf',argv=['/opt/conda310/bin/python','-B',str(Path(__file__).resolve()),'hf',str(out)],seconds=180,hf=True,pins=pins)]
    jobs +=[dict(name=c['name'],argv=[str(out/'native'),str(out/c['name']),*[str(c[k]) for k in ['rows','heads','kv_heads','dim','context','pages']]],seconds=90,pins=pins) for c in cases]
    jobs.append(dict(name='compare',argv=['/opt/conda310/bin/python','-B',str(Path(__file__).resolve()),'compare',str(out)],seconds=30,env=dict(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='void'),pins=pins));write_jobs(out,jobs)
    print('Prepared',len(cases),'candidate attention cases')

def hf(out):
    for c in json.loads((out/'cases.json').read_text()):oracle.run(out/c['name'])

def compare(out):
    import numpy as np
    result={}
    for c in json.loads((out/'cases.json').read_text()):
        d=out/c['name'];r={}
        for name in ['raw','gated']:
            actual=(np.fromfile(d/('candidate-'+name+'.bf16'),np.uint16).astype(np.uint32)<<16).view(np.float32)
            expected=np.fromfile(d/('hf-'+name+'.f32'),np.float32)
            ix=np.flatnonzero(actual.view(np.uint32)!=expected.view(np.uint32));r[name]=dict(unequal=len(ix),elements=len(actual),first=ix[:8].tolist(),max_abs=float(np.max(np.abs(actual-expected))))
        result[c['name']]=r
    write_json(out/'comparison.json',result);print(json.dumps(result,indent=2));assert all(v['unequal']==0 for c in result.values() for v in c.values())
if __name__=='__main__':{'prepare':prepare,'hf':hf,'compare':compare}[sys.argv[1]](Path(sys.argv[2]).resolve())
