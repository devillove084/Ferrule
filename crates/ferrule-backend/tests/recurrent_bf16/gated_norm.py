"""Fixed-input BF16 gated norm: pinned HF observation and native CUDA12.4.

No model. Prepare is CPU-only; run uses authenticated custody and fresh receipts.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
from performance import ROOT,HERE,NATIVE,NVCC,TOOLCHAIN,CUDA_LIB,execute,sha,write_json

CAPTURE=ROOT/'target/validation/capital-decode-state-20261007/norm-replay'
HELLO=ROOT/'target/validation/bf16-layer01-supported-retry-20261005'
BASE=ROOT/'target/validation/gdn-gated-norm-20261007'


def prepare(out):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=False)
    old=BASE/'before.cuh';assert sha(old)=='14695c068c9267ac2447a91f2726cbc3611e9633ba70507da5de0727c439e8b2'
    (out/'prior.cuh').write_text(old.read_text().replace('hf_bf16','prior_hf_bf16').replace('ferrule_core_bf16_','prior_core_bf16_'))
    header=NATIVE/'core/recurrent_bf16.cuh';source=header.read_text()
    # No recurrence/conv/single-row-fusion or scalar norm arithmetic edits.
    for name in ['conv','delta_scan','delta','gated_norm']:
        def body(s):
            a=s.index('__global__ void '+name+'(');return s[a:s.index('\n}',a)+2]
        assert body(source)==body(old.read_text()),name
    a=source.index('__global__ void gated_norm_tree(');b=source.index('\n}',a)+2
    observed=source[a:b].replace('gated_norm_tree(', 'gated_norm_observed(').replace('float mean_factor)', 'float mean_factor, float *trace)')
    anchor='  const float inv = rsqrtf(__fadd_rn(variance, a.epsilon));'
    assert observed.count(anchor)==1
    observed=observed.replace(anchor,anchor+'\n  if(lane==0){trace[row]=variance;trace[a.rows+row]=inv;}')
    anchor='    const float z = load(a.gate_bf16, base + d);'
    assert observed.count(anchor)==1
    observed=observed.replace(anchor,anchor+'''\n    const uint64_t n=uint64_t(a.rows)*a.width;
    const float silu=z / (1.0f + expf(-z));
    trace[2*a.rows+base+d]=normalized;
    trace[2*a.rows+n+base+d]=weighted;
    trace[2*a.rows+2*n+base+d]=silu;
    trace[2*a.rows+3*n+base+d]=__fmul_rn(weighted,silu);''')
    (out/'norm_observed.cuh').write_text('namespace ferrule::cuda::core::hf_bf16 {\n'+observed+'\n}\n')
    cases=[];pins={str(p):sha(p) for p in [header,old,HERE/'gated_norm.cu',Path(__file__),NATIVE/'core/bf16_rms_reduction.cuh',NATIVE/'core/recurrent_bf16_abi.h',NATIVE/'core/device.cuh',NATIVE/'core/validation.cuh']}
    def write(path,value):path.write_bytes(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    def case(name,x,z,w,expected=None,timed=False):
        d=out/name;d.mkdir();r,width=x.shape
        for key,t in [('input',x),('gate',z),('weight',w)]:write(d/(key+'.bf16'),t.bfloat16())
        if expected is not None:write(d/'retained.bf16',expected.bfloat16())
        cases.append(dict(name=name,rows=r,width=width,epsilon=1e-6,timed=timed))
    fixture=json.loads((CAPTURE/'fixture.json').read_text())
    for name,pin in fixture['pins'].items():assert sha(CAPTURE/name)==pin;pins[str(CAPTURE/name)]=pin
    def load(p):return torch.frombuffer(bytearray(p.read_bytes()),dtype=torch.bfloat16).clone()
    case('capital',load(CAPTURE/'input.bf16').reshape(32,128),load(CAPTURE/'gate.bf16').reshape(32,128),load(CAPTURE/'weight.bf16'),load(CAPTURE/'expected.bf16'),True)
    assert sha(HELLO/'hf-layer01.safetensors')=='6752faa07d618e884c6f4be51694aa071866fa88f4e1d0b7860e3423c9b21b07'
    pins[str(HELLO/'hf-layer01.safetensors')]=sha(HELLO/'hf-layer01.safetensors')
    payloads=json.loads((HELLO/'hf-source-payloads.json').read_text())['used_payloads']
    with safe_open(str(HELLO/'hf-layer01.safetensors'),framework='pt',device='cpu') as f:
        for layer in [0,1]:
            key='model.language_model.layers.'+str(layer)+'.linear_attn.norm.weight';meta=payloads[key]
            with safe_open(str(Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8')/meta['shard']),framework='pt',device='cpu') as weights:w=weights.get_tensor(key)
            import hashlib
            assert hashlib.sha256(w.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==meta['sha256']
            for stage in ['prefill','decode.0']:
                prefix='hello/'+stage+'.'+str(layer)+'.'
                case('hello-'+stage+'-L'+str(layer),f.get_tensor(prefix+'linear_attn.norm.input').reshape(32,128),f.get_tensor(prefix+'linear_attn.in_proj_z.output').reshape(32,128),w,f.get_tensor(prefix+'linear_attn.norm.output'),False)
    gen=torch.Generator().manual_seed(20261007)
    widths=[1,2,3,4,7,15,16,17,31,32,33,63,64,65,127,128,129,130,131,132,255,256,257,511,512,513,1023,1024,1025,2048,4095,4096,4097]
    for rows in [1,3,17,32]:
        for width in widths:
            x=(torch.randn(rows,width,generator=gen)*torch.logspace(-3,1,rows)[:,None]).bfloat16()
            z=(torch.randn(rows,width,generator=gen)*2).bfloat16();w=(torch.randn(width,generator=gen)*0.1+1).bfloat16()
            case('synthetic-'+str(rows)+'-'+str(width),x,z,w)
    write_json(out/'cases.json',cases)
    (out/'cases.tsv').write_text(''.join(f"{c['name']} {c['rows']} {c['width']} {c['epsilon']} {int(c['timed'])}\n" for c in cases))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=CUDA_LIB+':/usr/lib64')
    r=execute([NVCC,'-std=c++17','-O3','-arch=sm_86','--cudadevrt=none','--cudart=shared','-Xptxas=-v','-I',NATIVE,'-I',out,HERE/'gated_norm.cu','-L',TOOLCHAIN/'link','-o',out/'norm'],env,120)
    write_json(out/'build.json',r);assert r['returncode']==0,r['stderr']
    assert all(sha(p)==pin for p,pin in pins.items())
    write_json(out/'prepared.json',dict(source_pins=pins,artifact_pins={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},cases=len(cases),gpu_execution=False))
    print('Prepared',len(cases),'norm cases, GPU not run')


def hf(out):
    import torch
    from safetensors.torch import save_file
    sys.path.insert(0,str(ROOT/'scripts'));import qwen35_fp8_bf16_reference as ref
    pins=ref.initialize();fn=ref.hf.Qwen3_5MoeRMSNormGated.forward
    lines,start=inspect.getsourcelines(fn);locations={start+i:l.strip() for i,l in enumerate(lines)}
    with torch.inference_mode():
        for c in json.loads((out/'cases.json').read_text()):
            if c['width']>4096:continue
            d=out/c['name'];r,w=c['rows'],c['width']
            def load(n):return torch.frombuffer(bytearray((d/(n+'.bf16')).read_bytes()),dtype=torch.bfloat16).clone().cuda()
            x=load('input').reshape(r,w);z=load('gate').reshape(r,w);weight=load('weight')
            module=ref.hf.Qwen3_5MoeRMSNormGated(w,eps=c['epsilon']).cuda().bfloat16();module.weight=torch.nn.Parameter(weight)
            stages={}
            def trace(frame,event,arg):
                if frame.f_code is fn.__code__ and event=='line':
                    line=locations[frame.f_lineno];loc=frame.f_locals
                    if line.startswith('hidden_states = hidden_states * torch.rsqrt'):stages['variance']=loc['variance'].clone()
                    if line.startswith('hidden_states = self.weight'):stages['normalized']=loc['hidden_states'].bfloat16().clone()
                    if line.startswith('hidden_states = hidden_states * F.silu'):stages['weighted']=loc['hidden_states'].clone()
                    if line.startswith('return hidden_states.to'):stages['precast']=loc['hidden_states'].clone()
                return trace
            baseline=module(x,z)
            if (d/'retained.bf16').exists():assert baseline.cpu().contiguous().view(torch.uint8).numpy().tobytes()==(d/'retained.bf16').read_bytes(),c['name']
            assert sys.gettrace() is None;sys.settrace(trace)
            try:y=module(x,z)
            finally:sys.settrace(None)
            assert torch.equal(y,baseline)
            stages.update(output=y,inv=torch.rsqrt(stages['variance']+c['epsilon']),silu=ref.F.silu(z.float()))
            save_file({k:v.cpu().contiguous() for k,v in stages.items()},str(d/'hf-stages.safetensors'))
    write_json(out/'hf.json',dict(pins=pins,retained_real_outputs_exact=True,observer_output_exact=True))


def compare(out):
    import numpy as np
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1);result={}
    def bf16(p):return (np.fromfile(p,dtype=np.uint16).astype(np.uint32)<<16).view(np.float32)
    def metric(a,b):
        a=a.reshape(-1);b=b.reshape(-1);assert a.shape==b.shape
        ix=np.flatnonzero(a.view(np.uint32)!=b.view(np.uint32))
        return dict(elements=len(a),unequal=len(ix),max_abs=float(np.max(np.abs(a-b))),first=ix[:4].tolist())
    for c in json.loads((out/'cases.json').read_text()):
        d=out/c['name'];r,w=c['rows'],c['width'];n=r*w
        if w>4096:
            result[c['name']]={'unchanged_fallback':metric(bf16(d/'native.bf16'),bf16(d/'old.bf16'))};continue
        trace=np.fromfile(d/'native-stages.f32',np.float32);metrics={}
        with safe_open(str(d/'hf-stages.safetensors'),framework='pt',device='cpu') as f:
            for name,start,count in [('variance',0,r),('inv',r,r),*[(name,2*r+i*n,n) for i,name in enumerate(['normalized','weighted','silu','precast'])]]:
                metrics[name]=metric(trace[start:start+count],f.get_tensor(name).float().numpy())
            metrics['output']=metric(bf16(d/'native.bf16'),f.get_tensor('output').float().numpy())
            metrics['old_output']=metric(bf16(d/'old.bf16'),f.get_tensor('output').float().numpy())
        result[c['name']]=metrics
    write_json(out/'comparison.json',result)
    failures={name:{k:v for k,v in m.items() if k!='old_output' and v['unequal']} for name,m in result.items()}
    failures={k:v for k,v in failures.items() if v};print(json.dumps(failures,indent=2));print('cases',len(result),'failures',len(failures))
    return not failures


def run(out,evidence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and not os.environ.get('LD_PRELOAD')
    m=json.loads((out/'prepared.json').read_text())
    for p,pin in m['source_pins'].items():assert sha(p)==pin,p
    for p,pin in m['artifact_pins'].items():assert sha(out/p)==pin,p
    evidence.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'scripts'));import test_vllm_qwen35 as supervisor
    def snapshot():return {k:execute(['nvidia-smi',q,'--format=csv,noheader'],dict(os.environ),10) for k,q in [('gpu','--query-gpu=index,uuid,memory.used,utilization.gpu'),('apps','--query-compute-apps=pid,gpu_uuid,used_gpu_memory')]}
    def idle(s):return s['gpu']['returncode']==s['apps']['returncode']==0 and not s['apps']['stdout'].strip() and len(s['gpu']['stdout'].splitlines())==8 and all(l.split(',')[-2].strip()=='0 MiB' and l.split(',')[-1].strip()=='0 %' for l in s['gpu']['stdout'].splitlines())
    def custody(r):
        h,c=r['handshake'],r['custody'];return h.get('containment_verified') and c.get('containment_verified') and c.get('echild') and r['descendant_custody']=='proven_drained' and h.get('launch_id')==c.get('launch_id') and h.get('helper_identity')==c.get('helper_identity') and not r['cleanup_errors'] and not r.get('error')
    before=snapshot();write_json(evidence/'before.json',before);assert idle(before)
    receipts=[]
    try:
        libs=json.loads((HELLO/'environment-preflight.json').read_text())['resolved_libraries']
        for lib in libs:assert sha(lib['realpath'])==lib['sha256']
        write_json(evidence/'launch.json',dict(prepared_sha256=sha(out/'prepared.json'),libraries=libs))
        for name,cmd in [('hf',['/opt/conda310/bin/python','-B',str(Path(__file__).resolve()),'hf',str(out)]),('native',[str(out/'norm'),str(out)])]:
            os.environ['LD_LIBRARY_PATH']='' if name=='hf' else CUDA_LIB+':/usr/lib64';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            r=supervisor.run_process(cmd,120);receipts.append(r);write_json(evidence/(name+'-supervisor.json'),r)
            assert r['returncode']==0 and not r['timed_out'] and custody(r),r['stderr']
        assert compare(out),'Exact norm stage gate failed'
        import numpy as np
        lines=[json.loads(l) for l in receipts[-1]['stdout'].splitlines()]
        times={v:[r['us'] for r in lines if r['event']=='sample' and r['variant']==v] for v in ['old','new']}
        assert all(len(v)==30 for v in times.values())
        write_json(evidence/'timing.json',dict(resources=[r for r in lines if r['event']=='resources'],samples=[r for r in lines if r['event']=='sample'],median_us={k:float(np.median(v)) for k,v in times.items()},p95_us={k:float(np.percentile(v,95)) for k,v in times.items()},interpretation='Old norm differs at two real output words; timings are bounded norm calls, not throughput'))
    finally:
        after=[]
        for _ in range(3):time.sleep(1);after.append(snapshot())
        released=bool(receipts) and all(custody(r) for r in receipts) and all(idle(s) for s in after)
        write_json(evidence/'release.json',dict(after=after,drained=all(custody(r) for r in receipts),release='released' if released else 'unverified'));assert released


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','hf']);p.add_argument('directory',type=Path);p.add_argument('--evidence',type=Path);a=p.parse_args();out=a.directory.resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='hf':hf(out)
    else:
        if a.evidence is None:p.error('--evidence required')
        run(out,a.evidence.resolve())
