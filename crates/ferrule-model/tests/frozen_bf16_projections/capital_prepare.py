"""Supplemental independent Capital L0 operators, never a full-model replay.

Retains original six HF pairs; observes routed pairs from unmodified pinned HF
Experts.forward on retained HF input/routes. Aggregate must match original HF.
Only execute under authenticated supervision and explicit GPU ownership.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import prepare as common
ROOT=common.ROOT
SOURCE=ROOT/'target/validation/bf16-layer01-supported-retry-20261005'
PINS={'hf-layer01.safetensors':'6752faa07d618e884c6f4be51694aa071866fa88f4e1d0b7860e3423c9b21b07',
'hf-manifest.json':'281ad7cc617bd9b9b69413adb426170a86c883f019864c49f9f585209b0e5c26',
'hf-source-payloads.json':'e3bef78cb9a843717e283780f089ae9a771781f4e8e2d47807389868831aa4e0'}

def main(out):
    for name,pin in PINS.items(): assert common.util.sha256(SOURCE/name)==pin
    manifest=json.loads((SOURCE/'hf-manifest.json').read_text())
    assert len(manifest['checks'])==30 and all(x['unequal']==0 for x in manifest['checks'])
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'scripts'))
    import qwen35_fp8_bf16_reference as ref
    source_pins=ref.initialize()
    import torch
    from safetensors import safe_open
    from torch.overrides import TorchFunctionMode
    assert torch.cuda.get_device_capability()==(8,6)
    provenance=json.loads((SOURCE/'hf-source-payloads.json').read_text())['used_payloads']
    checkpoint=Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8')
    frozen=safe_open(str(SOURCE/'hf-layer01.safetensors'),framework='pt',device='cpu')
    get=lambda key:frozen.get_tensor('capital/prefill.0.'+key)
    cases=[]; profiles={}; hashes={}; epilogues={}
    def put(name,data):
        with (out/name).open('xb') as f:f.write(data)
        hashes[name]=hashlib.sha256(data).hexdigest()
    def data(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    def payload(name):
        pin=provenance[name];dtype='BF16' if pin['dtype']=='torch.bfloat16' else 'F8_E4M3'
        raw,_=common.util.tensor(checkpoint/pin['shard'],name,dtype,pin['shape'])
        assert hashlib.sha256(raw).hexdigest()==pin['sha256']
        return raw
    def operands(name):
        raw=payload(name);sc=payload(name+'_scale_inv');n,k=provenance[name]['shape']
        w=torch.frombuffer(bytearray(raw),dtype=torch.float8_e4m3fn).reshape(n,k).cuda()
        scale=torch.frombuffer(bytearray(sc),dtype=torch.bfloat16).reshape(n//128,k//128).cuda()
        return raw,sc,ref.dequantize_bf16(w,scale)
    def observe_linear(x,w,expected):
        shape=tuple(x.reshape(-1,x.shape[-1]).shape)+(w.shape[0],)
        if shape not in profiles:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                y=torch.nn.functional.linear(x,w)
            torch.cuda.synchronize();assert torch.equal(y,expected)
            profiles[shape]=sorted(set(e.name for e in p.events() if e.device_type==torch.autograd.DeviceType.CUDA))
            p.export_chrome_trace(str(out/('trace-'+'-'.join(map(str,shape))+'.json')))
        raw=torch.mm(x.reshape(-1,x.shape[-1]),w.T,out_dtype=torch.float32)
        epilogues[str(shape)]=int((raw.bfloat16().reshape(expected.shape)!=expected).sum())
        return raw
    def emit(identity,role,x,y,wname,decoded,raw_output):
        n,k=provenance[wname]['shape'];m=x.numel()//k
        for suffix,value in [('input.bf16',data(x)),('hf.bf16',data(y)),('weight.fp8',payload(wname)),('scale.bf16',payload(wname+'_scale_inv')),('decoded.bf16',data(decoded)),('native.f32',data(raw_output))]:put(identity+'.'+suffix,value)
        cases.append(dict(id=identity,role=role,m=m,n=n,k=k,weight=wname))
    with torch.inference_mode():
        for _,(role,hook,n,k) in common.SPECS.items():
            wname='model.language_model.layers.0.'+hook+'.weight'
            _,_,w=operands(wname);x=get(hook+'.input').cuda();y=get(hook+'.output').cuda()
            assert torch.equal(torch.nn.functional.linear(x,w),y)
            raw=observe_linear(x,w,y)
            emit(role,role,x,y,wname,w,raw)
        config=ref.read_config(checkpoint)
        with torch.device('meta'): experts=ref.hf.Qwen3_5MoeExperts(config)
        storage=ref.legacy.Checkpoint(checkpoint);weights=ref.DeviceWeights(storage,'cuda:0',128<<20)
        holder=torch.nn.Module();holder.experts=experts
        ref.bind(holder,'model.layers.0.mlp',weights,config)
        x=get('mlp.experts.input').cuda();ids=get('router.ids').cuda();routing=get('router.weights').cuda()
        hits=sorted(set(ids.cpu().reshape(-1).tolist()));captures=[]
        class Observe(TorchFunctionMode):
            def __torch_function__(self,func,types,args=(),kwargs=None):
                y=func(*args,**(kwargs or {}))
                if func.__name__=='linear': captures.append((args[0].detach().clone(),args[1].detach().clone(),y.detach().clone()))
                return y
        with Observe():aggregate=holder.experts(x,ids,routing)
        assert torch.equal(aggregate.cpu(),get('mlp.experts.output')),'supplemental native aggregate changed'
        assert len(captures)==2*len(hits)
        for i,expert in enumerate(hits):
            for j,parts in [(0,['gate','up']),(1,['down'])]:
                x,w,y=captures[2*i+j];raw=observe_linear(x,w,y)
                for part in parts:
                    start=512 if part=='up' else 0;end=start+(512 if j==0 else 2048)
                    name=f'model.language_model.layers.0.mlp.experts.{expert}.{part}_proj.weight'
                    _,_,independent=operands(name);assert torch.equal(w[start:end],independent)
                    emit(f'expert{expert}_{part}','RoutedExpert'+part.title(),x,y[...,start:end],name,w[start:end],raw[:,start:end])
        for name,entry in storage.payloads.items():assert entry['sha256']==provenance[name]['sha256']
    result=dict(schema='ferrule.capital-projection-matrix.v1',oracle='independent-pinned-hf-bf16',hf_sha256=PINS['hf-layer01.safetensors'],source_pins=source_pins,source_manifest_sha256=PINS['hf-manifest.json'],original_boundaries_exact=30,supplemental_experts_aggregate_exact=True,routed_experts=hits,cases=cases,profiles={str(k):v for k,v in profiles.items()},f32_epilogue_differences=epilogues,precision=dict(torch=torch.__version__,cuda=torch.version.cuda,tf32=False,bf16_reduced_precision_reduction=False))
    put('manifest.json',(json.dumps(result,indent=2)+'\n').encode())
    (out/'cases.txt').write_text(''.join(f"{c['id']} {c['m']} {c['n']} {c['k']}\n" for c in cases))
    hashes['cases.txt']=common.util.sha256(out/'cases.txt')
    (out/'SHA256SUMS').write_text(''.join(h+'  '+n+'\n' for n,h in sorted(hashes.items())))
    print(json.dumps(dict(cases=len(cases),routed_experts=len(hits),aggregate_exact=True,fixture_pin=common.util.sha256(out/'SHA256SUMS'),profiles=result['profiles']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--execute-cuda',action='store_true');a=p.parse_args();assert a.execute_cuda;main(a.output)
