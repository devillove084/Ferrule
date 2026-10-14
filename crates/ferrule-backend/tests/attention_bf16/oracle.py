"""Observe the configured eager HF attention, no SDPA or reference substitution."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[4]

def run(d):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    sys.path.insert(0,str(ROOT/'scripts'));import qwen35_fp8_bf16_reference as ref
    pins=ref.initialize()
    meta=json.loads((d/'case.json').read_text());r,h,kv,dim,n=[meta[k] for k in ['rows','heads','kv_heads','dim','context']]
    def load(name,shape):return torch.frombuffer(bytearray((d/(name+'.bf16')).read_bytes()),dtype=torch.bfloat16).reshape(shape).clone().cuda()
    q=load('query',(1,h,r,dim));k=load('key',(1,kv,n,dim));v=load('value',(1,kv,n,dim));g=load('gate',(1,r,h*dim))
    positions=torch.tensor(meta['positions'],device='cuda')
    mask=torch.where(torch.arange(n,device='cuda')[None,:]<=positions[:,None],0.,torch.finfo(torch.bfloat16).min).bfloat16().reshape(1,1,r,n)
    config=SimpleNamespace(_attn_implementation='eager')
    module=SimpleNamespace(num_key_value_groups=h//kv,training=False,config=config)
    fn=ref.hf.ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation,ref.hf.eager_attention_forward)
    assert fn is ref.hf.eager_attention_forward
    records=[]
    class Observer(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            if isinstance(result,torch.Tensor):
                records.append((str(func),result.detach().clone()))
            return result
    def forward():
        out,prob=fn(module,q,k,v,mask,scaling=dim**-.5,dropout=0.)
        gated=out.reshape(1,r,h*dim)*torch.sigmoid(g)
        return out,prob,gated
    with torch.inference_mode():
        baseline=forward()
        with Observer():out,prob,gated=forward()
        assert all(torch.equal(a.view(torch.int16),b.view(torch.int16)) for a,b in zip(baseline,[out,prob,gated]))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:forward()
    def save(name,t):
        (d/('hf-'+name+'.f32')).write_bytes(t.cpu().float().contiguous().numpy().tobytes())
    bmm=0;mul=0
    for op,t in records:
        if op in ['aten.bmm.default','aten.matmul.default']:name=['qk','pv'][bmm];bmm+=1
        elif op=='aten.mul.Tensor':name=['scaled','gated'][mul];mul+=1
        elif op=='aten.add.Tensor':name='masked'
        elif op in ['aten._softmax.default','aten.softmax.int']:name='prob-f32'
        elif op=='aten.sigmoid.default':name='sigmoid'
        else:continue
        save(name,t)
    save('prob-bf16',prob);save('raw',out);save('gated',gated)
    for name,t in [('expected_ungated',out),('expected_gated',gated)]:
        if (d/(name+'.bf16')).exists():assert t.cpu().contiguous().view(torch.uint8).numpy().tobytes()==(d/(name+'.bf16')).read_bytes(),name
    save('sigmoid-f32-diagnostic',torch.sigmoid(g.float()))
    (d/'hf.json').write_text(json.dumps(dict(pins=pins,implementation=config._attn_implementation,ops=[(op,list(t.shape),str(t.dtype)) for op,t in records],kernels=[e.name for e in prof.events() if 'CUDA' in str(e.device_type)],retained_unchanged=True),indent=2)+'\n')
    print('HF eager observed',[(op,list(t.shape)) for op,t in records])
if __name__=='__main__':run(Path(sys.argv[1]).resolve())
