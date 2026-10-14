"""Observe the unmodified pinned Qwen RMSNorm producer via Torch dispatch."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[4]

def observe(directory, rows, width, epsilon):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    sys.path.insert(0,str(ROOT/'scripts'));import qwen35_fp8_bf16_reference as ref
    pins=ref.initialize()
    def load(name):return torch.frombuffer(bytearray((directory/(name+'.bf16')).read_bytes()),dtype=torch.bfloat16).clone().reshape(rows,width).cuda() if name=='input' else torch.frombuffer(bytearray((directory/(name+'.bf16')).read_bytes()),dtype=torch.bfloat16).clone().cuda()
    x=load('input');weight=load('weight')
    module=ref.hf.Qwen3_5MoeRMSNorm(width,eps=epsilon).cuda().bfloat16()
    module.weight=torch.nn.Parameter(weight)
    records=[]
    class Observer(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            output=func(*args,**(kwargs or {}))
            records.append((str(func),output.detach().clone()))
            return output
    with torch.inference_mode():
        baseline=module(x)
        with Observer():observed=module(x)
    assert torch.equal(baseline.view(torch.int16),observed.view(torch.int16))
    raw=baseline.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    if (directory/'expected.bf16').exists():assert raw==(directory/'expected.bf16').read_bytes()
    (directory/'hf.bf16').write_bytes(raw)
    names=[];adds=0;muls=0;stages={}
    for name,tensor in records:
        names.append(dict(op=name,shape=list(tensor.shape),dtype=str(tensor.dtype)))
        if name=='aten.pow.Tensor_Scalar':key='square'
        elif name=='aten.mean.dim':key='variance'
        elif name=='aten.rsqrt.default':key='inverse'
        elif name=='aten.add.Tensor':key=['epsilon_added','affine'][adds];adds+=1
        elif name=='aten.mul.Tensor':key=['normalized','weighted'][muls];muls+=1
        else:continue
        stages[key]=tensor.cpu().float().contiguous()
        (directory/('hf-'+key+'.f32')).write_bytes(stages[key].numpy().tobytes())
    assert set(stages)=={'square','variance','epsilon_added','inverse','affine','normalized','weighted'},names
    (directory/'hf.json').write_text(json.dumps(dict(pins=pins,ops=names,observer_exact=True),indent=2)+'\n')
    print(json.dumps(names))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--rows',type=int,default=1);p.add_argument('--width',type=int,default=2048);p.add_argument('--epsilon',type=float,default=1e-6);a=p.parse_args();observe(a.directory,a.rows,a.width,a.epsilon)
