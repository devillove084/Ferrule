"""CUDA F32 product/reduction oracle for the GDN vectorized-column regime."""
import sys,json
from pathlib import Path
import torch
out=Path(sys.argv[1]); torch.manual_seed(6026);torch.set_num_threads(1)
assert torch.__version__=='2.13.0+cu129'
cases=[]
for dk in [64,65,67,127,128,129,255,256,257,511,512]:
    for hv,dv in [(32,4),(2,64),(1,128),(3,44),(32,128)]:
        prefix=out/str(len(cases))
        q=torch.randn(dk).bfloat16().cuda();s=torch.randn(hv,dk,dv).cuda()
        # Helper uses inv=1, scale=1: exact BF16 load then F32 product, no L2.
        expected=(s*q.float()[None,:,None]).sum(-2)
        for suffix,value in [('q.bf16',q),('state.f32',s),('expected.f32',expected)]:
            Path(str(prefix)+'-'+suffix).write_bytes(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        cases.append((hv,dk,dv))
(out/'cases.txt').write_text(''.join('%d %d %d\n'%c for c in cases))
print(json.dumps(dict(cases=len(cases),torch=torch.__version__,key_dimensions=sorted(set(c[1] for c in cases)))))
