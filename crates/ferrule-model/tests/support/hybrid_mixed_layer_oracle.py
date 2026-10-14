"""Independent Torch CPU oracle for gated GQA+dense TP2 block, not Ferrule math."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
from pathlib import Path
import torch

torch.set_num_threads(1)
t = json.loads((Path(__file__).parent / "../fixtures/hybrid_cpu/oracle.json").read_text())["tensors"]
def tensor(name):
    p = t["layers.1."+name+".weight"]
    return torch.tensor(p["values"],dtype=torch.bfloat16).reshape(p["shape"])
def norm(x,w):
    z=x.float()
    return (z*torch.rsqrt((z*z).mean(-1,keepdim=True)+1e-6)*(1+w.float())).bfloat16()
x=torch.tensor([((i*7)%19-9)/16 for i in range(16)],dtype=torch.bfloat16).reshape(2,8)
z=norm(x,tensor("input_norm"))
qg=torch.stack([z @ tensor("attention.query")[lane*16:(lane+1)*16].T for lane in range(2)],dim=1)
q,g=qg[...,:8],qg[...,8:]
k=(z @ tensor("attention.key").T).reshape(2,1,8)
v=(z @ tensor("attention.value").T).reshape(2,1,8)
q=norm(q,tensor("attention.query_norm"))
k=norm(k,tensor("attention.key_norm"))
angle=torch.arange(2,dtype=torch.float32)[:,None]*torch.tensor([1.,0.01])[None,:]
c,s=angle.cos().bfloat16()[:,None,:],angle.sin().bfloat16()[:,None,:]
def rope(x):
    a,b=x[...,:2],x[...,2:4]
    return torch.cat([a*c-b*s,b*c+a*s,x[...,4:]],dim=-1)
q,k=rope(q),rope(k)
parts=[]
for lane in range(2):
    score=(q[:,lane] @ k[:,0].T)*(8**-0.5)
    mask=torch.zeros((2,2),dtype=torch.bfloat16)
    mask[0,1]=torch.finfo(torch.bfloat16).min
    prob=torch.softmax((score+mask).float(),dim=-1).bfloat16()
    out=(prob @ v[:,0])*torch.sigmoid(g[:,lane])
    parts.append(out @ tensor("attention.output")[:,lane*8:(lane+1)*8].T)
x=x+(parts[0]+parts[1])
z=norm(x,tensor("post_attention_norm"))
parts=[]
for lane in range(2):
    a=z @ tensor("feed_forward.gate")[lane*6:(lane+1)*6].T
    b=z @ tensor("feed_forward.up")[lane*6:(lane+1)*6].T
    parts.append((torch.nn.functional.silu(a)*b) @ tensor("feed_forward.down")[:,lane*6:(lane+1)*6].T)
y=x+(parts[0]+parts[1])
Path(__file__).with_suffix(".json").write_text(json.dumps(dict(schema="ferrule.mixed-gqa-block.tp2.v1",oracle="independent-torch-cpu",torch=torch.__version__,
    output=y.view(torch.uint16).flatten().tolist()),indent=2)+"\n")
