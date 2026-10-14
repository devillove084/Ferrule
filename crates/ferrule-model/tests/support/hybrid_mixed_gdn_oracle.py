"""Independent CPU Torch/HF TP2 GQA->GDN+dense island; no Ferrule imports.

Run with /opt/conda310/bin/python. The pinned HF implementation is the same
BF16-normalization/chunk reference as existing Mixed, not newer HF F32 Q/K.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ATEN_CPU_CAPABILITY'] = 'avx512'
import hashlib
import inspect
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import transformers
from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

torch.set_num_threads(1)
HERE = Path(__file__).parent
source = Path(inspect.getfile(hf))
pin = 'aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048'
assert hashlib.sha256(source.read_bytes()).hexdigest() == pin
assert transformers.__version__ == '5.5.3'
raw = json.loads((HERE / '../fixtures/hybrid_cpu/oracle.json').read_text())['tensors']

def tensor(layer, name):
    p = raw[f'layers.{layer}.{name}.weight']
    return torch.tensor(p['values'], dtype=torch.float32).reshape(p['shape'])

def weight(layer, name):
    return tensor(layer, name).bfloat16()

def norm(x, w):
    z = x.float()
    return (z * torch.rsqrt((z*z).mean(-1, keepdim=True) + 1e-6) * (1+w.float())).bfloat16()

def dense(x, layer):
    z = norm(x, weight(layer, 'post_attention_norm'))
    parts = []
    for lane in range(2):
        a = z @ weight(layer, 'feed_forward.gate')[lane*6:(lane+1)*6].T
        b = z @ weight(layer, 'feed_forward.up')[lane*6:(lane+1)*6].T
        parts.append((F.silu(a)*b) @ weight(layer, 'feed_forward.down')[:,lane*6:(lane+1)*6].T)
    return x + (parts[0]+parts[1])

def gqa(x):
    n = x.shape[0]
    z = norm(x, weight(1, 'input_norm'))
    qg = torch.stack([z @ weight(1, 'attention.query')[lane*16:(lane+1)*16].T for lane in range(2)],dim=1)
    q,g = qg[...,:8], qg[...,8:]
    k = (z @ weight(1, 'attention.key').T).reshape(n,1,8)
    v = (z @ weight(1, 'attention.value').T).reshape(n,1,8)
    q,k = norm(q,weight(1,'attention.query_norm')),norm(k,weight(1,'attention.key_norm'))
    angle = torch.arange(n,dtype=torch.float32)[:,None]*torch.tensor([1.,0.01])[None,:]
    c,s = angle.cos().bfloat16()[:,None,:],angle.sin().bfloat16()[:,None,:]
    def rope(t):
        a,b=t[...,:2],t[...,2:4]
        return torch.cat([a*c-b*s,b*c+a*s,t[...,4:]],dim=-1)
    q,k=rope(q),rope(k)
    parts=[]
    for lane in range(2):
        score=(q[:,lane] @ k[:,0].T)*(8**-0.5)
        score=score.masked_fill(torch.ones(n,n,dtype=torch.bool).triu(1),torch.finfo(torch.bfloat16).min)
        prob=torch.softmax(score.float(),dim=-1).bfloat16()
        y=(prob @ v[:,0])*torch.sigmoid(g[:,lane])
        parts.append(y @ weight(1,'attention.output')[:,lane*8:(lane+1)*8].T)
    return dense(x+(parts[0]+parts[1]),1)

# Eight identical independent CPU batches select the established BF16 rsqrt
# vector path rather than scalar-tail double rounding. No CUDA is initialized.
BATCH=8

def gdn(x, cached=None):
    z=norm(x,weight(2,'input_norm'))
    parts=[]; states=[]
    for lane in range(2):
        def local(name):
            return (tensor(2,'attention.'+name)*(1.+lane*0.1)).bfloat16()
        qkv=(z @ local('qkv').T).unsqueeze(0).repeat(BATCH,1,1)
        a=(z @ local('a').T).unsqueeze(0).repeat(BATCH,1,1)
        b=(z @ local('beta').T).unsqueeze(0).repeat(BATCH,1,1)
        gate=(z @ local('z').T).reshape(1,len(x),2,2).repeat(BATCH,1,1,1)
        hist=torch.zeros(BATCH,10,3,dtype=torch.bfloat16) if cached is None else cached[lane][0].clone()
        state=torch.zeros(BATCH,2,3,2) if cached is None else cached[lane][1].clone()
        conv=hf.torch_causal_conv1d_update(qkv.transpose(1,2),hist,local('conv').squeeze(1),None,'silu').transpose(1,2)
        q,k,v=conv.split([3,3,4],dim=-1)
        q=q.reshape(BATCH,len(x),1,3).repeat_interleave(2,dim=2)
        k=k.reshape(BATCH,len(x),1,3).repeat_interleave(2,dim=2)
        v=v.reshape(BATCH,len(x),2,2)
        beta=b.sigmoid()
        g=-local('a_log').float().exp()*F.softplus(a.float()+local('dt_bias'))
        fn=hf.torch_recurrent_gated_delta_rule if cached is not None and len(x)==1 else hf.torch_chunk_gated_delta_rule
        core,final=fn(q,k,v,g,beta,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=True)
        rms=hf.Qwen3_5RMSNormGated(2,eps=1e-6).bfloat16()
        with torch.no_grad(): rms.weight.copy_(weight(2,'attention.norm'))
        out=rms(core.reshape(-1,2),gate.reshape(-1,2)).reshape(BATCH,len(x),4)[0]
        parts.append(out @ (tensor(2,'attention.output')/2).bfloat16().T)
        states.append((hist,final))
    return dense(x+(parts[0]+parts[1]),2),states

def record(output, states):
    return dict(output=output.contiguous().view(torch.uint16).flatten().tolist(),ranks=[dict(
        conv=h[0].contiguous().view(torch.uint16).flatten().tolist(),
        ssm=s[0].contiguous().view(torch.int32).flatten().tolist()) for h,s in states])

with torch.no_grad():
    x=torch.tensor([((i*7)%19-9)/16 for i in range(16)],dtype=torch.bfloat16).reshape(2,8)
    g=gqa(x)
    full,full_state=gdn(g)
    first,first_state=gdn(gqa(x[:1]))
    decode,decode_state=gdn(g[1:],first_state)
    result=dict(schema='ferrule.mixed-gqa-gdn.tp2.v1',oracle='independent-torch-cpu-hf',
        torch=torch.__version__,transformers=transformers.__version__,hf_source_sha256=pin,
        conv_dtype='bf16',ssm_dtype='f32',tp=2,ep=1,
        prefill=record(full,full_state),fresh_one=record(first,first_state),cached_one=record(decode,decode_state))
    HERE.joinpath('hybrid_mixed_gdn_oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:[len(v['output']),len(v['ranks'][0]['conv']),len(v['ranks'][0]['ssm'])] for k,v in result.items() if isinstance(v,dict)},indent=2))
