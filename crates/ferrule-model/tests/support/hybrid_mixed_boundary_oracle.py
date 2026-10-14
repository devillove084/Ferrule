"""Independent Torch endpoints over the immutable CUDA MoE program oracle.

Embedding rows reproduce the prefix's fixed inputs; final norm and ragged TP2
head are computed independently, BF16 on every producer/collective boundary.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json
from pathlib import Path
import torch

torch.set_num_threads(1)
here=Path(__file__).parent
prefix_file=here/'hybrid_mixed_moe_cuda_oracle.json'
prefix=json.loads(prefix_file.read_text())
raw=json.loads((here/'../fixtures/hybrid_cpu/oracle.json').read_text())['tensors']
def weight(name):
    p=raw[name+'.weight']
    return torch.tensor(p['values'],dtype=torch.bfloat16).reshape(p['shape'])
embedding=weight('token_embedding').float()
embedding[1:3]=torch.tensor([((i*7)%19-9)/16 for i in range(16)],dtype=torch.float32).reshape(2,8)
result=dict(schema='ferrule.mixed-boundary.tp2.v1',oracle='independent-torch-cpu-endpoints',torch=torch.__version__,
    prefix_sha256=hashlib.sha256(prefix_file.read_bytes()).hexdigest(),
    embedding=dict(shape=[11,8],values=embedding.flatten().tolist()),head_ranges=[[0,6],[6,11]])
for case in ['prefill','fresh_one','cached_one']:
    x=torch.tensor(prefix[case]['output'],dtype=torch.uint16).view(torch.bfloat16).reshape(-1,8)
    z=x.float()
    z=(z*torch.rsqrt((z*z).mean(-1,keepdim=True)+1e-6)*(1+weight('final_norm').float())).bfloat16()
    parts=[z@weight('output')[start:end].T for start,end in result['head_ranges']]
    logits=torch.cat(parts,dim=-1)
    result[case]=dict(output=logits.contiguous().view(torch.uint16).flatten().tolist(),local_head=[p.contiguous().view(torch.uint16).flatten().tolist() for p in parts])
here.joinpath('hybrid_mixed_boundary_oracle.json').write_text(json.dumps(result,indent=2)+'\n')
print({c:len(result[c]['output']) for c in ['prefill','fresh_one','cached_one']})
