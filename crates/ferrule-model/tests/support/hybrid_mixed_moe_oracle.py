"""Independent Torch GQA/MoE suffix over the immutable HF/Torch GQA-GDN oracle.

This validates a three-layer dense rank program and a second compact KV layer.
It does not change or regenerate the prefix's CPU or CUDA-state references.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
here = Path(__file__).parent
prefix_path = here / 'hybrid_mixed_gdn_oracle.json'
prefix_pin = '98884445fb90c34747980f2c6a30161532b7a7ae59f7fe7cd218b8a3577cf2b7'
assert hashlib.sha256(prefix_path.read_bytes()).hexdigest() == prefix_pin
prefix = json.loads(prefix_path.read_text())
tensors = json.loads((here / '../fixtures/hybrid_cpu/oracle.json').read_text())['tensors']

def weight(name):
    p = tensors['layers.3.' + name + '.weight']
    return torch.tensor(p['values'], dtype=torch.bfloat16).reshape(p['shape'])

def norm(x, w):
    z = x.float()
    return (z * torch.rsqrt((z*z).mean(-1, keepdim=True) + 1e-6) * (1+w.float())).bfloat16()

def gqa(x, start=0, past=None):
    n = len(x)
    z = norm(x, weight('input_norm'))
    qg = torch.stack([z @ weight('attention.query')[lane*16:(lane+1)*16].T for lane in range(2)], dim=1)
    q, gate = qg[..., :8], qg[..., 8:]
    k = (z @ weight('attention.key').T).reshape(n, 1, 8)
    v = (z @ weight('attention.value').T).reshape(n, 1, 8)
    q, k = norm(q, weight('attention.query_norm')), norm(k, weight('attention.key_norm'))
    angle = torch.arange(start, start+n, dtype=torch.float32)[:, None] * torch.tensor([1., .01])[None, :]
    c, s = angle.cos().bfloat16()[:, None, :], angle.sin().bfloat16()[:, None, :]
    def rope(t):
        a, b = t[..., :2], t[..., 2:4]
        return torch.cat([a*c-b*s, b*c+a*s, t[..., 4:]], dim=-1)
    q, k = rope(q), rope(k)
    if past is not None:
        k, v = torch.cat([past[0], k]), torch.cat([past[1], v])
    parts = []
    for lane in range(2):
        score = (q[:, lane] @ k[:, 0].T) * (8**-.5)
        causal = torch.arange(len(k))[None, :] > torch.arange(start, start+n)[:, None]
        prob = score.masked_fill(causal, torch.finfo(torch.bfloat16).min).float().softmax(-1).bfloat16()
        out = (prob @ v[:, 0]) * gate[:, lane].sigmoid()
        parts.append(out @ weight('attention.output')[:, lane*8:(lane+1)*8].T)
    x = x + (parts[0] + parts[1])
    z = norm(x, weight('post_attention_norm'))
    logits = z @ moe_weight('router').T
    ids = torch.argsort(logits.float(), dim=-1, descending=True, stable=True)[:, :8]
    probabilities = logits.float().softmax(-1).gather(1, ids)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    partials = []
    for lane in range(2):
        def expert(name):
            gate = z @ moe_weight(name+'.gate')[lane*6:(lane+1)*6].T
            up = z @ moe_weight(name+'.up')[lane*6:(lane+1)*6].T
            return (F.silu(gate)*up) @ moe_weight(name+'.down')[:,lane*6:(lane+1)*6].T
        shared = expert('shared') * (z @ moe_weight('shared_output_gate').T).sigmoid()
        routed = torch.zeros_like(x)
        for expert_id in range(16):
            output = expert('experts.'+str(expert_id))
            for row in range(n):
                selected = (ids[row] == expert_id).nonzero()
                if selected.numel():
                    w = probabilities[row,selected[0,0]]
                    routed[row] += (output[row].float()*w).bfloat16()
        partials.append(shared+routed)
    return x+(partials[0]+partials[1]), (k,v), dict(
        expert_ids=ids.flatten().tolist(), weight_bits=probabilities.contiguous().view(torch.int32).flatten().tolist(),
        partials=[words(p) for p in partials], logits=words(logits))

def input_for(case):
    return torch.tensor(prefix[case]['output'], dtype=torch.uint16).view(torch.bfloat16).reshape(-1, 8)

def words(x):
    return x.contiguous().view(torch.uint16).flatten().tolist()

# Source weights are independent of Ferrule preparation and retained in the fixture.
moe = {}
for matrix in ['gate','up','down']:
    original = tensors['layers.3.feed_forward.'+matrix+'.weight']
    base = torch.tensor(original['values'], dtype=torch.float32).reshape(original['shape'])
    moe['shared.'+matrix] = dict(shape=original['shape'], values=base.flatten().tolist())
    for expert in range(16):
        values = base*(0.5+expert/32.)
        moe['experts.'+str(expert)+'.'+matrix] = dict(shape=original['shape'], values=values.flatten().tolist())
# Two raw-logit tie groups; the selected half changes with the token's sign.
router = torch.zeros(16,8)
router[:8,0] = 0.5
router[8:,0] = -0.5
moe['router'] = dict(shape=[16,8],values=router.flatten().tolist())
moe['shared_output_gate'] = dict(shape=[1,8],values=[.25,-.125,.0625,0.,.125,0.,-.25,.0625])
def moe_weight(name):
    p=moe[name]
    return torch.tensor(p['values'],dtype=torch.bfloat16).reshape(p['shape'])

with torch.no_grad():
    prefill, _, route_full = gqa(input_for('prefill'))
    first, history, route_first = gqa(input_for('fresh_one'))
    cached, _, route_cached = gqa(input_for('cached_one'), start=1, past=history)
    assert route_first['expert_ids'] != route_cached['expert_ids']
    result = dict(schema='ferrule.mixed-rank-moe.tp2.v1',layers=[1,2,3],experts=16,top_k=8,
        oracle='independent-torch-cpu-hf-eager-tp2',torch=torch.__version__,prefix_sha256=prefix_pin,
        tensors={'layers.3.feed_forward.'+name+'.weight':p for name,p in moe.items()},
        prefill=dict(output=words(prefill),routes=route_full),fresh_one=dict(output=words(first),routes=route_first),
        cached_one=dict(output=words(cached),routes=route_cached))
    here.joinpath('hybrid_mixed_moe_oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    print({c:result[c]['routes']['expert_ids'] for c in ['prefill','fresh_one','cached_one']})
