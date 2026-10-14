"""Independent Torch GQA suffix over the immutable HF/Torch GQA-GDN oracle.

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
    parts = []
    for lane in range(2):
        gate = z @ weight('feed_forward.gate')[lane*6:(lane+1)*6].T
        up = z @ weight('feed_forward.up')[lane*6:(lane+1)*6].T
        parts.append((F.silu(gate) * up) @ weight('feed_forward.down')[:, lane*6:(lane+1)*6].T)
    return x + (parts[0] + parts[1]), (k, v)

def input_for(case):
    return torch.tensor(prefix[case]['output'], dtype=torch.uint16).view(torch.bfloat16).reshape(-1, 8)

def words(x):
    return x.contiguous().view(torch.uint16).flatten().tolist()

with torch.no_grad():
    prefill, _ = gqa(input_for('prefill'))
    first, history = gqa(input_for('fresh_one'))
    cached, _ = gqa(input_for('cached_one'), start=1, past=history)
    result = dict(schema='ferrule.mixed-rank-program.tp2.v1', layers=[1, 2, 3],
        oracle='independent-torch-cpu-suffix-over-frozen-hf-prefix', torch=torch.__version__,
        prefix_sha256=prefix_pin, prefill=dict(output=words(prefill)),
        fresh_one=dict(output=words(first)), cached_one=dict(output=words(cached)))
    here.joinpath('hybrid_mixed_program_oracle.json').write_text(json.dumps(result, indent=2)+'\n')
    print({case: len(result[case]['output']) for case in ['prefill', 'fresh_one', 'cached_one']})
