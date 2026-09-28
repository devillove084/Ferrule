"""Same-input Layer-0 conv/GDN/norm isolation using native HF Torch functions.
No model is constructed; only four small nonquantized tensors are read.
"""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule,
)
from analyze_numeric_trace import diff

p = argparse.ArgumentParser()
p.add_argument('--trace', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(8)
config = json.loads((a.model/'config.json').read_text())['text_config']
index = json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
def weight(suffix):
    name = 'model.language_model.layers.0.linear_attn.'+suffix
    with safe_open(a.model/index[name], framework='pt', device='cpu') as f:
        return f.get_tensor(name).float()
conv_w, alog, dt, norm = [weight(n) for n in ['conv1d.weight','A_log','dt_bias','norm.weight']]
kh, vh = config['linear_num_key_heads'], config['linear_num_value_heads']
kd, vd = config['linear_key_head_dim'], config['linear_value_head_dim']
events = [json.loads(line) for line in (a.trace/'events.jsonl').read_text().splitlines()]
def read(event):
    return torch.frombuffer(bytearray((a.trace/event['file']).read_bytes()), dtype=torch.float32).reshape(event['shape'])
report = []
previous = None
for stage in ['prefill','decode.0']:
    es = [e for e in events if e['stage']==stage and e['layer']==0]
    def get(name, role=None):
        matches = [e for e in es if e['event']==name and (role is None or e['role']==role)]
        assert len(matches)==1, (name,role,len(matches))
        return read(matches[0])
    def compare(label, actual, expected):
        result = dict(stage=stage, boundary=label, **diff(actual,expected))
        report.append(result)
        print(stage,label,{k:result[k] for k in ['max_abs','outside','max_ulp','crossed_bf16_bins']})
    history = get('gdn.conv_state').reshape(1,-1,config['linear_conv_kernel_dim'])
    cpu_conv = F.silu(F.conv1d(history,conv_w,groups=history.shape[1])).transpose(1,2)
    gpu_conv = get('gdn.convolved').reshape(1,1,-1)
    compare('conv+silu SAME history/weight',gpu_conv,cpu_conv)
    q,k,v = gpu_conv.split([kh*kd,kh*kd,vh*vd],dim=-1)
    q=q.reshape(1,1,kh,kd).repeat_interleave(vh//kh,dim=2)
    k=k.reshape(1,1,kh,kd).repeat_interleave(vh//kh,dim=2)
    v=v.reshape(1,1,vh,vd)
    decay=get('linear.output','LinearAttentionA').reshape(1,1,vh)
    beta=get('linear.output','LinearAttentionBeta').reshape(1,1,vh).sigmoid()
    g=-alog.exp()*F.softplus(decay+dt)
    fn=torch_chunk_gated_delta_rule if stage=='prefill' else torch_recurrent_gated_delta_rule
    out,state=fn(q,k,v,g,beta,initial_state=previous,output_final_state=True,use_qk_l2norm_in_kernel=True)
    compare('GDN output SAME convolved/projections/state',get('gdn.recurrent_output'),out)
    compare('GDN state SAME convolved/projections/state',get('gdn.recurrent_state'),state)
    previous=get('gdn.recurrent_state').reshape(1,vh,kd,vd)
    gpu_out=get('gdn.recurrent_output').reshape(-1,vd)
    normalized=gpu_out*torch.rsqrt(gpu_out.square().mean(-1,keepdim=True)+config['rms_norm_eps'])*norm
    compare('GDN RMSNorm SAME input',get('gdn.normalized'),normalized)
    z=get('linear.output','LinearAttentionZ').reshape(-1,vd)
    compare('GDN SiLU gate SAME input',get('gdn.gated'),get('gdn.normalized').reshape(-1,vd)*F.silu(z))
(a.trace/'gdn-isolation.json').write_text(json.dumps(report,indent=2)+'\n')
