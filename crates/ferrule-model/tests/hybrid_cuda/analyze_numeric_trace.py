"""Read opt-in CUDA trace; independently compare boundaries and exact-input operators.

Never loads a model. Only the selected projection/scale pairs are decoded at once.
All outputs belong under target/validation, not in version control.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def tensor(meta):
    with open(meta['path'], 'rb') as file:
        file.seek(meta['offset'])
        raw = bytearray(file.read(meta['bytes']))
    assert len(raw) == meta['bytes']
    dtype = {'F32': torch.float32, 'Bf16': torch.bfloat16,
             'F8E4M3': torch.float8_e4m3fn}[meta['dtype']]
    return torch.frombuffer(raw, dtype=dtype).reshape(meta['shape'])


def decoded(event):
    w = tensor(event['weight']).float()
    if event['numeric']:
        s = tensor(event['scale']).float()
        for row in range(s.shape[0]):
            for col in range(s.shape[1]):
                w[row*128:(row+1)*128, col*128:(col+1)*128] *= s[row, col]
        w = w.bfloat16().float()
    return w


def ordered(x):
    b = x.contiguous().view(torch.int32).long()
    return torch.where(b < 0, 0x80000000 - (b & 0x7fffffff), 0x80000000 + b)


def diff(actual, expected, atol=.002, rtol=.002):
    a, e = actual.flatten().float(), expected.flatten().float()
    assert a.shape == e.shape
    assert torch.isfinite(a).all() and torch.isfinite(e).all()
    d = (a.double() - e.double()).abs()
    u = (ordered(a) - ordered(e)).abs()
    bins = a.bfloat16().view(torch.int16) != e.bfloat16().view(torch.int16)
    idx = torch.topk(d, min(5, len(d))).indices.tolist()
    return dict(n=len(a), max_abs=d.max().item(), rms=d.square().mean().sqrt().item(),
                outside=int((d > atol + rtol*e.double().abs()).sum()),
                max_ulp=int(u.max()), unequal=int((u != 0).sum()),
                first_unequal=int(torch.nonzero(u).flatten()[0]) if torch.any(u) else None,
                first_outside=int(torch.nonzero(d > atol + rtol*e.double().abs()).flatten()[0]) if torch.any(d > atol + rtol*e.double().abs()) else None,
                top_ulp=[dict(index=i,ulp=int(u[i]),actual=a[i].item(),expected=e[i].item(),abs=d[i].item()) for i in torch.topk(u,min(5,len(u))).indices.tolist()],
                crossed_bf16_bins=int(bins.sum()), top=[dict(index=i, actual=a[i].item(),
                expected=e[i].item(), abs=d[i].item(), ulp=int(u[i]),
                a_bf16=a[i].bfloat16().float().item(), e_bf16=e[i].bfloat16().float().item()) for i in idx])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--isolate-layers', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision('highest')
    events = [json.loads(line) for line in (args.trace/'events.jsonl').read_text().splitlines()]
    def read(event):
        return torch.frombuffer(bytearray((args.trace/event['file']).read_bytes()), dtype=torch.float32).reshape(event['shape'])
    def one(es, name, role=None):
        matches = [v for v in es if v['event']==name and (role is None or v['role']==role)]
        assert len(matches)==1, (name, role, len(matches))
        return matches[0]
    report = dict(boundaries=[], routes=[], isolated=[])
    with safe_open(args.oracle, framework='pt', device='cpu') as oracle:
        for stage in ['prefill', 'decode.0']:
            se = [e for e in events if e['stage']==stage]
            embedding = read(one(se, 'embedding'))
            report['boundaries'].append(dict(stage=stage, layer=-1, boundary='embedding', **diff(embedding,oracle.get_tensor(stage+'.embedding'))))
            first = None
            for layer in range(40):
                es = [e for e in se if e['layer']==layer]
                prefix = f'{stage}.layers.{layer:02d}'
                residuals = [e for e in es if e['event']=='residual.update']
                hidden = [e for e in es if e['event']=='residual.output'][-1]
                assert len(residuals)==3, (layer, len(residuals))
                comparisons = [('attention', read(residuals[0]), oracle.get_tensor(prefix+'.attention')),
                    ('shared_gate_logit', read(one(es,'linear.output','SharedExpertOutputGate')), oracle.get_tensor(prefix+'.shared_gate_logit')),
                    ('moe',read(residuals[-1]),oracle.get_tensor(prefix+'.moe')),
                    ('hidden',read(hidden),oracle.get_tensor(prefix+'.hidden'))]
                ref_input=oracle.get_tensor(stage+'.embedding' if layer==0 else f'{stage}.layers.{layer-1:02d}.hidden').reshape(1,-1)
                for role, x in [('AttentionNorm', ref_input), ('FeedForwardNorm', ref_input+oracle.get_tensor(prefix+'.attention').reshape(1,-1))]:
                    ne = one(es,'norm.output',role)
                    w=tensor(ne['weight']).float()
                    normalized=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+ne['epsilon']))*(w+int(ne['one_plus_weight']))
                    comparisons.append((role,read(ne),normalized))
                qkv_events = [v for v in es if v["event"] == "linear.output" and v["role"] == "LinearAttentionQkv"]
                if qkv_events:
                    ref_qkv = oracle.get_tensor(f"{stage}.cache.conv_states.{layer:02d}")[0, :, -1]
                    comparisons.append(("QKV", read(qkv_events[0]), ref_qkv))
                order=['AttentionNorm','QKV','attention','FeedForwardNorm','shared_gate_logit','moe','hidden']
                comparisons.sort(key=lambda entry:order.index(entry[0]))
                for name,a,e in comparisons:
                    result = dict(stage=stage,layer=layer,boundary=name,**diff(a,e))
                    report['boundaries'].append(result)
                    if result['outside'] and first is None:
                        first=(layer,name,result['max_abs'],result['outside'])
                    if layer<3:
                        print(stage,layer,name,{k:result[k] for k in ['max_abs','outside','max_ulp','crossed_bf16_bins']})
                route=one(es,'router')
                expected_ids=oracle.get_tensor(prefix+'.selected_expert_ids').flatten().tolist()
                ids=route['ids']
                reference_weights=oracle.get_tensor(prefix+'.routing_weights').flatten()
                weights=diff(torch.tensor(route['weights']),reference_weights)
                router_projection=one(es,'linear.output','RouterLogits')
                rw=decoded(router_projection)
                ref_ffn=next(v for name,_,v in comparisons if name=='FeedForwardNorm')
                ref_logits=F.linear(ref_ffn.bfloat16().float() if router_projection['numeric'] else ref_ffn,rw)
                actual_logits=read(route)
                report['isolated'].append(dict(stage=stage,layer=layer,parameter='router logits ORACLE normalized input',**diff(actual_logits,ref_logits)))
                gv,gi=torch.topk(actual_logits.flatten(),9)
                rv,ri=torch.topk(ref_logits.flatten(),9)
                shared_ids=sorted(set(ids)&set(expected_ids))
                aligned=diff(torch.tensor([route['weights'][ids.index(i)] for i in shared_ids]),torch.tensor([reference_weights[expected_ids.index(i)] for i in shared_ids]))
                report['routes'].append(dict(stage=stage,layer=layer,actual=ids,expected=expected_ids,
                    positional_mismatch=sum(a!=e for a,e in zip(ids,expected_ids)),
                    set_mismatch=sorted(ids)!=sorted(expected_ids),weights=weights,weights_aligned_by_id=aligned,
                    gpu_top9_ids=gi.tolist(),gpu_top9_logits=gv.tolist(),reference_top9_ids=ri.tolist(),reference_top9_logits=rv.tolist(),
                    reference_router_reconstructed=ri[:8].tolist()==expected_ids))
                if ids!=expected_ids: print('ROUTE MISMATCH',stage,layer,ids,expected_ids)
                recomputed_ids=torch.topk(actual_logits,8,dim=-1).indices.flatten().tolist()
                assert recomputed_ids==ids, ('GPU router selector differs on same logits',stage,layer)
                recomputed_weights=torch.softmax(actual_logits,-1).flatten()[ids]
                recomputed_weights/=recomputed_weights.sum()
                report['isolated'].append(dict(stage=stage,layer=layer,parameter='router weights SAME logits',**diff(torch.tensor(route['weights']),recomputed_weights)))
                for ne in [e for e in es if e['event']=='norm.output']:
                    ni=one(es,'norm.input',ne['role'])
                    x=read(ni).reshape(-1,ne['weight']['shape'][0])
                    w=tensor(ne['weight']).float()
                    expected=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+ne['epsilon'])*(w+int(ne['one_plus_weight']))
                    report['isolated'].append(dict(stage=stage,layer=layer,parameter=ne['parameter']+' same-input norm',**diff(read(ne),expected)))
                if layer>=args.isolate_layers: continue
                inputs={e['parameter']:e for e in es if e['event']=='linear.input'}
                for out in [e for e in es if e['event']=='linear.output']:
                    inp=inputs[out['parameter']]
                    x=read(inp)
                    w=decoded(out)
                    product_input=x.bfloat16().float() if out['numeric'] else x
                    expected=F.linear(product_input,w)
                    result=dict(stage=stage,layer=layer,parameter=out['parameter'],numeric=out['numeric'],
                        **diff(read(out),expected))
                    report['isolated'].append(result)
                    if 'experts.' not in out['parameter']:
                        print('SAME INPUT',stage,out['parameter'],{k:result[k] for k in ['max_abs','outside','max_ulp','crossed_bf16_bins']})
                    if layer==0 and out['role']=='LinearAttentionQkv':
                        ref_norm = next(v for name,a,v in comparisons if name=='AttentionNorm')
                        ref_proj=F.linear(ref_norm.bfloat16().float(),w)
                        report['isolated'].append(dict(stage=stage,layer=layer,parameter=out['parameter']+' ORACLE INPUT',**diff(read(out),ref_proj)))
                        report['isolated'].append(dict(stage=stage,layer=layer,parameter=out['parameter']+' F64 SUM same input',**diff(read(out),F.linear(product_input.double(),w.double()).float())))
                    del w
            print('FIRST OUTSIDE',stage,first)
    path=args.trace/'analysis.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print('Wrote',path)

if __name__=='__main__': main()
