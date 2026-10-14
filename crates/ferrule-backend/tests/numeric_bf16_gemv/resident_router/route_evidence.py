"""Separate pinned HF process: observe full softmax/selected renorm on frozen raw logits."""
import sys,json
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts'));import qwen35_fp8_bf16_reference as ref
pins=ref.initialize();import torch
D=Path(sys.argv[1]);manifest=json.loads((D/'manifest.json').read_text());report=[]
def diff(a,b):
 a=a.detach().float().cpu().reshape(-1);b=b.detach().float().cpu().reshape(-1);assert a.shape==b.shape;idx=(a.view(torch.int32)!=b.view(torch.int32)).nonzero().reshape(-1);return dict(elements=a.numel(),unequal=idx.numel(),max_abs=float((a-b).abs().max()),first=[dict(index=int(i),actual=float(a[i]),expected=float(b[i]))for i in idx[:3]])
with torch.inference_mode():
 for c in manifest['cases']:
  if c['module']!='mlp.gate':continue
  name=c['id'];m,n=c['m'],c['n'];read=lambda s,dtype:torch.frombuffer(bytearray((D/(name+'.'+s)).read_bytes()),dtype=dtype).cuda()
  raw=read('actual.bf16',torch.bfloat16).reshape(m,n);hf=read('hf.bf16',torch.bfloat16).reshape(m,n);ids=read('hf.ids',torch.int64).reshape(m,8);expected=read('hf.weights',torch.float32).reshape(m,8);prod=json.loads((D/(name+'.routes.json')).read_text());actual=torch.tensor(prod['weights']).reshape(m,8)
  probs=torch.softmax(raw,dim=-1,dtype=torch.float32);selected=probs.gather(-1,ids);denom=selected.sum(-1,keepdim=True);normalized=selected/denom
  logits=raw.float().gather(-1,ids);exps=(logits-logits.max(-1,keepdim=True).values).exp();serial=torch.zeros((m,1),device='cuda:0')
  for i in range(8):serial=serial+exps[:,i:i+1]
  selected_serial=exps/serial;selected_tree=exps/exps.sum(-1,keepdim=True)
  # Do not replace returned IDs: retain the original raw-logit tie policy.
  rec=dict(case=name,input=c['input'],raw_projection=diff(raw,hf),actual_vs_full_softmax_renorm=diff(actual,normalized),full_softmax_renorm_vs_original=diff(normalized,expected),actual_vs_selected_exp_serial=diff(actual,selected_serial),actual_vs_selected_exp_tree=diff(actual,selected_tree),ids_equal=torch.equal(torch.tensor(prod['ids']).reshape(m,8),ids.cpu()))
  for key,value in [('probabilities',probs),('selected',selected),('denominator',denom),('normalized',normalized),('selected-exp-serial',selected_serial),('selected-exp-tree',selected_tree)]:(D/(name+'.'+key+'.f32')).write_bytes(value.cpu().contiguous().numpy().tobytes())
  report.append(rec)
(D/'route-evidence.json').write_text(json.dumps(dict(cases=report,source_pins=pins,production_algorithm='topk raw logits then selected-only exponential softmax, serial denominator; no full-expert softmax materialization',oracle='native full F32 softmax then topk and selected sum/divide; algebraically equivalent but different finite-precision arithmetic'),indent=2)+'\n');print(json.dumps(report,indent=2))
