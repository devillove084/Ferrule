"""CPU compare all captured boundaries through first hidden mismatch; no oracle edits."""
import json,sys
from pathlib import Path
import torch
from safetensors import safe_open
D=Path(sys.argv[1]);P=D/'producer/hello/prefill';torch.set_num_threads(1)
events=[json.loads(l)for l in(P/'events.jsonl').read_text().splitlines()]
def delta(a,b):
 a=a.float().reshape(-1);b=b.float().reshape(-1);assert a.shape==b.shape,(a.shape,b.shape)
 ids=(a.view(torch.int32)!=b.view(torch.int32)).nonzero().reshape(-1)
 return dict(elements=a.numel(),unequal=ids.numel(),max_abs=float((a-b).abs().max()),first=[dict(index=int(i),actual=float(a[i]),expected=float(b[i]))for i in ids[:4]])
rows=[];unmapped=[]
with safe_open(str(D/'hf-layers0-16.safetensors'),framework='pt')as hf:
 get=hf.get_tensor;keys=set(hf.keys());res={}
 for e in events:
  layer=e['layer']
  if layer is None or layer>16:continue
  prefix=f'{layer}.';event=e['event'];key=None;expected=None
  weight=(e.get('weight')or{}).get('name','')
  if event.startswith(('linear.','norm.')):
   module=weight.split(f'.layers.{layer}.')[-1].removesuffix('.weight')
   key=prefix+module+'.'+event.split('.')[-1]
   if module=='mlp.gate':key=prefix+('post_attention_layernorm.output'if event=='linear.input'else'router.raw')
   if key not in keys and '.mlp.experts.' in key and event=='linear.input' and not module.endswith('down_proj'):key=prefix+'mlp.experts.input'
  elif event=='router':
   key=prefix+'router.raw'
   rows.append(dict(layer=layer,event='router.weights',file=e['file'],reference=prefix+'router.weights',**delta(torch.tensor(e['weights']),get(prefix+'router.weights'))))
   rows.append(dict(layer=layer,event='router.ids',file=e['file'],reference=prefix+'router.ids',**delta(torch.tensor(e['ids']),get(prefix+'router.ids'))))
  elif event in ['gdn.recurrent_output','gdn.gated','gdn.convolved','gdn.conv_state','gdn.recurrent_state','shared_gate.output','routed.output','swiglu.product','attention_gate.output']:
   key=prefix+{'gdn.recurrent_output':'core.output','gdn.gated':'linear_attn.norm.output','gdn.convolved':'convolved','gdn.conv_state':'cache.conv_states','gdn.recurrent_state':'cache.recurrent_states','shared_gate.output':'shared_gated','routed.output':'mlp.experts.output','swiglu.product':'mlp.shared_expert.down_proj.input','attention_gate.output':'self_attn.o_proj.input'}[event]
  elif event.startswith('rope.'):
   which='query' if res.get((layer,event),0)==0 else 'key';res[(layer,event)]=res.get((layer,event),0)+1;key=prefix+'rope.'+which+'.'+event.split('.')[-1]
  elif event.startswith('residual.'):
   rank=res.get((layer,event),0);res[(layer,event)]=rank+1
   attn=get(prefix+('self_attn.output' if prefix+'self_attn.output'in keys else 'linear_attn.output'));h=get(prefix+'layer.input')
   branch=(h+attn).bfloat16()
   if event=='residual.output':expected=[branch,get(prefix+'mlp.output'),get(prefix+'layer.output')][rank]
   elif event=='residual.input':expected=[h,get(prefix+'mlp.experts.output'),branch][rank]
   else:expected=[attn,get(prefix+'shared_gated'),get(prefix+'mlp.output')][rank]
  if expected is None:
   if key not in keys:unmapped.append(dict(layer=layer,event=event,file=e['file'],key=key));continue
   expected=get(key)
  actual=torch.frombuffer(bytearray((P/e['file']).read_bytes()),dtype=torch.float32).reshape(e['shape'])
  rows.append(dict(layer=layer,event=event,file=e['file'],reference=key,**delta(actual,expected)))
rows.sort(key=lambda r:int(r['file'].split('.')[0]))
result=dict(compared=len(rows),unmapped=unmapped,different=[r for r in rows if r['unequal']],comparisons=rows)
(D/'CAUSE-COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(compared=len(rows),unmapped=unmapped,first_differences=result['different'][:15]),indent=2))
