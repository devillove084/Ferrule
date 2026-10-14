"""One pinned native HF layer15 replay on exact fresh QA decode input/cache."""
import json,sys
from pathlib import Path
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts'));import qwen35_fp8_bf16_reference as ref
pins=ref.initialize();import torch
from safetensors import safe_open
from safetensors.torch import save_file
D=Path(sys.argv[1]);P=D/'narrow';events=[json.loads(l)for l in(P/'events.jsonl').read_text().splitlines()if json.loads(l)['layer']==15]
store=ref.legacy.Checkpoint(Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8'));weights=ref.DeviceWeights(store,'cuda:0',256<<20);config=ref.read_config(store.directory);saved={};checks=[]
def cap(key,t):
 if isinstance(t,tuple):t=t[0]
 saved[key]=t.detach().cpu().contiguous().clone()
def delta(a,b):
 a=a.detach().float().cpu().reshape(-1);b=b.detach().float().cpu().reshape(-1);assert a.shape==b.shape
 ii=(a.view(torch.int32)!=b.view(torch.int32)).nonzero().reshape(-1);return dict(elements=a.numel(),unequal=ii.numel(),max_abs=float((a-b).abs().max()),first=[dict(index=int(i),actual=float(a[i]),expected=float(b[i]))for i in ii[:4]])
def actual(e):return torch.frombuffer(bytearray((P/e['file']).read_bytes()),dtype=torch.float32).reshape(e['shape'])
with torch.inference_mode(),safe_open(str(D/'oracle/qa.safetensors'),framework='pt')as hf:
 cache=ref.f32_ssm_cache(config,['cuda:0']*40)
 state=cache.layers[15];state.keys=hf.get_tensor('step.00.cache.key_cache.15').cuda();state.values=hf.get_tensor('step.00.cache.value_cache.15').cuda();state.is_initialized=True;state.dtype=torch.bfloat16;state.device=torch.device('cuda:0')
 x=hf.get_tensor('step.01.layers.14.hidden').cuda();cap('layer.input',x)
 with torch.device('meta'):layer=ref.hf.Qwen3_5MoeDecoderLayer(config,15)
 ref.bind(layer,'model.layers.15',weights,config)
 for name,module in layer.named_modules():
  if name and (module.__class__.__name__=='Linear' or name in ['input_layernorm','post_attention_layernorm','self_attn.q_norm','self_attn.k_norm','self_attn','mlp','mlp.experts','mlp.shared_expert','mlp.shared_expert_gate']):
   module.register_forward_pre_hook(lambda m,args,kwargs,key=name+'.input':cap(key,args[0]if args else kwargs['hidden_states']),with_kwargs=True)
   module.register_forward_hook(lambda m,args,result,key=name+'.output':cap(key,result))
 def route(m,args,result):
  raw,selected=ref.router_observation(m,args,result,ref.ROUTER_PROFILE);cap('router.raw',raw);cap('router.weights',selected[1]);cap('router.ids',selected[2]);return selected
 layer.mlp.gate.register_forward_hook(route)
 rope=ref.hf.Qwen3_5MoeTextRotaryEmbedding(config,device='cuda:0');ref.legacy.audit_rope_f32(rope);pos=torch.tensor([[5]],device='cuda:0')
 y=layer(x,position_embeddings=rope(x,pos),position_ids=pos,past_key_values=cache,attention_mask=None);cap('layer.output',y)
 for boundary,name in [('attention','self_attn.output'),('moe','mlp.output'),('hidden','layer.output')]:
  check=dict(boundary=boundary,**delta(saved[name],hf.get_tensor('step.01.layers.15.'+boundary)));checks.append(check);assert check['unequal']==0,check
 for field,name in [('key_cache','keys'),('value_cache','values')]:
  check=dict(boundary=field,**delta(getattr(cache.layers[15],name),hf.get_tensor('step.01.cache.'+field+'.15')));checks.append(check);assert check['unequal']==0,check
 for name,parameter in layer.named_parameters():cap('loaded.'+name,parameter)
 comparisons=[]
 for e in events:
  event=e['event'];weight=e.get('weight')or'';key=None
  if event.startswith(('linear.','norm.')):
   module=weight.split('.layers.15.')[-1].removesuffix('.weight');key=module+'.'+event.split('.')[-1]
   if module=='mlp.gate':key='post_attention_layernorm.output'if event=='linear.input'else'router.raw'
  elif event=='routed.output':key='mlp.experts.output'
  elif event=='router':key='router.raw'
  if key in saved:
   record=dict(file=e['file'],event=event,key=key,**delta(actual(e),saved[key]))
   if record['unequal']and event=='linear.output':
    inputevent=next(v for v in events if v['event']=='linear.input'and v.get('weight')==weight)
    if '.mlp.experts.'not in weight:record['sameinput']=delta(actual(e),ref.F.linear(actual(inputevent).cuda().bfloat16(),weights.matrix(weight)))
   if record['unequal']and event=='norm.output':
    inp=next(v for v in events if v['event']=='norm.input'and v.get('weight')==weight);norm=layer.get_submodule(module);record['sameinput']=delta(actual(e),norm(actual(inp).cuda().bfloat16()))
   comparisons.append(record)
 save_file(saved,str(D/'layer15-replay.safetensors'))
 result=dict(checks=checks,comparisons=comparisons,first_difference=next((c for c in comparisons if c['unequal']),None),source_pins=pins,oracle_sha256=ref.sha256(D/'oracle/qa.safetensors'),loaded_weight_rules='original maintained ref.bind, native BF16 and parameter-only BF16 cast',layer_input=delta(actual(next(e for e in events if e['event']=='norm.input')),x))
 ref.write_json(D/'LAYER15-REPLAY.json',result);print(json.dumps(result,indent=2))
