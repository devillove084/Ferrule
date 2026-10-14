"""CPU-only immutable BF16 router/control extraction with independent HF inputs."""
import json,hashlib,sys
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path.cwd();OUT=Path(sys.argv[1]);OUT.mkdir(exist_ok=False,parents=True);torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=ROOT/'target/validation/bf16-layer01-supported-retry-20261005';sup=ROOT/'target/validation/bf16-gqa-boundary-20261005/supplement'
assert sha(old/'hf-layer01.safetensors')=='6752faa07d618e884c6f4be51694aa071866fa88f4e1d0b7860e3423c9b21b07'
assert sha(sup/'layers234.safetensors')=='5b503e15030f9bd523c293f07c01ad9edde2c7d32b4497745f00c7eaa0b2dbe8'
def diff(a,b):
 a=a.float().reshape(-1);b=b.float().reshape(-1);assert a.shape==b.shape;return dict(elements=a.numel(),unequal=int((a.view(torch.int32)!=b.view(torch.int32)).sum()),max_abs=float((a-b).abs().max()))
cases=[];pins={}
for case,layer,stage,module in [('hello0',0,'hello/prefill','mlp.gate'),('hello3',3,'hello/prefill','mlp.gate'),('capital0',0,'capital/prefill','mlp.gate'),('capital1',1,'capital/prefill','mlp.gate'),('hello4a',4,'hello/prefill','linear_attn.in_proj_a'),('hello4b',4,'hello/prefill','linear_attn.in_proj_b')]:
 prod=ROOT/('target/validation/hello-producer-cause-20261007/producer'if stage.startswith('hello')else'target/validation/capital-decode-state-20261007/producer')/stage
 events=[json.loads(s)for s in(prod/'events.jsonl').read_text().splitlines()];weight=f'model.language_model.layers.{layer}.{module}.weight'
 get=lambda name:next(e for e in events if e['layer']==layer and e['event']==name and(e.get('weight')or{}).get('name')==weight)
 ei,eo=get('linear.input'),get('linear.output');load=lambda e:torch.frombuffer(bytearray((prod/e['file']).read_bytes()),dtype=torch.float32).reshape(e['shape']);x,y=load(ei),load(eo);desc=eo['weight'];assert desc['dtype']=='Bf16'and eo.get('scale')is None
 with Path(desc['path']).open('rb')as f:f.seek(desc['offset']);data=f.read(desc['bytes'])
 w=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(desc['shape']);assert torch.equal(x,x.bfloat16().float())
 capture=(sup/'layers234.safetensors')if layer in [3,4]else(old/'hf-layer01.safetensors');prefix=f'{layer}.'if layer in [3,4]else f'{stage}.{layer}.'
 with safe_open(str(capture),framework='pt')as hf:
  xi=hf.get_tensor(prefix+('post_attention_layernorm.output'if module=='mlp.gate'else module+'.input'));expected=hf.get_tensor(prefix+('router.raw'if module=='mlp.gate'else module+'.output'))
  if module=='mlp.gate':
   for name,key in [('hf.ids','router.ids'),('hf.weights','router.weights')]:
    t=hf.get_tensor(prefix+key);(OUT/(case+'.'+name)).write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
 m,k=x.shape;n=w.shape[0]
 for name,t in [('input.bf16',x.bfloat16()),('weight.bf16',w),('actual.bf16',y.bfloat16()),('hf.bf16',expected.bfloat16())]:
  p=OUT/(case+'.'+name);p.write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes());pins[p.name]=sha(p)
 rec=dict(id=case,m=m,n=n,k=k,module=module,input=diff(x,xi),output=diff(y,expected),weight=desc,weight_sha256=hashlib.sha256(data).hexdigest());cases.append(rec)
 if module=='mlp.gate':
  e=next(e for e in events if e['layer']==layer and e['event']=='router');(OUT/(case+'.routes.json')).write_text(json.dumps({'ids':e['ids'],'weights':e['weights']}))
for p in OUT.iterdir():
 if p.is_file():pins[p.name]=sha(p)
(OUT/'manifest.json').write_text(json.dumps(dict(cases=cases,pins=pins),indent=2)+'\n');print(json.dumps(cases,indent=2))
