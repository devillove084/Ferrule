"""CPU-only extraction of every captured native BF16 A/B, preserving actual inputs."""
import json, hashlib, sys
from pathlib import Path
import torch
from safetensors import safe_open
root=Path.cwd();out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
hf01=root/'target/validation/bf16-layer01-supported-retry-20261005/hf-layer01.safetensors'
hf234=root/'target/validation/bf16-gqa-boundary-20261005/supplement/layers234.safetensors'
hf511=root/'target/validation/hello-producer-cause-20261007/hf-layers5-11.safetensors'
assert sha(hf01)=='6752faa07d618e884c6f4be51694aa071866fa88f4e1d0b7860e3423c9b21b07'
assert sha(hf234)=='5b503e15030f9bd523c293f07c01ad9edde2c7d32b4497745f00c7eaa0b2dbe8'
proof=json.loads((hf511.parent/'hf-replay.json').read_text());assert proof['capture_sha256']==sha(hf511);assert all(c['unequal']==0 for c in proof['checks'])
def delta(a,b):
 a=a.float().reshape(-1);b=b.float().reshape(-1);assert a.shape==b.shape
 return dict(elements=a.numel(),unequal=int((a.view(torch.int32)!=b.view(torch.int32)).sum()),max_abs=float((a-b).abs().max()))
cases=[];pins={str(p):sha(p) for p in [hf01,hf234,hf511]}
for label,prod in [('hello',root/'target/validation/hello-producer-cause-20261007/producer/hello/prefill'),('capital',root/'target/validation/capital-decode-state-20261007/producer/capital/prefill')]:
 events=[json.loads(l)for l in(prod/'events.jsonl').read_text().splitlines()]
 for e in events:
  name=(e.get('weight')or{}).get('name','')
  if e['event']!='linear.output' or not name.endswith(('.in_proj_a.weight','.in_proj_b.weight')):continue
  layer=e['layer'];module='linear_attn.'+name.split('.')[-2];kind='a' if '.in_proj_a.' in name else 'b';ident=f'{label}{layer}{kind}'
  if label=='capital' and layer>1:continue
  inp=next(v for v in events if v['event']=='linear.input' and v['layer']==layer and (v.get('weight')or{}).get('name')==name)
  def read(v):return torch.frombuffer(bytearray((prod/v['file']).read_bytes()),dtype=torch.float32).reshape(v['shape'])
  x,y=read(inp),read(e);desc=e['weight'];assert desc['dtype']=='Bf16' and e.get('scale') is None
  with Path(desc['path']).open('rb')as f:f.seek(desc['offset']);data=f.read(desc['bytes'])
  w=torch.frombuffer(bytearray(data),dtype=torch.bfloat16).reshape(desc['shape']);assert torch.equal(x,x.bfloat16().float())
  cap=hf01 if layer<2 else hf234 if layer<=4 else hf511;prefix=f'{label}/prefill.{layer}.' if layer<2 else f'{layer}.'
  with safe_open(str(cap),framework='pt')as hf:xi=hf.get_tensor(prefix+module+'.input');yi=hf.get_tensor(prefix+module+'.output')
  for suffix,t in [('input.bf16',x.bfloat16()),('weight.bf16',w),('actual.bf16',y.bfloat16()),('original.bf16',yi.bfloat16())]:
   p=out/(ident+'.'+suffix);p.write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
  cases.append(dict(id=ident,m=x.shape[0],n=w.shape[0],k=w.shape[1],layer=layer,role=kind,input=delta(x,xi),legacy_original=delta(y,yi),weight=desc,weight_sha256=hashlib.sha256(data).hexdigest()))
for p in out.iterdir():pins[str(p.resolve())]=sha(p)
(out/'manifest.json').write_text(json.dumps(dict(cases=cases,pins=pins),indent=2)+'\n')
(out/'cases.txt').write_text(''.join(f"{c['id']} {c['m']} {c['n']} {c['k']}\n"for c in cases))
print(json.dumps(cases,indent=2))
