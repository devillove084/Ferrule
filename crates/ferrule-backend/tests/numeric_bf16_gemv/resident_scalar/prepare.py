"""CPU-only original scalar witnesses plus deterministic BF16 public-GemmEx operands."""
import json,hashlib,sys,shutil
from pathlib import Path
import torch
from safetensors import safe_open
root=Path.cwd();out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
frozen=root/'target/validation/bf16-fresh-quality-20261007/frozen-shared-output-gate';meta=json.loads((frozen/'manifest.json').read_text())
for name,h in meta['pins'].items():assert sha(frozen/name)==h
for src,dst in [('input','input'),('weight','weight'),('expected','hf')]:shutil.copyfile(frozen/(src+'.bf16'),out/('qa15gate.'+dst+'.bf16'))
cases=[dict(id='qa15gate',m=1,n=1,k=2048,source=str(frozen/'manifest.json'),input_unequal=0)]
hfpath=root/'target/validation/bf16-layer01-supported-retry-20261005/hf-layer01.safetensors';assert sha(hfpath)=='6752faa07d618e884c6f4be51694aa071866fa88f4e1d0b7860e3423c9b21b07'
for label,layer in [('hello',0),('capital',0),('capital',1)]:
 prod=root/('target/validation/hello-producer-cause-20261007/producer'if label=='hello'else'target/validation/capital-decode-state-20261007/producer')/label/'prefill';events=[json.loads(l)for l in(prod/'events.jsonl').read_text().splitlines()];weight=f'model.language_model.layers.{layer}.mlp.shared_expert_gate.weight'
 es={e['event']:e for e in events if e['layer']==layer and(e.get('weight')or{}).get('name')==weight};e=es['linear.input'];x=torch.frombuffer(bytearray((prod/e['file']).read_bytes()),dtype=torch.float32).reshape(e['shape']);desc=es['linear.output']['weight'];assert desc['dtype']=='Bf16'
 with Path(desc['path']).open('rb')as f:f.seek(desc['offset']);w=torch.frombuffer(bytearray(f.read(desc['bytes'])),dtype=torch.bfloat16).reshape(desc['shape'])
 with safe_open(str(hfpath),framework='pt')as hf:
  prefix=f'{label}/prefill.{layer}.mlp.shared_expert_gate.';xi=hf.get_tensor(prefix+'input');yi=hf.get_tensor(prefix+'output');assert torch.equal(x.reshape(-1),xi.float().reshape(-1))
 ident=f'{label}{layer}gate';m,k=x.shape;assert w.shape==(1,k)
 for name,t in [('input',x.bfloat16()),('weight',w),('hf',yi.bfloat16())]:(out/(ident+'.'+name+'.bf16')).write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
 cases.append(dict(id=ident,m=m,n=1,k=k,weight=desc,input_unequal=0))
for m in [1,5]:
 for n in [1,3,7,8,31]:
  for k in [1,7,31,2048]:
   ident=f'tail-{m}-{n}-{k}'
   x=(((torch.arange(m*k)*17)%41-20).float()/32).bfloat16();w=(((torch.arange(n*k)*13)%37-18).float()/32).bfloat16()
   for name,t in [('input',x),('weight',w)]:(out/(ident+'.'+name+'.bf16')).write_bytes(t.view(torch.uint8).numpy().tobytes())
   cases.append(dict(id=ident,m=m,n=n,k=k,reference='public CUDA12.4 BF16-output GemmEx, element-aligned+2B pointers'))
(out/'cases.txt').write_text(''.join(f"{c['id']} {c['m']} {c['n']} {c['k']}\n"for c in cases));(out/'actual-cases.txt').write_text(''.join(f"{c['id']} {c['m']} {c['n']} {c['k']}\n"for c in cases[:4]));(out/'manifest.json').write_text(json.dumps(dict(cases=cases,pins={p.name:sha(p)for p in out.iterdir()}),indent=2)+'\n');print('cases',len(cases),'actual',4)
