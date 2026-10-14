"""Pinned independent HF F.linear on actual captured input; originals immutable."""
import json,sys,hashlib
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'scripts'));import qwen35_fp8_bf16_reference as ref
pins=ref.initialize();import torch
d=Path(sys.argv[1]);manifest=json.loads((d/'manifest.json').read_text());reports=[]
for p,h in manifest['pins'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
with torch.inference_mode():
 for c in manifest['cases']:
  name=c['id'];read=lambda suffix:torch.frombuffer(bytearray((d/(name+'.'+suffix)).read_bytes()),dtype=torch.bfloat16)
  x=read('input.bf16').reshape(c['m'],c['k']).cuda();w=read('weight.bf16').reshape(c['n'],c['k']).cuda()
  desc=c['weight']
  with Path(desc['path']).open('rb')as f:f.seek(desc['offset']);weight=f.read(desc['bytes'])
  assert hashlib.sha256(weight).hexdigest()==c['weight_sha256']
  assert weight==(d/(name+'.weight.bf16')).read_bytes()
  y=ref.F.linear(x,w).cpu();raw=y.view(torch.uint8).numpy().tobytes()
  if c['input']['unequal']==0:assert raw==(d/(name+'.original.bf16')).read_bytes(),name
  with (d/(name+'.hf.bf16')).open('xb')as f:f.write(raw)
  old=read('actual.bf16');bad=int((y.reshape(-1).view(torch.int16)!=old.view(torch.int16)).sum());reports.append(dict(case=name,words=y.numel(),input_original_unequal=c['input']['unequal'],legacy_sameinput_unequal=bad,max_abs=float((y.reshape(-1).float()-old.float()).abs().max())))
(d/'oracle.json').write_text(json.dumps(dict(source_pins=pins,cases=reports),indent=2)+'\n');print(json.dumps(reports,indent=2))
