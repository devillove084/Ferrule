"""CPU extraction of independently captured layer2..4 projections; no expected synthesis."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
spec=importlib.util.spec_from_file_location('util',ROOT/'crates/ferrule-backend/tests/numeric_bf16_tiled/extract_qkv.py')
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
SUPPLEMENT=ROOT/'target/validation/bf16-gqa-boundary-20261005/supplement'
PIN='25296825802d6a76bb0d160d099fa02526b3bad560700010f9dac9f2f93aa5cd'
ROLES={'linear_attn.in_proj_qkv':'LinearAttentionQkv','linear_attn.in_proj_z':'LinearAttentionZ','linear_attn.out_proj':'AttentionOutput','self_attn.q_proj':'AttentionQuery','self_attn.k_proj':'AttentionKey','self_attn.v_proj':'AttentionValue','self_attn.o_proj':'AttentionOutput','mlp.shared_expert.gate_proj':'SharedExpertGate','mlp.shared_expert.up_proj':'SharedExpertUp','mlp.shared_expert.down_proj':'SharedExpertDown'}

def main(out):
 assert u.sha256(SUPPLEMENT/'manifest.json')==PIN
 src=json.loads((SUPPLEMENT/'manifest.json').read_text())
 assert src['layers']==[2,3,4] and not src['production_inputs_used']
 assert src['versions']=={'torch':'2.13.0+cu129','transformers':'5.5.3'}
 assert src['tf32'] is False and src['bf16_reduced_precision_reduction'] is False
 assert all(c['unequal']==0 for c in src['checks'])
 hf=SUPPLEMENT/'layers234.safetensors';assert u.sha256(hf)==src['capture_sha256']
 assert u.sha256(SUPPLEMENT/'source-payloads.json')==src['payloads_sha256']
 out.mkdir(parents=True,exist_ok=False);hashes={};cases=[]
 def put(name,data):
  with (out/name).open('xb') as f:f.write(data)
  hashes[name]=hashlib.sha256(data).hexdigest()
 headers,_=u.header(hf)
 for item in src['linears']:
  if 'scale' not in item:continue
  layer=item['layer'];role=ROLES[item['module']];n,k=item['weight']['shape'];identity=f'l{layer}_{role}'
  for kind,dtype,suffix in [('weight','F8_E4M3','weight.fp8'),('scale','BF16','scale.bf16')]:
   t=item[kind];data,offset=u.tensor(Path(t['path']),t['name'],dtype,t['shape'])
   assert offset==t['offset'] and len(data)==t['bytes'] and hashlib.sha256(data).hexdigest()==t['sha256']
   put(identity+'.'+suffix,data)
  for phase,suffix in [('input','input.bf16'),('output','hf.bf16')]:
   key=f'{layer}.{item["module"]}.{phase}';entry=headers[key]
   data,_=u.tensor(hf,key,'BF16',entry['shape']);assert len(data)==2*(k if phase=='input' else n)
   put(identity+'.'+suffix,data)
  cases.append(dict(id=identity,layer=layer,role=role,n=n,k=k,production_sameinput=False))
 assert len(cases)==19 and len({x['id'] for x in cases})==19
 manifest=dict(schema='ferrule.frozen-bf16-layers234.v1',oracle='independent-pinned-hf-bf16',hf_sha256=src['capture_sha256'],profile_sha256=src['original_manifest_sha256'],torch=src['versions']['torch'],transformers=src['versions']['transformers'],model_boundary='bf16',accumulation='f32',allow_tf32=False,allow_bf16_reduced_precision_reduction=False,numeric_count=91,missing_expert_pairs=72,cases=cases)
 put('manifest.json',(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode())
 put('audit.json',(json.dumps(dict(supplement_manifest_sha256=PIN,source=src,linears=src['linears'],production_sameinput='not observed in full-model trace; local test consumes HF inputs directly',missing='72 routed per-expert intermediate pairs unavailable'),sort_keys=True,indent=2)+'\n').encode())
 (out/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in sorted(hashes.items())))
 print(json.dumps(dict(cases=len(cases),hf_sha256=src['capture_sha256'],fixture_sha256=u.sha256(out/'SHA256SUMS'))))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
