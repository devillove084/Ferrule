"""Profile only the pinned gate/QKV F.linear calls under an authorized GPU supervisor.

Does not load a model or regenerate a reference. Uses the pinned producer's
Torch/HF initialization (CUDA 12.9), distinct from Ferrule's CUDA 12.4 build.
"""
import argparse
import os,sys,json,hashlib,struct
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--execute-cuda', action='store_true')
parser.add_argument('--sole-owner-permission', action='store_true')
args = parser.parse_args()
if not (args.execute_cuda and args.sole_owner_permission):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser.error('requires explicit CUDA and sole-owner permission; invoke under supervisor')
ROOT = Path(__file__).resolve().parents[4]
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0,str(ROOT/'scripts'))
import qwen35_fp8_bf16_reference as ref
pins=ref.initialize();import torch
assert torch.cuda.get_device_capability()==(8,6)
def load(p,dtype):return torch.frombuffer(bytearray(p.read_bytes()),dtype=dtype).clone()
def dump(name,t):
 b=t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
 with (OUT/name).open('xb') as f:f.write(b)
 return hashlib.sha256(b).hexdigest()
def diffs(a,b):
 aa=a.cpu().bfloat16().view(torch.int16).flatten();bb=b.cpu().bfloat16().view(torch.int16).flatten()
 return [dict(index=i,actual=f'0x{int(aa[i])&65535:04x}',expected=f'0x{int(bb[i])&65535:04x}') for i in (aa!=bb).nonzero().flatten().tolist()]
result=dict(torch=torch.__version__,cuda=torch.version.cuda,source=pins,projections={})
for name,dir,n,pin in [('gate',ROOT/'target/validation/bf16-tiled-model-20261005/gate-fixture',4096,'c6a90487a76ab30d5babb4e0154886c44024969d372a60833a3ff9c9a10190c5'),('qkv',ROOT/'target/validation/bf16-tiled-compat-20261004/qkv-fixture',8192,'a26fb38acfa8f3d05a8bc8923d99b51b9e0cedf535f8c5a456e3f0b9309a8936')]:
 assert hashlib.sha256((dir/'SHA256SUMS').read_bytes()).hexdigest()==pin
 for line in (dir/'SHA256SUMS').read_text().splitlines():
  h,f=line.split();assert hashlib.sha256((dir/f).read_bytes()).hexdigest()==h
 x=load(dir/'input.bf16',torch.bfloat16).reshape(1,2048)
 w=load(dir/'weight.fp8',torch.float8_e4m3fn).reshape(n,2048)
 s=load(dir/'scale.bf16',torch.bfloat16).reshape(n//128,16)
 w=(w.float()*s.float().repeat_interleave(128,0).repeat_interleave(128,1)).bfloat16()
 hf=load(dir/'hf.bf16',torch.bfloat16).reshape(1,n)
 decoded_sha=dump(name+'-decoded.bf16',w)
 x=x.cuda();w=w.cuda()
 with torch.inference_mode():
  for _ in range(3): torch.nn.functional.linear(x,w)
  torch.cuda.synchronize()
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   y=torch.nn.functional.linear(x,w)
  torch.cuda.synchronize();prof.export_chrome_trace(str(OUT/(name+'-hf-trace.json')))
  kernels=sorted(set(e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA))
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof32:
   raw=torch.mm(x,w.T,out_dtype=torch.float32)
  torch.cuda.synchronize();prof32.export_chrome_trace(str(OUT/(name+'-hf-f32-trace.json')))
  kernels32=sorted(set(e.name for e in prof32.events() if e.device_type==torch.autograd.DeviceType.CUDA))
  r=dict(decoded_sha256=decoded_sha,kernels=kernels,f32_kernels=kernels32,hf_differences=diffs(y,hf),f32_output_rounded_differences=diffs(raw,hf),output_sha256=dump(name+'-native.bf16',y),f32_sha256=dump(name+'-native.f32',raw),tiles={})
  for t in [64,257,1024]:
   yy=torch.cat([torch.nn.functional.linear(x,w[first:first+t]) for first in range(0,n,t)],1)
   r['tiles'][str(t)]=diffs(yy,hf)
  result['projections'][name]=r
  assert not r['hf_differences'],r
 torch.cuda.synchronize();del x,w,y,raw
with (OUT/'hf-probe.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
