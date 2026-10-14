"""CPU-only independent Torch BF16 TP2 fixture. Never imports Ferrule or opens CUDA."""
import json
import os
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

torch.set_num_threads(1)
m, n, k = 2, 128, 256
codes = [0x38, 0xB4, 0x29, 0x42, 0xBB, 0x13, 0x48, 0xAE]
x = torch.tensor([((i*29 + i//k*11) % 509 - 254)/128 for i in range(m*k)], dtype=torch.bfloat16).reshape(m,k)
w8 = torch.tensor([codes[(r*53+c*17+(r*c)%37)%8] for r in range(n) for c in range(k)], dtype=torch.uint8).view(torch.float8_e4m3fn).float().reshape(n,k)
scales = torch.tensor([0.75, 1.25], dtype=torch.bfloat16)
w = (w8 * scales.repeat_interleave(128)).to(torch.bfloat16)
parts = [x[:, i*128:(i+1)*128] @ w[:, i*128:(i+1)*128].T for i in range(2)]
reduced = parts[0] + parts[1]
deferred = sum(x[:,i*128:(i+1)*128].float() @ w[:,i*128:(i+1)*128].float().T for i in range(2)).bfloat16()
assert (reduced != deferred).any(), "fixture must distinguish projection writeback"
# Discriminates local shared+routed add from independent TP reductions.
shared = [torch.tensor([1., 1.], dtype=torch.bfloat16), torch.tensor([-1., -1.],dtype=torch.bfloat16)]
routed = [torch.tensor([0.00390625, 0.0078125],dtype=torch.bfloat16),torch.tensor([0.00390625, 0.00390625],dtype=torch.bfloat16)]
late = (shared[0]+routed[0])+(shared[1]+routed[1])
early = (shared[0]+shared[1])+(routed[0]+routed[1])
assert (late != early).any()
def words(t): return t.contiguous().view(torch.uint16).flatten().tolist()
result = dict(schema="ferrule.tp2.bf16-local.v1", oracle="independent-torch-cpu", torch=torch.__version__,
              m=m,n=n,k=k,part0=words(parts[0]),part1=words(parts[1]),reduced=words(reduced),
              deferred=words(deferred),shared=[words(s) for s in shared],routed=[words(r) for r in routed],
              late=words(late),early=words(early))
# Local EP=1 experts: gate/up N shards, down K shards of EVERY expert.
# Frozen eager weighting explicitly differs from a generic fused MoE claim.
mx = torch.tensor([[1/256]*256,[1/128]*256],dtype=torch.bfloat16)
def expert(factor,lane):
    g = torch.full((128,256),factor*(lane+1)/256,dtype=torch.bfloat16)
    u = torch.full((128,256),factor*(lane+1)/128,dtype=torch.bfloat16)
    d = torch.full((256,128),factor*(lane+1)/256,dtype=torch.bfloat16)
    gate, up = (mx @ torch.cat([g,u]).T).chunk(2,dim=-1)
    product = torch.nn.functional.silu(gate)*up
    return product @ d.T
moe = {}
for lane in range(2):
    # IDs were supplied [3,1], but eager accumulation runs in expert index order.
    routed = torch.zeros((2,256),dtype=torch.bfloat16)
    for e,weight in [(1,0.75),(3,0.25)]:
        routed += (expert(e+1,lane).float()*weight).bfloat16()
    shared = expert(1,lane)*torch.sigmoid(torch.zeros((2,1),dtype=torch.bfloat16))
    moe[str(lane)] = dict(routed=words(routed),shared=words(shared),combined=words(shared+routed))
result["moe"] = moe
Path(__file__).with_name("hybrid_rank_mixed_oracle.json").write_text(json.dumps(result,indent=2)+"\n")
print("frozen BF16 TP2 row outputs; differing deferred lanes:", int((reduced != deferred).sum()))
