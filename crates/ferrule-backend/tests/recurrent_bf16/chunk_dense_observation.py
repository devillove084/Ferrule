"""Locate the dense multi-chunk exact-parity gap on already-prepared operands.

Internal supervised CUDA child only; no replacement reference or acceptance
threshold. Compares original HF locals with native stage files from chunk.cu.
"""
import inspect
import json
from pathlib import Path
import sys
from performance import ROOT, sha, write_json


def observe(directory, evidence):
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'scripts'))
    import qwen35_fp8_bf16_reference as ref
    pins=ref.initialize()
    fn=ref.hf.torch_chunk_gated_delta_rule
    lines,start=inspect.getsourcelines(fn)
    locations={start+i:line.strip() for i,line in enumerate(lines)}
    stages={}
    def load(name,dtype=torch.float32):
        return torch.frombuffer(bytearray((directory/name).read_bytes()),dtype=dtype).clone()
    def capture(frame,event,arg):
        if frame.f_code is fn.__code__ and event=='line':
            line,loc=locations[frame.f_lineno],frame.f_locals
            pairs=[]
            if line.startswith('v_beta ='):pairs=[('qscaled','query')]
            elif line.startswith('g = g.reshape'):pairs=[('k_beta','k_beta'),('v_beta','v_beta')]
            elif line.startswith('decay_mask ='):pairs=[('g_prefix','g')]
            elif line.startswith('attn = -'):pairs=[('decay_mask','decay_mask')]
            elif line.startswith('for i in range(1, chunk_size)') and 'triangular_before' not in stages:pairs=[('triangular_before','attn')]
            elif line.startswith('value = attn @'):pairs=[('triangular_after','attn')]
            elif line.startswith('k_cumdecay ='):pairs=[('matmul1','value')]
            elif line.startswith('last_recurrent_state = (') and 'k_cumdecay' in loc and 'core_attn_out' not in loc:pairs=[('matmul2','k_cumdecay')]
            elif line.startswith('v_prime ='):pairs=[('chunk'+str(loc['i'])+'-qk_attn','attn')]
            elif line.startswith('attn_inter ='):pairs=[('chunk'+str(loc['i'])+'-v_new','v_new'),('chunk'+str(loc['i'])+'-matmul4','v_prime')]
            elif line.startswith('core_attn_out[:, :, i] ='):pairs=[('chunk'+str(loc['i'])+'-matmul5','attn_inter')]
            for name,key in pairs:
                assert name not in stages
                stages[name]=loc[key].clone()
        return capture
    with torch.inference_mode():
        qkv=load('qkv.bf16',torch.bfloat16).reshape(1,65,50).cuda()
        q,k,v=qkv.split([16,16,18],dim=-1)
        q=q.reshape(1,65,1,16).repeat_interleave(2,dim=2)
        k=k.reshape(1,65,1,16).repeat_interleave(2,dim=2)
        v=v.reshape(1,65,2,9)
        get=lambda name:load(name+'.bf16',torch.bfloat16).cuda()
        g=-get('a_log').float().exp()*ref.F.softplus(get('a').reshape(1,65,2).float()+get('dt_bias'))
        beta=get('b').reshape(1,65,2).sigmoid()
        assert sys.gettrace() is None
        sys.settrace(capture)
        try:output,state=fn(q,k,v,g,beta,initial_state=load('state.f32').reshape(1,2,16,9).cuda(),output_final_state=True,use_qk_l2norm_in_kernel=True)
        finally:sys.settrace(None)
    assert torch.equal(output.cpu().float().flatten().view(torch.int32),load('hf-output.f32').view(torch.int32))
    assert torch.equal(state.cpu().flatten().view(torch.int32),load('hf-state.f32').view(torch.int32))
    stages['g']=ref.F.pad(g.transpose(1,2),(0,63)).reshape(1,2,2,64)
    stages['beta']=ref.F.pad(beta.transpose(1,2),(0,63)).reshape(1,2,2,64)
    stages['qscaled']=stages['qscaled'].reshape(1,2,2,64,16)
    result={}
    for name,value in stages.items():
        values=[(name,value)] if name.startswith('chunk') else [('chunk'+str(i)+'-'+name,value[:,:,i]) for i in range(2)]
        for key,value in values:
            a=load(key+'.f32');b=value.cpu().float().flatten();assert a.shape==b.shape
            indices=(a.view(torch.int32)!=b.view(torch.int32)).nonzero().flatten()
            result[key]=dict(elements=a.numel(),unequal=indices.numel(),max_abs=float((a-b).abs().max()),first_indices=indices[:8].tolist())
    write_json(evidence/'dense-stage-comparison.json',dict(pins=pins,script_sha256=sha(__file__),native_fixture=str(directory),
        original_synthetic_hf_output_state_unchanged=True,comparisons=result))
    print(json.dumps(result,indent=2))


if __name__=='__main__':observe(Path(sys.argv[1]).resolve(),Path(sys.argv[2]).resolve())
