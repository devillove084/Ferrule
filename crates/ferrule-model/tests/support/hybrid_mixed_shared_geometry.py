"""Pinned same-input shared-MLP geometry diagnosis from retained full capture.

No full model construction/execution. Native HF shared MLP uses separate gate/up;
HF routed experts instead use a concatenated gate_up_proj. Keep both contracts.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import hybrid_mixed_production_compare as base

ROOT=Path(__file__).resolve().parents[4]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','reference','model','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--layer',type=int,required=True)
    parser.add_argument('--execute-cuda',action='store_true')
    a=parser.parse_args();assert a.execute_cuda and os.environ.get('FERRULE_PAPER_WORKER_TOKEN')
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8' and os.environ.get('CUDA_VISIBLE_DEVICES')
    assert not a.output.exists() and 0<=a.layer<40
    assert base.sha(a.reference/'manifest.json')==base.PIN
    manifest=json.loads((a.reference/'manifest.json').read_text());case=next(c for c in manifest['cases'] if c['id']=='capital')
    assert base.sha(a.reference/case['file'])==case['sha256']
    for key in ('source_payloads','operand_layout'):
        assert base.sha(a.reference/manifest[key]['file'])==manifest[key]['sha256']
    payloads=json.loads((a.reference/'source-payloads.json').read_text());layouts=json.loads((a.reference/'operand-layout.json').read_text())
    sys.path.insert(0,str(ROOT/'papers/arxiv/experiments/tp2_mixed/oracle'))
    import tp2_oracle as tp
    source=tp.initialize();torch,F,hf=tp.torch,tp.F,tp.native.hf
    from safetensors import safe_open
    from safetensors.torch import save_file
    config=tp.native.read_config(a.model);local=tp.local_config(config,2)
    ckpt=tp.native.legacy.Checkpoint(a.model);assert ckpt.index_hash==payloads['index_sha256']
    assert not torch.cuda.is_initialized()
    results={};expected={};fixed={};audit={};capture_pins={}
    def metric(x,y):
        x,y=[v.detach().float().cpu().contiguous().flatten() for v in (x,y)];assert x.shape==y.shape
        xb,yb=x.view(torch.int32),y.view(torch.int32);ix=(xb!=yb).nonzero().flatten()
        return dict(elements=x.numel(),unequal=ix.numel(),max_abs=float((x-y).abs().max()),
            first=[dict(index=int(i),actual_bits=f'{int(xb[i])&0xffffffff:08x}',expected_bits=f'{int(yb[i])&0xffffffff:08x}') for i in ix[:16]])
    def rawbytes(v):return v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    for rank in range(2):
        directory=a.capture/f'rank-{rank}';events=[json.loads(l) for l in (directory/'events.jsonl').read_text().splitlines()]
        events=[e for e in events if e['layer']==a.layer];capture_pins[str(rank)]={'events.jsonl':base.sha(directory/'events.jsonl')}
        device=f'cuda:{rank}'
        def get(name,role=None):
            es=[e for e in events if e['event']==name and e['role']==role];assert len(es)==1,(name,role,len(es));e=es[0]
            p=directory/e['file'];data=p.read_bytes();capture_pins[str(rank)][e['file']]=base.sha(p)
            assert all(v&65535==0 for (v,) in struct.iter_unpack('<I',data))
            return torch.frombuffer(bytearray(data),dtype=torch.float32).clone().reshape(e['shape']).bfloat16().to(device)
        operands=tp.Operands(ckpt,config,device,cache_bytes=64<<20)
        prefix=f'model.layers.{a.layer}'
        with torch.device('meta'):
            norm=hf.Qwen3_5MoeRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
            mlp=hf.Qwen3_5MoeMLP(local,intermediate_size=local.shared_expert_intermediate_size)
        tp.bind(norm,prefix+'.post_attention_layernorm',operands,0)
        tp.bind(mlp,prefix+'.mlp.shared_expert',operands,rank)
        with torch.inference_mode(),safe_open(str(a.reference/case['file']),framework='pt',device='cpu') as frozen:
            before=frozen.get_tensor('prefill.embedding' if a.layer==0 else f'prefill.layers.{a.layer-1:02d}.hidden').to(device)
            attention=frozen.get_tensor(f'prefill.layers.{a.layer:02d}.attention').to(device)
            norm_input=before+attention
            normalized=norm(norm_input).reshape(5,config.hidden_size)
            x=get('mixed.pair.input','SharedExpertGate')
            w=[operands.get(prefix+'.mlp.shared_expert.'+key+'_proj.weight',rank,matrix=True) for key in ('gate','up','down')]
            gate,up=F.linear(x,w[0]),F.linear(x,w[1])
            paired=F.linear(x,torch.cat(w[:2],dim=0));pg,pu=paired.chunk(2,dim=-1)
            product=F.silu(gate)*up;pair_product=F.silu(pg)*pu
            output=mlp(x);paired_output=F.linear(pair_product,w[2])
            cg,cu=get('mixed.pair.gate','SharedExpertGate'),get('mixed.pair.up','SharedExpertUp')
            cp=get('mixed.linear.input','SharedExpertDown');cy=get('mixed.swiglu.partial','SharedExpertDown')
            checks=dict(norm_input=metric(get('norm.input','FeedForwardNorm'),norm_input),normalized_input=metric(x,normalized),
                gate_vs_separate=metric(cg,gate),up_vs_separate=metric(cu,up),gate_vs_concat=metric(cg,pg),up_vs_concat=metric(cu,pu),
                product_vs_separate=metric(cp,product),product_vs_concat=metric(cp,pair_product),
                product_same_input=metric(cp,F.silu(cg)*cu),down_same_input=metric(cy,F.linear(cp,w[2])),
                output_vs_native_mlp=metric(cy,output),output_vs_concat=metric(cy,paired_output),
                native_mlp_vs_archive=metric(output,frozen.get_tensor(f'prefill.rank.{rank}.layers.{a.layer:02d}.shared_partial')))
            for name,value in dict(input=x,gate=gate,up=up,product=product,output=output,concat_gate=pg,concat_up=pu,concat_product=pair_product,concat_output=paired_output).items():
                expected[f'rank{rank}.{name}']=value.cpu().contiguous()
                fixed[f'rank{rank}.{name}.bf16']=rawbytes(value)
            results[str(rank)]=checks
            torch.cuda.synchronize(rank)
        for key,entry in operands.audit.items():assert json.loads(json.dumps(entry))==layouts[key],key
        audit.update(operands.audit)
    # Keep actual encoded full-source tensors, not an offline decoded substitute.
    for key in ('gate','up','down'):
        name=tp.native.legacy.source_name(f'model.layers.{a.layer}.mlp.shared_expert.{key}_proj.weight')
        raw=ckpt.read(name);scale=ckpt.read(name+'_scale_inv')
        assert raw.dtype==torch.float8_e4m3fn and scale.dtype==torch.bfloat16
        fixed[key+'.fp8']=rawbytes(raw);fixed[key+'.scales.bf16']=rawbytes(scale)
    for name,entry in ckpt.payloads.items():assert entry==payloads['used_payloads'][name],name
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'fixture').mkdir()
    for name,data in fixed.items():(a.output/'fixture'/name).write_bytes(data)
    save_file(expected,str(a.output/'independent.safetensors'))
    report=dict(schema='ferrule.mixed-shared-geometry.v1',acceptance=False,layer=a.layer,rows=5,results=results,
        source_policy=source,interception_audit=tp.source_audit(),operand_audit=audit,used_source_payloads=ckpt.payloads,capture_pins=capture_pins,
        reference_manifest_sha256=base.PIN,expected_sha256=base.sha(a.output/'independent.safetensors'),
        fixture_pins={name:base.sha(a.output/'fixture'/name) for name in fixed},
        arithmetic='unchanged native HF shared MLP separate gate/up; concat control with identical operands; BF16 producer/output; F32 accumulator')
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
