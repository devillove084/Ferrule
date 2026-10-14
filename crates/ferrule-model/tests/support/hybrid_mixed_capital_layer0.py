"""Capital M5 layer0-only supplemental pinned-HF diagnosis; never acceptance.

Archive comparison retains its original tolerances. All supplemental comparisons
are bit-exact; captured native tensors are operands, never expected outputs.
"""
import argparse
import json
import math
import os
from pathlib import Path
import struct
import sys
import hybrid_mixed_production_compare as archive_compare

ROOT=Path(__file__).resolve().parents[4]


def capture(directory):
    events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
    assert events and events[-1]['event']=='mixed.layer.output'
    assert all(e['layer'] in (None,0) for e in events)
    values={}
    for e in events:
        assert Path(e['file']).name==e['file']
        data=(directory/e['file']).read_bytes()
        assert len(data)==math.prod(e['shape'])*4
        assert all(math.isfinite(v) for (v,) in struct.iter_unpack('<f',data))
        key=e['event']
        if key=='norm.output':key+='.'+e['role']
        if key.startswith('gdn.') or key in ('embedding.output','norm.output.AttentionNorm','mixed.attention.partial'):
            assert key not in values
            values[key]=(data,e['shape'])
    for key,width in [('gdn.input.qkv',4096),('gdn.input.a',16),('gdn.input.b',16),('gdn.input.z',2048),('gdn.convolved',4096),('gdn.recurrent_output',2048),('gdn.gated',2048),('norm.output.AttentionNorm',2048)]:
        assert values[key][1]==[5,width],(key,values[key][1])
        assert all(v&65535==0 for (v,) in struct.iter_unpack('<I',values[key][0]))
    assert not any(values['gdn.before.ssm'][0]) and not any(values['gdn.before.conv'][0])
    return events,values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('capture','reference','model','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--execute-cuda',action='store_true')
    args=parser.parse_args()
    if not args.execute_cuda:os.environ['CUDA_VISIBLE_DEVICES']=''
    else:
        assert os.environ.get('FERRULE_PAPER_WORKER_TOKEN') and os.environ.get('CUDA_VISIBLE_DEVICES')
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert not args.output.exists()
    assert archive_compare.sha(args.reference/'manifest.json')==archive_compare.PIN
    manifest=json.loads((args.reference/'manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['id']=='capital')
    assert archive_compare.sha(args.reference/case['file'])==case['sha256']
    arch=archive_compare.Archive(args.reference/case['file'])
    captures=[capture(args.capture/f'rank-{r}') for r in range(2)]
    comparisons=[];covered=set()
    for rank,(events,_) in enumerate(captures):
        for order,e in enumerate(events):
            key=archive_compare.mapping(e,rank)
            if key:
                data=(args.capture/f'rank-{rank}'/e['file']).read_bytes()
                actual=[v for (v,) in struct.iter_unpack('<f',data)]
                comparisons.append(dict(rank=rank,order=order,event=e['event'],reference=key,**archive_compare.metrics(actual,arch.values(key))))
                covered.add(key)
            if e['event']=='router':
                for field,suffix in [('ids','selected_expert_ids'),('weights','routing_weights')]:
                    key=f'prefill.rank.{rank}.layers.00.{suffix}'
                    comparisons.append(dict(rank=rank,order=order,event=field,reference=key,**archive_compare.metrics(e[field],arch.values(key),field=='ids')))
                    covered.add(key)
    report=dict(acceptance=False,case='capital',rows=5,reference_manifest_sha256=archive_compare.PIN,
        expected_candidate_tensors=2178,distinct_compared_tensors=len(covered),archive_comparisons=comparisons,
        first_archive_inexact={str(r):next((v for v in comparisons if v['rank']==r and v['unequal']),None) for r in range(2)})
    if not args.execute_cuda:
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return
    sys.path.insert(0,str(ROOT/'papers/arxiv/experiments/tp2_mixed/oracle'))
    import tp2_oracle as tp
    source=tp.initialize()
    torch,F,hf=tp.torch,tp.F,tp.native.hf
    from safetensors import safe_open
    from safetensors.torch import save_file
    config=tp.native.read_config(args.model)
    local=tp.local_config(config,2)
    checkpoint=tp.native.legacy.Checkpoint(args.model)
    payloads=json.loads((args.reference/'source-payloads.json').read_text())
    layouts=json.loads((args.reference/'operand-layout.json').read_text())
    assert checkpoint.index_hash==payloads['index_sha256']
    expected={};results={};audit={};fixed={}
    def metric(x,y):
        x,y=[v.detach().float().cpu().contiguous().flatten() for v in (x,y)]
        assert x.shape==y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
        xb,yb=x.view(torch.int32),y.view(torch.int32);ix=(xb!=yb).nonzero().flatten()
        return dict(elements=x.numel(),unequal=ix.numel(),max_abs=float((x-y).abs().max()),
            first=[dict(index=int(i),actual_bits=f'{int(xb[i])&0xffffffff:08x}',expected_bits=f'{int(yb[i])&0xffffffff:08x}') for i in ix[:16]])
    for rank,(_,values) in enumerate(captures):
        device=f'cuda:{rank}'
        def get(name,bf16=True):
            raw,shape=values[name]
            t=torch.frombuffer(bytearray(raw),dtype=torch.float32).clone().reshape(shape)
            return (t.bfloat16() if bf16 else t).to(device)
        operands=tp.Operands(checkpoint,config,device,cache_bytes=128<<20)
        with torch.device('meta'):
            attn=hf.Qwen3_5MoeGatedDeltaNet(local,0)
            norm=hf.Qwen3_5MoeRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
        tp.bind(attn,'model.layers.0.linear_attn',operands,rank)
        tp.bind(norm,'model.layers.0.input_layernorm',operands,0)
        cache=tp.native.f32_ssm_cache(local,[device]*config.num_hidden_layers)
        stages={};handles=[]
        def record(name):
            def hook(module,args,out):stages[name]=out.detach().clone()
            return hook
        for key in ('qkv','a','b','z'):
            handles.append(getattr(attn,'in_proj_'+key).register_forward_hook(record('projection.'+key)))
        handles.append(attn.norm.register_forward_pre_hook(lambda module,args:stages.update(core=args[0].detach().clone())))
        handles.append(attn.norm.register_forward_hook(record('gated')))
        handles.append(attn.conv1d.register_forward_hook(record('pre_silu')))
        with torch.inference_mode(),safe_open(str(args.reference/case['file']),framework='pt',device='cpu') as frozen:
            embedding=frozen.get_tensor('prefill.embedding').to(device)
            normalized=norm(embedding)
            # No producer values substituted: this native branch uses archive embedding.
            partial=attn(normalized,cache_params=cache,attention_mask=torch.ones((1,5),dtype=torch.int64,device=device))
            conv=F.silu(stages['pre_silu'][:,:,:5]).transpose(1,2)
            state=cache.layers[0].recurrent_states;history=cache.layers[0].conv_states
            checks={'embedding':metric(get('embedding.output'),embedding),'normalized_input':metric(get('norm.output.AttentionNorm'),normalized)}
            for key in ('qkv','a','b','z'):checks['projection.'+key]=metric(get('gdn.input.'+key),stages['projection.'+key])
            for key,native,exp in [('conv','convolved',conv),('history','conv_state',history),('core','recurrent_output',stages['core']),('gated','gated',stages['gated']),('state','recurrent_state',state)]:
                checks[key]=metric(get('gdn.'+native,key!='state'),exp)
            for handle in handles:handle.remove()
            same_projection=attn.in_proj_qkv(get('norm.output.AttentionNorm').reshape(1,5,2048))
            checks['same_normalized_qkv']=metric(get('gdn.input.qkv'),same_projection)
            fixed[f'rank{rank}.normalized_input.bf16']=get('norm.output.AttentionNorm').cpu().contiguous().view(torch.uint8).numpy().tobytes()
            fixed[f'rank{rank}.hf_concat.qkv.f32']=same_projection.float().cpu().contiguous().numpy().tobytes()
            checks['attention_partial']=metric(get('mixed.attention.partial'),partial)
            checks['independent_attention_vs_archive']=metric(partial,frozen.get_tensor(f'prefill.rank.{rank}.layers.00.attention_partial'))
            checks['independent_state_vs_archive']=metric(state,frozen.get_tensor(f'prefill.rank.{rank}.cache.recurrent_states.00'))
            # Separately isolate downstream arithmetic with exactly Ferrule's operands.
            raw=get('gdn.input.qkv').reshape(1,5,4096)
            for h in handles:h.remove()
            aa,bb=[get('gdn.input.'+key).reshape(1,5,16) for key in ('a','b')]
            zz=get('gdn.input.z').reshape(5*16,128)
            def core_from(convolved):
                q,k,v=convolved.reshape(1,5,4096).split([1024,1024,2048],dim=-1)
                q,k=[t.reshape(1,5,8,128).repeat_interleave(2,dim=2) for t in (q,k)]
                g=-attn.A_log.float().exp()*F.softplus(aa.float()+attn.dt_bias)
                return hf.torch_chunk_gated_delta_rule(q,k,v.reshape(1,5,16,128),g,bb.sigmoid(),initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=True)
            same_conv=F.silu(attn.conv1d(raw.transpose(1,2))[:,:,:5]).transpose(1,2)
            same_core,same_state=core_from(same_conv)
            captured_conv_core,captured_conv_state=core_from(get('gdn.convolved'))
            same_gated=attn.norm(get('gdn.recurrent_output').reshape(5*16,128),zz)
            same_partial=attn.out_proj(get('gdn.gated').reshape(1,5,2048))
            checks.update(same_qkv_conv=metric(get('gdn.convolved'),same_conv),
                same_qkv_core=metric(get('gdn.recurrent_output'),same_core),same_qkv_state=metric(get('gdn.recurrent_state',False),same_state),
                same_conv_core=metric(get('gdn.recurrent_output'),captured_conv_core),same_conv_state=metric(get('gdn.recurrent_state',False),captured_conv_state),
                same_core_gated=metric(get('gdn.gated'),same_gated),same_gated_projection=metric(get('mixed.attention.partial'),same_partial))
            tensors=dict(embedding=embedding,normalized=normalized,conv=conv,history=history,core=stages['core'],gated=stages['gated'],state=state,attention_partial=partial,
                same_qkv_conv=same_conv,same_qkv_core=same_core,same_qkv_state=same_state,same_conv_core=captured_conv_core,same_conv_state=captured_conv_state,same_core_gated=same_gated,same_gated_projection=same_partial)
            tensors.update(stages)
            expected.update({f'rank{rank}.{k}':v.detach().cpu().contiguous().clone() for k,v in tensors.items()})
            results[str(rank)]=checks
            torch.cuda.synchronize(rank)
        for key,entry in operands.audit.items():
            assert json.loads(json.dumps(entry))==layouts[key],key
        audit.update(operands.audit)
    for name,entry in checkpoint.payloads.items():assert entry==payloads['used_payloads'][name],name
    report.update(supplemental=results,source_policy=source,source_audit=tp.source_audit(),operand_audit=audit,source_payloads=checkpoint.payloads,
        supplemental_policy=dict(torch=torch.__version__,cuda=torch.version.cuda,tf32=False,reduced_bf16=False,cublas_workspace=':4096:8',ssm='F32',fresh_initial_state=None))
    args.output.mkdir(parents=True,exist_ok=False)
    fixed_dir=args.output/'fixed-qkv';fixed_dir.mkdir()
    for name,data in fixed.items():(fixed_dir/name).write_bytes(data)
    report['fixed_qkv_pins']={name:archive_compare.sha(fixed_dir/name) for name in fixed}
    save_file(expected,str(args.output/'independent.safetensors'))
    report['expected_sha256']=archive_compare.sha(args.output/'independent.safetensors')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'first_archive_inexact':report['first_archive_inexact'],'supplemental':{r:{k:{f:v[f] for f in ('elements','unequal','max_abs')} for k,v in checks.items()} for r,checks in results.items()}},indent=2))

if __name__=='__main__':main()
