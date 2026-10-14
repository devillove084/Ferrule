"""Authenticated local HF subpath replay; no full-model forward or candidate import."""
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import routed_down_contract as rc
import tp2_oracle as o
from safetensors import safe_open
from safetensors.torch import save_file


def raw_bytes(value):
    return value.detach().cpu().contiguous().view(o.torch.uint8).numpy().tobytes()


class FrozenPayloads:
    def __init__(self, manifest):
        self.manifest = manifest

    def read(self, name):
        entry = self.manifest['original_payloads'][name]
        path = rc.BUNDLE / entry['file']
        rc.require(rc.sha(path) == entry['sha256'] == entry['original_used_payload']['sha256'], 'Original payload changed')
        raw = bytearray(path.read_bytes())
        dtype = {'BF16':o.torch.bfloat16, 'F8_E4M3':o.torch.float8_e4m3fn}[entry['original_header']['dtype']]
        return o.torch.frombuffer(raw, dtype=dtype).reshape(entry['original_header']['shape']).clone()


def export(p, launch_hash):
    auth = o.authenticate_export(p, launch_hash)  # Before source/model reads or CUDA initialization.
    rc.bounds(p)
    out = Path(p['directory']);write = o.native.write_json
    write(out/'numerical-authentication.json', auth)
    result = dict(schema='ferrule.paper.routed-down-fixture-release.v1',status='incomplete',model_fixture_ready=False,
                  workflow_pins=p['pins'],bundle_sha256=rc.BUNDLE_PIN,reference_sha256=rc.REFERENCE_PIN,
                  selected_rank_rows=0,down_calls=0,expert_forward_calls=0,fixtures=[],controls=[])
    start = time.monotonic();operands = None
    original_linear = o.F.linear
    captures = {};capture_bytes = 0;pending = {};current = {};replay_calls = []
    try:
        for path,pin in p['pins'].items():rc.require(rc.sha(path)==pin,'Source changed')
        m=rc.bundle();sources=o.initialize();audit=o.source_audit()
        rc.require(not o.torch.cuda.is_initialized(),'Unexpected CUDA initialization')
        rc.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Oracle workspace policy changed')
        config=o.native.hf.Qwen3_5MoeTextConfig(**m['checkpoint']['effective_text_config'])
        config._experts_implementation=config._attn_implementation='eager'
        local=o.local_config(config,2)
        operands=o.Operands(FrozenPayloads(m),config,'cuda:0',2,64*1024**2)
        rc.require(o.torch.cuda.device_count()==1,'Exactly one visible device required')
        o.torch.cuda.reset_peak_memory_stats()
        def exact(label, actual, expected):
            av=actual.detach().cpu().contiguous();ev=expected.cpu().contiguous()
            rc.require(av.shape==ev.shape and av.dtype==ev.dtype,'Control geometry: '+label)
            a=av.view(o.torch.uint8);e=ev.view(o.torch.uint8)
            ok=o.torch.equal(a,e)
            row=dict(label=label,exact=ok,shape=list(av.shape),dtype=str(av.dtype),actual_sha256=o.digest_tensor(av),expected_sha256=o.digest_tensor(ev))
            result['controls'].append(row)
            if not ok:
                row.update(nonexact_elements=int((av!=ev).sum()),max_abs=float((av.float()-ev.float()).abs().max()))
                save_file({'actual':av,'expected':ev},str(out/'failed-control.safetensors'))
                raise ValueError('Exact immutable upstream control failed: '+json.dumps(row))
        class ObservedDown(o.ExpertBank):
            def __getitem__(self, expert):
                weight=super().__getitem__(expert)
                rc.require(not pending,'Unconsumed down observation')
                pending.update(expert=int(expert),weight=weight)
                return weight
        def observe_linear(x, weight, bias=None):
            nonlocal capture_bytes
            if pending:
                rc.require(weight is pending['weight'] and bias is None,'Unexpected down linear interception')
                expert=pending['expert']
                rc.require(tuple(x.shape)==(1,256) and tuple(weight.shape)==(2048,256),'Actual down M1 geometry')
                value=original_linear(x,weight,bias)
                result['down_calls']+=1;replay_calls.append(expert)
                if expert==current['expert']:
                    key=(current['directory'],current['rank']);rc.require(key not in captures,'Duplicate target observation')
                    rc.require(x.dtype==value.dtype==o.torch.bfloat16 and tuple(value.shape)==(1,2048),'Native BF16 partial differs')
                    rc.require(bool(o.torch.isfinite(x).all() and o.torch.isfinite(value).all()),'Nonfinite expert observation')
                    captures[key]=(raw_bytes(x),raw_bytes(value));capture_bytes+=sum(map(len,captures[key]))
                    rc.require(capture_bytes<=16*1024**2,'Capture exceeds 16 MiB')
                    result['selected_rank_rows']+=1
                pending.clear()
                return value
            return original_linear(x,weight,bias)
        with o.torch.inference_mode():
            for fixture,replay in zip(m['fixtures'],m['gpu_replays']):
                name=fixture['directory'];prep=json.loads((rc.BUNDLE/name/'preparation.json').read_text())
                layer=fixture['layer'];base=f'model.layers.{layer}.'
                with safe_open(rc.BUNDLE/name/'observed.safetensors',framework='pt',device='cpu') as archive:
                    state=archive.get_tensor(replay['upstream_input_key']).to('cuda:0')
                    attention=archive.get_tensor(f'decode.0.layers.{layer:02d}.attention').to('cuda:0')
                    norm=o.native.hf.Qwen3_5MoeRMSNorm(2048,config.rms_norm_eps)
                    norm.weight=o.nn.Parameter(operands.get(base+'post_attention_layernorm.weight',0),requires_grad=False)
                    hidden=norm(state+attention).reshape(1,2048)
                    for rank in (0,1):
                        prefix=f'decode.0.rank.{rank}.layers.{layer:02d}.'
                        router=o.native.hf.Qwen3_5MoeTopKRouter(local)
                        router.weight=o.nn.Parameter(operands.get(base+'mlp.gate.weight',rank),requires_grad=False)
                        native=router(hidden)
                        raw,selected=o.native.router_observation(router,(hidden,),native,o.native.ROUTER_PROFILE)
                        for field,value in [('raw_router_logits',raw),('router_probabilities',selected[0]),('routing_weights',selected[1]),('selected_expert_ids',selected[2])]:
                            exact(name+f'.rank{rank}.'+field,value,archive.get_tensor(prefix+field))
                        with o.torch.device('meta'):experts=o.native.hf.Qwen3_5MoeExperts(local)
                        experts=o.bind(experts,base+'mlp.experts',operands,rank)
                        bank=experts.down_proj
                        experts.down_proj=ObservedDown(bank.prefix,bank.parts,bank.operands,bank.rank,bank.shape)
                        current.update(directory=name,rank=rank,expert=fixture['expert']);replay_calls.clear()
                        o.F.linear=observe_linear
                        try:aggregate=experts(hidden,selected[2],selected[1])
                        finally:o.F.linear=original_linear
                        rc.require(not pending and replay_calls==sorted(prep['selected_expert_ids']),'Expert call count/order changed')
                        result['expert_forward_calls']+=1
                        exact(name+f'.rank{rank}.routed_partial',aggregate,archive.get_tensor(prefix+'routed_partial'))
                        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(fixture=name,rank=rank,exact_controls=5,down_calls=list(replay_calls)))+'\n')
                        del experts,router,aggregate
                    del norm,hidden,state,attention
        o.torch.cuda.synchronize()
        rc.require(result['expert_forward_calls']==8 and result['down_calls']==64 and len(captures)==8,'Incomplete bounded replay')
        rc.require(o.source_audit()==audit and o.F.linear is original_linear,'Source/interception changed')
        rc.require(o.torch.cuda.max_memory_reserved()<=1024**3,'Device allocation bound exceeded')
        for fixture in m['fixtures']:
            name=fixture['directory'];directory=out/name;directory.mkdir()
            prep=json.loads((rc.BUNDLE/name/'preparation.json').read_text())
            for rank in (0,1):
                x,y=captures[(name,rank)]
                (directory/f'rank{rank}.input.bf16').write_bytes(x)
                (directory/f'rank{rank}.hf.partial.bf16').write_bytes(y)
                for field in ('weight.fp8','scale.bf16'):shutil.copyfile(rc.BUNDLE/name/f'rank{rank}.{field}',directory/f'rank{rank}.{field}')
            manifest=dict(schema='ferrule.mixed-routed-down.tp2.v1',role='RoutedExpertDown',tp=2,rows=1,oracle='hf-eager-bf16-rank-partial',
                n=2048,local_k=256,layer=fixture['layer'],expert=fixture['expert'],checkpoint_sha256=rc.REFERENCE_PIN,
                checkpoint_shard_set_sha256=m['checkpoint']['full_shard_set_sha256'],provenance=prep,
                original_down_payloads={k:m['original_payloads'][prep[k]] for k in ('raw_weight_source','raw_scale_source')},
                native_source=sources,workflow_pins=p['pins'],versions=dict(o.native.VERSIONS),
                source_bundle_sha256=rc.BUNDLE_PIN,launch_sha256=launch_hash,numerical_authentication_sha256=rc.sha(out/'numerical-authentication.json'),
                extraction='Actual unmodified native HF Experts.forward down F.linear input/output before route weighting; transparent observation only',
                control='Same CUDA BF16 norm/router and eight-expert weighted aggregate match immutable upstream oracle word-for-word on both logical ranks',
                execution=dict(device='cuda:0',gpu_uuid=p['gpu_uuid'],logical_tp=2,tf32=False,bf16_reduced_precision_reduction=False,cublas_workspace_config=':4096:8',production_imports=False,full40_forward=False),
                acceptance='Exact BF16 words; no tolerance or candidate data used',source_audit=audit)
            write(directory/'manifest.json',manifest)
            with (directory/'SHA256SUMS').open('x') as f:
                for filename in rc.FILES:f.write(rc.sha(directory/filename)+'  '+filename+'\n')
            result['fixtures'].append(dict(directory=name,sha256=rc.sha(directory/'SHA256SUMS'),layer=fixture['layer'],expert=fixture['expert']))
        rc.bundle()
        for path,pin in p['pins'].items():rc.require(rc.sha(path)==pin,'Source changed after replay')
        result.update(status='pending_custody',capture_bytes=capture_bytes,native_source=sources,versions=dict(o.native.VERSIONS),
                      interception_audit=audit,peak_allocated_bytes=o.torch.cuda.max_memory_allocated(),peak_reserved_bytes=o.torch.cuda.max_memory_reserved(),
                      numerical_authentication_sha256=rc.sha(out/'numerical-authentication.json'),
                      claim='Local independent HF CUDA replay only; native/W8 production operator gate not run')
    except BaseException as e:
        result.update(status='incomplete',error=f'{type(e).__name__}: {e}');raise
    finally:
        o.F.linear=original_linear
        result['wall_seconds']=time.monotonic()-start
        if operands is not None:operands.decoded.clear()
        gc.collect()
        if o.torch.cuda.is_initialized():o.torch.cuda.empty_cache()
        write(out/'export.json',result)
