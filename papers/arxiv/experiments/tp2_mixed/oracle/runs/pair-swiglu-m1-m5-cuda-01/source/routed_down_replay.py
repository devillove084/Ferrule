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
    def __init__(self, manifest, directory=rc.BUNDLE):
        self.manifest = manifest
        self.directory = directory

    def read(self, name):
        entry = self.manifest['original_payloads'][name]
        path = Path(entry.get('source_directory', self.directory)) / entry['file']
        rc.require(rc.sha(path) == entry['sha256'] == entry['original_used_payload']['sha256'], 'Original payload changed')
        raw = bytearray(path.read_bytes())
        dtype = {'BF16':o.torch.bfloat16, 'F8_E4M3':o.torch.float8_e4m3fn}[entry['original_header']['dtype']]
        return o.torch.frombuffer(raw, dtype=dtype).reshape(entry['original_header']['shape']).clone()


def export(p, launch_hash):
    auth = o.authenticate_export(p, launch_hash)  # Before source/model reads or CUDA initialization.
    rc.bounds(p)
    pair = rc.is_pair(p)
    bundle_dir = rc.PAIR_BUNDLE if pair else rc.BUNDLE
    out = Path(p['directory']);write = o.native.write_json
    write(out/'numerical-authentication.json', auth)
    result = dict(schema='ferrule.paper.routed-down-fixture-release.v1',status='incomplete',model_fixture_ready=False,
                  workflow_pins=p['pins'],bundle_sha256=p['bundle_sha256'],reference_sha256=rc.REFERENCE_PIN,
                  selected_rank_rows=0,down_calls=0,expert_forward_calls=0,fixtures=[],controls=[])
    start = time.monotonic();operands = None
    original_linear = o.F.linear
    captures = {};capture_bytes = 0;pending = {};current = {};replay_calls = []
    gate_inputs = {};normalized_inputs = {};row_orders = {}
    if pair:
        result.update(schema='ferrule.paper.pair-swiglu-fixture-release.v1',gate_up_calls=0,fixture_ready=False)
    try:
        for path,pin in p['pins'].items():rc.require(rc.sha(path)==pin,'Source changed')
        m=rc.task_bundle(p);sources=o.initialize();audit=o.source_audit()
        rc.require(sources==m['native_source'],'Pinned HF sources differ')
        rc.require(not o.torch.cuda.is_initialized(),'Unexpected CUDA initialization')
        rc.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Oracle workspace policy changed')
        config=o.native.hf.Qwen3_5MoeTextConfig(**m['checkpoint']['effective_text_config'])
        config._experts_implementation=config._attn_implementation='eager'
        local=o.local_config(config,2)
        operands=o.Operands(FrozenPayloads(m,bundle_dir),config,'cuda:0',2,64*1024**2)
        rc.require(o.torch.cuda.device_count()==1,'Exactly one visible device required')
        rc.require(os.environ.get('NVIDIA_VISIBLE_DEVICES')==p['gpu_uuid'],'NVIDIA visibility differs')
        device_uuid=str(o.torch.cuda.get_device_properties(0).uuid)
        rc.require(device_uuid.lower().removeprefix('gpu-')==p['gpu_uuid'].lower().removeprefix('gpu-'),'Selected CUDA device UUID differs')
        result['selected_device']=dict(logical_device='cuda:0',uuid=device_uuid,physical_index=p.get('physical_gpu_index'),cuda_visible=os.environ['CUDA_VISIBLE_DEVICES'],nvidia_visible=os.environ['NVIDIA_VISIBLE_DEVICES'])
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
        class ObservedGate(o.ExpertBank):
            def __getitem__(self, expert):
                weight=super().__getitem__(expert)
                rc.require(not pending,"Unconsumed gate observation")
                pending.update(expert=int(expert),weight=weight,kind="gate")
                return weight
        def observe_linear(x, weight, bias=None):
            nonlocal capture_bytes
            if pair and pending.get("kind")=="gate":
                rc.require(weight is pending["weight"] and bias is None,"Unexpected gate interception")
                rc.require(x.ndim==2 and x.shape[1]==2048 and tuple(weight.shape)==(512,2048),"Gate pair geometry")
                result["gate_up_calls"]+=1
                if pending["expert"]==current["expert"]:
                    key=(current["directory"],current["rank"])
                    rc.require(key not in gate_inputs and tuple(x.shape)==(current["rows"],2048),"Target gate geometry")
                    order=[token for slot in range(8) for token,row in enumerate(current["ids"]) if row[slot]==current["expert"]]
                    rc.require(sorted(order)==list(range(current["rows"])),"Target does not cover actual tokens once")
                    actual=raw_bytes(x);normalized=raw_bytes(current["hidden"])
                    rc.require(actual==b"".join(normalized[i*4096:(i+1)*4096] for i in order),"Actual gate argument is not native normalized hidden")
                    rc.require(x.dtype==o.torch.bfloat16 and bool(o.torch.isfinite(x).all()),"Nonfinite gate input")
                    gate_inputs[key]=actual;normalized_inputs[key]=normalized;row_orders[key]=order
                    capture_bytes+=len(actual)+len(normalized)
                pending.clear()
                return original_linear(x,weight,bias)
            if pending:
                rc.require(weight is pending['weight'] and bias is None,'Unexpected down linear interception')
                expert=pending['expert']
                rc.require(x.ndim==2 and x.shape[1]==256 and 1<=x.shape[0]<=current['rows'] and tuple(weight.shape)==(2048,256),'Actual down geometry')
                value=original_linear(x,weight,bias)
                result['down_calls']+=1;replay_calls.append(expert)
                if expert==current['expert']:
                    key=(current['directory'],current['rank']);rc.require(key not in captures,'Duplicate target observation')
                    rc.require(x.dtype==value.dtype==o.torch.bfloat16 and tuple(x.shape)==(current['rows'],256) and tuple(value.shape)==(current['rows'],2048),'Native BF16 partial differs')
                    rc.require(bool(o.torch.isfinite(x).all() and o.torch.isfinite(value).all()),'Nonfinite expert observation')
                    if pair:
                        rc.require(key in gate_inputs,'Missing actual gate input')
                        if current['rows']==1:
                            for suffix,actual in (('input.bf16',x),('hf.partial.bf16',value)):
                                path=rc.OLD_CUDA/current['directory']/('rank%d.%s' % (current['rank'],suffix))
                                expected=o.torch.frombuffer(bytearray(path.read_bytes()),dtype=o.torch.bfloat16).reshape(actual.shape)
                                exact(current['directory']+'.rank%d.old_hf.%s' % (current['rank'],suffix),actual,expected)
                    captures[key]=(raw_bytes(x),raw_bytes(value));capture_bytes+=sum(map(len,captures[key]))
                    rc.require(capture_bytes<=16*1024**2,'Capture exceeds 16 MiB')
                    result['selected_rank_rows']+=1
                pending.clear()
                return value
            return original_linear(x,weight,bias)
        with o.torch.inference_mode():
            for fixture,replay in zip(m['fixtures'],m['gpu_replays']):
                name=fixture['directory'];prep=json.loads((bundle_dir/name/'preparation.json').read_text())
                layer=fixture['layer'];base=f'model.layers.{layer}.'
                rows=fixture.get('rows',1);stage=fixture['stage']
                with safe_open(bundle_dir/name/'observed.safetensors',framework='pt',device='cpu') as archive:
                    state=archive.get_tensor(replay['upstream_input_key']).to('cuda:0')
                    attention=archive.get_tensor(f'{stage}.layers.{layer:02d}.attention').to('cuda:0')
                    norm=o.native.hf.Qwen3_5MoeRMSNorm(2048,config.rms_norm_eps)
                    norm.weight=o.nn.Parameter(operands.get(base+'post_attention_layernorm.weight',0),requires_grad=False)
                    hidden=norm(state+attention).reshape(rows,2048)
                    for rank in (0,1):
                        prefix=f'{stage}.rank.{rank}.layers.{layer:02d}.'
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
                        if pair:
                            gate_bank=experts.gate_up_proj
                            experts.gate_up_proj=ObservedGate(gate_bank.prefix,gate_bank.parts,gate_bank.operands,gate_bank.rank,gate_bank.shape)
                        current.update(directory=name,rank=rank,expert=fixture['expert'],rows=rows,hidden=hidden,ids=selected[2].cpu().tolist());replay_calls.clear()
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
        rc.require(result['expert_forward_calls']==(10 if pair else 8) and result['down_calls']==(112 if pair else 64) and len(captures)==(10 if pair else 8),'Incomplete bounded replay')
        rc.require(o.source_audit()==audit and o.F.linear is original_linear,'Source/interception changed')
        rc.require(o.torch.cuda.max_memory_reserved()<=1024**3,'Device allocation bound exceeded')
        if pair:
            export_pair_files(out,p,launch_hash,m,sources,audit,captures,gate_inputs,normalized_inputs,row_orders,result)
        for fixture in ([] if pair else m['fixtures']):
            name=fixture['directory'];directory=out/name;directory.mkdir()
            prep=json.loads((bundle_dir/name/'preparation.json').read_text())
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
        rc.task_bundle(p)
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


def export_pair_files(out,p,launch_hash,m,sources,audit,captures,gate_inputs,normalized_inputs,row_orders,result):
    rc.require(len(gate_inputs)==len(normalized_inputs)==len(captures)==10,"Incomplete pair captures")
    for fixture in m["fixtures"]:
        name=fixture["directory"];rows=fixture["rows"]
        prep=json.loads((rc.PAIR_BUNDLE/name/"preparation.json").read_text())
        for rank in (0,1):
            key=(name,rank);directory=out/(name+"-rank%d" % rank);directory.mkdir()
            files={}
            def store(filename,raw,dtype,shape,source):
                rc.require(len(raw)==__import__("math").prod(shape)*(1 if dtype=="F8_E4M3" else 2),"Export extent")
                (directory/filename).write_bytes(raw)
                files[filename]=dict(bytes=len(raw),sha256=rc.sha(directory/filename),dtype=dtype,shape=shape,source=source)
            store("gateupinput.bf16",gate_inputs[key],"BF16",[rows,2048],"Actual native HF gate/up F.linear argument, not CPU normalization")
            store("normalized_hidden.bf16",normalized_inputs[key],"BF16",[rows,2048],"Actual native HF CUDA norm output in original token order")
            store("hf.activation.bf16",captures[key][0],"BF16",[rows,256],"Actual native HF pre-down F.linear argument before route weighting")
            for part in ("gate","up"):
                base="model.language_model.layers.%d.mlp.experts.%d.%s_proj.weight" % (fixture["layer"],fixture["expert"],part)
                for scale in (False,True):
                    tensor=base+("_scale_inv" if scale else "")
                    entry=m["original_payloads"][tensor]
                    source=Path(entry.get("source_directory",rc.PAIR_BUNDLE))/entry["file"]
                    rc.require(rc.sha(source)==entry["sha256"]==entry["original_used_payload"]["sha256"],"Raw pair source differs")
                    raw=source.read_bytes();shape=[2,16] if scale else [256,2048]
                    rc.require(entry["original_header"]["shape"]==([4,16] if scale else [512,2048]),"Global pair shape")
                    size=64 if scale else 524288
                    rc.require(len(raw)==2*size,"Global pair extent")
                    store(part+(".scale.bf16" if scale else ".weight.fp8"),raw[rank*size:(rank+1)*size],
                        "BF16" if scale else "F8_E4M3",shape,
                        dict(tensor=tensor,original=entry,source_byte_range=[rank*size,(rank+1)*size],
                             global_rows=[rank*(2 if scale else 256),(rank+1)*(2 if scale else 256)]))
            manifest=dict(schema="ferrule.paper.pair-swiglu.hf-cuda.v1",fixture_ready=True,
                role="RoutedExpertGateUpSwiGLU",tp=2,rank=rank,rows=rows,n=256,k=2048,packed_n=512,
                case=fixture["case"],stage=fixture["stage"],layer=fixture["layer"],expert=fixture["expert"],
                gate_input_token_indices=row_orders[key],normalized_hidden_token_indices=list(range(rows)),
                files=files,provenance=prep,native_source=sources,workflow_pins=p["pins"],versions=dict(o.native.VERSIONS),
                source_bundle_sha256=rc.PAIR_PIN,source_reference_manifest_sha256=rc.REFERENCE_PIN,
                old_m1_cuda_manifest_sha256=rc.OLD_CUDA_PIN if rows==1 else None,
                source_audit=audit,launch_sha256=launch_hash,
                numerical_authentication_sha256=rc.sha(out/"numerical-authentication.json"),
                execution=dict(device="cuda:0",physical_gpu_index=0,gpu_uuid=p["gpu_uuid"],logical_tp=2,
                    tf32=False,bf16_reduced_precision_reduction=False,cublas_workspace_config=":4096:8",
                    production_imports=False,full40_forward=False),
                acceptance="Require complete parent release; exact BF16 words against independent native HF activation, never candidate expected",
                arithmetic="Unchanged HF packed 2N linear, native SiLU BF16 result, BF16 multiply result; captured before down/route weighting")
            o.native.write_json(directory/"manifest.json",manifest)
            with (directory/"SHA256SUMS").open("x") as f:
                for filename in rc.PAIR_FILES:
                    f.write(rc.sha(directory/filename)+"  "+filename+"\n")
            result["fixtures"].append(dict(directory=directory.name,sha256=rc.sha(directory/"SHA256SUMS"),
                rows=rows,n=256,k=2048,layer=fixture["layer"],expert=fixture["expert"],rank=rank))
