"""Authenticated pre-weight CUDA checks; frozen independent fixtures stay read-only."""
import ast
import copy
import json
import torch
from torch import nn
import torch.nn.functional as F
import tp2_oracle as o

SUPPORT=o.ROOT/'crates/ferrule-model/tests/support'
PINS={
 'hybrid_rank_mixed_oracle.json':'5a671d220bd4cecfee92a1614f637c0d59bb5feeff543240fd377bdb81a051af',
 'hybrid_mixed_gdn_oracle.py':'39ed35183ffe3693d4988b25fc17daf2176e44ecca06bdfc081f26bdf8354fc9',
 'hybrid_mixed_gdn_oracle.json':'98884445fb90c34747980f2c6a30161532b7a7ae59f7fe7cd218b8a3577cf2b7',
 'hybrid_mixed_gdn_cuda_oracle.json':'2baf60d8663ca330385d1fe3848e241a4c273706317396b7167b3b8f627c2a6c',
 'hybrid_mixed_moe_cuda_oracle.py':'feeb06173ba2c353a38a5a6cef09bac785f2a5ed509f754e2bdf0b2ea10250b4',
 'hybrid_mixed_moe_cuda_oracle.json':'5825d2005f4a127dfa2458898631aef04a2eed7432933b1e5f7e7a2beaeefe38',
 'hybrid_mixed_boundary_oracle.json':'6a9a7b2906783d3a5fdc72f8d65acff326e71ca1f2819b2e71fe9412cdb2a91b',
 '../fixtures/hybrid_cpu/oracle.json':'fd03595dbabb39c67401f43bda6b055ba5d1ec9b34623b2198a57e2004dc65a5'}


def words(v,dtype=torch.bfloat16):
    return torch.tensor(v,dtype=torch.uint16 if dtype==torch.bfloat16 else torch.int32).view(dtype)


class Store:
    def __init__(self, values):
        self.values={o.native.legacy.source_name(k):v.detach().cpu().bfloat16().clone() for k,v in values.items()}
    def read(self,name):
        o.require(name in self.values,'Missing synthetic operand: '+name)
        return self.values[name].clone()


def model_store(model):
    values={}
    for name,value in model.state_dict().items():
        if name.endswith('.experts.gate_up_proj'):
            for e,w in enumerate(value):
                for part,half in zip(('gate_proj','up_proj'),w.chunk(2,0)):
                    values[name.removesuffix('gate_up_proj')+f'{e}.{part}.weight']=half
        elif name.endswith('.experts.down_proj'):
            for e,w in enumerate(value):values[name.removesuffix('down_proj')+f'{e}.down_proj.weight']=w
        else:values[name]=value
    return Store(values)


def run(output):
    report=dict(schema='ferrule.paper.tp2-oracle.cuda-small-gate.v1',status='incomplete',metrics={},
        fixtures=PINS,device='cuda:0',production_imports=False,
        limits='small synthetic weights only; runs before full checkpoint reads',
        unresolved_sites=json.loads((o.HERE/'protocol.json').read_text())['unvalidated_math_sites'])
    def check(name,actual,expected,exact=False):
        actual=actual.detach().cpu();expected=expected.detach().cpu().reshape(actual.shape)
        m=o.tensor_metric(actual,expected,exact)
        report['metrics'][name]=m
        o.require(m['failures']==0,'CUDA small gate: '+name+': '+str(m))
    try:
        for name,pin in PINS.items():o.require(o.sha256(SUPPORT/name)==pin,'Changed independent fixture: '+name)
        o.require(torch.cuda.device_count()==1,'One granted visible GPU required')
        with torch.inference_mode():
            row=json.loads((SUPPORT/'hybrid_rank_mixed_oracle.json').read_text())
            m,n,k=row['m'],row['n'],row['k']
            x=torch.tensor([((i*29+i//k*11)%509-254)/128 for i in range(m*k)],dtype=torch.bfloat16,device='cuda').reshape(m,k)
            codes=[0x38,0xB4,0x29,0x42,0xBB,0x13,0x48,0xAE]
            raw=torch.tensor([codes[(r*53+c*17+(r*c)%37)%8] for r in range(n) for c in range(k)],dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(n,k)
            w=o.native.dequantize_bf16(raw.cuda(),torch.tensor([[.75,1.25]],dtype=torch.bfloat16,device='cuda'))
            parts=[F.linear(x[:,r*128:(r+1)*128],w[:,r*128:(r+1)*128]) for r in range(2)]
            for r in range(2):check(f'fp8.row.rank{r}',parts[r],words(row[f'part{r}']),True)
            check('fp8.row.reduce',o.rank_sum(parts),words(row['reduced']),True)
            shared=[words(v).cuda() for v in row['shared']];routed=[words(v).cuda() for v in row['routed']]
            check('local_shared_route_then_reduce',o.rank_sum([a+b for a,b in zip(shared,routed)]),words(row['late']),True)
            o.require(not torch.equal(o.rank_sum(shared)+o.rank_sum(routed),words(row['late']).cuda()),'No late-add witness')
            tp1_tp2(check)
            fixture_moe_head(check)
            fixture_gdn(check,report)
        torch.cuda.synchronize()
        report.update(status='pass',checks=len(report['metrics']),
                      scope='independent fixture/CUDA small gates only; no Ferrule/full40 or performance PASS')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        raise
    finally:o.native.write_json(output/'cuda-small-gate.json',report)
    return report


def tp1_tp2(check):
    torch.manual_seed(3501635)
    c=o.native.tiny_config();c.num_attention_heads=4;c.num_key_value_heads=2
    original=o.native.hf.Qwen3_5MoeForCausalLM(c).eval()
    store=model_store(original)
    for p in original.parameters():p.data=p.data.to(device='cuda',dtype=torch.bfloat16)
    original.model.rotary_emb=o.native.hf.Qwen3_5MoeTextRotaryEmbedding(c,device='cuda:0')
    handles=[layer.mlp.gate.register_forward_hook(lambda m,a,out:o.native.router_observation(m,a,out,o.native.ROUTER_PROFILE)[1]) for layer in original.model.layers]
    try:
        for world in (1,2):
            capture=o.Capture(device='cuda:0');operands=o.Operands(store,c,'cuda:0',world=world)
            model=o.build_model(c,operands,capture)
            try:
                for ids in ([1],[1,3,2,4,5]):
                    o.run_case(model,c,operands,capture,ids,decode_token=5)
                    if world==1:
                        cache=o.native.f32_ssm_cache(c,['cuda:0']*2)
                        for stage,fed,start in [('prefill',ids,0),('decode.0',[5],len(ids))]:
                            logits=original(torch.tensor([fed],device='cuda'),position_ids=torch.arange(start,start+len(fed),device='cuda')[None],
                                attention_mask=torch.ones(1,start+len(fed),dtype=torch.int64,device='cuda'),past_key_values=cache,use_cache=True).logits
                            check(f'tp1.native.{len(ids)}.{stage}.logits',capture.tensors[stage+'.logits'],logits,True)
                            for i,layer in enumerate(cache.layers):
                                for field,key in [('conv_states','conv_states'),('recurrent_states','recurrent_states'),('keys','key_cache'),('values','value_cache')]:
                                    v=getattr(layer,field,None)
                                    if v is not None:check(f'tp1.native.{len(ids)}.{stage}.{key}.{i}',capture.tensors[f'{stage}.cache.{key}.{i:02d}'],v,True)
                    else:
                        for stage in ('prefill','decode.0'):
                            for i in range(2):
                                for key,partial in [('attention','attention_partial'),('moe','moe_partial')]:
                                    parts=[capture.tensors[f'{stage}.rank.{r}.layers.{i:02d}.{partial}'] for r in range(2)]
                                    check(f'tp2.{len(ids)}.{stage}.{key}.{i}',capture.tensors[f'{stage}.layers.{i:02d}.{key}'],o.rank_sum(parts),True)
                            check(f'tp2.{len(ids)}.{stage}.state',capture.tensors[stage+'.cache.recurrent_states.00'],torch.cat([capture.tensors[f'{stage}.rank.{r}.cache.recurrent_states.00'] for r in range(2)],1),True)
            finally:capture.close();operands.decoded.clear()
    finally:
        for h in handles:h.remove()


def fixture_moe_head(check):
    raw=json.loads((SUPPORT/'../fixtures/hybrid_cpu/oracle.json').read_text())['tensors']
    frozen=json.loads((SUPPORT/'hybrid_mixed_moe_cuda_oracle.json').read_text())
    prefix=json.loads((SUPPORT/'hybrid_mixed_gdn_oracle.json').read_text())
    boundary=json.loads((SUPPORT/'hybrid_mixed_boundary_oracle.json').read_text())
    o.require(frozen['prefix_sha256']==PINS['hybrid_mixed_gdn_oracle.json'] and boundary['prefix_sha256']==PINS['hybrid_mixed_moe_cuda_oracle.json'],'Fixture chain differs')
    # Only audited function definitions; never execute fixture imports, main,
    # top-level generation, default-device changes, or file writers.
    path=SUPPORT/'hybrid_mixed_moe_cuda_oracle.py'
    tree=ast.parse(path.read_text(),filename=str(path));tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    moe={k.removeprefix('layers.3.feed_forward.').removesuffix('.weight'):v for k,v in frozen['tensors'].items()}
    env=dict(torch=torch,F=F,tensors=raw,prefix=prefix,moe=moe)
    exec(compile(tree,str(path),'exec'),env)
    values={}
    for key,entry in moe.items():
        if key=='router':target='gate'
        elif key=='shared_output_gate':target='shared_expert_gate'
        else:
            target=key.replace('shared.','shared_expert.')
            stem,part=target.rsplit('.',1);target=stem+'.'+{'gate':'gate_proj','up':'up_proj','down':'down_proj'}[part]
        values['model.layers.0.mlp.'+target+'.weight']=torch.tensor(entry['values']).reshape(entry['shape'])
    c=o.native.tiny_config();c.hidden_size=8;c.num_key_value_heads=2
    c.moe_intermediate_size=c.shared_expert_intermediate_size=12;c.num_experts=16;c.num_experts_per_tok=8
    local=o.local_config(c,2);operands=o.Operands(Store(values),c,'cuda:0')
    with torch.device('meta'):branches=[o.native.hf.Qwen3_5MoeSparseMoeBlock(local) for _ in range(2)]
    capture=o.Capture(device='cuda:0');capture.stage='small'
    try:
        for r,branch in enumerate(branches):
            o.bind(branch,'model.layers.0.mlp',operands,r);capture.router(branch.gate,f'rank.{r}')
        history=None
        for case,start in [('prefill',0),('fresh_one',0),('cached_one',1)]:
            norms=[];original_norm=env['norm']
            def observe(x,w):
                z=original_norm(x,w);norms.append((x,z));return z
            env['norm']=observe
            try:
                with torch.device('cuda:0'):
                    expected,history,routes=env['gqa'](env['input_for'](case),start=start,past=history if start else None)
            finally:env['norm']=original_norm
            check('independent_cuda_fixture.'+case,expected,words(frozen[case]['output']),True)
            capture.tensors.clear();capture.bytes=0
            parts=[b(norms[-1][1][None]) for b in branches]
            for r,p in enumerate(parts):
                check(f'moe.gate_up_down_local.rank{r}.{case}',p,words(routes['partials'][r]))
                check(f'moe.ascending_raw_tie_ids.rank{r}.{case}',capture.tensors[f'small.rank.{r}.selected_expert_ids'],torch.tensor(routes['expert_ids']),True)
                check(f'moe.routing_f32.rank{r}.{case}',capture.tensors[f'small.rank.{r}.routing_weights'],words(routes['weight_bits'],torch.float32),True)
            check('moe.late_add_reduce.'+case,norms[-1][0]+o.rank_sum(parts)[0],expected)
            entry=raw['final_norm.weight']
            norm=o.native.hf.Qwen3_5MoeRMSNorm(8,eps=1e-6).cuda().bfloat16()
            norm.weight.copy_(torch.tensor(entry['values'],device='cuda',dtype=torch.bfloat16))
            z=norm(words(frozen[case]['output']).reshape(-1,8).cuda())
            entry=raw['output.weight'];w=torch.tensor(entry['values'],device='cuda',dtype=torch.bfloat16).reshape(entry['shape'])
            head=[F.linear(z,w[lo:hi]) for lo,hi in boundary['head_ranges']]
            for r,h in enumerate(head):check(f'head.ragged.rank{r}.{case}',h,words(boundary[case]['local_head'][r]))
            # Production wire may pad last shard; padding must never become vocab.
            padded=[F.pad(h,(0,6-h.shape[-1])) for h in head]
            gathered=torch.cat([p[:,:hi-lo] for p,(lo,hi) in zip(padded,boundary['head_ranges'])],-1)
            check('head.padded_ordered_gather.'+case,gathered,words(boundary[case]['output']))
    finally:capture.close();operands.decoded.clear()


def fixture_gdn(check,report):
    import inspect
    import sys
    from transformers.models.qwen3_5 import modeling_qwen3_5 as dense_hf
    o.require(o.sha256(dense_hf.__file__)=='aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048','Dense fixture HF source changed')
    for name in ('l2norm','torch_causal_conv1d_update','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule'):
        o.require(inspect.getsource(getattr(dense_hf,name))==inspect.getsource(getattr(o.native.hf,name)), 'HF family function differs')
    path=SUPPORT/'hybrid_mixed_gdn_oracle.py'
    tree=ast.parse(path.read_text(),filename=str(path));tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    env=dict(torch=torch,F=F,hf=dense_hf,BATCH=8,
             raw=json.loads((SUPPORT/'../fixtures/hybrid_cpu/oracle.json').read_text())['tensors'])
    exec(compile(tree,str(path),'exec'),env)
    # Observe only independent CPU fixture operands, as the pinned CUDA state
    # reference did. No native-runtime/production captures participate.
    snapshots=[]
    source_lines=path.read_text().splitlines()
    def observe(frame,event,arg):
        if frame.f_code is env['gdn'].__code__ and event=='line':
            loc=frame.f_locals
            if source_lines[frame.f_lineno-1].strip().startswith('conv=hf.'):
                snapshots.append({k:loc[k].detach().clone() for k in ('qkv','a','b','hist','state') }|{'lane':loc['lane']})
        return observe
    x=torch.tensor([((i*7)%19-9)/16 for i in range(16)],dtype=torch.bfloat16).reshape(2,8)
    sys.settrace(observe)
    try:
        g=env['gqa'](x);env['gdn'](g)
        _,state=env['gdn'](env['gqa'](x[:1]));env['gdn'](g[1:],state)
    finally:sys.settrace(None)
    o.require(len(snapshots)==6,'Incomplete independent GDN operand trace')
    frozen=json.loads((SUPPORT/'hybrid_mixed_gdn_cuda_oracle.json').read_text())
    cpu=json.loads((SUPPORT/'hybrid_mixed_gdn_oracle.json').read_text())
    report['cpu_cuda_state_drift']={}
    for i,snapshot in enumerate(snapshots):
        case=('prefill','fresh_one','cached_one')[i//2];r=snapshot['lane'];n=2 if case=='prefill' else 1
        qkv,a,b,hist,state=[snapshot[k][:1].cuda() for k in ('qkv','a','b','hist','state')]
        def weight(name):return (env['tensor'](2,'attention.'+name)*(1+r*.1)).bfloat16().cuda()
        conv=o.native.hf.torch_causal_conv1d_update(qkv.transpose(1,2),hist,weight('conv').squeeze(1),None,'silu').transpose(1,2)
        q,k,v=conv.split([3,3,4],-1)
        q=q.reshape(1,n,1,3).repeat_interleave(2,dim=2);k=k.reshape(1,n,1,3).repeat_interleave(2,dim=2);v=v.reshape(1,n,2,2)
        decay=-weight('a_log').float().exp()*F.softplus(a.float()+weight('dt_bias'))
        fn=o.native.hf.torch_recurrent_gated_delta_rule if case=='cached_one' else o.native.hf.torch_chunk_gated_delta_rule
        _,final=fn(q,k,v,decay,b.sigmoid(),initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=True)
        check(f'gdn.cuda_f32_state.rank{r}.{case}',final,words(frozen[case]['ranks'][r]['ssm'],torch.float32),True)
        check(f'gdn.conv_bf16.rank{r}.{case}',hist,words(frozen[case]['ranks'][r]['conv']),True)
        report['cpu_cuda_state_drift'][f'{case}.rank{r}']=o.tensor_metric(final.cpu(),words(cpu[case]['ranks'][r]['ssm'],torch.float32).reshape(final.shape))
