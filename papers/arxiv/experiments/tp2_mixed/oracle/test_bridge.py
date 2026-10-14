"""Synthetic wire fixtures, NOT numerical model evidence; CPU only."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',ATEN_CPU_CAPABILITY='avx512')
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bridge as b
import compare as c
import tp2_oracle as o
from safetensors import safe_open

PIN='a'*64


def fixture_config(layers=2):
    cfg=o.native.tiny_config();cfg.num_key_value_heads=2;cfg.vocab_size=10000
    cfg.num_hidden_layers=layers
    cfg.layer_types=['linear_attention','full_attention'] if layers==2 else (['linear_attention']*3+['full_attention'])*10
    return cfg


def synthetic_bundle(root,cfg,mutation=None):
    root=Path(root);candidate=root/'candidate';candidate.mkdir()
    protocol=json.loads((o.HERE/'protocol.json').read_text())
    reference=dict(protocol=protocol,checkpoint=dict(config_sha256='c'*64,index_sha256='d'*64,full_shard_set_sha256='e'*64,effective_text_config=cfg.to_dict()))
    contract=dict(schema=b.RAW_SCHEMA,oracle_manifest_sha256=PIN,owners=[0,1],cases=['hello','capital'],phases=['prefill','decode.0'],
                  storage='f32le-native-host-view',complete_required_inventory=False,max_tensor_bytes=8<<20,max_bytes_per_owner=1<<30,max_events_per_owner=40000,max_invocations=4)
    (candidate/'contract.json').write_text(json.dumps(contract))
    owners=[];truth={}
    for rank in (0,1):
        rows=[dict(kind='owner_binding',binding=dict(owner=rank,device=rank,epoch=1,group=2,source=0,ordered_members=[0,1],
                    rank_layout=dict(lane=rank,layers=[0,cfg.num_hidden_layers],precision='Bf16',routed_execution='Local')))]
        for ci,(case,prompt) in enumerate(o.PROMPTS.items()):
            for step,phase in enumerate(('prefill','decode.0')):
                invocation=ci*2+step+1;tx=10+invocation;session=100+ci;start=0 if step==0 else len(prompt);ids=prompt if not step else [0];n=len(ids);end=start+n
                begin=dict(owner=rank,transaction=tx,sessions=[session],tokens=ids,positions=list(range(start,end)),
                           sequences=[dict(phase='Prefill' if not step else 'Decode',context_len=start,sequence_len=end,query=[0,n],page_generation=1,execution_generation=7,state_index=0)])
                rows.append(dict(kind='begin',invocation=invocation,identity=begin,committed=False))
                def event(name,layer,shape,dtype='Bf16',role=None,parameter=None,partition=None,value=None,**kw):
                    v=torch_value(shape,dtype) if value is None else value
                    e=dict(kind='tensor',invocation=invocation,event=name,layer=layer,shape=list(shape),strides=b.strides(shape),native_dtype=dtype,
                           committed=False,sequence=None,parameter_partition=partition,attention_partition=b.attention_geometry(cfg,layer,rank) if layer is not None else None,
                           parameter=parameter,role=role,ids=None,weight_bits=None,_value=v)
                    e.update(kw);rows.append(e);return e
                event('embedding.output',None,[n,cfg.hidden_size],role='TokenEmbedding',parameter='model.language_model.embed_tokens.weight',partition=dict(rows=[[0,cfg.vocab_size]],columns=[0,cfg.hidden_size],collective='None'))
                for layer,kind in enumerate(cfg.layer_types):
                    for name in ('mixed.layer.output','mixed.attention.partial','mixed.attention.reduced','mixed.moe.partial','mixed.moe.reduced','mixed.routed.partial'):
                        event(name,layer,[n,cfg.hidden_size])
                    event('mixed.swiglu.partial',layer,[n,cfg.hidden_size],role='SharedExpertDown',parameter=f'layers.{layer}.shared.down',
                          partition=dict(rows=[[0,cfg.hidden_size]],columns=list(o.bounds(cfg.shared_expert_intermediate_size,rank)),collective='Sum'))
                    event('mixed.linear.output',layer,[n,1],role='SharedExpertOutputGate',parameter=f'layers.{layer}.shared.output_gate',
                          partition=dict(rows=[[0,1]],columns=[0,cfg.hidden_size],collective='None'))
                    event('router',layer,[n,cfg.num_experts],ids=list(range(cfg.num_experts_per_tok))*n,weight_bits=[0x3e800000]*(n*cfg.num_experts_per_tok))
                    event('router.probabilities',layer,[n,cfg.num_experts],'F32')
                    if kind=='linear_attention':
                        sequence=dict(sequence=0,packed_rows=list(range(n)),positions=list(range(start,end)),input_frontier=start)
                        event('gdn.conv_state',layer,[sum(o.gdn_groups(cfg))//2,cfg.linear_conv_kernel_dim],sequence=sequence)
                        event('gdn.recurrent_state',layer,[cfg.linear_num_value_heads//2,cfg.linear_key_head_dim,cfg.linear_value_head_dim],'F32',sequence=sequence)
                event('norm.output',None,[n,cfg.hidden_size])
                event('mixed.linear.output',None,[n,cfg.vocab_size//2],role='OutputHead',parameter='lm_head.weight',
                      partition=dict(rows=[list(o.bounds(cfg.vocab_size,rank))],columns=[0,cfg.hidden_size],collective='Gather'))
                rows.append(dict(kind='execution_end',invocation=invocation,execution_ok=True,committed=False))
                source=dict(transaction=tx,source_owner=123,state_slot=ci,generation=1,incarnation=3+ci,committed_tokens=end,page_size=2)
                rows.append(dict(kind='committed_kv_begin',invocation=invocation,identity=source,committed=True))
                local=0
                for layer,kind in enumerate(cfg.layer_types):
                    if kind!='full_attention':continue
                    for plane in (0,1):
                        for lo in range(0,end,2):
                            hi=min(end,lo+2);heads=cfg.num_key_value_heads//2
                            event('committed.kv',layer,[hi-lo,heads,cfg.head_dim],committed=True,transaction=tx,local_kv_layer=local,
                                  plane=plane,positions=[lo,hi],kv_heads=list(o.bounds(cfg.num_key_value_heads,rank)),kv_owners=[rank],
                                  page=100+ci*10+lo//2,slot=lo//2,slot_epoch=4,source_owner=123,state_slot=ci,generation=1,incarnation=3+ci)
                    local+=1
                if rank==0:
                    parent=dict(case=case,phase=phase,session=session,input_token_ids=ids,published=True,transaction=tx,error=None)
                    (candidate/f'{case}.{phase}.receipt.json').write_text(json.dumps(parent))
                    kv=dict(schema=b.KV_SCHEMA,case=case,phase=phase,session=session,transaction=tx,owners=[0,1],export_completed=True,writer_drain_required=True,error=None)
                    (candidate/f'{case}.{phase}.kv-receipt.json').write_text(json.dumps(kv))
                    torch_value((n,cfg.vocab_size),'F32').numpy().astype('<f4').tofile(root/f'{case}.{phase}.actual.f32le')
        owners.append(rows)
    if mutation:mutation(owners,candidate)
    for rank,rows in enumerate(owners):
        directory=candidate/f'rank-{rank}';directory.mkdir();offset=0;count=0
        with (directory/'tensors.f32le').open('wb') as f:
            for e in rows:
                if e['kind']!='tensor':continue
                data=e.pop('_value').float().contiguous().numpy().astype('<f4').tobytes()
                e.update(order=count,offset=offset,bytes=len(data),storage='f32le-native-host-view')
                f.write(data);count+=1;offset+=len(data)
        (directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in rows))
        (directory/'capture-status.json').write_text(json.dumps(dict(schema=b.OWNER_SCHEMA,invocations=4,events=count,tensor_bytes=offset,
            complete_available_trace=True,capture_error=None,required_inventory_complete=False,committed=False)))
    return reference


def torch_value(shape,dtype):return o.torch.full(tuple(shape),.5,dtype=o.torch.bfloat16 if dtype=='Bf16' else o.torch.float32)


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):o.initialize()
    def tearDown(self):self.assertFalse(o.torch.cuda.is_initialized())

    def convert(self,layers=2,mutation=None,after=None):
        temp=tempfile.TemporaryDirectory(dir=o.HERE);self.addCleanup(temp.cleanup)
        root=Path(temp.name);raw=root/'raw';raw.mkdir();cfg=fixture_config(layers)
        ref=synthetic_bundle(raw,cfg,mutation)
        if after:after(raw)
        out=root/'canonical';manifest=b.convert_raw(raw,out,ref,PIN,b.snapshot(raw)['sha256'])
        return out,manifest,cfg

    def test_full40_inventory_2178_and_both_cases_distinct_shapes(self):
        out,manifest,cfg=self.convert(40)
        self.assertEqual(manifest['status'],'complete')
        for entry in manifest['cases']:
            report=b.read(c.checked_file(out,entry['report']))
            with safe_open(c.checked_file(out,entry),framework='pt',device='cpu') as archive:
                inventory=c.validate_case(archive,report,cfg,o.PROMPTS[entry['id']],False)
                self.assertEqual(len(inventory),2178)
                self.assertEqual(len(archive.keys()),2178)

    def test_native_nd_and_cache_prefix_reconstruction(self):
        out,manifest,cfg=self.convert()
        with safe_open(out/'capital.safetensors',framework='pt',device='cpu') as a:
            self.assertEqual(a.get_tensor('decode.0.rank.1.cache.key_cache.01').shape,(1,1,6,cfg.head_dim))
            self.assertEqual(a.get_tensor('decode.0.cache.key_cache.01').shape,(1,2,6,cfg.head_dim))
            self.assertEqual(a.get_tensor('prefill.rank.0.cache.recurrent_states.00').dtype,o.torch.float32)
            self.assertEqual(a.get_tensor('prefill.rank.0.layers.00.routing_weights').tolist(),[[.25]*cfg.num_experts_per_tok]*5)

    def test_missing_duplicate_and_unpublished_fail_closed(self):
        def missing(rows,_):rows[0][:]=[e for e in rows[0] if not(e.get('event')=='router.probabilities' and e.get('invocation')==1)]
        def duplicate(rows,_):
            i=next(i for i,e in enumerate(rows[0]) if e.get('event')=='embedding.output');rows[0].insert(i,copy.deepcopy(rows[0][i]))
        def unpublished(_,c):
            p=c/'hello.prefill.receipt.json';v=b.read(p);v['published']=False;p.write_text(json.dumps(v))
        for mutation,message in ((missing,'Missing'),(duplicate,'Duplicate'),(unpublished,'published')):
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):self.convert(mutation=mutation)

    def test_wrong_transaction_session_owner_and_kv_source(self):
        def wrong_tx(rows,_):next(e for e in rows[0] if e['kind']=='begin')['identity']['transaction']=999
        def wrong_session(rows,_):next(e for e in rows[0] if e['kind']=='begin')['identity']['sessions']=[999]
        def wrong_owner(rows,_):rows[1][0]['binding']['owner']=0
        def wrong_source(rows,_):next(e for e in rows[0] if e.get('event')=='committed.kv')['source_owner']=999
        def wrong_both_source(rows,_):next(e for e in rows[1] if e['kind']=='committed_kv_begin')['identity']['incarnation']=999
        for mutation in (wrong_tx,wrong_session,wrong_owner,wrong_source,wrong_both_source):
            with self.subTest(mutation=mutation.__name__),self.assertRaises(ValueError):self.convert(mutation=mutation)

    def test_missing_duplicate_prefix_and_bad_partition(self):
        def missing(rows,_):rows[0][:]=[e for e in rows[0] if not(e.get('event')=='committed.kv' and e.get('positions')==[0,2] and e['invocation']==4)]
        def duplicate(rows,_):
            i=next(i for i,e in enumerate(rows[0]) if e.get('event')=='committed.kv');rows[0].insert(i,copy.deepcopy(rows[0][i]))
        def partition(rows,_):next(e for e in rows[0] if e.get('event')=='gdn.conv_state')['attention_partition']['key_heads']=[7,8]
        for mutation in (missing,duplicate,partition):
            with self.subTest(mutation=mutation.__name__),self.assertRaises(ValueError):self.convert(mutation=mutation)

    def test_event_order_writer_status_and_receipt_are_required(self):
        def order(rows,_):
            i=next(i for i,e in enumerate(rows[0]) if e['kind']=='execution_end');rows[0].insert(1,rows[0].pop(i))
        def status(raw):
            p=raw/'candidate/rank-1/capture-status.json';s=b.read(p);s['complete_available_trace']=False;p.write_text(json.dumps(s))
        def receipt(raw):
            p=raw/'candidate/hello.prefill.kv-receipt.json';s=b.read(p);s['export_completed']=False;p.write_text(json.dumps(s))
        with self.assertRaises(ValueError):self.convert(mutation=order)
        for after in (status,receipt):
            with self.assertRaises(ValueError):self.convert(after=after)

    def test_owner_local_generations_and_indices_need_not_match(self):
        def mutate(rows,_):
            for e in rows[1]:
                if e['kind']=='begin':
                    e['identity']['sequences'][0]['execution_generation']=999
                    e['identity']['sequences'][0]['state_index']=3
        out,manifest,_=self.convert(mutation=mutate)
        self.assertEqual(manifest['status'],'complete')

    def test_missing_receipt_and_raw_pin_do_not_publish_complete_candidate(self):
        with tempfile.TemporaryDirectory(dir=o.HERE) as d:
            root=Path(d);raw=root/'raw';raw.mkdir();ref=synthetic_bundle(raw,fixture_config())
            with self.assertRaisesRegex(ValueError,'pin'):
                b.convert_raw(raw,root/'bad-pin',ref,PIN,'0'*64)
            self.assertFalse((root/'bad-pin').exists())
            p=raw/'candidate/hello.prefill.kv-receipt.json';data=b.read(p);data['export_completed']=False;p.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'KV receipt'):
                b.convert_raw(raw,root/'failed',ref,PIN,b.snapshot(raw)['sha256'])
            self.assertEqual(b.read(root/'failed/manifest.json')['status'],'incomplete')
            p.unlink()
            with self.assertRaisesRegex(ValueError,'Missing'):b.snapshot(raw)

    def test_nonfinite_inexact_bf16_and_stride_alias_rejected(self):
        for value in (float('nan'),1.0001):
            def mutate(rows,_,value=value):
                e=next(e for e in rows[0] if e.get('event')=='embedding.output');e['_value']=e['_value'].float();e['_value'][0,0]=value
            with self.assertRaises(ValueError):self.convert(mutation=mutate)
        def strides(rows,_):next(e for e in rows[0] if e.get('event')=='embedding.output')['strides']=[0,0]
        with self.assertRaisesRegex(ValueError,'layout'):self.convert(mutation=strides)


if __name__=='__main__':unittest.main()
