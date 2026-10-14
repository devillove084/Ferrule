#!/usr/bin/env python3
"""CPU-only v2 observations -> canonical archive, never an execution engine."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import compare as c
import tp2_oracle as o
import torch
from safetensors.torch import save_file

HERE=o.HERE
RAW_SCHEMA='ferrule.mixed-candidate.v2'
OWNER_SCHEMA='ferrule.mixed-owner-trace.v2'
KV_SCHEMA='ferrule.mixed-committed-kv-receipt.v1'
require=o.require


def read(path, cap=64*1024**2):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'Missing/oversize JSON: '+str(path))
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'Duplicate JSON key: '+k);out[k]=v
        return out
    def invalid(v):raise ValueError('Nonfinite JSON: '+v)
    return json.loads(path.read_text(),object_pairs_hook=pairs,parse_constant=invalid)


def strides(shape):
    result=[];width=1
    for d in reversed(shape):result.insert(0,width);width*=d
    return result


def decode(stream,e):
    shape=e['shape']
    require(isinstance(shape,list) and 1<=len(shape)<=5 and all(type(n) is int and n>0 for n in shape),'Invalid native N-D shape')
    # v2 explicitly exports contiguous native sources, not arbitrary padded views.
    require(e['strides']==strides(shape),'Non-contiguous/overlapping native layout')
    require(e['storage']=='f32le-native-host-view' and e['native_dtype'] in ('Bf16','F32'),'Wrong native storage/dtype')
    size=e['bytes'];off=e['offset']
    require(type(off) is int and off>=0 and off%4==0 and size==4*math.prod(shape) and size<=8*1024**2,'Native byte geometry')
    require(off+size<=stream.stat().st_size,'Tensor byte range exceeds file')
    with stream.open('rb') as f:f.seek(off);raw=f.read(size)
    import numpy as np
    value=torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy()).reshape(shape)
    require(bool(torch.isfinite(value).all()),'Nonfinite native observation')
    if e['native_dtype']=='Bf16':
        bf=value.bfloat16()
        require(torch.equal(bf.float().view(torch.int32),value.view(torch.int32)),'Not an exact BF16 host expansion')
        value=bf
    return value


def read_owner(candidate,rank,config,expected_invocations=4):
    directory=candidate/f'rank-{rank}';status=read(directory/'capture-status.json',1024**2)
    require(status['schema']==OWNER_SCHEMA and status['complete_available_trace'] is True and
            status['required_inventory_complete'] is False and not status.get('capture_error') and
            status['committed'] is False,'Owner writer incomplete/invalid status')
    stream=directory/'tensors.f32le';index=directory/'events.jsonl'
    require(stream.is_file() and not stream.is_symlink() and stream.stat().st_size<=1024**3,'Owner tensor file invalid')
    require(index.is_file() and not index.is_symlink() and index.stat().st_size<=64*1024**2,'Owner index invalid')
    binding=None;calls={};current=None;mode=None;order=0;offset=0
    with index.open() as f:
        for line in f:
            require(len(line.encode())<=256*1024,'Metadata record bound exceeded')
            e=json.loads(line);kind=e.get('kind')
            if kind=='owner_binding':
                require(binding is None and not calls,'Duplicate/out-of-order binding')
                binding=e['binding'];continue
            require(binding is not None,'Event before binding')
            if kind=='begin':
                require(mode in (None,'kv') and (current is None or current['kv'] is not None),'Out-of-order begin')
                inv=e['invocation'];require(inv==len(calls)+1 and e['committed'] is False,'Invocation order/commit flag')
                current=dict(begin=e['identity'],tensors=[],chunks=[],kv=None,end=None)
                calls[inv]=current;mode='execute';continue
            require(current is not None and e.get('invocation')==len(calls),'Orphan/wrong invocation')
            if kind=='execution_end':
                require(mode=='execute' and current['end'] is None and e['execution_ok'] is True and e['committed'] is False,'Execution end is not publication')
                current['end']=e;mode='ended'
            elif kind=='committed_kv_begin':
                require(mode=='ended' and current['kv'] is None and e['committed'] is True,'Out-of-order/duplicate KV export')
                current['kv']=e['identity'];mode='kv'
            elif kind=='tensor':
                require(type(e.get('order')) is int and e['order']==order and e['offset']==offset,'Tensor duplicate/order/offset gap')
                require(0<e['bytes']<=8*1024**2 and e['bytes']%4==0,'Tensor byte bound')
                order+=1;offset+=e['bytes'];require(order<=40000,'Event bound exceeded')
                if e['event']=='committed.kv':
                    require(mode=='kv' and e['committed'] is True,'KV before publication/export')
                    current['chunks'].append(e)
                else:
                    require(mode=='execute' and e['committed'] is False,'Operator event not provisional/in execution')
                    current['tensors'].append(e)
            else:raise ValueError('Unknown event kind: '+str(kind))
    require(binding is not None and len(calls)==expected_invocations and mode=='kv','Incomplete invocation stream')
    require(status['invocations']==len(calls) and status['events']==order and status['tensor_bytes']==offset==stream.stat().st_size,'Writer counts/ranges differ')
    require(binding['owner']==rank and binding['ordered_members']==[0,1] and binding['source'] in (0,1),'Wrong owner/group source')
    require(binding['rank_layout']['lane']==rank and binding['rank_layout']['layers']==[0,config.num_hidden_layers],'Owner layout differs')
    return dict(binding=binding,calls=calls,stream=stream,status=status)


def join_call(owners,parent,kv_receipt,config):
    require(parent.get('published') is True and parent.get('error') is None and type(parent.get('transaction')) is int,'Parent is not published')
    require(kv_receipt['schema']==KV_SCHEMA and kv_receipt['export_completed'] is True and kv_receipt['writer_drain_required'] is True and
            kv_receipt.get('error') is None and kv_receipt['owners']==[0,1],'KV receipt incomplete')
    for key in ('case','phase','session','transaction'):require(kv_receipt[key]==parent[key],'Parent/KV transaction/session differs')
    selected=[]
    for rank,owner in enumerate(owners):
        found=[call for call in owner['calls'].values() if call['begin']['transaction']==parent['transaction'] and call['begin']['sessions']==[parent['session']]]
        require(len(found)==1,'Missing/duplicate/wrong transaction or session join')
        call=found[0];b=call['begin'];seq=b['sequences']
        require(b['owner']==rank and b['tokens']==parent['input_token_ids'] and len(seq)==1,'Owner tokens/sequence identity differs')
        seq=seq[0];n=len(b['tokens']);positions=b['positions']
        require(n>0 and len(positions)==n and all(type(t) is int and 0<=t<config.vocab_size for t in b['tokens']),'Invalid packed tokens/positions')
        require(all(type(seq[k]) is int and seq[k]>=0 for k in ('page_generation','execution_generation','state_index')),'Invalid owner-local generation/index')
        start=positions[0]
        require(n>0 and positions==list(range(start,start+n)) and seq['query']==[0,n] and seq['context_len']==start and seq['sequence_len']==start+n,'Packed input frontier/geometry differs')
        require(seq['phase']==('Prefill' if start==0 else 'Decode'),'Packed phase differs')
        k=call['kv'];require(k is not None and k['transaction']==parent['transaction'] and k['committed_tokens']==start+n and k['generation']==seq['page_generation'] and k['page_size']>0,'KV source frontier/generation differs')
        for field in ('source_owner','state_slot','generation','incarnation'):require(type(k[field]) is int and k[field]>=0,'Invalid KV source identity')
        selected.append(call)
    for key in ('epoch','group','source','ordered_members'):
        require(owners[0]['binding'][key]==owners[1]['binding'][key],'Owner binding source/epoch disagreement')
    a,b=selected
    for field in ('transaction','sessions','tokens','positions'):
        require(a['begin'][field]==b['begin'][field],'Rank input/transaction disagreement')
    # Execution generations and packed state indices are owner-local identities,
    # not a replicated collective value. Preserve them without equating ranks.
    for field in ('phase','context_len','sequence_len','query','page_generation'):
        require(a['begin']['sequences'][0][field]==b['begin']['sequences'][0][field],'Rank packed geometry disagreement')
    require(a['kv']==b['kv'],'Rank committed source identity disagreement')
    return selected


def attention_geometry(config,layer,rank):
    if config.layer_types[layer]=='full_attention':
        return dict(kind='gqa',query_heads=list(o.bounds(config.num_attention_heads,rank)),
            kv_heads=list(o.bounds(config.num_key_value_heads,rank)),kv_owners=[rank],head_dim=config.head_dim)
    return dict(kind='gated_delta',key_heads=list(o.bounds(config.linear_num_key_heads,rank)),
        value_heads=list(o.bounds(config.linear_num_value_heads,rank)),
        qkv_channels=[list(x) for x in o.grouped_ranges(o.gdn_groups(config),rank,2)],
        shape=dict(key_heads=config.linear_num_key_heads//2,value_heads=config.linear_num_value_heads//2,
            key_dim=config.linear_key_head_dim,value_dim=config.linear_value_head_dim,kernel=config.linear_conv_kernel_dim))


def mapped_key(e,rank):
    """Only actual named model sources. Never select a boundary by flat length."""
    name=e['event'];layer=e.get('layer');role=e.get('role')
    if name=='embedding.output' and layer is None:return 'embedding'
    if name=='norm.output' and layer is None:return 'final_norm'
    if name=='mixed.linear.output' and role=='OutputHead' and layer is None:return f'rank.{rank}.logits'
    if layer is None:return None
    require(type(layer) is int and layer>=0,'Invalid layer ID')
    p=f'layers.{layer:02d}.';r=f'rank.{rank}.'
    if name=='mixed.linear.output' and role=='SharedExpertOutputGate':return r+p+'shared_gate_logit'
    if name=='mixed.swiglu.partial':
        require(role=='SharedExpertDown','Wrong SwiGLU source role')
        return r+p+'shared_partial'
    globals={'mixed.layer.output':'hidden','mixed.attention.reduced':'attention','mixed.moe.reduced':'moe'}
    locals={'mixed.attention.partial':'attention_partial','mixed.moe.partial':'moe_partial',
        'mixed.routed.partial':'routed_partial','router':'raw_router_logits','router.probabilities':'router_probabilities'}
    if name in globals:return p+globals[name]
    if name in locals:return r+p+locals[name]
    if name in ('gdn.recurrent_state','gdn.conv_state'):
        field='recurrent_states' if name=='gdn.recurrent_state' else 'conv_states'
        return r+f'cache.{field}.{layer:02d}'
    return None


def validate_metadata(e,rank,call,config):
    b=call['begin'];n=len(b['tokens']);start=b['positions'][0];layer=e.get('layer')
    if layer is not None:
        require(0<=layer<config.num_hidden_layers,'Layer outside model')
        require(e.get('attention_partition')==attention_geometry(config,layer,rank),'Wrong native attention partition/source')
    else:require(e.get('attention_partition') is None,'Unexpected root attention partition')
    seq=e.get('sequence')
    if e['event'] in ('gdn.recurrent_state','gdn.conv_state') or seq is not None:
        require(seq==dict(sequence=0,packed_rows=list(range(n)),positions=list(range(start,start+n)),input_frontier=start),
                'Sequence/frontier metadata differs')
    if e.get('role') in ('OutputHead','SharedExpertOutputGate','SharedExpertDown','TokenEmbedding'):
        p=e.get('parameter_partition');require(isinstance(p,dict) and bool(e.get('parameter')),'Parameter source metadata missing')
        role=e['role']
        expected_rows=([list(o.bounds(config.vocab_size,rank))] if role=='OutputHead' else
            [[0,1]] if role=='SharedExpertOutputGate' else [[0,config.hidden_size]] if role=='SharedExpertDown' else [[0,config.vocab_size]])
        expected_cols=list(o.bounds(config.shared_expert_intermediate_size,rank)) if role=='SharedExpertDown' else [0,config.hidden_size]
        require(p['rows']==expected_rows and p['columns']==expected_cols and isinstance(p['collective'],str),'Parameter shard geometry differs')


def kv_arrays(call,owner,rank,config):
    source=call['kv'];end=source['committed_tokens'];page_size=source['page_size']
    full_layers=[i for i,k in enumerate(config.layer_types) if k=='full_attention']
    tables={};pages={};physical={}
    for e in call['chunks']:
        require(e['transaction']==source['transaction'],'KV chunk wrong transaction')
        for field in ('source_owner','state_slot','generation','incarnation'):
            require(e[field]==source[field],'KV chunk source identity differs: '+field)
        layer=e['layer'];plane=e['plane']
        require(layer in full_layers and plane in (0,1) and e['local_kv_layer']==full_layers.index(layer),'KV layer/plane differs')
        lo,hi=e['positions'];heads=config.num_key_value_heads//2
        require(type(lo) is int and type(hi) is int and 0<=lo<hi<=end and lo//page_size==(hi-1)//page_size,'KV logical position range differs')
        require(e['kv_heads']==list(o.bounds(config.num_key_value_heads,rank)) and e['kv_owners']==[rank],'KV head partition differs')
        require(e['shape']==[hi-lo,heads,config.head_dim] and e['native_dtype']=='Bf16','KV native shape/dtype differs')
        page=lo//page_size;descriptor=tuple(e[f] for f in ('page','slot','slot_epoch'))
        require(all(type(v) is int and v>=0 for v in descriptor),'Invalid physical page metadata')
        require(pages.setdefault(page,descriptor)==descriptor,'Logical page maps to inconsistent physical identity')
        require(physical.setdefault(descriptor,page)==page,'Physical page aliases distinct logical pages')
        table=tables.setdefault((layer,plane),{})
        require(all(pos not in table for pos in range(lo,hi)),'Duplicate/overlapping KV chunk')
        v=decode(owner['stream'],e)
        for pos in range(lo,hi):table[pos]=v[pos-lo]
    output={}
    for layer in full_layers:
        for plane,name in ((0,'key_cache'),(1,'value_cache')):
            table=tables.get((layer,plane),{})
            require(set(table)==set(range(end)),'Missing committed KV prefix/chunk')
            output[f'rank.{rank}.cache.{name}.{layer:02d}']=torch.stack([table[t] for t in range(end)]).permute(1,0,2)[None].contiguous()
    return output


def assemble_call(owners,calls,parent,config,phase,full_logits):
    n=len(parent['input_token_ids']);start=calls[0]['begin']['positions'][0];end=start+n
    inventory=o.expected_tensors(config,parent['input_token_ids'],stages=[(phase,n,start)])
    inventory={k:v for k,v in inventory.items() if not k.endswith('.native_expert_ids')}
    local=[]
    for rank,(owner,call) in enumerate(zip(owners,calls)):
        output={}
        for e in call['tensors']:
            key=mapped_key(e,rank)
            if key is None:continue
            validate_metadata(e,rank,call,config)
            full=phase+'.'+key;require(full in inventory,'Unexpected mapped source: '+full)
            spec=inventory[full];batch_axis=key in ('embedding','final_norm',f'rank.{rank}.logits') or (
                key.startswith('layers.') or '.cache.' in key or key.endswith(('.attention_partial','.moe_partial')))
            raw_shape=spec['shape'][1:] if batch_axis else spec['shape']
            require(e['shape']==raw_shape,'Native N-D geometry does not match canonical boundary: '+key)
            value=decode(owner['stream'],e);require(str(value.dtype)==spec['dtype'],'Native boundary dtype differs: '+key)
            require(key not in output,'Duplicate canonical observation: '+key)
            output[key]=value[None] if batch_axis else value
            if e['event']=='router':
                ids=e.get('ids');bits=e.get('weight_bits');count=n*config.num_experts_per_tok
                require(isinstance(ids,list) and len(ids)==count and all(type(i) is int and 0<=i<config.num_experts for i in ids),'Invalid integer route IDs')
                require(isinstance(bits,list) and len(bits)==count and all(type(i) is int and 0<=i<2**32 for i in bits),'Invalid F32 route weight bits')
                rid=torch.tensor(ids,dtype=torch.int64).reshape(n,-1)
                require(all(len(set(row))==config.num_experts_per_tok for row in rid.tolist()),'Duplicate selected experts')
                import numpy as np
                weights=torch.from_numpy(np.array(bits,dtype='<u4').view('<f4').copy()).reshape(n,-1)
                require(bool(torch.isfinite(weights).all()),'Nonfinite route weights')
                base=key.removesuffix('raw_router_logits')
                output[base+'selected_expert_ids']=rid;output[base+'routing_weights']=weights
        output.update(kv_arrays(call,owner,rank,config));local.append(output)
    # Replicated outputs must agree exactly. Do not replace a missing observation
    # on one rank with the other's value, or compute a reduction to fill a gap.
    result={};global_keys={k for k in inventory if not k.startswith(phase+'.rank.') and '.cache.' not in k}
    input_keys={phase+'.'+k for k in ('input_ids','position_ids','cache_position','attention_mask','logits')}
    for full in global_keys-input_keys:
        key=full[len(phase)+1:]
        require(key in local[0] and key in local[1],'Missing replicated source: '+key)
        require(torch.equal(local[0][key],local[1][key]),'Replica boundary disagreement: '+key)
        result[full]=local[0][key]
    for rank in (0,1):
        result.update({phase+'.'+k:v for k,v in local[rank].items() if k.startswith(f'rank.{rank}.')})
    for layer,kind in enumerate(config.layer_types):
        for name in (('conv_states','recurrent_states') if kind=='linear_attention' else ('key_cache','value_cache')):
            keys=[f'{phase}.rank.{r}.cache.{name}.{layer:02d}' for r in (0,1)]
            require(all(k in result for k in keys),'Missing actual local state: '+str(keys))
            values=[result[k] for k in keys]
            result[f'{phase}.cache.{name}.{layer:02d}']=o.merge_gdn(values,config,2) if name=='conv_states' else torch.cat(values,1)
    # These integer tensors encode validated packed-batch metadata. The mask is
    # the v1 batch-one/no-padding all-ones representation, not a new model result.
    pos=torch.tensor(calls[0]['begin']['positions'],dtype=torch.int64)
    result.update({phase+'.input_ids':torch.tensor([parent['input_token_ids']],dtype=torch.int64),
        phase+'.position_ids':pos[None].clone(),phase+'.cache_position':pos,
        phase+'.attention_mask':torch.ones((1,end),dtype=torch.int64),phase+'.logits':full_logits})
    require(set(result)==set(inventory),'Missing required canonical inventory: '+str(sorted(set(inventory)-set(result))))
    for key,spec in inventory.items():
        value=result[key];require(list(value.shape)==spec['shape'] and str(value.dtype)==spec['dtype'] and bool(torch.isfinite(value).all()),'Invalid canonical boundary: '+key)
    gathered=torch.cat([result[f'{phase}.rank.{r}.logits'] for r in (0,1)],-1)
    require(torch.equal(gathered,full_logits),'Observed full vocabulary differs from ordered rank gather')
    call=dict(stage=phase,input_token_ids=parent['input_token_ids'],position_start=start,
              sequence_length_after=end,next_token_id=int(full_logits[0,-1].argmax()))
    return result,call,inventory


def source_files(root):
    root=Path(root).resolve();candidate=root/'candidate'
    files=[candidate/'contract.json']
    for r in (0,1):files += [candidate/f'rank-{r}'/name for name in ('events.jsonl','tensors.f32le','capture-status.json')]
    for case in ('hello','capital'):
        for phase in ('prefill','decode.0'):
            files += [candidate/f'{case}.{phase}.receipt.json',candidate/f'{case}.{phase}.kv-receipt.json',root/f'{case}.{phase}.actual.f32le']
    return root,files


def snapshot(root):
    root,files=source_files(root);result={}
    for p in files:
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root),'Missing/escaping raw artifact: '+str(p))
        cap=1024**3 if p.name=='tensors.f32le' else 64*1024**2
        require(p.stat().st_size<=cap,'Raw file byte cap: '+str(p))
        result[str(p.relative_to(root))]=dict(sha256=o.sha256(p),bytes=p.stat().st_size)
    digest=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return dict(schema='ferrule.paper.tp2-mixed-raw-snapshot.v1',files=result,sha256=digest)


def full_logits(path,n,vocab):
    import numpy as np
    require(path.stat().st_size==4*n*vocab,'Full-vocab observed file geometry')
    values=torch.from_numpy(np.fromfile(path,dtype='<f4').copy()).reshape(1,n,vocab)
    require(bool(torch.isfinite(values).all()),'Nonfinite full vocabulary')
    result=values.bfloat16()
    require(torch.equal(result.float().view(torch.int32),values.view(torch.int32)),'Logits file is not exact BF16 expansion')
    return result


def write_candidate_case(output,name,arrays,calls,inventory):
    path=output/(name+'.safetensors');save_file(arrays,str(path),metadata={'schema':c.CANDIDATE_SCHEMA,'profile':o.PROFILE})
    report=dict(id=name,calls=calls,tensors=inventory,**o.file_record(path))
    o.native.write_json(output/(name+'.json'),report)
    return dict(id=name,**o.file_record(path),report=o.file_record(output/(name+'.json')))


def convert_raw(root,output,ref,reference_pin,raw_pin):
    """Numerical format conversion. Runtime execution authorization remains external."""
    root=Path(root).resolve();output=Path(output).resolve();candidate=root/'candidate'
    require(output.is_relative_to(HERE) and not output.is_relative_to(root),'Fresh output must be separate and paper-local')
    before=snapshot(root);require(before['sha256']==raw_pin,'Raw source snapshot pin differs')
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema=c.CANDIDATE_SCHEMA,profile=o.PROFILE,status='incomplete',cases=[],
        protocol=ref['protocol'],checkpoint=ref['checkpoint'],reference_manifest_sha256=reference_pin,
        boundary_inventory_complete=False,bridge_source_sha256=o.sha256(__file__),
        execution_claim='Canonical observed tensors only; producer custody/authenticated execution must be checked separately',
        checkpoint_identity_scope='Runtime contract attests validation against pinned oracle checkpoint; bridge did not reopen model weights')
    try:
        contract=read(candidate/'contract.json')
        require(contract['schema']==RAW_SCHEMA and contract['oracle_manifest_sha256']==reference_pin and contract['owners']==[0,1] and
                contract['cases']==['hello','capital'] and contract['phases']==['prefill','decode.0'] and
                contract['storage']=='f32le-native-host-view' and contract['complete_required_inventory'] is False,'Raw v2 contract differs')
        require(contract['max_tensor_bytes']<=8*1024**2 and contract['max_bytes_per_owner']<=1024**3 and
                contract['max_events_per_owner']<=40000 and contract['max_invocations']==4,'Raw admission caps differ')
        config=o.native.hf.Qwen3_5MoeTextConfig(**ref['checkpoint']['effective_text_config'])
        owners=[read_owner(candidate,r,config) for r in (0,1)];used=set();sources={}
        manifest['owner_bindings']=[owner['binding'] for owner in owners]
        for case in ('hello','capital'):
            arrays={};calls=[];inventory={};session=None
            for phase in ('prefill','decode.0'):
                parent=read(candidate/f'{case}.{phase}.receipt.json');kv=read(candidate/f'{case}.{phase}.kv-receipt.json')
                require(parent['case']==case and parent['phase']==phase,'Receipt filename/identity differs')
                require(parent['transaction'] not in used,'Duplicate published transaction');used.add(parent['transaction'])
                if session is None:session=parent['session']
                require(parent['session']==session,'Prefill/decode changed session')
                joined=join_call(owners,parent,kv,config)
                expected_start=0 if phase=='prefill' else len(o.PROMPTS[case])
                require(joined[0]['begin']['positions'][0]==expected_start,'Legacy call frontier differs')
                source={k:joined[0]['kv'][k] for k in ('source_owner','state_slot','generation','incarnation')}
                require(sources.setdefault(session,source)==source,'Session committed source changed')
                n=len(parent['input_token_ids']);logits=full_logits(root/f'{case}.{phase}.actual.f32le',n,config.vocab_size)
                values,call,spec=assemble_call(owners,joined,parent,config,phase,logits)
                arrays.update(values);calls.append(call);inventory.update(spec)
            require(calls[0]['input_token_ids']==o.PROMPTS[case],'Wrong original prompt')
            require(set(inventory)==set(c.candidate_inventory(config,o.PROMPTS[case])),'Required legacy inventory differs')
            manifest['cases'].append(write_candidate_case(output,case,arrays,calls,inventory))
        for owner in owners:
            require({call['begin']['transaction'] for call in owner['calls'].values()}==used,'Extra/unjoined owner invocation')
        require(snapshot(root)==before,'Raw source changed during conversion')
        o.native.write_json(output/'raw-source.json',before)
        manifest.update(status='complete',boundary_inventory_complete=True,raw_source=o.file_record(output/'raw-source.json'),
                        candidate_kind='runtime-v2-publication-joined-native-trace')
    except BaseException as error:
        manifest['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        o.native.write_json(output/'manifest.json',manifest)
        with (output/'manifest.sha256').open('x') as f:f.write(o.sha256(output/'manifest.json')+'\n')
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--inspect',action='store_true')
    p.add_argument('--input-sha256');p.add_argument('--reference',type=Path);p.add_argument('--reference-sha256');p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    if a.inspect:
        snap=snapshot(a.input);print(json.dumps(dict(sha256=snap['sha256'],files=len(snap['files'])),indent=2));return 0
    require(all((a.input_sha256,a.reference,a.reference_sha256,a.output)),'Conversion needs both source/reference pins and fresh output')
    require(not a.output.resolve().is_relative_to(a.reference.resolve()),'Cannot write inside immutable reference')
    o.initialize();ref=c.load_manifest(a.reference,a.reference_sha256,True)
    manifest=convert_raw(a.input,a.output,ref,a.reference_sha256,a.input_sha256)
    print(json.dumps(dict(status=manifest['status'],manifest_sha256=o.sha256(a.output/'manifest.json'),cases=[e['id'] for e in manifest['cases']]),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
