#!/usr/bin/env python3
"""Numerical + logical-lifecycle comparison for existing multistep recipes.

No new executor. Candidate calls require their actual parent publication
receipt. Production COW/cancel/quarantine/cold-process proof remains separate.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none')
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import compare as c
import multistep as m
import tp2_oracle as o
from safetensors import safe_open

CANDIDATE_SCHEMA='ferrule.paper.tp2-mixed-lifecycle-candidate.v1'


def cache_hashes(archive,config):
    hashes={}
    for owner,prefix in enumerate(('', 'rank.0.', 'rank.1.')):
        for i,kind in enumerate(config.layer_types):
            fields=(('conv_states','conv_states'),('recurrent_states','recurrent_states')) if kind=='linear_attention' else (('keys','key_cache'),('values','value_cache'))
            for native,name in fields:
                hashes[f'{owner}.{i}.{native}']=o.digest_tensor(archive.get_tensor(f'call.{prefix}cache.{name}.{i:02d}'))
    return hashes


def validate_call(archive,report,event,config,reference):
    call=event['call'];ids=call['input_token_ids'];start=call['position_start'];n=len(ids)
    o.require(call['stage']=='call' and call['sequence_length_after']==start+n and n>0,'Invalid lifecycle call frontier')
    inventory=o.expected_tensors(config,ids,stages=[('call',n,start)])
    if not reference:inventory={k:v for k,v in inventory.items() if not k.endswith('.native_expert_ids')}
    o.require(report['calls']==[call] and report['tensors']==inventory and set(archive.keys())==set(inventory),'Lifecycle call/inventory mismatch')
    for key,spec in inventory.items():
        v=archive.get_tensor(key)
        o.require(list(v.shape)==spec['shape'] and str(v.dtype)==spec['dtype'] and bool(o.torch.isfinite(v).all()),'Lifecycle boundary geometry/dtype/nonfinite: '+key)
    pos=o.torch.arange(start,start+n,dtype=o.torch.int64)
    for key,v in [('input_ids',o.torch.tensor([ids],dtype=o.torch.int64)),('position_ids',pos[None]),('cache_position',pos),('attention_mask',o.torch.ones((1,start+n),dtype=o.torch.int64))]:
        o.require(o.torch.equal(archive.get_tensor('call.'+key),v),'Lifecycle input/position differs')
    logits=archive.get_tensor('call.logits')
    o.require(o.torch.equal(logits,o.torch.cat([archive.get_tensor(f'call.rank.{r}.logits') for r in (0,1)],-1)),'Lifecycle vocabulary gather differs')
    o.require(call['next_token_id']==int(logits[0,-1].argmax()),'Lifecycle reported greedy differs')
    o.require(event['state_after']==cache_hashes(archive,config),'Lifecycle state digest not bound to actual arrays')
    return inventory


def validate_suite(directory,suite,config,reference):
    plan=m.validate_plan(suite['plan']);events=suite['events'];cases=suite['cases']
    o.require(len(events)==len(plan['operations']) and [e['id'] for e in cases]==m.call_ids(plan),'Missing/duplicate lifecycle operations or captures')
    entries={e['id']:e for e in cases};sessions={};epoch=0;results={};transactions=set();live_runtime=set()
    for op,event in zip(plan['operations'],events):
        o.require(all(event.get(k)==v for k,v in op.items()),'Lifecycle event differs from canonical recipe/order')
        kind=op['op'];name=op.get('session')
        if kind=='recreate_model':
            o.require(not sessions and event['decoded_cache_cleared'] is True,'Recreate retained live state/decoded cache')
            epoch+=1
        o.require(event['epoch']==epoch,'Lifecycle owner epoch differs')
        if kind=='create':
            o.require(name not in sessions and event['frontier']==0,'Create must have fresh frontier')
            sessions[name]=dict(frontier=0,next=None,state=None,runtime=None)
        elif kind=='fork':
            src=sessions[op['source']]
            o.require(name not in sessions and event['frontier']==src['frontier'] and event['state_sha256']==src['state'],'Fork source/frontier digest differs')
            if reference:o.require(event['storage_disjoint'] is True,'HF fork must not alias')
            sessions[name]=dict(src,runtime=None)
        elif kind=='drop':
            runtime=sessions[name]['runtime']
            if runtime is not None:live_runtime.remove(runtime)
            del sessions[name]
        elif kind=='call':
            session=sessions[name];call=event['call'];ids=call['input_token_ids']
            o.require(call['position_start']==session['frontier'],'Call uses stale session frontier')
            if op['input']['kind']=='tokens':o.require(ids==op['input']['ids'],'Forced input differs')
            elif reference:o.require(ids==[session['next']],'Reference greedy continuation differs')
            # Candidate continuation is replayed from reference even on a failed
            # greedy prediction. Cross-reference input check happens below.
            if session['state'] is not None:o.require(event['state_before']==session['state'],'State changed between session/fork calls')
            o.require(event['untouched_sessions']==[n for n in sessions if n!=name],'Canary/nonactive session inventory differs')
            entry=entries[op['id']];report=json.loads(c.checked_file(directory,entry['report']).read_text())
            o.require(report['id']==op['id'],'Lifecycle report identity differs')
            if not reference:
                receipt=json.loads(c.checked_file(directory,entry['publication']).read_text())
                o.require(receipt.get('published') is True and receipt.get('error') is None and type(receipt.get('transaction')) is int,
                          'Lifecycle call has no successful parent publication')
                tx=(epoch,receipt['transaction']);o.require(tx not in transactions,'Duplicate publication transaction');transactions.add(tx)
                o.require(receipt['input_token_ids']==ids and receipt['phase']==('prefill' if call['position_start']==0 else 'decode') and receipt['case']==op['id'],
                          'Publication does not bind lifecycle call/input')
                runtime=(epoch,receipt['session'])
                if session['runtime'] is None:
                    o.require(runtime not in live_runtime,'Distinct logical sessions alias a live runtime session')
                    session['runtime']=runtime;live_runtime.add(runtime)
                o.require(session['runtime']==runtime,'Runtime session changed without recreation')
            with safe_open(c.checked_file(directory,entry),framework='pt',device='cpu') as archive:
                inventory=validate_call(archive,report,event,config,reference)
            session.update(frontier=call['sequence_length_after'],next=call['next_token_id'],state=event['state_after'])
            results[op['id']]=dict(entry=entry,event=event,inventory=inventory)
    o.require(not sessions,'Lifecycle left live sessions')
    return results


def compare_suites(reference,ref_suite,actual,actual_suite,config):
    o.require(ref_suite['plan']==actual_suite['plan'],'Different lifecycle recipes')
    ref=validate_suite(reference,ref_suite,config,True);got=validate_suite(actual,actual_suite,config,False)
    rows={}
    for key in ref:
        rc,ac=ref[key]['event']['call'],got[key]['event']['call']
        for field in ('stage','input_token_ids','position_start','sequence_length_after'):
            o.require(rc[field]==ac[field],'Candidate must replay reference lifecycle inputs/frontier')
        metrics={}
        with safe_open(c.checked_file(reference,ref[key]['entry']),framework='pt',device='cpu') as r, safe_open(c.checked_file(actual,got[key]['entry']),framework='pt',device='cpu') as a:
            for tensor in got[key]['inventory']:
                expected=r.get_tensor(tensor);value=a.get_tensor(tensor)
                metrics[tensor]=o.tensor_metric(value,expected,exact=expected.dtype==o.torch.int64)
        rows[key]=dict(metrics=metrics,greedy_match=rc['next_token_id']==ac['next_token_id'])
    passed=all(row['greedy_match'] and all(v['failures']==0 for v in row['metrics'].values()) for row in rows.values())
    return dict(schema='ferrule.paper.tp2-mixed-lifecycle-comparison.v1',status='pass' if passed else 'fail',passed=passed,
        scenario=ref_suite['plan']['scenario'],calls=rows,atol=o.ATOL,rtol=o.RTOL,
        claim='Complete numerical inventory and logical recipe/publication consistency; not physical COW, cancellation, quarantine, or cold-owner process proof')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('reference','actual','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('reference-sha256','actual-sha256'):p.add_argument('--'+key,required=True)
    a=p.parse_args(argv);o.require(a.output.resolve().is_relative_to(o.HERE),'Output must stay paper-local')
    o.require(not a.output.resolve().is_relative_to(a.reference.resolve()) and not a.output.resolve().is_relative_to(a.actual.resolve()),'Cannot modify input bundle')
    o.initialize();r=c.load_manifest(a.reference,a.reference_sha256,True,lifecycle=True);g=c.load_manifest(a.actual,a.actual_sha256,False,lifecycle=True)
    for key in ('config_sha256','index_sha256','full_shard_set_sha256','effective_text_config'):
        o.require(r['checkpoint'][key]==g['checkpoint'][key],'Lifecycle checkpoint differs')
    config=o.native.hf.Qwen3_5MoeTextConfig(**r['checkpoint']['effective_text_config'])
    o.require(config.num_hidden_layers==40 and config.vocab_size==248320 and config.num_experts==256,'Expected full40 lifecycle reference')
    result=compare_suites(a.reference,r['suite'],a.actual,g['suite'],config)
    result.update(reference_manifest_sha256=a.reference_sha256,actual_manifest_sha256=a.actual_sha256)
    o.native.write_json(a.output,result);print(json.dumps(dict(status=result['status'],calls=len(result['calls']))))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
