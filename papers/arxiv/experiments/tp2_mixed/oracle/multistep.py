"""Bounded lifecycle recipes for the existing authenticated HF oracle.

CLI only prints plans (stdlib, GPUs hidden). run() uses the SAME run_call,
Capture, operands, HF model, and serializer as legacy full40; no runtime loop.
"""
import argparse
import copy
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCHEMA='ferrule.paper.tp2-mixed-lifecycle-reference.v1'
PLAN_SCHEMA='ferrule.paper.tp2-mixed-lifecycle-plan.v1'
SCENARIOS=('greedy','canary','fork','recreate','cold-owner')
PROMPTS={'hello':[9419], 'capital':[760,6511,314,9338,369]}


def build_plan(name, decode_steps=4, prompts=None):
    if name not in SCENARIOS or type(decode_steps) is not int or not 1 <= decode_steps <= 8:
        raise ValueError('Unknown scenario or decode bound (1..8)')
    prompts=copy.deepcopy(PROMPTS if prompts is None else prompts)
    if set(prompts)!=set(PROMPTS) or any(not ids or len(ids)>32 or any(type(i) is not int or i<0 for i in ids) for ids in prompts.values()):
        raise ValueError('Explicit hello/capital prompt IDs required; max 32 each')
    ops=[]
    def add(op, **kw):ops.append(dict(op=op,**kw))
    def create(session,case):add('create',session=session,case=case)
    def call(session,ids=None):
        add('call',session=session,id=f'call-{sum(x["op"]=="call" for x in ops):02d}',
            input=dict(kind='greedy') if ids is None else dict(kind='tokens',ids=list(ids)))
    if name in ('greedy','cold-owner'):
        for case,ids in prompts.items():
            create(case,case);call(case,ids)
            for _ in range(decode_steps):call(case)
            add('drop',session=case)
    elif name=='canary':
        create('main','hello');call('main',prompts['hello'])
        create('canary','capital');call('canary',prompts['capital']);call('canary');add('drop',session='canary')
        for _ in range(decode_steps):call('main')
        add('drop',session='main')
    elif name=='fork':
        create('parent','hello');call('parent',prompts['hello'])
        add('fork',source='parent',session='left');add('fork',source='parent',session='right')
        for branch in ('left','right'):
            call(branch) # identical first continuation checks clone fidelity
            for step in range(1,decode_steps):
                # Right branch deliberately diverges, never silently called greedy.
                call(branch,prompts['capital'][:1] if branch=='right' and step==1 else None)
            add('drop',session=branch)
        call('parent');add('drop',session='parent')
    else:
        for epoch in range(2):
            if epoch:add('recreate_model')
            create('main','hello');call('main',prompts['hello'])
            for _ in range(decode_steps):call('main')
            add('drop',session='main')
    return dict(schema=PLAN_SCHEMA,scenario=name,decode_steps=decode_steps,prompts=prompts,operations=ops,
        limits=dict(max_calls=20,max_live_sessions=3,max_total_capture_bytes=4*1024**3),
        capture='every call; complete old inventory at stage call; cache reflects committed post-call state',
        fork='deep independent HF global/rank cache snapshots, not a runtime COW implementation',
        cold_owner='one fresh authenticated numerical child per launch; repeat in a second launch to establish different owner identity',
        stop='fixed bounded decode count, including EOS; not natural-stop quality/performance evaluation')


def validate_plan(plan, full_model=False):
    expected=build_plan(plan['scenario'],plan['decode_steps'],plan['prompts'])
    if plan!=expected:raise ValueError('Plan differs from bounded canonical recipe')
    if full_model and plan['prompts']!=PROMPTS:raise ValueError('Full-model prompt contract differs')
    if len(call_ids(plan))>plan['limits']['max_calls']:raise ValueError('Call bound exceeded')
    return plan


def call_ids(plan):return [op['id'] for op in plan['operations'] if op['op']=='call']


def state_tensors(state):
    global_cache,rank_caches=state
    return {f'{rank}.{i}.{name}':value for rank,cache in enumerate([global_cache,*rank_caches])
        for i,layer in enumerate(cache.layers) for name in ('keys','values','conv_states','recurrent_states')
        if (value:=getattr(layer,name,None)) is not None}


def state_hashes(state):
    import tp2_oracle as o
    return {k:o.digest_tensor(v) for k,v in state_tensors(state).items()}


def clone_state(state):
    import tp2_oracle as o
    copied=copy.deepcopy(state)
    old,new=state_tensors(state),state_tensors(copied)
    o.require(set(old)==set(new),'Fork cache inventory differs')
    o.require(all(old[k].data_ptr()!=new[k].data_ptr() and o.torch.equal(old[k],new[k]) for k in old),
              'Fork must copy every state tensor without storage aliasing')
    return copied


def run(plan, model, config, operands, capture, output=None, observe=None):
    """Recipe driver only. All numerical calls go through tp2_oracle.run_call."""
    import tp2_oracle as o
    validate_plan(plan)
    o.require(all(i<config.vocab_size for ids in plan['prompts'].values() for i in ids),'Prompt exceeds vocabulary')
    sessions={};entries=[];events=[];total=0;epoch=0
    try:
        for op in plan['operations']:
            kind=op['op'];name=op.get('session')
            if kind=='create':
                o.require(name not in sessions,'Duplicate session')
                sessions[name]=dict(state=o.reset_caches(model,config,operands.device,operands.world),next=None)
                events.append(dict(**op,epoch=epoch,frontier=0))
            elif kind=='fork':
                source=sessions[op['source']]
                sessions[name]=dict(state=clone_state(source['state']),next=source['next'])
                events.append(dict(**op,epoch=epoch,frontier=source['state'][0].get_seq_length(),
                                   state_sha256=state_hashes(source['state']),storage_disjoint=True))
            elif kind=='drop':
                del sessions[name];events.append(dict(**op,epoch=epoch))
            elif kind=='recreate_model':
                o.require(not sessions,'Cannot recreate with live sessions')
                capture.close();del model
                operands.decoded.clear()
                model=o.build_model(config,operands,capture);epoch+=1
                events.append(dict(**op,epoch=epoch,decoded_cache_cleared=True))
            else:
                current=sessions[name];state=current['state'];start=state[0].get_seq_length()
                fed=op['input']['ids'] if op['input']['kind']=='tokens' else [current['next']]
                o.require(all(type(i) is int for i in fed),'Greedy requires a committed predecessor')
                untouched={s:state_hashes(v['state']) for s,v in sessions.items() if s!=name}
                before=state_hashes(state)
                capture.tensors.clear();capture.bytes=0
                record=o.run_call(model,config,operands,capture,*state,fed,start,'call')
                inventory=o.expected_tensors(config,fed,operands.world,stages=[('call',len(fed),start)])
                o.require({k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in capture.tensors.items()}==inventory,
                          'Incomplete lifecycle capture')
                total+=capture.bytes
                o.require(total<=plan['limits']['max_total_capture_bytes'],'Total capture byte bound exceeded')
                for s,pins in untouched.items():o.require(state_hashes(sessions[s]['state'])==pins,'Cross-session state mutation')
                current['next']=record['next_token_id']
                events.append(dict(**op,epoch=epoch,call=record,state_before=before,state_after=state_hashes(state),
                                   untouched_sessions=list(untouched)))
                if output is not None:
                    entries.append(o.save_case(output,op['id'],config,fed,capture,[record],inventory=inventory))
                if observe is not None:observe(op,record,capture)
            o.require(len(sessions)<=plan['limits']['max_live_sessions'],'Live session bound exceeded')
        o.require(not sessions,'Undropped session')
        return dict(schema=SCHEMA,plan=plan,cases=entries,events=events,total_capture_bytes=total,
                    status='numerical_complete',owner_scope='fresh child; process identity in numerical-authentication.json')
    finally:
        capture.tensors.clear();capture.bytes=0
        sessions.clear()
        # Remove branch references retained by the last invocation.
        for layer in model.model.layers:
            (layer.linear_attn if layer.layer_type=='linear_attention' else layer.self_attn).caches=None


def main(argv=None):
    import os
    os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',choices=SCENARIOS,default='greedy')
    p.add_argument('--decode-steps',type=int,default=4)
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv);plan=validate_plan(build_plan(args.scenario,args.decode_steps))
    text=json.dumps(plan,indent=2)+'\n'
    if args.output:
        if not args.output.resolve().is_relative_to(HERE):raise ValueError('Plan output must stay paper-local')
        with args.output.open('x') as stream:stream.write(text)
    else:print(text,end='')
    return 0


if __name__=='__main__':raise SystemExit(main())
