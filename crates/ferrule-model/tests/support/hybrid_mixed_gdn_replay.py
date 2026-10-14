"""Same-input GDN diagnosis, not a replacement oracle or model implementation.

CPU mode never initializes CUDA. CUDA mode is explicit and uses the pinned
single-owner HF CUDA policy in a separate process from CUDA12.4 model gates.
All existing fixtures are read-only. Observations use sys.settrace on unchanged
HF functions, as the established actual-HF replay does.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CPU_PIN = '98884445fb90c34747980f2c6a30161532b7a7ae59f7fe7cd218b8a3577cf2b7'
SOURCE_PIN = '39ed35183ffe3693d4988b25fc17daf2176e44ecca06bdfc081f26bdf8354fc9'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('device',choices=['cpu','cuda'])
    p.add_argument('capture',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--freeze',type=Path,help='Create a NEW profile CUDA SSM reference; never overwrite the CPU fixture')
    a=p.parse_args()
    if a.device=='cpu': os.environ['CUDA_VISIBLE_DEVICES']=''
    else: assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0,1'
    os.environ['ATEN_CPU_CAPABILITY']='avx512'
    import torch
    import torch.nn.functional as F
    from transformers.models.qwen3_5 import modeling_qwen3_5 as dense_hf
    sys.path.insert(0,str(ROOT/'scripts'))
    import qwen35_fp8_bf16_reference as ref
    pins=ref.initialize()  # Pinned eager functions/flags; does not initialize CUDA.
    hf=ref.hf
    assert sha(HERE/'hybrid_mixed_gdn_oracle.json')==CPU_PIN
    assert sha(HERE/'hybrid_mixed_gdn_oracle.py')==SOURCE_PIN
    # Establish that family choice is not an arithmetic change.
    for name in ['l2norm','torch_causal_conv1d_update','torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule']:
        assert inspect.getsource(getattr(dense_hf,name))==inspect.getsource(getattr(hf,name)),name
    module={'__file__':str(HERE/'hybrid_mixed_gdn_oracle.py')}
    source=(HERE/'hybrid_mixed_gdn_oracle.py').read_text()
    exec(compile(source.split('\nwith torch.no_grad():\n')[0],module['__file__'],'exec'),module)
    # That CPU-only module hides devices; restore visibility before explicit CUDA.
    if a.device=='cuda': os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    oracle=json.loads((HERE/'hybrid_mixed_gdn_oracle.json').read_text())
    results={}
    input_checks={}
    def bits(t):return t.detach().float().cpu().contiguous().view(torch.int32).flatten().tolist()
    def metric(x,y):
        x=x.detach().float().cpu().contiguous().flatten();y=y.detach().float().cpu().contiguous().flatten()
        assert x.shape==y.shape,(x.shape,y.shape)
        xb,yb=x.view(torch.int32),y.view(torch.int32)
        different=xb!=yb
        return dict(elements=x.numel(),unequal=int(different.sum()),max_abs=float((x-y).abs().max()),
            differences=[dict(index=int(i),actual=int(xb[i]),expected=int(yb[i])) for i in different.nonzero().flatten()])
    def read(lane,tx,name):
        d=json.loads((a.capture/f'rank{lane}-tx{tx}-{name}.json').read_text())
        return torch.tensor(d['bits'],dtype=torch.uint32).view(torch.float32)
    # Re-evaluate independent CPU fixture operands without executing its writer.
    cpu_inputs=[]
    gdn_code=module['gdn'].__code__
    def cpu_observer(frame,event,arg):
        if frame.f_code is gdn_code and event=='line':
            text=source.splitlines()[frame.f_lineno-1].strip()
            if text.startswith('fn='):
                loc=frame.f_locals
                cpu_inputs.append({k:loc[k].detach().clone() for k in ['qkv','a','b','gate','hist','state','conv']} | {'lane':loc['lane']})
        return cpu_observer
    with torch.no_grad():
        x=torch.tensor([((i*7)%19-9)/16 for i in range(16)],dtype=torch.bfloat16).reshape(2,8)
        g=module['gqa'](x)
        sys.settrace(cpu_observer)
        try:
            module['gdn'](g)
            _,state=module['gdn'](module['gqa'](x[:1]))
            module['gdn'](g[1:],state)
        finally:sys.settrace(None)
    assert len(cpu_inputs)==6
    for i,c in enumerate(cpu_inputs):
        tx=[71,72,73][i//2];lane=c['lane']
        fields={}
        for key,name in [('qkv','gdn.input.qkv'),('a','gdn.input.a'),('b','gdn.input.b'),('gate','gdn.input.z'),('conv','gdn.convolved'),('state','gdn.before.ssm')]:
            fields[name]=metric(read(lane,tx,name),c[key][0])
        for key,name in [('conv','conv'),('a_log','a_log'),('dt_bias','dt_bias')]:
            expected=(module['tensor'](2,'attention.'+name)*(1.+lane*.1)).bfloat16()
            fields['weight.'+key]=metric(read(lane,tx,'gdn.weight.'+key),expected)
        fields['weight.norm']=metric(read(lane,tx,'gdn.weight.norm'),module['weight'](2,'attention.norm'))
        input_checks[f'rank{lane}-tx{tx}']=fields
    assert all(v['unequal']==0 for fields in input_checks.values() for v in fields.values())
    frozen=json.loads(json.dumps(oracle))
    for lane in range(2):
        device=f'cuda:{lane}' if a.device=='cuda' else 'cpu'
        for tx,case in [(71,'prefill'),(72,'fresh_one'),(73,'cached_one')]:
            rows=2 if tx==71 else 1
            batch=1 if a.device=='cuda' else 8
            # Reference operands come from independent CPU projection math,
            # not Ferrule capture. The capture is only a same-input assertion.
            independent=cpu_inputs[([71,72,73].index(tx))*2+lane]
            history_before=(torch.tensor(oracle['fresh_one']['ranks'][lane]['conv'],dtype=torch.uint16)
                            .view(torch.bfloat16) if tx==73 else torch.zeros(30,dtype=torch.bfloat16))
            operands={
                'gdn.input.qkv':independent['qkv'][0], 'gdn.input.a':independent['a'][0],
                'gdn.input.b':independent['b'][0], 'gdn.before.ssm':independent['state'][0],
                'gdn.before.conv':history_before,
            }
            for name in ['conv','a_log','dt_bias']:
                operands['gdn.weight.'+name]=(module['tensor'](2,'attention.'+name)*(1.+lane*.1)).bfloat16()
            def load(name,shape,bf16=True):
                v=operands[name].reshape(shape)
                assert metric(read(lane,tx,name),v)['unequal']==0,(lane,tx,name)
                assert v.dtype==(torch.bfloat16 if bf16 else torch.float32)
                return v.to(device)
            def repeat(v):return v.repeat(batch,*([1]*(v.ndim-1)))
            raw=repeat(load('gdn.input.qkv',(1,rows,10)))
            history=repeat(load('gdn.before.conv',(1,10,3)))
            state=repeat(load('gdn.before.ssm',(1,2,3,2),False))
            aa=repeat(load('gdn.input.a',(1,rows,2)));bb=repeat(load('gdn.input.b',(1,rows,2)))
            al=load('gdn.weight.a_log',(2,));dt=load('gdn.weight.dt_bias',(2,))
            conv=hf.torch_causal_conv1d_update(raw.transpose(1,2),history,load('gdn.weight.conv',(10,3)),None,'silu').transpose(1,2)
            q,k,v=conv.split([3,3,4],dim=-1)
            q=q.reshape(batch,rows,1,3).repeat_interleave(2,dim=2)
            k=k.reshape(batch,rows,1,3).repeat_interleave(2,dim=2);v=v.reshape(batch,rows,2,2)
            exp_a=al.float().exp();sp=F.softplus(aa.float()+dt);g=-exp_a*sp;beta=bb.sigmoid()
            stages=dict(conv=conv,g=g,beta=beta,exp_a=exp_a,softplus=sp,decay=g.exp())
            fn=hf.torch_recurrent_gated_delta_rule if tx==73 else hf.torch_chunk_gated_delta_rule
            lines,start=inspect.getsourcelines(fn);locations={start+i:line.strip() for i,line in enumerate(lines)}
            def observe(frame,event,arg):
                if frame.f_code is fn.__code__ and event=='line':
                    loc=frame.f_locals;line=locations[frame.f_lineno]
                    if line.startswith('kv_mem ='): stages['decayed']=loc['last_recurrent_state'].clone()
                    if line.startswith('delta ='): stages['memory']=loc['kv_mem'].clone()
                    if line.startswith('last_recurrent_state = last_recurrent_state +'): stages['correction']=loc['delta'].clone()
                    if line.startswith('if not output_final_state:'):
                        for key in ['query','key','g','beta','k_beta','v_beta','k_cumdecay','v_new','attn','value','attn_inter']:
                            if key in loc: stages['local.'+key]=loc[key].clone()
                return observe
            with torch.no_grad():
                sys.settrace(observe)
                try:out,final=fn(q,k,v,g,beta,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=True)
                finally:sys.settrace(None)
            stages.update(qnorm=hf.l2norm(q),knorm=hf.l2norm(k),output=out,ssm=final)
            metrics={name:metric(read(lane,tx,native),stages[name][0]) for name,native in [('conv','gdn.convolved'),('output','gdn.recurrent_output'),('ssm','gdn.recurrent_state')]}
            expected=torch.tensor(oracle[case]['ranks'][lane]['ssm'],dtype=torch.int32).view(torch.float32)
            metrics['cpu_fixture_ssm']=metric(final[0],expected)
            # F64 is diagnosis only; never an acceptance reference.
            g64=-al.double().exp()*F.softplus(aa.double()+dt.double())
            stages['diagnostic.f64_decay']=g64.exp().float()
            serialized={}
            for key,t in stages.items():
                t=t if key=='exp_a' else t[:1]
                serialized[key]=dict(shape=list(t.shape),bits=bits(t))
            results[f'rank{lane}-tx{tx}']=dict(metrics=metrics,stages=serialized)
            frozen[case]['ranks'][lane]['ssm']=bits(final[0])
    report=dict(device=a.device,pins=pins,fixture_sha256=CPU_PIN,fixture_source_sha256=SOURCE_PIN,
        capture_sha256={p.name:sha(p) for p in sorted(a.capture.glob('*.json'))},
        policy=dict(tf32=False,bf16_reduced_precision_reduction=False,deterministic=True,ssm='f32',cuda_batch=1,cpu_batch=8),
        independent_input_checks=input_checks,results=results)
    assert not a.output.exists()
    report['script_sha256']=sha(__file__)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    if a.freeze:
        assert a.device=='cuda' and not a.freeze.exists()
        assert all(v['metrics'][field]['unequal']==0 for v in results.values() for field in ['conv','output','ssm'])
        frozen.update(schema='ferrule.mixed-gqa-gdn.tp2.cuda-ssm.v1',
            oracle='independent-torch-cpu-bf16-and-pinned-hf-cuda-f32-ssm',
            cpu_fixture_sha256=CPU_PIN,cpu_generator_sha256=SOURCE_PIN,
            hf_cuda_source_pins=pins,policy=report['policy'],generator_sha256=sha(__file__),
            replay_sha256=sha(a.output),gpu_compute_capability=list(torch.cuda.get_device_capability(0)))
        a.freeze.write_text(json.dumps(frozen,indent=2)+'\n')
    print(json.dumps({k:v['metrics'] for k,v in results.items()},indent=2))

if __name__=='__main__':main()
