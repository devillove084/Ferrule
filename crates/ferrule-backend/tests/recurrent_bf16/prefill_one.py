"""Explicit one-row chunk semantics: frozen actual inputs, independent HF, no model."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from performance import ROOT, HERE, NATIVE, NVCC, TOOLCHAIN, CUDA_LIB, sha, execute, write_json

CAUSE = ROOT / 'target/validation/hello-producer-cause-20261007'
HELLO = ROOT / 'target/validation/bf16-layer01-supported-retry-20261005'
CAPITAL = ROOT / 'target/validation/capital-decode-state-20261007'


def prepare(out):
    os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void')
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    frozen = CAUSE / 'frozen-layer9'
    assert sha(frozen / 'manifest.json') == '9234657f9be9690c9defb0d81f1a92a696307cb4d2c654c615e0e3a39c37633a'
    assert sha(CAUSE / 'gdn-sameinput.safetensors') == '7934c8f2cf05e5c251a9453b53486a0be06876a10c47b6f52a1e21064b7ead90'
    m = json.loads((frozen / 'manifest.json').read_text())
    pins = {str(frozen / 'manifest.json'): sha(frozen / 'manifest.json'),
            str(CAUSE / 'gdn-sameinput.safetensors'): sha(CAUSE / 'gdn-sameinput.safetensors')}
    for name, pin in m['hashes'].items():
        assert sha(frozen / name) == pin
        pins[str(frozen / name)] = pin
    def save(path, tensor):
        path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    def load(path, dtype):
        return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).clone()
    actual = out / 'actual'; (actual / 'step0').mkdir(parents=True)
    for name in ['qkv.bf16','a.bf16','b.bf16','expected.bf16','expected-state.f32']:
        shutil.copyfile(frozen / name, actual / 'step0' / name)
        shutil.copyfile(frozen / name, actual / name)
    for name in ['a_log.bf16','dt_bias.bf16','state.f32','actual.bf16','actual-state.f32']:
        shutil.copyfile(frozen / name, actual / name)
    save(actual / 'expected-output.f32', load(frozen / 'expected.bf16', torch.bfloat16).float())
    cases = [dict(name='actual', geometry=[16,32,128,128], rows=[1], cached=False, retained=True)]
    for i, (kh, hv, dk, dv) in enumerate([(1,2,1,3),(1,2,17,9),(16,32,128,128),(1,2,511,512),(1,1,512,511)]):
        name = 'synthetic-' + str(dk); d = out / name; d.mkdir()
        gen = torch.Generator().manual_seed(2026100710 + i)
        save(d/'a_log.bf16', torch.linspace(-4,-2.5,hv).bfloat16())
        save(d/'dt_bias.bf16', torch.linspace(-.2,.2,hv).bfloat16())
        save(d/'state.f32', torch.randn(hv,dk,dv,generator=gen)*.003 + .000012345)
        for j, rows in enumerate([1,5,1]):
            step = d / ('step'+str(j)); step.mkdir()
            q=(torch.randn(rows,kh*dk,generator=gen)*.1+.2).bfloat16()
            k=(torch.randn(rows,kh*dk,generator=gen)*.02+.2).bfloat16()
            v=(torch.randn(rows,hv*dv,generator=gen)*.15).bfloat16()
            save(step/'qkv.bf16',torch.cat([q,k,v],dim=-1))
            save(step/'a.bf16',(torch.randn(rows,hv,generator=gen)*.2+j*.03125).bfloat16())
            save(step/'b.bf16',(torch.randn(rows,hv,generator=gen)*.3-j*.0625).bfloat16())
        cases.append(dict(name=name,geometry=[kh,hv,dk,dv],rows=[1,5,1],cached=False,retained=False))
    # Four exact cached layer0/1 controls, on independently retained HF operands.
    payloads=json.loads((HELLO/'hf-source-payloads.json').read_text())['used_payloads']
    for prompt, file in [('hello',HELLO/'hf-layer01.safetensors'),('capital',CAPITAL/'hf-decode-layer01.safetensors')]:
        pins[str(file)]=sha(file)
        with safe_open(str(file),framework='pt',device='cpu') as f:
            for layer in [0,1]:
                name=prompt+'-cached-L'+str(layer);d=out/name;step=d/'step0';step.mkdir(parents=True)
                prefix=prompt+'/decode.0.'+str(layer)+'.'
                get=lambda key:f.get_tensor(prefix+key)
                q=get('core.q')[:,:,::2,:].flatten(2)
                k=get('core.k')[:,:,::2,:].flatten(2)
                v=get('core.v').flatten(2)
                save(step/'qkv.bf16',torch.cat([q,k,v],dim=-1).bfloat16())
                for key in ['a','b']:
                    save(step/(key+'.bf16'),get('linear_attn.in_proj_'+key+'.output').bfloat16())
                save(d/'state.f32',get('before.recurrent_states').float())
                save(step/'expected.bf16',get('core.output').bfloat16())
                save(step/'expected-state.f32',get('after.recurrent_states').float())
                for key in ['A_log','dt_bias']:
                    param='model.language_model.layers.'+str(layer)+'.linear_attn.'+key
                    meta=payloads[param]
                    with safe_open(str(Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8')/meta['shard']),framework='pt',device='cpu') as weights:
                        value=weights.get_tensor(param)
                    import hashlib
                    assert hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==meta['sha256']
                    save(d/('a_log.bf16' if key=='A_log' else 'dt_bias.bf16'),value.bfloat16())
                cases.append(dict(name=name,geometry=[16,32,128,128],rows=[1],cached=True,retained=True))
    env=dict(os.environ,LD_LIBRARY_PATH=CUDA_LIB+':/usr/lib64')
    r=execute([NVCC,'-std=c++17','-O3','-arch=sm_86','--cudadevrt=none','--cudart=shared','-Xptxas=-v',
        '-I',NATIVE,'-I',HERE,HERE/'prefill_one.cu','-L',TOOLCHAIN/'link','-o',out/'native'],env,120)
    write_json(out/'build.json',r);assert r['returncode']==0,r['stderr']
    r=execute([out/'native','--contracts'],env,15);write_json(out/'contracts.json',r);assert r['returncode']==0,r
    files=[Path(__file__),HERE/'prefill_one.cu',HERE/'chunk.cu',HERE/'chunk_continuation.cu']
    files += [NATIVE/'core'/name for name in ['recurrent_bf16_chunk.cuh','recurrent_bf16_chunk_abi.h','recurrent_bf16.cuh','recurrent_bf16_abi.h','device.cuh','validation.cuh']]
    pins.update({str(p):sha(p) for p in files})
    artifacts={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    write_json(out/'prepared.json',dict(cases=cases,source_pins=pins,artifact_pins=artifacts))
    combined={**pins,**{str(out/p):h for p,h in artifacts.items()}}
    jobs=[dict(name='hf',argv=['/opt/conda310/bin/python','-B',str(Path(__file__).resolve()),'hf',str(out)],hf=True,seconds=120,pins=combined)]
    jobs += [dict(name=c['name'],argv=[str(out/'native'),str(out/c['name']),'cached' if c['cached'] else 'chunk',*map(str,c['geometry']),*map(str,c['rows'])],seconds=120,pins=combined) for c in cases]
    write_json(out/'jobs.json',jobs)
    print('CPU prepared:',len(cases),'cases, CUDA not run')


def hf(out):
    import torch
    sys.path.insert(0,str(ROOT/'scripts'));import qwen35_fp8_bf16_reference as ref
    pins=ref.initialize()
    report={}
    with torch.inference_mode():
        for c in json.loads((out/'prepared.json').read_text())['cases']:
            d=out/c['name'];kh,hv,dk,dv=c['geometry']
            def load(path,dtype=torch.bfloat16):
                return torch.frombuffer(bytearray(path.read_bytes()),dtype=dtype).clone().cuda()
            state=load(d/'state.f32',torch.float32).reshape(1,hv,dk,dv)
            al=load(d/'a_log.bf16');dt=load(d/'dt_bias.bf16')
            for i,rows in enumerate(c['rows']):
                step=d/('step'+str(i));qkv=load(step/'qkv.bf16').reshape(1,rows,-1)
                q,k,v=qkv.split([kh*dk,kh*dk,hv*dv],dim=-1)
                q=q.reshape(1,rows,kh,dk).repeat_interleave(hv//kh,dim=2)
                k=k.reshape(1,rows,kh,dk).repeat_interleave(hv//kh,dim=2)
                v=v.reshape(1,rows,hv,dv)
                g=-al.float().exp()*ref.F.softplus(load(step/'a.bf16').reshape(1,rows,hv).float()+dt)
                beta=load(step/'b.bf16').reshape(1,rows,hv).sigmoid()
                fn=ref.hf.torch_recurrent_gated_delta_rule if c['cached'] else ref.hf.torch_chunk_gated_delta_rule
                if c['name']=='actual':
                    recurrent,rs=ref.hf.torch_recurrent_gated_delta_rule(q,k,v,g,beta,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=True)
                    assert recurrent.cpu().contiguous().view(torch.uint8).numpy().tobytes()==(d/'actual.bf16').read_bytes()
                    assert rs.cpu().contiguous().view(torch.uint8).numpy().tobytes()==(d/'actual-state.f32').read_bytes()
                output,state=fn(q,k,v,g,beta,initial_state=state,output_final_state=True,use_qk_l2norm_in_kernel=True)
                for name,t in [('expected.bf16',output.bfloat16()),('expected-state.f32',state.float())]:
                    raw=t.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    if c['retained']:assert raw==(step/name).read_bytes(),(c['name'],name)
                    else:(step/name).write_bytes(raw)
                report[c['name']+'-'+str(i)]=dict(output_words=output.numel(),state_words=state.numel(),retained=c['retained'])
    write_json(out/'hf.json',dict(pins=pins,cases=report,retained_unchanged=True))
    print('Independent HF prepared/checked',len(report),'calls; retained actual/cached references unchanged')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','hf']);p.add_argument('directory',type=Path)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.directory.resolve())
    else:hf(a.directory.resolve())
