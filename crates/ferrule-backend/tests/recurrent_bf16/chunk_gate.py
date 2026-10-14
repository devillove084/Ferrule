"""Prepare and supervise fixed-input backend chunk gates; no model execution.

Same dedicated harness/custody as cached GDN. Exact differences are reported,
never converted to a tolerance pass. Synthetic HF runs remain separate/cu129.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from performance import ROOT, HERE, NVCC, NATIVE, TOOLCHAIN, CUDA_LIB, execute, sha, write_json

FROZEN = ROOT/'target/validation/capital-layer0-current-20261006/frozen-gdn'
STAGES = ROOT/'target/validation/gdn-chunk-stages-final-20261006'
DENSE_STAGE_RECEIPT = ROOT/'target/validation/gdn-chunk-dense-stage-fixed-20261006/dense-stage-comparison.json'
DENSE_STAGE_RECEIPT_SHA256 = 'd3d4b8375eb2707a864f33b6962d4c979edeb9a8a85a050336e3ee215c529291'
# Dense65: Hv=2, C=64, Dk=16, Dv=9, two chunks (including the padded tail).
DENSE_STAGE_ELEMENTS = {
    f'chunk{chunk}-{name}': elements
    for chunk in range(2)
    for name, elements in {
        'qscaled': 2048, 'k_beta': 2048, 'v_beta': 1152,
        'g_prefix': 128, 'decay_mask': 8192, 'triangular_before': 8192,
        'triangular_after': 8192, 'matmul1': 1152, 'matmul2': 2048,
        'qk_attn': 8192, 'v_new': 1152, 'matmul4': 1152,
        'matmul5': 1152, 'g': 128, 'beta': 128,
    }.items()
}
SOURCES = [NATIVE/'core/recurrent_bf16_chunk.cuh', NATIVE/'core/recurrent_bf16_chunk_abi.h',
           NATIVE/'core/device.cuh', NATIVE/'core/validation.cuh', HERE/'chunk.cu', Path(__file__),
           ROOT/'scripts/test_vllm_qwen35.py', ROOT/'scripts/_owned_subreaper.py']


def validate_dense_stage_receipt(path=None):
    """Validate retained stage evidence independently of final output/state gates."""
    path = DENSE_STAGE_RECEIPT if path is None else Path(path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f'dense-stage receipt unavailable: {path}') from error
    digest = hashlib.sha256(payload).hexdigest()
    if digest != DENSE_STAGE_RECEIPT_SHA256:
        raise ValueError(f'dense-stage receipt SHA256 mismatch: {path}')
    receipt = json.loads(payload)
    if receipt.get('original_synthetic_hf_output_state_unchanged') is not True:
        raise ValueError('dense-stage receipt must preserve original HF output/state')
    comparisons = receipt.get('comparisons', {})
    if comparisons.keys() != DENSE_STAGE_ELEMENTS.keys():
        raise ValueError('dense-stage receipt stage inventory mismatch')
    for name, elements in DENSE_STAGE_ELEMENTS.items():
        metric = comparisons[name]
        if type(metric.get('elements')) is not int or metric['elements'] != elements:
            raise ValueError(f'dense-stage receipt element count mismatch: {name}')
        if (type(metric.get('unequal')) is not int or metric['unequal'] != 0
                or metric.get('max_abs') != 0 or metric.get('first_indices') != []):
            raise ValueError(f'dense-stage receipt is not exact: {name}')
    return dict(path=str(path), sha256=digest, stages=len(comparisons))


def prepare(out):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((FROZEN/'inputs.json').read_text())
    for name,pin in manifest['sha256'].items():
        assert sha(FROZEN/name)==pin,name
    observation=json.loads((STAGES/'observation.json').read_text())
    assert sha(STAGES/'hf-stages.safetensors')==observation['stage_sha256']
    pins={str(p):sha(p) for p in SOURCES}
    def write(p,t):
        p.write_bytes(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    actual=out/'actual';actual.mkdir()
    for name in ['a.bf16','b.bf16','a_log.bf16','dt_bias.bf16','state.f32']:
        (actual/name).write_bytes((FROZEN/name).read_bytes())
    conv=torch.frombuffer(bytearray((FROZEN/'producer-conv.f32').read_bytes()),dtype=torch.float32)
    assert torch.equal(conv,conv.bfloat16().float())
    write(actual/'qkv.bf16',conv.bfloat16())
    for name in ['output','state']:
        (actual/('expected-'+name+'.f32')).write_bytes((FROZEN/('hf-chunk-'+name+'.f32')).read_bytes())
    hf=safe_open(str(STAGES/'hf-stages.safetensors'),framework='pt',device='cpu')
    for j in range(8):
        for side in ['lhs','rhs','result']:
            write(actual/('mm'+str(j)+'-'+side+'.f32'),hf.get_tensor('chunk.matmul.'+str(j)+'.'+side))
    cases=[dict(name='actual',shape=[5,16,32,128,128],actual=True)]
    for rows,kh,hv,dk,dv in [(2,1,2,1,3),(5,2,6,4,7),(63,1,2,16,9),(64,1,2,16,9),(65,1,2,16,9),(129,1,1,64,5),(67,1,2,17,9)]:
        name='synthetic-'+str(rows);directory=out/name;directory.mkdir()
        qkv=torch.zeros(rows,2*kh*dk+hv*dv,dtype=torch.bfloat16)
        for t in range(rows):
            for h in range(kh):
                qkv[t,h*dk+t%dk]=1
                qkv[t,kh*dk+h*dk+t%dk]=1
            qkv[t,2*kh*dk:]=torch.tensor([((t+i)%7-3)/8 for i in range(hv*dv)],dtype=torch.bfloat16)
        values=dict(qkv=qkv,a=torch.zeros(rows,hv,dtype=torch.bfloat16),b=torch.zeros(rows,hv,dtype=torch.bfloat16),
                    a_log=torch.full((hv,),-128,dtype=torch.bfloat16),dt_bias=torch.zeros(hv,dtype=torch.bfloat16),
                    state=(torch.arange(hv*dk*dv).reshape(1,hv,dk,dv)%5-2).float()/64+1/1048576)
        for key,value in values.items():write(directory/(key+('.f32' if key=='state' else '.bf16')),value)
        cases.append(dict(name=name,shape=[rows,kh,hv,dk,dv],actual=False))
    # Dense nonzero-decay inputs exercise reduction order and nonzero F32 state.
    # Structured cases above additionally isolate exact tail/chunk algebra.
    gen=torch.Generator().manual_seed(20261006)
    for rows in [5,65]:
        kh,hv,dk,dv=1,2,16,9
        name='dense-'+str(rows);directory=out/name;directory.mkdir()
        values=dict(qkv=(torch.randn(rows,2*kh*dk+hv*dv,generator=gen)*0.2).bfloat16(),
                    a=(torch.randn(rows,hv,generator=gen)*0.1).bfloat16(),
                    b=(torch.randn(rows,hv,generator=gen)*0.1).bfloat16(),
                    a_log=torch.full((hv,),-1,dtype=torch.bfloat16),dt_bias=torch.zeros(hv,dtype=torch.bfloat16),
                    state=torch.randn(1,hv,dk,dv,generator=gen)*0.01)
        for key,value in values.items():write(directory/(key+('.f32' if key=='state' else '.bf16')),value)
        cases.append(dict(name=name,shape=[rows,kh,hv,dk,dv],actual=False))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',LD_LIBRARY_PATH=CUDA_LIB+':/usr/lib64')
    cmd=[NVCC,'-std=c++17','-O3','-arch=sm_86','--cudadevrt=none','--cudart=shared','-Xptxas=-v','-I',NATIVE,HERE/'chunk.cu','-L',TOOLCHAIN/'link','-o',out/'chunk']
    result=execute(cmd,env,120);write_json(out/'build.json',result)
    if result['returncode']:raise RuntimeError(result['stderr'])
    contracts=execute([out/'chunk','--contracts'],env,10);write_json(out/'contracts.json',contracts)
    assert contracts['returncode']==0,contracts
    assert all(sha(p)==pin for p,pin in pins.items())
    write_json(out/'prepared.json',dict(cases=cases,source_pins=pins,fixture_manifest_sha256=sha(FROZEN/'inputs.json'),
        stage_sha256=sha(STAGES/'hf-stages.safetensors'),artifact_pins={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    print('CPU preparation/contracts passed: '+str(out))


def hf_synthetic(out):
    import torch
    sys.path.insert(0,str(ROOT/'scripts'))
    import qwen35_fp8_bf16_reference as ref
    pins=ref.initialize()
    with torch.inference_mode():
        for case in json.loads((out/'prepared.json').read_text())['cases']:
            if case['actual']:continue
            directory=out/case['name'];rows,kh,hv,dk,dv=case['shape']
            def load(name,dtype=torch.bfloat16):
                return torch.frombuffer(bytearray((directory/(name+('.f32' if dtype==torch.float32 else '.bf16'))).read_bytes()),dtype=dtype).clone().cuda()
            q,k,v=load('qkv').reshape(1,rows,-1).split([kh*dk,kh*dk,hv*dv],dim=-1)
            q=q.reshape(1,rows,kh,dk).repeat_interleave(hv//kh,dim=2)
            k=k.reshape(1,rows,kh,dk).repeat_interleave(hv//kh,dim=2)
            v=v.reshape(1,rows,hv,dv)
            beta=load('b').reshape(1,rows,hv).sigmoid()
            g=-load('a_log').float().exp()*ref.F.softplus(load('a').reshape(1,rows,hv).float()+load('dt_bias'))
            output,state=ref.hf.torch_chunk_gated_delta_rule(q,k,v,g,beta,initial_state=load('state',torch.float32).reshape(1,hv,dk,dv),output_final_state=True,use_qk_l2norm_in_kernel=True)
            assert state.dtype==torch.float32
            for name,value in [('hf-output',output.float()),('hf-state',state)]:
                (directory/(name+'.f32')).write_bytes(value.cpu().contiguous().numpy().tobytes())
    write_json(out/'synthetic-hf.json',dict(pins=pins,source='unmodified pinned HF chunk function, no replacement reference'))


def compare(out):
    dense_receipt = validate_dense_stage_receipt()
    import torch
    from safetensors import safe_open
    torch.set_num_threads(1)
    hf=safe_open(str(STAGES/'hf-stages.safetensors'),framework='pt',device='cpu')
    def load(path,dtype=torch.float32):return torch.frombuffer(bytearray(path.read_bytes()),dtype=dtype).clone()
    def metrics(a,b):
        a=a.float().flatten();b=b.float().flatten();assert a.shape==b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        ix=(a.view(torch.int32)!=b.view(torch.int32)).nonzero().flatten()
        return dict(elements=a.numel(),unequal=ix.numel(),max_abs=float((a-b).abs().max()),first=[dict(index=int(i),actual=float(a[i]),expected=float(b[i]),actual_bits=hex(int(a[i].view(torch.int32))&0xffffffff),expected_bits=hex(int(b[i].view(torch.int32))&0xffffffff)) for i in ix[:4]])
    result={}
    for case in json.loads((out/'prepared.json').read_text())['cases']:
        d=out/case['name'];c={}
        for name in ['output','state']:
            expected=load(FROZEN/('hf-chunk-'+name+'.f32')) if case['actual'] else load(d/('hf-'+name+'.f32'))
            c[name]=metrics(load(d/('native-'+name+('.bf16' if name=='output' else '.f32')),torch.bfloat16 if name=='output' else torch.float32),expected)
        if case['actual']:
            names=['qscaled','knorm','g','beta','k_beta','v_beta','g_prefix','decay_mask','triangular_before','triangular_after','qk_attn','v_new']
            for name in names:
                expected=hf.get_tensor('chunk.'+name) if name not in ['g','beta'] else load(FROZEN/('hf-'+name+'.f32')).reshape(1,5,32).transpose(1,2)
                if name=='knorm':expected=torch.nn.functional.pad(expected.transpose(1,2).float(),(0,0,0,59))
                if name in ['g','beta']:expected=torch.nn.functional.pad(expected,(0,59))
                c[name]=metrics(load(d/('chunk0-'+name+'.f32')),expected)
            for j in range(8):
                expected=load(d/('mm'+str(j)+'-result.f32'))
                c['matmul'+str(j)]=metrics(load(d/('chunk0-matmul'+str(j)+'.f32')),expected)
                c['isolated_matmul'+str(j)]=metrics(load(d/('isolated-matmul'+str(j)+'.f32')),expected)
        result[case['name']]=c
    write_json(out/'comparison.json',result)
    required=json.loads((out/'prepared.json').read_text())['cases']
    assert all(v['unequal']==0 for c in required for v in result[c['name']].values()), 'Exact actual/synthetic gate failed; see comparison.json'
    write_json(out/'exact-gates.json',dict(all_cases_exact=True,actual_and_structured_exact=True,
        dense_observations_exact=True, dense_stage_receipt=dense_receipt,
        tolerance_used=False))
    print(json.dumps({k:{n:v for n,v in c.items() if n in ['output','state']} for k,c in result.items()},indent=2))
    return result


def run(out,evidence):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and not os.environ.get('LD_PRELOAD')
    manifest=json.loads((out/'prepared.json').read_text())
    for name,pin in manifest['source_pins'].items():assert sha(name)==pin,name
    for name,pin in manifest['artifact_pins'].items():assert sha(out/name)==pin,name
    evidence.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'scripts'))
    import test_vllm_qwen35 as supervisor
    def snapshot():
        return {k:execute(['nvidia-smi',q,'--format=csv,noheader'],dict(os.environ),10) for k,q in [('gpu','--query-gpu=index,uuid,memory.used,utilization.gpu'),('apps','--query-compute-apps=pid,gpu_uuid,used_gpu_memory')]}
    def idle(s):
        lines=s['gpu']['stdout'].splitlines()
        return s['gpu']['returncode']==s['apps']['returncode']==0 and not s['apps']['stdout'].strip() and len(lines)==8 and all(l.split(',')[-2].strip()=='0 MiB' and l.split(',')[-1].strip()=='0 %' for l in lines)
    before=snapshot();write_json(evidence/'before.json',before);assert idle(before),'All eight GPUs must be idle'
    receipts=[]
    try:
        libraries=json.loads((ROOT/'target/validation/bf16-layer01-supported-retry-20261005/environment-preflight.json').read_text())['resolved_libraries']
        for lib in libraries:assert sha(lib['realpath'])==lib['sha256']
        write_json(evidence/'launch.json',dict(prepared_sha256=sha(out/'prepared.json'),authenticated_libraries=libraries))
        commands=[('hf',['/opt/conda310/bin/python','-B',str(Path(__file__).resolve()),'hf',str(out)],120)]
        commands += [(c['name'],[str(out/'chunk'),str(out/c['name']),*map(str,c['shape'])],60) for c in manifest['cases']]
        for name,cmd,limit in commands:
            os.environ['LD_LIBRARY_PATH']='' if name=='hf' else CUDA_LIB+':/usr/lib64'
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            receipt=supervisor.run_process(cmd,limit);receipts.append(receipt);write_json(evidence/(name+'-supervisor.json'),receipt)
            if (receipt['returncode'] or receipt['timed_out'] or receipt.get('error') or receipt['cleanup_errors'] or receipt['descendant_custody']!='proven_drained' or not receipt['custody'].get('echild')):
                raise RuntimeError(name+' failed, see authenticated receipt')
        compare(out)
    finally:
        after=[]
        for _ in range(3):time.sleep(1);after.append(snapshot())
        drained=bool(receipts) and all(r['descendant_custody']=='proven_drained' and r['custody'].get('echild') for r in receipts)
        release=drained and all(idle(s) for s in after)
        write_json(evidence/'release.json',dict(drained=drained,after=after,gpu_quiescence=all(idle(s) for s in after),release='released' if release else 'unverified'))
        assert release,'Drain/release unverified'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','hf','compare']);p.add_argument('output',type=Path);p.add_argument('--evidence',type=Path);a=p.parse_args();out=a.output.resolve()
    if a.mode=='prepare':prepare(out)
    elif a.mode=='hf':hf_synthetic(out)
    elif a.mode=='compare':compare(out)
    else:
        if a.evidence is None:p.error('run requires a fresh --evidence directory and sole GPU ownership')
        run(out,a.evidence.resolve())
