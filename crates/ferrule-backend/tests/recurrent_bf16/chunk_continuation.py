"""Additional chunk gates using chunk_gate fixtures and the existing supervisor.

prepare/test are CPU-only. run explicitly owns the GPU; independent new HF
expectations are generated then, never fabricated during CPU preparation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

from performance import ROOT, HERE, NATIVE, NVCC, TOOLCHAIN, CUDA_LIB, execute, sha, write_json

ACTUAL = ROOT / 'target/validation/gdn-chunk-backend-release-20261006/actual'
INPUTS = ('qkv.bf16', 'a.bf16', 'b.bf16')
SOURCES = [Path(__file__), HERE/'chunk_continuation.cu', HERE/'chunk.cu', HERE/'chunk_gate.py',
           HERE/'performance.py', HERE/'chunk.sha256', ROOT/'scripts/qwen35_fp8_bf16_reference.py',
           ROOT/'scripts/qwen35_fp8_reference.py', ROOT/'scripts/test_vllm_qwen35.py',
           ROOT/'scripts/_owned_subreaper.py'] + [NATIVE/'core'/name for name in (
               'recurrent_bf16_chunk.cuh', 'recurrent_bf16_chunk_abi.h', 'recurrent_bf16.cuh',
               'recurrent_bf16_abi.h', 'device.cuh', 'validation.cuh')]


def cases():
    return [
        dict(name='changed-dense', rows=[5, 65, 129], geometry=[1, 2, 16, 9], actual=False),
        dict(name='near-511-512', rows=[2, 3], geometry=[1, 2, 511, 512], actual=False),
        dict(name='near-512-511', rows=[2, 5], geometry=[1, 1, 512, 511], actual=False),
        dict(name='limit-512-512', rows=[2, 3], geometry=[1, 1, 512, 512], actual=False),
        dict(name='actual5', rows=[5], geometry=[16, 32, 128, 128], actual=True, performance=True),
        dict(name='dense64', rows=[64], geometry=[16, 32, 128, 128], actual=False, performance=True),
        dict(name='dense129', rows=[129], geometry=[16, 32, 128, 128], actual=False, performance=True),
    ]


def workspace_elements(geometry):
    kh, hv, dk, dv = geometry
    if not (0 < kh <= hv <= 65535 and hv % kh == 0 and 0 < dk <= 512 and 0 < dv <= 512):
        raise ValueError('unsupported geometry')
    elements = hv * (6*64*dk + 7*64*dv + 4*64*64 + 3*64 + dk*dv)
    if elements*4 > 2147483647:
        raise ValueError('workspace extent')
    return elements


def verify(prepared):
    m = json.loads((prepared/'prepared.json').read_text())
    for path, pin in m['source_pins'].items():
        if sha(path) != pin:
            raise ValueError('source changed; reprepare: '+path)
    for path, pin in m['artifact_pins'].items():
        if sha(prepared/path) != pin:
            raise ValueError('prepared artifact changed: '+path)
    return m


def native_command(prepared, evidence, case, performance=False):
    return [prepared/'chunk_continuation', 'performance' if performance else 'continuation',
            prepared/case['name'], evidence, *case['geometry'], *case['rows']]


def prepare(out):
    # The shared machinery also keeps compilation/contracts device-free.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'void'
    import torch
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    source_pins = {str(p): sha(p) for p in SOURCES}
    for line in (HERE/'chunk.sha256').read_text().splitlines():
        pin, name = line.split()
        if sha(ACTUAL/name) != pin:
            raise ValueError('original actual fixture pin: '+name)
        source_pins[str(ACTUAL/name)] = pin
    def write(path, tensor):
        path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    for number, case in enumerate(cases()):
        d = out/case['name']; d.mkdir()
        kh, hv, dk, dv = case['geometry']
        workspace_elements(case['geometry'])
        if case['actual']:
            for name in ('a_log.bf16', 'dt_bias.bf16', 'state.f32'):
                shutil.copyfile(ACTUAL/name, d/name)
            shutil.copyfile(ACTUAL/'state.f32', d/'reset-state.f32')
            (d/'step0').mkdir()
            for name in INPUTS:
                shutil.copyfile(ACTUAL/name, d/'step0'/name)
            # Only the existing independently retained actual expectations exist now.
            for name in ('output', 'state'):
                shutil.copyfile(ACTUAL/('expected-'+name+'.f32'), d/('retained-'+name+'.f32'))
            continue
        gen = torch.Generator().manual_seed(2026100600+number)
        # Alternating-sign non-BF16 F32 state: near-cancelling key-axis terms,
        # not zero state, identity Q/K, or zero/underflowed decay fixtures.
        signs = (1 - 2*(torch.arange(dk) % 2)).reshape(1, 1, dk, 1)
        magnitudes = 0.125 + torch.rand(1, hv, 1, dv, generator=gen)*0.375
        state = signs*magnitudes + torch.randn(1, hv, dk, dv, generator=gen)*0.0003
        reset_state = state*(-0.75) + 0.0001234567
        write(d/'state.f32', state.float()); write(d/'reset-state.f32', reset_state.float())
        write(d/'a_log.bf16', torch.linspace(-4.0, -2.5, hv).bfloat16())
        write(d/'dt_bias.bf16', torch.linspace(-0.2, 0.2, hv).bfloat16())
        for i, rows in enumerate(case['rows']):
            step = d/('step'+str(i)); step.mkdir()
            # Dense keys around a common positive component stress cancellation.
            q = (torch.randn(rows, kh*dk, generator=gen)*0.1 + 0.2).bfloat16()
            k = (torch.randn(rows, kh*dk, generator=gen)*0.02 + 0.2).bfloat16()
            v = (torch.randn(rows, hv*dv, generator=gen)*0.15).bfloat16()
            write(step/'qkv.bf16', torch.cat([q,k,v], dim=-1))
            write(step/'a.bf16', (torch.randn(rows,hv,generator=gen)*0.2+i*0.03125).bfloat16())
            write(step/'b.bf16', (torch.randn(rows,hv,generator=gen)*0.3-i*0.0625).bfloat16())
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void', LD_LIBRARY_PATH=CUDA_LIB+':/usr/lib64')
    command = [NVCC, '-std=c++17', '-O3', '-arch=sm_86', '--cudadevrt=none', '--cudart=shared',
               '-Xptxas=-v', '-I', NATIVE, '-I', HERE, HERE/'chunk_continuation.cu',
               '-L', TOOLCHAIN/'link', '-o', out/'chunk_continuation']
    build = execute(command, env, 120); write_json(out/'build.json', build)
    if build['returncode']:
        raise RuntimeError(build['stderr'])
    contracts = execute([out/'chunk_continuation', '--contracts'], env, 15)
    write_json(out/'contracts.json', contracts)
    if contracts['returncode']:
        raise RuntimeError(contracts['stderr'])
    for path, pin in source_pins.items():
        if sha(path) != pin:
            raise ValueError('source changed during preparation: '+path)
    write_json(out/'prepared.json', dict(status='CPU_PREPARED_GPU_NOT_RUN', cases=cases(),
        source_pins=source_pins, artifact_pins={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},
        independent_expected_status='New synthetic/carry/reset HF expectations NOT generated; GPU owner required',
        native_runtime='CUDA12.4 sm86', hf_runtime='Pinned Torch2.13 CUDA12.9/Transformers5.5.3 in separate process',
        timing_policy='Different numerical profiles; event timings plus separate errors, never same-quality speedup',
        workspace={c['name']:workspace_elements(c['geometry'])*4 for c in cases()}))
    for path in out.rglob('*'):
        if path.is_file():
            with path.open('rb') as f: os.fsync(f.fileno())
    for path in [p for p in out.rglob('*') if p.is_dir()]+[out]:
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    print('CPU fixture/build/contracts prepared; no new HF or GPU result: '+str(out))


def hf_worker(prepared, expected):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('FERRULE_CHUNK_CONTINUATION_HF_WORKER') != '1':
        raise RuntimeError('internal worker: use explicitly owned supervised run')
    manifest = verify(prepared)
    import torch
    sys.path.insert(0, str(ROOT/'scripts'))
    import qwen35_fp8_bf16_reference as ref
    pins = ref.initialize()
    expected.mkdir(exist_ok=False)
    def load(path, dtype=torch.bfloat16):
        return torch.frombuffer(bytearray(path.read_bytes()), dtype=dtype).clone().cuda()
    def write(path, tensor):
        path.write_bytes(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    with torch.inference_mode():
        for case in manifest['cases']:
            src = prepared/case['name']; dst = expected/case['name']; dst.mkdir()
            if case['actual']:
                for prefix in ['carry-0', 'reset']:
                    for name in ['output', 'state']:
                        shutil.copyfile(src/('retained-'+name+'.f32'), dst/(prefix+'-'+name+'.f32'))
                continue
            kh,hv,dk,dv = case['geometry']
            al=load(src/'a_log.bf16'); dt=load(src/'dt_bias.bf16')
            state=load(src/'state.f32',torch.float32).reshape(1,hv,dk,dv)
            def forward(i, state):
                rows=case['rows'][i]; step=src/('step'+str(i))
                q,k,v=load(step/'qkv.bf16').reshape(1,rows,-1).split([kh*dk,kh*dk,hv*dv],dim=-1)
                q=q.reshape(1,rows,kh,dk).repeat_interleave(hv//kh,dim=2)
                k=k.reshape(1,rows,kh,dk).repeat_interleave(hv//kh,dim=2)
                v=v.reshape(1,rows,hv,dv)
                beta=load(step/'b.bf16').reshape(1,rows,hv).sigmoid()
                g=-al.float().exp()*ref.F.softplus(load(step/'a.bf16').reshape(1,rows,hv).float()+dt)
                output,next_state=ref.hf.torch_chunk_gated_delta_rule(q,k,v,g,beta,
                    initial_state=state, output_final_state=True,use_qk_l2norm_in_kernel=True)
                assert next_state.dtype==torch.float32
                return output,next_state
            for i in range(len(case['rows'])):
                output,state=forward(i,state)
                write(dst/('carry-'+str(i)+'-output.f32'),output)
                write(dst/('carry-'+str(i)+'-state.f32'),state)
            output,state=forward(0,load(src/'reset-state.f32',torch.float32).reshape(1,hv,dk,dv))
            write(dst/'reset-output.f32',output);write(dst/'reset-state.f32',state)
    write_json(expected/'manifest.json',dict(source='unmodified independently pinned native HF chunk function',
        pins=pins,prepared_sha256=sha(prepared/'prepared.json'),native_results_used=False,
        artifacts={str(p.relative_to(expected)):sha(p) for p in expected.rglob('*.f32')}))


def metric(actual, expected):
    import numpy as np
    a=np.asarray(actual,dtype=np.float32).reshape(-1);b=np.asarray(expected,dtype=np.float32).reshape(-1)
    if a.shape != b.shape or not a.size:
        raise ValueError('comparison extent')
    different=np.flatnonzero(a.view(np.uint32)!=b.view(np.uint32))
    nonfinite=int((~np.isfinite(a) | ~np.isfinite(b)).sum())
    return dict(elements=a.size,unequal=int(different.size),nonfinite=nonfinite,
        max_abs=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),first_indices=different[:8].tolist())


def compare(prepared, evidence, performance=False):
    import numpy as np
    manifest=verify(prepared); expected=evidence/'expected'
    oracle=json.loads((expected/'manifest.json').read_text())
    if oracle['native_results_used'] or oracle['prepared_sha256'] != sha(prepared/'prepared.json'):
        raise ValueError('independent expected provenance')
    for name,pin in oracle['artifacts'].items():
        if sha(expected/name) != pin: raise ValueError('expected changed: '+name)
    def load(path):
        if path.suffix=='.bf16':
            return (np.fromfile(path,dtype='<u2').astype(np.uint32)<<16).view(np.float32)
        return np.fromfile(path,dtype='<f4')
    result={}
    for case in manifest['cases']:
        name=case['name'];d=evidence/name;metrics={}
        for prefix in [*['carry-'+str(i) for i in range(len(case['rows']))], 'reset']:
            for field in ['output','state']:
                metrics[prefix+'-'+field]=metric(load(d/(prefix+'-'+field+('.bf16' if field=='output' else '.f32'))),
                    load(expected/name/(prefix+'-'+field+'.f32')))
        result[name]=metrics
    exact=all(m['unequal']==0 and m['nonfinite']==0 for c in result.values() for m in c.values())
    comparison = dict(cases=result,exact=exact,tolerances=False)
    if (evidence/'comparison.json').exists():
        if json.loads((evidence/'comparison.json').read_text()) != comparison:
            raise ValueError('immutable comparison changed')
    else:
        write_json(evidence/'comparison.json',comparison)
    if not exact: raise RuntimeError('Exact continuation/reset gate failed; no tolerance fallback, inspect comparison.json')
    if performance:
        perf={}
        for case in [c for c in manifest['cases'] if c.get('performance')]:
            name=case['name'];d=evidence/('perf-'+name);errors={}
            for variant in ['scan','chunk']:
                errors[variant]={field:metric(load(d/(variant+'-'+field+('.bf16' if field=='output' else '.f32'))),
                    load(expected/name/('carry-0-'+field+'.f32'))) for field in ['output','state']}
            receipt=json.loads((evidence/('perf-'+name+'-supervisor.json')).read_text())
            samples=[json.loads(line) for line in receipt['stdout'].splitlines() if line.startswith('{')]
            times={v:[s['microseconds'] for s in samples if s.get('event')=='timing' and s['variant']==v] for v in ['scan','chunk']}
            if any(len(v)!=20 for v in times.values()): raise ValueError('incomplete timed samples')
            if any(m['unequal'] or m['nonfinite'] for m in errors['chunk'].values()):
                raise RuntimeError('timed chunk differs from independently pinned HF')
            perf[name]=dict(numerical_errors=errors,raw_samples=samples,median_us={k:statistics.median(v) for k,v in times.items()},
                interpretation='Different numerical profiles. Not an equal-quality speed comparison. Allocations/state resets/checks excluded from events.')
        write_json(evidence/'performance.json',perf)
    return result


def run(prepared,evidence,performance=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or os.environ.get('LD_PRELOAD'):
        raise RuntimeError('requires explicitly owned GPU 0, no LD_PRELOAD')
    manifest=verify(prepared);evidence.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'scripts'));import test_vllm_qwen35 as supervisor
    def snapshot():
        return {k:execute(['nvidia-smi',q,'--format=csv,noheader'],dict(os.environ),10) for k,q in [
            ('gpu','--query-gpu=index,uuid,memory.used,utilization.gpu'),('apps','--query-compute-apps=pid,gpu_uuid,used_gpu_memory')]}
    def idle(s):
        return (s['gpu']['returncode']==s['apps']['returncode']==0 and not s['apps']['stdout'].strip()
            and len(s['gpu']['stdout'].splitlines())==8 and all(l.split(',')[-2].strip()=='0 MiB'
            and l.split(',')[-1].strip()=='0 %' for l in s['gpu']['stdout'].splitlines()))
    def custody(r):
        h,c=r.get('handshake',{}),r.get('custody',{})
        return (h.get('containment_verified') is True and c.get('containment_verified') is True
            and c.get('status')=='proven_drained' and c.get('echild') is True
            and h.get('launch_id')==c.get('launch_id') and h.get('helper_identity')==c.get('helper_identity')
            and not r.get('cleanup_errors') and not r.get('error') and not r.get('reporting_errors'))
    before=snapshot();write_json(evidence/'before.json',before)
    if not idle(before): raise RuntimeError('all GPUs must be idle before ownership cycle')
    receipts=[]
    try:
        libraries=json.loads((ROOT/'target/validation/bf16-layer01-supported-retry-20261005/environment-preflight.json').read_text())['resolved_libraries']
        for lib in libraries:
            if sha(lib['realpath'])!=lib['sha256']: raise ValueError('authenticated library changed')
        link=execute(['ldd',prepared/'chunk_continuation'],dict(os.environ,LD_LIBRARY_PATH=CUDA_LIB+':/usr/lib64'),15)
        if link['returncode'] or 'not found' in link['stdout']: raise ValueError('native link preflight')
        elf=execute([NVCC.parent/'cuobjdump','--list-elf',prepared/'chunk_continuation'],dict(os.environ),30)
        if elf['returncode'] or 'sm_86' not in elf['stdout']: raise ValueError('native sm86 preflight')
        write_json(evidence/'launch.json',dict(prepared_sha256=sha(prepared/'prepared.json'),libraries=libraries,ldd=link,elf=elf,performance=performance))
        def execute_owned(name,cmd,timeout,hf=False):
            os.environ['LD_LIBRARY_PATH']='' if hf else CUDA_LIB+':/usr/lib64'
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            os.environ['FERRULE_CHUNK_CONTINUATION_HF_WORKER']='1' if hf else '0'
            r=supervisor.run_process(list(map(str,cmd)),timeout);receipts.append(r);write_json(evidence/(name+'-supervisor.json'),r)
            if r['returncode'] or r['timed_out'] or not custody(r): raise RuntimeError(name+' failed; receipt preserved')
        execute_owned('hf',['/opt/conda310/bin/python','-B',Path(__file__).resolve(),'hf',prepared,'--evidence',evidence],120,True)
        for case in manifest['cases']:
            d=evidence/case['name'];d.mkdir()
            execute_owned(case['name'],native_command(prepared,d,case),120)
        # Exact gate before optional timing. Its result file is immutable.
        compare(prepared,evidence)
        if performance:
            for case in [c for c in manifest['cases'] if c.get('performance')]:
                d=evidence/('perf-'+case['name']);d.mkdir()
                execute_owned('perf-'+case['name'],native_command(prepared,d,case,True),120)
            compare(prepared,evidence,True)
        verify(prepared)
    finally:
        after=[]
        for _ in range(3):time.sleep(1);after.append(snapshot())
        drained=bool(receipts) and all(custody(r) for r in receipts)
        release=drained and all(idle(s) for s in after)
        write_json(evidence/'release.json',dict(drained=drained,after=after,release='released' if release else 'unverified'))
        for p in evidence.rglob('*'):
            if p.is_file():
                with p.open('rb') as f:os.fsync(f.fileno())
        for p in [p for p in evidence.rglob('*') if p.is_dir()]+[evidence]:
            fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
        if not release: raise RuntimeError('custody or GPU release unverified')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','hf','compare'])
    p.add_argument('prepared',type=Path);p.add_argument('--evidence',type=Path);p.add_argument('--performance',action='store_true')
    a=p.parse_args();prepared=a.prepared.resolve()
    if a.mode=='prepare':prepare(prepared)
    else:
        if a.evidence is None:p.error('fresh --evidence is required')
        evidence=a.evidence.resolve()
        if a.mode=='hf':hf_worker(prepared,evidence/'expected')
        elif a.mode=='compare':compare(prepared,evidence,a.performance)
        else:run(prepared,evidence,a.performance)
