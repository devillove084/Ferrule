"""Bounded actual-input observation of pinned HF chunk/recurrent GDN, no model.

Run uses the existing authenticated supervisor; hf is its internal child mode.
Native replay consumes a fresh copy, never writes the retained independent fixture.
No rewritten reference, production changes, or acceptance-tolerance changes.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time

from performance import ROOT, CUDA_LIB, execute, sha, write_json

CAPTURE = ROOT / 'target/validation/capital-layer0-current-20261006'
FROZEN = CAPTURE / 'frozen-gdn'
HEADER = ROOT / 'crates/ferrule-backend/native/cuda/core/recurrent_bf16.cuh'
HF_SOURCE_SHA = '0f8ca77e875c31d448137c9ac291bed505c6b9927a85b5b8460af29d1e1ee647'


def verify():
    manifest = json.loads((FROZEN / 'inputs.json').read_text())
    handoff = json.loads((CAPTURE / 'GDN-HANDOFF.json').read_text())
    for name, pin in manifest['sha256'].items():
        assert sha(FROZEN / name) == pin, name
    for item in handoff['native_per_token_stages'].values():
        assert sha(ROOT / item['path']) == item['sha256'], item['path']
    assert sha(CAPTURE / 'scan-replay') == handoff['native_binary_sha256']
    assert sha(HEADER) == handoff['production_gdn_sha256']
    assert sha(FROZEN / 'observed.cuh') == manifest['observed_kernel_sha256']
    return manifest, handoff


def hf(out):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from safetensors.torch import save_file
    sys.path.insert(0, str(ROOT / 'scripts'))
    import qwen35_fp8_bf16_reference as ref
    pins = ref.initialize()
    assert sha(ref.hf.__file__) == HF_SOURCE_SHA
    manifest, handoff = verify()
    torch.set_num_threads(1)
    stages, metrics = {}, {}

    def load(name, dtype=torch.float32, directory=FROZEN):
        return torch.frombuffer(bytearray((directory / name).read_bytes()), dtype=dtype).clone()

    def check(name, actual, expected):
        a, b = actual.detach().float().cpu().flatten(), expected.detach().float().cpu().flatten()
        assert a.shape == b.shape, (name, a.shape, b.shape)
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
        ix = (a.view(torch.int32) != b.view(torch.int32)).nonzero().flatten()
        result = dict(elements=a.numel(), unequal=ix.numel(), max_abs=float((a-b).abs().max()),
            first=[dict(index=int(i), actual=float(a[i]), expected=float(b[i]),
                        actual_bits=hex(int(a[i].view(torch.int32)) & 0xffffffff),
                        expected_bits=hex(int(b[i].view(torch.int32)) & 0xffffffff)) for i in ix[:4]])
        metrics[name] = result
        return result['unequal']

    def exact(name, a, b):
        assert check(name, a, b) == 0, name

    def record(name, value):
        assert name not in stages, name
        stages[name] = value.detach().clone()

    def observe(fn, prefix, q, k, v, g, beta, state):
        lines, start = inspect.getsourcelines(fn)
        locations = {start+i: line.strip() for i, line in enumerate(lines)}
        norms = []
        def tracer(frame, event, arg):
            if frame.f_code is ref.hf.l2norm.__code__ and event == 'return':
                name = ['qnorm', 'knorm'][len(norms)]
                record(prefix+'.'+name, arg)
                norms.append(name)
            if frame.f_code is fn.__code__ and event == 'line':
                line, loc = locations[frame.f_lineno], frame.f_locals
                if prefix == 'chunk':
                    if line.startswith('v_beta ='):
                        record('chunk.qscaled', loc['query'])
                    elif line.startswith('g = g.reshape'):
                        for key in ['k_beta', 'v_beta']:
                            record('chunk.'+key, loc[key])
                    elif line.startswith('decay_mask ='):
                        record('chunk.g_prefix', loc['g'])
                    elif line.startswith('attn = -'):
                        record('chunk.decay_mask', loc['decay_mask'])
                    elif line.startswith('for i in range(1, chunk_size)') and 'chunk.triangular_before' not in stages:
                        record('chunk.triangular_before', loc['attn'])
                    elif line.startswith('value = attn @'):
                        record('chunk.triangular_after', loc['attn'])
                    elif line.startswith('k_cumdecay ='):
                        record('chunk.value', loc['value'])
                    elif line.startswith('last_recurrent_state = (') and 'k_cumdecay' in loc and 'core_attn_out' not in loc:
                        record('chunk.k_cumdecay', loc['k_cumdecay'])
                    elif line.startswith('v_prime ='):
                        record('chunk.qk_attn', loc['attn'])
                    elif line.startswith('attn_inter ='):
                        record('chunk.v_prime', loc['v_prime'])
                        record('chunk.v_new', loc['v_new'])
                    elif line.startswith('core_attn_out[:, :, i] ='):
                        record('chunk.attn_inter', loc['attn_inter'])
                    elif line.startswith('if not output_final_state:'):
                        record('chunk.dot', loc['core_attn_out'])
                else:
                    if line.startswith('kv_mem ='):
                        for key, source in [('decayed_state','last_recurrent_state'), ('qscaled','q_t'), ('decay','g_t')]:
                            record('recurrent.'+str(loc['i'])+'.'+key, loc[source])
                    elif line.startswith('delta ='):
                        record('recurrent.'+str(loc['i'])+'.memory', loc['kv_mem'])
                    elif line.startswith('last_recurrent_state = last_recurrent_state +'):
                        record('recurrent.'+str(loc['i'])+'.correction', loc['delta'])
                    elif line.startswith('core_attn_out[:, :, i] ='):
                        record('recurrent.'+str(loc['i'])+'.updated_state', loc['last_recurrent_state'])
                    elif line.startswith('if not output_final_state:'):
                        record('recurrent.dot', loc['core_attn_out'])
            return tracer
        class MatrixObserver(TorchDispatchMode):
            # Redispatch the original op unchanged. This exposes the actual F32
            # matmul boundary, not a diagnostic replacement for HF's matmul.
            def __init__(self):
                super().__init__()
                self.count = 0
                self.operations = set()

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                self.operations.add(str(func))
                result = func(*args, **(kwargs or {}))
                if func is torch.ops.aten.matmul.default:
                    name = prefix+'.matmul.'+str(self.count)
                    record(name+'.lhs', args[0])
                    record(name+'.rhs', args[1])
                    record(name+'.result', result)
                    self.count += 1
                return result

        assert sys.gettrace() is None
        sys.settrace(tracer)
        try:
            with MatrixObserver() as matrix_observer:
                result = fn(q, k, v, g=g, beta=beta, initial_state=state.clone(),
                            output_final_state=True, use_qk_l2norm_in_kernel=True)
            assert matrix_observer.count == (8 if prefix == 'chunk' else 0), (matrix_observer.count, sorted(matrix_observer.operations))
        finally:
            sys.settrace(None)
        assert len(norms) == 2
        record(prefix+'.output', result[0])
        record(prefix+'.state', result[1])
        assert result[1].dtype == torch.float32
        return result

    with torch.inference_mode():
        raw = load('raw.bf16', torch.bfloat16).reshape(1,5,8192).cuda()
        weight = load('weight.bf16', torch.bfloat16).reshape(8192,1,4).cuda()
        # Original prefill producer expression, unlike the cached update helper.
        conv = ref.F.silu(ref.F.conv1d(raw.transpose(1,2), weight, padding=3, groups=8192)[:, :, :5]).transpose(1,2)
        exact('producer.conv', conv, load('hf-conv.f32'))
        q, k, v = conv.split([2048,2048,4096], dim=-1)
        q = q.reshape(1,5,16,128).repeat_interleave(2, dim=2)
        k = k.reshape(1,5,16,128).repeat_interleave(2, dim=2)
        v = v.reshape(1,5,32,128)
        for name, value in [('q',q), ('k',k), ('v',v)]:
            exact('producer.'+name, value, load('hf-'+name+'.f32'))
        a = load('a.bf16', torch.bfloat16).reshape(1,5,32).cuda()
        b = load('b.bf16', torch.bfloat16).reshape(1,5,32).cuda()
        al = load('a_log.bf16', torch.bfloat16).cuda()
        dt = load('dt_bias.bf16', torch.bfloat16).cuda()
        g = -al.float().exp() * ref.F.softplus(a.float()+dt)
        beta = b.sigmoid()
        exact('producer.g', g, load('hf-g.f32'))
        exact('producer.beta', beta, load('hf-beta.f32'))
        state = load('state.f32').reshape(1,32,128,128).cuda()
        assert not torch.count_nonzero(state)
        chunk_out, chunk_state = observe(ref.hf.torch_chunk_gated_delta_rule, 'chunk', q,k,v,g,beta,state)
        exact('independent.chunk_output', chunk_out, load('hf-chunk-output.f32'))
        exact('independent.chunk_state', chunk_state, load('hf-chunk-state.f32'))
        rec_out, rec_state = observe(ref.hf.torch_recurrent_gated_delta_rule, 'recurrent', q,k,v,g,beta,state)
        check('recurrent_vs_chunk.output', rec_out, chunk_out)
        check('recurrent_vs_chunk.state', rec_state, chunk_state)
        for name in ['qnorm','knorm']:
            exact('normalization.chunk_vs_recurrent.'+name, stages['chunk.'+name], stages['recurrent.'+name])
            check('scan_vs_chunk.'+name, load('scan-'+name+'.f32'), stages['chunk.'+name])
        # Canonical token/head/column order; crop padding only, no new arithmetic.
        def tokens(x):
            return x.reshape(1,32,64,128)[:, :, :5].transpose(1,2).contiguous()
        check('scan_vs_chunk.qscaled', load('scan-qscaled.f32'), tokens(stages['chunk.qscaled']))
        for name in ['g', 'beta']:
            exact('scan_vs_producer.'+name, load('scan-'+name+'.f32'), {'g':g,'beta':beta}[name])
        for name in ['decay','decayed_state','memory','correction','qscaled']:
            native = load('scan-'+name+'.f32').reshape(5,-1)
            for t in range(5):
                check('scan_vs_recurrent.'+str(t)+'.'+name, native[t], stages['recurrent.'+str(t)+'.'+name])
        rec_dot = stages['recurrent.dot'].transpose(1,2).contiguous().reshape(5,-1)
        chunk_dot = tokens(stages['chunk.dot']).reshape(5,-1)
        for t in range(5):
            check('scan_vs_recurrent.'+str(t)+'.dot', load('scan-dot.f32').reshape(5,-1)[t], rec_dot[t])
            # With zero initial state, transformed chunk values equal serial
            # corrections over real arithmetic only; reassociation is intentional.
            check('scan_vs_chunk_reassociated.'+str(t)+'.correction', load('scan-correction.f32').reshape(5,-1)[t], tokens(stages['chunk.value']).reshape(5,-1)[t])
            check('scan_vs_chunk_reassociated.'+str(t)+'.dot', load('scan-dot.f32').reshape(5,-1)[t], chunk_dot[t])
        for name, value in [('output',chunk_out), ('state',chunk_state)]:
            check('production_vs_chunk.'+name, load('producer-'+name+'.f32'), value)
        for name, dtype in [('conv',torch.bfloat16), ('history',torch.bfloat16), ('output',torch.bfloat16), ('state',torch.float32)]:
            exact('fresh_native_vs_production.'+name, load('fresh-'+name+('.bf16' if dtype==torch.bfloat16 else '.f32'),dtype,out), load('producer-'+name+'.f32'))
        # Authenticate every fresh native trace field, not just output/state.
        native_trace = load('fresh-trace.f32', directory=out).reshape(5,10,524288)
        for index, name in enumerate(['qnorm','knorm','qscaled','g','beta','decay','decayed_state','memory','correction','dot']):
            original = load('scan-'+name+'.f32').reshape(5,-1)
            exact('fresh_trace.'+name, native_trace[:,index,:original.shape[1]], original)
        focus = {}
        for name, value in [('scan',load('scan-dot.f32')), ('recurrent',rec_dot), ('chunk',chunk_dot)]:
            x = value.float().flatten()[10476].cpu()
            focus[name] = dict(dot=float(x), bits=hex(int(x.view(torch.int32)) & 0xffffffff),
                               output_word=hex(int(x.bfloat16().view(torch.int16)) & 65535))
        save_file({k:v.cpu().contiguous() for k,v in stages.items()}, str(out/'hf-stages.safetensors'))
    write_json(out/'comparison.json', metrics)
    write_json(out/'observation.json', dict(pins=pins, hf_source_sha256=HF_SOURCE_SHA,
        handoff_sha256=sha(CAPTURE/'GDN-HANDOFF.json'), input_manifest_sha256=sha(FROZEN/'inputs.json'),
        stage_sha256=sha(out/'hf-stages.safetensors'), script_sha256=sha(__file__), focus=focus,
        stage_shapes={k:list(v.shape) for k,v in stages.items()},
        limitation='Chunk transformed values vs scan corrections are algebraically equivalent only with zero initial state; this is not an exact-stage contract.'))
    print(json.dumps(dict(focus=focus, comparisons={k:v for k,v in metrics.items() if not k.startswith('fresh_trace.')}), indent=2))


def run(out, execute_cuda):
    if not execute_cuda or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('LD_PRELOAD'):
        raise ValueError('Explicit sole GPU0 ownership and clean CUDA environment required')
    manifest, handoff = verify()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Fresh empty evidence directory required')
    sys.path.insert(0, str(ROOT/'scripts'))
    import test_vllm_qwen35 as supervisor
    env = dict(os.environ, LD_LIBRARY_PATH=CUDA_LIB+':/usr/lib64', CUDA_DEVICE_ORDER='PCI_BUS_ID')
    def snapshot():
        return {name:execute(['nvidia-smi',query,'--format=csv,noheader'],env,10)
                for name,query in [('gpu','--query-gpu=index,uuid,memory.used,utilization.gpu'),
                                   ('apps','--query-compute-apps=pid,gpu_uuid,used_gpu_memory')]}
    def idle(s):
        lines=s['gpu']['stdout'].splitlines()
        return (s['gpu']['returncode']==s['apps']['returncode']==0 and not s['apps']['stdout'].strip()
                and len(lines)==8 and all(l.split(',')[-2].strip()=='0 MiB' and l.split(',')[-1].strip()=='0 %' for l in lines))
    before = snapshot()
    write_json(out/'before.json',before)
    if not idle(before):
        raise ValueError('All eight GPUs must be idle')
    receipts = []
    try:
        prior = json.loads((ROOT/'target/validation/bf16-layer01-supported-retry-20261005/environment-preflight.json').read_text())
        for lib in prior['resolved_libraries']:
            assert sha(lib['realpath']) == lib['sha256']
        binary = CAPTURE/'scan-replay'
        ldd = execute(['ldd',binary],env,10)
        write_json(out/'ldd.json',ldd)
        assert ldd['returncode']==0 and 'not found' not in ldd['stdout']
        for name in ['raw.bf16','a.bf16','b.bf16','a_log.bf16','dt_bias.bf16','weight.bf16','history.bf16','state.f32']:
            shutil.copyfile(FROZEN/name,out/name)
            assert sha(out/name)==manifest['sha256'][name]
        write_json(out/'launch.json',dict(authenticated_libraries=prior['resolved_libraries'],
            native_binary_sha256=sha(binary), production_sha256=sha(HEADER),
            script_sha256=sha(__file__), supervisor_sha256=sha(ROOT/'scripts/test_vllm_qwen35.py'),
            subreaper_sha256=sha(ROOT/'scripts/_owned_subreaper.py'), limits_seconds=[60,120]))
        os.environ.update(LD_LIBRARY_PATH=env['LD_LIBRARY_PATH'],CUDA_DEVICE_ORDER='PCI_BUS_ID')
        commands = [('native',[str(binary),str(out),'fresh'],60),
                    ('hf',['/opt/conda310/bin/python','-B',str(Path(__file__).resolve()),'hf',str(out)],120)]
        for label, command, seconds in commands:
            if label=='hf':
                os.environ.update(LD_LIBRARY_PATH='',CUBLAS_WORKSPACE_CONFIG=':4096:8')
            result=supervisor.run_process(command,seconds)
            receipts.append(result)
            write_json(out/(label+'-supervisor.json'),result)
            if (result['returncode'] or result['timed_out'] or result.get('error') or result['cleanup_errors'] or
                result['descendant_custody']!='proven_drained' or not result['custody'].get('echild') or result.get('stdout_truncated')):
                raise RuntimeError(label+' failed; see authenticated receipt')
        verify()
        print('Actual frozen native and independent HF chunk observations complete: '+str(out))
    finally:
        after=[]
        for _ in range(3):
            time.sleep(1)
            after.append(snapshot())
        drained=bool(receipts) and all(r['descendant_custody']=='proven_drained' and r['custody'].get('echild') for r in receipts)
        released=drained and all(idle(s) for s in after)
        write_json(out/'release.json',dict(after=after,drained=drained,gpu_quiescence=all(idle(s) for s in after),release='released' if released else 'unverified'))
        if not released:
            raise RuntimeError('GPU drain/release unverified')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['run','hf'])
    parser.add_argument('output',type=Path)
    parser.add_argument('--execute-cuda',action='store_true')
    args=parser.parse_args()
    if args.mode=='run':
        run(args.output.resolve(),args.execute_cuda)
    else:
        hf(args.output.resolve())
