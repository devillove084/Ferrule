"""Bounded same-input Hello/layer0 diagnosis; never an acceptance oracle.

Inputs are immutable Ferrule operands, but every expected output is independently
computed by pinned HF/Torch. Fresh prefill and cached-update convolution dispatch
are kept distinct. No production implementation is imported or emulated.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[4]
ORACLE_PIN = 'fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d'
BF16_FIELDS = {
    'input.qkv': 4096, 'input.a': 16, 'input.b': 16, 'input.z': 2048,
    'weight.conv': 16384, 'weight.a_log': 16, 'weight.dt_bias': 16,
    'weight.norm': 128, 'before.conv': 16384, 'convolved': 4096,
    'recurrent_output': 2048, 'gated': 2048, 'conv_state': 16384,
}
F32_FIELDS = {'before.ssm': 262144, 'recurrent_state': 262144}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_events(directory):
    events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
    assert events and events[-1]['event'] == 'mixed.layer.output'
    assert all(e['layer'] in (None, 0) for e in events)
    result, pins = {}, {'events.jsonl': sha(directory / 'events.jsonl')}
    for event in events:
        file = event['file']
        assert Path(file).name == file and file.endswith('.f32')
        data = (directory / file).read_bytes()
        assert len(data) == 4 * math.prod(event['shape'])
        assert all(math.isfinite(v) for (v,) in struct.iter_unpack('<f', data))
        pins[file] = sha(directory / file)
        if event['event'].startswith('gdn.') and event['event'] != 'gdn.chunk.workspace':
            key = event['event'][4:]
            assert key not in result, key
            result[key] = data
        if event['event'] == 'norm.output' and event['role'] == 'AttentionNorm':
            assert 'normalized_input' not in result
            result['normalized_input'] = data
    for key, count in (BF16_FIELDS | F32_FIELDS | {'normalized_input': 2048}).items():
        assert len(result[key]) == count * 4, key
        if key not in F32_FIELDS:
            assert all(word & 0xffff == 0 for (word,) in struct.iter_unpack('<I', result[key])), key
    assert not any(result['before.ssm']) and not any(result['before.conv']), 'Not fresh zero-state prefill'
    return result, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--execute-cuda', action='store_true')
    args = parser.parse_args()
    if not args.execute_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        assert os.environ.get('FERRULE_PAPER_WORKER_TOKEN'), 'Authenticated child required'
        assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'Explicit devices required'
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'Pinned HF deterministic workspace policy'
    assert not args.output.exists(), 'Never overwrite receipts or references'
    assert sha(args.reference / 'manifest.json') == ORACLE_PIN
    manifest = json.loads((args.reference / 'manifest.json').read_text())
    for name in ['source_payloads', 'operand_layout']:
        record = manifest[name]
        assert sha(args.reference / record['file']) == record['sha256']
    hello = next(case for case in manifest['cases'] if case['id'] == 'hello')
    assert hello['file'] == 'hello.safetensors'
    assert sha(args.reference / hello['file']) == hello['sha256']
    payloads = json.loads((args.reference / manifest['source_payloads']['file']).read_text())
    layouts = json.loads((args.reference / manifest['operand_layout']['file']).read_text())
    captures = [read_events(args.capture / f'rank-{rank}') for rank in range(2)]
    sys.path.insert(0, str(ROOT / 'scripts'))
    import qwen35_fp8_bf16_reference as ref
    pins = ref.initialize()
    torch, F, hf = ref.torch, ref.F, ref.hf
    from safetensors import safe_open
    from safetensors.torch import save_file
    assert not torch.cuda.is_initialized()
    checkpoint = ref.legacy.Checkpoint(args.model)
    assert checkpoint.index_hash == payloads['index_sha256']
    expected_tensors, results, source_checks = {}, {}, {}

    def metric(actual, expected):
        x = actual.detach().float().cpu().contiguous().flatten()
        y = expected.detach().float().cpu().contiguous().flatten()
        assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
        xb, yb = x.view(torch.int32), y.view(torch.int32)
        ix = (xb != yb).nonzero().flatten()
        return dict(elements=x.numel(), unequal=ix.numel(), max_abs=float((x-y).abs().max()),
                    indices=ix.tolist(), first=[dict(index=int(i), actual=float(x[i]), expected=float(y[i]),
                    actual_bits=f'{int(xb[i]) & 0xffffffff:08x}', expected_bits=f'{int(yb[i]) & 0xffffffff:08x}') for i in ix[:16]])

    def source(name):
        value = checkpoint.read(name)
        assert checkpoint.payloads[name] == payloads['used_payloads'][name], name
        return value

    def local(name, rank, matrix=False):
        key = f'rank.{rank}.model.layers.0.linear_attn.{name}'
        entry = layouts[key]
        value = source(entry['source'])
        original_dtype = str(value.dtype)
        if matrix and value.dtype == torch.float8_e4m3fn:
            value = ref.dequantize_bf16(value, source(entry['source'] + '_scale_inv'))
        else:
            assert value.dtype == torch.bfloat16 or (name in ('A_log', 'norm.weight') and value.dtype == torch.float32)
            value = value.bfloat16()
        if entry['axis'] is not None:
            value = torch.cat([value.narrow(entry['axis'], lo, hi-lo) for lo, hi in entry['ranges']], dim=entry['axis'])
        value = value.contiguous()
        digest = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
        assert digest == entry['decoded_bf16_sha256'], key
        source_checks[key] = dict(source_dtype=original_dtype, resident_dtype='BF16', resident_sha256=digest,
                                  axis=entry['axis'], ranges=entry['ranges'], shape=list(value.shape))
        return value

    for rank, (capture, _) in enumerate(captures):
        def get(name, bf16=True):
            value = torch.frombuffer(bytearray(capture[name]), dtype=torch.float32).clone()
            return value.bfloat16() if bf16 else value
        vectors = {key: local(name, rank) for key, name in [
            ('conv', 'conv1d.weight'), ('a_log', 'A_log'), ('dt_bias', 'dt_bias'), ('norm', 'norm.weight')]}
        checks = {key: metric(get('weight.' + key), value) for key, value in vectors.items()}
        assert all(v['unequal'] == 0 for v in checks.values()), checks
        matrices = {key: local('in_proj_' + key + '.weight', rank, True) for key in ('qkv', 'a', 'b', 'z')}
        results[str(rank)] = dict(vector_source_checks=checks, key_heads=list(range(rank*8, (rank+1)*8)),
                                  value_heads=list(range(rank*16, (rank+1)*16)), zero_initial_state=True)
        if not args.execute_cuda:
            continue
        device = f'cuda:{rank}'
        def gpu(name, shape, bf16=True):
            return get(name, bf16).reshape(shape).to(device)
        vectors = {k: v.to(device) for k, v in vectors.items()}
        raw = gpu('input.qkv', (1, 1, 4096))
        aa, bb = [gpu('input.' + key, (1, 1, 16)) for key in ('a', 'b')]
        zz = gpu('input.z', (16, 128))
        history = gpu('before.conv', (1, 4096, 4))
        state = gpu('before.ssm', (1, 16, 128, 128), False)
        normalized = gpu('normalized_input', (1, 2048))
        stages = {}
        with torch.inference_mode():
            projections = {key: F.linear(normalized, weight.to(device)) for key, weight in matrices.items()}
            projection_checks = {key: metric(get('input.'+key), value) for key, value in projections.items()}
            # Exactly the fresh HF forward branch: padding=kernel-1, slice to sequence length.
            fresh_history = F.pad(raw.transpose(1, 2), (3, 0))
            pre_silu = F.conv1d(raw.transpose(1, 2), vectors['conv'], padding=3, groups=4096)[:, :, :1]
            conv = F.silu(pre_silu).transpose(1, 2)
            update = hf.torch_causal_conv1d_update(raw.transpose(1, 2), history, vectors['conv'].squeeze(1), None, 'silu').transpose(1, 2)
            q, k, v = conv.split([1024, 1024, 2048], dim=-1)
            q = q.reshape(1, 1, 8, 128).repeat_interleave(2, dim=2)
            k = k.reshape(1, 1, 8, 128).repeat_interleave(2, dim=2)
            v = v.reshape(1, 1, 16, 128)
            g = -vectors['a_log'].float().exp() * F.softplus(aa.float() + vectors['dt_bias'])
            beta = bb.sigmoid()
            fn = hf.torch_chunk_gated_delta_rule
            def observe(frame, event, arg):
                if frame.f_code is fn.__code__ and event == 'return':
                    for name in ['query', 'key', 'g', 'beta', 'k_beta', 'v_beta', 'k_cumdecay', 'v_new', 'attn', 'value', 'attn_inter']:
                        if name in frame.f_locals:
                            stages['chunk.' + name] = frame.f_locals[name].clone()
                return observe
            assert sys.gettrace() is None
            sys.settrace(observe)
            try:
                # Fresh HF ignores zero allocated cache: initial_state=None, not cached recurrence.
                output, final = fn(q, k, v, g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
            finally:
                sys.settrace(None)
            rms = hf.Qwen3_5MoeRMSNormGated(128, eps=1e-6).to(device=device, dtype=torch.bfloat16)
            rms.weight.copy_(vectors['norm'])
            gated = rms(output.reshape(16, 128), zz)
            same_core_gated = rms(gpu('recurrent_output', (16, 128)), zz)
            stages.update(conv=conv, pre_silu=pre_silu, history=fresh_history, output=output, state=final,
                          gated=gated, g=g, beta=beta, qnorm=hf.l2norm(q), knorm=hf.l2norm(k),
                          exp_a=vectors['a_log'].float().exp(), decay=g.exp(), initial_ssm=state)
            stages.update({'projection.' + key: value for key, value in projections.items()})
            comparisons = {key: metric(get(native, key != 'state'), stages[key]) for key, native in [
                ('history', 'conv_state'), ('conv', 'convolved'), ('output', 'recurrent_output'),
                ('state', 'recurrent_state'), ('gated', 'gated')]}
            comparisons['gate_same_core'] = metric(get('gated'), same_core_gated)
            comparisons['fresh_vs_update_conv'] = metric(conv, update)
            comparisons['fresh_vs_update_history'] = metric(fresh_history, history)
            with safe_open(str(args.reference / 'hello.safetensors'), framework='pt', device='cpu') as archive:
                archived_history = archive.get_tensor(f'prefill.rank.{rank}.cache.conv_states.00')
                comparisons['captured_qkv_vs_archived_history_tail'] = metric(raw.transpose(1, 2), archived_history[:, :, -1:])
                comparisons['independent_projection_vs_archived_history_tail'] = metric(projections['qkv'].reshape(1,4096,1), archived_history[:, :, -1:])
                comparisons['hf_same_input_vs_archived_ssm'] = metric(final, archive.get_tensor(f'prefill.rank.{rank}.cache.recurrent_states.00'))
            results[str(rank)].update(projections=projection_checks, comparisons=comparisons,
                                      gpu_name=torch.cuda.get_device_name(rank), gpu_capability=list(torch.cuda.get_device_capability(rank)))
            expected_tensors.update({f'rank{rank}.{key}': value.detach().cpu().contiguous() for key, value in stages.items()})
            torch.cuda.synchronize(rank)
        # CUDA memory belongs solely to this short-lived authenticated child.
    assert args.execute_cuda or not torch.cuda.is_initialized()
    for rank, (_, before) in enumerate(captures):
        assert read_events(args.capture / f'rank-{rank}')[1] == before
    assert sha(args.reference / 'manifest.json') == ORACLE_PIN
    args.output.mkdir(parents=True, exist_ok=False)
    if args.execute_cuda:
        save_file(expected_tensors, str(args.output / 'hf-intermediates.safetensors'))
    report = dict(schema='ferrule.mixed-layer0-same-input-hf.v1', acceptance=False,
                  mode='cuda' if args.execute_cuda else 'cpu_preflight', script_sha256=sha(__file__), source_pins=pins,
                  reference_manifest_sha256=ORACLE_PIN, capture_pins={str(r): p for r, (_, p) in enumerate(captures)},
                  source_checks=source_checks, used_source_payloads=checkpoint.payloads, results=results,
                  policy=dict(torch=torch.__version__, cuda=torch.version.cuda, tf32=False, reduced_bf16=False,
                              deterministic=True, cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                              ssm='F32', fresh_chunk=True, expected_from='unchanged HF functions on captured inputs; independent checkpoint projections'))
    if args.execute_cuda:
        report['expected_sha256'] = sha(args.output / 'hf-intermediates.safetensors')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({rank: {section: {k: {f: v[f] for f in ('elements','unequal','max_abs')} for k,v in result.get(section,{}).items()}
                            for section in ('vector_source_checks','projections','comparisons')} for rank,result in results.items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
