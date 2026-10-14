"""Additive fixed-input geometry experiment, not a model acceptance reference.

prepare is strictly CPU-only. replay requires authenticated GPU custody and uses
unchanged pinned HF on original operands and, separately, on native QKV results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import hybrid_mixed_gdn_full40_replay as retained


def tensor_bytes(value):
    return value.detach().cpu().contiguous().view(dtype=__import__('torch').uint8).numpy().tobytes()


def metric(torch, actual, expected):
    x, y = [v.detach().float().cpu().contiguous().flatten() for v in (actual, expected)]
    assert x.shape == y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
    xb, yb = x.view(torch.int32), y.view(torch.int32)
    ix = (xb != yb).nonzero().flatten()
    return dict(elements=x.numel(), unequal=ix.numel(), max_abs=float((x-y).abs().max()),
                indices=ix.tolist(), first=[dict(index=int(i), actual_bits=f'{int(xb[i]) & 0xffffffff:08x}',
                expected_bits=f'{int(yb[i]) & 0xffffffff:08x}') for i in ix[:16]])


def prepare(args, ref):
    torch = ref.torch
    assert not torch.cuda.is_initialized()
    manifest = json.loads((args.reference / 'manifest.json').read_text())
    assert retained.sha(args.reference / 'manifest.json') == retained.ORACLE_PIN
    for name in ('source_payloads', 'operand_layout'):
        record = manifest[name]
        assert retained.sha(args.reference / record['file']) == record['sha256']
    payloads = json.loads((args.reference / manifest['source_payloads']['file']).read_text())
    layouts = json.loads((args.reference / manifest['operand_layout']['file']).read_text())
    ckpt = ref.legacy.Checkpoint(args.model)
    assert ckpt.index_hash == payloads['index_sha256']
    args.output.mkdir(exist_ok=False, parents=True)
    pins, ranks = {}, {}
    def write(name, data):
        (args.output / name).write_bytes(data)
        pins[name] = hashlib.sha256(data).hexdigest()
    for rank in range(2):
        capture, capture_pins = retained.read_events(args.capture / f'rank-{rank}')
        entry = layouts[f'rank.{rank}.model.layers.0.linear_attn.in_proj_qkv.weight']
        assert entry['axis'] == 0 and entry['shape'] == [4096, 2048]
        ranges = entry['ranges']
        assert [hi-lo for lo, hi in ranges] == [1024, 1024, 2048]
        assert all(lo % 128 == hi % 128 == 0 for lo, hi in ranges)
        raw = ckpt.read(entry['source'])
        scales = ckpt.read(entry['source'] + '_scale_inv')
        assert raw.dtype == torch.float8_e4m3fn and scales.dtype == torch.bfloat16
        full = ref.dequantize_bf16(raw, scales)
        local = torch.cat([full[lo:hi] for lo, hi in ranges]).contiguous()
        assert hashlib.sha256(tensor_bytes(local)).hexdigest() == entry['decoded_bf16_sha256']
        packed_raw = torch.cat([raw.view(torch.uint8)[lo:hi] for lo, hi in ranges]).contiguous()
        packed_scales = torch.cat([scales[lo//128:hi//128] for lo, hi in ranges]).contiguous()
        assert torch.equal(ref.dequantize_bf16(packed_raw.view(torch.float8_e4m3fn), packed_scales), local)
        prefix = f'rank{rank}.'
        write(prefix+'weight.fp8', tensor_bytes(packed_raw))
        write(prefix+'scales.bf16', tensor_bytes(packed_scales))
        write(prefix+'weight.bf16', tensor_bytes(local))
        for key in ('normalized_input', 'input.qkv', 'input.a', 'input.b', 'input.z',
                    'weight.conv', 'weight.a_log', 'weight.dt_bias', 'weight.norm', 'before.conv', 'before.ssm',
                    'convolved', 'conv_state', 'recurrent_output', 'recurrent_state'):
            value = torch.frombuffer(bytearray(capture[key]), dtype=torch.float32).clone()
            is_f32 = key in retained.F32_FIELDS
            write(prefix+key+('.f32' if is_f32 else '.bf16'), tensor_bytes(value if is_f32 else value.bfloat16()))
        ranks[str(rank)] = dict(source=entry, capture_pins=capture_pins)
    for name, record in ckpt.payloads.items():
        assert record == payloads['used_payloads'][name], name
    assert not torch.cuda.is_initialized()
    report = dict(schema='ferrule.mixed-qkv-geometry-input.v1', acceptance=False, pins=pins,
                  ranks=ranks, reference_manifest_sha256=retained.ORACLE_PIN, source_payloads=ckpt.payloads,
                  source_policy=ref.initialize(), generator_sha256=retained.sha(__file__))
    (args.output/'manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(cpu_prepared=True, files=len(pins), ranks=2)))


def replay(args, ref):
    torch, F, hf = ref.torch, ref.F, ref.hf
    assert os.environ.get('FERRULE_PAPER_WORKER_TOKEN')
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    fixture = json.loads((args.fixture/'manifest.json').read_text())
    for name, digest in fixture['pins'].items():
        assert retained.sha(args.fixture/name) == digest
    assert retained.sha(args.reference/'manifest.json') == retained.ORACLE_PIN
    from safetensors import safe_open
    from safetensors.torch import save_file
    args.output.mkdir(exist_ok=False, parents=True)
    report, tensors = {}, {}
    for rank in range(2):
        device = f'cuda:{rank}'
        def load(name, shape, native=False, f32=False):
            path = (args.native if native else args.fixture) / (f'rank{rank}.'+name+('.f32' if f32 or native else '.bf16'))
            value = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32 if f32 or native else torch.bfloat16).clone()
            return value.reshape(shape).to(device)
        x = load('normalized_input', (1, 2048))
        w = load('weight', (4096, 2048))
        conv_w = load('weight.conv', (4096, 1, 4))
        a, b = [load('input.'+name, (1, 1, 16)) for name in ('a', 'b')]
        al, dt = [load('weight.'+name, (16,)) for name in ('a_log', 'dt_bias')]
        z = load('input.z', (16, 128))
        rms = hf.Qwen3_5MoeRMSNormGated(128, eps=1e-6).to(device=device, dtype=torch.bfloat16)
        def downstream(raw):
            raw = raw.bfloat16().reshape(1, 1, 4096)
            history = F.pad(raw.transpose(1, 2), (3, 0))
            conv = F.silu(F.conv1d(raw.transpose(1, 2), conv_w, padding=3, groups=4096)[:, :, :1]).transpose(1, 2)
            q, k, v = conv.split([1024, 1024, 2048], dim=-1)
            q, k = [t.reshape(1, 1, 8, 128).repeat_interleave(2, dim=2) for t in (q, k)]
            g = -al.float().exp() * F.softplus(a.float()+dt)
            output, state = hf.torch_chunk_gated_delta_rule(q, k, v.reshape(1, 1, 16, 128), g, b.sigmoid(),
                initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
            return dict(qkv=raw, conv=conv, history=history, output=output, state=state,
                        gated=rms(output.reshape(16, 128), z))
        with torch.inference_mode():
            rms.weight.copy_(load('weight.norm', (128,)))
            projected = dict(hf_concat=F.linear(x, w),
                             hf_segmented=torch.cat([F.linear(x, p.contiguous()) for p in w.split([1024, 1024, 2048])], dim=-1))
            independent = {name: downstream(value) for name, value in projected.items()}
            with safe_open(str(args.reference/'hello.safetensors'), framework='pt', device='cpu') as archive:
                tail = archive.get_tensor(f'prefill.rank.{rank}.cache.conv_states.00')[:, :, -1:]
                archived_state = archive.get_tensor(f'prefill.rank.{rank}.cache.recurrent_states.00')
            checks = dict(hf_concat_vs_archive=metric(torch, projected['hf_concat'].reshape(1, 4096, 1), tail),
                          hf_concat_state_vs_archive=metric(torch, independent['hf_concat']['state'], archived_state),
                          hf_segmented_vs_concat=metric(torch, projected['hf_segmented'], projected['hf_concat']))
            for candidate in ('numeric_segmented', 'resident_segmented', 'numeric_concat', 'resident_concat'):
                actual = {name: load(candidate+'.'+name, (count,), native=True) for name, count in
                          [('qkv',4096),('conv',4096),('history',16384),('output',2048),('state',262144)]}
                same = downstream(actual['qkv'])
                checks[candidate] = dict(
                    vs_hf_concat={key:metric(torch,value,independent['hf_concat'][key]) for key,value in actual.items()},
                    vs_hf_segmented={key:metric(torch,value,independent['hf_segmented'][key]) for key,value in actual.items()},
                    same_input_hf={key:metric(torch,value,same[key]) for key,value in actual.items()})
                tensors.update({f'rank{rank}.{candidate}.same_input_hf.{key}':value.cpu().contiguous() for key,value in same.items()})
            tensors.update({f'rank{rank}.{name}.{key}':value.cpu().contiguous() for name, stages in independent.items() for key,value in stages.items()})
            report[str(rank)] = checks
            torch.cuda.synchronize(rank)
    save_file(tensors, str(args.output/'independent.safetensors'))
    result = dict(schema='ferrule.mixed-qkv-geometry-comparison.v1', acceptance=False, results=report,
                  source_policy=ref.initialize(), script_sha256=retained.sha(__file__),
                  fixture_manifest_sha256=retained.sha(args.fixture/'manifest.json'),
                  expected_sha256=retained.sha(args.output/'independent.safetensors'),
                  policy=dict(torch=torch.__version__, cuda=torch.version.cuda, tf32=False, reduced_bf16=False,
                              workspace=':4096:8', native_workspace=':0:0', initial_state=None, ssm='F32'))
    (args.output/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({rank:{name:({section:{k:v['unequal'] for k,v in values.items()} for section,values in check.items()}
                      if name in ('numeric_segmented','resident_segmented','numeric_concat','resident_concat') else check)
                      for name,check in checks.items()} for rank,checks in report.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare','replay'])
    for name in ('capture','reference','model','output','fixture','native'):
        parser.add_argument('--'+name, type=Path, required=name in ('reference','output'))
    args = parser.parse_args()
    if args.mode == 'prepare':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        assert os.environ.get('FERRULE_PAPER_WORKER_TOKEN') and os.environ.get('CUDA_VISIBLE_DEVICES')
    sys.path.insert(0, str(retained.ROOT/'scripts'))
    import qwen35_fp8_bf16_reference as ref
    ref.initialize()
    (prepare if args.mode == 'prepare' else replay)(args, ref)


if __name__ == '__main__':
    main()
