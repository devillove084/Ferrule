"""CPU preparation / explicitly owned GPU execution of the matched GDN microbench.

Uses the existing authenticated subreaper, not a new supervisor. No installs,
model execution, GPU work in prepare, or production/header restoration.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'target/validation/gdn-actual-20261006'
CAPTURE = ROOT / 'target/validation/bf16-layer01-supported-retry-20261005/producer'
NATIVE = ROOT / 'crates/ferrule-backend/native/cuda'
TOOLCHAIN = ROOT / 'target/validation/qwen35-35b-current/toolchain'
NVCC = Path('/home/admin/wxw403883/anaconda3/envs/tao/bin/nvcc')
CUDA_LIB = '/home/admin/wxw403883/anaconda3/envs/tao/lib'
PRIOR_SHA = 'da079c6199d884173798aca67effd462c9115414bc46559dc8891d5dbbca85b2'
CAPTURE_PINS = {
    'hello/decode.0/0012.f32': '0053b7f318b1a4ec553460f40955628d7b66fc38380b1c9e163516617a8a2493',
    'hello/decode.0/0015.f32': '502fd2ef4f2d6911237bdaeafffcaadf2437ab2d5821573d83f9b2c184fdb483',
    'capital/prefill/0011.f32': '7a940a1597c09912b6de9cf67b7c126e1d93a0e13f9928bae72a61d5b328d7ce',
    'capital/prefill/0008.f32': '68eabf5ab3212506a2ff26c3f0f94cf734ac78b9ddcbcfaa54667e2681e142f8',
    'capital/prefill/0010.f32': '95b2dfda200fba288b0c0b0029d8e66a8a7d87819ed1f0ea27bb3587af731de6',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def execute(argv, env, timeout=60):
    r = subprocess.run(list(map(str, argv)), env=env, capture_output=True,
                       text=True, timeout=timeout)
    return dict(command=list(map(str, argv)), returncode=r.returncode,
                stdout=r.stdout, stderr=r.stderr)


def bf16(data):
    if len(data) % 4:
        raise ValueError('F32 extent')
    words = [v[0] for v in struct.iter_unpack('<I', data)]
    if any(v & 65535 or v & 0x7f800000 == 0x7f800000 for v in words):
        raise ValueError('Not a finite exact BF16 expansion')
    return b''.join(struct.pack('<H', v >> 16) for v in words)


def tile(data, width, rows):
    stride = width * 2
    if len(data) != 5 * stride:
        raise ValueError('Expected five captured rows')
    return b''.join(data[(i % 5)*stride:(i % 5+1)*stride] for i in range(rows))


def prepare(out):
    # Even if an owner sourced env-cuda124.sh (which exports device 0), all
    # subprocesses here remain CPU-only. No nvidia-smi or CUDA device queries.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    env['LD_LIBRARY_PATH'] = CUDA_LIB + ':/usr/lib64'
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Preparation requires an empty output directory')
    archive = EVIDENCE / 'baseline-source/core/recurrent_bf16.cuh'
    if sha(archive) != PRIOR_SHA:
        raise ValueError('Archived prechange kernel pin mismatch')
    prior = archive.read_text().replace('hf_bf16', 'prior_hf_bf16').replace(
        'ferrule_core_bf16_', 'prior_core_bf16_')
    (out / 'prior.cuh').write_text(prior)
    fixture = EVIDENCE / 'final'
    sources = [archive, HERE / 'actual.sha256', HERE / 'performance.py',
               HERE / 'performance.cu', HERE / 'stability.cu',
               ROOT / 'scripts/test_vllm_qwen35.py', ROOT / 'scripts/_owned_subreaper.py',
               TOOLCHAIN / 'env-cuda124.sh']
    # Pin GDN's direct implementation/header dependencies, not unrelated GEMMs.
    sources += [NATIVE / ('core/' + f) for f in
                ['recurrent_bf16.cuh', 'recurrent_bf16_abi.h', 'device.cuh', 'validation.cuh']]
    for line in (HERE / 'actual.sha256').read_text().splitlines():
        pin, name = line.split()
        if sha(fixture / name) != pin:
            raise ValueError('Actual/HF fixture changed: ' + name)
        sources.append(fixture / name)
    for name, pin in CAPTURE_PINS.items():
        if sha(CAPTURE / name) != pin:
            raise ValueError('Captured producer payload changed: ' + name)
        sources.append(CAPTURE / name)
    source_pins = {str(p): sha(p) for p in sources}
    for name in ['a_log.bf16', 'dt_bias.bf16']:
        (out / name).write_bytes((fixture / name).read_bytes())
    for rows in [1, 5, 32]:
        directory = out / ('rows' + str(rows))
        directory.mkdir()
        if rows == 1:
            payload = dict(qkv=bf16((fixture / 'expected-conv.f32').read_bytes()),
                           a=(fixture / 'a.bf16').read_bytes(),
                           b=(fixture / 'b.bf16').read_bytes())
            initial = (fixture / 'state.f32').read_bytes()
            for version, base, output, state in [
                    ('old', CAPTURE / 'hello/decode.0', '0012.f32', '0015.f32'),
                    ('new', fixture, 'expected-output.f32', 'expected-state.f32')]:
                (directory / ('expected-'+version+'-output.bf16')).write_bytes(
                    bf16((base / output).read_bytes()))
                (directory / ('expected-'+version+'-state.f32')).write_bytes((base / state).read_bytes())
        else:
            base = CAPTURE / 'capital/prefill'
            payload = {name: tile(bf16((base / file).read_bytes()), width, rows)
                       for name, file, width in [('qkv', '0011.f32', 8192),
                                                 ('a', '0008.f32', 32), ('b', '0010.f32', 32)]}
            initial = bytes(32 * 128 * 128 * 4)
        for name, data in payload.items():
            (directory / (name+'.bf16')).write_bytes(data)
        (directory / 'state.f32').write_bytes(initial)
    version = execute([NVCC, '--version'], env)
    if version['returncode'] or 'V12.4.99' not in version['stdout']:
        raise ValueError('Requires validated CUDA12.4.99 compiler')
    write_json(out / 'compiler.json', version)
    builds = {}
    for name in ['stability', 'performance']:
        argv = [NVCC, '-std=c++17', '-O3', '-arch=sm_86', '--cudadevrt=none',
                '--cudart=shared', '-Xptxas=-v', '-I', NATIVE, '-I', out,
                HERE / (name+'.cu'), '-L', TOOLCHAIN / 'link', '-o', out / name]
        result = execute(argv, env)
        write_json(out / (name+'-build.json'), result)
        (out / (name+'-ptxas.log')).write_text(result['stderr'])
        if result['returncode']:
            raise RuntimeError(result['stderr'])
        builds[name] = result
    # Host-only CLI paths: neither invokes a CUDA API or tests a CUDA device.
    listing = execute([out / 'stability', '--list-cases'], env)
    lines = listing['stdout'].splitlines()
    tuples = [tuple(map(int, line.split())) for line in lines[:-1]]
    excluded = [c for c in tuples if c[0] == 1 and c[3] % 4 == 0 and 64 <= c[2] <= 512 and c[1]*c[3] < 128]
    if listing['returncode'] or len(tuples) != 107 or len(excluded) != 10:
        raise ValueError('Missing below-threshold stability cases')
    description = execute([out / 'performance', '--describe'], env)
    if description['returncode']:
        raise ValueError('Benchmark description failed')
    write_json(out / 'cpu-checks.json', dict(stability=listing, benchmark=description,
                                           below_threshold_cases=excluded, gpus_hidden=True))
    if any(sha(p) != pin for p, pin in source_pins.items()):
        raise ValueError('Source/input changed during prepare')
    pins = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    write_json(out / 'prepared.json', dict(schema='ferrule.gdn-matched-performance.v1',
        status='prepared_not_executed', gpu_execution=False, archived_kernel_sha256=PRIOR_SHA,
        source_pins=source_pins, artifact_pins=pins,
        protocol=json.loads(description['stdout']),
        inputs={'rows1': 'actual Hello/decode.0/L0, original F32 state',
                'rows5': 'actual capital/prefill/L0 convolved QKV/A/B, zero initial F32 state',
                'rows32': 'five captured capital rows cyclically tiled to 32, zero initial F32 state; not a real 32-token capture'},
        timing_scope='GDN launch only; no conv/projections/full40; reset before every event, same buffer addresses',
        limits={'stability_seconds': 60, 'benchmark_seconds': 120},
        source_unchanged_during_prepare=True))
    print(json.dumps(dict(status='prepared_not_executed', output=str(out),
                         stability_cases=len(tuples), below_threshold_cases=len(excluded),
                         gpu_execution=False), indent=2))


def run(out, evidence, execute_cuda):
    if not execute_cuda or os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Requires explicit --execute-cuda and sole ownership of visible GPU0')
    if os.environ.get('LD_PRELOAD'):
        raise ValueError('Refuse LD_PRELOAD')
    manifest = json.loads((out / 'prepared.json').read_text())
    for file, pin in manifest['source_pins'].items():
        if sha(file) != pin:
            raise ValueError('Source/input changed since prepare: ' + file)
    for file, pin in manifest['artifact_pins'].items():
        if sha(out / file) != pin:
            raise ValueError('Prepared artifact changed: ' + file)
    evidence.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / 'scripts'))
    import test_vllm_qwen35 as supervisor
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = CUDA_LIB + ':/usr/lib64'
    os.environ['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH']
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    def snapshot():
        return {name: execute(['nvidia-smi', query, '--format=csv,noheader'], env, 10)
                for name, query in [('gpu', '--query-gpu=index,uuid,name,driver_version,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw,memory.used,utilization.gpu'),
                                    ('apps', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory')]}
    def idle(s):
        lines = s['gpu']['stdout'].splitlines()
        return (s['gpu']['returncode'] == s['apps']['returncode'] == 0 and
                not s['apps']['stdout'].strip() and len(lines) == 8 and
                all(l.split(',')[-2].strip() == '0 MiB' and l.split(',')[-1].strip() == '0 %' for l in lines))
    before = snapshot()
    write_json(evidence / 'before.json', before)
    if not idle(before):
        raise ValueError('All GPUs must be idle/released before exclusive measurement')
    results = {}
    try:
        ldd = execute(['ldd', out / 'performance'], env, 10)
        write_json(evidence / 'ldd.json', ldd)
        if ldd['returncode'] or 'not found' in ldd['stdout']:
            raise ValueError('Unresolved native dependencies')
        prior = json.loads((ROOT / 'target/validation/bf16-layer01-supported-retry-20261005/environment-preflight.json').read_text())
        for lib in prior['resolved_libraries']:
            if sha(lib['realpath']) != lib['sha256']:
                raise ValueError('CUDA library authentication failed')
        write_json(evidence / 'launch.json', dict(prepared_sha256=sha(out/'prepared.json'),
            authenticated_libraries=prior['resolved_libraries'], protocol=manifest['protocol'],
            note='Runtime attributes report registers/local bytes; spill stores/loads are in prepared PTXAS logs'))
        for name, seconds, args in [('stability', 60, []), ('performance', 120, [str(out)])]:
            result = supervisor.run_process([str(out/name), *args], seconds)
            results[name] = result
            write_json(evidence / (name+'-supervisor.json'), result)
            (evidence / (name+'.stdout')).write_text(result['stdout'])
            if (result['returncode'] != 0 or result['timed_out'] or result.get('error') or
                    result['cleanup_errors'] or result['descendant_custody'] != 'proven_drained' or
                    not result['custody'].get('echild') or result.get('stdout_truncated')):
                raise RuntimeError('Execution/drain failed: ' + name)
        rows = [json.loads(line) for line in results['performance']['stdout'].splitlines()]
        paired = [r for r in rows if r.get('event') == 'paired_sample']
        correctness = [r for r in rows if r.get('event') == 'correctness']
        if len(paired) != 90 or [r['rows'] for r in correctness] != [1, 5, 32]:
            raise ValueError('Incomplete benchmark')
        write_json(evidence / 'timing.json', dict(resources=[r for r in rows if r.get('event') == 'resources'],
            summaries=[r for r in rows if r.get('event') == 'summary'], paired_samples=paired,
            correctness=correctness, performance_claim='Bounded matched GDN operator only; no full40/serving claim'))
        print(json.dumps([r for r in rows if r.get('event') in ['summary', 'resources', 'correctness']], indent=2))
    finally:
        after = []
        for _ in range(3):
            time.sleep(1)
            after.append(snapshot())
        drained = all(r['descendant_custody'] == 'proven_drained' and r['custody'].get('echild') for r in results.values())
        write_json(evidence / 'release.json', dict(after=after, drained=drained,
            gpu_quiescence=all(idle(s) for s in after),
            release='released' if drained and all(idle(s) for s in after) else 'unverified'))
        if not drained or not all(idle(s) for s in after):
            raise RuntimeError('Drain/GPU release not proven')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--evidence', type=Path)
    parser.add_argument('--execute-cuda', action='store_true')
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.mode == 'prepare':
        prepare(directory)
    else:
        if args.evidence is None:
            parser.error('run requires a fresh --evidence directory')
        run(directory, args.evidence.resolve(), args.execute_cuda)
