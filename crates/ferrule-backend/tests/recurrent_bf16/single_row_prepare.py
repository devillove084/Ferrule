"""CPU-only preparation for the single-row fusion candidate; never runs CUDA.

Existing performance.py run remains the authenticated entry for original-scalar
versus candidate timing/stability. Extra exact-unfused and synthetic commands
are emitted as plans for the current GPU owner's authenticated supervisor.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
NATIVE = ROOT / 'crates/ferrule-backend/native/cuda'
PIN = '15a6e71ea871e2bdf37be48fa65f3afcfdb874c2e54d3847f430ee87b3a4b844'
OBSERVED_PIN = '2ed0f364975b05d978f49950890212ed01badd24b4c3f765b60f79b1743473ea'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(out, before):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'void'
    import performance as shared
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, LD_LIBRARY_PATH=shared.CUDA_LIB + ':/usr/lib64')
    def execute(label, argv, timeout=120, success=True):
        r = subprocess.run(list(map(str, argv)), env=env, capture_output=True,
                           text=True, timeout=timeout)
        report = dict(command=list(map(str, argv)), returncode=r.returncode,
                      stdout=r.stdout, stderr=r.stderr)
        (out / (label + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        if success != (r.returncode == 0):
            raise RuntimeError(label + ': ' + r.stdout + r.stderr)
        return report
    source = (NATIVE / 'core/recurrent_bf16.cuh').read_text()
    source_pin = sha(NATIVE / 'core/recurrent_bf16.cuh')
    source_files = [HERE / f for f in ['single_row_prepare.py', 'single_row_order.cpp',
        'single_row_numeric.py', 'single_row_regression.cu', 'performance.py',
        'performance.cu', 'actual_replay.cu', 'actual_replay.py', 'stability.cu',
        'reduction_matrix.cu', 'actual.sha256']]
    source_files += [NATIVE / 'core' / f for f in ['recurrent_bf16.cuh',
        'recurrent_bf16_abi.h', 'device.cuh', 'validation.cuh']]
    source_pins = {str(p): sha(p) for p in source_files}
    if sha(before) != PIN:
        raise ValueError('Requires the pinned exact unfused header')
    previous = before.read_text()
    scalar = (shared.EVIDENCE / 'baseline-source/core/recurrent_bf16.cuh').read_text()
    def kernel(s, name):
        start = s.index('__global__ void ' + name + '(')
        end = s.index('\n}', start) + 2
        return s[start:end]
    if kernel(source, 'delta_scan').replace('delta_scan(', 'delta(') != kernel(scalar, 'delta'):
        raise ValueError('Excluded/multirow scalar body changed')
    ref = source[source.index('__device__ inline float column_sum4('):source.index('// Fuse each independent')]
    oldref = previous[previous.index('__device__ inline float column_sum4('):previous.index('// A thread owns')]
    if ref.strip() != oldref.strip():
        raise ValueError('Reference reduction helper changed')
    fused = source[source.index('template <bool Update>'):source.index('// Separate entry points')]
    helpers = (ref + fused).replace('__device__ ', '').replace('float', 'Expr')
    (out / 'single_row_helpers.inc').write_text(helpers)
    compile_host = ['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Wno-unknown-pragmas',
                    '-I', out, HERE / 'single_row_order.cpp', '-o', out / 'order-proof']
    execute('order-build', compile_host)
    proof = execute('order-proof', [out / 'order-proof'])
    # Sanity-check that the structural checker rejects actual arithmetic changes.
    for name, bad in [
        ('stride', fused.replace('d += 16', 'd += 32')),
        ('merge', fused.replace('partial[0], partial[2]', 'partial[2], partial[0]')),
        ('bf16', fused.replace('bf16_round(__fmul_rn(load(qkv, k + d), kn))',
                               '__fmul_rn(load(qkv, k + d), kn)')),
    ]:
        inc = (ref + bad).replace('__device__ ', '').replace('float', 'Expr')
        (out / 'single_row_helpers.inc').write_text(inc)
        execute('negative-' + name + '-build', compile_host)
        execute('negative-' + name, [out / 'order-proof'], success=False)
    (out / 'single_row_helpers.inc').write_text(helpers)
    execute('order-final-build', compile_host)
    execute('order-final', [out / 'order-proof'])

    execute('cpu-frozen-numeric', [sys.executable, '-B', HERE / 'single_row_numeric.py',
                                  shared.EVIDENCE / 'final'])
    # Read-only reuse of shared preparation and protocols, no shared edits.
    shared.prepare(out / 'legacy')
    (out / 'prior.cuh').write_text(previous.replace('hf_bf16', 'prior_hf_bf16')
                                 .replace('ferrule_core_bf16_', 'prior_core_bf16_'))
    exact = out / 'exact-inputs'
    exact.mkdir()
    for name in ['a_log.bf16', 'dt_bias.bf16']:
        shutil.copyfile(out / 'legacy' / name, exact / name)
    for rows in [1, 5, 32]:
        shutil.copytree(out / 'legacy' / ('rows' + str(rows)), exact / ('rows' + str(rows)))
    # This second comparator is exact-unfused vs fused, so BOTH use the same
    # pinned HF expected outputs. Independent source fixtures remain read-only.
    for suffix in ['output.bf16', 'state.f32']:
        shutil.copyfile(exact / 'rows1' / ('expected-new-' + suffix),
                        exact / 'rows1' / ('expected-old-' + suffix))
    observed = shared.EVIDENCE / 'observed.cuh'
    if sha(observed) != OBSERVED_PIN:
        raise ValueError('Pinned exact instrumented reference changed')
    shutil.copyfile(observed, out / 'observed.cuh')
    actual = out / 'actual'
    actual.mkdir()
    for line in (HERE / 'actual.sha256').read_text().splitlines():
        pin, name = line.split()
        shutil.copyfile(shared.EVIDENCE / 'final' / name, actual / name)
        if sha(actual / name) != pin:
            raise ValueError('Pinned actual input mismatch')
    for name in ['hf-intermediates.safetensors', 'hf-replay.json']:
        shutil.copyfile(shared.EVIDENCE / 'final' / name, actual / name)

    # Renaming main loses C++'s implicit return-zero rule: add it explicitly
    # in the generated test-only copy, leaving the shared benchmark untouched.
    benchmark = (HERE / 'performance.cu').read_text().replace(
        'int main(int argc, char **argv)', 'int shared_performance_main(int argc, char **argv)')
    last = benchmark.rindex('}')
    benchmark = benchmark[:last] + '  return 0;\n' + benchmark[last:]
    (out / 'single_row_performance.inc').write_text(benchmark)
    resources = {}
    for name, src in [('exact-performance', HERE / 'single_row_regression.cu'),
                      ('actual-replay', HERE / 'actual_replay.cu'),
                      ('reduction-matrix', HERE / 'reduction_matrix.cu')]:
        argv = [shared.NVCC, '-std=c++17', '-O3', '-arch=sm_86', '--cudadevrt=none',
                '--cudart=shared', '-Xptxas=-v', '-I', NATIVE, '-I', out,
                src, '-L', shared.TOOLCHAIN / 'link', '-o', out / name]
        built = execute(name + '-build', argv)
        entries = re.findall(r"Function properties for ([^\n]+)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers", built['stderr'])
        resources[name] = [dict(symbol=n, stack=int(st), spill_stores=int(ss),
                                spill_loads=int(sl), registers=int(reg)) for n, st, ss, sl, reg in entries]
        if not entries or any(int(ss) or int(sl) for _, _, ss, sl, _ in entries):
            raise ValueError('Missing resource evidence or spill detected: ' + name)
    execute('exact-describe', [out / 'exact-performance', '--describe'])
    plans = {}
    runtime = {'CUDA_VISIBLE_DEVICES': '0', 'LD_LIBRARY_PATH': shared.CUDA_LIB + ':/usr/lib64'}
    for label, seconds, command in [
        ('actual', 60, [out / 'actual-replay', actual, 'fused']),
        ('actual-compare', 60, [sys.executable, '-B', HERE / 'actual_replay.py', 'compare', actual, '--label', 'fused', '--require-exact']),
        ('synthetic', 90, [out / 'exact-performance', '--synthetic']),
        ('exact-performance', 120, [out / 'exact-performance', exact]),
        ('matrix', 60, [out / 'reduction-matrix', shared.EVIDENCE / 'matrix']),
    ]:
        plan = dict(command=list(map(str, command)), timeout=seconds, environment=runtime,
                    requires_explicit_gpu_grant=True, requires_authenticated_supervisor=True)
        (out / (label + '-plan.json')).write_text(json.dumps(plan, indent=2) + '\n')
        plans[label] = plan
    if any(sha(p) != pin for p, pin in source_pins.items()):
        raise ValueError('Production source changed during prepare')
    report = dict(status='CPU_PREPARED_GPU_NOT_RUN', gpu_execution=False,
                  source_sha256=source_pin, source_pins=source_pins, unfused_sha256=PIN,
                  scalar_body_unchanged=True, ordered_dag=json.loads(proof['stdout']),
                  negative_checks=['stride', 'merge', 'bf16'], resources=resources,
                  legacy_suite=str(out / 'legacy'), plans=plans,
                  artifact_pins={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    (out / 'PREPARED.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['artifact_pins', 'plans']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--exact-unfused-header', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output, args.exact_unfused_header)
