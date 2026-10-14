"""Short, pinned, equal-result 15/19 chunk benchmark; prepare is CPU-only.

Reuses chunk_continuation buffers/events and the existing authenticated wrapper.
No HF execution, new reference, model, production edit, or GPU work in prepare.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys

from performance import ROOT, HERE, NATIVE, NVCC, TOOLCHAIN, CUDA_LIB, execute, sha, write_json, bf16

ARCHIVE_SHA = '2182f58b6f5543d681dc6fbd1873bb0d4de1e27a3b4e265f85af001af1c5d0f7'
EXPECTED_SHA = '2d4cdcf4c6dba60cea08620c40c02fcdc05ef674a8e9cce6b0ffdd240e9e6c1d'
PRIOR = ROOT / 'target/validation/chunk-continuation-cpu-20261006/prepared'
EXPECTED = ROOT / 'target/validation/chunk-continuation-performance-gpu-20261006/expected'
ACCEPTED = ROOT / 'target/validation/gdn-prefill-one-20261007/FINAL.json'
WRAPPER = ROOT / 'target/validation/gdn-chunk-splitk-20261006/owned.py'
PREFLIGHT = ROOT / 'target/validation/bf16-layer01-supported-retry-20261005/environment-preflight.json'
CASES = [('actual5', 5), ('dense64', 64), ('dense129', 129)]


def definition(text, marker):
    start = text.index(marker)
    end = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def prepare(out):
    os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void')
    out.mkdir(parents=True, exist_ok=False)
    current = NATIVE / 'core/recurrent_bf16_chunk.cuh'
    archive = HERE / 'chunk_unfused.cuh'
    accepted = json.loads(ACCEPTED.read_text())
    assert accepted['status'] == 'BACKEND_EXPLICIT_M1_PREFILL_EXACT_FUSION_GATES_PASS_GPU_RELEASED'
    assert sha(current) == accepted['source_pins'][str(current.relative_to(ROOT))]
    assert sha(archive) == ARCHIVE_SHA
    old, new = archive.read_text(), current.read_text()
    unchanged = ['struct Workspace', '__device__ inline float inv_l2(',
                 'inline float query_scale(', 'inline uint32_t prefix_window(',
                 '__global__ void prepare(', '__global__ void prefix(',
                 '__global__ void matmul(', '__global__ void triangular(']
    for marker in unchanged:
        assert definition(old, marker) == definition(new, marker), marker
    assert 'launches_per_chunk = 19;' in old and 'launches_per_chunk = 15;' in new
    renamed = old.replace('bf16_chunk', 'prior_bf16_chunk').replace(
        'core/recurrent_prior_bf16_chunk_abi.h', 'core/recurrent_bf16_chunk_abi.h')
    (out / 'chunk_prior.cuh').write_text(renamed)
    assert sha(EXPECTED / 'manifest.json') == EXPECTED_SHA
    reference = json.loads((EXPECTED / 'manifest.json').read_text())
    assert not reference['native_results_used']
    assert sha(PRIOR / 'prepared.json') == reference['prepared_sha256']
    original = json.loads((PRIOR / 'prepared.json').read_text())
    sources = [Path(__file__), HERE/'chunk_fusion_performance.cu', HERE/'chunk_continuation.cu',
               HERE/'chunk.cu', HERE/'performance.py', archive, current, ACCEPTED,
               PRIOR/'prepared.json', EXPECTED/'manifest.json', WRAPPER, PREFLIGHT,
               ROOT/'scripts/test_vllm_qwen35.py', ROOT/'scripts/_owned_subreaper.py',
               TOOLCHAIN/'env-cuda124.sh', NVCC]
    sources += [NATIVE/'core'/name for name in ['recurrent_bf16_chunk_abi.h',
                'recurrent_bf16.cuh', 'recurrent_bf16_abi.h', 'device.cuh', 'validation.cuh']]
    libraries = json.loads(PREFLIGHT.read_text())['resolved_libraries']
    for library in libraries:
        assert sha(library['realpath']) == library['sha256']
        sources.append(Path(library['realpath']))
    for name, rows in CASES:
        d = out / name; (d/'step0').mkdir(parents=True)
        for relative in ['state.f32', 'a_log.bf16', 'dt_bias.bf16',
                         'step0/qkv.bf16', 'step0/a.bf16', 'step0/b.bf16']:
            src = PRIOR / name / relative
            assert sha(src) == original['artifact_pins'][name+'/'+relative]
            shutil.copyfile(src, d/relative); sources.append(src)
        for field in ['output', 'state']:
            relative = name+'/carry-0-'+field+'.f32'; src=EXPECTED/relative
            assert sha(src) == reference['artifacts'][relative]
            sources.append(src)
            if field == 'output':
                (d/'expected-output.bf16').write_bytes(bf16(src.read_bytes()))
                assert (d/'expected-output.bf16').stat().st_size == rows*4096*2
            else:
                shutil.copyfile(src, d/'expected-state.f32')
    env = dict(os.environ, LD_LIBRARY_PATH=CUDA_LIB+':/usr/lib64')
    build = execute([NVCC, '-std=c++17', '-O3', '-arch=sm_86', '--cudadevrt=none',
        '--cudart=shared', '-Xptxas=-v', '-I', NATIVE, '-I', HERE, '-I', out,
        HERE/'chunk_fusion_performance.cu', '-L', TOOLCHAIN/'link', '-o', out/'benchmark'], env, 120)
    write_json(out/'build.json', build)
    assert build['returncode'] == 0, build['stderr']
    contracts = execute([out/'benchmark', '--contracts'], env, 15)
    write_json(out/'contracts.json', contracts)
    assert contracts['returncode'] == 0, contracts
    write_json(out/'prepared.json', dict(status='CPU_READY_GPU_NOT_RUN', cases=CASES,
        archive_sha256=ARCHIVE_SHA, unchanged_math=unchanged,
        pairs_per_case=30, samples_per_variant=30, warmups_per_variant=10,
        gpu_child_timeout_seconds=60, whole_wrapper_timeout_seconds=240,
        same_addresses=True, independent_state_reset_outside_events=True,
        independent_HF_check_every_sample=True, library_pins=libraries,
        source_pins={str(p):sha(p) for p in sources},
        artifact_pins={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
    manifest = verify(out)
    pins = {**manifest['source_pins'], **{str(out/p):h for p,h in manifest['artifact_pins'].items()},
            str(out/'prepared.json'):sha(out/'prepared.json')}
    jobs = [dict(name=name, argv=[str(out/'benchmark'), str(out/name), str(rows), str(i%2)],
                 seconds=60, pins=pins) for i,(name,rows) in enumerate(CASES)]
    write_json(out/'jobs.json', jobs)
    print('CPU READY: 3 cases, 90 pairs, 180 individually checked timed calls; GPU NOT run')


def verify(out):
    m = json.loads((out/'prepared.json').read_text())
    for path,pin in m['source_pins'].items():
        assert sha(path) == pin, 'source/resource changed: '+path
    for path,pin in m['artifact_pins'].items():
        assert sha(out/path) == pin, 'artifact changed: '+path
    return m


def summarize(out, evidence):
    import numpy as np
    verify(out)
    assert json.loads((evidence/'release.json').read_text())['release'] == 'released'
    result = {}
    for case, rows in CASES:
        receipt = json.loads((evidence/(case+'-supervisor.json')).read_text())
        h,c = receipt['handshake'],receipt['custody']
        assert receipt['returncode'] == 0 and not receipt['timed_out']
        assert h['containment_verified'] and c['containment_verified'] and c['echild']
        assert h['launch_id'] == c['launch_id'] and h['helper_identity'] == c['helper_identity']
        assert receipt['descendant_custody'] == 'proven_drained' and not receipt['cleanup_errors'] and not receipt.get('error')
        records = [json.loads(line) for line in receipt['stdout'].splitlines()]
        samples = [r for r in records if r['event'] == 'timing']
        complete = [r for r in records if r['event'] == 'complete']
        assert len(samples) == 60 and len(complete) == 1
        assert complete[0]['rows'] == rows and complete[0]['correctness_checked_every_sample']
        assert all(r['correctness_exact'] for r in samples)
        for pair in range(30):
            p = [r for r in samples if r['pair'] == pair]
            assert len(p) == 2 and {r['variant'] for r in p} == {'unfused19','fused15'}
            assert {r['order'] for r in p} == {0,1}
        times = {v:[r['microseconds'] for r in samples if r['variant']==v]
                 for v in ['unfused19','fused15']}
        assert all(len(t)==30 and all(np.isfinite(x) and x>0 for x in t) for t in times.values())
        resources = [r for r in records if r['event']=='resource']
        assert len(resources) == 18
        result[case] = dict(rows=rows, samples=samples, resources=resources,
            median_us={k:float(np.median(v)) for k,v in times.items()},
            p95_us={k:float(np.percentile(v,95)) for k,v in times.items()},
            paired_delta_us=[next(r['microseconds'] for r in samples if r['pair']==i and r['variant']=='fused15') -
                             next(r['microseconds'] for r in samples if r['pair']==i and r['variant']=='unfused19') for i in range(30)])
    write_json(evidence/'timing.json',dict(prepared_sha256=sha(out/'prepared.json'),cases=result,
        interpretation='Same-result chunk-only CUDA events, no scan, no throughput projection'))
    print(json.dumps({k:{n:v[n] for n in ['median_us','p95_us']} for k,v in result.items()},indent=2))


def run(out, evidence, execute_cuda):
    if not execute_cuda or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('LD_PRELOAD'):
        raise RuntimeError('requires --execute-cuda, CUDA_VISIBLE_DEVICES=0 and sole GPU ownership')
    verify(out)
    jobs=json.loads((out/'jobs.json').read_text())
    assert len(jobs)==3
    for i,(case,rows) in enumerate(CASES):
        assert jobs[i]['argv']==[str(out/'benchmark'),str(out/case),str(rows),str(i%2)]
        assert jobs[i]['seconds']==60
    # Existing wrapper verifies all eight idle BEFORE any GPU child, authenticates
    # library hashes/custody, then verifies ECHILD and three idle release samples.
    r=execute(['/opt/conda310/bin/python','-B',WRAPPER,out/'jobs.json',evidence],dict(os.environ),240)
    print(r['stdout']);print(r['stderr'],file=sys.stderr)
    if evidence.is_dir():write_json(evidence/'wrapper-result.json',r)
    assert r['returncode']==0, 'owned wrapper failed; inspect receipts before retry'
    summarize(out,evidence)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run','summarize'])
    p.add_argument('directory',type=Path);p.add_argument('--evidence',type=Path)
    p.add_argument('--execute-cuda',action='store_true');a=p.parse_args()
    out=a.directory.resolve()
    if a.mode=='prepare':prepare(out)
    else:
        if a.evidence is None:p.error('--evidence required')
        if a.mode=='run':run(out,a.evidence.resolve(),a.execute_cuda)
        else:summarize(out,a.evidence.resolve())
