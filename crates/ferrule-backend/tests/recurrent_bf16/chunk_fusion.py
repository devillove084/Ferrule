"""CPU preparation for the 19->15 chunk fusion. GPU runs use existing owners.

No oracle regeneration during preparation, new supervisor, or timing claim.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from performance import ROOT, HERE, NATIVE, NVCC, TOOLCHAIN, CUDA_LIB, execute, sha, write_json
import chunk_gate
import chunk_math_regression
import chunk_splitk

ARCHIVE_SHA = '2182f58b6f5543d681dc6fbd1873bb0d4de1e27a3b4e265f85af001af1c5d0f7'
PREVIOUS = ROOT / 'target/validation/gdn-chunk-splitk-20261006'
STAGES = chunk_math_regression.CAPTURE / 'stage-handoff'
SHAPES = {'near-511-512': [2, 1, 2, 511, 512],
          'near-512-511': [2, 1, 1, 512, 511],
          'limit-512-512': [2, 1, 1, 512, 512],
          'dense129': [129, 16, 32, 128, 128]}


def prepare(out):
    os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void')
    assert sha(HERE / 'chunk_unfused.cuh') == ARCHIVE_SHA
    out.mkdir(parents=True, exist_ok=False)
    chunk_gate.prepare(out / 'baseline')
    chunk_math_regression.prepare(out / 'continuation')
    chunk_splitk.prepare(out / 'matrices')
    prior = (HERE / 'chunk_unfused.cuh').read_text().replace(
        'bf16_chunk', 'prior_bf16_chunk')
    # Do not rename the shared ABI include; only the namespace/export is private.
    prior = prior.replace('core/recurrent_prior_bf16_chunk_abi.h',
                          'core/recurrent_bf16_chunk_abi.h')
    (out / 'chunk_prior.cuh').write_text(prior)
    env = dict(os.environ, LD_LIBRARY_PATH=CUDA_LIB + ':/usr/lib64')
    for name, source in [('fusion', 'chunk_fusion.cu'), ('math_cpu', 'chunk_math_cpu.cu')]:
        result = execute([NVCC, '-std=c++17', '-O3', '-arch=sm_86',
                          '--cudadevrt=none', '--cudart=shared', '-Xptxas=-v',
                          '-I', NATIVE, '-I', HERE, '-I', out, HERE / source,
                          '-L', TOOLCHAIN / 'link', '-o', out / name], env, 120)
        write_json(out / (name + '-build.json'), result)
        assert result['returncode'] == 0, result['stderr']
    source_pins = {str(p): sha(p) for p in [Path(__file__), HERE / 'chunk_fusion.cu',
        HERE / 'chunk_unfused.cuh', HERE / 'chunk.cu',
        NATIVE / 'core/recurrent_bf16_chunk.cuh',
        NATIVE / 'core/recurrent_bf16_chunk_abi.h',
        NATIVE / 'core/device.cuh', NATIVE / 'core/validation.cuh']}
    # Freeze actual panel operands from the previously exact, independently checked
    # production replay. HF references themselves remain in their original files.
    previous = PREVIOUS / 'baseline-prepared/actual'
    actual = out / 'panels'; actual.mkdir()
    for p in previous.glob('chunk0-*.f32'):
        shutil.copyfile(p, actual / p.name); source_pins[str(p)] = sha(p)
    shutil.copyfile(previous / 'state.f32', actual / 'state.f32')
    source_pins[str(previous / 'state.f32')] = sha(previous / 'state.f32')
    handoff = json.loads((chunk_math_regression.CAPTURE / 'BACKEND-HANDOFF.json').read_text())
    stage_dir = out / 'stages'; stage_dir.mkdir()
    references = {}
    for name, shape in SHAPES.items():
        src = out / 'continuation' / name
        dst = stage_dir / name; dst.mkdir()
        for filename in ['qkv.bf16', 'a.bf16', 'b.bf16']:
            shutil.copyfile(src / 'step0' / filename, dst / filename)
        for filename in ['state.f32', 'a_log.bf16', 'dt_bias.bf16']:
            shutil.copyfile(src / filename, dst / filename)
        reference = STAGES / name / 'hf-observed-stages.safetensors'
        assert sha(reference) == handoff['cases'][name]['observed_hf_stages_sha256']
        references[name] = dict(path=str(reference), sha256=sha(reference), shape=shape)
        source_pins[str(reference)] = sha(reference)
    write_json(out / 'stage-references.json', references)
    artifacts = {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}
    write_json(out / 'fusion-prepared.json', dict(status='CPU_PREPARED_GPU_NOT_RUN',
        launches_per_chunk=15, prior_launches_per_chunk=19,
        source_pins=source_pins, artifact_pins=artifacts,
        reference_changes=False, workspace_changes=False))
    pins = {**source_pins, **{str(out / p): h for p, h in artifacts.items()}}
    jobs = [dict(name='fused-panels', argv=[str(out / 'fusion'), str(actual)],
                 seconds=120, pins=pins)]
    jobs += [dict(name=name, argv=[str(out / 'baseline/chunk'),
                 str(stage_dir / name), *map(str, shape)], seconds=120, pins=pins)
             for name, shape in SHAPES.items()]
    jobs.append(dict(name='compare-stages', argv=['/opt/conda310/bin/python', '-B',
        str(Path(__file__).resolve()), 'compare', str(out)], seconds=120,
        env=dict(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void'), pins=pins))
    write_json(out / 'extra-jobs.json', jobs)
    print('Prepared 49 fused-panel cases, 120 fixed HF stages, 10 baseline, '
          '7 continuation and 14 matrix cases; GPU NOT run')


def compare(out):
    import numpy as np
    from safetensors import safe_open
    # CPU conversion only: the retained beta tensor is BF16.
    import torch
    torch.set_num_threads(1)
    references = json.loads((out / 'stage-references.json').read_text())
    result = {}
    for name, meta in references.items():
        assert sha(meta['path']) == meta['sha256']
        result[name] = {}
        with safe_open(meta['path'], framework='pt', device='cpu') as f:
            for key in f.keys():
                value = f.get_tensor(key).float().numpy()
                panels = [(key, value)] if key.startswith('chunk') else [
                    ('chunk' + str(i) + '-' + key, value[:, :, i])
                    for i in range((meta['shape'][0] + 63) // 64)]
                for stage, expected in panels:
                    actual = np.fromfile(out / 'stages' / name / (stage + '.f32'), np.float32)
                    expected = expected.reshape(-1)
                    assert actual.shape == expected.shape, (name, stage)
                    unequal = int(np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32)))
                    result[name][stage] = dict(elements=int(actual.size), unequal=unequal)
    write_json(out / 'stage-comparison.json', result)
    assert sum(map(len, result.values())) == 120
    assert all(v['unequal'] == 0 for stages in result.values() for v in stages.values())
    print('120/120 retained independent HF stages bit-exact')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'compare'])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.directory.resolve())
    else:
        compare(args.directory.resolve())
