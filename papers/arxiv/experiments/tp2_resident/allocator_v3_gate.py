"""Small real-backend V3 qualification; no model import or model execution.

Reuses the paper UUID flock and authenticated owned-subprocess custodian.
Each GPU selector (including every fault) is a fresh process, never a worker pool.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(HERE.parent), str(ROOT / 'scripts')]
import contract as ct
import test_vllm_qwen35 as supervisor

UUID = 'GPU-b980f22b-58a3-fc88-21c1-ae86053d8535'
FREEZE = ROOT / 'crates/ferrule-backend/tests/allocator_batch_v3/FREEZE_V3.sha256'
FREEZE_HASH = '44a551c3772e5456a842d95133e9cbdd29dadafcc3d8fb29f15205cdb7574344'
NEW_GPU = [
    'gpu_n8_two_streams_batch_and_reuse',
    'gpu_pending_byte_pressure_allows_safe_extent_not_growth',
    'gpu_pending_batch_pressure_allows_safe_extent_not_growth',
    'gpu_weak_gone_pending_sync_new_registration_is_not_quiescent',
    'gpu_capture_preserves_recorded_proof',
    'gpu_capture_preserves_complete_proof',
    'gpu_retirement_query_and_fallback_unknown_retains_whole_batch',
    'gpu_shutdown_known_drains_and_releases',
    'gpu_shutdown_fault_retains_primary_and_segments',
]
OLD_GPU = [
    'gpu_context_drop_quiescent_releases_primary_once',
    'gpu_context_drop_shutdown_sync_fault_retains_primary_and_segments',
    'gpu_context_drop_unknown_before_does_not_attempt_release',
    'gpu_context_drop_query_fault_cannot_be_washed_by_later_event',
]
PUBLIC = [
    ('public_allocator_diagnostics_track_reuse_and_trim', False),
    ('shutdown_rejects_new_ordinary_device_buffers', False),
    ('pr01_capture_guard_precedes_allocator_and_preserves_failpoint', True),
    ('pr01_omitted_public_entries_reject_capture_assertion', True),
    ('pr01_sync_failpoint_precedes_copy_and_guard_does_not_consume_it', True),
]

def freeze():
    ct.require(ct.sha(FREEZE) == FREEZE_HASH, 'V3 freeze manifest drift')
    for line in FREEZE.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        ct.require(ct.sha(ROOT / name) == digest, 'Frozen source drift: ' + name)


def environment(gpu=False):
    env = ct.build_environment(ct.EXCLUSIVE)
    ct.require(env['CUDA_HOME'] == '/home/admin/wxw403883/anaconda3/envs/tao', 'Not tao CUDA')
    env.update(FERRULE_CUDA_TEST_ORACLE='0', CARGO_TARGET_DIR=str(ROOT / 'target/validation/allocator-batch-v3/cargo-cuda124'),
               CUDA_VISIBLE_DEVICES=UUID if gpu else '', NVIDIA_VISIBLE_DEVICES=UUID if gpu else 'void')
    if not gpu:
        env['FERRULE_NO_CUDA'] = '1'
    return env


def execute(out, name, argv, env, timeout=60):
    result = supervisor.run_process(ct.isolated_argv(list(map(str, argv)), env), timeout)
    ct.write(out / (name + '.json'), dict(argv=list(map(str, argv)), environment=env, supervisor=result))
    (out / (name + '.stdout')).write_text(result['stdout'])
    (out / (name + '.stderr')).write_text(result['stderr'])
    print(name, 'PASS' if ct.successful_process(result) else 'FAIL', 'seconds=', round(result['wall_seconds'], 3), flush=True)
    ct.require(ct.successful_process(result), 'Process failed/timed out/custody incomplete: ' + name)
    return result['stdout']


def prepare(out):
    freeze()
    env = environment()
    version = subprocess.check_output([env['NVCC'], '--version'], env=env, text=True, timeout=10)
    ct.require('release 12.4' in version, 'CUDA12.4 required')
    sources = ct.source_pins()
    args = [ct.EXCLUSIVE['cargo'], 'test', '--locked', '--offline', '-p', 'ferrule-backend', '--features', 'cuda', '--lib']
    for name in ('allocator_batch_v3', 'allocator_context_drop', 'allocator_retirement', 'cuda_allocator', 'allocator_retirement_gpu'):
        args += ['--test', name]
    text = execute(out, 'build', args + ['--no-run', '--message-format=json'], env, 600)
    binaries = {}
    for line in text.splitlines():
        if not line.startswith('{'):
            continue
        row = json.loads(line)
        if row.get('reason') == 'compiler-artifact' and row.get('executable'):
            key = 'lib' if row['target']['kind'] == ['lib'] else row['target']['name']
            binaries[key] = dict(path=row['executable'], sha256=ct.sha(row['executable']))
    ct.require(len(binaries) == 6, 'Missing expected test binaries')
    freeze()
    ct.require(sources == ct.source_pins(), 'Sources changed during preparation')
    ct.write(out / 'prepared.json', dict(freeze_sha256=FREEZE_HASH, sources=sources, binaries=binaries,
        environment=env, nvcc=version, runner_sha256=ct.sha(__file__),
        rustc=subprocess.check_output([env['RUSTC'], '-Vv'], env=env, text=True, timeout=10)))


def prepared(out):
    freeze()
    p = ct.load(out / 'prepared.json')
    ct.require(p['sources'] == ct.source_pins(), 'Current source differs from prepared')
    ct.require(p['runner_sha256'] == ct.sha(__file__), 'Runner differs from prepared')
    for row in p['binaries'].values():
        ct.require(ct.sha(row['path']) == row['sha256'], 'Binary drift')
    return p


def cpu(out):
    p = prepared(out)
    rows = [('allocator_batch_v3', 'tests::v3_', 10), ('allocator_context_drop', 'tests::context_drop_', 14),
            ('allocator_retirement', '', 29), ('lib', 'cuda::allocator::', 7)]
    for binary, selector, count in rows:
        text = execute(out, 'cpu-' + binary, [p['binaries'][binary]['path'], selector, '--nocapture', '--test-threads=1'], environment())
        ct.require(f'test result: ok. {count} passed; 0 failed;' in text, 'CPU selector/count mismatch')
    ct.write(out / 'cpu-result.json', dict(passed=True, phases=rows, total=60))


def snapshot():
    g = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,compute_cap,memory.used,utilization.gpu',
                                '--format=csv,noheader,nounits'], text=True, timeout=10)
    p = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = list(csv.reader(io.StringIO(g), skipinitialspace=True))
    ct.require(len(rows) == 8 and rows[0][1] == UUID and rows[0][3] == '8.6', 'GPU inventory/UUID/SM mismatch')
    return dict(gpus=rows, compute_processes=p.strip(), idle=not p.strip() and all(r[4] == '0' and r[5] == '0' for r in rows))


def gpu(out):
    p = prepared(out)
    ct.require(ct.load(out / 'cpu-result.json')['passed'], 'CPU qualification required')
    env = environment(True)
    ct.require(env['CUDA_VISIBLE_DEVICES'] == env['NVIDIA_VISIBLE_DEVICES'] == UUID, 'selected/visible mismatch')
    rows = [('allocator_retirement_gpu', 'tests::' + name, True) for name in NEW_GPU]
    rows += [('lib', 'cuda::runtime::context_drop_gpu_tests::' + name, True) for name in OLD_GPU]
    rows += [('cuda_allocator', name, ignored) for name, ignored in PUBLIC]
    result = dict(passed=False, selected=[UUID], phases=[], full_model_run=False)
    try:
        with ct.exclusive_owner_lock(UUID) as lock:
            ct.validate_owner_lock(lock, UUID, os.getpid())
            ct.write(out / 'gpu-owner.json', dict(lock=lock, selected=[UUID], environment=env,
                authorization='Current user: sole fresh delegated GPU owner; backend only, no full model'))
            before = snapshot(); ct.write(out / 'gpu-before.json', before)
            ct.require(before['idle'], 'Eight-GPU idle prerequisite not met; do not kill foreign work')
            for binary, selector, ignored in rows:
                freeze()
                name = 'gpu-' + selector.rsplit('::', 1)[-1]
                args = [p['binaries'][binary]['path'], selector, '--exact', '--nocapture', '--test-threads=1']
                if ignored:
                    args.append('--ignored')
                listing = subprocess.check_output(args + ['--list'], env=env, text=True, timeout=10)
                ct.require(selector + ': test' in listing and '1 test,' in listing, 'Exact selector missing')
                try:
                    text = execute(out, name, args, env, 60)
                    ct.require('test result: ok. 1 passed; 0 failed; 0 ignored;' in text, 'Zero/ignored/wrong count GPU run')
                finally:
                    deadline = time.monotonic() + 30
                    while True:
                        after = snapshot()
                        if after['idle'] or time.monotonic() >= deadline:
                            break
                        time.sleep(1)
                    ct.write(out / (name + '-after.json'), after)
                    ct.require(after['idle'], 'GPU did not drain; preserve evidence; no foreign kill')
                result['phases'].append(dict(binary=binary, selector=selector, passed=True))
            result['passed'] = True
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        result['final_snapshot'] = snapshot()
        # Acquiring the same existing lock proves that the owner's fd was closed.
        with ct.exclusive_owner_lock(UUID):
            result['owner_lock_released'] = True
        freeze()
        ct.write(out / 'gpu-result.json', result)


def main():
    ct.require_cpu_python()
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('prepare', 'cpu', 'gpu'))
    ap.add_argument('--id', required=True)
    args = ap.parse_args()
    ct.require(re.fullmatch(r'[A-Za-z0-9_-]+', args.id), 'Unsafe identity')
    out = HERE / 'runs' / args.id
    if args.phase == 'prepare':
        out.mkdir(exist_ok=False)
    globals()[args.phase](out)

if __name__ == '__main__':
    main()
