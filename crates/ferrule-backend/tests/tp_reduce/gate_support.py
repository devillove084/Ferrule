"""Backend-only adapter to the existing paper locks/authenticated custodian/observer."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PAPER = ROOT / 'papers/arxiv/experiments'
sys.path[:0] = [str(PAPER), str(PAPER / 'tp2_resident'), str(ROOT / 'scripts')]
import contract as ct
import launch_guard as guard
import observer
import test_vllm_qwen35 as supervisor

UUIDS = ['GPU-b980f22b-58a3-fc88-21c1-ae86053d8535', 'GPU-63f1bd01-ddf9-a742-ee48-9389892493d9']
NCCL = ROOT / 'target/vendor/vllm-baseline/lib/python3.10/site-packages/nvidia/nccl'
NCCL_SHA = '1d9c10322efacde695cf640db1f29fc1586c929f1087f0e1202d2d5e70fb7168'
HEADER_SHA = '4b240a1fa79858ca8cdffcf5c2fc40aeaca8b3eaec61728213715f0902aba3a8'
TAO = Path('/home/admin/wxw403883/anaconda3/envs/tao')
CONFIG = dict(limits=dict(case_seconds=120, idle_seconds=30), resources=dict(
    gpu_expected_total_bytes=24 << 30, gpu_target_bytes=22 << 30,
    host_target_bytes=128 << 30, sample_interval_seconds=.05))
PREFIX = 'cuda::operators::tp_reduce::gpu_tests::'
CASES = {
    'finite': PREFIX + 'two_card_mapping_m1_m5_rows8_edges_and_normal_owner_drop',
    'rejection': PREFIX + 'numeric::nonfinite_and_finite_overflow_never_publish_or_clear_sticky_status',
    'two-sites': PREFIX + 'numeric::two_sites_enqueue_without_checkpoint_and_rejection_is_sticky_across_sites',
    'unknown': PREFIX + 'post_enqueue_unknown_abort_is_not_completion_joint_retained_until_process_exit',
}

OWNERSHIP_CASES = {
    name: PREFIX + 'ownership::control_' + suffix
    for name, suffix in [('A','a'),('B','b'),('C','c'),('A-counts','a_counts'),('B-counts','b_counts'),('C-counts','c_counts')]
}

def case_rows(suite):
    ct.require(suite in ('numerical','ownership'), 'Unknown capabilities policy')
    return OWNERSHIP_CASES if suite == 'ownership' else CASES

def directory(identity, create=False):
    import re
    ct.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', identity or ''), 'Unsafe identity')
    path = ROOT / 'target/validation' / ('tp-reduce-gpu-' + identity)
    if create:
        path.mkdir(exist_ok=False)
    return path


def sources():
    backend = ROOT / 'crates/ferrule-backend'
    paths = [ROOT / 'Cargo.toml', ROOT / 'Cargo.lock', ROOT / 'rust-toolchain.toml']
    paths += [p for p in backend.rglob('*') if p.is_file() and p.suffix in ('.rs', '.cu', '.cuh', '.h', '.cc', '.toml', '.py')]
    paths += [p for p in (ROOT / 'crates/ferrule-common').rglob('*') if p.is_file() and p.suffix in ('.rs', '.toml')]
    paths += list((ROOT / '.cargo').glob('*.toml'))
    paths += [PAPER / 'contract.py', PAPER / 'launch_guard.py', PAPER / 'runner.py', PAPER / 'telemetry.py',
              PAPER / 'tp2_resident/observer.py', PAPER / 'tp2_resident/sampling_v4.py',
              ROOT / 'scripts/test_vllm_qwen35.py', ROOT / 'scripts/_owned_subreaper.py']
    return ct.inventory(paths)


def environment(out):
    env = ct.build_environment(ct.EXCLUSIVE)
    ct.require(env['CUDA_HOME'] == str(TAO) and env['FERRULE_CUDA_ARCH'] == 'sm_86', 'Pinned CUDA12.4/sm86 required')
    aliases = out / 'cuda12-lib'
    aliases.mkdir(exist_ok=True)
    for stem, filename in [('libcudart', 'libcudart.so.12.4.99'), ('libcublas', 'libcublas.so.12.4.5.8'),
                           ('libcublasLt', 'libcublasLt.so.12.4.5.8')]:
        source = TAO / 'lib' / filename
        ct.require(source.is_file(), 'Pinned CUDA12 library missing: ' + str(source))
        for suffix in ('.so', '.so.12'):
            link = aliases / (stem + suffix)
            if not link.exists():
                link.symlink_to(source)
            ct.require(link.resolve() == source.resolve(), 'CUDA alias drift')
    env.update(LD_LIBRARY_PATH=str(aliases) + ':/usr/lib64', LIBRARY_PATH=str(aliases),
               RUSTFLAGS='-L native=' + str(aliases) + ' ' + env['RUSTFLAGS'],
               CARGO_TARGET_DIR=str(ROOT / 'target'), FERRULE_CUDA_TEST_ORACLE='0',
               FERRULE_NCCL_INCLUDE_DIR=str(NCCL / 'include'))
    ct.require(not any(k.startswith('NCCL_') for k in env), 'No process-global NCCL overrides')
    return env


def cpu(out, name, argv, env, timeout=600):
    result = supervisor.run_process(ct.isolated_argv(list(map(str, argv)), env), timeout)
    ct.write(out / (name + '.json'), dict(argv=list(map(str, argv)), environment=env, supervisor=result))
    (out / (name + '.stdout')).write_text(result['stdout'])
    (out / (name + '.stderr')).write_text(result['stderr'])
    ct.require(ct.successful_process(result), 'CPU process failed/not drained: ' + name)
    return result['stdout']


def idle(note):
    row = observer.idle(CONFIG, UUIDS, note)
    ct.require(row['status'] == 'verified' and len(row['baseline']) == 8
               and all(d['gpu_bytes'] == 0 for d in row['baseline'].values()), 'All-eight pre/post idle required; never kill foreign work')
    return row


def wait_observed(proc, monitor, seconds=120):
    # Exact existing paper ownership pattern: the observer is the sole poll/reaper.
    import time
    deadline = time.monotonic() + seconds
    while monitor.thread.is_alive():
        if proc.returncode is not None:
            return proc.returncode
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, seconds)
        monitor.thread.join(timeout=min(.05, remaining))
    return proc.wait(timeout=max(.001, deadline - time.monotonic()))


def parse_log(path):
    events = []
    import re
    for line in path.read_text(errors='replace').splitlines():
        found = re.search(r'(TP_[A-Z_]+) (\{.*\})$', line)
        if found:
            events.append(dict(kind=found[1], **json.loads(found[2])))
    return events


def memory_report(events):
    stages = {e['stage']: e for e in events if e['kind'] == 'TP_MEMORY'}
    ordered = ['aftercontext', 'beforeNccl_afterOwnedGrant', 'afterinit_async_ready_beforeWarm',
               'afterwarm_rows8_count16384', 'afterfinite_checks', 'afterNcclDrop_beforeOwnerDrop']
    deltas = []
    for before, after in zip(ordered, ordered[1:]):
        if before not in stages or after not in stages:
            continue
        for a, b in zip(stages[before]['selected_cu'], stages[after]['selected_cu']):
            deltas.append(dict(before=before, after=after, ordinal=a['ordinal'],
                measured_device_free_drop_bytes=a['cu_free_bytes'] - b['cu_free_bytes'],
                measured_allocator_reserved_delta_bytes=b['reserved_bytes'] - a['reserved_bytes']))
    return dict(stages=stages, deltas=deltas, known_storage_grant_per_rank=262144,
                native_metadata_minimum_bytes=168, private_device_bound='Unknown', private_host_bound='Unknown',
                private_channel_sliding_buffer_bound='Unknown', measured_only=True, hard_cap_claim=False,
                attribution='Global free-memory deltas combine NCCL, driver/runtime/module overhead; NOT attributable allocation bounds',
                beforecontext_precision='all-eight NVML MiB; no CU context created for a byte-precision baseline')
