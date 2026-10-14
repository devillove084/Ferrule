#!/usr/bin/env python3
"""Check or explicitly execute prepared attention jobs using the existing custodian.

Without --sole-gpu-owner-granted this only checks pins; no telemetry or child is
started. Preparation must be repeated after workflow changes. Historical jobs
and receipts are never rewritten. Drain evidence and GPU release are separate.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import test_vllm_qwen35 as supervisor

SCRIPT = Path(__file__).resolve()
WORKFLOW = (SCRIPT, SCRIPT.with_name('test_vllm_qwen35.py'), SCRIPT.with_name('_owned_subreaper.py'))
ENV_PROGRAM = Path('/usr/bin/env')
# A frozen allowlist, not an archive of the caller's credentials or full environment.
ENV_KEYS = ('PATH', 'HOME', 'TMPDIR', 'PYTHONPATH', 'CUDA_HOME', 'CUDA_PATH',
            'CUDA_DEVICE_ORDER', 'FERRULE_CUDA_ARCH', 'HF_HOME', 'HF_HUB_OFFLINE',
            'TRANSFORMERS_OFFLINE', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
NATIVE_LIBRARIES = '/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def libraries(path):
    rows = json.loads(Path(path).read_text())['resolved_libraries']
    require(isinstance(rows, list) and rows, 'Need pinned resolved runtime libraries')
    for row in rows:
        require(sha(row['realpath']) == row['sha256'], 'Runtime library changed: ' + row['realpath'])
    return rows


def launch_sha(job):
    return hashlib.sha256(json.dumps({k: v for k, v in job.items() if k != 'launch_sha256'},
                                     sort_keys=True, allow_nan=False).encode()).hexdigest()


def pin_jobs(jobs, libraries_file, extra_paths=()):
    """Preparation only: freeze each job environment independently, including tools."""
    require(not os.environ.get('LD_PRELOAD'), 'LD_PRELOAD is not admitted')
    libraries_file = Path(libraries_file).resolve(strict=True)
    libs = libraries(libraries_file)
    common = {str(p.resolve()): sha(p) for p in
              (*WORKFLOW, ENV_PROGRAM, Path(sys.executable), libraries_file, *extra_paths)}
    common.update({str(Path(row['realpath']).resolve()): row['sha256'] for row in libs})
    for job in jobs:
        executable = Path(job['argv'][0]).resolve(strict=True)
        pins = {str(Path(p).resolve()): value for p, value in job['pins'].items()}
        for path, pin in pins.items():
            require(sha(path) == pin, 'Prepared input changed before launch freeze: ' + path)
        pins.update(common)
        pins[str(executable)] = sha(executable)
        job['pins'] = pins
        environment = {key: os.environ[key] for key in ENV_KEYS if key in os.environ}
        environment.update(CUDA_VISIBLE_DEVICES='0', NVIDIA_VISIBLE_DEVICES='all',
                           CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONDONTWRITEBYTECODE='1',
                           LD_LIBRARY_PATH='' if job.get('hf') else NATIVE_LIBRARIES,
                           CUBLAS_WORKSPACE_CONFIG=':4096:8')
        environment.update(job.get('env', {}))
        require('LD_PRELOAD' not in environment, 'Job LD_PRELOAD is not admitted')
        job['environment'] = environment
        job['launch_sha256'] = launch_sha(job)
    return jobs


def validate(jobs, libraries_file):
    require(isinstance(jobs, list) and jobs, 'Need a nonempty prepared job list')
    require(not os.environ.get('LD_PRELOAD'), 'LD_PRELOAD is not admitted')
    libs = libraries(libraries_file)
    required = [*WORKFLOW, ENV_PROGRAM, Path(sys.executable), Path(libraries_file),
                *[Path(row['realpath']) for row in libs]]
    names = set()
    for job in jobs:
        name = job['name']
        require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)
                and name not in names, 'Unsafe or duplicate job name')
        names.add(name)
        argv = job['argv']
        require(isinstance(argv, list) and argv and all(isinstance(a, str) and '\0' not in a for a in argv),
                'Need literal shell-free argv')
        require(Path(argv[0]).is_absolute(), 'Executable must be absolute')
        seconds = job['seconds']
        require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 3600,
                'Job timeout must be finite and within 0..3600 seconds')
        environment = job.get('environment')
        require(isinstance(environment, dict) and all(isinstance(k, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', k)
                and isinstance(v, str) and '\0' not in v for k, v in environment.items()),
                'Missing frozen environment; reprepare, do not patch historical jobs')
        require('LD_PRELOAD' not in environment, 'Job LD_PRELOAD is not admitted')
        require(environment.get('CUDA_VISIBLE_DEVICES') in ('0', '')
                and environment.get('NVIDIA_VISIBLE_DEVICES') in ('all', 'void', '')
                and environment.get('CUBLAS_WORKSPACE_CONFIG') in (':0:0', ':4096:8')
                and environment.get('PYTHONDONTWRITEBYTECODE') == '1', 'Invalid device/arithmetic environment')
        require(all(environment.get(k) == v for k, v in job.get('env', {}).items()), 'Job override differs from frozen environment')
        pins = job['pins']
        for path in [*required, Path(argv[0])]:
            require(str(path.resolve()) in pins, 'Missing workflow/library/executable pin: ' + str(path))
        for path, pin in pins.items():
            require(Path(path).is_absolute() and sha(path) == pin, 'Pinned input changed: ' + path)
        require(job.get('launch_sha256') == launch_sha(job), 'Frozen argv/environment/timeout changed; reprepare')
    return libs


def drained(result):
    h, c = result.get('handshake', {}), result.get('custody', {})
    return (h.get('containment_verified') is True and c.get('containment_verified') is True
            and c.get('echild') is True and c.get('status') == 'proven_drained'
            and result.get('descendant_custody') == 'proven_drained'
            and bool(h.get('launch_id')) and h.get('launch_id') == c.get('launch_id')
            and bool(h.get('helper_identity')) and h.get('helper_identity') == c.get('helper_identity')
            and result.get('cleanup_errors') == [] and not result.get('error')
            and not result.get('reporting_errors'))


def sample():
    result = {}
    for key, query in [('gpu', '--query-gpu=index,uuid,memory.used,utilization.gpu'),
                       ('apps', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory')]:
        value = subprocess.run(['nvidia-smi', query, '--format=csv,noheader'],
                               capture_output=True, text=True, timeout=10, check=True)
        result[key] = value.stdout
    return result


def idle(snapshot):
    rows = [line.split(',') for line in snapshot['gpu'].splitlines()]
    # Preserve the original owner's eight-card, whole-host quiescence gate.
    return (not snapshot['apps'].strip() and len(rows) == 8
            and all(len(row) == 4 and row[-2].strip() == '0 MiB'
                    and row[-1].strip() == '0 %' for row in rows))


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def execute(jobs, output, libraries_file, granted=False):
    require(granted, 'Explicit sole GPU owner grant required')
    libs = validate(jobs, libraries_file)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    receipts, after, release_error = [], [], None
    write(output / 'launch.json', dict(jobs=jobs, libraries=libs,
          supervisor_sha256=sha(WORKFLOW[1]), subreaper_sha256=sha(WORKFLOW[2]),
          runner_sha256=sha(SCRIPT)))
    try:
        before = sample()
        write(output / 'before.json', before)
        require(idle(before), 'Whole-host GPU quiescence not established; no jobs launched')
        for job in jobs:
            validate([job], libraries_file)
            # Each child receives its own frozen environment. CPU compare jobs
            # cannot hide GPUs or change arithmetic for a subsequent GPU job.
            argv = [str(ENV_PROGRAM), '-i',
                    *[key + '=' + value for key, value in sorted(job['environment'].items())],
                    *job['argv']]
            try:
                result = supervisor.run_process(argv, job['seconds'])
            except BaseException as error:
                result = getattr(error, 'supervisor_result', dict(custody={'status': 'unknown'},
                                 error=type(error).__name__))
                receipts.append(result)
                write(output / (job['name'] + '-supervisor.json'), result)
                raise
            receipts.append(result)
            write(output / (job['name'] + '-supervisor.json'), result)
            require(result.get('returncode') == 0 and result.get('timed_out') is False
                    and drained(result), 'Job failed or descendant custody unknown: ' + job['name'])
    finally:
        try:
            for _ in range(3):
                time.sleep(1)
                after.append(sample())
        except Exception as error:
            release_error = str(error)
        proven = bool(receipts) and all(drained(result) for result in receipts)
        released = proven and len(after) == 3 and all(idle(snapshot) for snapshot in after)
        write(output / 'release.json', dict(after=after, drained=proven,
              release='released' if released else 'unverified', telemetry_error=release_error))
    require(released, 'GPU release is unverified; no further jobs authorized')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', type=Path, required=True)
    parser.add_argument('--libraries', type=Path, required=True, help='pinned environment-preflight JSON')
    parser.add_argument('--output', type=Path, help='new receipt directory; never overwrite')
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    args = parser.parse_args(argv)
    jobs = json.loads(args.jobs.read_text())
    if not args.sole_gpu_owner_granted:
        validate(jobs, args.libraries)
        print('CHECK PASS: pins/environment only; no GPU telemetry or child launched')
        return 0
    require(args.output is not None, 'Execution requires a new --output directory')
    return execute(jobs, args.output, args.libraries, granted=True)


if __name__ == '__main__':
    raise SystemExit(main())
