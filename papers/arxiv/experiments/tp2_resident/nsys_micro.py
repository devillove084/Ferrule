#!/usr/bin/env python3
"""Official vectorAdd micro acceptance; existing authenticated custody, no model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrumentation_capture as capture
import instrumentation_live as live

ct, guard, observer, shared = live.ct, live.guard, live.observer, live.shared
HERE = Path(__file__).resolve().parent
PYTHON = '/home/admin/wxw403883/anaconda3/bin/python3.12'
SAMPLE = Path('/home/admin/wxw403883/anaconda3/envs/tao/demo_suite/vectorAdd')
TOOL = ct.ROOT/'target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1'
NSYS = TOOL/'target-linux-x64/nsys'
INVENTORY = sorted([*shared.UUIDS,
    'GPU-0bc01d70-2ade-0714-8959-b25ae75469b6', 'GPU-dc825838-7c6c-2683-71e7-9ee0ac61b011',
    'GPU-6c86e874-0ba8-b45a-a604-bca3b934bb86', 'GPU-ff61a997-b3fc-fd3d-800d-63bad94fd685',
    'GPU-2cdfe4c5-38b1-ef88-fbdd-f6848ddbae9d', 'GPU-0ac006a6-932d-f77d-9abd-ea242dd3ba4f'])
NOTE = 'User sole GPU owner; official Nsight micro only; eight instrumentation locks; workload GPU0/1; no foreign signals'


def profile_argv(case, digest):
    # 2025.3 has application-scoped CUDA trace, not the later cuda-trace-scope switch.
    return [str(NSYS), 'profile', '--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=none',
        '--gpu-metrics-devices=none', '--gpuctxsw=false', '--duration=150', '--kill=none',
        '--wait=all', '--stop-on-exit=true', '--export=sqlite', '--force-overwrite=false',
        '--output='+str(case/'micro'), PYTHON, '-S', '-B', str(Path(__file__).resolve()),
        'target', str(case), '--digest', digest]


def selected_environment(inherited, uuid):
    capture.require_selected_visibility(shared.UUIDS)
    ct.require(uuid in shared.UUIDS, 'Sample must stay within selected GPU0/1')
    # Preserve Nsight's injected loader environment; narrow each official sample to one GPU.
    return dict(inherited, CUDA_VISIBLE_DEVICES=uuid, NVIDIA_VISIBLE_DEVICES=uuid)


def target(case, digest):
    manifest = live.sealed(case/'collector-manifest.json', digest)
    ct.require(manifest['command'] == [str(SAMPLE)] and manifest['command_sha256'] == ct.digest([str(SAMPLE)]),
               'Official sample command differs')
    live.live_auth(manifest)
    capture.require_selected_visibility(shared.UUIDS)
    identity = dict(identity=observer.proc_identity(os.getpid()), manifest_sha256=digest, nonce=manifest['nonce'])
    live.descendant(identity['identity'], manifest['authenticated']['worker'])
    live.readonly(case/'collector-target-ready.json', identity)
    gate = capture.wait_gate(case, 'model-release.json', manifest,
                             manifest['created_monotonic']+live.scope.PROBE_SECONDS)
    capture.check_gate(gate, manifest, identity, digest)
    ct.require(not (case/'profile-stop').exists(), 'Cancelled before micro')
    ct.require(ct.sha(SAMPLE) == manifest['binary_sha256'], 'Sample binary changed')
    rows = []
    for index, uuid in enumerate(shared.UUIDS):
        live.live_auth(manifest)
        live.descendant(identity['identity'], gate['collector'])
        with (case/('vectorAdd-'+str(index)+'.log')).open('x') as out:
            result = subprocess.run([str(SAMPLE)], env=selected_environment(os.environ, uuid),
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, timeout=30)
        text = (case/('vectorAdd-'+str(index)+'.log')).read_text()
        rows.append(dict(index=index, uuid=uuid, returncode=result.returncode, passed='Test PASSED' in text))
        ct.require(result.returncode == 0 and rows[-1]['passed'], 'Official vectorAdd failed')
    ct.write(case/'profile-server.json', dict(returncode=0, samples=rows, workload='NVIDIA CUDA12.4 demo_suite/vectorAdd'))
    live.readonly(case/'collector-model-ended.json', dict(target=identity, manifest_sha256=digest,
        wrapper_sha256=ct.sha(case/'profile-server.json'), result=0))
    released = capture.wait_gate(case, 'collector-release.json', manifest, time.monotonic()+100)
    ct.require(released['manifest_sha256'] == digest, 'Collector release differs')
    return 0


def check_progress(server, monitor, deadline):
    ct.require(server.poll() is None and not monitor.failed.is_set() and time.monotonic() < deadline,
               'Micro deadline/collector/observer failure')


def worker(digest):
    ct.require_cpu_python()
    payload = json.load(sys.stdin)
    auth = guard.authenticate(payload, digest)
    case = Path(payload['directory'])
    live.validate_locks(payload['owner_locks'], auth['coordinator']['pid'])
    ct.check_pins(payload['pins'])
    ct.require(ct.sha(case/'baseline.json') == payload['baseline_sha256'], 'B0 receipt differs')
    before = observer.idle(shared.config(), shared.UUIDS, NOTE, payload['baseline'])
    ct.write(case/'prelaunch.json', before)
    ct.require(before['status'] == 'verified' and all(d['gpu_bytes'] == 0 for d in before['baseline'].values()),
               'Need original all-eight B0 before collector')
    digest = live.target_manifest(case, [str(SAMPLE)])
    env = dict(shared.base.cpu_environment(), CUDA_VISIBLE_DEVICES=','.join(shared.UUIDS),
               NVIDIA_VISIBLE_DEVICES=','.join(shared.UUIDS), TMPDIR=str(case/'nsys-tmp'),
               FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
    argv = profile_argv(case, digest)
    ct.write(case/'collector-command.json', dict(argv=argv, environment=env))

    monitor = server = None
    outcome = dict(status='failed')
    try:
        with (case/'server.log').open('x') as out:
            server = subprocess.Popen(ct.isolated_argv(argv, env), stdin=subprocess.DEVNULL,
                                      stdout=out, stderr=subprocess.STDOUT)
            monitor = live.CollectorMonitor(case, server, shared.config(), auth, shared.UUIDS, NOTE, payload['baseline'])
            monitor.start()
            deadline = time.monotonic()+65
            while not (case/'collector-model-ended.json').exists():
                live.live_auth(monitor.manifest)
                check_progress(server, monitor, deadline)
                time.sleep(.1)
            outcome['status'] = 'work_complete'
    except BaseException as error:
        outcome['error'] = type(error).__name__+': '+str(error)
    finally:
        if monitor:
            outcome['monitor'] = monitor.finish()
        if server:
            try:
                outcome['shutdown'] = live.shutdown(server, case, 100)
            except BaseException as error:
                outcome['cleanup_error'] = type(error).__name__+': '+str(error)
        ct.write(case/'micro-worker.json', outcome)
    good = (outcome['status'] == 'work_complete' and not outcome.get('cleanup_error')
        and not outcome['monitor']['errors'] and outcome['monitor'].get('gpu_samples', 0) > 0
        and outcome['shutdown']['returncode'] == 0 and not outcome['shutdown']['timed_out'])
    return 0 if good else 1


def run(pair):
    ct.require_cpu_python()
    case = pair/'runs/P1/session'
    ct.require((case/'nsys-tmp').is_dir(), 'Create fresh run directories before launch')
    ct.require(not (pair/'prepared.json').exists(), 'Fresh run required')
    pins = ct.inventory([SAMPLE, NSYS, TOOL/'host-linux-x64/QdstrmImporter',
        *TOOL.glob('target-linux-x64/libcupti.so.*'), Path(__file__), HERE/'instrumentation_capture.py',
        HERE/'instrumentation_live.py', HERE/'instrumentation_scope.py', HERE/'observer.py',
        ct.HERE/'launch_guard.py', ct.HERE/'contract.py', ct.ROOT/'scripts/_owned_subreaper.py',
        ct.ROOT/'scripts/test_vllm_qwen35.py'])
    ct.write(pair/'prepared.json', dict(kind='official_micro_only', pins=pins, workload=str(SAMPLE),
        model_started=False, supervisor_seconds=240, sample_seconds_each=30,
        instrumentation_ceiling_bytes=live.scope.PROBE_CEILING_BYTES,
        nonselected_delta_tolerance_bytes=live.scope.TOLERANCE_BYTES))
    grant = dict(scope='all_eight_instrumentation_probe', granted=True, note=NOTE,
        gpu_uuids=INVENTORY, model_gpu_uuids=list(shared.UUIDS), run_id=pair.name,
        prepared_sha256=ct.sha(pair/'prepared.json'))
    ct.write(pair/'probe-grant.json', grant)
    result = dict(status='failed', model_started=False)
    with live.scope.probe_locks(INVENTORY, shared.UUIDS, grant, ct.exclusive_owner_lock) as locks:
        ct.write(case/'probe-locks.json', dict(locks=locks, scope=grant['scope']))
        before = observer.idle(shared.config(), shared.UUIDS, NOTE)
        ct.write(case/'before.json', before)
        ct.require(before['status'] == 'verified' and all(d['gpu_bytes'] == 0 for d in before['baseline'].values()),
                   'Original all-eight B0 required')
        ct.write(case/'baseline.json', before['baseline'])
        payload = dict(directory=str(case), owner_locks=locks, baseline=before['baseline'],
                       baseline_sha256=ct.sha(case/'baseline.json'), pins=pins)
        payload, digest = guard.seal(payload, case)
        env = dict(shared.base.cpu_environment(), FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
        argv = ct.isolated_argv([PYTHON, '-S', '-B', str(Path(__file__).resolve()), 'worker', '--digest', digest], env)
        try:
            receipt = shared.base.supervisor.run_process(argv, 240, payload)
        except BaseException as error:
            receipt = getattr(error, 'supervisor_result', {})
            receipt.setdefault('error', str(error))
        ct.write(case/'supervisor.json', receipt)
        after = observer.idle(shared.config(), shared.UUIDS, NOTE, before['baseline'])
        ct.write(case/'after.json', after)
        original = (after['status'] == 'verified' and all(d['gpu_bytes'] == 0 for d in after['baseline'].values()))
        result.update(authenticated_drained=ct.drained(receipt), supervisor_success=ct.successful_process(receipt),
                      original_b0_verified=original, b1_returned=(case/'collector-after-model.json').exists())
        if all(result[k] for k in ('authenticated_drained','supervisor_success','original_b0_verified','b1_returned')):
            result['status'] = 'capture_complete_requires_sqlite_acceptance'
    ct.write(pair/'gpu-release.json', dict(result, local_contexts_closed=True, gpu_uuids=INVENTORY))
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['status'] == 'capture_complete_requires_sqlite_acceptance' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'worker', 'target'])
    parser.add_argument('directory', nargs='?', type=Path)
    parser.add_argument('--digest')
    parser.add_argument('--sole-gpu-owner', action='store_true')
    args = parser.parse_args()
    if args.mode == 'run':
        ct.require(args.sole_gpu_owner and args.directory, 'Explicit sole-owner grant and fresh directory required')
        return run(args.directory.resolve())
    if args.mode == 'worker':
        return worker(args.digest)
    return target(args.directory.resolve(), args.digest)


if __name__ == '__main__':
    raise SystemExit(main())
