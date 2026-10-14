"""CPU prepare / authenticated finite-time two-GPU backend microgate. No model route."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_support as s
ct = s.ct


def prepare(identity, suite):
    cases = s.case_rows(suite)
    out = s.directory(identity, True)
    env = s.environment(out)
    before = s.sources()
    ct.require(ct.sha(s.NCCL / 'include/nccl.h') == s.HEADER_SHA, 'Pinned header SHA drift')
    ct.require(ct.sha(s.NCCL / 'lib/libnccl.so.2') == s.NCCL_SHA, 'Pinned CUDA12 NCCL library SHA drift')
    metadata = (s.NCCL.parent.parent / 'nvidia_nccl_cu12-2.27.5.dist-info/METADATA').read_text()
    ct.require('Name: nvidia-nccl-cu12\n' in metadata and 'Version: 2.27.5\n' in metadata, 'NCCL CUDA12 metadata differs')
    version = subprocess.check_output([env['NVCC'], '--version'], env=env, text=True, timeout=10)
    ct.require('V12.4.99' in version, 'Exact NVCC12.4.99 required')
    tools = ct.inventory([Path(env['NVCC']), Path(env['RUSTC']), Path(ct.EXCLUSIVE['cargo']),
        s.TAO / 'lib/libcudart.so.12.4.99', s.TAO / 'lib/libcublas.so.12.4.5.8',
        s.TAO / 'lib/libcublasLt.so.12.4.5.8', Path('/lib64/libcuda.so.1'),
        s.NCCL / 'include/nccl.h', s.NCCL / 'lib/libnccl.so.2',
        s.TAO / 'lib/libcupti.so.2024.1.0', s.TAO / 'include/cupti_callbacks.h', s.TAO / 'include/cupti_version.h',
        s.TAO / 'conda-meta/cuda-cupti-12.4.99-0.json'])
    cargo = ct.EXCLUSIVE['cargo']
    args = [cargo, 'test', '--offline', '--locked', '-p', 'ferrule-backend', '--features', 'cuda',
            '--lib', '--test', 'tp_reduce', '--no-run', '--message-format=json']
    text = s.cpu(out, 'build', args, env)
    binaries = {}
    for line in text.splitlines():
        if line.startswith('{'):
            row = json.loads(line)
            if row.get('reason') == 'compiler-artifact' and row.get('executable'):
                key = 'lib' if row['target']['kind'] == ['lib'] else row['target']['name']
                slot = out / key
                slot.mkdir()
                path = ct.stage_binary(row['executable'], slot)
                binaries[key] = dict(path=path, sha256=ct.sha(path))
    ct.require(set(binaries) == {'lib', 'tp_reduce'}, 'Missing backend/CPU test binaries')
    linkage = subprocess.check_output(['/usr/bin/ldd', binaries['lib']['path']], env=env, text=True, timeout=10)
    (out / 'ldd.txt').write_text(linkage)
    ct.require('libcudart.so.12 =>' in linkage and 'libcudart.so.13' not in linkage, 'Mixed/absent CUDA12 runtime')
    for name, selector in cases.items():
        listing = s.cpu(out, 'list-' + name, [binaries['lib']['path'], selector, '--exact', '--ignored', '--list'], env, 20)
        ct.require(selector + ': test' in listing and '1 test,' in listing, 'Wrong exact selector')
    cpu_text = s.cpu(out, 'cpu-contracts', [binaries['tp_reduce']['path'], '--test-threads=1', '--nocapture'], env, 30)
    ct.require('8 passed; 0 failed' in cpu_text, 'CPU qualification missing')
    s.cpu(out, 'clippy', [cargo, 'clippy', '--offline', '--locked', '-p', 'ferrule-backend', '--features', 'cuda',
                         '--lib', '--test', 'tp_reduce', '--', '-D', 'warnings'], env)
    ct.require(before == s.sources(), 'Source drift during CPU preparation')
    ct.check_pins(tools)
    ct.write(out / 'prepared.json', dict(sources=before, tools=tools, binaries=binaries, environment=env,
        compiler_version=version, linkage=linkage, nccl_version=22705, selected=s.UUIDS, suite=suite, cases=cases,
        driver_compatibility='UNTESTED: CPU compilation/library load does not establish driver575/GPU support',
        hard_cap_claim=False, public_route_enabled=False, full_model_run=False))
    print(json.dumps(dict(prepared=str(out), gpu_executed=False)), flush=True)


def worker(pin):
    p = json.load(sys.stdin)
    auth = s.guard.authenticate(p, pin)
    ct.check_pins(p['sources']); ct.check_pins(p['tools']); ct.check_pins(p['pins'])
    ct.validate_staged_binary(p['binary'], p['binary_sha256'])
    ct.require(p['selected'] == s.UUIDS and p['selector'] == s.case_rows(p['suite'])[p['case']], 'Sealed selection/selector differs')
    for uuid in s.UUIDS:
        ct.validate_owner_lock(p['locks'][uuid], uuid, p['coordinator']['pid'])
    out = Path(p['directory'])
    ct.write(out / 'authentication.json', auth)
    env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), CUDA_VISIBLE_DEVICES=','.join(s.UUIDS),
        NVIDIA_VISIBLE_DEVICES=','.join(s.UUIDS), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        LD_LIBRARY_PATH=p['ld_library_path'], CUBLAS_WORKSPACE_CONFIG=':0:0',
        FERRULE_TP_REDUCE_GPU_AUTH='main-authorized-two-card-proof',
        FERRULE_TP_REDUCE_AUTH_RECEIPT=str(out / 'authentication.json'),
        FERRULE_TP_REDUCE_MEMORY_PROBE='1' if p['case'] == 'finite' else '0',
        FERRULE_PAPER_WORKER_TOKEN=p['worker_token'],
        FERRULE_TP_PROBE_CUPTI_PATH=str((s.TAO / 'lib/libcupti.so.2024.1.0').resolve()))
    ct.require(not any(k.startswith('NCCL_') for k in env), 'NCCL env override forbidden')
    ct.write(out / 'environment.json', env)
    with (out / 'test.log').open('x') as log:
        with ct.pinned_executable(p['binary'], p['binary_sha256']) as fd:
            proc = subprocess.Popen(['/proc/self/fd/' + str(fd), p['selector'], '--exact', '--ignored', '--nocapture', '--test-threads=1'],
                env=env, cwd=s.ROOT, stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
        monitor = s.observer.Monitor(out, proc, s.CONFIG, auth, s.UUIDS, p['owner_note'], p['baseline'],
            ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'FERRULE_TP_REDUCE_GPU_AUTH'))
        monitor.start()
        try:
            code = s.wait_observed(proc, monitor)
        finally:
            resources = monitor.finish()
            ct.write(out / 'resources.json', resources)
    text = (out / 'test.log').read_text(errors='replace')
    events = s.parse_log(out / 'test.log')
    ct.write(out / 'events.json', events)
    ct.write(out / 'memory-report.json', s.memory_report(events))
    loaded = {path for e in events for path in e.get('libraries', []) if Path(path).is_file()}
    ct.write(out / 'observed-library-sha256.json', ct.inventory(loaded))
    one = code == 0 and 'running 1 test' in text and '1 passed; 0 failed; 0 ignored' in text
    ct.write(out / 'test-exit.json', dict(returncode=code, one_test_passed=one, selector=p['selector'],
        binary_sha256=p['binary_sha256'], log_sha256=ct.sha(out / 'test.log'), public_route_enabled=False,
        hard_cap_claim=False, full_model_run=False, performance_claim=False))
    ct.check_pins(p['sources']); ct.check_pins(p['tools']); ct.check_pins(p['pins'])
    return 0 if one and resources['status'] == 'observed_within_targets' else 1


def run(identity, granted, note):
    ct.require(granted and note, 'Explicit sole GPU owner/backend microgate authorization required')
    root = s.directory(identity)
    prepared_path = root / 'prepared.json'
    prepared = ct.load(prepared_path)
    ct.check_pins(prepared['sources']); ct.check_pins(prepared['tools'])
    ct.require(prepared['sources'] == s.sources(), 'Fresh source set differs')
    cases = s.case_rows(prepared['suite'])
    ct.require(cases == prepared['cases'], 'Prepared capabilities policy differs')
    passed = []
    with s.contextlib.ExitStack() as stack:
        locks = {u: stack.enter_context(ct.exclusive_owner_lock(u)) for u in sorted(s.UUIDS)}
        for u in s.UUIDS:
            ct.validate_owner_lock(locks[u], u, os.getpid())
        before = s.idle(note)
        ct.write(root / 'preflight-eight-idle.json', before)
        inventory = subprocess.check_output(['/usr/bin/nvidia-smi',
            '--query-gpu=index,uuid,name,compute_cap,driver_version,memory.total,memory.free,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True, timeout=10)
        rows = list(s.csv.reader(s.io.StringIO(inventory), skipinitialspace=True))
        ct.require(len(rows) == 8 and [r[1] for r in rows[:2]] == s.UUIDS and all(r[3] == '8.6' for r in rows[:2]), 'Actual selected physical 0,1 UUID/SM roster mismatch')
        ct.write(root / 'actual-inventory.json', dict(rows=rows, selected=s.UUIDS, owner_note=note, locks=locks))
        for case, selector in cases.items():
            out = root / case
            out.mkdir(exist_ok=False)
            before = s.idle(note)
            ct.write(out / 'before.json', before)
            binary = prepared['binaries']['lib']
            p = dict(directory=str(out), case=case, selector=selector, binary=binary['path'], binary_sha256=binary['sha256'],
                suite=prepared['suite'], sources=prepared['sources'], tools=prepared['tools'], pins={str(prepared_path):ct.sha(prepared_path)},
                selected=s.UUIDS, locks=locks, owner_note=note, baseline=before['baseline'],
                ld_library_path=prepared['environment']['LD_LIBRARY_PATH'], backend_only=True)
            p, pin = s.guard.seal(p, out)
            env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
                       FERRULE_PAPER_WORKER_TOKEN=p['worker_token'], PYTHONDONTWRITEBYTECODE='1')
            result = s.supervisor.run_process(ct.isolated_argv([sys.executable, '-S', '-B', str(Path(__file__).resolve()),
                'worker', '--pin', pin], env), 180, p)
            ct.write(out / 'supervisor.json', result)
            ct.require(ct.drained(result), 'Custody Unknown: STOP; no cleanup/replay claim')
            after = s.idle(note)
            ct.write(out / 'after.json', after)
            okay = ct.successful_process(result)
            ct.write(out / 'result.json', dict(passed=okay, process_tree_drained=True, observer_thread_joined=True,
                all_eight_idle_after_process_exit=True, private_bound='Unknown', hard_cap_claim=False,
                public_route_enabled=False, full_model_run=False, performance_claim=False))
            print(case, 'PASS' if okay else 'FAIL', 'seconds=', round(result['wall_seconds'], 3), flush=True)
            if not okay:
                if (out / 'test.log').exists():
                    print((out / 'test.log').read_text(errors='replace'), flush=True)
                ct.require(False, 'Backend gate failed: stop diagnostic; no CUDA/NCCL env mixing or further GPU cases')
            passed.append(case)
        final = s.idle(note)
        ct.write(root / 'final-eight-idle.json', final)
    # Reacquiring both same UUID locks confirms the coordinator released the FDs.
    with s.contextlib.ExitStack() as stack:
        for uuid in sorted(s.UUIDS):
            stack.enter_context(ct.exclusive_owner_lock(uuid))
    ct.write(root / 'result.json', dict(passed=True, cases=passed, selected=s.UUIDS, two_uuid_locks_released=True,
        all_eight_idle=True, suite=prepared['suite'], private_bound='Unknown', hard_cap_claim=False, public_route_enabled=False,
        full_model_run=False, performance_claim=False))
    print(json.dumps(ct.load(root / 'result.json')), flush=True)


def main():
    ct.require_cpu_python()
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('prepare', 'run', 'worker'))
    ap.add_argument('--id')
    ap.add_argument('--suite', choices=('numerical','ownership'), default='numerical')
    ap.add_argument('--pin')
    ap.add_argument('--sole-gpu-owner-granted', action='store_true')
    ap.add_argument('--owner-note')
    args = ap.parse_args()
    if args.mode == 'prepare': prepare(args.id, args.suite)
    elif args.mode == 'run': run(args.id, args.sole_gpu_owner_granted, args.owner_note)
    else: sys.exit(worker(args.pin))

if __name__ == '__main__': main()
