"""Explicitly authorized, bounded single-context PID-mapping observation; no model."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract as ct
import launch_guard
import telemetry
sys.path.insert(0, str(ct.ROOT / 'scripts'))
import test_vllm_qwen35 as supervisor


def observe(payload):
    telemetry.authorize(payload['granted'], payload['uuid'], payload['owner_note'])
    ct.validate_owner_lock(payload['owner_lock'], payload['uuid'], payload['coordinator']['pid'])
    ct.require(os.environ.get('CUDA_VISIBLE_DEVICES') == payload['uuid'], 'Probe visibility mismatch')
    driver = ctypes.CDLL('/usr/lib64/libcuda.so.1')
    context, memory = ctypes.c_void_p(), ctypes.c_uint64()
    def call(name, types, *args):
        fn = getattr(driver, name)
        fn.argtypes, fn.restype = types, ctypes.c_int
        code = fn(*args)
        ct.require(code == 0, name + ' CUDA status ' + str(code))
    result = dict(status='failed', mapping_proven=False, local_identity=launch_guard.identity(os.getpid()),
                  pid_namespace=os.readlink('/proc/self/ns/pid'), nspid=[], samples=[])
    result['nspid'] = [int(x) for line in Path('/proc/self/status').read_text().splitlines()
                       if line.startswith('NSpid:') for x in line.split()[1:]]
    try:
        call('cuInit', [ctypes.c_uint], 0)
        device = ctypes.c_int()
        call('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), 0)
        uuid_bytes = (ctypes.c_ubyte * 16)()
        call('cuDeviceGetUuid_v2', [ctypes.c_void_p, ctypes.c_int], ctypes.byref(uuid_bytes), device)
        raw = bytes(uuid_bytes).hex()
        actual = 'GPU-' + '-'.join((raw[:8], raw[8:12], raw[12:16], raw[16:20], raw[20:]))
        ct.require(actual == payload['uuid'], 'Probe device UUID differs from authorized GPU')
        call('cuCtxCreate_v2', [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int],
             ctypes.byref(context), 0, device)
        call('cuMemAlloc_v2', [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t], ctypes.byref(memory), 1 << 20)
        for _ in range(2):
            result['samples'].append(telemetry.gpu_sample('/usr/bin/nvidia-smi', payload['uuid'], True, payload['owner_note']))
            time.sleep(.25)
        driver_pids = sorted({a['pid'] for s in result['samples'] for a in s['apps']})
        result['driver_reported_pids'] = driver_pids
        result['proc_evidence'] = {}
        for pid in driver_pids:
            status = Path(f'/proc/{pid}/status')
            result['proc_evidence'][str(pid)] = (
                [line for line in status.read_text().splitlines() if line.startswith(('Name:', 'Pid:', 'PPid:', 'NSpid:'))]
                if status.exists() else 'not visible in mounted /proc')
        result['status'] = 'observed'
        result['reason'] = ('Driver PID/local probe PID observation is not a reusable namespace translation; '
                            'need kernel NSpid mapping from host /proc or another independent mapping source')
        result['probe_pid_equals_driver_pid'] = driver_pids == [os.getpid()]
    finally:
        if memory.value:
            call('cuMemFree_v2', [ctypes.c_uint64], memory)
        if context.value:
            call('cuCtxDestroy_v2', [ctypes.c_void_p], context)
        ct.write(Path(payload['directory']) / 'observation.json', result)
    return 0


def main():
    ct.require_cpu_python()
    if len(sys.argv) == 3 and sys.argv[1] == '--_child':
        payload = json.load(sys.stdin)
        launch_guard.authenticate(payload, sys.argv[2])
        return observe(payload)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--uuid')
    parser.add_argument('--owner-note')
    parser.add_argument('--id', required=True)
    args = parser.parse_args()
    telemetry.authorize(args.execute, args.uuid, args.owner_note)
    directory = ct.new_run(args.id)
    report = dict(status='incomplete', gpu_uuid=args.uuid, mapping_proven=False,
                  purpose='single owned CUDA context PID mapping observation only; not inference')
    with ct.exclusive_owner_lock(args.uuid) as lock:
        before = telemetry.idle('/usr/bin/nvidia-smi', args.uuid, True, args.owner_note, ct.BASE)
        ct.write(directory / 'before.json', before)
        ct.require(before['status'] == 'verified', 'GPU idle prerequisite failed')
        payload, pin = launch_guard.seal(dict(directory=str(directory), granted=True, uuid=args.uuid,
                                             owner_note=args.owner_note, owner_lock=lock), directory)
        env = dict(PATH='/usr/bin:/bin', HOME='/home/admin', CUDA_VISIBLE_DEVICES=args.uuid,
                   NVIDIA_VISIBLE_DEVICES=args.uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONDONTWRITEBYTECODE='1',
                   FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
        argv = ct.isolated_argv([sys.executable, '-S', '-B', str(Path(__file__).resolve()), '--_child', pin], env)
        try:
            result = supervisor.run_process(argv, 60, payload)
            ct.write(directory / 'supervisor.json', result)
            report['supervisor'] = result
            report['status'] = 'observed' if ct.successful_process(result) else 'failed'
        finally:
            after = telemetry.idle('/usr/bin/nvidia-smi', args.uuid, True, args.owner_note, ct.BASE)
            ct.write(directory / 'after.json', after)
            report['release'] = after['status']
            ct.write(directory / 'result.json', report)
    print(json.dumps(dict(directory=str(directory), status=report['status'], release=report['release'])))
    return 0 if report['status'] == 'observed' and report['release'] == 'verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
