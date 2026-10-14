#!/usr/bin/env python3
"""Non-CUDA gated Nsight target, not a second server supervisor.

No GPU query or lock is acquired here. The sealed worker measures B1 and opens
one identity-bound gate. Actual model lifetime/signals use the existing root
_profile_capture.run_target; descendant cleanup belongs to the outer custodian.
"""
import argparse
import os
from pathlib import Path
import time

import instrumentation_live as live

ct, observer = live.ct, live.observer


def require_selected_visibility(selected):
    """Require exact inherited selected UUIDs; never widen or hide them here."""
    expected = ','.join(selected)
    ct.require(os.environ.get('CUDA_VISIBLE_DEVICES') == expected
        and os.environ.get('NVIDIA_VISIBLE_DEVICES') == expected,
        'Target must retain exactly selected model visibility')


def wait_gate(case, name, manifest, deadline):
    while not (case/name).exists():
        live.live_auth(manifest)
        if (case/'profile-stop').exists() and name == 'model-release.json':
            raise ValueError('Cancelled before model admission')
        ct.require(time.monotonic() < deadline, 'Bounded collector gate timeout: '+name)
        time.sleep(.05)
    path = case/name
    return live.sealed(path, ct.sha(path))


def check_gate(gate, manifest, target, digest):
    ct.require(gate.get('schema') == live.SCHEMA and gate.get('manifest_sha256') == digest
        and gate.get('nonce') == manifest['nonce'] and gate.get('target') == target
        and gate.get('launch_id') == manifest['authenticated']['launch_id']
        and gate.get('model_gpu_uuids') == list(live.shared.UUIDS), 'Forged/replayed model release gate')
    live.live_auth(manifest)
    live.descendant(target['identity'], gate['collector'])
    live.descendant(gate['collector'], manifest['authenticated']['worker'])
    frozen = live.sealed(Path(manifest['directory'])/'collector-baseline.json', gate['baseline_sha256'])
    ct.require(frozen['collector_identity'] == dict(gate['collector'], launch_id=gate['launch_id']),
               'Collector B1 identity differs')
    state = live.state_from_manifest(manifest)
    state.collector_started(frozen['collector_identity'], frozen['samples'][0]['monotonic'])
    state.freeze(frozen['samples'], frozen['collector_identity'])
    state.admit_model(live.shared.UUIDS, frozen['collector_identity'])
    return state


def run(case, digest, command):
    manifest = live.sealed(case/'collector-manifest.json', digest)
    ct.require(manifest.get('schema') == live.SCHEMA and manifest['directory'] == str(case)
        and manifest['command'] == command and manifest['command_sha256'] == ct.digest(command),
        'Sealed model command/directory differs')
    live.live_auth(manifest)
    target = dict(identity=observer.proc_identity(os.getpid()), manifest_sha256=digest, nonce=manifest['nonce'])
    live.descendant(target['identity'], manifest['authenticated']['worker'])
    require_selected_visibility(live.shared.UUIDS)
    live.readonly(case/'collector-target-ready.json', target)
    deadline = manifest['created_monotonic'] + live.scope.PROBE_SECONDS
    gate = wait_gate(case, 'model-release.json', manifest, deadline)
    check_gate(gate, manifest, target, digest)
    ct.require(not (case/'profile-stop').exists(), 'Cancelled before model launch')
    ct.validate_staged_binary(command[0], manifest['binary_sha256'])
    # Import is CPU-only. No second Popen/kill/wait loop exists in this adapter.
    wrapper = live.load_wrapper()
    result = wrapper.run_target(case, command, 1450, 45)
    live.readonly(case/'collector-model-ended.json', dict(target=target,
        manifest_sha256=digest, wrapper_sha256=ct.sha(case/'profile-server.json'), result=result))
    # Hold the collector alive for the worker's post-model B1 sample. The worker
    # stops its normal monitor before shutdown, hence this explicit handshake.
    released = wait_gate(case, 'collector-release.json', manifest, time.monotonic()+100)
    ct.require(released['manifest_sha256'] == digest, 'Collector release binding differs')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    ct.require(args.command and args.command[0] == '--' and len(args.command)>1,
               'Requires -- followed by pinned model command')
    try:
        return run(args.directory, args.manifest_sha256, args.command[1:])
    except BaseException as error:
        ct.write(args.directory/'collector-target-failure.json', dict(error=type(error).__name__+': '+str(error)))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
