"""Private live PID/birth/executable receipts bridged to raw SQL process keys.

No process discovery, signaling, GPU work, or acceptance from a caller's PID
list. Only the maintained authenticated collector's target and model qualify.
"""
from __future__ import annotations

from pathlib import Path

from profile_overlap_v1 import protocol as p, tp1_collector as tc
from runnerhelpers.evidence_v2 import ct, require

SCHEMA = p.ID + '.live-process-roles.v1'


def same_live(actual, expected):
    return (isinstance(actual, dict) and isinstance(expected, dict)
            and all(type(expected.get(k)) is int and expected[k] > 0 and actual.get(k) == expected[k]
                    for k in ('pid', 'birth_ticks', 'ppid'))
            and actual.get('state') in ('R', 'S', 'D', 'I', 'T', 't')
            and expected.get('state') not in ('Z', 'X'))


def expected(case, worker, prepared, *, generation=None):
    generation = p.GENERATION if generation is None else generation
    case = Path(case)
    pair = case.parent.parent.parent
    require(pair == Path(prepared['directory']) and pair.name == generation
            and worker['mode'] in p.ORDER and case == pair / 'runs' / worker['mode'] / 'session',
            'process roles from wrong generation/arm')
    manifest_path = case / 'collector-manifest.json'
    digest = ct.sha(manifest_path)
    manifest = tc.live.sealed(manifest_path, digest)
    launch = tc.live.sealed(case / 'worker-launch.json', manifest['worker_launch_sha256'])
    require(manifest['directory'] == str(case) and manifest['authenticated'] == worker['authenticated_worker']
            and manifest['command'] == prepared['specs'][worker['mode']]['argv']
            and manifest['command_sha256'] == ct.digest(manifest['command'])
            and manifest['binary_sha256'] == p.BINARY_SHA
            and launch['prepared_sha256'] == ct.sha(pair / 'prepared.json')
            and launch['directory'] == str(case) and launch['mode'] == worker['mode'],
            'process role source/launch/command binding differs')
    target = tc.live.sealed(case / 'collector-target-ready.json', ct.sha(case / 'collector-target-ready.json'))
    gate = tc.live.sealed(case / 'model-release.json', ct.sha(case / 'model-release.json'))
    require(target['identity'] == worker['target_identity'] and gate['target'] == target
            and target['manifest_sha256'] == gate['manifest_sha256'] == digest
            and target['nonce'] == gate['nonce'] == manifest['nonce']
            and gate['launch_id'] == manifest['authenticated']['launch_id']
            and gate['collector'] == worker['collector_identity']
            and gate['model_gpu_uuids'] == [tc.UUID], 'process roles lack authenticated target gate')
    root, wrapper = worker['server_identity'], worker['target_identity']
    require(root['ppid'] == wrapper['pid'] and root['birth_ticks'] >= wrapper['birth_ticks']
            and len({root['pid'], wrapper['pid'], worker['collector_identity']['pid']}) == 3,
            'model role is not unique birth-bound wrapper child')
    python_sha = prepared['pins'][str(Path(p.PYTHON).resolve())]
    return manifest, dict(
        model=dict(identity=root, executable_sha256=p.BINARY_SHA,
                   source_binding_sha256=prepared['binding_sha256'], command=manifest['command']),
        non_cuda_wrapper=dict(identity=wrapper, executable_sha256=python_sha,
            source_binding_sha256=ct.sha(pair / 'prepared.json'),
            command=p.profile_argv(case, digest)[p.profile_argv(case, digest).index(p.PYTHON):]))


def capture(case, label, worker, prepared):
    manifest, roles = expected(case, worker, prepared)
    tc.live.live_auth(manifest)
    tc.live.descendant(worker['target_identity'], worker['collector_identity'])
    records = {}
    for name, role in roles.items():
        identity = role['identity']
        before = tc.live.observer.proc_identity(identity['pid'])
        require(same_live(before, identity), 'role not live before executable observation')
        proc = Path('/proc') / str(identity['pid'])
        command = (proc / 'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        sha = ct.sha(proc / 'exe')
        after = tc.live.observer.proc_identity(identity['pid'])
        require(same_live(after, identity), 'role birth/parent/liveness changed during observation')
        require(sha == role['executable_sha256'] and command == role['command'], 'actual role executable/argv differs')
        records[name] = dict(role, before=before, after=after, observed_executable_sha256=sha, observed_command=command)
    tc.live.live_auth(manifest)
    receipt = dict(schema=SCHEMA, directory=str(case), phase=label, generation=p.GENERATION,
        manifest_sha256=ct.sha(Path(case) / 'collector-manifest.json'),
        prepared_sha256=ct.sha(Path(prepared['directory']) / 'prepared.json'), roles=records)
    path = Path(case) / (label + '-process-roles.json')
    tc.live.readonly(path, receipt)
    return dict(path=str(path), sha256=ct.sha(path))


def bind(case, worker, prepared, processes, *, generation=None):
    """Offline: verify both live observations, then uniquely join raw SQL PIDs."""
    generation = p.GENERATION if generation is None else generation
    _, wanted = expected(case, worker, prepared, generation=generation)
    for label in ('ready', 'after-requests'):
        path = Path(case) / (label + '-process-roles.json')
        pin = worker['process_role_receipts'][label]
        require(pin['path'] == str(path), 'role receipt from foreign directory')
        receipt = tc.live.sealed(path, pin['sha256'])
        require(receipt['schema'] == SCHEMA and receipt['generation'] == generation
                and receipt['directory'] == str(case) and receipt['phase'] == label
                and receipt['manifest_sha256'] == ct.sha(Path(case) / 'collector-manifest.json')
                and receipt['prepared_sha256'] == ct.sha(Path(prepared['directory']) / 'prepared.json')
                and set(receipt['roles']) == set(wanted), 'role receipt binding differs')
        for name, role in wanted.items():
            actual = receipt['roles'][name]
            require(all(actual.get(k) == v for k, v in role.items())
                    and same_live(actual['before'], role['identity']) and same_live(actual['after'], role['identity'])
                    and actual['observed_executable_sha256'] == role['executable_sha256']
                    and actual['observed_command'] == role['command'], 'raw role observation changed or not live')
    bound = []
    for name, role in wanted.items():
        matches = [r for r in processes if r['pid'] == role['identity']['pid']]
        require(len(matches) == 1, 'SQL process PID absent/reused/ambiguous for authenticated role')
        bound.append(dict(role= name, globalPid=matches[0]['globalPid'],
            **{k: role[k] for k in ('identity', 'executable_sha256', 'source_binding_sha256')}))
    return dict(process_roles=bound,
        role_executable_sha256={k: v['executable_sha256'] for k, v in wanted.items()},
        role_source_binding_sha256={k: v['source_binding_sha256'] for k, v in wanted.items()})
