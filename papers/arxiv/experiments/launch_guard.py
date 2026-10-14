"""Kernel-identity and sealed-receipt checks for the direct worker."""
import os
import secrets
from pathlib import Path
import time

import contract as ct


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, birth_ticks=int(fields[19]), ppid=int(fields[1]),
                pgrp=int(fields[2]), session=int(fields[3]))


def seal(payload, directory):
    payload = dict(payload, worker_token=secrets.token_hex(32), manifest={'protocol': ct.SCHEMA + 'worker.v2'},
                   coordinator=identity(os.getpid()),
                   worker_source_sha256=ct.sha(ct.HERE / 'runner.py'))
    path = Path(directory) / 'worker-launch.json'
    ct.write(path, payload)
    path.chmod(0o400)
    return payload, ct.sha(path)


def authenticate(payload, expected_hash):
    """Reject free --_worker invocation; name/argv alone are not authority."""
    path = Path(payload['directory']) / 'worker-launch.json'
    ct.require(path.is_file() and not path.is_symlink() and path.stat().st_mode & 0o222 == 0,
               'Worker needs a sealed read-only launch receipt')
    ct.require(ct.sha(path) == expected_hash and ct.load(path) == payload,
               'Worker launch receipt/hash differs')
    ct.require(ct.sha(ct.HERE / 'runner.py') == payload['worker_source_sha256'],
               'Worker source changed')
    token = os.environ.get('FERRULE_PAPER_WORKER_TOKEN')
    ct.require(token and token == payload.get('worker_token'), 'Worker token missing or mismatched')
    parent_pid = os.getppid()
    parent = identity(parent_pid)
    command = Path(f'/proc/{parent_pid}/cmdline').read_bytes().split(b'\0')
    helper = ct.ROOT / 'scripts/_owned_subreaper.py'
    ct.require(str(helper).encode() in command, 'Worker parent is not the maintained authenticated supervisor')
    ct.require(os.readlink('/proc/self/ns/pid') == os.readlink(f'/proc/{parent_pid}/ns/pid'),
               'Worker PID namespace identity is unprovable')
    journal_path = Path(payload['directory']) / 'custody.json'
    current = identity(os.getpid())
    deadline = time.monotonic() + 5
    journal = {}
    while time.monotonic() < deadline:
        try:
            journal = ct.load(journal_path)
            root = journal.get('root_identity') or {}
            if (journal.get('containment_verified') is True and journal.get('method') == 'linux-subreaper-waitid'
                    and journal.get('helper_identity') == parent and root.get('pid') == current['pid']
                    and root.get('birth_ticks') == current['birth_ticks']):
                break
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(.01)
    else:
        raise ValueError('Authenticated custodian/root kernel identity is missing')
    owner = payload['coordinator']
    ct.require(identity(owner['pid']) == owner and parent['ppid'] == owner['pid'],
               'Worker coordinator identity changed')
    return dict(worker=current, helper=parent, coordinator=owner, launch_id=journal['launch_id'])
