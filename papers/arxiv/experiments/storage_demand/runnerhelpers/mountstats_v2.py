"""Passive, timestamp-bracketed Linux NFS-client mountstats capture and deltas."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import time

from .evidence_v2 import ct, finite, integer, require

NFS_BYTES = ('normalreadbytes', 'normalwritebytes', 'directreadbytes', 'directwritebytes',
             'serverreadbytes', 'serverwritebytes', 'readpages', 'writepages')
HEADER = re.compile(r'^device (.+?) mounted on (.+?) with fstype (\S+)(?: statvers=(\S+))?\s*$')


def _unescape(value):
    return re.sub(r'\\([0-7]{3})', lambda match: chr(int(match.group(1), 8)), value)


def parse_mountstats(raw_text, captured_unix_ns=None, captured_monotonic=None, source=None):
    require(isinstance(raw_text, str), 'mountstats must be raw text')
    mounts = []
    current = None
    for line in raw_text.splitlines():
        header = HEADER.fullmatch(line)
        if header:
            current = dict(device=_unescape(header[1]), mountpoint=_unescape(header[2]),
                           fstype=header[3], statvers=header[4], age_seconds=None,
                           counters={}, errors=[], raw_lines=[line])
            mounts.append(current)
            continue
        if current is None:
            continue
        current['raw_lines'].append(line)
        key, colon, tail = line.strip().partition(':')
        if not colon or key not in ('age', 'bytes', 'READ'):
            continue
        parts = tail.split()
        if not parts or not all(part.isdigit() for part in parts):
            current['errors'].append('invalid ' + key)
            continue
        values = [int(part) for part in parts]
        if key == 'age':
            current['age_seconds'] = values[0]
        elif key == 'bytes':
            if len(values) != len(NFS_BYTES):
                current['errors'].append('unsupported bytes counter layout')
            else:
                current['counters'].update(zip(NFS_BYTES, values))
        elif key == 'READ':
            if len(values) < 8:
                current['errors'].append('incomplete READ RPC counters')
            else:
                current['counters'].update({f'read_rpc_{index}': value for index, value in enumerate(values)})
    return dict(source=source, captured_unix_ns=captured_unix_ns,
                captured_monotonic=captured_monotonic, mounts=mounts,
                raw_text=raw_text, raw_sha256=hashlib.sha256(raw_text.encode()).hexdigest())


def timestamped_mountstats(raw_text, phase, source='/proc/self/mountstats'):
    """For injected fixture/sample text; real acquisition uses capture_mountstats."""
    return dict(parse_mountstats(raw_text, time.time_ns(), time.monotonic(), source), phase=phase)


def select_nfs_mount(snapshot, checkpoint_paths):
    require(isinstance(checkpoint_paths, list) and checkpoint_paths, 'checkpoint paths missing')
    candidates = []
    for mount in snapshot.get('mounts', []):
        point = mount['mountpoint']
        if mount['fstype'] in ('nfs', 'nfs4') and all(
                str(path) == point or str(path).startswith(point.rstrip('/') + '/')
                for path in checkpoint_paths):
            candidates.append(mount)
    require(candidates, 'no single NFS mount covers all checkpoint paths')
    longest = max(len(mount['mountpoint']) for mount in candidates)
    selected = [mount for mount in candidates if len(mount['mountpoint']) == longest]
    require(len(selected) == 1, 'ambiguous NFS mount identity')
    return selected[0]


def capture_mountstats(directory, label, reader_identity, verify_root, checkpoint_paths):
    """Caller supplies maintained observer.verify_root; capture grants no authority.

    Use the authenticated worker/sampler as reader so it survives server exit.
    Prove its mount namespace matches the server at Ready; preserve that binding.
    Each local read is size bounded; the maintained worker absolute deadline
    bounds acquisition. This helper never touches checkpoint tensor payloads.
    """
    require(re.fullmatch(r'[A-Za-z0-9_-]{1,80}', label or ''), 'unsafe sample label')
    verify_root(reader_identity)
    pid = reader_identity['pid']
    begin = dict(monotonic=time.monotonic(), unix_ns=time.time_ns())
    namespace = str(Path(f'/proc/{pid}/ns/mnt').readlink())
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    with Path(f'/proc/{pid}/mountstats').open('rb') as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    require(len(raw) <= 2 * 1024 * 1024, 'mountstats raw capture size exceeded')
    verify_root(reader_identity)
    require(str(Path(f'/proc/{pid}/ns/mnt').readlink()) == namespace, 'mount namespace changed during sample')
    end = dict(monotonic=time.monotonic(), unix_ns=time.time_ns())
    snapshot = parse_mountstats(raw.decode('utf-8'), end['unix_ns'], end['monotonic'], f'/proc/{pid}/mountstats')
    selected = select_nfs_mount(snapshot, checkpoint_paths)
    snapshot.update(label=label, acquisition_begin=begin, acquisition_end=end,
                    reader_identity=dict(reader_identity), boot_id=boot_id, mount_namespace=namespace,
                    selected_mount=selected, acquisition_status='captured',
                    shared_client_traffic=True, owned_server_physical_bytes=None,
                    physical_nas_backend_bytes=None)
    raw_path = Path(directory) / (label + '-mountstats.txt')
    with raw_path.open('xb') as stream:
        stream.write(raw)
    ct.write(Path(directory) / (label + '-mountstats.json'), snapshot)
    return snapshot


def mountstats_delta(before, after):
    result = dict(status='missing_data', deltas=None, reasons=[],
                  shared_client_traffic=True, owned_server_physical_bytes=None,
                  physical_nas_backend_bytes=None,
                  scope='client mount counters; all shared traffic, not server-attributable physical NAS disk IO')
    if not isinstance(before, dict) or not isinstance(after, dict):
        result['reasons'].append('missing sample')
        return result
    result.update(before_window=before.get('acquisition_begin'), after_window=after.get('acquisition_end'))
    for key in ('boot_id', 'mount_namespace'):
        if not before.get(key) or before.get(key) != after.get(key):
            result['reasons'].append(key + ' missing or changed')
    for clock in ('captured_unix_ns', 'captured_monotonic'):
        if not finite(before.get(clock)) or not finite(after.get(clock)) or after[clock] <= before[clock]:
            result['reasons'].append(clock + ' missing/nonincreasing')
    old, new = before.get('selected_mount'), after.get('selected_mount')
    if not isinstance(old, dict) or not isinstance(new, dict):
        result['reasons'].append('selected mount missing')
        return result
    for key in ('device', 'mountpoint', 'fstype', 'statvers'):
        if not old.get(key) or old.get(key) != new.get(key):
            result['reasons'].append('mount identity ' + key + ' changed/missing')
    if old.get('errors') or new.get('errors'):
        result['reasons'].append('malformed/unsupported counters')
    counters_old, counters_new = old.get('counters', {}), new.get('counters', {})
    names = list(NFS_BYTES) + ['read_rpc_' + str(index) for index in range(8)]
    if set(counters_old) != set(counters_new) or not all(
            integer(counters_old.get(key)) and integer(counters_new.get(key)) for key in names):
        result['reasons'].append('counter fields missing/changed')
    reset = (integer(old.get('age_seconds')) and integer(new.get('age_seconds'))
             and new['age_seconds'] < old['age_seconds'])
    reset = reset or any(integer(value) and integer(counters_new.get(key))
                         and counters_new[key] < value for key, value in counters_old.items())
    if not integer(old.get('age_seconds')) or not integer(new.get('age_seconds')):
        result['reasons'].append('mount age missing')
    if reset:
        result.update(status='counter_reset', reasons=result['reasons'] + ['counter decrease or mount-age reset'])
    elif not result['reasons']:
        result.update(status='shared_client_delta',
                      deltas={key: counters_new[key] - value for key, value in counters_old.items()})
    return result
