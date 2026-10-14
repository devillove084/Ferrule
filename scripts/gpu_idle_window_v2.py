"""Versioned validation-only all-eight idle window; no launch/kill/reset authority.

Historical fixed-three-sample receipts and paper protocols are not upgraded.
Only authenticated ECHILD plus absent recorded owners admits utilization-only
stabilization. Every observation is retained; memory/apps stop immediately.
NVML idle and process drain are not a physical CUDA resource-release certificate.
"""
import re
import subprocess
import time
from pathlib import Path

from run_pinned_jobs import drained

SCHEMA = 'ferrule.gpu-idle-window.v2'
SECONDS = 15
CONSECUTIVE = 3
INTERVAL = 1


def no_recorded_owner(receipt):
    """ECHILD authority comes from the maintained authenticated supervisor."""
    if not drained(receipt):
        return False
    custody = receipt['custody']
    entries = custody.get('entries', [])
    roots = [row for row in entries if row.get('role') == 'root']
    if not entries or len(roots) != 1 or roots[0].get('identity') != custody.get('root_identity'):
        return False
    if any(row.get('reaped') is not True for row in entries):
        return False
    identities = [row.get('identity', {}) for row in entries] + [custody.get('helper_identity', {})]
    for ident in identities:
        if (type(ident.get('pid')) is not int
                or ident['pid'] <= 0 or type(ident.get('birth_ticks')) is not int
                or ident['birth_ticks'] <= 0):
            return False
        try:
            fields = Path(f'/proc/{ident["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[19]) == ident['birth_ticks']:
                return False
        except FileNotFoundError:
            pass  # Recorded PID/birth is absent; PID reuse is not our owner.
    return True


def sample(deadline, snapshot):
    """One shared deadline includes both queries; preserve partial raw output."""
    snapshot['recorded_unix_ns'] = time.time_ns()
    for key, query in [('gpu', '--query-gpu=index,uuid,memory.used,utilization.gpu'),
                       ('apps', '--query-compute-apps=gpu_uuid,pid,used_memory')]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Idle observation deadline before telemetry query')
        row = subprocess.run(['/usr/bin/nvidia-smi', query, '--format=csv,noheader'],
                             capture_output=True, text=True, timeout=min(10, remaining), check=True)
        snapshot[key] = row.stdout


def classify(snapshot, uuids):
    rows = [[part.strip() for part in line.split(',')] for line in snapshot['gpu'].splitlines()]
    if (len(rows) != 8 or any(len(row) != 4 or row[:2] != [str(i), uuids[i]]
                             for i, row in enumerate(rows))):
        return 'inventory_error'
    if snapshot['apps'].strip():
        return 'compute_apps_present'
    if any(row[2] != '0 MiB' for row in rows):
        return 'memory_nonzero_or_unknown'
    if any(not re.fullmatch(r'(?:[0-9]|[1-9][0-9]|100) %', row[3]) for row in rows):
        return 'utilization_unknown'
    return 'idle' if all(row[3] == '0 %' for row in rows) else 'utilization_only_transient'


def observe(uuids, *, custody=None):
    """Strict preflight without custody; bounded post-ECHILD stabilization with it.

    Return a fail-closed receipt (including partial telemetry), never discard a
    transient or extend the deadline. Callers must persist it before asserting.
    """
    started = time.monotonic()
    deadline = started + SECONDS
    result = dict(schema=SCHEMA, status='blocked', all_eight_idle=False, samples=[],
                  deadline_seconds=SECONDS, required_consecutive=CONSECUTIVE,
                  interval_seconds=INTERVAL, zero_memory_mib=0, zero_utilization_percent=0,
                  no_compute_apps_required=True, physical_owner_release='unverified',
                  mode='post_authenticated_echild' if custody is not None else 'strict_preflight',
                  authenticated_echild=False, recorded_owners_absent=False)
    consecutive = 0
    try:
        if (len(uuids) != 8 or len(set(uuids)) != 8
                or any(not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uuid) for uuid in uuids)):
            raise ValueError('Need exact ordered eight-UUID inventory')
        if custody is not None:
            result['authenticated_echild'] = drained(custody)
            result['recorded_owners_absent'] = no_recorded_owner(custody)
            if not result['authenticated_echild'] or not result['recorded_owners_absent']:
                raise ValueError('Authenticated ECHILD/absent recorded owners required before stabilization')
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Fixed 15-second idle stabilization deadline')
            if custody is not None and not no_recorded_owner(custody):
                raise ValueError('Recorded owner presence/custody became unknown')
            snapshot = dict(monotonic_started=time.monotonic())
            result['samples'].append(snapshot)
            sample(deadline, snapshot)
            snapshot['monotonic_finished'] = time.monotonic()
            state = classify(snapshot, uuids)
            snapshot['classification'] = state
            consecutive = consecutive + 1 if state == 'idle' else 0
            snapshot['consecutive_idle'] = consecutive
            if state not in ('idle', 'utilization_only_transient'):
                raise ValueError('Immediate idle stop: ' + state)
            if state != 'idle' and custody is None:
                raise ValueError('Preflight is not strict all-eight idle; no waiting for foreign work')
            # A late query cannot turn an expired window into a passing receipt.
            if time.monotonic() >= deadline:
                raise TimeoutError('Fixed 15-second idle stabilization deadline')
            if consecutive == CONSECUTIVE:
                result.update(status='verified', all_eight_idle=True,
                              accepted_sample_indices=list(range(len(result['samples']) - CONSECUTIVE,
                                                                len(result['samples']))))
                break
            time.sleep(min(INTERVAL, max(0, deadline - time.monotonic())))
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        result.update(consecutive_idle=consecutive, elapsed_seconds=time.monotonic() - started)
    return result
