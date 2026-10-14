"""Narrow CPU-only evidence adapters for storage-demand protocol v2.

This module deliberately does not own a process, query a GPU, send HTTP,
signal a PID, drop caches, or retry a lifecycle. Future execution code must
reuse the maintained contract, launch-guard, observer, target, supervisor and
HTTP modules named by ``protocol.json``.
"""
from __future__ import annotations

import math
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
EXPERIMENTS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXPERIMENTS))
import contract as ct

PHASE_COUNTS = dict(initial=4, qa_excluded_first=1, qa_warmup=2, qa_measured=5)
COUNTERS = (
    'checkpoint_read_calls', 'checkpoint_read_bytes', 'checkpoint_source_checks',
    'checkpoint_path_metadata_calls', 'checkpoint_handle_metadata_calls',
    'h2d_copies', 'h2d_bytes', 'artifact_uploads', 'artifact_upload_bytes',
    'expert_hits', 'expert_uploads', 'expert_evictions',
)
UUID_RE = re.compile(r'^GPU-[0-9a-fA-F-]{36}$')


class ProtocolError(ValueError):
    """Malformed or insufficient evidence; never a launch instruction."""


def require(condition, reason):
    if not condition:
        raise ProtocolError(reason)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_deadlines(limits):
    """Validate the literal finite envelope before any future authorization."""
    required = (
        'startup_seconds', 'request_seconds', 'all_requests_seconds',
        'worker_bookkeeping_seconds', 'target_lifetime_seconds', 'shutdown_seconds',
        'supervisor_seconds', 'supervisor_call_overhead_seconds',
        'outer_lifecycle_seconds', 'pilot_seconds',
    )
    require(isinstance(limits, dict), 'limits must be an object')
    require(all(integer(limits.get(name), 1) for name in required),
            'all lifecycle limits must be positive integer seconds')
    require(limits['request_seconds'] <= limits['all_requests_seconds'],
            'per-request deadline exceeds aggregate request deadline')
    require(limits['startup_seconds'] + limits['all_requests_seconds']
            + limits['worker_bookkeeping_seconds'] <= limits['target_lifetime_seconds'],
            'startup + requests + bookkeeping exceed target lifetime')
    require(limits['target_lifetime_seconds'] + limits['shutdown_seconds']
            <= limits['outer_lifecycle_seconds'],
            'target lifetime + shutdown exceed outer lifecycle deadline')
    require(limits['supervisor_seconds'] + limits['supervisor_call_overhead_seconds']
            <= limits['outer_lifecycle_seconds'],
            'supervisor deadline and call overhead exceed outer deadline')
    require(limits['outer_lifecycle_seconds'] * 5 <= limits['pilot_seconds'],
            'five finite lifecycles exceed pilot wall budget')
    return dict(status='pass', checked=list(required), absolute=True, sliding=False)


def _payload(http, model, prompt, max_tokens):
    return http.make_payload(model, prompt, max_tokens, 1)


def build_schedule(refs, model='qwen35-storage-demand-paper'):
    """Return the preregistered 12-request order without executing anything."""
    sys.path.insert(0, str(ct.ROOT / 'scripts'))
    import bench_ferrule_vllm as http

    by_id = {row['id']: row for row in refs}
    require(set(by_id) == {'qa', 'arithmetic', 'chinese'} and len(refs) == 3,
            'need exactly the three frozen generation8 references')
    rows = [
        ('hello2', 'initial', 'initial_canary', False, 'Hello', 1, 2, None),
        ('canary-qa', 'initial', 'initial_canary', False, by_id['qa']['prompt'], 5, 8, 'qa'),
        ('canary-arithmetic', 'initial', 'initial_canary', False, by_id['arithmetic']['prompt'], 5, 8, 'arithmetic'),
        ('canary-chinese', 'initial', 'initial_canary', False, by_id['chinese']['prompt'], 3, 8, 'chinese'),
        ('qa-first-excluded', 'qa_excluded_first', 'excluded_first', False, by_id['qa']['prompt'], 5, 8, 'qa'),
        ('qa-warm-1', 'qa_warmup', 'warmup', False, by_id['qa']['prompt'], 5, 8, 'qa'),
        ('qa-warm-2', 'qa_warmup', 'warmup', False, by_id['qa']['prompt'], 5, 8, 'qa'),
        ('qa-measured-1', 'qa_measured', 'measured', True, by_id['qa']['prompt'], 5, 8, 'qa'),
        ('qa-measured-2', 'qa_measured', 'measured', True, by_id['qa']['prompt'], 5, 8, 'qa'),
        ('qa-measured-3', 'qa_measured', 'measured', True, by_id['qa']['prompt'], 5, 8, 'qa'),
        ('qa-measured-4', 'qa_measured', 'measured', True, by_id['qa']['prompt'], 5, 8, 'qa'),
        ('qa-measured-5', 'qa_measured', 'measured', True, by_id['qa']['prompt'], 5, 8, 'qa'),
    ]
    result = []
    for index, (name, phase, population, measured, prompt, input_tokens, max_tokens, ref_id) in enumerate(rows):
        result.append(dict(
            index=index, name=name, phase=phase, population=population,
            measured=measured, prompt=prompt, input_tokens=input_tokens,
            max_tokens=max_tokens, reference_id=ref_id, reference=by_id.get(ref_id),
            payload=_payload(http, model, prompt, max_tokens),
            status='scheduled', result=None,
        ))
    validate_schedule(result)
    return result


def validate_schedule(rows, ceiling=12):
    require(type(ceiling) is int and ceiling == 12, 'v2 ceiling must be exactly 12')
    require(isinstance(rows, list) and len(rows) == ceiling,
            'schedule must contain exactly 12 requests')
    require([row.get('index') for row in rows] == list(range(ceiling)),
            'schedule indices must be contiguous and zero-based')
    names = [row.get('name') for row in rows]
    expected = [
        'hello2', 'canary-qa', 'canary-arithmetic', 'canary-chinese',
        'qa-first-excluded', 'qa-warm-1', 'qa-warm-2',
        'qa-measured-1', 'qa-measured-2', 'qa-measured-3',
        'qa-measured-4', 'qa-measured-5',
    ]
    require(names == expected, 'request order differs from preregistration')
    counts = request_phase_counts(rows)
    require(counts == PHASE_COUNTS, 'phase counts do not resolve 4+1+2+5=12')
    require([row['phase'] for row in rows] == ['initial'] * 4 + ['qa_excluded_first'] + ['qa_warmup'] * 2 + ['qa_measured'] * 5,
            'phase order differs')
    for row in rows:
        require(type(row.get('measured')) is bool and row['measured'] is (row['phase'] == 'qa_measured'), 'measured labels drift')
        payload = row.get('payload', {})
        require(integer(row.get('input_tokens'), 1) and integer(row.get('max_tokens'), 1) and row['input_tokens'] + row['max_tokens'] <= 128, 'request context bound drift')
        require(type(payload.get('seed')) is int and payload['seed'] == 1 and type(payload.get('n')) is int and payload['n'] == 1 and payload.get('stream') is True and payload.get('temperature') == 0 and payload.get('stream_options') == dict(include_usage=True) and payload.get('prompt') == row['prompt'] and type(payload.get('max_tokens')) is int and payload['max_tokens'] == row['max_tokens'], 'frozen serial request controls drift')
    return dict(status='pass', count=len(rows), phase_counts=counts, ceiling=ceiling)


def request_phase_counts(rows):
    counts = {key: 0 for key in PHASE_COUNTS}
    for row in rows:
        phase = row.get('phase')
        require(phase in counts, 'unknown request phase: ' + str(phase))
        counts[phase] += 1
    return counts


def _outcome(row):
    if row.get('unissued') is True or row.get('status') == 'unissued':
        return 'unissued'
    if row.get('right_censored') is True or row.get('status') in (
            'timeout', 'censored', 'transport_timeout', 'custody_unknown'):
        return 'censored'
    status = row.get('status')
    if status in ('success', 'pass'):
        return 'success'
    if status in ('failed', 'fail', 'canary_failed', 'error', 'protocol_error', 'http_error'):
        return 'failed'
    return 'censored'


def censoring_summary(rows, schedule=None):
    """Keep scheduled, issued, terminal, censored and unissued denominators."""
    schedule = list(schedule or rows)
    by_name = {row.get('name'): row for row in rows}
    require(len(by_name) == len(rows), 'duplicate request observations')
    require(set(by_name) <= {row.get('name') for row in schedule}, 'unregistered request observation')
    all_rows = []
    for planned in schedule:
        observed = by_name.get(planned.get('name'))
        row = dict(planned)
        if observed is None:
            row.update(status='censored' if planned.get('issued') is True else 'unissued',
                       unissued=planned.get('issued') is not True,
                       censor_reason='issued journal lacks terminal observation' if planned.get('issued') is True else 'not dispatched')
        else:
            row.update(observed)
        require(not (row.get('unissued') is True and (row.get('issued') is True or row.get('sent_at') is not None)), 'issued request cannot be unissued')
        row['outcome'] = _outcome(row)
        all_rows.append(row)
    counts = {key: 0 for key in ('success', 'failed', 'censored', 'unissued')}
    phases = {phase: {key: 0 for key in counts} for phase in PHASE_COUNTS}
    reasons = {}
    for row in all_rows:
        outcome = row['outcome']
        counts[outcome] += 1
        phases[row['phase']][outcome] += 1
        if outcome in ('failed', 'censored', 'unissued'):
            reason = row.get('censor_reason') or row.get('error') or outcome
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
    issued = counts['success'] + counts['failed'] + counts['censored']
    terminal = counts['success'] + counts['failed']
    measured = [row for row in all_rows if row.get('phase') == 'qa_measured']
    measured_success = sum(row['outcome'] == 'success' for row in measured)
    eligible = sum(row['outcome'] == 'success' and row.get('metric_eligible') is True for row in measured)
    return dict(
        scheduled=len(all_rows), issued=issued, terminal=terminal,
        successful=counts['success'], failed=counts['failed'],
        censored=counts['censored'], unissued=counts['unissued'],
        conservation=issued + counts['unissued'] == len(all_rows),
        counts=counts, by_phase=phases, reasons=reasons,
        measured_qa=dict(scheduled=len(measured), success=measured_success,
                         eligible_success=eligible),
        denominator_policy='No failed/censored/unissued slot is removed or relabeled',
    )


def validate_custody_identity(identity, lock, uuid):
    """Validate identity binding; a UUID flock is local serialization, not global reservation."""
    require(isinstance(identity, dict) and integer(identity.get('pid'), 1), 'missing positive owner PID')
    birth = identity.get('birth_ticks', identity.get('start_time'))
    require(integer(birth, 1), 'missing PID birth/start identity')
    require(integer(identity.get('ppid'), 1), 'missing owner parent identity')
    require(isinstance(uuid, str) and UUID_RE.fullmatch(uuid), 'invalid physical GPU UUID')
    require(isinstance(lock, dict) and lock.get('mode') == 'local_nonblocking_flock'
            and lock.get('global_reservation') is False, 'UUID lock is not explicitly local-only')
    return dict(status='pass', pid=identity['pid'], birth_ticks=birth, uuid=uuid,
                local_lock=True, global_reservation=False,
                live_lock_validation_required='contract.validate_owner_lock plus authenticated launch_guard',
                note='PID/birth/parent and UUID are evidence bindings; NFS shared client traffic remains unattributable')


def request_budget_seconds(limits, now, requests_started, target_deadline, outer_deadline, pilot_deadline):
    """A cutoff never slides on progress; zero means do not dispatch."""
    require(all(finite(value) for value in (now, requests_started, target_deadline, outer_deadline, pilot_deadline)), 'finite monotonic deadlines required')
    require(now >= requests_started, 'request clock precedes phase start')
    return max(0.0, min(float(limits['request_seconds']), requests_started + limits['all_requests_seconds'] - now, target_deadline - now, outer_deadline - now, pilot_deadline - now))
