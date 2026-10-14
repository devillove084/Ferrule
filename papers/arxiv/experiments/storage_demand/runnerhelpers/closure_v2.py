"""Read actual owner-local closure JSON; never issue a cleanup/release certificate."""
from __future__ import annotations

import json

from .evidence_v2 import integer, require


def evidence(value):
    if not isinstance(value, dict):
        return 'missing', None
    state = value.get('state')
    if state not in ('unobserved', 'observed', 'verified', 'unknown'):
        return 'invalid', None
    if state in ('observed', 'verified') and 'value' not in value:
        return 'invalid', None
    return state, value.get('value')


def unknown_paths(value, path=''):
    result = []
    if isinstance(value, dict):
        if value.get('state') == 'unknown':
            result.append(path)
        for key, child in value.items():
            result.extend(unknown_paths(child, path + '.' + key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(unknown_paths(child, path + '.' + str(index)))
    return result


def closure_from_resource_summary(summary, owner_retained=None, mapped_owner=False):
    """Known close permits retained resources, but missing is never quiescent."""
    require(isinstance(summary, dict), 'resource shutdown summary must be an object')
    runtime = summary.get('runtime') if isinstance(summary.get('runtime'), dict) else {}
    physical = summary.get('physical') if isinstance(summary.get('physical'), dict) else {}
    projections = {name: evidence(value) for name, value in (
        ('runtime_fence', runtime.get('fence')), ('physical_fence', physical.get('quiescence')),
        ('runtime_closure', runtime.get('closure')), ('physical_close', physical.get('close')),
        ('residency', physical.get('residency')), ('release', physical.get('release')),
        ('pending', runtime.get('pending_operations')), ('runtime_quarantine', runtime.get('quarantine')),
        ('physical_quarantine', physical.get('quarantine')), ('scope', physical.get('scope')),
        ('expert_cache', physical.get('expert_cache')), ('allocator', physical.get('allocator')),
    )}
    states = {key: state for key, (state, _) in projections.items()}
    values = {key: value for key, (_, value) in projections.items()}
    cache = values['expert_cache'] if isinstance(values['expert_cache'], dict) else {}
    scope = values['scope'] if isinstance(values['scope'], dict) else {}
    require(owner_retained is None, 'owner retention must come from actual summary, not a caller assertion')
    retained = values['residency'] == 'retained' if states['residency'] in ('observed', 'verified') else None
    reasons = unknown_paths(summary)
    for name in ('runtime_fence', 'physical_fence'):
        if values[name] == 'unknown':
            reasons.append(name + '.value')
    if cache.get('quarantined') is True or cache.get('unknown_quarantine_bytes', 0) != 0:
        reasons.append('physical.expert_cache.quarantine')
    for name in ('runtime_quarantine', 'physical_quarantine'):
        if values[name] is True:
            reasons.append(name)
    custody = 'unknown' if reasons else 'not_established'
    missing = []
    criteria = dict(
        schema=type(summary.get('schema_version')) is int and summary['schema_version'] == 1,
        error=summary.get('error') is None,
        outcome=summary.get('outcome') == 'complete',
        mapped_owner=mapped_owner is True,
        scope=states['scope'] in ('observed', 'verified')
            and integer(scope.get('owner'), 1) and integer(scope.get('generation'), 1),
        runtime_closure=values['runtime_closure'] == 'closed',
        physical_close=values['physical_close'] == 'closed',
        physical_fence=values['physical_fence'] == 'quiescent',
        pending=type(values['pending']) is int and values['pending'] == 0,
        physical_quarantine=values['physical_quarantine'] is False,
        expert_cache=type(cache.get('pending_upload_bytes')) is int and cache['pending_upload_bytes'] == 0
            and type(cache.get('unknown_quarantine_bytes')) is int and cache['unknown_quarantine_bytes'] == 0
            and cache.get('quarantined') is False,
    )
    missing.extend(key for key, valid in criteria.items() if not valid)
    pending = (summary.get('outcome') == 'pending' or values['physical_fence'] == 'pending' or values['runtime_fence'] == 'pending'
               or (integer(values['pending']) and values['pending'] > 0)
               or (integer(cache.get('pending_upload_bytes')) and cache['pending_upload_bytes'] > 0))
    allocator = values['allocator'] if isinstance(values['allocator'], dict) else {}
    pending = pending or any(integer(allocator.get(key)) and allocator[key] > 0 for key in ('pending_unfenced_bytes', 'pending_retirement_bytes'))
    if not reasons and not missing and not pending:
        custody = 'known_closed_owner_local'
    # The report is explicitly not a universal/final-owner release certificate,
    # even if individual source fields carry the Verified qualifier.
    return dict(
        schema='ferrule.storage-demand.closure-evidence.v2',
        owner_retained=retained, mapped_owner=mapped_owner,
        fences={
            'runtime': dict(state=states['runtime_fence'], value=values['runtime_fence']),
            'physical': dict(state=states['physical_fence'], value=values['physical_fence']),
        },
        owner_close={key: dict(state=states[key], value=values[key]) for key in ('runtime_closure', 'physical_close')},
        release=dict(state=states['release'], value=values['release']),
        physical_scope=dict(state=states['scope'], value=values['scope']),
        allocator=physical.get('allocator'), broker_domains=summary.get('broker_domains'),
        coverage=summary.get('coverage'),
        resource_contract_complete='failed' if summary.get('outcome') == 'error' else 'unverified',
        custody_state=custody, unknown_reasons=reasons, missing_close_evidence=missing,
        pending=pending, can_continue=custody == 'known_closed_owner_local',
        universal_final_release_verified=False,
        interpretation='Process exit/drain/post-idle remain independent gates; retained residency and unobserved final release are not unknown custody',
    )


def stop_on_unknown_custody(closure):
    """Do not conflate missing coverage or retained residency with Unknown."""
    return closure.get('custody_state') == 'unknown'


def parse_resource_shutdown_summaries(lines, mapped_owner=False):
    rows = []
    errors = []
    sticky_unknown = []
    for number, raw in enumerate(lines, 1):
        try:
            event = json.loads(raw)
        except (TypeError, ValueError, UnicodeError):
            continue
        fields = event.get('fields', event) if isinstance(event, dict) else {}
        if not isinstance(fields, dict):
            continue
        if fields.get('event') == 'resource_shutdown_summary_serialization_error':
            errors.append(dict(line=number, reason='producer serialization error'))
        if fields.get('event') != 'resource_shutdown_summary':
            continue
        try:
            summary = json.loads(fields['summary_json'])
            require(isinstance(summary, dict), 'nonobject shutdown summary')
            require(summary.get('attempt') == fields.get('attempt')
                    and summary.get('outcome') == fields.get('outcome'), 'summary envelope mismatch')
            interpreted = closure_from_resource_summary(summary, mapped_owner=mapped_owner)
            sticky_unknown.extend(interpreted['unknown_reasons'])
            rows.append(dict(line=number, raw=event, summary=summary, interpreted=interpreted))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(dict(line=number, reason=str(error)))
    last = rows[-1]['interpreted'].copy() if rows else dict(
        custody_state='not_established', resource_contract_complete='unverified', can_continue=False)
    if sticky_unknown:
        last.update(custody_state='unknown', can_continue=False, unknown_reasons=sticky_unknown)
    if errors:
        last['can_continue'] = False
    return dict(attempts=rows, errors=errors, final=last, sticky_unknown=bool(sticky_unknown))
