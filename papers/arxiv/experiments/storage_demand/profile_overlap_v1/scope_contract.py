"""Claim-specific diagnostic policy; never upgrades a historical or partial trace.

SQL evidence alone cannot authenticate process birth, source, HTTP quality, or
cleanup. Those facts belong to the sealed outer lifecycle. This module keeps
that distinction explicit rather than treating warning classification as PASS.
"""
from __future__ import annotations

import re

SCHEMA = 'ferrule.storage-demand.model-inference-overlap-scope.v1'
SCHEDULING = ('Scheduling information is absent. The thread activity is deduced based on OS runtime libraries traces. '
              'This is inaccurate and does not take into account asynchronous interrupts and exception faults.')
NO_PAYLOAD = ('No NVTX events collected. Does the process use NVTX?',
              'CUDA profiling might have not been started correctly.',
              'No CUDA events collected. Does the process use CUDA?')
GLOBAL_DIAGNOSTIC_PID = -144115188075855872
LOSS = re.compile(r'not all|dropp?ed|lost|loss|incomplete|overflow|truncat|buffer.*fail|'
                  r'profiling.*fail|collection.*fail|partial.*collect|events.*skip|stringstorage|numtpcs|targetprofilingfailed', re.I)


def policy(cpuctxsw):
    if cpuctxsw not in ('none', 'process-tree'):
        raise ValueError('unsupported CPU context-switch policy')
    return dict(schema=SCHEMA, claim='model inference userspace pread wall intervals intersect actual expert GPU math kernels',
        cpuctxsw=cpuctxsw, cpu_sampling='none', gpu_metrics=False,
        thread_activity_claim=False, cpu_scheduling_claim=False, active_io_claim=False,
        physical_nas_claim=False, read_envelope_includes_allocation=True,
        pread_wall_includes_preemption=True, dropped_event_count=None,
        full_trace_warning_policy='every official warning rejects full-trace acceptance',
        scheduling_warning_policy=('reject: collection was explicitly requested' if cpuctxsw == 'process-tree'
                                   else 'only exact official global absent-scheduling diagnostic is outside this claim'),
        wrapper_warning_policy='only exact no-payload diagnostics on birth/source-bound non-CUDA wrapper roles; never model',
        unknown_warning_policy='reject', loss_policy='reject in every process and diagnostic severity',
        closure_policy='whole-export bidirectional process-key/API/kernel closure plus expected QA ranges and typed read counters',
        historical_acceptance_transferred=False)


def scoped_diagnostics(diagnostics, severities, context, model, processes):
    """Classify exact raw rows without deleting them or relaxing global rejection."""
    declared = context.get('scope_contract')
    valid = (isinstance(declared, dict) and declared.get('cpuctxsw') in ('none', 'process-tree')
             and declared == policy(declared['cpuctxsw']))
    issues = [] if valid else ['missing or unrecognized predeclared overlap scope']
    roles = context.get('process_roles', [])
    model_roles = [r for r in roles if r.get('role') == 'model']
    by_key = {r['globalPid']: r['pid'] for r in processes}
    seen = set()
    bound = {}
    for role in roles:
        key = role.get('globalPid')
        identity = role.get('identity', {})
        good = (type(key) is int and key > 0 and key not in seen
                and key & ((1 << 24) - 1) == 0
                and key & ~((1 << 48) - 1) == model & ~((1 << 48) - 1)
                and (key >> 24) & ((1 << 24) - 1) == identity.get('pid')
                and by_key.get(key) == identity.get('pid')
                and all(type(identity.get(k)) is int and identity[k] > 0 for k in ('pid', 'birth_ticks', 'ppid'))
                and identity.get('state') not in ('Z', 'X')
                and role.get('role') in ('model', 'non_cuda_wrapper')
                and role.get('executable_sha256') == context.get('role_executable_sha256', {}).get(role.get('role'))
                and role.get('source_binding_sha256') == context.get('role_source_binding_sha256', {}).get(role.get('role'))
                and all(isinstance(role.get(k), str) and re.fullmatch(r'[0-9a-f]{64}', role[k])
                        for k in ('executable_sha256', 'source_binding_sha256')))
        seen.add(key)
        if not good:
            issues.append('process role lacks unique SQL PID and external PID/birth/source binding')
        else:
            bound[key] = role
    if not (len(model_roles) == 1 and model in bound and bound[model]['role'] == 'model'
            and bound[model]['identity']['pid'] == context.get('model_pid')
            and bound[model]['identity']['birth_ticks'] == context.get('model_birth_ticks')):
        issues.append('actual model role binding missing or inconsistent')
    if model in bound:
        wrappers = [r for r in bound.values() if r['role'] == 'non_cuda_wrapper']
        if not (len(wrappers) == 1 and bound[model]['identity']['ppid'] == wrappers[0]['identity']['pid']
                and bound[model]['identity']['birth_ticks'] >= wrappers[0]['identity']['birth_ticks']):
            issues.append('model is not the birth-bound wrapper direct child')
    classifications = []
    for row in diagnostics:
        severity = severities.get(row['severity'])
        message = row['text']
        classification, affects_scope = 'informational', False
        if LOSS.search(message) or severity not in ('Info', 'Verbose', 'Warning'):
            classification, affects_scope = 'loss_error_or_unknown_severity', True
        elif severity == 'Warning':
            classification, affects_scope = 'unknown_or_model_warning', True
            official = row.get('source') == 3 and row.get('timestampType') == 2
            if (valid and official and message == SCHEDULING
                    and row.get('globalPid') == GLOBAL_DIAGNOSTIC_PID and declared['cpuctxsw'] == 'none'):
                classification, affects_scope = 'global_thread_state_not_measured', False
            elif (valid and official and message in NO_PAYLOAD and row.get('globalPid') != model
                  and bound.get(row.get('globalPid'), {}).get('role') == 'non_cuda_wrapper'):
                classification, affects_scope = 'authenticated_non_cuda_wrapper_no_payload', False
        if affects_scope:
            issues.append('scope-invalid diagnostic: ' + classification)
        classifications.append(dict(raw=row, classification=classification, affects_scoped_overlap=affects_scope,
                                    affects_full_trace=severity == 'Warning' or affects_scope))
    return dict(policy_valid=valid, issues=sorted(set(issues)), classifications=classifications,
                process_roles=roles, dropped_event_count=None)


def sql_eligible(report):
    return (report.get('scoped_sql_evidence_consistent') is True and report.get('scoped_issues') == []
            and report.get('log_issues') == [] and report.get('loss_diagnostic_count') == 0
            and 'dropped_event_count' in report and report['dropped_event_count'] is None)


def acceptance(sql_report, gates):
    """Final predicate, called only after independent raw lifecycle checks.

    A caller cannot infer any gate from matching CUDA/OSRT table row counts.
    The worker's SQL audit is eligible evidence, not final pair acceptance.
    """
    required = ('one_frozen_qa_http_token_join', 'authenticated_model_pid_birth_source',
                'ready_and_after_raw_maps', 'dynamic_owner_pair_closed',
                'authenticated_echild', 'original_all_eight_stable_idle', 'collector_export_closed')
    complete = all(gates.get(k) is True for k in required) and sql_eligible(sql_report)
    return dict(scoped_overlap_accepted=complete,
                full_trace_accepted=complete and sql_report.get('status') == 'sql_evidence_consistent'
                    and sql_report.get('warnings') == [],
                missing_gates=[k for k in required if gates.get(k) is not True],
                dropped_event_count=sql_report.get('dropped_event_count'),
                simultaneous_unreported_loss_excluded=False, universal_trace_completeness_established=False)


