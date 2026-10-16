"""CPU-only completeness layer for optional official exports; never launches nsys.

The maintained auditor handles direct API/activity correlations. Graph launches
need explicit exported node-to-launch relations and may not be treated as lost
one-to-one direct submissions. Missing coverage is unverified, not PASS.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tp2_resident'))
import profile_trace_audit_v2 as direct
from runnerhelpers.evidence_v2 import require

GRAPH_APIS = {'cuGraphLaunch', 'cudaGraphLaunch'}


def audit(activities, api_rows, windows, model_pid, graph_expansions=None, completeness=None):
    require(isinstance(model_pid, int) and model_pid > 0, 'actual model PID required')
    graph_expansions = graph_expansions or []
    graph_apis = [r for r in api_rows if r.get('name', '').split('_v')[0] in GRAPH_APIS
                  and direct.process_id(r['globalTid']) == model_pid]
    graph_keys = {(direct.process_id(r['globalTid']), r['correlationId']) for r in graph_apis}
    issues, consumed = [], set()
    require(len(graph_keys) == len(graph_apis), 'duplicate graph launch correlation key')
    expansions = {(r.get('pid'), r.get('launch_correlation_id')): r for r in graph_expansions}
    require(len(expansions) == len(graph_expansions), 'duplicate graph expansion relation')
    graph_nodes = set()
    for launch in graph_apis:
        key = (model_pid, launch['correlationId'])
        expansion = expansions.get(key)
        if not expansion or expansion.get('official_export_relation') is not True:
            issues.append('missing official exported graph expansion: ' + str(key))
            continue
        nodes = expansion.get('activities', [])
        if launch.get('returnValue') != 0 or not nodes:
            issues.append('failed/empty graph launch expansion: ' + str(key))
            continue
        label = direct.containing_window(launch, windows)
        if label is None:
            issues.append('graph launch outside registered request window')
        for node in nodes:
            node_key = (node.get('globalPid'), node.get('kind'), node.get('start'), node.get('end'), node.get('correlationId'))
            matching = [r for r in activities if (r.get('globalPid'), r.get('kind'), r.get('start'), r.get('end'), r.get('correlationId')) == node_key]
            if len(matching) != 1 or node_key in graph_nodes or direct.process_id(node.get('globalPid', 0)) != model_pid:
                issues.append('graph node missing/duplicate/foreign process')
                continue
            if direct.containing_window(matching[0], windows) != label:
                issues.append('graph node/request window mismatch')
            graph_nodes.add(node_key)
        if expansion.get('expected_node_count') != len(nodes):
            issues.append('graph expansion node denominator unknown/mismatch')
        consumed.add(key)
    if set(expansions) != consumed:
        issues.append('unbound graph expansion relation')
    ordinary = [r for r in activities if (r.get('globalPid'), r.get('kind'), r.get('start'), r.get('end'), r.get('correlationId')) not in graph_nodes]
    result = direct.audit_correlations(ordinary, [r for r in api_rows if r not in graph_apis], windows, model_pid)
    completeness = completeness or {}
    complete = (completeness.get('schema') == 'ferrule.storage-demand.official-nsys-completeness.v1'
        and completeness.get('official_full_export') is True
        and completeness.get('collector_custody_drained') is True
        and completeness.get('clock_pid_uuid_request_bindings_verified') is True
        and completeness.get('loss_diagnostics_available') is True
        and completeness.get('dropped_events') == 0
        and completeness.get('unknown_event_loss') is False
        and completeness.get('warnings') == []
        and completeness.get('independent_collection_coverage_verified') is True)
    # This deliberately cannot manufacture collection coverage from correlated rows.
    return dict(direct_correlation=result, graph_launches=len(graph_apis), graph_nodes=len(graph_nodes),
        graph_issues=issues, official_collection_completeness_verified=complete,
        full_trace_accepted=False, scientific_acceptance=False,
        interpretation='Requires independent official coverage and collector acceptance; this layer alone never authorizes a trace or overlap claim')


def process_api_summary(db, model_pid, wrapper_pid, uuid):
    """Observed whole-process APIs, not complete coverage or prediction ranges.

    Use the maintained official schema/process-key parser. In particular, an
    absent DRIVER table does not mean zero driver calls: this Nsight release
    also serializes cu* calls in its RUNTIME table.
    """
    from collections import Counter, defaultdict
    import re
    import sqlite3
    from profile_overlap_v1 import analyze as maintained, scope_contract

    db.row_factory = sqlite3.Row
    tables, metadata = maintained.inspect_schema(db)
    processes = maintained.rows(db, 'PROCESSES')
    keys = [r['globalPid'] for r in processes if r['pid'] == model_pid]
    wrappers = [r['globalPid'] for r in processes if r['pid'] == wrapper_pid]
    require(model_pid != wrapper_pid and len(keys) == len(wrappers) == 1,
            'Unique birth-bound model and wrapper SQL process keys required')
    model, wrapper = keys[0], wrappers[0]
    require(maintained.pid(model) == model_pid and maintained.pid(wrapper) == wrapper_pid,
            'SQL process encoding differs')
    gpus = {r['id']: r for r in maintained.rows(db, 'TARGET_INFO_GPU')}
    devices = [r for r in maintained.rows(db, 'TARGET_INFO_CUDA_DEVICE') if r['pid'] == model_pid]
    require(len(devices) == 1 and all(r['cudaId'] == 0 and 'GPU-' + gpus[r['gpuId']]['uuid'].removeprefix('GPU-') == uuid
                and gpus[r['gpuId']]['vmId'] == model & ~((1 << 48) - 1) for r in devices),
            'Actual model CUDA ordinal/UUID differs')
    strings = {r['id']: r['value'] for r in maintained.rows(db, 'StringIds')}
    api = defaultdict(lambda: dict(count=0, host_duration_ns=0, return_values=Counter()))
    duration_rows = maintained.rows(db, 'ANALYSIS_DETAILS')
    require(duration_rows and all(r['duration'] > 0 for r in duration_rows), 'Missing finite capture duration')
    duration = max(r['duration'] for r in duration_rows)
    family = (
        ('event_query', r'^(?:cu|cuda)EventQuery'),
        ('event_create', r'^(?:cu|cuda)EventCreate'),
        ('event_record', r'^(?:cu|cuda)EventRecord'),
        ('event_destroy', r'^(?:cu|cuda)EventDestroy'),
        ('event_synchronize', r'^(?:cu|cuda)EventSynchronize'),
        ('stream_wait_event', r'^(?:cu|cuda)StreamWaitEvent'),
        ('host_synchronize', r'^(?:cuCtx|cudaDevice|cuStream|cudaStream)Synchronize'),
        ('host_pin_unpin', r'^(?:cuMemHost|cuMemAllocHost|cuMemFreeHost|cudaHost|cudaMallocHost|cudaFreeHost)'),
        ('allocation_free', r'^(?:cuMemAlloc|cuMemFree|cudaMalloc|cudaFree)'),
        ('memcpy', r'^(?:cu|cuda)Memcpy'),
        ('kernel_launch', r'^(?:cu|cuda)Launch'),
    )
    observed_tables = {}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            observed_tables[table] = 'absent; not a zero-count assertion'
            continue
        observed_tables[table] = db.execute('SELECT count(*) FROM ' + table).fetchone()[0]
        # Aggregate in SQLite instead of materializing potentially large serial
        # ready-query spin traces in Python. Table names above are closed literals.
        records = db.execute('SELECT nameId,returnValue,count(*) AS calls,sum(end-start) AS ns,'
            'min(start) AS first,max(end) AS last,sum(start>=end) AS invalid FROM ' + table +
            ' WHERE (globalTid & ?) = ? GROUP BY nameId,returnValue', (~maintained.MASK, model))
        for row in records:
            require(0 <= row['first'] < row['last'] <= duration and row['invalid'] == 0,
                    'API outside finite capture')
            name = strings[row['nameId']]
            value = api[(table, name)]
            value['count'] += row['calls']
            value['host_duration_ns'] += row['ns']
            value['return_values'][str(row['returnValue'])] += row['calls']
            # Both pinned CUDA12.4 headers define NOT_READY as600; the legacy
            # runtime34 value must not hide an error on this exact toolchain.
            require(row['returnValue'] == 0 or ('EventQuery' in name and row['returnValue'] == 600),
                    'CUDA API failure outside explicit readiness-query not-ready codes')
    api_rows = [dict(table=k[0], name=k[1], family=next((n for n, pattern in family if re.match(pattern, k[1])), 'other'),
                     **dict(v, return_values=dict(v['return_values']))) for k, v in sorted(api.items())]
    require(api_rows, 'No model CUDA API observations')
    kernels = maintained.rows(db, 'CUPTI_ACTIVITY_KIND_KERNEL')
    copies = maintained.rows(db, 'CUPTI_ACTIVITY_KIND_MEMCPY')
    require(all(r['globalPid'] == model and r['deviceId'] == 0 for r in kernels + copies),
            'GPU activity from foreign/unbound model process/device')
    require(kernels and copies, 'Positive real model kernel/copy evidence required')
    kernel_counts = Counter()
    for row in kernels:
        require(0 <= row['start'] < row['end'] <= duration, 'Kernel outside capture')
        kernel_counts[strings[row['demangledName']]] += 1
    copy_kinds = {r['id']: r['name'] for r in maintained.rows(db, 'ENUM_CUDA_MEMCPY_OPER')}
    copy_counts = defaultdict(lambda: dict(count=0, bytes=0, device_duration_ns=0))
    for row in copies:
        require(0 <= row['start'] < row['end'] <= duration and type(row['bytes']) is int and row['bytes'] > 0,
                'Copy outside capture or bytes unknown')
        kind = copy_kinds[row['copyKind']]
        value = copy_counts[kind]
        value['count'] += 1
        value['bytes'] += row['bytes']
        value['device_duration_ns'] += row['end'] - row['start']
    require(copy_counts.get('CUDA_MEMCPY_KIND_HTOD', {}).get('bytes', 0) > 0, 'No actual model H2D bytes')
    severities = {r['id']: r['name'] for r in maintained.rows(db, 'ENUM_DIAGNOSTIC_SEVERITY_LEVEL')}
    diagnostics = maintained.rows(db, 'DIAGNOSTIC_EVENT')
    classifications = []
    for row in diagnostics:
        severity, text = severities.get(row['severity']), row['text']
        require(severity in ('Info', 'Verbose', 'Warning') and not scope_contract.LOSS.search(text),
                'Official loss/error/unknown diagnostic; preserve and stop')
        classification = 'informational'
        if severity == 'Warning':
            # These exact predeclared absent-payload warnings do not assert CUDA
            # loss. They still prohibit full-trace acceptance, as in the old auditor.
            official = row.get('source') == 3 and row.get('timestampType') == 2
            allowed = official and (
                (row['globalPid'] == wrapper and text in scope_contract.NO_PAYLOAD)
                or (row['globalPid'] == model and text == 'No NVTX events collected. Does the process use NVTX?')
                or (row['globalPid'] == scope_contract.GLOBAL_DIAGNOSTIC_PID and text == scope_contract.SCHEDULING))
            require(allowed, 'Unclassified official warning affects CUDA-count scope')
            classification = 'declared_absent_nonmeasured_payload_not_full_trace_waiver'
        classifications.append(dict(raw=row, classification=classification))
    return dict(schema='ferrule.retained-upload.observed-cuda-api-summary.v1',
        model_pid=model_pid, wrapper_pid=wrapper_pid, model_global_pid=model, uuid=uuid,
        api_tables=observed_tables, apis=api_rows, kernels=dict(kernel_counts), copies=dict(copy_counts),
        diagnostics=classifications, official_export_metadata=metadata,
        event_query_counts_scope='observed readiness queries including not-ready returns; serial spin counts, not waits',
        duration_scope='API host duration sums and copy device duration sums may overlap; neither is elapsed latency',
        full_trace_accepted=False, api_count_exhaustiveness='unproven; never infer absent calls or dropped_events=0',
        ttft_token_or_per_request_latency=False, overlap_claim=False, performance_claim=False)
