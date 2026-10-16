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


def model_summary_policy():
    return dict(status="PARTIAL_OBSERVED_MODEL_SUMMARY",
        device_binding="unique model CUDA0 to exact VM/GPU-id/UUID; inventory is not activity",
        cuda_provenance="all exported runtime/driver APIs and kernel/memcpy/memset/sync rows must bind model/device0",
        memcpy_endpoints="explicit off-device rejects; ordinary null stays unobserved; peer requires both endpoints",
        warning_mechanism="exact official Analysis/HostTimestamp only",
        model_warning="exact no-NVTX only; no expert/range attribution",
        nonmodel_warning="exact no-payload with no observed CUDA activity; SQL-unbound subjects are not owned",
        scheduling_warning="exact global absent scheduling only when collection disabled; no CPU scheduling claims",
        unknown_warning_or_loss="reject", full_trace_accepted=False,
        dropped_event_count=None, api_count_exhaustiveness="unproven",
        cpu_scheduling_claim=False, expert_range_claim=False)


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
    gpu_rows = maintained.rows(db, "TARGET_INFO_GPU")
    gpus = {(r["vmId"], r["id"]): r for r in gpu_rows}
    require(len(gpus) == len(gpu_rows), "Duplicate VM/GPU inventory identity")
    vm = model & ~((1 << 48) - 1)
    devices = [r for r in maintained.rows(db, "TARGET_INFO_CUDA_DEVICE") if r["pid"] == model_pid]
    selected = [r for r in devices if r["cudaId"] == 0]
    require(len({r["cudaId"] for r in devices}) == len(devices) and len(selected) == 1,
            "Unique actual model CUDA0 mapping required; inventory is not activity")
    gpu = gpus.get((vm, selected[0]["gpuId"]))
    matching = [r for r in gpu_rows if r["vmId"] == vm and isinstance(r["uuid"], str)
                and "GPU-" + r["uuid"].removeprefix("GPU-") == uuid]
    require(gpu is not None and matching == [gpu] and gpu["id"] == 0, "Actual model CUDA ordinal/UUID differs")
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
    observed_tables, api_provenance = {}, {}
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            observed_tables[table] = 'absent; not a zero-count assertion'
            continue
        observed_tables[table] = db.execute('SELECT count(*) FROM ' + table).fetchone()[0]
        # Validate the whole table before filtering for model aggregation. A
        # wrapper/unbound CUDA caller is not an absent-payload warning waiver.
        provenance = [dict(r) for r in db.execute(
            "SELECT (globalTid & ?) AS process_key,count(*) AS calls,"
            "sum(globalTid IS NULL OR globalTid<=0 OR typeof(globalTid)!='integer') AS invalid "
            "FROM " + table + " GROUP BY process_key", (~maintained.MASK,))]
        require(all(r["process_key"] == model and r["invalid"] == 0 for r in provenance),
                "Foreign/unbound CUDA API provenance")
        api_provenance[table] = provenance
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
    activity_tables = {}
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY",
                  "CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        if table not in tables:
            activity_tables[table] = dict(state="unobserved", reason="table absent; not zero")
            continue
        columns = {r[1] for r in db.execute("PRAGMA table_info(" + table + ")")}
        require({"globalPid", "deviceId", "start", "end"} <= columns, "Unknown activity provenance schema")
        row = dict(db.execute("SELECT count(*) AS count,sum(end-start) AS observed_interval_sum_ns,"
            "sum(globalPid IS NULL OR globalPid!=? OR deviceId IS NULL OR deviceId!=0 "
            "OR typeof(start)!='integer' OR typeof(end)!='integer' OR start<0 OR start>=end OR end>?) AS invalid "
            "FROM " + table, (model, duration)).fetchone())
        require(row["invalid"] in (None, 0), "GPU activity from foreign/unbound model process/device or outside capture")
        activity_tables[table] = dict(state="observed", **row)
        if table == "CUPTI_ACTIVITY_KIND_MEMSET":
            require("bytes" in columns, "Unknown memset byte schema")
            total = db.execute("SELECT sum(bytes),sum(typeof(bytes)!='integer' OR bytes<=0) FROM " + table).fetchone()
            require(total[1] in (None, 0), "Unknown/invalid memset bytes")
            activity_tables[table]["bytes"] = total[0]
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
    endpoints = {key: Counter() for key in ('srcDeviceId', 'dstDeviceId')}
    for row in copies:
        require(0 <= row['start'] < row['end'] <= duration and type(row['bytes']) is int and row['bytes'] > 0,
                'Copy outside capture or bytes unknown')
        kind = copy_kinds.get(row["copyKind"])
        require(kind is not None and kind != "CUDA_MEMCPY_KIND_UNKNOWN", "Unknown memcpy kind")
        for field, counts in endpoints.items():
            value = row.get(field)
            require(value is None or type(value) is int and value == 0, "Explicit off-device memcpy endpoint")
            counts["unobserved" if value is None else "observed_device0"] += 1
        if kind == "CUDA_MEMCPY_KIND_PTOP":
            require(all(row.get(field) == 0 for field in endpoints), "Unknown peer memcpy endpoint blocks one-GPU claim")
        value = copy_counts[kind]
        value['count'] += 1
        value['bytes'] += row['bytes']
        value['device_duration_ns'] += row['end'] - row['start']
    require(copy_counts.get('CUDA_MEMCPY_KIND_HTOD', {}).get('bytes', 0) > 0, 'No actual model H2D bytes')
    severities = {r["id"]: r["name"] for r in maintained.rows(db, "ENUM_DIAGNOSTIC_SEVERITY_LEVEL")}
    origins = {}
    for table in ("ENUM_DIAGNOSTIC_SOURCE_TYPE", "ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE"):
        entries = maintained.rows(db, table) if table in tables else []
        origins[table] = {r["id"]: r["name"] for r in entries}
        require(len(origins[table]) == len(entries), "Ambiguous diagnostic origin enum")
    capture = maintained.rows(db, "META_DATA_CAPTURE")
    scheduling_disabled = [r["value"] for r in capture if r["name"] == "COLLECT_THREAD_STATE_TRACE"] == ["false"]
    diagnostics = maintained.rows(db, "DIAGNOSTIC_EVENT")
    classifications = []
    for row in diagnostics:
        severity, text = severities.get(row["severity"]), row["text"]
        require(severity in ("Info", "Verbose", "Warning") and isinstance(text, str) and not scope_contract.LOSS.search(text),
                "Official loss/error/unknown diagnostic; preserve and stop")
        classification, mechanism, binding, effect = "informational", None, None, None
        subject = row["globalPid"]
        if severity == "Warning":
            require(row["source"] == 3 and row["timestampType"] == 2
                and origins["ENUM_DIAGNOSTIC_SOURCE_TYPE"].get(3) == "Analysis"
                and origins["ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE"].get(2) == "HostTimestamp",
                "Unknown official warning origin/mechanism")
            mechanism = "official_analysis_host_timestamp"
            if text == scope_contract.SCHEDULING:
                require(subject == scope_contract.GLOBAL_DIAGNOSTIC_PID and scheduling_disabled,
                        "Scheduling absent without exact global subject/disabled collection")
                classification, binding, effect = "absent_scheduling", "global", "no_CPU_scheduling_or_thread_activity_claim"
            else:
                require(type(subject) is int and subject > 0 and subject == maintained.process_key(subject),
                        "Unknown warning process subject")
                matches = [r for r in processes if r["globalPid"] == subject]
                require(len(matches) <= 1, "Ambiguous warning SQL process binding")
                require(not matches or matches[0]["pid"] == maintained.pid(subject), "Warning SQL process encoding differs")
                binding = ("model_sql_key_externally_bound_by_caller" if subject == model else
                           "wrapper_sql_key_externally_bound_by_caller" if subject == wrapper else
                           "sql_process_only_not_owned" if matches else "unbound_not_owned")
                require(text in scope_contract.NO_PAYLOAD and
                        (subject != model or text == scope_contract.NO_PAYLOAD[0]),
                        "Unclassified/model CUDA-failure official warning affects CUDA-count scope")
                classification = "model_absent_nvtx" if subject == model else "nonmodel_absent_payload"
                effect = ("no_expert_or_range_claim" if subject == model else
                          "partial_model_summary_only_no_observed_foreign_CUDA_activity; no_ownership_inference")
        classifications.append(dict(raw=row, classification=classification, mechanism=mechanism,
                                    subject_global_pid=subject, binding=binding, effect=effect))
    return dict(schema='ferrule.retained-upload.observed-cuda-api-summary.v1',
        status='PARTIAL_OBSERVED_MODEL_SUMMARY', policy=model_summary_policy(),
        model_pid=model_pid, wrapper_pid=wrapper_pid, model_global_pid=model, uuid=uuid,
        device_binding=dict(vm_id=vm, cuda_id=0, gpu_id=gpu['id'], uuid=uuid, model_inventory_rows=len(devices)),
        api_provenance=api_provenance, activity_tables=activity_tables,
        memcpy_endpoint_observations={key: dict(value) for key, value in endpoints.items()},
        api_tables=observed_tables, apis=api_rows, kernels=dict(kernel_counts), copies=dict(copy_counts),
        diagnostics=classifications, official_export_metadata=metadata,
        event_query_counts_scope='observed readiness queries including not-ready returns; serial spin counts, not waits',
        duration_scope='API host duration sums and copy device duration sums may overlap; neither is elapsed latency',
        dropped_event_count=None, cpu_scheduling_claim=False, expert_range_claim=False,
        full_trace_accepted=False, api_count_exhaustiveness='unproven; never infer absent calls or dropped_events=0',
        ttft_token_or_per_request_latency=False, overlap_claim=False, performance_claim=False)
