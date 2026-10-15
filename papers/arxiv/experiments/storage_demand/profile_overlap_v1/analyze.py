"""Read-only Nsight 2025.3 overlap analysis; row coverage is not acceptance.

Only actual GPU kernel intervals enter the compute union. NVTX layer.expert is
an API submission attribution range, never a GPU-duration proxy. OSRT pread time
is userspace file-API time, not physical NAS service time.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3

from profile_overlap_v1 import scope_contract

MASK = (1 << 24) - 1
LAUNCH = re.compile(r'^(?:cudaLaunch(?:Kernel(?:ExC)?|CooperativeKernel)|cuLaunch(?:Kernel(?:Ex)?|CooperativeKernel))(?:_v[0-9]+|_ptsz|_ptds)*$')
LABELS = {
    'read_subset': ('operation', 'source_generation', 'requested_bytes'),
    'layer.expert': ('image_generation', 'layer', 'expert', 'rows'),
    'layer.expert.upload': ('image_generation', 'layer', 'expert', 'rows'),
}
REQUIRED = {
    'StringIds': 'id value', 'PROCESSES': 'globalPid pid',
    'TARGET_INFO_GPU': 'id uuid vmId', 'TARGET_INFO_CUDA_DEVICE': 'pid cudaId gpuId',
    'NVTX_EVENTS': 'start end eventType rangeId text textId globalTid endGlobalTid domainId',
    'CUPTI_ACTIVITY_KIND_RUNTIME': 'start end globalTid correlationId nameId returnValue',
    'CUPTI_ACTIVITY_KIND_KERNEL': 'start end globalPid deviceId correlationId demangledName graphNodeId',
    'OSRT_API': 'start end eventClass globalTid nameId returnValue',
        'ENUM_NSYS_EVENT_CLASS': 'id name',
    'DIAGNOSTIC_EVENT': 'timestamp timestampType source severity text globalPid',
    'ENUM_DIAGNOSTIC_SEVERITY_LEVEL': 'id name', 'ENUM_NSYS_EVENT_TYPE': 'id name',
    'META_DATA_EXPORT': 'name value', 'META_DATA_CAPTURE': 'name value',
    'ANALYSIS_DETAILS': 'duration startTime stopTime', 'TARGET_INFO_SESSION_START_TIME': 'utcEpochNs',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def process_key(global_id):
    require(type(global_id) is int and global_id > 0, 'missing/invalid serialized process/thread identity')
    return global_id & ~MASK


def pid(global_id):
    return (process_key(global_id) >> 24) & MASK


def epoch_ns(text):
    delta = datetime.fromisoformat(text.replace('Z', '+00:00')) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000


def union(intervals):
    result = []
    for start, end in sorted(intervals):
        require(type(start) is int and type(end) is int and start < end, 'empty/reversed interval')
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def intersection(left, right):
    left, right = union(left), union(right)
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if a < b:
            result.append((a, b))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return result


def length(intervals):
    return sum(b - a for a, b in union(intervals))


def label(text):
    parts = text.split()
    require(parts and parts[0] in LABELS, 'unknown Ferrule profile range')
    name, fields = parts[0], {}
    for item in parts[1:]:
        match = re.fullmatch(r'([a-z_]+)=([0-9]+)', item)
        require(match and match[1] not in fields, 'malformed/duplicate numeric NVTX field')
        fields[match[1]] = int(match[2])
    require(tuple(fields) == LABELS[name] and all(v < 1 << 64 for v in fields.values()),
            'missing/reordered/unknown NVTX numeric metadata')
    require(all(v > 0 for k, v in fields.items() if k not in ('layer', 'expert')), 'zero NVTX identity/size')
    return name, fields


def rows(db, table, budget=1_000_000):
    require(re.fullmatch(r'[A-Za-z0-9_]+', table), 'unsafe table identifier')
    count = db.execute('SELECT count(*) FROM ' + table).fetchone()[0]
    require(count <= budget, 'offline analysis row budget exceeded: ' + table)
    return [dict(r) for r in db.execute('SELECT * FROM ' + table)]


def inspect_schema(db):
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for name, required in REQUIRED.items():
        require(name in tables, 'unknown coverage: missing table ' + name)
        columns = {r[1] for r in db.execute('PRAGMA table_info(' + name + ')')}
        require(set(required.split()) <= columns, 'unknown schema columns: ' + name)
    meta = dict((r['name'], r['value']) for r in rows(db, 'META_DATA_EXPORT'))
    for key, value in dict(EXPORT_PRODUCT_VERSION='2025.3.1.90', EXPORT_SCHEMA_VERSION='3.20.2',
                           EXPORT_PARAM_LAZY='false', EXPORT_PARAM_TIME_NORMALIZE='false',
                           EXPORT_PARAM_TIME_SHIFT='0', EXPORT_PARAM_TABLES_TO_EXPORT='',
                           EXPORT_PARAM_TIMES_TO_EXPORT='').items():
        require(meta.get(key) == value, 'unknown/filtered export coverage: ' + key)
    return tables, meta


def osrt_evidence(db):
    """Count real API records, not the exporter start/finish rows in OSRT_API."""
    classes = {r['id']: r['name'] for r in rows(db, 'ENUM_NSYS_EVENT_CLASS')}
    records = rows(db, 'OSRT_API')
    apis, markers, issues = [], [], []
    by_process = defaultdict(Counter)
    for row in records:
        name = classes.get(row['eventClass'])
        key = process_key(row['globalTid'])
        by_process[key][name or 'unknown'] += 1
        if name == 'TRACE_PROCESS_EVENT_OS_RUNTIME':
            apis.append(row)
        elif name in ('TRACE_PROCESS_EVENT_OS_RUNTIME_START', 'TRACE_PROCESS_EVENT_OS_RUNTIME_FINISH'):
            markers.append(row)
            if row['start'] != row['end']:
                issues.append('OSRT lifecycle marker has an interval')
        else:
            issues.append('unknown OSRT event class')
    reported = defaultdict(list)
    for row in rows(db, 'DIAGNOSTIC_EVENT'):
        match = re.fullmatch(r'Number of OS runtime libraries events collected:\s*([0-9]+)\.', row['text'])
        if match:
            reported[row['globalPid']].append(int(match[1]))
    counters = []
    for key in sorted(set(by_process) | set(reported)):
        actual = by_process[key]['TRACE_PROCESS_EVENT_OS_RUNTIME']
        matches = reported[key] == [actual]
        if not matches:
            issues.append('official OSRT API counter missing/duplicate/mismatched')
        counters.append(dict(globalPid=key, pid=pid(key), api_rows=actual,
                             event_class_counts=dict(by_process[key]), official_counts=reported[key], matched=matches))
    return apis, dict(total_rows=len(records), api_rows=len(apis), lifecycle_marker_rows=len(markers),
        counters=counters, issues=sorted(set(issues)), dropped_event_count=None,
        count_scope='OSRT API class only; exporter START/FINISH markers retained separately, not dropped APIs')


def diagnostic_evidence(db, context, model):
    severities = {r['id']: r['name'] for r in rows(db, 'ENUM_DIAGNOSTIC_SEVERITY_LEVEL')}
    diagnostics, warnings, issues = rows(db, 'DIAGNOSTIC_EVENT'), [], []
    loss = scope_contract.LOSS
    dispositions = scope_contract.scoped_diagnostics(diagnostics, severities, context, model, rows(db, 'PROCESSES'))
    for row, disposition in zip(diagnostics, dispositions['classifications']):
        severity, message = severities.get(row['severity']), row['text']
        if severity not in ('Info', 'Verbose', 'Warning'):
            issues.append('collector error/unknown diagnostic severity')
        if loss.search(message):
            issues.append('collector reports incomplete events')
        if severity != 'Warning':
            continue
        classification = disposition['classification'] if not disposition['affects_scoped_overlap'] else None
        warnings.append(dict(row, classification=classification))
        issues.append('official collector warning; classification does not waive full-trace rejection')
    return dict(diagnostics=diagnostics, warnings=warnings, issues=sorted(set(issues)),
                loss_diagnostic_count=sum(bool(loss.search(r['text'])) for r in diagnostics),
                dropped_event_count=None, simultaneous_unreported_loss_excluded=False)


def raw_summary(db, context):
    """Positive observed facts survive missing QA/ranges; never full-trace acceptance."""
    db.row_factory = sqlite3.Row
    tables, metadata = inspect_schema(db)
    keys = [r['globalPid'] for r in rows(db, 'PROCESSES') if r['pid'] == context['model_pid']]
    require(len(keys) == 1 and keys[0] == process_key(keys[0]), 'ambiguous raw model process key')
    model = keys[0]
    diagnostics = diagnostic_evidence(db, context, model)
    _, osrt = osrt_evidence(db)
    strings = {r['id']: r['value'] for r in rows(db, 'StringIds')}
    by_api, by_kernel = defaultdict(list), defaultdict(list)
    counts, api_names = {}, Counter()
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            continue
        records = rows(db, table)
        counts[table] = len(records)
        for row in records:
            name = strings[row['nameId']]
            api_names[name] += 1
            if LAUNCH.fullmatch(name):
                by_api[(process_key(row['globalTid']), row['correlationId'])].append(row)
    kernels = rows(db, 'CUPTI_ACTIVITY_KIND_KERNEL')
    for row in kernels:
        by_kernel[(row['globalPid'], row['correlationId'])].append(row)
    correlation = dict(
        kernel_without_unique_math_api=sum(len(v) for k, v in by_kernel.items() if len(by_api.get(k, [])) != 1),
        math_api_without_unique_kernel=sum(len(v) for k, v in by_api.items() if len(by_kernel.get(k, [])) != 1),
        duplicate_math_api_keys=sum(len(v) != 1 for v in by_api.values()),
        duplicate_kernel_keys=sum(len(v) != 1 for v in by_kernel.values()),
        foreign_kernel_process=sum(r['globalPid'] != model for r in kernels),
        foreign_launch_process=sum(k[0] != model for k in by_api),
        failed_launches=sum(r['returnValue'] != 0 for items in by_api.values() for r in items),
        invalid_kernel_intervals=sum(r['start'] >= r['end'] for r in kernels),
        graph_node_kernels=sum(r['graphNodeId'] not in (None, 0) for r in kernels),
        kernel_precedes_api=sum(r['start'] < by_api[k][0]['start'] for k, items in by_kernel.items()
                                if len(by_api.get(k, [])) == 1 for r in items))
    counts.update(CUPTI_ACTIVITY_KIND_KERNEL=len(kernels), NVTX_EVENTS=len(rows(db, 'NVTX_EVENTS')),
                  OSRT_API=osrt['total_rows'])
    return dict(schema='ferrule.storage-demand.profile-raw-diagnostic.v2', status='diagnostic_partial',
        full_trace_accepted=False, scoped_overlap_accepted=False, scoped_sql_evidence_consistent=False,
                model_global_pid=model, model_pid=context['model_pid'],
        model_birth_ticks=context.get('model_birth_ticks'), counts=counts, api_names=dict(api_names),
        correlation_counts=correlation, observed_launch_kernel_bijection=bool(kernels) and not any(correlation.values()),
        osrt_counter_coverage=osrt, export_metadata=metadata, collector=diagnostics,
        warnings=diagnostics['warnings'], diagnostics=diagnostics['diagnostics'],
        issues=sorted(set(diagnostics['issues'] + osrt['issues'])),
        loss_diagnostic_count=diagnostics['loss_diagnostic_count'], dropped_event_count=None,
        attributed_kernel_count=None, read_envelope_kernel_intersection_ns=None, pread_kernel_intersection_ns=None,
        counter_coverage='OSRT reported API counts reconciled separately; CUPTI produced events are not an SQLite row total',
        request_coverage='unestablished: no QA clock, final read counters or complete range audit in this diagnostic',
        physical_uuid_verified=False, physical_nas_service_time=False, ordinary_performance=False,
        simultaneous_unreported_loss_excluded=False)


def interval_index(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[row['globalTid']].append(row)
    indexed = {}
    for tid, items in grouped.items():
        items.sort(key=lambda r: r['start'])
        require(all(a['end'] <= b['start'] for a, b in zip(items, items[1:])),
                'overlapping same-thread ranges/pread aliases; attribution or bytes ambiguous')
        indexed[tid] = ([r['start'] for r in items], items)
    return indexed


def containing(index, row):
    starts, items = index.get(row['globalTid'], ([], []))
    i = bisect_right(starts, row['start']) - 1
    if i >= 0 and row['start'] < items[i]['end']:
        require(row['end'] <= items[i]['end'], 'API crosses attribution range boundary')
        return items[i]
    require(i + 1 == len(items) or row['end'] <= items[i + 1]['start'], 'API crosses attribution range boundary')
    return None


def audit_sqlite(db, context):
    """context comes from owned raw lifecycle/QA/map evidence, not a PASS toggle."""
    db.row_factory = sqlite3.Row
    tables, export = inspect_schema(db)
    issues, warnings = [], []
    counts = {}

    def check(value, reason):
        if not value:
            issues.append(reason)

    strings = {r['id']: r['value'] for r in rows(db, 'StringIds')}

    def string(identity):
        require(identity in strings, 'unresolved StringIds entry')
        return strings[identity]

    def text(row):
        direct = row.get('text')
        indirect = string(row['textId']) if row.get('textId') is not None else None
        require(direct is not None or indirect is not None, 'missing NVTX text/StringIds')
        require(direct is None or indirect is None or direct == indirect, 'ambiguous NVTX string')
        return direct if direct is not None else indirect

    processes = rows(db, 'PROCESSES')
    keys = [r['globalPid'] for r in processes if r['pid'] == context['model_pid']]
    require(len(keys) == 1 and keys[0] == process_key(keys[0]) and pid(keys[0]) == context['model_pid'],
            'model PID/globalPid absent, reused, or ambiguous')
    model = keys[0]
    for row in processes:
        require(pid(row['globalPid']) == row['pid'], 'PROCESSES serialized PID disagreement')
    require(len({r['globalPid'] for r in processes}) == len(processes), 'duplicate process identity')
    gpus = rows(db, 'TARGET_INFO_GPU')
    require(len(gpus) == 8 and len({r['id'] for r in gpus}) == 8, 'incomplete physical GPU inventory')
    devices = {r['id']: r for r in gpus}
    mappings = {}
    for row in rows(db, 'TARGET_INFO_CUDA_DEVICE'):
        key = row['pid'], row['cudaId']
        require(key not in mappings, 'duplicate per-process CUDA mapping')
        mappings[key] = row['gpuId']

    def physical(row):
        require(row['globalPid'] == model, 'GPU activity from unbound process/VM')
        require(row['deviceId'] == 0, 'TP1 logical device must be zero')
        gpu = devices.get(mappings.get((context['model_pid'], row['deviceId'])))
        require(gpu and 'GPU-' + gpu['uuid'].removeprefix('GPU-') == context['uuid']
                and gpu['vmId'] == model & ~((1 << 48) - 1), 'actual CUDA device/VM/physical UUID mismatch')

    capture = rows(db, 'META_DATA_CAPTURE')
    values = defaultdict(list)
    for r in capture:
        values[r['name']].append(r['value'])
    check({'Cuda', 'NvtxEvents', 'OSRuntime'} <= set(values['CAPTURE_EVENT_TYPE']), 'missing capture event types')
    check(values['OS_RUNTIME_OPTIONS:DURATION_THRESHOLD_NS'] == ['1000'], 'unknown OSRT threshold')
    # The CLI receipt is also checked; API bytes/counts below must reconcile. Never
    # infer file-API completeness merely because the OSRT table is nonempty.
    details = rows(db, 'ANALYSIS_DETAILS')
    session = rows(db, 'TARGET_INFO_SESSION_START_TIME')
    require(len(details) == len(session) == 1, 'ambiguous capture clock/session')
    duration = details[0]['duration']
    begin, end = context['request_begin_unix_ns'] - session[0]['utcEpochNs'], context['request_end_unix_ns'] - session[0]['utcEpochNs']
    require(0 <= begin < end <= duration, 'request outside full capture clock/duration')

    event_types = {r['id']: r['name'] for r in rows(db, 'ENUM_NSYS_EVENT_TYPE')}
    nvtx = rows(db, 'NVTX_EVENTS')
    wrapper_keys = {r.get('globalPid') for r in context.get('process_roles', [])
                    if r.get('role') == 'non_cuda_wrapper'}
    check(all(process_key(r['globalTid']) not in wrapper_keys for r in nvtx),
          'declared non-CUDA wrapper has NVTX payload; role warning exclusion invalid')
    domains = {}
    for row in nvtx:
        if event_types.get(row['eventType']) == 'NvtxDomainCreate':
            key = process_key(row['globalTid']), row['domainId']
            require(key not in domains, 'duplicate NVTX domain/process identity')
            domains[key] = text(row)
    ranges = defaultdict(list)
    seen_ranges, seen_reads = set(), set()
    for row in nvtx:
        domain = domains.get((process_key(row['globalTid']), row['domainId']))
        if domain != 'ferrule_profile' or event_types.get(row['eventType']) == 'NvtxDomainCreate':
            continue
        require(process_key(row['globalTid']) == model, 'Ferrule range belongs to foreign process')
        require(event_types.get(row['eventType']) == 'NvtxStartEndRange', 'unclosed/unsupported Ferrule NVTX range kind')
        require(row['endGlobalTid'] in (None, row['globalTid']), 'range changed threads; attribution unknown')
        require(type(row['end']) is int and begin <= row['start'] < row['end'] <= end,
                'Ferrule range incomplete/outside the only QA request')
        key = model, row['domainId'], row['rangeId']
        require(row['rangeId'] is not None and key not in seen_ranges, 'duplicate NVTX range identity')
        seen_ranges.add(key)
        name, fields = label(text(row))
        row = dict(row, fields=fields)
        if name == 'read_subset':
            key = model, fields['source_generation'], fields['operation']
            require(key not in seen_reads, 'duplicate worker operation/source/PID identity')
            seen_reads.add(key)
        ranges[name].append(row)
    for name in LABELS:
        check(bool(ranges[name]), 'expected positive range absent: ' + name)
    images = {r['fields']['image_generation'] for r in ranges['layer.expert'] + ranges['layer.expert.upload']}
    sources = {r['fields']['source_generation'] for r in ranges['read_subset']}
    check(len(images) == 1 and sources == images, 'source generation/image binding mismatch')
    if context.get('scope_contract'):
        check(type(context.get('model_source_generation')) is int
              and sources == {context['model_source_generation']}, 'ranges differ from actual owner image generation')

    indexes = {name: interval_index(records) for name, records in ranges.items()}
    apis, by_api = [], defaultdict(list)
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            continue
        table_rows = rows(db, table)
        counts[table] = len(table_rows)
        for row in table_rows:
            row = dict(row, name=string(row['nameId']), table=table)
            if process_key(row['globalTid']) != model:
                check(False, 'CUDA API from unbound process')
                continue
            require(row['start'] <= row['end'], 'reversed CUDA API interval')
            check(not re.search(r'Graph|(?:Begin|End)Capture', row['name']), 'graph/capture API unsupported: ' + row['name'])
            if 'Launch' in row['name'] and 'HostFunc' not in row['name']:
                check(bool(LAUNCH.fullmatch(row['name'])), 'unknown launch API: ' + row['name'])
            if LAUNCH.fullmatch(row['name']):
                check(row['returnValue'] == 0, 'failed math launch')
                by_api[(process_key(row['globalTid']), row['correlationId'])].append(row)
            apis.append(row)
    kernels = rows(db, 'CUPTI_ACTIVITY_KIND_KERNEL')
    counts['CUPTI_ACTIVITY_KIND_KERNEL'] = len(kernels)
    by_kernel = defaultdict(list)
    attributed = []
    kernel_shapes = Counter()
    range_hits = Counter()
    for row in kernels:
        physical(row)
        require(0 <= row['start'] < row['end'] <= duration, 'kernel outside trace')
        string(row['demangledName'])
        check(row['graphNodeId'] in (None, 0), 'graph node kernel unsupported')
        key = row['globalPid'], row['correlationId']
        by_kernel[key].append(row)
        matches = by_api.get(key, [])
        check(len(matches) == 1, 'kernel has missing/ambiguous math API correlation')
        if len(matches) != 1:
            continue
        api = matches[0]
        check(api['start'] <= row['start'], 'kernel precedes correlated API')
        expert = containing(indexes.get('layer.expert', {}), api)
        if expert:
            check(begin <= row['start'] < row['end'] <= end, 'expert kernel crosses request boundary')
            check(containing(indexes.get('layer.expert.upload', {}), api) is None, 'compute attributed to upload')
            attributed.append((row['start'], row['end']))
            shape = (string(row['demangledName']), *(row.get(k) for k in
                     ('gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ')))
            kernel_shapes[shape] += 1
            range_hits[expert['rangeId']] += 1
    for key, api in by_api.items():
        check(len(api) == 1 and len(by_kernel.get(key, [])) == 1,
              'math API orphan/duplicate/unknown one-to-many kernel correlation')
    check(bool(kernels) and bool(attributed), 'no actual attributed expert math kernels')
    for r in ranges['layer.expert']:
        check(range_hits[r['rangeId']] > 0, 'expert range without actual correlated math kernel')
    for table in tables:
        if table.startswith('CUDA_GRAPH') or table == 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE':
            check(db.execute('SELECT count(*) FROM ' + table).fetchone()[0] == 0, 'captured graph table unsupported')
        if table.startswith('CUPTI_ACTIVITY_KIND_') and table not in counts:
            records = rows(db, table)
            counts[table] = len(records)
            if table in ('CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
                for row in records:
                    physical(row)

    osrt, osrt_coverage = osrt_evidence(db)
    issues.extend(osrt_coverage['issues'])
    counts['OSRT_API'] = osrt_coverage['total_rows']
    osrt_names = Counter(string(r['nameId']) for r in osrt)
    preads, pread_records = [], []
    read_bytes, read_calls = Counter(), Counter()
    read_threads = {r['globalTid'] for r in ranges['read_subset']}
    compute_threads = {r['globalTid'] for r in ranges['layer.expert']}
    check(not read_threads & compute_threads, 'read worker and compute submission thread not separated')
    seen_preads = set()
    for row in osrt:
        name = string(row['nameId'])
        if name not in ('pread', 'pread64') or row['globalTid'] not in read_threads:
            continue
        require(row['start'] < row['end'], 'invalid pread interval')
        read = containing(indexes.get('read_subset', {}), row)
        check(read is not None, 'orphan/ambiguous worker pread')
        key = row['globalTid'], row['start'], row['end'], row['returnValue']
        require(key not in seen_preads, 'duplicate pread interval/alias; bytes cannot be counted twice')
        seen_preads.add(key)
        if read is None:
            continue
        check(row['returnValue'] > 0, 'failed/empty worker pread')
        rid = read['rangeId']
        read_bytes[rid] += row['returnValue']
        read_calls[rid] += 1
        preads.append((row['start'], row['end']))
        pread_records.append(row)
    interval_index(pread_records)  # Reject nested libc aliases before summing bytes.
    for row in ranges['read_subset']:
        check(read_calls[row['rangeId']] > 0 and read_bytes[row['rangeId']] == row['fields']['requested_bytes'],
              'worker pread bytes/counts missing or differ from NVTX requested bytes')
    stats = context['read_stats']
    check(len(ranges['read_subset']) == stats['io_started_reads'] > 0, 'NVTX reads differ from current final read ledger')
    check(sum(r['fields']['requested_bytes'] for r in ranges['read_subset']) == stats['proved_read_bytes'] > 0,
          'NVTX bytes differ from current final read ledger')
    last = context.get('last_receipt')
    check(isinstance(last, dict), 'last actual worker receipt missing')
    if last:
        identity = last['identity']
        matching = [r for r in ranges['read_subset'] if r['fields']['operation'] == identity['operation']
                    and r['fields']['source_generation'] == identity['image_generation']]
        check(len(matching) == 1 and matching[0]['fields']['requested_bytes'] == last['requested_bytes'] == last['io_bytes'],
              'last actual worker receipt/source/operation differs')

    counts['NVTX_EVENTS'] = len(nvtx)
    collector = diagnostic_evidence(db, context, model)
    diagnostics, warnings = collector['diagnostics'], collector['warnings']
    scope = scope_contract.scoped_diagnostics(diagnostics,
        {r['id']: r['name'] for r in rows(db, 'ENUM_DIAGNOSTIC_SEVERITY_LEVEL')}, context, model, processes)
    cupti = [r['text'].removeprefix('Loaded CUPTI library: ') for r in diagnostics
             if r['globalPid'] == model and r['text'].startswith('Loaded CUPTI library: ')]
    check(cupti == [context['cupti_path']], 'actual model CUPTI library diagnostic missing/ambiguous/mismatched')
    read_envelopes = [(r['start'], r['end']) for r in ranges['read_subset']]
    kernel_union = union(attributed)
    scoped_issues = sorted(set(issues + scope['issues']))
    issues.extend(collector['issues'])
    summary = dict(schema='ferrule.storage-demand.profile-overlap-sql.v1',
        status='sql_evidence_consistent' if not issues else 'rejected', full_trace_accepted=False,
        scoped_sql_evidence_consistent=not scoped_issues, scoped_overlap_accepted=False,
        scoped_issues=scoped_issues, diagnostic_scope=scope,
        thread_activity='not established by this overlap analysis', thread_state_analysis_required=False,
        math_scope_criterion=not scoped_issues, cpu_scheduling_claim=False, active_io_claim=False,
        issues=sorted(set(issues)), warnings=warnings, diagnostics=diagnostics,
        model_global_pid=model, model_pid=context['model_pid'], model_birth_ticks=context['model_birth_ticks'],
        physical_uuid=context['uuid'], request_window_ns=[begin, end],
        counts=counts, api_names=dict(Counter(r['name'] for r in apis)), osrt_api_names=dict(osrt_names),
        correlation_counts=dict(
            kernel_without_unique_math_api=sum(len(v) for k, v in by_kernel.items() if len(by_api.get(k, [])) != 1),
            math_api_without_unique_kernel=sum(len(v) for k, v in by_api.items() if len(by_kernel.get(k, [])) != 1),
            duplicate_math_api_keys=sum(len(v) != 1 for v in by_api.values()),
            duplicate_kernel_keys=sum(len(v) != 1 for v in by_kernel.values())),
        osrt_counter_coverage=osrt_coverage,
        loss_diagnostic_count=collector['loss_diagnostic_count'],
        dropped_event_count=None, simultaneous_unreported_loss_excluded=False,
        nvtx_range_counts={k: len(ranges[k]) for k in LABELS}, worker_pread_calls=len(preads),
                source_generations=sorted(sources), image_generations=sorted(images),
                expert_range_counts_by_layer=dict(Counter(r['fields']['layer'] for r in ranges['layer.expert'])),
                attributed_kernel_shapes=[dict(name=shape[0], grid=list(shape[1:4]), block=list(shape[4:7]), count=count)
                                          for shape, count in sorted(kernel_shapes.items(), key=lambda item: repr(item[0]))],
        requested_read_bytes=sum(r['fields']['requested_bytes'] for r in ranges['read_subset']),
        attributed_kernel_count=len(attributed), expert_kernel_union_ns=length(kernel_union),
        worker_read_envelope_union_ns=length(read_envelopes), worker_pread_union_ns=length(preads),
        read_envelope_kernel_intersection_ns=length(intersection(read_envelopes, kernel_union)),
        pread_kernel_intersection_ns=length(intersection(preads, kernel_union)),
        read_envelope_overlap_percent=(100 * length(intersection(read_envelopes, kernel_union)) / length(read_envelopes)
                                       if read_envelopes else None),
        pread_overlap_percent=(100 * length(intersection(preads, kernel_union)) / length(preads) if preads else None),
        overlap_fraction_denominator='union of respective worker intervals, not sum of duplicate intervals',
        operation_identity_coverage='numeric source/operation for every observed read; final last receipt is one positive check, '
                                    'not a full per-operation layer/expert journal',
        osrt_threshold_ns=1000, file_access_threshold_bypass=True, export_metadata=export,
        full_memory_bound=False, physical_nas_service_time=False, ordinary_performance=False,
        limitations='Observed userspace pread/kernel concurrency only. Read envelope includes allocation; '
                     'pread wall time includes CPU preemption and can be page-cache service, not active IO or physical NAS service. '
                                          'No copy enqueue, wait, upload, or host submission duration is GPU time. '
                     'Bidirectional closure does not exclude simultaneous API/activity loss; lifecycle, diagnostics, '
                     'full export, expected ranges and current read counters are independently required.')
    return summary


def analyze(path, context):
    path = Path(path).resolve()
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        return dict(audit_sqlite(db, context), private_sqlite_path=str(path))
