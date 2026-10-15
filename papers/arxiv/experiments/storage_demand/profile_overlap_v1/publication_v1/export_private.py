#!/usr/bin/env python3
"""Curate existing actual-05 files, read-only and CPU-only; never collect a trace.

Public reproduction uses check.py, not this private-input exporter. Queries below
are the exact executed SQL projections. No SELECT * or system-environment query.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
import re
import sqlite3

import check as c

NVTX_SQL = '''SELECT n.rowid AS row_id, n.start, n.end, n.eventType,
 n.rangeId, n.globalTid, n.endGlobalTid, n.domainId, n.text, s.value AS indirect_text
 FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId = s.id
 ORDER BY n.start, n.end, n.rowid'''
PREAD_SQL = '''SELECT o.rowid AS row_id, o.start, o.end, o.globalTid,
 o.eventClass, o.nameId, s.value AS name, o.returnValue
 FROM OSRT_API o JOIN StringIds s ON o.nameId = s.id
 JOIN ENUM_NSYS_EVENT_CLASS e ON o.eventClass = e.id
 WHERE e.name = 'TRACE_PROCESS_EVENT_OS_RUNTIME' AND s.value IN ('pread', 'pread64')
 ORDER BY o.start, o.end, o.rowid'''
KERNEL_SQL = '''SELECT rowid AS row_id, start, end, globalPid, deviceId,
 contextId, streamId, correlationId, demangledName, graphNodeId
 FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start, end, rowid'''
API_SQL = '''SELECT a.rowid AS row_id, a.start, a.end, a.globalTid,
 a.correlationId, a.nameId, s.value AS name, a.returnValue
 FROM {table} a JOIN StringIds s ON a.nameId = s.id
 ORDER BY a.start, a.end, a.rowid'''
DIAGNOSTIC_SQL = '''SELECT rowid AS row_id, timestamp, timestampType, source,
 severity, text, globalPid FROM DIAGNOSTIC_EVENT ORDER BY rowid'''
LABEL_FIELDS = {'read_subset': ('operation', 'source_generation', 'requested_bytes'),
                'layer.expert': ('image_generation', 'layer', 'expert', 'rows'),
                'layer.expert.upload': ('image_generation', 'layer', 'expert', 'rows')}


def load(path):
    return json.loads(path.read_text())


def pick(obj, names):
    return {k: obj[k] for k in names.split()}


def csv_bytes(kind, rows):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, c.FIELDS[kind], lineterminator='\n')
    writer.writeheader()
    for r in sorted(rows, key=lambda r: (r['start_ns'], r['end_ns'], r['row_id'])):
        writer.writerow({k: 'null' if v is None else v for k, v in r.items()})
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode='wb', filename='', mtime=0, compresslevel=9) as f:
        f.write(text.getvalue().encode('ascii'))
    return out.getvalue()


def extract_arm(root, arm, handoff, source):
    session = root / 'runs' / arm / 'session'
    audit = load(session / 'worker-profile-audit.json')
    analysis = load(session / 'overlap-analysis.json')
    ledger = load(session / 'read-ledger.json')
    h = next(a for a in handoff['arms'] if a['mode'] == arm)
    c.require(c.file_record(session / 'overlap-analysis.json')['sha256'] == h['analysis_sha256'], 'final analysis seal changed')
    for n in ('sqlite', 'report'):
        path = session / ('nsys.sqlite' if n == 'sqlite' else 'nsys.nsys-rep')
        c.require(c.file_record(path)['sha256'] == h[n + '_sha256'] == analysis[n + '_sha256'], 'raw capture seal changed')
    begin, end = analysis['request_window_ns']
    model = analysis['model_global_pid']
    wrapper = next(r['globalPid'] for r in analysis['diagnostic_scope']['process_roles'] if r['role'] == 'non_cuda_wrapper')
    command = load(session / 'collector-command.json')['argv']
    expected_flags = {'--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=process-tree',
                      '--gpu-metrics-devices=none', '--gpuctxsw=false',
                      '--osrt-file-access=true', '--osrt-threshold=1000'}
    c.require(expected_flags <= set(command), 'collector receipt options changed')
    quality = load(session / 'canary-qa/quality.json')
    data = {k: [] for k in c.FIELDS}
    with sqlite3.connect((session / 'nsys.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        clock = db.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchall()
        c.require(len(clock) == 1 and [quality['request_begin_unix_ns'] - clock[0][0],
                  quality['request_end_unix_ns'] - clock[0][0]] == [begin, end], 'request clock binding changed')
        gpus = [dict(r) for r in db.execute('SELECT id, uuid, vmId FROM TARGET_INFO_GPU')]
        mapping = db.execute('SELECT gpuId FROM TARGET_INFO_CUDA_DEVICE WHERE pid=? AND cudaId=0',
                             (analysis['model_pid'],)).fetchall()
        c.require(len(gpus) == 8 and len(mapping) == 1, 'physical mapping inventory')
        physical = next(r for r in gpus if r['id'] == mapping[0][0])
        c.require('GPU-' + physical['uuid'].removeprefix('GPU-') == analysis['physical_uuid']
                  and physical['vmId'] == model & ~((1 << 48) - 1), 'physical UUID/VM binding changed')
        full_nvtx = [dict(r) for r in db.execute(NVTX_SQL)]
        for r in full_nvtx:
            text = r['text'] if r['text'] is not None else r['indirect_text']
            c.require(r['text'] is None or r['indirect_text'] is None or r['text'] == r['indirect_text'], 'ambiguous NVTX text')
            if r['eventType'] == 75:
                c.require(text == 'ferrule_profile', 'unexpected NVTX domain')
                continue
            parts = text.split()
            c.require(parts[0] in LABEL_FIELDS and len(parts) == len(LABEL_FIELDS[parts[0]]) + 1, 'unknown NVTX label')
            fields = {}
            for token, expected in zip(parts[1:], LABEL_FIELDS[parts[0]]):
                m = re.fullmatch(r'([a-z_]+)=([0-9]+)', token)
                c.require(m and m[1] == expected, 'unknown/reordered source label')
                fields[m[1]] = int(m[2])
            row = dict.fromkeys(c.FIELDS['nvtx'])
            row.update(row_id=r['row_id'], kind=parts[0], start_ns=r['start'] - begin, end_ns=r['end'] - begin,
                       global_pid=r['globalTid'] & ~c.MASK, global_tid=r['globalTid'], end_global_tid=r['endGlobalTid'],
                       event_type=r['eventType'], domain_id=r['domainId'], range_id=r['rangeId'], **fields)
            data['nvtx'].append(row)
        reads = [r for r in data['nvtx'] if r['kind'] == 'read_subset']
        experts = [r for r in data['nvtx'] if r['kind'] == 'layer.expert']
        read_idx, expert_idx = c.index(reads), c.index(experts)
        excluded_preads = Counter()
        for r in db.execute(PREAD_SQL):
            pid = r['globalTid'] & ~c.MASK
            if pid != model:
                excluded_preads['non_model'] += 1
                continue
            if r['end'] <= begin or r['start'] >= end:
                excluded_preads['model_outside_request'] += 1
                continue
            read = c.containing(read_idx, r['globalTid'], r['start'] - begin, r['end'] - begin)
            c.require(read, 'orphan request pread')
            data['pread'].append(dict(row_id=r['row_id'], start_ns=r['start'] - begin, end_ns=r['end'] - begin,
                global_pid=pid, global_tid=r['globalTid'], event_class=r['eventClass'], name_id=r['nameId'],
                name=r['name'], return_value=r['returnValue'], domain_id=read['domain_id'],
                read_range_id=read['range_id'], osrt_bytes=None))
        by_api, all_api_count, wrapper_apis = {}, 0, 0
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
            if table not in tables:
                continue
            for item in db.execute(API_SQL.format(table=table)):
                r = dict(item)
                pid = r['globalTid'] & ~c.MASK
                wrapper_apis += int(pid == wrapper)
                c.require(pid == model, 'foreign CUDA API')
                c.require(not re.search(r'Graph|(?:Begin|End)Capture', r['name']), 'unsupported graph API')
                if 'Launch' in r['name'] and 'HostFunc' not in r['name']:
                    c.require(c.LAUNCH.fullmatch(r['name']), 'unsupported launch API')
                if not c.LAUNCH.fullmatch(r['name']):
                    continue
                key = pid, r['correlationId']
                c.require(key not in by_api and r['returnValue'] == 0, 'duplicate/failed math launch')
                by_api[key] = dict(r, table=table)
                all_api_count += 1
        seen_kernel, excluded_kernels = set(), Counter()
        all_kernel_count = wrapper_kernels = 0
        for r in db.execute(KERNEL_SQL):
            all_kernel_count += 1
            wrapper_kernels += int(r['globalPid'] == wrapper)
            key = r['globalPid'], r['correlationId']
            c.require(key not in seen_kernel and key in by_api, 'kernel duplicate/orphan')
            seen_kernel.add(key)
            api = by_api[key]
            c.require(r['globalPid'] == model and r['deviceId'] == 0 and r['graphNodeId'] in (None, 0)
                      and api['start'] <= r['start'] < r['end'], 'unsupported kernel')
            expert = c.containing(expert_idx, api['globalTid'], api['start'] - begin, api['end'] - begin)
            if expert is None:
                excluded_kernels['outside_request' if r['end'] <= begin or r['start'] >= end else 'request_nonexpert'] += 1
                continue
            data['kernel'].append(dict(row_id=r['row_id'], start_ns=r['start'] - begin, end_ns=r['end'] - begin,
                global_pid=r['globalPid'], device_id=r['deviceId'], context_id=r['contextId'], stream_id=r['streamId'],
                correlation_id=r['correlationId'], name_id=r['demangledName'], graph_node_id=r['graphNodeId'],
                api_row_id=api['row_id'], api_table=api['table'], api_start_ns=api['start'] - begin,
                api_end_ns=api['end'] - begin, api_global_tid=api['globalTid'], api_name_id=api['nameId'],
                api_name=api['name'], api_return_value=api['returnValue'], domain_id=expert['domain_id'],
                expert_range_id=expert['range_id']))
        c.require(seen_kernel == set(by_api), 'whole-export API orphan')
        diagnostics = []
        classes = {c.digest(c.encode(d['raw'])): d for d in analysis['diagnostic_scope']['classifications']}
        for item in db.execute(DIAGNOSTIC_SQL):
            raw = dict(item); row_id = raw.pop('row_id')
            raw_hash = c.digest(c.encode(raw))
            cl = classes[raw_hash]
            text = raw['text']
            redacted = '/' in text
            if redacted:
                if text.startswith('Process was launched by the profiler, see '):
                    text = 'Process was launched by the profiler; output locations [REDACTED]'
                elif text.startswith('Loaded CUPTI library: '):
                    text = 'Loaded CUPTI library: [REDACTED]'
                else:
                    raise ValueError('unreviewed diagnostic redaction')
            diagnostics.append(dict(row_id=row_id, **pick(raw, 'timestampType source severity globalPid'), text=text,
                **pick(cl, 'classification affects_scoped_overlap affects_full_trace'), text_redacted=redacted,
                original_text_sha256=c.digest(raw['text'].encode()), original_row_sha256=raw_hash))
        c.require(len(diagnostics) == len(classes), 'diagnostic coverage mismatch')
        export = dict(db.execute('SELECT name, value FROM META_DATA_EXPORT'))
        capture = defaultdict(list)
        for name, value in db.execute('SELECT name, value FROM META_DATA_CAPTURE'):
            if name in c.CAPTURE:
                capture[name].append(value)
        osrt_columns = [r[1] for r in db.execute('PRAGMA table_info(OSRT_API)')]
        table_counts = {name: db.execute('SELECT count(*) FROM ' + name).fetchone()[0] for name in analysis['counts']}
        c.require(table_counts == analysis['counts'], 'whole SQL table counts changed')
        osrt_counts = defaultdict(dict)
        for tid, event, count in db.execute('''SELECT o.globalTid, e.name, count(*)
                FROM OSRT_API o JOIN ENUM_NSYS_EVENT_CLASS e ON o.eventClass=e.id
                GROUP BY o.globalTid, e.name'''):
            pid = tid & ~c.MASK
            osrt_counts[pid][event] = osrt_counts[pid].get(event, 0) + count
        c.require(dict(osrt_counts) == {v['globalPid']: v['event_class_counts']
                  for v in analysis['osrt_counter_coverage']['counters']}, 'raw OSRT process/class coverage changed')
    original = {
        'sqlite': 'nsys.sqlite', 'report': 'nsys.nsys-rep', 'worker_audit': 'worker-profile-audit.json',
        'final_analysis': 'overlap-analysis.json', 'read_ledger': 'read-ledger.json', 'quality': 'canary-qa/quality.json',
        'collector_manifest': 'collector-manifest.json', 'collector_command': 'collector-command.json',
        'ready_roles': 'ready-process-roles.json', 'after_roles': 'after-requests-process-roles.json',
        'ready_maps': 'ready-library-evidence/maps.raw', 'after_maps': 'after-requests-library-evidence/maps.raw',
        'worker_capture': 'worker-capture.json', 'custody': 'custody.json', 'supervisor': 'supervisor.json',
        'export_result': 'export-result.json', 'server_log': 'server.log',
    }
    for relative in original.values():
        path = session / relative
        c.require(c.file_record(path)['sha256'] == handoff['evidence_pins'][str(path.resolve())],
                  'original handoff evidence pin changed: ' + path.name)
    receipt = ledger['read']['last_receipt']
    proof = dict(schema='ferrule.overlap.public-projection.v1', arm=arm,
        preliminary_worker=pick(audit, 'status full_trace_accepted scoped_sql_evidence_consistent scoped_overlap_accepted'),
        final_scope=pick(analysis, 'scoped_overlap_accepted full_trace_accepted dropped_event_count simultaneous_unreported_loss_excluded universal_trace_completeness_established lifecycle_gates missing_gates scoped_issues log_issues loss_diagnostic_count'),
        request_duration_ns=end - begin,
        process_roles=[dict(**pick(r, 'role globalPid executable_sha256 source_binding_sha256'),
                            **pick(r['identity'], 'pid ppid birth_ticks')) for r in analysis['diagnostic_scope']['process_roles']],
        diagnostics=diagnostics, export_header={k: export[k] for k in c.EXPORT}, capture_header=dict(capture),
        osrt_schema_columns=osrt_columns, osrt_counter_coverage=analysis['osrt_counter_coverage'],
        full_export_projection=dict(table_counts=table_counts, all_kernel_rows=all_kernel_count,
            all_math_launch_rows=all_api_count, correlation_counts=analysis['correlation_counts'],
            retained_expert_kernels=len(data['kernel']), excluded_kernels=dict(excluded_kernels),
            retained_preads=len(data['pread']), excluded_preads=dict(excluded_preads),
            wrapper_nvtx_rows=sum(r['globalTid'] & ~c.MASK == wrapper for r in full_nvtx),
            wrapper_cuda_api_rows=wrapper_apis, wrapper_kernel_rows=wrapper_kernels, all_diagnostic_rows=len(diagnostics)),
        ledger=dict(**pick(ledger, 'model_source_generation invocation_count'),
            **pick(ledger['read'], 'status effective_mode stats'),
            last_receipt=pick(receipt, 'identity outcome requested_bytes io_bytes cancellation_requested')),
        maps=h['maps'], qa=h['qa'], metrics=h['metrics'],
        original_files={name: c.file_record(session / relative) for name, relative in original.items()},
        source_binding=dict(cli_sha256=source['cli_sha256'], source_commit=c.SOURCE, measurement_fix_commit=handoff['source_commit'],
                            source_generation=ledger['model_source_generation'], image_generation=1),
        physical_mapping=dict(model_pid=analysis['model_pid'], logical_device_id=0, physical_uuid_bound=True,
                              uuid_redacted=True, gpu_inventory_count=8),
        collector_options=dict(trace=['cuda', 'nvtx', 'osrt'], sample='none', cpuctxsw='process-tree',
                               osrt_file_access=True, osrt_threshold_ns=1000))
    c.require(excluded_preads == analysis['excluded_pread_calls'], 'excluded pread counts changed')
    c.require(ledger['read']['stats'] == analysis['read_ledger_stats'] == h['read_stats'], 'typed ledger mismatch')
    # Check every retained range against the independently stored SQL auditor,
    # not only the five aggregate measurements.
    per_read = defaultdict(list)
    for r in data['pread']:
        per_read[r['read_range_id']].append((r['start_ns'] + begin, r['end_ns'] + begin))
    for r, stored in zip(reads, analysis['read_range_metrics']):
        c.require((r['range_id'], r['operation'], r['start_ns'] + begin, r['end_ns'] + begin) ==
                  (stored['range_id'], stored['operation'], stored['start_ns'], stored['end_ns'])
                  and per_read[r['range_id']] == [tuple(v) for v in stored['pread_intervals_ns']], 'per-read auditor mismatch')
    per_expert = defaultdict(list)
    for r in data['kernel']:
        per_expert[r['expert_range_id']].append((r['start_ns'] + begin, r['end_ns'] + begin))
    for stored in analysis['expert_range_metrics']:
        rows = per_expert[stored['range_id']]
        c.require(len(rows) == stored['kernel_count'] and c.union(rows) == [tuple(v) for v in stored['kernel_union_intervals_ns']], 'per-expert auditor mismatch')
    c.validate_arm(proof, data)
    return {arm + '-proof.json': c.encode(proof), **{arm + '-' + kind + '.csv.gz': csv_bytes(kind, rows) for kind, rows in data.items()}}


def build(private_root):
    root = private_root / 'artifacts-go-20261015-05'
    handoff = load(root / 'final-handoff.json')
    source = load(root / 'nvtx-source-qualification.json')
    candidate = load(root / 'candidate-binding.json')
    c.require(handoff['status'] == 'two_fresh_scoped_captures_accepted' and handoff['source_commit'] == c.FIX
              and handoff['accepted_arms'] == handoff['captures_launched'] == 2
              and handoff['full_trace_accepted'] is False and handoff['dropped_event_count'] is None, 'not actual-05 scoped pair')
    c.require(c.file_record(root / 'candidate-binding.json')['sha256'] == handoff['binding_sha256'], 'candidate seal changed')
    identity = candidate['candidate_identity']
    c.require(identity['commit'] == c.SOURCE and identity['cli_binary_sha256'] == source['cli_sha256'] == c.CLI
              and identity['source_archive_sha256'] == source['source_archive_sha256'], 'frozen CLI/source identity mismatch')
    for name in ('nvtx-source-qualification.json', 'prepared.json', 'runtime-provenance.json', 'collector-contract.json'):
        path = root / name
        c.require(c.file_record(path)['sha256'] == handoff['evidence_pins'][str(path.resolve())], 'source metadata seal changed')
    pins = {}
    excerpts = []
    slices = {'crates/ferrule-model/src/checkpoint/read_pipeline.rs': [(1184, 1219)],
              'crates/ferrule-model/src/checkpoint/read_session.rs': [(98, 135), (316, 355)]}
    for raw_path, sha in source['source_pins'].items():
        path = Path(raw_path)
        c.require(c.file_record(path)['sha256'] == sha, 'frozen source pin changed')
        name = raw_path.split('/frozen-source/', 1)[1]
        pins[name] = sha
        lines = path.read_text().splitlines()
        for first, last in slices.get(name, []):
            excerpts.append(dict(source_member=name, first_line=first, last_line=last,
                                 lines=lines[first - 1:last], whole_file_sha256=sha))
    source_public = dict(schema='ferrule.overlap.public-source.v1', generation='actual-05',
        candidate_identity=pick(candidate['candidate_identity'], 'commit source_digest source_archive_sha256 cli_binary_sha256 source_members'),
        measurement_fix_commit=handoff['source_commit'], source_pins=pins, source_boundary_excerpts=excerpts,
        read_source_boundary=source['read_source_boundary'],
        read_measurement=load(root / 'runs/blocking/session/overlap-analysis.json')['read_measurement'],
        final_handoff=pick(handoff, 'status captures_launched captures_completed_and_exported qa_requests_issued qa_requests_completed accepted_arms scoped_overlap_accepted full_trace_accepted dropped_event_count simultaneous_unreported_loss_excluded validation'),
        original_files={name: c.file_record(root / name) for name in
            ('final-handoff.json', 'candidate-binding.json', 'nvtx-source-qualification.json', 'prepared.json', 'runtime-provenance.json', 'collector-contract.json')},
        redaction=dict(method='explicit positive field projection, not recursive scrubbing',
            omitted=['worker authorization credentials', 'environment and command arguments', 'host and filesystem paths',
                     'physical GPU UUIDs', 'application maps and private logs', 'absolute UTC and capture origins',
                     'diagnostic timestamps (mixed clocks; not relabeled as request-relative)',
                     'kernel symbols and callchains', 'unrelated/out-of-request SQL rows'],
            retained='Numeric SQL process/thread/source-row/correlation identities are local to each capture.',
            private_replay='Hashes commit to originals but public projection cannot independently replay omitted authentication, map validation, or simultaneous missing events.',
            map_hash_scope='Backing-file/source hashes and observed map checks; live mapped-memory content and physical final release remain unobserved.'),
        timestamp_policy='Every interval start/end is signed integer Nsight ns minus the private request-window start. UTC and the subtracted origin are omitted.',
        zoom_rule='First source-generation-1 read_subset in start order after request bound; [its start, its start + 20000000 ns). No overlap-based selection.',
        gpu_queries=0, captures_issued=0)
    results = {'source.json': c.encode(source_public)}
    for arm in c.ARMS:
        results.update(extract_arm(root, arm, handoff, source))
    old1 = private_root / 'artifacts-go-20261015-01/capture-result.json'
    old2 = private_root / 'artifacts-go-20261015-02/closeout.json'
    old3 = private_root / 'artifacts-go-20261015-03/collector-contract.json'
    old4 = private_root / 'artifacts-go-20261015-04/final-handoff.json'
    retro = private_root / 'artifacts-offline-audit-v2-20261015-04/offlineaudit_v2.json'
    history = []
    for gen, path, captures in [('-01', old1, 1), ('-02', old2, 0), ('-03', old3, 0), ('-04', old4, 1),
                                ('-04-retrospective-v2', retro, 0), ('-05', root / 'final-handoff.json', 2)]:
        d = load(path)
        row = dict(generation=gen, status=d['status'], new_captures=captures, accepted_arms=2 if gen == '-05' else 0,
                   full_trace_accepted=False, original_file=c.file_record(path))
        if gen == '-04':
            row.update(rejection_reasons=d['blocking']['rejection_reasons'], lookahead_status=d['lookahead']['status'])
        if gen == '-04-retrospective-v2':
            row.update(pick(d, 'retrospective historical_prepared_upgraded historical_acceptance_transferred corrected_predicate original_status original_files_unchanged'))
        history.append(row)
    results['history.json'] = c.encode(history)
    results['manifest.json'] = c.encode(dict(schema='ferrule.overlap.archive.v1',
        files={name: dict(sha256=c.digest(data), bytes=len(data)) for name, data in results.items()}))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-root', type=Path, default=c.HERE.parent)
    parser.add_argument('--check', action='store_true', help='Re-extract and compare without writing anything')
    args = parser.parse_args()
    results = build(args.private_root)
    for name, data in results.items():
        path = c.HERE / name
        if args.check:
            c.require(path.read_bytes() == data, 'public projection differs: ' + name)
        else:
            c.require(not path.exists(), 'refuse overwrite: use --check for an existing archive')
    if not args.check:
        for name, data in results.items():
            (c.HERE / name).write_bytes(data)
    print(json.dumps(dict(status='exact_private_projection_verified' if args.check else 'curated',
                          files=len(results), bytes=sum(map(len, results.values())), gpu_queries=0, captures_issued=0)))


if __name__ == '__main__':
    main()
