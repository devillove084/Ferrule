#!/usr/bin/env python3
"""Offline, CPU-only verification of the deliberately pruned actual-05 archive.

No Nsight, CUDA, private evidence, repository imports, or third-party packages.
This verifies the public projection, not omitted private lifecycle evidence.
"""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ARMS = ('blocking', 'lookahead')
MASK = (1 << 24) - 1
CLI = '7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01'
SOURCE = '86c58a017d7c4b53490f536a3b1a133f2f7848cc'
FIX = '5e4e29cb883c84392dfaff15120ee910dac093ad'
QA_IDS = [25358, 13, 198, 760, 6511, 314, 6124, 369]
WARNINGS = (
    'No NVTX events collected. Does the process use NVTX?',
    'CUDA profiling might have not been started correctly.',
    'No CUDA events collected. Does the process use CUDA?',
)
LAUNCH = re.compile(r'^(?:cudaLaunch(?:Kernel(?:ExC)?|CooperativeKernel)|cuLaunch(?:Kernel(?:Ex)?|CooperativeKernel))(?:_v[0-9]+|_ptsz|_ptds)*$')
FIELDS = {
    'nvtx': 'row_id kind start_ns end_ns global_pid global_tid end_global_tid event_type domain_id range_id operation source_generation requested_bytes image_generation layer expert rows'.split(),
    'pread': 'row_id start_ns end_ns global_pid global_tid event_class name_id name return_value domain_id read_range_id osrt_bytes'.split(),
    'kernel': 'row_id start_ns end_ns global_pid device_id context_id stream_id correlation_id name_id graph_node_id api_row_id api_table api_start_ns api_end_ns api_global_tid api_name_id api_name api_return_value domain_id expert_range_id'.split(),
}
TEXT_FIELDS = {'kind', 'name', 'api_table', 'api_name'}
METRICS = ('worker_read_envelope_union_ns', 'worker_pread_union_ns', 'expert_kernel_union_ns',
           'read_envelope_kernel_intersection_ns', 'pread_kernel_intersection_ns')
EXPECTED = {
    'blocking': (2982515538, 2299019830, 194741187, 0, 0),
    'lookahead': (2989969321, 2295077102, 194217266, 174409210, 751507),
}
GATES = {'one_frozen_qa_http_token_join', 'authenticated_model_pid_birth_source',
         'ready_and_after_raw_maps', 'dynamic_owner_pair_closed', 'authenticated_echild',
         'original_all_eight_stable_idle', 'collector_export_closed'}
EXPORT = {
    'EXPORT_PRODUCT_NAME': 'NVIDIA Nsight Systems', 'EXPORT_PRODUCT_VERSION': '2025.3.1.90',
    'EXPORT_SCHEMA_VERSION': '3.20.2', 'EXPORT_PARAM_LAZY': 'false',
    'EXPORT_PARAM_TIME_NORMALIZE': 'false', 'EXPORT_PARAM_TIME_SHIFT': '0',
    'EXPORT_PARAM_TABLES_TO_EXPORT': '', 'EXPORT_PARAM_TIMES_TO_EXPORT': '',
}
CAPTURE = {
    'CAPTURE_EVENT_TYPE': ['OSRuntime', 'NvtxEvents', 'Cuda'],
    'OS_RUNTIME_OPTIONS:DURATION_THRESHOLD_NS': ['1000'],
    'RATE_HZ': ['0'], 'COLLECT_GPU_CTX_SW_TRACE': ['false'],
    'LINUX_PERF_OPTIONS:COLLECT_CPU_CTXSW_TRACE': ['true'],
    'LINUX_PERF_OPTIONS:COLLECTION_MODE': ['ProcessTree'],
    'NVTX_OPTIONS:NVTX_DOMAIN_FILTER_STRING': [''], 'INCLUDE_CHILDREN': ['true'],
}
PROOF_KEYS = set('schema arm preliminary_worker final_scope request_duration_ns process_roles diagnostics export_header capture_header osrt_schema_columns osrt_counter_coverage full_export_projection ledger maps qa metrics original_files source_binding physical_mapping collector_options'.split())


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(data):
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + '\n').encode()


def file_record(path):
    data = path.read_bytes()
    return {'sha256': digest(data), 'bytes': len(data)}


def union(intervals):
    out = []
    for a, b in sorted(intervals):
        require(type(a) is int and type(b) is int and a < b, 'invalid interval')
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def intersection(left, right):
    left, right = union(left), union(right)
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if a < b:
            out.append((a, b))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return out


def length(intervals):
    return sum(b - a for a, b in union(intervals))


def intervals(rows):
    return [(r['start_ns'], r['end_ns']) for r in rows]


def index(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['global_tid']].append(r)
    result = {}
    for tid, items in groups.items():
        items = sorted(items, key=lambda r: r['start_ns'])
        require(all(a['end_ns'] <= b['start_ns'] for a, b in zip(items, items[1:])), 'overlapping same-thread ranges')
        result[tid] = ([r['start_ns'] for r in items], items)
    return result


def containing(idx, tid, start, end):
    starts, rows = idx.get(tid, ([], []))
    i = bisect_right(starts, start) - 1
    if i >= 0 and start < rows[i]['end_ns']:
        require(end <= rows[i]['end_ns'], 'interval crosses range boundary')
        return rows[i]
    require(i + 1 == len(rows) or end <= rows[i + 1]['start_ns'], 'interval crosses range boundary')
    return None


def read_csv(path, kind):
    with gzip.open(path, 'rt', newline='', encoding='ascii') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == FIELDS[kind], 'missing/unknown/reordered CSV columns')
        result = []
        for row in reader:
            require(None not in row and None not in row.values(), 'cut CSV row')
            parsed = {}
            for k, v in row.items():
                if v == 'null':
                    parsed[k] = None
                elif k in TEXT_FIELDS:
                    parsed[k] = v
                else:
                    require(re.fullmatch(r'-?(?:0|[1-9][0-9]*)', v), 'non-integer CSV field')
                    parsed[k] = int(v)
            result.append(parsed)
    return result


def assert_hash(value):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value), 'invalid source SHA256')


def validate_proof(p):
    require(set(p) == PROOF_KEYS, 'cut/unknown raw evidence field')
    require(p['schema'] == 'ferrule.overlap.public-projection.v1' and p['arm'] in ARMS, 'projection identity')
    pre, final = p['preliminary_worker'], p['final_scope']
    require(pre == dict(status='rejected', full_trace_accepted=False,
                       scoped_sql_evidence_consistent=True, scoped_overlap_accepted=False), 'preliminary verdict changed')
    require(set(final) == {'scoped_overlap_accepted', 'full_trace_accepted', 'dropped_event_count',
                          'simultaneous_unreported_loss_excluded', 'universal_trace_completeness_established',
                          'lifecycle_gates', 'missing_gates', 'scoped_issues', 'log_issues', 'loss_diagnostic_count'}, 'cut scope evidence')
    require(final['scoped_overlap_accepted'] is True and final['full_trace_accepted'] is False,
            'scoped-only acceptance required')
    require(final['dropped_event_count'] is None and final['simultaneous_unreported_loss_excluded'] is False
            and final['universal_trace_completeness_established'] is False, 'loss unknown must remain unknown')
    require(set(final['lifecycle_gates']) == GATES and all(v is True for v in final['lifecycle_gates'].values())
            and final['missing_gates'] == final['scoped_issues'] == final['log_issues'] == []
            and final['loss_diagnostic_count'] == 0, 'missing scoped gates')
    require(p['export_header'] == EXPORT and p['capture_header'] == CAPTURE, 'cut/filtered collector header')
    require(p['osrt_schema_columns'] == ['start', 'end', 'eventClass', 'globalTid', 'nameId', 'returnValue', 'nestingLevel', 'callchainId'], 'OSRT schema changed')
    require(p['collector_options'] == dict(trace=['cuda', 'nvtx', 'osrt'], sample='none',
                cpuctxsw='process-tree', osrt_file_access=True, osrt_threshold_ns=1000), 'collector options changed')
    roles = p['process_roles']
    require(len(roles) == 2 and {r['role'] for r in roles} == {'model', 'non_cuda_wrapper'}, 'role coverage')
    for r in roles:
        require(set(r) == {'role', 'globalPid', 'pid', 'ppid', 'birth_ticks', 'executable_sha256', 'source_binding_sha256'}, 'cut role projection')
        require(r['globalPid'] & MASK == 0 and (r['globalPid'] >> 24) & MASK == r['pid'], 'process key mismatch')
        assert_hash(r['executable_sha256']); assert_hash(r['source_binding_sha256'])
    model = next(r for r in roles if r['role'] == 'model')
    wrapper = next(r for r in roles if r['role'] == 'non_cuda_wrapper')
    require(model['ppid'] == wrapper['pid'] and model['executable_sha256'] == CLI, 'unbound model role')
    require(p['source_binding'] == dict(cli_sha256=CLI, source_commit=SOURCE, measurement_fix_commit=FIX,
                                       source_generation=1, image_generation=1), 'source binding changed')
    require(p['physical_mapping'] == dict(model_pid=model['pid'], logical_device_id=0,
                physical_uuid_bound=True, uuid_redacted=True, gpu_inventory_count=8), 'device binding projection')
    warnings = []
    require(len(p['diagnostics']) == 20, 'cut diagnostic rows')
    for d in p['diagnostics']:
        require(set(d) == {'row_id', 'timestampType', 'source', 'severity', 'text', 'globalPid',
                          'classification', 'affects_scoped_overlap', 'affects_full_trace',
                          'text_redacted', 'original_text_sha256', 'original_row_sha256'}, 'cut diagnostic fields')
        assert_hash(d['original_text_sha256']); assert_hash(d['original_row_sha256'])
        if not d['text_redacted']:
            require(digest(d['text'].encode()) == d['original_text_sha256'], 'diagnostic text changed')
        if d['severity'] > 1:
            require(d['text'] in WARNINGS and d['globalPid'] == wrapper['globalPid'] and d['source'] == 3
                    and d['severity'] == 2 and d['timestampType'] == 2 and d['text_redacted'] is False
                    and d['classification'] == 'authenticated_non_cuda_wrapper_no_payload'
                    and d['affects_scoped_overlap'] is False and d['affects_full_trace'] is True,
                    'unknown/target/global warning: scope rejected')
            warnings.append(d['text'])
        else:
            require(d['classification'] == 'informational' and d['affects_scoped_overlap'] is False
                    and d['affects_full_trace'] is False, 'informational diagnostic scope changed')
    require(Counter(warnings) == Counter(WARNINGS), 'missing/duplicate wrapper warning')
    coverage = p['osrt_counter_coverage']
    require(coverage['issues'] == [] and coverage['dropped_event_count'] is None
            and coverage['total_rows'] == coverage['api_rows'] + coverage['lifecycle_marker_rows']
            and coverage['lifecycle_marker_rows'] == 6, 'OSRT counter coverage')
    require({c['globalPid'] for c in coverage['counters']} == {r['globalPid'] for r in roles}, 'OSRT roles missing')
    require(sum(c['api_rows'] for c in coverage['counters']) == coverage['api_rows'], 'OSRT counter sum')
    for c in coverage['counters']:
        require(c['matched'] is True and c['official_counts'] == [c['api_rows']]
                and c['event_class_counts'] == dict(TRACE_PROCESS_EVENT_OS_RUNTIME=c['api_rows'],
                    TRACE_PROCESS_EVENT_OS_RUNTIME_START=1, TRACE_PROCESS_EVENT_OS_RUNTIME_FINISH=2), 'OSRT count mismatch')
    full = p['full_export_projection']
    require(set(full) == {'table_counts', 'all_kernel_rows', 'all_math_launch_rows', 'correlation_counts',
            'retained_expert_kernels', 'excluded_kernels', 'retained_preads', 'excluded_preads',
            'wrapper_nvtx_rows', 'wrapper_cuda_api_rows', 'wrapper_kernel_rows', 'all_diagnostic_rows'}, 'cut whole-export proof')
    require(full['all_kernel_rows'] == full['all_math_launch_rows'] == 63042
            and full['retained_expert_kernels'] == 33403 and full['retained_preads'] == 22068
            and sum(full['excluded_kernels'].values()) + 33403 == 63042, 'whole-export projection counts')
    require(full['correlation_counts'] == dict(kernel_without_unique_math_api=0, math_api_without_unique_kernel=0,
            duplicate_math_api_keys=0, duplicate_kernel_keys=0), 'whole-export closure failed')
    require(full['wrapper_nvtx_rows'] == full['wrapper_cuda_api_rows'] == full['wrapper_kernel_rows'] == 0
            and full['all_diagnostic_rows'] == 20, 'wrapper payload or diagnostic loss')
    require(full['table_counts']['OSRT_API'] == coverage['total_rows']
            and full['table_counts']['NVTX_EVENTS'] == 11035
            and full['table_counts']['CUPTI_ACTIVITY_KIND_KERNEL'] == 63042, 'table count proof')
    require(set(p['maps']) == {'ready', 'after-requests'}, 'map phases missing')
    for m in p['maps'].values():
        require(m['status'] == 'passed' and m['required'] == 16 and m['observed'] == 19, 'map summary failed')
        assert_hash(m['raw_sha256'])
    qa = p['qa']
    require(qa['status'] == 'passed' and qa['actual_generated_ids'] == QA_IDS
            and qa['decoder_invocations'] == 9 and qa['usage'] == dict(prompt_tokens=5, completion_tokens=8, total_tokens=13), 'QA changed')
    for name, record in p['original_files'].items():
        require('/' not in name and set(record) == {'sha256', 'bytes'} and record['bytes'] > 0, 'private source descriptor')
        assert_hash(record['sha256'])
    required = {'sqlite', 'report', 'worker_audit', 'final_analysis', 'read_ledger', 'quality',
                'collector_manifest', 'collector_command', 'ready_roles', 'after_roles', 'ready_maps', 'after_maps',
                'worker_capture', 'custody', 'supervisor', 'export_result', 'server_log'}
    require(set(p['original_files']) == required, 'cut raw-source inventory')
    return model['globalPid']


def validate_arm(p, data):
    model = validate_proof(p)
    require(set(data) == set(FIELDS), 'missing interval table')
    for kind, rows in data.items():
        require(rows and all(set(r) == set(FIELDS[kind]) for r in rows), 'cut/unknown interval fields')
        require(rows == sorted(rows, key=lambda r: (r['start_ns'], r['end_ns'], r['row_id'])), 'noncanonical interval order')
        require(len({r['row_id'] for r in rows}) == len(rows), 'duplicate source row')
        for r in rows:
            require(type(r['start_ns']) is int and type(r['end_ns']) is int
                    and 0 <= r['start_ns'] < r['end_ns'] <= p['request_duration_ns'], 'invalid request interval')
            require(r['global_pid'] == model, 'foreign process interval')
    nvtx = data['nvtx']
    require(Counter(r['kind'] for r in nvtx) == Counter({'read_subset': 3678, 'layer.expert': 3678, 'layer.expert.upload': 3678}), 'range counts')
    require(len({(r['domain_id'], r['range_id']) for r in nvtx}) == len(nvtx), 'duplicate NVTX range')
    reads = [r for r in nvtx if r['kind'] == 'read_subset']
    experts = [r for r in nvtx if r['kind'] == 'layer.expert']
    uploads = [r for r in nvtx if r['kind'] == 'layer.expert.upload']
    for r in nvtx:
        require(r['global_tid'] & ~MASK == model and r['end_global_tid'] in (None, r['global_tid'])
                and r['event_type'] == 60 and r['domain_id'] == 1, 'NVTX identity/type')
        if r['kind'] == 'read_subset':
            require(r['source_generation'] == 1 and r['requested_bytes'] == 3146112
                    and all(r[k] is None for k in ('image_generation', 'layer', 'expert', 'rows')), 'read source/byte/identity changed')
        else:
            require(r['image_generation'] == 1 and 0 <= r['layer'] < 40 and 0 <= r['expert'] < 256
                    and r['rows'] > 0 and all(r[k] is None for k in ('operation', 'source_generation', 'requested_bytes')), 'expert source/identity changed')
    require([r['operation'] for r in reads] == list(range(1, 3679)), 'source operations not contiguous/unique')
    require(not {r['global_tid'] for r in reads} & {r['global_tid'] for r in experts}, 'worker/submit thread conflated')
    read_idx, expert_idx, upload_idx = index(reads), index(experts), index(uploads)
    calls, seen_preads = Counter(), set()
    for r in data['pread']:
        require(r['global_tid'] & ~MASK == model and r['name'] in ('pread', 'pread64')
                and r['return_value'] == 0 and r['osrt_bytes'] is None, 'OSRT bytes/return encoding is not ssize_t')
        read = containing(read_idx, r['global_tid'], r['start_ns'], r['end_ns'])
        require(read and (read['domain_id'], read['range_id']) == (r['domain_id'], r['read_range_id']), 'orphan/misassigned pread')
        key = r['global_tid'], r['start_ns'], r['end_ns']
        require(key not in seen_preads, 'duplicate pread alias'); seen_preads.add(key)
        calls[r['read_range_id']] += 1
    index(data['pread'])
    require(len(data['pread']) == 22068 and len(calls) == 3678 and set(calls.values()) == {6}, 'six preads per range')
    seen_corr, seen_api, hits = set(), set(), Counter()
    for r in data['kernel']:
        key = r['global_pid'], r['correlation_id']
        require(key not in seen_corr, 'duplicate kernel/API correlation'); seen_corr.add(key)
        api_key = r['api_table'], r['api_row_id']
        require(api_key not in seen_api, 'duplicate launch row'); seen_api.add(api_key)
        require(r['api_global_tid'] & ~MASK == model and r['device_id'] == 0
                and r['graph_node_id'] in (None, 0) and r['api_table'] in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER')
                and LAUNCH.fullmatch(r['api_name']) and r['api_return_value'] == 0, 'unsupported kernel/API identity')
        require(0 <= r['api_start_ns'] <= r['api_end_ns'] <= p['request_duration_ns']
                and r['api_start_ns'] <= r['start_ns'], 'invalid kernel/API order')
        expert = containing(expert_idx, r['api_global_tid'], r['api_start_ns'], r['api_end_ns'])
        require(expert and (expert['domain_id'], expert['range_id']) == (r['domain_id'], r['expert_range_id']), 'orphan/misassigned kernel')
        require(containing(upload_idx, r['api_global_tid'], r['api_start_ns'], r['api_end_ns']) is None, 'upload is not math')
        hits[r['expert_range_id']] += 1
    require(len(data['kernel']) == 33403 and len(hits) == 3678, 'expert launch/kernel coverage')
    ledger = p['ledger']
    require(ledger['model_source_generation'] == 1 and ledger['invocation_count'] == 9
            and ledger['status'] == 'closed' and ledger['effective_mode'] == p['arm'], 'source ledger owner')
    stats = ledger['stats']
    for k in ('submitted_reads', 'worker_started_reads', 'io_started_reads', 'completed_reads', 'verified_reads'):
        require(stats[k] == len(reads), 'unverified read ledger')
    for k in ('failed_reads', 'cancelled_reads', 'unknown_reads', 'unobserved_terminal_reads', 'unobserved_reader_intervals', 'unknown_bytes'):
        require(stats[k] == 0, 'unknown/failed read ledger')
    requested = sum(r['requested_bytes'] for r in reads)
    require(requested == stats['proved_read_bytes'] == stats['reader_interval_bytes'] == 11571399936
            and stats['reader_interval_calls'] == len(data['pread']), 'typed bytes/calls mismatch')
    receipt = ledger['last_receipt']
    require(receipt['outcome'] == 'source_verified' and receipt['identity']['operation'] == reads[-1]['operation']
            and receipt['identity']['image_generation'] == 1 and receipt['requested_bytes'] == receipt['io_bytes'] == 3146112,
            'last receipt mismatch (not a full operation/expert journal)')
    r, c, k = intervals(reads), intervals(data['pread']), intervals(data['kernel'])
    values = (length(r), length(c), length(k), length(intersection(r, k)), length(intersection(c, k)))
    result = dict(zip(METRICS, values))
    require(values == EXPECTED[p['arm']], 'interval unions differ from sealed actual-05 result')
    require(all(p['metrics'][name] == v for name, v in result.items()), 'source aggregate mismatch')
    require(p['metrics']['osrt_read_bytes'] is None and p['metrics']['source_verified_read_bytes'] == requested,
            'source-bound bytes are not OSRT bytes')
    result.update(read_ranges=len(reads), pread_calls=len(c), expert_kernels=len(k),
                  source_verified_read_bytes=requested, osrt_read_bytes=None,
                  read_envelope_overlap_percent=100 * values[3] / values[0],
                  pread_overlap_percent=100 * values[4] / values[1],
                  zoom_start_ns=reads[0]['start_ns'], zoom_end_ns=reads[0]['start_ns'] + 20000000)
    return result


def load(root=HERE):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    require(set(manifest) == {'schema', 'files'} and manifest['schema'] == 'ferrule.overlap.archive.v1', 'manifest schema')
    names = {'source.json', 'history.json'} | {a + '-' + n for a in ARMS for n in
            ('proof.json', 'nvtx.csv.gz', 'pread.csv.gz', 'kernel.csv.gz')}
    require(set(manifest['files']) == names, 'archive file set changed')
    for name, record in manifest['files'].items():
        require(file_record(root / name) == record, 'archive hash/size mismatch: ' + name)
    source = json.loads((root / 'source.json').read_text())
    require(source['candidate_identity']['commit'] == SOURCE and source['candidate_identity']['cli_binary_sha256'] == CLI
            and source['measurement_fix_commit'] == FIX and source['final_handoff']['accepted_arms'] == 2
            and source['final_handoff']['captures_launched'] == 2 and source['final_handoff']['full_trace_accepted'] is False,
            'actual-05 source/final-handoff changed')
    require(source['read_measurement']['osrt_read_bytes'] is None
            and source['read_measurement']['call_success_proved'] is False
            and source['read_measurement']['physical_io_bytes_claim'] is False, 'byte semantics changed')
    history = json.loads((root / 'history.json').read_text())
    require([h['generation'] for h in history] == ['-01', '-02', '-03', '-04', '-04-retrospective-v2', '-05'], 'history missing')
    require([h['accepted_arms'] for h in history] == [0, 0, 0, 0, 0, 2]
            and [h['new_captures'] for h in history] == [1, 0, 0, 1, 0, 2]
            and all(h['full_trace_accepted'] is False for h in history), 'historical verdict upgraded')
    require(history[3]['status'] == 'rejected_stop_lookahead_unissued'
            and history[4]['historical_acceptance_transferred'] is False
            and history[4]['corrected_predicate']['scoped_overlap_accepted'] is True, 'retrospective distinction missing')
    proofs, data, results = {}, {}, {}
    for arm in ARMS:
        proofs[arm] = json.loads((root / (arm + '-proof.json')).read_text())
        data[arm] = {kind: read_csv(root / (arm + '-' + kind + '.csv.gz'), kind) for kind in FIELDS}
        results[arm] = validate_arm(proofs[arm], data[arm])
    require(proofs['blocking']['original_files']['sqlite']['sha256'] != proofs['lookahead']['original_files']['sqlite']['sha256'], 'fresh pair must be distinct')
    return manifest, proofs, data, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=HERE)
    args = parser.parse_args()
    manifest, _, _, results = load(args.directory)
    print(json.dumps(dict(status='public_projection_verified', scoped_overlap_accepted=True,
          full_trace_accepted=False, dropped_event_count=None, private_lifecycle_replayed=False,
          compressed_interval_bytes=sum(r['bytes'] for n, r in manifest['files'].items() if n.endswith('.gz')),
          arms=results), indent=2))


if __name__ == '__main__':
    main()
