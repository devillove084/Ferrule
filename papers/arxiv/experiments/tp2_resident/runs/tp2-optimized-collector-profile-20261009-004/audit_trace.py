"""Run004 offline audit; stdlib only, no GPU, process launch, or source mutation."""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from datetime import datetime

D = Path(__file__).resolve().parent
S = D/'runs/P1/session'
C = sqlite3.connect('file:'+str(S/'nsight/nsys.sqlite')+'?mode=ro', uri=True)
C.row_factory = sqlite3.Row
queries = []
def query(sql, parameters=()):
    queries.append(dict(sql=sql, parameters=parameters))
    return [dict(row) for row in C.execute(sql, parameters)]
def load(path):
    return json.loads(path.read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()
def save(name, value):
    with (D/name).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

tables = {row['name'] for row in query("SELECT name FROM sqlite_master WHERE type='table'")}
analysis = query('SELECT * FROM ANALYSIS_DETAILS')[0]
clock = query('SELECT * FROM TARGET_INFO_SESSION_START_TIME')[0]
requests = []
for label in ('warmup', 'measured'):
    row = load(S/'profile'/(label+'.json'))
    start = row['start_clock']['monotonic_ns']-analysis['startTime']
    end = row['end_clock']['monotonic_ns']-analysis['startTime']
    unix_start = row['start_clock']['unix_ns']-clock['utcEpochNs']
    unix_end = row['end_clock']['unix_ns']-clock['utcEpochNs']
    assert 0 <= start < end <= analysis['duration']
    assert abs(unix_start-start) < 1000000 and abs(unix_end-end) < 1000000
    requests.append(dict(label=label,status=row['status'],passed=row['passed'],
        actual_output_tokens=row['actual_output_tokens'],ttft_seconds=row['ttft_seconds'],
        e2e_seconds=row['request_e2e_seconds'],start_ns=start,end_ns=end,
        inside_capture=True,clock_crosscheck_ns=[unix_start-start,unix_end-end],
        partial_api_by_thread=query('SELECT globalTid, count(*) calls, sum(end-start) duration_ns FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start >= ? AND end <= ? GROUP BY globalTid',(start,end)),
        partial_copies=query('SELECT deviceId,copyKind,count(*) calls,sum(bytes) bytes,sum(end-start) duration_ns FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ? GROUP BY deviceId,copyKind',(start,end))))
api = query('SELECT s.value name,count(*) calls,sum(r.end-r.start) duration_ns,max(r.end-r.start) max_ns FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId GROUP BY s.value ORDER BY duration_ns DESC')
threads = query('SELECT r.globalTid,s.value name,count(*) calls,min(r.start) start_ns,max(r.end) end_ns FROM CUPTI_ACTIVITY_KIND_RUNTIME r LEFT JOIN ThreadNames t ON t.globalTid=r.globalTid LEFT JOIN StringIds s ON s.id=t.nameId GROUP BY r.globalTid')
copies = query('SELECT deviceId,copyKind,count(*) calls,sum(bytes) bytes,sum(end-start) duration_ns FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY deviceId,copyKind')
logs = []
for line in (S/'server.log').read_text().splitlines():
    if line.startswith('{'):
        try: logs.append(json.loads(line))
        except json.JSONDecodeError: pass
milestones = [{k:row[k] for k in ('timestamp','fields')} for row in logs if any(t in row['fields'].get('message','') for t in ('build entering','experimental source-bound','image/KV ready','claim completion','server listening','shutdown fence'))]
init = next(row for row in logs if 'build entering' in row['fields'].get('message',''))
source = next(row for row in logs if 'experimental source-bound' in row['fields'].get('message',''))
ready = [row for row in logs if 'image/KV ready' in row['fields'].get('message','')]
def seconds(a,b):
    return (datetime.fromisoformat(b['timestamp'])-datetime.fromisoformat(a['timestamp'])).total_seconds()
case = load(S/'case.json')
wrapper = load(S/'profile-server.json')
release = load(D/'final-owner-release.json')
baseline = load(S/'collector-baseline.json')
after_model = load(S/'collector-after-model.json')
assert release['all_eight_idle'] and release['all_locks_available_and_released'] and release['authenticated_drained']
assert case['request_gate']['status']=='complete' and wrapper['returncode']==0
assert len(after_model['samples']) >= 2
selected = baseline['model_gpu_uuids']
max_nonselected = {u:0 for u in baseline['instrumentation'] if u not in selected}
valid_gpu_samples = 0
for line in (S/'samples.jsonl').read_text().splitlines():
    sample = json.loads(line)
    gpu = sample.get('gpu')
    if not isinstance(gpu,dict) or 'devices' not in gpu: continue
    valid_gpu_samples += 1
    for device in gpu['devices']:
        u = device['uuid']
        if u in max_nonselected:
            max_nonselected[u] = max(max_nonselected[u],device['gpu_bytes']-baseline['instrumentation'][u]['gpu_bytes'])
            assert device['utilization']==0
assert valid_gpu_samples > 0
assert max(max_nonselected.values()) <= 1 << 20
for sample in after_model['samples'][-2:]:
    assert not sample['apps']
    for row in sample['devices']:
        assert row['gpu_bytes']==baseline['instrumentation'][row['uuid']]['gpu_bytes'] and row['utilization']==0
result = dict(schema='ferrule.paper.offline-trace-audit.v1',run_id=D.name,
    status='rejected_incomplete_cuda_trace',lifecycle_status='pass',benchmark=False,fairmatch=False,exact_speedup=None,
    release_binary_sha256=load(D/'profiling.json')['release_binary_sha256'],
    trace='cuda-sw,nvtx,osrt',kernel_table_present='CUPTI_ACTIVITY_KIND_KERNEL' in tables,
    nvtx_table_present='NVTX_EVENTS' in tables,kernel_duration_ns=None,host_gpu_gap_ns=None,
    kernel_bottleneck_ranking=None,all_api_metrics_complete=False,
    reason='Missing kernel table; NumTpcs and StringStorage importer exceptions persist with explicit software trace. Root cause inside installed collector/importer unresolved; not a production CUDA failure or proof hardware selection caused earlier failures.',
    diagnostics=query('SELECT severity,text,count(*) count FROM DIAGNOSTIC_EVENT GROUP BY severity,text ORDER BY count(*) DESC'),
    analysis=analysis,clock_anchor=clock,requests=requests,
    gpu_metadata=query('SELECT id,uuid,name FROM TARGET_INFO_GPU ORDER BY id'),
    cuda_device_metadata=query('SELECT * FROM TARGET_INFO_CUDA_DEVICE'),
    partial_api_threads=threads,partial_api_summary=api,partial_copy_summary=copies,
    partial_osrt_summary=query('SELECT s.value name,count(*) calls,sum(r.end-r.start) duration_ns FROM OSRT_API r JOIN StringIds s ON s.id=r.nameId GROUP BY s.value ORDER BY duration_ns DESC'),
    startup_seconds=case['startup_seconds'],startup_milestones=milestones,
    startup_observed_intervals=dict(build_entry_to_source_admission_seconds=seconds(init,source),
        source_admission_to_owner_image_ready_seconds={str(r['fields']['owner']):seconds(source,r) for r in ready}),
    baseline_mib={u:v/(1 << 20) for u,v in baseline['instrumentation_delta_bytes'].items()},
    nonselected_delta_max_bytes=max_nonselected,valid_gpu_samples=valid_gpu_samples,
    b1_return_samples=len(after_model['samples']),cleanup='model B1, collector original B0, authenticated drain, final independent all-eight lock availability/release verified',
    caveats=['Partial imported records are not full request counts or workload totals.',
        'Synchronization API waits include GPU work; cross-thread durations overlap.',
        'No kernel or scheduling trace means no measured idle gaps or decode/GEMM/pack attribution.',
        'Startup intervals are log milestones, not isolated optimization costs.',
        'Installed CUDA device metadata contains an inconsistent UUID; do not infer owner mapping from that row. Physical TARGET_INFO_GPU and authenticated visibility retained.'])
paths = [D/'audit_trace.py',D/'prepared.json',D/'profiling.json',D/'final-owner-release.json',S/'server.log',S/'case.json',S/'collector-baseline.json',S/'collector-after-model.json',S/'samples.jsonl',*list((S/'profile').glob('*.json')),*list((S/'nsight').glob('*'))]
result['artifact_pins']={str(p):sha(p) for p in paths if p.is_file()}
save('trace-queries.json',queries)
result['artifact_pins'][str(D/'trace-queries.json')]=sha(D/'trace-queries.json')
save('trace-audit.json',result)
print(json.dumps({k:result[k] for k in ('status','lifecycle_status','startup_seconds','startup_observed_intervals','nonselected_delta_max_bytes','valid_gpu_samples','partial_api_threads')},indent=2))
