"""Fixed 9ef/7e1c info-only profile policy and offline audit; not a launcher."""
from copy import deepcopy
from pathlib import Path
import json
import sqlite3

import pre_serving as pre
import operator_retirement_ab as ab
import allocator_v3_release as release
import profile_2025 as tools
from profile_trace_audit_v2 import audit_correlations, process_id

ct = pre.ct
CONFIG = dict(release=tools.TOOL_CONFIG['release'], policy='operator_retirement_9ef_7e1c_info_only_v1',
    before_binary_sha256=ab.BEFORE_SHA, after_binary_sha256=release.BINARY_SHA,
    rankperf=False, log='info', window='actual_HTTP_start_end_unix_clock_anchors')


def selected(admission):
    return admission.get('profiler') == CONFIG


def validate(admission, purpose):
    import paired
    ct.require(purpose == 'profiling' and set(admission) == {'protocol','engines','profiler'}
        and admission['protocol'] == pre.PROFILE_PROTOCOL and set(admission['engines']) == {'ferrule'}
        and selected(admission), 'Unreviewed retirement profile shape/purpose')
    m = admission['engines']['ferrule']; sha = m['binary']['sha256']
    ct.require(sha in (ab.BEFORE_SHA,release.BINARY_SHA), 'Only fixed 9ef/7e1c profile artifacts are reviewed')
    ct.require(m['spec'] == paired.ferrule_spec(m['binary']['path'],paired.CACHE_WORKSPACE_SCOPE)
        and m['model'] == paired.model_contract() and m['decoded_cache'] == release.d.CACHE
        and m['serving_scope'] == paired.CACHE_WORKSPACE_SCOPE, 'Profile model/spec/geometry differs')
    if sha == ab.BEFORE_SHA:
        ct.require(m == ab.frozen_metadata(), 'Frozen 9ef profile metadata must be exact, not relabelled current sources')
        ab.frozen_audit()
        ct.require(all(m['source_pins'].get(p) == h for p,h in paired.source_audit()['source_pins'].items()),
                   'Frozen public gate/CLI changed')
    else:
        current = dict(admission,profiler=dict(tools.TOOL_CONFIG))
        paired.validate_admission(current,'profiling')
    tools.configure(tools.TOOL_CONFIG,pre)
    ct.check_pins(tools.TOOL_PINS)
    return admission


def admission(metadata):
    result = dict(protocol=pre.PROFILE_PROTOCOL,engines={'ferrule':deepcopy(metadata)},profiler=dict(CONFIG))
    validate(result,'profiling')
    return result


def request_windows(case, epoch):
    """Actual client request windows; no server DEBUG or invented decode markers."""
    result = {}
    for label in ('warmup','measured'):
        http = ct.load(case/'profile'/(label+'.json'))
        ct.require(http['passed'] is True and http['actual_input_tokens'] == 5 and http['actual_output_tokens'] == 8,
                   'Profile actual QA8 tokens/canary failed')
        start, end = http['start_clock'],http['end_clock']
        ct.require(start['monotonic_ns'] < end['monotonic_ns'] and start['unix_ns'] < end['unix_ns'], 'Invalid HTTP clocks')
        drift = abs((end['unix_ns']-start['unix_ns'])-(end['monotonic_ns']-start['monotonic_ns']))
        ct.require(drift <= 1000000, 'Wall/monotonic request clock drift exceeds 1ms')
        result[label] = dict(start_ns=start['unix_ns']-epoch,end_ns=end['unix_ns']-epoch,
            actual_input_tokens=5,actual_output_tokens=8,clock_delta_disagreement_ns=drift,
            scope='Client HTTP start/end anchors; includes transport/terminal response, not pure prefill/decode')
    ct.require(result['warmup']['end_ns'] <= result['measured']['start_ns'], 'Profile request windows overlap')
    return result


def event_kind(name):
    for prefix in ('cuda','cu'):
        for operation in ('Create','Record','Query','Destroy'):
            if name.startswith(prefix+'Event'+operation):
                return ('runtime' if prefix == 'cuda' else 'driver'),operation.lower()
    return None


def api_counts(rows, windows, model_pid):
    result = {label:dict(events={layer:{op:dict(count=0,duration_ns=0,returns={},names={})
        for op in ('create','record','query','destroy')} for layer in ('runtime','driver')},
        synchronization={},launches={},boundary_crossing_calls=[]) for label in windows}
    for row in rows:
        if process_id(row['globalTid']) != model_pid: continue
        for label,w in windows.items():
            if max(w['start_ns'],row['start']) >= min(w['end_ns'],row['end']): continue
            if not (w['start_ns'] <= row['start'] and row['end'] <= w['end_ns']):
                result[label]['boundary_crossing_calls'].append(dict(row)); continue
            name = row['name']; kind = event_kind(name)
            if kind:
                dest = result[label]['events'][kind[0]][kind[1]]
                dest['count'] += 1; dest['duration_ns'] += row['end']-row['start']
                value = str(row.get('returnValue')); dest['returns'][value] = dest['returns'].get(value,0)+1
                dest['names'][name] = dest['names'].get(name,0)+1
            group = 'launches' if 'Launch' in name else 'synchronization' if ('Synchronize' in name or 'StreamWaitEvent' in name) else None
            if group:
                dest = result[label][group].setdefault(name,dict(count=0,duration_ns=0))
                dest['count'] += 1; dest['duration_ns'] += row['end']-row['start']
    return result


def audit(pair):
    import paired
    pair = Path(pair).resolve(); prepared = ct.load(pair/'prepared.json')
    ct.require(selected(prepared['admission']), 'Offline retirement audit requires its typed info-only policy')
    paired.assess_slot(pair/'runs/P1','ferrule')
    case = pair/'runs/P1/session'; log = (case/'server.log').read_text()
    ct.require('mixed_rank_performance' not in log and 'request_worker_terminal' not in log,
               'Profile unexpectedly enabled performance/request DEBUG')
    release.d.snapshot_rows(log,profiling=False)
    model_pid = ct.load(case/'profile/model-identity.json')['pid']
    ct.require(model_pid == ct.load(case/'profile-server.json')['pid'], 'Profile actual model PID differs')
    database = case/'nsight/nsys.sqlite'
    db = sqlite3.connect(database.as_uri()+'?mode=ro',uri=True); db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    strings = {r['id']:r['value'] for r in db.execute('SELECT id,value FROM StringIds')}
    epoch = db.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    windows = request_windows(case,epoch)
    mapping = {(r['pid'],r['cudaId']):r['gpuId'] for r in db.execute('SELECT * FROM TARGET_INFO_CUDA_DEVICE')}
    gpus = {r['id']:'GPU-'+r['uuid'].removeprefix('GPU-') for r in db.execute('SELECT id,uuid FROM TARGET_INFO_GPU')}
    activities = []; apis = []; coverage = {}
    counts = {label:{uuid:dict(kernels=0,kernel_duration_ns=0,memcpy={},memsets=0) for uuid in paired.shared.UUIDS} for label in windows}
    for kind,table in (('kernel','CUPTI_ACTIVITY_KIND_KERNEL'),('memcpy','CUPTI_ACTIVITY_KIND_MEMCPY'),('memset','CUPTI_ACTIVITY_KIND_MEMSET')):
        ct.require(table in tables, 'Required profile activity table absent: '+table)
        for row in db.execute('SELECT * FROM '+table):
            row = dict(row); pid = process_id(row['globalPid'])
            ct.require(pid == model_pid, 'Profile activity belongs to a different PID')
            uuid = gpus[mapping[pid,row['deviceId']]]
            ct.require(uuid in paired.shared.UUIDS, 'Profile activity on a nonselected physical UUID')
            activities.append(dict(row,kind=kind,uuid=uuid))
            for label,w in windows.items():
                if not (w['start_ns'] <= row['start'] and row['end'] <= w['end_ns']): continue
                dest = counts[label][uuid]
                if kind == 'kernel':
                    dest['kernels'] += 1; dest['kernel_duration_ns'] += row['end']-row['start']
                elif kind == 'memcpy':
                    direction = {1:'H2D',2:'D2H',8:'D2D'}.get(row['copyKind'],'other_'+str(row['copyKind']))
                    copy = dest['memcpy'].setdefault(direction,dict(count=0,bytes=0,duration_ns=0))
                    copy['count'] += 1; copy['bytes'] += row['bytes']; copy['duration_ns'] += row['end']-row['start']
                else: dest['memsets'] += 1
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        coverage[table] = table in tables
        if table not in tables: continue
        for r in db.execute('SELECT start,end,globalTid,correlationId,nameId,returnValue FROM '+table):
            apis.append(dict(r,name=strings[r['nameId']],table=table))
    ct.require(apis, 'No exported CUDA API rows; counts unknown, not zero')
    correlation = audit_correlations(activities,apis,windows,model_pid)
    counts_api = api_counts(apis,windows,model_pid)
    diagnostics = [dict(r) for r in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else None
    db.close()
    for label in windows:
        ct.require(all(row['kernels'] > 0 for row in counts[label].values()), 'No observed two-UUID kernel coverage')
    result = dict(schema='ferrule.paper.operator-retirement-official-profile.v1',
        status='observed_closed' if all(v['observed_correlation_closed'] for v in correlation.values()) else 'observed_not_closed',
        binary_sha256=prepared['admission']['engines']['ferrule']['binary']['sha256'],policy=CONFIG,
        model_pid=model_pid,session_epoch_ns=epoch,actual_request_windows=windows,
        bidirectional_correlation=correlation,per_uuid_activity=counts,api_counts=counts_api,api_table_coverage=coverage,
        diagnostics=diagnostics,full_trace_accepted=False,
        full_trace_warning='Retain all Nsight/DIAGNOSTIC_EVENT completeness warnings; exported bidirectional closure does not exclude paired API/activity loss',
        unique_cuda_events=None,completed_retirements=None,unknown_pressure=None,pending_peak=None,
        scope='Same actual HTTP request bounds for APIs and activities, exact model PID and physical UUID mapping; info-only, rankperf off',
        limitations=['Not pure prefill/decode windows; no server request DEBUG markers.',
            'Runtime/driver call layers must not be added as unique events; query not-ready is polling, not retirement completion.',
            'Host API durations include waits, overlap across threads/devices and cannot be added into throughput benefits.',
            'Stripped release, no CPU stack attribution; full-trace completeness unproven.'],
        artifact_pins=ct.inventory([database,case/'nsight/nsys.nsys-rep',case/'server.log',pair/'prepared.json',
            case/'profile/warmup.json',case/'profile/measured.json',pair/'runs/P1/probe-lock-release.json']))
    output = pair/'retirement-profile-audit.json'
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result
