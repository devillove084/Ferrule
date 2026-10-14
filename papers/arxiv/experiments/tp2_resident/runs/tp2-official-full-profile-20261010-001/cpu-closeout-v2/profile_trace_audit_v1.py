#!/usr/bin/env python3
"""Offline official SQLite audit. Missing trace/phase markers never become PASS."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

EXPECTED = {'GPU-b980f22b-58a3-fc88-21c1-ae86053d8535', 'GPU-63f1bd01-ddf9-a742-ee48-9389892493d9'}


def require(value, message):
    if not value:
        raise ValueError(message)


def epoch_ns(text):
    dt = datetime.fromisoformat(text.replace('Z', '+00:00'))
    delta = dt - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*1000000000 + delta.microseconds*1000


def pid(global_id):
    return global_id//0x1000000 % 0x1000000


def physical(global_pid, device_id, mapping, gpus):
    key = pid(global_pid), device_id
    require(key in mapping, 'Missing per-process CUDA device mapping')
    uuid = gpus[mapping[key]]
    require(uuid in EXPECTED, 'Nonselected physical GPU activity')
    return uuid


def distribution(values):
    values = sorted(values)
    if not values:
        return dict(count=0, sum_ns=0, p50_ns=None, p95_ns=None, max_ns=None)
    return dict(count=len(values), sum_ns=sum(values), p50_ns=values[len(values)//2],
                p95_ns=values[min(len(values)-1, int(len(values)*.95))], max_ns=values[-1])


def interval_union(intervals):
    merged = []
    for start,end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start,end])
    return merged


def kernel_group(name):
    if 'decode_weights' in name:
        return 'decode_weights'
    if 'pack_bf16_activation' in name:
        return 'pack_bf16_activation'
    if 'bf16_gemmex::split_pair' in name:
        return 'gemmex_split_pair_helper_not_GemmEx_API'
    if any(word in name.lower() for word in ('gemv','gemm','cutlass::kernel','splitkreduce','dot_kernel','reduce_1block_kernel')):
        return 'matrix_math_named_kernels_not_GemmEx_API'
    return 'other'


def add(table, key, duration, size=0):
    row = table.setdefault(key, dict(count=0, sum_ns=0, bytes=0, max_ns=0))
    row['count'] += 1
    row['sum_ns'] += duration
    row['bytes'] += size
    row['max_ns'] = max(row['max_ns'], duration)


def audit(pair):
    case = pair/'runs/P1/session'
    load = lambda path: json.loads(path.read_text())
    dbpath = case/'nsight/nsys.sqlite'
    db = sqlite3.connect(dbpath.resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    strings = {r['id']:r['value'] for r in db.execute('SELECT id,value FROM StringIds')}
    session = db.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    maps = {(r['pid'],r['cudaId']):r['gpuId'] for r in db.execute('SELECT * FROM TARGET_INFO_CUDA_DEVICE')}
    gpus = {r['id']:'GPU-'+r['uuid'].removeprefix('GPU-') for r in db.execute('SELECT id,uuid FROM TARGET_INFO_GPU')}
    model_pid = load(case/'profile/model-identity.json')['pid']
    log = [json.loads(line) for line in (case/'server.log').read_text().splitlines() if line.startswith('{')]
    markers = [r for r in log if r.get('target') == 'ferrule_request']
    requests = {}
    for label in ('warmup','measured'):
        http = load(case/'profile'/f'{label}.json')
        require(http['passed'], 'HTTP canary failed')
        ids = {json.loads(r['data'])['id'] for r in http['raw_sse_events'] if r['data'] != '[DONE]'}
        require(len(ids) == 1, 'Ambiguous HTTP request identity')
        rid = int(next(iter(ids)).rsplit('-',1)[1])
        own = {r['fields']['event']:r for r in markers if r['fields']['request_id'] == rid}
        require(len(own) == 4, 'Incomplete request/HTTP admission and terminal markers')
        start = epoch_ns(own['request_admitted']['timestamp'])-session
        end = epoch_ns(own['request_worker_terminal']['timestamp'])-session
        terminal = own['request_worker_terminal']['fields']
        # Derived emission boundary is a proxy, NOT a prefill/decode kernel marker.
        first = end-(terminal['worker_request_us']-terminal['worker_first_emission_elapsed_us'])*1000
        http_first = http['start_clock']['unix_ns'] + round(http['content_event_times'][0]*1e9)-http['start_clock']['monotonic_ns']-session
        require(0 <= start < first < end, 'Invalid request clock mapping')
        requests[label] = dict(request_id=rid, start_ns=start, first_emission_proxy_ns=first, end_ns=end,
            first_emission_vs_client_receive_ns=http_first-first, marker_events=own,
            phase_status='exact prefill/decode unavailable; before_first_emission and after_first_emission are explicit proxies only',
            window_ns=end-start, bounds_method='server request ID + UTC markers minus official TARGET_INFO_SESSION_START_TIME.utcEpochNs; log resolution 1us',
            kernels={}, kernel_categories={}, memcpy={}, apis={}, osrt={}, per_uuid={},
            phases={phase:dict(kernels={},memcpy={},apis={}) for phase in ('before_first_emission_proxy','after_first_emission_proxy')})

    def window(start,end):
        for label,w in requests.items():
            if w['start_ns'] <= start and end <= w['end_ns']:
                phase='before_first_emission_proxy' if end <= w['first_emission_proxy_ns'] else 'after_first_emission_proxy' if start >= w['first_emission_proxy_ns'] else None
                return label,phase
        return None,None

    activity=[]; totals={}; gpu_intervals=defaultdict(list); correlation_keys=set(); missing=[]
    for kind,table in [('kernel','CUPTI_ACTIVITY_KIND_KERNEL'),('memcpy','CUPTI_ACTIVITY_KIND_MEMCPY'),('memset','CUPTI_ACTIVITY_KIND_MEMSET')]:
        n=0
        for r in db.execute('SELECT * FROM '+table):
            r=dict(r); n+=1
            require(pid(r['globalPid']) == model_pid, 'GPU activity from another process')
            uuid=physical(r['globalPid'],r['deviceId'],maps,gpus)
            label,phase=window(r['start'],r['end'])
            key=(model_pid,r['correlationId']);correlation_keys.add(key)
            activity.append((kind,r,label,phase,uuid,key))
            if label is None:
                continue
            w=requests[label];duration=r['end']-r['start']
            gpu_intervals[label,uuid].append((r['start'],r['end']))
            by=w['per_uuid'].setdefault(uuid,dict(kernel_count=0,kernel_ns=0,memcpy={},memset_count=0))
            if kind == 'kernel':
                name=strings[r['demangledName']]
                add(w['kernels'],name,duration);add(w['kernel_categories'],kernel_group(name),duration)
                by['kernel_count']+=1;by['kernel_ns']+=duration
                if phase: add(w['phases'][phase]['kernels'],name,duration)
            elif kind == 'memcpy':
                direction={1:'H2D',2:'D2H',8:'D2D'}.get(r['copyKind'],'other_'+str(r['copyKind']))
                add(w['memcpy'],direction,duration,r['bytes']);add(by['memcpy'],direction,duration,r['bytes'])
                if phase: add(w['phases'][phase]['memcpy'],direction,duration,r['bytes'])
            else:
                by['memset_count']+=1
        totals[kind]=n

    api_matches=defaultdict(list);returns=Counter();total_api=0
    for r in db.execute('SELECT start,end,globalTid,correlationId,nameId,returnValue FROM CUPTI_ACTIVITY_KIND_RUNTIME'):
        total_api+=1
        name=strings[r['nameId']];key=(pid(r['globalTid']),r['correlationId'])
        if key in correlation_keys:
            api_matches[key].append(dict(r,name=name))
        if r['returnValue']:
            returns[name,str(r['returnValue'])]+=1
        label,phase=window(r['start'],r['end'])
        if label:
            require(pid(r['globalTid']) == model_pid, 'API from another process')
            add(requests[label]['apis'],name,r['end']-r['start'])
            if phase: add(requests[label]['phases'][phase]['apis'],name,r['end']-r['start'])
    totals['runtime_api']=total_api
    launches=defaultdict(list);owner_tids=defaultdict(Counter);correlation_counts=defaultdict(Counter)
    for kind,r,label,phase,uuid,key in activity:
        matches=api_matches.get(key,[])
        matches=[a for a in matches if a['returnValue']==0]
        if label:
            correlation_counts[label][kind+'_observed']+=1
            if len(matches)==1:
                correlation_counts[label][kind+'_correlated']+=1
            else:
                missing.append(dict(kind=kind,request=label,uuid=uuid,correlation_id=r['correlationId'],start_ns=r['start'],matches=len(matches)))
        if kind=='kernel' and label and len(matches)==1:
            a=matches[0];tid=a['globalTid']%0x1000000
            owner_tids[label,uuid][tid]+=1
            launches[label,uuid].append((a['start'],a['end'],r['start']))
    for label,w in requests.items():
        w['observed_activity_correlation']=dict(correlation_counts[label])
        for uuid,by in w['per_uuid'].items():
            intervals=interval_union(gpu_intervals[label,uuid])
            gaps=[b[0]-a[1] for a,b in zip(intervals,intervals[1:])]
            ordered=sorted(launches[label,uuid])
            by.update(owner_launch_tids=dict(owner_tids[label,uuid]),
                observed_gpu_active_union_ns=sum(b-a for a,b in intervals),
                observed_gpu_gaps=distribution(gaps),
                observed_launch_interarrival=distribution([b[0]-a[0] for a,b in zip(ordered,ordered[1:])]),
                observed_launch_api_to_kernel_start=distribution([max(0,g-a) for _,a,g in ordered]),
                gap_caveat='Observed intervals only; missing GPU events inflate gaps; gaps are NOT proven CPU/collective wait')
        w['kernel_sum_ns']=sum(x['sum_ns'] for x in w['kernels'].values())
        w['sync_api_host_duration_including_gpu_wait']={n:v for n,v in w['apis'].items() if 'Synchronize' in n or 'StreamWait' in n}
        w['phase_pure_prefill']=None;w['phase_pure_decode']=None

    thread_cols=[r[1] for r in db.execute('PRAGMA table_info(ThreadNames)')]
    thread_names={}
    for r in db.execute('SELECT * FROM ThreadNames'):
        if pid(r['globalTid'])==model_pid:
            name_id=r[0];thread_names[r['globalTid']%0x1000000]=strings.get(name_id,str(name_id))
    for r in db.execute('SELECT * FROM OSRT_API'):
        if pid(r['globalTid']) != model_pid: continue
        label,phase=window(r['start'],r['end'])
        if label:
            tid=r['globalTid']%0x1000000
            name=strings[r['nameId']]
            owners=[u for u in EXPECTED if tid in owner_tids[label,u]]
            key=str(tid)+':'+thread_names.get(tid,'unknown')+':'+name
            add(requests[label]['osrt'],key,r['end']-r['start'])
    diagnostics=[dict(r) for r in db.execute('SELECT * FROM DIAGNOSTIC_EVENT')]
    failures=[]
    for r in diagnostics:
        if r['severity']==3 or ('Not all CUDA events' in r['text'] and pid(r['globalPid'])==model_pid):
            failures.append(r['text'])
    if missing:failures.append('Uncorrelated observed activity rows: '+str(len(missing)))
    failures.append('No exact request prefill/decode markers; no cublasGemmEx API ranges exported')
    for label,w in requests.items():
        require(set(w['per_uuid'])==EXPECTED and all(v['kernel_count']>0 and {'H2D','D2H'}<=set(v['memcpy']) for v in w['per_uuid'].values()),
                'Missing two-owner kernel/memcpy coverage')
    release=load(case.parent/'probe-lock-release.json')
    require(all(release.get(k) is True for k in ('local_contexts_closed','authenticated_drained','original_b0_verified')), 'Release chain failed')
    result=dict(status='rejected_incomplete_trace', full_trace_accepted=False, benchmark=False, model_starts=1,
        failures=failures, model_pid=model_pid, session_utc_epoch_ns=session, totals=totals,
        gpu_mapping=[dict(pid=p,cudaId=c,gpuId=g,uuid=gpus[g]) for (p,c),g in maps.items()],
        requests=requests, diagnostics=diagnostics, missing_correlations=missing,
        nonzero_api_returns=[dict(api=n,return_value=v,count=c) for (n,v),c in returns.items()],
        cuEventQuery_600='CUDA_ERROR_NOT_READY polling, not execution failure',
        gemmex_api_sum=None, gemmex_reason='Only cuBLAS NVTX domain-create metadata exists; no cuBLAS API ranges. Matrix-named kernel sums are NOT GemmEx API timing.',
        cpu_collective_wait_ns=None, cpu_collective_reason='Owner-thread OSRT waits and launch gaps are visible, but no collective marker/callsite attribution; CUDA host API durations include GPU work and overlap across threads.',
        release=release, raw_sha256={str(p.relative_to(pair)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (dbpath,case/'nsight/nsys.nsys-rep',case/'server.log')})
    db.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair',type=Path)
    args=parser.parse_args()
    result=audit(args.pair.resolve())
    with (args.pair/'profile-audit.json').open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps(dict(status=result['status'],totals=result['totals'],failures=result['failures'],
        requests={k:dict(id=w['request_id'],kernel_sum_ns=w['kernel_sum_ns'],categories=w['kernel_categories'],
            memcpy=w['memcpy'],correlation=w['observed_activity_correlation']) for k,w in result['requests'].items()}),indent=2))


if __name__ == '__main__':
    main()
