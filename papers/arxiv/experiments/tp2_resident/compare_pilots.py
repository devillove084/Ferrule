#!/usr/bin/env python3
"""Offline, descriptive old/new paired receipts; never launch or validate old live sources."""
import argparse
from pathlib import Path
import paired as p

ct = p.ct
OLD = p.HERE/'runs/tp2-fv-qa8-20261009-001'


def same_scope(old, new):
    ct.require(old['protocol'] == new['protocol'] == p.PROTOCOL
        and old['purpose'] == new['purpose'] == 'paired'
        and old['order'] == new['order'] == list(p.SLOTS)
        and old['config'] == new['config'], 'Different workload/order/config; no comparison')
    for engine in ('ferrule','vllm'):
        a,b=(x['admission']['engines'][engine] for x in (old,new))
        for key in ('topology','boundaries','workload','resources','limits','model','cache'):
            ct.require(a[key]==b[key], 'Different paired scope: '+engine+'/'+key)
        ct.require(a['spec']['argv'][1:]==b['spec']['argv'][1:]
            and a['spec']['environment']==b['spec']['environment']
            and a['runtime_pins']==b['runtime_pins'], 'Inference parameters/environment/runtime changed: '+engine)
        if engine=='vllm':
            ct.require(a['binary']==b['binary'] and a['source_pins']==b['source_pins'], 'vLLM baseline changed')


def compare(new_directory):
    new_directory=new_directory.resolve()
    ct.require(new_directory.is_relative_to(p.HERE/'runs') and new_directory != OLD, 'Fresh paper pilot required')
    old_prepared,new_prepared=(ct.load(d/'prepared.json') for d in (OLD,new_directory))
    same_scope(old_prepared,new_prepared)
    historical=ct.load(OLD/'serving-pilot-summary.json')
    pins={str(OLD/'serving-pilot-summary.json'):ct.sha(OLD/'serving-pilot-summary.json')}
    populations={}
    for name,directory in (('old',OLD),('optimized',new_directory)):
        rows=[]
        def read(path):
            digest=ct.sha(path)
            if name=='old':
                ct.require(historical['artifact_pins'].get(str(path))==digest,'Historical receipt changed: '+str(path))
            pins[str(path)]=digest
            return ct.load(path)
        read(directory/'prepared.json')
        ct.require(read(directory/'execution.json')['status']=='complete_descriptive_only','Pilot failed/incomplete')
        for slot in p.SLOTS:
            cell=directory/'runs'/slot
            result=read(cell/'execution.json')
            supervisor=read(cell/'session/supervisor.json')
            release=read(cell/'lock-release.json')
            ct.require(result['status']=='complete' and ct.successful_process(supervisor)
                and result['authenticated_drained'] is True and result['after']['status']=='verified'
                and release['released'] is True and release['all_gpu_idle'] is True, 'Pilot custody/release failure')
            for phase in ('hello2','canary8','qa8-V1'):
                ct.require(read(cell/'session'/phase/'gate.json')['status']=='pass','Canary or QA failed')
            report=read(cell/'session/qa8-V1/http.json')
            requests=[row for run in report['runs'] for row in run['requests']]
            ct.require(report['status']=='complete' and len(requests)==5 and len(report['warmups'])==2
                and report['first_after_ready']['status']=='success'
                and all(row['status']=='success' and row['actual_output_tokens']==8 for row in requests),
                'QA population changed')
            case=read(cell/'session/case.json');resources=case['resources']
            log_path=cell/'session/server.log';digest=ct.sha(log_path);pins[str(log_path)]=digest
            if name=='old':ct.require(historical['artifact_pins'][str(log_path)]==digest,'Old log changed')
            text=log_path.read_text(errors='replace')
            if slot.startswith('F'):
                import json
                reserves=[json.loads(line)['fields'] for line in text.splitlines() if 'Mixed optional diagnostic reserves' in line]
                ct.require(len(reserves)==1 and reserves[0]['trace_host']==0 and reserves[0]['trace_device_per_owner']==0
                    and 'request_worker_terminal' not in text, 'Ferrule diagnostic trace/debug active')
            else:
                ct.require('collect_detailed_traces=None' in text and 'enable_layerwise_nvtx_tracing=False' in text,
                           'vLLM tracing unexpectedly enabled')
            rows.append(dict(slot=slot,launch_id=result['launch_id'],startup_seconds=case['startup_seconds'],
                summary=report['runs'][0]['summary'],actual_measured_tokens=sum(row['actual_output_tokens'] for row in requests),
                requests=[{key:row.get(key) for key in ('status','actual_output_tokens','ttft_seconds','request_e2e_seconds')}
                          for row in requests],
                gpu_peak_bytes=resources['gpu_peak_bytes'],pss_peak_bytes=resources['tree_pss_peak_bytes'],
                phase_memory=resources.get('phase_memory'),memory_components=resources.get('memory_components')))
        populations[name]=rows
    result=dict(status='descriptive_only',old_release_sha256=old_prepared['admission']['engines']['ferrule']['binary']['sha256'],
        optimized_release_sha256=new_prepared['admission']['engines']['ferrule']['binary']['sha256'],
        populations=populations,artifact_pins=pins,fairmatch=False,exact_speedup=None,e2_complete=False,
        limitations=['Five same-server measured requests per slot; no statistics, parity, optimality, extrapolation or causal attribution',
            'Historical sources/scripts are not revalidated against current worktree; consumed raw receipts checked against original summary pins',
            'Old phase smaps absent is null, not zero; after-first samples are not proof of steady state',
            'CPU collective vs NCCL; Mixed/HF-eager reference vs Marlin; effective decodebatch1 vs requested seq2',
            'OS/JIT/page cache uncontrolled; profiler and correctness durations excluded'])
    ct.write(new_directory/'old-vs-optimized.json',result)
    return result


def main():
    ct.require_cpu_python()
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--new',required=True,type=Path)
    args=cli.parse_args();compare(args.new);print(args.new/'old-vs-optimized.json')


if __name__=='__main__':main()
