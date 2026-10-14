#!/usr/bin/env python3
"""One maintained TP2 lifecycle, reusing frozen pins/custody/HTTP primitives."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import experiment_v4 as shared
import observer

ct, base, guard, http = shared.ct, shared.base, shared.guard, shared.http
UUIDS, NOTE = shared.UUIDS, shared.NOTE
PRIOR = HERE.parent/'runs/tp2-resident-20261008-004/prepared.json'
PROTOCOL = dict(id='tp2-resident-single-lifecycle', server_starts=1, startup_seconds=900,
                request_seconds=120, all_requests_seconds=440, teardown_seconds=120,
                supervisor_seconds=1600, supervisor_ceiling_seconds=1700,
                phases=['hello2', 'canary8', 'qa8-V1'],
                qa='same-server Japan max8: excluded first QA request, 2 warmups, 5 measured; NOT cold/first-after-server-ready',
                ferrule='TP2 absent; no F slot, pair, order-effect estimate or speedup',
                nonselected_delta_tolerance_bytes=1<<20, valid_sample_deadline_seconds=60,
                nvml='normal 3s/query; one diagnostic pair with 15s total deadline on timeout, in observer thread',
                proc='root PID/starttime/parent bound once; unavailable leaf branches retained as incomplete, PSS null')


def comparison():
    entries = []
    first = ct.load(HERE.parent/'runs/tp2-resident-20261008-001/prepared.json')
    for suffix in ('001','002','003','004'):
        directory = HERE.parent/'runs'/('tp2-resident-20261008-'+suffix)
        p = ct.load(directory/'prepared.json')
        log = (directory/'hello2/server.log').read_text(errors='replace')
        entries.append(dict(run=directory.name, environment_equal=p['spec']['environment']==first['spec']['environment'],
            argv_except_unique_stage_equal=p['spec']['argv'][1:]==first['spec']['argv'][1:],
            executable_hash_equal=p['executable_sha256']==first['executable_sha256'],
            runtime_inventory_equal=p['runtime_pins']==first['runtime_pins'], gpu_uuids=p['gpu_uuids'],
            log_sha256=ct.sha(directory/'hello2/server.log'),
            initialization_evidence=[dict(line=i,text=line) for i,line in enumerate(log.splitlines(),1)
                if any(x in line for x in ('CUDA driver initialization failed','KeyboardInterrupt: terminated',
                    'set_device_index','_cuda_init()', 'world_size=2 rank=', 'NCCL version','MarlinFP8','Using MARLIN'))]))
    return dict(runs=entries, exact_environment=first['spec']['environment'],
                actual_historical_loaded_library_mappings='unavailable; identical installed inventory is not proof of identical dynamic resolution',
                run003='Observer 3s nvidia-smi timeout caused SIGTERM during API-server FLA/Triton CUDA lazy init; terminal exception KeyboardInterrupt: terminated',
                run004='Worker PID150547 failed at gpu_worker.py:379 set_device_index -> torch._C._cuda_init before NCCL. PID150546 reached rank0 distributed init. PSS observer subsequently exhausted ESRCH retries.',
                cuda_numeric_error='Not emitted in historical logs; underlying CUresult/runtime error code unavailable',
                root_cause='Observer-induced abort proven for 003. Underlying driver failure 004 not proven; no env/config/library-input drift detected. One authorized same-config retry, no fallback.')


def prepare(run_id):
    ct.require_cpu_python()
    directory = ct.new_run(run_id)
    prior = ct.load(PRIOR)
    ct.check_pins(prior['runtime_pins'])
    ct.require(ct.inventory(shared.runtime_paths(prior['config'])) == prior['runtime_pins'], 'Prior runtime changed')
    ct.require(ct.model_identity(prior['config'])==prior['model_identity'], 'Prior model changed')
    canaries, refs = shared.references()
    staged = ct.stage_binary(ct.absolute(prior['config']['vllm']), directory)
    spec = shared.spec_for(prior['config'], staged)
    ct.require(spec['environment']==prior['spec']['environment'] and spec['argv'][1:]==prior['spec']['argv'][1:],
               'Inference environment/flags must remain exact')
    ct.write(directory/'comparison.json', comparison())
    ct.write(directory/'canaries.json', canaries)
    pins = ct.inventory(shared.workflow_paths())
    pins[str(PRIOR)] = ct.sha(PRIOR)
    loader = subprocess.run(['/sbin/ldconfig','-p'],capture_output=True,text=True,timeout=5,check=True)
    driver_pins = ct.inventory([Path('/lib64/libcuda.so.1'),Path('/lib64/libnvidia-ml.so.1')])
    ct.write(directory/'driver-loader.json',dict(version=Path('/proc/driver/nvidia/version').read_text(),
        libraries=[line for line in loader.stdout.splitlines() if any(n in line for n in ('libcuda','libnccl','libnvidia-ml'))],
        note='Host cache has CUDA13 default too; server env -i has no LD_LIBRARY_PATH; actual mappings sampled during the single launch'))
    prepared = dict(protocol=PROTOCOL, config=prior['config'], spec=spec, source_pins=pins,
        runtime_pins=prior['runtime_pins'], driver_pins=driver_pins, reference_pins=refs,
        model_identity=prior['model_identity'], executable_sha256=ct.sha(staged),
        canaries_sha256=ct.sha(directory/'canaries.json'), reference_manifest_sha256=shared.references()[1].get(
            str(ct.ROOT/'target/validation/bf16-fresh-quality-20261007/oracle/manifest.json')),
        prior_manifest=str(PRIOR), prior_sha256=ct.sha(PRIOR),
        historical_pins={str(p):ct.sha(p) for p in (HERE.parent/'runs').rglob('*') if p.is_file() and not p.is_relative_to(directory)},
        fairmatch=False, exact_speedup=None)
    prepared['digest'] = ct.digest(prepared)
    ct.write(directory/'prepared.json',prepared)
    print(json.dumps(dict(status='prepared', directory=str(directory), same_inference_config=True,
                         server_starts=1, protocol=PROTOCOL)),flush=True)


def validate(directory):
    p = ct.load(directory/'prepared.json')
    ct.require(ct.digest({k:v for k,v in p.items() if k!='digest'})==p['digest'], 'Prepared changed')
    ct.require(p['protocol']==PROTOCOL,'Protocol differs')
    ct.require({str(f.resolve()) for f in shared.runtime_paths(p['config'])}==set(p['runtime_pins']), 'Runtime coverage differs')
    ct.require({str(f.resolve()) for f in shared.workflow_paths()} | {str(PRIOR)} == set(p['source_pins']), 'Source coverage differs')
    for key in ('source_pins','runtime_pins','driver_pins','reference_pins'):
        ct.check_pins(p[key])
    prior = ct.load(PRIOR)
    ct.require(p['config']==prior['config'] and p['runtime_pins']==prior['runtime_pins'], 'Pinned inference contract differs')
    ct.require(p['spec']==shared.spec_for(p['config'],p['spec']['argv'][0]),'Launch spec differs')
    ct.require(ct.model_identity(p['config'])==p['model_identity'],'Model metadata changed')
    ct.require(ct.sha(directory/'canaries.json')==p['canaries_sha256'],'Canaries changed')
    ct.validate_staged_binary(p['spec']['argv'][0],p['executable_sha256'])
    return p


def requests(case, prepared, monitor):
    result = dict(status='incomplete', phases=[], same_server=True)
    ct.write(case/'canaries.json',ct.load(case.parent/'canaries.json'))
    def alarm(signum, frame):
        raise TimeoutError('Same-lifecycle total request budget 440 seconds exceeded')
    previous = signal.signal(signal.SIGALRM,alarm)
    signal.setitimer(signal.ITIMER_REAL,PROTOCOL['all_requests_seconds'])
    try:
        for stage in PROTOCOL['phases']:
            ct.require(not monitor.failed.is_set(),'Observer coverage/resource gate failed')
            directory=case/stage
            directory.mkdir(exist_ok=False)
            ct.write(directory/'metadata.json',dict(model=prepared['model_identity'],precision=prepared['config']['protocol'],
                topology=dict(tp=2,pp=1,dp=1,ep=1,gpu_uuids=list(UUIDS)),resources=prepared['config']['resources'],
                source=prepared['source_pins'],fairmatch=False,exact_speedup=None,
                lifecycle='same server for Hello2, all canaries and QA',
                qa_first_label='HTTP helper first_after_ready here denotes excluded first QA-phase request, NOT first request after server readiness'))
            gate=shared.request_block(directory,prepared,stage)
            ct.write(directory/'gate.json',gate)
            result['phases'].append(dict(stage=stage,gate=gate))
            print(json.dumps(dict(stage=stage,gate=gate),ensure_ascii=False),flush=True)
            if gate['status']!='pass':
                result['status']='canary_failed'
                return result
        result['status']='complete'
        return result
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        ct.write(case/'requests.json',result)


def worker(seal):
    ct.require_cpu_python()
    payload=json.load(sys.stdin)
    authenticated=guard.authenticate(payload,seal)
    case=Path(payload['directory'])
    p=validate(case.parent)
    shared.validate_locks(payload['owner_locks'],payload['coordinator']['pid'])
    ct.require(ct.sha(case/'baseline.json')==payload['baseline_sha256'],'Baseline changed')
    ct.require(payload['prepared_sha256']==ct.sha(case.parent/'prepared.json'),'Prepared launch binding differs')
    ct.require(os.getpid()==os.getpgrp(),'Worker not in supervised session')
    result=dict(status='failed',started_monotonic=time.monotonic(),fairmatch=False,exact_speedup=None)
    server=monitor=None
    try:
        before=observer.idle(p['config'],UUIDS,NOTE,payload['baseline'])
        ct.write(case/'prelaunch.json',before)
        ct.require(before['status']=='verified','Immediate prelaunch idle missing')
        base.require_free_port(p['spec'])
        with (case/'server.log').open('xb') as log:
            result['launch_monotonic']=time.monotonic()
            with ct.pinned_executable(p['spec']['argv'][0],p['executable_sha256']) as fd:
                server=subprocess.Popen(ct.isolated_argv(base.bound_server_argv(p['spec'],fd),p['spec']['environment']),
                                        pass_fds=(fd,),cwd=ct.ROOT,stdout=log,stderr=subprocess.STDOUT)
            result['server_pid']=server.pid
            monitor=observer.Monitor(case,server,p['config'],authenticated,UUIDS,NOTE,payload['baseline'],p['spec']['environment'].keys())
            result['server_identity']=monitor.root
            monitor.start()
            ready=http.wait_ready(p['spec'],base.StartupGuard(server,monitor),900)
            result['startup_seconds']=ready-result['launch_monotonic']
            result['request_gate']=requests(case,p,monitor)
            result['status']=result['request_gate']['status']
    except BaseException as error:
        result['error']=type(error).__name__+': '+str(error)
        if 'launch_monotonic' in result and 'startup_seconds' not in result:
            result['startup_failure_seconds']=time.monotonic()-result['launch_monotonic']
    finally:
        cleanup_deadline=time.monotonic()+112  # eight seconds reserved for the authenticated custodian
        if monitor:
            result['resources']=monitor.finish()
            if result['resources']['status']!='observed_within_targets' and result['status']=='complete':
                result['status']='resource_evidence_blocked'
        if server:
            try:
                result['shutdown']=http.shutdown_owned_group(server,max(.1,cleanup_deadline-time.monotonic()))
                if result['shutdown']['timed_out']:
                    result['status']='cleanup_timeout'
            except BaseException as error:
                result.update(status='cleanup_unknown',cleanup_error=str(error))
        result['wall_seconds']=time.monotonic()-result['started_monotonic']
        ct.write(case/'case.json',result)
    return 0 if result['status']=='complete' else 1


def run(run_id):
    directory=HERE.parent/'runs'/run_id
    p=validate(directory)
    ct.require(not (directory/'execution.json').exists(),'No resume/retry')
    case=directory/'session'
    case.mkdir(exist_ok=False)
    result=dict(status='incomplete',server_starts=0,fairmatch=False,exact_speedup=None)
    anchor=None
    with shared.owner_locks(UUIDS) as locks:
        try:
            before=observer.idle(p['config'],UUIDS,NOTE)
            ct.write(case/'before.json',before)
            ct.require(before['status']=='verified','Pre-launch idle unavailable')
            anchor=before['baseline']
            ct.write(case/'baseline.json',anchor)
            topology={}
            for name,args in (('matrix',['-m']),('p2p',['-p2p','r'])):
                topology[name]=subprocess.run(['/usr/bin/nvidia-smi','topo',*args],capture_output=True,text=True,timeout=5,check=True).stdout
            ct.write(directory/'topology.json',topology)
            ct.require('CNS' in topology['p2p'],'Declared historical P2P policy evidence missing')
            payload=dict(directory=str(case),owner_locks=locks,baseline=anchor,baseline_sha256=ct.sha(case/'baseline.json'),
                         prepared_sha256=ct.sha(directory/'prepared.json'))
            payload,seal=guard.seal(payload,case)
            env=dict(base.cpu_environment(),FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
            argv=ct.isolated_argv([sys.executable,'-S','-B',str(Path(__file__).resolve()),'--worker',seal],env)
            print('START single lifecycle; no retry',flush=True)
            try:
                receipt=base.supervisor.run_process(argv,1600,payload)
            except BaseException as error:
                receipt=getattr(error,'supervisor_result',{})
                receipt.setdefault('error',str(error))
            ct.write(case/'supervisor.json',receipt)
            result['launch_id']=receipt.get('handshake',{}).get('launch_id')
            result['authenticated_drained']=ct.drained(receipt)
            observation=ct.load(case/'case.json') if (case/'case.json').exists() else dict(status='missing')
            result['observation']=observation
            result['server_starts']=int('server_pid' in observation)
            result['status']=observation['status']
            result['supervisor_success']=ct.successful_process(receipt)
            if (case/'server.log').exists():
                result['runtime']=shared.runtime_evidence((case/'server.log').read_text(errors='replace'))
                result['final_log_sha256']=ct.sha(case/'server.log')
            if not result['authenticated_drained']:
                result['status']='custody_unknown'
        except BaseException as error:
            result['error']=type(error).__name__+': '+str(error)
        finally:
            try:
                result['after']=observer.idle(p['config'],UUIDS,NOTE,anchor)
                ct.write(case/'after.json',result['after'])
                ct.check_pins(p['historical_pins'])
                result['historical_preservation']='all_prior_run_files_unchanged'
                ct.check_pins(p['source_pins'])
                if result['after']['status']!='verified':
                    result['status']='release_unknown'
            except BaseException as error:
                result.update(status='finalization_failed',finalization_error=str(error))
            ct.write(directory/'execution.json',result)
    ct.write(directory/'lock-release.json',dict(released=True,gpu_uuids=list(UUIDS),all_gpu_idle=result.get('after',{}).get('status')=='verified'))
    print(json.dumps(dict(status=result['status'],launch_id=result.get('launch_id'),server_starts=result['server_starts'],
                         authenticated_drained=result.get('authenticated_drained'),after=result.get('after',{}).get('status'))),flush=True)
    return 0 if result['status']=='complete' else 1


if __name__=='__main__':
    ct.require_cpu_python()
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        raise SystemExit(worker(sys.argv[2]))
    cli=argparse.ArgumentParser()
    cli.add_argument('mode',choices=('prepare','run'))
    cli.add_argument('--id',required=True)
    args=cli.parse_args()
    ct.require(__import__('re').fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',args.id),'Unsafe run ID')
    if args.mode=='prepare':
        prepare(args.id)
    else:
        raise SystemExit(run(args.id))
