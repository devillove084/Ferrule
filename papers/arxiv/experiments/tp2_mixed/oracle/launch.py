#!/usr/bin/env python3
"""Paper custodian. Default preflight is CPU-only, with no model-file reads."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none')
HERE=Path(__file__).resolve().parent
PAPER=HERE.parents[1]
ROOT=HERE.parents[4]
sys.path.insert(0,str(PAPER))
import contract as ct
import launch_guard as guard
import telemetry
sys.path.insert(0,str(ROOT/'scripts'))
import test_vllm_qwen35 as supervisor


def protocol():return ct.load(HERE/'protocol.json')


def pins():
    paths=list(HERE.glob('*.py'))+[HERE/'protocol.json',PAPER/'contract.py',PAPER/'launch_guard.py',
        PAPER/'telemetry.py',PAPER/'runner.py',ROOT/'scripts/test_vllm_qwen35.py',ROOT/'scripts/_owned_subreaper.py',
        ROOT/'scripts/qwen35_fp8_bf16_reference.py',ROOT/'scripts/qwen35_fp8_cuda_reference.py']
    return {str(p):ct.sha(p) for p in sorted(paths)}


def validate_payload(p):
    if p.get('scenario_plan') is not None:
        import multistep
        multistep.validate_plan(p['scenario_plan'],full_model=True)
        ct.require(not p.get('unsharded_reference'),'Historical drift is legacy two-stage only')
    ct.require(p['protocol']==protocol(),'Protocol changed')
    ct.require(p['pins']==pins(),'Workflow pins changed')
    ct.require(Path(p['directory']).resolve().is_relative_to(HERE),'Output must stay inside this paper oracle folder')
    ct.require(p['execute_cuda'] is True,'Worker is export-only')
    telemetry.authorize(p['sole_gpu_owner_granted'],p['gpu_uuid'],p['owner_note'],'device_exclusive_local_custody')
    ct.require(1<=p['timeout']<=1700 and 1<=p['cache_mib']<=1024 and 128<=p['capture_mib']<=2048,'Invalid resource bound')
    ct.require(Path(p['python']).is_file(),'Pinned inference Python missing')
    ct.require(ct.sha(p['python'])==p['python_sha256'],'Inference Python changed')
    ct.require(bool(p.get('unsharded_reference'))==bool(p.get('unsharded_sha256')),'Baseline requires path AND hash')


def worker(launch_hash):
    ct.require_cpu_python()
    p=json.load(sys.stdin)
    authenticated=guard.authenticate(p,launch_hash)
    validate_payload(p)
    ct.validate_owner_lock(p['owner_lock'],p['gpu_uuid'],p['coordinator']['pid'])
    ct.require(os.getpid()==os.getpgrp(),'Worker not in owned session')
    ct.write(Path(p['directory'])/'authentication.json',authenticated)
    env=dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),PYTHONDONTWRITEBYTECODE='1',
        CUDA_VISIBLE_DEVICES=p['gpu_uuid'],NVIDIA_VISIBLE_DEVICES=p['gpu_uuid'],
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        FERRULE_PAPER_WORKER_TOKEN=p['worker_token'],
        TOKENIZERS_PARALLELISM='false',ATEN_CPU_CAPABILITY='avx512',CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1')
    # The numerical child is a descendant of the existing authenticated subreaper;
    # loss/timeout is handled by that helper, never by chasing other owners' PIDs.
    code='import json,sys; sys.path.insert(0,sys.argv[1]); import tp2_oracle; tp2_oracle.export(json.load(sys.stdin),sys.argv[2])'
    result=subprocess.run([p['python'],'-B','-c',code,str(HERE),launch_hash],input=json.dumps(p),text=True,
                          env=env,cwd=ROOT,timeout=p['timeout'])
    return result.returncode


def idle(uuid,note):
    sample=telemetry.gpu_sample('/usr/bin/nvidia-smi',uuid,True,note)
    ct.require(not sample['apps'] and sample['gpu_bytes']==0 and sample['utilization']==0,
               'Granted device is not idle; no launch or foreign signals')
    return sample


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--model',type=Path)
    parser.add_argument('--execute-cuda',action='store_true')
    parser.add_argument('--sole-gpu-owner-granted',action='store_true')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--owner-note',default='')
    parser.add_argument('--python',default='/opt/conda310/bin/python')
    parser.add_argument('--timeout',type=int,default=1700)
    parser.add_argument('--cache-mib',type=int,default=256)
    parser.add_argument('--capture-mib',type=int,default=512)
    parser.add_argument('--unsharded-reference',type=Path)
    parser.add_argument('--unsharded-sha256')
    parser.add_argument('--scenario',choices=('greedy','canary','fork','recreate','cold-owner'))
    parser.add_argument('--decode-steps',type=int,default=4)
    parser.add_argument('--_worker',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    ct.require_cpu_python()
    if args._worker:return worker(args._worker)
    import multistep
    plan=multistep.validate_plan(multistep.build_plan(args.scenario,args.decode_steps),full_model=True) if args.scenario else None
    if not args.execute_cuda:
        record=dict(schema='ferrule.paper.tp2-oracle.preflight.v1',status='cpu_only_not_executed',
            protocol=protocol(),pins=pins(),scenario_plan=plan,full_reference_complete=False,model_files_read=False,gpu_queries=False,
            readiness='CPU contracts are separate; fresh CUDA export and runtime full40 gate remain required')
        if args.output:
            out=args.output.resolve();ct.require(out.is_relative_to(HERE),'Output must stay paper-local')
            out.mkdir(parents=True,exist_ok=False);ct.write(out/'preflight.json',record)
        else:print(json.dumps(record,indent=2))
        return 0
    # Fail BEFORE filesystem creation, weight reads, telemetry, locks or launch.
    telemetry.authorize(args.sole_gpu_owner_granted,args.gpu_uuid,args.owner_note,'device_exclusive_local_custody')
    ct.require(args.output is not None,'Fresh paper-local output required')
    out=args.output.resolve();ct.require(out.is_relative_to(HERE),'Output must stay paper-local')
    p=dict(directory=str(out),model=str((args.model or Path(ct.config(ct.EXCLUSIVE_PATH)['model'])).resolve()),
        scenario_plan=plan,execute_cuda=True,sole_gpu_owner_granted=args.sole_gpu_owner_granted,gpu_uuid=args.gpu_uuid,
        owner_note=args.owner_note,python=args.python,python_sha256=ct.sha(args.python),timeout=args.timeout,
        cache_mib=args.cache_mib,capture_mib=args.capture_mib,protocol=protocol(),pins=pins(),
        unsharded_reference=str(args.unsharded_reference.resolve()) if args.unsharded_reference else None,
        unsharded_sha256=args.unsharded_sha256)
    validate_payload(p)
    # Use the SAME shared paper lock namespace as the existing GPU owner.
    with ct.exclusive_owner_lock(args.gpu_uuid) as lock:
        before=idle(args.gpu_uuid,args.owner_note)
        out.mkdir(parents=True,exist_ok=False)
        ct.write(out/'before.json',before)
        p['owner_lock']=lock
        p,launch_hash=guard.seal(p,out)
        env=dict(PATH='/usr/bin:/bin',HOME=str(Path.home()),CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',
                 PYTHONDONTWRITEBYTECODE='1',FERRULE_PAPER_WORKER_TOKEN=p['worker_token'])
        argv=ct.isolated_argv([sys.executable,'-S','-B',str(HERE/'launch.py'),'--_worker',launch_hash],env)
        result={}
        final=dict(schema='ferrule.paper.tp2-mixed-independent-oracle.v1',profile='SystemMixedTP2EP1',
                   status='incomplete',full_reference_complete=False,strict_ferrule_full40=False)
        try:
            result=supervisor.run_process(argv,args.timeout,p)
            ct.write(out/'supervisor.json',result)
            if ct.drained(result):ct.write(out/'after.json',idle(args.gpu_uuid,args.owner_note))
            ct.require(ct.successful_process(result),'Worker failed/timed out/custody unknown')
            final=ct.load(out/'export.json')
            ct.require(final['status']=='export_complete_pending_custody' and final['boundary_inventory_complete'],
                       'Missing complete numerical export')
            expected_ids=multistep.call_ids(plan) if plan else ['hello','capital']
            ct.require([c['id'] for c in final['cases']]==expected_ids,'Missing required case')
            if plan:
                ct.require(final['suite']['plan']==plan and final['suite']['status']=='numerical_complete','Scenario incomplete')
            ct.require(final['workflow_pins']==pins(),'Source changed after export')
            artifacts = [record for case in final['cases'] for record in (case, case['report'], case['statistics'])]
            artifacts += [final[key] for key in ('source_payloads','operand_layout','numerical_authentication','cuda_small_gate','checkpoint_metadata','interception_after')]
            for record in artifacts:
                path = (out/record['file']).resolve()
                ct.require(path.parent == out and ct.sha(path) == record['sha256'], 'Export artifact changed/escaped')
            final.update(status='complete',full_reference_complete=True,
                         custody=dict(file='supervisor.json',sha256=ct.sha(out/'supervisor.json')),
                         authentication=dict(file='authentication.json',sha256=ct.sha(out/'authentication.json')),
                         launch=dict(file='worker-launch.json',sha256=launch_hash),
                         idle_before=dict(file='before.json',sha256=ct.sha(out/'before.json')),
                         idle_after=dict(file='after.json',sha256=ct.sha(out/'after.json')))
        except BaseException as error:
            if (out/'export.json').is_file():final=ct.load(out/'export.json')
            final.update(status='incomplete',full_reference_complete=False,error=f'{type(error).__name__}: {error}')
            raise
        finally:
            ct.write(out/'manifest.json',final)
            with (out/'manifest.sha256').open('x') as stream:stream.write(ct.sha(out/'manifest.json')+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
