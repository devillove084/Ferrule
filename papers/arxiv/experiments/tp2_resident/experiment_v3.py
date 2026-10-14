#!/usr/bin/env python3
"""Separate, append-only TP2 resident (not EP2) vLLM paper experiment."""
import argparse
import ast
from contextlib import ExitStack, contextmanager
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import runner as base
import sampling_v3
ct, telemetry, guard, http = base.ct, base.telemetry, base.launch_guard, base.http
UUIDS = ('GPU-b980f22b-58a3-fc88-21c1-ae86053d8535', 'GPU-63f1bd01-ddf9-a742-ee48-9389892493d9')
NOTE = 'User explicitly assigned sole GPU ownership for TP2 resident vLLM; other agent CPU-only; no foreign signals'
PROFILE = 'vllm-tp2-resident-noep-observer-v3-baseline-delta'
HISTORY = ct.ROOT / 'target/validation/qwen35-35b-current/runtime-tp-ep-owner/source-pinned/docs/temp_ferrule_impl.md'
STAGES = ('hello2', 'canary8', 'qa8-V1', 'qa8-V2')
ORDER_BLOCKS = (('F1', 'V1'), ('V2', 'F2'))
PRIOR_PREPARED = HERE.parent / 'runs/tp2-resident-20261008-002/prepared.json'


def config():
    old = ct.config(ct.EXCLUSIVE_PATH)
    flags = dict(old['vllm_flags'])
    for name in ('--offload-backend', '--cpu-offload-params'):
        del flags[name]
    flags.update({'--tensor-parallel-size': '2', '--cpu-offload-gb': '0',
                  '--gpu-memory-utilization': '0.8', '--port': '18367',
                  '--served-model-name': 'qwen35-vllm-tp2-resident-paper'})
    return dict(schema='ferrule.paper.tp2-resident.config.v1', profile=PROFILE,
                model=old['model'], vllm=old['vllm'], vllm_fix_environment=old['vllm_fix_environment'],
                protocol=dict(old['protocol'], protocol_id=PROFILE, tp=2, pp=1, dp=1, ep=1,
                              enable_expert_parallel=False, expert_policy='MoE weights tensor-sharded; NOT TP2EP2'),
                resources=dict(old['resources'], gpu_target_bytes=22*(1 << 30), host_target_bytes=128*(1 << 30),
                               gpu_policy='22 GiB/card sampled ceiling; 0.8*24=19.2 GiB/card vLLM planning; not a partition',
                               host_policy='128 GiB sampled server-tree PSS, predeclared common future Ferrule budget; not total physical host cap',
                               idle_max_device_mib=0, nonselected_gpu_delta_tolerance_bytes=1 << 20),
                limits=dict(startup_seconds=900, request_seconds=120, cleanup_seconds=120,
                            case_seconds=1600, idle_seconds=20, supervisor_ceiling_seconds=1700),
                workloads=dict(hello2=dict(max_tokens=2, requests=1, warmup=0),
                               canary8=dict(max_tokens=8, requests=3, warmup=0),
                               qa8=dict(max_tokens=8, requests=5, warmup=2, first_after_ready=True)),
                vllm_flags=flags, transport=dict(NCCL_P2P_DISABLE='1', NCCL_DEBUG='INFO',
                       NCCL_DEBUG_SUBSYS='INIT,GRAPH,P2P,SHM,NET',
                       policy='Fixed before first launch; historical and live CNS evidence; no retry/transport override'))


def validate_config(c):
    ct.require(c == config(), 'Unknown TP2 resident profile; no overrides or offload/EP fallback')
    return c


@contextmanager
def owner_locks(uuids):
    ct.require(len(uuids) == 2 and len(set(uuids)) == 2, 'Need two distinct physical UUIDs')
    for uuid in uuids:
        telemetry.authorize(True, uuid, NOTE, 'device_exclusive_local_custody')
    # ExitStack releases the first lock if acquisition of the second fails.
    with ExitStack() as stack:
        yield {uuid: stack.enter_context(ct.exclusive_owner_lock(uuid)) for uuid in sorted(uuids)}


def validate_locks(locks, coordinator):
    ct.require(list(locks) == sorted(UUIDS), 'Missing/unsorted two-UUID lock set')
    for uuid in sorted(UUIDS):
        ct.validate_owner_lock(locks[uuid], uuid, coordinator)


def references():
    path = HERE.parent / 'runs/g10-exclusive-001/g10.py'
    spec = importlib.util.spec_from_file_location('historical_g10_reference_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, pins = module.references()
    pins[str(path)] = ct.sha(path)
    return rows, pins


def audit(c):
    site = ct.absolute(c['vllm']).parent.parent / 'lib/python3.12/site-packages'
    root = site / 'vllm'
    sources = {p: ast.parse((root / p).read_text()) for p in
               ('engine/arg_utils.py', 'entrypoints/openai/cli_args.py')}
    declarations = {n.value for tree in sources.values() for n in ast.walk(tree)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith('--')}
    fields = {n.target.id: ast.unparse(n.annotation) for tree in sources.values() for n in ast.walk(tree)
              if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)}
    evidence = {}
    for flag, value in c['vllm_flags'].items():
        positive = '--' + flag[5:] if flag.startswith('--no-') else flag
        field = positive[2:].replace('-', '_')
        ct.require(positive in declarations or flag in ('--host', '--port'), 'CLI missing: ' + flag)
        annotation = fields.get(field, '')
        kind = ct.VLLM_EXPECTED_FLAGS[flag]
        if kind == 'enum':
            source, alias = base.cli_audit.ENUMS[flag]
            ct.require(value in base.cli_audit.literal_choices(ast.parse((root / source).read_text()), alias), 'Enum mismatch')
        elif kind == 'bool':
            ct.require(value is True and 'bool' in annotation, 'Invalid bool')
        elif kind == 'int':
            ct.require('int' in annotation and isinstance(value, str) and value.isdigit(), 'Invalid integer')
        elif kind == 'float':
            ct.require(annotation == 'float' and float(value) >= 0, 'Invalid float')
        else:
            ct.require('str' in annotation and isinstance(value, str), 'Invalid string')
        evidence[flag] = dict(annotation=annotation, value=value)
    offload = (root / 'config/offload.py').read_text()
    parallel = (root / 'config/parallel.py').read_text()
    selection = (root / 'model_executor/offloader/base.py').read_text()
    for literal in ('cpu_offload_gb: float = Field(default=0, ge=0)',
                    'offload_group_size: int = Field(default=0, ge=0)',
                    'offload_backend: OffloadBackend = "auto"'):
        ct.require(literal in offload, 'Offload default changed: ' + literal)
    ct.require('enable_expert_parallel: bool = False' in parallel, 'EP default changed')
    ct.require('elif uva.cpu_offload_gb > 0:' in selection and 'return NoopOffloader()' in selection,
               'Offloader selection changed')
    versions = {}
    for package, version in (('vllm', '0.28.0+cu129'), ('torch', '2.13.0+cu129')):
        metadata = site / (package + '-' + version + '.dist-info/METADATA')
        ct.require(('Version: ' + version + '\n') in metadata.read_text(), 'Runtime version mismatch')
        versions[package] = version
    return dict(status='pass_static_not_device_proof', flags=evidence, versions=versions,
                no_offload='Explicit CPU offload 0; auto backend + prefetch group0 defaults select NoopOffloader',
                expert_parallel='Default false; no --enable-expert-parallel; TP2EP2 requires a separate profile')


def runtime_paths(c):
    site = ct.absolute(c['vllm']).parent.parent / 'lib/python3.12/site-packages'
    paths = [p for p in site.rglob('*') if p.is_file() and
             (p.suffix in ('.py', '.pth', '.so') or '.so.' in p.name or p.name in ('METADATA', 'RECORD'))]
    ct.require(paths, 'Runtime inventory empty')
    fix = ct.validate_vllm_fix_env(c)
    cuda = Path(fix['CUDA_HOME'])
    paths += [p for p in cuda.rglob('*') if p.is_file()]
    paths += [ct.absolute(c['vllm']), ct.absolute(c['vllm']).parent / 'python',
              ct.absolute(c['vllm_fix_environment']), Path('/usr/bin/nvidia-smi'),
              Path('/usr/bin/env'), Path(sys.executable)]
    paths += list(Path('/home/admin/wxw403883/anaconda3/lib/python3.12/site-packages').glob('*.pth'))
    return paths


def workflow_paths():
    return [*sorted(HERE.glob('*.py')),
            *[HERE.parent / n for n in ('runner.py', 'contract.py', 'telemetry.py', 'launch_guard.py', 'cli_audit.py')],
            *[ct.ROOT / 'scripts' / n for n in ('bench_ferrule_vllm.py', 'test_vllm_qwen35.py', '_owned_subreaper.py')]]


def spec_for(c, staged):
    env = dict(HOME='/home/admin', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               CUDA_VISIBLE_DEVICES=','.join(UUIDS), NVIDIA_VISIBLE_DEVICES=','.join(UUIDS),
               CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.update(ct.validate_vllm_fix_env(c))
    env.update({k: v for k, v in c['transport'].items() if k.startswith('NCCL_')})
    ct.require(not any(k in env for k in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUBLAS_WORKSPACE_CONFIG')),
               'Inherited Ferrule runtime pollution')
    flags = c['vllm_flags']
    return dict(engine='vllm', argv=[staged, 'serve', c['model'], *ct.flags_argv(flags)], environment=env,
                model=flags['--served-model-name'], url='http://127.0.0.1:'+flags['--port']+'/v1/completions')


def prior_runtime_binding(c):
    prior = ct.load(PRIOR_PREPARED)
    ct.check_pins(prior['runtime_pins'])
    ct.require(c['vllm_flags'] == prior['config']['vllm_flags']
               and c['transport'] == prior['config']['transport'], 'Pinned inference/transport changed')
    ct.require(ct.model_identity(c) == prior['model_identity'], 'Prior model identity changed')
    ct.require(ct.inventory(runtime_paths(c)) == prior['runtime_pins'], 'Prior runtime inventory changed')
    return dict(path=str(PRIOR_PREPARED), sha256=ct.sha(PRIOR_PREPARED),
                runtime_inventory_sha256=ct.digest(prior['runtime_pins']), inference_and_transport_unchanged=True)


def prepare(run_id):
    ct.require_cpu_python()
    c = validate_config(config())
    directory = ct.new_run(run_id)
    rows, refs = references()
    ct.write(directory / 'canaries.json', rows)
    ct.write(directory / 'config.json', c)
    history = HISTORY.read_text()
    ct.require('NCCL_P2P_DISABLE=1' in history and 'CNS' in history, 'Historical P2P evidence missing')
    # Pin frozen historical source, not the other agent's changing current docs.
    refs[str(HISTORY)] = ct.sha(HISTORY)
    staged = ct.stage_binary(ct.absolute(c['vllm']), directory)
    print('Hashing installed runtime/source/JIT inventories (CPU only)', flush=True)
    p = dict(schema='ferrule.paper.tp2-resident.prepared.v1', profile=PROFILE, config=c,
             config_sha256=ct.digest(c), gpu_uuids=list(UUIDS), gpu_indices=[0, 1], owner_note=NOTE,
             workflow_pins=ct.inventory(workflow_paths()), runtime_pins=ct.inventory(runtime_paths(c)),
             reference_pins=refs, model_identity=ct.model_identity(c), cli_audit=audit(c),
             staged=staged, executable_sha256=ct.sha(staged), spec=spec_for(c, staged),
             baseline_preservation_pins={str(f):ct.sha(f) for f in (HERE.parent / 'runs').rglob('*')
                                        if f.is_file() and not f.is_relative_to(directory)},
             canaries_sha256=ct.sha(directory / 'canaries.json'), fairmatch=False, exact_speedup=None,
             inventory_limitations=['Checkpoint tensor payloads NOT hashed; metadata + shard stat identity only',
                 'Installed Python/shared-library and CUDA JIT input inventory; dynamically generated cache/driver closure unknown',
                 'No Ferrule binary executed or pinned; concurrent Ferrule CPU work is outside this vLLM profile'],
             host_budget_common_future_ferrule_bytes=128*(1 << 30),
             reference_limitations='Original HF-derived manifest uses seed0, ignore_eos=true and canonical tie policy; this limited text-prefix canary uses greedy seed1 natural EOS, not full vLLM numerical equivalence',
             stages=list(STAGES), no_retry=True, no_transport_fallback=True, no_foreign_kill=True,
             observer_revision='v3 per-UUID pre-launch baseline + 1 MiB nonselected positive memory delta tolerance; v2 ESRCH unchanged',
             prior_observer_run='tp2-resident-20261008-002',
             order_blocks=ORDER_BLOCKS, absent_engine='Ferrule TP2 Mixed; F1/F2 not run, never a paired block',
             prior_runtime_binding=prior_runtime_binding(c))
    p['prepared_sha256'] = ct.digest(p)
    ct.write(directory / 'prepared.json', p)
    print(json.dumps(dict(status='prepared', directory=str(directory), flags=c['vllm_flags'], canaries=rows), ensure_ascii=False), flush=True)


def validate(directory):
    p = ct.load(directory / 'prepared.json')
    ct.require(ct.digest({k:v for k,v in p.items() if k != 'prepared_sha256'}) == p['prepared_sha256'], 'Prepared changed')
    c = validate_config(p['config'])
    ct.require(c == ct.load(directory / 'config.json') and ct.digest(c) == p['config_sha256'], 'Config changed')
    for key, paths in (('workflow_pins', workflow_paths()), ('runtime_pins', runtime_paths(c))):
        ct.require(set(p[key]) == {str(f.resolve()) for f in paths}, 'Missing mandatory pin coverage: '+key)
        ct.check_pins(p[key])
    ct.check_pins(p['reference_pins'])
    ct.require(ct.model_identity(c) == p['model_identity'], 'Model identity changed')
    ct.require(ct.sha(directory / 'canaries.json') == p['canaries_sha256'], 'Canaries changed')
    ct.require(ct.sha(PRIOR_PREPARED) == p['prior_runtime_binding']['sha256'], 'Prior prepared receipt changed')
    prior = ct.load(PRIOR_PREPARED)
    ct.require(p['runtime_pins'] == prior['runtime_pins'] and c['vllm_flags'] == prior['config']['vllm_flags']
               and c['transport'] == prior['config']['transport'], 'Prior pinned runtime/inference/transport differs')
    ct.validate_staged_binary(p['staged'], p['executable_sha256'])
    ct.require(spec_for(c, p['staged']) == p['spec'], 'Spec changed')
    return p


class Monitor(sampling_v3.Monitor):
    def __init__(self, directory, server, config, authenticated, baseline):
        super().__init__(directory, server, config, authenticated, UUIDS, NOTE, baseline)


def request_block(directory, p, stage):
    c, spec = p['config'], p['spec']
    workload = 'qa8' if stage.startswith('qa8-') else stage
    w = c['workloads'][workload]
    refs = ct.load(directory.parent / 'canaries.json')
    args = ['--engine','vllm','--url',spec['url'],'--model',spec['model'],'--seed','1',
            '--max-tokens',str(w['max_tokens']),'--concurrency','1','--requests',str(w['requests']),
            '--warmup',str(w['warmup']),'--repeat','1','--timeout','120', '--window-kind','fixed-work',
            '--measurement-seconds','600','--metadata',str(directory / 'metadata.json'),
            '--output',str(directory / 'http.json')]
    if stage == 'hello2':
        args += ['--prompt','Hello','--input-tokens','1']
    else:
        corpus = [{k:r[k] for k in ('id','prompt','input_tokens')} for r in (refs if stage == 'canary8' else refs[:1])]
        ct.write(directory / 'prompts.json', corpus)
        args += ['--prompts-json',str(directory / 'prompts.json')]
    if workload == 'qa8':
        args += ['--first-after-ready']
    code = http.main(args)
    report = ct.load(directory / 'http.json')
    rows = [r for run in report['runs'] for r in run['requests']]
    all_rows = rows + report['warmups'] + ([report['first_after_ready']] if 'first_after_ready' in report else [])
    ct.require(code == 0 and report['status'] == 'complete' and len(rows) == w['requests']
               and len(report['warmups']) == w['warmup'], 'HTTP incomplete')
    ct.require(all(r['status'] == 'success' and r['input_length_matches'] is True
                   and type(r['actual_output_tokens']) is int and 0 < r['actual_output_tokens'] <= w['max_tokens']
                   and r['finish_reason'] in ('length','stop') for r in all_rows), 'HTTP status/token usage invalid')
    if stage != 'canary8':
        return dict(status='pass', quality_claim=False)
    checks = [dict(id=ref['id'], expected_text=ref['reference_text'], observed_text=row['text'],
                   reference_ids=ref['reference_ids'], observed_ids=None,
                   actual_output_tokens=row['actual_output_tokens'], actual_input_tokens=row['actual_input_tokens'],
                   passed=row['text'] == ref['reference_text'] and row['actual_output_tokens'] == len(ref['reference_ids']))
              for row,ref in zip(rows,refs,strict=True)]
    return dict(status='pass' if all(r['passed'] for r in checks) else 'fail', checks=checks,
                limitation='Exact visible first8 text + usage only; not observed token IDs/logits/routes or full numerical oracle')


def runtime_evidence(log):
    patterns = dict(linear_marlin=r'[^\n]*MarlinFP8ScaledMMLinearKernel[^\n]*',
                    moe_marlin=r'[^\n]*Using MARLIN Fp8 MoE backend[^\n]*',
                    model_load=r'[^\n]*(?:Model loading took|Loading model weights took)[^\n]*',
                    offload=r'[^\n]*(?:Offloader set to|CPU offloaded parameters)[^\n]*',
                    topology=r'[^\n]*(?:tensor_parallel_size=2|world_size=2|TP rank)[^\n]*',
                    gdn=r'[^\n]*(?:GDN|gated_delta|gated delta|FusedGDN)[^\n]*',
                    transport=r'[^\n]*(?:NCCL|via SHM|via NET|P2P|Socket|IPC)[^\n]*',
                    transport_channels=r'[^\n]*(?:via SHM|via NET|via P2P|Custom allreduce is disabled|NCCL_P2P_DISABLE set)[^\n]*',
                    cache_dtype=r'[^\n]*(?:mamba_cache_dtype|mamba_ssm_cache_dtype|dtype=torch.bfloat16|kv_cache_dtype)[^\n]*')
    found = {key:re.findall(pattern,log,re.I)[:100] for key,pattern in patterns.items()}
    contradictions = re.findall(r'[^\n]*(?:Offloader set to (?:UVA|Prefetch)Offloader|Total CPU offloaded parameters: [1-9])[^\n]*',log)
    return dict(evidence=found, contradictions=contradictions,
                no_cpu_offload='source-default/config verified; Noop selection may be debug-only; residency telemetry/logs retained',
                exact_arithmetic_match=False, effective_full_dtype_and_kernel_closure='unknown; logged operations only')


def worker(launch_hash):
    ct.require_cpu_python()
    payload = json.load(sys.stdin)
    authenticated = guard.authenticate(payload, launch_hash)
    directory = Path(payload['directory'])
    p = validate(directory.parent)
    validate_locks(payload['owner_locks'], payload['coordinator']['pid'])
    ct.require(payload['prepared_sha256'] == ct.sha(directory.parent / 'prepared.json'), 'Launch prepared differs')
    baseline_receipt = ct.load(directory / 'baseline.json')
    ct.require(payload['baseline'] == baseline_receipt and ct.sha(directory / 'baseline.json') == payload['baseline_sha256'],
               'Pre-launch baseline changed')
    sampling_v3.device_deltas(dict(devices=list(baseline_receipt.values()), apps=[]), baseline_receipt, UUIDS)
    stage = payload['stage']
    ct.require(stage in STAGES and directory.name == stage, 'Unknown stage')
    ct.require(os.getpid() == os.getpgrp(), 'Not in authenticated worker session')
    c, spec = p['config'], p['spec']
    result = dict(status='failed', started_monotonic=time.monotonic(), fairmatch=False, exact_speedup=None)
    server, monitor = None, None
    try:
        ct.write(directory / 'metadata.json', dict(source=p['workflow_pins'], model=p['model_identity'],
                 binary=dict(path=p['staged'], sha256=p['executable_sha256']), dependencies=p['cli_audit']['versions'],
                 precision=dict(declared=c['protocol'], effective='logged operations, not full arithmetic verification'),
                 topology=dict(tp=2,pp=1,dp=1,ep=1,enable_expert_parallel=False,physical_gpu_uuids=list(UUIDS)),
                 resources=c['resources'], cache=dict(prefix_mode='off',state_reset='per_request',os_cache='uncontrolled'),
                 runtime_contract=dict(identity_mode='device_exclusive_local_custody',owned_gpu_attribution='unknown'),
                 quality_gate=dict(status='independent_limited_canary_required'), fairmatch=False,exact_speedup=None))
        base.require_free_port(spec)
        prelaunch = idle(c, payload['baseline'])
        ct.write(directory / 'prelaunch-recheck.json', prelaunch)
        ct.require(prelaunch['status'] == 'verified', 'Pre-launch baseline delta/PID idle check failed')
        with (directory / 'server.log').open('xb') as log:
            result['server_launch_monotonic'] = time.monotonic()
            with ct.pinned_executable(spec['argv'][0], p['executable_sha256']) as fd:
                server = subprocess.Popen(ct.isolated_argv(base.bound_server_argv(spec, fd), spec['environment']),
                                          pass_fds=(fd,), cwd=ct.ROOT, stdout=log, stderr=subprocess.STDOUT)
            result['server_pid'] = server.pid
            monitor = Monitor(directory, server, c, authenticated, payload['baseline'])
            monitor.start()
            ready = http.wait_ready(spec, base.StartupGuard(server, monitor), 900)
            result['startup_seconds'] = ready - result['server_launch_monotonic']
            ct.require(not monitor.is_set(), 'Resource monitor failed')
            result['request_gate'] = request_block(directory, p, stage)
            ct.require(not monitor.is_set(), 'Resource monitor failed during requests')
            result['status'] = 'complete' if result['request_gate']['status'] == 'pass' else 'canary_failed'
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        if 'server_launch_monotonic' in result and 'startup_seconds' not in result:
            result['startup_failure_seconds'] = time.monotonic() - result['server_launch_monotonic']
    finally:
        if monitor:
            result['resources'] = monitor.finish()
            if result['resources']['status'] != 'observed_within_targets' and result['status'] == 'complete':
                result['status'] = 'resource_failed'
        if server:
            try:
                result['shutdown'] = http.shutdown_owned_group(server, 112)
                if result['shutdown']['timed_out']:
                    result['status'] = 'cleanup_timeout'
            except BaseException as error:
                result.update(status='cleanup_unknown', cleanup_error=str(error))
        log = directory / 'server.log'
        if log.exists():
            result['runtime'] = runtime_evidence(log.read_text(errors='replace'))
            result['log_sha256'] = ct.sha(log)
            if result['runtime']['contradictions']:
                result['status'] = 'runtime_contradiction'
        result['wall_seconds'] = time.monotonic() - result['started_monotonic']
        ct.write(directory / 'case.json', result)
    return 0 if result['status'] == 'complete' else 1


def idle(c, anchor=None):
    return sampling_v3.idle(c, UUIDS, NOTE, anchor)


def run_case(directory, stage, p, locks, anchor):
    case = directory / stage
    case.mkdir(exist_ok=False)
    before = idle(p['config'], anchor)
    ct.write(case / 'before.json', before)
    ct.require(before['status'] == 'verified', 'All-device idle gate failed')
    baseline = before['baseline']
    ct.write(case / 'baseline.json', baseline)
    payload = dict(directory=str(case), stage=stage, prepared_sha256=ct.sha(directory / 'prepared.json'),
                   owner_locks=locks, baseline=baseline, baseline_sha256=ct.sha(case / 'baseline.json'))
    payload, launch_hash = guard.seal(payload, case)
    env = dict(base.cpu_environment(), FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
    argv = ct.isolated_argv([sys.executable,'-S','-B',str(Path(__file__).resolve()),'--worker',launch_hash],env)
    print('START '+stage, flush=True)
    try:
        receipt = base.supervisor.run_process(argv, 1600, payload)
    except BaseException as error:
        receipt = getattr(error,'supervisor_result',{})
        receipt.setdefault('error',str(error))
    ct.write(case / 'supervisor.json',receipt)
    after = idle(p['config'], anchor)
    after['case_baseline_deltas'] = [sampling_v3.device_deltas(row, baseline, UUIDS) for row in after['observations']]
    ct.write(case / 'after.json',after)
    observation = ct.load(case / 'case.json') if (case / 'case.json').exists() else dict(status='missing')
    assessed = dict(stage=stage, observation=observation, launch_id=receipt.get('handshake',{}).get('launch_id'),
                    drained=ct.drained(receipt), all_gpu_idle=after['status']=='verified',
                    eligible=ct.successful_process(receipt) and observation['status']=='complete'
                        and observation.get('resources',{}).get('status')=='observed_within_targets' and after['status']=='verified')
    if (case / 'server.log').exists():
        assessed['final_drained_server_log_sha256'] = ct.sha(case / 'server.log')
        assessed['final_runtime'] = runtime_evidence((case / 'server.log').read_text(errors='replace'))
    ct.write(case / 'assessment.json',assessed)
    print(json.dumps({k:v for k,v in assessed.items() if k not in ('observation', 'final_runtime')} |
                     {'status':observation['status'], 'startup_seconds':observation.get('startup_seconds'),
                      'error':observation.get('error'), 'request_gate':observation.get('request_gate')}),flush=True)
    ct.require(assessed['drained'] and assessed['all_gpu_idle'], 'Fail-stop: cleanup/idle unknown')
    return assessed


def run(run_id):
    ct.require_cpu_python()
    ct.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',run_id or ''), 'Unsafe run ID')
    directory = HERE.parent / 'runs' / run_id
    ct.require(not (directory / 'execution.json').exists() and not (directory / 'hello2').exists(), 'Run already started; no resume')
    p = validate(directory)
    report = dict(schema='ferrule.paper.tp2-resident.execution.v1', status='incomplete', cases=[],
                  profile=PROFILE, fairmatch=False, exact_speedup=None, performance_status='not_run',
                  order_blocks=[dict(order=list(block), ferrule_status='not_run_TP2_absent', vllm_status='not_run_gate_required')
                                for block in ORDER_BLOCKS],
                  identity_mode='device_exclusive_local_custody', owned_gpu_attribution='unknown',
                  owned_process_gpu_peak_bytes=None, absence_of_foreign_interference_proven=False,
                  performance_basis='exclusive_operator_allocation_assumption')
    with owner_locks(UUIDS) as locks:
        report['owner_locks'] = locks
        anchor = None
        try:
            initial = idle(p['config'])
            ct.write(directory / 'initial-baseline.json', initial)
            ct.require(initial['status'] == 'verified', 'Initial idle baseline unavailable')
            anchor = initial['baseline']
            topology = {}
            for key, args in (('matrix',['-m']),('p2p_read',['-p2p','r'])):
                query = subprocess.run(['/usr/bin/nvidia-smi','topo',*args],capture_output=True,text=True,timeout=5,check=True)
                topology[key] = query.stdout
            ct.write(directory / 'topology.json',topology)
            ct.require('CNS' in topology['p2p_read'], 'Declared P2P-disabled evidence missing')
            for stage in STAGES:
                row = run_case(directory,stage,p,locks,anchor)
                report['cases'].append(row)
                ct.require(row['eligible'], stage+' failed; subsequent stages blocked, no retry')
                if stage.startswith('qa8-'):
                    report['order_blocks'][0 if stage.endswith('V1') else 1]['vllm_status'] = 'complete_descriptive_only'
            report.update(status='complete',performance_status='complete_descriptive_only')
        except BaseException as error:
            report['error'] = type(error).__name__ + ': ' + str(error)
        finally:
            try:
                report['final_release'] = idle(p['config'], anchor)
                ct.write(directory / 'final-release.json',report['final_release'])
                ct.check_pins(p['baseline_preservation_pins'])
                report['baseline_preservation'] = 'all_prior_run_files_unchanged'
                if report['final_release']['status'] != 'verified':
                    report['status'] = 'cleanup_unknown'
            except BaseException as error:
                report.update(status='failed',finalization_error=str(error))
            ct.write(directory / 'execution.json',report)
    ct.write(directory / 'lock-release.json',dict(status='released',gpu_uuids=list(UUIDS),
             local_locks_released=True, all_gpu_idle=report.get('final_release',{}).get('status')=='verified'))
    print(json.dumps(dict(status=report['status'],error=report.get('error'),performance_status=report['performance_status'],
                         final_release=report.get('final_release',{}).get('status'))),flush=True)
    return 0 if report['status']=='complete' else 1


if __name__ == '__main__':
    ct.require_cpu_python()
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        raise SystemExit(worker(sys.argv[2]))
    cli=argparse.ArgumentParser()
    cli.add_argument('mode',choices=('prepare','run'))
    cli.add_argument('--id',required=True)
    args=cli.parse_args()
    if args.mode=='prepare':
        prepare(args.id)
    else:
        raise SystemExit(run(args.id))
