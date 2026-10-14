#!/usr/bin/env python3
"""Paper-only M7 orchestration. Default plan is CPU-only; never installs packages."""
import argparse
import ast
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import contract as ct
import cli_audit
import launch_guard
import telemetry

DIAGNOSTIC_LOG = 'info,ferrule_performance=debug'

sys.path.insert(0, str(ct.ROOT / 'scripts'))
import bench_ferrule_vllm as http
import test_vllm_qwen35 as supervisor


def cpu_environment():
    return dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), CUDA_VISIBLE_DEVICES='',
                NVIDIA_VISIBLE_DEVICES='none', PYTHONDONTWRITEBYTECODE='1')


def static_cli_audit(c):
    return cli_audit.audit(c)


def quality_history(c):
    path = ct.absolute(c['quality_history'])
    data = ct.load(path)
    ct.require(data['status'] == 'PASS_mixed_prepared_attention_original_full40_and_fresh_quality',
               'Authoritative historical quality result changed')
    return dict(path=str(path), sha256=ct.sha(path), status=data['status'],
                full40=data.get('full40'), fresh_quality=data.get('fresh_quality'),
                cold_qa_restart=data.get('cold_qa_restart'),
                applicability='historical numerical PASS; subsequent composition-only refactor reported numerically unchanged; new release basic quality still required')


def plan(c):
    clock = time.get_clock_info('monotonic')
    return dict(schema=ct.SCHEMA + 'plan.v1', status='cpu_only_not_executed', config=c,
                config_sha256=ct.digest(c), protocol_sha256=ct.digest(c['protocol']),
                cli_audit=static_cli_audit(c), quality_history=quality_history(c),
                model_identity=ct.model_identity(c),
                build=dict(argv=ct.build_argv(c), environment=ct.build_environment(c),
                           execute_requires='build --execute-build; optional, no GPU queries'),
                host=dict(monotonic_clock=dict(implementation=clock.implementation,
                                              resolution=clock.resolution, adjustable=clock.adjustable),
                          memlock=list(resource.getrlimit(resource.RLIMIT_MEMLOCK))),
                gates=['fresh release build pin', 'explicit sole-owner grant + physical UUID',
                       'independent idle before/after each engine', 'authenticated descendant drain',
                       'observed GPU/PSS within targets', 'release basic quality before Hello32'],
                deferred=['p128o32 needs a separately versioned ctx160 admission/quality profile'],
                exact_speedup=None, hard_memory_partition=False)


def supervise(payload, seconds):
    ct.require_cpu_python()
    payload, launch_hash = launch_guard.seal(payload, payload['directory'])
    environment = dict(cpu_environment(), FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
    argv = ct.isolated_argv([sys.executable, '-S', '-B', str(Path(__file__).resolve()),
                             '--_worker', launch_hash], environment)
    try:
        return supervisor.run_process(argv, seconds, payload)
    except BaseException as error:
        receipt = getattr(error, 'supervisor_result', {})
        receipt.setdefault('error', type(error).__name__ + ': ' + str(error))
        return receipt


def build(c, directory):
    before = ct.source_pins()
    environment = ct.build_environment(c)
    pins = ct.workflow_pins()
    tools = [Path(c['cargo']), Path(environment['NVCC']), ct.absolute(c['ferrule_build_environment'])]
    tools += [Path(environment['CUDA_HOME']) / 'lib' / n for n in ('libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12')]
    pins.update({str(p): ct.sha(p) for p in tools})
    inventories = ct.required_inventory(c)
    spec = dict(mode='build', config=c, directory=str(directory), argv=ct.build_argv(c), environment=environment,
                pins=pins, sources=before, inventories=inventories)
    spec['contract_sha256'] = ct.digest(spec)
    spec['config_sha256'] = ct.digest(c)
    ct.write(directory / 'launch.json', spec)
    result = supervise(spec, c['limits']['build_seconds'])
    receipt = dict(schema=ct.SCHEMA + 'build.v1', status='failed', argv=spec['argv'],
                   environment=environment, supervisor=result, sources_before=before,
                   sources_after=ct.source_pins(), pins=pins, inventories=inventories,
                   inventory_status='complete_declared_build_inputs', lto='fat via pinned environment + Cargo.toml')
    try:
        ct.require(ct.successful_process(result), 'Build failed/timed out/custody unknown')
        ct.require(receipt['sources_after'] == before, 'Source changed during build')
        ct.check_pins(pins)
        rows = []
        with (directory / 'cargo.jsonl').open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get('reason') == 'compiler-artifact' and row.get('target', {}).get('name') == 'ferrule':
                    rows.append(row)
        receipt['artifact'] = ct.release_artifact(rows)
        receipt['binary_sha256'] = ct.sha(receipt['artifact']['executable'])
        ct.validate_inventory(c, inventories)
        artifact_dir = ct.ROOT / 'target/paper-m7-release-artifacts' / directory.name
        artifact_dir.mkdir(parents=True, exist_ok=False)
        receipt['staged_binary'] = ct.stage_binary(receipt['artifact']['executable'], artifact_dir)
        receipt['status'] = 'pass'
    except Exception as error:
        receipt['error'] = str(error)
    ct.validate_schema(receipt)
    ct.write(directory / 'build.json', receipt)
    return receipt


def prepare(c, directory, build_path):
    receipt = ct.load(build_path)
    binary = ct.validate_build(receipt, c)
    manifest = plan(c)
    manifest.update(schema=ct.SCHEMA + 'prepared.v1', status='prepared_cpu_only',
                    binary=binary, binary_sha256=receipt['binary_sha256'],
                    build_receipt=str(Path(build_path).resolve()), build_receipt_sha256=ct.sha(build_path),
                    source_pins=receipt['sources_after'], workflow_pins=ct.workflow_pins(),
                    runtime_pins=ct.vllm_pins(c), inventories=ct.required_inventory(c, runtime=True),
                    inventory_status='complete_declared_runtime_inputs', release_quality='not_yet_bound')
    manifest['runtime_pins'].update(ct.build_runtime_pins(receipt))
    manifest['historical_build_workflow_pins'] = {p: h for p, h in receipt['pins'].items()
                                                   if Path(p).is_relative_to(HERE)}
    manifest['runtime_pins'][str(ct.absolute(c['quality_history']))] = manifest['quality_history']['sha256']
    artifact_dir = ct.ROOT / 'target/paper-m7-release-artifacts' / directory.name
    artifact_dir.mkdir(parents=True, exist_ok=False)
    manifest['vllm_staged'] = ct.stage_binary(ct.absolute(c['vllm']), artifact_dir)
    manifest['vllm_binary_sha256'] = ct.sha(manifest['vllm_staged'])
    manifest['model_identity_sha256'] = ct.digest(manifest['model_identity'])
    manifest['manifest_sha256'] = ct.digest(manifest)
    ct.validate_schema(manifest)
    ct.write(directory / 'prepared.json', manifest)
    return manifest


def validate_manifest(path):
    m = ct.load(path)
    ct.validate_schema(m)
    ct.require(m.get('schema') == ct.SCHEMA + 'prepared.v1', 'Need prepared manifest')
    c = m['config']
    ct.config_value(c)
    ct.require(m['config_sha256'] == ct.digest(c), 'Stale/unknown prepared config')
    unsigned = {k: v for k, v in m.items() if k != 'manifest_sha256'}
    ct.require(ct.digest(unsigned) == m['manifest_sha256'], 'Manifest was modified')
    ct.require(set(m['workflow_pins']) == set(ct.workflow_pins()), 'Missing mandatory workflow pin inventory')
    for key in ('workflow_pins', 'runtime_pins'):
        ct.check_pins(m[key])
    ct.require(all(m['runtime_pins'].get(p) == h for p, h in ct.vllm_pins(c).items()), 'Missing mandatory entrypoint/runtime pins')
    ct.validate_inventory(c, m['inventories'], runtime=True)
    ct.validate_staged_binary(m['vllm_staged'], m['vllm_binary_sha256'])
    ct.require(m['source_pins'] == ct.source_pins(), 'Current sources differ from prepared release')
    ct.require(ct.sha(m['build_receipt']) == m['build_receipt_sha256'], 'Build receipt changed')
    ct.require(ct.validate_build(ct.load(m['build_receipt']), c) == m['binary'], 'Build executable changed')
    ct.require(ct.model_identity(c) == m['model_identity'], 'Checkpoint/tokenizer metadata or shard stats changed')
    ct.require(m['model_identity_sha256'] == ct.digest(m['model_identity'])
               and m['protocol_sha256'] == ct.digest(c['protocol']), 'Identity digest mismatch')
    return m


def metadata(m, engine, uuid):
    return dict(comparison_intent='system-level', source={'prepared_sha256': m['manifest_sha256']},
                model=m['model_identity'], precision=dict(declared=m['config']['protocol'], effective='unverified'),
                topology=dict(tp=1, pp=1, ep=1, dp=1, physical_gpu_uuid=uuid),
                resources=m['config']['resources'], cache=dict(prefix_mode='off', state_reset='per_request', os_cache='uncontrolled'),
                quality_gate=dict(historical=m['quality_history']['status'], release='separate owner evidence'),
                runtime_contract=dict(status='unverified', identity_mode=ct.identity_mode(m['config']),
                                      performance_basis=('exclusive_operator_allocation_assumption'
                                          if ct.identity_mode(m['config']) == 'device_exclusive_local_custody'
                                          else 'hostpid_verified'), owned_process_gpu_peak_bytes=None),
                operator_note='Boundary-aligned candidate only; exact arithmetic/speedup not claimed', engine=engine)


def require_free_port(spec):
    port = int(spec['url'].split(':')[2].split('/')[0])
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', port))
        probe.listen(1)


def bound_server_argv(spec, executable_fd):
    if spec['engine'] == 'vllm':
        # Python spawn reopens __main__.__file__ in a fresh interpreter. An fd
        # pathname is not stable there; use the unique read-only staged script.
        path = Path(spec['argv'][0])
        current, opened = path.stat(), os.fstat(executable_fd)
        ct.require((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
                   == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns),
                   'Staged Python entrypoint inode changed before launch')
        ct.require(not path.is_symlink() and current.st_mode & 0o222 == 0
                   and path.parent.stat().st_mode & 0o222 == 0, 'Staged Python entrypoint became writable')
        return list(spec['argv'])
    return ['/proc/self/fd/' + str(executable_fd), *spec['argv'][1:]]


class StartupGuard:
    """Abort the maintained readiness poll immediately on telemetry failure."""
    def __init__(self, server, monitor):
        self.server, self.monitor = server, monitor

    def poll(self):
        ct.require(not self.monitor.failed.is_set(), 'Resource monitor failed during startup')
        return self.server.poll()


def case_worker(payload):
    c, directory = payload['config'], Path(payload['directory'])
    telemetry.authorize(payload.get('granted'), payload['uuid'], payload['owner_note'], ct.identity_mode(c))
    ct.require(os.getpid() == os.getpgrp(), 'Worker must be in supervisor-owned fresh session')
    spec, workload = payload['spec'], c['workloads'][payload['mode']]
    result = dict(status='failed', started_monotonic=time.monotonic(), exact_speedup=None)
    server, monitor = None, None
    try:
        ct.write(directory / 'metadata.json', payload['metadata'])
        require_free_port(spec)
        with (directory / 'server.log').open('xb') as log:
            worker_contract(payload)
            result['contract_validation_seconds'] = time.monotonic() - result['started_monotonic']
            result['server_launch_monotonic'] = time.monotonic()
            with ct.pinned_executable(spec['argv'][0], payload['executable_sha256']) as executable_fd:
                bound_argv = bound_server_argv(spec, executable_fd)
                server = subprocess.Popen(ct.isolated_argv(bound_argv, spec['environment']),
                                          pass_fds=(executable_fd,), cwd=ct.ROOT, stdout=log, stderr=subprocess.STDOUT)
            result['executable_binding'] = ('verified read-only staged Python path, matched to open hashed inode; spawn-compatible'
                                            if spec['engine'] == 'vllm' else
                                            'verified read-only staged inode executed through inherited fd')
            monitor = telemetry.Monitor(directory, server, payload['gpu_tool'], payload['uuid'], payload['owner_note'], c,
                                                    payload['authenticated_worker'])
            monitor.start()
            ready = http.wait_ready(spec, StartupGuard(server, monitor), c['limits']['startup_seconds'])
            result['startup_seconds'] = ready - result['server_launch_monotonic']
            ct.require(not monitor.failed.is_set(), 'Resource monitor failed during startup')
            code = http.main(['--engine', spec['engine'], '--url', spec['url'], '--model', spec['model'],
                              '--prompt', 'Hello', '--input-tokens', '1', '--seed', '1',
                              '--max-tokens', str(workload['max_tokens']), '--concurrency', '1',
                              '--requests', str(workload['requests']), '--warmup', str(workload['warmup']),
                              '--repeat', '1', '--timeout', str(c['limits']['request_seconds']),
                              '--window-kind', 'fixed-work', '--measurement-seconds', str(c['limits']['measurement_seconds']),
                              '--metadata', str(directory / 'metadata.json'), '--output', str(directory / 'http.json')])
            ct.require(code == 0 and ct.http_success(ct.load(directory / 'http.json'), workload), 'HTTP incomplete/failed/missing token counts')
            ct.require(not monitor.failed.is_set(), 'Resource target exceeded or telemetry failed')
            result['status'] = 'complete'
    except BaseException as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        if 'server_launch_monotonic' in result and 'startup_seconds' not in result:
            result['startup_failure_seconds'] = time.monotonic() - result['server_launch_monotonic']
    finally:
        # Finish request-lifetime sampling before graceful root shutdown; custody
        # and independent post-case GPU idle are decided by the parent supervisor.
        if monitor:
            result['resources'] = monitor.finish()
            if result['resources']['status'] != 'observed_within_targets' and result['status'] == 'complete':
                result.update(status='resource_failed', error='Final telemetry failed or exceeded target')
        if server:
            try:
                result['shutdown'] = http.shutdown_owned_group(server, c['limits']['cleanup_seconds'])
                if result['shutdown']['timed_out']:
                    result['status'] = 'cleanup_timeout'
            except BaseException as error:
                result.update(status='cleanup_unknown', cleanup_error=str(error))
        log_path = directory / 'server.log'
        if log_path.exists():
            with log_path.open('rb') as stream:
                log = stream.read(16 * 1024 * 1024)
                truncated = bool(stream.read(1))
            result['runtime'] = ct.runtime_evidence(spec['engine'], log.decode('utf-8', 'replace'))
            result['runtime'].update(log_sha256=ct.sha(log_path), scanned_prefix_bytes=len(log), truncated=truncated)
            result['runtime']['performance_fields_observed'] = False
            result['runtime']['performance_parser'] = 'profile/backend evidence only; no performance-field extraction implemented'
            if result['runtime']['contradictions'] and result['status'] == 'complete':
                result.update(status='runtime_contradiction', error='Explicit backend/offload contract contradicted in logs')
            elif result['runtime']['contradictions']:
                result['runtime']['contract_warning'] = 'Runtime warning retained; primary case failure preserved'
        result['wall_seconds'] = time.monotonic() - result['started_monotonic']
        ct.write(directory / 'case.json', result)
    return 0 if result['status'] == 'complete' else 1


def worker_contract(payload):
    c = payload.get('config')
    ct.require(c in (ct.BASE, ct.EXCLUSIVE) and payload.get('config_sha256') == ct.digest(c),
               'Worker config contract mismatch')
    ct.config_value(c)
    mode = payload.get('mode')
    if mode == 'build':
        expected = dict(mode='build', config=c, directory=payload['directory'], argv=payload['argv'],
                        environment=payload['environment'], pins=payload['pins'], sources=payload['sources'],
                        inventories=payload['inventories'])
        ct.require(payload.get('contract_sha256') == ct.digest(expected), 'Build worker contract mismatch')
        return
    ct.require(mode in ('feasibility', 'run'), 'Unknown worker mode')
    ct.validate_owner_lock(payload['owner_lock'], payload['uuid'], payload['coordinator']['pid'])
    prepared = Path(payload['prepared_path']).resolve()
    ct.require(ct.sha(prepared) == payload['prepared_sha256'], 'Prepared receipt changed before worker launch')
    manifest = validate_manifest(prepared)
    ct.require(manifest['manifest_sha256'] == payload['manifest_sha256'], 'Prepared manifest digest mismatch')
    ct.validate_staged_binary(payload['binary'], payload['binary_sha256'])
    ct.require(manifest['config'] == c, 'Worker/prepared protocol mismatch')
    telemetry.authorize(payload.get('granted'), payload['uuid'], payload['owner_note'], ct.identity_mode(c))
    expected = ct.server_spec(c, payload['spec']['engine'], payload['binary'], payload['uuid'])
    if payload.get('diagnostic_performance') and payload['spec']['engine'] == 'ferrule':
        expected['environment']['FERRULE_LOG'] = DIAGNOSTIC_LOG
    if payload['spec']['engine'] == 'vllm':
        expected['argv'][0] = manifest['vllm_staged']
    ct.require(payload['spec'] == expected, 'Worker staged entrypoint differs')
    expected_executable_hash = manifest['vllm_binary_sha256'] if payload['spec']['engine'] == 'vllm' else manifest['binary_sha256']
    ct.require(payload['executable_sha256'] == expected_executable_hash, 'Worker executable hash mismatch')
    if mode == 'run':
        ct.require(payload.get('quality_path') and ct.sha(payload['quality_path']) == payload['quality_sha256'],
                   'Quality receipt missing or changed')
        ct.quality_receipt(payload['quality_path'], manifest)
        ct.require(payload.get('feasibility_path') and ct.sha(payload['feasibility_path']) == payload['feasibility_sha256'],
                   'Feasibility receipt missing or changed')
        validate_feasibility(payload['feasibility_path'], manifest, payload['uuid'])
    contract = dict(mode=mode, config=c, directory=payload['directory'], prepared_path=str(prepared),
                    prepared_sha256=payload['prepared_sha256'], manifest_sha256=payload['manifest_sha256'],
                    binary=payload['binary'], binary_sha256=payload['binary_sha256'], spec=payload['spec'],
                    executable_sha256=payload['executable_sha256'], diagnostic_performance=payload['diagnostic_performance'],
                    uuid=payload['uuid'], quality_path=payload.get('quality_path'), quality_sha256=payload.get('quality_sha256'),
                    feasibility_path=payload.get('feasibility_path'), feasibility_sha256=payload.get('feasibility_sha256'))
    ct.require(payload.get('contract_sha256') == ct.digest(contract), 'Execution worker contract mismatch')


def worker(launch_hash):
    ct.require_cpu_python()
    payload = json.load(sys.stdin)
    c = ct.config_value(payload['config'])
    payload['authenticated_worker'] = launch_guard.authenticate(payload, launch_hash)
    directory = Path(payload['directory']).resolve()
    ct.require(directory.is_relative_to(HERE / 'runs'), 'Worker artifacts must remain in paper runs')
    if payload['mode'] == 'build':
        worker_contract(payload)
        ct.require(payload['argv'] == ct.build_argv(c) and payload['environment'] == ct.build_environment(c), 'Unsafe build command')
        ct.require(set(payload['pins']) >= set(ct.workflow_pins()), 'Missing worker build pins')
        ct.check_pins(payload['pins'])
        ct.require(payload['sources'] == ct.source_pins(), 'Worker source snapshot changed')
        ct.validate_inventory(c, payload['inventories'])
        with (directory / 'cargo.jsonl').open('xb') as out, (directory / 'cargo.stderr.log').open('xb') as err:
            child = subprocess.Popen(ct.isolated_argv(payload['argv'], payload['environment']),
                                     cwd=ct.ROOT, stdout=out, stderr=err)
            return child.wait()  # bounded by authenticated outer supervisor
    ct.require(payload['mode'] in ('feasibility', 'run'), 'Unknown worker mode')
    # case_worker checks the full contract once, immediately before server Popen.
    return case_worker(payload)


def validate_feasibility(path, m, uuid):
    receipt = ct.load(path)
    ct.validate_schema(receipt)
    ct.require(receipt.get('schema') == ct.SCHEMA + ('execution.v2' if ct.identity_mode(m['config']) == 'device_exclusive_local_custody' else 'execution.v1') and receipt.get('mode') == 'feasibility'
               and receipt.get('status') == 'complete' and receipt.get('manifest_sha256') == m['manifest_sha256']
               and receipt.get('gpu_uuid') == uuid, 'Need successful same-manifest/same-GPU feasibility receipt')
    ct.require(receipt.get('observer_effect') == 'baseline_ferrule_log_info_only',
               'Diagnostic receipts cannot qualify baseline execution')
    ct.require(len(receipt.get('cases', [])) == 2 and {x['engine'] for x in receipt['cases']} == {'ferrule', 'vllm'}, 'Both engines must pass feasibility')
    ct.check_pins(receipt['artifact_pins'])
    for case in receipt['cases']:
        folder = Path(path).resolve().parent / case['engine']
        for name in ('case.json', 'http.json', 'supervisor.json', 'before.json', 'after.json', 'launch.json'):
            ct.require(str(folder / name) in receipt['artifact_pins'], 'Missing feasibility artifact pin: ' + name)
        ct.require(ct.load(folder / 'supervisor.json') == case['supervisor']
                   and ct.load(folder / 'case.json') == case['observation']
                   and ct.load(folder / 'after.json') == case['after'], 'Feasibility receipt/artifact mismatch')
        ct.require(case['eligible_observation'] is True and ct.successful_process(case['supervisor'])
                   and case['after']['status'] == 'verified'
                   and ct.load(folder / 'before.json')['status'] == 'verified'
                   and case['observation']['status'] == 'complete'
                   and case['observation']['resources']['status'] == 'observed_within_targets'
                   and ct.http_success(ct.load(folder / 'http.json'), m['config']['workloads']['feasibility']),
                   'Feasibility custody/resources/HTTP incomplete')
    return dict(path=str(Path(path).resolve()), sha256=ct.sha(path))


def execute(args, directory):
    telemetry.authorize(args.sole_gpu_owner_granted, args.gpu_uuid, args.owner_note)
    m = validate_manifest(args.prepared)
    c = m['config']
    identity_mode = ct.identity_mode(c)
    ct.require(getattr(args, 'identity_mode', 'hostpid_verified') == identity_mode,
               'Explicit identity mode must match prepared protocol')
    if hasattr(args, 'config'):
        ct.require(ct.config(args.config) == c, 'Requested/prepared config differs')
    exclusive = identity_mode == 'device_exclusive_local_custody'
    tool = '/usr/bin/nvidia-smi'
    quality = ct.quality_receipt(args.release_quality, m) if args.release_quality else None
    previous = None
    if args.mode == 'run':
        ct.require(quality is not None, 'Hello32 requires --release-quality tied to this release binary')
        ct.require(args.feasibility is not None, 'Hello32 requires --feasibility receipt')
        previous = validate_feasibility(args.feasibility, m, args.gpu_uuid)
    report = dict(schema=ct.SCHEMA + ('execution.v2' if exclusive else 'execution.v1'), mode=args.mode, status='incomplete',
                  manifest_sha256=m['manifest_sha256'], gpu_uuid=args.gpu_uuid,
                  owner_note=args.owner_note, release_quality=quality, feasibility=previous,
                  gpu_tool_sha256=ct.sha(tool), cases=[], exact_speedup=None, fairmatch=False,
                  owner_lock=None, observer_effect=('diagnostic_ferrule_performance_debug_not_baseline'
                      if getattr(args, 'diagnostic_performance', False) else 'baseline_ferrule_log_info_only'),
                  comparison_kind='descriptive boundary-aligned M7 candidate; no hard partition',
                  comparison_status='pending',
                  fairness_blockers=['partial checkpoint identity', 'effective runtime evidence incomplete',
                                     'arithmetic not matched', 'pinned/total physical host memory unverified'])
    if exclusive:
        report.update(identity_mode=identity_mode, protocol_id=c['protocol']['protocol_id'],
                      sole_gpu_owner_granted=True, performance_basis='exclusive_operator_allocation_assumption',
                      owned_gpu_attribution='unknown', owned_process_gpu_peak_bytes=None,
                      absence_of_foreign_interference_proven=False)
        report['fairness_blockers'].append('NVML PIDs may be unmapped; exclusive allocation is an operator assumption')
    ct.write(directory / 'launch.json', dict(prepared=str(Path(args.prepared).resolve()), manifest=m,
                                            mode=args.mode, owner_note=args.owner_note, gpu_uuid=args.gpu_uuid))
    lock_cm = ct.exclusive_owner_lock(args.gpu_uuid)
    lock_info = lock_cm.__enter__()
    report['owner_lock'] = lock_info
    try:
        for engine in ('ferrule', 'vllm'):
            # Parent pins were verified above; each worker independently verifies
            # its sealed prepared hash and full inventories immediately before launch.
            case_dir = directory / engine
            case_dir.mkdir()
            before = telemetry.idle(tool, args.gpu_uuid, True, args.owner_note, c)
            ct.write(case_dir / 'before.json', before)
            ct.require(before['status'] == 'verified', 'Independent pre-case GPU idle not established')
            spec = ct.server_spec(c, engine, m['binary'], args.gpu_uuid)
            if getattr(args, 'diagnostic_performance', False) and engine == 'ferrule':
                spec['environment']['FERRULE_LOG'] = DIAGNOSTIC_LOG
            if engine == 'vllm':
                spec['argv'][0] = m['vllm_staged']
            payload = dict(mode=args.mode, config=c, config_sha256=ct.digest(c), directory=str(case_dir),
                           binary=m['binary'], binary_sha256=m['binary_sha256'], granted=True,
                           executable_sha256=m['binary_sha256'] if engine == 'ferrule' else m['vllm_binary_sha256'],
                           diagnostic_performance=getattr(args, 'diagnostic_performance', False),
                           uuid=args.gpu_uuid, owner_note=args.owner_note, gpu_tool=tool, spec=spec, owner_lock=lock_info,
                           prepared_path=str(Path(args.prepared).resolve()), prepared_sha256=ct.sha(args.prepared),
                           manifest_sha256=m['manifest_sha256'], quality_path=str(Path(args.release_quality).resolve()) if args.release_quality else None,
                           quality_sha256=ct.sha(args.release_quality) if args.release_quality else None,
                           feasibility_path=str(Path(args.feasibility).resolve()) if args.feasibility else None,
                           feasibility_sha256=ct.sha(args.feasibility) if args.feasibility else None,
                           metadata=metadata(m, engine, args.gpu_uuid))
            contract_payload = {k: payload[k] for k in ('mode', 'config', 'directory', 'prepared_path', 'prepared_sha256',
                              'manifest_sha256', 'binary', 'binary_sha256', 'executable_sha256', 'diagnostic_performance', 'spec', 'uuid', 'quality_path',
                              'quality_sha256', 'feasibility_path', 'feasibility_sha256')}
            payload['contract_sha256'] = ct.digest(contract_payload)
            ct.write(case_dir / 'launch.json', payload)
            result = supervise(payload, c['limits']['case_seconds'])
            ct.write(case_dir / 'supervisor.json', result)
            try:
                after = telemetry.idle(tool, args.gpu_uuid, True, args.owner_note, c)
            except Exception as error:
                after = dict(status='unverified', error=str(error))
            ct.write(case_dir / 'after.json', after)
            observed = ct.load(case_dir / 'case.json') if (case_dir / 'case.json').exists() else {'status': 'missing'}
            eligible = (ct.successful_process(result) and after['status'] == 'verified'
                        and observed['status'] == 'complete'
                        and observed.get('resources', {}).get('status') == 'observed_within_targets')
            case = dict(engine=engine, supervisor=result, after=after, observation=observed,
                        eligible_observation=eligible, exact_speedup=None)
            report['cases'].append(case)
            ct.write(case_dir / 'assessment.json', case)
            if exclusive and args.mode == 'feasibility':
                # A safely drained infeasible case is a result, not a reason to
                # substitute a policy. The next frozen engine still needs idle.
                ct.require(ct.drained(result) and after['status'] == 'verified',
                           'Fail-stop: authenticated custody or all-device release failed')
            else:
                ct.require(eligible, 'Fail-stop: request/resource/custody/release gate failed; no next engine')
        report['status'] = 'complete' if all(case['eligible_observation'] for case in report['cases']) else 'incomplete'
        if any(ct.resource_infeasible(case.get('observation', {})) for case in report['cases']):
            report['comparison_status'] = 'infeasible'
            report['infeasible_engines'] = [case['engine'] for case in report['cases']
                                            if ct.resource_infeasible(case.get('observation', {}))]
        elif report['status'] == 'complete':
            report['comparison_status'] = 'complete_observed'
        else:
            report['comparison_status'] = 'blocked'
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['comparison_status'] = 'blocked'
    finally:
        report['artifact_pins'] = {str(p): ct.sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
        try:
            ct.validate_schema(report)
            ct.write(directory / 'execution.json', report)
        finally:
            lock_cm.__exit__(None, None, None)
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == '--_worker':
        ct.require(len(argv) == 2, 'Free --_worker entry is forbidden; sealed receipt hash required')
        return worker(argv[1])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'prepare', 'build', 'feasibility', 'run'), nargs='?', default='plan')
    parser.add_argument('--config', type=Path, default=ct.DEFAULT)
    parser.add_argument('--id', help='New append-only directory under experiments/runs')
    parser.add_argument('--build-receipt', type=Path)
    parser.add_argument('--execute-build', action='store_true')
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--feasibility', type=Path)
    parser.add_argument('--release-quality', type=Path)
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--owner-note')
    parser.add_argument('--identity-mode', choices=ct.IDENTITY_MODES, default='hostpid_verified',
                        help='Explicit opt-in must match the prepared versioned protocol')
    parser.add_argument('--diagnostic-performance', action='store_true',
                        help='Separate diagnostic profile: JSON ferrule_performance=debug, never a baseline result')
    args = parser.parse_args(argv)
    directory = None
    try:
        ct.require_cpu_python()
        c = ct.config(args.config)
        if args.mode in ('feasibility', 'run'):
            telemetry.authorize(args.sole_gpu_owner_granted, args.gpu_uuid, args.owner_note, args.identity_mode)
            ct.require(args.identity_mode == ct.identity_mode(c), 'Explicit identity mode must match config')
            ct.require(args.prepared is not None, 'Need --prepared')
        if args.mode == 'build':
            ct.require(args.execute_build, 'Build requires explicit --execute-build; default plan is CPU-only')
        if args.mode == 'prepare':
            ct.require(args.build_receipt is not None, 'Need --build-receipt from this runner or identical external pinned build contract')
        if args.mode == 'plan':
            result = plan(c)
            if args.id:
                directory = ct.new_run(args.id)
                ct.write(directory / 'plan.json', result)
            else:
                print(json.dumps(result, indent=2, allow_nan=False))
            return 0
        directory = ct.new_run(args.id)
        if args.mode == 'build':
            result = build(c, directory)
        elif args.mode == 'prepare':
            result = prepare(c, directory, args.build_receipt)
        else:
            result = execute(args, directory)
        print(json.dumps(dict(directory=str(directory), status=result['status'], exact_speedup=None)))
        return 0 if result['status'] in ('pass', 'prepared_cpu_only', 'complete') else 1
    except (ValueError, OSError, KeyError, TypeError) as error:
        if directory is not None:
            ct.write(directory / 'blocked.json', dict(status='blocked', error=str(error), exact_speedup=None))
        print('BLOCKED: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
