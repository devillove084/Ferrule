#!/usr/bin/env python3
"""Frozen paper-only TP1 middle case; reuse the maintained custody/target/HTTP stack."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import runner as base
ct, guard, http = base.ct, base.launch_guard, base.http
sys.path.insert(0, str(HERE.parent / 'tp2_resident'))
import observer

PREPARED_SHA = '5c7a01147084aa95f6a6604cf727f69754c7e7b51539afd7423539723e4bc696'
FINALIZED = HERE / 'cpu-revisions/middle-finalized-20261013-02'
TP1_IMPORTS = ('papers/arxiv/experiments/runner.py', 'papers/arxiv/experiments/launch_guard.py',
    'papers/arxiv/experiments/contract.py', 'papers/arxiv/experiments/cli_audit.py',
    'papers/arxiv/experiments/telemetry.py', 'papers/arxiv/experiments/tp2_resident/observer.py',
    'papers/arxiv/experiments/tp2_resident/sampling_v4.py', 'papers/arxiv/experiments/tp2_resident/sampling_v2.py',
    'papers/arxiv/experiments/m7.json', 'papers/arxiv/experiments/m7-device-exclusive.json',
    'scripts/_profile_capture.py', 'scripts/_owned_subreaper.py',
    'scripts/test_vllm_qwen35.py', 'scripts/bench_ferrule_vllm.py')
RUN = HERE / 'runs/storage-demand-tp1-middle-20261013-02'
PURPOSE = 'storage_demand_tp1_middle_four_requests_v1'
COUNTERS = ('checkpoint_read_calls', 'checkpoint_read_bytes', 'checkpoint_source_checks',
            'checkpoint_path_metadata_calls', 'checkpoint_handle_metadata_calls', 'h2d_copies',
            'h2d_bytes', 'artifact_uploads', 'artifact_upload_bytes', 'expert_hits',
            'expert_uploads', 'expert_evictions')
GAUGES = ('expert_resident_entries', 'expert_resident_bytes', 'expert_pending_upload_bytes',
          'expert_scratch_bytes', 'expert_owner_peak_bytes', 'expert_unknown_quarantine_bytes')
RESOURCE_COMPLETION_FIELDS = dict(pending_requests_observed=0, pending_operations_observed=0,
    custody_unknown_observed=False, quarantine_observed=False, ledger_credits_outstanding_observed=0,
    fence_quiescent_observed=True, source_completion_observed=True)


def shutdown_contract(returncode, resource_sample=None):
    """Assess explicit final owner evidence, never infer resource release from exit."""
    process_exit_success = returncode == 0 if type(returncode) is int else None
    sample = resource_sample if isinstance(resource_sample, dict) else {}
    observed = {name:sample.get(name) for name in RESOURCE_COMPLETION_FIELDS}
    missing = [name for name, value in observed.items() if value is None]
    invalid = [name for name, value in observed.items() if value is not None
        and (type(value) is not type(RESOURCE_COMPLETION_FIELDS[name])
             or (type(value) is int and value < 0))]
    failed = [name for name, value in observed.items() if value is not None
        and name not in invalid and value != RESOURCE_COMPLETION_FIELDS[name]]
    if process_exit_success is False or invalid or failed or (resource_sample is not None and not isinstance(resource_sample, dict)):
        complete = 'failed'
    elif process_exit_success is True and not missing:
        complete = 'verified'
    else:
        complete = 'unknown'
    return dict(normal_exit_observed=process_exit_success, process_exit_success=process_exit_success,
        resource_contract_complete=complete, **observed,
        resource_evidence_coverage=dict(required_fields=list(RESOURCE_COMPLETION_FIELDS),
            missing_fields=missing, invalid_fields=invalid, failed_fields=failed),
        basis='Process exit is supervision evidence only; resource verification requires explicit final owner ledger credits, fence quiescence, pending, Unknown/quarantine and current source-completion observations')


def frozen():
    ct.require(ct.sha(HERE / 'prepared.json') == PREPARED_SHA, 'Prepared SHA differs')
    prepared = ct.load(HERE / 'prepared.json')
    pins = ct.load(HERE / 'pins.json')['pins']
    ct.require(len(pins) == 702 and len(prepared['artifact_pins']) == 8, 'Pin coverage differs')
    ct.check_pins({p:h for p,h in prepared['artifact_pins'].items() if Path(p).name != 'README.md'})
    ct.check_pins({str(ct.ROOT / p):pins[str(ct.ROOT / p)] for p in TP1_IMPORTS})
    case = ct.load(HERE / 'case-middle.json')
    handoff = ct.load(ct.absolute(case['handoff']))
    ct.check_pins({str(ct.absolute(case['handoff'])):case['handoff_sha256']})
    relation = handoff['source_relation']
    ct.check_pins({str(ct.absolute(relation[key]['path'])):relation[key]['sha256'] for key in
        ('production_pins', 'source_snapshot', 'source_snapshot_manifest')})
    production = ct.load(ct.absolute(case['source_pin_manifest']))
    archive_pins = ct.load(ct.absolute(relation['source_snapshot_manifest']['path']))
    ct.require(len(production) == 597 and all(pins.get(p) == h == archive_pins.get(p) for p,h in production.items()),
        '597 production pins must match the sealed artifact source archive and legacy 702 manifest')
    ct.require(ct.sha(ct.absolute(case['binary'])) == case['binary_sha256'], 'Current binary differs')
    ct.require(case['topology'] == dict(pp=1, tp=1, ep=1, rank_backend='thread', owners=1, visible_ordinals=[0]), 'TP1 topology differs')
    shards = ct.load(HERE / 'checkpoint-metadata.json')['shards']
    ct.require(len(shards) == 14, 'Checkpoint coverage differs')
    for row in shards:
        path = Path(row['path'])
        st = path.stat()
        ct.require([st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns] == row['identity'], 'Checkpoint identity differs')
        ct.require(type(row.get('header_bytes')) is int and row['header_bytes'] > 8
            and isinstance(row.get('header_sha256'), str) and len(row['header_sha256']) == 64,
            'Frozen checkpoint header snapshot malformed')
    # Header bytes and hashes above are reused from the frozen CPU snapshot;
    # this finalization pass intentionally performs no tensor/header payload read.
    refs = ct.load(HERE / 'canaries.json')
    ct.check_pins(refs['reference_pins'])
    cpu = ct.load(HERE / 'cpu-checks.json')
    ct.require(len(cpu['checks']) == 15 and all(v is True for v in cpu['checks'].values()), 'CPU contracts incomplete')
    ct.require(handoff['build']['unknown_nccl_enabled'] is False, 'NCCL build qualification differs')
    return case, refs['rows']


def schedule(case, refs):
    rows = [dict(name='hello2', prompt='Hello', input_tokens=1, max_tokens=2, reference=None)]
    rows += [dict(name='canary-' + r['id'], prompt=r['prompt'], input_tokens=r['input_tokens'], max_tokens=8, reference=r) for r in refs]
    ct.require(len(rows) == 4 and [r['name'] for r in rows] == ['hello2', 'canary-qa', 'canary-arithmetic', 'canary-chinese'], 'Four-request schedule differs')
    for row in rows:
        ct.require(type(row['input_tokens']) is int and 0 < row['input_tokens'] + row['max_tokens'] <= 128, 'Context bounds differ')
        ref = row['reference']
        ct.require(ref is None or len(ref['reference_ids']) == 8, 'Generation8 reference missing')
        row['payload'] = http.make_payload('qwen35-storage-demand-paper', row['prompt'], row['max_tokens'], 1)
    return rows


def config(total):
    ct.require(type(total) is int and total >= 7168 << 20, 'Actual capacity insufficient')
    return dict(resources=dict(gpu_expected_total_bytes=total, gpu_target_bytes=7168 << 20,
        host_target_bytes=(1 << 63) - 1, sample_interval_seconds=2,
        host_policy='observation-only sentinel; not an application/physical host cap',
        physical_host_cap_bytes=None), limits=dict(idle_seconds=20, case_seconds=1100))


def strict_idle(value, uuid):
    ct.require(value.get('status') == 'verified' and len(value.get('baseline', {})) == 8, 'All-eight idle unverified')
    ct.require(all(d['gpu_bytes'] == 0 and d['utilization'] == 0 for d in value['baseline'].values())
        and all(not s['apps'] for s in value['observations']), 'All-eight zero idle required')
    ct.require(value['baseline'][uuid]['index'] == 0, 'Physical UUID/index0 differs')


def discover(note):
    ct.require(isinstance(note, str) and note.strip(), 'Owner authorization is required before inventory')
    out = subprocess.run(['/usr/bin/nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3, check=True)
    devices = []
    import csv, io
    for row in csv.reader(io.StringIO(out.stdout)):
        ct.require(len(row) == 5, 'Malformed eight-GPU inventory')
        index, uuid, total, used, util = (x.strip() for x in row)
        devices.append(dict(index=int(index), uuid=uuid, total_bytes=int(total) << 20,
                            gpu_bytes=int(used) << 20, utilization=int(util)))
    ct.require(len(devices) == 8 and sorted(d['index'] for d in devices) == list(range(8))
        and len({d['uuid'] for d in devices}) == 8, 'Expected physical eight-GPU inventory')
    selected = next(d for d in devices if d['index'] == 0)
    ct.require(selected['gpu_bytes'] == 0 and selected['utilization'] == 0
        and all(d['gpu_bytes'] == 0 and d['utilization'] == 0 for d in devices), 'Preflight is not all-eight idle')
    sample = observer.all_gpu_sample(selected['uuid'], note)
    ct.require(sample['index'] == 0 and sample['uuid'] == selected['uuid'] and not sample['apps'], 'Selected UUID/index0 changed')
    return sample


def binding(payload):
    ct.require(payload.get('purpose') == PURPOSE and payload.get('request_count') == 4, 'Purpose/count guard')
    ct.require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', payload.get('uuid', '')) and payload.get('gpu_mask') == '0', 'UUID/mask guard')
    ct.require(payload.get('prepared_sha256') == PREPARED_SHA and payload.get('wrapper_sha256') == ct.sha(__file__), 'Prepared/wrapper source guard')
    ct.require(payload.get('worker_token') and os.environ.get('FERRULE_PAPER_WORKER_TOKEN') == payload['worker_token'], 'Worker nonce guard')
    ct.require(payload.get('config') == config(payload.get('actual_total_bytes')), 'Actual capacity/config guard')
    ct.require(Path(payload['directory']).resolve() == RUN.resolve(), 'Fixed run directory guard')
    ct.require(payload.get('granted') is True and isinstance(payload.get('owner_note'), str) and payload['owner_note'].strip(), 'Owner grant guard')


def cpu_runtime_pins(case):
    libraries = ct.load(ct.absolute(case['runtime_pin_manifest']))['libraries']
    pins = {path:value['sha256'] for path, value in libraries.items()}
    ct.check_pins(pins)
    driver_paths = [p for p in Path('/usr/lib64').glob('libnvidia-*.so.*') if p.is_file()]
    pins.update(ct.inventory(driver_paths) if driver_paths else {})
    return dict(schema='ferrule.paper.storage-demand.runtime-pins.v1', pins=pins,
        scope='frozen twelve loader libraries plus CPU-hashed installed driver auxiliaries; no CUDA load during pinning')


def runtime_environment_contract(actual, expected):
    ct.require(all(actual.get(k) == v for k, v in expected.items()) and not actual.get('LD_PRELOAD'),
        'Actual CUDA environment/loader differs')


def request_contract(payload, row):
    http.strict_json(json.dumps(payload, allow_nan=False))
    ct.require(payload == http.make_payload('qwen35-storage-demand-paper', row['prompt'], row['max_tokens'], 1),
        'Unknown or changed request controls')
    ct.require(type(row['input_tokens']) is int and type(row['max_tokens']) is int
        and row['input_tokens'] > 0 and row['max_tokens'] > 0
        and row['input_tokens'] + row['max_tokens'] <= 128, 'Request context exceeds frozen bounds')


def map_snapshot(server, identity, case, directory, name):
    observer.verify_root(identity)
    paths = sorted({line.split()[-1] for line in Path(f'/proc/{server.pid}/maps').read_text().splitlines()
                    if len(line.split()) >= 6 and line.split()[-1].startswith('/')})
    observer.verify_root(identity)
    libraries = ct.load(directory / 'current-runtime-pins.json')['pins']
    environment = dict(x.split('=', 1) for x in Path(f'/proc/{server.pid}/environ').read_text().split('\0') if '=' in x)
    runtime_environment_contract(environment, case['environment'])
    observer.verify_root(identity)
    rows = []
    for path in paths:
        resolved = str(Path(path).resolve())
        if '.so' in Path(path).name or Path(path).name.startswith('ld-'):
            expected = libraries.get(resolved)
            actual = ct.sha(resolved)
            rows.append(dict(path=path, resolved=resolved, sha256=actual, expected_sha256=expected, pinned=actual == expected))
    value = dict(identity=identity, monotonic=time.monotonic(), unix_ns=time.time_ns(), libraries=rows,
                 actual_environment={k:environment.get(k) for k in case['environment']}, visible_gpu_mask='0',
                 nccl_mapped=any('nccl' in r['path'].lower() for r in rows),
                 all_mapped_libraries_pinned=bool(rows) and all(r['pinned'] for r in rows))
    ct.write(directory / (name + '-library-maps.json'), value)
    ct.require(value['all_mapped_libraries_pinned'] and not value['nccl_mapped'], 'Runtime library map pin/NCCL guard')
    return value


def invocations(path, start=0, end=None):
    data = path.read_bytes()
    end = len(data) if end is None else end
    prefix_lines = data[:start].count(b'\n')
    rows = []
    for index, raw in enumerate(data[start:end].splitlines(), prefix_lines + 1):
        try:
            event = json.loads(raw)
        except (ValueError, UnicodeError):
            continue
        fields = event.get('fields', {})
        fields = event.get('fields') if isinstance(event.get('fields'), dict) else event
        if fields.get('event') == 'mixed_decoder_invocation':
            rows.append(dict(invocation_id='server-log-line-' + str(index), line=index, raw=event,
                             fields=fields))
    return rows


def counter_summary(rows):
    fields = [r['fields'] for r in rows]
    valid = bool(fields) and all(f.get('counters_available') is True and f.get('counter_reset_detected') is False for f in fields)
    deltas = {k:sum(f[k] for f in fields) if valid and all(type(f.get(k)) is int and f[k] >= 0 for f in fields) else None for k in COUNTERS}
    return dict(invocations=len(rows), valid=valid, deltas=deltas,
        last_gauges={k:fields[-1].get(k) if fields else None for k in GAUGES},
        expert_cache_miss_counter=None, io_credits=None,
        physical_nas_bytes=None, proc_read_bytes=None, nfs_shared_mount_bytes=None,
        scope='logical checkpoint reader and facade deltas; not physical NAS/device-kernel counts')


def one_request(directory, row, spec, monitor):
    request_contract(row['payload'], row)
    directory.mkdir(exist_ok=False)
    ct.write(directory / 'generated-controls.json', row['payload'])
    ct.write(directory / 'prompts.json', [dict(id=row['name'], prompt=row['prompt'], input_tokens=row['input_tokens'])])
    code = http.main(['--engine', 'ferrule', '--url', spec['url'], '--model', spec['model'],
        '--prompts-json', str(directory / 'prompts.json'), '--seed', '1', '--max-tokens', str(row['max_tokens']),
        '--concurrency', '1', '--requests', '1', '--warmup', '0', '--repeat', '1', '--timeout', '120',
        '--window-kind', 'fixed-work', '--measurement-seconds', '120', '--output', str(directory / 'http.json')])
    report = ct.load(directory / 'http.json')
    rows = [r for run in report.get('runs', []) for r in run.get('requests', [])]
    ct.require(code == 0 and report['status'] == 'complete' and len(rows) == 1 and report.get('warmups') == []
        and 'first_after_ready' not in report, 'One-request HTTP guard')
    actual = rows[0]
    ct.require(actual['status'] == 'success' and actual['input_length_matches'] is True
        and actual['actual_input_tokens'] == row['input_tokens'] and 0 < actual['actual_output_tokens'] <= row['max_tokens']
        and actual['finish_reason'] in ('length', 'stop') and not monitor.failed.is_set(), 'HTTP/observer gate')
    ref = row['reference']
    passed = ref is None or (actual['text'] == ref['reference_text'] and actual['actual_output_tokens'] == 8)
    return dict(name=row['name'], status='pass' if passed else 'canary_failed', actual=actual,
        generated_controls=row['payload'], reference=ref, observed_token_ids=actual.get('output_token_ids'),
        exact_token_pass=None, full_current_math_pass=False,
        quality_scope='actual visible greedy text and usage versus frozen HF generation8; not observed IDs/logits')


def worker(seal):
    ct.require_cpu_python()
    payload = http.strict_json(sys.stdin.read())
    authenticated = guard.authenticate(payload, seal)
    binding(payload)
    ct.require(ct.sha(payload['finalized_prepared']) == payload['finalized_prepared_sha256'], 'Sealed finalized preparation binding differs')
    validate_finalized(payload['finalized_prepared'])
    case, refs = frozen()
    directory = Path(payload['directory'])
    uuid = payload['uuid']
    ct.validate_owner_lock(payload['owner_lock'], uuid, payload['coordinator']['pid'])
    ct.require(os.getpid() == os.getpgrp(), 'Authenticated fresh session required')
    ct.require(ct.sha(directory / 'baseline.json') == payload['baseline_sha256']
        and ct.load(directory / 'baseline.json') == payload['baseline'], 'Sealed baseline guard')
    ct.require(ct.sha(directory / 'current-runtime-pins.json') == payload['runtime_pins_sha256'], 'Runtime pin receipt guard')
    ct.check_pins(ct.load(directory / 'current-runtime-pins.json')['pins'])
    argv = list(case['argv'])
    argv[0] = ct.validate_staged_binary(directory / 'staged/ferrule', case['binary_sha256'])
    spec = dict(engine='ferrule', argv=argv, environment=dict(base.cpu_environment(), **case['environment']),
                model='qwen35-storage-demand-paper', url='http://127.0.0.1:18369/v1/completions')
    ct.require(spec['environment']['CUDA_VISIBLE_DEVICES'] == spec['environment']['NVIDIA_VISIBLE_DEVICES'] == '0', 'Single GPU mask guard')
    before = observer.idle(payload['config'], (uuid,), payload['owner_note'], payload['baseline'])
    ct.write(directory / 'prelaunch.json', before)
    strict_idle(before, uuid)
    base.require_free_port(spec)
    registration = dict(purpose=PURPOSE, prepared_sha256=PREPARED_SHA, wrapper_sha256=ct.sha(__file__),
        original_binary=str(ct.absolute(case['binary'])), staged_binary_sha256=case['binary_sha256'], spec=spec,
        schedule=schedule(case, refs), startup_seconds=700, request_seconds=120, all_requests_seconds=300,
        lifetime_seconds=1100, shutdown_seconds=45, supervisor_seconds=1100, n=1,
        observer_effect='ferrule_performance DEBUG overhead unmeasured; not a fast-performance baseline',
        physical_host_cap_bytes=None, PssAnon=None, physical_nas_bytes=None)
    ct.write(directory / 'registration.json', registration)
    result = dict(status='failed', requests=[], started_monotonic=time.monotonic(), token_ids_exposed=False)
    module_spec = importlib.util.spec_from_file_location('storage_existing_target', ct.ROOT / 'scripts/_profile_capture.py')
    target = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(target)
    monitor = None
    server = None
    logpath = directory / 'server.log'
    with logpath.open('xb') as log:
        def launch(command, **kwargs):
            nonlocal server, monitor
            ct.require(command == ct.isolated_argv(spec['argv'], spec['environment']), 'Target argv binding differs')
            with ct.pinned_executable(argv[0], case['binary_sha256']) as fd:
                server = subprocess.Popen(ct.isolated_argv(base.bound_server_argv(spec, fd), spec['environment']),
                    pass_fds=(fd,), cwd=ct.ROOT, stdout=log, stderr=subprocess.STDOUT, **kwargs)
            result.update(server_pid=server.pid, launch_monotonic=time.monotonic(), launch_unix_ns=time.time_ns())
            try:
                monitor = observer.Monitor(directory, server, payload['config'], authenticated, (uuid,),
                    payload['owner_note'], payload['baseline'], spec['environment'].keys())
                result['server_identity'] = monitor.root
                monitor.start()
                ready = http.wait_ready(spec, base.StartupGuard(server, monitor), 700)
                monitor.mark_ready(ready)
                result.update(ready_monotonic=ready, ready_unix_ns=time.time_ns(), startup_seconds=ready - result['launch_monotonic'])
                ct.write(directory / 'ready.json', dict(monotonic=ready, unix_ns=time.time_ns(), server_identity=monitor.root,
                    observed=observer.all_gpu_sample(uuid, payload['owner_note']), declared_argv=spec['argv']))
                map_snapshot(server, monitor.root, case, directory, 'ready')
                def expired(number, frame):
                    raise TimeoutError('Four-request total budget exceeded')
                previous = signal.signal(signal.SIGALRM, expired)
                signal.setitimer(signal.ITIMER_REAL, 300)
                try:
                    for row in registration['schedule']:
                        ct.require(server.poll() is None and not monitor.failed.is_set(), 'Server/observer failed before request')
                        start = logpath.stat().st_size
                        anchor = dict(name=row['name'], local_request_id=PURPOSE + '/' + row['name'],
                            before_monotonic=time.monotonic(), before_unix_ns=time.time_ns(), log_start_byte=start)
                        observed = one_request(directory / row['name'], row, spec, monitor)
                        anchor.update(after_monotonic=time.monotonic(), after_unix_ns=time.time_ns(), log_end_byte=logpath.stat().st_size,
                            http_sent_monotonic=observed['actual']['sent_at'], http_ended_monotonic=observed['actual']['ended_at'])
                        events = invocations(logpath, start, anchor['log_end_byte'])
                        counters = counter_summary(events)
                        observed.update(anchor=anchor, counters=counters, invocation_mapping='serialized named request/log-byte interval; transaction is NOT request ID')
                        ct.write(directory / row['name'] / 'invocations.json', events)
                        ct.write(directory / row['name'] / 'observed.json', observed)
                        result['requests'].append(observed)
                        print(json.dumps(dict(request=row['name'], status=observed['status'], counters=counters), ensure_ascii=False), flush=True)
                        ct.require(observed['status'] == 'pass', 'Frozen canary visible text/usage failure')
                        ct.require(counters['valid'] and all(v is not None for v in counters['deltas'].values())
                            and all(e['fields'].get('device') == 0 and e['fields'].get('status') == 'completed'
                                    and e['fields'].get('quarantined') is False for e in events)
                            and counters['last_gauges']['expert_pending_upload_bytes'] == 0
                            and counters['last_gauges']['expert_unknown_quarantine_bytes'] == 0, 'Invocation source/device/completion counter gate')
                    ct.require(len(result['requests']) == 4
                        and [r['name'] for r in result['requests']] == [r['name'] for r in registration['schedule']],
                        'Final exact four-request schedule guard')
                    map_snapshot(server, monitor.root, case, directory, 'after-requests')
                    result['status'] = 'complete'
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)
            except BaseException as error:
                result.update(status='failed', error=type(error).__name__ + ': ' + str(error))
            finally:
                if monitor:
                    result['resources'] = monitor.finish()
                    if result['resources']['status'] != 'observed_within_targets':
                        result['status'] = 'resource_evidence_blocked'
                (directory / 'profile-stop').write_text('four-request admission closed; no QA or subsequent point\n')
            return server
        # Module-local launch binding only; existing target owns finite wait/TERM.
        # HTTP executes on the authenticated main thread, retaining its SIGALRM contracts.
        target.subprocess = SimpleNamespace(Popen=launch, DEVNULL=subprocess.DEVNULL,
            TimeoutExpired=subprocess.TimeoutExpired)
        try:
            result['target_returncode'] = target.run_target(directory, ct.isolated_argv(argv, spec['environment']), 1100, 45)
        except BaseException as error:
            result.update(status='target_failed', target_error=type(error).__name__ + ': ' + str(error))
    if (directory / 'profile-server.json').exists():
        result['target'] = ct.load(directory / 'profile-server.json')
        if result.get('target_returncode') != 0:
            result['status'] = 'normal_shutdown_unproven'
    # No final owner ledger/fence/source-completion sample is exposed by this path.
    result['shutdown_contract'] = shutdown_contract(server.returncode if server is not None else None)
    result['all_invocation_counters'] = counter_summary(invocations(logpath))
    text = logpath.read_text(errors='replace')
    result['runtime_log_evidence'] = [dict(line=i, text=line) for i, line in enumerate(text.splitlines(), 1)
        if any(word in line.lower() for word in ('device budget', 'required_device_bytes', 'loading ', 'mixed bf16',
            'unknown', 'admission', 'shutdown', 'error', 'quarantin', 'cuda owner'))]
    result['log_sha256'] = ct.sha(logpath)
    try:
        frozen()
        result['pins_after'] = dict(status='verified_scoped', legacy_manifest_entries=702,
            production_source_scope='597 pins via sealed compiled-source archive; current worktree not rehashed',
            active_workflow_scope='explicit TP1 import/assets; unrelated paper workflow excluded',
            tensor_payload_bytes_hashed=0, shard_headers_reread=False)
    except BaseException as error:
        result.update(status='pin_failed', pins_after_error=str(error))
    result['wall_seconds'] = time.monotonic() - result['started_monotonic']
    ct.write(directory / 'case.json', result)
    return 0 if result['status'] == 'complete' else 1


def cpu_contracts(case=None, refs=None):
    if case is None or refs is None:
        case, refs = frozen()
    rows = schedule(case, refs)
    fixture = dict(purpose=PURPOSE, request_count=4, uuid='GPU-00000000-0000-0000-0000-000000000000', gpu_mask='0', prepared_sha256=PREPARED_SHA,
        wrapper_sha256=ct.sha(__file__), worker_token='cpu-fixture-not-authority', actual_total_bytes=24 << 30,
        config=config(24 << 30), directory=str(RUN), granted=True, owner_note='CPU contract fixture')
    prior = os.environ.get('FERRULE_PAPER_WORKER_TOKEN')
    os.environ['FERRULE_PAPER_WORKER_TOKEN'] = fixture['worker_token']
    tests = dict(historical_cpu_contracts_15_bound=True, legacy_702_manifest_and_parent_entries_bound=True, exact_four_schedule=len(rows) == 4,
        greedy_payloads=all(r['payload']['temperature'] == 0 and r['payload']['seed'] == 1 and r['payload']['n'] == 1 for r in rows),
        physical_unknowns_null=case['physical_host_cap_bytes'] is None)
    try:
        binding(fixture)
        tests['binding_accepts_frozen_case'] = True
        for key, value in [('request_count', 10), ('request_count', 12), ('gpu_mask', '0,1'), ('uuid', 'GPU-bad'),
                           ('purpose', 'tp2'), ('wrapper_sha256', '0' * 64), ('prepared_sha256', '0' * 64),
                           ('worker_token', None), ('worker_token', 'wrong-nonce'), ('actual_total_bytes', float('nan')), ('granted', False)]:
            candidate = copy.deepcopy(fixture)
            candidate[key] = value
            try:
                binding(candidate)
            except (ValueError, KeyError, TypeError):
                tests['reject_' + key + '_' + str(value)] = True
            else:
                raise ValueError('Negative guard accepted: ' + key)
    finally:
        if prior is None:
            os.environ.pop('FERRULE_PAPER_WORKER_TOKEN', None)
        else:
            os.environ['FERRULE_PAPER_WORKER_TOKEN'] = prior
    ct.require(all(tests.values()), 'CPU guard tests failed')
    return dict(status='passed_cpu_only', checks=tests, existing_contracts=15, wrapper_source_sha256=ct.sha(__file__),
        gpu_queries=0, gpu_locks=0, gpu_runs=0)


def finalized_pin_scope(case):
    """Pin only this TP1 entry/import/runtime/case closure, not all paper files."""
    imports = [ct.ROOT / p for p in TP1_IMPORTS] + [HERE / name for name in
        ('run_middle.py', 'test_run_middle_semantics.py', 'test_run_middle_target.py')]
    assets = [HERE / name for name in ('case-middle.json', 'canaries.json', 'requests.json',
        'cpu-help.json', 'cpu-checks.json', 'checkpoint-metadata.json', 'prepared.json', 'prepared.sha256')]
    assets += [ct.absolute(case['handoff']), ct.absolute(case['source_pin_manifest']),
        ct.absolute(case['runtime_pin_manifest']), ct.absolute(case['binary']), HERE / 'pins.json']
    relation = ct.load(ct.absolute(case['handoff']))['source_relation']
    assets += [ct.absolute(relation[key]['path']) for key in ('source_snapshot', 'source_snapshot_manifest')]
    refs = [ct.absolute(p) for p in ct.load(HERE / 'canaries.json')['reference_pins']]
    actual_imports = ct.inventory(imports)
    actual_assets = ct.inventory(assets + refs)
    runtime_doc = ct.load(ct.absolute(case['runtime_pin_manifest']))
    ct.require(len(runtime_doc['libraries']) == 12, 'Expected twelve runtime library pins')
    runtime_pins = {path:row['sha256'] for path,row in runtime_doc['libraries'].items()}
    ct.check_pins(runtime_pins)
    for path, row in runtime_doc['libraries'].items():
        ct.require(str(Path(row['load_path']).resolve()) == path, 'Runtime loader alias resolution changed')
    legacy_pins = ct.load(HERE / 'pins.json')['pins']
    ct.require(len(legacy_pins) == 702 and ct.sha(HERE / 'pins.json') == ct.load(HERE / 'prepared.json')['artifact_pins'][str(HERE / 'pins.json')],
        'Frozen 702-pin manifest changed')
    environment = case['environment']
    active = set(actual_imports) | set(actual_assets) | set(runtime_pins)
    production = ct.load(ct.absolute(case['source_pin_manifest']))
    excluded = {p:'not imported/executed by frozen TP1 storage workload' for p in legacy_pins
        if p not in active and p not in production}
    ct.require(set(environment) == {'CUBLAS_WORKSPACE_CONFIG', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
        'FERRULE_LOG', 'FERRULE_LOG_FORMAT', 'LD_LIBRARY_PATH'} and environment['CUDA_VISIBLE_DEVICES'] == '0'
        and environment['NVIDIA_VISIBLE_DEVICES'] == '0' and environment['LD_LIBRARY_PATH'],
        'TP1 loader environment differs')
    return dict(schema='ferrule.paper.storage-demand.tp1-finalized-pins.v1',
        production_source_manifest=dict(path=case['source_pin_manifest'], count=597,
            sha256=ct.sha(ct.absolute(case['source_pin_manifest'])),
            revalidated='597 declared hashes match verified compiled-source archive manifest; current worktree content not rehashed'),
        legacy_preparation_pins=dict(path=str(HERE / 'pins.json'), count=702, sha256=ct.sha(HERE / 'pins.json'),
            role='historical CPU preparation receipt; not a blanket paper-workflow dependency'),
        actual_tp1_import_pins=actual_imports, runtime_library_pins=runtime_pins,
        case_asset_pins=actual_assets,
        excluded_legacy_workflow_pins=excluded,
        out_of_scope='TP2 qualification scripts and unrelated paper documents are not current TP1 gates',
        binary=dict(path=str(ct.absolute(case['binary'])), sha256=case['binary_sha256'],
            source_directory_writable=bool(ct.absolute(case['binary']).parent.stat().st_mode & 0o222),
            readonly_stage_required=True),
        loader_environment=environment,
        counts=dict(production_source=597, legacy_preparation=702, runtime_libraries=12,
            actual_tp1_imports=len(actual_imports), case_assets=len(actual_assets)))


def repeated_preparation_contracts(case, refs, help_receipt):
    rows = schedule(case, refs)
    def rejects(action):
        try:
            action()
        except (ValueError, KeyError, TypeError):
            return True
        return False
    def exact_history(actual, expected):
        return isinstance(actual, list) and len(actual) == 8 and all(type(v) is int for v in actual) and actual == expected
    bad_payload = dict(rows[1]['payload'], temperature=float('nan'))
    unknown_payload = dict(rows[1]['payload'], unrecognized_storage_option=1)
    oversized = dict(rows[1], max_tokens=129)
    tests = dict(
        current_binary_pin=ct.sha(ct.absolute(case['binary'])) == case['binary_sha256'],
        handoff_pin=ct.sha(ct.absolute(case['handoff'])) == case['handoff_sha256'],
        help_passed=help_receipt['returncode'] == 0,
        three_generation8_references=len(refs) == 3 and all(len(r['reference_ids']) == 8 for r in refs),
        context_bounds=all(r['input_tokens'] + r['max_tokens'] <= 128 for r in rows),
        exact_history_accepts_reference=exact_history(list(refs[0]['reference_ids']), refs[0]['reference_ids']),
        exact_history_rejects_missing=not exact_history(None, refs[0]['reference_ids']),
        exact_history_rejects_short=not exact_history(refs[0]['reference_ids'][:7], refs[0]['reference_ids']),
        physical_missing_is_null=case['physical_host_cap_bytes'] is None and counter_summary([])['physical_nas_bytes'] is None,
        full_schedule_blocked=rejects(lambda: schedule(case, refs + refs)),
        finite_greedy=all(request_contract(r['payload'], r) is None for r in rows),
        reject_nonfinite=rejects(lambda: request_contract(bad_payload, rows[1])),
        reject_unknown=rejects(lambda: request_contract(unknown_payload, rows[1])),
        reject_oversize=rejects(lambda: request_contract(http.make_payload('qwen35-storage-demand-paper', oversized['prompt'], 129, 1), oversized)),
        reject_changed_pin=rejects(lambda: ct.check_pins({str(HERE / 'case-middle.json'):'0' * 64})))
    ct.require(len(tests) == 15 and all(tests.values()), 'Repeated CPU preparation contracts failed')
    return tests


def finalize_cpu():
    case, refs = frozen()
    checks = cpu_contracts(case, refs)
    scope = finalized_pin_scope(case)
    for name in ('prepared.json', 'pins.json', 'cpu-checks.json', 'serve-help.json', 'semantics-tests.json', 'case.json'):
        ct.require(not (FINALIZED / name).exists(), 'CPU revision is append-only; no retry/overwrite')
    staged = ct.stage_binary(ct.absolute(case['binary']), FINALIZED)
    ct.validate_staged_binary(staged, case['binary_sha256'])
    cpu_env = dict(base.cpu_environment(), LD_LIBRARY_PATH=case['environment']['LD_LIBRARY_PATH'],
        CUBLAS_WORKSPACE_CONFIG=case['environment']['CUBLAS_WORKSPACE_CONFIG'])
    help_argv = [staged, 'serve', '--help']
    help_result = subprocess.run(ct.isolated_argv(help_argv, cpu_env), cwd=ct.ROOT,
        capture_output=True, text=True, timeout=15)
    flags = case['argv'][3::2]
    values = dict(zip(flags, case['argv'][4::2], strict=True))
    expected_limits = {'--cuda-device-budget-mb':'7168', '--cuda-workspace-mb':'2048',
        '--ctx-size':'128', '--max-tensor-mb':'1024', '--cuda-expert-device-entries':'64',
        '--cuda-expert-device-bytes':'335544320', '--cuda-expert-scratch-mb':'8',
        '--max-active-sequences':'2', '--max-batch-tokens':'8', '--prefill-chunk-size':'8',
        '--expert-prewarm':'lazy', '--backend':'cuda', '--engine':'resident', '--model-profile':'qwen35-mixed-bf16'}
    ct.require(all(values.get(k) == v for k,v in expected_limits.items())
        and 335544320 > 8 << 20 and case['model'] == '/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8',
        'Frozen software budgets/model/profile differ')
    ct.require(help_result.returncode == 0 and all(flag in help_result.stdout for flag in flags),
        'Same-SHA readonly stage GPU-hidden help/flags check failed')
    help_receipt = dict(argv=help_argv, environment=cpu_env, returncode=help_result.returncode,
        stdout=help_result.stdout, stderr=help_result.stderr, gpu_hidden=True,
        invocation_scope='CLI help only; no CUDA admission, model load or GPU query')
    repeated = repeated_preparation_contracts(case, refs, help_receipt)
    runtime_environment_contract(dict(case['environment']), case['environment'])
    environment_rejections = []
    for key, value in (('CUDA_VISIBLE_DEVICES', '0,1'), ('NVIDIA_VISIBLE_DEVICES', '1'),
                       ('LD_LIBRARY_PATH', '/invalid/loader'), ('LD_PRELOAD', '/invalid/inject.so')):
        try:
            runtime_environment_contract(dict(case['environment'], **{key:value}), case['environment'])
        except ValueError:
            environment_rejections.append(key)
        else:
            raise ValueError('Environment negative guard accepted: ' + key)
    semantics_argv = [sys.executable, '-S', '-B', '-m', 'unittest', 'discover',
        '-s', str(HERE), '-p', 'test_run_middle*.py', '-v']
    semantics = subprocess.run(ct.isolated_argv(semantics_argv, base.cpu_environment()), cwd=ct.ROOT,
        capture_output=True, text=True, timeout=15)
    ct.require(semantics.returncode == 0 and 'Ran 6 tests' in semantics.stderr and semantics.stderr.rstrip().endswith('OK'),
        'Five resource semantics tests and target startup integration test failed')
    checks.update(repeated_preparation_contracts=repeated, repeated_contract_count=15,
        wrapper_check_count=len(checks['checks']), semantics_test_count=5,
        target_integration_test_count=1, total_cpu_test_count=6,
        environment_negative_guards=environment_rejections, runtime_library_pin_count=12,
        actual_cuda_environment_observed=False, actual_cuda_library_maps_observed=False,
        shard_stat_snapshots_verified=14, shard_header_bytes_reread=0, tensor_payload_bytes_read=0,
        readonly_stage_validated=True, actual_generated_ids=None,
        source_scope='compiled-source archive matched 597 production pins; working-tree source contents not rehashed')
    staged_argv = [staged, *case['argv'][1:]]
    finalized_case = dict(schema='ferrule.paper.storage-demand.tp1-finalized-case.v1',
        parent_case_sha256=ct.sha(HERE / 'case-middle.json'), purpose=PURPOSE,
        argv=staged_argv, environment=case['environment'], topology=case['topology'], model=case['model'],
        request_count=4, schedule=schedule(case, refs), expert_prewarm='lazy',
        expert_entries=64, expert_device_bytes=335544320, expert_scratch_mib=8,
        warmup_requests=0, qa_extension_authorized=False, actual_generated_ids=None,
        resource_contract_complete='unknown', physical_nas_read='unknown', os_pagecache='uncontrolled',
        cache_policy='software GPU expert cache; not a physical host cap or disk-necessity claim')
    ct.write(FINALIZED / 'pins.json', scope)
    ct.write(FINALIZED / 'cpu-checks.json', checks)
    ct.write(FINALIZED / 'serve-help.json', help_receipt)
    ct.write(FINALIZED / 'semantics-tests.json', dict(argv=semantics_argv, returncode=semantics.returncode,
        stdout=semantics.stdout, stderr=semantics.stderr, tests=6,
        test_counts=dict(resource_semantics=5, target_integration=1),
        source_pins={str(HERE / name):scope['actual_tp1_import_pins'][str(HERE / name)]
            for name in ('test_run_middle_semantics.py', 'test_run_middle_target.py')},
        fixture_scope='mock owner samples and Popen target startup; no actual GPU evidence'))
    ct.write(FINALIZED / 'case.json', finalized_case)
    final = dict(schema='ferrule.paper.storage-demand.tp1-finalized-cpu-prepared.v1',
        revision='storage-demand-tp1-middle-finalized-cpu-20261013-02',
        status='CPU_PREPARED_EXECUTION_BLOCKED', parent_prepared_sha256=PREPARED_SHA,
        binary=dict(path=staged, sha256=case['binary_sha256'], source='same-SHA readonly stage of new-36e6 release artifact',
            readonly_stage_validated=True, original_source_directory_not_immutable=True),
        model=case['model'], topology=case['topology'], argv=staged_argv, environment=case['environment'],
        fixed_limits=dict(device_budget_mib=7168, workspace_mib=2048, context=128, max_tensor_mib=1024,
            expert_entries=64, expert_device_bytes=335544320, expert_scratch_mib=8,
            max_active_sequences=2, max_batch_tokens=8, prefill_chunk_size=8),
        workload=dict(request_count=4, exact=['Hello max_tokens=2', 'qa max_tokens=8',
            'arithmetic max_tokens=8', 'chinese max_tokens=8'], qa_extension_executed=False,
            warmups=0, measured_canaries=3, generation_ids_exposed=False),
        claims=dict(physical_nas_read='unknown', os_pagecache='uncontrolled',
            software_gpu_expert_cache='logical regulated application cache', host_physical_cap=None,
            current_gpu_quality='pending; CPU flags/help and source metadata are not GPU quality'),
        cpu_checks=checks, pin_scope=scope,
        artifact_pins=ct.inventory([FINALIZED / name for name in ('pins.json', 'cpu-checks.json',
            'serve-help.json', 'semantics-tests.json', 'case.json')] + [Path(staged)]),
        gpu_queries=0, gpu_locks=0, gpu_runs=0,
        blocked=['GPU source-bound admission and actual checkpoint load pending',
            'read counters remain logical reader events, not physical NAS IO',
            'current canary text/usage and token IDs not observed',
            'normal exit cannot establish resource_contract_complete without final owner evidence'])
    final_path = FINALIZED / 'prepared.json'
    pins_path = FINALIZED / 'pins.json'
    checks_path = FINALIZED / 'cpu-checks.json'
    sha_path = FINALIZED / 'prepared.sha256'
    ct.write(final_path, final)
    for name in ('prepared.json', 'pins.json', 'cpu-checks.json', 'serve-help.json', 'semantics-tests.json', 'case.json'):
        (FINALIZED / name).chmod(0o400)
    with sha_path.open('x') as stream:
        stream.write(ct.sha(final_path) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return dict(status=final['status'], prepared=str(final_path), prepared_sha256=ct.sha(final_path),
        pins=str(pins_path), checks=str(checks_path), gpu_queries=0, gpu_locks=0, gpu_runs=0,
        request_count=4, qa_extension_executed=False, repeated_cpu_contracts=15,
        wrapper_checks=len(checks['checks']), semantics_tests=5, target_integration_tests=1, total_cpu_tests=6,
        environment_negative_guards=4,
        runtime_library_pins=12, shard_stat_snapshots=14, stage_sha256=case['binary_sha256'])


def validate_finalized(path):
    path = Path(path).resolve()
    ct.require(path == FINALIZED / 'prepared.json' and path.is_file() and not path.is_symlink()
        and path.stat().st_mode & 0o222 == 0, 'Need this immutable finalized CPU revision')
    ct.require(ct.sha(path) == (FINALIZED / 'prepared.sha256').read_text().strip(), 'Finalized CPU revision SHA differs')
    final = ct.load(path)
    ct.require(final['status'] == 'CPU_PREPARED_EXECUTION_BLOCKED' and final['parent_prepared_sha256'] == PREPARED_SHA
        and final['workload']['request_count'] == 4 and final['workload']['warmups'] == 0
        and final['workload']['qa_extension_executed'] is False, 'Finalized four-request contract differs')
    scope = final['pin_scope']
    expected = {str((ct.ROOT / p).resolve()) for p in TP1_IMPORTS}
    expected.update(str((HERE / p).resolve()) for p in ('run_middle.py', 'test_run_middle_semantics.py', 'test_run_middle_target.py'))
    ct.require(set(scope['actual_tp1_import_pins']) == expected, 'Finalized TP1 import closure differs')
    for pins in (final['artifact_pins'], scope['actual_tp1_import_pins'], scope['case_asset_pins'], scope['runtime_library_pins']):
        ct.check_pins(pins)
    ct.validate_staged_binary(final['binary']['path'], final['binary']['sha256'])
    case = ct.load(HERE / 'case-middle.json')
    ct.require(final['binary']['sha256'] == case['binary_sha256'] and final['argv'][1:] == case['argv'][1:]
        and final['environment'] == case['environment'] and final['topology'] == case['topology'], 'Finalized staged case differs')
    return final


def run(note, granted, finalized_prepared):
    ct.require(granted is True and isinstance(note, str) and bool(note.strip()), 'Explicit owner authorization required')
    final = validate_finalized(finalized_prepared)
    case, refs = frozen()
    checks = cpu_contracts(case, refs)
    RUN.parent.mkdir(exist_ok=True)
    RUN.mkdir(exist_ok=False)
    report = dict(status='blocked', purpose=PURPOSE, owner_note=note, request_count=4,
        prepared_sha256=PREPARED_SHA, n=1, full_current_math_pass=False, exact_token_pass=None,
        physical_host_cap_bytes=None, physical_nas_bytes=None, PssAnon=None, small_big_authorized=False)
    ct.write(RUN / 'cpu-contracts.json', checks)
    ct.stage_binary(final['binary']['path'], RUN)
    ct.write(RUN / 'current-runtime-pins.json', cpu_runtime_pins(case))
    (RUN / 'current-runtime-pins.json').chmod(0o400)
    try:
        sample = discover(note)
        uuid = sample['uuid']
        ct.require(sample['index'] == 0 and not sample['apps'] and len(sample['devices']) == 8
            and all(d['gpu_bytes'] == 0 and d['utilization'] == 0 for d in sample['devices']), 'Independent all-eight zero preflight failed')
        cfg = config(sample['total_bytes'])
        ct.write(RUN / 'preflight.json', sample)
        (RUN / 'preflight.json').chmod(0o400)
        with ct.exclusive_owner_lock(uuid) as lock:
            ct.validate_owner_lock(lock, uuid, os.getpid())
            idle = observer.idle(cfg, (uuid,), note)
            ct.write(RUN / 'lock-preflight.json', idle)
            (RUN / 'lock-preflight.json').chmod(0o400)
            strict_idle(idle, uuid)
            ct.write(RUN / 'baseline.json', idle['baseline'])
            (RUN / 'baseline.json').chmod(0o400)
            payload = dict(directory=str(RUN), purpose=PURPOSE, request_count=4, uuid=uuid, gpu_mask='0',
                actual_total_bytes=sample['total_bytes'], config=cfg, baseline=idle['baseline'],
                baseline_sha256=ct.sha(RUN / 'baseline.json'), owner_lock=lock, granted=True, owner_note=note,
                runtime_pins_sha256=ct.sha(RUN / 'current-runtime-pins.json'),
                prepared_sha256=PREPARED_SHA, wrapper_sha256=ct.sha(__file__),
                finalized_prepared=str(Path(finalized_prepared).resolve()), finalized_prepared_sha256=ct.sha(finalized_prepared))
            payload, seal = guard.seal(payload, RUN)
            env = dict(base.cpu_environment(), FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
            argv = ct.isolated_argv([sys.executable, '-S', '-B', str(Path(__file__).resolve()), '--worker', seal], env)
            print('START authorized TP1 middle only; exactly Hello2 + 3 canaries; no retries', flush=True)
            try:
                receipt = base.supervisor.run_process(argv, 1100, payload)
            except BaseException as error:
                receipt = getattr(error, 'supervisor_result', {})
                receipt.setdefault('error', str(error))
            ct.write(RUN / 'supervisor.json', receipt)
            report['authenticated_drained'] = ct.drained(receipt)
            report['supervision_complete'] = report['authenticated_drained']
            report['supervisor_success'] = ct.successful_process(receipt)
            report['case'] = ct.load(RUN / 'case.json') if (RUN / 'case.json').exists() else dict(status='missing')
            report['resource_contract_complete'] = report['case'].get('shutdown_contract', {}).get('resource_contract_complete', 'unknown')
            frozen()
            ct.check_pins(ct.load(RUN / 'current-runtime-pins.json')['pins'])
            report['pins_post_supervisor'] = 'sealed_702_history_and_597_build_archive_plus_explicit_TP1_import_assets_and_runtime_pins_verified'
            post = observer.idle(cfg, (uuid,), note, idle['baseline'])
            ct.write(RUN / 'post-idle.json', post)
            strict_idle(post, uuid)
            report['all_eight_post_idle'] = True
            report['status'] = ('custody_unknown' if not report['authenticated_drained'] else
                'complete' if report['supervisor_success'] and report['case']['status'] == 'complete' else 'failed')
        report['selected_local_lock_released'] = True
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    report['artifact_pins'] = {str(p):ct.sha(p) for p in sorted(RUN.rglob('*')) if p.is_file()}
    ct.write(RUN / 'report.json', report)
    print(json.dumps(dict(directory=str(RUN), status=report['status'], actual_requests=len(report.get('case', {}).get('requests', [])),
        small_big_may_be_considered=report['status'] == 'complete', no_further_runs=True)), flush=True)
    return 0 if report['status'] == 'complete' else 1


def main():
    ct.require_cpu_python()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu-contracts', action='store_true')
    mode.add_argument('--finalize-cpu', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--worker')
    parser.add_argument('--owner-note')
    parser.add_argument('--finalized-prepared', type=Path)
    parser.add_argument('--sole-gpu-owner-granted', action='store_true')
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.cpu_contracts:
        print(json.dumps(cpu_contracts(), indent=2), flush=True)
        return 0
    if args.finalize_cpu:
        print(json.dumps(finalize_cpu(), indent=2), flush=True)
        return 0
    ct.require(args.sole_gpu_owner_granted and args.finalized_prepared is not None,
        'Explicit --sole-gpu-owner-granted and --finalized-prepared are required')
    return run(args.owner_note, args.sole_gpu_owner_granted, args.finalized_prepared)


if __name__ == '__main__':
    raise SystemExit(main())
