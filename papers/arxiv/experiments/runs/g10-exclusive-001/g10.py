"""Append-only G10 experiment; import existing M7 primitives without mutating them."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
sys.path.insert(0, str(EXPERIMENTS))
import runner as base
ct, telemetry, guard, http = base.ct, base.telemetry, base.launch_guard, base.http
PREPARED = EXPERIMENTS / 'runs/m7-prepared-exclusive-003/prepared.json'
UUID = 'GPU-b980f22b-58a3-fc88-21c1-ae86053d8535'
NOTE = 'User delegated sole ownership of all eight idle GPUs for NEW G10; no foreign signals, no fallback/retry'
BINARY_SHA = '219d8b77bbeceecedf83272b880cd3a8175db7bf5bedea5dcfc537888461ff80'
REFERENCE = ct.ROOT / 'target/validation/bf16-fresh-quality-20261007/oracle/manifest.json'
REFERENCE_SHA = '102c798baeb5200bc8ed8386eabed6146870b3b6da2a90fc6010677041352c7e'
INPUT = REFERENCE.parent.parent / 'INPUT.json'
STAGES = [('feasibility-ferrule', 'ferrule', 'feasibility'),
          ('feasibility-vllm', 'vllm', 'feasibility'),
          ('canary-ferrule', 'ferrule', 'canary'),
          ('canary-vllm', 'vllm', 'canary'),
          ('performance-A1', 'ferrule', 'performance'),
          ('performance-B1', 'vllm', 'performance'),
          ('performance-B2', 'vllm', 'performance'),
          ('performance-A2', 'ferrule', 'performance')]


def config():
    c = copy.deepcopy(ct.config(ct.EXCLUSIVE_PATH))
    c['schema'] = 'ferrule.paper.g10.config.v1'
    c['protocol']['protocol_id'] = 'g10-device-exclusive-local-custody-v1'
    c['resources']['gpu_target_bytes'] = 10 * (1 << 30)
    c['resources']['gpu_policy'] = '10 GiB observed whole-device ceiling; sampled, not a hard partition; vLLM planning remains M7'
    c['ferrule_flags']['--cuda-device-budget-mb'] = '10240'
    c['workloads'] = dict(feasibility=dict(max_tokens=2, requests=1, warmup=0),
                          canary=dict(max_tokens=8, requests=3, warmup=0),
                          performance=dict(max_tokens=8, requests=2, warmup=1))
    return c


def references():
    ct.require(ct.sha(REFERENCE) == REFERENCE_SHA, 'Independent manifest hash differs')
    ref = ct.load(REFERENCE)
    ct.require(ct.sha(INPUT) == ref['input_sha256'], 'Reference input differs')
    inp = ct.load(INPUT)
    pins = {str(REFERENCE): REFERENCE_SHA, str(INPUT): ct.sha(INPUT)}
    model = ct.absolute(config()['model'])
    for name, expected in inp['tokenizer']['files'].items():
        ct.require(ct.sha(model / name) == expected, 'Reference tokenizer differs')
        pins[str(model / name)] = expected
    tokenizer = ct.load(model / 'tokenizer.json')
    ct.require(tokenizer['model']['type'] == 'BPE' and tokenizer['decoder']['type'] == 'ByteLevel',
               'Expected byte-level BPE tokenizer')
    vocab = {value: key for key, value in tokenizer['model']['vocab'].items()}
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = byte_values.copy()
    n = 0
    for b in range(256):
        if b not in byte_values:
            byte_values.append(b)
            characters.append(256 + n)
            n += 1
    decode_byte = dict(zip(map(chr, characters), byte_values))
    def decode(ids):
        return bytes(decode_byte[ch] for token in ids for ch in vocab[token]).decode('utf-8')
    rows = []
    for item, manifest_case in zip(inp['cases'], ref['cases'], strict=True):
        report_path = REFERENCE.parent / manifest_case['report']
        ct.require(ct.sha(report_path) == manifest_case['report_sha256'], 'Reference report differs')
        pins[str(report_path)] = ct.sha(report_path)
        report = ct.load(report_path)
        ct.require(item['id'] == manifest_case['id'] == report['id'], 'Reference order differs')
        ct.require(decode(item['input_ids']) == item['prompt'] == report['prompt'], 'Prompt decode differs')
        ct.require(decode(report['generated_ids']) == report['generated_text'], 'Reference full decode differs')
        rows.append(dict(id=item['id'], prompt=item['prompt'], input_tokens=len(item['input_ids']),
                         reference_ids=report['generated_ids'][:8], reference_text=decode(report['generated_ids'][:8])))
    return rows, pins


def prepare():
    ct.require_cpu_python()
    m = base.validate_manifest(PREPARED)
    ct.require(m['binary_sha256'] == BINARY_SHA, 'Wrong prepared release')
    rows, refs = references()
    c = config()
    ct.require(c['vllm_flags'] == m['config']['vllm_flags'], 'vLLM flags changed')
    ct.write(HERE / 'config.json', c)
    ct.write(HERE / 'canaries.json', rows)
    receipt = dict(schema='ferrule.paper.g10.prepared.v1', config=c, config_sha256=ct.digest(c),
                   original_prepared=str(PREPARED), original_prepared_sha256=ct.sha(PREPARED),
                   original_manifest_sha256=m['manifest_sha256'], binary_sha256=BINARY_SHA,
                   reference_sha256=REFERENCE_SHA, reference_pins=refs,
                   driver_sha256=ct.sha(__file__), stages=STAGES, gpu_uuid=UUID, owner_note=NOTE,
                   canaries_sha256=ct.sha(HERE / 'canaries.json'),
                   baseline_preservation_pins={str(p): ct.sha(p) for p in EXPERIMENTS.rglob('*')
                                               if p.is_file() and not p.is_relative_to(HERE) and p.suffix != '.lock'},
                   fairmatch=False, exact_speedup=None, identity_mode='device_exclusive_local_custody',
                   owned_gpu_attribution='unknown', hard_memory_partition=False,
                   performance='One short ABBA fresh-process block; Hello max8, 1 warmup then 2 measured per process; descriptive only',
                   canary_gate='Exact visible 8-token reference prefix and usage counts; not logits/routes/state arithmetic equivalence',
                   no_hf_regeneration=True, no_config_fallback=True, no_retry=True)
    ct.write(HERE / 'prepared.json', receipt)
    print(json.dumps(dict(status='prepared',config_sha256=receipt['config_sha256'],canaries=rows), ensure_ascii=False), flush=True)


def validate():
    p = ct.load(HERE / 'prepared.json')
    ct.require(ct.sha(__file__) == p['driver_sha256'], 'G10 driver changed')
    ct.require(ct.load(HERE / 'config.json') == config() == p['config'], 'G10 config changed')
    ct.require(ct.digest(config()) == p['config_sha256'], 'G10 config digest differs')
    ct.require(ct.sha(PREPARED) == p['original_prepared_sha256'], 'Original prepared changed')
    ct.require(ct.sha(HERE / 'canaries.json') == p['canaries_sha256'], 'Canary inputs changed')
    ct.check_pins(p['reference_pins'])
    m = base.validate_manifest(PREPARED)
    ct.require(m['binary_sha256'] == BINARY_SHA and m['manifest_sha256'] == p['original_manifest_sha256'], 'Release identity differs')
    return p, m


def server_spec(engine, m):
    spec = ct.server_spec(config(), engine, m['binary'], UUID)
    if engine == 'vllm':
        spec['argv'][0] = m['vllm_staged']
    return spec


def request_block(directory, spec, mode):
    c = config()
    workload = c['workloads'][mode]
    output = directory / 'http.json'
    args = ['--engine', spec['engine'], '--url', spec['url'], '--model', spec['model'],
            '--seed', '1', '--max-tokens', str(workload['max_tokens']), '--concurrency', '1',
            '--requests', str(workload['requests']), '--warmup', str(workload['warmup']), '--repeat', '1',
            '--timeout', '120', '--window-kind', 'fixed-work', '--measurement-seconds', '360',
            '--metadata', str(directory / 'metadata.json'), '--output', str(output)]
    if mode == 'canary':
        corpus = [{k: row[k] for k in ('id', 'prompt', 'input_tokens')} for row in ct.load(HERE / 'canaries.json')]
        ct.write(directory / 'prompts.json', corpus)
        args += ['--prompts-json', str(directory / 'prompts.json')]
    else:
        args += ['--prompt', 'Hello', '--input-tokens', '1']
    code = http.main(args)
    report = ct.load(output)
    ct.require(code == 0 and report['status'] == 'complete', 'HTTP incomplete or failed')
    rows = [row for run in report['runs'] for row in run['requests']]
    ct.require(len(rows) == workload['requests'] and len(report['warmups']) == workload['warmup'], 'Request count differs')
    ct.require(all(row['status'] == 'success' and row['input_length_matches'] is True
                   and type(row['actual_output_tokens']) is int and 0 < row['actual_output_tokens'] <= workload['max_tokens']
                   and row['finish_reason'] in ('length', 'stop') for row in rows + report['warmups']), 'HTTP counts/status invalid')
    if mode != 'canary':
        return dict(status='pass', quality_claim=False)
    checks = []
    for row, ref in zip(rows, ct.load(HERE / 'canaries.json'), strict=True):
        checks.append(dict(id=ref['id'], expected_text=ref['reference_text'], observed_text=row['text'],
                           reference_token_ids=ref['reference_ids'], observed_token_ids=None,
                           actual_input_tokens=row['actual_input_tokens'], actual_output_tokens=row['actual_output_tokens'],
                           pass_visible_prefix_and_count=row['text'] == ref['reference_text'] and row['actual_output_tokens'] == 8))
    return dict(status='pass' if all(x['pass_visible_prefix_and_count'] for x in checks) else 'fail',
                reference_sha256=REFERENCE_SHA, checks=checks, independent_reference=True,
                limitation='Visible text + usage only; token IDs unavailable over maintained SSE client; not a full numerical gate')


def worker(launch_hash):
    ct.require_cpu_python()
    payload = json.load(sys.stdin)
    authenticated = guard.authenticate(payload, launch_hash)
    p, m = validate()
    ct.require(payload['g10_prepared_sha256'] == ct.sha(HERE / 'prepared.json'), 'G10 prepared changed')
    ct.validate_owner_lock(payload['owner_lock'], UUID, payload['coordinator']['pid'])
    directory = Path(payload['directory'])
    ct.require(directory.parent == HERE and payload['stage'] in [list(s) for s in STAGES], 'Unknown G10 stage')
    ct.require(directory.name == payload['stage'][0], 'Stage path mismatch')
    _, engine, mode = payload['stage']
    spec = server_spec(engine, m)
    ct.require(payload['spec'] == spec, 'Server spec changed')
    telemetry.authorize(True, UUID, NOTE, 'device_exclusive_local_custody')
    ct.require(os.getpid() == os.getpgrp(), 'Worker session not owned')
    result = dict(status='failed', fairmatch=False, exact_speedup=None, started_monotonic=time.monotonic())
    server, monitor = None, None
    try:
        metadata = base.metadata(m, engine, UUID)
        metadata.update(resources=config()['resources'], source={'original_manifest_sha256':m['manifest_sha256'], 'g10_prepared_sha256':payload['g10_prepared_sha256']},
                        quality_gate={'status':'independent_canary_required'}, protocol_id=config()['protocol']['protocol_id'])
        ct.write(directory / 'metadata.json', metadata)
        base.require_free_port(spec)
        with (directory / 'server.log').open('xb') as log:
            expected = m['binary_sha256'] if engine == 'ferrule' else m['vllm_binary_sha256']
            result['server_launch_monotonic'] = time.monotonic()
            with ct.pinned_executable(spec['argv'][0], expected) as fd:
                server = subprocess.Popen(ct.isolated_argv(base.bound_server_argv(spec, fd), spec['environment']),
                                          pass_fds=(fd,), cwd=ct.ROOT, stdout=log, stderr=subprocess.STDOUT)
            monitor = telemetry.Monitor(directory, server, '/usr/bin/nvidia-smi', UUID, NOTE, config(), authenticated)
            monitor.start()
            ready = http.wait_ready(spec, base.StartupGuard(server, monitor), 900)
            result['startup_seconds'] = ready - result['server_launch_monotonic']
            ct.require(not monitor.failed.is_set(), 'Resource monitor failed during startup')
            result['request_gate'] = request_block(directory, spec, mode)
            ct.require(not monitor.failed.is_set(), 'Resource monitor failed during requests')
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
                # Leave eight seconds of the 120s cleanup budget to the existing subreaper.
                result['shutdown'] = http.shutdown_owned_group(server, 112)
                if result['shutdown']['timed_out']:
                    result['status'] = 'cleanup_timeout'
            except BaseException as error:
                result.update(status='cleanup_unknown', cleanup_error=str(error))
        log_path = directory / 'server.log'
        if log_path.exists():
            log = log_path.read_text(errors='replace')
            result['runtime'] = ct.runtime_evidence(engine, log)
            result['runtime']['log_sha256'] = ct.sha(log_path)
            if result['runtime']['contradictions'] and result['status'] == 'complete':
                result['status'] = 'runtime_contradiction'
        result['wall_seconds'] = time.monotonic() - result['started_monotonic']
        ct.write(directory / 'case.json', result)
    return 0 if result['status'] == 'complete' else 1


def case(stage, m, lock):
    directory = HERE / stage[0]
    directory.mkdir(exist_ok=False)
    before = telemetry.idle('/usr/bin/nvidia-smi', UUID, True, NOTE, config())
    ct.write(directory / 'before.json', before)
    ct.require(before['status'] == 'verified', 'All-eight pre-case idle missing')
    payload = dict(directory=str(directory), stage=stage, spec=server_spec(stage[1], m), owner_lock=lock,
                   g10_prepared_sha256=ct.sha(HERE / 'prepared.json'))
    ct.write(directory / 'launch.json', payload)
    payload, launch_hash = guard.seal(payload, directory)
    env = dict(base.cpu_environment(), FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
    argv = ct.isolated_argv([sys.executable, '-S', '-B', str(Path(__file__).resolve()), '--worker', launch_hash], env)
    print('START ' + stage[0], flush=True)
    receipt = base.supervisor.run_process(argv, 1600, payload)
    ct.write(directory / 'supervisor.json', receipt)
    after = telemetry.idle('/usr/bin/nvidia-smi', UUID, True, NOTE, config())
    ct.write(directory / 'after.json', after)
    observation = ct.load(directory / 'case.json') if (directory / 'case.json').exists() else {'status':'missing'}
    result = dict(stage=stage, observation=observation, supervisor=receipt, after=after,
                  eligible_observation=ct.successful_process(receipt) and observation['status'] == 'complete'
                                       and observation.get('resources',{}).get('status') == 'observed_within_targets'
                                       and after['status'] == 'verified', fairmatch=False, exact_speedup=None)
    ct.write(directory / 'assessment.json', result)
    print(json.dumps(dict(stage=stage[0], status=observation['status'], error=observation.get('error'),
                          resources=observation.get('resources'), drained=ct.drained(receipt), after=after['status'])), flush=True)
    ct.require(ct.drained(receipt) and after['status'] == 'verified', 'Fail-stop: custody/release unknown')
    return result


def run():
    ct.require_cpu_python()
    p, m = validate()
    report = dict(schema='ferrule.paper.g10.execution.v1', status='incomplete', cases=[],
                  identity_mode='device_exclusive_local_custody', owned_gpu_attribution='unknown',
                  owned_process_gpu_peak_bytes=None, absence_of_foreign_interference_proven=False,
                  performance_basis='exclusive_operator_allocation_assumption', fairmatch=False, exact_speedup=None,
                  prepared_sha256=ct.sha(HERE / 'prepared.json'), performance_status='not_run',
                  arithmetic_match=False, no_foreign_kill=True, no_config_fallback=True, no_retry=True)
    with ct.exclusive_owner_lock(UUID) as lock:
        report['owner_lock'] = lock
        try:
            for stage in STAGES:
                if stage[0] == 'canary-ferrule':
                    ct.require(all(c['eligible_observation'] for c in report['cases']), 'Feasibility failed; canary blocked')
                if stage[0] == 'performance-A1':
                    ct.require(all(c['eligible_observation'] for c in report['cases']), 'Independent canary failed; performance blocked')
                    report['performance_status'] = 'incomplete'
                observation = case(stage, m, lock)
                report['cases'].append(observation)
                if stage[2] == 'performance':
                    ct.require(observation['eligible_observation'], 'Performance case failed; no retry')
            report.update(status='complete', performance_status='complete_descriptive_only')
        except BaseException as error:
            report['error'] = type(error).__name__ + ': ' + str(error)
        finally:
            try:
                report['final_release'] = telemetry.idle('/usr/bin/nvidia-smi', UUID, True, NOTE, config())
                ct.write(HERE / 'final-release.json', report['final_release'])
                ct.check_pins(p['baseline_preservation_pins'])
                report['baseline_preservation'] = 'all_existing_experiment_files_unchanged'
            except BaseException as error:
                report.update(status='failed', finalization_error=str(error))
            report['artifact_pins'] = {str(f): ct.sha(f) for f in HERE.rglob('*') if f.is_file()}
            ct.write(HERE / 'execution.json', report)
    ct.write(HERE / 'lock-release.json', dict(status='released', gpu_uuid=UUID, local_lock_released=True,
                                             all_gpu_idle=report.get('final_release',{}).get('status') == 'verified'))
    print(json.dumps(dict(status=report['status'], error=report.get('error'), performance_status=report['performance_status'],
                          final_release=report.get('final_release',{}).get('status'), exact_speedup=None)), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    ct.require(os.environ.get('HOME') == '/home/admin', 'Use original prepared HOME=/home/admin')
    if sys.argv[1:] == ['prepare']:
        prepare()
    elif sys.argv[1:] == ['run']:
        raise SystemExit(run())
    elif len(sys.argv) == 3 and sys.argv[1] == '--worker':
        raise SystemExit(worker(sys.argv[2]))
    else:
        raise SystemExit('Need prepare or run; sealed worker entry only')
