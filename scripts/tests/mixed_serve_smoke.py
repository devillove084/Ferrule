"""Opt-in Mixed public-CLI lifecycle test; prepare/check are CPU-only.

Reuses the maintained HTTP parser, readiness/shutdown helpers and authenticated
owned subreaper. No benchmark semantics, inference APIs or installers. Explicit execution uses
read-only GPU telemetry to gate startup, restart and final release.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import bench_ferrule_vllm as http
import test_vllm_qwen35 as supervisor

SCRIPT = Path(__file__).resolve()
MODEL = Path('/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8')
PROFILE = 'cuda-hybrid-numeric-fp8-model-bf16-mixed-f32-ssm'
MODEL_ID = 'qwen35-mixed-smoke'
PORT = 18359
CONFIG = ROOT / 'crates/ferrule-cli/tests/support/mixed_serve_smoke.json'
CONTRACT = json.loads(CONFIG.read_text())
LIMITS, FLAGS, POLICY = (CONTRACT[k] for k in ('limits', 'serve_flags', 'policy'))
WORKFLOW = [str(SCRIPT.relative_to(ROOT)), 'scripts/bench_ferrule_vllm.py',
            'scripts/test_vllm_qwen35.py', 'scripts/_owned_subreaper.py', str(CONFIG.relative_to(ROOT))]
require = http.require
write = http.dump_json


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_snapshot():
    paths = [p for base in ('crates', '.cargo') for p in (ROOT / base).rglob('*')
             if p.is_file() and p.suffix in ('.rs', '.cu', '.cuh', '.h', '.cpp', '.c', '.toml', '.lock', '.json')]
    paths += [ROOT / p for p in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml')]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def production_inputs(snapshot):
    # Test-only attention fixtures are not linked into the non-test CLI binary.
    # Retain every other crate/build/config file conservatively, including additions.
    return {p: h for p, h in snapshot.items() if '/tests/' not in p}


def gpu_snapshot():
    def query(fields):
        result = subprocess.run(['nvidia-smi', fields, '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=5, check=True)
        return result.stdout.strip()
    devices = []
    for line in query('--query-gpu=index,uuid,memory.used,memory.total').splitlines():
        index, uuid, used, total = [x.strip() for x in line.split(',')]
        devices.append(dict(index=int(index), uuid=uuid, used_mib=int(used), total_mib=int(total)))
    require(devices, 'No GPU telemetry; cannot establish quiescence')
    return dict(devices=devices, compute_apps=query('--query-compute-apps=pid,gpu_uuid,used_memory'),
                monotonic=time.monotonic())


def wait_gpu_quiescent(label, seconds=30):
    started, consecutive, observations = time.monotonic(), 0, []
    while time.monotonic() - started < seconds:
        snapshot = gpu_snapshot()
        observations.append(snapshot)
        idle = not snapshot['compute_apps'] and all(d['used_mib'] == 0 for d in snapshot['devices'])
        consecutive = consecutive + 1 if idle else 0
        if consecutive >= 2:
            print('GPU QUIESCENT', label, 'all devices 0 MiB; no compute apps; two observations', flush=True)
            return dict(status='verified', label=label, observations=observations)
        time.sleep(.25)
    raise RuntimeError(f'GPU quiescence {label} not established; no restart/no foreign PID cleanup: {observations[-1:]}')


def serve_argv(binary, model):
    return [str(binary), 'serve', str(model), *[x for item in FLAGS.items() for x in item]]


def prepare(directory):
    directory = Path(directory).resolve()
    build = json.loads((directory / 'build-final-report.json').read_text())
    receipt = json.loads((directory / 'build-final-supervisor.json').read_text())
    require(build['sources_stable'], 'Source changed during build; rebuild before GPU')
    require(receipt['returncode'] == 0 and not receipt['timed_out']
            and receipt['custody']['status'] == 'proven_drained', 'CUDA build did not complete')
    artifacts = [a for a in build['artifacts'] if a.get('executable')
                 and a['target']['name'] == 'ferrule' and 'cuda' in a['features']]
    require(len(artifacts) == 1, 'Need one actual CUDA Ferrule Cargo executable')
    binary = directory / 'ferrule-cuda124-mixed'
    require(sha(binary) == sha(artifacts[0]['executable']), 'Pinned copy differs from resolved Cargo executable')
    env = json.loads((directory / 'build-environment.json').read_text())
    env.update(CUDA_VISIBLE_DEVICES='0', NVIDIA_VISIBLE_DEVICES='0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUBLAS_WORKSPACE_CONFIG=':0:0', FERRULE_LOG='info,ferrule_request=debug', FERRULE_LOG_FORMAT='json')
    plan = dict(schema='ferrule.mixed-public-serve-smoke.v1', status='prepared_cpu_only',
                executable=str(binary), executable_sha256=sha(binary),
                cargo_executable=artifacts[0]['executable'], cargo_profile=artifacts[0]['profile'],
                python=str(Path(sys.executable).resolve()), python_sha256=sha(Path(sys.executable).resolve()),
                runtime_libraries={str(p): sha(p) for p in (
                    Path(env['CUDA_HOME']) / 'lib/libcudart.so.12',
                    Path(env['CUDA_HOME']) / 'lib/libcublas.so.12',
                    Path(env['CUDA_HOME']) / 'lib/libcublasLt.so.12', Path('/usr/lib64/libcuda.so.1'))},
                environment=env, model=str(MODEL), argv=serve_argv(binary, MODEL),
                policy=POLICY, limits=LIMITS, source_sha256=build['source_after'], source_scope='production_inputs_excluding_test_dirs',
                workflow_sha256={p: sha(ROOT / p) for p in WORKFLOW},
                model_metadata_sha256={p: sha(MODEL / p) for p in (
                    'config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json')},
                rounds=2, prompt='Hello', input_tokens=1, normal_tokens=2, cancel_tokens=4,
                quality='Frozen full40 804 metrics PASS; scalar-fix first QA 8 steps PASS reported by owner; fresh attention6 still blocking; lifecycle smoke is not quality acceptance',
                gpu_validation='not_run; attention agent owns GPU execution')
    validate(plan)
    path = directory / 'serve-smoke-plan.json'
    require(not path.exists(), 'Refuse to overwrite a prepared plan')
    write(path, plan)
    print('PREPARED (no GPU/server):', path, '\nExecutable:', binary,
          '\nSHA256:', plan['executable_sha256'], '\nFresh quality: attention6 BLOCKING', flush=True)
    return plan


def validate(plan):
    require(plan.get('schema') == 'ferrule.mixed-public-serve-smoke.v1', 'Unknown smoke schema')
    require(plan['policy'] == POLICY and plan['limits'] == LIMITS, 'Precision/budget/deadline contract changed')
    require(plan['argv'] == serve_argv(plan['executable'], plan['model']), 'Not the public Mixed serve command')
    require((plan['rounds'], plan['prompt'], plan['input_tokens'], plan['normal_tokens'], plan['cancel_tokens'])
            == (2, 'Hello', 1, 2, 4), 'Only frozen Hello/2-token completion and 4-token cancellation are admitted')
    require(plan['environment']['CUDA_VISIBLE_DEVICES'] == '0'
            and plan['environment']['CUBLAS_WORKSPACE_CONFIG'] == ':0:0'
            and plan['environment']['FERRULE_CUDA_ARCH'] == 'sm_86', 'Wrong startup environment')
    require(sha(plan['python']) == plan['python_sha256'], 'Python interpreter changed')
    require(all(sha(p) == pin for p, pin in plan['runtime_libraries'].items()), 'CUDA loader library changed')
    require(sha(plan['executable']) == plan['executable_sha256'], 'Binary changed; rebuild/reprepare')
    current = source_snapshot()
    expected = plan['source_sha256']
    if plan.get('source_scope') == 'production_inputs_excluding_test_dirs':
        current, expected = production_inputs(current), production_inputs(expected)
    require(current == expected, 'Production build inputs changed since CUDA build; MUST rebuild/reprepare before GPU')
    require(all(sha(ROOT / p) == pin for p, pin in plan['workflow_sha256'].items()), 'Workflow changed; reprepare')
    require(all(sha(Path(plan['model']) / p) == pin for p, pin in plan['model_metadata_sha256'].items()), 'Checkpoint/tokenizer metadata changed')


@contextlib.contextmanager
def deadline(seconds):
    prior = signal.signal(signal.SIGALRM, http.deadline_signal)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)


def get_json(path, timeout=2):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), http.NoRedirect())
    with opener.open(f'http://127.0.0.1:{PORT}{path}', timeout=timeout) as response:
        body = response.read(http.MAX_BODY + 1)
        require(response.status == 200 and len(body) <= http.MAX_BODY, 'Invalid API response: ' + path)
        return http.strict_json(body.decode())


def complete(plan):
    messages = []
    http.http_request(dict(url=f'http://127.0.0.1:{PORT}/v1/completions',
                           timeout=LIMITS['request_seconds'], payload=http.make_payload(
                               MODEL_ID, plan['prompt'], plan['normal_tokens'], 0)), messages.append)
    require(messages[-1].get('status') == 'success', 'SSE failed: ' + str(messages[-1]))
    stream = http.Stream()
    for message in messages:
        if message['kind'] == 'event':
            stream.event(message['data'], message['event'], message['at'])
    stream.eof()
    require(stream.finish_reason == 'length' and stream.usage['prompt_tokens'] == 1
            and stream.usage['completion_tokens'] == 2, 'Wrong normal completion/usage')
    result = dict(text=''.join(stream.content), usage=stream.usage, finish_reason=stream.finish_reason)
    print('NORMAL SSE', json.dumps(result), flush=True)
    return result


def cancel(plan):
    payload = http.make_payload(MODEL_ID, plan['prompt'], plan['cancel_tokens'], 0)
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/completions',
                                    data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), http.NoRedirect())
    with deadline(LIMITS['request_seconds']), opener.open(request, timeout=LIMITS['request_seconds']) as response:
        require(response.status == 200 and response.headers.get_content_type() == 'text/event-stream', 'Cancellation request failed')
        stream, total = http.Stream(), 0
        for line in response:
            total += len(line)
            require(total <= http.MAX_BODY, 'Cancellation SSE exceeds bound')
            if not line.startswith(b'data: '):
                continue
            data = line[6:].decode().strip()
            stream.event(data, '', time.monotonic())
            require(stream.finish_reason is None, 'Cancellation raced normal completion')
            if stream.content:
                request_id = int(json.loads(data)['id'].removeprefix('cmpl-ferrule-'))
                print('CANCEL disconnect after first content, request_id=', request_id, flush=True)
                return request_id
    raise RuntimeError('No first content for cancellation')


def wait_released(log, request_id=None):
    with deadline(LIMITS['cancellation_seconds']):
        while True:
            # Snapshot delivery shares the serial owner with an in-flight CUDA step.
            # The outer phase alarm, not a shorter socket timeout, owns this deadline.
            snapshot = get_json('/admission', timeout=LIMITS['cancellation_seconds'])
            runtime = snapshot.get('runtime', {})
            released = snapshot['requests']['held'] == 0 and snapshot['prompt_bytes']['held'] == 0
            released &= runtime.get('status') == 'available' and all(
                runtime.get(key, {}).get('held') == 0 for key in ('waiting_requests', 'request_identities', 'session_identities'))
            observed = request_id is None
            if request_id is not None:
                with log.open() as records:
                    for line in records:
                        try: fields = json.loads(line).get('fields', {})
                        except ValueError: continue
                        observed |= (fields.get('event') == 'request_http_terminal'
                                     and fields.get('request_id') == request_id
                                     and fields.get('finish_reason') == 'client_disconnect')
            if released and observed:
                print('ADMISSION released; cancellation trace=', observed, flush=True)
                return snapshot
            time.sleep(.1)


def port_released():
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(('127.0.0.1', PORT))
        probe.listen(1)


def worker(config):
    plan, directory = config['manifest'], Path(config['directory'])
    server = None
    report = dict(status='starting', gpu_quiescence='unverified; no GPU queries')
    log = directory / 'server.log'
    try:
        validate(plan)
        require(os.getpgrp() == os.getpid(), 'Worker requires owned supervisor session')
        port_released()
        print('STARTUP', json.dumps(plan['argv']), 'deadline=', LIMITS['startup_seconds'], flush=True)
        with log.open('xb') as output:
            server = subprocess.Popen(plan['argv'], env=dict(os.environ, **plan['environment']),
                                      stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        http.wait_ready(dict(url=f'http://127.0.0.1:{PORT}/v1/completions', model=MODEL_ID), server, LIMITS['startup_seconds'])
        with deadline(10):
            report['health'] = get_json('/health')
            report['ready'] = get_json('/readyz')
            report['models'] = get_json('/v1/models')
        require(report['health'].get('status') == 'ok' and report['ready'].get('ready') is True, 'Health/readiness failed')
        require(any(row.get('id') == MODEL_ID for row in report['models'].get('data', [])), 'Wrong model API')
        startup = log.read_text(errors='replace')
        require(PROFILE in startup and 'activation/KV/conv=BF16 SSM/accumulation=F32' in startup, 'Mixed effective startup profile missing')
        print('READY health/model API PASS; effective profile=', PROFILE, flush=True)
        report['memory_after_ready'] = gpu_snapshot()
        report['effective_startup_log'] = startup
        print('GPU MEMORY AFTER READY', json.dumps(report['memory_after_ready']), flush=True)
        report['normal'] = complete(plan)
        wait_released(log)
        report['cancel_request_id'] = cancel(plan)
        report['after_cancel'] = wait_released(log, report['cancel_request_id'])
        report['recovery'] = complete(plan)
        wait_released(log)
        report['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        print('SMOKE ERROR', report['error'], flush=True)
    finally:
        if server is not None:
            try:
                report['shutdown'] = http.shutdown_owned_group(server, LIMITS['teardown_seconds'])
                report['server_returncode'] = server.returncode
                require(not report['shutdown']['timed_out'] and server.returncode == 0, 'Non-graceful server exit')
                port_released()
                report['port_released'] = True
            except Exception as error:
                report.update(status='failed', teardown_error=str(error))
        if log.exists():
            with log.open('rb') as stream:
                stream.seek(max(0, log.stat().st_size - 16384))
                print('SERVER LOG TAIL\n' + stream.read().decode(errors='replace'), flush=True)
        print('TEARDOWN', report.get('shutdown'), 'exit=', report.get('server_returncode'), flush=True)
        write(directory / 'lifecycle.json', report)
    return 0 if report['status'] == 'complete' else 1


def restart_allowed(result, lifecycle):
    custody, handshake = result.get('custody', {}), result.get('handshake', {})
    return (result.get('returncode') == 0 and not result.get('timed_out') and not result.get('error')
            and not result.get('cleanup_errors') and custody.get('status') == 'proven_drained'
            and custody.get('echild') is True and custody.get('containment_verified') is True
            and handshake.get('containment_verified') is True and bool(handshake.get('launch_id'))
            and custody.get('launch_id') == handshake.get('launch_id')
            and custody.get('helper_identity') == handshake.get('helper_identity')
            and lifecycle.get('status') == 'complete' and lifecycle.get('server_returncode') == 0
            and lifecycle.get('port_released') is True)


def execute(plan, output):
    validate(plan)
    directory = Path(output).resolve()
    directory.mkdir(exist_ok=False)
    report = dict(status='in_progress', rounds=[], cold_restart='not_run', gpu_quiescence='unverified')
    try:
        report['gpu_before_launch'] = wait_gpu_quiescent('before launch')
        for index in range(2):
            validate(plan)
            child = directory / f'round-{index}'
            child.mkdir()
            print('RUN', index, '(cold process reload)' if index else '(initial startup)', flush=True)
            write(directory / 'smoke.json', report)
            try:
                result = supervisor.run_process([plan['python'], '-B', str(SCRIPT), '--_worker'],
                                                LIMITS['supervisor_seconds'], dict(manifest=plan, directory=str(child)))
            except BaseException as error:
                if hasattr(error, 'supervisor_result'):
                    write(child / 'supervisor.json', error.supervisor_result)
                raise
            write(child / 'supervisor.json', result)
            print(result.get('stdout', ''), result.get('stderr', ''), flush=True)
            lifecycle = json.loads((child / 'lifecycle.json').read_text()) if (child / 'lifecycle.json').exists() else {}
            allowed = restart_allowed(result, lifecycle)
            report['rounds'].append(dict(index=index, status='pass' if allowed else 'failed',
                                        custody=result.get('custody'), lifecycle=lifecycle))
            quiescence = wait_gpu_quiescent('before restart' if index == 0 else 'final release')
            report['rounds'][-1]['gpu_quiescence'] = quiescence
            report['gpu_quiescence'] = 'verified'
            require(allowed, 'HALT: work/exit/deadline/authenticated custody incomplete; no next restart; do not kill foreign PIDs')
        report.update(status='pass', cold_restart='pass')
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        print('SMOKE FAILED', report['error'], flush=True)
    finally:
        try:
            report['gpu_final'] = wait_gpu_quiescent('final ownership release')
            report['gpu_quiescence'] = 'verified'
        except Exception as error:
            report.update(status='failed', gpu_quiescence='unknown', gpu_release_error=str(error))
            print('GPU RELEASE UNKNOWN', error, 'do not start another owner', flush=True)
        write(directory / 'smoke.json', report)
    print('SMOKE', report['status'], 'rounds=', len(report['rounds']), 'report=', directory / 'smoke.json', flush=True)
    return 0 if report['status'] == 'pass' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'run'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--sole-owner-permission', action='store_true')
    args = parser.parse_args()
    if args.mode == 'prepare':
        require(args.directory is not None and not args.execute, 'prepare needs --directory, never --execute')
        prepare(args.directory)
        return 0
    require(args.plan is not None, 'run needs --plan')
    plan = json.loads(args.plan.read_text())
    validate(plan)
    if not args.execute:
        print('CHECK PASS; no GPU/server launched. Rebuild after concurrent attention changes. Use both execution authorization flags.')
        return 0
    require(args.sole_owner_permission and args.output is not None, 'Execution requires --sole-owner-permission and a fresh --output')
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Owner signal {signum}')
    prior = signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute(plan, args.output)
    finally:
        signal.signal(signal.SIGTERM, prior)


if __name__ == '__main__':
    try:
        raise SystemExit(worker(json.load(sys.stdin)) if sys.argv[1:] == ['--_worker'] else main())
    except (OSError, ValueError, http.ProtocolError) as error:
        print('BLOCKED:', error, file=sys.stderr, flush=True)
        raise SystemExit(1)
