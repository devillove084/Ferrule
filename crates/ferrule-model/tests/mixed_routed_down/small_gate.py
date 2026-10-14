"""Pinned, bounded small tests only; reuse maintained authentication/lock/subreaper."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / 'papers/arxiv/experiments'
sys.path[:0] = [str(PAPER), str(PAPER / 'tp2_resident'), str(ROOT / 'scripts')]
import contract as ct
import launch_guard as guard
import observer
import test_vllm_qwen35 as supervisor

RELEASE = PAPER / 'tp2_mixed/oracle/runs/routed-down-m1-cuda-01'
RELEASE_PIN = '4f3081e16cc496121817bd49e4732cb5b49a030ca9853f616751c17d23ba1704'
PREFIX = 'transformer::standard::cuda::hybrid_rank::mixed::candidate_tests::'
FROZEN = 'frozen_tp2_down_w8a16_and_native_match_every_word'
PREFLIGHT = 'frozen_tp2_down_fixture_preflight'
DISPATCH = PREFIX + 'retired_candidate_dispatches_only_down_and_preserves_routed_layer_words'
UNKNOWN = PREFIX + 'retired_candidate_unknown_quarantines_without_native_retry_or_release'
RETIRE = PREFIX + 'retired_metadata_preserves_exact_selector_identity_without_payload_or_source_io'
NOTE = 'Explicit sole GPU owner: four frozen TP2 routed-down M1 cases and retired dispatch/Unknown only; no full model'
CONFIG = dict(limits=dict(case_seconds=300, idle_seconds=30), resources=dict(
    gpu_expected_total_bytes=24 << 30, gpu_target_bytes=2 << 30,
    host_target_bytes=8 << 30, sample_interval_seconds=.5))


def fixtures():
    ct.require(ct.sha(RELEASE / 'manifest.json') == RELEASE_PIN, 'Release manifest changed')
    release = ct.load(RELEASE / 'manifest.json')
    ct.require(release['status'] == 'complete' and len(release['fixtures']) == 4, 'Incomplete release')
    for row in release['fixtures']:
        ct.require(ct.sha(RELEASE / row['directory'] / 'SHA256SUMS') == row['sha256'], 'Child pin changed')
    return release['fixtures']


def hidden_env():
    env = ct.build_environment(ct.config(ct.EXCLUSIVE_PATH))
    env['NVIDIA_VISIBLE_DEVICES'] = 'void'
    return env


def select(binary, selector, env):
    r = subprocess.run([binary, selector, '--exact', '--list'], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=20, check=True)
    ct.require(r.stdout.splitlines().count(selector + ': test') == 1
               and '1 test, 0 benchmarks' in r.stdout, 'Selector must match exactly one test')
    return r.stdout


def prepare(out):
    rows = fixtures()
    out.mkdir(exist_ok=False)
    env = hidden_env()
    sources = ct.source_pins()
    pins = ct.inventory([Path(__file__), Path(sys.executable), PAPER / 'contract.py',
                         PAPER / 'launch_guard.py', PAPER / 'telemetry.py', PAPER / 'runner.py',
                         PAPER / 'tp2_resident/observer.py', ROOT / 'scripts/test_vllm_qwen35.py',
                         ROOT / 'scripts/_owned_subreaper.py', Path('/usr/bin/env'), Path('/usr/bin/sha256sum'),
                         Path('/usr/bin/nvidia-smi'), RELEASE / 'manifest.json',
                         *[p for row in rows for p in (RELEASE / row['directory']).iterdir()]])
    tools = ct.required_inventory(ct.config(ct.EXCLUSIVE_PATH))
    ct.write(out / 'source-before.json', sources)
    ct.write(out / 'tools.json', tools)
    ct.write(out / 'environment.json', env)
    version = subprocess.check_output([env['NVCC'], '--version'], env=env, text=True, timeout=10)
    ct.require('release 12.4' in version and env['FERRULE_CUDA_ARCH'] == 'sm_86', 'Wrong CUDA toolchain')
    argv = [ct.EXCLUSIVE['cargo'], 'test', '--offline', '--locked', '-p', 'ferrule-model',
            '--features', 'cuda', '--lib', '--test', 'mixed_routed_down_frozen', '--no-run',
            '--message-format=json', '--target-dir', str(ROOT / 'target/validation/qwen35-35b-current/toolchain/cargo-cuda124')]
    code = 'import subprocess,sys; f=open(sys.argv[1],"x"); e=open(sys.argv[2],"x"); sys.exit(subprocess.call(sys.argv[3:],stdout=f,stderr=e))'
    result = supervisor.run_process(ct.isolated_argv([sys.executable, '-S', '-B', '-c', code,
        str(out / 'build.jsonl'), str(out / 'build.stderr'), *argv], env), 240)
    ct.write(out / 'build-supervisor.json', result)
    ct.require(ct.successful_process(result), 'Build failed/timed out; inspect build.stderr')
    ct.require(sources == ct.source_pins(), 'Sources changed during build')
    ct.validate_inventory(ct.config(ct.EXCLUSIVE_PATH), tools)
    binaries = {}
    for line in (out / 'build.jsonl').read_text().splitlines():
        artifact = json.loads(line)
        if artifact.get('reason') != 'compiler-artifact' or not artifact.get('executable'):
            continue
        key = artifact['target']['name']
        ct.require(key in ('ferrule_model', 'mixed_routed_down_frozen') and artifact['profile']['test'], 'Unexpected build target')
        directory = out / ('binary-' + key)
        directory.mkdir()
        staged = ct.stage_binary(artifact['executable'], directory)
        binaries[key] = dict(path=staged, sha256=ct.sha(staged))
    ct.require(len(binaries) == 2, 'Missing exact small binaries')
    listings = {}
    for selector in (PREFLIGHT, FROZEN, RETIRE, DISPATCH, UNKNOWN):
        key = 'ferrule_model' if selector.startswith(PREFIX) else 'mixed_routed_down_frozen'
        listings[selector] = select(binaries[key]['path'], selector, env)
    for row in rows:
        case_env = dict(env, FERRULE_MIXED_DOWN_FIXTURE=str(RELEASE / row['directory']),
                        FERRULE_MIXED_DOWN_SHA256=row['sha256'])
        result = supervisor.run_process(ct.isolated_argv([binaries['mixed_routed_down_frozen']['path'],
            PREFLIGHT, '--exact', '--ignored', '--nocapture', '--test-threads=1'], case_env), 20)
        ct.write(out / (row['directory'] + '-preflight.json'), result)
        ct.require(ct.successful_process(result) and '1 passed; 0 failed; 0 ignored' in result['stdout'], 'CPU fixture preflight failed')
    result = supervisor.run_process(ct.isolated_argv([binaries['ferrule_model']['path'], RETIRE,
        '--exact', '--nocapture', '--test-threads=1'], env), 20)
    ct.write(out / 'retirement-cpu.json', result)
    ct.require(ct.successful_process(result) and '1 passed; 0 failed; 0 ignored' in result['stdout'], 'CPU retirement failed')
    ct.check_pins(pins)
    ct.require(sources == ct.source_pins(), 'Sources changed during preparation')
    # Pin the actual dynamic resolution, not just the chosen CUDA directory.
    shared = {}
    for key, binary in binaries.items():
        linkage = subprocess.check_output(['/usr/bin/ldd', binary['path']], env=env, text=True, timeout=10)
        ct.require('not found' not in linkage, 'Missing shared library')
        ct.write(out / (key + '-linkage.json'), dict(ldd=linkage))
        for line in linkage.splitlines():
            for token in line.split():
                if token.startswith('/') and Path(token).is_file():
                    shared[str(Path(token).resolve())] = ct.sha(token)
    prepared = dict(directory=str(out), binaries=binaries, sources=sources, pins=pins, tools=tools,
                    shared=shared, environment=env, nvcc=version, listings=listings, fixtures=rows,
                    cpu_preflight_passed=4, production_enabled=False)
    ct.write(out / 'prepared.json', prepared)
    print(json.dumps(dict(prepared=str(out / 'prepared.json'), sha256=ct.sha(out / 'prepared.json'),
                         cpu_preflight_passed=4, selectors=len(listings))))


def check(p):
    ct.check_pins(p['pins'])
    ct.check_pins(p['sources'])
    ct.check_pins(p['shared'])
    for group in p['tools'].values():
        ct.check_pins(group)
    for binary in p['binaries'].values():
        ct.validate_staged_binary(binary['path'], binary['sha256'])
    ct.require(fixtures() == p['fixtures'] and p['cpu_preflight_passed'] == 4, 'Preflight/release differs')


def idle(uuids):
    r = observer.idle(CONFIG, uuids, NOTE)
    ct.require(r['status'] == 'verified' and all(d['gpu_bytes'] == 0 for d in r['baseline'].values()), 'All-eight idle required')
    return r


def worker(pin):
    p = json.load(sys.stdin)
    auth = guard.authenticate(p, pin)
    check(p)
    for uuid, lock in p['locks'].items():
        ct.validate_owner_lock(lock, uuid, p['coordinator']['pid'])
    out = Path(p['directory'])
    ct.write(out / 'authentication.json', auth)
    env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), CUDA_VISIBLE_DEVICES=','.join(p['uuids']),
               NVIDIA_VISIBLE_DEVICES=','.join(p['uuids']), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               LD_LIBRARY_PATH=p['environment']['LD_LIBRARY_PATH'], CUBLAS_WORKSPACE_CONFIG=':0:0',
               FERRULE_RUN_MIXED_RANK_GPU='1', RUST_BACKTRACE='1')
    ct.write(out / 'gpu-environment.json', env)
    jobs = [(row['directory'], 'mixed_routed_down_frozen', FROZEN, row) for row in p['fixtures']]
    jobs += [('retired-dispatch', 'ferrule_model', DISPATCH, None), ('retired-unknown', 'ferrule_model', UNKNOWN, None)]
    exits = []
    for name, key, selector, row in jobs:
        directory = out / name
        directory.mkdir()
        case_env = dict(env)
        if row:
            case_env.update(FERRULE_MIXED_DOWN_FIXTURE=str(RELEASE / name),
                            FERRULE_MIXED_DOWN_SHA256=row['sha256'], FERRULE_MIXED_DOWN_REPORT_DIR=str(directory))
        binary = p['binaries'][key]
        with (directory / 'test.log').open('x') as log:
            with ct.pinned_executable(binary['path'], binary['sha256']) as fd:
                proc = subprocess.Popen(['/proc/self/fd/' + str(fd), selector, '--exact', '--ignored',
                    '--nocapture', '--test-threads=1'], cwd=ROOT, env=case_env, stdout=log,
                    stderr=subprocess.STDOUT, pass_fds=(fd,))
            monitor = observer.Monitor(directory, proc, CONFIG, auth, p['uuids'], NOTE, p['baseline'],
                                       ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'LD_LIBRARY_PATH'))
            monitor.start()
            try:
                code = proc.wait(timeout=45)
            finally:
                ct.write(directory / 'resources.json', monitor.finish())
        text = (directory / 'test.log').read_text()
        ct.require('running 1 test' in text and ('1 passed; 0 failed; 0 ignored' in text
                   or '0 passed; 1 failed; 0 ignored' in text), 'Selector count/run incomplete')
        result = dict(name=name, selector=selector, selector_count=1, returncode=code,
                      exact_pass=code == 0, log_sha256=ct.sha(directory / 'test.log'))
        ct.write(directory / 'exit.json', result)
        exits.append(result)
        # Numerical failure is still failure, but authorized collection may continue.
        if row:
            reports = ct.load(directory / 'exact.json')
            ct.require(len(reports) == 4, 'Missing both independent rank comparisons')
        ct.check_pins(p['pins'])
        ct.check_pins(p['sources'])
    ct.write(out / 'tests.json', exits)
    return int(any(r['returncode'] for r in exits))


def run(out, prepared_pin, granted):
    ct.require(granted, 'Explicit renewed sole-owner grant required')
    ct.require(ct.sha(out / 'prepared.json') == prepared_pin, 'Preparation pin mismatch')
    p = ct.load(out / 'prepared.json')
    check(p)
    all_uuids = [d['uuid'] for d in ct.load(RELEASE / 'all-before.json')]
    ct.require(len(set(all_uuids)) == 8, 'Eight lock identities required')
    p['uuids'] = all_uuids[:2]
    with contextlib.ExitStack() as stack:
        p['locks'] = {u: stack.enter_context(ct.exclusive_owner_lock(u)) for u in sorted(all_uuids)}
        before = idle(p['uuids'])
        ct.write(out / 'all-before.json', before)
        p['baseline'] = before['baseline']
        p, pin = guard.seal(p, out)
        env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), CUDA_VISIBLE_DEVICES='',
                   NVIDIA_VISIBLE_DEVICES='void', PYTHONDONTWRITEBYTECODE='1', FERRULE_PAPER_WORKER_TOKEN=p['worker_token'])
        result = supervisor.run_process(ct.isolated_argv([sys.executable, '-S', '-B', str(Path(__file__).resolve()),
            '--worker', pin], env), 300, p)
        ct.write(out / 'supervisor.json', result)
        ct.require(ct.drained(result), 'Custody unknown; release forbidden')
        ct.write(out / 'all-after.json', idle(p['uuids']))
        check(p)
    # Reacquire all local locks and verify final idle, then release every lock.
    with contextlib.ExitStack() as stack:
        for u in sorted(all_uuids):
            stack.enter_context(ct.exclusive_owner_lock(u))
        ct.write(out / 'all-final.json', idle(p['uuids']))
    ct.write(out / 'release.json', dict(proven_drained=True, echild=True, all_eight_idle=True,
        all_local_locks_reacquired_and_released=True, gpu_owner_released=True,
        small_gates_passed=ct.successful_process(result), full_model_run=False,
        production_enabled=False, supervisor_sha256=ct.sha(out / 'supervisor.json')))
    print(json.dumps(ct.load(out / 'release.json')))
    if result['stderr']:
        print(result['stderr'])
    return int(not ct.successful_process(result))


def main():
    ct.require_cpu_python()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare', type=Path)
    ap.add_argument('--run', type=Path)
    ap.add_argument('--prepared-sha256')
    ap.add_argument('--sole-gpu-owner-granted', action='store_true')
    ap.add_argument('--worker')
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker)
    if a.prepare:
        return prepare(a.prepare.resolve())
    if a.run:
        return run(a.run.resolve(), a.prepared_sha256, a.sole_gpu_owner_granted)
    ap.error('Choose --prepare or --run')


if __name__ == '__main__':
    raise SystemExit(main())
