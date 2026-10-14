"""Internal decoded-cache qualification; never a public factory or benchmark.

CPU prepare builds/pins one composition test binary and runs exact metadata gates.
GPU run requires a fresh owner grant AND authenticated small-operator evidence.
The existing supervisor, locks, observer, bridge and oracle remain authoritative.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(PAPER), str(HERE), str(ROOT / 'scripts')]
import contract as ct
import launch_guard as guard
import observer
import test_vllm_qwen35 as supervisor

ORACLE = PAPER / 'tp2_mixed/oracle'
MODEL = '/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8'
UUIDS = ['GPU-b980f22b-58a3-fc88-21c1-ae86053d8535', 'GPU-63f1bd01-ddf9-a742-ee48-9389892493d9']
FULL = ('full40-20261009-03', 'fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d')
RECIPES = {
    'greedy4': ('lifecycle-greedy4-01', '4fafda33e51d7cec3d82beb59aac73eb12dc6b9904dc0a74535ea35e9d68df81'),
    'fork2': ('lifecycle-fork2-01', '098820b7839ed74b643066dcebd6e7e7b86d0716a8cbcecec708068d37b2a6dd'),
    'canary2': ('lifecycle-canary2-01', '4069b0e3f4dbead859654855a576c402867fa6ea2908d3dc67cf78c3ea2f672f'),
    'recreate1': ('lifecycle-recreate1-01', '5747d1ee3793d629aff5d87488d2dbd970fb02be11742b88ac16e3dca05e5a59'),
    'cold-owner1-01': ('lifecycle-cold-owner1-01', 'fe7fb83bf8cae99164209aad077f5c1ff92adf902aaca43e8dbb99f8af8166ae'),
    'cold-owner1-02': ('lifecycle-cold-owner1-02', 'a67492ddb2a5a3d4a4f40abc1ca92c672f74605238d2f06ea62827f17cda3d54'),
}
GATES = {'full40': 'decoded_cache::full40_cache_candidate_boundaries_two_cases_two_phases',
         **{name: 'lifecycle::full40_decoded_cache_bounded_multistep_fork_recreate_cold_owner' for name in RECIPES}}
BACKEND_PREFIX = 'cuda::providers::bf16_gemmex::mixed_decode_reuse_tests::'
MODEL_PREFIX = 'transformer::standard::cuda::hybrid_rank::mixed::decoded_gpu_tests::'
SMALL_GATES = {
    'backend-exact': ('backend', BACKEND_PREFIX + 'mixed_decode_resident_single_pair_m1_m5', 1),
    'backend-preflight': ('backend', BACKEND_PREFIX + 'mixed_decode_pair_rejections', 1),
    'model-cold-hot-reload': ('model', MODEL_PREFIX + 'decoded_pair_single_cold_hot_reload_exact_and_no_cache_dma', 2),
    'model-fixed-pins': ('model', MODEL_PREFIX + 'decoded_fixed_pins_prepare_once_and_shared_stays_separate', 2),
    'model-unknown-panic': ('model', MODEL_PREFIX + 'decoded_pair_unknown_and_panic_retain_physical_backing', 1),
    'model-physical-full': ('model', MODEL_PREFIX + 'decoded_physical_full_native_fallback_before_submit_exact', 2),
}
HOST_SMALL = [
    ('backend', BACKEND_PREFIX + name) for name in (
        'mixed_decode_preflight_without_device_calls', 'mixed_pair_decode_preflight_without_device_calls',
        'mixed_pair_preflight_without_device_calls')
] + [
    ('model', 'transformer::standard::cuda::hybrid_rank::mixed::preparation_tests::decoded_claim_tests::decoded_claim_compares_full_admission_even_when_grants_match'),
    ('model', MODEL_PREFIX + 'decoded_fixture_descriptors_preserve_pair_and_separate_fixed_roles'),
]
ALL_GATES = {**{k: v[1] for k, v in SMALL_GATES.items()}, **GATES}

CPU = {
    'decoded_cache::native_default_off_and_cache_report_claim_share_exact_grant': False,
    'decoded_cache::cache_rejects_low_budget_capacity_and_configuration_drift': False,
    'decoded_cache::cache_capacity_follows_device_placement_not_lane_number': False,
    'decoded_cache::full40_cache_header_ledger': True,
    'decoded_cache::lifecycle_cache_header_ledger': True,
}
CONFIG = dict(limits=dict(case_seconds=1700, idle_seconds=30), resources=dict(
    gpu_expected_total_bytes=24 << 30, gpu_target_bytes=22 << 30,
    host_target_bytes=128 << 30, sample_interval_seconds=.5))
SMALL_CASES = {f'{form}-m{rows}-{phase}' for form in ('single', 'pair') for rows in (1, 5)
               for phase in ('cold', 'hot', 'eviction-reload', 'after-retire')}
SMALL_CASES |= {'known-failure', 'unknown-retains', 'borrowed-retains'}


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=ct.sha(path))


def checked(row):
    ct.check_pins({row['path']: row['sha256']})
    return Path(row['path'])


def run_directory(identity, create=False):
    ct.require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', identity or ''), 'Need safe new id')
    root = HERE / 'runs'
    ct.require(root.resolve() == root and not root.is_symlink(), 'Run root must be local')
    path = root / identity
    if create:
        path.mkdir(exist_ok=False)
    ct.require(path.resolve() == path and path.is_dir(), 'Run path must be local')
    return path


def manifest():
    return dict(schema='ferrule.paper.decoded-cache-plan.v1', public_enabled=False,
        cache_grants_per_owner=3 << 30, device_capacity=22 << 30,
        pinned_grants=1340 << 20, routed_working_set_grants=768 << 20,
        cpu_selectors=CPU, host_small_selectors=HOST_SMALL, gpu_selectors=GATES,
        small_selectors=SMALL_GATES, small_required=sorted(SMALL_CASES),
        numerical='native BF16 bit-exact; M1/M5, ordered full-2N routed pair GEMM; no W8',
        custody='healthy completion only; Unknown/borrowed roots never refunded or reused',
        performance='GPU-correlated prefill/decode launch counts, not host misses or historical shape motifs',
        progression='small -> full40 hello/capital each 2178 exact -> six recipes -> fresh public admission/E2E',
        status='plan_only_no_candidate_comparison')


def workflow_paths():
    return [Path(__file__), PAPER / 'contract.py', PAPER / 'launch_guard.py',
        PAPER / 'telemetry.py', HERE / 'observer.py', HERE / 'sampling_v4.py',
        HERE / 'sampling_v2.py', HERE / 'sampling_v3.py',
        ROOT / 'scripts/test_vllm_qwen35.py', ROOT / 'scripts/_owned_subreaper.py',
        ROOT / 'scripts/qwen35_fp8_bf16_reference.py', ROOT / 'scripts/qwen35_fp8_cuda_reference.py',
        Path(sys.executable), Path('/usr/bin/env'), Path('/usr/bin/nvidia-smi'),
        *ORACLE.glob('*.py'), ORACLE / 'protocol.json',
        ROOT / 'crates/ferrule-composition/tests/tp2_mixed_execution/lifecycle_bridge.py',
        HERE / 'decoded_cache_compare.py']


def cpu_command(out, name, argv, env, timeout):
    # Reuse bounded process-tree supervision for compilers as well as test children.
    code = 'import subprocess,sys; o=open(sys.argv[1],"x"); e=open(sys.argv[2],"x"); sys.exit(subprocess.call(sys.argv[3:],stdout=o,stderr=e))'
    result = supervisor.run_process(ct.isolated_argv([sys.executable, '-S', '-B', '-c', code,
        str(out / (name + '.stdout')), str(out / (name + '.stderr')), *map(str, argv)], env), timeout)
    ct.write(out / (name + '.json'), dict(argv=list(map(str, argv)), supervisor=result))
    ct.require(ct.successful_process(result), 'CPU step failed/timed out: ' + name)
    return (out / (name + '.stdout')).read_text()


def prepare(identity):
    out = run_directory(identity, True)
    env = ct.build_environment(ct.config(ct.EXCLUSIVE_PATH))
    env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
    sources = ct.source_pins()
    ct.write(out / 'sources-before.json', sources)
    ct.write(out / 'environment.json', env)
    ct.write(out / 'plan.json', manifest())
    tools = ct.required_inventory(ct.config(ct.EXCLUSIVE_PATH))
    ct.require('release 12.4' in subprocess.check_output([env['NVCC'], '--version'], env=env, text=True, timeout=10), 'Need CUDA12.4')
    binaries = {}
    targets = [('backend', 'ferrule-backend', 'ferrule_backend', ['--lib']),
               ('model', 'ferrule-model', 'ferrule_model', ['--lib']),
               ('composition', 'ferrule-composition', 'tp2_mixed_execution', ['--test', 'tp2_mixed_execution'])]
    for key, package, name, flags in targets:
        text = cpu_command(out, 'build-' + key, [ct.EXCLUSIVE['cargo'], 'test', '--locked', '--offline',
            '-p', package, '--features', 'cuda', *flags, '--no-run', '--message-format=json', '--target-dir',
            ROOT / 'target/validation/qwen35-35b-current/toolchain/cargo-cuda124'], env, 240)
        artifacts = [json.loads(line) for line in text.splitlines()]
        matches = [a['executable'] for a in artifacts if a.get('reason') == 'compiler-artifact'
                   and a.get('target', {}).get('name') == name and a.get('executable') and a['profile']['test']]
        ct.require(len(matches) == 1, 'Need exact test artifact: ' + name)
        directory = out / ('binary-' + key)
        directory.mkdir()
        binaries[key] = record(ct.stage_binary(matches[0], directory))
    selectors = [(key, selector) for key, selector, _ in SMALL_GATES.values()]
    selectors += [('composition', selector) for selector in GATES.values()]
    selectors += HOST_SMALL + [('composition', selector) for selector in CPU]
    for key, selector in sorted(set(selectors)):
        listed = subprocess.check_output([binaries[key]['path'], selector, '--exact', '--list'], env=env, text=True, timeout=20)
        ct.require(listed.splitlines().count(selector + ': test') == 1 and '1 test, 0 benchmarks' in listed,
                   'Selector must resolve to exactly one test: ' + selector)
    receipts = []
    cpu = [(key, selector, False) for key, selector in HOST_SMALL]
    cpu += [('composition', selector, ignored) for selector, ignored in CPU.items()]
    for index, (key, selector, ignored) in enumerate(cpu):
        argv = [binaries[key]['path'], selector, '--exact', '--nocapture', '--test-threads=1']
        if ignored:
            argv.append('--ignored')
        text = cpu_command(out, 'cpu-' + str(index), argv,
            dict(env, FERRULE_MIXED_HEADER_DIR=MODEL), 180)
        ct.require('1 passed; 0 failed; 0 ignored' in text, 'CPU selector did not execute')
        receipts.append(record(out / ('cpu-' + str(index) + '.json')))
    ct.require(sources == ct.source_pins(), 'Sources changed during preparation; no promotion')
    ct.validate_inventory(ct.config(ct.EXCLUSIVE_PATH), tools)
    pins = ct.inventory(workflow_paths())
    pins.update(ct.inventory([out / 'plan.json', out / 'environment.json']))
    for pattern in ('build-*', 'cpu-*'):
        for file in out.glob(pattern):
            pins[str(file)] = ct.sha(file)
    for ref, sha in [FULL, *RECIPES.values()]:
        path = ORACLE / 'runs' / ref / 'manifest.json'
        ct.require(ct.sha(path) == sha, 'Frozen reference changed')
        pins[str(path)] = sha
    ct.write(out / 'prepared.json', dict(schema='ferrule.paper.decoded-cache-prepared.v2',
        binary=binaries['composition'], binaries=binaries, sources=sources, pins=pins, tools=tools, cpu_receipts=receipts,
        public_enabled=False, gpu_run=False))
    return out


def validate(out):
    p = ct.load(out / 'prepared.json')
    ct.require(p['schema'] == 'ferrule.paper.decoded-cache-prepared.v2', 'Wrong preparation')
    ct.check_pins(p['pins'])
    ct.check_pins(p['sources'])
    ct.require(p['sources'] == ct.source_pins(), 'Source inventory drift')
    ct.validate_inventory(ct.config(ct.EXCLUSIVE_PATH), p['tools'])
    for binary in p['binaries'].values():
        ct.validate_staged_binary(binary['path'], binary['sha256'])
    return p


def validate_resources(resources):
    owners = resources['owners']
    ct.require(len(owners) == 2 and {r['owner'] for r in owners} == {0, 1}, 'Missing owner resource ledger')
    for row in owners:
        fields = ('cache_peak_grants', 'cache_grant', 'full_total_with_margins_and_trace', 'new_artifact_h2d_bytes')
        ct.require(all(type(row[k]) is int and row[k] >= 0 for k in fields), 'Invalid resource count')
        ct.require(0 < row['cache_grant'] <= 3 << 30 and row['cache_peak_grants'] <= row['cache_grant']
                   <= row['full_total_with_margins_and_trace'] <= 22 << 30
                   and row['new_artifact_h2d_bytes'] == 0, 'Cache resource/transfer gate failed')


def validate_phase_decode(report):
    """A separate performance gate; host misses/old shape motifs cannot pass it."""
    ct.require(report['measurement'] == 'gpu_correlated_launches' and report['complete'] is True,
               'Need complete phase-bound GPU launch evidence')
    rows = report['rows']
    ct.require(len(rows) == 4 and {(r['owner'], r['phase']) for r in rows}
               == {(owner, phase) for owner in (0, 1) for phase in ('prefill', 'decode')}, 'Missing rank/phase')
    for row in rows:
        ct.require(type(row['native_decode_launches']) is int and type(row['cache_decode_launches']) is int
                   and 0 <= row['cache_decode_launches'] < row['native_decode_launches'], 'GPU decode count did not fall')
    ct.check_pins(report['trace_pins'])


def validate_small(path, sources):
    """Consume real owner receipts; this runner does not fabricate small GPU tests."""
    q = ct.load(path)
    ct.require(q['schema'] == 'ferrule.paper.decoded-cache-small.v1' and q['sources'] == sources,
               'Small qualification must use current sources')
    ct.check_pins(q['sources'])
    ct.check_pins(q['artifact_pins'])
    ct.require(set(q['cases']) == SMALL_CASES, 'Missing/extra small qualification coverage')
    for name, row in q['cases'].items():
        launch = ct.load(checked(row['supervisor']))
        ct.require(row['selector'] and ct.successful_process(launch), 'Unauthenticated small gate: ' + name)
        preparation = ct.load(checked(row['preparation']))
        ct.require(preparation['sources'] == sources and row['binary'] in preparation['binaries'].values(),
                   'Small binary/source preparation mismatch')
        result = ct.load(checked(row['test_exit']))
        ct.require(result['returncode'] == 0 and result['one_test_passed'] is True
                   and result['selector'] == row['selector'], 'Small selector not executed')
        checked(row['binary'])
        evidence = ct.load(checked(row['evidence']))
        ct.require(evidence['passed'] is True and evidence['owners'] == [0, 1], 'Small evidence incomplete')
        ct.require(evidence['coverage'] == name and evidence['binary_sha256'] == row['binary']['sha256']
                   and evidence['source_digest'] == ct.digest(sources)
                   and evidence['launch_id'] == launch['handshake']['launch_id'],
                   'Small evidence relabelled or detached from authenticated execution')
        if '-m' in name:
            pairs = evidence['native_pairs']
            ct.require(len(pairs) == 2 and {r['owner'] for r in pairs} == {0, 1}, 'Missing rank words')
            for pair in pairs:
                a, b = checked(pair['actual']), checked(pair['native'])
                ct.require(a.stat().st_size > 0 and a.stat().st_size % 2 == 0
                           and a.stat().st_size == b.stat().st_size and ct.sha(a) == ct.sha(b),
                           'Native BF16 words differ')
    validate_resources(ct.load(checked(q['resources'])))
    # Hash every referenced artifact again inside the worker seal, including
    # nested native/actual word files, not only the qualification envelope.
    def bind_records(value):
        if isinstance(value, dict):
            if set(value) == {'path', 'sha256'}:
                checked(value)
                ct.require(q['artifact_pins'].get(value['path']) == value['sha256'], 'Unsealed small artifact')
            else:
                for item in value.values():
                    bind_records(item)
        elif isinstance(value, list):
            for item in value:
                bind_records(item)
    bind_records(q)
    for row in q['cases'].values():
        bind_records(ct.load(checked(row['evidence'])))
    bind_records(ct.load(checked(q['resources'])))
    return q


def idle(note):
    result = observer.idle(CONFIG, UUIDS, note)
    ct.require(result['status'] == 'verified' and all(r['gpu_bytes'] == 0 for r in result['baseline'].values()),
               'All-eight idle required')
    return result


def trace_budgets(out, gate):
    roots = [out / 'candidate'] if gate == 'full40' else sorted((out / 'lifecycle-raw').glob('epoch-*'))
    ct.require(len(roots) == (2 if gate == 'recreate1' else 1), 'Missing/reordered owner epochs')
    result = []
    for root in roots:
        for lane in (0, 1):
            path = root / ('rank-' + str(lane)) / 'events.jsonl'
            with path.open() as stream:
                bindings = [json.loads(line)['binding'] for line in stream if '"owner_binding"' in line]
            ct.require(len(bindings) == 1, 'Missing/duplicate owner binding')
            ct.require(bindings[0]['owner'] == lane, 'Owner trace identity drift')
            budget = bindings[0].get('decoded_weight_cache')
            ct.require(budget == dict(capacity_grants=3 << 30, pinned_grants=1340 << 20,
                routed_working_set_grants=768 << 20, routed_entries=20480), 'Cache silently disabled/drifted')
            result.append(dict(events=record(path), budget=budget))
    return result


def worker(pin):
    p = json.load(sys.stdin)
    auth = guard.authenticate(p, pin)
    ct.check_pins(p['pins'])
    ct.check_pins(p['sources'])
    ct.validate_staged_binary(p['binary'], p['binary_sha256'])
    for uuid in UUIDS:
        ct.validate_owner_lock(p['locks'][uuid], uuid, p['coordinator']['pid'])
    out, gate = Path(p['directory']), p['gate']
    ct.write(out / 'authentication.json', auth)
    env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), CUDA_VISIBLE_DEVICES=','.join(UUIDS),
        NVIDIA_VISIBLE_DEVICES=','.join(UUIDS), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        LD_LIBRARY_PATH='/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64',
        CUBLAS_WORKSPACE_CONFIG=':0:0', FERRULE_MIXED_GPU_RUN='1', FERRULE_RUN_MIXED_RANK_GPU='1',
        FERRULE_MIXED_MODEL=MODEL, FERRULE_MIXED_ORACLE_DIR=str(ORACLE / 'runs' / FULL[0]),
        FERRULE_MIXED_ORACLE_SHA256=FULL[1], FERRULE_MIXED_EVIDENCE_DIR=str(out),
        FERRULE_PAPER_WORKER_TOKEN=p['worker_token'], FERRULE_LOG='info,ferrule_performance=debug')
    if gate in RECIPES:
        ref, sha = RECIPES[gate]
        env.update(FERRULE_MIXED_LIFECYCLE_ORACLE_DIR=str(ORACLE / 'runs' / ref), FERRULE_MIXED_LIFECYCLE_ORACLE_SHA256=sha)
    ct.write(out / 'environment.json', env)
    with (out / 'test.log').open('x') as log:
        with ct.pinned_executable(p['binary'], p['binary_sha256']) as fd:
            proc = subprocess.Popen(['/proc/self/fd/' + str(fd), ALL_GATES[gate], '--exact', '--ignored',
                '--nocapture', '--test-threads=1'], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
        monitor = observer.Monitor(out, proc, CONFIG, auth, UUIDS[:SMALL_GATES[gate][2]] if gate in SMALL_GATES else UUIDS,
                                   p['owner_note'], p['baseline'],
                                   ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'LD_LIBRARY_PATH'))
        monitor.start()
        try:
            code = proc.wait(timeout=1600)
        finally:
            resources = monitor.finish()
            ct.write(out / 'resources.json', resources)
    text = (out / 'test.log').read_text()
    one = code == 0 and '1 passed; 0 failed; 0 ignored' in text and 'running 1 test' in text
    ct.write(out / 'test-exit.json', dict(returncode=code, one_test_passed=one, selector=ALL_GATES[gate],
        binary_sha256=p['binary_sha256']))
    ct.require(one, 'Exact GPU selector failed')
    ct.require(resources['status'] == 'observed_within_targets', 'Resource observation incomplete/exceeded target')
    if gate not in SMALL_GATES:
        ct.write(out / 'cache-bindings.json', trace_budgets(out, gate))
    else:
        ct.write(out / 'small-receipt.json', dict(selector=ALL_GATES[gate], binary_sha256=p['binary_sha256'],
            source_digest=ct.digest(p['sources']), assertions_passed=True,
            cache_values='not exported by model-owned tests; no fabricated numeric snapshot',
            gpu_memory=record(out / 'resources.json'), log=record(out / 'test.log')))
    ct.check_pins(p['pins'])
    ct.check_pins(p['sources'])
    return 0


def run(identity, gate, qualification, granted, note):
    ct.require(granted and bool(note.strip()), 'Fresh explicit sole GPU owner grant required')
    ct.require(gate in ALL_GATES, 'Need a known exact gate')
    root = run_directory(identity)
    ct.require(not (root / 'STOP.json').exists(), 'Prior failure: no further GPU work before root fix/reseal')
    prepared = validate(root)
    preceding = list(ALL_GATES)[:list(ALL_GATES).index(gate)]
    evidence_paths = [root / 'prepared.json']
    for prior in preceding:
        previous = root / prior
        result = ct.load(previous / 'result.json')
        ct.require(result['passed'] is True and ct.successful_process(ct.load(previous / 'supervisor.json')),
                   'Prior exact gate/custody failed: ' + prior)
        exit = ct.load(previous / 'test-exit.json')
        key = SMALL_GATES[prior][0] if prior in SMALL_GATES else 'composition'
        ct.require(exit['one_test_passed'] is True and exit['selector'] == ALL_GATES[prior]
                   and exit['binary_sha256'] == prepared['binaries'][key]['sha256'], 'Prior binary/selector drift')
        evidence_paths.extend(previous / name for name in ('result.json', 'supervisor.json', 'test-exit.json', 'resources.json', 'test.log'))
        if prior not in SMALL_GATES:
            receipt = ct.load(previous / 'comparison-receipt.json')
            ct.require(receipt['all_exact'] is True, 'Prior independent comparison missing')
            checked(receipt['report'])
            evidence_paths.extend([previous / 'comparison-receipt.json', Path(receipt['report']['path'])])
    out = root / gate
    out.mkdir(exist_ok=False)
    pins = dict(prepared['pins'])
    pins.update(ct.inventory(evidence_paths))
    key = SMALL_GATES[gate][0] if gate in SMALL_GATES else 'composition'
    binary = prepared['binaries'][key]
    p = dict(gate=gate, directory=str(out), binary=binary['path'],
        binary_sha256=binary['sha256'], pins=pins, sources=prepared['sources'], owner_note=note)
    with contextlib.ExitStack() as stack:
        p['locks'] = {uuid: stack.enter_context(ct.exclusive_owner_lock(uuid)) for uuid in sorted(UUIDS)}
        before = idle(note)
        ct.write(out / 'before.json', before)
        p['baseline'] = before['baseline']
        p, pin = guard.seal(p, out)
        env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), CUDA_VISIBLE_DEVICES='',
                   NVIDIA_VISIBLE_DEVICES='none', FERRULE_PAPER_WORKER_TOKEN=p['worker_token'])
        result = supervisor.run_process(ct.isolated_argv([sys.executable, '-S', '-B', str(Path(__file__).resolve()),
            'worker', '--worker-pin', pin], env), 1700, p)
        ct.write(out / 'supervisor.json', result)
        if not ct.drained(result):
            ct.write(root / 'STOP.json', dict(gate=gate, reason='unknown_custody', supervisor=record(out / 'supervisor.json')))
            raise ValueError('Unknown custody; no further GPU work')
        ct.write(out / 'after.json', idle(note))
        ct.write(out / 'result.json', dict(passed=ct.successful_process(result), drained=True, comparison_pending=gate not in SMALL_GATES))
    ct.write(out / 'release.json', dict(gpu_owner_released=True, drained=True, all_eight_idle=True))
    if not ct.successful_process(result):
        ct.write(root / 'STOP.json', dict(gate=gate, reason='exact_gate_or_resource_failure',
            supervisor=record(out / 'supervisor.json'), gpu_owner_released=True))
        raise ValueError('GPU gate failed; stopped before all later gates')
    return out


def exact_inventory(report, gate):
    ct.require(report.get('passed') is True, 'Independent comparison failed')
    rows = report.get('cases', report.get('calls', {}))
    ct.require(bool(rows), 'Missing comparison rows')
    if gate == 'full40':
        ct.require(set(rows) == {'hello', 'capital'} and all(len(r['metrics']) == 2178 for r in rows.values()),
                   'Need all 2178 fields per case')
    if gate != 'full40':
        expected_calls = {'greedy4': 10, 'fork2': 6, 'canary2': 5, 'recreate1': 4,
                          'cold-owner1-01': 4, 'cold-owner1-02': 4}[gate]
        ct.require(len(rows) == expected_calls and all(len(r['metrics']) == 1089 for r in rows.values()),
                   'Lifecycle inventory incomplete')
    for row in rows.values():
        greedy = row.get('greedy_ids_match') if gate == 'full40' else [row.get('greedy_match')]
        ct.require(bool(greedy) and all(v is True for v in greedy), 'Greedy token mismatch')
        ct.require(bool(row['metrics']) and all(v['exact'] is True and v['failures'] == 0 and v['max_abs'] == 0
                   for v in row['metrics'].values()), 'Tolerance-only pass is not exact')


def compare(identity, gate):
    root = run_directory(identity)
    validate(root)
    out = root / gate
    ct.require(ct.successful_process(ct.load(out / 'supervisor.json')), 'GPU custody/result incomplete')
    trace_budgets(out, gate)
    ref, pin = FULL if gate == 'full40' else RECIPES[gate]
    env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    # The output-only adapter keeps fresh archives beside raw runs, never inside
    # raw/oracle bundles. All tensor mapping and numerical checks remain frozen.
    canonical, report = root / (gate + '-canonical'), root / (gate + '-comparison.json')
    text = cpu_command(out, 'canonical-compare', ['/opt/conda310/bin/python', '-B',
        HERE / 'decoded_cache_compare.py', '--input', out, '--reference', ORACLE / 'runs' / ref,
        '--reference-sha256', pin, '--canonical', canonical, '--output', report,
        *(['--lifecycle'] if gate != 'full40' else [])], env, 180)
    raw = json.loads(text)['raw_sha256']
    exact_inventory(ct.load(report), gate)
    validate(root)
    ct.write(out / 'comparison-receipt.json', dict(all_exact=True, report=record(report),
        raw_sha256=raw, reference_sha256=pin, public_e2e_accepted=False))
    return report


def performance(identity, evidence):
    root = run_directory(identity)
    prepared = validate(root)
    ct.require(evidence is not None, 'Need a pinned phase evidence report')
    report = ct.load(evidence)
    ct.require(report['sources'] == prepared['sources'], 'Performance source drift')
    for gate in GATES:
        receipt = ct.load(root / gate / 'comparison-receipt.json')
        checked(receipt['report'])
        ct.require(receipt['all_exact'] is True, 'Fresh numerical suite incomplete')
    validate_phase_decode(report)
    ct.write(root / 'performance-receipt.json', dict(phase_decode_reduced=True,
        evidence=record(evidence), public_e2e_accepted=False, public_cache_enabled=False))
    return root / 'performance-receipt.json'


def main():
    ct.require_cpu_python()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('plan', 'prepare', 'run', 'compare', 'performance', 'worker'))
    ap.add_argument('--id')
    ap.add_argument('--gate', choices=tuple(ALL_GATES))
    ap.add_argument('--small-qualification', type=Path)
    ap.add_argument('--phase-evidence', type=Path)
    ap.add_argument('--sole-gpu-owner-granted', action='store_true')
    ap.add_argument('--owner-note', default='')
    ap.add_argument('--worker-pin')
    a = ap.parse_args()
    if a.mode == 'worker':
        return worker(a.worker_pin)
    if a.mode == 'plan':
        print(json.dumps(manifest(), indent=2))
    elif a.mode == 'prepare':
        print(prepare(a.id))
    elif a.mode == 'compare':
        print(compare(a.id, a.gate))
    elif a.mode == 'performance':
        print(performance(a.id, a.phase_evidence))
    else:
        print(run(a.id, a.gate, a.small_qualification, a.sole_gpu_owner_granted, a.owner_note))
    return 0


if __name__ == '__main__':
    sys.exit(main())
