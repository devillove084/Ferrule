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
import time

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
# The pinned dadf quality adapter audits only its original thirteen gates.
# New arena qualification must execute all current gates with fresh source pins.
HISTORICAL_SMALL_GATES = dict(SMALL_GATES)
HISTORICAL_ALL_GATES = {**{k: v[1] for k, v in HISTORICAL_SMALL_GATES.items()}, **GATES}
ARENA_SMALL_GATES = {
    'model-arena-fit': ('model', MODEL_PREFIX + 'decoded_arena_fit_working_set_hot_zero_eviction_decode_fallback', 1),
    'model-arena-lru': ('model', MODEL_PREFIX + 'decoded_arena_true_class_full_lru_and_size_class_fairness', 2),
    'model-arena-alias': ('model', MODEL_PREFIX + 'decoded_arena_alias_retains_whole_root', 1),
}
ARENA_SMALL_GATES['model-fixed-unknown'] = (
    'model', MODEL_PREFIX + 'decoded_fixed_pin_unknown_and_panic_never_publish_arena', 1)
ARENA_SMALL_GATES.update({
    'model-uninitialized-views': ('model', MODEL_PREFIX + 'decoded_uninitialized_arena_view_faults_retain_whole_root', 1),
    'model-first-fixed-failure': ('model', MODEL_PREFIX + 'decoded_first_fixed_failure_retained_until_completion_not_ready', 1),
    'model-uninitialized-slots': ('model', MODEL_PREFIX + 'decoded_uninitialized_slots_reject_consumers_until_ready', 1),
})
SMALL_GATES = {
    'backend-exact': HISTORICAL_SMALL_GATES['backend-exact'],
    'backend-preflight': HISTORICAL_SMALL_GATES['backend-preflight'],
    'backend-typed-view': ('backend-slice', 'bf16_slice_gpu_shared_pointer_no_work_parent_drop', 1),
    'backend-arena-allocation': ('backend-arena', 'gpu::arena_allocates_then_retains_root_before_full_decode', 1),
    'backend-arena-errors': ('backend-arena', 'gpu::arena_errors_never_submit_initialization', 1),
    'model-cold-hot-fit': ('model', MODEL_PREFIX + 'decoded_pair_single_cold_hot_fit_exact_and_no_cache_dma', 2),
    **{k: HISTORICAL_SMALL_GATES[k] for k in ('model-fixed-pins','model-unknown-panic','model-physical-full')},
    **ARENA_SMALL_GATES,
}
HOST_SMALL = [
    ('backend', BACKEND_PREFIX + name) for name in (
        'mixed_decode_preflight_without_device_calls', 'mixed_pair_decode_preflight_without_device_calls',
        'mixed_pair_preflight_without_device_calls')
] + [
    ('model', 'transformer::standard::cuda::hybrid_rank::mixed::preparation_tests::decoded_claim_tests::decoded_claim_compares_full_admission_even_when_grants_match'),
    ('model', MODEL_PREFIX + 'decoded_fixture_descriptors_preserve_pair_and_separate_fixed_roles'),
]
ARENA_HOST = [
    ('model', 'transformer::standard::mixed_decoded::tests::' + name) for name in (
        'pair_down_fit_set_is_stable_after_warmup_and_real_3366_mib_set_evicts',
        'every_class_at_capacity_has_no_warm_eviction_with_healthy_boundaries',
        'unknown_and_partial_install_failure_retain_root_and_all_slots',
        'failed_or_unsettled_root_allocation_never_refunds_or_retries',
        'existing_allocator_holes_do_not_change_full_root_charge',
    )
] + [
    ('model', 'transformer::standard::mixed_decoded::snapshot_tests::' + name) for name in (
        'decoded_snapshot_prepare_execute_aftercomplete_are_disjoint_and_read_only',
        'decoded_snapshot_pool_failed_reservations_and_unknown_remain_visible',
        'decoded_snapshot_partial_view_failure_partitions_installed_root_once',
    )
] + [('model', 'transformer::hybrid_parallel::mixed_decoded_arena::tests::'
      'three_gib_plan_has_disjoint_fixed_and_at_least_577_complete_experts')]
HOST_SMALL += ARENA_HOST + [
    ('model', MODEL_PREFIX + 'decoded_fixture_real_class_inventory_and_same_config_plan'),
    ('backend', 'cuda::allocator::dedicated_kv_tests::dedicated_kv_segment_never_lends_free_extent_to_weights_or_scratch'),
    *[('backend-slice', name) for name in (
        'bf16_slice_element_bounds_and_empty_policy',
        'bf16_slice_rejects_u16_element_byte_and_address_overflow',
        'bf16_slice_custody_delegates_to_existing_shared_root')],
    *[('backend-arena', name) for name in (
        'arena_checks_word_bytes_rounded_grant_and_budget',
        'arena_has_no_initialization_path_before_return_or_on_error')],
    *[('model-handoff', name) for name in (
        'dedicated_arena_handoff_precedes_any_post_allocation_observation',
        'slot_readiness_follows_full_decode_and_completion_classification')],
    ('runtime', 'absent_cache_snapshot_requires_no_cuda_owner'),
    *[('runtime', 'observation::decoded_snapshot_tests::' + name) for name in (
        'decoded_snapshot_runtime_json_preserves_physical_fields_and_unobserved_activity',
        'decoded_snapshot_runtime_json_preserves_profile_coverage_and_quarantine',
        'decoded_snapshot_runtime_json_preserves_preallocation_and_failed_root_reservation')],
]
GROUPED_PREFIX = 'transformer::standard::expert_cache::grouped_tests::'
GROUPED_CPU = [GROUPED_PREFIX + name for name in (
    'randomized_operations_match_old_linear_victims_credits_and_leases',
    'registration_is_frozen_duplicate_or_unregistered_keys_cannot_rebind',
    'completion_order_ties_epoch_overflow_and_live_leases_preserve_index',
    'cancel_consuming_and_uploading_never_publish_and_stale_proof_cannot_reload',
)]
GROUPED_BENCH = GROUPED_PREFIX + 'cpu_grouped_selector_benchmark'
HOST_SMALL += [('model', selector) for selector in (*GROUPED_CPU, GROUPED_BENCH)]
IGNORED_HOST = {('model', GROUPED_BENCH)}
INDEXED_LRU_SOURCES = {
    str(ROOT/'crates/ferrule-model/src/transformer/standard/expert_cache.rs'):
        'cbb6b8cf01119ca649f3446ecf5a189fcd0e3d0886d449b1aa0a9989428be229',
    str(ROOT/'crates/ferrule-model/src/transformer/standard/mixed_decoded.rs'):
        'bfb5faa7f7098b067df7e2e463a281e43547d1c95314d2aa6289aa2c9f28859f',
}


def validate_indexed_sources(sources):
    ct.require(all(sources.get(path) == pin for path, pin in INDEXED_LRU_SOURCES.items()),
               'Indexed LRU production source differs from the requested frozen SHA')


def benchmark_summary(text):
    rows = [line for line in text.splitlines() if line.startswith('CPU only indexed=')]
    ct.require(len(rows) == 2 and rows[0].startswith('CPU only indexed=false ')
        and rows[1].startswith('CPU only indexed=true ')
        and all('entries=1155 classes=2 descriptors=20000 misses=50000 ' in row
                and 'not E2E' in row for row in rows), 'CPU benchmark summary missing or changed scope')
    return rows


ALL_GATES = {**{k: v[1] for k, v in SMALL_GATES.items()}, **GATES}
V3_FREEZE = ROOT / 'crates/ferrule-backend/tests/allocator_batch_v3/FREEZE_V3.sha256'
V3_FREEZE_SHA = '44a551c3772e5456a842d95133e9cbdd29dadafcc3d8fb29f15205cdb7574344'
V3_SMALL = HERE / 'runs/allocator-v3-fresh-20261011-02/CLOSEOUT.json'
V3_SMALL_SHA = 'afe4787a8762d08d46eaeaa41887973aeb00ea7857ac5cfcbbbded9c4eac058f'
V3_PREPARED_SCHEMA = 'ferrule.paper.allocator-v3-prepared.v1'
V3_ACCEPTANCE_SCHEMA = 'ferrule.paper.allocator-v3-acceptance.v1'


def validate_allocator_v3():
    ct.require(ct.sha(V3_FREEZE) == V3_FREEZE_SHA, 'Allocator V3 freeze manifest drift')
    for line in V3_FREEZE.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        ct.require(ct.sha(ROOT / path) == digest, 'Allocator V3 frozen input drift: ' + path)
    ct.require(ct.sha(V3_SMALL) == V3_SMALL_SHA, 'Allocator small-gate closeout drift')
    small = ct.load(V3_SMALL)
    ct.require(small['gpu_passed'] == 18 and small['gpu_failed'] == 0
        and small['freeze_sha256'] == V3_FREEZE_SHA and not small['production_freeze_drift'],
        'Allocator supplemental qualification incomplete')
    ct.check_pins({str(ROOT / path): sha for path, sha in small['artifacts_sha256'].items()})
    return dict(freeze=record(V3_FREEZE), supplemental_small_gate=record(V3_SMALL),
        supplemental_gate_count=18, counted_toward_named_23=False,
        reason='Different exact selectors; all five required backend gates rerun fresh')


def allocator_v3_rows(log, epochs):
    messages = {'Mixed owner preparation claim completion': 'ready',
                'Mixed owner shutdown fence; image retained until worker retirement': 'shutdown'}
    rows, by_owner = [], {0: [], 1: []}
    required = {'retired_blocks', 'retirement_batches', 'retirement_created_fences',
        'pending_retirement_blocks', 'pending_retirement_bytes', 'pending_retirement_batches',
        'pending_unfenced_bytes', 'reserved_bytes', 'peak_reserved_bytes', 'allocation_failures',
        'capture_retirement_bytes', 'shutdown_quiescent', 'shutdown_retained'}
    ct.require('driver growth deferred by ordinary retirement pressure' not in log,
        'Normal mixed operation hit allocator retirement pressure; no threshold relaxation')
    for line in log.splitlines():
        if 'CudaAllocatorMetrics {' not in line:
            continue
        event = json.loads(line[line.find('{'):])['fields']
        ct.require(event.get('message') in messages and event.get('owner') in by_owner,
                   'Unrecognized allocator observation boundary')
        raw = event['allocator']
        ct.require(raw.startswith('CudaAllocatorMetrics { ') and raw.endswith(' }'),
                   'Allocator Debug format changed')
        fields = dict(part.split(': ', 1) for part in raw[len('CudaAllocatorMetrics { '):-2].split(', '))
        values = {k: json.loads(v) for k, v in fields.items()}
        ct.require(required <= values.keys(), 'Missing V3 allocator metric fields')
        ct.require(all(type(values[k]) is int and values[k] >= 0 for k in required
            if k not in ('shutdown_quiescent', 'shutdown_retained')), 'Invalid allocator metric type')
        ct.require(values['allocation_failures'] == 0 and values['peak_reserved_bytes'] <= 22 << 30,
                   'Allocator allocation failure or real reservation cap exceeded')
        ct.require(not event['device_quarantine'] and not values['shutdown_retained'],
                   'Normal owner became quarantined')
        stage = messages[event['message']]
        ct.require(event['claim_finished'] if stage == 'ready' else event['quiescent'],
                   'Owner boundary is not known complete')
        ct.require(all(values[k] == 0 for k in ('pending_retirement_blocks', 'pending_retirement_bytes',
            'pending_retirement_batches', 'pending_unfenced_bytes', 'capture_retirement_bytes')),
            'Quiescent owner boundary still has pending retirement')
        owner = event['owner']
        row = dict(owner=owner, epoch=len(by_owner[owner]) // 2, stage=stage, metrics=values)
        by_owner[owner].append(row); rows.append(row)
    ct.require(all([r['stage'] for r in group] == ['ready', 'shutdown'] * epochs
        for group in by_owner.values()), 'Missing/duplicate allocator owner epochs')
    for group in by_owner.values():
        for start in range(0, len(group), 2):
            before, after = (r['metrics'] for r in group[start:start+2])
            ct.require(all(after[k] >= before[k] for k in
                ('retired_blocks', 'retirement_batches', 'retirement_created_fences')),
                'Allocator cumulative counters regressed within owner epoch')
    return dict(schema='ferrule.paper.allocator-v3-observations.v1', rows=rows,
        observed_pending_peak_bytes=max(r['metrics']['pending_retirement_bytes'] for r in rows),
        exact_pending_high_watermark_bytes=None,
        observation_scope='Existing ready/shutdown boundary samples, not an all-operation pending high-watermark',
        peak_reserved_bytes=max(r['metrics']['peak_reserved_bytes'] for r in rows),
        ordinary_pressure_error=False, quiescent_pending_zero=True)

CPU = {
    'decoded_cache::native_default_off_and_cache_report_claim_share_exact_grant': False,
    'decoded_cache::cache_rejects_low_budget_capacity_and_configuration_drift': False,
    'decoded_cache::cache_capacity_follows_device_placement_not_lane_number': False,
    'decoded_cache::admitted_cache_programs_survive_runtime_binding_and_reject_owner_geometry_drift': False,
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


def execution_devices(gate):
    ct.require(gate in ALL_GATES, 'Unknown gate device contract')
    selected = UUIDS[:SMALL_GATES[gate][2]] if gate in SMALL_GATES else list(UUIDS)
    return dict(selected=selected, environment=dict(
        CUDA_VISIBLE_DEVICES=','.join(selected), NVIDIA_VISIBLE_DEVICES=','.join(selected)))


def validate_execution_devices(gate, sealed):
    ct.require(sealed == execution_devices(gate), 'Visible/selected device contract drift')
    return sealed


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
    return dict(schema='ferrule.paper.allocator-v3-plan.v1', public_enabled=False,
        allocator_v3=dict(freeze_sha256=V3_FREEZE_SHA, supplemental_small_sha256=V3_SMALL_SHA,
            named_gates=list(ALL_GATES), supplemental_not_replacement=True,
            pending_bytes_growth_watermark=256 << 20, pending_batches_growth_watermark=256),
        cache_grants_per_owner=3 << 30, device_capacity=22 << 30,
        pinned_grants=1340 << 20, routed_working_set_grants=768 << 20,
        cpu_selectors=CPU, host_small_selectors=HOST_SMALL, ignored_host_selectors=sorted(IGNORED_HOST),
        indexed_lru_source_pins=INDEXED_LRU_SOURCES, gpu_selectors=GATES,
        small_selectors=SMALL_GATES, small_required=sorted(SMALL_CASES),
        numerical='native BF16 bit-exact; M1/M5, ordered full-2N routed pair GEMM; no W8',
        custody='healthy completion only; Unknown/borrowed roots never refunded or reused',
        arena_required=dict(physical_root_cap_bytes=3 << 30,
            allocation_policy='dedicated_uninitialized_root_retained_before_initialization',
            driver_growth_equals_full_root_grant=True, hot_fit_evictions=0,
            hot_fit_decodes=0, accounting='schema v2: full root once; logical slots/free/tail/pending partition'),
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
        ROOT / 'crates/ferrule-model/tests/support/mixed_decoded_arena_handoff.md',
        ROOT / 'crates/ferrule-backend/tests/numeric_bf16_gemv/mixed_decode_handoff.md',
        *sorted(HERE.glob('*.py'))]


def cpu_command(out, name, argv, env, timeout):
    # Reuse bounded process-tree supervision for compilers as well as test children.
    code = 'import subprocess,sys; o=open(sys.argv[1],"x"); e=open(sys.argv[2],"x"); sys.exit(subprocess.call(sys.argv[3:],stdout=o,stderr=e))'
    print('CPU start:', name, flush=True)
    result = supervisor.run_process(ct.isolated_argv([sys.executable, '-S', '-B', '-c', code,
        str(out / (name + '.stdout')), str(out / (name + '.stderr')), *map(str, argv)], env), timeout)
    ct.write(out / (name + '.json'), dict(argv=list(map(str, argv)), supervisor=result))
    ct.require(ct.successful_process(result), 'CPU step failed/timed out: ' + name)
    print('CPU passed:', name, 'seconds=', round(result['wall_seconds'], 3), flush=True)
    return (out / (name + '.stdout')).read_text()


def prepare(identity):
    out = run_directory(identity, True)
    env = ct.build_environment(ct.config(ct.EXCLUSIVE_PATH))
    env.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', FERRULE_CUDA_TEST_ORACLE='0', FERRULE_NO_CUDA='1')
    allocator_v3 = validate_allocator_v3()
    sources = ct.source_pins()
    validate_indexed_sources(sources)
    ct.write(out / 'sources-before.json', sources)
    ct.write(out / 'environment.json', env)
    ct.write(out / 'plan.json', manifest())
    tools = ct.required_inventory(ct.config(ct.EXCLUSIVE_PATH))
    ct.require('release 12.4' in subprocess.check_output([env['NVCC'], '--version'], env=env, text=True, timeout=10), 'Need CUDA12.4')
    cpu_command(out, 'backend-unit-release', [ct.EXCLUSIVE['cargo'], 'test', '--locked', '--offline',
        '--release', '-p', 'ferrule-backend', '--features', 'cuda', '--lib', '--target-dir',
        ROOT / 'target/validation/qwen35-35b-current/toolchain/cargo-cuda124', '--', '--test-threads=1'], env, 600)
    cpu_command(out, 'workflow-tests', [sys.executable, '-S', '-B', '-m', 'unittest', 'discover',
        '-s', HERE, '-p', 'test_decoded_cache_gate.py', '-v'], env, 60)
    for label, subcommand in (('cuda-check', 'check'), ('cuda-clippy', 'clippy')):
        argv = [ct.EXCLUSIVE['cargo'], subcommand, '--locked', '--offline', '-p', 'ferrule-model',
            '--features', 'cuda', '--lib', '--tests', '--target-dir',
            ROOT / 'target/validation/qwen35-35b-current/toolchain/cargo-cuda124']
        if subcommand == 'clippy':
            argv += ['--', '-D', 'warnings']
        cpu_command(out, label, argv, env, 600)
    binaries = {}
    targets = [('backend', 'ferrule-backend', 'ferrule_backend', ['--lib']),
               ('backend-slice', 'ferrule-backend', 'cuda_bf16_slice', ['--test', 'cuda_bf16_slice']),
               ('backend-arena', 'ferrule-backend', 'cuda_bf16_arena', ['--test', 'cuda_bf16_arena']),
               ('model', 'ferrule-model', 'ferrule_model', ['--lib']),
               ('model-handoff', 'ferrule-model', 'mixed_decoded_arena_handoff', ['--test', 'mixed_decoded_arena_handoff']),
               ('runtime', 'ferrule-runtime', 'mixed_decoded_snapshot', ['--test', 'mixed_decoded_snapshot']),
               ('composition', 'ferrule-composition', 'tp2_mixed_execution', ['--test', 'tp2_mixed_execution'])]
    for key, package, name, flags in targets:
        text = cpu_command(out, 'build-' + key, [ct.EXCLUSIVE['cargo'], 'test', '--locked', '--offline',
            '-p', package, '--features', 'cuda', *flags, '--no-run', '--message-format=json', '--target-dir',
            ROOT / 'target/validation/qwen35-35b-current/toolchain/cargo-cuda124'], env, 600)
        artifacts = [json.loads(line) for line in text.splitlines() if line.startswith('{')]
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
    cpu = [(key, selector, (key, selector) in IGNORED_HOST) for key, selector in HOST_SMALL]
    cpu += [('composition', selector, ignored) for selector, ignored in CPU.items()]
    for index, (key, selector, ignored) in enumerate(cpu):
        argv = [binaries[key]['path'], selector, '--exact', '--nocapture', '--test-threads=1']
        if ignored:
            argv.append('--ignored')
        text = cpu_command(out, 'cpu-' + str(index), argv,
            dict(env, FERRULE_MIXED_HEADER_DIR=MODEL), 600 if ignored else 180)
        ct.require('1 passed; 0 failed; 0 ignored' in text, 'CPU selector did not execute')
        receipt = record(out / ('cpu-' + str(index) + '.json'))
        receipts.append(receipt)
        if selector == GROUPED_BENCH:
            stderr = out / ('cpu-' + str(index) + '.stderr')
            ct.write(out / 'cpu-indexed-lru-benchmark.json', dict(
                schema='ferrule.paper.indexed-lru-cpu-benchmark.v1', selector=selector,
                binary=binaries[key], source_digest=ct.digest(sources), production_sources=INDEXED_LRU_SOURCES,
                test_source=record(ROOT/'crates/ferrule-model/tests/unit/transformer/grouped_expert_cache.rs'),
                supervisor_receipt=receipt, raw_stderr=record(stderr),
                summary_lines=benchmark_summary(stderr.read_text()),
                equal_victim_sequence_stats_and_final_entries_asserted=True,
                timing_assertion=False, gpu_used=False, end_to_end=False,
                scope='1155 resident entries, 20000 descriptors, 50000 misses; excludes preparation, includes lease/completion/index maintenance; not E2E'))
    ct.require(sources == ct.source_pins(), 'Sources changed during preparation; no promotion')
    ct.validate_inventory(ct.config(ct.EXCLUSIVE_PATH), tools)
    pins = ct.inventory(workflow_paths())
    pins.update(ct.inventory([out / 'plan.json', out / 'environment.json']))
    for pattern in ('build-*', 'cpu-*', 'cuda-*', 'backend-unit-release.*', 'workflow-tests.*'):
        for file in out.glob(pattern):
            pins[str(file)] = ct.sha(file)
    for ref, sha in [FULL, *RECIPES.values()]:
        path = ORACLE / 'runs' / ref / 'manifest.json'
        ct.require(ct.sha(path) == sha, 'Frozen reference changed')
        pins[str(path)] = sha
    ct.write(out / 'prepared.json', dict(schema=V3_PREPARED_SCHEMA, allocator_v3=allocator_v3,
        binary=binaries['composition'], binaries=binaries, sources=sources, pins=pins, tools=tools, cpu_receipts=receipts,
        public_enabled=False, gpu_run=False))
    return out


def reseal(identity, previous_identity):
    """Workflow-only correction: preserve CPU/build proof, inherit no GPU pass."""
    previous = run_directory(previous_identity)
    old = ct.load(previous / 'prepared.json')
    ct.require(old['schema'] == V3_PREPARED_SCHEMA, 'Need current three-binary preparation')
    ct.require(old['sources'] == ct.source_pins(), 'Production/test sources changed: rebuild required')
    validate_required_plan(previous, old)
    ct.check_pins(old['sources'])
    ct.validate_inventory(ct.config(ct.EXCLUSIVE_PATH), old['tools'])
    allowed = {str(HERE / name) for name in ('decoded_cache_gate.py', 'decoded_cache_compare.py')}
    preserved = {path: sha for path, sha in old['pins'].items() if path not in allowed}
    ct.check_pins(preserved)
    for binary in old['binaries'].values():
        ct.validate_staged_binary(binary['path'], binary['sha256'])
    for receipt in old['cpu_receipts']:
        ct.require(ct.successful_process(ct.load(checked(receipt))['supervisor']), 'Prior CPU gate not passed')
    out = run_directory(identity, True)
    binaries = {}
    for key, binary in old['binaries'].items():
        directory = out / ('binary-' + key)
        directory.mkdir()
        binaries[key] = record(ct.stage_binary(binary['path'], directory))
    env = ct.load(previous / 'environment.json')
    ct.require(env['CUDA_VISIBLE_DEVICES'] == '' and env['FERRULE_CUDA_ARCH'] == 'sm_86', 'Hidden SM86 preparation required')
    ct.write(out / 'environment.json', env)
    ct.write(out / 'plan.json', manifest())
    cpu_command(out, 'workflow-tests', [sys.executable, '-S', '-B', '-m', 'unittest', 'discover',
        '-s', HERE, '-p', 'test_decoded_cache_gate.py', '-v'], env, 60)
    pins = dict(preserved)
    pins.update(ct.inventory(workflow_paths()))
    pins.update(ct.inventory([previous / 'prepared.json', out / 'plan.json', out / 'environment.json',
        HERE / 'test_decoded_cache_gate.py', *out.glob('workflow-tests.*')]))
    ct.write(out / 'prepared.json', dict(schema=old['schema'], allocator_v3=old['allocator_v3'], binary=binaries['composition'], binaries=binaries,
        sources=old['sources'], tools=old['tools'], pins=pins, cpu_receipts=old['cpu_receipts'],
        reused_cpu_build=record(previous / 'prepared.json'), inherited_gpu_passes=[],
        workflow_changes={path: dict(before=old['pins'].get(path), after=ct.sha(path)) for path in allowed},
        public_enabled=False, gpu_run=False))
    validate(out)
    return out


def validate_required_plan(out, prepared):
    ct.require(ct.digest(ct.load(out / 'plan.json')) == ct.digest(manifest()),
               'Required arena gate contract changed: fresh build/CPU preparation required')
    receipts = prepared['cpu_receipts']
    ct.require(len(receipts) == len(HOST_SMALL) + len(CPU)
        and len({r['path'] for r in receipts}) == len(receipts), 'Required CPU gate receipts missing/duplicated')


def validate(out):
    p = ct.load(out / 'prepared.json')
    ct.require(p['schema'] == V3_PREPARED_SCHEMA, 'Wrong preparation')
    ct.check_pins(p['pins'])
    validate_required_plan(out, p)
    ct.check_pins(p['sources'])
    ct.require(p['sources'] == ct.source_pins(), 'Source inventory drift')
    validate_indexed_sources(p['sources'])
    ct.require(p['allocator_v3'] == validate_allocator_v3(), 'V3 supplemental/freeze binding drift')
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


def internal_snapshot_rows(log, bindings):
    """Reuse the actual release ledger validator, once per recreated owner epoch."""
    import decoded_cache_release as release
    by_owner = {0: [], 1: []}
    for line in log.splitlines():
        if 'mixed_decoded_cache_snapshot' not in line:
            continue
        # libtest may prefix the first JSON event with "test selector ... ".
        start = line.find('{')
        ct.require(start >= 0, 'Missing actual cache snapshot JSON')
        record = json.loads(line[start:])
        fields = record.get('fields', {})
        ct.require(fields.get('event') == 'mixed_decoded_cache_snapshot'
            and fields.get('owner') in by_owner and fields.get('schema_version') == 2,
            'Internal arena snapshot missing/foreign/old schema')
        by_owner[fields['owner']].append(record)
    ct.require(len(bindings) > 0 and len(bindings) % 2 == 0, 'Missing paired owner bindings')
    epochs = len(bindings) // 2
    ct.require(all(len(rows) == 2 * epochs for rows in by_owner.values()), 'Missing/duplicate internal arena snapshots')
    result = []
    generations = {0: set(), 1: set()}
    for epoch in range(epochs):
        records = []
        for owner in (0, 1):
            pair = by_owner[owner][2*epoch:2*epoch+2]
            ct.require([r['fields']['stage'] for r in pair] == ['ready', 'shutdown'],
                       'Internal arena owner did not close before recreate')
            records.extend(pair)
        rows = release.snapshot_rows('\n'.join(json.dumps(r) for r in records))
        for row in rows:
            snapshot = row['snapshot']
            ct.require(snapshot['driver_allocation_growth_bytes'] == snapshot['arena_backing_grants']
                == snapshot['capacity_grants'], 'Dedicated root driver growth must equal the full arena grant')
            owner = row['owner']
            generation = row['snapshot']['image_generation']
            if row['stage'] == 'ready':
                ct.require(generation not in generations[owner], 'Recreated cache reused image identity')
                generations[owner].add(generation)
            result.append(dict(epoch=epoch, **row))
    return dict(schema='ferrule.paper.arena-snapshots.v1', rows=result,
        measurement='actual dedicated cache-only host ledger; driver growth equals full root grant, root counted once; not global device memory')


def wait_observed_test(proc, monitor, timeout):
    # The observer is the sole poll/waitpid caller while it samples. Popen.wait
    # on this thread would make its concurrent poll spuriously return None.
    deadline = time.monotonic() + timeout
    while monitor.thread.is_alive():
        if proc.returncode is not None:
            return proc.returncode
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(proc.args, timeout)
        monitor.thread.join(timeout=min(.05, remaining))
    # After sampling stops, no observer can race this final bounded wait.
    return proc.wait(timeout=max(.001, deadline - time.monotonic()))


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
    devices = validate_execution_devices(gate, p['devices'])
    env = dict(PATH='/usr/bin:/bin', HOME=str(Path.home()), **devices['environment'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
        LD_LIBRARY_PATH='/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64',
        CUBLAS_WORKSPACE_CONFIG=':0:0', FERRULE_MIXED_GPU_RUN='1', FERRULE_RUN_MIXED_RANK_GPU='1',
        FERRULE_MIXED_MODEL=MODEL, FERRULE_MIXED_ORACLE_DIR=str(ORACLE / 'runs' / FULL[0]),
        FERRULE_MIXED_ORACLE_SHA256=FULL[1], FERRULE_MIXED_EVIDENCE_DIR=str(out),
        FERRULE_PAPER_WORKER_TOKEN=p['worker_token'], FERRULE_LOG='info', FERRULE_LOG_FORMAT='json')
    if gate in RECIPES:
        ref, sha = RECIPES[gate]
        env.update(FERRULE_MIXED_LIFECYCLE_ORACLE_DIR=str(ORACLE / 'runs' / ref), FERRULE_MIXED_LIFECYCLE_ORACLE_SHA256=sha)
    ct.write(out / 'environment.json', env)
    with (out / 'test.log').open('x') as log:
        with ct.pinned_executable(p['binary'], p['binary_sha256']) as fd:
            proc = subprocess.Popen(['/proc/self/fd/' + str(fd), ALL_GATES[gate], '--exact', '--ignored',
                '--nocapture', '--test-threads=1'], env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
        monitor = observer.Monitor(out, proc, CONFIG, auth, devices['selected'],
                                   p['owner_note'], p['baseline'],
                                   ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'LD_LIBRARY_PATH'))
        monitor.start()
        try:
            code = wait_observed_test(proc, monitor, 1600)
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
        bindings = trace_budgets(out, gate)
        ct.write(out / 'cache-bindings.json', bindings)
        ct.write(out / 'cache-snapshots.json', internal_snapshot_rows(text, bindings))
        ct.write(out / 'allocator-v3-observations.json', allocator_v3_rows(text, len(bindings) // 2))
    else:
        ct.write(out / 'small-receipt.json', dict(selector=ALL_GATES[gate], binary_sha256=p['binary_sha256'],
            source_digest=ct.digest(p['sources']), assertions_passed=True,
            cache_values='source-pinned model assertions; no raw per-step counters exported',
            assertion_contract=(dict(hot_compute_launch_delta=0, evictions=0, native_fallbacks=0,
                postprepare_decode_allocation_request_delta=0, postprepare_decode_granted_bytes_delta=0,
                postprepare_decode_driver_bytes_delta=0, basis='successful exact source-pinned assertions, not sampled counters')
                if gate == 'model-arena-fit' else None),
            gpu_memory=record(out / 'resources.json'), log=record(out / 'test.log')))
    ct.check_pins(p['pins'])
    ct.check_pins(p['sources'])
    return 0


REUSABLE_BACKEND = ('backend-exact', 'backend-preflight', 'backend-typed-view')
SCOPE_FIX_SOURCES = tuple(str(ROOT / p) for p in (
    'crates/ferrule-model/src/transformer/standard/cuda/hybrid_rank/mixed.rs',
    'crates/ferrule-model/tests/unit/transformer/mixed_decoded_gpu.rs'))


def backend_reuse_relation(old, current):
    changed = {p for p in set(old['sources']) | set(current['sources'])
               if old['sources'].get(p) != current['sources'].get(p)}
    ct.require(set(old['sources']) == set(current['sources'])
        and changed == set(SCOPE_FIX_SOURCES), 'Backend reuse permits only the reviewed two-file model test-scope fix')
    ct.require(old['tools'] == current['tools'], 'Backend tool/dependency inventory changed')
    for kind in ('backend', 'backend-slice'):
        ct.require(old['binaries'][kind]['sha256'] == current['binaries'][kind]['sha256'],
                   'Backend test binary changed: '+kind)
    ct.require(old['binaries']['model']['sha256'] != current['binaries']['model']['sha256'],
               'Test-scope fix requires a freshly linked model binary')
    return {p:dict(before=old['sources'][p], after=current['sources'][p]) for p in sorted(changed)}


def reuse_backend(identity, previous_identity):
    """Reuse only three unchanged backend tests, never a model/source-wide PASS."""
    root, previous = run_directory(identity), run_directory(previous_identity)
    current = validate(root)
    old = ct.load(previous / 'prepared.json')
    changes = backend_reuse_relation(old, current)
    ct.require(ct.load(root/'environment.json') == ct.load(previous/'environment.json'), 'Build environment changed')
    old_plan = ct.load(previous/'plan.json')
    closure = ct.load(previous/'failure-closeout.json')
    prior_rows = {r['gate']:r for r in closure['rows']}
    paths = [root/'prepared.json', previous/'prepared.json', previous/'failure-closeout.json',
             previous/'plan.json', previous/'environment.json']
    for name in REUSABLE_BACKEND:
        ct.require(not (root/name).exists(), 'Cannot mix reused and freshly executed backend gates')
        ct.require(old_plan['small_selectors'][name] == list(SMALL_GATES[name]), 'Historical backend selector/roster differs')
        ct.require(prior_rows[name]['passed'] is True, 'Prior backend gate not passed')
        for link in prior_rows[name]['evidence'].values():
            checked(link)
        kind = SMALL_GATES[name][0]
        ct.validate_staged_binary(old['binaries'][kind]['path'], old['binaries'][kind]['sha256'])
        paths.append(Path(old['binaries'][kind]['path']))
        directory = previous/name
        worker = ct.load(directory/'worker-launch.json')
        launch = ct.load(directory/'supervisor.json')
        auth = ct.load(directory/'authentication.json')
        exit = ct.load(directory/'test-exit.json')
        result = ct.load(directory/'result.json')
        release = ct.load(directory/'release.json')
        resource = ct.load(directory/'resources.json')
        ct.require(ct.successful_process(launch) and launch['custody']['echild'] is True
            and auth['launch_id'] == launch['handshake']['launch_id']
            and result['passed'] is True and result['drained'] is True
            and all(release[k] is True for k in ('drained','gpu_owner_released','all_eight_idle')),
            'Prior backend authenticated closure incomplete')
        ct.require(exit['one_test_passed'] is True and exit['returncode'] == 0
            and exit['selector'] == ALL_GATES[name]
            and exit['binary_sha256'] == current['binaries'][kind]['sha256']
            and worker['binary_sha256'] == exit['binary_sha256'] and worker['sources'] == old['sources'],
            'Prior backend source/binary/selector relationship differs')
        validate_execution_devices(name, worker['devices'])
        ct.require(resource['status'] == 'observed_within_targets' and not resource['errors'], 'Prior backend resource failure')
        for phase in ('before','after'):
            ct.require(ct.load(directory/(phase+'.json'))['status'] == 'verified', 'Prior backend idle proof missing')
        paths.extend(directory/n for n in ('worker-launch.json','supervisor.json','authentication.json',
            'test-exit.json','result.json','release.json','resources.json','before.json','after.json','test.log','small-receipt.json'))
    proof = dict(schema='ferrule.paper.unchanged-backend-reuse.v1', gates=list(REUSABLE_BACKEND),
        previous_prepared=record(previous/'prepared.json'), current_prepared=record(root/'prepared.json'),
        changed_model_test_sources=changes, pins=ct.inventory(paths),
        scope='Only unchanged backend sources/tools/bit-identical binaries and original authenticated receipts; all eight model gates must run fresh')
    ct.write(root/'backend-reuse.json', proof)
    return root/'backend-reuse.json'


def backend_reuse(root, current):
    path = root/'backend-reuse.json'
    if not path.exists():
        return None
    proof = ct.load(path)
    ct.require(proof['schema'] == 'ferrule.paper.unchanged-backend-reuse.v1'
        and proof['gates'] == list(REUSABLE_BACKEND)
        and proof['current_prepared'] == record(root/'prepared.json'), 'Backend reuse scope differs')
    ct.check_pins(proof['pins'])
    old_path = checked(proof['previous_prepared'])
    old = ct.load(old_path)
    ct.require(proof['changed_model_test_sources'] == backend_reuse_relation(old, current), 'Backend reuse source relation changed')
    return old_path.parent, old


def run(identity, gate, qualification, granted, note):
    ct.require(granted and bool(note.strip()), 'Fresh explicit sole GPU owner grant required')
    ct.require(gate in ALL_GATES, 'Need a known exact gate')
    root = run_directory(identity)
    ct.require(not (root / 'STOP.json').exists(), 'Prior failure: no further GPU work before root fix/reseal')
    prepared = validate(root)
    inherited = backend_reuse(root, prepared)
    ct.require(not inherited or gate not in REUSABLE_BACKEND, 'Backend gate already bound as explicitly reused')
    preceding = list(ALL_GATES)[:list(ALL_GATES).index(gate)]
    evidence_paths = [root / 'prepared.json']
    if inherited:
        evidence_paths.append(root/'backend-reuse.json')
        evidence_paths.extend(Path(p) for p in ct.load(root/'backend-reuse.json')['pins'])
    for prior in preceding:
        previous = (inherited[0] if inherited and prior in REUSABLE_BACKEND else root) / prior
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
        binary_sha256=binary['sha256'], pins=pins, sources=prepared['sources'], owner_note=note,
        devices=execution_devices(gate))
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
    bindings = trace_budgets(out, gate)
    ct.require(ct.load(out / 'cache-snapshots.json') == internal_snapshot_rows((out / 'test.log').read_text(), bindings),
               'Actual arena snapshot evidence changed')
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


def closeout(identity):
    """CPU-only closure of fresh executed gates; never inherits old dadf PASS."""
    root = run_directory(identity)
    ct.require(not (root / 'STOP.json').exists(), 'Failed preparation cannot close out')
    prepared = validate(root)
    inherited = backend_reuse(root, prepared)
    reused = list(REUSABLE_BACKEND) if inherited else []
    rows = []
    calls = metrics = 0
    for name, selector in ALL_GATES.items():
        directory = (inherited[0] if name in reused else root) / name
        evidence_sources = inherited[1]['sources'] if name in reused else prepared['sources']
        kind = SMALL_GATES[name][0] if name in SMALL_GATES else 'composition'
        binary = prepared['binaries'][kind]
        launch = ct.load(directory / 'supervisor.json')
        auth = ct.load(directory / 'authentication.json')
        worker = ct.load(directory / 'worker-launch.json')
        exit = ct.load(directory / 'test-exit.json')
        release = ct.load(directory / 'release.json')
        result = ct.load(directory / 'result.json')
        resources = ct.load(directory / 'resources.json')
        ct.require(ct.successful_process(launch) and launch['custody']['echild'] is True
            and auth['launch_id'] == launch['handshake']['launch_id']
            and result['passed'] is True and result['drained'] is True
            and all(release[k] is True for k in ('gpu_owner_released','drained','all_eight_idle')),
            'Gate lacks authenticated drain/release: '+name)
        ct.require(exit['one_test_passed'] is True and exit['returncode'] == 0
            and exit['selector'] == selector and exit['binary_sha256'] == binary['sha256']
            and worker['gate'] == name and worker['binary_sha256'] == binary['sha256']
            and worker['sources'] == evidence_sources, 'Gate source/selector/binary mismatch: '+name)
        validate_execution_devices(name, worker['devices'])
        ct.require(resources['status'] == 'observed_within_targets' and not resources['errors'],
                   'Resource gate failed: '+name)
        for phase in ('before','after'):
            ct.require(ct.load(directory / (phase+'.json'))['status'] == 'verified', 'Idle proof missing: '+name)
        item = dict(gate=name, selector=selector, binary_sha256=binary['sha256'],
            reused_unchanged_backend=name in reused,
            launch_id=auth['launch_id'], selected_and_visible=execution_devices(name),
            **{key: record(directory / (key+'.json')) for key in (
                'supervisor','authentication','worker-launch','test-exit','resources','release','before','after','result')},
            log=record(directory / 'test.log'))
        if name in SMALL_GATES:
            item['assertions'] = record(directory / 'small-receipt.json')
        else:
            receipt = ct.load(directory / 'comparison-receipt.json')
            comparison = ct.load(checked(receipt['report']))
            ct.require(receipt['all_exact'] is True, 'Independent comparator incomplete')
            exact_inventory(comparison, name)
            groups = comparison['cases'] if name == 'full40' else comparison['calls']
            ct.require(all(m['rmse'] == 0 for g in groups.values() for m in g['metrics'].values()),
                       'Nonzero exact comparison RMSE')
            count = sum(len(g['metrics']) for g in groups.values())
            call_count = 4 if name == 'full40' else len(groups)
            bindings = trace_budgets(directory, name)
            snapshots = internal_snapshot_rows((directory / 'test.log').read_text(), bindings)
            ct.require(ct.load(directory / 'cache-snapshots.json') == snapshots, 'Snapshot evidence drift')
            allocator = allocator_v3_rows((directory / 'test.log').read_text(), len(bindings) // 2)
            ct.require(ct.load(directory / 'allocator-v3-observations.json') == allocator, 'Allocator observations drift')
            item['allocator_v3'] = record(directory / 'allocator-v3-observations.json')
            item.update(calls=call_count, metrics=count, all_exact=True, comparison=receipt['report'],
                comparison_receipt=record(directory / 'comparison-receipt.json'),
                snapshots=record(directory / 'cache-snapshots.json'))
            calls += call_count
            metrics += count
        rows.append(item)
    ct.require((calls, metrics) == (37, 40293), 'Full40/six-recipe exact inventory incomplete')
    validate(root)
    summary = dict(schema=V3_ACCEPTANCE_SCHEMA, allocator_v3=prepared['allocator_v3'],
        named_required_gates=list(ALL_GATES),
        status='fresh_allocator_v3_indexed_arena_numerical_quality_pass', prepared=record(root / 'prepared.json'),
        sources=prepared['sources'], binaries=prepared['binaries'], tools=prepared['tools'],
        fresh_gpu_gates=len(rows)-len(reused), fresh_small_gates=len(SMALL_GATES)-len(reused),
        reused_backend_gates=reused, backend_reuse=record(root/'backend-reuse.json') if inherited else None, full_model_calls=calls,
        quality_metrics=metrics, all_exact=True, failures=0, max_abs=0, rows=rows,
        prior_gpu_passes_inherited=bool(reused), performance_accepted=False, public_e2e_accepted=False,
        gpu_owner_released=True, all_eight_idle=True,
        cache_observation='Actual ready/shutdown root partitions; small hot deltas proved by source-pinned assertions, raw per-step counters not exported')
    ct.write(root / 'acceptance-summary.json', summary)
    return root / 'acceptance-summary.json'


def init_probe_worker(pin):
    """One driver initialization and, only on success, the pinned official sample."""
    import ctypes
    p = json.load(sys.stdin)
    auth = guard.authenticate(p, pin)
    ct.check_pins(p['pins'])
    ct.check_pins(p['sources'])
    out = Path(p['directory'])
    ct.write(out / 'authentication.json', auth)
    for uuid in UUIDS:
        ct.validate_owner_lock(p['locks'][uuid], uuid, p['coordinator']['pid'])
    devices = execution_devices('backend-exact')
    ct.require(all(p['environment'][k] == v for k,v in devices['environment'].items()), 'Probe roster differs')
    # main's CPU-only import boundary hides devices. Restore only the sealed,
    # authenticated roster before this worker's first CUDA library load.
    os.environ.update(devices['environment'])
    for key, value in p['environment'].items():
        ct.require(os.environ.get(key) == value, 'Probe environment differs: '+key)
    ct.write(out / 'environment.json', p['environment'])
    nodes = []
    for name in ('/dev/nvidiactl','/dev/nvidia0','/dev/nvidia-uvm','/dev/nvidia-uvm-tools'):
        item = dict(path=name)
        try:
            stat = os.stat(name)
            item.update(mode=oct(stat.st_mode), major=os.major(stat.st_rdev), minor=os.minor(stat.st_rdev))
            fd = os.open(name, os.O_RDWR | os.O_CLOEXEC)
            os.close(fd)
            item['open_rdwr_errno'] = 0
        except OSError as error:
            item.update(open_rdwr_errno=error.errno, error=str(error))
        nodes.append(item)
    lib = ctypes.CDLL('libcuda.so.1')
    lib.cuInit.argtypes = [ctypes.c_uint]
    lib.cuInit.restype = ctypes.c_int
    code = lib.cuInit(0)
    name, message = ctypes.c_char_p(), ctypes.c_char_p()
    lib.cuGetErrorName(code, ctypes.byref(name))
    lib.cuGetErrorString(code, ctypes.byref(message))
    maps = [line for line in Path('/proc/self/maps').read_text().splitlines() if 'libcuda.so' in line]
    loaded = sorted({line.split()[-1] for line in maps})
    ct.require(loaded == [p['driver_path']], 'cuInit used a different driver or stub')
    report = dict(cuInit=code, name=name.value.decode() if name.value else None,
        message=message.value.decode() if message.value else None, maps=maps, nodes=nodes,
        driver=record(p['driver_path']), kernel=Path('/proc/driver/nvidia/version').read_text(),
        process_status=Path('/proc/self/status').read_text(),
        root_cause='not established; success/failure does not prove a transient driver cause')
    if code == 0:
        count = ctypes.c_int()
        report['cuDeviceGetCount'] = lib.cuDeviceGetCount(ctypes.byref(count))
        report['device_count'] = count.value
    ct.write(out / 'driver-probe.json', report)
    if code != 0:
        return 1
    ct.require(report['cuDeviceGetCount'] == 0 and report['device_count'] == 1, 'Probe device roster differs')
    with (out / 'vectorAdd.log').open('x') as log:
        sample = subprocess.run([p['sample']], stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, env=dict(os.environ), timeout=30)
    text = (out / 'vectorAdd.log').read_text()
    passed = sample.returncode == 0 and 'Test PASSED' in text
    ct.write(out / 'sample-result.json', dict(returncode=sample.returncode, passed=passed,
        sample=record(p['sample']), log=record(out / 'vectorAdd.log')))
    ct.check_pins(p['pins'])
    return 0 if passed else 1


def init_probe(identity, previous_identity, granted, note):
    ct.require(granted and bool(note.strip()), 'Fresh sole-owner probe grant required')
    previous = run_directory(previous_identity)
    old = ct.load(previous / 'prepared.json')
    ct.require(old['sources'] == ct.source_pins(), 'Frozen sources changed before initialization probe')
    ct.check_pins(old['sources'])
    ct.validate_inventory(ct.config(ct.EXCLUSIVE_PATH), old['tools'])
    for binary in old['binaries'].values():
        ct.validate_staged_binary(binary['path'], binary['sha256'])
    ct.check_pins({k:v for k,v in old['pins'].items() if k != str(Path(__file__).resolve())})
    sample = Path('/home/admin/wxw403883/anaconda3/envs/tao/demo_suite/vectorAdd')
    driver = Path('/usr/lib64/libcuda.so.1').resolve()
    ct.require('stubs' not in str(driver), 'CUDA stub is not a driver')
    out = run_directory(identity, True)
    original = ct.load(previous / 'backend-exact/environment.json')
    environment = {k:v for k,v in original.items() if k != 'FERRULE_PAPER_WORKER_TOKEN'}
    environment['FERRULE_MIXED_EVIDENCE_DIR'] = str(out)
    ldd = subprocess.run(['/usr/bin/ldd', str(sample)], env=environment,
        capture_output=True, text=True, timeout=15, check=True)
    (out / 'vectorAdd-ldd.txt').write_text(ldd.stdout+ldd.stderr)
    libraries = [Path(v) for line in ldd.stdout.splitlines() for v in line.split()
                 if v.startswith('/') and Path(v).is_file()]
    pins = ct.inventory([sample, driver, Path('/usr/bin/ldd'), *libraries, *workflow_paths()])
    ct.write(out / 'probe-plan.json', dict(sample=record(sample), driver=record(driver), pins=pins,
        failed_preparation=record(previous / 'prepared.json'), sources=old['sources'],
        environment=environment, original_environment=record(previous / 'backend-exact/environment.json'),
        environment_changes=['FERRULE_PAPER_WORKER_TOKEN: fresh authentication', 'FERRULE_MIXED_EVIDENCE_DIR: fresh output'],
        driver_attempts=1, official_sample_attempts_at_most=1, seconds=120, no_gpu_reset=True))
    payload = dict(directory=str(out), sample=str(sample), driver_path=str(driver),
        sources=old['sources'], pins=pins, environment=environment)
    with contextlib.ExitStack() as stack:
        payload['locks'] = {u:stack.enter_context(ct.exclusive_owner_lock(u)) for u in sorted(UUIDS)}
        ct.write(out / 'before.json', idle(note))
        payload, pin = guard.seal(payload, out)
        env = dict(environment, FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
        result = supervisor.run_process(ct.isolated_argv([sys.executable,'-S','-B',str(Path(__file__).resolve()),
            'init-probe-worker','--worker-pin',pin], env), 120, payload)
        ct.write(out / 'supervisor.json', result)
        ct.require(ct.drained(result), 'Probe custody unknown; no further GPU action')
        ct.write(out / 'after.json', idle(note))
    passed = ct.successful_process(result)
    ct.write(out / 'release.json', dict(passed=passed, drained=True, gpu_owner_released=True, all_eight_idle=True))
    ct.require(passed, 'Initialization probe failed: no gate retry; inspect driver-probe/node/loader evidence')
    return out


def main():
    ct.require_cpu_python()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=('plan', 'prepare', 'reseal', 'run', 'compare', 'performance', 'closeout', 'worker', 'init-probe', 'init-probe-worker', 'reuse-backend'))
    ap.add_argument('--id')
    ap.add_argument('--from-id')
    ap.add_argument('--gate', choices=tuple(ALL_GATES))
    ap.add_argument('--small-qualification', type=Path)
    ap.add_argument('--phase-evidence', type=Path)
    ap.add_argument('--sole-gpu-owner-granted', action='store_true')
    ap.add_argument('--owner-note', default='')
    ap.add_argument('--worker-pin')
    a = ap.parse_args()
    if a.mode == 'init-probe-worker':
        return init_probe_worker(a.worker_pin)
    if a.mode == 'init-probe':
        print(init_probe(a.id, a.from_id, a.sole_gpu_owner_granted, a.owner_note))
        return 0
    if a.mode == 'worker':
        return worker(a.worker_pin)
    if a.mode == 'plan':
        print(json.dumps(manifest(), indent=2))
    elif a.mode == 'prepare':
        print(prepare(a.id))
    elif a.mode == 'reseal':
        print(reseal(a.id, a.from_id))
    elif a.mode == 'reuse-backend':
        print(reuse_backend(a.id, a.from_id))
    elif a.mode == 'closeout':
        print(closeout(a.id))
    elif a.mode == 'compare':
        print(compare(a.id, a.gate))
    elif a.mode == 'performance':
        print(performance(a.id, a.phase_evidence))
    else:
        print(run(a.id, a.gate, a.small_qualification, a.sole_gpu_owner_granted, a.owner_note))
    return 0


if __name__ == '__main__':
    sys.exit(main())
