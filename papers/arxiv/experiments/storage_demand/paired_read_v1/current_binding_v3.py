#!/usr/bin/env python3
"""V3 read-only review/binding of a real completed corrected-source candidate.

Reuses current-builder semantics, not its old candidate acceptance. No build,
executable/help probe, GPU query, lock or HTTP call is made here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import tarfile
import tomllib
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import plan_v3 as plan, protocol as historical
from runnerhelpers.evidence_v2 import ct, require
from runnerhelpers.build_receipts_v1 import immutable, inventory_tree

ADAPTER = 'ferrule.storage-demand.current-qualification-cli-review.v3'
QUALIFICATION = ct.ROOT / 'target/validation/fixesmetadataCLIcurrent-quality-20261015-01'
EXPECTED_IDENTITY = dict(qualification_directory=str(QUALIFICATION),
    commit='86c58a017d7c4b53490f536a3b1a133f2f7848cc',
    source_digest='10183d246a1278248273e7d3055d1690cb56854cbff30f9f0662f18813dc8188',
    source_archive_sha256='e0d160996147820fcc64cfb29070b265b2297a6d9265a21de541ed77e8f2fa65',
    cli_binary_sha256='7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01',
    source_members=1484)
BINDING_PATH = HERE / 'actual-binding-v3.json'
REVIEW_PATH = HERE / 'actual-build-review-v3.json'
EXECUTION_PATH = HERE / 'execution-sealed-v3.json'
FILES = ('current_binding_v3.py', 'run_v3.py', 'test_run_v3.py', 'EXECUTION_V3.md')


def case_for_mode(mode):
    require(mode in plan.MODES, 'unknown V3 read mode')
    cases = plan.validate_cases(ct.load(plan.CASES_PATH))
    return copy.deepcopy(cases[mode])


def raw_options(mode):
    options = case_for_mode(mode)['read_options']
    options['schedule'] = mode
    return options


def candidate_identity(root):
    root = Path(root)
    require(root.is_absolute() and not root.is_symlink() and root.resolve() == root
            and root.parent == ct.ROOT / 'target/validation', 'explicit real qualification directory required')
    source = ct.load(root / 'source-freeze.json')
    commit = source.get('commit', '')
    require(re.fullmatch(r'[0-9a-f]{40}', commit) is not None and commit == EXPECTED_IDENTITY['commit'],
            'unknown source candidate; only the reviewed 86c58a0 cut is enabled')
    artifact = ct.load(root / 'artifact-cli.json')
    value = dict(qualification_directory=str(root), commit=commit,
        source_digest=source.get('source_pins_digest'), source_archive_sha256=source.get('archive_sha256'),
        cli_binary_sha256=artifact.get('sha256'), source_members=source.get('file_count'))
    require(all(re.fullmatch(r'[0-9a-f]{64}', value[key] or '') is not None for key in
        ('source_digest', 'source_archive_sha256', 'cli_binary_sha256')), 'candidate hashes missing/unknown')
    require(type(value['source_members']) is int and value['source_members'] > 0, 'source inventory absent')
    require(value == EXPECTED_IDENTITY, 'unknown exact source/archive/CLI candidate provenance')
    return value


def review_source(root, identity):
    """Review the new exact archive without claiming old compiled-source equality."""
    source = ct.load(root / 'source-freeze.json')
    require(source.get('schema') == 'ferrule.expert-read.committed-source-freeze.v1'
            and source.get('immutable_readonly_inputs') is True
            and source['commit'] == identity['commit'], 'new immutable committed-source freeze required')
    frozen = root / 'frozen-source'
    require(source['frozen_source'] == str(frozen) and not frozen.is_symlink()
            and frozen.stat().st_mode & 0o222 == 0, 'new source root mutable/aliased')
    relative = source['pins_relative_to_frozen_source']
    require(isinstance(relative, dict) and len(relative) == identity['source_members'], 'new source count differs')
    for name in relative:
        require(isinstance(name, str) and name and not Path(name).is_absolute()
                and '..' not in Path(name).parts and str(Path(name)) == name, 'unsafe new source member')
        current = frozen
        for part in Path(name).parts:
            current /= part
            require(not current.is_symlink(), 'new source symlink forbidden')
    originals = {str(ct.ROOT / name): pin for name, pin in relative.items()}
    require(source['original_source_pins'] == originals
            and ct.digest(originals) == identity['source_digest']
            and ct.digest(relative) == source['source_pins_digest_relative'], 'new source digest relation differs')
    require(inventory_tree(frozen) == {str(frozen / name) for name in relative}, 'new source coverage differs')
    for name, pin in relative.items():
        immutable(frozen / name)
        require(ct.sha(frozen / name) == pin, 'new frozen source bytes differ: ' + name)
    archive = root / 'source.tar'
    immutable(archive)
    require(source['archive'] == str(archive) and ct.sha(archive) == identity['source_archive_sha256'],
            'new source archive binding differs')
    members = {}
    with tarfile.open(archive, 'r|') as stream:
        for member in stream:
            require(not Path(member.name).is_absolute() and '..' not in Path(member.name).parts,
                    'unsafe new archive path')
            if member.isdir():
                continue
            require(member.isfile() and member.name in relative and member.name not in members,
                    'unsafe/extra/duplicate new archive member')
            with stream.extractfile(member) as data:
                pin = hashlib.file_digest(data, 'sha256').hexdigest()
            require(pin == relative[member.name], 'new archive bytes differ')
            members[member.name] = pin
    require(members == relative, 'new archive coverage incomplete')
    profile = tomllib.loads((frozen / 'Cargo.toml').read_text())['profile']['release']
    require(all(profile.get(k) == v for k, v in {'opt-level': 3, 'lto': 'fat', 'codegen-units': 1,
            'panic': 'unwind', 'strip': 'symbols'}.items()), 'new release profile differs')
    return source



def review_lowering(root):
    frozen = root / 'frozen-source'
    paths = [frozen / name for name in ('crates/ferrule-cli/src/args.rs',
        'crates/ferrule-cli/src/commands/serve.rs', 'crates/ferrule-runtime/src/scheduling/resident.rs')]
    args, serve, runtime = (path.read_text() for path in paths)
    block = re.search(r'scheduler_config: ResidentSchedulerConfig \{(.*?)\n        \}', serve, re.S)
    require(block is not None, 'new CLI scheduler lowering missing')
    mappings = dict(prefill_chunk_size='prefill_chunk_size', max_active_sequences='max_active_sequences',
        max_decode_batch='max_active_sequences', decode_cohort_target='decode_cohort_target',
        decode_cohort_max_deferrals='decode_cohort_max_deferrals', max_batch_tokens='max_batch_tokens')
    for field, argument in mappings.items():
        require(re.search(r'\b' + field + r': args\.' + argument + r'\s*,', block[1]) is not None,
                'new CLI lowering differs: ' + field)
    require('..ResidentSchedulerConfig::default()' in block[1], 'new CLI inherited defaults differ')
    defaults = re.search(r'impl Default for ResidentSchedulerConfig \{(.*?)\n\}', runtime, re.S)
    require(defaults is not None, 'new scheduler defaults missing')
    for field, value in dict(prefill_reserve_tokens='1', allow_mixed_batches='true', prefix_cache_capacity_pages='0').items():
        require(re.search(r'\b' + field + r': ' + value + r'\s*,', defaults[1]) is not None,
                'new inherited scheduler field differs: ' + field)
    for flag, default in (('decode-cohort-target', '4'), ('decode-cohort-max-deferrals', '3')):
        require('#[arg(long = "' + flag + '", default_value_t = ' + default + ')]' in args,
                'public CLI default changed: ' + flag)
    return dict(scope='Exact archived CLI lowering and runtime defaults, not live factory/serving PASS',
        source_pins=ct.inventory(paths), scheduler_fields=sorted(plan.SCHEDULER), public_defaults_unchanged=True)


def validate_gate_plan(gates):
    expected = [('01-constructor', 'constructor', None, 'constructor',
        'normal_cuda_constructor_reaches_configured_provider_for_both_model_boundaries')]
    for kind, numbers in (('full40', ('02', '03')), ('fresh', ('04', '05')),
                          ('ordinary', ('06', '07')), ('cold', ('08', '09'))):
        for mode, number in zip(plan.MODES, numbers):
            if kind == 'full40':
                selector = 'model_factory::dense_bf16_tests::full40::numeric_bf16_mixed_storage_middle64_' + mode + '_public_factory_full40_logits_routes_states_match_profile_reference'
            else:
                suffix = dict(fresh='retained_generation', ordinary='no_observer_equality', cold='cold_owner_restart')[kind]
                selector = 'model_factory::dense_bf16_tests::full40::fresh_quality::mixed_storage_middle64_' + mode + '_fresh_quality_' + suffix
            expected.append((number + '-' + mode + '-' + kind, kind, mode, 'composition', selector))
    require([(g.get('id'), g.get('kind'), g.get('schedule'), g.get('role'), g.get('selector')) for g in gates]
            == expected and all(g.get('seconds') == 1200 for g in gates), 'unknown nine-gate selector/role/order/deadline')


def reviewer(root, identity):
    # Isolated globals: never monkeypatch the historical reviewer or receipts.
    spec = importlib.util.spec_from_file_location('_paired_read_current_review_v3', HERE / 'current_binding_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.QUALIFICATION = root
    module.COMMIT = identity['commit']
    module.SOURCE_SHA = identity['source_digest']
    module.ARCHIVE_SHA = identity['source_archive_sha256']
    module.CLI_SHA = identity['cli_binary_sha256']
    module.ADAPTER = ADAPTER
    module.ORIGINAL_VALIDATE_PREPARED = plan.validate_prepared
    module.prereg = SimpleNamespace(MODES=plan.MODES, case_for_mode=case_for_mode)
    module.review_source = lambda directory: review_source(directory, identity)
    return module


def review(root):
    plan.validate_prepared()
    root = Path(root)
    identity = candidate_identity(root)
    require((root / 'FINAL.json').is_file() and (root / 'evidence-manifest.json').is_file(),
            'candidate pending: completed immutable nine-gate PASS required before binding')
    manifest = ct.load(root / 'evidence-manifest.json')
    for name in manifest['pins_relative_to_directory']:
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe evidence member')
    gates = ct.load(root / 'qualification-plan-v2.json')['gates']
    validate_gate_plan(gates)
    module = reviewer(root, identity)
    value = module.review(root)
    lowering = review_lowering(root)
    for gate in gates:
        if gate['kind'] != 'constructor':
            path = root / 'gates' / gate['id'] / (
                'numeric-report/full40-bf16-profile.json' if gate['kind'] == 'full40' else 'quality.json')
            plan.compare_report_configuration(case_for_mode(gate['schedule']), ct.load(path))
    value.update(schema='ferrule.storage-demand.actual-current-build-review.v3', protocol_id=plan.ID,
        protocol_sha256=ct.sha(plan.PROTOCOL_PATH), cases_sha256=ct.sha(plan.CASES_PATH),
        source_members=identity['source_members'], preregistered_source_overlap=None,
        raw_source_inventory_scope='New exact corrected-cut immutable archive; no old-source equality or acceptance transfer',
        candidate_identity=identity, cli_lowering=lowering, all_scheduler_fields_compared=sorted(plan.SCHEDULER))
    value['artifact_pins'].update(ct.inventory([root / 'evidence-manifest.json']))
    return value


def binding_value(record):
    identity = record['candidate_identity']
    root = Path(identity['qualification_directory'])
    return dict(schema='ferrule.storage-demand.actual-current-candidate-binding.v3', adapter=ADAPTER,
        protocol_id=plan.ID, status='current_nine_gate_bound_cli_serving_pending',
        protocol_sha256=ct.sha(plan.PROTOCOL_PATH), cases_sha256=ct.sha(plan.CASES_PATH),
        candidate_identity=identity, review_path=str(REVIEW_PATH), review_sha256=ct.sha(REVIEW_PATH),
        qualification_directory=str(root), uuid_inventory=record['uuid_inventory'],
        cli_binary_sha256=identity['cli_binary_sha256'], source_digest=identity['source_digest'],
        candidate=dict(binary=str(root / 'staged/cli'), binary_sha256=identity['cli_binary_sha256'],
            runtime_pin_manifest=str(root / 'runtime-library-pins.json'),
            artifact_pins=dict(record['artifact_pins'], **ct.inventory([REVIEW_PATH]))),
        cli_serving_accepted=False, old_adapter_acceptance_transferred=False,
        gpu_authorized_by_this_receipt=False, gpu_runs_by_this_helper=0)


def validate_binding(path=BINDING_PATH):
    value = historical.unseal(path, 'ferrule.storage-demand.actual-current-candidate-binding.v3')
    require(value.get('adapter') == ADAPTER and value.get('protocol_id') == plan.ID
            and value.get('protocol_sha256') == ct.sha(plan.PROTOCOL_PATH)
            and value.get('cases_sha256') == ct.sha(plan.CASES_PATH), 'unknown V3 protocol/candidate binding')
    require(value['candidate_identity'] == candidate_identity(Path(value['qualification_directory'])),
            'exact source/candidate provenance changed')
    record = ct.load(REVIEW_PATH)
    require(record.get('schema') == 'ferrule.storage-demand.actual-current-build-review.v3'
            and record.get('adapter') == ADAPTER and record.get('candidate_identity') == value['candidate_identity']
            and record.get('protocol_id') == plan.ID, 'unknown V3 review provenance')
    require({k: v for k, v in value.items() if k != 'digest'} == binding_value(record), 'V3 candidate binding stale')
    ct.check_pins(value['candidate']['artifact_pins'])
    ct.check_pins(ct.load(value['candidate']['runtime_pin_manifest'])['pins'])
    plan.validate_prepared()
    return value


def execution_value():
    candidate = validate_binding()
    prepared = plan.validate_prepared()
    # Reuse maintained lifecycle pins, not historical source validation/PASS.
    original = historical.unseal(HERE / 'cpu-sealed.json', 'ferrule.storage-demand.paired-read-cpu-prepared.v1')
    pins = dict(original['artifact_pins'])
    pins.update(prepared['artifact_pins'])
    pins.update(ct.inventory([HERE / name for name in (*FILES, 'run_actual_v2.py', 'current_binding_v2.py',
                                                      'cpu-sealed-v3.json', 'cpu-validation-v3.json')]))
    ct.check_pins(pins)
    return dict(schema='ferrule.storage-demand.paired-read-reviewed-execution.v3', protocol_id=plan.ID,
        status='preregistered_actual_binding_ready_owner_authorization_required',
        protocol_sha256=ct.sha(plan.PROTOCOL_PATH), cases_sha256=ct.sha(plan.CASES_PATH),
        binding_path=str(BINDING_PATH), binding_sha256=ct.sha(BINDING_PATH),
        candidate_identity=candidate['candidate_identity'], limits=prepared['limits'],
        trial_order=prepared['trial_order'], order=prepared['order'], artifact_pins=pins,
        scheduled_blocks=5, scheduled_starts=10, scheduled_requests=120, scheduled_measured_qa=50,
        retries=0, replacements=0, profiling_captures=0, historical_resume_permitted=False,
        process_policy='Maintained finite-arm coordinator and lifecycle worker; new exclusive V3 journal only')


def validate_execution(path=EXECUTION_PATH):
    value = historical.unseal(path, 'ferrule.storage-demand.paired-read-reviewed-execution.v3')
    require({k: v for k, v in value.items() if k != 'digest'} == execution_value(), 'V3 execution seal stale')
    ct.check_pins(value['artifact_pins'])
    return value


def main(argv=None):
    ct.require_cpu_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('review-and-bind', 'seal-execution', 'check'))
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'review-and-bind':
        require(args.qualification is not None, '--qualification is required; no default/future candidate')
        require(not any(path.exists() for path in (REVIEW_PATH, BINDING_PATH, EXECUTION_PATH)),
                'V3 binding/review/execution exclusive; never replace')
        record = review(args.qualification)
        ct.write(REVIEW_PATH, record)
        historical.seal_to(BINDING_PATH, binding_value(record))
    elif args.mode == 'seal-execution':
        historical.seal_to(EXECUTION_PATH, execution_value())
    else:
        validate_execution()
    print(json.dumps(dict(status=args.mode + '_complete', protocol_id=plan.ID,
                         gpu_workloads=0, cli_serving_accepted=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
