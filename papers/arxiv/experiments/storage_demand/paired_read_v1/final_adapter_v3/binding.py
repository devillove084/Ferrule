#!/usr/bin/env python3
"""Append-only binding for the reviewed immutable corrected-cut final layout."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import current_binding_v3 as original, plan_v3 as plan
from paired_read_v1.final_adapter_v3 import evidence
from runnerhelpers.evidence_v2 import ct, require
from runnerhelpers.build_receipts_v1 import immutable

ADAPTER = 'ferrule.storage-demand.final-layout-reviewed-cli.v3'
OUT = HERE / 'bindings-final-v3-20261015-03'
BINDING_PATH = OUT / 'actual-binding-v3.json'
REVIEW_PATH = OUT / 'actual-build-review-v3.json'
EXECUTION_PATH = OUT / 'execution-sealed-v3.json'
ADMISSION = ct.ROOT / 'target/validation/cli-source-admission-20261015-01'
EXTRAS = {
    'scripts/__pycache__/gpu_idle_window_v2.cpython-310.pyc': 'fcdc41de311a8faa25dc911f58e34954231d80c9f5a1091c50c14f7cecf7d5f9',
    'scripts/__pycache__/run_pinned_jobs.cpython-310.pyc': '29127c251e279ea67e9232eef750399535dfdd1b127b05c98e2c9f7d5003f4c8',
    'scripts/__pycache__/test_vllm_qwen35.cpython-310.pyc': '08f3f4bce348ca95d5ecae726473605de4abc51e98356ea8c492a2cf655fa86b',
}
SOURCE_FILES = tuple('final_adapter_v3/' + name for name in
                     ('binding.py', 'evidence.py', 'run.py', 'test_adapter.py', 'README.md'))


def disposition(inventory, manifest):
    require(inventory['archive_and_all_pinned_source_bytes_verified'] is True
            and inventory['archive_members'] == 1484 and inventory['missing'] == [],
            'immutable source/archive incomplete')
    rows = inventory['extra_files']
    require(len(rows) == len(EXTRAS)
            and {r['relative_path']: r['sha256'] for r in rows} == EXTRAS,
            'unclassified source extra or generated evidence hash differs')
    for row in rows:
        name = row['relative_path']
        require(row['source_pin_member'] is False and row['archive_member'] is False
                and row['evidence_manifest_pin'] == EXTRAS[name]
                and manifest['pins_relative_to_directory'].get('frozen-source/' + name) == EXTRAS[name],
                'generated evidence lacks exact manifest pin')
        immutable(row['path'])
        require(ct.sha(row['path']) == EXTRAS[name], 'generated evidence bytes changed')
    return dict(compiled_archive_members_unchanged=1484, generated_noncompiled_evidence=EXTRAS,
                unclassified_extras=0, preserved_in_place=True, manifest_rewritten=False)


def source_admission(identity):
    final = ct.load(ADMISSION / 'FINAL.json')
    require(final['status'] == 'PASS' and final['source_admission_blocks_v3'] is False
            and final['frozen_cli_sha256'] == identity['cli_binary_sha256']
            and final['v3_cases_sha256'] == ct.sha(plan.CASES_PATH)
            and final['production_changes'] is False
            and final['production_source_file_count_verified'] == 545
            and final['current_targeted_failures'] == 0, 'actual CLI source admission absent/stale')
    name = 'actual-headers-v3-full-argv-final'
    command = ct.load(ADMISSION / (name + '-command.json'))
    receipt = ct.load(ADMISSION / (name + '-receipt.json'))
    require(all(receipt.get(k) == v for k, v in command.items())
            and receipt['returncode'] == 0 and receipt['gpu_hidden'] is True
            and receipt['profiling_enabled'] is False and receipt['timeout_seconds'] == 300,
            'source admission command failed/unbound')
    stdout = (ADMISSION / (name + '.stdout')).read_text()
    require((ADMISSION / (name + '.stderr')).read_bytes() == b''
            and 'test result: ok. 1 passed; 0 failed; 0 ignored;' in stdout,
            'source admission test did not actually pass')
    actual = [json.loads(s.split('=', 1)[1]) for s in stdout.splitlines() if s.startswith('source_admission=')]
    require(actual == final['actual_model_source_admission'] and len(actual) == 2,
            'source admission summary differs from raw output')
    for mode, row, argv in zip(plan.MODES, actual, final['v3_literal_argv']):
        require(argv == ['ferrule', *original.case_for_mode(mode)['argv_after_binary']]
                and row['schedule'] == mode and row['status'] == 'PASS'
                and row['metadata_admitted'] is True and row['cuda_context_created'] is False
                and row['required_composition_metadata_bytes'] == 7704576
                and row['effective_cli_max_metadata_bytes'] == 8388608
                and row['decode_cohort_target'] == 1 and row['decode_cohort_max_deferrals'] == 0,
                'literal V3 CLI/source limits differ')
    comparison = ct.load(ADMISSION / 'production-and-original-receipts-unchanged.json')
    require(len(comparison['production_source_files']) == 545 and comparison['production_mismatches'] == []
            and comparison['frozen_cli_sha256'] == identity['cli_binary_sha256']
            and comparison['original_failures_preserved'] is True, 'production/failure preservation differs')
    source = ct.load(original.QUALIFICATION / 'source-freeze.json')
    production = {n: pin for n, pin in source['pins_relative_to_frozen_source'].items()
                  if n.startswith('crates/') and '/src/' in n}
    require(len(production) == 545 and {r['path']: r['sha256'] for r in comparison['production_source_files']} == production, 'production comparison coverage differs')
    ct.check_pins({str(ct.ROOT / n): pin for n, pin in production.items()})
    for n, pin in comparison['original_failure_sha256'].items():
        require(ct.sha(original.QUALIFICATION / n) == pin, 'original CPU failure receipt changed')
    pins = ct.inventory(p for p in ADMISSION.iterdir() if p.is_file())
    return dict(status='actual_header_only_source_admission_reviewed', production_files=545,
                metadata_required=7704576, metadata_limit=8388608, cpu_build_custody='unobserved',
                full_cpu_suite_claim=False, artifact_pins=pins)


def review(root):
    plan.validate_prepared()
    root = Path(root)
    identity = original.candidate_identity(root)
    require(root == evidence.ROOT, 'final-layout candidate directory differs')
    manifest = evidence.load('evidence-manifest.json')
    require(manifest['schema'] == 'ferrule.expert-read.immutable-evidence-manifest.v1', 'evidence manifest schema')
    for name in manifest['pins_relative_to_directory']:
        evidence.safe_member(name)
        immutable(root / name)
    pins = {str(root / n): pin for n, pin in manifest['pins_relative_to_directory'].items()}
    ct.check_pins(pins)
    qualification, final, audit = (evidence.load(n) for n in
                                 ('qualification-plan-v2.json', 'FINAL.json', 'receipt-audit.json'))
    original.validate_gate_plan(qualification['gates'])
    require(qualification['schema'] == 'ferrule.expert-read.corrected-metadata-current-nine-gates.v1'
            and qualification['commit'] == final['commit'] == audit['commit'] == identity['commit']
            and qualification['source_digest'] == final['source_digest'] == audit['source_pins_digest'] == identity['source_digest']
            and final['status'] == audit['status'] == 'passed' and final['actual_gpu_tests_issued'] == 9
            and final['all_nine_qualified'] is True and final['not_issued_gates'] == [], 'nine-gate final incomplete')
    require(final['qualification_plan_sha256'] == audit['qualification_plan_sha256'] == ct.sha(root / 'qualification-plan-v2.json')
            and audit['final_receipt_sha256'] == ct.sha(root / 'FINAL.json')
            and all(final[k] == 0 for k in ('retries', 'conditioning_probes', 'warmups', 'profiling_captures')),
            'qualification hash/scope differs')
    ct.check_pins(qualification['pins'])
    inventory, source = evidence.source_inventory(identity, manifest)
    preservation = disposition(inventory, manifest)
    artifacts = {role: evidence.load('artifact-final-' + role + '.json') for role in evidence.ROLES}
    require(artifacts == qualification['artifacts'] and final['artifact_sha256'] == audit['artifact_sha256']
            == {r: a['sha256'] for r, a in artifacts.items()}, 'qualified artifact relation differs')
    module = original.reviewer(root, identity)
    runtime = evidence.load('runtime-library-pins-final.json')
    build = evidence.build_review(module, source, artifacts, runtime)
    gates = evidence.gate_review(module, qualification, source, runtime)
    admission = source_admission(identity)
    require(final['final_authenticated_echild'] is True and final['final_recorded_owners_absent'] is True
            and evidence.load('final-idle.json')['status'] == 'verified'
            and evidence.load('final-lock-release.json')['context_exited'] is True
            and evidence.load('final-lock-probe.json')['all_eight_nonblocking_reacquired'] is True,
            'qualification cleanup missing')
    pins.update(admission['artifact_pins'])
    pins.update(ct.inventory([root / 'evidence-manifest.json', *(HERE / n for n in SOURCE_FILES)]))
    return dict(schema='ferrule.storage-demand.actual-current-build-review.v3', adapter=ADAPTER,
        status='actual_build_and_nine_gate_evidence_reviewed_cli_serving_pending', protocol_id=plan.ID,
        protocol_sha256=ct.sha(plan.PROTOCOL_PATH), cases_sha256=ct.sha(plan.CASES_PATH),
        candidate_identity=identity, commit=identity['commit'], source_digest=identity['source_digest'],
        source_archive_sha256=identity['source_archive_sha256'], source_members=1484,
        source_disposition=preservation, build_review=build, quality_review=gates, cli_source_admission=admission,
        cli_lowering=original.review_lowering(root), artifact_pins=pins, runtime_pins=runtime['pins'],
        uuid_inventory=qualification['uuid_inventory'], qualification_directory=str(root),
        build_custody='unobserved; bounded CPU command receipts, not authenticated GPU producer custody',
        cli_serving_accepted=False, physical_release='unobserved', old_adapter_acceptance_transferred=False)


# Isolate V3 globals; never mutate the published historical adapter or its pins.
_spec = importlib.util.spec_from_file_location('_paired_final_layout_binding_v3', HERE / 'current_binding_v3.py')
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.ADAPTER = ADAPTER
_base.BINDING_PATH, _base.REVIEW_PATH, _base.EXECUTION_PATH = BINDING_PATH, REVIEW_PATH, EXECUTION_PATH
_original_binding_value = _base.binding_value
_original_execution_value = _base.execution_value


def binding_value(record):
    value = _original_binding_value(record)
    root = Path(record['candidate_identity']['qualification_directory'])
    value['candidate'].update(binary=str(root / 'staged-final/cli'),
        runtime_pin_manifest=str(root / 'runtime-library-pins-final.json'))
    return value


def execution_value():
    value = _original_execution_value()
    value['artifact_pins'].update(ct.inventory(HERE / name for name in SOURCE_FILES))
    value['final_layout_adapter'] = ADAPTER
    return value


_base.review, _base.binding_value, _base.execution_value = review, binding_value, execution_value
_validate_binding, _validate_execution = _base.validate_binding, _base.validate_execution


def validate_binding(path=None):
    return _validate_binding(BINDING_PATH if path is None else path)


def validate_execution(path=None):
    return _validate_execution(EXECUTION_PATH if path is None else path)


_base.validate_binding, _base.validate_execution = validate_binding, validate_execution
raw_options, case_for_mode = original.raw_options, original.case_for_mode
def seal_to(path, value):
    path = Path(path)
    require(path in (BINDING_PATH, EXECUTION_PATH) and path.parent == OUT
            and not OUT.is_symlink() and OUT.resolve() == OUT and not path.exists(),
            'new exclusive final-layout binding output required')
    ct.write(path, dict(value, digest=ct.digest(value)))


def main(argv=None):
    ct.require_cpu_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('review-and-bind', 'seal-execution', 'check'))
    parser.add_argument('--qualification', type=Path)
    args = parser.parse_args(argv)
    if args.mode == 'review-and-bind':
        require(args.qualification is not None and not OUT.is_symlink()
                and OUT.resolve() == OUT and OUT.is_dir(), 'explicit candidate/output directory required')
        require(not any(p.exists() for p in (REVIEW_PATH, BINDING_PATH, EXECUTION_PATH)),
                'final-layout review/binding exclusive; never replace')
        record = review(args.qualification)
        ct.write(REVIEW_PATH, record)
        seal_to(BINDING_PATH, binding_value(record))
    elif args.mode == 'seal-execution':
        seal_to(EXECUTION_PATH, execution_value())
    else:
        validate_execution()
    print(json.dumps(dict(status=args.mode + '_complete', protocol_id=plan.ID,
                          gpu_workloads=0, cli_serving_accepted=False)))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
