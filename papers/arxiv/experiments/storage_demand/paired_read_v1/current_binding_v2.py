#!/usr/bin/env python3
"""Explicit read-only review of the actual committed-source NVTX builder and nine gates.

No raw schema is rewritten, no old adapter acceptance transfers, and CPU command
return receipts do not become authenticated build-custody proof. CLI serving is
qualified only by the first planned lifecycle, not by the distinct test binaries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import tomllib

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import protocol as prereg, quality
from runnerhelpers.evidence_v2 import ct, require, integer
from runnerhelpers.quality_reports_v1 import validate_quality_reports, token_checks
from runnerhelpers.closure_v2 import evidence
from runnerhelpers.build_receipts_v1 import immutable, inventory_tree, RUST_COMMIT, CUTLASS_COMMIT
from newversionedprotocol import run_pilot as pilot

QUALIFICATION = ct.ROOT / 'target/validation/expert-read-current-quality-20261014-02'
COMMIT = '033519c3def098ce417e4b6ff0f141a9f30bddb1'
CLI_SHA = 'de5159951fe724e97e35c0fc86367642d33735994328eee1a5e604cf9d2b037b'
SOURCE_SHA = 'a4dceb084e21b80f6b45ff7ac980c6fa80ac8f17e902fd28c65f3d5bc7bb0efa'
ARCHIVE_SHA = 'ac072e0d1bac7efcc85b7f26ce68723f412b0e22b5ee868104b1f6073dd2a82f'
ADAPTER = 'ferrule.storage-demand.current-qualification-cli-review.v2'
BINDING_PATH = HERE / 'actual-binding-v2.json'
EXECUTION_PATH = HERE / 'execution-sealed-v2.json'
ORIGINAL_VALIDATE_PREPARED = prereg.validate_prepared
GATE_IDS = ['01-constructor', '02-blocking-full40', '03-lookahead-full40', '04-blocking-fresh',
            '05-lookahead-fresh', '06-blocking-ordinary', '07-lookahead-ordinary',
            '08-blocking-cold', '09-lookahead-cold']
EXTRA_FILES = ('current_binding_v2.py', 'run_actual_v2.py', 'test_actual_v2.py', 'EXECUTION_V2.md')


def raw_options(mode, constructor=False):
    require(mode in prereg.MODES, 'disabled/unknown mode')
    return dict(schedule=mode, max_host_bytes=(2 if constructor else 64) << 20,
                max_metadata_bytes=(256 << 10) if constructor else 8 << 20,
                worker_stack_bytes=(1 if constructor else 2) << 20)


def command(root, name, argv, environment=None):
    value = ct.load(root / (name + '-receipt.json'))
    spec = ct.load(root / (name + '-command.json'))
    require(all(value.get(k) == v for k, v in spec.items()), 'raw command/receipt relation differs: ' + name)
    require(value.get('argv') == argv and value.get('gpu_hidden') is True
            and value.get('profiling_enabled') is False and type(value.get('returncode')) is int
            and value['returncode'] == 0 and value.get('timed_out') is False
            and integer(value.get('timeout_seconds'), 1), 'actual CPU command failed/unbound: ' + name)
    env = value.get('environment', {})
    require(env.get('CUDA_VISIBLE_DEVICES') == '' and env.get('NVIDIA_VISIBLE_DEVICES') == 'none'
            and env.get('FERRULE_PROFILE_NVTX') == '0', 'CPU command visibility/profiling differs')
    if environment is not None:
        require(env == environment, 'actual build environment differs')
    for suffix in ('stdout', 'stderr'):
        path = root / (name + '.' + suffix)
        require(ct.sha(path) == value.get(suffix + '_sha256'), 'actual CPU output hash differs: ' + name)
    return value


def review_source(root):
    source = ct.load(root / 'source-freeze.json')
    require(source.get('schema') == 'ferrule.expert-read.committed-source-freeze.v1'
            and source.get('commit') == COMMIT and source.get('source_pins_digest') == SOURCE_SHA
            and source.get('file_count') == 1484 and source.get('immutable_readonly_inputs') is True,
            'actual committed source freeze differs')
    frozen = Path(source['frozen_source'])
    require(frozen == root / 'frozen-source' and frozen.stat().st_mode & 0o222 == 0, 'frozen root mutable/mismatched')
    relative = source['pins_relative_to_frozen_source']
    originals = {str(ct.ROOT / name): pin for name, pin in relative.items()}
    require(source['original_source_pins'] == originals and ct.digest(originals) == SOURCE_SHA
            and ct.digest(relative) == source['source_pins_digest_relative'] and len(relative) == 1484,
            'source inventory/digest relation differs')
    require(inventory_tree(frozen) == {str(frozen / name) for name in relative}, 'extra/missing frozen source member')
    old = ct.load(HERE / 'source-cut.json')['production_source_pins']
    shared = set(originals) & set(old)
    require(len(shared) == 1208 and all(originals[name] == old[name] for name in shared),
            'preregistered compiled-source overlap changed')
    for name, pin in relative.items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe source member')
        immutable(frozen / name)
        require(ct.sha(frozen / name) == pin, 'compiled source changed: ' + name)
        if name.startswith(('crates/', '.cargo/')) or name in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'):
            require(ct.sha(ct.ROOT / name) == pin, 'live compiled production source differs: ' + name)
    archive = Path(source['archive'])
    immutable(archive)
    require(archive == root / 'source.tar' and source['archive_sha256'] == ct.sha(archive) == ARCHIVE_SHA, 'archive binding differs')
    members = {}
    with tarfile.open(archive, 'r|') as stream:
        for member in stream:
            if member.isdir():
                require(not Path(member.name).is_absolute() and '..' not in Path(member.name).parts, 'unsafe archive directory')
                continue
            require(member.isfile() and member.name in relative and member.name not in members, 'unsafe/extra/duplicate archive member')
            with stream.extractfile(member) as data:
                pin = hashlib.file_digest(data, 'sha256').hexdigest()
            require(pin == relative[member.name], 'archive bytes differ')
            members[member.name] = pin
    require(members == relative, 'archive coverage incomplete')
    profile = tomllib.loads((frozen / 'Cargo.toml').read_text())['profile']['release']
    require(all(profile.get(k) == v for k, v in dict(**{'opt-level': 3, 'lto': 'fat', 'codegen-units': 1,
            'panic': 'unwind', 'strip': 'symbols'}).items()), 'release profile differs')
    return source


def review_build(root, source):
    env = ct.load(root / 'build-environment.json')
    allowed = set('CARGO_TARGET_DIR CARGO_HOME RUSTUP_HOME RUSTUP_TOOLCHAIN RUSTUP_AUTO_INSTALL CARGO_BUILD_JOBS '
        'CARGO_INCREMENTAL CUBLAS_WORKSPACE_CONFIG CUDA_HOME NVCC FERRULE_CUDA_ARCH FERRULE_CUDA_TEST_ORACLE '
        'FERRULE_CUTLASS_DIR CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES LD_LIBRARY_PATH LIBRARY_PATH PATH '
        'PYTHONDONTWRITEBYTECODE HOME CC CXX CUDAHOSTCXX RUSTC RUSTDOC FERRULE_NVTX_INCLUDE_DIR '
        'FERRULE_PROFILE_NVTX TMPDIR'.split())
    require(set(env) == allowed and all(isinstance(v, str) for v in env.values()), 'actual build allowlist differs')
    require(env['CARGO_TARGET_DIR'] == str(root / 'cargo-target') and env['CARGO_INCREMENTAL'] == '0'
            and env['FERRULE_CUDA_ARCH'] == 'sm_86' and env['FERRULE_CUDA_TEST_ORACLE'] == '0'
            and env['RUSTUP_TOOLCHAIN'] == 'nightly-2026-08-01' and env['RUSTUP_AUTO_INSTALL'] == '0'
            and env['CUBLAS_WORKSPACE_CONFIG'] == ':0:0' and env['CC'] == '/usr/bin/gcc'
            and env['CXX'] == env['CUDAHOSTCXX'] == '/usr/bin/g++', 'build/compiler/CUDA controls differ')
    tools = ct.load(root / 'toolchain.json')
    require(set(tools) == {'cargo', 'rustc', 'nvcc', 'cutlass'} and all(v.get('returncode') == 0
            and v.get('stderr') == '' for v in tools.values()), 'actual toolchain receipts failed')
    require('commit-hash: ' + RUST_COMMIT + '\n' in tools['rustc']['stdout']
            and 'V12.4.99' in tools['nvcc']['stdout'] and tools['cutlass']['stdout'] == CUTLASS_COMMIT + '\n', 'toolchain version differs')
    external = ct.load(root / 'external-input-pins.json')['pins']
    ct.check_pins(external)
    for key in ('NVCC', 'RUSTC', 'CC', 'CXX'):
        require(env[key] in external or str(Path(env[key]).resolve()) in external, 'compiler input unpinned: ' + key)
    for key in ('FERRULE_CUTLASS_DIR', 'FERRULE_NVTX_INCLUDE_DIR'):
        require(inventory_tree(env[key]) <= set(external), 'CUTLASS/NVTX tree unpinned')
    require((Path(env['FERRULE_NVTX_INCLUDE_DIR']) / 'nvtx3/nvToolsExt.h').is_file(), 'official NVTX header absent')
    artifacts, cargo_rows, packages = {}, {}, {}
    for role in ('cli', 'constructor', 'composition'):
        artifact = ct.load(root / ('artifact-' + role + '.json'))
        receipt = ct.load(root / ('build-' + role + '-receipt.json'))
        command(root, 'build-' + role, receipt['argv'], env)
        expected = ['/root/.cargo/bin/cargo', 'build' if role == 'cli' else 'test', '--locked', '--offline',
                    '--release', '--no-default-features', '-p', {'cli': 'ferrule-cli', 'constructor': 'ferrule-model',
                    'composition': 'ferrule-composition'}[role], '--features',
                    'cuda,nvtx-profile' if role == 'cli' else 'cuda,ferrule-common/nvtx-profile']
        expected += [] if role == 'cli' else (['--test', 'expert_read_constructor', '--no-run'] if role == 'constructor'
                                            else ['--lib', 'full40', '--no-run'])
        expected += ['--message-format=json']
        require(receipt['argv'] == expected and receipt['cwd'] == source['frozen_source']
                and receipt['timeout_seconds'] == 1800, 'actual build argv/cwd/deadline differs')
        rows = [json.loads(line) for line in (root / ('build-' + role + '.stdout')).read_text().splitlines() if line.startswith('{')]
        require(rows and rows[-1] == dict(reason='build-finished', success=True), 'Cargo completion missing')
        compiled = artifact['compiled_artifact']
        require(compiled in rows and artifact['source_digest'] == SOURCE_SHA and artifact['nvtx_compiled'] is True
                and artifact['qualification_nvtx_runtime_enabled'] is False, 'actual linked artifact/source/NVTX differs')
        workspace = [row for row in rows if row.get('reason') == 'compiler-artifact'
                     and row.get('manifest_path', '').startswith(source['frozen_source'] + '/')]
        require(workspace and all('cpu-reference' not in row.get('features', []) for row in workspace), 'CPU reference/unreviewed feature linked')
        require(all('nvtx-profile' in row['features'] for row in artifact['common_artifacts'])
                and all(row in rows for row in artifact['common_artifacts']), 'actual common NVTX feature missing')
        if role == 'cli':
            require(compiled['features'] == ['cuda', 'nvtx-profile'] and compiled['fresh'] is False
                    and compiled['profile']['opt_level'] == '3' and compiled['profile']['debug_assertions'] is False
                    and compiled['profile']['test'] is False, 'CLI release features/profile differs')
        staged = Path(artifact['executable'])
        immutable(staged)
        output = Path(compiled['executable'])
        require(staged == root / 'staged' / role and ct.sha(staged) == artifact['sha256'] == ct.sha(output)
                and staged.stat().st_size == artifact['bytes'] and not staged.samefile(output), 'staged/output actual bytes/inodes differ')
        with staged.open('rb') as stream:
            require(stream.read(4) == b'\x7fELF', 'linked binary is not ELF')
        artifacts[role], cargo_rows[role] = artifact, rows
        for row in rows:
            if row.get('reason') == 'compiler-artifact' and row.get('package_id', '').startswith('registry+'):
                packages[row['manifest_path']] = row['package_id']
    require(artifacts['cli']['sha256'] == CLI_SHA, 'wrong requested immutable CLI')
    dependencies = ct.load(root / 'compiled-dependency-pins.json')
    require(packages == dependencies['packages'] and set(dependencies['pins']) == set().union(
        *(inventory_tree(Path(manifest).parent) for manifest in packages)), 'actual dependency coverage differs')
    ct.check_pins(dependencies['pins'])
    native = ct.load(root / 'native-archive-pins.json')
    originals, copies = native['original_pins'], native['immutable_copies']
    require(native['all_roles_same_new_target'] is True and len(originals) == len(copies)
            and set(originals) == {str(p) for p in (root / 'cargo-target').rglob('*.a')}, 'native bundle coverage differs')
    ct.check_pins(originals)
    ct.check_pins(copies)
    for original, pin in originals.items():
        matching = [Path(p) for p, h in copies.items() if h == pin and Path(p).name.endswith('-' + Path(original).name)]
        require(len(matching) == 1 and not matching[0].samefile(original), 'independent native copy relation missing')
        immutable(matching[0])
    require({'libferrule_cpu_native.a', 'libferrule_cuda_core.a', 'libferrule_cuda_cutlass.a', 'libferrule_nvtx_host.a'}
            <= {Path(p).name for p in originals}, 'CUDA/NVTX native bundle missing')
    runtime = ct.load(root / 'runtime-library-pins.json')
    ct.check_pins(runtime['pins'])
    for role, artifact in artifacts.items():
        command(root, 'ldd-' + role, ['/usr/bin/ldd', artifact['executable']])
        text = (root / ('ldd-' + role + '.stdout')).read_text()
        require('not found' not in text, 'linked runtime unresolved')
        libraries = {str(Path(path).resolve()) for path in re.findall(r'(/[^\s()]+)\s+\(', text)}
        require(libraries == set(runtime['per_role'][role]) and all(runtime['pins'].get(path) == pin
                for path, pin in runtime['per_role'][role].items()), 'actual per-role loader pins differ')
    cli = artifacts['cli']['executable']
    for name, args in (('help', ['--help']), ('serve-help', ['serve', '--help']), ('version', ['--version'])):
        command(root, 'cli-' + name, [cli, *args])
        require((root / ('cli-' + name + '.stderr')).read_bytes() == b'', 'exact CLI help stderr nonempty')
    flags = set(re.findall(r'--[a-z][a-z0-9-]*', (root / 'cli-serve-help.stdout').read_text()))
    require({x for x in prereg.case_for_mode('blocking')['argv_after_binary'] if x.startswith('--')} <= flags,
            'exact staged CLI read/config field-help missing')
    return artifacts, runtime


def validate_reads(report, mode):
    requested = raw_options(mode)
    q = report.get('expert_read_quality', {})
    require(q.get('schema') == 'ferrule.expert-read-quality-evidence.v1' and q.get('interval_validation_revision') == 2
            and q.get('passed') is True and q.get('requested') == requested, 'actual read evidence/config differs')
    calls = q.get('calls', [])
    require(calls and q.get('sampled_operations') == len(calls), 'read operation denominator differs')
    for row in calls:
        require(row.get('requested') == requested and all(row.get(k) is True for k in
            ('same_owner', 'configuration_matches', 'within_charged_limits', 'unambiguous', 'evidence_complete')),
            'raw interval not qualified')
        for endpoint in ('before', 'after'):
            require(all(row.get('configuration_pair', {}).get(endpoint, {}).get(k) is True for k in
                ('configuration_matches', 'within_charged_limits', 'unambiguous')), 'raw endpoint unqualified')
        quality.interval_endpoints(row, requested)
    positive = sum(row.get('positive_worker_read') is True for row in calls)
    require(positive > 0 and q.get('positive_worker_read_calls') == positive, 'no positive worker-read evidence')
    expected = prereg.case_for_mode(mode)['quality_test_config']['expert_read_pipeline']
    effective = report['effective_config']
    require(effective.get('assertions_passed') is True and effective.get('requested_expert_read_pipeline') == expected
            and effective.get('effective_expert_read_pipeline') == expected, 'actual factory lowering differs')


def validate_constructor(report):
    require(report.get('schema') == 'ferrule.expert-read-constructor-quality.v1'
            and report.get('complete') is True and report.get('passed') is True
            and report.get('expected_cells') == 4 and report.get('total_logit_values_checked') == 128
            and report.get('diagnostic_trace_installed') is False, 'actual constructor incomplete')
    cells = report.get('cells', [])
    require(len(cells) == 4 and {(c['model_boundary'], c['requested']['schedule']) for c in cells}
            == {(b, m) for b in ('Bf16ModelF32Accumulate', 'W8A16Bf16F32Accumulate') for m in prereg.MODES},
            'constructor mode/boundary cells differ')
    for cell in cells:
        options = raw_options(cell['requested']['schedule'], constructor=True)
        require(cell['requested'] == options and all(cell.get(k) is True for k in
                ('passed', 'same_owner', 'positive_worker_read', 'charged_limits_satisfied'))
                and cell.get('logit_values_checked') == 32 and cell.get('quarantine') is False
                and type(cell.get('state_bytes_after_shutdown')) is int and cell['state_bytes_after_shutdown'] == 0,
                'actual constructor cell not qualified')
        quality.read_observation(cell['observed'], options, effective_required=True)
        require(cell['observed']['stats']['submitted_reads'] == 6, 'constructor read denominator differs')


def validate_ordinary(report, prior, prior_hash, mode):
    require(report.get('schema') == 'ferrule.bf16-fresh-quality-result.v1'
            and report.get('complete') is True and report.get('passed') is True
            and report.get('test_config') == prior['test_config'] == prereg.case_for_mode(mode)['quality_test_config']
            and report.get('effective_config') == prior['effective_config']
            and report.get('diagnostic_primary_report_sha256') == prior_hash
            and report.get('oracle_manifest_sha256') == prior['oracle_manifest_sha256'], 'ordinary primary/config differs')
    require(report.get('observer_mode') == 'ordinary_no_diagnostic_observer' and report.get('ordinary_no_observer_claim') is True
            and report.get('cold_owner_restart') is False and report.get('failures') == []
            and report.get('quarantine') is False and report.get('state_bytes_after_shutdown') == 0
            and report.get('atol') == .00025 and report.get('rtol') == .012
            and report.get('peak_reserved_bytes', 1 << 40) <= 7 << 30, 'ordinary numerical/budget/closure failed')
    fp = report.get('prediction_fingerprints', {})
    require(fp.get('paired_exact_logits_and_greedy_equal') is True and fp.get('failures') == []
            and isinstance(fp.get('calls'), dict) and len(fp['calls']) == 64
            and fp['calls'] == prior['prediction_fingerprints']['calls'], 'ordinary actual fingerprints differ')
    require([r['case'] for r in report['cases']] == ['qa', 'arithmetic', 'chinese'], 'ordinary case coverage')
    for actual, reference in zip(report['cases'], prior['cases']):
        ids = reference['oracle_generated_ids']
        require(actual['oracle_generated_ids'] == actual['common_history_generated_ids'] == actual['free_greedy_ids'] == ids
                and actual['free_greedy_equal'] is True and actual['top1_failures'] == [], 'ordinary actual tokens differ')
        token_checks(actual['next_token_checks'], ids)
        token_checks(actual['free_greedy_next_token_checks'], ids)
    streams = report['prediction_denominators']['streams']
    require(set(streams) == {'common_history', 'free_feedback', 'reset_replay', 'fork_replay'}, 'ordinary stream coverage')
    for name, row in streams.items():
        count = 27 if name in ('common_history', 'free_feedback') else 9 if name == 'reset_replay' else 1
        require(all(row.get(k) == count for k in ('prediction_calls', 'greedy_checks', 'input_history_matched_reference_calls'))
                and integer(row.get('logit_values_compared'), 1) and row.get('failures') == [], 'ordinary stream denominator/failure')
    validate_reads(report, mode)


def review_gate(root, gate, artifacts, runtime, source):
    directory = root / 'gates' / gate['id']
    result = ct.load(directory / 'gate-result.json')
    custody = ct.load(directory / 'supervisor.json')
    launch = ct.load(directory / 'authenticated-launch.json')
    sealed = ct.load(directory / 'worker-launch.json')
    require(ct.successful_process(custody) and result.get('status') == 'passed'
            and result.get('actual_test_issued') is True and result.get('dispatched') == 1
            and result.get('natural_exit') is True and result.get('timed_out') is False
            and result.get('retries') == 0 and result.get('source_digest') == SOURCE_SHA, 'actual producer failed/unknown')
    entries = custody['custody']['entries']
    require(all(not row.get('signals') for row in entries), 'quality producer did not exit naturally')
    identity = launch['identity']
    require(all(identity['worker'][key] == custody['custody']['root_identity'][key] for key in ('pid', 'birth_ticks', 'ppid'))
            and identity['worker']['pgrp'] == identity['worker']['session'] == identity['worker']['pid']
            and identity['helper'] == custody['custody']['helper_identity']
            and identity['launch_id'] == custody['custody']['launch_id']
            and identity['coordinator'] == sealed['coordinator'], 'actual producer PID/birth/parent/launch relation differs')
    require(launch.get('all_eight_uuid_locks_validated') is True
            and launch.get('exec_preserves_authenticated_root_pid_birth') is True
            and launch['argv'] == [gate['executable'], gate['selector'], '--exact', '--ignored', '--nocapture', '--test-threads=1']
            and gate['executable'] == artifacts[gate['role']]['executable']
            and launch['executable_sha256'] == gate['binary_sha256'] == artifacts[gate['role']]['sha256'],
            'actual selected test binary/argv differs')
    require(sealed['gate'] == gate and sealed['qualification_plan_sha256'] == ct.sha(root / 'qualification-plan-v2.json')
            and launch['environment']['FERRULE_PROFILE_NVTX'] == '0', 'raw qualification contract differs')
    listed = ct.load(directory / 'exact-list-receipt.json')
    require(listed.get('returncode') == 0 and listed.get('stderr') == ''
            and [s.strip() for s in listed['stdout'].splitlines() if s.strip()]
            == [gate['selector'] + ': test', '1 test, 0 benchmarks'], 'exact test selector not one')
    require(re.search(r'test result: ok\. 1 passed; 0 failed; 0 ignored;', (directory / 'test.stdout').read_text()) is not None,
            'ignored/false test success')
    for name in ('pre-idle', 'authenticated-pre-cuda-idle', 'post-idle'):
        value = ct.load(directory / (name + '.json'))
        require(value.get('status') == 'verified' and value.get('required_consecutive') == 3
                and value.get('deadline_seconds') == 15 and all(value['samples'][i]['classification'] == 'idle'
                for i in value['accepted_sample_indices']), 'actual quality stable idle differs')
    actual_paths = set()
    mapping = ct.load(directory / 'runtime-mapping-observations.json')
    require(mapping['actual_loaded_mappings_claim'] is True and mapping['samples'], 'actual test runtime maps missing')
    for sample in mapping['samples']:
        require(sample.get('identity', {}).get('pid') == identity['worker']['pid']
                and sample['identity']['birth_ticks'] == identity['worker']['birth_ticks'], 'mapped owner differs')
        for line in sample['maps'].splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and parts[-1].startswith('/') and '.so' in parts[-1]:
                require(not parts[-1].endswith(' (deleted)'), 'deleted actual runtime mapping')
                actual_paths.add(str(Path(parts[-1]).resolve()))
    require(actual_paths <= set(runtime['pins']) and any('libcudart.so.12.4.99' in path for path in actual_paths)
            and any('libcuda.so.' in path for path in actual_paths), 'actual native/CUDA runtime unbound')
    report_path = directory / ('numeric-report/full40-bf16-profile.json' if gate['kind'] == 'full40' else 'quality.json')
    require(result['report'] == str(report_path) and result['report_sha256'] == ct.sha(report_path), 'actual raw report relation differs')
    return ct.load(report_path), dict(gate=gate['id'], report=str(report_path), report_sha256=ct.sha(report_path),
        supervisor=str(directory / 'supervisor.json'), supervisor_sha256=ct.sha(directory / 'supervisor.json'),
        executable_sha256=gate['binary_sha256'], authenticated_echild=True, natural_exit=True,
        actual_runtime_libraries=len(actual_paths), physical_release='unobserved')


def review(root=QUALIFICATION):
    ORIGINAL_VALIDATE_PREPARED()
    require(Path(root).resolve() == QUALIFICATION.resolve(), 'unreviewed qualification directory')
    manifest = ct.load(root / 'evidence-manifest.json')
    require(manifest.get('schema') == 'ferrule.expert-read.immutable-evidence-manifest.v1', 'unknown evidence manifest')
    evidence_pins = {str(root / name): pin for name, pin in manifest['pins_relative_to_directory'].items()}
    ct.check_pins(evidence_pins)
    plan = ct.load(root / 'qualification-plan-v2.json')
    final = ct.load(root / 'FINAL.json')
    audit = ct.load(root / 'receipt-audit.json')
    require(plan['commit'] == final['commit'] == audit['commit'] == COMMIT
            and plan['source_digest'] == final['source_digest'] == audit['source_pins_digest'] == SOURCE_SHA
            and final['status'] == audit['status'] == 'passed' and final['actual_gpu_tests_issued'] == 9
            and final['all_nine_qualified'] is True and final['not_issued_gates'] == []
            and [gate['id'] for gate in plan['gates']] == GATE_IDS, 'current nine-gate final incomplete')
    require(all(final[k] == 0 for k in ('retries', 'conditioning_probes', 'warmups', 'profiling_captures'))
            and final['final_authenticated_echild'] is True and final['final_recorded_owners_absent'] is True,
            'qualification scope/custody differs')
    ct.check_pins(plan['pins'])
    source = review_source(root)
    artifacts, runtime = review_build(root, source)
    require(artifacts == plan['artifacts'], 'actual qualification linked artifact manifest differs')
    references = ct.load(root / 'reference-pins.json')
    ct.check_pins(references['reference_files'])
    ct.check_pins(references['model_metadata'])
    reports, rows = {}, []
    for gate in plan['gates']:
        report, row = review_gate(root, gate, artifacts, runtime, source)
        reports[gate['id']] = report
        rows.append(row)
    validate_constructor(reports['01-constructor'])
    for mode, numbers in (('blocking', ('02', '04', '06', '08')), ('lookahead', ('03', '05', '07', '09'))):
        gates = {label: reports[number + '-' + mode + '-' + label] for label, number in zip(
            ('full40', 'fresh', 'ordinary', 'cold'), numbers)}
        validate_quality_reports({label: gates[label] for label in ('full40', 'fresh', 'cold')}, prereg.case_for_mode(mode))
        require(gates['full40']['reference_manifest_sha256'] == references['numeric_manifest_sha256']
                and gates['fresh']['oracle_manifest_sha256'] == gates['cold']['oracle_manifest_sha256']
                == references['fresh_manifest_sha256'], 'both arms independent references differ')
        for label in ('full40', 'fresh', 'cold'):
            validate_reads(gates[label], mode)
        fresh_path = root / 'gates' / (numbers[1] + '-' + mode + '-fresh') / 'quality.json'
        validate_ordinary(gates['ordinary'], gates['fresh'], ct.sha(fresh_path), mode)
        oracles = {row['id']: row for row in ct.load(HERE / 'oracles.json')['references']}
        for case in gates['fresh']['cases']:
            require(case['oracle_generated_ids'][:8] == oracles[case['case']]['reference_ids'], 'frozen generation8 oracle differs')
        qa = gates['fresh']['cases'][0]
        require(qa['prompt_ids'] == ct.load(HERE / 'oracles.json')['qa']['input_ids'], 'actual QA input token IDs differ')
    final_idle = ct.load(root / 'final-idle.json')
    require(final_idle['status'] == 'verified' and ct.load(root / 'final-lock-probe.json')['all_eight_nonblocking_reacquired'] is True,
            'qualification final idle/lock release missing')
    return dict(schema='ferrule.storage-demand.actual-current-build-review.v2', adapter=ADAPTER,
        status='actual_build_and_nine_gate_evidence_reviewed_cli_serving_pending', commit=COMMIT,
        source_digest=SOURCE_SHA, source_archive_sha256=ARCHIVE_SHA, source_members=1484,
        preregistered_source_overlap=1208, raw_source_inventory_scope='Exact committed archive; CPU preregistration also inventoried untracked fixture files. Shared compiled inputs byte-identical; no source hashes normalized.',
        artifact_sha256={role: artifact['sha256'] for role, artifact in artifacts.items()},
        gates=rows, build_custody='unobserved; actual bounded subprocess command receipts, not authenticated CPU build proof',
        cli_qualification='pending first planned serving lifecycle; separate test binaries are not mathematical CLI equivalence',
        old_build_adapter_acceptance_transferred=False, source_native_runtime_relation='actual same committed source/target/native/runtime bundle',
        qualification_directory=str(root), artifact_pins=evidence_pins, runtime_pins=runtime['pins'],
        uuid_inventory=plan['uuid_inventory'], gpu_workloads_by_this_reviewer=0, server_runs_by_this_reviewer=0,
        physical_release='unobserved', interpretation='Evidence review only; renewed owner authorization required for actual study')


def execution_value(binding_path=BINDING_PATH):
    original = ORIGINAL_VALIDATE_PREPARED()
    return dict(schema='ferrule.storage-demand.paired-read-reviewed-execution.v2',
        status='preregistered_actual_binding_ready_owner_authorization_required',
        original_cpu_seal_path=str(prereg.PREPARED_PATH), original_cpu_seal_sha256=ct.sha(prereg.PREPARED_PATH),
        binding_path=str(binding_path), binding_sha256=ct.sha(binding_path),
        limits=original['limits'], trial_order=original['trial_order'], order=original['order'],
        scheduled_blocks=5, scheduled_starts=10, scheduled_requests=120, scheduled_measured_qa=50,
        artifact_pins=dict(original['artifact_pins'], **ct.inventory([HERE / name for name in EXTRA_FILES])),
        raw_read_observation_adapter='serde lowercase schedule; factory test_config uses Debug title-case; raw evidence is never rewritten',
        process_policy='one finite arm per command; permanent exclusive attempt journal; no resume after any incomplete/failed attempt',
        retries=0, replacements=0, profiling_captures=0)


def validate_execution(path=EXECUTION_PATH):
    value = prereg.unseal(path, 'ferrule.storage-demand.paired-read-reviewed-execution.v2')
    require({k: v for k, v in value.items() if k != 'digest'} == execution_value(Path(value['binding_path'])), 'reviewed execution seal stale')
    ct.check_pins(value['artifact_pins'])
    return value


def validate_binding(path=BINDING_PATH):
    value = prereg.unseal(path, 'ferrule.storage-demand.actual-current-candidate-binding.v2')
    require(value.get('adapter') == ADAPTER and value.get('status') == 'current_nine_gate_bound_cli_serving_pending'
            and value.get('cli_binary_sha256') == CLI_SHA and value.get('source_digest') == SOURCE_SHA
            and value.get('protocol_sha256') == ct.sha(prereg.PROTOCOL_PATH)
            and value.get('cases_sha256') == ct.sha(prereg.CASE_PATH), 'actual candidate binding stale/unknown')
    ct.check_pins(value['candidate']['artifact_pins'])
    review_path = Path(value['review_path'])
    review_record = ct.load(review_path)
    require(ct.sha(review_path) == value['review_sha256'] and review_record['adapter'] == ADAPTER
            and review_record['artifact_sha256']['cli'] == CLI_SHA and review_record['source_digest'] == SOURCE_SHA,
            'actual read-only build/quality review changed')
    ct.check_pins(ct.load(value['candidate']['runtime_pin_manifest'])['pins'])
    return value


def main(argv=None):
    ct.require_cpu_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('review-and-bind', 'seal-execution', 'check'))
    args = parser.parse_args(argv)
    if args.mode == 'review-and-bind':
        record = review()
        review_path = HERE / 'actual-build-review-v2.json'
        ct.write(review_path, record)
        pins = dict(record['artifact_pins'], **ct.inventory([review_path]))
        value = dict(schema='ferrule.storage-demand.actual-current-candidate-binding.v2', adapter=ADAPTER,
            status='current_nine_gate_bound_cli_serving_pending', cli_binary_sha256=CLI_SHA, source_digest=SOURCE_SHA,
            protocol_sha256=ct.sha(prereg.PROTOCOL_PATH), cases_sha256=ct.sha(prereg.CASE_PATH),
            review_path=str(review_path), review_sha256=ct.sha(review_path), qualification_directory=str(QUALIFICATION),
            uuid_inventory=record['uuid_inventory'], candidate=dict(binary=str(QUALIFICATION / 'staged/cli'),
                binary_sha256=CLI_SHA, runtime_pin_manifest=str(QUALIFICATION / 'runtime-library-pins.json'), artifact_pins=pins),
            sealed_current_evidence_binding=True, cli_serving_accepted=False, old_adapter_acceptance_transferred=False,
            gpu_authorized_by_this_receipt=False, gpu_runs_by_this_helper=0)
        prereg.seal_to(BINDING_PATH, value)
    elif args.mode == 'seal-execution':
        validate_binding()
        prereg.seal_to(EXECUTION_PATH, execution_value())
    else:
        validate_binding()
        validate_execution()
    print(json.dumps(dict(status=args.mode + '_complete', gpu_workloads=0, cli_serving_accepted=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
