#!/usr/bin/env python3
"""Final-layout evidence readers recovered from the non-admitting CPU review.

No historical receipt is changed. CPU commands are not authenticated GPU custody.
Admission and exact generated-evidence disposition live in binding.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import tomllib

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
HERE = OUT.parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1 import current_binding_v3 as binding, plan_v3 as plan
from paired_read_v1 import quality
from runnerhelpers.evidence_v2 import ct, require
from runnerhelpers.build_receipts_v1 import immutable, inventory_tree, RUST_COMMIT, CUTLASS_COMMIT
from runnerhelpers.quality_reports_v1 import validate_quality_reports

ROOT = binding.QUALIFICATION
ROLES = ('cli', 'constructor', 'composition')
REVIEW_PATH = OUT / 'review-manifest.json'
ENV_KEYS = set(('CARGO_TARGET_DIR CARGO_HOME RUSTUP_HOME RUSTUP_TOOLCHAIN RUSTUP_AUTO_INSTALL '
    'CARGO_BUILD_JOBS CARGO_INCREMENTAL CUBLAS_WORKSPACE_CONFIG CUDA_HOME NVCC FERRULE_CUDA_ARCH '
    'FERRULE_CUDA_TEST_ORACLE FERRULE_CUTLASS_DIR CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES '
    'LD_LIBRARY_PATH LIBRARY_PATH PATH PYTHONDONTWRITEBYTECODE HOME CC CXX CUDAHOSTCXX '
    'RUSTC RUSTDOC FERRULE_NVTX_INCLUDE_DIR FERRULE_PROFILE_NVTX TMPDIR').split())


def load(name):
    return ct.load(ROOT / name)


def safe_member(name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and '..' not in Path(name).parts and str(Path(name)) == name, 'unsafe member')


def source_inventory(identity, manifest):
    source = load('source-freeze.json')
    frozen = ROOT / 'frozen-source'
    require(source['schema'] == 'ferrule.expert-read.committed-source-freeze.v1'
            and source['immutable_readonly_inputs'] is True and source['worktree_edits_excluded'] is True
            and source['frozen_source'] == str(frozen) and not frozen.is_symlink()
            and frozen.stat().st_mode & 0o222 == 0, 'source freeze/root differs')
    relative = source['pins_relative_to_frozen_source']
    require(len(relative) == identity['source_members'], 'source member count differs')
    for name, pin in relative.items():
        safe_member(name)
        current = frozen
        for part in Path(name).parts:
            current /= part
            require(not current.is_symlink(), 'source symlink')
        immutable(current)
        require(ct.sha(current) == pin, 'pinned source bytes differ: ' + name)
    original = {str(ct.ROOT / name): pin for name, pin in relative.items()}
    require(source['original_source_pins'] == original and ct.digest(original) == identity['source_digest']
            and ct.digest(relative) == source['source_pins_digest_relative'], 'source digest relation differs')
    archive = ROOT / 'source.tar'
    immutable(archive)
    require(source['archive'] == str(archive) and ct.sha(archive) == identity['source_archive_sha256'],
            'archive identity differs')
    members = {}
    with tarfile.open(archive, 'r|') as stream:
        for member in stream:
            safe_member(member.name.rstrip('/') if member.isdir() else member.name)
            if member.isdir():
                continue
            require(member.isfile() and member.name in relative and member.name not in members,
                    'archive link/extra/duplicate member')
            with stream.extractfile(member) as data:
                pin = hashlib.file_digest(data, 'sha256').hexdigest()
            require(pin == relative[member.name], 'archive member bytes differ')
            members[member.name] = pin
    require(members == relative, 'archive coverage differs')
    recorded = load('archive-exact-members.json')
    require(recorded['commit'] == identity['commit'] and recorded['file_count'] == len(members)
            and {n: v['sha256'] for n, v in recorded['members'].items()} == members,
            'owner archive review relation differs')
    profile = tomllib.loads((frozen / 'Cargo.toml').read_text())['profile']['release']
    require(all(profile.get(k) == v for k, v in {'opt-level': 3, 'lto': 'fat', 'codegen-units': 1,
            'panic': 'unwind', 'strip': 'symbols'}.items()), 'release profile differs')
    actual = {str(Path(p).relative_to(frozen)) for p in inventory_tree(frozen)}
    extras = []
    for name in sorted(actual - set(relative)):
        safe_member(name)
        path = frozen / name
        extras.append(dict(path=str(path), relative_path=name, sha256=ct.sha(path),
            source_pin_member=name in relative, archive_member=name in members,
            evidence_manifest_pin=manifest['pins_relative_to_directory'].get('frozen-source/' + name),
            cleanup_performed=False, deletion_authorized=False))
    return dict(archive_and_all_pinned_source_bytes_verified=True, archive_members=len(members),
        actual_regular_files=len(actual), missing=sorted(set(relative) - actual), extra_files=extras,
        strict_exact_tree_pass=actual == set(relative),
        preservation_required='Build owner must preserve evidence-manifest-pinned bytecode; do not delete pinned files'), source


def build_review(module, source, artifacts, runtime):
    env = load('build-environment.json')
    require(set(env) == ENV_KEYS and all(isinstance(v, str) for v in env.values()), 'build environment allowlist')
    expected = dict(CARGO_TARGET_DIR=str(ROOT / 'cargo-target'), CARGO_INCREMENTAL='0',
        FERRULE_CUDA_ARCH='sm_86', FERRULE_CUDA_TEST_ORACLE='0', RUSTUP_TOOLCHAIN='nightly-2026-08-01',
        RUSTUP_AUTO_INSTALL='0', CUBLAS_WORKSPACE_CONFIG=':0:0', CC='/usr/bin/gcc', CXX='/usr/bin/g++',
        CUDAHOSTCXX='/usr/bin/g++', CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
        FERRULE_PROFILE_NVTX='0', PYTHONDONTWRITEBYTECODE='1')
    require(all(env[k] == v for k, v in expected.items()), 'build/compiler/CUDA controls differ')
    tools = load('toolchain.json')
    require(set(tools) == {'cargo', 'rustc', 'nvcc', 'cutlass'} and all(v['returncode'] == 0
            and v['stderr'] == '' for v in tools.values()), 'toolchain receipt failed')
    require('commit-hash: ' + RUST_COMMIT + '\n' in tools['rustc']['stdout']
            and 'V12.4.99' in tools['nvcc']['stdout'] and tools['cutlass']['stdout'] == CUTLASS_COMMIT + '\n',
            'toolchain version differs')
    external = load('external-input-pins.json')['pins']
    ct.check_pins(external)
    for key in ('NVCC', 'RUSTC', 'CC', 'CXX'):
        require(env[key] in external or str(Path(env[key]).resolve()) in external, 'unbound compiler input')
    for key in ('FERRULE_CUTLASS_DIR', 'FERRULE_NVTX_INCLUDE_DIR'):
        require(inventory_tree(env[key]) <= set(external), 'unbound CUTLASS/NVTX tree')
    native = load('native-archive-pins-final.json')
    require(set(native['per_role']) == set(ROLES) and native['all_roles_same_new_target'] is True
            and native['captured_immediately_after_each_role_link'] is True
            and native['cargo_graph_and_executable_equality_not_claimed'] is True, 'native provenance differs')
    packages, copies, role_pins, review_rows = {}, {}, {}, {}
    for role in ROLES:
        artifact = artifacts[role]
        prefix = 'build-final-' + role
        args = ['/root/.cargo/bin/cargo', 'build' if role == 'cli' else 'test', '--locked', '--offline',
            '--release', '--no-default-features', '-p',
            dict(cli='ferrule-cli', constructor='ferrule-model', composition='ferrule-composition')[role],
            '--features', 'cuda,ferrule-common/nvtx-profile']
        args += [] if role == 'cli' else (['--test', 'expert_read_constructor', '--no-run']
                                         if role == 'constructor' else ['--lib', 'full40', '--no-run'])
        args += ['--message-format=json']
        receipt = module.command(ROOT, prefix, args, env)
        require(receipt['cwd'] == source['frozen_source'] and receipt['timeout_seconds'] == 1800, 'build cwd/deadline')
        rows = [json.loads(s) for s in (ROOT / (prefix + '.stdout')).read_text().splitlines() if s.startswith('{')]
        require(rows and rows[-1] == dict(reason='build-finished', success=True), 'Cargo completion absent')
        compiled = artifact['compiled_artifact']
        require(compiled in rows and compiled['profile']['opt_level'] == '3'
                and compiled['profile']['debug_assertions'] is False
                and compiled['profile']['test'] is (role != 'cli')
                and artifact['source_digest'] == source['source_pins_digest'] and artifact['nvtx_compiled'] is True
                and artifact['qualification_nvtx_runtime_enabled'] is False, 'linked artifact/profile/source differs')
        require('cuda' in compiled['features'] and (role != 'cli' or compiled['features'] == ['cuda']),
                'CUDA role feature differs')
        workspace = [r for r in rows if r.get('reason') == 'compiler-artifact'
                     and r.get('manifest_path', '').startswith(source['frozen_source'] + '/')]
        require(workspace and all('cpu-reference' not in r['features'] for r in workspace), 'CPU reference linked')
        common = [r for r in workspace if r['manifest_path'] == source['frozen_source'] + '/crates/ferrule-common/Cargo.toml']
        require(common == artifact['common_artifacts'] and common and all('nvtx-profile' in r['features'] for r in common),
                'actual common NVTX feature missing')
        staged = Path(artifact['executable'])
        immutable(staged)
        require(staged == ROOT / 'staged-final' / role and ct.sha(staged) == artifact['sha256']
                and staged.stat().st_size == artifact['bytes'], 'final staged artifact relation differs')
        with staged.open('rb') as stream:
            require(stream.read(4) == b'\x7fELF', 'not ELF')
        scripts = [r for r in rows if r.get('reason') == 'build-script-executed']
        snapshot = load('native-role-final-' + role + '.json')
        require(snapshot == native['per_role'][role] and snapshot['role'] == role
                and snapshot['binary_sha256'] == artifact['sha256']
                and snapshot['source_digest'] == source['source_pins_digest']
                and snapshot['captured_immediately_after_successful_link'] is True, 'per-role postlink snapshot differs')
        linked_static = {str(Path(r['out_dir']) / ('lib' + lib.split('=', 1)[1] + '.a'))
                         for r in scripts for lib in r.get('linked_libs', []) if lib.startswith('static=')}
        originals = set()
        for name, row in snapshot['pins'].items():
            copied, original = Path(name), Path(row['original_path'])
            immutable(copied)
            require(copied.parent == ROOT / 'native-role-final' / role
                    and original.is_relative_to(ROOT / 'cargo-target')
                    and copied.name.endswith('-' + original.name)
                    and row['sha256'] == row['original_sha256'] == ct.sha(copied), 'native copy/hash relation differs')
            originals.add(str(original))
            copies[name] = row['sha256']
            role_pins[role + ':' + str(original)] = row['sha256']
        require(linked_static <= originals and {'libferrule_cpu_native.a', 'libferrule_cuda_core.a',
                'libferrule_cuda_cutlass.a', 'libferrule_nvtx_host.a'} <= {Path(n).name for n in linked_static},
                'Cargo-linked static archives not covered by role snapshot')
        for r in rows:
            if r.get('reason') == 'compiler-artifact' and r.get('package_id', '').startswith('registry+'):
                packages[r['manifest_path']] = r['package_id']
        module.command(ROOT, 'final-ldd-' + role, ['/usr/bin/ldd', str(staged)])
        text = (ROOT / ('final-ldd-' + role + '.stdout')).read_text()
        libraries = {str(Path(n).resolve()) for n in re.findall(r'(/[^\s()]+)\s+\(', text)}
        require('not found' not in text and libraries == set(runtime['per_role'][role])
                and all(runtime['pins'][n] == pin for n, pin in runtime['per_role'][role].items()), 'per-role loader binding differs')
        review_rows[role] = dict(executable=str(staged), sha256=artifact['sha256'],
            feature_argv='cuda,ferrule-common/nvtx-profile', common_nvtx_compiled=True,
            native_snapshot_sha256=ct.sha(ROOT / ('native-role-final-' + role + '.json')),
            native_copy_count=len(snapshot['pins']), cargo_linked_static_archives=sorted(linked_static),
            runtime_library_count=len(libraries), build_custody='unobserved; bounded CPU command receipts only')
    require(copies == native['immutable_copies'] and role_pins == native['original_role_pins'], 'native aggregate relation')
    require(artifacts['cli']['sha256'] == binding.EXPECTED_IDENTITY['cli_binary_sha256'], 'wrong CLI')
    dependencies = load('compiled-dependency-pins-final.json')
    require(packages == dependencies['packages'] and set(dependencies['pins']) == set().union(
        *(inventory_tree(Path(n).parent) for n in packages)), 'compiled dependency coverage differs')
    ct.check_pins(dependencies['pins'])
    ct.check_pins(runtime['pins'])
    for name, args in (('help', ['--help']), ('serve-help', ['serve', '--help']), ('version', ['--version'])):
        module.command(ROOT, 'final-cli-' + name, [artifacts['cli']['executable'], *args])
        require((ROOT / ('final-cli-' + name + '.stderr')).read_bytes() == b'', 'CLI help stderr nonempty')
    flags = set(re.findall(r'--[a-z][a-z0-9-]*', (ROOT / 'final-cli-serve-help.stdout').read_text()))
    require({s for s in binding.case_for_mode('blocking')['argv_after_binary'] if s.startswith('--')} <= flags,
            'final CLI field help missing')
    compile_map = load('nvtx-compile-test-map.json')
    require(compile_map['status'] == 'passed' and compile_map['unit_tests_actual_passed'] == 12
            and compile_map['unit_runtime_environment_nvtx'] == '0'
            and compile_map['explicit_sdk_include'] == env['FERRULE_NVTX_INCLUDE_DIR'], 'NVTX compile map differs')
    ct.check_pins(compile_map['official_sdk_header_pins'])
    for role in ROLES:
        require(compile_map['qualification_roles'][role]['executable_sha256'] == artifacts[role]['sha256']
                and compile_map['qualification_roles'][role]['common_nvtx_feature_artifacts'] == artifacts[role]['common_artifacts'],
                'NVTX role compile mapping differs')
    return dict(per_role=review_rows, dependency_files=len(dependencies['pins']), external_input_files=len(external),
        native_scope='Recorded immediate per-role immutable copies; current expendable Cargo archives need not retain prelink bytes',
        whole_graph_or_binary_equality_claim=False, cli_serving_accepted=False)


def paired_fingerprints(actual, primary, count, subset=False):
    calls = actual.get('prediction_fingerprints', {}).get('calls', {})
    prior = primary.get('prediction_fingerprints', {}).get('calls', {})
    require(isinstance(calls, dict) and len(calls) == count and isinstance(prior, dict)
            and (all(k in prior and v == prior[k] for k, v in calls.items()) if subset else calls == prior),
            'actual primary fingerprints differ')
    return count


def gate_review(module, qualification, source, runtime):
    reports, rows = {}, []
    references = load('reference-pins.json')
    ct.check_pins(references['reference_files'])
    ct.check_pins(references['model_metadata'])
    require(references['bf16_atol'] == .00025 and references['bf16_rtol'] == .012, 'reference tolerances changed')
    audit = {r['gate_id']: r for r in load('receipt-audit.json')['gates']}
    for gate in qualification['gates']:
        report, row = module.review_gate(ROOT, gate, qualification['artifacts'], runtime, source)
        require(report['complete'] is True and report['passed'] is True, 'raw report incomplete/failed')
        directory = ROOT / 'gates' / gate['id']
        result = ct.load(directory / 'gate-result.json')
        ct.check_pins(result['log_pins'])
        require(result['binary_sha256'] == gate['binary_sha256'] and result['failures'] == []
                and result['recorded_owners_absent'] is True, 'producer result differs')
        require(audit[gate['id']]['report_sha256'] == row['report_sha256']
                and audit[gate['id']]['executable_sha256'] == row['executable_sha256'], 'owner audit relation differs')
        for name in ('pre-idle', 'authenticated-pre-cuda-idle', 'post-idle'):
            idle = ct.load(directory / (name + '.json'))
            indices = idle['accepted_sample_indices']
            require(len(indices) == 3 and indices == list(range(indices[0], indices[0] + 3))
                    and all(s['classification'] in ('idle', 'utilization_only_transient') for s in idle['samples']),
                    'idle acceptance/transient scope differs')
            if name != 'post-idle':
                require(len(idle['samples']) == 3, 'preflight stabilization forbidden')
        reports[gate['id']] = report
        if gate['kind'] == 'constructor':
            module.validate_constructor(report)
            row['actual_verified_reads_per_cell'] = [c['observed']['stats']['verified_reads'] for c in report['cells']]
        else:
            plan.compare_report_configuration(binding.case_for_mode(gate['schedule']), report)
            module.validate_reads(report, gate['schedule'])
            row.update(scheduler_fields_compared=sorted(plan.SCHEDULER),
                fingerprint_calls=len(report.get('prediction_fingerprints', {}).get('calls', {})),
                metric_count=len(report['metrics']), main_logit_values=report['logits_compared'],
                actual_ids={r['case']: r['free_greedy_ids'] for r in report.get('cases', [])},
                resource_release='unobserved; do not infer physical release from idle/ECHILD')
        rows.append(row)
    oracles = {r['id']: r for r in ct.load(HERE / 'oracles.json')['references']}
    for mode, numbers in (('blocking', ('02', '04', '06', '08')), ('lookahead', ('03', '05', '07', '09'))):
        group = {label: reports[n + '-' + mode + '-' + label]
                 for label, n in zip(('full40', 'fresh', 'ordinary', 'cold'), numbers)}
        validate_quality_reports({k: group[k] for k in ('full40', 'fresh', 'cold')}, binding.case_for_mode(mode))
        require(group['full40']['reference_manifest_sha256'] == references['numeric_manifest_sha256']
                and group['fresh']['oracle_manifest_sha256'] == group['cold']['oracle_manifest_sha256']
                == references['fresh_manifest_sha256'], 'canonical references differ')
        primary = ROOT / 'gates' / (numbers[1] + '-' + mode + '-fresh') / 'quality.json'
        module.validate_ordinary(group['ordinary'], group['fresh'], ct.sha(primary), mode)
        paired_fingerprints(group['ordinary'], group['fresh'], 64)
        paired_fingerprints(group['cold'], group['fresh'], 28, subset=True)
        for label, n in (('ordinary', numbers[2]), ('cold', numbers[3])):
            launch = load('gates/' + n + '-' + mode + '-' + label + '/worker-launch.json')
            require(launch['environment']['FERRULE_BF16_FRESH_PRIMARY_REPORT'] == str(primary), 'current primary path differs')
            if label == 'ordinary':
                require(launch['environment']['FERRULE_BF16_FRESH_OBSERVED_REPORT_SHA256'] == ct.sha(primary), 'primary hash differs')
        for case in group['fresh']['cases']:
            require(case['oracle_generated_ids'][:8] == oracles[case['case']]['reference_ids'], 'generation8 oracle differs')
        require(group['fresh']['cases'][0]['prompt_ids'] == ct.load(HERE / 'oracles.json')['qa']['input_ids'], 'QA prompt IDs differ')
    return dict(all_nine_raw_producers_and_quality_reviewed=True, gates=rows,
        both_ordinary_primary_fingerprints=64, both_cold_primary_subset_fingerprints=28,
        reference_sha256=dict(full40=references['numeric_manifest_sha256'], fresh=references['fresh_manifest_sha256']),
        cli_http_or_enqueue_join_observed=False)
