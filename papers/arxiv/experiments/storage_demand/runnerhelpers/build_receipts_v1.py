"""Read-only, explicitly selected adapter for the immutable CPU CLI builder.

This reviews recorded evidence, never builds, runs help, authenticates a producer,
queries a GPU, or promotes build facts to numerical/serving qualification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tarfile
import tomllib

from .evidence_v2 import ct, integer, require

ADAPTER = 'ferrule.storage-demand.cuda-cli-build-adapter.v1'
BUILD_SCHEMA = 'ferrule.cuda-only-cli-release-build.v1'
SOURCE_SCHEMA = 'ferrule.cuda-cli-source-freeze.v1'
PIN_GROUPS = ('compiler-input-pins.json', 'cutlass-tree-pins.json',
              'compiled-dependency-pins.json', 'runtime-library-pins.json')
RUST_COMMIT = 'ad3d0bc141a02cf446e384136d250a1f6950fed5'
CUTLASS_COMMIT = 'e05f953a5b3d38adc240df2ff928e0421c2abba3'
ENV_KEYS = frozenset(('CARGO_BUILD_JOBS CARGO_HOME CARGO_INCREMENTAL CARGO_TARGET_DIR CC '
    'CUBLAS_WORKSPACE_CONFIG CUDAHOSTCXX CUDA_HOME CUDA_VISIBLE_DEVICES CXX '
    'FERRULE_CUDA_ARCH FERRULE_CUDA_TEST_ORACLE FERRULE_CUTLASS_DIR HOME LD_LIBRARY_PATH '
    'LIBRARY_PATH NVCC NVIDIA_VISIBLE_DEVICES PATH RUSTC RUSTDOC RUSTUP_AUTO_INSTALL '
    'RUSTUP_HOME RUSTUP_TOOLCHAIN').split())
SOURCE_SUFFIXES = {'.rs', '.cu', '.cuh', '.cc', '.cpp', '.c', '.h', '.toml', '.json',
                   '.ptx', '.md', '.txt', '.bin', '.csv', '.yaml', '.yml'}


def pinned_path(candidate, path, label):
    require(isinstance(path, str) and Path(path).is_absolute(), 'missing absolute evidence path: ' + label)
    pins = candidate.get('artifact_pins', {})
    require(path in pins, 'unpinned build evidence: ' + label)
    ct.check_pins({path: pins[path]})
    return Path(path)


def load_evidence(candidate, path, label):
    return ct.load(pinned_path(candidate, path, label))


def immutable(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_mode & 0o222 == 0,
            'not an immutable regular file: ' + str(path))


def inventory_tree(root):
    return {str(p) for p in Path(root).rglob('*') if p.is_file() and '.git' not in p.parts}


def review_source(candidate, build):
    source = load_evidence(candidate, candidate.get('source_archive_manifest'), 'source_archive_manifest')
    require(source.get('schema') == SOURCE_SCHEMA and source.get('immutable_readonly_inputs') is True
            and source.get('archive_members_exactly_match_copied_source') is True, 'unknown source freeze evidence')
    root = Path(build['cwd'])
    require(source.get('frozen_source') == str(root) and root.is_absolute() and root.is_dir()
            and not root.is_symlink() and root.stat().st_mode & 0o222 == 0, 'compiled frozen-source root mismatch')
    pins = source.get('pins_relative_to_frozen_source')
    require(isinstance(pins, dict) and pins and integer(source.get('file_count'), 1)
            and source['file_count'] == len(pins) == source.get('archive_members_read_and_hashed'),
            'incomplete full source pins')
    for name, expected in pins.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and str(path) == name,
                'unsafe source member: ' + name)
        target = root / path
        require(target.resolve().is_relative_to(root.resolve()), 'source escaped frozen root')
        immutable(target)
        ct.check_pins({str(target): expected})
    required = {'Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml'}
    required |= {str(p.relative_to(root)) for p in (root / '.cargo').rglob('*.toml')}
    required |= {str(p.relative_to(root)) for p in (root / 'crates').rglob('*')
                 if p.is_file() and p.suffix in SOURCE_SUFFIXES}
    require(required <= set(pins), 'missing compiled source inputs')
    original = {str(ct.ROOT / name): expected for name, expected in pins.items()}
    require(source.get('source_pins_digest_original_absolute_keys') == ct.digest(original)
            and source.get('source_pins_digest_relative_keys') == ct.digest(pins), 'source cut digest mismatch')
    archive = pinned_path(candidate, candidate.get('source_archive'), 'source_archive')
    immutable(archive)
    require(source.get('source_archive') == str(archive)
            and ct.sha(archive) == source.get('source_archive_sha256'), 'source archive hash/relation mismatch')
    seen, payload_bytes = set(), 0
    with tarfile.open(archive, 'r|gz') as stream:
        for member in stream:
            require(member.isfile() and member.name in pins and member.name not in seen,
                    'source archive member mismatch/link/duplicate: ' + member.name)
            with stream.extractfile(member) as data:
                actual = hashlib.file_digest(data, 'sha256').hexdigest()
            require(actual == pins[member.name], 'source archive member hash mismatch: ' + member.name)
            seen.add(member.name)
            payload_bytes += member.size
    require(seen == set(pins) and payload_bytes == source.get('archive_payload_bytes'), 'source archive coverage mismatch')
    profile = tomllib.loads((root / 'Cargo.toml').read_text()).get('profile', {}).get('release', {})
    require(all(profile.get(key.replace('_', '-')) == value for key, value in build['release_profile'].items()),
            'frozen Cargo release profile mismatch')
    return original


def review_command(build):
    require(build.get('schema') == BUILD_SCHEMA, 'unsupported CPU CLI build schema')
    require(type(build.get('returncode')) is int and build['returncode'] == 0
            and build.get('timed_out') is False and integer(build.get('timeout_seconds'), 1)
            and build['timeout_seconds'] <= 1800, 'CPU build failed/unknown/unbounded')
    argv = build.get('argv', [])
    require(len(argv) == 11 and Path(argv[0]).is_absolute() and Path(argv[0]).name == 'cargo'
            and argv[1:] == ['build', '--locked', '--offline', '--release', '-p', 'ferrule-cli',
                             '--no-default-features', '--features', 'cuda', '--message-format=json'],
            'unreviewed CUDA-only build argv')
    require(build.get('outer_supervisor_argv') == ['/usr/bin/timeout', str(build['timeout_seconds']), *argv],
            'CPU timeout command mismatch')
    require(build.get('environment_policy') ==
            'env -i exact allowlist; no inherited RUSTFLAGS, wrappers, CPU reference feature or GPU visibility',
            'unknown CPU environment isolation policy')
    require(build.get('external_input_pin_receipt_timing') ==
            'Post-build reverified byte-for-byte against previously sealed quality pins; not a newly captured pre-build receipt',
            'unknown external-input pin timing')
    env = build.get('environment', {})
    require(set(env) == ENV_KEYS and all(isinstance(v, str) for v in env.values()), 'unreviewed build environment allowlist')
    require(env['CUDA_VISIBLE_DEVICES'] == '' and env['NVIDIA_VISIBLE_DEVICES'] == 'none'
            and env['FERRULE_CUDA_ARCH'] == 'sm_86' and env['FERRULE_CUDA_TEST_ORACLE'] == '0'
            and env['RUSTUP_TOOLCHAIN'] == 'nightly-2026-08-01' and env['RUSTUP_AUTO_INSTALL'] == '0'
            and env['CARGO_INCREMENTAL'] == '0' and env['CUBLAS_WORKSPACE_CONFIG'] == ':0:0'
            and env['CC'] == '/usr/bin/gcc' and env['CXX'] == env['CUDAHOSTCXX'] == '/usr/bin/g++',
            'CPU visibility/compiler/runtime pins differ')
    require(build.get('compiled_cuda') == '12.4.99' and build.get('compiled_arch') == 'sm_86'
            and build.get('source_frozen_before_build') is True and build.get('shared_build_target_reused') is False
            and build.get('gpu_tools_or_workloads_launched') is False, 'unknown build isolation/source/GPU scope')
    require(build.get('release_profile') == dict(opt_level=3, lto='fat', codegen_units=1, panic='unwind', strip='symbols'),
            'unknown release profile')
    tools = build.get('toolchain', {})
    require(set(tools) == {'cargo', 'rustc', 'nvcc', 'cutlass_head', 'cutlass_status'}, 'full toolchain evidence missing')
    for name, tool in tools.items():
        require(type(tool.get('returncode')) is int and tool['returncode'] == 0 and tool.get('stderr') == ''
                and tool.get('exact_quality_tool_output_match') is True, 'toolchain check failed: ' + name)
    require(tools['cargo'].get('argv') == [argv[0], '-V']
            and tools['rustc'].get('argv') == [env['RUSTC'], '-vV']
            and tools['nvcc'].get('argv') == [env['NVCC'], '--version'], 'toolchain executable relation missing')
    require('commit-hash: ' + RUST_COMMIT + '\n' in tools['rustc'].get('stdout', '')
            and tools['cargo'].get('stdout') == 'cargo 1.99.0-nightly (7c83d4cc0 2026-07-29)\n'
            and 'V12.4.99' in tools['nvcc'].get('stdout', '')
            and tools['cutlass_head'].get('stdout') == CUTLASS_COMMIT + '\n'
            and tools['cutlass_status'].get('stdout') == '', 'toolchain version/full commit mismatch')


def review_cli_build(candidate, build, case):
    """Return factual build coverage only, not a candidate seal or a PASS upgrade."""
    require(candidate.get('build_adapter') == ADAPTER, 'explicit reviewed build adapter required')
    require(load_evidence(candidate, candidate.get('build_receipt'), 'build_receipt') == build, 'raw build receipt changed')
    review_command(build)
    evidence = candidate.get('build_evidence', {})
    def receipt(key):
        return load_evidence(candidate, evidence.get(key), key)
    def raw(key, expected):
        path = pinned_path(candidate, evidence.get(key), key)
        require(ct.sha(path) == expected, 'raw build evidence hash mismatch: ' + key)
        return path
    stdout = raw('build_stdout', build.get('full_stdout_sha256'))
    raw('build_stderr', build.get('full_stderr_sha256'))
    rows = [json.loads(line)
            for line in stdout.read_text().splitlines() if line.startswith('{')]
    require(rows and rows[-1] == dict(reason='build-finished', success=True), 'Cargo completion evidence missing')
    artifact = receipt('artifact')
    cli = ct.release_artifact(rows)
    require(cli == artifact.get('compiled_artifact') and cli.get('features') == ['cuda']
            and cli.get('fresh') is False, 'CLI Cargo artifact mismatch/reused/CPU-reference')
    root, target = Path(build['cwd']), Path(build['environment']['CARGO_TARGET_DIR'])
    require(root.is_absolute() and target.is_absolute() and not target.is_relative_to(root), 'isolated build roots required')
    workspace = [r for r in rows if r.get('reason') == 'compiler-artifact'
                 and Path(r.get('manifest_path', '/')).is_relative_to(root)]
    require(workspace == artifact.get('workspace_artifacts') and workspace
            and all('cpu-reference' not in r.get('features', []) for r in workspace), 'workspace feature/artifact coverage mismatch')
    require(cli.get('manifest_path') == str(root / 'crates/ferrule-cli/Cargo.toml')
            and cli.get('target', {}).get('src_path') == str(root / 'crates/ferrule-cli/src/main.rs')
            and cli.get('executable') == str(target / 'release/ferrule'), 'Cargo source/output relation mismatch')
    binary = pinned_path(candidate, candidate.get('binary'), 'binary')
    expected = artifact.get('staged_binary_sha256')
    ct.validate_staged_binary(binary, expected)
    require(candidate.get('binary_sha256') == expected and artifact.get('staged_binary') == str(binary)
            and binary.stat().st_size == artifact.get('staged_binary_bytes'), 'staged binary relation mismatch')
    output = Path(cli['executable'])
    require(output.is_file() and not output.is_symlink() and ct.sha(output) == expected
            and not output.samefile(binary) and artifact.get('independent_inode_from_build_output') is True,
            'binary copy hash/inode mismatch')
    sources = review_source(candidate, build)
    proof = receipt('compiler_pins')
    require(set(proof) == set(PIN_GROUPS), 'full compiler/CUTLASS/dependency/runtime pin groups missing')
    for name, group in proof.items():
        ct.check_pins(group.get('pins'))
        require(group.get('matched') == len(group['pins']), 'pin group coverage mismatch: ' + name)
        reference = Path(group.get('reference_receipt', ''))
        require(reference.is_absolute() and ct.sha(reference) == group.get('reference_receipt_sha256')
                and ct.load(reference).get('pins') == group['pins'], 'previous sealed pin relation mismatch: ' + name)
    compiler = proof['compiler-input-pins.json']['pins']
    for path in (build['argv'][0], *(build['environment'][k] for k in ('NVCC', 'RUSTC', 'CC', 'CXX'))):
        require(path in compiler or str(Path(path).resolve()) in compiler, 'compiler executable pin missing: ' + path)
    cutlass = proof['cutlass-tree-pins.json']['pins']
    require(set(cutlass) == inventory_tree(build['environment']['FERRULE_CUTLASS_DIR']), 'full CUTLASS tree pins missing')
    dependencies = receipt('dependencies')
    ct.check_pins(dependencies.get('pins'))
    dep_rows = [r for r in rows if r.get('reason') == 'compiler-artifact' and r.get('package_id', '').startswith('registry+')]
    manifests = {r['manifest_path'] for r in dep_rows}
    require(dep_rows == dependencies.get('artifacts') and manifests
            and len(manifests) == dependencies.get('package_count')
            and len(dependencies['pins']) == dependencies.get('file_count'), 'actual dependency artifact/pin coverage missing')
    dep_files = set().union(*(inventory_tree(Path(m).parent) for m in manifests))
    require(set(dependencies['pins']) == dep_files, 'full actual dependency source pins missing')
    native, copies = receipt('native_archives'), receipt('immutable_native_archives')
    ct.check_pins(native.get('pins'))
    ct.check_pins(copies.get('pins'))
    require(set(native['pins']) == {str(p) for p in target.rglob('*.a')}
            and native['pins'] == copies.get('original_build_archives'), 'native archive coverage/relation missing')
    by_name = {Path(p).name: h for p, h in native['pins'].items()}
    require(len(by_name) == len(native['pins']) == len(copies['pins'])
            and {Path(p).name: h for p, h in copies['pins'].items()} == by_name
            and {'libferrule_cpu_native.a', 'libferrule_cuda_core.a', 'libferrule_cuda_cutlass.a'} <= set(by_name),
            'independent native archive copies missing')
    for path in copies['pins']:
        immutable(path)
        require(not Path(path).samefile(next(p for p in native['pins'] if Path(p).name == Path(path).name)),
                'native archive copy shares build inode')
    runtime = receipt('runtime_resolution')
    libraries = runtime.get('libraries', {})
    runtime_pins = {p: row.get('sha256') for p, row in libraries.items()}
    require(libraries and runtime.get('count') == len(libraries)
            and runtime.get('required_ld_library_path') == build['environment']['LD_LIBRARY_PATH']
            and runtime.get('ldd_is_cpu_only_static_loader_resolution') is True
            and runtime.get('gpu_context_and_actual_server_maps_unobserved') is True
            and runtime.get('explicit_model_vendor_dlopen_libraries_not_exercised') is True
            and runtime.get('all_resolved_libraries_match_quality_runtime_pins') is True,
            'unknown runtime resolution/loaded-runtime coverage')
    ct.check_pins(runtime_pins)
    prior_runtime = proof['runtime-library-pins.json']['pins']
    require(all(prior_runtime.get(p) == h for p, h in runtime_pins.items()), 'runtime pins differ from sealed inputs')
    for path, row in libraries.items():
        require(Path(row.get('load_path', '')).resolve() == Path(path)
                and Path(path).stat().st_size == row.get('bytes'), 'runtime load-path/hash relation mismatch')
    ldd = receipt('ldd_receipt')
    review_probe(ldd, ['/usr/bin/ldd', str(binary)], build)
    text = raw('ldd_stdout', ldd.get('stdout_sha256')).read_text()
    require(raw('ldd_stderr', ldd.get('stderr_sha256')).read_bytes() == b'', 'ldd stderr not empty')
    require('not found' not in text and {str(Path(p).resolve()) for p in re.findall(r'(/[^\s()]+)\s+\(', text)}
            == set(runtime_pins), 'ldd runtime pin coverage missing')
    for label, argv in (('help', ['--help']), ('serve_help', ['serve', '--help']), ('version', ['--version'])):
        probe_paths = evidence.get('help', {}).get(label, {})
        probe = load_evidence(candidate, probe_paths.get('receipt'), label)
        review_probe(probe, [str(binary), *argv], build)
        text_path = pinned_path(candidate, probe_paths.get('stdout'), label + ' stdout')
        error_path = pinned_path(candidate, probe_paths.get('stderr'), label + ' stderr')
        require(ct.sha(text_path) == probe.get('stdout_sha256') and text_path.stat().st_size > 0
                and ct.sha(error_path) == probe.get('stderr_sha256') and error_path.read_bytes() == b'',
                'help/version raw evidence hash mismatch: ' + label)
        if label == 'serve_help':
            require(probe_paths['receipt'] == candidate.get('serve_help_receipt'), 'serve-help receipt relation mismatch')
            flags = {part for part in case['argv_after_binary'] if part.startswith('--')}
            help_flags = set(re.findall(r'--[a-z][a-z0-9-]*', text_path.read_text()))
            require(flags <= help_flags, 'exact CLI field-help flags missing')
    return dict(adapter=ADAPTER, raw_schema=BUILD_SCHEMA, binary=str(binary), binary_sha256=expected,
                sources=sources, source_digest=ct.digest(sources), runtime_pins=runtime_pins,
                coverage=dict(authenticated_build_custody='unobserved', loaded_model_runtime='unobserved',
                              gpu_context='unobserved', field_help='observed_cpu_only',
                              external_pin_timing=build.get('external_input_pin_receipt_timing')),
                quality_approved=False, gpu_authorized=False)


def review_probe(probe, argv, build):
    env = probe.get('environment', {})
    require(probe.get('argv') == argv and probe.get('cwd') == build['cwd']
            and probe.get('gpu_hidden') is True and type(probe.get('returncode')) is int and probe['returncode'] == 0
            and probe.get('timed_out') is False and integer(probe.get('timeout_seconds'), 1)
            and probe['timeout_seconds'] <= 30 and env.get('CUDA_VISIBLE_DEVICES') == ''
            and env.get('NVIDIA_VISIBLE_DEVICES') == 'none'
            and env.get('LD_LIBRARY_PATH') == build['environment']['LD_LIBRARY_PATH'], 'CPU help/ldd probe failed/unbound')


def validate_candidate_build(candidate, build, config, case):
    """Select by explicit version, never reinterpret an unknown raw schema."""
    if candidate.get('build_adapter') == ADAPTER:
        return review_cli_build(candidate, build, case)
    require(candidate.get('build_adapter') is None and build.get('schema') == 'ferrule.paper.m7.build.v1',
            'unknown build schema/adapter; explicit reviewed version required')
    binary = ct.validate_build(build, config)
    return dict(adapter='legacy_m7_unchanged', raw_schema=build['schema'], binary=binary,
                binary_sha256=build['binary_sha256'], sources=build['sources_before'],
                source_digest=ct.digest(build['sources_before']), runtime_pins=None,
                coverage=dict(authenticated_build_custody='verified_by_legacy_contract'),
                quality_approved=False, gpu_authorized=False)
