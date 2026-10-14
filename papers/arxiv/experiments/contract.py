"""CPU-only pins and fail-closed contracts; no model/runtime imports."""
import contextlib
import fcntl
import hashlib
import json
import os
import shutil

from pathlib import Path
import re
import stat
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DEFAULT = HERE / 'm7.json'
BASE = json.loads(DEFAULT.read_text())
EXCLUSIVE_PATH = HERE / 'm7-device-exclusive.json'
EXCLUSIVE = json.loads(EXCLUSIVE_PATH.read_text())
IDENTITY_MODES = ('hostpid_verified', 'device_exclusive_local_custody')
EXCLUSIVE_SHA256 = 'cb843e69454cc29c3e710586bc2e8335a99e7d275df164de8dcf5be6c629fd4c'
SCHEMA = 'ferrule.paper.m7.'
VLLM_FIX_KEYS = frozenset(('CUDA_HOME', 'CUDA_PATH', 'CUDACXX', 'NVCC',
                           'FLASHINFER_NVCC', 'PATH', 'FLASHINFER_WORKSPACE_BASE', 'MAX_JOBS'))
VLLM_FORBIDDEN_KEYS = frozenset(('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES',
                                 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'CUBLAS_WORKSPACE_CONFIG',
                                 'VLLM_MARLIN_INPUT_DTYPE', 'VLLM_WEIGHT_OFFLOADING_DISABLE_UVA',
                                 'VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY'))
VLLM_EXPECTED_FLAGS = {
    '--host': 'str', '--port': 'int', '--served-model-name': 'str', '--language-model-only': 'bool',
    '--tensor-parallel-size': 'int', '--pipeline-parallel-size': 'int', '--data-parallel-size': 'int',
    '--dtype': 'enum', '--kv-cache-dtype': 'enum', '--mamba-cache-dtype': 'enum',
    '--mamba-ssm-cache-dtype': 'enum', '--linear-backend': 'enum', '--moe-backend': 'enum',
    '--offload-backend': 'enum', '--cpu-offload-gb': 'float', '--cpu-offload-params': 'list',
    '--max-model-len': 'int', '--max-num-seqs': 'int', '--max-num-batched-tokens': 'int',
    '--enable-chunked-prefill': 'bool', '--gpu-memory-utilization': 'float',
    '--no-enable-prefix-caching': 'bool', '--generation-config': 'str', '--seed': 'int',
    '--enforce-eager': 'bool',
}

CONFIG_SHA256 = '806fd7b808657abb59929c4426cdedae9dc2ea532e003d79eb22f79186beb3d6'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def load(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def absolute(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def write(path, value):
    """Final receipts are exclusive-create, never rewritten by this runner."""
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def exclusive_owner_lock(uuid):
    """Local nonblocking serialization, not a global GPU reservation."""
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uuid or ''), 'Invalid lock UUID')
    path = HERE / (uuid + '.lock')
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Local owner lock is held; no telemetry or launch') from error
        st = os.fstat(fd)
        yield dict(path=str(path), fd=fd, device=st.st_dev, inode=st.st_ino,
                   coordinator_pid=os.getpid(), mode='local_nonblocking_flock', global_reservation=False)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def new_run(identity):
    require(isinstance(identity, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', identity),
            'Need a new filesystem-safe --id (no dots/slashes)')
    runs = HERE / 'runs'
    runs.mkdir(exist_ok=True)
    require(not runs.is_symlink(), 'runs must not be a symlink')
    directory = runs / identity
    directory.mkdir(exist_ok=False)
    return directory


def require_cpu_python():
    """Enter an explicitly CPU-only orchestration mode, never an import hook.

    CLI/worker entry points call this before work; GPU model/collector targets
    instead inherit their sealed allowlisted environment without calling it.
    CPU subprocesses additionally use runner.cpu_environment()/env -i.
    """
    base = Path('/home/admin/wxw403883/anaconda3/bin/python3.12')
    require(Path(sys.executable).absolute() == base and sys.flags.no_site == 1,
            'Use the exact base Python with -S; maintained helper subprocesses must not use the vLLM venv')
    os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
    os.environ['PYTHONNOUSERSITE'] = '1'
    os.environ.pop('PYTHONPATH', None)
    os.environ.pop('PYTHONHOME', None)
    for p in Path('/home/admin/wxw403883/anaconda3/lib/python3.12/site-packages').glob('*.pth'):
        text = p.read_text(errors='replace').lower()
        require(not any(word in text for word in ('cuda', 'torch', 'vllm')), 'GPU startup hook in base interpreter: ' + str(p))


def identity_mode(c):
    return c['protocol'].get('identity_mode', 'hostpid_verified')


def config(path=DEFAULT):
    return config_value(load(path))


def config_value(c):
    require((c == BASE and digest(c) == CONFIG_SHA256)
            or (c == EXCLUSIVE and digest(c) == EXCLUSIVE_SHA256),
            'Unknown/stale parameters: only the two checksum-bound M7 protocols are supported')
    require(c['protocol']['seed'] == 1 and c['protocol']['context'] == 128,
            'Protocol revision required for changed seed/context')
    lim = c['limits']
    require(all(type(x) is int and 0 < x <= 3600 for x in lim.values()), 'Invalid limits')
    require(lim['case_seconds'] + 2 * lim['idle_seconds'] + 60 <= 1700, 'Per-case total exceeds 1700s')
    require(lim['startup_seconds'] + lim['request_seconds'] + lim['measurement_seconds']
            + lim['cleanup_seconds'] + 60 <= lim['case_seconds'], 'Insufficient lifecycle budget')
    require(c['resources']['hard_memory_partition'] is False, 'No hard partition is implemented')
    return c


def source_pins():
    paths = [ROOT / p for p in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml')]
    paths += sorted((ROOT / '.cargo').rglob('*.toml'))
    paths += sorted(p for p in (ROOT / 'crates').rglob('*') if p.is_file()
                    and p.suffix in ('.rs', '.cu', '.cuh', '.cpp', '.c', '.h', '.toml', '.json', '.ptx'))
    return {str(p): sha(p) for p in paths}


def workflow_pins():
    paths = sorted(p for p in HERE.iterdir() if p.suffix in ('.py', '.json', '.md'))
    paths += [ROOT / 'scripts' / n for n in
              ('bench_ferrule_vllm.py', 'test_vllm_qwen35.py', '_owned_subreaper.py')]
    paths += [Path(sys.executable).resolve(), Path('/usr/bin/env')]
    return {str(p): sha(p) for p in paths}


def check_pins(pins):
    require(isinstance(pins, dict) and pins, 'Missing pins')
    for name, expected in pins.items():
        require(Path(name).is_absolute() and re.fullmatch(r'[0-9a-f]{64}', expected or '')
                and sha(name) == expected, 'Stale or malformed pin: ' + name)


def validate_owner_lock(lock, uuid, coordinator_pid):
    expected = HERE / (uuid + '.lock')
    require(lock.get('path') == str(expected) and lock.get('coordinator_pid') == coordinator_pid,
            'Worker GPU lock binding differs')
    st = Path(f'/proc/{coordinator_pid}/fd/{lock["fd"]}').stat()
    require((st.st_dev, st.st_ino) == (lock['device'], lock['inode'])
            == (expected.stat().st_dev, expected.stat().st_ino), 'Worker owner lock inode changed')
    info = Path(f'/proc/{coordinator_pid}/fdinfo/{lock["fd"]}').read_text()
    require('FLOCK' in info and 'WRITE' in info, 'Coordinator does not hold exclusive GPU lock')


def inventory(paths):
    rows = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        require(path.is_file() and not path.is_symlink(), 'Inventory path must be a regular file: ' + str(path))
        rows[str(path)] = sha(path)
    return rows


def inventory_paths(c, runtime=False):
    env = build_environment(c)
    toolchain = Path('/root/.rustup/toolchains/nightly-2026-08-01-x86_64-unknown-linux-gnu')
    cuda = Path(env['CUDA_HOME'])
    cutlass = Path(env['FERRULE_CUTLASS_DIR']).resolve()
    regular = lambda root: [p for p in root.rglob('*') if p.is_file() and '.git' not in p.parts]
    result = {
        'compiler': [Path(c['cargo']), Path('/usr/bin/cc'), Path('/usr/bin/c++'), Path('/usr/bin/ld'),
                     Path('/usr/libexec/gcc/x86_64-redhat-linux/10/cc1'),
                     Path('/usr/libexec/gcc/x86_64-redhat-linux/10/cc1plus'),
                     *[p for p in (cuda / 'bin').glob('*') if p.is_file()],
                     *regular(cuda / 'nvvm/bin')],
        'rust_toolchain': [ROOT / 'rust-toolchain.toml', *regular(toolchain / 'bin'), *regular(toolchain / 'lib')],
        'cutlass_tree': regular(cutlass),
        'cuda_headers': [*regular(cuda / 'include'), *regular(cuda / 'nvvm/libdevice'), *regular(cuda / 'nvvm/lib64')],
        'cuda_shared': [cuda / 'lib' / n for n in ('libcudart.so.12', 'libcublas.so.12', 'libcublasLt.so.12')],
        'cpu_python': [Path('/home/admin/wxw403883/anaconda3/bin/python3.12'),
                       *Path('/home/admin/wxw403883/anaconda3/lib/python3.12/site-packages').glob('*.pth')],
    }
    if runtime:
        site = absolute(c['vllm']).parent.parent / 'lib/python3.12/site-packages'
        result['runtime_python_shared'] = [p for p in regular(site) if p.suffix in ('.py', '.pth', '.so')
                                           or '.so.' in p.name or p.name in ('METADATA', 'RECORD')]
        fix = validate_vllm_fix_env(c)
        result['jit_tools'] = [Path(fix['NVCC']), Path(fix['CUDACXX']), Path(fix['FLASHINFER_NVCC'])]
    for name, files in result.items():
        require(files, 'Mandatory inventory is empty: ' + name)
    return result


def required_inventory(c, runtime=False):
    return {name: inventory(paths) for name, paths in inventory_paths(c, runtime).items()}


def validate_inventory(c, recorded, runtime=False):
    required = {k: {str(p.resolve()) for p in paths} for k, paths in inventory_paths(c, runtime).items()}
    require(set(recorded) == set(required) and all(set(recorded[k]) == paths for k, paths in required.items()),
            'Mandatory compiler/CUTLASS/runtime/shared-library inventory missing or changed')
    for pins in recorded.values():
        check_pins(pins)
    return recorded


def model_identity(c):
    root = Path(c['model'])
    index = load(root / 'model.safetensors.index.json')
    metadata = {n: sha(root / n) for n in
                ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json')}
    shards = {}
    for name in sorted(set(index['weight_map'].values())):
        require(Path(name).name == name, 'Unsafe shard path')
        st = (root / name).stat()
        shards[name] = dict(bytes=st.st_size, mtime_ns=st.st_mtime_ns, inode=st.st_ino, device=st.st_dev)
    return dict(path=str(root), metadata_sha256=metadata, shard_stats=shards,
                index_payload_bytes=index['metadata']['total_size'],
                identity_strength='partial: metadata hashes + shard stat pins; payloads NOT hashed',
                full_checkpoint_sha256=None)


def vllm_pins(c):
    executable = absolute(c['vllm'])
    files = [executable, executable.parent / 'python', absolute(c['vllm_fix_environment'])]
    return inventory(files)


def build_environment(c):
    original = load(absolute(c['ferrule_build_environment']))
    keys = ('PATH', 'CUDA_HOME', 'NVCC', 'LD_LIBRARY_PATH', 'FERRULE_CUDA_ARCH', 'FERRULE_CUTLASS_DIR', 'RUSTFLAGS')
    env = {k: original[k] for k in keys}
    env.update(HOME=str(Path.home()), CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
               CUBLAS_WORKSPACE_CONFIG=':0:0', CARGO_BUILD_JOBS='2',
               CARGO_PROFILE_RELEASE_OPT_LEVEL='3', CARGO_PROFILE_RELEASE_LTO='fat',
               CARGO_PROFILE_RELEASE_CODEGEN_UNITS='1', CARGO_PROFILE_RELEASE_DEBUG_ASSERTIONS='false',
               PYTHONDONTWRITEBYTECODE='1', RUSTUP_TOOLCHAIN='nightly-2026-08-01',
               CC='/usr/bin/gcc', CXX='/usr/bin/g++', CUDAHOSTCXX='/usr/bin/g++',
               CARGO_HOME='/root/.cargo', RUSTUP_HOME='/root/.rustup', RUSTUP_AUTO_INSTALL='0',
               RUSTC='/root/.rustup/toolchains/nightly-2026-08-01-x86_64-unknown-linux-gnu/bin/rustc',
               RUSTDOC='/root/.rustup/toolchains/nightly-2026-08-01-x86_64-unknown-linux-gnu/bin/rustdoc')
    require(env['FERRULE_CUDA_ARCH'] == 'sm_86', 'Need explicit SM86 build (no GPU discovery)')
    require('LD_PRELOAD' not in original and 'CUDA_VISIBLE_DEVICES' in original,
            'Build environment inventory is missing explicit hidden-GPU policy')
    return env


def build_argv(c):
    return [c['cargo'], 'build', '--offline', '--locked', '--release', '-p', 'ferrule-cli',
            '--bin', 'ferrule', '--no-default-features', '--features', 'cuda',
            '--message-format=json', '--target-dir', str(absolute(c['build_target']))]


def stage_binary(executable, directory):
    source = Path(executable).resolve()
    require(source.is_file() and not Path(executable).is_symlink(), 'Cargo executable must be a regular file')
    staged_dir = Path(directory) / 'staged'
    staged_dir.mkdir(exist_ok=False)
    staged = staged_dir / 'ferrule'
    shutil.copyfile(source, staged)
    os.chmod(staged, stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
    require(sha(staged) == sha(source) and staged.stat().st_mode & 0o222 == 0, 'Immutable staged executable failed')
    staged_dir.chmod(0o555)
    return str(staged.resolve())


def validate_staged_binary(path, expected):
    p = Path(path).resolve()
    require(p.is_file() and not Path(path).is_symlink() and p.stat().st_mode & 0o222 == 0
            and p.parent.stat().st_mode & 0o222 == 0,
            'Executable is not an immutable staged regular file')
    require(re.fullmatch(r'[0-9a-f]{64}', expected or '') and sha(p) == expected, 'Staged binary hash mismatch')
    return str(p)


@contextlib.contextmanager
def pinned_executable(path, expected):
    validate_staged_binary(path, expected)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(os.dup(fd), 'rb') as stream:
            require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected,
                    'Opened executable descriptor differs from receipt')
        os.lseek(fd, 0, os.SEEK_SET)
        yield fd
    finally:
        os.close(fd)


def release_artifact(rows):
    candidates = [r for r in rows if r.get('reason') == 'compiler-artifact'
                  and r.get('target', {}).get('name') == 'ferrule' and r.get('executable')]
    require(len(candidates) == 1, 'Expected one Cargo ferrule executable artifact')
    row = candidates[0]
    p = row['profile']
    require(p['opt_level'] == '3' and p['debug_assertions'] is False and p['test'] is False
            and 'cuda' in row['features'] and 'cpu-reference' not in row['features'],
            'Need CUDA release opt3 non-test binary without debug assertions/CPU inference')
    return row


def drained(result):
    h, c = result.get('handshake', {}), result.get('custody', {})
    return (h.get('containment_verified') is True and c.get('containment_verified') is True
            and c.get('echild') is True and c.get('status') == 'proven_drained'
            and result.get('descendant_custody') == 'proven_drained'
            and bool(h.get('launch_id')) and h.get('launch_id') == c.get('launch_id')
            and bool(h.get('helper_identity')) and h.get('helper_identity') == c.get('helper_identity')
            and result.get('cleanup_errors') == [] and not result.get('error')
            and not result.get('reporting_errors'))


def successful_process(result):
    return result.get('returncode') == 0 and result.get('timed_out') is False and drained(result)


def validate_build(receipt, c):
    require(receipt.get('schema') == SCHEMA + 'build.v1' and receipt.get('status') == 'pass', 'Missing passing build receipt')
    require(receipt.get('argv') == build_argv(c) and receipt.get('environment') == build_environment(c),
            'Build argv/environment not the pinned CUDA12.4 opt3/fat-LTO build')
    require(successful_process(receipt.get('supervisor', {})), 'Build custody/failure gate')
    require(receipt.get('sources_before') == receipt.get('sources_after') == source_pins(), 'Build sources changed')
    # Paper orchestration is not linked into the Rust executable. Preserve its
    # historical receipt hashes, but bind current orchestration in the new
    # prepared manifest rather than forcing a numerical rebuild after doc/tool edits.
    historical = receipt['pins']
    required = {p: h for p, h in workflow_pins().items() if not Path(p).is_relative_to(HERE)}
    require(all(historical.get(p) == h for p, h in required.items()), 'Missing mandatory build tool/helper pins')
    for name in ('runner.py', 'contract.py', 'm7.json'):
        require(re.fullmatch(r'[0-9a-f]{64}', historical.get(str(HERE / name), '')),
                'Missing historical build orchestration pin: ' + name)
    check_pins(build_runtime_pins(receipt))
    validate_inventory(c, receipt['inventories'])
    release_artifact([receipt['artifact']])
    return validate_staged_binary(receipt['staged_binary'], receipt['binary_sha256'])


def build_runtime_pins(receipt):
    return {p: h for p, h in receipt['pins'].items() if not Path(p).is_relative_to(HERE)}


def flags_argv(flags):
    result = []
    for flag, value in flags.items():
        result.append(flag)
        if value is not True:
            result.extend(value if isinstance(value, list) else [value])
    return result


def validate_vllm_fix_env(c, selected_uuid=None):
    fix = load(absolute(c['vllm_fix_environment']))
    require(set(fix) == VLLM_FIX_KEYS, 'vLLM fix-env must contain exactly the reviewed JIT allowlist')
    require(not set(fix) & VLLM_FORBIDDEN_KEYS, 'vLLM fix-env contains forbidden visibility/preload/override key')
    for key, value in fix.items():
        require(isinstance(value, str) and '\\0' not in value, 'Invalid vLLM environment value')
    if selected_uuid is not None:
        require(fix.get('CUDA_VISIBLE_DEVICES') is None and fix.get('NVIDIA_VISIBLE_DEVICES') is None,
                'vLLM fix-env must not override selected GPU')
    return fix


def server_spec(c, engine, binary, uuid):
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', uuid) is not None, 'Need literal physical GPU UUID')
    env = dict(HOME=str(Path.home()), PATH='/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES=uuid, NVIDIA_VISIBLE_DEVICES=uuid, CUDA_DEVICE_ORDER='PCI_BUS_ID',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if engine == 'ferrule':
        build = build_environment(c)
        env.update({k: build[k] for k in ('CUDA_HOME', 'NVCC', 'LD_LIBRARY_PATH', 'PATH')})
        env.update(CUBLAS_WORKSPACE_CONFIG=':0:0', FERRULE_LOG='info', FERRULE_LOG_FORMAT='json')
        executable = binary
    else:
        require(engine == 'vllm', 'Unknown engine')
        env.update(validate_vllm_fix_env(c, uuid))
        # env -i: no inherited CUDA12.4 library path, preload, Marlin input override,
        # offload-disable variable, or Ferrule cuBLAS workspace policy.
        require(env.get('CUDA_VISIBLE_DEVICES') == uuid and env.get('NVIDIA_VISIBLE_DEVICES') == uuid,
                'Selected UUID must be applied after vLLM fix-env merge')
        require('LD_LIBRARY_PATH' not in env and 'CUBLAS_WORKSPACE_CONFIG' not in env
                and 'LD_PRELOAD' not in env, 'Polluted vLLM environment')
        executable = str(absolute(c['vllm']))
    flags = c[engine + '_flags']
    require(env.get('CUDA_VISIBLE_DEVICES') == uuid and env.get('NVIDIA_VISIBLE_DEVICES') == uuid,
            'Final selected UUID environment mismatch')
    return dict(engine=engine, environment=env,
                argv=[executable, 'serve', c['model'], *flags_argv(flags)],
                model=flags['--served-model-name'],
                url='http://127.0.0.1:' + flags['--port'] + '/v1/completions')


def isolated_argv(argv, environment):
    return ['/usr/bin/env', '-i', *[k + '=' + v for k, v in sorted(environment.items())], *argv]


def validate_schema(value):
    # Existing CPU validator only; expose packages without running .pth/site hooks.
    site = absolute(BASE['vllm']).parent.parent / 'lib/python3.12/site-packages'
    if str(site) not in sys.path:
        sys.path.append(str(site))
    from jsonschema import Draft202012Validator
    errors = sorted(Draft202012Validator(load(HERE / 'schema.json')).iter_errors(value), key=lambda e: str(e.path))
    require(not errors, 'Artifact schema invalid: ' + (errors[0].message[:500] if errors else ''))
    if value.get('schema') in (SCHEMA + 'prepared.v1', SCHEMA + 'plan.v1'):
        config_value(value['config'])
    if value.get('schema') in (SCHEMA + 'execution.v1', SCHEMA + 'execution.v2') and value.get('status') == 'complete':
        require(len(value['cases']) == 2 and {c['engine'] for c in value['cases']} == {'ferrule', 'vllm'},
                'Complete execution requires exactly one case per engine')
        require(all(c.get('eligible_observation') is True and successful_process(c.get('supervisor', {}))
                    and c.get('after', {}).get('status') == 'verified' for c in value['cases']),
                'Complete execution contains a failed/unknown case')


def quality_receipt(path, manifest):
    q = load(path)
    validate_schema(q)
    require(set(q) == {'schema', 'status', 'binary_sha256', 'model_identity_sha256',
                       'protocol_sha256', 'checks', 'independent_reference', 'artifact_pins'},
            'Unknown/incomplete release-quality schema')
    require(q.get('schema') == SCHEMA + 'release-quality.v1' and q.get('status') == 'pass', 'Release basic quality is not pass')
    for key in ('binary_sha256', 'model_identity_sha256', 'protocol_sha256'):
        require(q.get(key) == manifest[key], 'Release quality mismatch: ' + key)
    require(q.get('checks') == dict(raw_hello_greedy=True, reset_repeat=True, cold_repeat=True), 'Incomplete basic quality coverage')
    require(q.get('independent_reference') is True, 'Need independent reference, not cross-engine agreement alone')
    check_pins(q['artifact_pins'])
    require(len(q['artifact_pins']) >= 2, 'Pin reference and actual release-quality result artifacts')
    return dict(path=str(Path(path).resolve()), sha256=sha(path), evidence=q)


def http_success(report, workload):
    runs = report.get('runs', [])
    rows = [r for run in runs for r in run.get('requests', [])]
    valid = lambda r: (r.get('status') == 'success' and r.get('actual_input_tokens') == 1
                       and type(r.get('actual_output_tokens')) is int
                       and 0 < r['actual_output_tokens'] <= workload['max_tokens']
                       and r.get('finish_reason') in ('stop', 'length'))
    return (report.get('status') == 'complete' and len(runs) == 1
            and len(rows) == workload['requests'] and all(valid(r) for r in rows)
            and len(report.get('warmups', [])) == workload['warmup']
            and all(valid(r) for r in report.get('warmups', []))
            and report.get('measurements', {}).get('all_requests_successful') is True
            and all(run.get('not_issued_indices') == [] for run in runs))


def resource_evidence(samples, errors, resources):
    valid = bool(samples) and not errors and all(
        type(s.get('gpu_bytes')) is int and type(s.get('pss_bytes')) is int
        and s['gpu_bytes'] >= 0 and s['pss_bytes'] > 0 for s in samples)
    peak = lambda key: max((s[key] for s in samples if type(s.get(key)) is int), default=None)
    gpu, host = peak('gpu_bytes'), peak('pss_bytes')
    return dict(status=('observed_within_targets' if valid and 0 < gpu <= resources['gpu_target_bytes']
                        and host <= resources['host_target_bytes'] else 'blocked'),
                sampled_device_high_water_bytes=gpu, sampled_tree_pss_high_water_bytes=host,
                sampled_tree_rss_high_water_bytes=peak('rss_bytes'), samples=len(samples), errors=errors,
                sample_interval_seconds=resources['sample_interval_seconds'], exact_peaks=False,
                hard_memory_partition=False, pinned_bytes=None,
                caveats=['CUDA pinning not measured by VmLck/VmPin alone',
                         'PSS excludes unmapped OS file cache; not a total physical-host-memory proof',
                         'Device samples include baseline; between-sample peaks may be missed'])


def resource_infeasible(observation):
    errors = observation.get('resources', {}).get('errors', [])
    return any(error in ('Sampled GPU target exceeded', 'Sampled host PSS target exceeded') for error in errors)


def runtime_evidence(engine, log):
    patterns = ({'moe_marlin': r'Using MARLIN Fp8 MoE backend',
                 'linear_marlin': r'MarlinFP8ScaledMMLinearKernel',
                 'offloader_uva': r'Offloader set to UVAOffloader',
                 'offloaded_amount': r'Total CPU offloaded parameters:.*',
                 'model_bf16': r'dtype=torch.bfloat16'} if engine == 'vllm' else
                {'mixed_profile': r'cuda-hybrid-numeric-fp8-model-bf16-mixed-f32-ssm'})
    found = {k: re.findall(p, log)[:4] for k, p in patterns.items()}
    contradictions = re.findall(r'[^\n]*(?:(?:falling back (?!to the Triton GDN decode path))|W8A8|Offloader set to PrefetchOffloader|functional_call-based)[^\n]*', log, re.I)[:10]
    return dict(status='blocked', evidence=found, contradictions=contradictions,
                fallback_observations=re.findall(r'[^\n]*falling back[^\n]*', log, re.I)[:10],
                missing=[k for k, v in found.items() if not v],
                reasons=['Effective KV/conv/SSM dtype and physical pinning need runtime evidence',
                         'UVA selection log alone does not prove UVA path rather than functional_call fallback'],
                exact_arithmetic_match=False)
