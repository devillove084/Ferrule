"""Stdlib, fail-closed manifest checks for the bounded routed-down replay."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / 'runs/routed-down-m1-cpu-01'
BUNDLE_PIN = '6fe6b0c5b8d886586fef422be52f61151fc72345b4077eae33987e8fa325c65f'
REFERENCE_PIN = 'fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d'
TARGETS = [('hello', 0, 21), ('hello', 3, 37), ('capital', 0, 72), ('capital', 3, 10)]
FILES = ['manifest.json'] + [f'rank{r}.{s}' for r in (0, 1) for s in ('input.bf16', 'weight.fp8', 'scale.bf16', 'hf.partial.bf16')]


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_sums(directory, sums_name, pin, expected=None, max_bytes=192*1024**2):
    require(sha(directory / sums_name) == pin, 'Bundle checksum pin differs')
    seen = set(); total = 0
    for line in (directory / sums_name).read_text().splitlines():
        digest, name = line.split('  ', 1)
        p = (directory / name).resolve()
        require(p.is_relative_to(directory.resolve()) and name not in seen and p.is_file(), 'Invalid/duplicate artifact')
        total += p.stat().st_size
        require(total <= max_bytes, 'Artifact byte bound')
        require(sha(p) == digest, 'Artifact differs: ' + name)
        seen.add(name)
    if expected is not None:
        require(seen == set(expected), 'Incomplete fixture files')
    return total


def bundle():
    check_sums(BUNDLE, 'SHA256SUMS.cpu-extract', BUNDLE_PIN)
    m = json.loads((BUNDLE / 'manifest.json').read_text())
    require(m['reference_manifest_sha256'] == REFERENCE_PIN and not m['model_fixture_ready'], 'Unexpected preparation')
    require([(f['case'],f['layer'],f['expert']) for f in m['fixtures']] == TARGETS, 'Different bounded selection')
    require(sha(HERE/'runs/full40-20261009-03/manifest.json') == REFERENCE_PIN, 'Old reference changed')
    for f in m['fixtures']:
        require((f['n'],f['local_k'],f['stage']) == (2048,256,'decode.0'), 'Wrong M1 geometry')
    return m


def bounds(p):
    require(p['task'] == 'routed-down-m1' and p['timeout'] == 120 and p['capture_mib'] == 16 and p['cache_mib'] == 64, 'Task/resource bound differs')
    require(p['bundle_sha256'] == BUNDLE_PIN and p['reference_sha256'] == REFERENCE_PIN, 'Input pins differ')
    out = Path(p['directory']).resolve()
    require(out.is_relative_to(HERE/'runs') and not out.is_relative_to(BUNDLE), 'Fresh sibling paper output required')
