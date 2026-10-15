"""Standard-library-only portable archive primitives; never imports the runner."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
RUN = 'artifacts-v3-mapfixed-20261015-01'
PLAN = 'plan-mapfixed-20261015-01'
PINNED = {
    RUN + '/paired-final.json': '33459c536d5a736bb4843e80e2cf3e79016fae054e5e5664cfe859cfb089e357',
    PLAN + '/poststudy-audit.json': '30d123fac5be3df3eae9a0241f75897ea24a2d30cec254c41456baed85dd17c5',
    RUN + '/final-cleanup.json': '0f1f42d854af3e27eba09a1eb2df0dbd7ae19860015de589910b3ae556254563',
}
CLI_SHA256 = '7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01'
SOURCE_COMMIT = '86c58a017d7c4b53490f536a3b1a133f2f7848cc'
SOURCE_ARCHIVE_SHA256 = 'e0d160996147820fcc64cfb29070b265b2297a6d9265a21de541ed77e8f2fa65'
MAX_BYTES = 20 * 1024 * 1024
SECRET_KEY = re.compile(r'(?:worker.?token$|(?:^|_)(?:nonce|launch_id|password|passwd|secret|api_key|authorization|cookie|access_token|refresh_token|private_key)(?:$|_))', re.I)
PRIVATE_KEYS = {'environment', 'env', 'environ', 'raw_text', 'raw_lines', 'mountpoint', 'mount_namespace', 'boot_id', 'hostname', 'handshake'}
SECRET_ASSIGNMENT = re.compile(r'''(?:worker.?token|nonce|launch_id|password|api_key|authorization|access_token)["']?\s*[:=]''', re.I)
PRIVATE_TEXT = re.compile(r'(?:/(?:home|mnt|root|Users)/|https?://|(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}\b|\bGPU-[0-9a-f-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+[A-Za-z0-9])', re.I)


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(Invalid('nonfinite JSON')))


def safe_data(value):
    """Scan keys recursively, including JSON objects embedded in string fields."""
    if isinstance(value, dict):
        for key, child in value.items():
            require(not SECRET_KEY.search(key) and key.lower() not in PRIVATE_KEYS,
                    'private field rejected (value not printed)')
            require(not PRIVATE_TEXT.search(key), 'private identity in key')
            safe_data(child)
    elif isinstance(value, list):
        for child in value:
            safe_data(child)
    elif isinstance(value, str):
        require(not PRIVATE_TEXT.search(value) and not SECRET_ASSIGNMENT.search(value), 'private location/credential marker in string')
        if value.lstrip().startswith(('{', '[')):
            try:
                nested = decode(value)
            except (ValueError, TypeError):
                return
            safe_data(nested)


def regular(root, name):
    path = Path(name)
    require(not path.is_absolute() and path.parts and '..' not in path.parts,
            'unsafe archive member')
    result = root / path
    require(result.is_file() and not result.is_symlink(), 'missing/nonregular archive member: ' + name)
    require(all(not parent.is_symlink() for parent in result.parents if parent != root.parent),
            'symlink archive parent')
    return result


def unpack(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def load(root, name):
    return decode(unpack(regular(root, name)))


def pick(value, keys):
    return {key: value[key] for key in keys.split()}


def write(root, name, value, *, raw=False):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if raw else encode(value)
    if name.endswith('.gz'):
        data = gzip.compress(data, compresslevel=9, mtime=0)
    if path.exists():
        require(path.read_bytes() == data, 'existing publication member differs; never overwrite: ' + name)
        return
    with path.open('xb') as stream:
        stream.write(data)


def seal(root):
    """Only the new bundle is mutable. Manifest is deliberately not self-hashed."""
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        if name == 'manifest.json':
            continue
        regular(root, name)
        data = path.read_bytes()
        plain = unpack(path)
        plain.decode('utf-8')
        files[name] = dict(bytes=len(data), sha256=sha(data),
                           uncompressed_bytes=len(plain), uncompressed_sha256=sha(plain),
                           encoding='gzip' if name.endswith('.gz') else 'utf-8')
    manifest = dict(schema='ferrule.paired-http-publication-manifest.v1',
                    files=files, source_pins=PINNED, byte_budget_exclusive=MAX_BYTES,
                    hash_scope='Member hashes authenticate this public copy, not omitted private originals',
                    original_input_inventory='provenance.json.gz',
                    redaction_rules='REDACTION.md', scientific_acceptance=False)
    (root / 'manifest.json').write_bytes(encode(manifest))


def verify_members(root):
    manifest = load(root, 'manifest.json')
    require(manifest['source_pins'] == PINNED, 'immutable source pins changed')
    expected = set(manifest['files']) | {'manifest.json'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if not p.is_dir()}
    require(actual == expected, 'missing or unlisted bundle files')
    total = 0
    expanded = 0
    for name, entry in manifest['files'].items():
        path = regular(root, name)
        data, plain = path.read_bytes(), unpack(path)
        require(len(data) == entry['bytes'] and sha(data) == entry['sha256'], 'stale copy hash: ' + name)
        require(len(plain) == entry['uncompressed_bytes'] and sha(plain) == entry['uncompressed_sha256'],
                'stale uncompressed hash: ' + name)
        text = plain.decode('utf-8')
        if name.endswith(('.json', '.json.gz')):
            safe_data(decode(text))
        elif name.endswith('.jsonl.gz'):
            for line in text.splitlines():
                safe_data(decode(line))
        require(not data.startswith((b'\x7fELF', b'MZ', b'!<arch>')), 'binary payload prohibited')
        total += len(data)
        expanded += len(plain)
    total += (root / 'manifest.json').stat().st_size
    require(total < MAX_BYTES, 'bundle exceeds 20 MiB budget')
    return dict(files=len(expected), bytes=total, uncompressed_member_bytes=expanded)
