"""Execution-role pins for the preserved CLI86 and the live pressure controller.

This is an input-binding revision, not a new protocol, build, or quality gate.
The private seal is exclusive-create; neither predecessor seals nor raw evidence
are recaptured. Compiled-source relations never consume the current checkout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

from evidence import require

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
ROLE_PATH = HERE / 'binding-roles-v3.json'
BINDING_PATH = HERE / 'bindings-v3-private-02.json'
SCHEMA = 'ferrule.storage-demand.pressure-fault-inputs.v3'
ROLE_SCHEMA = 'ferrule.storage-demand.pressure-input-roles.v1'
ROLES = frozenset(('compiled_source', 'host_control', 'source_model', 'references', 'protocol'))


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def safe_relative(name):
    path = Path(name)
    require(isinstance(name, str) and name and not path.is_absolute()
            and '..' not in path.parts and str(path) == name, 'unsafe binding path')
    require('profile_overlap_v1' not in path.parts, 'active profiler source cannot be bound')
    return path


def role_root():
    value = load(ROLE_PATH)
    require(value.get('schema') == ROLE_SCHEMA and set(value['roles'].values()) <= ROLES,
            'unknown binding role/schema')
    for name in value['roles']:
        safe_relative(name)
    for name, pin in value['predecessor_pins'].items():
        require(sha(ROOT / safe_relative(name)) == pin, 'predecessor evidence changed: ' + name)
    return value


def source_context(spec):
    """Reuse the exact archive verifier, including the three preserved pyc pins."""
    sys.path.insert(0, str(HERE.parent))
    from paired_read_v1 import current_binding_v3 as qualified
    from paired_read_v1.final_adapter_v3 import evidence, binding
    from runnerhelpers.build_receipts_v1 import immutable

    identity = spec['candidate_identity']
    require(identity == qualified.candidate_identity(qualified.QUALIFICATION), 'candidate identity changed')
    for name, pin in spec['producer_pins'].items():
        path = ROOT / safe_relative(name)
        immutable(path)
        require(sha(path) == pin, 'candidate source producer changed: ' + name)
    inventory, source = evidence.source_inventory(identity, evidence.load('evidence-manifest.json'))
    disposition = binding.disposition(inventory, evidence.load('evidence-manifest.json'))
    artifact = evidence.load('artifact-final-cli.json')
    require(artifact['sha256'] == identity['cli_binary_sha256']
            and artifact['source_digest'] == identity['source_digest']
            and source['commit'] == identity['commit'], 'compiled artifact/source producer relation changed')
    immutable(artifact['executable'])
    require(sha(artifact['executable']) == artifact['sha256'], 'compiled CLI bytes changed')
    return source, disposition


def validate_entries(files, roles, root, source):
    """Resolve what is actually consumed; do not treat live build inputs as runtime."""
    require(set(files) == set(roles) and set(roles.values()) <= ROLES, 'unknown/missing input role')
    pins, relations = {}, []
    members = source['pins_relative_to_frozen_source']
    for name, pin in files.items():
        path = root / safe_relative(name)
        role = roles[name]
        require(isinstance(pin, str) and re.fullmatch('[0-9a-f]{64}', pin), 'invalid input pin: ' + name)
        if role == 'compiled_source':
            require(name.startswith('crates/') and path.suffix == '.rs',
                    'host/control input cannot be compiled_source: ' + name)
            require(name in members and pin == members[name], 'compiled member/pin not in candidate archive: ' + name)
            consumed = Path(source['frozen_source']) / name
            require(consumed.is_file() and not consumed.is_symlink()
                    and consumed.stat().st_mode & 0o222 == 0, 'mutable compiled member: ' + name)
            require(sha(consumed) == pin, 'changed compiled member: ' + name)
            live = sha(path) if path.is_file() else None
            relations.append(dict(path=name, role=role, expected_sha256=pin, consumed_path=str(consumed),
                current_live_sha256=live, live_matches_compiled=live == pin,
                live_checkout_consumed=False, relation='candidate archive member, not live compilation'))
        else:
            consumed = path
            require(sha(consumed) == pin, 'changed ' + role + ' input: ' + name)
        pins[str(consumed)] = pin
        if role == 'source_model':
            validate_model_identity(load(consumed))
    return pins, relations


def validate_model_identity(rows):
    require(isinstance(rows, dict) and len(rows) == 14, 'exact fourteen source model identities required')
    for name, expected in rows.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'source model physical path changed')
        stat = path.stat()
        actual = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino]
        require(actual == expected, 'source model physical identity changed: ' + name)
    # Identity stability is not a full shard rehash or an OS read-only guarantee.


def execution_pins(value):
    source = load(ROOT / value['source_freeze'])
    return {str(Path(source['frozen_source']) / name if value['roles'][name] == 'compiled_source'
                else ROOT / name): pin for name, pin in value['files'].items()}


def check_bindings(path=BINDING_PATH):
    value = load(path)
    require(value.get('schema') == SCHEMA and value.get('digest') == digest(
        {k: v for k, v in value.items() if k != 'digest'}), 'role binding schema/digest changed')
    spec = role_root()
    require(value['role_root_sha256'] == sha(ROLE_PATH) and value['roles'] == spec['roles']
            and value['source_freeze'] == spec['source_freeze'], 'binding role root changed')
    # Check all live inputs before executing the shared archive-review helpers.
    source = load(ROOT / spec['source_freeze'])
    validate_entries(value['files'], value['roles'], ROOT, source)
    source, _ = source_context(spec)
    old = load(ROOT / spec['predecessor_binding'])
    require(value['candidate'] == old['candidate'] and value['workflow_tools'] == old['workflow_tools']
            and value['budget_variant'] == old['budget_variant'], 'candidate/tools/protocol drift')
    require(set(old['files']) <= set(value['files']), 'predecessor input omitted')
    for name, pin in old['files'].items():
        expected = source['pins_relative_to_frozen_source'][name] if value['roles'][name] == 'compiled_source' else pin
        if name not in spec['revised_control_sources']:
            require(value['files'][name] == expected, 'unapproved input pin revision: ' + name)
    return value


def check_source_model():
    spec = role_root()
    value = load(BINDING_PATH)
    for name, role in spec['roles'].items():
        if role == 'source_model':
            require(sha(ROOT / name) == value['files'][name], 'source model identity receipt changed')
            validate_model_identity(load(ROOT / name))


def create_binding():
    """Explicit CPU preparation only; no launch, GPU query, or old-seal rewrite."""
    spec = role_root()
    source, _ = source_context(spec)
    old = load(ROOT / spec['predecessor_binding'])
    files = {name: source['pins_relative_to_frozen_source'][name] if role == 'compiled_source'
             else sha(ROOT / name) for name, role in spec['roles'].items()}
    value = dict(schema=SCHEMA, role_root_sha256=sha(ROLE_PATH), roles=spec['roles'], files=files,
        source_freeze=spec['source_freeze'], candidate=old['candidate'], workflow_tools=old['workflow_tools'],
        budget_variant=old['budget_variant'], predecessor_sha256=sha(ROOT / spec['predecessor_binding']),
        historical_resume_permitted=False, gpu_authorized=False)
    value['digest'] = digest(value)
    with BINDING_PATH.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    BINDING_PATH.chmod(0o400)
    check_bindings()
    return dict(path=str(BINDING_PATH), sha256=sha(BINDING_PATH), gpu_queries=0, gpu_runs=0)


if __name__ == '__main__':
    require(sys.argv[1:] == ['prepare'], 'only explicit CPU prepare is supported')
    print(json.dumps(create_binding(), indent=2))
