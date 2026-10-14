"""Use the existing authenticated bounded supervisor; never runs without grant flags."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('label')
parser.add_argument('--execute-cuda', action='store_true')
parser.add_argument('--sole-owner-permission', action='store_true')
args = parser.parse_args()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'numeric_bf16_gemv'))
from contract_run import run
run(args.directory.resolve(), args.label, args.execute_cuda, args.sole_owner_permission)

# Expected synthetic stages do not exist at CPU readiness. Pin them only after
# the authenticated independent oracle has succeeded; retain the original plan.
if args.label == 'oracle':
    directory = args.directory.resolve()
    plan_path = directory / 'PREPARED.json'
    initial = plan_path.read_bytes()
    archive = directory / 'PREPARED-BEFORE-ORACLE.json'
    with archive.open('xb') as f:
        f.write(initial)
    plan = json.loads(initial)
    generated_path = directory / 'fixture' / 'GENERATED-SHA256.json'
    generated = json.loads(generated_path.read_text())
    for path, expected in generated.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Generated fixture changed: ' + path)
        if path in plan['pins'] and plan['pins'][path] != expected:
            raise ValueError('Original fixture changed: ' + path)
    plan['pins'].update(generated)
    plan['pins'][str(generated_path)] = hashlib.sha256(generated_path.read_bytes()).hexdigest()
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
