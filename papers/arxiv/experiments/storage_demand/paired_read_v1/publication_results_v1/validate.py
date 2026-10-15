#!/usr/bin/env python3
"""Owner-side release validation. Updates only this bundle's validation receipts."""
from __future__ import annotations

import ast
import re
import subprocess
import sys

from common import HERE, encode, require, seal, sha
from check import verify
from audit_sources import audit


def main():
    input_manifest = sha((HERE / 'manifest.json').read_bytes())
    result, inventory = verify(HERE)
    code = {}
    for path in sorted(HERE.glob('*.py')):
        content = path.read_bytes()
        ast.parse(content, filename=path.name)
        code[path.name] = sha(content)
    completed = subprocess.run([sys.executable, '-S', '-B', str(HERE / 'test_publication.py')],
                               cwd=HERE, capture_output=True, text=True, timeout=120)
    log = completed.stdout + completed.stderr
    require(completed.returncode == 0, 'unit tests failed; no success receipt written\n' + log)
    count = re.search(r'^Ran (\d+) tests in ', log, re.M)
    require(count is not None and '\nOK\n' in log, 'unit test completion absent')
    private = audit()
    receipt = dict(schema='ferrule.paired-http-publication-validation.v1', status='passed',
                   python_version=sys.version.split()[0], stdlib_only=True,
                   check='check.verify: membership, original pointers, full reconstruction, all bootstrap draws',
                   tests=dict(command='python -S -B test_publication.py', returncode=completed.returncode,
                              tests_run=int(count.group(1)), isolated_copy_check=True,
                              negative_cases='See individual test names in validation-tests.txt'),
                   code_sha256=code, tested_input_manifest_sha256=input_manifest,
                   tested_input_inventory=inventory, raw_hash_and_private_scan=private,
                   coverage=result['coverage'], bootstrap_draws=result['bootstrap_draw_count'],
                   gpu_workloads=0, network_requests=0, original_sources_modified=False,
                   receipt_scope='Input manifest/code checked before writing these two receipts; final membership resealed and verified afterward')
    (HERE / 'validation-tests.txt').write_text(log)
    (HERE / 'validation.json').write_bytes(encode(receipt))
    seal(HERE)
    _, final_inventory = verify(HERE)
    final_scan = audit()
    print(encode(dict(status='passed', python_version=receipt['python_version'], tests=receipt['tests']['tests_run'],
                      final_inventory=final_inventory, final_private_scan=final_scan)).decode(), end='')


if __name__ == '__main__':
    main()
