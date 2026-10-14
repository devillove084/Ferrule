"""Review-scoped 25-option audit against installed source, never vendor imports."""
import ast
import math
import re

import contract as ct

ENUMS = {
    '--dtype': ('config/model.py', 'ModelDType'),
    '--kv-cache-dtype': ('config/cache.py', 'CacheDType'),
    '--mamba-cache-dtype': ('config/cache.py', 'MambaDType'),
    '--mamba-ssm-cache-dtype': ('config/cache.py', 'MambaDType'),
    '--offload-backend': ('config/offload.py', 'OffloadBackend'),
    '--linear-backend': ('config/kernel.py', 'LinearBackend'),
    '--moe-backend': ('config/kernel.py', 'MoEBackend'),
}


def literal_choices(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            ct.require(isinstance(node.value, ast.Subscript) and ast.unparse(node.value.value) == 'Literal',
                       'Installed enum is no longer a Literal: ' + name)
            nodes = node.value.slice.elts if isinstance(node.value.slice, ast.Tuple) else [node.value.slice]
            return [ast.literal_eval(n) for n in nodes]
    raise ValueError('Installed enum declaration missing: ' + name)


def audit(c):
    root = ct.absolute(c['vllm']).parent.parent / 'lib/python3.12/site-packages/vllm'
    paths = {'engine/arg_utils.py', 'entrypoints/openai/cli_args.py', *[p for p, _ in ENUMS.values()]}
    trees = {p: ast.parse((root / p).read_text()) for p in paths}
    fields = {}
    for path in ('engine/arg_utils.py', 'entrypoints/openai/cli_args.py'):
        fields.update({n.target.id: ast.unparse(n.annotation) for n in ast.walk(trees[path])
                       if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)})
    declarations = {n.value for n in ast.walk(trees['engine/arg_utils.py']) if isinstance(n, ast.Constant)
                    and isinstance(n.value, str) and n.value.startswith('--')}
    flags = c['vllm_flags']
    ct.require(set(flags) == set(ct.VLLM_EXPECTED_FLAGS), 'Only the reviewed 25 installed CLI options are accepted')
    evidence = {}
    for flag, value in flags.items():
        kind = ct.VLLM_EXPECTED_FLAGS[flag]
        positive = '--' + flag[5:] if flag.startswith('--no-') else flag
        field = positive[2:].replace('-', '_')
        annotation = fields.get(field, '')
        ct.require(positive in declarations or flag in ('--host', '--port'), 'Missing CLI declaration: ' + flag)
        if flag.startswith('--no-'):
            ct.require(kind == 'bool' and 'bool' in annotation, 'Negation requires a declared bool option')
        if kind == 'bool':
            ct.require('bool' in annotation and value is True, 'Boolean switch must be literal true: ' + flag)
        elif kind == 'int':
            ct.require('int' in annotation and isinstance(value, str) and re.fullmatch(r'[0-9]+', value), 'Invalid integer: ' + flag)
            ct.require(int(value) >= (0 if flag == '--seed' else 1), 'Integer below range: ' + flag)
            if flag == '--port':
                ct.require(int(value) <= 65535, 'Port out of range')
        elif kind == 'float':
            ct.require(annotation == 'float' and isinstance(value, str), 'Invalid float type: ' + flag)
            try:
                number = float(value)
            except ValueError as error:
                raise ValueError('Invalid float: ' + flag) from error
            ct.require(math.isfinite(number) and number >= 0, 'Invalid float value: ' + flag)
            if flag == '--gpu-memory-utilization':
                ct.require(0 < number <= 1, 'GPU utilization outside (0,1]')
        elif kind == 'list':
            ct.require(annotation == 'set[str]' and isinstance(value, list) and value
                       and all(isinstance(x, str) and re.fullmatch(r'[a-zA-Z0-9_.]+', x) for x in value)
                       and len(set(value)) == len(value), 'Invalid offload parameter collection')
        elif kind == 'enum':
            path, alias = ENUMS[flag]
            choices = literal_choices(trees[path], alias)
            ct.require(isinstance(value, str) and value in choices, 'Invalid installed enum: ' + flag)
        else:
            ct.require('str' in annotation and isinstance(value, str) and value and '\0' not in value,
                       'Invalid string option: ' + flag)
        evidence[flag] = dict(annotation=annotation, value=value)
    ferrule = (ct.ROOT / 'crates/ferrule-cli/src/args.rs').read_text()
    for flag in c['ferrule_flags']:
        ct.require(flag in ferrule or flag[2:].replace('-', '_') in ferrule, 'Ferrule option removed: ' + flag)
    return dict(status='pass', method='25-option typed/value audit; static, not runtime/device validation',
                option_evidence=evidence, source_pins={str(root / p): ct.sha(root / p) for p in sorted(paths)})
