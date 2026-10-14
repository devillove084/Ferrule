"""CPU-only output-location adapter around the frozen TP2 numerical comparators.

No oracle/protocol/tensor/tolerance changes. The legacy bridge's output root is
scoped to this experiment; tp2_oracle.HERE (and its immutable protocol) is not.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORACLE = HERE.parent / 'tp2_mixed/oracle'


def output_paths(raw, reference, canonical, report):
    raw, reference, canonical, report = (p.resolve() for p in (raw, reference, canonical, report))
    for path in (canonical, report):
        if (not path.is_relative_to(HERE / 'runs') or path.is_relative_to(raw)
                or path.is_relative_to(reference) or path.exists()):
            raise ValueError('Fresh output must be experiment-local and outside immutable inputs')
    if report.is_relative_to(canonical) or canonical.is_relative_to(report):
        raise ValueError('Report and canonical bundle must be separate')
    return raw, reference, canonical, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('input', 'reference', 'canonical', 'output'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--reference-sha256', required=True)
    ap.add_argument('--lifecycle', action='store_true')
    a = ap.parse_args()
    raw, reference, canonical, output = output_paths(a.input, a.reference, a.canonical, a.output)
    sys.path.insert(0, str(ORACLE))
    import bridge as b
    import compare as c
    import tp2_oracle as o
    o.initialize()
    ref = c.load_manifest(reference, a.reference_sha256, True, lifecycle=a.lifecycle)
    if a.lifecycle:
        spec = importlib.util.spec_from_file_location('decoded_cache_lifecycle_bridge',
            ROOT / 'crates/ferrule-composition/tests/tp2_mixed_execution/lifecycle_bridge.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = module.snapshot(raw)
        module.convert(raw, canonical, ref, a.reference_sha256, before['sha256'])
        actual_pin = o.sha256(canonical / 'manifest.json')
        actual = c.load_manifest(canonical, actual_pin, False, lifecycle=True)
        for key in ('config_sha256', 'index_sha256', 'full_shard_set_sha256', 'effective_text_config'):
            o.require(ref['checkpoint'][key] == actual['checkpoint'][key], 'Lifecycle checkpoint differs')
        config = o.native.hf.Qwen3_5MoeTextConfig(**ref['checkpoint']['effective_text_config'])
        o.require(config.num_hidden_layers == 40 and config.vocab_size == 248320 and config.num_experts == 256,
                  'Expected full40 lifecycle reference')
        import lifecycle_compare as lc
        result = lc.compare_suites(reference, ref['suite'], canonical, actual['suite'], config)
        result.update(reference_manifest_sha256=a.reference_sha256, actual_manifest_sha256=actual_pin)
        o.require(module.snapshot(raw) == before, 'Raw lifecycle evidence changed')
    else:
        before = b.snapshot(raw)
        old_root = b.HERE
        try:
            b.HERE = HERE
            b.convert_raw(raw, canonical, ref, a.reference_sha256, before['sha256'])
        finally:
            b.HERE = old_root
        result = c.compare(reference, a.reference_sha256, canonical, o.sha256(canonical / 'manifest.json'))
        o.require(b.snapshot(raw) == before, 'Raw evidence changed')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(raw_sha256=before['sha256'], passed=result['passed'])))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
