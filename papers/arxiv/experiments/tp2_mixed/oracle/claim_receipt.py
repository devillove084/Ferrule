#!/usr/bin/env python3
"""Read-only inspector for a runtime logits claim receipt.

This is deliberately not the full TP2 comparator. It proves what a logits-only
claim contains and fail-closed reports the boundaries still needed by the
paper-local full-inventory comparison.
"""
import json
from pathlib import Path

EXPECTED_CASES = ('hello', 'capital')
EXPECTED_STAGES = ('prefill', 'decode.0')
REQUIRED_FULL_BOUNDARIES = (
    'embedding', 'final_norm', 'layers.*.hidden', 'layers.*.attention',
    'layers.*.moe', 'rank.*.layers.*.attention_partial',
    'rank.*.layers.*.moe_partial', 'rank.*.layers.*.shared_partial',
    'rank.*.layers.*.routed_partial', 'rank.*.layers.*.raw_router_logits',
    'rank.*.layers.*.router_probabilities', 'rank.*.layers.*.routing_weights',
    'rank.*.layers.*.selected_expert_ids', 'cache.*', 'rank.*.cache.*',
    'rank.*.logits', 'logits',
)


def _load(path):
    return json.loads(Path(path).read_text())


def inspect_receipt(directory):
    """Return a capability report without changing the receipt directory."""
    directory = Path(directory).resolve()
    result = _load(directory/'result.json')
    exit_record = _load(directory/'test-exit.json')
    metrics = {}
    actual_files = []
    for case in EXPECTED_CASES:
        for stage in EXPECTED_STAGES:
            metric_path = directory/f'{case}.{stage}.metrics.json'
            actual_path = directory/f'{case}.{stage}.actual.f32le'
            metric = _load(metric_path)
            metrics[f'{case}.{stage}'] = dict(
                failures=metric['failures'], compared=metric['compared'],
                finite_compared=metric['finite_compared'], nonfinite=metric['nonfinite'],
                max_abs_finite=metric['max_abs_finite'], rmse_finite=metric['rmse_finite'],
                actual_greedy=metric['actual_greedy'], expected_greedy=metric['expected_greedy'],
                atol=metric['atol'], rtol=metric['rtol'])
            actual_files.append(dict(case=case, stage=stage, file=actual_path.name,
                bytes=actual_path.stat().st_size))
    passed = bool(result.get('passed') is True and result.get('drained') is True and
                  result.get('all_eight_idle') is True and exit_record.get('returncode') == 0 and
                  all(v['failures'] == 0 and v['nonfinite'] == 0 and
                      v['actual_greedy'] == v['expected_greedy'] for v in metrics.values()))
    return dict(
        schema='ferrule.paper.tp2-mixed-claim-receipt-inspection.v1',
        status='logits_only_partial' if passed else 'invalid_claim_receipt',
        source_directory=str(directory),
        claim_passed=passed,
        cases=list(EXPECTED_CASES), stages=list(EXPECTED_STAGES), metrics=metrics,
        actual_files=actual_files,
        full_inventory_compatible=False,
        missing_full_inventory=list(REQUIRED_FULL_BOUNDARIES),
        missing_provenance=['per-rank route IDs/weights', 'per-rank attention/MoE partials',
                            'global and rank-local KV/GDN F32 state', 'full checkpoint/shard identity',
                            'candidate safetensors/report inventory'],
        admission='Logits-only claim is not a candidate for compare.py and cannot establish TP2 route/state/shard equivalence')


def main(argv=None):
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(inspect_receipt(args.directory),indent=2,sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
