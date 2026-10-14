"""Offline Capital full-image diagnostic comparison; not acceptance.

Maps only immutable archive fields. Unknown production projection/intermediate
observations remain diagnostic-only and never become expected values.
"""
import argparse
import json
import struct
from pathlib import Path
import hybrid_mixed_production_compare as base


def mapping(event, rank):
    layer = event.get('layer')
    name = event['event']
    if name == 'embedding.output':
        return 'prefill.embedding'
    if layer is None:
        if name == 'norm.output' and event.get('role') == 'OutputNorm':
            return 'prefill.final_norm'
        if name == 'mixed.linear.output' and event.get('role') == 'OutputHead':
            return f'prefill.rank.{rank}.logits'
        return None
    prefix = f'prefill.rank.{rank}.layers.{layer:02d}.'
    full = f'prefill.layers.{layer:02d}.'
    rank_cache = f'prefill.rank.{rank}.cache.'
    if name == 'mixed.layer.input':
        return 'prefill.embedding' if layer == 0 else f'prefill.layers.{layer-1:02d}.hidden'
    if name == 'router':
        return prefix + 'raw_router_logits'
    if name == 'gqa.key':
        return rank_cache + f'key_cache.{layer:02d}'
    if name == 'gqa.value':
        return rank_cache + f'value_cache.{layer:02d}'
    if name == 'mixed.layer.output':
        return full + 'hidden'
    if name == 'mixed.attention.partial':
        return prefix + 'attention_partial'
    if name == 'mixed.attention.reduced':
        return full + 'attention'
    if name == 'mixed.moe.partial':
        return prefix + 'moe_partial'
    if name == 'mixed.moe.reduced':
        return full + 'moe'
    if name == 'mixed.routed.partial':
        return prefix + 'routed_partial'
    if name == 'mixed.swiglu.partial' and event.get('role') == 'SharedExpertDown':
        return prefix + 'shared_partial'
    if name == 'mixed.linear.output' and event.get('role') == 'SharedExpertOutputGate':
        return prefix + 'shared_gate_logit'
    if name == 'gdn.conv_state':
        return rank_cache + f'conv_states.{layer:02d}'
    if name == 'gdn.recurrent_state':
        return rank_cache + f'recurrent_states.{layer:02d}'
    return None


def compare(reference, capture):
    assert base.sha(reference / 'manifest.json') == base.PIN
    manifest = json.loads((reference / 'manifest.json').read_text())
    case = next(c for c in manifest['cases'] if c['id'] == 'capital')
    assert base.sha(reference / case['file']) == case['sha256']
    archive = base.Archive(reference / case['file'])
    inventory = {k for k in archive.header if k != '__metadata__' and not k.endswith('.native_expert_ids')}
    assert len(inventory) == 2178
    rows, unknown, covered = [], [], set()
    for rank in range(2):
        directory = capture / f'rank-{rank}'
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        assert events
        assert [e['layer'] for e in events if e['event']=='mixed.layer.output']==list(range(40))
        assert max((e['layer'] for e in events if e['layer'] is not None), default=0) == 39
        for order, event in enumerate(events):
            key = mapping(event, rank)
            if key is not None and key in archive.header:
                data = (directory / event['file']).read_bytes()
                actual = [v[0] for v in struct.iter_unpack('<f', data)]
                if event['event'] in ('gqa.key','gqa.value'):
                    batch,heads,tokens,dim=archive.header[key]['shape']
                    assert batch==1 and event['shape']==[tokens,heads*dim]
                    actual=[actual[t*heads*dim+h*dim+d] for h in range(heads) for t in range(tokens) for d in range(dim)]
                row = dict(rank=rank, layer=event.get('layer'), order=order,
                           event=event['event'], reference=key,
                           **base.metrics(actual, archive.values(key)))
                rows.append(row)
                covered.add(key)
            else:
                unknown.append(dict(rank=rank, layer=event.get('layer'), order=order,
                                    event=event['event'], role=event.get('role'),
                                    shape=event['shape']))
            if event['event'] == 'router':
                for field, suffix in [('ids', 'selected_expert_ids'), ('weights', 'routing_weights')]:
                    key = f'prefill.rank.{rank}.layers.{event["layer"]:02d}.{suffix}'
                    row = dict(rank=rank, layer=event['layer'], order=order,
                               event=field, reference=key,
                               **base.metrics(event[field], archive.values(key), field == 'ids'))
                    rows.append(row)
                    covered.add(key)
        data=(directory/'logits.f32').read_bytes()
        rows.append(dict(rank=rank,layer=None,order=len(events),event='logits.export',reference='prefill.logits',
                         **base.metrics([v[0] for v in struct.iter_unpack('<f',data)],archive.values('prefill.logits'))))
        covered.add('prefill.logits')
    def chronological(row):
        layer=row['layer']
        if layer is None:layer=-1 if row['event']=='embedding.output' else 40
        return layer,row['order'],row['rank']
    rows.sort(key=chronological)
    mismatches = [row for row in rows if row['unequal']]
    first = min(mismatches, key=chronological, default=None)
    preceding = []
    if first is not None:
        for row in rows:
            if chronological(row) < chronological(first):
                preceding.append(row)
    result = dict(schema='ferrule.capital-full-image-diagnostic.v1', acceptance=False,
                  reference_manifest_sha256=base.PIN, expected_candidate_tensors=2178,
                  distinct_compared_tensors=len(covered), missing_expected_count=len(inventory - covered),
                  compared_observations=len(rows), unknown_observations=len(unknown),
                  first_mismatch=first, preceding_exact_count=sum(r['unequal'] == 0 for r in preceding),
                  preceding_last=preceding[-1] if preceding else None,
                  mismatches=mismatches, comparisons=rows, unknown=unknown,
                  note='Full Capital forward completed; unknown projections/states are diagnostics only and were not synthesized into expected values')
    with (capture / 'full-boundary-comparison.json').open('x') as out:
        json.dump(result,out,indent=2);out.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--capture', required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.capture)
    print(json.dumps({k: v for k, v in result.items() if k not in ('comparisons', 'unknown', 'mismatches')}, indent=2))
