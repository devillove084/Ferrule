#!/usr/bin/env python3
"""CPU test adapter for production lifecycle observations; no inference execution.

Reuses the frozen paper's tensor mapper and suite validator. Raw originals are
never modified; per-call bounded projections only renumber file offsets/order.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[4]
PAPER = ROOT / 'papers/arxiv/experiments/tp2_mixed/oracle'
sys.path.insert(0, str(PAPER))
import bridge as b
import compare as c
import lifecycle_compare as lc
import multistep as m
import tp2_oracle as o
from safetensors.torch import save_file


def snapshot(root):
    root = Path(root).resolve()
    raw = root / 'lifecycle-raw'
    files = sorted(p for p in raw.rglob('*') if p.is_file())
    files += sorted(root.glob('call-*.call.actual.f32le'))
    b.require(files, 'No lifecycle raw capture')
    pins = {}
    for p in files:
        b.require(not p.is_symlink() and p.resolve().is_relative_to(root), 'Escaping raw file')
        pins[str(p.relative_to(root))] = dict(bytes=p.stat().st_size, sha256=o.sha256(p))
    return dict(files=pins, sha256=hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest())


def owner_index(directory, rank, contract):
    directory = directory / f'rank-{rank}'
    status = b.read(directory / 'capture-status.json')
    b.require(status['schema'] == b.OWNER_SCHEMA and status['complete_available_trace'] is True
              and status['required_inventory_complete'] is False and not status['capture_error'], 'Incomplete epoch writer')
    stream = directory / 'tensors.f32le'
    b.require(stream.stat().st_size <= contract['max_bytes_per_owner'] <= 2 << 30, 'Epoch payload budget')
    b.require((directory/'events.jsonl').stat().st_size <= 64 << 20, 'Epoch index budget')
    binding = None
    calls = []
    offset = order = 0
    with (directory/'events.jsonl').open() as f:
        for line in f:
            b.require(len(line.encode()) <= 256 << 10, 'Metadata record cap')
            e = json.loads(line)
            if e['kind'] == 'owner_binding':
                b.require(binding is None and not calls, 'Duplicate binding')
                binding = e
                continue
            if e['kind'] == 'begin':
                b.require(e['invocation'] == len(calls)+1, 'Invocation order')
                calls.append([])
            b.require(binding is not None and calls and e['invocation'] == len(calls), 'Orphan event')
            if e['kind'] == 'tensor':
                b.require(e['order'] == order and e['offset'] == offset, 'Epoch tensor range/order gap')
                order += 1
                offset += e['bytes']
            calls[-1].append(e)
    b.require(order == status['events'] and order <= contract['max_events_per_owner'] <= 100000, 'Epoch event count')
    b.require(offset == status['tensor_bytes'] == stream.stat().st_size, 'Epoch byte count')
    b.require(len(calls) == status['invocations'] == contract['max_invocations'], 'Epoch call count')
    return dict(binding=binding, calls=calls, stream=stream, status=status)


def project_call(index, call, destination):
    """Copy observed bytes only; normalize a single call for existing strict parser."""
    destination.mkdir()
    rows = [index['binding']]
    offset = order = 0
    with index['stream'].open('rb') as source, (destination/'tensors.f32le').open('xb') as sink:
        for original in call:
            e = copy.deepcopy(original)
            e['invocation'] = 1
            if e['kind'] == 'tensor':
                source.seek(e['offset'])
                remaining = e['bytes']
                while remaining:
                    data = source.read(min(remaining, 1 << 20))
                    b.require(data, 'Truncated raw source')
                    sink.write(data)
                    remaining -= len(data)
                e['offset'], e['order'] = offset, order
                offset += e['bytes']
                order += 1
            rows.append(e)
    b.require(order <= 40000 and offset <= 1 << 30, 'Single-call legacy-parser projection cap')
    (destination/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in rows))
    status = dict(index['status'], events=order, tensor_bytes=offset, invocations=1)
    (destination/'capture-status.json').write_text(json.dumps(status))


def state_hashes(arrays, config, prefix='call.'):
    result = {}
    for owner, rank in enumerate(('', 'rank.0.', 'rank.1.')):
        for layer, kind in enumerate(config.layer_types):
            fields = [('conv_states', 'conv_states'), ('recurrent_states', 'recurrent_states')] if kind == 'linear_attention' else [('keys', 'key_cache'), ('values', 'value_cache')]
            for native, key in fields:
                result[f'{owner}.{layer}.{native}'] = o.digest_tensor(arrays[f'{prefix}{rank}cache.{key}.{layer:02d}'])
    return result


def before_arrays(owners, calls, after, config):
    """Actual GDN before-observations; retained KV prefix of authenticated export.

No zero state synthesis, reference digest copies, or KV reconstruction from new
K/V projections. The retained prefix is sliced from the real committed snapshot.
"""
    result = {}
    start = calls[0]['begin']['positions'][0]
    for rank, (owner, call) in enumerate(zip(owners, calls)):
        for e in call['tensors']:
            if e['event'] not in ('gdn.before.conv', 'gdn.before.ssm'):
                continue
            b.validate_metadata(e, rank, call, config)
            field = 'conv_states' if e['event'] == 'gdn.before.conv' else 'recurrent_states'
            key = f'call.rank.{rank}.cache.{field}.{e["layer"]:02d}'
            b.require(key not in result, 'Duplicate before state')
            result[key] = b.decode(owner['stream'], e)[None]
            b.require(result[key].shape == after[key].shape and result[key].dtype == after[key].dtype, 'Before-state geometry/dtype')
        for layer, kind in enumerate(config.layer_types):
            if kind == 'full_attention':
                for name in ('key_cache', 'value_cache'):
                    key = f'call.rank.{rank}.cache.{name}.{layer:02d}'
                    result[key] = after[key][:, :, :start, :].contiguous()
    for layer, kind in enumerate(config.layer_types):
        for name in (('conv_states', 'recurrent_states') if kind == 'linear_attention' else ('key_cache', 'value_cache')):
            values = [result[f'call.rank.{r}.cache.{name}.{layer:02d}'] for r in (0, 1)]
            result[f'call.cache.{name}.{layer:02d}'] = o.merge_gdn(values, config, 2) if name == 'conv_states' else o.torch.cat(values, 1)
    return result


def convert(root, output, reference, reference_pin, raw_pin):
    root, output = Path(root).resolve(), Path(output).resolve()
    b.require(not output.is_relative_to(root), 'Cannot write inside raw evidence')
    before = snapshot(root)
    b.require(before['sha256'] == raw_pin, 'Raw snapshot changed')
    raw = root/'lifecycle-raw'
    contract, result = b.read(raw/'contract.json'), b.read(raw/'result.json')
    b.require(contract['schema'] == result['schema'] == 'ferrule.mixed-lifecycle-raw.v1', 'Raw schema')
    b.require(result['status'] == 'captured' and result['shutdown_ok'] and result['writer_complete'], 'Raw lifecycle incomplete')
    b.require(contract['reference_manifest_sha256'] == reference_pin, 'Reference pin differs')
    for field in ('checkpoint', 'protocol'):
        b.require(contract[field] == reference[field], 'Source/protocol differs')
    # The authenticated release manifest is the canonical recipe.  Do not
    # regenerate it through the mutable paper helper: prompt-map ordering is
    # part of the pinned operation sequence and must not be reinterpreted.
    plan = reference['suite']['plan']
    b.require(contract['plan'] == plan, 'Different recipes')
    b.require(plan.get('schema') == m.PLAN_SCHEMA and isinstance(plan.get('operations'), list),
              'Malformed pinned lifecycle plan')
    b.require(len(result['events']) == len(plan['operations']), 'Missing raw operations')
    config = o.native.hf.Qwen3_5MoeTextConfig(**reference['checkpoint']['effective_text_config'])
    epoch_indices = {}
    total_payload = 0
    for epoch, count in enumerate(contract['epoch_calls']):
        directory = raw/f'epoch-{epoch}'
        ec = b.read(directory/'contract.json')
        b.require(ec['epoch'] == epoch and ec['plan'] == plan and ec['oracle_manifest_sha256'] == reference_pin and ec['max_invocations'] == count, 'Epoch binding')
        epoch_indices[epoch] = [owner_index(directory, rank, ec) for rank in (0, 1)]
        total_payload += sum(i['status']['tensor_bytes'] for i in epoch_indices[epoch])
    b.require(total_payload <= plan['limits']['max_total_capture_bytes'], 'Aggregate raw tensor budget')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=lc.CANDIDATE_SCHEMA, profile=o.PROFILE, protocol=reference['protocol'],
                    checkpoint=reference['checkpoint'], reference_manifest_sha256=reference_pin,
                    status='incomplete', cases=[], suite=dict(plan=plan, events=[], cases=[]),
                    bridge_source_sha256=o.sha256(__file__), boundary_inventory_complete=False)
    sessions = {}
    epoch = 0
    used = set()
    canonical_bytes = 0
    try:
        for op, observed in zip(plan['operations'], result['events']):
            b.require(all(observed.get(k) == v for k, v in op.items()), 'Raw operation differs')
            kind, name = op['op'], op.get('session')
            event = copy.deepcopy(observed)
            if kind == 'recreate_model':
                b.require(not sessions and observed['decoded_cache_cleared'], 'Recreation with live state')
                epoch += 1
            b.require(event['epoch'] == epoch, 'Wrong epoch')
            if kind == 'create':
                sessions[name] = None
            elif kind == 'fork':
                b.require(sessions.get(op['source']) is not None, 'Uncaptured fork source')
                event['state_sha256'] = sessions[op['source']]
                sessions[name] = dict(event['state_sha256'])
            elif kind == 'drop':
                del sessions[name]
            elif kind == 'call':
                call_id = op['id']
                directory = raw/f'epoch-{epoch}'
                parent = b.read(directory/f'{call_id}.receipt.json')
                kv = b.read(directory/f'{call_id}.kv-receipt.json')
                b.require(parent['case'] == call_id and observed['transaction'] == parent['transaction'] and observed['runtime_session'] == parent['session'], 'Raw call publication identity')
                indices = epoch_indices[epoch]
                selected = []
                for index in indices:
                    found = [call for call in index['calls'] if call[0]['identity']['transaction'] == parent['transaction'] and call[0]['identity']['sessions'] == [parent['session']]]
                    b.require(len(found) == 1, 'Missing/duplicate raw call')
                    selected.append(found[0])
                identity = epoch, parent['transaction']
                b.require(identity not in used, 'Repeated transaction'); used.add(identity)
                with tempfile.TemporaryDirectory(dir=output) as tmp:
                    tmp = Path(tmp)
                    for rank in (0, 1): project_call(indices[rank], selected[rank], tmp/f'rank-{rank}')
                    owners = [b.read_owner(tmp, rank, config, expected_invocations=1) for rank in (0, 1)]
                    calls = b.join_call(owners, parent, kv, config)
                    logits = b.full_logits(root/f'{call_id}.call.actual.f32le', len(parent['input_token_ids']), config.vocab_size)
                    arrays, record, inventory = b.assemble_call(owners, calls, parent, config, 'call', logits)
                    b.require(record == observed['call'], 'Observed call differs from actual tensors')
                    event['state_before'] = state_hashes(before_arrays(owners, calls, arrays, config), config)
                    event['state_after'] = state_hashes(arrays, config)
                if sessions[name] is not None:
                    b.require(event['state_before'] == sessions[name], 'Actual before state differs from preceding source/fork')
                sessions[name] = event['state_after']
                canonical_bytes += sum(v.numel()*v.element_size() for v in arrays.values())
                b.require(canonical_bytes <= plan['limits']['max_total_capture_bytes'], 'Canonical tensor budget')
                path = output/(call_id+'.safetensors')
                save_file(arrays, str(path), metadata={'schema':lc.CANDIDATE_SCHEMA, 'profile':o.PROFILE})
                report = dict(id=call_id, calls=[record], tensors=inventory, **o.file_record(path))
                o.native.write_json(output/(call_id+'.json'), report)
                receipt = output/(call_id+'.publication.json')
                shutil.copyfile(directory/f'{call_id}.receipt.json', receipt)
                entry = dict(id=call_id, **o.file_record(path), report=o.file_record(output/(call_id+'.json')), publication=o.file_record(receipt))
                manifest['cases'].append(entry)
            manifest['suite']['events'].append(event)
        b.require(not sessions and len(used) == sum(contract['epoch_calls']), 'Unjoined capture/session')
        manifest['suite']['cases'] = manifest['cases']
        lc.validate_suite(output, manifest['suite'], config, False)
        b.require(snapshot(root) == before, 'Raw changed during conversion')
        o.native.write_json(output/'raw-source.json', before)
        manifest.update(status='complete', boundary_inventory_complete=True,
                        raw_source=o.file_record(output/'raw-source.json'), total_capture_bytes=canonical_bytes)
    except BaseException as error:
        manifest['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        o.native.write_json(output/'manifest.json', manifest)
        (output/'manifest.sha256').write_text(o.sha256(output/'manifest.json')+'\n')
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True);p.add_argument('--inspect', action='store_true')
    p.add_argument('--input-sha256');p.add_argument('--reference',type=Path)
    p.add_argument('--reference-sha256');p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.inspect:
        s=snapshot(a.input);print(json.dumps(dict(sha256=s['sha256'], files=len(s['files']))));return
    b.require(all((a.input_sha256,a.reference,a.reference_sha256,a.output)), 'Missing pins/output')
    b.require(not a.output.resolve().is_relative_to(a.reference.resolve()), 'Cannot overwrite reference')
    o.initialize()
    reference=c.load_manifest(a.reference,a.reference_sha256,True,lifecycle=True)
    result=convert(a.input,a.output,reference,a.reference_sha256,a.input_sha256)
    print(json.dumps(dict(status=result['status'],cases=len(result['cases']),manifest_sha256=o.sha256(a.output/'manifest.json'))))

if __name__=='__main__':main()
