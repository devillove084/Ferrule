#!/usr/bin/env python3
"""CPU-only preparation; never invent missing per-expert CUDA observations."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
import compare as c
import tp2_oracle as o
from safetensors import safe_open
from safetensors.torch import save_file

REFERENCE_PIN = 'fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d'
LAYERS = (0, 3)
STAGE = 'decode.0'
CAP = 192 * 1024**2


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def row_slice(raw, rows, columns, low, high, word_bytes):
    check(len(raw) == rows * columns * word_bytes, 'Wrong source byte geometry')
    check(0 <= low < high <= columns, 'Invalid column range')
    return b''.join(raw[(r * columns + low) * word_bytes:
                       (r * columns + high) * word_bytes] for r in range(rows))


def down_slices(weight, scales, n, global_k, rank):
    check(rank in (0, 1) and n > 0 and global_k > 0 and
          n % 128 == 0 and global_k % 256 == 0, 'Incomplete TP2 scale blocks')
    k = global_k // 2
    low, high = rank * k, (rank + 1) * k
    return (row_slice(weight, n, global_k, low, high, 1),
            row_slice(scales, n // 128, global_k // 128, low // 128, high // 128, 2))


class PinnedPayloads:
    """Bounded original bytes, authenticated against old used-payload digests."""
    def __init__(self, directory, checkpoint, metadata, used):
        self.directory, self.checkpoint, self.metadata, self.used = directory, checkpoint, metadata, used
        self.headers, self.identities, self.cache = {}, {}, {}
        self.bytes = 0

    def read(self, name):
        if name in self.cache:
            return self.cache[name]
        entry = self.metadata['tensors'][name]
        recorded = self.used[name]  # Missing independently authenticated payload is fatal.
        shard = entry['shard']
        path = (self.directory / shard).resolve()
        check(path.parent == self.directory.resolve(), 'Shard path escape')
        st = path.stat()
        identity = (st.st_size, st.st_mtime_ns, st.st_ino)
        check(self.identities.get(shard, identity) == identity, 'Shard changed')
        self.identities[shard] = identity
        original = self.metadata['shards'][shard]
        check(st.st_size == original['bytes'] and shard in self.checkpoint['full_shards'], 'Shard size/identity differs')
        with path.open('rb') as f:
            if shard not in self.headers:
                length = struct.unpack('<Q', f.read(8))[0]
                check(length == original['header_bytes'] and 0 < length <= 64 * 1024**2, 'Header size differs')
                header = f.read(length)
                check(digest(header) == original['header_sha256'], 'Original header hash differs')
                self.headers[shard] = (length, json.loads(header))
            length, header = self.headers[shard]
            check(header[name] == {k: entry[k] for k in ('dtype', 'shape', 'data_offsets')}, 'Tensor header differs')
            lo, hi = entry['data_offsets']
            check(0 <= lo < hi <= st.st_size - length - 8 and hi - lo <= 2 * 1024**2, 'Payload byte bound')
            check(self.bytes + hi - lo <= CAP, 'Total payload cap')
            f.seek(8 + length + lo)
            raw = f.read(hi - lo)
        check(len(raw) == hi - lo and digest(raw) == recorded['sha256'], 'Original payload hash differs: ' + name)
        check(recorded['shard'] == shard and recorded['shape'] == entry['shape'], 'Payload identity differs')
        self.cache[name] = raw
        self.bytes += len(raw)
        return raw

    def finish(self):
        for shard, identity in self.identities.items():
            s = (self.directory / shard).stat()
            check((s.st_size, s.st_mtime_ns, s.st_ino) == identity, 'Shard changed during extraction')


def extract(output):
    check(sys.byteorder == 'little', 'Little-endian host required')
    check(output.resolve().is_relative_to(o.HERE / 'runs') and not output.exists(), 'Fresh paper run required')
    check(not o.torch.cuda.is_initialized(), 'Unexpected CUDA initialization')
    o.initialize()
    ref = o.HERE / 'runs/full40-20261009-03'
    m = c.load_manifest(ref, REFERENCE_PIN, True)
    metadata = json.loads(c.checked_file(ref, m['checkpoint_metadata']).read_text())
    used = json.loads(c.checked_file(ref, m['source_payloads']).read_text())['used_payloads']
    model = Path(m['checkpoint']['directory'])
    check(o.sha256(model / 'config.json') == m['checkpoint']['config_sha256'], 'Config differs')
    check(o.sha256(model / 'model.safetensors.index.json') == m['checkpoint']['index_sha256'], 'Index differs')
    mapping = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    cfg = m['checkpoint']['effective_text_config']
    n, global_k = cfg['hidden_size'], cfg['moe_intermediate_size']
    check((n, global_k, cfg['num_experts_per_tok']) == (2048, 512, 8), 'Unexpected actual geometry')
    payloads = PinnedPayloads(model, m['checkpoint'], metadata, used)
    output.mkdir()
    pending = dict(schema='ferrule.paper.mixed-routed-down.cpu-preparation.v1', status='incomplete',
                   model_fixture_ready=False, cuda_executed=False)
    try:
        (output / 'payloads').mkdir()
        sources, fixtures, replays = {}, [], []
        def store(name):
            if name in sources:
                return payloads.read(name)
            check(mapping[name] == metadata['tensors'][name]['shard'], 'Index/header mismatch')
            raw = payloads.read(name)
            filename = 'payloads/' + name + '.bin'
            (output / filename).write_bytes(raw)
            sources[name] = dict(file=filename, bytes=len(raw), sha256=digest(raw),
                                 original_header=metadata['tensors'][name],
                                 original_used_payload=used[name],
                                 historical_whole_shard_sha256=m['checkpoint']['full_shards'][mapping[name]])
            return raw
        for case in m['cases']:
            report = json.loads(c.checked_file(ref, case['report']).read_text())
            with safe_open(c.checked_file(ref, case), framework='pt', device='cpu') as archive:
                for layer in LAYERS:
                    prefix = f'{STAGE}.rank.0.layers.{layer:02d}.'
                    ids = archive.get_tensor(prefix + 'selected_expert_ids')
                    check(ids.shape == (1, 8) and len(set(ids[0].tolist())) == 8, 'M1 actual route required')
                    other = archive.get_tensor(prefix.replace('rank.0.', 'rank.1.') + 'selected_expert_ids')
                    check(o.torch.equal(ids, other), 'Rank routes differ')
                    expert = min(ids[0].tolist())  # Deterministic selection before candidate execution.
                    name = f'{case["id"]}-decode0-layer{layer:02d}-expert{expert:03d}'
                    folder = output / name
                    folder.mkdir()
                    input_key = STAGE + ('.embedding' if layer == 0 else f'.layers.{layer-1:02d}.hidden')
                    keys = [input_key, f'{STAGE}.layers.{layer:02d}.attention', f'{STAGE}.layers.{layer:02d}.moe']
                    for rank in (0, 1):
                        keys += [f'{STAGE}.rank.{rank}.layers.{layer:02d}.' + field for field in
                                 ('routed_partial', 'selected_expert_ids', 'routing_weights', 'raw_router_logits', 'router_probabilities')]
                    boundary = {k: archive.get_tensor(k).contiguous() for k in keys}
                    check(all(bool(o.torch.isfinite(v).all()) for v in boundary.values()), 'Nonfinite observed boundary')
                    check(all('expert_input' not in k and 'swiglu' not in k for k in archive.keys()), 'Inventory changed; reevaluate direct extraction')
                    save_file(boundary, str(folder / 'observed.safetensors'))
                    base = f'model.language_model.layers.{layer}.'
                    # Freeze all eight actually routed experts for a later native HF replay/control;
                    # no CPU norm, SwiGLU, projection or expected value is computed here.
                    replay_names = [base + 'post_attention_layernorm.weight', base + 'mlp.gate.weight']
                    for e in sorted(ids[0].tolist()):
                        for part in ('gate_proj', 'up_proj', 'down_proj'):
                            weight_name = base + f'mlp.experts.{e}.{part}.weight'
                            replay_names += [weight_name, weight_name + '_scale_inv']
                    for key in replay_names:
                        store(key)
                    weight_name = base + f'mlp.experts.{expert}.down_proj.weight'
                    scale_name = weight_name + '_scale_inv'
                    check(metadata['tensors'][weight_name]['dtype'] == 'F8_E4M3' and
                          metadata['tensors'][weight_name]['shape'] == [n, global_k], 'Down raw FP8 geometry')
                    check(metadata['tensors'][scale_name]['dtype'] == 'BF16' and
                          metadata['tensors'][scale_name]['shape'] == [n // 128, global_k // 128], 'Original BF16 scale grid')
                    partitions = []
                    for rank in (0, 1):
                        w, s = down_slices(store(weight_name), store(scale_name), n, global_k, rank)
                        (folder / f'rank{rank}.weight.fp8').write_bytes(w)
                        (folder / f'rank{rank}.scale.bf16').write_bytes(s)
                        coords = [[row, col] for row in range(n // 128) for col in range(rank * 2, (rank + 1) * 2)]
                        partitions.append(dict(rank=rank, rows=[0, n], columns=[rank * 256, (rank + 1) * 256],
                                               scale_shape=[16, 2], global_128_block_coordinates=coords,
                                               block_coordinates_sha256=digest(json.dumps(coords).encode())))
                    prep = dict(schema='ferrule.paper.mixed-routed-down.pending-fixture.v1',
                                status='blocked_missing_cuda_expert_observations', model_fixture_ready=False,
                                role='RoutedExpertDown', tp=2, rows=1, n=n, local_k=global_k // 2, layer=layer, expert=expert,
                                checkpoint_sha256=REFERENCE_PIN, checkpoint_shard_set_sha256=m['checkpoint']['full_shard_set_sha256'],
                                case=case['id'], stage=STAGE, call=report['calls'][1],
                                selected_expert_ids=ids[0].tolist(), selection='minimum actually selected expert ID, one per case/layer',
                                route_slot=ids[0].tolist().index(expert), reference_case=case,
                                observed_tensor_keys=keys, raw_weight_source=weight_name, raw_scale_source=scale_name,
                                partitions=partitions,
                                missing=[f'rank{r}.{s}.bf16' for r in (0, 1) for s in ('input', 'hf.partial')],
                                warning='routed_partial is weighted expert accumulation, NOT an unweighted expert down partial')
                    write_json(folder / 'preparation.json', prep)
                    fixtures.append(dict(directory=name, **{k:prep[k] for k in ('case','stage','layer','expert','n','local_k','missing')}))
                    replays.append(dict(fixture=name, upstream_input_key=input_key, payload_names=replay_names,
                                        calls='two native rank Experts.forward calls, eight actual routed experts each'))
        payloads.finish()
        check(not o.torch.cuda.is_initialized(), 'Extraction initialized CUDA')
        check(o.sha256(ref / 'manifest.json') == REFERENCE_PIN, 'Historical reference changed')
        source_paths = [Path(__file__), o.HERE / 'tp2_oracle.py', o.HERE / 'compare.py',
                        Path(o.native.__file__), Path(o.native.legacy.__file__)]
        (output / 'extractor-source.py').write_bytes(Path(__file__).read_bytes())
        pending.update(status='cpu_extracted_pending_authorized_cuda', reference_manifest_sha256=REFERENCE_PIN,
                       protocol_sha256=o.sha256(o.HERE / 'protocol.json'), native_source=m['native_source'],
                       extraction_source_pins={str(p):o.sha256(p) for p in source_paths},
                       checkpoint=m['checkpoint'], checkpoint_metadata=m['checkpoint_metadata'], source_payloads=m['source_payloads'],
                       current_checkpoint_check='config/index, touched shard sizes/header hashes and every used original payload SHA verified; full shards NOT reread/rehashed',
                       raw_payload_bytes=payloads.bytes, payload_byte_cap=CAP, original_payloads=sources,
                       fixtures=fixtures, gpu_replays=replays,
                       future_cuda_task=dict(max_seconds=120, outer_timeout_ms=180000, physical_devices=1, logical_tp=2,
                           native_rank_expert_calls=8, down_projections=64, target_fixtures=4,
                           estimated_device_budget_mib=1024, capture_budget_mib=16,
                           authentication='existing paper lock/idle/authenticated subreaper; a small replay worker must be wired before execution',
                           controls='GPU BF16 residual + pinned HF norm/router/Experts.forward; exact old router/weighted routed_partial control; intercept actual down F.linear input/output before weighting',
                           prohibition='no candidate kernels, no CPU numerical expected, no full model replay, no threshold substitutions'),
                       acceptance='future model gate requires exact BF16 words for native AND W8 candidate; mismatch blocks, no tolerance')
        write_json(output / 'manifest.json', pending)
        files=sorted(p for p in output.rglob('*') if p.is_file())
        with (output / 'SHA256SUMS.cpu-extract').open('x') as f:
            for p in files:
                f.write(o.sha256(p) + '  ' + str(p.relative_to(output)) + '\n')
        print(json.dumps(dict(status=pending['status'], directory=str(output), fixtures=len(fixtures),
                              raw_payload_bytes=payloads.bytes, sha256=o.sha256(output / 'SHA256SUMS.cpu-extract'),
                              cuda_initialized=o.torch.cuda.is_initialized())))
    except BaseException as e:
        write_json(output / 'failure.json', dict(status='incomplete', error=str(e), model_fixture_ready=False))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    extract(parser.parse_args().output)
