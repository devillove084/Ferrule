#!/usr/bin/env python3
"""CPU-only, source-qualified replay of the public actual-05 NVTX projection.

Example (all inputs are read-only; JSON goes to stdout):
  python3 -B scripts/replay_expert_cache.py \
    --archive papers/arxiv/experiments/storage_demand/profile_overlap_v1/publication_v1 \
    --source-manifest target/validation/fixesmetadataCLIcurrent-quality-20261015-01/source-freeze.json

No CUDA, network, model loading, kernel reordering, or new experiment harness.
Canonical parameter IDs are NOT exported by this archive. Keys below are explicitly
lower-qualified: immutable source/image + audited precision + layer/expert/roles,
not fabricated canonical IDs and not a replacement for the GPU cache's full key.
Only the reviewed Qwen3.5 capture is supported; DeepSeek V4 remains untested.

Policies use only prior history and the CURRENT layer's already-routed selection.
Layer quotas are fixed from the declared model layer count, never future demand.
Both bypass policies reserve one transient slot INSIDE the slot/byte budget.
Frequency admission compares prior counts, admits only on strict improvement,
then increments a fixed 4x2048 byte count-min sketch (seed 0, cap 255, halved every
4096 demands). Counts are approximate/aged, never evidence of coldness.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys

MIB = 1 << 20
MAX_INPUT_BYTES = 32 * MIB
MAX_ROWS = 200_000
PAYLOAD_BYTES = 3_146_112
PRECISION = 'Bf16ModelMixedF32Accumulate'
ROLES = ('RoutedExpertGate', 'RoutedExpertUp', 'RoutedExpertDown')
POLICIES = ('global_lru', 'layer_protected_bypass', 'history_frequency_bypass')
SOURCE_COMMIT = '86c58a017d7c4b53490f536a3b1a133f2f7848cc'
# Reviewed immutable binding/identity checks, linked to the captured source manifest.
BINDING_PINS = {
    'crates/ferrule-composition/src/model_factory/mixed_serving.rs': '12ee977819eb651733b01feaa2b7205fc84aa75e678488d36bf3bbbb27330900',
    'crates/ferrule-model/src/transformer/standard/expert_cache.rs': 'cbb6b8cf01119ca649f3446ecf5a189fcd0e3d0886d449b1aa0a9989428be229',
    'crates/ferrule-model/src/transformer/standard/cuda/residency.rs': '493cbf427816b0be25e1d843840d66a60d98bfc17cdd7ce2ac2f0edf10a8fd8c',
    'crates/ferrule-model/src/transformer/standard/cuda/binding.rs': '4ab70e10fccd2cc40c1b8b8c787a0d6f89b3890203ce96ded932d08726f67c18',
    'crates/ferrule-model/src/transformer/operators.rs': 'dd04423f4d99596599d5fcba489459b89ed70cb0f0275bd1042f8ccf49a05d58',
    'crates/ferrule-model/src/transformer/standard/cuda.rs': 'db6835984a6dece5d31eb48398cb510263cab27b0f30409f2aff5b04da8247f3',
}
# Audited collector commands: qwen35-mixed-bf16, lazy, 64 entries, 320 MiB, 8 MiB scratch.
# Precision is reconstructed from this binding, NOT read from a nonexistent CSV column.
COLLECTOR_PINS = {
    'blocking': 'a5535d66a11ac19abb5171cbe785c14599186204c437a1849bf8bc5a87a212a3',
    'lookahead': '0c1ece3395512b522d842bc5b2e28cdcbaf1296151e74573ae8e6bafb6834798',
}
NVTX_FIELDS = ('row_id kind start_ns end_ns global_pid global_tid end_global_tid '
               'event_type domain_id range_id operation source_generation requested_bytes '
               'image_generation layer expert rows').split()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def bounded_read(path, limit=MAX_INPUT_BYTES):
    with Path(path).open('rb') as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, 'input exceeds byte bound; refusing truncated history')
    return data


def decode(data, compressed=False):
    if compressed:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            data = stream.read(MAX_INPUT_BYTES + 1)
        require(len(data) <= MAX_INPUT_BYTES, 'decompressed input exceeds byte bound')
    return data.decode('utf-8')


def json_bytes(data, compressed=False):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(decode(data, compressed), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'nonfinite JSON number'))


def verify_source_manifest(source, manifest):
    identity = source['candidate_identity']
    require(identity['commit'] == SOURCE_COMMIT == manifest['commit'], 'unreviewed source commit')
    require(manifest['archive_sha256'] == identity['source_archive_sha256'], 'source archive mismatch')
    original = manifest['original_source_pins']
    relative = manifest['pins_relative_to_frozen_source']
    require(sha256(json.dumps(original, sort_keys=True).encode()) == identity['source_digest']
            == manifest['source_pins_digest'], 'source manifest digest mismatch')
    require(sha256(json.dumps(relative, sort_keys=True).encode()) == manifest['source_pins_digest_relative'],
            'relative source manifest digest mismatch')
    roots = [name[:-len('Cargo.toml')] for name in original
             if name.endswith('/Cargo.toml') and original[name] == relative['Cargo.toml']]
    require(len(roots) == 1, 'ambiguous source root')
    require({roots[0] + name: digest for name, digest in relative.items()} == original,
            'relative/original source manifest disagreement')
    require(len(relative) == identity['source_members'] == manifest['file_count'], 'source member count mismatch')
    for pins in (source['source_pins'], BINDING_PINS):
        for name, digest in pins.items():
            require(relative.get(name) == digest, 'unreviewed immutable binding: ' + name)
    return dict(scope='source_manifest_verified_immutable_binding_not_exported_canonical_ids',
                source_commit=identity['commit'], source_archive_sha256=identity['source_archive_sha256'],
                source_digest=identity['source_digest'], binding_pins=BINDING_PINS)


def load_public_archive(directory, arm):
    require(arm in COLLECTOR_PINS, 'unsupported arm')
    directory = Path(directory)
    manifest_data = bounded_read(directory / 'manifest.json')
    manifest = json_bytes(manifest_data)
    require(manifest['schema'] == 'ferrule.overlap.archive.v1', 'unsupported archive schema')
    inputs = {}
    for name in ('source.json', arm + '-proof.json', arm + '-nvtx.csv.gz'):
        data = bounded_read(directory / name)
        require(manifest['files'][name] == dict(bytes=len(data), sha256=sha256(data)),
                'archive digest/length mismatch: ' + name)
        inputs[name] = data
    source = json_bytes(inputs['source.json'])
    proof = json_bytes(inputs[arm + '-proof.json'])
    require(source['schema'] == 'ferrule.overlap.public-source.v1'
            and proof['schema'] == 'ferrule.overlap.public-projection.v1', 'unsupported evidence schema')
    require(proof['arm'] == arm and proof['final_scope']['scoped_overlap_accepted'] is True,
            'capture not accepted in its limited scope')
    identity, binding = source['candidate_identity'], proof['source_binding']
    require(identity['commit'] == SOURCE_COMMIT == binding['source_commit']
            and identity['cli_binary_sha256'] == binding['cli_sha256'], 'capture/source binding mismatch')
    require(proof['original_files']['collector_manifest']['sha256'] == COLLECTOR_PINS[arm],
            'unreviewed collector precision/budget binding')
    models = [role for role in proof['process_roles'] if role['role'] == 'model']
    require(len(models) == 1 and models[0]['executable_sha256'] == binding['cli_sha256']
            and models[0]['source_binding_sha256'] == source['original_files']['candidate-binding.json']['sha256'],
            'model process/source binding mismatch')
    require(proof['ledger']['status'] == 'closed' and proof['qa']['status'] == 'passed', 'unclosed/unqualified trace')
    text = decode(inputs[arm + '-nvtx.csv.gz'], compressed=True)
    reader = csv.DictReader(io.StringIO(text))
    require(reader.fieldnames == NVTX_FIELDS, 'requires full NVTX CSV, not a summary or invented key list')
    rows = []
    for row in reader:
        require(len(rows) < MAX_ROWS, 'row bound exceeded; refusing truncated history')
        require(None not in row and None not in row.values(), 'incomplete CSV row')
        rows.append(row)
    provenance = dict(archive_manifest_sha256=sha256(manifest_data),
                      inputs={name: sha256(data) for name, data in inputs.items()},
                      collector_manifest_sha256=COLLECTOR_PINS[arm],
                      key_scope='source_qualified_logical_reconstruction_not_full_canonical_GPU_key',
                      precision=PRECISION, canonical_roles=ROLES,
                      source_qualifier='software_archive_and_frozen_image_not_exported_checkpoint_digest',
                      full_trace_accepted=proof['final_scope']['full_trace_accepted'],
                      universal_trace_completeness_established=False)
    return rows, source, proof, models[0]['globalPid'], provenance


def parse_demands(rows, source, proof, model_pid, layer_count=40):
    """Reject incomplete/ambiguous ranges; never infer demand from upload/read rows."""
    require(0 < layer_count <= 4096, 'invalid declared layer count')
    by_kind = {kind: [] for kind in ('layer.expert', 'layer.expert.upload', 'read_subset')}
    row_ids, ranges = set(), set()
    binding = proof['source_binding']
    image, generation = binding['image_generation'], binding['source_generation']
    require(type(image) is int and image > 0 and type(generation) is int and generation > 0,
            'invalid frozen image/source generation')
    for raw in rows:
        require(raw['kind'] in by_kind, 'unexpected NVTX kind')
        row = {name: (raw[name] if name == 'kind' else None if raw[name] == 'null' else int(raw[name]))
               for name in NVTX_FIELDS}
        require(row['row_id'] not in row_ids and row['row_id'] > 0, 'duplicate/invalid source row')
        row_ids.add(row['row_id'])
        require(row['global_pid'] == model_pid and row['global_tid'] >> 24 == model_pid >> 24,
                'range outside authenticated model process')
        require(row['event_type'] == 60 and row['end_global_tid'] is None
                and row['domain_id'] > 0 and row['range_id'] > 0, 'incomplete/nonlocal NVTX range')
        rid = (row['domain_id'], row['range_id'])
        require(rid not in ranges, 'duplicate NVTX range identity')
        ranges.add(rid)
        require(0 <= row['start_ns'] < row['end_ns'] <= proof['request_duration_ns'],
                'range outside finite request window')
        if row['kind'] == 'read_subset':
            require(row['source_generation'] == generation and row['requested_bytes'] == PAYLOAD_BYTES
                    and row['operation'] > 0, 'read source/payload mismatch')
            require(all(row[f] is None for f in ('image_generation', 'layer', 'expert', 'rows')),
                    'read range cannot supply an expert identity')
        else:
            require(row['image_generation'] == image and 0 <= row['layer'] < layer_count
                    and 0 <= row['expert'] < 256 and row['rows'] > 0, 'expert identity mismatch')
            require(all(row[f] is None for f in ('operation', 'source_generation', 'requested_bytes')),
                    'unexpected expert/read identity mixture')
        by_kind[row['kind']].append(row)
    counts = {kind: len(values) for kind, values in by_kind.items()}
    require(counts == proof['metrics']['nvtx_range_counts'], 'NVTX count mismatch')
    n = counts['layer.expert']
    require(n > 0 and counts['layer.expert.upload'] == counts['read_subset'] == n,
            'this audited version requires one proved read and upload per demand')
    stats = proof['ledger']['stats']
    for name in ('verified_reads', 'submitted_reads', 'completed_reads', 'demanded_reads'):
        require(stats[name] == n, 'ledger count mismatch: ' + name)
    for name in ('failed_reads', 'unknown_reads', 'unknown_bytes', 'resident_skips', 'materialization_failures'):
        require(stats[name] == 0, 'incomplete demand ledger: ' + name)
    require(stats['proved_read_bytes'] == n * PAYLOAD_BYTES, 'proved payload byte mismatch')
    reads = sorted(by_kind['read_subset'], key=lambda row: row['start_ns'])
    require([r['operation'] for r in reads] == list(range(1, n + 1)), 'incomplete read operations')
    demand_rows = sorted(by_kind['layer.expert'], key=lambda row: (row['start_ns'], row['row_id']))
    uploads = sorted(by_kind['layer.expert.upload'], key=lambda row: (row['start_ns'], row['row_id']))
    require(len({r['global_tid'] for r in demand_rows + uploads}) == 1, 'concurrent demand owners unsupported')
    keys, batches = [], []
    previous_end = -1
    for row, upload in zip(demand_rows, uploads):
        require(all(row[f] == upload[f] for f in ('image_generation', 'layer', 'expert', 'rows')),
                'upload/demand identity or order mismatch')
        require(previous_end <= upload['start_ns'] < upload['end_ns'] <= row['start_ns'],
                'overlapping/reordered expert execution')
        previous_end = row['end_ns']
        key = (source['candidate_identity']['source_archive_sha256'], image, PRECISION,
               row['layer'], row['expert'], ROLES)
        if not batches or batches[-1][0][3] != row['layer']:
            batches.append([])
        require(key not in batches[-1], 'duplicate expert in a routed layer selection')
        batches[-1].append(key)
        keys.append(key)
    require(len(batches) == stats['sequences_started'] == stats['sequences_finished'],
            'ambiguous layer boundaries')
    require(len(batches) == layer_count * proof['ledger']['invocation_count'], 'incomplete invocation layers')
    require([b[0][3] for b in batches] == list(range(layer_count)) * proof['ledger']['invocation_count'],
            'noncanonical layer traversal')
    return keys, batches


def trace_diagnostics(keys):
    last, distances = {}, []
    tree = [0] * (len(keys) + 1)

    def prefix(index):
        total = 0
        while index:
            total += tree[index]
            index -= index & -index
        return total

    def add(index, amount):
        while index < len(tree):
            tree[index] += amount
            index += index & -index

    for index, key in enumerate(keys, 1):
        if key in last:
            # One active position per distinct key: exact distinct-key stack distance.
            distances.append(prefix(index - 1) - prefix(last[key]))
            add(last[key], -1)
        add(index, 1)
        last[key] = index
    return dict(demands=len(keys), distinct_keys=len(last), repeated_demands=len(distances),
                reuse_distance_min=min(distances, default=None), reuse_distance_max=max(distances, default=None),
                history_scope='complete_observed_window_only_not_cold_since_owner_creation')


class FrequencySketch:
    __slots__ = ('counts', 'age')
    WIDTH, DEPTH, PERIOD, SEED = 2048, 4, 4096, 0

    def __init__(self):
        self.counts = bytearray(self.WIDTH * self.DEPTH)
        self.age = 0

    def indices(self, key):
        data = json.dumps(key, separators=(',', ':')).encode()
        digest = hashlib.blake2b(data, digest_size=16, key=self.SEED.to_bytes(8, 'little')).digest()
        return [row * self.WIDTH + int.from_bytes(digest[row * 4:row * 4 + 4], 'little') % self.WIDTH
                for row in range(self.DEPTH)]

    def prior(self, key):
        return min(self.counts[index] for index in self.indices(key))

    def increment(self, key):
        for index in self.indices(key):
            self.counts[index] = min(255, self.counts[index] + 1)
        self.age += 1
        if self.age == self.PERIOD:
            for index, count in enumerate(self.counts):
                self.counts[index] = count // 2
            self.age = 0


def deep_size(value, seen=None):
    """Python implementation footprint, not a prediction of Rust B-tree allocation."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, FrequencySketch):
        size += deep_size(value.counts, seen) + deep_size(value.age, seen)
    elif isinstance(value, (tuple, list, set)):
        size += sum(deep_size(item, seen) for item in value)
    return size


def replay(batches, policy, slots=64, max_bytes=320 * MIB, scratch_bytes=8 * MIB,
           payload_bytes=PAYLOAD_BYTES, layer_count=40):
    require(policy in POLICIES and type(slots) is int and 0 < slots <= 65536, 'invalid policy/slots')
    require(0 <= scratch_bytes < max_bytes and payload_bytes > 0 and 0 < layer_count <= 4096,
            'invalid byte/layer budget')
    effective = min(slots, (max_bytes - scratch_bytes) // payload_bytes)
    transient = int(policy != 'global_lru')
    require(effective > transient, 'budget cannot fit persistent plus charged transient payload')
    persistent = effective - transient
    quotas = [persistent // layer_count + int(layer < persistent % layer_count) for layer in range(layer_count)]
    resident, sketch = OrderedDict(), FrequencySketch() if policy == 'history_frequency_bypass' else None
    counters = Counter(hits=0, misses=0, bypasses=0, evictions=0, selected_at_start_hits=0,
                       evicted_selected=0, evicted_future_hint=0, entry_limit_evictions=0, byte_limit_evictions=0,
                       layer_quota_evictions=0, frequency_replacements=0)
    peak_slots, peak_metadata, peaks = 0, 0, []
    by_layer = [Counter(hits=0, misses=0) for _ in range(layer_count)]
    for batch in batches:
        require(batch and len(batch) <= 256 and len(set(batch)) == len(batch), 'invalid layer selection')
        layer = batch[0][3]
        require(0 <= layer < layer_count and all(key[3] == layer for key in batch), 'mixed layer selection')
        selected = set(batch)
        counters['selected_at_start_hits'] += sum(key in resident for key in batch)
        for index, key in enumerate(batch):
            hit, victim = key in resident, None
            if hit:
                resident.move_to_end(key)
            else:
                if policy == 'global_lru':
                    if len(resident) == persistent:
                        victim = next(iter(resident))
                        counters['entry_limit_evictions'] += int(effective == slots)
                        counters['byte_limit_evictions'] += int((len(resident) + 1) * payload_bytes + scratch_bytes > max_bytes)
                    admit = True
                elif policy == 'layer_protected_bypass':
                    local = [k for k in resident if k[3] == layer]
                    eligible = [k for k in local if k not in selected]
                    admit = len(local) < quotas[layer] or bool(eligible)
                    if len(local) >= quotas[layer] and eligible:
                        victim = eligible[0]
                        counters['layer_quota_evictions'] += 1
                else:
                    admit = len(resident) < persistent
                    if not admit:
                        eligible = [k for k in resident if k not in selected]
                        if eligible:
                            candidate = min(eligible, key=sketch.prior)
                            # Candidate and victim BOTH use counts before this demand.
                            admit = sketch.prior(key) > sketch.prior(candidate)
                            if admit:
                                victim = candidate
                                counters['frequency_replacements'] += 1
                # Bypass payload is charged even when it will not be retained.
                # Promotion removes a completed victim before transferring the transient lease.
                if transient:
                    peak_slots = max(peak_slots, len(resident) + 1)
                if victim is not None:
                    counters['evictions'] += 1
                    counters['evicted_selected'] += int(victim in selected)
                    counters['evicted_future_hint'] += int(index + 1 < len(batch) and victim == batch[index + 1])
                    del resident[victim]
                if admit:
                    resident[key] = None
                else:
                    counters['bypasses'] += 1
            counters['hits' if hit else 'misses'] += 1
            by_layer[layer]['hits' if hit else 'misses'] += 1
            if sketch:
                sketch.increment(key)
            peak_slots = max(peak_slots, len(resident))
            require(peak_slots <= effective and len(resident) <= persistent, 'slot accounting violated')
            peak_metadata = max(peak_metadata, deep_size((resident, selected, key, quotas, sketch)))
        peaks.append(counters['misses'])
    charged = peak_slots * payload_bytes + scratch_bytes
    require(charged <= max_bytes, 'byte accounting violated')
    return dict(policy=policy, requested_slots=slots, effective_slots=effective,
                full_requested_capacity_fits=slots * payload_bytes + scratch_bytes <= max_bytes,
                persistent_slots=persistent, transient_slots=transient, max_bytes=max_bytes,
                scratch_bytes=scratch_bytes, peak_charged_payload_and_scratch_bytes=charged,
                **counters, estimated_expert_reads=counters['misses'],
                estimated_requested_read_bytes=counters['misses'] * payload_bytes,
                estimated_uploaded_bytes=counters['misses'] * payload_bytes,
                zero_h2d_retained_hits=counters['hits'],
                layer_results=[dict(layer=layer, **counts) for layer, counts in enumerate(by_layer)],
                cumulative_misses_after_layer=peaks,
                policy_metadata_python_peak_bytes=peak_metadata,
                metadata_scope='Python_state_snapshot_keys_LRU_selection_quotas_transient_sketch_not_temporary_lists_trace_or_Rust_allocator',
                eviction_selection_scope='selected_current_layer_and_immediate_next_expert_hint_only',
                layer_quota_rule='equal_share_with_remainder_to_low_layer_ids' if policy == 'layer_protected_bypass' else None,
                max_history_bytes=deep_size(sketch) if sketch else 0,
                history_sketch_storage_bytes=len(sketch.counts) if sketch else 0,
                history_metadata_scope='Python_fixed_sketch_object_bytearray_and_bounded_age_integer' if sketch else 'none',
                frequency=dict(width=sketch.WIDTH, depth=sketch.DEPTH, seed=sketch.SEED,
                               decay_every=sketch.PERIOD, cap=255, tie='bypass', compare='before_increment') if sketch else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(COLLECTOR_PINS), default='lookahead')
    parser.add_argument('--capacities', nargs='+', type=int, default=[64, 128, 320])
    args = parser.parse_args(argv)
    try:
        require(0 < len(args.capacities) <= 16 and len(set(args.capacities)) == len(args.capacities),
                'capacities must be a bounded unique list')
        rows, source, proof, pid, provenance = load_public_archive(args.archive, args.arm)
        data = bounded_read(args.source_manifest)
        binding = verify_source_manifest(source, json_bytes(data, args.source_manifest.suffix == '.gz'))
        keys, batches = parse_demands(rows, source, proof, pid)
        results = [replay(batches, policy, slots) for slots in args.capacities for policy in POLICIES]
        for result in results:
            endpoints = result['cumulative_misses_after_layer'][39::40]
            result['estimated_reads_per_invocation'] = [end - start for start, end in
                                                       zip([0] + endpoints[:-1], endpoints)]
        best = {str(slots): min((r for r in results if r['requested_slots'] == slots),
                               key=lambda r: (r['misses'], r['policy_metadata_python_peak_bytes']))['policy']
                for slots in args.capacities}
        tradeoffs = []
        for policy in POLICIES:
            ordered = sorted((r for r in results if r['policy'] == policy), key=lambda r: r['requested_slots'])
            for smaller, larger in zip(ordered, ordered[1:]):
                tradeoffs.append(dict(policy=policy, smaller_slots=smaller['requested_slots'],
                    larger_slots=larger['requested_slots'], miss_change=larger['misses'] - smaller['misses'],
                    layers_with_more_misses=[b['layer'] for a, b in zip(smaller['layer_results'], larger['layer_results'])
                                            if b['misses'] > a['misses']]))
        policy_tradeoffs = []
        for slots in args.capacities:
            candidates = {r['policy']: r for r in results if r['requested_slots'] == slots}
            a, b = candidates['layer_protected_bypass'], candidates['history_frequency_bypass']
            policy_tradeoffs.append(dict(slots=slots, reference=a['policy'], candidate=b['policy'],
                miss_change=b['misses'] - a['misses'],
                layers_with_more_misses=[y['layer'] for x, y in zip(a['layer_results'], b['layer_results'])
                                        if y['misses'] > x['misses']],
                invocations_with_more_misses=[i for i, (x, y) in
                    enumerate(zip(a['estimated_reads_per_invocation'], b['estimated_reads_per_invocation'])) if y > x]))
        report = dict(schema='ferrule.expert-cache-replay.v1', arm=args.arm,
                      evidence='analytical_CPU_counterfactual_not_measured_IO_H2D_or_GPU_benefit',
                      initial_state='empty_at_observed_window_start_not_proven_owner_cold',
                      completion_model='serial_completed_consumers_assumption_not_NVTX_completion_proof',
                      horizon='current_routed_layer_only_no_future_layers_or_Belady',
                      kernel_order='unchanged', client_cache_model='one_same_source_payload_request_per_miss_no_OS_or_NAS_cache_model',
                      scope='Qwen35_only_DeepSeekV4_untested', trace=trace_diagnostics(keys),
                      metadata_interpreter=dict(implementation=sys.implementation.name, version=sys.version),
                      provenance={**provenance, **binding, 'source_manifest_sha256': sha256(data)},
                      best_by_same_budget=best, capacity_tradeoffs=tradeoffs,
                      policy_tradeoffs=policy_tradeoffs, results=results,
                      offline_input_limits=dict(max_file_or_expanded_bytes=MAX_INPUT_BYTES, max_nvtx_rows=MAX_ROWS,
                                                overflow='reject_not_truncate'),
                      caveats=['Layer quotas and aged frequency admission are not stack policies; larger capacity need not improve every phase/layer.',
                               'Trace starts inside an existing owner; absent pre-window history is not coldness.',
                               'A future CUDA bypass needs explicit completed-entry retirement; no runtime policy is installed here.'])
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, EOFError, csv.Error) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
