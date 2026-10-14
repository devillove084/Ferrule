"""CPU-only post-processing of the existing request-6 export; never a collector.

Launch signatures are not tensor identities. Source-qualified logical bytes are
not measured DRAM traffic, and adjacent exported kernels do not prove dataflow.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3

from profile_trace_audit_v2 import process_id

WINDOW = (2622335777, 5112115777)
PID = 167217
PREFIX = 'ferrule::cuda::cutlass::numeric_fp8::'
PACK = PREFIX + 'pack_bf16_activation(FerruleCutlassNumericFp8Args)'
DECODE = PREFIX + 'decode_weights(FerruleCutlassNumericFp8Args, unsigned int, unsigned int)'
DIMS = ('gridX', 'gridY', 'gridZ', 'blockX', 'blockY', 'blockZ')


def element_bounds(grid):
    """Invert numeric_fp8::blocks, including both rounding and saturation."""
    if not 1 <= grid <= 65535:
        raise ValueError('Not a numeric_fp8::blocks launch')
    return [(grid - 1) * 256 + 1, grid * 256 if grid < 65535 else None]


def signature(row):
    return (row['name'], *(row[d] for d in DIMS))


def stream_key(row):
    return tuple(row[k] for k in ('globalPid', 'deviceId', 'contextId', 'streamId'))


def matrix_kernel(row):
    return any(s in row['name'] for s in (
        'internal::gemvx::', 'ampere_bf16_', 'cutlass::Kernel2<',
        'cublasLt::splitKreduce_kernel<', 'dot_kernel<', 'reduce_1Block_kernel<'))


def analyze_kernels(rows):
    streams = defaultdict(list)
    for row in rows:
        streams[stream_key(row)].append(row)
    groups = defaultdict(list)
    sequences = Counter()
    roles = defaultdict(list)
    routed_shapes = Counter()
    examples = []
    unmatched_pairs = []
    paired_ids = set()
    shared_ids = set()
    for key, kernels in sorted(streams.items()):
        kernels.sort(key=lambda r: (r['start'], r['end'], r['correlationId']))
        for i, row in enumerate(kernels):
            if row['name'] in (PACK, DECODE):
                groups[signature(row)].append(row)
            if row['name'] != PACK:
                continue
            j = i + 1
            decodes = []
            while j < len(kernels) and kernels[j]['name'] == DECODE:
                decodes.append(kernels[j])
                j += 1
            math_start = j
            while j < len(kernels) and matrix_kernel(kernels[j]):
                j += 1
            sequences[(row['gridX'], tuple(d['gridX'] for d in decodes),
                       tuple(signature(r) for r in kernels[math_start:j]))] += 1
            if len(decodes) != 2:
                continue
            valid = (j > math_start and j + 3 < len(kernels)
                     and '::bf16_gemmex::split_pair(' in kernels[j]['name']
                     and '::bf16_boundary_kernel(' in kernels[j + 1]['name']
                     and kernels[j + 2]['name'] == PACK
                     and kernels[j + 3]['name'] == DECODE
                     and all(d['gridX'] == 2048 for d in decodes + [kernels[j + 3]])
                     and row['gridX'] == 8 * kernels[j + 2]['gridX'])
            if not valid:
                unmatched_pairs.append(row['correlationId'])
                continue
            paired_ids.add((key, i))
            roles['routed_gate_up_motif'].extend(decodes)
            roles['routed_down_motif'].append(kernels[j + 3])
            roles['routed_pack_motif'].extend([row, kernels[j + 2]])
            routed_shapes[kernels[j + 2]['gridX']] += 1
            if len(examples) < 4:
                examples.append(dict(stream=list(key), correlation_ids=[
                    r['correlationId'] for r in kernels[i:j + 4]],
                    kernel_names=[r['name'] for r in kernels[i:j + 4]]))
        # Shared MLP: three singles after the router and before its first routed
        # pair. Stop at another router; never bridge a missing whole layer.
        for i, row in enumerate(kernels):
            if '::full_router_kernel<' not in row['name']:
                continue
            between = []
            for j in range(i + 1, len(kernels)):
                if '::full_router_kernel<' in kernels[j]['name']:
                    break
                if (key, j) in paired_ids:
                    dec = [r for r in between if r['name'] == DECODE]
                    pack = [r for r in between if r['name'] == PACK]
                    if (len(dec) == len(pack) == 3
                            and all(r['gridX'] == 2048 for r in dec)
                            and pack[0]['gridX'] == pack[1]['gridX'] == 8 * pack[2]['gridX']):
                        roles['shared_three_single_motif'].extend(dec)
                        roles['shared_pack_motif'].extend(pack)
                        shared_ids.update((stream_key(r), r['correlationId']) for r in dec)
                    break
                between.append(kernels[j])
    claimed = {(stream_key(r), r['correlationId']) for name, rs in roles.items()
               if 'pack' not in name for r in rs}
    for row in rows:
        if row['name'] == DECODE and (stream_key(row), row['correlationId']) not in claimed:
            roles['unassigned_decode_grid_' + str(row['gridX'])].append(row)
    output_groups = []
    for sig, rs in groups.items():
        owner_counts = Counter((r['deviceId'], r['contextId'], r['streamId'], r.get('owner_tid')) for r in rs)
        output_groups.append(dict(name=sig[0], launch_dims=list(sig[1:]), count=len(rs),
            sum_ns=sum(r['end'] - r['start'] for r in rs), element_bounds=element_bounds(sig[1]),
            owner_stream_counts=[dict(device=d, context=c, stream=s, owner_tid=t, count=n)
                                for (d, c, s, t), n in sorted(owner_counts.items())]))
    return dict(groups=sorted(output_groups, key=lambda r: -r['sum_ns']),
        motifs=[dict(pack_grid=p, decode_grids=list(d), matrix_launches=[list(x) for x in m], count=n)
                for (p, d, m), n in sorted(sequences.items(), key=lambda item: -item[1])],
        motif_roles={name: dict(count=len(rs), sum_ns=sum(r['end'] - r['start'] for r in rs))
                     for name, rs in sorted(roles.items())},
        routed_m_histogram=dict(sorted(routed_shapes.items())),
        routed_examples=examples, unmatched_pair_correlations=unmatched_pairs,
        repetition={name: dict(count=sum(r['name'] == name for r in rows),
            unique_launch_signatures=sum(sig[0] == name for sig in groups),
            pointer_repetition_rate=None, tensor_shape_repetition_rate=None,
            limitation='Launch-signature recurrence only; no weight/activation addresses or kernel arguments exported.')
            for name in (PACK, DECODE)})


def analyze_sqlite(path, window=WINDOW, pid=PID):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        names = dict(connection.execute('SELECT id, value FROM StringIds'))
        rows = [dict(r) for r in connection.execute(
            'SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?', window)
                if process_id(r['globalPid']) == pid]
        for r in rows:
            r['name'] = names[r['demangledName']]
        correlations = {r['correlationId'] for r in rows}
        launches = defaultdict(list)
        api_stats = defaultdict(lambda: Counter(count=0, sum_ns=0))
        previous = {}
        preceding = defaultdict(Counter)
        return_values = defaultdict(Counter)
        callchains = 0
        # Full-export counterpart search, not a time-limited join.
        for api in connection.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY globalTid,start'):
            if process_id(api['globalTid']) != pid:
                continue
            name = names[api['nameId']]
            if api['correlationId'] in correlations:
                launches[api['correlationId']].append(dict(api, name=name))
            if not window[0] <= api['start'] <= api['end'] <= window[1]:
                continue
            api_stats[name]['count'] += 1
            api_stats[name]['sum_ns'] += api['end'] - api['start']
            if 'Event' in name:
                preceding[name][previous.get(api['globalTid'], 'window_start')] += 1
                return_values[name][api['returnValue']] += 1
            previous[api['globalTid']] = name
            callchains += api['callchainId'] is not None
        errors = []
        for row in rows:
            apis = launches[row['correlationId']]
            if (len(apis) != 1 or apis[0]['returnValue'] != 0
                    or not apis[0]['name'].startswith(('cudaLaunchKernel', 'cuLaunchKernel'))
                    or not window[0] <= apis[0]['start'] <= apis[0]['end'] <= window[1]):
                errors.append(row['correlationId'])
            else:
                row['owner_tid'] = apis[0]['globalTid']
        if errors:
            raise ValueError(f'Kernel/API qualification failed: {errors[:10]}')
        result = analyze_kernels(rows)
        result['kernel_rows'] = len(rows)
        result['kernel_api_unique_success_same_window'] = len(rows)
        result['api_stats'] = dict(sorted(api_stats.items(), key=lambda item: -item[1]['count']))
        result['event_api_previous_call'] = {k: dict(v) for k, v in preceding.items()}
        result['event_api_return_values'] = {k: dict(v) for k, v in return_values.items()}
        result['runtime_callchain_rows'] = callchains
        result['schema_columns'] = {table: [r[1] for r in connection.execute(f'PRAGMA table_info({table})')]
            for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_MEMCPY')}
        # Check whether adjacency in the kernel-only view hides a copy/set.
        activities = defaultdict(list)
        for row in rows:
            activities[stream_key(row)].append(row)
        for kind in ('MEMCPY', 'MEMSET'):
            for row in connection.execute(f'SELECT * FROM CUPTI_ACTIVITY_KIND_{kind} WHERE start>=? AND end<=?', window):
                if process_id(row['globalPid']) == pid:
                    activities[stream_key(row)].append(dict(row, name=kind))
        next_activity = Counter()
        for stream in activities.values():
            stream.sort(key=lambda r: (r['start'], r['end'], r['correlationId']))
            for a, b in zip(stream, stream[1:]):
                if a['name'] in (PACK, DECODE):
                    next_activity[(a['name'], b['name'])] += 1
        result['immediate_next_gpu_activity'] = [dict(first=a, next=b, count=n)
            for (a, b), n in sorted(next_activity.items(), key=lambda item: -item[1])]
        result['owner_names'] = {str(r['globalTid']): names[r['nameId']]
            for r in connection.execute('SELECT * FROM ThreadNames')
            if r['globalTid'] in {row.get('owner_tid') for row in rows}}
        return result
    finally:
        connection.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze_sqlite(args.sqlite), indent=2))
