"""No CUDA imports, devices, subprocesses or model execution."""
import json
from pathlib import Path
import unittest

import profile_shape_audit as audit


def row(name, index, grid=2048, stream=22, device=0, context=1):
    return dict(name=name, start=index * 10, end=index * 10 + 3,
                correlationId=index, globalPid=(1 << 48) + (167217 << 24),
                deviceId=device, contextId=context, streamId=stream,
                gridX=grid, gridY=1, gridZ=1, blockX=256, blockY=1, blockZ=1)


def pair():
    return [row(audit.PACK, 0, 8), row(audit.DECODE, 1), row(audit.DECODE, 2),
            row('ampere_bf16_example', 3),
            row('ferrule::cuda::cutlass::bf16_gemmex::split_pair(x)', 4, 1),
            row('ferrule::cuda::core::bf16_boundary_kernel(x)', 5, 1),
            row(audit.PACK, 6, 1), row(audit.DECODE, 7),
            row('ampere_bf16_example', 8)]


class ShapeAudit(unittest.TestCase):
    def test_ceiling_is_an_interval_not_an_exact_size(self):
        self.assertEqual(audit.element_bounds(1), [1, 256])
        self.assertEqual(audit.element_bounds(2048), [524033, 524288])

    def test_capped_grid_has_no_upper_bound(self):
        self.assertEqual(audit.element_bounds(65535), [16776705, None])
        for invalid in (0, -1, 65536):
            with self.assertRaises(ValueError):
                audit.element_bounds(invalid)

    def test_paired_sequence_reconciles_three_decodes_two_packs(self):
        result = audit.analyze_kernels(pair())
        roles = result['motif_roles']
        self.assertEqual(roles['routed_gate_up_motif'], dict(count=2, sum_ns=6))
        self.assertEqual(roles['routed_down_motif'], dict(count=1, sum_ns=3))
        self.assertEqual(roles['routed_pack_motif'], dict(count=2, sum_ns=6))
        self.assertEqual(result['routed_m_histogram'], {1: 1})

    def test_never_stitch_devices_contexts_streams_or_processes(self):
        for field in ('globalPid', 'deviceId', 'contextId', 'streamId'):
            rows = pair()
            for r in rows[4:]:
                r[field] += 1
            self.assertNotIn('routed_down_motif', audit.analyze_kernels(rows)['motif_roles'])

    def test_missing_split_is_not_a_routed_attribution(self):
        rows = pair()
        del rows[4]
        result = audit.analyze_kernels(rows)
        self.assertNotIn('routed_gate_up_motif', result['motif_roles'])
        self.assertEqual(result['unmatched_pair_correlations'], [0])

    def test_missing_matrix_is_not_a_routed_attribution(self):
        rows = pair()
        del rows[3]
        self.assertNotIn('routed_gate_up_motif', audit.analyze_kernels(rows)['motif_roles'])

    def test_equal_launches_do_not_prove_equal_pointers(self):
        result = audit.analyze_kernels([row(audit.DECODE, i) for i in range(10)])
        stats = result['repetition'][audit.DECODE]
        self.assertEqual(stats['unique_launch_signatures'], 1)
        self.assertIsNone(stats['pointer_repetition_rate'])
        self.assertIsNone(stats['tensor_shape_repetition_rate'])

    def test_distinct_transpose_shapes_share_a_launch_signature(self):
        self.assertEqual(256 * 2048, 2048 * 256)
        self.assertEqual(audit.signature(row(audit.DECODE, 0)),
                         audit.signature(row(audit.DECODE, 1)))

    def test_official_report_totals_and_caveats(self):
        path = Path(__file__).parent / 'runs/tp2-official-full-profile-20261010-001/profile-shape-audit.json'
        report = json.loads(path.read_text())
        self.assertEqual(report['request_window_ns'], list(audit.WINDOW))
        self.assertEqual(report['status'], 'observed_correlation_closed')
        self.assertFalse(report['full_trace_accepted'])
        self.assertFalse(report['paired_api_activity_loss_excluded'])
        groups = report['observed']['groups']
        for name, count, ns in ((audit.DECODE, 26592, 270507218), (audit.PACK, 19228, 30012786)):
            selected = [g for g in groups if g['name'] == name]
            self.assertEqual(sum(g['count'] for g in selected), count)
            self.assertEqual(sum(g['sum_ns'] for g in selected), ns)
        roles = report['observed']['motif_roles']
        decodes = [v for k, v in roles.items() if 'pack' not in k]
        self.assertEqual(sum(v['count'] for v in decodes), 26592)
        self.assertEqual(sum(v['sum_ns'] for v in decodes), 270507218)
        self.assertEqual(sum(report['observed']['routed_m_histogram'].values()), 7364)
        self.assertTrue(any(not p['matches_admission'] for p in report['source_pins']))


if __name__ == '__main__':
    unittest.main()
