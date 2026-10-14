"""CPU-only validation of the prepared GDN benchmark; never invokes CUDA work."""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

import performance as bench

parser = argparse.ArgumentParser()
parser.add_argument('--prepared', required=True, type=Path)
options, remaining = parser.parse_known_args()
PREPARED = options.prepared.resolve()


class PreparationTests(unittest.TestCase):
    def test_all_pins_and_production_unchanged(self):
        m = json.loads((PREPARED / 'prepared.json').read_text())
        self.assertFalse(m['gpu_execution'])
        self.assertEqual(m['status'], 'prepared_not_executed')
        for path, pin in m['source_pins'].items():
            self.assertEqual(bench.sha(path), pin, path)
        for path, pin in m['artifact_pins'].items():
            self.assertEqual(bench.sha(PREPARED/path), pin, path)
        self.assertEqual(bench.sha(bench.NATIVE/'core/recurrent_bf16.cuh'),
                         '15a6e71ea871e2bdf37be48fa65f3afcfdb874c2e54d3847f430ee87b3a4b844')
        archive = bench.EVIDENCE/'baseline-source/core/recurrent_bf16.cuh'
        self.assertEqual(bench.sha(archive), bench.PRIOR_SHA)
        self.assertEqual((PREPARED/'prior.cuh').read_text(), archive.read_text().replace(
            'hf_bf16', 'prior_hf_bf16').replace('ferrule_core_bf16_', 'prior_core_bf16_'))

    def test_excluded_threshold_and_protocol(self):
        checks = json.loads((PREPARED/'cpu-checks.json').read_text())
        expected = [[1, 4, dk, dv] for dv in [4, 28] for dk in [64, 65, 128, 511, 512]]
        self.assertEqual(checks['below_threshold_cases'], expected)
        self.assertIn('cases=107 below-threshold=10', checks['stability']['stdout'])
        p = json.loads(checks['benchmark']['stdout'])
        self.assertEqual(p['rows'], [1, 5, 32])
        self.assertEqual((p['key_dim'], p['value_dim'], p['value_heads']), (128, 128, 32))
        self.assertEqual((p['warmups_per_variant'], p['paired_rounds'], p['launches_per_sample']), (20, 30, 8))

    def test_fixture_shapes_and_row32_cycling(self):
        for rows in [1, 5, 32]:
            p = PREPARED/('rows'+str(rows))
            self.assertEqual((p/'qkv.bf16').stat().st_size, rows*8192*2)
            for name in ['a.bf16', 'b.bf16']:
                self.assertEqual((p/name).stat().st_size, rows*32*2)
            self.assertEqual((p/'state.f32').stat().st_size, 32*128*128*4)
            if rows != 1:
                self.assertFalse(any((p/'state.f32').read_bytes()))
        for name, width in [('qkv.bf16', 8192), ('a.bf16', 32), ('b.bf16', 32)]:
            self.assertEqual((PREPARED/'rows32'/name).read_bytes(),
                             bench.tile((PREPARED/'rows5'/name).read_bytes(), width, 32))

    def test_bf16_conversion_rejects_rounding_and_nonfinite(self):
        self.assertEqual(bench.bf16(struct.pack('<II', 0x3f800000, 0x80000000)),
                         struct.pack('<HH', 0x3f80, 0x8000))
        for bits in [0x3f800001, 0x7f800000, 0x7fc00000]:
            with self.assertRaises(ValueError):
                bench.bf16(struct.pack('<I', bits))
        with self.assertRaises(ValueError):
            bench.tile(bytes(8), 1, 32)

    def test_gpu_mode_refuses_hidden_or_missing_authorization(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(bench, 'execute') as execute:
            evidence = Path(temporary)/'never-created'
            with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
                with self.assertRaises(ValueError):
                    bench.run(PREPARED, evidence, True)
            with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
                with self.assertRaises(ValueError):
                    bench.run(PREPARED, evidence, False)
            execute.assert_not_called()
            self.assertFalse(evidence.exists())


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0], *remaining])
