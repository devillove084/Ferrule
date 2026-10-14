"""CPU-only integrity and coverage checks for the supplemental Capital fixture."""
import hashlib
import json
import os
from pathlib import Path
import unittest
import capital_prepare as exporter

FIXTURE=exporter.ROOT/'target/validation/capital-projections-20261006/fixture-v2'
PIN='0381178a03f7c848a73b153ddf3cec3284c316814ef3007f3f7a8043f96e984f'

class CapitalFixture(unittest.TestCase):
    def test_original_oracle_unchanged(self):
        for name,pin in exporter.PINS.items():
            self.assertEqual(exporter.common.util.sha256(exporter.SOURCE/name),pin)
        m=json.loads((exporter.SOURCE/'hf-manifest.json').read_text())
        self.assertEqual(len(m['checks']),30)
        self.assertTrue(all(x['unequal']==0 for x in m['checks']))

    def test_frozen_payloads_and_independent_coverage(self):
        self.assertEqual(exporter.common.util.sha256(FIXTURE/'SHA256SUMS'),PIN)
        for line in (FIXTURE/'SHA256SUMS').read_text().splitlines():
            pin,name=line.split();self.assertEqual(exporter.common.util.sha256(FIXTURE/name),pin)
        m=json.loads((FIXTURE/'manifest.json').read_text())
        self.assertEqual(len(m['cases']),114);self.assertEqual(len(m['routed_experts']),36)
        self.assertTrue(m['supplemental_experts_aggregate_exact'])
        events=[json.loads(x) for x in (exporter.SOURCE/'producer/capital/prefill/events.jsonl').read_text().splitlines()]
        expected={e['weight']['name'] for e in events if e.get('layer')==0 and e['event']=='linear.input' and e.get('scale')}
        self.assertEqual({c['weight'] for c in m['cases']},expected)
        self.assertEqual(len({c['id'] for c in m['cases']}),114)
        for c in m['cases']:
            for suffix,size in [('input.bf16',c['m']*c['k']*2),('hf.bf16',c['m']*c['n']*2),('decoded.bf16',c['n']*c['k']*2),('native.f32',c['m']*c['n']*4)]:
                self.assertEqual((FIXTURE/(c['id']+'.'+suffix)).stat().st_size,size)

    def test_epilogue_diagnostic_is_not_oracle(self):
        m=json.loads((FIXTURE/'manifest.json').read_text())
        self.assertEqual(m['f32_epilogue_differences']['(5, 4096, 2048)'],17)
        self.assertEqual(m['hf_sha256'],exporter.PINS['hf-layer01.safetensors'])

if __name__=='__main__': unittest.main()
