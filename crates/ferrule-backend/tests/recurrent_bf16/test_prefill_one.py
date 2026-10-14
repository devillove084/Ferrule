"""CPU admission/fixture integrity for explicit M1 chunk semantics, no GPU fallback."""
import argparse
import json
from pathlib import Path
import unittest
import numpy as np
from performance import ROOT, HERE, NATIVE, sha

p=argparse.ArgumentParser();p.add_argument('--prepared',required=True,type=Path)
args,rest=p.parse_known_args();OUT=args.prepared.resolve()


class PrefillOneCPU(unittest.TestCase):
    def test_frozen_mismatch_is_preserved_not_referenced_away(self):
        d=OUT/'actual'
        expected=np.fromfile(d/'expected.bf16',np.uint16)
        captured=np.fromfile(d/'actual.bf16',np.uint16)
        self.assertEqual(expected.size,4096)
        self.assertEqual(np.flatnonzero(expected!=captured).tolist(),[1575,1596])
        self.assertEqual((d/'expected-state.f32').read_bytes(),(d/'actual-state.f32').read_bytes())
        self.assertEqual((d/'state.f32').read_bytes(),bytes(524288*4))
        for line in (HERE/'prefill_one.sha256').read_text().splitlines():
            pin,name=line.split();self.assertEqual(sha(d/name),pin)

    def test_explicit_entry_preserves_legacy_and_descriptor(self):
        header=(NATIVE/'core/recurrent_bf16_chunk.cuh').read_text()
        abi=(NATIVE/'core/recurrent_bf16_chunk_abi.h').read_text()
        self.assertIn('return ferrule_core_bf16_chunk_launch_impl(a, false);',header)
        self.assertIn('return ferrule_core_bf16_chunk_launch_impl(a, true);',header)
        self.assertIn('a.reserved || !a.rows || (!prefill && a.rows == 1)',header)
        self.assertIn('sizeof(FerruleCoreBf16ChunkArgs) == 104',abi)
        self.assertIn('workspace_f32) == 80',abi)
        self.assertIn('stream) == 96',abi)
        op=(ROOT/'crates/ferrule-backend/src/cuda/operators/recurrent_bf16_chunk.rs').read_text()
        self.assertIn('plan.execution() == Bf16ChunkExecution::CachedRecurrent',op)
        self.assertNotIn('if l.rows == 1',op)
        self.assertIn('workspace.plan != plan',op)
        self.assertIn('submit_operator(plan.launch_count()',op)
        self.assertIn('disjoint(',op)
        self.assertIn('check_buffer_owner',op)

    def test_native_prefill_rejection_checks_ran_without_cuda(self):
        r=json.loads((OUT/'contracts.json').read_text())
        self.assertEqual(r['returncode'],0)
        self.assertIn('prefill native guards passed without CUDA calls',r['stdout'])
        source=(HERE/'prefill_one.cu').read_text()
        self.assertIn('b.reserved = 1; reject(b);',source)
        self.assertIn('b.workspace_elements = n - 1; reject(b);',source)
        self.assertIn('b.output_bf16 = a.state_f32; reject(b);',source)

    def test_prepared_sources_and_payloads_are_pinned(self):
        m=json.loads((OUT/'prepared.json').read_text())
        for name,pin in m['source_pins'].items():self.assertEqual(sha(name),pin,name)
        for name,pin in m['artifact_pins'].items():self.assertEqual(sha(OUT/name),pin,name)
        self.assertEqual(json.loads((OUT/'build.json').read_text())['returncode'],0)

    def test_independent_cases_cover_semantics_and_carry(self):
        cases=json.loads((OUT/'prepared.json').read_text())['cases']
        self.assertEqual(len(cases),10)
        self.assertEqual(sum(len(c['rows']) for c in cases),20)
        self.assertEqual(sum(c['cached'] for c in cases),4)
        for c in cases:
            if c['name'].startswith('synthetic'):
                self.assertEqual(c['rows'],[1,5,1])
                s=np.fromfile(OUT/c['name']/'state.f32',np.float32)
                self.assertTrue(np.any(s!=0))
                self.assertTrue(np.any(s.view(np.uint32)&65535))
        self.assertEqual([c['geometry'][2] for c in cases if c['name'].startswith('synthetic')],[1,17,128,511,512])


if __name__=='__main__':unittest.main(argv=[__file__,*rest])
