"""CPU evidence checks, not a CPU norm fallback or new reference."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
p=argparse.ArgumentParser();p.add_argument('--evidence',required=True,type=Path);args,rest=p.parse_known_args();BASE=args.evidence.resolve()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

class OrdinaryNormEvidence(unittest.TestCase):
    def test_actual_parameter_offset_and_affine_semantics(self):
        d=BASE/'probe';m=json.loads((d/'weight-offset.json').read_text())
        self.assertEqual(m['absolute_offset'],12170912)
        self.assertTrue(m['payload_exact'])
        x=(np.fromfile(d/'input.bf16',np.uint16).astype(np.uint32)<<16).view(np.float32)
        square=x*x
        self.assertTrue(np.array_equal(x.astype(np.float64)**2,square.astype(np.float64)))
        self.assertTrue(np.array_equal(square.view(np.uint32),np.fromfile(d/'hf-square.f32',np.uint32)))
        weight=(np.fromfile(d/'weight.bf16',np.uint16).astype(np.uint32)<<16).view(np.float32)
        self.assertTrue(np.array_equal((weight+np.float32(1)).view(np.uint32),np.fromfile(d/'hf-affine.f32',np.uint32)))
        c=json.loads((d/'comparison.json').read_text())
        self.assertEqual(c['old']['variance']['actual'],['0x3f88be2a'])
        self.assertEqual(c['old']['variance']['expected'],['0x3f88be29'])
        self.assertEqual(c['old']['affine']['unequal'],0)
        self.assertEqual(c['old']['output']['first'],[452])
        self.assertTrue(all(v['unequal']==0 for v in c['candidate'].values()))

    def test_full_ordinary_and_gated_gates(self):
        ordinary=json.loads((BASE/'prepared-v2/comparison.json').read_text())
        self.assertEqual(len(ordinary),105)
        self.assertTrue(all(v['unequal']==0 for c in ordinary.values() for v in c.values()))
        cases=json.loads((BASE/'prepared-v2/cases.json').read_text())
        self.assertEqual(sum(c['width']>4096 for c in cases),6)
        self.assertEqual(sum(c['nonfinite'] for c in cases),3)
        gated=json.loads((BASE/'gated-prepared/comparison.json').read_text())
        self.assertEqual(len(gated),137)
        self.assertTrue(all(v['unequal']==0 for c in gated.values() for k,v in c.items() if k!='old_output'))

    def test_scoped_sources_and_shared_helper(self):
        before=json.loads((BASE/'source-before.json').read_text())
        tensor='crates/ferrule-backend/native/cuda/core/tensor_ops.cuh'
        self.assertEqual(sha(ROOT/tensor),before[tensor])
        a=ROOT/'crates/ferrule-backend/native/cuda/core'
        gated=(a/'recurrent_bf16.cuh').read_text()
        self.assertIn('bf16_rms::lane_sum(a.input_bf16, a.width, row, lane, stride)',gated)
        ordinary=(a/'bf16_boundary.cuh').read_text()
        self.assertIn('bf16_rms::lane_sum(a.input, a.width, row, lane, blockDim.x)',ordinary)
        self.assertIn('const float affine = __fadd_rn(1.0f,',ordinary)
        self.assertIn('const float value = __fmul_rn(normalized, affine);',ordinary)
        model=(ROOT/'crates/ferrule-model/src/transformer/standard/cuda/bf16.rs').read_text()
        branch=model[model.index('let output = if norm.one_plus_weight()'):model.index('rows_bf16(input.shape(), arena, output)')]
        self.assertIn('offset_rms_norm_bf16_into',branch)
        self.assertNotIn('offset_rms_norm_f32_into',branch)
        self.assertIn('this.ops.rms_norm_bf16(',branch)

    def test_source_artifact_pins_and_contracts(self):
        for folder in ['prepared-v2','gated-prepared']:
            d=BASE/folder;m=json.loads((d/'prepared.json').read_text())
            for p,h in m['source_pins'].items():self.assertEqual(sha(p),h,p)
            for p,h in m['artifact_pins'].items():self.assertEqual(sha(d/p),h,p)
        self.assertEqual(json.loads((BASE/'prepared-v2/contracts.json').read_text())['returncode'],0)

    def test_authenticated_receipts_and_timing_inventory(self):
        for name in ['probe-run','suite-run','gated-run','rust-run']:
            d=BASE/name;self.assertEqual(json.loads((d/'release.json').read_text())['release'],'released')
            for p in d.glob('*-supervisor.json'):
                r=json.loads(p.read_text());h,c=r['handshake'],r['custody']
                self.assertEqual(r['returncode'],0,p)
                self.assertFalse(r['timed_out'])
                self.assertEqual(r['descendant_custody'],'proven_drained')
                self.assertTrue(c['echild'] and c['containment_verified'] and h['containment_verified'])
                self.assertEqual(h['launch_id'],c['launch_id'])
                self.assertEqual(h['helper_identity'],c['helper_identity'])
        t=json.loads((BASE/'timing.json').read_text())
        self.assertEqual(len(t['samples']),60)
        self.assertEqual({v['variant'] for v in t['samples']},{'old4','new1'})
        self.assertIn('not an equal-quality speedup',t['meaning'])

if __name__=='__main__':unittest.main(argv=[__file__,*rest])
