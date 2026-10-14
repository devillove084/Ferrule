"""CPU integrity/coverage checks only. No synthetic expected or GPU pass claim."""
import argparse
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['NVIDIA_VISIBLE_DEVICES'] = 'void'
import numpy as np
import chunk_continuation as gate

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prepared',required=True,type=Path)
args,rest=parser.parse_known_args()
OUT=args.prepared.resolve()


def bf16(path):
    return (np.fromfile(path,dtype='<u2').astype(np.uint32)<<16).view(np.float32)


class ContinuationCPU(unittest.TestCase):
    def test_pins_build_contracts_and_pending_gpu_status(self):
        m=gate.verify(OUT)
        self.assertEqual(m['status'],'CPU_PREPARED_GPU_NOT_RUN')
        self.assertEqual(json.loads((OUT/'build.json').read_text())['returncode'],0)
        contract=json.loads((OUT/'contracts.json').read_text())
        self.assertEqual(contract['returncode'],0)
        self.assertIn('no CUDA calls',contract['stdout'])
        self.assertIn('NOT generated',m['independent_expected_status'])
        self.assertFalse(list(OUT.rglob('carry-*-output.f32')))
        self.assertFalse(list(OUT.rglob('native-*')))
        for case in m['cases']:
            if not case['actual']:
                self.assertFalse(list((OUT/case['name']).glob('*output*')))

    def test_changed_input_calls_and_independent_reset_seed(self):
        for case in gate.cases():
            if case['actual']:continue
            d=OUT/case['name'];kh,hv,dk,dv=case['geometry']
            state=np.fromfile(d/'state.f32',dtype='<f4')
            reset=np.fromfile(d/'reset-state.f32',dtype='<f4')
            self.assertEqual(state.size,hv*dk*dv)
            self.assertTrue(np.isfinite(state).all())
            self.assertTrue(np.any(state>0) and np.any(state<0))
            self.assertTrue(np.any(state.view(np.uint32)&65535))
            self.assertFalse(np.array_equal(state.view(np.uint32),reset.view(np.uint32)))
            self.assertGreater(float(np.abs(state).sum()),20*float(np.abs(state.reshape(hv,dk,dv).sum(axis=1)).sum()))
            signatures=[]
            for i,rows in enumerate(case['rows']):
                step=d/('step'+str(i))
                self.assertEqual((step/'qkv.bf16').stat().st_size,rows*(2*kh*dk+hv*dv)*2)
                for name in ('a','b'):
                    self.assertEqual((step/(name+'.bf16')).stat().st_size,rows*hv*2)
                signatures.append(tuple(gate.sha(step/name) for name in gate.INPUTS))
            self.assertEqual(len(set(signatures)),len(signatures))

    def test_nonzero_nonunderflow_decay_and_multichunk(self):
        for case in gate.cases():
            if case['actual']:continue
            d=OUT/case['name'];hv=case['geometry'][1]
            al=bf16(d/'a_log.bf16');dt=bf16(d/'dt_bias.bf16')
            for i,rows in enumerate(case['rows']):
                a=bf16(d/('step'+str(i))/'a.bf16').reshape(rows,hv)
                # Input characterization, not a CPU oracle or CUDA parity claim.
                g=-np.exp(al.astype(np.float64))*np.logaddexp(0,a.astype(np.float64)+dt)
                self.assertTrue(np.all(g<-.001) and np.all(g>-.2))
                self.assertTrue(np.all((np.exp(g)>0)&(np.exp(g)<1)))
        self.assertEqual(next(c for c in gate.cases() if c['name']=='changed-dense')['rows'],[5,65,129])
        self.assertEqual(next(c for c in gate.cases() if c['name']=='dense129')['rows'],[129])

    def test_near_limit_admission_and_workspace_rows_independent(self):
        self.assertEqual(gate.workspace_elements([16,32,128,128])*4,17_850_368)
        for dk,dv in [(511,512),(512,511),(512,512)]:
            self.assertLess(gate.workspace_elements([1,2,dk,dv])*4,2147483647)
        for geometry in [[1,1,513,128],[1,1,128,513],[3,2,128,128]]:
            with self.assertRaises(ValueError):gate.workspace_elements(geometry)

    def test_actual_reference_unchanged_and_performance_opt_in(self):
        self.assertEqual(gate.sha(OUT/'actual5/retained-output.f32'),gate.sha(gate.ACTUAL/'expected-output.f32'))
        self.assertEqual(gate.sha(OUT/'actual5/retained-state.f32'),gate.sha(gate.ACTUAL/'expected-state.f32'))
        performance=[c for c in gate.cases() if c.get('performance')]
        self.assertEqual([c['rows'] for c in performance],[[5],[64],[129]])
        for c in performance:
            self.assertEqual(gate.native_command(OUT,OUT,c)[1],'continuation')
            self.assertEqual(gate.native_command(OUT,OUT,c,True)[1],'performance')
        self.assertFalse(list(OUT.rglob('performance.json')))

    def test_exact_comparator_detects_one_bit_and_nonfinite(self):
        a=np.array([1,2],np.float32);b=a.copy();b.view(np.uint32)[0]+=1
        self.assertEqual(gate.metric(a,a)['unequal'],0)
        self.assertEqual(gate.metric(a,b)['unequal'],1)
        self.assertEqual(gate.metric(a,np.array([np.inf,2],np.float32))['nonfinite'],1)
        with self.assertRaises(ValueError):gate.metric(a,a[:1])

    def test_pin_tampering_rejected_without_touching_original(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'payload').write_bytes(b'original')
            m=dict(source_pins={},artifact_pins={'payload':gate.sha(root/'payload')})
            (root/'prepared.json').write_text(json.dumps(m))
            gate.verify(root)
            (root/'payload').write_bytes(b'changed')
            with self.assertRaises(ValueError):gate.verify(root)

    def test_source_is_parsable_and_unsupervised_hf_refused(self):
        ast.parse(Path(gate.__file__).read_text())
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaises(RuntimeError):gate.hf_worker(OUT,OUT/'must-not-exist')
        self.assertFalse((OUT/'must-not-exist').exists())


if __name__=='__main__':unittest.main(argv=[__file__,*rest])
