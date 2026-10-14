"""CPU integrity checks for exact split-K backend evidence, not a GPU fallback."""
import argparse
import json
import os
from pathlib import Path
import unittest
os.environ['CUDA_VISIBLE_DEVICES']=''
from performance import ROOT,sha

p=argparse.ArgumentParser();p.add_argument('--evidence-root',required=True,type=Path);args,rest=p.parse_known_args();BASE=args.evidence_root.resolve()


def read(p):return json.loads(p.read_text())


class SplitKTests(unittest.TestCase):
    def test_current_production_and_frozen_input_pins(self):
        for name in ['all-matrices-prepared','final-prepared','baseline-prepared']:
            out=BASE/name;m=read(out/'prepared.json')
            for path,pin in m['source_pins'].items():self.assertEqual(sha(path),pin,path)
            for path,pin in m['artifact_pins'].items():self.assertEqual(sha(out/path),pin,path)
            for path,pin in m.get('retained_pins',{}).items():self.assertEqual(sha(path),pin,path)
        self.assertFalse(read(BASE/'final-prepared/retained-reference-proof.json')['references_regenerated'])

    def test_all_fourteen_ieee_matrix_cases_exact(self):
        c=read(BASE/'all-matrices-run/comparison.json')
        self.assertEqual(len(c),14)
        self.assertEqual(sum(v['elements'] for v in c.values()),241472)
        for name,v in c.items():self.assertEqual(v['unequal'],0,name)
        kernels=read(BASE/'all-matrices-prepared/hf.json')['kernels']
        self.assertTrue(any('sliced1x4_tn' in n for n in kernels['random-signed']))
        self.assertTrue(any('splitKreduce' in n for n in kernels['random-signed']))
        self.assertTrue(any('128x32_8x5_nn' in n for n in kernels['random-state-512']))

    def test_continuation_and_all_frozen_stages_exact(self):
        c=read(BASE/'final-run/comparison.json')
        self.assertTrue(c['exact']);self.assertFalse(c['tolerances']);self.assertFalse(c['references_regenerated'])
        self.assertEqual(len(c['cases']),7)
        for case,metrics in c['cases'].items():
            for name,v in metrics.items():self.assertEqual(v['unequal'],0,(case,name))
        stages=read(BASE/'stage-final/comparison.json')
        self.assertEqual(sum(len(v) for v in stages.values()),120)
        for c in stages.values():
            for v in c.values():self.assertEqual(v['unequal'],0)

    def test_timing_has_exact_chunk_guards_and_separate_scan_errors(self):
        perf=read(BASE/'final-run/performance.json')
        self.assertEqual(set(perf),{'actual5','dense64','dense129'})
        for case,r in perf.items():
            samples=[s for s in r['raw_samples'] if s.get('event')=='timing']
            for variant in ['scan','chunk']:
                times=[s['microseconds'] for s in samples if s['variant']==variant]
                self.assertEqual(len(times),20)
                self.assertTrue(all(t>0 for t in times))
            for field in ['output','state']:self.assertEqual(r['numerical_errors']['chunk'][field]['unequal'],0)
            self.assertGreater(r['numerical_errors']['scan']['state']['unequal'],0)

    def test_authenticated_final_receipts(self):
        for directory in ['all-matrices-run','final-run','baseline-run','stage-final-run','rust-run']:
            d=BASE/directory
            self.assertEqual(read(d/'release.json')['release'],'released')
            for p in d.glob('*-supervisor.json'):
                r=read(p)
                self.assertEqual(r['returncode'],0,p)
                self.assertFalse(r['timed_out'])
                self.assertFalse(r['cleanup_errors'])
                h,c=r['handshake'],r['custody']
                self.assertTrue(h['containment_verified']);self.assertTrue(c['containment_verified'])
                self.assertTrue(c['echild']);self.assertEqual(r['descendant_custody'],'proven_drained')
                self.assertEqual(h['launch_id'],c['launch_id']);self.assertEqual(h['helper_identity'],c['helper_identity'])


if __name__=='__main__':unittest.main(argv=[__file__,*rest])
