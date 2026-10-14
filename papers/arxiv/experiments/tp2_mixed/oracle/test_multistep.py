"""Tiny CPU lifecycle tests: real HF calls, no model files or GPU queries."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',ATEN_CPU_CAPABILITY='avx512')
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import multistep as m
import tp2_oracle as o
from test_tp2_oracle import Store,tiny


class MultistepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):o.initialize()

    def tearDown(self):self.assertFalse(o.torch.cuda.is_initialized())

    def execute(self,name,steps=2):
        o.torch.manual_seed(529)
        c=tiny();native=o.native.hf.Qwen3_5MoeForCausalLM(c).eval()
        operands=o.Operands(Store(native),c,'cpu');capture=o.Capture()
        model=o.build_model(c,operands,capture);values={};records={}
        plan=m.build_plan(name,steps,{'hello':[1],'capital':[2,3,4]})
        def observe(op,record,capture):
            values[op['id']]={k:v.clone() for k,v in capture.tensors.items()}
            records[op['id']]=record
        with tempfile.TemporaryDirectory(dir=m.HERE) as directory:
            try:
                result=m.run(plan,model,c,operands,capture,Path(directory),observe)
                self.assertEqual([e['id'] for e in result['cases']],m.call_ids(plan))
                for entry in result['cases']:
                    for artifact in (entry,entry['report'],entry['statistics']):
                        self.assertEqual(o.sha256(Path(directory)/artifact['file']),artifact['sha256'])
            finally:capture.close()
        return result,values,records

    def assert_captures_equal(self,a,b):
        self.assertEqual(set(a),set(b))
        for key in a:self.assertTrue(o.torch.equal(a[key],b[key]),key)

    def test_greedy_four_steps_and_no_placeholder_tokens(self):
        result,values,records=self.execute('greedy',4)
        self.assertEqual(len(values),10)
        calls=list(records.values())
        for group in (calls[:5],calls[5:]):
            for prev,nxt in zip(group,group[1:]):
                self.assertEqual(nxt['input_token_ids'],[prev['next_token_id']])
                self.assertEqual(nxt['position_start'],prev['sequence_length_after'])
        self.assertTrue(all(min(c['input_token_ids'])>=0 for c in calls))

    def test_interleaved_canary_does_not_change_main_outputs_or_state(self):
        _,baseline,_=self.execute('greedy')
        result,actual,_=self.execute('canary')
        self.assert_captures_equal(baseline['call-00'],actual['call-00'])
        self.assert_captures_equal(baseline['call-01'],actual['call-03'])
        self.assert_captures_equal(baseline['call-02'],actual['call-04'])
        self.assertTrue(any(e.get('untouched_sessions')==['main'] for e in result['events']))

    def test_fork_branches_diverge_without_mutating_parent(self):
        result,values,records=self.execute('fork')
        self.assert_captures_equal(values['call-01'],values['call-03'])
        self.assert_captures_equal(values['call-01'],values['call-05'])
        forks=[e for e in result['events'] if e['op']=='fork']
        self.assertEqual(len(forks),2)
        self.assertTrue(all(e['storage_disjoint'] for e in forks))
        self.assertEqual(records['call-04']['input_token_ids'],[2])

    def test_recreate_is_fresh_and_replays_identically(self):
        result,values,_=self.execute('recreate')
        for a,b in zip(('call-00','call-01','call-02'),('call-03','call-04','call-05')):
            self.assert_captures_equal(values[a],values[b])
        self.assertEqual(sum(e['op']=='recreate_model' for e in result['events']),1)

    def test_cold_owner_recipe_has_no_in_process_cold_claim(self):
        plan=m.build_plan('cold-owner')
        self.assertIn('second launch',plan['cold_owner'])
        self.assertFalse(any(e['op']=='recreate_model' for e in plan['operations']))
        _,a,_=self.execute('cold-owner');_,b,_=self.execute('greedy')
        for k in a:self.assert_captures_equal(a[k],b[k])

    def test_plan_tampering_bounds_and_frontier_fail_closed(self):
        plan=m.build_plan('fork');plan['operations'][1]['input']['ids']=[99]
        with self.assertRaisesRegex(ValueError,'canonical'):m.validate_plan(plan)
        for n in (0,9,True):
            with self.assertRaises(ValueError):m.build_plan('greedy',n)
        with self.assertRaisesRegex(ValueError,'prompt'):
            m.validate_plan(m.build_plan('greedy',1,{'hello':[1],'capital':[2]}),full_model=True)

    def test_plan_output_is_fresh_and_stays_paper_local(self):
        with tempfile.TemporaryDirectory(dir=m.HERE) as directory:
            p=Path(directory)/'plan.json';self.assertEqual(m.main(['--scenario','canary','--output',str(p)]),0)
            with self.assertRaises(FileExistsError):m.main(['--output',str(p)])
            with self.assertRaisesRegex(ValueError,'paper-local'):m.main(['--output',str(m.HERE.parent/'not-created.json')])

    def test_claim_receipt_is_not_full_inventory(self):
        import claim_receipt
        p=o.ROOT/'target/validation/tp2-mixed-claim-gates/shared-full40-01'
        if not (p/'result.json').is_file():self.skipTest('receipt not installed')
        receipt=claim_receipt.inspect_receipt(p)
        self.assertFalse(receipt['full_inventory_compatible'])
        self.assertEqual(sum(v['compared'] for v in receipt['metrics'].values()),1986560)


if __name__=='__main__':unittest.main()
