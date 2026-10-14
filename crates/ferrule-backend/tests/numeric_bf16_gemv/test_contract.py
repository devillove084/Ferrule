"""CPU-only checks for the bounded contract probe and unchanged native BF16 oracle."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import contract_run

ROOT=contract_run.ROOT
OUT=ROOT/'target/validation/capital-gemv-contract-20261006'
SOURCE=ROOT/'target/validation/capital-projections-20261006/fixture-v2'
class Contract(unittest.TestCase):
    def test_fixture_and_original_oracle_pins(self):
        self.assertEqual(contract_run.sha(SOURCE/'SHA256SUMS'),'0381178a03f7c848a73b153ddf3cec3284c316814ef3007f3f7a8043f96e984f')
        manifest=json.loads((OUT/'fixture/manifest.json').read_text())
        self.assertEqual(len(manifest['cases']),8)
        for name,pin in manifest['payload_pins'].items():self.assertEqual(contract_run.sha(OUT/'fixture'/name),pin)
        for c in manifest['cases']:
            id=c['id'];expert,part=id.split('_')
            for suffix in ['decoded.bf16','hf.bf16']:
                expected=(SOURCE/(id+'.'+suffix)).read_bytes()
                if part=='up':expected=(SOURCE/(expert+'_gate.'+suffix)).read_bytes()+expected
                self.assertEqual((OUT/'fixture'/(id+'.'+suffix)).read_bytes(),expected)
            self.assertEqual((OUT/'fixture'/(id+'.input.bf16')).read_bytes(),(SOURCE/(id+'.input.bf16')).read_bytes())

    def test_contract_accepts_only_matched_public_api_and_layout(self):
        cases=[dict(n=2048,k=512),dict(n=1024,k=2048)]
        calls=[dict(api='cublasGemmEx',transa=1,transb=0,m=c['n'],n=1,k=c['k'],atype=14,btype=14,ctype=14,lda=c['k'],ldb=c['k'],ldc=c['n'],compute=68,algo=99,math=16,atomics=0,pointer_mode=0,alpha=1,beta=0,a_mod256=0,b_mod256=0,c_mod256=0,workspace_known=True,workspace_bytes=32<<20,status=0) for c in cases]
        self.assertTrue(contract_run.contract(calls,cases)['matched'])
        for field,value in [('math',0),('workspace_bytes',0),('algo',-1),('lda',513),('transa',0),('atomics',1),('a_mod256',2),('ctype',0)]:
            changed=copy.deepcopy(calls);changed[0][field]=value
            with self.assertRaises(ValueError,msg=field):contract_run.contract(changed,cases)

    def test_run_requires_permission_before_device_query(self):
        for execute,owner in [(False,False),(False,True),(True,False)]:
            with self.assertRaises(ValueError):contract_run.run(OUT,'hf-plain',execute,owner)

    def test_mock_observer_forwards_the_bf16_call(self):
        calls=[json.loads(l) for l in (OUT/'mock-calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls),1)
        self.assertTrue(contract_run.contract(calls,[dict(n=2048,k=512)])['matched'])

    def test_existing_repeats_are_repeatable_not_a_nondeterminism_proof(self):
        base=ROOT/'target/validation/capital-projections-20261006'
        reports=[]
        for name in ['operands-v2-supervisor.json','operands-final-supervisor.json']:
            r=json.loads((base/name).read_text());self.assertTrue(r['passed'])
            reports.append({x['case']:x for l in r['supervisor']['stdout'].splitlines() if 'case' in (x:=json.loads(l))})
        self.assertEqual(reports[0],reports[1])
        for id,count in [('expert146_down',1),('expert238_down',3),('expert200_down',1)]:
            self.assertEqual(reports[0][id]['tst_full'],count)

if __name__=='__main__':unittest.main()
