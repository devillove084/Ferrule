"""CPU synthetic protocol regression, not model-quality evidence."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import lifecycle_bridge as l
import test_bridge as fixture

class LifecycleBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): l.o.initialize()
    def tearDown(self): self.assertFalse(l.o.torch.cuda.is_initialized())

    def make(self, root):
        legacy=root/'legacy';legacy.mkdir()
        def before(rows, _):
            for rank in (0,1):
                for e in list(rows[rank]):
                    names={'gdn.conv_state':'gdn.before.conv','gdn.recurrent_state':'gdn.before.ssm'}
                    if e.get('event') in names:
                        extra=copy.deepcopy(e);extra['event']=names[e['event']]
                        rows[rank].insert(rows[rank].index(e),extra)
        config=fixture.fixture_config()
        ref=json.loads(json.dumps(fixture.synthetic_bundle(legacy,config,before)))
        plan=l.m.build_plan('greedy',1)
        ref['suite']={'plan':plan}
        raw=root/'raw';raw.mkdir();base=raw/'lifecycle-raw';base.mkdir()
        epoch=base/'epoch-0';shutil.copytree(legacy/'candidate',epoch)
        ec=l.b.read(epoch/'contract.json');ec.update(epoch=0,plan=plan)
        (epoch/'contract.json').write_text(json.dumps(ec))
        (base/'contract.json').write_text(json.dumps(dict(schema='ferrule.mixed-lifecycle-raw.v1',
            reference_manifest_sha256=fixture.PIN,plan=plan,epoch_calls=[4],checkpoint=ref['checkpoint'],protocol=ref['protocol'])))
        events=[];index=0
        for op in plan['operations']:
            event=dict(op,epoch=0)
            if op['op']=='create':event['frontier']=0
            elif op['op']=='call':
                phase='prefill' if op['input']['kind']=='tokens' else 'decode.0'
                case=op['session'];id=op['id']
                parent=l.b.read(epoch/f'{case}.{phase}.receipt.json')
                kv=l.b.read(epoch/f'{case}.{phase}.kv-receipt.json')
                parent.update(case=id,phase='prefill' if phase=='prefill' else 'decode')
                kv.update(case=id,phase=parent['phase'])
                (epoch/f'{id}.receipt.json').write_text(json.dumps(parent))
                (epoch/f'{id}.kv-receipt.json').write_text(json.dumps(kv))
                shutil.copyfile(legacy/f'{case}.{phase}.actual.f32le',raw/f'{id}.call.actual.f32le')
                start=0 if phase=='prefill' else len(l.o.PROMPTS[case])
                event.update(runtime_session=parent['session'],transaction=parent['transaction'],
                    call=dict(stage='call',input_token_ids=parent['input_token_ids'],position_start=start,
                        sequence_length_after=start+len(parent['input_token_ids']),next_token_id=0),untouched_sessions=[])
                index+=1
            events.append(event)
        (base/'result.json').write_text(json.dumps(dict(schema='ferrule.mixed-lifecycle-raw.v1',status='captured',
            shutdown_ok=True,writer_complete=True,events=events)))
        return raw,ref

    def test_raw_to_real_lifecycle_schema_and_actual_state_digests(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw,ref=self.make(root);pin=l.snapshot(raw)['sha256']
            result=l.convert(raw,root/'canonical',ref,fixture.PIN,pin)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['schema'],l.lc.CANDIDATE_SCHEMA)
            self.assertEqual(len(result['cases']),4)
            self.assertTrue(all(e['publication']['sha256'] for e in result['cases']))
            self.assertTrue(all(e['state_after'] for e in result['suite']['events'] if e['op']=='call'))

    def test_sorted_raw_prompt_keys_preserve_pinned_operation_order(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw,ref=self.make(root)
            path=raw/'lifecycle-raw/contract.json'
            path.write_text(json.dumps(l.b.read(path), sort_keys=True))
            self.assertEqual(list(l.b.read(path)['plan']['prompts']), ['capital', 'hello'])
            self.assertEqual(l.b.read(path)['plan'], ref['suite']['plan'])
            result=l.convert(raw,root/'canonical',ref,fixture.PIN,l.snapshot(raw)['sha256'])
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['suite']['plan']['operations'], ref['suite']['plan']['operations'])

    def test_changed_raw_recipe_is_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw,ref=self.make(root)
            path=raw/'lifecycle-raw/contract.json';contract=l.b.read(path)
            contract['plan']['operations'].reverse()
            path.write_text(json.dumps(contract))
            with self.assertRaisesRegex(ValueError, 'Different recipes'):
                l.convert(raw,root/'rejected',ref,fixture.PIN,l.snapshot(raw)['sha256'])
            self.assertFalse((root/'rejected').exists())

    def test_missing_before_state_and_failed_kv_never_complete_candidate(self):
        for fault in ('state','kv'):
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as d:
                root=Path(d);raw,ref=self.make(root);epoch=raw/'lifecycle-raw/epoch-0'
                if fault=='kv':
                    p=epoch/'call-00.kv-receipt.json';v=l.b.read(p);v['export_completed']=False;p.write_text(json.dumps(v))
                else:
                    p=epoch/'rank-0/events.jsonl';p.write_text(p.read_text().replace('gdn.before.ssm','unused.before.ssm'))
                with self.assertRaises((ValueError,KeyError)):
                    l.convert(raw,root/'rejected',ref,fixture.PIN,l.snapshot(raw)['sha256'])
                self.assertEqual(l.b.read(root/'rejected/manifest.json')['status'],'incomplete')

if __name__=='__main__':unittest.main()
