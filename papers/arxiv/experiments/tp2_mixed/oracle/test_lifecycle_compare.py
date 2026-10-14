"""Real tiny HF CPU recipe captures; synthetic candidate packaging, not runtime proof."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',ATEN_CPU_CAPABILITY='avx512')
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import lifecycle_compare as lc
import multistep as m
import tp2_oracle as o
from test_tp2_oracle import tiny,Store
from safetensors.torch import load_file,save_file


def generate(directory,scenario,candidate):
    o.torch.manual_seed(7781);config=tiny()
    native=o.native.hf.Qwen3_5MoeForCausalLM(config).eval();operands=o.Operands(Store(native),config,'cpu')
    capture=o.Capture();model=o.build_model(config,operands,capture)
    plan=m.build_plan(scenario,2,{'hello':[1],'capital':[2,3,4]})
    try:suite=m.run(plan,model,config,operands,capture,directory)
    finally:capture.close()
    if candidate:
        events={e['id']:e for e in suite['events'] if e['op']=='call'}
        for index,entry in enumerate(suite['cases']):
            path=directory/entry['file'];values=load_file(str(path));values={k:v for k,v in values.items() if not k.endswith('.native_expert_ids')}
            save_file(values,str(path));entry.update(o.file_record(path))
            report=json.loads((directory/entry['report']['file']).read_text());report['tensors']={k:v for k,v in report['tensors'].items() if k in values}
            report.update(o.file_record(path));(directory/entry['report']['file']).write_text(json.dumps(report))
            entry['report']=o.file_record(directory/entry['report']['file'])
            event=events[entry['id']];call=event['call']
            session={'main':101,'hello':101,'capital':102,'canary':102,'parent':101,'left':102,'right':103}[event['session']]
            receipt=dict(case=entry['id'],phase='prefill' if call['position_start']==0 else 'decode',session=session,
                input_token_ids=call['input_token_ids'],transaction=index+1,published=True,error=None)
            file=directory/(entry['id']+'.publication.json');file.write_text(json.dumps(receipt));entry['publication']=o.file_record(file)
    return config,suite


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):o.initialize()
    def tearDown(self):self.assertFalse(o.torch.cuda.is_initialized())
    def bundles(self,scenario='greedy'):
        temp=tempfile.TemporaryDirectory(dir=o.HERE);self.addCleanup(temp.cleanup)
        root=Path(temp.name);r=root/'reference';a=root/'candidate';r.mkdir();a.mkdir()
        config,ref=generate(r,scenario,False);_,got=generate(a,scenario,True)
        return r,ref,a,got,config

    def test_all_defined_recipes_full_numerical_inventory(self):
        for name in m.SCENARIOS:
            with self.subTest(name=name):
                args=self.bundles(name);result=lc.compare_suites(*args)
                self.assertTrue(result['passed'])
                self.assertEqual(len(result['calls']),len(args[1]['cases']))
                self.assertTrue(all(v['exact'] for row in result['calls'].values() for v in row['metrics'].values()))

    def test_float_failure_and_exact_route_failure_are_reported(self):
        r,ref,a,got,config=self.bundles();entry=got['cases'][-1];path=a/entry['file'];values=load_file(str(path))
        values['call.rank.0.layers.00.shared_partial']+=1
        values['call.rank.1.layers.01.selected_expert_ids'][0,0]+=1
        save_file(values,str(path));entry.update(o.file_record(path))
        result=lc.compare_suites(r,ref,a,got,config);self.assertFalse(result['passed'])
        metrics=result['calls'][entry['id']]['metrics']
        self.assertGreater(metrics['call.rank.0.layers.00.shared_partial']['failures'],0)
        self.assertEqual(metrics['call.rank.1.layers.01.selected_expert_ids']['atol'],0)
        self.assertEqual((result['atol'],result['rtol']),(.00025,.012))

    def test_missing_duplicate_out_of_order_wrong_source_and_unpublished(self):
        for kind in ('missing','duplicate','order','frontier','unpublished','source','fork'):
            with self.subTest(kind=kind):
                r,ref,a,got,config=self.bundles('fork' if kind=='fork' else 'greedy')
                calls=[e for e in got['events'] if e['op']=='call']
                if kind=='missing':got['cases'].pop()
                elif kind=='duplicate':got['events'].append(copy.deepcopy(got['events'][-1]))
                elif kind=='order':got['events'][0],got['events'][1]=got['events'][1],got['events'][0]
                elif kind=='frontier':calls[1]['call']['position_start']+=1
                elif kind=='fork':next(e for e in got['events'] if e['op']=='fork')['state_sha256']={}
                else:
                    entry=got['cases'][1];path=a/entry['publication']['file'];receipt=json.loads(path.read_text())
                    if kind=='unpublished':receipt['published']=False
                    else:receipt['session']=999
                    path.write_text(json.dumps(receipt));entry['publication']=o.file_record(path)
                with self.assertRaises((ValueError,KeyError)):lc.compare_suites(r,ref,a,got,config)

    def test_state_digest_tied_to_arrays_and_fork_history(self):
        r,ref,a,got,config=self.bundles('fork')
        entry=got['cases'][-1];path=a/entry['file'];values=load_file(str(path))
        values['call.cache.recurrent_states.00']+=1
        save_file(values,str(path));entry.update(o.file_record(path))
        with self.assertRaisesRegex(ValueError,'digest'):lc.compare_suites(r,ref,a,got,config)

    def test_legacy_historical_reference_still_loads(self):
        import compare as c
        path=o.HERE/'runs/full40-20261009-03'
        if not path.is_dir():self.skipTest('historical reference not installed')
        manifest=c.load_manifest(path,'fb96ba4dd134761af30b4025b4cce86f96136cb01f05b5b4fca3e74587caa71d',True)
        self.assertEqual(manifest['protocol'],json.loads((o.HERE/'protocol.json').read_text()))
        self.assertEqual(o.sha256(o.HERE/'protocol.json'),'c6e7836f3797c16502dc21b8c756c87495525c2cf2faac26ba11077f62cfbb57')


if __name__=='__main__':unittest.main()
