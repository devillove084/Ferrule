"""CPU-only typed arena index admission/wiring tests; no GPU or server launch."""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import arena_index_ab as a
import paired as p


class ArenaIndexAB(unittest.TestCase):
    def before(self):
        return a.frozen_metadata()

    def source_fixture(self):
        before={name:'1'*64 for name in a.ALLOWED_PRODUCTION}
        before[str(p.ct.ROOT/'Cargo.toml')]='2'*64
        after=dict(before,**{name:'3'*64 for name in a.ALLOWED_PRODUCTION})
        delta=dict(schema=a.DELTA_SCHEMA,before_binary_sha256=a.BEFORE_SHA,after_binary_sha256='4'*64,
            before_production_source_digest=a.ct.digest(before),after_production_source_digest=a.ct.digest(after),
            production_changes=[dict(path=k,before_sha256=before[k],after_sha256=after[k])
                                for k in sorted(before) if before[k]!=after[k]])
        return before,after,delta

    def quality_fixture(self):
        prior=p.ct.load(self.before()['test_quality']['path'])
        after=deepcopy(prior);after['qualification_handoff_sha256']='4'*64
        for i,row in enumerate(after['rows']):row['launch_id']='fresh-'+str(i)
        return prior,after

    def fixture(self,cycle=1):
        before=self.before();after=deepcopy(before)
        after['binary']={'path':'/fixture/indexed/ferrule','sha256':'4'*64,'kind':'public_serving'}
        after['spec']=p.ferrule_spec(after['binary']['path'],p.CACHE_WORKSPACE_SCOPE)
        after['test_quality']={'path':'/fixture/quality.json','sha256':'5'*64}
        delta={'path':'/fixture/delta.json','sha256':'6'*64}
        after['build_evidence_pins']={delta['path']:delta['sha256']}
        return dict(protocol=a.PROTOCOL,engines={'ferrule':after},before=before,cycle=cycle,
            arena_index_ab=a.contract(cycle),source_delta=delta,
            after_admission={'path':'/fixture/promoted.json','sha256':'7'*64})

    def validation_context(self,admission):
        stack=ExitStack();before,after,delta=self.source_fixture();prior,q=self.quality_fixture()
        m=admission['engines']['ferrule']
        build={'receipts':{'build-inputs':{'path':'/fixture/inputs.json','sha256':'8'*64}}}
        inputs=dict(environment={'CUDA_VISIBLE_DEVICES':''},release_profile={'lto':'fat'},
                    toolchain={k:k for k in ('cargo','rustc','nvcc','cxx','cutlass_head')})
        docs={'/fixture/promoted.json':{'engines':{'ferrule':m}},'/fixture/quality.json':q,
            admission['before']['test_quality']['path']:prior,'/fixture/delta.json':delta,'/fixture/inputs.json':inputs}
        stack.enter_context(mock.patch.object(a,'frozen_audit',return_value={'frozen':True}))
        stack.enter_context(mock.patch.object(a,'production_sources',side_effect=[(before,build),(after,build)]))
        stack.enter_context(mock.patch.object(a.pre,'checked_document',side_effect=lambda link:deepcopy(docs[link['path']])))
        stack.enter_context(mock.patch.object(a.ct,'check_pins'))
        release=stack.enter_context(mock.patch.object(p,'validate_admission',return_value={}))
        smoke=stack.enter_context(mock.patch.object(a.pre,'validate_smoke',return_value={'status':'pass'}))
        return stack,release,smoke

    def test_actual_frozen_arena_is_closed_source_not_current_tree(self):
        with mock.patch.object(a.pre,'current_release_sources',side_effect=AssertionError('no current substitution')):
            evidence=a.frozen_audit()
        self.assertEqual(evidence['binary_sha256'],a.BEFORE_SHA)
        self.assertFalse(evidence['current_tree_qualified'])
        self.assertTrue(evidence['cache_enabled'])
        self.assertEqual(evidence['smoke_status'],'pass')
        self.assertIn(str(a.WORKFLOW_ARCHIVE),evidence['evidence_pins'])
        self.assertEqual(evidence['closed_historical_workflow_source_pins'],a.CLOSED_WORKFLOW_SOURCES)

    def test_frozen_metadata_is_unchanged_and_not_cacheoff(self):
        m=self.before()
        self.assertEqual(m,p.ct.load(a.BASELINE)['engines']['ferrule'])
        self.assertTrue(m['decoded_cache']['enabled'])
        self.assertEqual(p.ct.sha(a.BASELINE),a.BASELINE_SHA)
        self.assertNotEqual(a.POLICY,p.cache_ab.POLICY)
        self.assertEqual(p.experiment_slots('cache_ab'),('OFF','ON'))

    def test_crossed_cycles_and_exact_contract(self):
        for cycle,order in ((1,('BEFORE','AFTER')),(2,('AFTER','BEFORE'))):
            admission=self.fixture(cycle)
            self.assertEqual(a.slots(cycle),order)
            self.assertEqual(p.experiment_slots('arena_index_ab',admission),order)
            self.assertEqual(a.contract(cycle)['concurrency'],1)
            self.assertEqual(a.contract(cycle)['arena_bytes_per_owner'],3<<30)
            self.assertEqual(a.contract(cycle)['workspace_mib'],4096)
            self.assertEqual(a.contract(cycle)['device_cap_mib'],22528)
            self.assertEqual(a.contract(cycle)['qa8_measured'],5)
            self.assertIsNone(a.contract(cycle)['cpu_observation']['thread_count'])
        for bad in (0,3,True,'1'):
            with self.assertRaises(ValueError):a.slots(bad)
        with self.assertRaisesRegex(ValueError,'cycle admission'):p.experiment_slots('arena_index_ab')

    def test_exact_two_file_delta_and_native_cc_inventory_not_exempt(self):
        before,after,delta=self.source_fixture()
        self.assertEqual(len(a.validate_source_delta(before,after,delta,'4'*64)),2)
        for change in ('third','added','removed','one','identity','digest','row'):
            candidate=deepcopy(after);receipt=deepcopy(delta)
            if change=='third':candidate[str(p.ct.ROOT/'Cargo.toml')]='9'*64
            if change=='added':candidate[str(p.ct.ROOT/'crates/ferrule-backend/native/cpu/extra.cc')]='9'*64
            if change=='removed':candidate.pop(str(p.ct.ROOT/'Cargo.toml'))
            if change=='one':candidate[a.ALLOWED_PRODUCTION[0]]=before[a.ALLOWED_PRODUCTION[0]]
            if change=='identity':receipt['after_binary_sha256']='9'*64
            if change=='digest':receipt['after_production_source_digest']='9'*64
            if change=='row':receipt['production_changes'][0]['before_sha256']='9'*64
            with self.subTest(change=change),self.assertRaises(ValueError):a.validate_source_delta(before,candidate,receipt,'4'*64)

    def test_named_fresh_23_gates_not_count_or_inheritance(self):
        prior,q=self.quality_fixture();a.require_fresh_quality(q,prior)
        for change in ('duplicate','selector','launch','reused','handoff','count','inherit'):
            bad=deepcopy(q)
            if change=='duplicate':bad['rows'][-1]=deepcopy(bad['rows'][0])
            if change=='selector':bad['rows'][0]['selector']='wrong'
            if change=='launch':bad['rows'][0]['launch_id']=prior['rows'][0]['launch_id']
            if change=='reused':bad['rows'][0]['launch_id']=bad['rows'][1]['launch_id']
            if change=='handoff':bad['qualification_handoff_sha256']=prior['qualification_handoff_sha256']
            if change=='count':bad['calls']=36
            if change=='inherit':bad['numerical_pass_inherited']=True
            with self.subTest(change=change),self.assertRaises(ValueError):a.require_fresh_quality(bad,prior)

    def test_after_requires_reviewed_release_and_actual_smoke(self):
        admission=self.fixture();context,release,smoke=self.validation_context(admission)
        with context:
            self.assertEqual(a.validate(admission),admission)
            release.assert_called_once_with(dict(protocol=a.pre.PROTOCOL,engines=admission['engines']),'serving_smoke')
            smoke.assert_called_once_with(admission['engines']['ferrule'])
        for failure in ('release','smoke'):
            context,release,smoke=self.validation_context(admission)
            with context:
                (release if failure=='release' else smoke).side_effect=ValueError('not qualified')
                with self.assertRaisesRegex(ValueError,'not qualified'):a.validate(admission)

    def test_delta_must_be_in_after_builder_evidence(self):
        admission=self.fixture();admission['engines']['ferrule']['build_evidence_pins']={}
        context,_,_=self.validation_context(admission)
        with context,self.assertRaisesRegex(ValueError,'pinned in AFTER builder'):a.validate(admission)

    def test_no_profiler_extra_engine_or_relabelled_baseline(self):
        for change in ('profiler','engine','baseline','cycle','off'):
            admission=self.fixture()
            if change=='profiler':admission['profiler']={}
            if change=='engine':admission['engines']['vllm']={}
            if change=='baseline':admission['before']['binary']['sha256']='9'*64
            if change=='cycle':admission['cycle']=2
            if change=='off':admission['before']['decoded_cache']['enabled']=False
            with self.subTest(change=change),self.assertRaises(ValueError):a.validate(admission)

    def test_scope_and_debug_drift_blocked(self):
        for change in ('workspace','debug','runtime','cache'):
            admission=self.fixture();m=admission['engines']['ferrule']
            if change=='workspace':m['spec']['argv'][m['spec']['argv'].index('--cuda-workspace-mb')+1]='3072'
            if change=='debug':m['spec']['environment']['FERRULE_LOG']='info,ferrule_performance=debug'
            if change=='runtime':m['runtime_pins']['unreviewed']='9'*64
            if change=='cache':m['decoded_cache']['enabled']=False
            context,_,_=self.validation_context(admission)
            with self.subTest(change=change),context,self.assertRaises(ValueError):a.validate(admission)

    def test_existing_lifecycle_and_slot_metadata_are_reused(self):
        admission=self.fixture(2)
        self.assertEqual(p.slot_metadata(admission,'BEFORE'),admission['before'])
        self.assertEqual(p.slot_metadata(admission,'AFTER'),admission['engines']['ferrule'])
        with self.assertRaises(ValueError):p.slot_metadata(admission,'OFF')
        module=p.lifecycle(Path('/fixture'),'ferrule','CPU wiring only','arena_index_ab')
        self.assertEqual(module.PROTOCOL['id'],a.PROTOCOL)
        self.assertEqual(module.PROTOCOL['phases'],['hello2','canary8','qa8-V1'])
        self.assertEqual(module.validate,p.validate_slot)
        self.assertIn(p.HERE/'arena_index_ab.py',p.execution_paths('arena_index_ab'))

    def test_prepare_two_cycles_uses_existing_receipts_with_correct_order(self):
        with tempfile.TemporaryDirectory(dir=p.HERE/'runs') as tmp:
            root=Path(tmp)
            for cycle in (1,2):
                admission=self.fixture(cycle);path=root/('admission-'+str(cycle)+'.json');p.ct.write(path,admission)
                directory=root/('cycle-'+str(cycle))
                with mock.patch.object(p,'strict_gate'),mock.patch.object(p,'validate_admission',return_value=admission), \
                        mock.patch.object(p,'pair_path',return_value=directory),mock.patch.object(a,'frozen_audit',return_value={'closed':True}):
                    self.assertEqual(p.prepare('fixture',path,'arena_index_ab'),directory)
                    prepared=p.validate_pair(directory)
                    self.assertEqual(prepared['order'],list(a.slots(cycle)))
                    self.assertNotIn('serving_scope_difference',prepared)
                    for slot in a.SLOTS:
                        v=p.validate_slot(directory/'runs'/slot)
                        self.assertEqual(v['engine_metadata'],p.slot_metadata(admission,slot))
                    self.assertFalse((directory/'launch.json').exists())

    def test_missing_final_hash_or_smoke_cannot_write_admission(self):
        with tempfile.TemporaryDirectory(dir=p.HERE/'admission') as tmp:
            output=Path(tmp)/'new.json'
            with self.assertRaisesRegex(ValueError,'Explicit final'):
                a.admission(output,Path('/missing'),None,Path('/missing'),None,1)
            self.assertFalse(output.exists())
        with mock.patch.object(p,'validate_admission',side_effect=ValueError('AFTER pending')):
            with self.assertRaisesRegex(ValueError,'AFTER pending'):
                p.prepare('never-created',a.BASELINE,'arena_index_ab')
            self.assertFalse((p.HERE/'runs/never-created').exists())

    def test_crossed_runner_schedule_fail_stop_and_unchanged_supervisor(self):
        for cycle in (1,2):
            for fail in (False,True):
                with self.subTest(cycle=cycle,fail=fail),tempfile.TemporaryDirectory(dir=p.HERE/'runs') as tmp:
                    directory=Path(tmp);starts=[]
                    def lifecycle(root,engine,note,purpose):
                        self.assertEqual((engine,purpose),('ferrule','arena_index_ab'))
                        return SimpleNamespace(run=lambda slot:starts.append(slot) or int(fail))
                    prepared=dict(purpose='arena_index_ab',protocol=a.PROTOCOL,admission=self.fixture(cycle))
                    with mock.patch.object(p,'strict_gate'),mock.patch.object(p,'pair_path',return_value=directory), \
                            mock.patch.object(p,'validate_pair',return_value=prepared),mock.patch.object(p,'lifecycle',side_effect=lifecycle), \
                            mock.patch.object(p,'assess_slot',return_value={'status':'complete'}), \
                            mock.patch.object(a,'finish',return_value={'status':'complete_descriptive_only'}) as finish:
                        code=p.run('fixture',True,'CPU fixture, launcher mocked','arena_index_ab')
                    self.assertEqual(code,int(fail))
                    self.assertEqual(starts,list(a.slots(cycle))[:1] if fail else list(a.slots(cycle)))
                    if fail:finish.assert_not_called()
                    else:finish.assert_called_once()
                    result=p.ct.load(directory/'execution.json')
                    if fail:self.assertEqual([r['status'] for r in result['slots']],['failed','not_run'])
                    else:self.assertEqual(result['status'],'complete_descriptive_only')
        module=p.lifecycle(Path('/fixture'),'ferrule','CPU fixture','arena_index_ab')
        for method in ('run','worker','requests'):
            self.assertEqual(getattr(module,method).__code__.co_code,getattr(p.baseline,method).__code__.co_code)
        self.assertIs(module.base.supervisor,p.base.supervisor)
        self.assertIs(module.observer,p.baseline.observer)
        with mock.patch.object(p,'strict_gate'),mock.patch.object(p,'validate_pair') as validate:
            with self.assertRaisesRegex(ValueError,'sole-owner'):p.run('fixture',purpose='arena_index_ab')
            validate.assert_not_called()

    def test_finish_retains_run_window_counts_only_samples_and_both_arena_roots(self):
        with tempfile.TemporaryDirectory(dir=p.HERE/'runs') as tmp:
            directory=Path(tmp);admission=self.fixture(2)
            request=dict(status='success',actual_input_tokens=5,actual_output_tokens=8,text='exact')
            report=dict(first_after_ready=request,warmups=[request,request],runs=[dict(requests=[request]*5,
                summary=dict(success=5,wall_seconds=2.0,successful_output_tokens_per_second=20.0))])
            for slot in a.SLOTS:
                session=directory/'runs'/slot/'session';session.mkdir(parents=True)
                (session/'server.log').write_text('fixture log')
                (session/'samples.jsonl').write_text(json.dumps(dict(kind='sample',gpu={'devices':[]}))+'\n'+json.dumps(dict(kind='monitor_terminal'))+'\n')
                p.ct.write(session/'case.json',dict(startup_seconds=1,resources=dict(status='observed_within_targets',errors=[])))
                for phase in ('hello2','canary8','qa8-V1'):
                    (session/phase).mkdir()
                    p.ct.write(session/phase/'http.json',report)
                    p.ct.write(session/phase/'gate.json',dict(status='pass',http_receipt_sha256=p.ct.sha(session/phase/'http.json')))
            with mock.patch.object(p,'assess_slot',side_effect=lambda cell,engine:dict(launch_id=cell.name)), \
                    mock.patch.object(p.shared,'references',return_value=([dict(reference_text='exact')],{})), \
                    mock.patch.object(a.release,'snapshot_rows',return_value=[dict(schema_version=2,snapshot=dict(arena_backing_grants=3<<30))]):
                result=a.finish(directory,dict(admission=admission))
            self.assertEqual([r['slot'] for r in result['rows']],['AFTER','BEFORE'])
            for row in result['rows']:
                self.assertEqual((row['gpu_samples'],row['missing_gpu_samples']),(1,0))
                self.assertEqual(row['summary'],report['runs'][0]['summary'])
                self.assertIsNone(row['cpu_observation']['affinity'])
                self.assertIsNone(row['activity'])
            self.assertFalse(result['statistical_significance_claim'])
            self.assertIn('executor exit',result['rate_window'])

    def test_plan_has_no_candidate_or_launch_authority(self):
        plan=a.plan()
        self.assertFalse(plan['execute_allowed']);self.assertFalse(plan['gpu_started'])
        self.assertIsNone(plan['after_binary_sha256'])
        with mock.patch.object(p.ct,'require_cpu_python'),mock.patch('builtins.print') as output:
            self.assertEqual(p.main(['plan','--purpose','arena_index_ab']),0)
            self.assertEqual(json.loads(output.call_args.args[0])['protocol'],a.PROTOCOL)


if __name__=='__main__':unittest.main()
