"""CPU-only short A/B admission and existing lifecycle wiring regression tests."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import cache_ab as ab
import paired as p


class CacheAB(unittest.TestCase):
    def candidate(self):
        return deepcopy(p.ct.load(p.HERE/'admission/decoded-cache-20261010-03/smoke-admission.json')['engines']['ferrule'])

    def admission(self):
        return dict(protocol=ab.PROTOCOL,engines={'ferrule':self.candidate()},
                    cacheoff=ab.frozen_metadata(),cache_ab=ab.contract())

    def test_actual_old_help_and_cpu_preflight_are_not_GPU_admission(self):
        receipt=p.ct.load(ab.PREFLIGHT)
        self.assertEqual(receipt['status'],'workspace4096_cli_and_pre_model_arguments_accepted')
        self.assertNotEqual(receipt['returncode'],0)
        self.assertFalse(receipt['full_model_resource_admission_proven'])
        self.assertFalse(receipt['server_started']);self.assertFalse(receipt['gpu_workload_started'])
        self.assertIn('--cuda-workspace-mb',(ab.PREFLIGHT.parent/'serve-help.stdout').read_text())
        self.assertEqual(receipt['environment']['CUDA_VISIBLE_DEVICES'],'')

    def test_frozen_audit_never_qualifies_current_source_tree(self):
        with mock.patch.object(p.pre_serving,'current_release_sources',side_effect=AssertionError('no current source substitution')):
            audit=ab.frozen_audit()
        self.assertEqual(audit['binary_sha256'],ab.historical.BINARY_SHA)
        self.assertFalse(audit['current_tree_qualified'])
        self.assertFalse(audit['workspace4096_serving_prequalified'])
        self.assertIsNone(audit['profiled_activity'])

    def test_derived_4g_preserves_old_receipt_and_historical_sources(self):
        original=p.ct.load(ab.historical.ADMISSION)['engines']['ferrule']
        off=ab.frozen_metadata()
        self.assertEqual(off['source_pins'],original['source_pins'])
        self.assertEqual(off['binary'],original['binary'])
        self.assertEqual(off['historical_serving_smoke'],original['serving_smoke'])
        self.assertNotIn('serving_smoke',off)
        a=original['spec']['argv'];b=off['spec']['argv']
        self.assertEqual([(x,y) for x,y in zip(a,b) if x!=y],[('1024','4096')])
        self.assertEqual(p.ct.sha(ab.historical.ADMISSION),ab.historical.ADMISSION_SHA)

    def test_frozen_exception_rejects_other_binary_or_changed_evidence(self):
        original=p.ct.load(ab.historical.ADMISSION)
        original['engines']['ferrule']['binary']['sha256']='0'*64
        with mock.patch.object(p.ct,'load',return_value=original):
            with self.assertRaisesRegex(ValueError,'only for qualified 4d40'):ab.frozen_metadata()
        with mock.patch.object(p.ct,'sha',return_value='0'*64):
            with self.assertRaisesRegex(ValueError,'Historical admission changed'):ab.frozen_metadata()

    def test_matched_flags_runtime_scope_and_no_one_sided_debug(self):
        off=ab.frozen_metadata();on=self.candidate();ab.common_specs(off,on)
        for change in ('workspace','debug','runtime','context'):
            bad=deepcopy(on)
            if change=='workspace':bad['spec']['argv'][bad['spec']['argv'].index('--cuda-workspace-mb')+1]='1024'
            if change=='debug':bad['spec']['environment']['FERRULE_LOG']='info,ferrule_performance=debug'
            if change=='runtime':bad['runtime_pins']['/unreviewed']='0'*64
            if change=='context':bad['workload']['context']=256
            with self.subTest(change=change),self.assertRaises(ValueError):ab.common_specs(off,bad)

    def test_frozen_old_does_not_bypass_new_candidate_release_audit(self):
        a=self.admission()
        with mock.patch.object(ab,'frozen_audit'),mock.patch.object(p,'validate_admission',side_effect=ValueError('candidate not qualified')) as validate:
            with self.assertRaisesRegex(ValueError,'candidate not qualified'):ab.validate(a)
            validate.assert_called_once_with(dict(protocol=p.pre_serving.PROTOCOL,engines=a['engines']),'serving_smoke')

    def test_no_profiler_extra_engine_or_changed_frozen_metadata(self):
        for change in ('profiler','engine','source'):
            a=self.admission()
            if change=='profiler':a['profiler']={}
            if change=='engine':a['engines']['vllm']={}
            if change=='source':a['cacheoff']['source_pins']={'invented':'0'*64}
            with self.subTest(change=change),mock.patch.object(ab,'frozen_audit'),mock.patch.object(p,'validate_admission'),self.assertRaises(ValueError):ab.validate(a)

    def test_protocol_has_exactly_two_F_lifecycles_and_three_unmeasured_QA8(self):
        self.assertEqual(p.experiment_slots('cache_ab'),('OFF','ON'))
        self.assertEqual([p.slot_engine(s) for s in ab.SLOTS],['ferrule','ferrule'])
        spec=self.candidate()['spec']
        plan=p.contract.http_plan('ferrule',spec,'qa8',Path('/fixture'))
        argv=plan['argv']
        self.assertEqual(argv[argv.index('--warmup')+1],'2')
        self.assertIn('--first-after-ready',argv)
        self.assertEqual(argv[argv.index('--requests')+1],'5')
        self.assertEqual(ab.contract()['qa8_unmeasured'],3)

    def test_cacheoff_slot_uses_frozen_binary_not_candidate_metadata(self):
        a=self.admission()
        with tempfile.TemporaryDirectory(dir=p.HERE) as tmp:
            root=Path(tmp);prepared=dict(protocol=ab.PROTOCOL,purpose='cache_ab',admission=a)
            p.ct.write(root/'prepared.json',prepared)
            for slot in ab.SLOTS:
                d=root/'runs'/slot;d.mkdir(parents=True)
                p.ct.write(d/'prepared.json',dict(pair_directory=str(root),slot=slot,pair_sha256=p.ct.sha(root/'prepared.json')))
                p.ct.write(d/'canaries.json',p.shared.references()[0])
                with mock.patch.object(p,'validate_pair',return_value=prepared):result=p.validate_slot(d)
                expected=a['cacheoff'] if slot=='OFF' else a['engines']['ferrule']
                self.assertEqual(result['engine_metadata'],expected)
                self.assertEqual(result['executable_sha256'],expected['binary']['sha256'])
                self.assertEqual(result['spec']['environment']['FERRULE_LOG'],'info')
                self.assertNotIn('TMPDIR',result['spec']['environment'])

    def test_same_supervisor_observer_and_HTTP_implementation(self):
        module=p.lifecycle(Path('/fixture'),'ferrule','CPU fixture','cache_ab')
        for method in ('run','worker','requests'):
            self.assertEqual(getattr(module,method).__code__.co_code,getattr(p.baseline,method).__code__.co_code)
        self.assertIs(module.base.supervisor,p.base.supervisor)
        self.assertIs(module.observer,p.baseline.observer)
        self.assertIs(module.shared.request_block,p.request_block)
        self.assertEqual(module.PROTOCOL['phases'],['hello2','canary8','qa8-V1'])
        self.assertEqual(module.PROTOCOL['id'],ab.PROTOCOL)

    def schedule(self,fail):
        with tempfile.TemporaryDirectory(dir=p.HERE) as tmp:
            directory=Path(tmp);starts=[]
            def lifecycle(root,engine,note,purpose):
                self.assertEqual((engine,purpose),('ferrule','cache_ab'))
                return SimpleNamespace(run=lambda slot: starts.append(slot) or int(fail))
            with mock.patch.object(p,'strict_gate'),mock.patch.object(p,'pair_path',return_value=directory), \
                    mock.patch.object(p,'validate_pair',return_value={'purpose':'cache_ab','protocol':ab.PROTOCOL}), \
                    mock.patch.object(p,'lifecycle',side_effect=lifecycle),mock.patch.object(p,'assess_slot',return_value={'status':'complete'}), \
                    mock.patch.object(ab,'finish',return_value={'status':'complete_descriptive_only'}) as finish:
                code=p.run('fixture',True,'CPU fixture grant','cache_ab')
                if fail:finish.assert_not_called()
                else:finish.assert_called_once()
            return code,starts,p.ct.load(directory/'execution.json')

    def test_schedule_runs_only_OFF_ON_and_no_vllm(self):
        code,starts,result=self.schedule(False)
        self.assertEqual(code,0);self.assertEqual(starts,['OFF','ON'])
        self.assertEqual(result['status'],'complete_descriptive_only')
        self.assertFalse(result['fairmatch']);self.assertIsNone(result['exact_speedup'])

    def test_old_failure_stops_before_candidate(self):
        code,starts,result=self.schedule(True)
        self.assertEqual(code,1);self.assertEqual(starts,['OFF'])
        self.assertEqual([r['status'] for r in result['slots']],['failed','not_run'])

    def test_missing_GPU_grant_blocks_before_any_validation(self):
        with mock.patch.object(p,'strict_gate'),mock.patch.object(p,'validate_pair') as validate:
            with self.assertRaisesRegex(ValueError,'sole-owner'):p.run('fixture',purpose='cache_ab')
            validate.assert_not_called()


if __name__=='__main__':unittest.main()
