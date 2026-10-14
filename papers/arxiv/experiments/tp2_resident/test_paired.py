"""CPU-only wiring tests: never acquire physical GPU locks or launch a server."""
from contextlib import contextmanager
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as p


class Paired(unittest.TestCase):
    def test_routebatch_pair_fixed_v_flags_and_isolation(self):
        import copy
        admission = p.ct.load(p.HERE/'admission/route-batch-release-20261012-01/paired-admission.json')
        with mock.patch.object(p,'fused_after_policy'):
            p.routebatch_pair_policy(admission)
            for kind in ('memory','marlin','environment'):
                bad = copy.deepcopy(admission); v = bad['engines']['vllm']['spec']
                if kind == 'memory':v['argv'][v['argv'].index('--gpu-memory-utilization')+1] = '0.9'
                elif kind == 'marlin':v['argv'][v['argv'].index('--linear-backend')+1] = 'auto'
                else:v['environment']['LD_LIBRARY_PATH'] = '/unreviewed-ferrule-cuda'
                with self.assertRaises(ValueError):p.routebatch_pair_policy(bad)
        self.assertEqual(p.experiment_slots('paired'),('F1','V1','V2','F2'))

    def test_routebatch_pair_matches_previous_actual_formal_flags(self):
        current = p.ct.load(p.HERE/'admission/route-batch-release-20261012-01/paired-admission.json')
        path = p.HERE/'runs/route-batch-info-after-20261012-01'
        report = p.ct.load(path/'routebatch-performance.json');p.ct.check_pins(report['raw_pins'])
        prior = p.ct.load(path/'prepared.json')
        self.assertEqual(current['engines']['ferrule']['spec'],prior['admission']['engines']['ferrule']['spec'])
        self.assertEqual(report['policy']['unmeasured_QA_requests'],3)
        self.assertEqual((report['measured_requests'],report['actual_input_tokens'],report['actual_output_tokens']),(5,25,40))

    def fused_after_fixture(self):
        import copy
        admission = copy.deepcopy(p.ct.load(p.HERE/'admission/pair-swiglu-fused-release-20261011-01/smoke-admission.json'))
        admission['protocol'] = p.PROTOCOL
        return admission

    def test_fused_after_info_only_and_actual_smoke_required(self):
        admission = self.fused_after_fixture()
        with mock.patch.object(p.pre_serving, 'validate_smoke', return_value={'status':'pass'}) as gate:
            self.assertEqual(p.fused_after_policy(admission), {'status':'pass'})
            gate.assert_called_once_with(admission['engines']['ferrule'])
        with mock.patch.object(p.pre_serving, 'validate_smoke', side_effect=ValueError('No actual same-binary smoke')):
            with self.assertRaises(ValueError):
                p.fused_after_policy(admission)

    def test_fused_after_rejects_request_debug_and_rankperf(self):
        for key,value in (('FERRULE_LOG','info,ferrule_request=debug'), ('FERRULE_CUDA_RANK_PERF','1')):
            admission = self.fused_after_fixture()
            admission['engines']['ferrule']['spec']['environment'][key] = value
            with mock.patch.object(p.pre_serving, 'validate_smoke') as gate:
                with self.assertRaises(ValueError):
                    p.fused_after_policy(admission)
                gate.assert_not_called()

    def test_fused_after_rejects_artifact_or_handoff_override(self):
        for field in ('binary','build_receipt'):
            admission = self.fused_after_fixture()
            admission['engines']['ferrule'][field]['sha256'] = '0'*64
            with self.assertRaises(ValueError):
                p.fused_after_policy(admission)

    def test_fused_after_single_slot_and_smoke_has_no_qa(self):
        self.assertEqual(p.experiment_slots('fused_after'), ('F1',))
        directory = p.HERE/'runs/pair-swiglu-fused-after-20261011-01'
        smoke = p.lifecycle(directory,'ferrule','CPU test only','serving_smoke')
        formal = p.lifecycle(directory,'ferrule','CPU test only','fused_after')
        self.assertEqual(smoke.PROTOCOL['phases'], list(p.pre_serving.PHASES))
        self.assertEqual(formal.PROTOCOL['phases'], ['hello2','canary8','qa8'])
        self.assertNotIn('qa8',smoke.PROTOCOL['phases'])

    def test_cache_workspace_scope_changes_only_workspace_not_device_cap(self):
        old = p.ferrule_flags()
        new = p.ferrule_flags(p.CACHE_WORKSPACE_SCOPE)
        self.assertEqual({key:(old[key],new[key]) for key in old if old[key] != new[key]},
                         {'--cuda-workspace-mb':('1024','4096')})
        self.assertEqual(new['--cuda-device-budget-mb'], '22528')
        before = p.ferrule_spec('/fixture/never-executed')
        after = p.ferrule_spec('/fixture/never-executed', p.CACHE_WORKSPACE_SCOPE)
        self.assertEqual(before['environment'], after['environment'])
        self.assertNotIn('RUST_LOG', after['environment'])
        self.assertIn('4096', after['argv'])

    def test_unreviewed_workspace_or_device_ceiling_rejected(self):
        for change in ({'cuda_workspace_mb':3072}, {'cuda_device_budget_mb':24576},
                       {'policy':'arbitrary'}, {'enable_rank_performance':True}):
            with self.assertRaisesRegex(ValueError, 'Unreviewed Ferrule serving scope'):
                p.ferrule_spec('/fixture/never-executed', dict(p.CACHE_WORKSPACE_SCOPE, **change))

    def test_plan_scope_cannot_override_prepared_admission(self):
        with mock.patch.object(p.ct, 'require_cpu_python'), mock.patch.object(p, 'prepare') as prepare:
            with self.assertRaisesRegex(ValueError, 'plan-only'):
                p.main(['prepare','--id','fixture','--serving-scope','decoded-cache-4g'])
            prepare.assert_not_called()

    def test_actual_source_profile_and_no_invented_cli(self):
        audit = p.source_audit()
        flags = audit['flags']
        self.assertIsInstance(audit['public_quality_accepted'], bool)
        for key,value in {'--tensor-parallel':'2', '--pipeline-parallel':'1',
                '--expert-parallel':'1', '--rank-backend':'thread', '--devices':'0,1',
                '--ctx-size':'128', '--max-batch-tokens':'8', '--max-active-sequences':'2',
                '--prefill-chunk-size':'8', '--kv-execution-policy':'resident'}.items():
            self.assertEqual(flags[key], value)
        for invented in ('--strict', '--no-offload', '--cpu-offload-gb', '--model-precision',
                         '--numeric-fp8-precision', '--enable-expert-parallel'):
            self.assertNotIn(invented, flags)

    def test_false_const_blocks_prepare_run_and_admission_before_side_effects(self):
        with mock.patch.object(p, 'source_audit', return_value={'public_quality_accepted':False}), \
                mock.patch.object(p.ct, 'load', side_effect=AssertionError('must gate before receipt read')), \
                mock.patch.object(p.shared, 'owner_locks', side_effect=AssertionError('GPU lock forbidden')), \
                mock.patch.object(p.base.supervisor, 'run_process', side_effect=AssertionError('launch forbidden')):
            for call in (lambda:p.prepare('blocked', Path('/not-read')),
                         lambda:p.run('blocked', True, 'even a grant cannot bypass'),
                         lambda:p.validate_admission({})):
                with self.assertRaisesRegex(ValueError, 'PUBLIC_QUALITY_ACCEPTED=false'):
                    call()

    def test_no_owner_grant_blocks_before_validation(self):
        with mock.patch.object(p, 'strict_gate'), mock.patch.object(p, 'validate_pair') as validate:
            with self.assertRaisesRegex(ValueError, 'sole-owner'):
                p.run('blocked')
            validate.assert_not_called()

    def test_cli_environment_isolated_and_order_preserved(self):
        f = p.ferrule_spec('/fixture/never-executed')
        v = p.shared.spec_for(p.shared.config(), '/fixture/never-executed')
        self.assertEqual(f['environment']['CUDA_VISIBLE_DEVICES'], ','.join(p.shared.UUIDS))
        self.assertEqual(f['environment']['CUBLAS_WORKSPACE_CONFIG'], ':0:0')
        self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', v['environment'])
        self.assertNotIn('LD_LIBRARY_PATH', v['environment'])
        self.assertEqual(v['environment']['NCCL_P2P_DISABLE'], '1')
        self.assertEqual(f['argv'][f['argv'].index('--devices')+1], '0,1')

    def test_private_adapter_reuses_code_without_mutating_pinned_modules(self):
        original_file, original_validate = p.baseline.__file__, p.baseline.validate
        original_block = p.shared.request_block
        m = p.lifecycle(Path('/not-created'), 'ferrule', 'fixture authorization')
        self.assertEqual(m.run.__code__.co_code, p.baseline.run.__code__.co_code)
        self.assertEqual(m.worker.__code__.co_code, p.baseline.worker.__code__.co_code)
        self.assertEqual(m.requests.__code__.co_code, p.baseline.requests.__code__.co_code)
        self.assertIs(m.shared.owner_locks, p.shared.owner_locks)
        self.assertIs(m.base.supervisor, p.base.supervisor)
        self.assertIs(m.observer, p.baseline.observer)
        self.assertIs(m.shared.request_block, p.request_block)
        self.assertEqual(m.__file__, str(Path(p.__file__).resolve()))
        self.assertEqual(p.baseline.__file__, original_file)
        self.assertIs(p.baseline.validate, original_validate)
        self.assertIs(p.shared.request_block, original_block)

    def report(self, stage):
        refs = p.shared.references()[0]
        def row(ref):
            return dict(status='success', input_length_matches=True,
                actual_input_tokens=ref['input_tokens'], actual_output_tokens=8,
                finish_reason='length', text=ref['reference_text'], ttft_ms=1, e2e_ms=2)
        if stage == 'canary8':
            return dict(status='complete', runs=[dict(requests=[row(r) for r in refs])], warmups=[])
        return dict(status='complete', runs=[dict(requests=[row(refs[0]) for _ in range(5)])],
                    warmups=[row(refs[0]) for _ in range(2)], first_after_ready=row(refs[0]))

    def request(self, stage, report):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            d = Path(name)
            spec = dict(engine='ferrule', url='http://127.0.0.1:18367/v1/completions', model='fixture')
            prepared = dict(spec=spec, engine_metadata={'gates':{}})
            def fake_http(args):
                self.assertEqual(args[args.index('--engine')+1], 'ferrule')
                p.ct.write(d/'http.json', report)
                return 0
            with mock.patch.object(p.baseline.http, 'main', side_effect=fake_http):
                return p.request_block(d, prepared, stage)

    def test_all_frozen_canaries_and_all_qa_rows_checked(self):
        for stage in ('canary8', 'qa8'):
            gate = self.request(stage, self.report(stage))
            self.assertEqual(gate['status'], 'pass')
            self.assertEqual(len(gate['checks']), 3 if stage == 'canary8' else 8)
            self.assertFalse(gate['fairmatch'])
        r = self.report('qa8')
        r['warmups'][0]['text'] = 'wrong'
        self.assertEqual(self.request('qa8', r)['status'], 'fail')
        r = self.report('qa8')
        del r['first_after_ready']
        self.assertEqual(self.request('qa8', r)['status'], 'fail')
        r = self.report('canary8')
        r['runs'][0]['requests'][1]['actual_output_tokens'] = 7
        self.assertEqual(self.request('canary8', r)['status'], 'fail')

    def test_canary_failure_stops_same_server_before_qa(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            d = Path(name)
            p.ct.write(d/'canaries.json', p.shared.references()[0])
            case = d/'session';case.mkdir()
            module = p.lifecycle(d, 'ferrule', 'fixture')
            stages = []
            def block(directory, prepared, stage):
                stages.append(stage)
                return {'status':'fail' if stage == 'canary8' else 'pass'}
            module.shared.request_block = block
            prepared = dict(config=p.shared.config(), model_identity={}, source_pins={})
            monitor = SimpleNamespace(failed=SimpleNamespace(is_set=lambda:False))
            r = module.requests(case, prepared, monitor)
            self.assertEqual(r['status'], 'canary_failed')
            self.assertEqual(stages, ['hello2','canary8'])
            self.assertTrue((case/'requests.json').exists())

    def test_authentication_precedes_worker_manifest_reads(self):
        with mock.patch.object(sys, 'stdin', io.StringIO('{}')), \
                mock.patch.object(p.guard, 'authenticate', side_effect=ValueError('unauthenticated')), \
                mock.patch.object(p, 'validate_slot') as validate:
            with self.assertRaisesRegex(ValueError, 'unauthenticated'):
                p.worker('forged')
            validate.assert_not_called()

    def test_schedule_fail_stop_retains_remaining_not_run(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            d = Path(name)
            starts = []
            def lifecycle(directory, engine, note):
                def run(slot):
                    starts.append(slot)
                    return 1
                return SimpleNamespace(run=run)
            with mock.patch.object(p, 'strict_gate'), mock.patch.object(p, 'validate_pair', return_value={}), \
                    mock.patch.object(p, 'pair_path', return_value=d), \
                    mock.patch.object(p, 'lifecycle', side_effect=lifecycle):
                self.assertEqual(p.run('fixture', True, 'fixture grant'), 1)
            r = p.ct.load(d/'execution.json')
            self.assertEqual(starts, ['F1'])
            self.assertEqual([x['status'] for x in r['slots']], ['failed']+['not_run']*3)
            self.assertFalse(r['fairmatch'])
            with mock.patch.object(p, 'strict_gate'), mock.patch.object(p, 'validate_pair', return_value={}), \
                    mock.patch.object(p, 'pair_path', return_value=d):
                with self.assertRaisesRegex(ValueError, 'No retry'):
                    p.run('fixture', True, 'fixture grant')

    def test_schedule_balanced_sequential_and_no_implicit_speedup(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            d = Path(name);starts=[]
            def lifecycle(directory, engine, note):
                return SimpleNamespace(run=lambda slot:(starts.append((slot,engine)) or 0))
            with mock.patch.object(p, 'strict_gate'), mock.patch.object(p, 'validate_pair', return_value={}), \
                    mock.patch.object(p, 'pair_path', return_value=d), \
                    mock.patch.object(p, 'lifecycle', side_effect=lifecycle), \
                    mock.patch.object(p, 'assess_slot', return_value={'status':'complete'}):
                self.assertEqual(p.run('fixture', True, 'fixture grant'), 0)
            self.assertEqual(starts, [('F1','ferrule'),('V1','vllm'),('V2','vllm'),('F2','ferrule')])
            r = p.ct.load(d/'execution.json')
            self.assertEqual(r['status'], 'complete_descriptive_only')
            self.assertIsNone(r['exact_speedup'])

    def test_existing_run_uses_sealed_supervisor_and_release_without_gpu_calls(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            d = Path(name); slot = d/'runs/F1';slot.mkdir(parents=True)
            p.ct.write(slot/'prepared.json', {'fixture':True})
            module = p.lifecycle(d, 'ferrule', 'fixture')
            pins = {str(Path(p.__file__).resolve()):p.ct.sha(Path(p.__file__))}
            module.validate = lambda directory:dict(config=p.shared.config(), historical_pins=pins, source_pins=pins)
            events = []
            @contextmanager
            def locks(uuids):
                self.assertEqual(uuids, p.shared.UUIDS)
                events.append('locks')
                try:
                    yield {'fixture':'not a real GPU lock'}
                finally:
                    events.append('release')
            def idle(*args):
                events.append('idle')
                return dict(status='verified', baseline={})
            def seal(payload, directory):
                events.append('seal')
                return dict(payload, worker_token='fixture-token'), 'fixture-seal'
            def supervise(argv, seconds, payload):
                events.append('supervisor')
                self.assertEqual(seconds, 1600)
                self.assertIn(str(Path(p.__file__).resolve()), argv)
                self.assertEqual(argv[-2:], ['--worker','fixture-seal'])
                p.ct.write(Path(payload['directory'])/'case.json', dict(status='complete', server_pid=123))
                identity = {'pid':1,'birth_ticks':2}
                return dict(returncode=0,timed_out=False,cleanup_errors=[],
                    descendant_custody='proven_drained',
                    handshake=dict(containment_verified=True,launch_id='fixture',helper_identity=identity),
                    custody=dict(containment_verified=True,echild=True,status='proven_drained',
                                 launch_id='fixture',helper_identity=identity))
            module.shared.owner_locks = locks
            with mock.patch.object(module.observer, 'idle', side_effect=idle), \
                    mock.patch.object(module.guard, 'seal', side_effect=seal), \
                    mock.patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout='CNS')), \
                    mock.patch.object(module.base.supervisor, 'run_process', side_effect=supervise):
                self.assertEqual(module.run('F1'), 0)
            self.assertEqual(events, ['locks','idle','seal','supervisor','idle','release'])
            self.assertTrue(p.ct.load(slot/'lock-release.json')['all_gpu_idle'])
            self.assertTrue(p.ct.load(slot/'execution.json')['authenticated_drained'])

    def test_unknown_drain_or_idle_is_never_eligible(self):
        for field in ('authenticated_drained','supervisor_success','after'):
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=p.HERE) as name:
                d = Path(name); (d/'session').mkdir()
                receipt = dict(status='complete', authenticated_drained=True, supervisor_success=True,
                               after={'status':'verified'})
                receipt[field] = {'status':'unverified'} if field == 'after' else False
                p.ct.write(d/'execution.json',receipt)
                p.ct.write(d/'session/supervisor.json',{})
                p.ct.write(d/'lock-release.json',dict(released=True,all_gpu_idle=True))
                with mock.patch.object(p.ct,'successful_process',return_value=True):
                    with self.assertRaisesRegex(ValueError,'no next slot'):
                        p.assess_slot(d,'ferrule')

    def test_receipt_hash_status_binary_source_spec_and_evidence_are_required(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            d = Path(name); evidence = d/'cpu-fixture.txt';evidence.write_text('synthetic, not acceptance')
            pins = {str(evidence):p.ct.sha(evidence)}
            m = dict(binary={'sha256':'a'*64},source_pins=pins,spec={'engine':'ferrule'},gates={})
            good = dict(schema='ferrule.paper.tp2-serving-admission.v1',engine='ferrule',
                gate='multistep',status='pass',binary_sha256='a'*64,source_pins=pins,
                spec_digest=p.ct.digest(m['spec']),evidence_pins=pins)
            for i, mutation in enumerate([{}, {'status':'pending'}, {'binary_sha256':'b'*64},
                    {'source_pins':{}}, {'spec_digest':'c'*64}, {'evidence_pins':{}},
                    {'evidence_pins':{str(evidence):'d'*64}}]):
                receipt = d/(str(i)+'.json');p.ct.write(receipt,dict(good,**mutation))
                m['gates']['multistep'] = dict(receipt_path=str(receipt),receipt_sha256=p.ct.sha(receipt))
                if not mutation:
                    p.validate_gate('ferrule','multistep',m)
                else:
                    with self.assertRaises(ValueError):
                        p.validate_gate('ferrule','multistep',m)
            m['gates']['multistep']['receipt_sha256'] = 'e'*64
            with self.assertRaisesRegex(ValueError,'receipt changed'):
                p.validate_gate('ferrule','multistep',m)

    def test_engine_specific_runtime_evidence(self):
        log = p.PROFILE+'\n[Mixed TP2 device budget] resident local expert shards; BF16 activation/KV/conv, F32 SSM/accumulation'
        evidence = p.runtime_evidence('ferrule', log)
        self.assertEqual(evidence['missing'], [])
        self.assertFalse(evidence['exact_arithmetic_match'])
        self.assertTrue(p.runtime_evidence('ferrule', 'wrong TP1 profile')['missing'])
        self.assertNotIn('all_boundary', p.required_gates('vllm'))
        self.assertEqual(p.required_gates('ferrule'), ())  # separate test-quality + output smoke


if __name__ == '__main__':
    unittest.main()
