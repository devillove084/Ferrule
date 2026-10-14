"""CPU fixtures only; no server, GPU, socket, lock or real supervisor launch."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import paired as p
import pre_serving as pre
from test_pair_contract import declaration


class ServingBootstrap(unittest.TestCase):
    def test_no_same_binary_http_receipt_needed_to_start_smoke(self):
        m = declaration('ferrule')
        m['gates'] = {}
        m['arithmetic'] = {'profile':p.PROFILE,'effective_evidence':'unknown'}
        m['transport'] = {'observed_channels':'unknown'}
        a = dict(protocol=pre.PROTOCOL,engines={'ferrule':m})
        with ExitStack() as stack:
            for name in ('strict_gate',):stack.enter_context(mock.patch.object(p,name))
            stack.enter_context(mock.patch.object(p,'model_contract',return_value=m['model']))
            stack.enter_context(mock.patch.object(p,'ferrule_spec',return_value=m['spec']))
            stack.enter_context(mock.patch.object(p,'source_audit',return_value={'source_pins':m['source_pins']}))
            stack.enter_context(mock.patch.object(p.ct,'check_pins'))
            stack.enter_context(mock.patch.object(pre,'current_release_sources',return_value=m['source_pins']))
            stack.enter_context(mock.patch.object(p.ct,'validate_staged_binary'))
            quality = stack.enter_context(mock.patch.object(pre,'validate_release'))
            smoke = stack.enter_context(mock.patch.object(pre,'validate_smoke',side_effect=AssertionError('circular admission')))
            stack.enter_context(mock.patch.object(p.shared,'owner_locks',side_effect=AssertionError('GPU forbidden')))
            stack.enter_context(mock.patch.object(p.base.supervisor,'run_process',side_effect=AssertionError('launch forbidden')))
            self.assertEqual(p.validate_admission(a,'serving_smoke'),a)
            quality.assert_called_once_with(m)
            smoke.assert_not_called()

    def test_false_quality_const_or_missing_owner_grant_blocks_smoke(self):
        with mock.patch.object(p,'source_audit',return_value={'public_quality_accepted':False}), \
                mock.patch.object(p.ct,'load',side_effect=AssertionError('no reads')):
            with self.assertRaisesRegex(ValueError,'PUBLIC_QUALITY_ACCEPTED=false'):
                p.run('no-run',True,'not sufficient',purpose='serving_smoke')
        with mock.patch.object(p,'strict_gate'),mock.patch.object(p,'validate_pair') as validate:
            with self.assertRaisesRegex(ValueError,'sole-owner'):
                p.run('no-run',purpose='serving_smoke')
            validate.assert_not_called()

    def test_smoke_adapter_reuses_same_process_loop_and_bounded_phases(self):
        m = p.lifecycle(Path('/not-created'),'ferrule','fixture','serving_smoke')
        self.assertEqual(m.worker.__code__.co_code,p.baseline.worker.__code__.co_code)
        self.assertEqual(m.run.__code__.co_code,p.baseline.run.__code__.co_code)
        self.assertEqual(m.PROTOCOL['phases'],list(pre.PHASES))
        self.assertEqual(m.PROTOCOL['all_requests_seconds'],440)
        self.assertIs(m.shared.request_block,pre.request_block)
        self.assertEqual(p.experiment_slots('serving_smoke'),('S1','S2'))
        self.assertEqual(p.slot_engine('S1'),'ferrule')

    def test_smoke_logging_is_only_spec_difference(self):
        original = p.ferrule_spec('/fixture/nonexecutable')
        changed = pre.smoke_spec(original)
        self.assertEqual(changed['argv'],original['argv'])
        expected = copy.deepcopy(original)
        expected['environment']['FERRULE_LOG'] = 'info,ferrule_request=debug'
        self.assertEqual(changed,expected)
        self.assertEqual(original['environment']['FERRULE_LOG'],'info')
        self.assertEqual(changed['environment']['CUDA_VISIBLE_DEVICES'],','.join(p.shared.UUIDS))

    def test_execution_pins_exclude_unrelated_paper_tests_and_docs(self):
        before = p.ct.inventory(p.execution_paths())
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            (Path(name)/'unrelated-paper.py').write_text('unrelated edit')
            self.assertEqual(before,p.ct.inventory(p.execution_paths()))
        self.assertTrue(all(Path(path).suffix != '.md' and not Path(path).name.startswith('test_')
                            for path in before if Path(path).parent == p.HERE))
        self.assertIn(str(p.HERE/'pre_serving.py'),before)
        self.assertIn(str(p.ct.ROOT/'scripts/test_vllm_qwen35.py'),before)

    def test_release_build_cannot_be_test_or_change_numerical_sources(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            d=Path(name);binary=d/'binary';binary.write_text('CPU fixture, never executed')
            numeric=str(p.ct.ROOT/'crates/ferrule-model/src/transformer/standard/cuda/hybrid_rank/mixed.rs')
            sources={numeric:'a'*64}
            runtime={str(binary):p.ct.sha(binary)}
            artifact=dict(reason='compiler-artifact',target={'name':'ferrule'},executable=str(binary),
                features=['cuda'],profile=dict(opt_level='3',debug_assertions=False,test=False))
            build=dict(status='pass',supervisor={},artifact=artifact,sources_before=sources,
                sources_after=sources,binary_sha256=p.ct.sha(binary),runtime_pins=runtime)
            quality=dict(schema=pre.QUALITY_SCHEMA,status='pass',kind='authenticated_test_binary_not_serving',
                calls=33,metrics=35937,artifact_pins=runtime,numerical_source_pins=sources,
                rows=[{'test_binary_sha256':'b'*64}])
            m=dict(source_pins=sources,runtime_pins=runtime,binary={'sha256':p.ct.sha(binary)},
                   test_quality={'path':'fixture-quality','sha256':'c'*64},
                   build_receipt={'path':'fixture-build','sha256':'d'*64})
            def document(link):return quality if link['path']=='fixture-quality' else build
            with mock.patch.object(pre,'checked_document',side_effect=document), \
                    mock.patch.object(p.ct,'successful_process',return_value=True):
                result=pre.validate_release(m)
                self.assertEqual(result['serving_observation'],'pending')
                artifact['profile']['test']=True
                with self.assertRaisesRegex(ValueError,'non-test'):
                    pre.validate_release(m)
                artifact['profile']['test']=False
                quality['numerical_source_pins']={numeric:'e'*64}
                with self.assertRaisesRegex(ValueError,'numerical/model/runtime'):
                    pre.validate_release(m)

    def test_catalog_handoff_bootstrap_reports_unauthenticated_build_not_fake_pass(self):
        numeric = str(p.ct.ROOT/'crates/ferrule-model/src/model.rs')
        sources = {numeric:'a'*64}
        quality = dict(schema=pre.QUALITY_SCHEMA,status='pass',kind='authenticated_test_binary_not_serving',
            calls=33,metrics=35937,artifact_pins={'fixture':'b'*64},numerical_source_pins=sources,
            rows=[{'test_binary_sha256':'c'*64}])
        evidence = dict(source_pins=sources,binary={'sha256':'d'*64},runtime_pins={'lib':'e'*64},
            evidence_pins={'raw-build-log':'f'*64},build_custody='unavailable: no authenticated build supervisor')
        m = dict(source_pins=sources,binary=evidence['binary'],runtime_pins=evidence['runtime_pins'],
            test_quality={'path':'quality','sha256':'1'*64},build_receipt={'path':'build','sha256':'2'*64},
            build_evidence_pins=evidence['evidence_pins'])
        with mock.patch.object(pre,'checked_document',side_effect=lambda link: quality if link['path']=='quality' else {'schema':pre.CATALOG_SCHEMA}), \
                mock.patch.object(pre,'catalog_build',return_value=evidence), mock.patch.object(p.ct,'check_pins'):
            self.assertIn('unavailable',pre.validate_release(m)['build_custody'])
            self.assertNotIn('serving_smoke',m)
            m['build_evidence_pins'] = {'changed':'0'*64}
            with self.assertRaisesRegex(ValueError,'evidence inventory changed'):
                pre.validate_release(m)
            m['build_evidence_pins'] = evidence['evidence_pins']
            m['runtime_pins'] = {'other':'3'*64}
            with self.assertRaisesRegex(ValueError,'release identity mismatch'):
                pre.validate_release(m)

    def test_smoke_checks_normal_shutdown_before_second_owner(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            d = Path(name)
            adapter = SimpleNamespace(run=mock.Mock(return_value=0))
            with mock.patch.object(p,'strict_gate'), mock.patch.object(p,'pair_path',return_value=d), \
                    mock.patch.object(p,'validate_pair',return_value={'purpose':'serving_smoke'}), \
                    mock.patch.object(p,'lifecycle',return_value=adapter), \
                    mock.patch.object(pre,'assess_lifecycle',side_effect=ValueError('normal shutdown missing')), \
                    mock.patch.object(pre,'finish') as finish:
                self.assertEqual(p.run('fixture',True,'fixture',purpose='serving_smoke'),1)
                finish.assert_not_called()
            adapter.run.assert_called_once_with('S1')
            self.assertEqual(p.ct.load(d/'execution.json')['slots'][1]['status'],'not_run')

    def test_shutdown_records_actual_root_code_without_replacing_drain(self):
        server = SimpleNamespace(returncode=0)
        original = dict(status='root_exited_descendants_unknown', timed_out=False)
        with mock.patch.object(p.baseline.http, 'shutdown_owned_group', return_value=original) as shutdown:
            self.assertEqual(p.shutdown_receipt(server, 112), dict(original, returncode=0))
            shutdown.assert_called_once_with(server, 112)
        self.assertNotIn('returncode', original)

    def test_source_projection_excludes_generated_tests_and_paper_not_production(self):
        model = str(p.ct.ROOT/'crates/ferrule-model/src/model.rs')
        build = str(p.ct.ROOT/'crates/ferrule-backend/build.rs')
        sources = {model:'a'*64, build:'b'*64,
            str(p.ct.ROOT/'crates/ferrule-model/target/stale.json'):'c'*64,
            str(p.ct.ROOT/'crates/ferrule-model/tests/fixture.rs'):'d'*64,
            str(p.HERE/'CURRENT.md'):'e'*64}
        self.assertEqual(pre.release_sources(sources), {model:'a'*64,build:'b'*64})

    def test_cancellation_requires_matching_worker_not_just_client_disconnect(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            log=Path(name)/'server.log'
            event=dict(event='request_worker_terminal',request_id=17,status='finished',
                cancellation_submitted=True,cancellation_failed=False,requires_restore_or_shutdown=False)
            log.write_text(json.dumps({'fields':event})+'\n')
            self.assertEqual(pre.terminal_evidence(log,17)['status'],'fail')
            event['status']='cancelled';log.write_text(json.dumps({'fields':event})+'\n')
            self.assertEqual(pre.terminal_evidence(log,17)['status'],'pass')
            self.assertEqual(pre.terminal_evidence(log,18,seconds=0)['status'],'fail')

    def test_cancel_reuses_http_helper_restores_alarm_and_retains_unknown_usage(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            d=Path(name);case=d/'cancel';case.mkdir()
            spec=dict(engine='ferrule',url='http://127.0.0.1:18367/v1/completions',model='fixture')
            def request(config,emit):
                self.assertEqual(config['payload']['max_tokens'],64)
                self.assertEqual(config['payload']['temperature'],0)
                try:
                    emit(dict(kind='event',event='',at=1,data=json.dumps(dict(
                        id='cmpl-ferrule-7',choices=[dict(index=0,text=' Tokyo',finish_reason=None)]))))
                except Exception:
                    emit(dict(kind='end',status='transport_error',error='intentional close'))
            previous=signal.getsignal(signal.SIGALRM)
            with mock.patch.object(pre.http,'http_request',side_effect=request), \
                    mock.patch.object(pre,'terminal_evidence',return_value={'status':'pass'}) as terminal, \
                    mock.patch.object(pre,'cleanup_snapshot',return_value={'status':'pass'}):
                r=pre.cancel_request(case,{'spec':spec})
            self.assertTrue(r['intentional_disconnect'])
            self.assertEqual(r['status'],'pass')
            self.assertIsNone(r['actual_output_tokens'])
            terminal.assert_called_once_with(d/'server.log',7)
            self.assertEqual(signal.getsignal(signal.SIGALRM),previous)

    def test_smoke_fail_stop_does_not_attempt_rebuild_after_failure(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            d=Path(name)
            adapter=SimpleNamespace(run=mock.Mock(return_value=1))
            with mock.patch.object(p,'strict_gate'), \
                    mock.patch.object(p,'validate_pair',return_value={'purpose':'serving_smoke'}), \
                    mock.patch.object(p,'pair_path',return_value=d), \
                    mock.patch.object(p,'lifecycle',return_value=adapter), \
                    mock.patch.object(pre,'finish') as finish:
                self.assertEqual(p.run('fixture',True,'fixture',purpose='serving_smoke'),1)
                finish.assert_not_called()
            adapter.run.assert_called_once_with('S1')
            r=p.ct.load(d/'execution.json')
            self.assertEqual([s['status'] for s in r['slots']],['failed','not_run'])

    def test_pair_requires_post_smoke_receipt_and_never_auto_admits(self):
        with self.assertRaisesRegex(ValueError,'receipt link'):
            pre.validate_smoke({'binary':{'sha256':'a'*64}})

    def test_metadata_numerical_pin_selection_excludes_catalog_and_paper(self):
        model=str(p.ct.ROOT/'crates/ferrule-model/src/transformer/standard/cuda/hybrid_rank/mixed.rs')
        pins={model:'a'*64,str(p.FACTORY):'b'*64,str(p.HERE/'paired.py'):'c'*64}
        self.assertEqual(pre.numerical_sources(pins),{model:'a'*64})


if __name__=='__main__':
    unittest.main()
