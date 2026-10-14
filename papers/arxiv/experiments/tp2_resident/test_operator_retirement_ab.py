"""CPU-only retirement admission/wiring regressions; every launcher is mocked."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import operator_retirement_ab as a
import paired as p


class OperatorRetirementAB(unittest.TestCase):
    def fixture(self, cycle=1):
        before = a.frozen_metadata(); after = deepcopy(before)
        after['binary'] = dict(path='/fixture/after/ferrule', sha256='4'*64, kind='public_serving')
        after['spec'] = p.ferrule_spec(after['binary']['path'], p.CACHE_WORKSPACE_SCOPE)
        after['test_quality'] = dict(path='/fixture/quality.json', sha256='5'*64)
        after['build_receipt'] = dict(path='/fixture/build.json', sha256='6'*64)
        review_link = dict(path='/fixture/review.json', sha256='7'*64)
        delta_link = dict(path='/fixture/delta.json', sha256='8'*64)
        after['build_evidence_pins'] = {v['path']:v['sha256'] for v in (review_link, delta_link)}
        return dict(protocol=a.PROTOCOL, engines={'ferrule':after}, before=before, cycle=cycle,
            operator_retirement_ab=a.contract(cycle), review_manifest=review_link,
            after_admission=dict(path='/fixture/promoted.json', sha256='9'*64))

    def source_fixture(self):
        m = a.frozen_metadata(); b = a.pre.checked_document(m['build_receipt'])
        before = a.pre.checked_document(b['source_pins_before']); after = dict(before)
        h, summary, relation = a.qualification_identity()
        for path, change in relation['production_changes'].items():
            after[path.removeprefix(str(a.ct.ROOT)+'/')] = change['after']
        after[a.TEST_PATCH['path'].removeprefix(str(a.ct.ROOT)+'/')] = a.TEST_PATCH['after_sha256']
        return before, after, b

    def validation_context(self, admission):
        before, after, build = self.source_fixture()
        old_inputs = a.pre.checked_document(build['receipts']['build-inputs'])
        new_inputs = deepcopy(old_inputs)
        new_inputs['environment']['FERRULE_CUDA_TEST_ORACLE'] = '0'
        old_quality = a.pre.checked_document(admission['before']['test_quality'])
        h, summary, _ = a.qualification_identity()
        q = dict(schema=p.retirement_profile.release.QUALITY_SCHEMA, status='pass', kind='authenticated_test_binary_not_serving',
            numerical_pass_inherited=False, source_relation={'inherited_gpu_pass':False}, calls=37, metrics=40293,
            qualification_handoff_sha256=a.QUALIFICATION['sha256'], rows=[dict(gate=r['gate'], selector=r['selector'],
                launch_id=r['launch_id'], test_binary_sha256=r['binary_sha256']) for r in summary['rows']])
        old_build = deepcopy(build); new_build = deepcopy(build)
        new_build['receipts']['build-inputs'] = dict(path="/fixture/new-inputs.json", sha256="c"*64)
        old_build['source_pins_before'] = dict(path='/fixture/old-sources.json', sha256='a'*64)
        new_build['source_pins_before'] = dict(path='/fixture/new-sources.json', sha256='b'*64)
        review = dict(schema=a.REVIEW_SCHEMA, optimization_class='operator_retirement_v3', before_binary_sha256=a.BEFORE_SHA,
            after_binary_sha256='4'*64, qualification_handoff=dict(a.QUALIFICATION),
            release_handoff=admission['engines']['ferrule']['build_receipt'],
            source_delta=dict(path='/fixture/delta.json', sha256='8'*64), rationale='fixture', safety_invariants=['fixture'])
        docs = {'/fixture/new-inputs.json':new_inputs, '/fixture/quality.json':q, '/fixture/promoted.json':{'engines':admission['engines']},
            '/fixture/delta.json':a.source_delta(before, after, '4'*64), '/fixture/old-sources.json':before,
            '/fixture/new-sources.json':after, admission['before']['test_quality']['path']:old_quality,
            build['receipts']['build-inputs']['path']:old_inputs}
        stack = ExitStack(); original = a.pre.checked_document
        stack.enter_context(mock.patch.object(a, 'reviewed_build', return_value=review))
        stack.enter_context(mock.patch.object(a, 'frozen_audit', return_value={'closed':True}))
        stack.enter_context(mock.patch.object(a.indexed, 'production_sources', side_effect=[
            (a.pre.release_sources(before),old_build),(a.pre.release_sources(after),new_build)]))
        stack.enter_context(mock.patch.object(a.pre, 'checked_document', side_effect=lambda link:
            deepcopy(docs[link['path']]) if link['path'] in docs else original(link)))
        stack.enter_context(mock.patch.object(a.ct, 'check_pins'))
        release = stack.enter_context(mock.patch.object(p, 'validate_admission', return_value={}))
        smoke = stack.enter_context(mock.patch.object(a.pre, 'validate_smoke', return_value={'status':'pass'}))
        return stack, release, smoke, docs

    def test_actual_frozen_9ef_audit_never_substitutes_current_sources(self):
        with mock.patch.object(a.pre, 'current_release_sources', side_effect=AssertionError('No current substitution')):
            evidence = a.frozen_audit()
        self.assertEqual(evidence['binary_sha256'], a.BEFORE_SHA)
        self.assertEqual(evidence['smoke_status'], 'pass')
        self.assertFalse(evidence['current_tree_qualified'])
        self.assertFalse(evidence['official_profile_available'])
        self.assertIn(str(a.WORKFLOW_ARCHIVE), evidence['evidence_pins'])

    def test_actual_qualification_is_eligibility_not_release_or_smoke(self):
        h, summary, relation = a.qualification_identity()
        self.assertEqual(h['schema'], a.QUALIFICATION_SCHEMA)
        self.assertEqual((len(summary['rows']),summary['full_model_calls'],summary['quality_metrics']), (23,37,40293))
        self.assertFalse(h['supplemental_replaces_named_gates'])
        self.assertEqual(set(relation['production_changes']), set(a.PRODUCTION))
        a.validate_test_patch()

    def test_named_selector_duplicates_and_inheritance_rejected(self):
        original = a.pre.checked_document
        for change in ('selector','duplicate','inherited','supplement','count'):
            def changed(link):
                result = original(link)
                if link == a.QUALIFICATION:
                    if change == 'supplement': result['supplemental_replaces_named_gates'] = True
                elif Path(link['path']).name == 'acceptance-summary.json':
                    if change == 'selector': result['rows'][0]['selector'] = 'aggregate18'
                    if change == 'duplicate': result['rows'][-1] = result['rows'][0]
                    if change == 'inherited': result['prior_gpu_passes_inherited'] = True
                    if change == 'count': result['quality_metrics'] -= 1
                return result
            with self.subTest(change=change), mock.patch.object(a.pre, 'checked_document', side_effect=changed), self.assertRaises(ValueError):
                a.qualification_identity()

    def test_full_source_delta_separates_exact_test_patch(self):
        before, after, _ = self.source_fixture()
        delta = a.source_delta(before, after, '4'*64)
        self.assertEqual({r['path'] for r in delta['production_changes']}, set(a.PRODUCTION))
        self.assertEqual(delta['test_only_source_changes'], [a.TEST_PATCH])
        self.assertEqual(sum(k.endswith('.cc') for k in a.pre.release_sources(before)), 4)
        test = 'crates/ferrule-backend/tests/allocator_batch_v3/new_fixture.rs'
        after[test] = 'c'*64
        delta = a.source_delta(before, after, '4'*64)
        self.assertEqual(delta['test_source_changes'], [dict(path=str(a.ct.ROOT/test), before_sha256=None, after_sha256='c'*64)])

    def test_unexpected_model_native_build_and_missing_deltas_rejected(self):
        before, after, _ = self.source_fixture()
        for change in ('model','native','cargo','patch','missing','added','removed'):
            modified = dict(after)
            if change in ('model','native'):
                key = next(k for k in modified if ('ferrule-model/src/' in k if change == 'model' else k.endswith('.cc')))
                modified[key] = 'd'*64
            if change == 'cargo': modified['Cargo.toml'] = 'd'*64
            if change == 'patch': modified[a.TEST_PATCH['path'].removeprefix(str(a.ct.ROOT)+'/')] = 'd'*64
            if change == 'missing': modified[a.PRODUCTION[0].removeprefix(str(a.ct.ROOT)+'/')] = before[a.PRODUCTION[0].removeprefix(str(a.ct.ROOT)+'/')]
            if change == 'added': modified['crates/ferrule-backend/native/cpu/unreviewed.cc'] = 'd'*64
            if change == 'removed': del modified[next(k for k in modified if k.endswith('.cc'))]
            with self.subTest(change=change), self.assertRaises(ValueError): a.source_delta(before, modified, '4'*64)

    def test_contract_crosses_order_and_keeps_nulls_and_geometry(self):
        for cycle, order in ((1,('BEFORE','AFTER')),(2,('AFTER','BEFORE'))):
            a_contract = a.contract(cycle)
            self.assertEqual(a.slots(cycle), order)
            self.assertEqual((a_contract['qa8_unmeasured'],a_contract['qa8_measured']), (3,5))
            self.assertEqual((a_contract['arena_bytes_per_owner'],a_contract['workspace_mib'],a_contract['device_cap_mib']),
                             (3<<30,4096,22528))
            self.assertFalse(a_contract['rankperf']); self.assertFalse(a_contract['profiler'])
            self.assertIsNone(a_contract['unknown_pressure']); self.assertIsNone(a_contract['pending_peak'])
        for cycle in (0,3,True,'1'):
            with self.assertRaises(ValueError): a.slots(cycle)

    def test_unregistered_manifest_and_missing_hash_fail_closed(self):
        self.assertEqual(a.reviewed_build(a.REVIEWED_BUILD)["after_binary_sha256"], p.retirement_profile.release.BINARY_SHA)
        with mock.patch.object(a, 'frozen_audit') as audit, self.assertRaisesRegex(ValueError, 'Unregistered'):
            a.validate(self.fixture())
        audit.assert_not_called()
        with mock.patch.object(p, 'strict_gate'), self.assertRaisesRegex(ValueError, 'Unregistered'):
            p.validate_admission(self.fixture(), 'operator_retirement_ab')
        with tempfile.TemporaryDirectory(dir=p.HERE/'admission') as tmp:
            output = Path(tmp)/'unaccepted.json'
            with self.assertRaisesRegex(ValueError, 'Explicit final'):
                a.admission(output, Path('/fixture/pending'), None, Path('/fixture/pending'), None, 1)
            self.assertFalse(output.exists())

    def test_registered_review_requires_exact_path_hash_and_shape(self):
        link = dict(path='/fixture/review.json',sha256='1'*64)
        with mock.patch.object(a, 'REVIEWED_BUILD', link):
            for wrong in (dict(link,sha256='2'*64),dict(link,path='/fixture/other.json')):
                with self.assertRaisesRegex(ValueError, 'Unregistered'): a.reviewed_build(wrong)
            with mock.patch.object(a.pre, 'checked_document', return_value={}), self.assertRaises(ValueError):
                a.reviewed_build(link)

    def test_admission_requires_release_audit_then_same_binary_smoke(self):
        admission = self.fixture(); stack, release, smoke, _ = self.validation_context(admission)
        with stack:
            self.assertEqual(a.validate(admission), admission)
            release.assert_called_once_with(dict(protocol=a.pre.PROTOCOL, engines=admission['engines']), 'serving_smoke')
            smoke.assert_called_once_with(admission['engines']['ferrule'])
        for failed in ('release','smoke'):
            stack, release, smoke, _ = self.validation_context(admission)
            (release if failed == 'release' else smoke).side_effect = ValueError(failed+' pending')
            with stack, self.assertRaisesRegex(ValueError, failed+' pending'): a.validate(admission)
            if failed == 'release': smoke.assert_not_called()

    def test_arbitrary_override_and_scope_drift_rejected(self):
        for change in ('allowlist','profiler','baseline','cycle','workspace','rankperf','runtime','cache','model'):
            admission = self.fixture(); m = admission['engines']['ferrule']
            if change == 'allowlist': admission['allow_sources'] = ['*']
            if change == 'profiler': admission['profiler'] = {}
            if change == 'baseline': admission['before']['binary']['sha256'] = 'd'*64
            if change == 'cycle': admission['cycle'] = 2
            if change == 'workspace': m['spec']['argv'][m['spec']['argv'].index('--cuda-workspace-mb')+1] = '3072'
            if change == 'rankperf': m['spec']['environment']['FERRULE_LOG'] = 'info,ferrule_performance=debug'
            if change == 'runtime': m['runtime_pins']['unreviewed'] = 'd'*64
            if change == 'cache': m['decoded_cache']['enabled'] = False
            if change == 'model': m['model'] = {}
            stack, _, _, _ = self.validation_context(admission)
            with self.subTest(change=change), stack, self.assertRaises(ValueError): a.validate(admission)

    def test_review_delta_and_actual_quality_launch_identity_are_bound(self):
        for change in ('delta','quality','unpin'):
            admission = self.fixture(); stack, _, _, docs = self.validation_context(admission)
            if change == 'delta': docs['/fixture/delta.json']['test_source_changes'] = [{'ignored':'*'}]
            if change == 'quality': docs['/fixture/quality.json']['rows'][0]['launch_id'] = 'another-fresh-but-unqualified-launch'
            if change == 'unpin': admission['engines']['ferrule']['build_evidence_pins'].clear()
            with self.subTest(change=change), stack, self.assertRaises(ValueError): a.validate(admission)

    def test_existing_lifecycle_slot_and_execution_closure(self):
        admission = self.fixture(2)
        self.assertEqual(p.experiment_slots('operator_retirement_ab', admission), ('AFTER','BEFORE'))
        self.assertEqual(p.slot_metadata(admission, 'BEFORE'), admission['before'])
        self.assertEqual(p.slot_metadata(admission, 'AFTER'), admission['engines']['ferrule'])
        with self.assertRaises(ValueError): p.slot_metadata(admission, 'OFF')
        module = p.lifecycle(Path('/fixture'), 'ferrule', 'CPU fixture', 'operator_retirement_ab')
        self.assertEqual(module.PROTOCOL['id'], a.PROTOCOL)
        self.assertEqual(module.PROTOCOL['phases'], ['hello2','canary8','qa8-V1'])
        for method in ('run','worker','requests'):
            self.assertEqual(getattr(module, method).__code__.co_code, getattr(p.baseline, method).__code__.co_code)
        self.assertIs(module.base.supervisor, p.base.supervisor)
        self.assertIs(module.observer, p.baseline.observer)
        self.assertIn(p.HERE/'operator_retirement_ab.py', p.execution_paths('operator_retirement_ab'))

    def test_prepare_two_cycles_existing_slot_binding(self):
        with tempfile.TemporaryDirectory(dir=p.HERE/'runs') as tmp:
            root = Path(tmp)
            for cycle in (1,2):
                admission = self.fixture(cycle); path = root/(str(cycle)+'.json'); p.ct.write(path, admission)
                directory = root/('cycle-'+str(cycle))
                with mock.patch.object(p, 'strict_gate'), mock.patch.object(p, 'validate_admission', return_value=admission), \
                        mock.patch.object(p, 'pair_path', return_value=directory), mock.patch.object(a, 'frozen_audit', return_value={'closed':True}):
                    p.prepare('fixture', path, 'operator_retirement_ab')
                    prepared = p.validate_pair(directory)
                    self.assertEqual(prepared['order'], list(a.slots(cycle)))
                    self.assertNotIn('serving_scope_difference', prepared)
                    for slot in a.SLOTS:
                        self.assertEqual(p.validate_slot(directory/'runs'/slot)['engine_metadata'], p.slot_metadata(admission, slot))
                    self.assertFalse((directory/'launch.json').exists())

    def test_crossed_runner_fail_stop_without_launching(self):
        for cycle in (1,2):
            for fail in (False,True):
                with tempfile.TemporaryDirectory(dir=p.HERE/'runs') as tmp:
                    directory = Path(tmp); starts = []
                    def lifecycle(root, engine, note, purpose):
                        self.assertEqual((engine,purpose), ('ferrule','operator_retirement_ab'))
                        return SimpleNamespace(run=lambda slot: starts.append(slot) or int(fail))
                    prepared = dict(purpose='operator_retirement_ab', protocol=a.PROTOCOL, admission=self.fixture(cycle))
                    with mock.patch.object(p, 'strict_gate'), mock.patch.object(p, 'pair_path', return_value=directory), \
                            mock.patch.object(p, 'validate_pair', return_value=prepared), mock.patch.object(p, 'lifecycle', side_effect=lifecycle), \
                            mock.patch.object(p, 'assess_slot', return_value={'status':'complete'}), \
                            mock.patch.object(a, 'finish', return_value={'status':'complete_descriptive_only'}) as finish:
                        code = p.run('fixture', True, 'CPU fixture, launcher mocked', 'operator_retirement_ab')
                    self.assertEqual(code, int(fail))
                    self.assertEqual(starts, list(a.slots(cycle))[:1] if fail else list(a.slots(cycle)))
                    if fail: finish.assert_not_called()
                    else: finish.assert_called_once()

    def test_plan_does_not_audit_query_lock_or_launch(self):
        with mock.patch.object(a, 'frozen_audit', side_effect=AssertionError('No IO plan')), \
                mock.patch.object(a, 'qualification_identity', side_effect=AssertionError('No IO plan')):
            plan = a.plan()
        self.assertFalse(plan['execute_allowed']); self.assertEqual(plan['after_binary_sha256'], p.retirement_profile.release.BINARY_SHA)
        self.assertFalse(plan['profile']['rankperf']); self.assertFalse(plan['profile']['cache_activity_required'])
        self.assertFalse(plan['profile']['historical_4d40_is_retirement_baseline'])
        self.assertIsNone(plan['unknown_pressure']); self.assertIsNone(plan['pending_peak'])


    def test_prepare_pending_review_and_run_without_grant_have_no_side_effects(self):
        with tempfile.TemporaryDirectory(dir=p.HERE/'runs') as tmp:
            root = Path(tmp); path = root/'admission.json'; p.ct.write(path, self.fixture())
            destination = root/'never-created'
            with mock.patch.object(p, 'strict_gate'), mock.patch.object(p, 'pair_path', return_value=destination), \
                    mock.patch.object(p, 'lifecycle', side_effect=AssertionError('No launch')):
                with self.assertRaisesRegex(ValueError, 'Unregistered'):
                    p.prepare('fixture', path, 'operator_retirement_ab')
                with self.assertRaisesRegex(ValueError, 'Explicit renewed'):
                    p.run('fixture', False, '', 'operator_retirement_ab')
            self.assertFalse(destination.exists())

    def test_report_checks_actual_tokens_and_preserves_unknowns(self):
        for wrong_tokens in (False,True):
            with tempfile.TemporaryDirectory(dir=p.HERE/'runs') as tmp:
                root = Path(tmp); admission = self.fixture()
                request = dict(status='success', actual_input_tokens=5, actual_output_tokens=8,
                    text=p.shared.references()[0][0]['reference_text'])
                http = dict(first_after_ready=request, warmups=[request,request],
                    runs=[dict(requests=[deepcopy(request) for _ in range(5)], summary={'success':5})])
                if wrong_tokens: http['runs'][0]['requests'][0]['actual_output_tokens'] = 7
                for slot in a.SLOTS:
                    session = root/'runs'/slot/'session'; session.mkdir(parents=True)
                    (session/'server.log').write_text('info only')
                    (session/'samples.jsonl').write_text('')
                    p.ct.write(session/'case.json', dict(startup_seconds=1,
                        resources=dict(status='observed_within_targets', errors=[])))
                    for phase in ('hello2','canary8','qa8-V1'):
                        (session/phase).mkdir()
                        p.ct.write(session/phase/'http.json', http)
                        p.ct.write(session/phase/'gate.json', dict(status='pass',
                            http_receipt_sha256=p.ct.sha(session/phase/'http.json')))
                snapshots = [dict(schema_version=2, snapshot={'arena_backing_grants':3<<30})]
                with mock.patch.object(p, 'assess_slot', side_effect=lambda cell, engine: dict(launch_id=cell.name,
                        runtime=dict(allocator_retirement=dict(pending_peak=None, unknown_peak=None)))), \
                        mock.patch.object(a.release, 'snapshot_rows', return_value=snapshots):
                    if wrong_tokens:
                        with self.assertRaisesRegex(ValueError, 'actual tokens'): a.finish(root, {'admission':admission})
                        self.assertFalse((root/'operator-retirement-summary.json').exists())
                    else:
                        result = a.finish(root, {'admission':admission})
                        self.assertEqual(result['status'], 'complete_descriptive_only')
                        self.assertFalse(result['isolated_retirement_effect'])
                        for row in result['rows']:
                            self.assertEqual(row["allocator_retirement"], dict(pending_peak=None, unknown_peak=None))
                            for key in ("unknown_peak", 'activity','unknown_pressure','pending_peak','retirement_event_counts','cuda_api_event_counts'):
                                self.assertIsNone(row[key])
                        with self.assertRaisesRegex(ValueError, 'No retirement summary overwrite'):
                            a.finish(root, {'admission':admission})


if __name__ == '__main__':
    unittest.main()
