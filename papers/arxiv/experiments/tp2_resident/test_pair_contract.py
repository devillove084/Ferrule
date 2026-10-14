"""CPU-only planner tests; fixtures are declarations, never serving receipts."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pair_contract as pair


def declaration(engine):
    """Synthetic shape-complete metadata; must NEVER authorize execution."""
    value = dict(pair.common_contract(), engine=engine, origin='fresh_preparation',
        binary=dict(path='/fixture/not-executed', sha256='a'*64, kind='public_serving'),
        source_pins={'fixture-source': 'b'*64}, runtime_pins={'fixture-runtime': 'c'*64},
        environment_policy='isolated_allowlist',
        spec=dict(engine=engine, argv=['/fixture/not-executed'],
                  environment={'CUDA_VISIBLE_DEVICES': ','.join(pair.shared.UUIDS)}),
        model=dict(checkpoint_identity='fixture-only', identity_strength='fixture-only',
                   tokenizer_identity='fixture-only'),
        arithmetic=dict(profile=engine+'-distinct-arithmetic', scales='fixture',
            dequant_rounding='fixture', reduction='fixture', gdn='fixture', effective_evidence='fixture'),
        transport=dict(declared_environment='fixture', observed_channels='fixture', evidence_sha256='d'*64),
        cache=dict(state_reset='per_request', os_jit_page_cache_policy='record-uncontrolled-not-cold'),
        custody_policy=dict(identity_mode='device_exclusive_local_custody',
            sorted_physical_uuid_locks=True, atomic_release_on_failure=True,
            supervisor='authenticated_linux_subreaper', root_identity='pid_starttime',
            pre_post_idle=True, nonselected_baseline_delta_bytes=1 << 20,
            foreign_signals=False, owned_gpu_attribution='unknown', stop_on_unknown_drain=True,
            sole_gpu_owner=True))
    value['gates'] = {name: dict(status='pass', receipt_sha256='e'*64,
        binary_sha256=value['binary']['sha256'], source_pins=copy.deepcopy(value['source_pins']))
        for name in pair.REQUIRED_GATES}
    return value


class PairContracts(unittest.TestCase):
    def review(self, f=None, v=None):
        return pair.review_pair(f if f is not None else declaration('ferrule'),
                                v if v is not None else declaration('vllm'))

    def test_empty_is_blocked_and_never_claims_speedup(self):
        report = pair.review_pair()
        self.assertFalse(report['metadata_complete'])
        self.assertFalse(report['execute_allowed'])
        self.assertFalse(report['fairmatch'])
        self.assertIsNone(report['exact_speedup'])
        self.assertIn('public_serving_cli', ' '.join(report['metadata_blockers']))

    def test_even_complete_declarations_cannot_authorize_execution(self):
        report = self.review()
        self.assertTrue(report['metadata_complete'], report['metadata_blockers'])
        self.assertEqual(report['status'], 'blocked')
        self.assertFalse(report['execute_allowed'])
        self.assertFalse(report['arithmetic_equivalence'])
        self.assertTrue(report['execution_blockers'])
        self.assertEqual(report['proposed_order'], [['F1', 'V1'], ['V2', 'F2']])

    def test_test_binary_and_logits_only_gate_are_insufficient(self):
        f = declaration('ferrule')
        f['binary']['kind'] = 'test'
        f['gates'] = {'logits': dict(status='pass', compared=1986560)}
        reasons = ' '.join(self.review(f)['metadata_blockers'])
        for word in ('test binary', 'public_serving_cli', 'all_boundary', 'multistep',
                     'service_lifecycle', 'serving_quality'):
            self.assertIn(word, reasons)

    def test_gates_must_bind_same_binary_and_sources(self):
        for key, value in (('binary_sha256', 'f'*64), ('source_pins', {'other': 'f'*64}),
                           ('receipt_sha256', 'unknown'), ('status', 'fail')):
            f = declaration('ferrule')
            f['gates']['serving_quality'][key] = value
            self.assertIn('serving_quality', ' '.join(self.review(f)['metadata_blockers']))

    def test_profile_mismatch_and_offload_fail_closed(self):
        mutations = [('topology', 'tp', 1), ('topology', 'ep', 2),
            ('topology', 'enable_expert_parallel', True),
            ('topology', 'gpu_uuids', list(reversed(pair.shared.UUIDS))),
            ('boundaries', 'kv', 'unknown'), ('workload', 'cpu_offload_gb', 1),
            ('workload', 'eos', 'ignore'), ('workload', 'seed', 0),
            ('resources', 'host_target_bytes', 64*(1 << 30)),
            ('limits', 'startup_seconds', 901)]
        for group, key, value in mutations:
            with self.subTest(group=group, key=key):
                f = declaration('ferrule')
                f[group][key] = value
                self.assertFalse(self.review(f)['metadata_complete'])

    def test_unknown_effective_arithmetic_and_transport_block(self):
        for group, key in (('arithmetic', 'effective_evidence'), ('arithmetic', 'dequant_rounding'),
                           ('transport', 'observed_channels'), ('model', 'identity_strength')):
            f = declaration('ferrule')
            f[group][key] = 'unknown'
            self.assertFalse(self.review(f)['metadata_complete'])

    def test_bad_custody_does_not_admit(self):
        for key, value in (('foreign_signals', True), ('sole_gpu_owner', False),
                           ('supervisor', 'unknown'), ('stop_on_unknown_drain', False)):
            f = declaration('ferrule')
            f['custody_policy'][key] = value
            self.assertIn('custody', ' '.join(self.review(f)['metadata_blockers']))

    def test_pins_and_environment_are_engine_specific(self):
        for field in ('source_pins', 'runtime_pins'):
            f = declaration('ferrule')
            f[field] = {'path': 'x'}
            self.assertFalse(self.review(f)['metadata_complete'])
        v = declaration('vllm')
        v['spec']['environment']['LD_LIBRARY_PATH'] = '/fixture/CUDA12.4'
        self.assertIn('leak', ' '.join(self.review(v=v)['metadata_blockers']))
        v = declaration('vllm')
        v['spec']['engine'] = 'ferrule'
        self.assertFalse(self.review(v=v)['metadata_complete'])

    def test_archived_baseline_cannot_supply_new_pair(self):
        v = declaration('vllm')
        v['origin'] = 'tp2-resident-single-lifecycle-20261009-001'
        self.assertIn('archived timings', ' '.join(self.review(v=v)['metadata_blockers']))

    def test_model_and_cache_scope_must_match(self):
        for group, key in (('model', 'tokenizer_identity'), ('cache', 'os_jit_page_cache_policy')):
            f = declaration('ferrule')
            f[group][key] = 'different'
            self.assertIn('paired '+group, ' '.join(self.review(f)['metadata_blockers']))

    def test_http_plan_reuses_parser_and_original_reference_without_execution(self):
        shared = pair.shared
        with mock.patch.object(shared.http, 'main', side_effect=AssertionError('HTTP forbidden')), \
                mock.patch.object(shared, 'owner_locks', side_effect=AssertionError('locks forbidden')), \
                mock.patch.object(shared.base.supervisor, 'run_process', side_effect=AssertionError('launch forbidden')):
            for engine in ('ferrule', 'vllm'):
                spec = dict(engine=engine, url='http://127.0.0.1:18367/v1/completions', model='fixture')
                for stage in ('hello2', 'canary8', 'qa8'):
                    plan = pair.http_plan(engine, spec, stage, Path('/fixture/not-created')/stage)
                    args = shared.http.parser().parse_args(plan['argv'])
                    self.assertEqual(args.engine, engine)
                    self.assertEqual(args.timeout, 120)
                    self.assertEqual(args.seed, 1)
                    self.assertEqual(args.concurrency, 1)
                    if stage == 'qa8':
                        self.assertEqual((args.requests, args.warmup, args.repeat), (5, 2, 1))
                        self.assertTrue(args.first_after_ready)
                    if stage == 'canary8':
                        refs = plan['canary_expectations']
                        self.assertEqual([r['id'] for r in refs], ['qa', 'arithmetic', 'chinese'])
                        self.assertTrue(all(len(r['reference_ids']) == 8 for r in refs))
                        self.assertEqual(refs[0]['reference_text'], ' Tokyo.\nThe capital of Japan is')
            self.review()

    def test_http_identity_endpoint_and_reference_fail_closed(self):
        spec = dict(engine='ferrule', url='http://127.0.0.1:18367/v1/completions', model='fixture')
        for engine, url in [('vllm', spec['url']), ('ferrule', 'https://remote/v1/completions'),
                            ('ferrule', 'http://user@127.0.0.1:18367/v1/completions')]:
            with self.assertRaises(ValueError):
                pair.http_plan(engine, dict(spec, url=url), 'canary8', '/fixture')
        with mock.patch.object(pair.shared, 'references', return_value=([], {})):
            with self.assertRaises(ValueError):
                pair.http_plan('ferrule', spec, 'canary8', '/fixture')


if __name__ == '__main__':
    unittest.main()
