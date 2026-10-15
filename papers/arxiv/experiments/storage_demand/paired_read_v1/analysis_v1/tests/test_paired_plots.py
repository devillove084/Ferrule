"""Offline stdlib regressions; all mutations are in-memory or temporary, never raw inputs."""
import copy
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import derive as D
import receipts as R
import study as S
import plot_paired_read as P


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources, cls.manifest, cls.manifest_hash = D.load_sources()
        cls.data = S.build_data(cls.sources, cls.manifest, cls.manifest_hash)
        cls.final = json.loads(cls.sources[D.RAW + 'paired-final.json'])
        cls.audit = json.loads(cls.sources[D.PLAN + 'poststudy-audit.json'])
        cls.request = json.loads(cls.sources[D.RAW + 'P1-blocking/qa-measured-1/observed.json'])
        log = cls.sources[D.RAW + 'P1-blocking/server.log']
        cls.window = D.log_rows(log[cls.request['log_start_byte']:cls.request['log_end_byte']])

    def test_exact_study_inventory_and_denominators(self):
        self.assertEqual(len(self.sources), 840)
        self.assertEqual(self.data['design']['n'], 5)
        self.assertEqual(self.data['design']['total_measured_requests'], 50)
        self.assertEqual(self.data['paired']['qualified_blocks'], 5)
        self.assertEqual(self.data['paired']['failed_or_incomplete_blocks'], 0)
        self.assertEqual(self.data['denominators'], dict(scheduled=120, issued=120, terminal=120,
                                                       successful=120, failed=0, censored=0, unissued=0))
        self.assertEqual([r['modes'] for r in self.data['design']['order']],
                         [['blocking', 'lookahead'], ['blocking', 'lookahead'], ['lookahead', 'blocking'],
                          ['lookahead', 'blocking'], ['blocking', 'lookahead']])
        self.assertFalse(any('artifacts-v3-final-' in p or 'artifacts-current-' in p for p in self.sources))
        self.assertEqual(len(self.data['excluded_historical_attempts']), 2)
        self.assertTrue(all(not r['pooled'] and r['unissued_requests'] == 120
                            for r in self.data['excluded_historical_attempts']))

    def test_exact_raw_estimates_and_nonpooled_goodput(self):
        expected = (.9182346584653581, .8875530391241149, 1.1090822446721382)
        for key, value in zip(D.METRICS, expected):
            metric = self.data['paired']['aggregate'][key]
            self.assertEqual(metric['ratio'], value)
            self.assertEqual(metric['draws'], 3125)
            self.assertEqual(metric['nearest_ranks_1based'], [79, 3047])
        for arm in self.data['arms']:
            measured = [q for q in arm['requests'] if q['measured']]
            self.assertEqual(len(measured), 5)
            self.assertEqual(arm['metrics']['request_e2e_seconds'], statistics.median(q['request_e2e_seconds'] for q in measured))
            self.assertEqual(arm['metrics']['qualified_goodput_tokens_per_second'],
                             40 / sum(q['request_e2e_seconds'] for q in measured))
        constant = D.paired_estimate([math.log(.8)] * 5)
        self.assertEqual(constant['descriptive_range'], [.8, .8])
        ordered = D.paired_estimate([math.log(v) for v in (.7, .8, .9, 1., 1.1)])
        self.assertEqual(ordered['descriptive_range'], [.7, 1.1])
        with self.assertRaises(D.EvidenceError):
            D.paired_estimate([0.] * 50)

    def test_failed_pair_is_retained_and_suppresses_aggregate(self):
        arms = {r['trial']: r['metrics'] for r in self.data['arms']}
        arms['P3-lookahead'] = None
        summary = D.paired_summary(arms)
        self.assertIsNone(summary['aggregate'])
        self.assertEqual(summary['qualified_blocks'], 4)
        self.assertEqual(summary['failed_or_incomplete_blocks'], 1)
        self.assertEqual(len(summary['blocks']), 5)
        self.assertTrue(summary['blocks'][2]['failure_pair_retained'])
        del arms['P3-lookahead']
        with self.assertRaisesRegex(D.EvidenceError, 'all ten'):
            D.paired_summary(arms)

    def test_input_copy_mutation_cannot_feed_plots(self):
        sources = dict(self.sources)
        sources[D.RAW + 'P1-blocking/server.log'] += b'\n'
        with self.assertRaisesRegex(D.EvidenceError, 'input copy hash mismatch'):
            S.build_data(sources, self.manifest, self.manifest_hash)
        sources = dict(self.sources)
        sources.pop(D.RAW + 'P5-lookahead/requests-final.json')
        with self.assertRaisesRegex(D.EvidenceError, 'input copy set differs'):
            S.build_data(sources, self.manifest, self.manifest_hash)

    def check_first_arm(self, sources):
        load = lambda p: json.loads(sources[p])
        return R.arm_data('P1-blocking', 'blocking', self.final['lifecycles'][0], self.audit['arms'][0],
                          {(r['trial'], r['name']): r for r in self.audit['requests']},
                          load(D.BASE + 'requests.json')['schedule'],
                          {r['id']: r for r in load(D.BASE + 'oracles.json')['references']},
                          sources, D.RAW + 'P1-blocking/', load(D.BASE + 'cases-v3.json'))

    def test_copied_request_and_runtime_timeline_are_reconciled(self):
        sources = dict(self.sources)
        path = D.RAW + 'P1-blocking/qa-measured-1/observed.json'
        changed = copy.deepcopy(self.request)
        changed['actual']['request_e2e_seconds'] += 1
        sources[path] = D.encode(changed)
        with self.assertRaisesRegex(D.EvidenceError, 'observed request input copy differs'):
            self.check_first_arm(sources)
        sources = dict(self.sources)
        sources[D.RAW + 'P1-blocking/server.log'] += b'{}\n'
        with self.assertRaisesRegex(D.EvidenceError, 'server timeline hash differs'):
            self.check_first_arm(sources)

    def test_actual_tokens_ordinals_http_ids_and_oracles(self):
        self.assertEqual(self.data['coverage'], dict(requests=120, measured_qa=50,
            actual_worker_generated_tokens=900, exact_eight_token_oracles=110,
            length_only_two_token_smoke=10, pipeline_invocation_jsons=1020))
        oracle = self.request['reference']
        good = R.check_tokens(self.request, self.window, oracle)
        self.assertEqual(good['completion_tokens'], 8)
        for field, value, error in (
            ('enqueue_ordinals', [1, 2, 3, 4, 5, 6, 8, 7], 'ordinals'),
            ('actual_generated_ids', [0] * 8, 'token IDs'),
            ('actual_http_completion_ids', ['cmpl-ferrule-999'], 'HTTP/worker'),
            ('exact_token_pass', None, 'oracle')):
            changed = copy.deepcopy(self.request)
            changed['trace'][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(D.EvidenceError, error):
                R.check_tokens(changed, self.window, oracle)
        rows = copy.deepcopy(self.window)
        next(r for r in rows if r['fields'].get('event') == 'request_worker_token_enqueued')['fields']['actual_token_id'] = 0
        with self.assertRaisesRegex(D.EvidenceError, 'token IDs'):
            R.check_tokens(self.request, rows, oracle)
        changed = copy.deepcopy(self.request)
        changed['actual']['usage']['completion_tokens'] = 9
        with self.assertRaisesRegex(D.EvidenceError, 'usage'):
            R.check_tokens(changed, self.window, oracle)

    def test_smoke_exact_acceptance_is_unknown_not_false_or_true(self):
        for arm in self.data['arms']:
            smoke = arm['requests'][0]
            self.assertEqual(smoke['completion_tokens'], 2)
            self.assertIsNone(smoke['exact_token_oracle'])
            self.assertTrue(all(q['exact_token_oracle'] is True for q in arm['requests'][1:]))

    def test_invocation_ordinals_are_not_tokens_or_replicates(self):
        for arm in self.data['arms']:
            self.assertEqual([r['ordinal'] for r in arm['timeline']], list(range(1, 103)))
            self.assertEqual([r['source_log_line'] for r in arm['timeline']],
                             [line for q in arm['requests'] for line in q['source_log_lines']])
            self.assertEqual([q['decoder_invocations'] for q in arm['requests']], [3] + [9] * 11)
            self.assertEqual(sum(q['completion_tokens'] for q in arm['requests']), 90)

    def test_wait_is_sum_within_request_then_median_within_arm(self):
        for arm in self.data['arms']:
            measured = [q for q in arm['requests'] if q['measured']]
            waits = []
            for q in measured:
                raw_ns = sum(r['demand_wait_ns'] for r in arm['timeline'] if r['request'] == q['name'])
                self.assertEqual(raw_ns, q['pipeline_nonpeak_deltas']['demand_wait_ns'])
                waits.append(raw_ns / 1e9)
            self.assertEqual(arm['median_request_demand_wait_seconds'], statistics.median(waits))
        self.assertTrue(all(r['wait_change_seconds'] < 0 and r['e2e_change_seconds'] < 0
                            for r in self.data['wait_latency_changes']))
        self.assertIsNone(self.data['unknowns']['causal_critical_path_savings_seconds'])

    def test_binary_units_gauges_and_zero_versus_unknown(self):
        self.assertEqual(D.units(16777984, D.MIB), 16.000732421875)
        self.assertEqual(D.units(4325376000, D.MIB), 4125)
        self.assertEqual(D.units(7 * D.GIB, D.MIB), 7168)
        self.assertEqual(D.units(0, D.MIB), 0)
        for v in (None, False, -1, float('nan'), float('inf')):
            with self.subTest(v=v), self.assertRaises(D.EvidenceError):
                D.units(v, D.MIB)
        with self.assertRaises(D.EvidenceError):
            D.units(1000000, 1000000)
        self.assertEqual(D.gauge([10, 7]), dict(last=7, maximum=10))
        self.assertEqual(D.gauge([None, 0]), dict(last=0, maximum=None))
        self.assertEqual(D.gauge([0, False]), dict(last=None, maximum=None))
        self.assertEqual(D.gauge([]), dict(last=None, maximum=None))
        self.assertTrue(all(v is None for v in self.data['unknowns'].values()))

    def test_equal_work_zero_waste_and_separate_memory_scopes(self):
        for arm in self.data['arms']:
            self.assertEqual(arm['logical_counters']['checkpoint_read_bytes'], 128811263616)
            self.assertEqual(arm['logical_counters']['h2d_bytes'], 128811927312)
            self.assertEqual(arm['logical_counters']['h2d_bytes'] - arm['logical_counters']['checkpoint_read_bytes'], 663696)
            self.assertEqual(arm['pipeline_stats']['wasted_read_bytes'], 0)
            self.assertEqual(arm['pipeline_stats']['unknown_bytes'], 0)
            self.assertEqual(arm['host_logical_peak_bytes'], 16777984)
            self.assertEqual(arm['host_logical_limit_bytes'], 64 * D.MIB)
            self.assertEqual(arm['sampled_device_max_bytes'], 4125 * D.MIB)
            self.assertEqual(arm['configured_device_target_bytes'], 7 * D.GIB)
            self.assertIsNone(arm['owned_process_gpu_peak_bytes'])
            self.assertIsNone(arm['startup_pid_io_delta'])
            self.assertIsNone(arm['shutdown_pid_io_delta'])
            self.assertGreater(arm['startup_probe_seconds'], 12)
            self.assertLess(arm['startup_probe_seconds'], 14)
            self.assertNotIn('pss_bytes', arm)

    def test_nfs_phase_is_not_identical_to_sum_of_request_windows(self):
        p1 = self.data['arms'][0]
        self.assertEqual(p1['nfs_phase_client_serverreadbytes']['requests_nfs'], 314825320)
        self.assertEqual(p1['nfs_request_window_sum_bytes'], 312732264)
        self.assertEqual(p1['nfs_between_request_windows_bytes'], 2093056)
        self.assertTrue(all(r['nfs_phase_client_serverreadbytes']['requests_nfs'] == 0 for r in self.data['arms'][1:]))
        self.assertTrue(all(r['physical_nas_backend_bytes'] is None for r in self.data['arms']))

    def test_actual_source_identity_does_not_reuse_historical_member_count(self):
        binding = json.loads(self.sources[D.PLAN + 'actual-binding-v3.json'])
        review = json.loads(self.sources[D.PLAN + 'actual-build-review-v3.json'])
        self.assertEqual(S.validate_identity(self.audit, binding, review)['source_members'], 1484)
        changed = copy.deepcopy(review)
        changed['source_members'] -= 1
        with self.assertRaisesRegex(D.EvidenceError, 'member count'):
            S.validate_identity(self.audit, binding, changed)

    def test_no_private_raw_projection_and_hash_binding(self):
        serialized = D.encode(self.data).decode()
        for forbidden in ('/home/', 'mountpoint', 'raw_sse_events', 'environment', 'NAS_EXPORT', 'maps.raw'):
            # The prose privacy policy says "environment"; no environment field may be emitted.
            needle = '"' + forbidden + '"' if forbidden == 'environment' else forbidden
            self.assertNotIn(needle, serialized)
        self.assertEqual(self.data['provenance']['input_manifest_sha256'], self.manifest_hash)
        self.assertEqual(self.data['provenance']['source_hashes'], S.source_hashes())
        self.assertEqual(len(self.data['provenance']['source_hashes']), 6)

    def test_check_is_byte_exact_and_never_rewrites_stale_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'plot-data.json'
            path.write_bytes(b'old\n')
            self.assertEqual(P.check_outputs(root, {'plot-data.json': b'new\n'}), ['plot-data.json'])
            self.assertEqual(path.read_bytes(), b'old\n')
            self.assertEqual(P.check_outputs(root, {'plot-data.json': b'old\n'}), [])
            self.assertEqual(P.check_outputs(root, {'missing.png': b''}), ['missing.png'])

    def test_caption_claim_boundaries_and_exact_labels(self):
        captions = P.captions(self.data)
        self.assertIn('3,125', captions['latency'])
        self.assertIn('n=5', captions['latency'])
        self.assertIn('not proof', captions['read-wait'])
        self.assertIn('unobserved (null), not zero', captions['accounting'])
        guide = P.guide(self.data).decode()
        for label in ('f:paired-latency', 'f:paired-read-wait', 'f:paired-accounting'):
            self.assertIn(label, guide)


if __name__ == '__main__':
    unittest.main()
