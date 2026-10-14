"""Small offline regressions; receipt mutations stay in memory, never in runs/."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('storage_demand_analysis', ANALYSIS / 'summarize.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class SummarizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = M.load_sources()
        cls.bindings = json.loads((ANALYSIS / 'inputs.json').read_text(encoding='utf-8'))['files']
        cls.summary = M.analyze(cls.sources, cls.bindings)

    def fields(self):
        return dict({key: 1 for key in M.DELTAS}, **{key: 7 for key in M.GAUGES},
                    counters_available=True, counter_reset_detected=False)

    def test_missing_counter_and_reset_are_not_zero(self):
        fields = self.fields()
        del fields['h2d_bytes']
        result = M.aggregate([{'fields': fields}])
        self.assertFalse(result['valid'])
        self.assertIsNone(result['deltas']['h2d_bytes'])
        self.assertEqual(result['deltas']['h2d_copies'], 1)
        fields['counter_reset_detected'] = True
        self.assertTrue(all(value is None for value in M.aggregate([{'fields': fields}])['deltas'].values()))
        fields['counter_reset_detected'] = False
        fields['expert_hits'] = False
        self.assertIsNone(M.aggregate([{'fields': fields}])['deltas']['expert_hits'])

    def test_gauges_are_last_and_maximum_never_sums(self):
        a, b = self.fields(), self.fields()
        a.update(expert_resident_entries=2, expert_owner_peak_bytes=10)
        b.update(expert_resident_entries=3, expert_owner_peak_bytes=9)
        result = M.aggregate([{'fields': a}, {'fields': b}])
        self.assertEqual(result['deltas']['h2d_copies'], 2)
        self.assertEqual(result['last_gauges']['expert_resident_entries'], 3)
        self.assertEqual(result['observed_gauge_maxima']['expert_resident_entries'], 3)
        self.assertEqual(result['last_gauges']['expert_owner_peak_bytes'], 9)
        self.assertEqual(result['observed_gauge_maxima']['expert_owner_peak_bytes'], 10)

    def test_failed_before_launch_keeps_planned_not_run_and_null_counters(self):
        attempt = self.summary['attempts'][0]
        self.assertEqual(attempt['denominator'], dict(planned=4, dispatched=0, not_run=4,
                         success=0, failed_dispatched=0, successful_output_tokens=0, invocations=0))
        self.assertEqual(attempt['failure_stage'], 'before_server_launch')
        self.assertIn('DEVNULL', attempt['failure_error'])
        self.assertTrue(all(value is None for value in attempt['additive_deltas'].values()))
        self.assertIsNone(attempt['timing']['startup_seconds'])
        self.assertEqual(attempt['shutdown']['resource_contract_complete'], 'unknown')

    def test_actual_totals_memory_and_no_generated_ids(self):
        self.assertEqual(self.summary['consistency']['status'], 'pass')
        self.assertEqual(self.summary['consistency']['divergences'], [])
        attempt = self.summary['attempts'][1]
        expected = dict(checkpoint_read_calls=69114, checkpoint_read_bytes=36240064128,
                        h2d_copies=88763, h2d_bytes=36240248688, expert_uploads=11519,
                        expert_evictions=11455, artifact_uploads=34557, expert_hits=0)
        for key, value in expected.items():
            self.assertEqual(attempt['additive_deltas'][key], value)
        self.assertEqual(attempt['additive_deltas'], attempt['per_request_delta_sum'])
        self.assertEqual(attempt['denominator'], dict(planned=4, dispatched=4, not_run=0,
                         success=4, failed_dispatched=0, successful_output_tokens=26, invocations=30))
        self.assertEqual(attempt['sampled_memory']['sample_count'], 32)
        self.assertEqual(attempt['sampled_memory']['tree_pss_peak_bytes'], 7980569600)
        self.assertIsNone(attempt['sampled_memory']['tree_pss_anon_peak_bytes'])
        self.assertTrue(all(row['observed_generated_token_ids'] is None for row in attempt['requests']))
        self.assertTrue(all(row['exact_token_pass'] is None for row in attempt['requests']))
        self.assertTrue(all(row['reference_ids_are_observed_generated_ids'] is False for row in attempt['requests']))

    def test_success_without_reference_is_not_reference_quality_passed(self):
        leaf = M.BASE + M.RUNS[1] + '/hello2/observed.json'
        frozen = json.loads(self.sources[leaf])
        self.assertEqual(frozen['status'], 'pass')
        self.assertEqual(frozen['actual']['status'], 'success')
        self.assertIsNone(frozen['reference'])
        self.assertIn('frozen HF generation8', frozen['quality_scope'])
        quality = M.request_quality(frozen)
        self.assertEqual(quality['quality_scope'],
                         'HTTP completion and usage checks only; no reference comparison')
        self.assertFalse(quality['reference_comparison_performed'])
        self.assertIsNone(quality['reference_text_match'])
        self.assertIsNone(quality['reference_output_usage_match'])
        self.assertIsNone(quality['reference_quality_passed'])
        attempt = self.summary['attempts'][1]
        hello = next(row for row in attempt['requests'] if row['name'] == 'hello2')
        for key, value in quality.items():
            self.assertEqual(hello[key], value)
        passed = [row['name'] for row in attempt['requests'] if row['reference_quality_passed'] is True]
        self.assertEqual(passed, ['canary-qa', 'canary-arithmetic', 'canary-chinese'])
        self.assertEqual(attempt['quality']['reference_comparison_requests'], 3)
        self.assertEqual(attempt['quality']['reference_quality_passed_requests'], 3)
        self.assertEqual(attempt['quality']['http_only_success_requests'], 1)
        self.assertEqual(attempt['denominator']['success'], 4)
        report = M.report_text(self.summary).decode('utf-8')
        self.assertIn('Only 3 canaries', report)
        self.assertIn('No exact generated IDs were observed', report)
        self.assertIn('none of these checks establishes numeric acceptance', report)
        self.assertEqual(frozen, json.loads(self.sources[leaf]))

    def test_stale_hash_and_semantic_report_divergence(self):
        sources = dict(self.sources)
        path = M.BASE + M.RUNS[1] + '/report.json'
        report = json.loads(sources[path])
        report['case']['all_invocation_counters']['deltas']['h2d_bytes'] += 1
        sources[path] = M.encode(report)
        result = M.analyze(sources, self.bindings)
        checks = [row['check'] for row in result['consistency']['divergences']]
        self.assertIn('input_sha256/' + path, checks)
        self.assertIn(M.RUNS[1] + '/case_report', checks)
        updated = dict(self.bindings, **{path: M.digest(sources[path])})
        result = M.analyze(sources, updated)
        self.assertEqual(result['consistency']['status'], 'fail')
        self.assertIn(M.RUNS[1] + '/case_report', [row['check'] for row in result['consistency']['divergences']])
        leaf = M.BASE + M.RUNS[1] + '/hello2/observed.json'
        sources = dict(self.sources, **{leaf: self.sources[leaf] + b'\n'})
        result = M.analyze(sources, self.bindings)
        self.assertIn('receipt_pin/' + leaf, [row['check'] for row in result['consistency']['divergences']])

    def test_deterministic_outputs_and_stale_summary(self):
        expected = M.render(self.summary)
        self.assertEqual(expected, M.render(M.analyze(self.sources, self.bindings)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as name:
            directory = Path(name)
            for leaf, data in expected.items():
                (directory / leaf).write_bytes(data)
            self.assertEqual(M.check_outputs(directory, expected), [])
            changed = json.loads(expected['summary.json'])
            changed['attempts'][1]['additive_deltas']['h2d_bytes'] += 1
            (directory / 'summary.json').write_bytes(M.encode(changed))
            self.assertEqual(M.check_outputs(directory, expected), ['summary.json'])

    def test_malformed_log_or_byte_anchor_fails_closed(self):
        with self.assertRaises(M.SchemaError):
            M.invocations(b'{"event":"mixed_decoder_invocation", broken}\n')
        with self.assertRaises(M.SchemaError):
            M.invocations(b'plan\n', 1, 5)
        with self.assertRaises(M.SchemaError):
            M.invocations(b'plan\n', 0, 99)


if __name__ == '__main__':
    unittest.main()
