"""Receipt-grounded plot regressions; no output or raw-receipt mutation."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('storage_demand_plot', ANALYSIS / 'plot_storage_demand.py')
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class PlotDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary, cls.sources = P.checked_summary()
        cls.data = P.build_data(cls.summary, cls.sources)

    def test_binary_units_and_paired_byte_equation(self):
        self.assertEqual(P.units(2**30, P.GIB), 1)
        self.assertEqual(P.units(2**20, P.MIB), 1)
        self.assertEqual(P.units(2**10, 2**10), 1)
        requests = self.data['requests']
        self.assertEqual(sum(r['logical_read_bytes'] for r in requests), 36240064128)
        self.assertEqual(sum(r['facade_h2d_bytes'] for r in requests), 36240248688)
        self.assertEqual(sum(r['h2d_minus_read_bytes'] for r in requests), 184560)
        for row in requests:
            self.assertEqual(row['facade_h2d_bytes'] - row['logical_read_bytes'], row['h2d_minus_read_bytes'])
            self.assertAlmostEqual(row['h2d_minus_read_kib'], row['h2d_minus_read_bytes'] / 1024)

    def test_upload_eviction_entry_reconciliation_is_not_byte_conservation(self):
        self.assertEqual([r['uploads'] - r['evictions'] for r in self.data['requests']], [64, 0, 0, 0])
        self.assertEqual(self.data['reconciliation']['total_net_entries'], 64)
        self.assertEqual(self.data['reconciliation']['last_observed_entries'], 64)
        self.assertEqual(self.data['reconciliation']['implied_initial_entries'], 0)
        self.assertIsNone(self.data['reconciliation']['initial_entries_observed'])
        self.assertTrue(all(point['balance_residual'] in (None, 0) for point in self.data['timeline']))
        self.assertEqual(sum(r['uploads'] for r in self.data['requests']), 11519)
        self.assertEqual(sum(r['evictions'] for r in self.data['requests']), 11455)

    def test_gauges_use_last_and_max_not_additive_sums(self):
        by_key = {g['key']: g for g in self.data['gauges']}
        self.assertEqual(by_key['expert_resident_entries']['last'], 64)
        self.assertEqual(by_key['expert_resident_entries']['maximum'], 64)
        self.assertEqual(by_key['expert_resident_bytes']['maximum'], 201351168)
        self.assertEqual(by_key['expert_scratch_bytes']['maximum'], 8388608)
        self.assertEqual(by_key['cache.peak_bytes']['source_key'], 'expert_owner_peak_bytes')
        self.assertEqual(by_key['cache.peak_bytes']['maximum'], 219754496)
        self.assertEqual(by_key['cache.peak_bytes']['configured_limit'], 335544320)
        self.assertEqual(by_key['cache.peak_bytes']['configured_limit_label'], 'Cache accounting ceiling')
        # The four request-end gauges are observations, not four cache copies.
        self.assertNotEqual(sum(r['last_resident_entries'] for r in self.data['requests']),
                            by_key['expert_resident_entries']['maximum'])

    def test_cache_peak_is_charged_high_water_not_owner_physical_memory(self):
        accounting = self.data['cache_accounting']
        self.assertEqual(accounting['field'], 'cache.peak_bytes')
        self.assertEqual(accounting['source_field'], 'expert_owner_peak_bytes')
        self.assertEqual(accounting['meaning'], 'cache charged-byte high-water: resident + pending + scratch')
        self.assertEqual(accounting['peak_bytes'], 219754496)
        self.assertAlmostEqual(P.units(accounting['peak_bytes'], P.MIB), 209.57421875)
        self.assertEqual(accounting['accounting_ceiling_field'], 'ExpertCacheLimits.max_bytes')
        self.assertEqual(accounting['accounting_ceiling_bytes'], 335544320)
        self.assertEqual(accounting['accounting_ceiling_label'], 'Cache accounting ceiling')
        self.assertIsNone(accounting['owner_physical_peak_bytes'])
        self.assertIsNone(accounting['owner_physical_ceiling_bytes'])
        self.assertTrue(all(point['cache_peak_bytes'] >= point['cache_charged_bytes']
                            for point in self.data['timeline']))
        self.assertTrue(all(point['cache_peak_bytes'] <= 335544320 for point in self.data['timeline']))
        for point in self.data['timeline']:
            gauges = point['gauges']
            self.assertEqual(point['cache_charged_bytes'], gauges['expert_resident_bytes']
                             + gauges['expert_pending_upload_bytes'] + gauges['expert_scratch_bytes'])
            self.assertEqual(point['cache_peak_bytes'], gauges['expert_owner_peak_bytes'])
        self.assertEqual(self.data['timeline'][-1]['cache_charged_bytes'], 209739776)
        self.assertLess(self.data['timeline'][-1]['cache_charged_bytes'], accounting['peak_bytes'])
        self.assertIsNone(self.data['unobserved']['owner_physical_peak_bytes'])
        self.assertIsNone(self.data['unobserved']['owner_physical_ceiling_bytes'])
        self.assertTrue(all(g['configured_limit_label'] != 'Owner physical ceiling' for g in self.data['gauges']))

    def test_cache_caption_uses_exact_terminology_and_latex_safe_identifiers(self):
        caption = self.data['captions']['cache-reconciliation']
        self.assertIn('cache.peak_bytes value, 209.574 MiB', caption)
        self.assertIn('resident + pending + scratch', caption)
        self.assertIn('ExpertCacheLimits.max_bytes = 320 MiB, a cache accounting ceiling', caption)
        self.assertIn('No independently recorded owner physical peak or owner physical ceiling', caption)
        self.assertIn('no contradiction is implied', caption)
        latex = P.latex_caption(caption)
        self.assertIn(r'\texttt{cache.peak\_bytes}', latex)
        self.assertIn(r'\texttt{ExpertCacheLimits.max\_bytes}', latex)
        guide = P.plot_guide(self.data).decode()
        self.assertNotIn('resident-byte cap', guide)
        self.assertNotIn('The owner-peak gauge has no separately recorded ceiling', guide)
        self.assertIn('cache accounting ceiling', guide)

    def test_configured_ceilings_have_distinct_domains_and_units(self):
        limits = self.data['configured_limits']
        self.assertEqual(limits['entries'], 64)
        self.assertEqual(limits['cache_accounting_bytes'], 335544320)
        self.assertEqual(limits['scratch_bytes'], 8 * 2**20)
        self.assertIsNone(limits['owner_physical_ceiling_bytes'])
        self.assertNotIn('owner_peak_bytes', limits)
        self.assertNotIn('resident_bytes', limits)
        self.assertEqual(self.data['units']['legacy_scratch_mb'], 'binary MiB; confirmed by bound startup plan')
        self.assertNotIn('device_budget_bytes', limits)

    def test_reference_quality_and_failed_attempt_statuses_are_explicit(self):
        self.assertEqual(self.data['measured_run'], 'storage-demand-tp1-middle-20261013-02')
        self.assertEqual(self.data['attempts'][0]['status'], 'failed')
        self.assertEqual(self.data['attempts'][0]['denominator']['dispatched'], 0)
        self.assertEqual(self.data['attempts'][1]['status'], 'complete')
        self.assertEqual(self.data['quality']['reference_comparison_requests'], 3)
        self.assertEqual(self.data['quality']['reference_quality_passed_requests'], 3)
        self.assertEqual(self.data['quality']['http_only_success_requests'], 1)
        self.assertIsNone(self.data['quality']['numeric_acceptance_pass'])
        self.assertTrue(all(row['reference_quality_passed'] is True for row in self.data['requests'][1:]))
        self.assertIsNone(self.data['requests'][0]['reference_quality_passed'])
        self.assertTrue(all(row['observed_generated_token_ids'] is None for row in self.data['requests']))

    def test_timeline_is_anchored_and_complete(self):
        timeline = self.data['timeline']
        self.assertEqual(len(timeline), 30)
        self.assertEqual([r['ordinal'] for r in timeline], list(range(1, 31)))
        self.assertEqual([r['source_log_line'] for r in timeline], list(range(13, 43)))
        self.assertEqual([r['request'] for r in timeline].count('hello2'), 3)
        self.assertEqual([r['request'] for r in timeline].count('canary-qa'), 9)
        self.assertEqual([r['request'] for r in timeline].count('canary-arithmetic'), 9)
        self.assertEqual([r['request'] for r in timeline].count('canary-chinese'), 9)
        self.assertTrue(all(r['local_request_id'].endswith('/' + r['request']) for r in timeline))
        self.assertEqual(self.data['debug_logging'], 'info,ferrule_performance=debug')

    def test_gauge_fall_and_missing_sample_never_become_sums_or_zero(self):
        rows = [{'fields': {'value': 10}}, {'fields': {'value': 7}}]
        self.assertEqual(P.gauge_observations(rows, 'value'), {'last': 7, 'maximum': 10})
        rows[0]['fields'].clear()
        self.assertEqual(P.gauge_observations(rows, 'value'), {'last': 7, 'maximum': None})
        rows[1]['fields']['value'] = False
        self.assertEqual(P.gauge_observations(rows, 'value'), {'last': None, 'maximum': None})
        self.assertEqual(P.gauge_observations([], 'value'), {'last': None, 'maximum': None})

    def test_unknown_invalid_and_boolean_quantities_are_not_zero(self):
        for value in (None, False, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(P.S.SchemaError):
                P.units(value, P.MIB)
        with self.assertRaises(P.S.SchemaError):
            P.units(1000000, 1000000)
        self.assertEqual(P.units(0, P.MIB), 0)  # A recorded zero is distinct from unknown.

    def test_reconciliation_fails_closed_and_does_not_assume_empty_start(self):
        rows = [{'line': 1, 'fields': {'expert_uploads': 2, 'expert_evictions': 1,
                                      'expert_resident_entries': 4}},
                {'line': 2, 'fields': {'expert_uploads': 5, 'expert_evictions': 5,
                                      'expert_resident_entries': 4}}]
        self.assertEqual(P.reconcile(rows)['implied_initial_entries'], 3)
        self.assertIsNone(P.reconcile(rows)['initial_entries_observed'])
        changed = copy.deepcopy(rows)
        changed[1]['fields']['expert_resident_entries'] = 5
        with self.assertRaisesRegex(P.S.SchemaError, 'entry reconciliation failed'):
            P.reconcile(changed)
        for value in (None, False, -1):
            changed = copy.deepcopy(rows)
            changed[1]['fields']['expert_uploads'] = value
            with self.assertRaisesRegex(P.S.SchemaError, 'unknown entry'):
                P.reconcile(changed)
        with self.assertRaises(P.S.SchemaError):
            P.reconcile([])

    def test_unknown_physical_and_quality_quantities_are_excluded(self):
        self.assertTrue(all(value is None for value in self.data['unobserved'].values()))
        failed = self.data['attempts'][0]
        self.assertIsNone(failed['startup_seconds'])
        self.assertIsNone(failed['logical_read_bytes'])
        self.assertEqual(failed['denominator']['not_run'], 4)
        hello = self.data['requests'][0]
        for key in ('reference_quality_passed', 'reference_text_match',
                    'reference_output_usage_match', 'exact_token_pass', 'numeric_acceptance_pass'):
            self.assertIsNone(hello[key])
        self.assertFalse(hello['reference_comparison_performed'])
        self.assertEqual(self.data['quality']['numeric_acceptance_status'], 'not_established')
        self.assertFalse(self.data['quality']['full_current_math_pass'])
        self.assertTrue(all(r['numeric_acceptance_pass'] is None for r in self.data['requests']))
        self.assertEqual(self.data['independent_successful_runs'], 1)

    def test_startup_and_http_timing_equations_are_exact(self):
        prefix = self.summary['attempts'][1]['source_directory'] + '/'
        case = P.json.loads(self.sources[prefix + 'case.json'])
        self.assertEqual(self.data['startup_seconds'], case['ready_monotonic'] - case['launch_monotonic'])
        for plotted, observed in zip(self.data['requests'], case['requests']):
            actual = observed['actual']
            self.assertEqual(plotted['ttft_seconds'], actual['content_event_times'][0] - actual['sent_at'])
            self.assertEqual(plotted['e2e_seconds'], actual['ended_at'] - actual['sent_at'])
            self.assertLessEqual(plotted['ttft_seconds'], plotted['e2e_seconds'])
        self.assertAlmostEqual(sum(r['e2e_seconds'] for r in self.data['requests']), 51.537303840741515)
        # Startup, HTTP windows and supervisor wall time have distinct denominators.
        self.assertNotEqual(sum(r['e2e_seconds'] for r in self.data['requests']),
                            self.summary['attempts'][1]['timing']['summed_http_fixed_work_windows_seconds'])

    def test_plot_derivation_rejects_drifted_input_and_false_quality_claim(self):
        sources = dict(self.sources)
        leaf = self.summary['attempts'][1]['source_directory'] + '/hello2/observed.json'
        sources[leaf] += b'\n'
        with self.assertRaisesRegex(P.S.SchemaError, 'input hash mismatch'):
            P.build_data(self.summary, sources)
        summary = copy.deepcopy(self.summary)
        summary['attempts'][1]['requests'][0]['reference_quality_passed'] = True
        with self.assertRaisesRegex(P.S.SchemaError, 'derived quality differs'):
            P.build_data(summary, self.sources)
        summary = copy.deepcopy(self.summary)
        summary['attempts'][1]['requests'][0]['observed_generated_token_ids'] = [123]
        with self.assertRaisesRegex(P.S.SchemaError, 'do not expose generated IDs'):
            P.build_data(summary, self.sources)

    def test_plot_data_is_deterministic_and_stale_check_does_not_write(self):
        self.assertEqual(P.S.encode(self.data), P.S.encode(P.build_data(self.summary, self.sources)))
        with tempfile.TemporaryDirectory(dir=ANALYSIS / 'tests') as name:
            root = Path(name)
            expected = {'plot.json': b'checked', 'missing.pdf': b'pdf bytes'}
            self.assertEqual(P.check_outputs(root, expected), list(expected))
            (root / 'plot.json').write_bytes(b'stale')
            self.assertEqual(P.check_outputs(root, expected), list(expected))
            self.assertEqual((root / 'plot.json').read_bytes(), b'stale')
            (root / 'plot.json').write_bytes(expected['plot.json'])
            self.assertEqual(P.check_outputs(root, expected), ['missing.pdf'])

    def test_captions_carry_measurement_limits_and_include_exact_paths(self):
        self.assertEqual(set(self.data['captions']),
                         {'logical-transfer', 'cache-reconciliation', 'timing', 'invocation-timeline'})
        self.assertIn('not physical storage I/O', self.data['captions']['logical-transfer'])
        self.assertIn('cache charged-byte high-water', self.data['captions']['cache-reconciliation'])
        self.assertIn('cache accounting ceiling', self.data['captions']['cache-reconciliation'])
        self.assertIn('not owner physical memory', self.data['captions']['cache-reconciliation'])
        self.assertIn('n=1', self.data['captions']['timing'])
        self.assertIn('not elapsed time', self.data['captions']['invocation-timeline'])
        guide = P.plot_guide(self.data).decode()
        self.assertIn('reference null', guide)
        self.assertIn('numeric acceptance: unknown', guide)
        for name in self.data['captions']:
            self.assertIn('figures/storage-demand-' + name + '.pdf}', guide)
            self.assertNotIn('confidence interval)', guide)

    def test_provenance_binds_all_inputs_and_plot_generator(self):
        provenance = self.data['provenance']
        self.assertEqual(len(provenance['inputs']), 41)
        self.assertEqual(len(provenance['generator_sha256']), 64)
        self.assertEqual(len(provenance['summarizer']['generator_sha256']), 64)
        self.assertEqual(provenance['generator_path'], P.GENERATOR)
        self.assertEqual(provenance['summary_sha256'], P.S.digest(P.S.encode(self.summary)))


if __name__ == '__main__':
    unittest.main()
