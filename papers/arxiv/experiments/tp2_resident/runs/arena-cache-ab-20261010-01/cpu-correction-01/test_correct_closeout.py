"""CPU tests for correction accounting; no serving or GPU imports."""
from copy import deepcopy
import unittest
import correct_closeout as c


class SampleCounts(unittest.TestCase):
    def test_terminal_without_gpu_is_not_a_missing_sample(self):
        result = c.gpu_sample_counts([{'kind':'sample','gpu':{'devices':[]}}, {'kind':'monitor_terminal'}])
        self.assertEqual((result['sample_rows'],result['raw_gpu_samples'],result['missing_gpu_samples']), (1,1,0))
        self.assertEqual(result['excluded_non_sample_rows'], {'monitor_terminal':1})

    def test_actual_null_or_absent_gpu_sample_is_missing(self):
        result = c.gpu_sample_counts([{'kind':'sample','gpu':None}, {'kind':'sample'}, {'kind':'monitor_terminal'}])
        self.assertEqual((result['sample_rows'],result['raw_gpu_samples'],result['missing_gpu_samples']), (2,0,2))

    def test_other_kinds_are_excluded_even_when_gpu_is_present(self):
        result = c.gpu_sample_counts([{'kind':'monitor_terminal','gpu':{}}, {'kind':'diagnostic'}, {}])
        self.assertEqual((result['sample_rows'],result['raw_gpu_samples'],result['missing_gpu_samples']), (0,0,0))

    def test_empty_stream_has_no_samples_or_missing(self):
        result = c.gpu_sample_counts(iter(()))
        self.assertEqual(result['sample_rows'], 0)
        self.assertEqual(result['missing_gpu_samples'], 0)


class RunWindow(unittest.TestCase):
    def report(self):
        return {'runs':[{'measurements':dict(window_kind='fixed-work',denominator_seconds=2.0,
            dispatch_start_at=10.0,terminal_observed_at=12.0),
            'summary':dict(denominator_seconds=2.0,wall_seconds=2.0,successful_output_tokens=8,
                           successful_output_tokens_per_second=4.0),
            'requests':[dict(client_observed_at=11.968,ended_at=11.967)]}]}

    def test_tail_is_disclosed_without_replacing_rate_or_window(self):
        report=self.report();before=deepcopy(report)
        result=c.run_window_annotation(report)
        self.assertAlmostEqual(result['tail_after_last_client_observed_ms'],32)
        self.assertEqual(result['recorded_run_window_seconds'],2)
        self.assertEqual(report,before)

    def test_formula_or_timing_drift_rejected(self):
        for key,value in (('terminal_observed_at',13),('denominator_seconds',1)):
            report=self.report();report['runs'][0]['measurements'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.run_window_annotation(report)
        report=self.report();report['runs'][0]['summary']['successful_output_tokens_per_second']=5
        with self.assertRaises(ValueError):c.run_window_annotation(report)


class HistoricalCorrection(unittest.TestCase):
    def test_actual_history_counts_rates_and_hashes_are_preserved(self):
        path=c.HERE.parent/'owner-closeout.json'
        old=c.load(path);new=c.build_correction(path)
        expected={'S1':161,'S2':159,'OFF':113,'ON':162}
        for row in new['lifecycles']:
            self.assertEqual(row['raw_gpu_samples'],expected[row['slot']])
            self.assertEqual(row['missing_gpu_samples'],0)
        for row in new['cpu_correction']['samples']:
            self.assertEqual(row['sample_rows'],expected[row['slot']])
            self.assertEqual(row['excluded_non_sample_rows'],{'monitor_terminal':1})
        self.assertTrue(all(r['missing_gpu_samples']==1 for r in old['lifecycles']))
        for key in ('metrics','descriptive_ON_OFF_output_rate_ratio','contract','binary_sha256','limitations'):
            self.assertEqual(new[key],old[key])
        self.assertEqual(c.sha(path),c.ORIGINAL_SHA)
        self.assertEqual(c.sha(c.HERE.parent/'RESULTS.md'),c.REPORT_SHA)


if __name__ == '__main__':
    unittest.main()
