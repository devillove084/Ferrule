"""CPU fixtures exercise the real full-export bidirectional correlation helper."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_trace_audit_v2 as audit


class BidirectionalAudit(unittest.TestCase):
    def setUp(self):
        self.pid = 10
        self.gpid = (1 << 48) + (self.pid << 24)
        self.windows = {'measured': dict(start_ns=100, end_ns=200)}
        self.activities = [dict(kind=kind, globalPid=self.gpid, correlationId=i, start=110+i*10, end=112+i*10)
                           for i,kind in enumerate(('kernel','memcpy','memset'),1)]
        self.apis = [dict(name=name, globalTid=self.gpid+1, correlationId=i, start=105+i*10, end=108+i*10, returnValue=0)
                     for i,name in enumerate(('cudaLaunchKernel_v7000','cuMemcpyHtoDAsync_v2','cuMemsetD8Async'),1)]

    def run_audit(self):
        return audit.audit_correlations(self.activities, iter(self.apis), self.windows, self.pid)['measured']

    def test_closed_export_is_never_complete_trace(self):
        result = self.run_audit()
        self.assertEqual(result['status'], 'observed_correlation_closed')
        self.assertFalse(result['full_trace_accepted'])
        self.assertFalse(result['paired_api_activity_loss_excluded'])
        for kind in ('kernel','memcpy','memset'):
            self.assertEqual(result['counts'][kind+'_activity_to_api_unique'], 1)
            self.assertEqual(result['counts'][kind+'_api_to_activity_unique'], 1)

    def test_regression_two_successful_warmup_driver_launches_have_no_activity(self):
        self.windows = {'measured': dict(start_ns=0, end_ns=900000000)}
        for corr,start in ((687294,816884800),(687352,816930550)):
            self.apis.append(dict(name='cuLaunchKernel',globalTid=self.gpid+2,
                                 correlationId=corr,start=start,end=start+2000,returnValue=0))
        result = self.run_audit()
        reverse = [r for r in result['issues'] if r['direction']=='api_to_activity']
        self.assertEqual([(r['correlation_id'],r['start_ns']) for r in reverse],
                         [(687294,816884800),(687352,816930550)])
        self.assertTrue(all(r['reason']=='missing_activity_in_full_export' for r in reverse))
        self.assertFalse(result['observed_correlation_closed'])

    def test_missing_copy_and_memset_apis_are_detected(self):
        self.apis = self.apis[:1]
        result = self.run_audit()
        self.assertEqual({r['kind'] for r in result['issues']}, {'memcpy','memset'})
        self.assertTrue(all(r['reason']=='missing_api_in_full_export' for r in result['issues']))

    def test_reverse_checks_memcpy_and_memset_not_only_launches(self):
        self.activities = self.activities[:1]
        result = self.run_audit()
        self.assertEqual({r['kind'] for r in result['issues']}, {'memcpy','memset'})
        self.assertTrue(all(r['direction']=='api_to_activity' for r in result['issues']))

    def test_same_correlation_in_another_pid_cannot_close(self):
        self.apis[0]['globalTid'] += 1 << 24
        result = self.run_audit()
        self.assertEqual(result['issues'][0]['reason'], 'missing_api_in_full_export')

    def test_duplicate_api_or_activity_is_not_unique(self):
        self.apis.append(dict(self.apis[0], globalTid=self.gpid+2))
        self.assertIn('nonunique_api_correlation', {r['reason'] for r in self.run_audit()['issues']})
        self.apis.pop()
        self.activities.append(dict(self.activities[0]))
        self.assertIn('nonunique_activity_correlation', {r['reason'] for r in self.run_audit()['issues']})

    def test_failed_submission_does_not_count_as_missing_successful_launch(self):
        self.apis[0]['returnValue'] = 1
        result = self.run_audit()
        self.assertIn('api_not_successful', {r['reason'] for r in result['issues']})
        self.assertEqual(result['counts'].get('kernel_successful_api',0), 0)

    def test_wrong_or_unsupported_api_kind_cannot_close(self):
        self.apis[0]['name'] = 'cuEventRecord'
        self.assertIn('unsupported_or_wrong_api_kind', {r['reason'] for r in self.run_audit()['issues']})

    def test_startup_counterpart_is_found_but_rejected_not_called_missing(self):
        self.apis[0].update(start=10, end=20)
        result = self.run_audit()
        self.assertIn('api_outside_request_window', {r['reason'] for r in result['issues']})
        self.assertNotIn('missing_api_in_full_export', {r['reason'] for r in result['issues']})

    def test_reverse_counterpart_in_another_window_is_rejected(self):
        self.activities[0].update(start=210,end=220)
        self.assertIn('activity_outside_request_window', {r['reason'] for r in self.run_audit()['issues']})

    def test_cross_boundary_row_is_not_silently_filtered(self):
        self.activities[0].update(start=95,end=120)
        self.assertIn('row_crosses_request_boundary', {r['reason'] for r in self.run_audit()['issues']})

    def test_overlap_same_thread_only_and_touching_end_is_allowed(self):
        row = dict(name='cuEventQuery',globalTid=self.gpid+1,correlationId=4,start=117,end=119,returnValue=600)
        self.apis.append(row)
        self.assertEqual(self.run_audit()['same_thread_runtime_overlaps'],1)
        row['globalTid'] = self.gpid+2
        self.assertEqual(self.run_audit()['same_thread_runtime_overlaps'],0)
        row.update(globalTid=self.gpid+1,start=118)
        self.assertEqual(self.run_audit()['same_thread_runtime_overlaps'],0)

    def test_empty_export_cannot_qualify(self):
        self.activities=[];self.apis=[]
        self.assertFalse(self.run_audit()['observed_correlation_closed'])


if __name__ == '__main__':
    unittest.main()
