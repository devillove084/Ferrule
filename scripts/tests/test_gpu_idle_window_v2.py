"""Dedicated CPU fixtures only; no GPU queries, locks, model or child launch."""
import copy
import sys
from pathlib import Path
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gpu_idle_window_v2 as idle

UUIDS = [f'GPU-{i:08x}-0123-0123-0123-0123456789ab' for i in range(8)]


def receipt():
    helper = dict(pid=123, birth_ticks=456)
    root = dict(pid=789, birth_ticks=1011)
    return dict(cleanup_errors=[], descendant_custody='proven_drained',
                handshake=dict(containment_verified=True, launch_id='fixture', helper_identity=helper),
                custody=dict(containment_verified=True, launch_id='fixture', helper_identity=helper,
                             status='proven_drained', echild=True, root_identity=root,
                             entries=[dict(role='root', identity=root, reaped=True)]))


def snapshot(util=0, memory=0, apps=''):
    return dict(gpu='\n'.join(f'{i}, {uuid}, {memory if i == 0 else 0} MiB, '
                             f'{util if i == 0 else 0} %' for i, uuid in enumerate(UUIDS)), apps=apps)


class IdleWindow(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(idle.subprocess, 'run', side_effect=AssertionError('No GPU query')).start()
        mock.patch.object(idle.time, 'monotonic', side_effect=lambda: self.now).start()
        mock.patch.object(idle.time, 'sleep', side_effect=self.sleep).start()
        mock.patch.object(idle.Path, 'read_text', side_effect=FileNotFoundError('CPU fixture')).start()

    def sleep(self, seconds):
        self.now += seconds

    def observe(self, rows, custody=None):
        values = iter(rows)
        def sample(deadline, target):
            target.update(next(values))
        with mock.patch.object(idle, 'sample', side_effect=sample) as query:
            result = idle.observe(UUIDS, custody=custody)
        return result, query.call_count

    def test_exact_three_strict_preflight_samples(self):
        result, count = self.observe([snapshot()] * 3)
        self.assertEqual((result['status'], count, result['accepted_sample_indices']), ('verified', 3, [0, 1, 2]))
        self.assertEqual(result['physical_owner_release'], 'unverified')

    def test_trailing_one_and_six_percent_restart_window_keep_every_sample(self):
        for util in (1, 6):
            self.now = 0
            result, count = self.observe([snapshot(util), snapshot(), snapshot(6), snapshot(), snapshot(), snapshot()], receipt())
            self.assertEqual((result['status'], count), ('verified', 6))
            self.assertEqual(result['accepted_sample_indices'], [3, 4, 5])
            self.assertEqual([s['consecutive_idle'] for s in result['samples']], [0, 1, 0, 1, 2, 3])
            self.assertIn(f'{util} %', result['samples'][0]['gpu'])
            self.assertTrue(result['authenticated_echild'] and result['recorded_owners_absent'])

    def test_memory_or_apps_stop_immediately_do_not_wait_for_later_zero(self):
        for busy, reason in [(snapshot(memory=1), 'memory_nonzero'), (snapshot(apps='GPU-foreign, 7, 1 MiB'), 'compute_apps')]:
            self.now = 0
            result, count = self.observe([busy, snapshot(), snapshot(), snapshot()], receipt())
            self.assertEqual((result['status'], count), ('blocked', 1))
            self.assertIn(reason, result['error'])
            self.assertEqual(self.now, 0)

    def test_stuck_utilization_hits_fixed_deadline_without_extension(self):
        result, count = self.observe([snapshot(6)] * 30, receipt())
        self.assertEqual((result['status'], count, result['elapsed_seconds']), ('blocked', 15, 15))
        self.assertEqual(len(result['samples']), 15)
        self.assertIn('deadline', result['error'])

    def test_query_completing_at_deadline_cannot_pass(self):
        def sample(deadline, target):
            target.update(snapshot())
            self.now += 5
        with mock.patch.object(idle, 'sample', side_effect=sample):
            result = idle.observe(UUIDS, custody=receipt())
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('deadline', result['error'])
        self.assertGreaterEqual(result['elapsed_seconds'], 15)

    def test_no_echild_unknown_or_forged_custody_never_queries(self):
        for mutation in ('echild', 'unknown', 'identity', 'launch', 'reaped', 'root', 'cleanup'):
            r = receipt()
            if mutation == 'echild': r['custody']['echild'] = False
            if mutation == 'unknown': r['custody']['status'] = 'unknown'
            if mutation == 'identity': r['custody']['helper_identity'] = {'pid': 1}
            if mutation == 'launch': r['custody']['launch_id'] = 'foreign'
            if mutation == 'reaped': r['custody']['entries'][0]['reaped'] = False
            if mutation == 'root': r['custody']['root_identity'] = None
            if mutation == 'cleanup': r['cleanup_errors'] = ['unknown']
            with self.subTest(mutation=mutation), mock.patch.object(idle, 'sample') as query:
                result = idle.observe(UUIDS, custody=r)
                self.assertEqual(result['status'], 'blocked')
                query.assert_not_called()
                self.assertEqual(result['samples'], [])

    def test_live_recorded_owner_is_not_absent_but_recycled_pid_is(self):
        fields = ['0'] * 20
        fields[19] = '1011'
        with mock.patch.object(idle.Path, 'read_text', return_value='789 (fixture) ' + ' '.join(fields)), \
                mock.patch.object(idle, 'sample') as query:
            result = idle.observe(UUIDS, custody=receipt())
            self.assertEqual(result['status'], 'blocked')
            query.assert_not_called()
        fields[19] = '9999'
        with mock.patch.object(idle.Path, 'read_text', return_value='789 (recycled) ' + ' '.join(fields)):
            self.assertTrue(idle.no_recorded_owner(receipt()))

    def test_live_recorded_helper_and_unreadable_procfs_block_before_query(self):
        fields = ['0'] * 20
        fields[19] = '456'
        for outcome in ('123 (helper) ' + ' '.join(fields), PermissionError('procfs unknown')):
            kwargs = {'side_effect': outcome} if isinstance(outcome, Exception) else {'return_value': outcome}
            with mock.patch.object(idle.Path, 'read_text', **kwargs), mock.patch.object(idle, 'sample') as query:
                result = idle.observe(UUIDS, custody=receipt())
                self.assertEqual(result['status'], 'blocked')
                query.assert_not_called()

    def test_preflight_never_waits_out_utilization(self):
        result, count = self.observe([snapshot(1), snapshot(), snapshot(), snapshot()])
        self.assertEqual((result['status'], count), ('blocked', 1))
        self.assertIn('Preflight', result['error'])

    def test_unknown_inventory_or_utilization_fails_closed(self):
        for field, bad in [('gpu', snapshot()['gpu'].replace(UUIDS[0], UUIDS[1])),
                           ('gpu', snapshot()['gpu'].replace('0 %', 'N/A', 1))]:
            row = copy.deepcopy(snapshot())
            row[field] = bad
            result, count = self.observe([row, snapshot()], receipt())
            self.assertEqual((result['status'], count), ('blocked', 1))

    def test_telemetry_error_retains_partial_sample(self):
        def sample(deadline, target):
            target['gpu'] = snapshot()['gpu']
            raise TimeoutError('apps query timed out')
        with mock.patch.object(idle, 'sample', side_effect=sample):
            result = idle.observe(UUIDS, custody=receipt())
        self.assertEqual(result['status'], 'blocked')
        self.assertEqual(result['samples'][0]['gpu'], snapshot()['gpu'])
        self.assertIn('apps query timed out', result['error'])


if __name__ == '__main__':
    unittest.main()
