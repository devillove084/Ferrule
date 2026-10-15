"""CPU-only regressions from the rejected official 2025.3.1 capture."""
from __future__ import annotations

import copy
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from profile_overlap_v1 import analyze as a, protocol as p, run
from profile_overlap_v1.test_profile_overlap import fixture, GPID, WORKER


class CollectorContractTests(unittest.TestCase):
    def setUp(self):
        self.db, self.context = fixture()
        self.db.row_factory = sqlite3.Row
        self.addCleanup(self.db.close)

    def test_three_lifecycle_rows_are_not_three_missing_apis(self):
        self.db.executemany('INSERT INTO OSRT_API VALUES(?,?,?,?,3,0,NULL,0)',
                           [(0, 0, 31, WORKER), (900, 900, 32, WORKER), (901, 901, 32, WORKER)])
        report = a.audit_sqlite(self.db, self.context)
        coverage = report['osrt_counter_coverage']
        self.assertEqual(report['status'], 'sql_evidence_consistent')
        self.assertEqual((coverage['total_rows'], coverage['api_rows'], coverage['lifecycle_marker_rows']), (5, 2, 3))
        self.assertTrue(coverage['counters'][0]['matched'])
        self.assertEqual(report['worker_pread_calls'], 2)
        self.assertIsNone(report['dropped_event_count'])

    def test_unknown_class_never_subtracted_to_force_count(self):
        self.db.execute('UPDATE OSRT_API SET eventClass=999 WHERE start=250')
        report = a.audit_sqlite(self.db, self.context)
        self.assertEqual(report['status'], 'rejected')
        self.assertIn('unknown OSRT event class', report['issues'])

    def test_missing_counter_is_not_zero_loss(self):
        self.db.execute("DELETE FROM DIAGNOSTIC_EVENT WHERE text LIKE 'Number of OS%'")
        report = a.audit_sqlite(self.db, self.context)
        self.assertEqual(report['status'], 'rejected')
        self.assertIsNone(report['dropped_event_count'])

    def test_missing_api_and_range_even_if_nvtx_balanced(self):
        self.db.execute('DELETE FROM NVTX_EVENTS WHERE rangeId=1')
        self.db.execute('DELETE FROM OSRT_API')
        report = a.audit_sqlite(self.db, self.context)
        self.assertEqual(report['status'], 'rejected')
        self.assertIn('NVTX reads differ from current final read ledger', report['issues'])

    def test_lost_events_reject_even_when_all_rows_and_counters_match(self):
        self.db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(1,1,1,1,?,?)', ('Dropped events: count unknown', GPID))
        report = a.audit_sqlite(self.db, self.context)
        self.assertEqual(report['status'], 'rejected')
        self.assertEqual(report['loss_diagnostic_count'], 1)
        self.assertIsNone(report['dropped_event_count'])

    def test_partial_does_not_need_request_or_read_ranges(self):
        self.db.execute('DELETE FROM NVTX_EVENTS WHERE rangeId IS NOT NULL')
        self.db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(1,2,3,2,?,?)', ('new warning', GPID))
        report = a.raw_summary(self.db, self.context)
        self.assertTrue(report['observed_launch_kernel_bijection'])
        self.assertEqual(report['status'], 'diagnostic_partial')
        self.assertEqual(len(report['warnings']), 1)
        self.assertFalse(report['full_trace_accepted'])
        self.assertIsNone(report['pread_kernel_intersection_ns'])
        self.assertIsNone(report['attributed_kernel_count'])

    def test_partial_orphan_does_not_claim_bijection(self):
        self.db.execute('DELETE FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE correlationId=1')
        report = a.raw_summary(self.db, self.context)
        self.assertFalse(report['observed_launch_kernel_bijection'])
        self.assertEqual(report['correlation_counts']['kernel_without_unique_math_api'], 1)

    def test_gate_precedes_grant_lock_or_gpu_work(self):
        blocked = dict(collector_contract=dict(status='blocked_before_gpu', launch_allowed=False, blockers=['warning']))
        with (patch.object(p, 'prepared', return_value=blocked),
              patch.object(run.live, 'sealed', side_effect=AssertionError('grant read')),
              patch.object(run.tc.scope, 'probe_locks', side_effect=AssertionError('GPU lock')),
              patch.object(run.live.observer, 'idle', side_effect=AssertionError('GPU query'))):
            with self.assertRaisesRegex(ValueError, 'collector contract unresolved'):
                run.run(p.HERE / p.GENERATION, None, None, True)
        with self.assertRaises(ValueError):
            p.require_launch({})


class HistoricalRawTests(unittest.TestCase):
    def test_actual_osrt_classes_and_kernel_correlation(self):
        if not (p.PRIOR / 'runs/blocking/session/nsys.sqlite').exists():
            self.skipTest('private rejected capture not installed')
        with patch('subprocess.run', side_effect=AssertionError('no tool commands in preflight')):
            report = p.collector_preflight()
        self.assertTrue(report['launch_allowed'])
        self.assertTrue(report['cpu_context_switch_compatible'])
        self.assertTrue(report['measurement_methodology_corrected'])
        self.assertFalse(report['direct_adapter_verified'])
        self.assertTrue(report['cpu_environment_check_reused'])
        case = p.PRIOR / 'runs/blocking/session'
        worker = p.ct.load(case / 'worker-capture.json')
        with sqlite3.connect((case / 'nsys.sqlite').resolve().as_uri() + '?mode=ro', uri=True) as db:
            raw = a.raw_summary(db, dict(model_pid=worker['server_identity']['pid'],
                model_birth_ticks=worker['server_identity']['birth_ticks']))
        self.assertEqual(len(raw['warnings']), 4)
        self.assertEqual(raw['counts']['CUPTI_ACTIVITY_KIND_KERNEL'], 4458)
        self.assertTrue(raw['observed_launch_kernel_bijection'])
        counters = raw['osrt_counter_coverage']
        self.assertEqual((counters['api_rows'], counters['lifecycle_marker_rows']), (82330, 6))
        self.assertTrue(all(r['matched'] for r in counters['counters']))
        self.assertFalse(raw['full_trace_accepted'])

    def test_raw_map_policy_stays_fail_closed(self):
        policy_path = p.HERE / 'artifacts-go-20261015-02/runtime-policy.json'
        if not policy_path.exists():
            self.skipTest('new private CPU runtime policy not prepared')
        folder = p.PRIOR / 'runs/blocking/session/ready-library-evidence'
        policy = p.ct.load(policy_path)
        snapshot = p.ct.load(folder / 'snapshot.json')
        report = p.maps.validate_snapshot(snapshot, policy)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(len(report['libraries']), 19)
        self.assertIsNone(report['live_mapped_memory_sha256'])
        self.assertTrue(report['overlay_device_aliases'])
        original = p.ct.load(p.ct.load(p.PRIOR / 'candidate-binding.json')['candidate']['runtime_pin_manifest'])
        self.assertEqual(len(original['per_role']['cli']), 12)
        self.assertTrue(set(original['per_role']['cli']) <= set(policy['required']))
        missing = copy.deepcopy(policy)
        keyboard = str(p.TOOL / 'target-linux-x64/libLinuxKeyboardInterceptorProxy.so')
        del missing['pins'][keyboard]
        del missing['optional'][keyboard]
        self.assertEqual(p.maps.validate_snapshot(snapshot, missing)['status'], 'failed')
        reused = copy.deepcopy(snapshot)
        reused['before']['birth_ticks'] += 1
        self.assertEqual(p.maps.validate_snapshot(reused, policy)['status'], 'failed')
        raw = (folder / 'maps.raw').read_text()
        unknown = '\n1000-2000 r-xp 00000000 00:00 1 /unknown/libsurprise.so\n'
        self.assertEqual(p.maps.assess(raw + unknown, policy)['status'], 'failed')
