"""CPU-only custody/scope policy fixtures; never lock or query GPUs."""
import sys
from pathlib import Path
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrumentation_scope as scope


UUIDS = [f"GPU-00000000-0000-0000-0000-{i:012x}" for i in range(8)]


def sample(uuid, index, used=0, at=0):
    return dict(index=index, uuid=uuid, total_bytes=24 << 30, gpu_bytes=used,
                utilization=0)


class InstrumentationScope(unittest.TestCase):
    def grant(self):
        return dict(scope='all_eight_instrumentation_probe', granted=True,
                    note='fixture explicit all-eight probe grant', gpu_uuids=sorted(UUIDS), model_gpu_uuids=UUIDS[:2])

    def idle(self, at=0, used=0):
        return dict(monotonic=at, apps=[], devices=[sample(uuid, i, used, at)
            for i, uuid in enumerate(UUIDS)])

    def test_plan_disables_metrics_but_does_not_claim_no_enumeration(self):
        plan = scope.plan()
        self.assertFalse(plan['execute_allowed'])
        self.assertIn('--gpu-metrics-devices=none', plan['metrics_flags'])
        self.assertIn('No documented no-enumeration switch', plan['enumeration_suppression'])
        self.assertEqual(plan['probe']['model_visible_gpus'], 2)

    def test_stable_instrumentation_baseline_is_separate_from_original_zero(self):
        state = scope.BaselineScope(UUIDS, UUIDS[:2], self.grant())
        state.raw_idle(self.idle())
        identity = dict(pid=77, birth_ticks=900, launch_id='collector')
        state.collector_started(identity, 1)
        frozen = state.freeze([self.idle(2, 4 << 20), self.idle(4, 4 << 20),
                               self.idle(6, 4 << 20)], identity)
        self.assertEqual(frozen['raw'][UUIDS[0]]['gpu_bytes'], 0)
        self.assertEqual(frozen['instrumentation'][UUIDS[0]]['gpu_bytes'], 4 << 20)
        self.assertEqual(frozen['instrumentation_delta_bytes'][UUIDS[3]], 4 << 20)
        self.assertFalse(frozen['execute_allowed'])

    def test_unstable_baseline_does_not_rebaseline_or_relax_tolerance(self):
        state = scope.BaselineScope(UUIDS, UUIDS[:2], self.grant())
        state.raw_idle(self.idle())
        identity = dict(pid=77, birth_ticks=900, launch_id='collector')
        state.collector_started(identity, 1)
        with self.assertRaisesRegex(ValueError, 'stable'):
            state.freeze([self.idle(2, 2 << 20), self.idle(4, 3 << 20),
                          self.idle(6, 2 << 20)], identity)

    def test_model_visibility_stays_two_cards_and_final_returns_original_baseline(self):
        state = scope.BaselineScope(UUIDS, UUIDS[:2], self.grant())
        state.raw_idle(self.idle())
        identity = dict(pid=77, birth_ticks=900, launch_id='collector')
        state.collector_started(identity, 1)
        state.freeze([self.idle(2), self.idle(4), self.idle(6)], identity)
        with self.assertRaisesRegex(ValueError, 'must not expand'):
            state.admit_model(UUIDS, identity)
        state.admit_model(UUIDS[:2], identity)
        state.after_model(True, True)
        result = state.finish(True, self.idle(8))
        self.assertEqual(result['status'], 'release_admissible')

    def test_locks_are_sorted_and_atomic_via_existing_context(self):
        calls = []
        class Lock:
            def __enter__(self): calls.append('enter'); return object()
            def __exit__(self, *args): calls.append('exit')
        with scope.probe_locks(UUIDS, UUIDS[:2], self.grant(), lambda uuid: Lock()):
            self.assertEqual(len(calls), 8)
        self.assertEqual(calls.count('exit'), 8)


    def frozen(self):
        state=scope.BaselineScope(UUIDS,UUIDS[:2],self.grant())
        state.raw_idle(self.idle())
        identity=dict(pid=77,birth_ticks=900,launch_id='collector')
        state.collector_started(identity,1)
        state.freeze([self.idle(2,4<<20),self.idle(4,4<<20),self.idle(6,4<<20)],identity)
        return state,identity

    def test_missing_probe_grant_blocks_before_acquisition(self):
        acquire=mock.Mock(side_effect=AssertionError('no real lock'))
        for grant in ({},dict(self.grant(),scope='two_gpu_model_owner'),dict(self.grant(),granted=False)):
            with self.assertRaisesRegex(ValueError,'eight-GPU probe grant'):
                with scope.probe_locks(UUIDS,UUIDS[:2],grant,acquire):pass
        acquire.assert_not_called()

    def test_partial_lock_failure_releases_in_reverse_order(self):
        from contextlib import contextmanager
        events=[]
        @contextmanager
        def acquire(uuid):
            events.append(('acquire',uuid))
            if uuid==sorted(UUIDS)[2]:raise BlockingIOError('foreign lock holder')
            try:yield uuid
            finally:events.append(('release',uuid))
        with self.assertRaises(BlockingIOError):
            with scope.probe_locks(list(reversed(UUIDS)),UUIDS[:2],self.grant(),acquire):pass
        self.assertEqual(events,[('acquire',UUIDS[0]),('acquire',UUIDS[1]),('acquire',UUIDS[2]),
                                 ('release',UUIDS[1]),('release',UUIDS[0])])

    def test_compute_unknown_missing_or_unstable_probe_never_admitted(self):
        for kind in ('foreign_pid','missing','timeout','overhead'):
            state=scope.BaselineScope(UUIDS,UUIDS[:2],self.grant());state.raw_idle(self.idle())
            identity=dict(pid=77,birth_ticks=900,launch_id='collector');state.collector_started(identity,1)
            samples=[self.idle(2),self.idle(4),self.idle(6)]
            if kind=='foreign_pid':samples[1]['apps']=[dict(uuid=UUIDS[5],pid=99)]
            if kind=='missing':samples[1]['devices'].pop()
            if kind=='timeout':samples[-1]['monotonic']=62
            if kind=='overhead':samples=[self.idle(at,17<<20) for at in (2,4,6)]
            with self.subTest(kind=kind),self.assertRaises(ValueError):state.freeze(samples,identity)

    def test_model_delta_remains_one_mib_no_rebaseline_no_visibility_expansion(self):
        state,identity=self.frozen();state.admit_model(UUIDS[:2],identity)
        row=self.idle(8,4<<20);row['devices'][2]['gpu_bytes']+=1<<20
        self.assertFalse(state.check_model_sample(row,identity)['violations'])
        row['devices'][2]['gpu_bytes']+=1
        with self.assertRaisesRegex(ValueError,'Nonselected'):state.check_model_sample(row,identity)
        with self.assertRaisesRegex(ValueError,'reset'):state.freeze([self.idle(10),self.idle(12),self.idle(14)],identity)

    def test_root_identity_and_original_final_idle_required(self):
        state,identity=self.frozen();state.admit_model(UUIDS[:2],identity)
        with self.assertRaisesRegex(ValueError,'identity lost'):
            state.check_model_sample(self.idle(8,4<<20),dict(identity,birth_ticks=901))
        with self.assertRaisesRegex(ValueError,'drain'):state.after_model(False,True)
        state.after_model(True,True)
        with self.assertRaisesRegex(ValueError,'original baseline'):state.finish(True,self.idle(10,4<<20))
        self.assertEqual(state.finish(True,self.idle(12))['status'],'release_admissible')

    def test_selected_utilization_after_reap_is_pending_not_idle(self):
        state,identity=self.frozen()
        row=self.idle(8,4<<20);row['devices'][0]['utilization']=8;row['devices'][1]['utilization']=5
        self.assertFalse(scope.returned_to_instrumentation(row,state.anchor,UUIDS[:2]))
        self.assertEqual(row['devices'][0]['utilization'],8)
        row['devices'][0]['utilization']=row['devices'][1]['utilization']=0
        self.assertTrue(scope.returned_to_instrumentation(row,state.anchor,UUIDS[:2]))

    def test_b1_return_foreign_nonselected_and_missing_still_abort(self):
        state,identity=self.frozen()
        for case in ('foreign','nonselected_util','nonselected_mem','missing','unknown'):
            row=self.idle(8,4<<20)
            if case=='foreign':row['apps']=[dict(pid=88,uuid=UUIDS[0])]
            elif case=='nonselected_util':row['devices'][7]['utilization']=1
            elif case=='nonselected_mem':row['devices'][7]['gpu_bytes']+=1
            elif case=='missing':row['devices'].pop()
            else:row['devices'][0]['utilization']=None
            with self.subTest(case=case),self.assertRaises(ValueError):
                scope.returned_to_instrumentation(row,state.anchor,UUIDS[:2])

    def test_actual_failed_cleanup_sample_reproduces_pending_without_relaxation(self):
        import json
        directory=Path(__file__).resolve().parent/'runs/tp2-optimized-collector-profile-20261009-002/runs/P1/session'
        frozen=json.loads((directory/'collector-baseline.json').read_text())
        samples=[json.loads(line) for line in (directory/'collector-samples.jsonl').read_text().splitlines()]
        row=next(x['gpu'] for x in samples if x['phase']=='after_model_B1')
        self.assertFalse(scope.returned_to_instrumentation(row,frozen['instrumentation'],frozen['model_gpu_uuids']))


if __name__ == '__main__':
    unittest.main()
