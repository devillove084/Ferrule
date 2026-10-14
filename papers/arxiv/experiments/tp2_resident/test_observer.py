"""CPU regressions for the shared observer and one-server orchestration."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment_current as current
import observer


class Snapshot(unittest.TestCase):
    def test_proc_status_rss_anon_never_substitutes_missing_rollup_pss(self):
        status = 'VmSize: 20 kB\nVmRSS: 8 kB\nRssAnon: 5 kB\nRssFile: 3 kB\nThreads: 2\nCpus_allowed_list: 0-3\nMems_allowed_list: 0\nvoluntary_ctxt_switches: 9\nnonvoluntary_ctxt_switches: 4\n'
        row = observer._parse_proc_status(status)
        self.assertEqual((row['RssAnon'],row['RssFile']),(5120,3072))
        self.assertEqual(row['voluntary_ctxt_switches'],9)
        self.assertEqual(row['Mems_allowed_list'],'0')
        self.assertIsNone(row['RssShmem'])
        memory = observer.parse_memory('Rss: 8 kB\nPss: 7 kB\nAnonymous: 5 kB\nPrivate_Dirty: 5 kB\n',status)
        self.assertIsNone(memory['Pss_Anon'])
        self.assertIsNone(memory['Pss_File'])
        self.assertEqual(memory['Anonymous'],5120)

    def test_proc_timeline_self_snapshot_has_no_sampler_or_gpu_query(self):
        import os
        with mock.patch.object(observer.subprocess,'run') as external:
            snapshot = observer.process_timeline_snapshot(os.getpid(),'CPU_self_schema_only')
        external.assert_not_called()
        self.assertEqual(snapshot['schema'],observer.PROC_TIMELINE_SCHEMA)
        self.assertTrue(snapshot['read_only'])
        self.assertIsNone(snapshot['heap_live_bytes'])
        self.assertIsNone(snapshot['pss_anon_bytes'])
        self.assertGreaterEqual(snapshot['sample_end_monotonic'],snapshot['sample_start_monotonic'])
        main = next(row for row in snapshot['threads'] if row['is_main'])
        self.assertEqual(main['starttime_ticks'],snapshot['root_identity']['birth_ticks'])
        self.assertIsInstance(main['minor_faults'],int)
        self.assertIsInstance(main['major_faults'],int)
        self.assertGreater(main['Threads'],0)

    def setUp(self):
        self.root=dict(pid=10,birth_ticks=100,ppid=1,state='S')
        self.leaf=dict(pid=11,birth_ticks=101,ppid=10,state='S')
        self.ids={10:self.root,11:self.leaf}

    def memory(self,pid):
        return dict(pid=pid,Pss=4096,Rss=8192,Anonymous=2048,Pss_Anon=1024,
                    Pss_File=3072,Private_Dirty=512)

    def test_memory_components_are_additive_optional_fields(self):
        smaps='Rss: 8 kB\nPss: 4 kB\nAnonymous: 2 kB\nPss_Anon: 1 kB\nPss_File: 3 kB\nPrivate_Dirty: 1 kB\n'
        status='VmLck: 0 kB\nVmPin: 0 kB\n'
        row=observer.parse_memory(smaps,status)
        self.assertEqual(row['Anonymous'],2048)
        self.assertEqual(row['Pss_Anon'],1024)
        self.assertEqual(row['Pss_File'],3072)
        self.assertEqual(row['Private_Dirty'],1024)

    def test_missing_component_is_null_not_zero(self):
        row=observer.parse_memory('Rss: 8 kB\nPss: 4 kB\n','')
        self.assertIsNone(row['Anonymous'])
        self.assertIsNone(row['Pss_File'])

    def test_departed_leaf_missing_not_zero_and_root_survives(self):
        def memory(pid):
            if pid==11: raise ProcessLookupError('leaf exited')
            return self.memory(pid)
        s=observer.host_snapshot(self.root,self.ids.__getitem__,memory,pids=[10,11])
        self.assertEqual(s['status'],'incomplete')
        self.assertIsNone(s['pss_bytes'])
        self.assertEqual(s['partial_tree_pss_lower_bound_bytes'],4096)
        self.assertIn(11,s['omitted_branches'])
        self.assertEqual([r['pid'] for r in s['processes']],[10])
        self.assertEqual(s['component_coverage']['pss_anon_bytes']['sampled_processes'], 1)

    def test_next_complete_snapshot_recovers_without_restart(self):
        s=observer.host_snapshot(self.root,self.ids.__getitem__,self.memory,pids=[10])
        self.assertEqual(s['status'],'complete')
        self.assertEqual(s['pss_bytes'],4096)
        self.assertEqual(s['pss_anon_bytes'],1024)

    def test_phase_memory_distinguishes_startup_and_after_first_request(self):
        samples=[]
        for start,end in ((1,2),(6,7),(12,13)):
            host=dict(status='complete',pss_bytes=10,rss_bytes=20,pss_anon_bytes=4,
                      pss_file_bytes=6,anonymous_bytes=8,private_dirty_bytes=2)
            samples.append(dict(monotonic=start,sample_end_monotonic=end,host=host,gpu=None))
        phases=observer.phase_memory(samples,[],ready=5,first_sent=10,first_terminal=11)
        self.assertEqual(phases['phases']['startup']['complete_host_samples'],1)
        self.assertEqual(phases['phases']['after_first_request']['complete_host_samples'],1)
        self.assertFalse(phases['steady_state_claim'])

    def test_missing_root_memory_is_incomplete_not_custody_loss(self):
        def memory(pid): raise ProcessLookupError('smaps temporarily gone')
        s=observer.host_snapshot(self.root,self.ids.__getitem__,memory,pids=[10])
        self.assertEqual(s['status'],'incomplete')
        self.assertIsNone(s['pss_bytes'])

    def test_root_reuse_is_fatal(self):
        with self.assertRaises(observer.CustodyLost):
            observer.host_snapshot(self.root,lambda pid:dict(self.root,birth_ticks=999),self.memory,pids=[10])

    def test_dead_intermediate_omits_whole_branch(self):
        ids=dict(self.ids,**{})
        ids[12]=dict(pid=12,birth_ticks=102,ppid=11,state='S')
        def memory(pid):
            if pid==11: raise FileNotFoundError('dead branch')
            return self.memory(pid)
        s=observer.host_snapshot(self.root,ids.__getitem__,memory,pids=[10,11,12])
        self.assertEqual(set(s['omitted_branches']),{11,12})
        self.assertEqual([r['pid'] for r in s['processes']],[10])

    def test_enumeration_foreign_disappearance_does_not_lose_root(self):
        def identity(pid):
            if pid==99: raise ProcessLookupError('unrelated proc exit')
            return self.ids[pid]
        s=observer.host_snapshot(self.root,identity,self.memory,pids=[10,99])
        self.assertEqual(s['status'],'complete')
        self.assertEqual(s['unclassified_proc_enumeration_disappearances'],[99])


    def test_optional_components_do_not_invalidate_old_memory_rows(self):
        def old(pid):return dict(pid=pid,Pss=4096,Rss=8192)
        snapshot=observer.host_snapshot(self.root,self.ids.__getitem__,old,pids=[10])
        self.assertEqual(snapshot['status'],'complete')
        for name in observer.MEMORY_COMPONENTS.values():self.assertIsNone(snapshot[name])

    def test_mixed_component_coverage_nulls_only_missing_component(self):
        def memory(pid):
            row=self.memory(pid)
            if pid==11:row.pop('Pss_File')
            return row
        snapshot=observer.host_snapshot(self.root,self.ids.__getitem__,memory,pids=[10,11])
        self.assertEqual(snapshot['status'],'complete')
        self.assertEqual(snapshot['pss_anon_bytes'],2048)
        self.assertIsNone(snapshot['pss_file_bytes'])
        self.assertEqual(snapshot['component_coverage']['pss_file_bytes']['missing_pids'],[11])

    def test_no_startup_or_steady_fabrication_for_straddling_missing_samples(self):
        host=dict(status='complete',pss_bytes=5,rss_bytes=8)
        rows=[dict(monotonic=4,sample_end_monotonic=6,host=host,gpu=None),
              dict(monotonic=10,sample_end_monotonic=12,host=host,gpu=None),
              dict(monotonic=15,host=host,gpu=None)]
        phases=observer.phase_memory(rows,[],5,9,11)
        self.assertEqual(phases['unassigned_or_boundary_samples'],3)
        self.assertIsNone(phases['phases']['after_first_request']['tree_pss_peak_bytes'])
        self.assertIsNone(phases['phases']['startup']['tree_pss_anon_peak_bytes'])
        with self.assertRaises(ValueError):observer.phase_memory(rows,[],5,9,8)

    def test_status_cannot_override_rollup_fields(self):
        row=observer.parse_memory('Rss: 8 kB\nPss: 4 kB\nAnonymous: 2 kB\n',
                                  'Anonymous: 99 kB\nRss: 999 kB\nVmLck: 1 kB\n')
        self.assertEqual(row['Rss'],8192);self.assertEqual(row['Anonymous'],2048)
        self.assertEqual(row['VmLck'],1024)



class ProcTimelineHooks(unittest.TestCase):
    def test_all_five_anchors_are_finite_and_gpu_free(self):
        import os
        with tempfile.TemporaryDirectory(dir=current.HERE) as directory:
            root = observer.proc_identity(os.getpid())
            with mock.patch.object(observer.subprocess,'run') as external:
                timeline = observer.ProcTimeline(directory,root)
                for phase in observer.PROC_TIMELINE_PHASES:
                    timeline.capture(phase,{'kind':'CPU-only synthetic hook anchor'})
                with self.assertRaises(ValueError):timeline.capture('postmeasured')
            external.assert_not_called()
            receipt = current.ct.load(Path(directory)/'proc-timeline.json')
            self.assertEqual(len(receipt['records']),5)
            self.assertFalse(receipt['policy']['measured_window_scans'])
            self.assertTrue(all(row['read_cost_seconds'] >= 0 for row in receipt['records']))

    def test_measured_body_contains_no_proc_scan_and_result_is_unchanged(self):
        calls = []; timeline = mock.Mock()
        timeline.capture.side_effect = lambda phase,anchor: calls.append(phase)
        result = {'started_at':10,'observed_end_at':11,'measurements':{'denominator_seconds':1}}
        def measure(*args,**kwargs):
            calls.append('original_measure_enter')
            self.assertEqual(timeline.capture.call_count,1)
            calls.append('original_measure_exit')
            return result
        wrapped = observer.measure_with_proc_timeline(measure,timeline)
        self.assertIs(wrapped([],None,[],0),result)
        self.assertEqual(calls,['afterwarm_before_measured','original_measure_enter','original_measure_exit','postmeasured'])
        self.assertEqual(result['measurements']['denominator_seconds'],1)

    def test_departed_shutdown_is_unavailable_not_zero_or_heap_proof(self):
        with tempfile.TemporaryDirectory(dir=current.HERE) as directory:
            timeline = observer.ProcTimeline(directory,{'pid':10,'birth_ticks':100})
            with mock.patch.object(observer,'process_timeline_snapshot',side_effect=ProcessLookupError('root exited')):
                timeline.capture('shutdown_after_exit',{'kind':'shutdown returned'})
            row = timeline.records[0]
            self.assertEqual(row['availability'],'unavailable')
            self.assertIsNone(row['data'])
            self.assertIn('ProcessLookupError',row['error'])

    def test_private_lifecycle_hooks_preserve_shared_measure_and_require_opt_in(self):
        import paired
        directory = paired.HERE/'runs/route-batch-vllm-crossblocks-20261012-01'
        original_load = paired.ct.load; original_measure = paired.baseline.http.measure
        def load(path):
            value = original_load(path)
            if Path(path) == directory/'prepared.json':
                import copy
                value = copy.deepcopy(value); value['admission']['proc_timeline'] = observer.PROC_TIMELINE_POLICY
            return value
        with mock.patch.object(paired.ct,'load',side_effect=load):
            module = paired.lifecycle(directory,'ferrule','CPU hook wiring only','paired')
        self.assertIs(paired.baseline.http.measure,original_measure)
        self.assertEqual(module.observer.Monitor.__name__,'TimelineMonitor')
        self.assertIsNot(module.observer.Monitor,observer.Monitor)
        self.assertEqual(module.shared.request_block.__name__,'timeline_requests')
        self.assertEqual(module.http.shutdown_owned_group.__name__,'timeline_shutdown')
        ordinary = paired.lifecycle(directory,'ferrule','CPU default hooks off','paired')
        self.assertIs(ordinary.observer.Monitor,observer.Monitor)


class Queries(unittest.TestCase):
    def test_single_diagnostic_retry_has_15s_total_budget(self):
        uuids=[*current.UUIDS,*['GPU-'+str(i).zfill(36) for i in range(2,8)]]
        output='\n'.join(f'{i}, {u}, 24576, 0, 0' for i,u in enumerate(uuids))
        run=mock.Mock(side_effect=[subprocess.TimeoutExpired('gpu',3),
                                  mock.Mock(stdout=output),mock.Mock(stdout='')])
        events=[]
        s=observer.all_gpu_sample(current.UUIDS[0],current.NOTE,event=events.append,run=run,clock=lambda:0)
        self.assertEqual(run.call_count,3)
        self.assertEqual([c.kwargs['timeout'] for c in run.call_args_list],[3,15,15])
        self.assertTrue(s['diagnostic_retry'])
        self.assertEqual(len(events),1)
        self.assertIsNone(events[0]['gpu_bytes'])

    def test_exhausted_query_records_both_missing_attempts(self):
        events=[]
        run=mock.Mock(side_effect=subprocess.TimeoutExpired('gpu',3))
        with self.assertRaises(subprocess.TimeoutExpired):
            observer.all_gpu_sample(current.UUIDS[0],current.NOTE,event=events.append,run=run,clock=lambda:0)
        self.assertEqual(run.call_count,2)
        self.assertEqual(len(events),2)
        self.assertTrue(all(e['gpu_bytes'] is None for e in events))


class Lifecycle(unittest.TestCase):
    def test_canary_failure_blocks_qa_in_same_session(self):
        with tempfile.TemporaryDirectory(dir=current.HERE) as name:
            root=Path(name); case=root/'session'; case.mkdir()
            current.ct.write(root/'canaries.json',[])
            p=dict(config=current.shared.config(),source_pins={},model_identity={})
            gates=[dict(status='pass'),dict(status='fail')]
            with mock.patch.object(current.shared,'request_block',side_effect=gates) as request:
                monitor=mock.Mock(); monitor.failed.is_set.return_value=False
                result=current.requests(case,p,monitor)
            self.assertEqual(result['status'],'canary_failed')
            self.assertEqual(request.call_count,2)
            self.assertFalse((case/'qa8-V1').exists())
            self.assertEqual(current.PROTOCOL['server_starts'],1)


if __name__=='__main__': unittest.main()
