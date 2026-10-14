"""CPU fixtures: GPU queries/locks/server processes are always replaced by mocks."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as p
import instrumentation_live as live
import instrumentation_capture as target
from test_instrumentation_scope import UUIDS, sample


class CollectorHandoff(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=p.HERE)
        self.addCleanup(self.temp.cleanup)
        self.pair = Path(self.temp.name)
        self.case = self.pair/'runs/P1/session'
        self.case.mkdir(parents=True)
        p.ct.write(self.pair/'prepared.json', {'fixture': True})
        self.inventory = sorted([*p.shared.UUIDS, *UUIDS[2:]])
        self.grant = dict(scope='all_eight_instrumentation_probe',granted=True,note='fixture',
            gpu_uuids=self.inventory,model_gpu_uuids=list(p.shared.UUIDS),run_id=self.pair.name,
            prepared_sha256=p.ct.sha(self.pair/'prepared.json'))
        p.ct.write(self.pair/'probe-grant.json',self.grant)

    def gpu(self, at=1, used=0):
        ordered = [*p.shared.UUIDS, *UUIDS[2:]]
        return dict(monotonic=at,apps=[],devices=[sample(u,i,used) for i,u in enumerate(ordered)])

    def test_explicit_run_bound_eight_scope_before_any_lock(self):
        for key,value in [('run_id','old'),('prepared_sha256','0'*64),('model_gpu_uuids',list(reversed(p.shared.UUIDS))),
                          ('scope','two_gpu_model_owner'),('granted',False)]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                live.probe_grant(dict(self.grant,**{key:value}),self.pair)
        p.ct.write(self.case.parent/'prepared.json',{})
        m = p.lifecycle(self.pair,'ferrule','fixture','profiling')
        (self.pair/'probe-grant.json').unlink()
        with mock.patch.object(p.ct,'exclusive_owner_lock') as acquire, mock.patch.object(live.observer,'idle') as query:
            with self.assertRaises(FileNotFoundError):
                with m.shared.owner_locks(p.shared.UUIDS): pass
            acquire.assert_not_called();query.assert_not_called()

    def test_atomic_readonly_publication_never_overwrites(self):
        path=self.case/'gate.json'
        digest=live.readonly(path,{'first':True})
        self.assertEqual(live.sealed(path,digest),{'first':True})
        with self.assertRaises(FileExistsError):live.readonly(path,{'second':True})
        with self.assertRaisesRegex(ValueError,'changed'):live.sealed(path,'0'*64)
        path.chmod(0o600)
        with self.assertRaisesRegex(ValueError,'read-only'):live.sealed(path,digest)

    def test_eighth_lock_failure_releases_first_seven(self):
        from contextlib import contextmanager
        events=[]
        @contextmanager
        def acquire(uuid):
            events.append(('enter',uuid))
            if uuid==self.inventory[-1]:raise BlockingIOError('foreign owner')
            try:yield uuid
            finally:events.append(('exit',uuid))
        m=p.lifecycle(self.pair,'ferrule','fixture','profiling')
        with mock.patch.object(p.ct,'exclusive_owner_lock',side_effect=acquire):
            with self.assertRaises(BlockingIOError):
                with m.shared.owner_locks(p.shared.UUIDS):pass
        self.assertEqual([u for kind,u in events if kind=='exit'],list(reversed(self.inventory[:-1])))
        r=p.ct.load(self.case.parent/'probe-lock-release.json')
        self.assertFalse(r['original_b0_verified']);self.assertEqual(len(r['probe_gpu_uuids']),8)

    def test_original_supervisor_loops_and_wrapper_reused(self):
        m=p.lifecycle(self.pair,'ferrule','fixture','profiling')
        for name in ('run','worker','requests'):
            self.assertEqual(getattr(m,name).__code__.co_code,getattr(p.baseline,name).__code__.co_code)
        self.assertIs(m.base.supervisor,p.base.supervisor)
        self.assertIs(m.observer.Monitor,live.CollectorMonitor)
        self.assertIs(live.CollectorMonitor.collect,live.observer.Monitor.collect)
        wrapper=live.load_wrapper()
        self.assertEqual(Path(wrapper.__file__),p.ct.ROOT/'scripts/_profile_capture.py')

    def test_final_idle_rejects_nonzero_on_any_card_and_retains_sample(self):
        m=p.lifecycle(self.pair,'ferrule','fixture','profiling')
        row=self.gpu();row['devices'][7]['gpu_bytes']=1<<20
        result=dict(status='verified',observations=[row],baseline={d['uuid']:d for d in row['devices']})
        with mock.patch.object(live.observer,'idle',return_value=result):
            actual=m.observer.idle({},p.shared.UUIDS,'fixture')
        self.assertEqual(actual['status'],'blocked');self.assertEqual(actual['observations'],[row])

    def test_descendant_rejects_foreign_or_reused_target(self):
        root=dict(pid=10,birth_ticks=1,ppid=9,state='S')
        child=dict(pid=11,birth_ticks=2,ppid=10,state='S')
        with mock.patch.object(live.observer,'verify_root'),mock.patch.object(live.observer,'proc_identity',return_value=root):
            live.descendant(child,root)
            with self.assertRaisesRegex(ValueError,'outside'):
                live.descendant(dict(child,ppid=1),root)
        with mock.patch.object(live.observer,'verify_root',side_effect=ValueError('PID reused')):
            with self.assertRaisesRegex(ValueError,'reused'):live.descendant(child,root)

    def test_gate_replay_rejected_before_live_auth_or_launch(self):
        manifest=dict(nonce='fresh',authenticated={'launch_id':'fresh'})
        with mock.patch.object(live,'live_auth') as auth, mock.patch.object(live,'load_wrapper') as wrapper:
            with self.assertRaisesRegex(ValueError,'replayed'):
                target.check_gate({'schema':live.SCHEMA},manifest,{},'0'*64)
            auth.assert_not_called();wrapper.assert_not_called()

    def test_cancel_before_gate_does_not_launch(self):
        (self.case/'profile-stop').touch()
        with mock.patch.object(live,'live_auth'),mock.patch.object(live,'load_wrapper') as wrapper:
            with self.assertRaisesRegex(ValueError,'Cancelled'):
                target.wait_gate(self.case,'model-release.json',{},float('inf'))
            wrapper.assert_not_called()

    def test_raw_query_failure_is_missing_not_zero(self):
        with mock.patch.object(live.observer,'all_gpu_sample',side_effect=TimeoutError('missing GPU query')):
            with self.assertRaises(TimeoutError):live.sample(self.case,'fixture','B1')
        rows=[json.loads(line) for line in (self.case/'collector-samples.jsonl').read_text().splitlines()]
        self.assertIsNone(rows[-1]['gpu']);self.assertEqual(rows[-1]['kind'],'sample_failed')

    def monitor_fixture(self, samples):
        # Instantiate without /proc; every live operation below is replaced.
        m=object.__new__(live.CollectorMonitor)
        m.directory=self.case;m.root=dict(pid=10,birth_ticks=100,ppid=9,state='S')
        m.started=1;m.authenticated={'launch_id':'fixture'};m.note='fixture';m.uuids=tuple(p.shared.UUIDS)
        m.config={'limits':{'case_seconds':60},'resources':{'gpu_target_bytes':22<<30,'host_target_bytes':128<<30}}
        m.thread=SimpleNamespace(ident=None)
        manifest=dict(grant=self.grant,directory=str(self.case),nonce='fresh')
        p.ct.write(self.case/'collector-manifest.json',manifest)
        p.ct.write(self.case/'prelaunch.json',dict(observations=[self.gpu(0)]))
        handshake=dict(identity={'pid':11,'birth_ticks':101,'ppid':10,'state':'S'},nonce='fresh',
            manifest_sha256=p.ct.sha(self.case/'collector-manifest.json'))
        p.ct.write(self.case/'collector-target-ready.json',handshake)
        return m

    def test_b1_freezes_before_atomic_model_gate_and_model_tolerance_stays_fixed(self):
        rows=[self.gpu(t,4<<20) for t in (2,4,6)]
        m=self.monitor_fixture(rows)
        with mock.patch.object(live,'live_auth'),mock.patch.object(live,'descendant'), \
                mock.patch.object(live,'sample',side_effect=rows),mock.patch.object(live.time,'sleep'), \
                mock.patch.object(live.time,'monotonic',return_value=6), \
                mock.patch.object(live.observer.Monitor,'start') as start:
            self.assertFalse((self.case/'model-release.json').exists())
            m.start()
        start.assert_called_once()
        gate=p.ct.load(self.case/'model-release.json');self.assertEqual(gate['model_gpu_uuids'],list(p.shared.UUIDS))
        self.assertEqual(m.baseline[p.shared.UUIDS[0]]['gpu_bytes'],4<<20)
        row=self.gpu(7,4<<20);row['devices'][2]['gpu_bytes']+=2<<20
        with mock.patch.object(live,'live_auth'),mock.patch.object(live.observer,'verify_root'):
            self.assertTrue(m.gpu_delta_check(row)['violations'])

    def test_unstable_b1_failure_never_starts_model_or_monitor(self):
        rows=[self.gpu(t,used<<20) for t,used in ((2,4),(4,5),(6,4))]
        m=self.monitor_fixture(rows)
        with mock.patch.object(live,'live_auth'),mock.patch.object(live,'descendant'), \
                mock.patch.object(live,'sample',side_effect=rows),mock.patch.object(live.time,'sleep'), \
                mock.patch.object(live.time,'monotonic',return_value=6), \
                mock.patch.object(live.observer.Monitor,'start') as start:
            with self.assertRaisesRegex(ValueError,'stable'):m.start()
        start.assert_not_called();self.assertFalse((self.case/'model-release.json').exists())
        self.assertEqual(m.finish()['status'],'blocked')
        self.assertTrue((self.case/'collector-handoff-failure.json').exists())

    def test_missing_probe_grant_cli_rejected_before_dispatch(self):
        with mock.patch.object(p,'strict_gate'),mock.patch.object(p,'pair_path',return_value=self.pair), \
                mock.patch.object(p,'validate_pair',return_value={'purpose':'profiling'}), \
                mock.patch.object(p,'lifecycle') as launch:
            with self.assertRaisesRegex(ValueError,'probe-grant'):p.run('fixture',True,'fixture','profiling')
        launch.assert_not_called()

    def test_live_auth_stale_worker_fails_before_lock_validation(self):
        manifest={'authenticated':{'worker':{'pid':3}}}
        with mock.patch.object(live.guard,'identity',return_value={'pid':4}),mock.patch.object(live,'validate_locks') as locks:
            with self.assertRaisesRegex(ValueError,'identity changed'):live.live_auth(manifest)
        locks.assert_not_called()

    def test_model_exit_and_b1_are_observed_before_collector_release(self):
        server=mock.Mock();server.returncode=0
        manifest=dict(grant=self.grant,directory=str(self.case))
        p.ct.write(self.case/'collector-manifest.json',manifest)
        target_id={'identity':{'pid':11,'birth_ticks':101,'ppid':10}}
        root={'pid':10,'birth_ticks':100,'ppid':9}
        row=self.gpu(10,4<<20)
        anchor={d['uuid']:d for d in row['devices']}
        digest=live.readonly(self.case/'collector-baseline.json',dict(instrumentation=anchor))
        p.ct.write(self.case/'model-release.json',dict(target=target_id,collector=root,baseline_sha256=digest))
        p.ct.write(self.case/'profile-server.json',dict(returncode=0))
        p.ct.write(self.case/'collector-model-ended.json',dict(target=target_id,
            manifest_sha256=p.ct.sha(self.case/'collector-manifest.json'),
            wrapper_sha256=p.ct.sha(self.case/'profile-server.json')))
        def sample(*args):
            self.assertTrue((self.case/'profile-stop').exists())
            self.assertFalse((self.case/'collector-release.json').exists())
            return row
        def wait(timeout):
            self.assertTrue((self.case/'collector-after-model.json').exists())
            self.assertTrue((self.case/'collector-release.json').exists())
            return 0
        server.wait.side_effect=wait
        with mock.patch.object(live,'live_auth'),mock.patch.object(live,'descendant'), \
                mock.patch.object(live,'sample',side_effect=sample),mock.patch.object(live.time,'sleep'):
            r=live.shutdown(server,self.case,110)
        self.assertFalse(r['timed_out']);server.terminate.assert_not_called()

    def test_cleanup_failure_keeps_failure_and_drains_owned_root_without_success(self):
        server=mock.Mock();server.returncode=1
        p.ct.write(self.case/'collector-manifest.json',{})
        p.ct.write(self.case/'model-release.json',{})
        with mock.patch.object(live,'live_auth',side_effect=ValueError('lost custody')):
            with self.assertRaisesRegex(ValueError,'lost custody'):live.shutdown(server,self.case,110)
        self.assertTrue((self.case/'collector-cleanup-failure.json').exists())
        self.assertTrue((self.case/'collector-release.json').exists())
        self.assertFalse((self.case/'collector-after-model.json').exists())
        server.wait.assert_called_once();server.terminate.assert_not_called()

    def test_target_reuses_wrapper_without_starting_before_gate(self):
        command=['/fixture/immutable','serve']
        manifest=dict(schema=live.SCHEMA,directory=str(self.case),command=command,
            command_sha256=p.ct.digest(command),nonce='fresh',created_monotonic=1,
            binary_sha256='0'*64,authenticated={'worker':{'pid':9},'launch_id':'fixture'})
        digest=live.readonly(self.case/'collector-manifest.json',manifest)
        wrapper=mock.Mock()
        def run_target(case,args,lifetime,shutdown):
            self.assertEqual(args,command);self.assertEqual((lifetime,shutdown),(1450,45))
            self.assertTrue((case/'collector-target-ready.json').exists())
            p.ct.write(case/'profile-server.json',dict(returncode=0))
            return 0
        wrapper.run_target.side_effect=run_target
        with mock.patch.object(live,'live_auth'),mock.patch.object(live,'descendant'), \
                mock.patch.object(live.observer,'proc_identity',return_value={'pid':11}), \
                mock.patch.object(target,'wait_gate',side_effect=[{}, {'manifest_sha256':digest}]) as gate, \
                mock.patch.object(target,'check_gate') as check, \
                mock.patch.object(live,'load_wrapper',return_value=wrapper), \
                mock.patch.object(live.ct,'validate_staged_binary'), \
                mock.patch.dict(target.os.environ,{key:','.join(p.shared.UUIDS) for key in ('CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES')}):
            self.assertEqual(target.run(self.case,digest,command),0)
        wrapper.run_target.assert_called_once();check.assert_called_once()
        self.assertEqual([c.args[1] for c in gate.call_args_list],['model-release.json','collector-release.json'])


if __name__=='__main__':unittest.main()
