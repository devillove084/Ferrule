"""Synthetic CPU mutations of same-cut controls, not performance/GPU evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import performance as p
import qualify as q
import test_quality_v1 as fixture
from profile_overlap_v1.test_profile_overlap import fixture as sql_fixture, PID, GPID


def plan():
    return dict(schema='ferrule.retained-upload.performance-plan.v1', commit=p.COMMIT,
        binary_sha256=p.BINARY_SHA, source_archive_sha256=p.SOURCE_SHA, jobs=p.fixed_jobs(),
        executable=p.fixed_jobs()[0]['executable'], source_digest='a' * 64,
        retries=0, extra_warmups=0, calibration=0, stop_on_first_failure=True,
        atol=.00025, rtol=.012, host_options=dict(max_host_bytes=64 << 20,
            max_metadata_bytes=8 << 20, worker_stack_bytes=2 << 20),
        nsys_version=p.VERSION, nsys='/official/nsight-systems-cli/2025.3.1/target-linux-x64/nsys',
        uuid_inventory=['GPU-b980f22b-58a3-fc88-21c1-ae86053d8535'] + [f'GPU-00000000-0000-0000-0000-{i:012d}' for i in range(1, 8)],
        measurement='whole_test_libtest_footer_only_not_token_TTFT_or_request_latency',
        capacity_and_options=dict(frozen_fixture_only=True), fresh_reference_sha256='b' * 64,
        full40_reference_sha256='c' * 64)


def profile_plan():
    value=plan();value.update(schema='ferrule.retained-upload.profile-only-plan.v1',
        jobs=p.profile_jobs(), measurement='official_nsight_process_wide_counts_and_host_durations_not_unprofiled_latency',
        completed_performance_source=dict(directory=str(q.PARENT),
            plan_sha256='875957578ca9479e553d5017b04bff4c8b3dd44bff270c86211c6b129109b991',
            final_sha256='9e080fa6149832d2d2b3c6962048a2ecc9dc4364ac5cacdd520a887d393b52e2'),
        diagnostic_primaries=[], pins={})
    for filename,key in (('performance-plan.json','plan_sha256'),('FINAL.json','final_sha256')):
        value['pins'][str(q.PARENT/filename)]=value['completed_performance_source'][key]
    for row in p.fixed_jobs()[:4]:
        binding=dict(job=row)
        for field,name in (('report','quality.json'),('gate_result','gate-result.json')):
            path=str(q.PARENT/'gates'/row['id']/name)
            binding[field]=path;binding[field+'_sha256']='e'*64;value['pins'][path]='e'*64
        value['diagnostic_primaries'].append(binding)
    return value


def report(row, manifest):
    mode, retained = row['schedule'], row['transfer_mode'] == 'retained'
    opts = fixture.options(mode)
    value = fixture.report(mode)
    calls = []
    before = fixture.observation(opts, configured=True)
    for i in range(64):
        after = fixture.observation(opts, reads=(i + 1) * 6, reuses=i + 1)
        if retained:
            after['stats']['peak_host_charged_bytes'] = 36_701_896
        call = fixture.interval(opts, copy.deepcopy(before), copy.deepcopy(after))
        call['label'] = f'prediction.{i:02d}'
        for endpoint, uploads in (('before', i * 6), ('after', (i + 1) * 6)):
            call[endpoint]['expert_cache'] = dict(state='observed', value=dict(uploads=uploads,
                evictions=0, hits=0, pending_upload_bytes=0, unknown_quarantine_bytes=0,
                quarantined=False, counters_overflowed=False, resident_entries=1,
                resident_bytes=1024, scratch_bytes=8 << 20, charged_bytes=(8 << 20) + 1024,
                peak_bytes=(8 << 20) + 1024))
        calls.append(call)
        before = after
    value['expert_read_quality'].update(calls=calls, sampled_operations=64, positive_worker_read_calls=64)
    diagnostic = row['phase'] == 'primary'
    value.update(complete=True, passed=True, failures=[], atol=.00025, rtol=.012,
        state_bytes_after_shutdown=0, quarantine=False, device_budget_bytes=7 << 30,
        peak_reserved_bytes=4 << 30, metrics=[dict(failures=0, nonfinite=0) for _ in range(5427 if diagnostic else 27)],
        logits_compared=9187840, oracle_manifest_sha256=manifest['fresh_reference_sha256'],
        cold_owner_restart=False, observer_mode='diagnostic_routes_and_states' if diagnostic else 'ordinary_no_diagnostic_observer',
        ordinary_no_observer_claim=not diagnostic,
        cases=[dict(free_greedy_equal=True, top1_failures=[], free_greedy_ids=list(range(9)),
            oracle_generated_ids=list(range(9)), common_history_generated_ids=list(range(9)),
            next_token_checks=[dict(equal=True)] * 9, free_greedy_next_token_checks=[dict(equal=True)] * 9) for _ in range(3)],
        prediction_fingerprints=dict(calls={f'prediction.{i:02d}': dict(f32_le_sha256='d'*64, next_token_id=i, rows=1, values=248320) for i in range(64)},
            failures=[], paired_exact_logits_and_greedy_equal=not diagnostic),
        prediction_denominators=dict(streams={name:dict(prediction_calls=n, greedy_checks=n,
            input_history_matched_reference_calls=n, logit_values_compared=1, failures=[])
            for name,n in (('common_history',27),('free_feedback',27),('reset_replay',9),('fork_replay',1))}),
        diagnostic_primary_report_sha256='e'*64,
        test_config=dict(name='storage-middle64-' + ('retained-' if retained else '') + mode,
            numeric_precision='Bf16ModelMixedF32Accumulate', capacity_and_options=manifest['capacity_and_options'],
            expert_read_pipeline=dict(schedule=mode.capitalize(), max_host_bytes=64 << 20,
                max_metadata_bytes=8 << 20, worker_stack_bytes=2 << 20,
                metadata_and_stack_included_in_host_cap=True, device_budget_separate=True)),
        effective_config=dict(assertions_passed=True))
    if retained:
        for container in ('test_config','effective_config'):
            value[container]['expert_transfer_policy'] = copy.deepcopy(q.retained_evidence.POLICY)
        value['effective_config']['expert_transfer_selection'] = 'explicit existing-owner capability before first prediction'
    return value


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.plan = plan()
        for name in ('subprocess.Popen','subprocess.run','os.kill','os.execve'):
            patch = mock.patch(name, side_effect=AssertionError('CPU-only control mutation: no execution/signal'))
            patch.start(); self.addCleanup(patch.stop)
        for obj, name in ((q.idle,'observe'), (q.ct,'exclusive_owner_lock'), (q.supervisor,'run_process')):
            patch = mock.patch.object(obj,name,side_effect=AssertionError('No GPU/locks/custody during contract tests'))
            patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(q, 'PLAN', dict(self.plan, gates=self.plan['jobs']))
        patch.start(); self.addCleanup(patch.stop)

    def validate(self, value, row, prior=None):
        with mock.patch.object(q,'primary',return_value=(Path('/synthetic/primary.json'),'e'*64,prior or value)):
            q.validate_composition(value,row)

    def test_fixed24_and_all_four_pairs_per_schedule(self):
        p.validate_plan(self.plan)
        self.assertEqual([r['phase'] for r in self.plan['jobs']].count('primary'),4)
        for mode in ('blocking','lookahead'):
            rows = [r for r in self.plan['jobs'] if r['phase']=='latency' and r['schedule']==mode]
            self.assertEqual([r['transfer_mode'] for r in rows], [v for pair in p.ORDER for v in pair])
        for index in range(24):
            bad=copy.deepcopy(self.plan); bad['jobs'][index]['selector'] += '_wrong'
            with self.assertRaises(ValueError):p.validate_plan(bad)

    def test_cut_tool_order_policy_deadline_and_claim_mutations(self):
        for field, bad in (('commit','f'*40),('binary_sha256','f'*64),('source_archive_sha256','f'*64),
                ('nsys_version','2025.4'),('retries',1),('extra_warmups',1),('calibration',1),
                ('atol',.000251),('rtol',.01201),('measurement','TTFT')):
            value=copy.deepcopy(self.plan);value[field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_plan(value)
        for field,bad in (('phase','profile'),('transfer_mode','retained'),('seconds',0),('id','foreign-job')):
            value=copy.deepcopy(self.plan['jobs'][0]);value[field]=bad
            with self.assertRaises(ValueError):p.job(self.plan,value)

    def test_valid_both_policies_and_schedules_all_job_shapes(self):
        for row in self.plan['jobs']:
            self.validate(report(row,self.plan),row)

    def test_policy_crossbinding_missing_retained_and_sync_injection(self):
        for row in self.plan['jobs'][:4]:
            value=report(row,self.plan)
            if row['transfer_mode']=='retained':
                del value['test_config']['expert_transfer_policy']
            else:
                value['effective_config']['expert_transfer_policy']=q.retained_evidence.POLICY
            with self.assertRaises(ValueError):self.validate(value,row)

    def test_primary_new_cut_policy_schedule_phase_and_report_sha_required(self):
        row=self.plan['jobs'][4]
        primary=self.plan['jobs'][0]
        data=report(primary,self.plan)
        good=dict(status='passed',report_sha256='e'*64,commit=p.COMMIT,binary_sha256=p.BINARY_SHA,
                          source_digest=self.plan['source_digest'],job=primary)
        p.validate_primary(good,data,row,self.plan,'e'*64)
        for field,bad in (('status','failed'),('report_sha256','f'*64),('commit','f'*40),('binary_sha256','f'*64),('source_digest','f'*64),('job',self.plan['jobs'][1]),('job',self.plan['jobs'][3]),('job',row)):
            changed=copy.deepcopy(good);changed[field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_primary(changed,data,row,self.plan,'e'*64)

    def test_v3_unknown_pending_uploads_revision_and_budget_mutations(self):
        for row in self.plan['jobs'][:4]:
            value=report(row,self.plan)
            for field,bad in (('worker_unknown',True),('worker_join_pending',True)):
                changed=copy.deepcopy(value);changed['expert_read_quality']['calls'][1]['after']['expert_read_pipeline']['value'][field]=bad
                with self.assertRaises(ValueError):self.validate(changed,row)
            changed=copy.deepcopy(value);changed['expert_read_quality']['interval_validation_revision']=2
            with self.assertRaises(ValueError):self.validate(changed,row)
            for field in q.retained_evidence.UPLOAD_FIELDS:
                changed=copy.deepcopy(value);changed['expert_read_quality']['calls'][1]['after']['expert_read_pipeline']['value']['current_charge'][field]=1
                with self.assertRaises(ValueError):self.validate(changed,row)

    def test_numeric_quota_stream_route_and_fingerprint_mutations(self):
        row=self.plan['jobs'][4]
        data=report(row,self.plan)
        for field,bad in (('atol',.001),('rtol',.02),('logits_compared',1),('device_budget_bytes',8<<30)):
            changed=copy.deepcopy(data);changed[field]=bad
            with self.assertRaises(ValueError):self.validate(changed,row)
        changed=copy.deepcopy(data);del changed['prediction_denominators']['streams']['fork_replay']
        with self.assertRaises(ValueError):self.validate(changed,row)
        changed=copy.deepcopy(data);changed['cases'][0]['free_greedy_ids']=[99]*9
        with self.assertRaises(ValueError):self.validate(changed,row)
        changed=copy.deepcopy(data);changed['prediction_fingerprints']['calls']['prediction.00']['next_token_id']=99
        with self.assertRaises(ValueError):self.validate(changed,row,data)

    def test_exact_ordered_work_read_upload_and_check_arrays_not_just_totals(self):
        a=report(self.plan['jobs'][0],self.plan);b=report(self.plan['jobs'][1],self.plan)
        self.assertTrue(p.compare_work(a,b)['equal'])
        for mutate in (
            lambda v:v['expert_read_quality']['calls'].reverse(),
            lambda v:v['expert_read_quality']['calls'][0]['delta'].__setitem__('reader_interval_calls',999),
            lambda v:v['expert_read_quality']['calls'][0]['after']['expert_cache']['value'].__setitem__('uploads',999),
            lambda v:v['prediction_denominators']['streams']['common_history'].__setitem__('greedy_checks',1)):
            changed=copy.deepcopy(b);mutate(changed)
            with self.assertRaises(ValueError):p.compare_work(a,changed)

    def test_cache_health_rejects_unchanged_work_all_jobs_both_endpoints(self):
        def update(item, **values):
            item['value'].update(values)
        mutations = [
            ('unobserved', lambda item: item.update(state='unobserved')),
            ('unknown', lambda item: item.update(state='unknown')),
            ('over_cap', lambda item: update(item, resident_bytes=320 << 20,
                charged_bytes=(320 << 20) + (8 << 20), peak_bytes=(320 << 20) + (8 << 20))),
            ('pending', lambda item: update(item, pending_upload_bytes=1,
                charged_bytes=(8 << 20) + 1025, peak_bytes=(8 << 20) + 1025)),
            ('quarantine_overflow', lambda item: update(item, quarantined=True, counters_overflowed=True)),
            ('quarantine', lambda item: update(item, quarantined=True)),
            ('overflow', lambda item: update(item, counters_overflowed=True)),
            ('unknown_bytes', lambda item: update(item, unknown_quarantine_bytes=1)),
            ('entries', lambda item: update(item, resident_entries=65)),
            ('scratch', lambda item: update(item, scratch_bytes=(8 << 20) + 1,
                charged_bytes=(8 << 20) + 1025, peak_bytes=(8 << 20) + 1025)),
            ('conservation', lambda item: update(item, charged_bytes=(8 << 20) + 1025,
                peak_bytes=(8 << 20) + 1025)),
            ('current_over_peak', lambda item: update(item, peak_bytes=(8 << 20) + 1023)),
            ('peak_over_cap', lambda item: update(item, peak_bytes=(320 << 20) + 1)),
            ('quarantine_not_bool', lambda item: update(item, quarantined=0)),
            ('overflow_not_bool', lambda item: update(item, counters_overflowed=0)),
        ]
        def unchecked_work(value):
            # Deliberately reproduce the old counter-only projection: invalid
            # evidence must not qualify merely because these arrays still match.
            return [(call['label'], dict(call['delta']),
                {key: call['after']['expert_cache']['value'][key] - call['before']['expert_cache']['value'][key]
                 for key in p.CACHE_WORK}) for call in value['expert_read_quality']['calls']]
        checked = 0
        for row in self.plan['jobs']:
            value=report(row,self.plan); baseline=unchecked_work(value)
            for endpoint in ('before','after'):
                physical=value['expert_read_quality']['calls'][1][endpoint]
                original=physical['expert_cache']
                for name,mutate in mutations:
                    bad=copy.deepcopy(original);mutate(bad);physical['expert_cache']=bad
                    try:
                        with self.subTest(job=row['id'],endpoint=endpoint,mutation=name):
                            self.assertEqual(unchecked_work(value),baseline)
                            with self.assertRaises(ValueError):self.validate(value,row)
                            with self.assertRaises(ValueError):p.work_vector(value)
                            with self.assertRaises(ValueError):p.compare_work(value,value)
                        checked += 1
                    finally:
                        physical['expert_cache']=original
        self.assertEqual(checked,24*2*len(mutations))
        print('CACHE_HEALTH_UNCHANGED_WORK_MUTATIONS_REJECTED',checked)

    def test_cache_evidence_shapes_and_all_counter_charge_types_reject(self):
        checked = 0
        for row in self.plan['jobs']:
            value=report(row,self.plan)
            for endpoint in ('before','after'):
                physical=value['expert_read_quality']['calls'][1][endpoint]
                original=physical['expert_cache']
                for malformed in (None, [], {}, dict(state='observed',value=None), dict(state='observed',value=[])):
                    physical['expert_cache']=malformed
                    with self.subTest(job=row['id'],endpoint=endpoint,shape=malformed):
                        with self.assertRaises(ValueError):self.validate(value,row)
                        with self.assertRaises(ValueError):p.work_vector(value)
                    checked += 1
                for field in (*p.CACHE_WORK, 'pending_upload_bytes','unknown_quarantine_bytes',
                        'resident_entries','resident_bytes','scratch_bytes','charged_bytes','peak_bytes'):
                    for bad in (None,False,True,float(original['value'][field]),str(original['value'][field]),[],{},-1):
                        changed=copy.deepcopy(original);changed['value'][field]=bad;physical['expert_cache']=changed
                        with self.subTest(job=row['id'],endpoint=endpoint,field=field,bad=bad):
                            with self.assertRaises(ValueError):self.validate(value,row)
                            with self.assertRaises(ValueError):p.work_vector(value)
                        checked += 1
                    changed=copy.deepcopy(original);del changed['value'][field];physical['expert_cache']=changed
                    with self.subTest(job=row['id'],endpoint=endpoint,missing=field):
                        with self.assertRaises(ValueError):self.validate(value,row)
                        with self.assertRaises(ValueError):p.work_vector(value)
                    checked += 1
                for field in ('quarantined','counters_overflowed'):
                    for bad in (True,None,0,0.0,'false',[],{}):
                        changed=copy.deepcopy(original);changed['value'][field]=bad;physical['expert_cache']=changed
                        with self.subTest(job=row['id'],endpoint=endpoint,field=field,bad=bad):
                            with self.assertRaises(ValueError):self.validate(value,row)
                            with self.assertRaises(ValueError):p.work_vector(value)
                        checked += 1
                physical['expert_cache']=original
        self.assertEqual(checked,24*2*(5+9*10+7*2))
        print('CACHE_HEALTH_SHAPE_TYPE_MUTATIONS_REJECTED',checked)

    def test_cache_upload_reset_between_calls_rejects_equal_per_call_arrays(self):
        for row in self.plan['jobs']:
            value=report(row,self.plan);expected=p.work_vector(value)
            for endpoint in ('before','after'):
                value['expert_read_quality']['calls'][1][endpoint]['expert_cache']['value']['uploads'] -= 6
            call=value['expert_read_quality']['calls'][1]
            self.assertEqual(call['after']['expert_cache']['value']['uploads'] - call['before']['expert_cache']['value']['uploads'],
                expected[1]['cache']['uploads'])
            with self.subTest(job=row['id']):
                with self.assertRaisesRegex(ValueError,'counters reset'):self.validate(value,row)
                with self.assertRaisesRegex(ValueError,'counters reset'):p.work_vector(value)

    def test_shared_cache_health_never_imposes_retained_minimum_or_positive_uploads_on_sync(self):
        for row in self.plan['jobs']:
            value=report(row,self.plan)
            if row['transfer_mode']=='synchronous':
                for call in value['expert_read_quality']['calls']:
                    self.assertLess(call['after']['expert_read_pipeline']['value']['stats']['peak_host_charged_bytes'],
                        q.retained_evidence.MINIMUM_AGGREGATE_BYTES)
                self.validate(value,row)
            for call in value['expert_read_quality']['calls']:
                for endpoint in ('before','after'):
                    call[endpoint]['expert_cache']['value']['uploads']=0
            self.assertEqual(len(p.work_vector(value)),64)
            if row['transfer_mode']=='synchronous':
                self.validate(value,row)
            else:
                with self.assertRaisesRegex(ValueError,'upload path not actually exercised'):self.validate(value,row)

    def test_retained_minimum_stays_separate_from_shared_cache_health(self):
        for row in self.plan['jobs']:
            if row['transfer_mode']!='retained':continue
            value=report(row,self.plan)
            for call in value['expert_read_quality']['calls']:
                for endpoint in ('before','after'):
                    read=call[endpoint]['expert_read_pipeline']['value']
                    if read['stats']['workers_created']:
                        read['stats']['peak_host_charged_bytes']=q.retained_evidence.MINIMUM_AGGREGATE_BYTES-1
                call['delta']['peak_host_charged_bytes']=call['after']['expert_read_pipeline']['value']['stats']['peak_host_charged_bytes']
            self.assertEqual(len(p.work_vector(value)),64)
            with self.assertRaisesRegex(ValueError,'positive retained aggregate'):self.validate(value,row)

    def test_whole_test_footer_resolution_no_fabricated_token_latency(self):
        footer='test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 256 filtered out; finished in 101.45s'
        value=p.libtest_elapsed(footer)
        self.assertEqual(value['printed_seconds'],'101.45');self.assertEqual(value['printed_resolution_seconds'],'0.01')
        for bad in ('',footer+'\n'+footer,footer.replace('1 passed','0 passed'),footer.replace('101.45','NaN'),footer.replace('101.45','0.00')):
            with self.assertRaises(ValueError):p.libtest_elapsed(bad)

    def test_profiler_only_wraps_four_declared_jobs_and_keeps_safe_flags(self):
        for row in self.plan['jobs']:
            if row['phase']!='profile':
                with self.assertRaises(ValueError):p.profile_argv(self.plan,row,Path('/new'),['cpu-wrapper'])
            else:
                argv=p.profile_argv(self.plan,row,Path('/new'),['cpu-wrapper'])
                for flag in ('--kill=none','--wait=primary','--trace-fork-before-exec=false','--duration=1080','--force-overwrite=false'):
                    self.assertIn(flag,argv)
                self.assertNotIn('--kill=sigterm',argv)

    def test_owner_pid_birth_parent_executable_selector_and_zombie_mutations(self):
        identity=dict(pid=101,birth_ticks=200,ppid=100,state='R')
        role=dict(before=identity,after=copy.deepcopy(identity),executable_sha256=p.BINARY_SHA,command=['exact-selector'])
        p.validate_role(role,identity,p.BINARY_SHA,['exact-selector'])
        for field,bad in (('pid',102),('birth_ticks',201),('ppid',99),('state','Z'),('state',None)):
            changed=copy.deepcopy(role);changed['after'][field]=bad
            with self.assertRaises(ValueError):p.validate_role(changed,identity,p.BINARY_SHA,['exact-selector'])
        for field,bad in (('executable_sha256','f'*64),('command',['foreign-selector'])):
            changed=copy.deepcopy(role);changed[field]=bad
            with self.assertRaises(ValueError):p.validate_role(changed,identity,p.BINARY_SHA,['exact-selector'])

    def test_profile_auth_rejects_job_root_birth_lock_scope_and_plan_crossbinding(self):
        row=self.plan['jobs'][20]
        auth=dict(worker=dict(pid=100,birth_ticks=10,ppid=99,pgrp=100,session=100),
            helper=dict(pid=99,birth_ticks=9,ppid=98,pgrp=99,session=99),
            coordinator=dict(pid=98,birth_ticks=8,ppid=97,pgrp=98,session=98), launch_id='cpu-launch')
        root=Path('/synthetic-stage');directory=root/'gates'/row['id']
        payload=dict(gate=row,gate_id=row['id'],qualification_plan_sha256='a'*64,
                     environment={},all_uuid_locks=[{} for _ in range(8)])
        journal=dict(containment_verified=True,method='linux-subreaper-waitid',
                     launch_id='cpu-launch',helper_identity=auth['helper'],root_identity=auth['worker'],
                     root_session_ready=True,status='tracking',echild=False,errors=[],reporting_errors=[])
        live=mock.Mock();live.sealed.side_effect=lambda *args:payload
        def load(path):
            return dict(identity=auth) if path.name=='authenticated-launch.json' else journal
        with mock.patch.object(q,'OUT',root), mock.patch.object(q,'profile_modules',return_value=(live,None)), \
                mock.patch.object(q,'require_independent_review'),mock.patch.object(q,'environment',return_value={}), \
                mock.patch.object(q.ct,'load',side_effect=load),mock.patch.object(q.ct,'sha',return_value='a'*64), \
                mock.patch.object(q.ct,'validate_owner_lock'),mock.patch.object(q.guard,'identity',side_effect=lambda pid:next(v for v in auth.values() if isinstance(v,dict) and v['pid']==pid)):
            q.PLAN['nsys_sha256']='a'*64
            self.assertEqual(q.profile_auth(directory,'sealed')[1],row)
            for key,bad in (('qualification_plan_sha256','b'*64),('gate_id','foreign'),('all_uuid_locks',[])):
                old=payload[key];payload[key]=bad
                with self.subTest(key=key),self.assertRaises(ValueError):q.profile_auth(directory,'sealed')
                payload[key]=old
            # Stable PID/birth/parent are not permission to ignore a stale
            # pre-setsid group/session snapshot. The handshake must repair it.
            for field,bad in (('pid',101),('birth_ticks',11),('ppid',98),('pgrp',99),('session',99)):
                journal['root_identity']=dict(auth['worker'],**{field:bad})
                with self.subTest(field=field),self.assertRaises(ValueError):q.profile_auth(directory,'sealed')
            journal['root_identity']=auth['worker']
            for field in ('pgrp','session'):
                saved=auth['worker'][field];auth['worker'][field]=99
                with self.subTest(matching_but_not_session_leader=field),self.assertRaises(ValueError):
                    q.profile_auth(directory,'sealed')
                auth['worker'][field]=saved
            self.assertEqual(q.profile_auth(directory,'sealed')[1],row)
            for field,bad in (('root_session_ready',False),('root_session_ready',1),
                    ('status','unknown'),('status','cleaning'),('echild',True),
                    ('errors',['unsafe cleanup']),('reporting_errors',['unpersisted identity']),
                    ('launch_id','foreign'),('helper_identity',dict(auth['helper'],birth_ticks=99))):
                old=journal[field];journal[field]=bad
                with self.subTest(field=field,bad=bad),self.assertRaises(ValueError):q.profile_auth(directory,'sealed')
                journal[field]=old


    def test_fresh_profile_only_exact4_completed_source_primaries_and_mutations(self):
        value=profile_plan();p.validate_plan(value)
        self.assertEqual(value['jobs'],p.fixed_jobs()[20:])
        self.assertTrue(all(r['phase']=='profile' for r in value['jobs']))
        mutations=(
            lambda v:v['jobs'].append(p.fixed_jobs()[0]),
            lambda v:v['jobs'].reverse(),
            lambda v:v['jobs'][0].__setitem__('seconds',1201),
            lambda v:v['jobs'][0].__setitem__('phase','latency'),
            lambda v:v['completed_performance_source'].__setitem__('final_sha256','f'*64),
            lambda v:v['completed_performance_source'].__setitem__('plan_sha256','f'*64),
            lambda v:v['diagnostic_primaries'].reverse(),
            lambda v:v['diagnostic_primaries'][0].__setitem__('report',v['diagnostic_primaries'][1]['report']),
            lambda v:v['diagnostic_primaries'][0].__setitem__('gate_result_sha256','f'*64),
            lambda v:v['pins'].clear(),
            lambda v:v.__setitem__('measurement','whole_test_libtest_footer_only_not_token_TTFT_or_request_latency'))
        for mutate in mutations:
            bad=copy.deepcopy(value);mutate(bad)
            with self.assertRaises(ValueError):p.validate_plan(bad)
        for row in value['jobs']:
            self.validate(report(row,value),row)

    def test_profile_only_primary_keeps_exact_policy_schedule_cut_and_report_binding(self):
        value=profile_plan()
        for row in value['jobs']:
            binding=next(b for b in value['diagnostic_primaries'] if p.primary_key(b['job'])==p.primary_key(row))
            source=report(binding['job'],value)
            accepted=dict(status='passed',report_sha256='e'*64,commit=p.COMMIT,
                binary_sha256=p.BINARY_SHA,source_digest=value['source_digest'],job=binding['job'])
            p.validate_primary(accepted,source,row,value,'e'*64)
            for key,bad in (('commit','f'*40),('binary_sha256','f'*64),
                    ('source_digest','f'*64),('report_sha256','f'*64),
                    ('job',value['jobs'][0]),('job',value['diagnostic_primaries'][(value['diagnostic_primaries'].index(binding)+1)%4]['job'])):
                changed=dict(accepted,**{key:bad})
                with self.subTest(job=row['id'],field=key),self.assertRaises(ValueError):
                    p.validate_primary(changed,source,row,value,'e'*64)

    def test_consumed24_directory_rejected_before_plan_load_or_any_gpu_action(self):
        before=q.OUT
        with self.assertRaisesRegex(ValueError,'fresh separately sealed'):q.configure(q.PARENT)
        self.assertEqual(q.OUT,before)

    def test_old24_review_cannot_authorize_fresh4_scope(self):
        with tempfile.TemporaryDirectory() as folder,mock.patch.object(q,'OUT',Path(folder)), \
                mock.patch.object(q,'PLAN',profile_plan()),mock.patch.object(q.ct,'sha',return_value='a'*64):
            record=dict(schema='ferrule.worker-reuse.performance-control-review.v1',status='passed',
                gpu_execution_authorized=True,reviewer='fixture',commit=p.COMMIT,
                qualification_plan_sha256='a'*64,control_sha256='a'*64,jobs=p.fixed_jobs(),review_packet_sha256='a'*64)
            path=Path(folder)/'independent-review.json';path.write_text(json.dumps(record));path.chmod(0o444)
            with self.assertRaisesRegex(ValueError,'bind this exact'):q.require_independent_review()

    def test_profiling_samples_never_count_as_unprofiled_pairs(self):
        results=[dict(job=row,status='passed',whole_test_elapsed=dict(seconds=2 if row['transfer_mode']=='retained' else 4)) for row in self.plan['jobs']]
        value=p.pair_summary(results)
        self.assertEqual(len(value['pairs']),8)
        self.assertTrue(all(v['retained_over_synchronous']==.5 for v in value['pairs']))
        self.assertFalse(value['significance_claim'])
        with self.assertRaises(ValueError):p.pair_summary(results[5:])

    def test_no_review_no_locks_gpu_queries_dispatch_or_final(self):
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(q,'OUT',Path(folder)):
            with self.assertRaisesRegex(ValueError,'Independent review/GO missing'):q.main()
            self.assertFalse((Path(folder)/'FINAL.json').exists())


class ProfileSQLiteTests(unittest.TestCase):
    def setUp(self):
        self.db,self.context=sql_fixture();self.addCleanup(self.db.close)
        self.db.execute('INSERT INTO PROCESSES VALUES(?,?,?)',(GPID-(1<<24),PID-1,'traced-python-wrapper'))
        self.db.execute('ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN copyKind INTEGER')
        self.db.execute('UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET copyKind=1')
        self.db.execute('CREATE TABLE ENUM_CUDA_MEMCPY_OPER(id INTEGER,name TEXT,label TEXT)')
        self.db.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(1,'CUDA_MEMCPY_KIND_HTOD','H2D')")
        self.db.execute("INSERT INTO StringIds VALUES(20,'cuEventQuery')")
        self.db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(430,440,0,?,20,20,600,NULL)',(GPID|PID,))

    def audit(self):
        return q.profile_audit.process_api_summary(self.db,PID,PID-1,self.context['uuid'])

    def test_observed_query_spin_not_ready_codes_h2d_bytes_and_no_complete_trace_claim(self):
        value=self.audit()
        queries=[r for r in value['apis'] if r['family']=='event_query']
        self.assertEqual(queries[0]['return_values'],{'600':1})
        self.assertEqual(value['copies']['CUDA_MEMCPY_KIND_HTOD']['bytes'],16)
        self.assertFalse(value['full_trace_accepted']);self.assertFalse(value['ttft_token_or_per_request_latency'])
        self.assertEqual(value['api_tables']['CUPTI_ACTIVITY_KIND_DRIVER'],'absent; not a zero-count assertion')

    def test_unknown_warning_loss_wrong_device_foreign_pid_and_api_error_reject(self):
        mutations=(
            "INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,2,'unknown warning',-1)",
            "INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,1,'lost events',-1)",
            'UPDATE TARGET_INFO_CUDA_DEVICE SET gpuId=1',
            'UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET globalPid=globalPid+(1<<24)',
            'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=1 WHERE nameId=20',
            'UPDATE CUPTI_ACTIVITY_KIND_RUNTIME SET returnValue=34 WHERE nameId=20',
            'INSERT INTO PROCESSES SELECT * FROM PROCESSES')
        for sql in mutations:
            self.db.execute('SAVEPOINT mutation');self.db.execute(sql)
            with self.subTest(sql=sql),self.assertRaises(ValueError):self.audit()
            self.db.execute('ROLLBACK TO mutation');self.db.execute('RELEASE mutation')

    def test_exact_model_no_nvtx_and_wrapper_no_payload_never_upgrade_full_trace(self):
        for global_pid,text in ((GPID,'No NVTX events collected. Does the process use NVTX?'),
                (GPID-(1<<24),'No CUDA events collected. Does the process use CUDA?')):
            self.db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,2,?,?)',(text,global_pid))
        self.assertFalse(self.audit()['full_trace_accepted'])
        self.db.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,2,?,?)',('No CUDA events collected. Does the process use CUDA?',GPID))
        with self.assertRaises(ValueError):self.audit()


class CuratedEvidenceTests(unittest.TestCase):
    def test_bounded_relative_source_pins_and_actual21_results_are_not_performance(self):
        folder=HERE/'engineering-20261016-v1'
        results=json.loads((folder/'completed-stages.json').read_text())
        source=json.loads((folder/'source-pins.json').read_text())
        self.assertEqual(results['actual_gpu_tests'],21)
        self.assertFalse(results['performance_accepted'])
        self.assertEqual([len(s['actual_invocations']) for s in results['stages']],[9,1,11])
        self.assertEqual([s['summary']['status'] for s in results['stages']],['passed']*3)
        for stage in source['stages']:
            self.assertEqual(len(stage['commit']),40)
            self.assertEqual(len(stage['source_archive_sha256']),64)
            for receipt in stage['raw_artifacts']:
                self.assertTrue(receipt['path'].startswith('target/validation/'))
                self.assertFalse(Path(receipt['path']).is_absolute())
                self.assertEqual(len(receipt['sha256']),64)
        for filename in ('completed-stages.json','source-pins.json'):
            self.assertLess((folder/filename).stat().st_size,200_000)


if __name__=='__main__':
    unittest.main()
