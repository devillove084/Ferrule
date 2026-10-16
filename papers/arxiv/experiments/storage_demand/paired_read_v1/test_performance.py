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
        diagnostic_primaries=[], pins={}, observed_model_summary_policy=q.profile_audit.model_summary_policy())
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

    def test_partial_model_summary_policy_is_predeclared_exact_and_not_full_trace(self):
        value=profile_plan();p.validate_plan(value)
        for field,original in value["observed_model_summary_policy"].items():
            variants = (True, 0) if original is False else (0,) if original is None else ("relaxed",)
            for variant in variants:
                bad=copy.deepcopy(value);bad["observed_model_summary_policy"][field]=variant
                with self.subTest(field=field,variant=variant),self.assertRaises(ValueError):p.validate_plan(bad)
        del value["observed_model_summary_policy"]
        with self.assertRaises(ValueError):p.validate_plan(value)

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
        for consumed in (q.PARENT, q.ROOT / 'target/validation/retained-profile-57e4629-postsetsid-20261016-01'):
            with self.subTest(consumed=str(consumed)),self.assertRaisesRegex(ValueError,'fresh separately sealed'):q.configure(consumed)
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

        self.db.executescript("CREATE TABLE ENUM_DIAGNOSTIC_SOURCE_TYPE(id INTEGER,name TEXT); INSERT INTO ENUM_DIAGNOSTIC_SOURCE_TYPE VALUES(3,'Analysis'); CREATE TABLE ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE(id INTEGER,name TEXT); INSERT INTO ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE VALUES(2,'HostTimestamp'); INSERT INTO META_DATA_CAPTURE VALUES('COLLECT_THREAD_STATE_TRACE','false'); ALTER TABLE CUPTI_ACTIVITY_KIND_MEMSET ADD COLUMN bytes INTEGER; CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER); ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN srcDeviceId INTEGER; ALTER TABLE CUPTI_ACTIVITY_KIND_MEMCPY ADD COLUMN dstDeviceId INTEGER;")
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(450,460,0,?,NULL,16)",(GPID,))
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES(460,470,0,?)",(GPID,))

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


    def mutation_rejected(self, sql, params=()):
        self.db.execute("SAVEPOINT mutation")
        try:
            self.db.execute(sql, params)
            with self.subTest(sql=sql, params=params), self.assertRaises(ValueError):self.audit()
        finally:
            self.db.execute("ROLLBACK TO mutation");self.db.execute("RELEASE mutation")

    def test_eight_inventory_rows_and_other_vm_same_gpu_id_are_not_activity(self):
        for i in range(1,8):self.db.execute("INSERT INTO TARGET_INFO_CUDA_DEVICE VALUES(?,?,?)",(i,i,PID))
        self.db.execute("INSERT INTO TARGET_INFO_GPU VALUES(?,?,?)",(2<<48,0,"foreign-vm-uuid"))
        value=self.audit()
        self.assertEqual(value["device_binding"]["model_inventory_rows"],8)
        self.assertEqual(value["device_binding"]["vm_id"],1<<48)
        for sql in ("INSERT INTO TARGET_INFO_CUDA_DEVICE SELECT * FROM TARGET_INFO_CUDA_DEVICE WHERE cudaId=0",
                "INSERT INTO TARGET_INFO_GPU SELECT * FROM TARGET_INFO_GPU WHERE vmId=(1<<48) AND id=0",
                "UPDATE TARGET_INFO_GPU SET uuid='wrong' WHERE vmId=(1<<48) AND id=0",
                "DELETE FROM TARGET_INFO_GPU WHERE vmId=(1<<48) AND id=0",
                "UPDATE TARGET_INFO_CUDA_DEVICE SET cudaId=8 WHERE cudaId=0"):
            self.mutation_rejected(sql)

    def test_consistent_cuda_mapping_and_physical_uuid_swap_cannot_relabel_device0_activity(self):
        self.audit()
        uuids = dict(self.db.execute("SELECT id,uuid FROM TARGET_INFO_GPU WHERE vmId=?",(1<<48,)))
        self.db.execute("UPDATE TARGET_INFO_GPU SET uuid=CASE id WHEN 0 THEN ? WHEN 1 THEN ? END WHERE vmId=? AND id IN (0,1)",
                        (uuids[1],uuids[0],1<<48))
        self.db.execute("UPDATE TARGET_INFO_CUDA_DEVICE SET gpuId=1 WHERE pid=? AND cudaId=0",(PID,))
        self.assertEqual(self.db.execute("SELECT uuid FROM TARGET_INFO_GPU WHERE vmId=? AND id=1",(1<<48,)).fetchone()[0],
                         self.context["uuid"].removeprefix("GPU-"))
        with self.assertRaisesRegex(ValueError,"Actual model CUDA ordinal/UUID differs"):
            self.audit()

    def test_all_four_activity_families_reject_foreign_device_process_and_null(self):
        for table in ("KERNEL","MEMCPY","MEMSET","SYNCHRONIZATION"):
            for change in ("globalPid=globalPid+(1<<24)","globalPid=globalPid+(1<<48)",
                    "globalPid=NULL","deviceId=1","deviceId=NULL","start=end"):
                if table == "KERNEL" and change == "deviceId=NULL":change="deviceId='unknown'"
                self.mutation_rejected("UPDATE CUPTI_ACTIVITY_KIND_"+table+" SET "+change)

    def test_memcpy_endpoints_explicit_foreign_reject_and_null_never_becomes_zero(self):
        value=self.audit()
        for field in ("srcDeviceId","dstDeviceId"):
            self.assertEqual(value["memcpy_endpoint_observations"][field],{"unobserved":1})
            for wrong in (1,-1):self.mutation_rejected("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET "+field+"=?",(wrong,))
        self.db.execute("INSERT INTO ENUM_CUDA_MEMCPY_OPER VALUES(10,'CUDA_MEMCPY_KIND_PTOP','Peer')")
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY SELECT start+1,end+1,deviceId,globalPid,correlationId+1,bytes,10,0,0 FROM CUPTI_ACTIVITY_KIND_MEMCPY")
        self.assertEqual(self.audit()["copies"]["CUDA_MEMCPY_KIND_PTOP"]["count"],1)
        for field in ("srcDeviceId","dstDeviceId"):
            for wrong in (None,1):self.mutation_rejected("UPDATE CUPTI_ACTIVITY_KIND_MEMCPY SET "+field+"=? WHERE copyKind=10",(wrong,))

    def test_runtime_and_driver_provenance_never_silently_filter_foreign_cuda(self):
        self.db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_DRIVER AS SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE 0")
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_DRIVER SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId=20")
        self.audit()
        for table in ("RUNTIME","DRIVER"):
            for key in (GPID-(1<<24),GPID+(1<<24),GPID+(1<<48),None,0):
                self.mutation_rejected("UPDATE CUPTI_ACTIVITY_KIND_"+table+" SET globalTid=?",(key,))



    def test_248_exact_absent_payload_warnings_preserve_unbound_not_owned_subjects(self):
        messages=q.profile_audit.model_summary_policy()
        from profile_overlap_v1 import scope_contract as scope
        subjects=[GPID-(1<<24)]+[GPID+((i+1)<<24) for i in range(81)]
        for i,key in enumerate(subjects[1:9]):
            self.db.execute("INSERT INTO PROCESSES VALUES(?,?,?)",(key,PID+i+1,"SQL-only-not-owned"))
        for key in subjects:
            for text in scope.NO_PAYLOAD:
                self.db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,2,?,?)",(text,key))
        self.db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,2,?,?)",(scope.NO_PAYLOAD[0],GPID))
        self.db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,2,?,?)",(scope.SCHEDULING,scope.GLOBAL_DIAGNOSTIC_PID))
        value=self.audit();warnings=[r for r in value["diagnostics"] if r["mechanism"]]
        self.assertEqual(len(warnings),248)
        unbound={r["subject_global_pid"] for r in warnings if r["binding"]=="unbound_not_owned"}
        self.assertEqual(len(unbound),73)
        self.assertTrue(all(r["raw"] and r["effect"] for r in warnings))
        self.assertEqual(value["policy"],messages)
        self.assertEqual(value["status"],"PARTIAL_OBSERVED_MODEL_SUMMARY")
        self.assertFalse(value["full_trace_accepted"])
        self.assertIsNone(value["dropped_event_count"])
        self.assertFalse(value["cpu_scheduling_claim"]);self.assertFalse(value["expert_range_claim"])
        self.assertIn("unproven",value["api_count_exhaustiveness"])

    def test_model_cuda_failure_unknown_message_origin_subject_and_loss_reject(self):
        from profile_overlap_v1 import scope_contract as scope
        for text in (*scope.NO_PAYLOAD[1:],"unknown warning","dropped events"):
            self.mutation_rejected("INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,2,?,?)",(text,GPID))
        for severity in (1,2,3):
            self.mutation_rejected("INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,?,?,?)",(severity,"lost events",GPID+(1<<24)))
        for source,timestamp,subject in ((1,2,GPID),(3,1,GPID),(3,2,None),(3,2,-1),(3,2,GPID+1)):
            self.mutation_rejected("INSERT INTO DIAGNOSTIC_EVENT VALUES(0,?,?,2,?,?)",(timestamp,source,scope.NO_PAYLOAD[0],subject))
        self.db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,2,?,?)",(scope.NO_PAYLOAD[0],GPID))
        for table in ("ENUM_DIAGNOSTIC_SOURCE_TYPE","ENUM_DIAGNOSTIC_TIMESTAMP_SOURCE"):
            self.mutation_rejected("UPDATE "+table+" SET name='UnknownOrigin'")
        self.mutation_rejected("INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,2,?,?)",("unknown warning",GPID+(1<<24)))

    def test_absent_scheduling_requires_disabled_collection_and_exact_global_subject(self):
        from profile_overlap_v1 import scope_contract as scope
        self.db.execute("INSERT INTO DIAGNOSTIC_EVENT VALUES(0,2,3,2,?,?)",(scope.SCHEDULING,scope.GLOBAL_DIAGNOSTIC_PID))
        self.audit()
        for sql in ("UPDATE META_DATA_CAPTURE SET value='true' WHERE name='COLLECT_THREAD_STATE_TRACE'",
                "DELETE FROM META_DATA_CAPTURE WHERE name='COLLECT_THREAD_STATE_TRACE'",
                "UPDATE DIAGNOSTIC_EVENT SET globalPid="+str(GPID)+" WHERE severity=2"):
            self.mutation_rejected(sql)

    def test_absent_optional_activity_tables_stay_unobserved_not_zero(self):
        for table in ("CUPTI_ACTIVITY_KIND_MEMSET","CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
            self.db.execute("DROP TABLE "+table)
        value=self.audit()
        for table in ("CUPTI_ACTIVITY_KIND_MEMSET","CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
            self.assertEqual(value["activity_tables"][table]["state"],"unobserved")
            self.assertNotIn("count",value["activity_tables"][table])



def grouped_plan():
    r=q.retained_evidence
    value=dict(schema='ferrule.grouped-upload.current-quality-plan.v1', commit=r.GROUPED_COMMIT,
        source_archive_sha256=r.GROUPED_SOURCE_SHA, source_digest=r.GROUPED_SOURCE_DIGEST,
        jobs=r.grouped_jobs(), atol=.00025, rtol=.012, transfer_policy=copy.deepcopy(r.GROUPED_POLICY),
        capacity_and_options=copy.deepcopy(r.GROUPED_FIXED_INPUTS['composition_capacity_and_options']),
        retries=0, extra_warmups=0, calibration=0, profiling_captures=0, profiling=False,
        stop_on_first_failure=True, independent_review_required=True, measurement='quality_only_no_performance_claim',
        pins={str(q.ROOT/'target/validation/retained-upload-57e4629-gpu-20261016-01/retained-plan.json'):
            '6041cae37cdd08795a980e912af3ab82b52cefe3a795f42fa2631513e284da76'})
    value.update(copy.deepcopy(r.GROUPED_FIXED_INPUTS))
    value['artifacts']={role:dict(executable=str(r.GROUPED_STAGE/'staged'/role),sha256=sha) for role,sha in r.GROUPED_BINARIES.items()}
    value['host_options']=dict(max_host_bytes=64<<20,max_metadata_bytes=8<<20,worker_stack_bytes=2<<20)
    return value


def grouped_report(row, manifest):
    value=report(row,manifest);kind=row['kind'];cold=kind=='cold'
    value['test_config'].update(name='storage-middle64-grouped-'+row['schedule'],
        expert_transfer_policy=copy.deepcopy(q.retained_evidence.GROUPED_POLICY))
    value['effective_config'].update(expert_transfer_policy=copy.deepcopy(q.retained_evidence.GROUPED_POLICY),
        expert_transfer_selection='explicit existing-owner capability before first prediction')
    for call in value['expert_read_quality']['calls']:
        for endpoint in ('before','after'):
            read=call[endpoint]['expert_read_pipeline']['value']
            if read['stats']['workers_created']:
                read['stats']['peak_host_charged_bytes']=q.retained_evidence.GROUPED_MINIMUM_AGGREGATE_BYTES
        call['delta']['peak_host_charged_bytes']=call['after']['expert_read_pipeline']['value']['stats']['peak_host_charged_bytes']
    diagnostic=kind!='ordinary'
    value.update(actual_peak_reserved_bytes=4<<30,reference_manifest_sha256=manifest['full40_reference_sha256'],
        next_token_checks=[dict(equal=True)]*4,cold_owner_restart=cold,
        observer_mode='diagnostic_routes_and_states' if diagnostic else 'ordinary_no_diagnostic_observer',
        ordinary_no_observer_claim=not diagnostic,
        metrics=[dict(failures=0,nonfinite=0)]*(804 if kind=='full40' else 1809 if cold else 5427 if diagnostic else 27),
        logits_compared=1986560 if kind=='full40' else 3228160 if cold else 9187840)
    if cold:
        value['cases']=value['cases'][:1]
        value['prediction_fingerprints']['calls']=dict(list(value['prediction_fingerprints']['calls'].items())[:28])
        for name,stream in value['prediction_denominators']['streams'].items():
            if name in ('common_history','free_feedback'):
                for key in ('prediction_calls','greedy_checks','input_history_matched_reference_calls'):stream[key]=9
    return value


class GroupedQualityTests(unittest.TestCase):
    def setUp(self):
        self.plan=grouped_plan()
        patch=mock.patch.object(q,'PLAN',dict(self.plan,gates=self.plan['jobs']));patch.start();self.addCleanup(patch.stop)
        for name in ('subprocess.Popen','subprocess.run','os.kill','os.execve'):
            patch=mock.patch(name,side_effect=AssertionError('CPU mutation: no execution/signal'));patch.start();self.addCleanup(patch.stop)
        for obj,name in ((q.idle,'observe'),(q.ct,'exclusive_owner_lock'),(q.supervisor,'run_process')):
            patch=mock.patch.object(obj,name,side_effect=AssertionError('No GPU/locks/custody'));patch.start();self.addCleanup(patch.stop)

    def validate(self,value,row,prior=None):
        with mock.patch.object(q,'primary',return_value=(Path('/synthetic/primary.json'),'e'*64,prior or value)):
            q.validate_composition(value,row)

    def test_exact14_backend_five_model_eight_quality_selectors_and_cut_mutations(self):
        r=q.retained_evidence;r.validate_grouped_plan(self.plan)
        self.assertEqual([sum(g['role']==role for g in self.plan['jobs']) for role in ('backend','model','composition')],[1,5,8])
        for field,bad in (('commit','f'*40),('source_archive_sha256','f'*64),
                ('source_digest','f'*64),('atol',.000251),('rtol',.01201),
                ('retries',1),('extra_warmups',1),('calibration',1),('profiling',True),
                ('profiling_captures',1),('measurement','latency'),('independent_review_required',False),('artifacts',{}),('host_options',{})):
            value=copy.deepcopy(self.plan);value[field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):r.validate_grouped_plan(value)
        for index in range(14):
            for field,bad in (('selector','wrong'),('seconds',1201),('binary_sha256','f'*64),
                    ('transfer_mode','retained'),('role','foreign'),('phase','profile')):
                value=copy.deepcopy(self.plan);value['jobs'][index][field]=bad
                with self.subTest(index=index,field=field),self.assertRaises(ValueError):r.validate_grouped_plan(value)
                with self.assertRaises(ValueError):r.grouped_job(self.plan,value['jobs'][index])
        for key in r.GROUPED_FIXED_INPUTS:
            value=copy.deepcopy(self.plan);value[key]=None
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_grouped_plan(value)
        for mutate in (lambda v:v['jobs'].reverse(),lambda v:v['pins'].clear(),lambda v:v['capacity_and_options'].update(max_bytes=321<<20)):
            value=copy.deepcopy(self.plan);mutate(value)
            with self.assertRaises(ValueError):r.validate_grouped_plan(value)

    def test_exact_per_role_payload_argv_never_uses_a_shared_composition_executable(self):
        self.assertNotIn("executable", self.plan)
        for row in self.plan["jobs"]:
            self.assertEqual(q.payload_argv(row), [row["executable"], row["selector"],
                "--exact", "--ignored", "--nocapture", "--test-threads=1"])
            for field, bad in (("executable", "/foreign/binary"), ("selector", "wrong"), ("seconds", 1200.0)):
                with self.subTest(job=row["id"], field=field), self.assertRaises(ValueError):
                    q.payload_argv(dict(row, **{field: bad}))
        with mock.patch.object(q, "PLAN", plan()):
            for row in p.fixed_jobs():
                self.assertEqual(q.payload_argv(row), p.payload_argv(q.PLAN, row))

    def test_all_eight_grouped_quality_shapes_accept_baseline_peak_not_retained_minimum(self):
        for row in self.plan['jobs'][6:]:
            value=grouped_report(row,self.plan);self.validate(value,row)
            self.assertEqual(q.retained_evidence.GROUPED_MINIMUM_AGGREGATE_BYTES,16777984)
            self.assertLess(q.retained_evidence.GROUPED_MINIMUM_AGGREGATE_BYTES,q.retained_evidence.MINIMUM_AGGREGATE_BYTES)

    def test_exact_grouped_policy_types_missing_and_cross_policy_reject_both_manifests(self):
        r=q.retained_evidence
        for row in self.plan['jobs'][6:]:
            for container in ('test_config','effective_config'):
                for bad in (None,{},r.POLICY,dict(r.GROUPED_POLICY,mode='synchronous'),
                        dict(r.GROUPED_POLICY,additional_pinned_bytes=1),dict(r.GROUPED_POLICY,additional_pinned_bytes=False),
                        dict(r.GROUPED_POLICY,host_quota_unchanged=1),dict(r.GROUPED_POLICY,physical_overhead_measured=0),
                        dict(r.GROUPED_POLICY,source_custody='borrowed_unbounded')):
                    value=grouped_report(row,self.plan);value[container]['expert_transfer_policy']=copy.deepcopy(bad)
                    with self.subTest(job=row['id'],container=container,bad=bad),self.assertRaises(ValueError):self.validate(value,row)

    def test_positive_real_upload_and_minimum_peak_mandatory(self):
        for row in self.plan['jobs'][6:]:
            for change in ('zero_upload','low_peak'):
                value=grouped_report(row,self.plan)
                for call in value['expert_read_quality']['calls']:
                    for endpoint in ('before','after'):
                        physical=call[endpoint]
                        if change=='zero_upload':physical['expert_cache']['value']['uploads']=0
                        elif physical['expert_read_pipeline']['value']['stats']['workers_created']:
                            physical['expert_read_pipeline']['value']['stats']['peak_host_charged_bytes']=16777983
                    call['delta']['peak_host_charged_bytes']=call['after']['expert_read_pipeline']['value']['stats']['peak_host_charged_bytes']
                with self.subTest(job=row['id'],change=change),self.assertRaises(ValueError):self.validate(value,row)

    def test_shared_cache_health_unknown_pending_and_upload_custody_reject_all_cells_endpoints(self):
        for row in self.plan['jobs'][6:]:
            for endpoint in ('before','after'):
                for field,bad in (('state','unobserved'),('pending_upload_bytes',1),('unknown_quarantine_bytes',1),
                        ('peak_bytes',(320<<20)+1),('resident_entries',65),('scratch_bytes',(8<<20)+1),
                        ('quarantined',True),('counters_overflowed',True),('uploads',True)):
                    value=grouped_report(row,self.plan);item=value['expert_read_quality']['calls'][1][endpoint]['expert_cache']
                    (item if field=='state' else item['value'])[field]=bad
                    with self.subTest(job=row['id'],endpoint=endpoint,field=field),self.assertRaises(ValueError):self.validate(value,row)
                for field in ('worker_unknown','worker_join_pending'):
                    value=grouped_report(row,self.plan);value['expert_read_quality']['calls'][1][endpoint]['expert_read_pipeline']['value'][field]=True
                    with self.assertRaises(ValueError):self.validate(value,row)
                for field in q.retained_evidence.UPLOAD_FIELDS:
                    value=grouped_report(row,self.plan);value['expert_read_quality']['calls'][1][endpoint]['expert_read_pipeline']['value']['current_charge'][field]=1
                    with self.assertRaises(ValueError):self.validate(value,row)

    def test_same_cut_grouped_primary_rejects_report_hash_policy_schedule_and_source_crossbinding(self):
        for row in self.plan['jobs'][10:]:
            source=next(g for g in self.plan['jobs'] if g['phase']=='primary' and g['schedule']==row['schedule'])
            data=grouped_report(source,self.plan)
            accepted=dict(status='passed',report_sha256='e'*64,commit=self.plan['commit'],source_digest=self.plan['source_digest'],
                binary_sha256=source['binary_sha256'],job=source)
            def load(path):return accepted if path.name=='gate-result.json' else data
            with mock.patch.object(q.ct,'load',side_effect=load),mock.patch.object(q.ct,'sha',return_value='e'*64):
                self.assertEqual(q.primary(row)[2],data)
                for field,bad in (('status','failed'),('report_sha256','f'*64),('commit',p.COMMIT),
                        ('binary_sha256',p.BINARY_SHA),('source_digest','f'*64),('job',self.plan['jobs'][0]),('job',dict(source,seconds=1200.0)),
                        ('job',next(g for g in self.plan['jobs'] if g['phase']=='primary' and g['schedule']!=row['schedule']))):
                    original=accepted[field];accepted[field]=bad
                    with self.subTest(job=row['id'],field=field),self.assertRaises(ValueError):q.primary(row)
                    accepted[field]=original
                for container in ('test_config','effective_config'):
                    original=data[container]['expert_transfer_policy'];data[container]['expert_transfer_policy']=q.retained_evidence.POLICY
                    with self.assertRaises(ValueError):q.primary(row)
                    data[container]['expert_transfer_policy']=original

    def test_numeric_reference_routes_greedy_streams_and_no_observer_fingerprints_unchanged(self):
        for row in self.plan['jobs'][6:]:
            for field,bad in (('atol',.001),('rtol',.02),('logits_compared',1),('device_budget_bytes',8<<30),
                    ('reference_manifest_sha256','wrong') if row['kind']=='full40' else ('oracle_manifest_sha256','wrong')):
                value=grouped_report(row,self.plan);value[field]=bad
                with self.subTest(job=row['id'],field=field),self.assertRaises(ValueError):self.validate(value,row)
            value=grouped_report(row,self.plan);value['metrics'][0]['failures']=1
            with self.assertRaises(ValueError):self.validate(value,row)
            if row['kind']=='full40':
                value=grouped_report(row,self.plan);value['next_token_checks'][0]['equal']=False
                with self.assertRaises(ValueError):self.validate(value,row)
            else:
                value=grouped_report(row,self.plan);value['cases'][0]['free_greedy_ids']=[99]*9
                with self.assertRaises(ValueError):self.validate(value,row)
        row=self.plan['jobs'][10];value=grouped_report(row,self.plan);prior=copy.deepcopy(value)
        value['prediction_fingerprints']['calls']['prediction.00']['next_token_id']=999
        with self.assertRaises(ValueError):self.validate(value,row,prior)

    def test_no_grouped_review_blocks_before_any_gpu_or_final_and_profile_go_is_wrong_scope(self):
        with tempfile.TemporaryDirectory() as folder,mock.patch.object(q,'OUT',Path(folder)):
            with self.assertRaisesRegex(ValueError,'Independent review/GO missing'):q.main()
            self.assertFalse((Path(folder)/'FINAL.json').exists())
            record=dict(schema='ferrule.worker-reuse.profile-only-control-review.v1',status='passed',
                gpu_execution_authorized=True,reviewer='synthetic-only',commit=p.COMMIT,
                qualification_plan_sha256='e'*64,control_sha256='e'*64,jobs=p.profile_jobs(),review_packet_sha256='e'*64)
            path=Path(folder)/'independent-review.json';path.write_text(json.dumps(record));path.chmod(0o444)
            with mock.patch.object(q.ct,'sha',return_value='e'*64),self.assertRaises(ValueError):q.require_independent_review()


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





def grouped_performance_plan():
    value=grouped_plan();r=q.retained_evidence
    value.update(schema='ferrule.grouped-upload.performance-plan.v1',jobs=p.fixed_jobs('grouped'),
        executable=p.fixed_jobs('grouped')[0]['executable'],binary_sha256=r.GROUPED_BINARIES['composition'],
        nsys_version=p.VERSION,nsys=str(q.ROOT/'target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/nsys'),
        nsys_sha256='525f35a10de49b396e91a3831ae40e98a8631d619b7c8bbb4c54469b8c08a72e',
        observed_model_summary_policy=q.profile_audit.model_summary_policy(),
        policies=dict(synchronous='unchanged_default',grouped=copy.deepcopy(r.GROUPED_POLICY)),
        completed_quality_source=p.grouped_quality_source(),profiling_captures=4,profiling=True,
        exact_cross_policy_fingerprints_required=True,historical_results_in_population=False,
        measurement='whole_test_libtest_footer_only_not_token_TTFT_or_request_latency')
    for name,field in (('grouped-plan.json','plan_sha256'),('FINAL.json','final_sha256'),('independent-review.json','review_sha256')):
        value['pins'][str(r.GROUPED_STAGE/name)]=value['completed_quality_source'][field]
    value['pins'][value['nsys']]=value['nsys_sha256']
    return value


def comparison_report(row, manifest):
    return grouped_report(row,manifest) if row['transfer_mode']=='grouped' else report(row,manifest)


class GroupedPerformanceTests(unittest.TestCase):
    def setUp(self):
        ContractTests.setUp(self)
        self.plan=grouped_performance_plan()
        q.PLAN=dict(self.plan,gates=self.plan['jobs'])

    validate=ContractTests.validate

    def test_generic_arm_preserves_historical_jobs_and_fixes_same_cut24_abba(self):
        p.validate_plan(self.plan)
        self.assertEqual(p.fixed_jobs(),p.fixed_jobs('retained'))
        self.assertEqual([r['phase'] for r in self.plan['jobs']].count('primary'),4)
        self.assertEqual([r['phase'] for r in self.plan['jobs']].count('profile'),4)
        for schedule in ('blocking','lookahead'):
            rows=[r for r in self.plan['jobs'] if r['phase']=='latency' and r['schedule']==schedule]
            self.assertEqual([r['transfer_mode'] for r in rows],['synchronous','grouped','grouped','synchronous','grouped','synchronous','synchronous','grouped'])
        for index in range(24):
            for field,bad in (('binary_sha256',p.BINARY_SHA),('executable',p.fixed_jobs()[0]['executable']),
                    ('transfer_mode','retained'),('selector','wrong'),('seconds',1200.0),('phase','foreign')):
                changed=copy.deepcopy(self.plan);changed['jobs'][index][field]=bad
                with self.subTest(index=index,field=field),self.assertRaises(ValueError):p.validate_plan(changed)

    def test_cut_quota_policy_tool_scope_and_completed_quality_pin_mutations(self):
        for field,bad in (('commit',p.COMMIT),('source_digest','f'*64),('source_archive_sha256',p.SOURCE_SHA),
                ('binary_sha256',p.BINARY_SHA),('executable',p.fixed_jobs()[0]['executable']),
                ('completed_quality_source',{}),('policies',{}),('profiling',False),('profiling_captures',3),
                ('retries',False),('calibration',1),('extra_warmups',1),('atol',.000251),('rtol',.01201),
                ('independent_review_required',False),('exact_cross_policy_fingerprints_required',False),
                ('historical_results_in_population',True),('measurement','TTFT'),('nsys_version','2025.4'),
                ('nsys','/foreign/nsight-systems-cli/2025.3.1/target-linux-x64/nsys'),('nsys_sha256','f'*64),
                ('observed_model_summary_policy',dict(q.profile_audit.model_summary_policy(),full_trace_accepted=True))):
            changed=copy.deepcopy(self.plan);changed[field]=bad
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_plan(changed)
        for name in (*q.retained_evidence.GROUPED_FIXED_INPUTS,'host_options','capacity_and_options'):
            changed=copy.deepcopy(self.plan);changed[name]={}
            with self.subTest(name=name),self.assertRaises(ValueError):p.validate_plan(changed)
        for path in self.plan['pins']:
            if path.endswith('retained-plan.json'):continue
            changed=copy.deepcopy(self.plan);changed['pins'][path]='f'*64
            with self.subTest(pin=path),self.assertRaises(ValueError):p.validate_plan(changed)

    def test_all24_reports_keep_baseline_host_peak_and_exact_grouped_or_sync_policy(self):
        for row in self.plan['jobs']:
            value=comparison_report(row,self.plan);self.validate(value,row)
            self.assertLess(value['expert_read_quality']['calls'][-1]['after']['expert_read_pipeline']['value']['stats']['peak_host_charged_bytes'],q.retained_evidence.MINIMUM_AGGREGATE_BYTES)
            for container in ('test_config','effective_config'):
                changed=copy.deepcopy(value);changed[container]['expert_transfer_policy']=q.retained_evidence.POLICY
                with self.assertRaises(ValueError):self.validate(changed,row)

    def test_all24_cache_health_both_endpoints_reject_with_unchanged_ordered_work(self):
        for row in self.plan['jobs']:
            base=comparison_report(row,self.plan);expected=p.work_vector(base)
            for endpoint in ('before','after'):
                for field,bad in (('state','unobserved'),('state','unknown'),('peak_bytes',(320<<20)+1),
                        ('pending_upload_bytes',1),('unknown_quarantine_bytes',1),('quarantined',True),
                        ('counters_overflowed',True),('resident_entries',65),('scratch_bytes',(8<<20)+1),
                        ('charged_bytes',0),('uploads',False),('uploads',None)):
                    value=copy.deepcopy(base);item=value['expert_read_quality']['calls'][1][endpoint]['expert_cache']
                    (item if field=='state' else item['value'])[field]=bad
                    # The ordered work arrays themselves are unchanged for the historical invalid-evidence mutations.
                    if field not in p.CACHE_WORK:
                        self.assertEqual([c['delta'] for c in value['expert_read_quality']['calls']], [c['delta'] for c in base['expert_read_quality']['calls']])
                    with self.subTest(job=row['id'],endpoint=endpoint,field=field):
                        with self.assertRaises(ValueError):self.validate(value,row)
                        with self.assertRaises(ValueError):p.work_vector(value)
                for field in ('worker_unknown','worker_join_pending'):
                    value=copy.deepcopy(base);value['expert_read_quality']['calls'][1][endpoint]['expert_read_pipeline']['value'][field]=True
                    with self.assertRaises(ValueError):self.validate(value,row)
        self.assertEqual(len(expected),64)

    def test_same_cut_primary_rejects_historical_cut_foreign_policy_job_and_report(self):
        for row in self.plan['jobs'][4:]:
            primary=next(g for g in self.plan['jobs'][:4] if p.primary_key(g)==p.primary_key(row))
            data=comparison_report(primary,self.plan)
            accepted=dict(status='passed',report_sha256='e'*64,commit=self.plan['commit'],binary_sha256=self.plan['binary_sha256'],source_digest=self.plan['source_digest'],job=primary)
            p.validate_primary(accepted,data,row,self.plan,'e'*64)
            for field,bad in (('commit',p.COMMIT),('binary_sha256',p.BINARY_SHA),('source_digest','f'*64),
                    ('report_sha256','f'*64),('job',p.fixed_jobs()[0]),('job',dict(primary,seconds=1200.0)),
                    ('job',next(g for g in self.plan['jobs'][:4] if p.primary_key(g)!=p.primary_key(row)))):
                changed=copy.deepcopy(accepted);changed[field]=bad
                with self.subTest(job=row['id'],field=field),self.assertRaises(ValueError):p.validate_primary(changed,data,row,self.plan,'e'*64)

    def test_exact_work_checks_and_cross_policy_fingerprints_required(self):
        left=comparison_report(self.plan['jobs'][0],self.plan);right=comparison_report(self.plan['jobs'][1],self.plan)
        self.assertTrue(p.compare_work(left,right,require_fingerprints=True)['fingerprints_equal'])
        changed=copy.deepcopy(right);changed['prediction_fingerprints']['calls']['prediction.00']['next_token_id']=999
        with self.assertRaisesRegex(ValueError,'fingerprints differ'):p.compare_work(left,changed,require_fingerprints=True)
        changed=copy.deepcopy(right);changed['expert_read_quality']['calls'][1]['label']='foreign'
        with self.assertRaises(ValueError):p.compare_work(left,changed,require_fingerprints=True)
        changed=copy.deepcopy(right);changed['prediction_denominators']['streams']['common_history']['greedy_checks']+=1
        with self.assertRaises(ValueError):p.compare_work(left,changed,require_fingerprints=True)

    def test_profile_four_only_and_same_cut_role_custody_stays_exact(self):
        for row in self.plan['jobs']:
            self.assertEqual(q.payload_argv(row),p.payload_argv(self.plan,row))
            if row['phase']!='profile':
                with self.assertRaises(ValueError):p.profile_argv(self.plan,row,Path('/fresh'),['wrapper'])
                continue
            argv=p.profile_argv(self.plan,row,Path('/fresh'),['wrapper'])
            for flag in ('--kill=none','--wait=primary','--sample=none','--cpuctxsw=none','--duration=1080','--force-overwrite=false'):
                self.assertIn(flag,argv)
            identity=dict(pid=123,birth_ticks=456,ppid=122,state='R')
            record=dict(before=identity,after=copy.deepcopy(identity),executable_sha256=row['binary_sha256'],command=q.payload_argv(row))
            p.validate_role(record,identity,row['binary_sha256'],q.payload_argv(row))
            record['after']['birth_ticks']+=1
            with self.assertRaises(ValueError):p.validate_role(record,identity,row['binary_sha256'],q.payload_argv(row))

    def test_pairs_exclude_profiles_and_historical_population_and_preserve_regression(self):
        rows=[dict(job=row,status='passed',whole_test_elapsed=dict(seconds=999 if row['phase']=='profile' else 5 if row['transfer_mode']=='grouped' else 4)) for row in self.plan['jobs']]
        summary=p.pair_summary(rows,'grouped')
        self.assertEqual(len(summary['pairs']),8);self.assertFalse(summary['significance_claim'])
        self.assertTrue(all(v['grouped_over_synchronous']==1.25 for v in summary['pairs']))
        self.assertNotIn('retained_over_synchronous',summary['pairs'][0])
        for bad in (rows[4:],rows+[rows[-1]],[dict(row,job=p.fixed_jobs()[i]) for i,row in enumerate(rows)]):
            with self.assertRaises(ValueError):p.pair_summary(bad,'grouped')

    def test_no_review_or_quality14_profile4_historical24_go_cannot_authorize_new24(self):
        with tempfile.TemporaryDirectory() as folder,mock.patch.object(q,'OUT',Path(folder)):
            with self.assertRaisesRegex(ValueError,'Independent review/GO missing'):q.main()
            self.assertFalse((Path(folder)/'FINAL.json').exists())
            for schema,jobs,commit in (('ferrule.worker-reuse.control-review.v1',q.retained_evidence.grouped_jobs(),self.plan['commit']),
                    ('ferrule.worker-reuse.profile-only-control-review.v1',p.profile_jobs(),p.COMMIT),
                    ('ferrule.worker-reuse.performance-control-review.v1',p.fixed_jobs(),p.COMMIT)):
                record=dict(schema=schema,status='passed',gpu_execution_authorized=True,reviewer='synthetic-only',commit=commit,
                    qualification_plan_sha256='e'*64,control_sha256='e'*64,jobs=jobs,review_packet_sha256='e'*64)
                path=Path(folder)/'independent-review.json';path.chmod(0o644) if path.exists() else None;path.write_text(json.dumps(record));path.chmod(0o444)
                with mock.patch.object(q.ct,'sha',return_value='e'*64),self.assertRaises(ValueError):q.require_independent_review()


if __name__=='__main__':
    unittest.main()
