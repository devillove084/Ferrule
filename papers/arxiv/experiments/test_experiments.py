"""CPU-only contracts, installed-source syntax checks, and real CPU custody smoke."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import contract as ct
import runner
import telemetry

UUID = 'GPU-b980f22b-58a3-fc88-21c1-ae86053d8535'


def good_process():
    identity = dict(launch_id='nonce', helper_identity={'pid': 123, 'start_time': 456}, containment_verified=True)
    return dict(returncode=0, timed_out=False, handshake=identity.copy(),
                custody=dict(identity, status='proven_drained', echild=True),
                descendant_custody='proven_drained', cleanup_errors=[])


def good_http():
    row = dict(status='success', actual_input_tokens=1, actual_output_tokens=2, finish_reason='length')
    return dict(status='complete', runs=[dict(requests=[row], not_issued_indices=[])],
                warmups=[], measurements={'all_requests_successful': True})


class Contracts(unittest.TestCase):
    def test_default_plan_never_spawns_or_queries(self):
        with mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('no subprocess')):
            p = runner.plan(ct.config())
        self.assertEqual(p['status'], 'cpu_only_not_executed')
        self.assertIsNone(p['exact_speedup'])
        self.assertIsNone(p['model_identity']['full_checkpoint_sha256'])
        self.assertTrue(p['quality_history']['status'].startswith('PASS_'))

    def test_authorization_precedes_telemetry_and_launch(self):
        for mode in ('feasibility', 'run'):
            with mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('must not spawn')):
                self.assertEqual(runner.main([mode]), 2)
        with mock.patch.object(subprocess, 'run', side_effect=AssertionError('must not query')):
            with self.assertRaisesRegex(ValueError, 'grant'):
                telemetry.gpu_sample('/usr/bin/nvidia-smi', UUID, False, 'test')

    def test_unsafe_ids_and_append_only(self):
        for name in ('../escape', '/tmp/escape', '', 'x.y', 'a/b'):
            with self.assertRaises(ValueError):
                ct.new_run(name)
        with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp:
            path = Path(tmp) / 'receipt.json'
            ct.write(path, {'value': 1})
            with self.assertRaises(FileExistsError):
                ct.write(path, {'value': 2})
            self.assertEqual(ct.load(path), {'value': 1})

    def test_unknown_and_stale_parameters(self):
        for modify in (
                lambda c: c['protocol'].update(seed=0),
                lambda c: c['vllm_flags'].update({'--dtype': 'bf16'}),
                lambda c: c['vllm_flags'].update({'--enable-expert-parallel': True}),
                lambda c: c['vllm_flags'].update({'--kv-cache-memory-bytes': '16777216'}),
                lambda c: c['protocol'].update(context=160),
                lambda c: c['resources'].update(hard_memory_partition=True)):
            c = copy.deepcopy(ct.BASE)
            modify(c)
            with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp:
                path = Path(tmp) / 'config.json'
                ct.write(path, c)
                with self.assertRaisesRegex(ValueError, 'Unknown/stale'):
                    ct.config(path)

    def test_cpu_build_is_explicit_release_offline(self):
        c = ct.config()
        argv, env = ct.build_argv(c), ct.build_environment(c)
        for token in ('--release', '--offline', '--locked', 'cuda'):
            self.assertIn(token, argv)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['NVIDIA_VISIBLE_DEVICES'], 'none')
        self.assertEqual(env['CARGO_PROFILE_RELEASE_LTO'], 'fat')
        self.assertEqual(env['CARGO_PROFILE_RELEASE_OPT_LEVEL'], '3')
        with mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('no build')):
            self.assertEqual(runner.main(['build']), 2)

    def test_release_artifact_rejects_historical_debug(self):
        row = dict(reason='compiler-artifact', target={'name': 'ferrule'}, executable='/release/ferrule',
                   profile=dict(opt_level='3', debug_assertions=False, test=False), features=['cuda'])
        self.assertEqual(ct.release_artifact([row]), row)
        for change in ({'opt_level': '1'}, {'debug_assertions': True}, {'test': True}):
            bad = copy.deepcopy(row)
            bad['profile'].update(change)
            with self.assertRaises(ValueError):
                ct.release_artifact([bad])

    def test_authenticated_custody_not_exit_code(self):
        self.assertTrue(ct.successful_process(good_process()))
        for change in (dict(timed_out=True), dict(returncode=None), dict(returncode=1), dict(custody={}),
                       dict(descendant_custody='unknown'), dict(cleanup_errors=['failure'])):
            self.assertFalse(ct.successful_process(dict(good_process(), **change)))
        bad = good_process()
        bad['custody']['launch_id'] = 'other'
        self.assertFalse(ct.drained(bad))

    def test_missing_http_counts_or_timeout_is_not_zero_success(self):
        work = ct.BASE['workloads']['feasibility']
        self.assertTrue(ct.http_success(good_http(), work))
        for change in ({'status': 'timeout'}, {'actual_output_tokens': None},
                       {'actual_output_tokens': 0}, {'actual_input_tokens': None}, {'finish_reason': None}):
            bad = good_http()
            bad['runs'][0]['requests'][0].update(change)
            self.assertFalse(ct.http_success(bad, work))
        self.assertFalse(ct.http_success({}, work))
        eos = good_http()
        eos['runs'][0]['requests'][0].update(actual_output_tokens=1, finish_reason='stop')
        self.assertTrue(ct.http_success(eos, work))

    def test_missing_telemetry_and_budget_excess_block(self):
        r = ct.BASE['resources']
        for samples, errors in (([], []), ([{}], []), ([dict(gpu_bytes=0, pss_bytes=0)], []),
                                ([dict(gpu_bytes=1, pss_bytes=1)], ['query timeout']),
                                ([dict(gpu_bytes=r['gpu_target_bytes']+1, pss_bytes=1)], [])):
            self.assertEqual(ct.resource_evidence(samples, errors, r)['status'], 'blocked')
        result = ct.resource_evidence([dict(gpu_bytes=100, pss_bytes=200, rss_bytes=300)], [], r)
        self.assertEqual(result['status'], 'observed_within_targets')
        self.assertIsNone(result['pinned_bytes'])
        self.assertFalse(result['exact_peaks'])

    def test_vllm_environment_isolated_and_dtype_flags_actual(self):
        with mock.patch.dict(os.environ, {'LD_LIBRARY_PATH': 'cuda12.4', 'LD_PRELOAD': 'bad',
                                         'VLLM_WEIGHT_OFFLOADING_DISABLE_UVA': '1', 'CUBLAS_WORKSPACE_CONFIG': ':0:0'}):
            spec = ct.server_spec(ct.BASE, 'vllm', '/unused', UUID)
        for key in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'CUBLAS_WORKSPACE_CONFIG', 'VLLM_WEIGHT_OFFLOADING_DISABLE_UVA'):
            self.assertNotIn(key, spec['environment'])
        self.assertEqual(spec['environment']['CUDA_VISIBLE_DEVICES'], UUID)
        self.assertIn('bfloat16', spec['argv'])
        self.assertIn('w13_weight', spec['argv'])
        self.assertNotIn('--enable-expert-parallel', spec['argv'])
        self.assertEqual(runner.static_cli_audit(ct.BASE)['status'], 'pass')

    def test_diagnostic_target_matches_actual_rust_source(self):
        source = ct.ROOT / 'crates/ferrule-model/src/transformer/standard/cuda/invocation_trace.rs'
        self.assertIn('target: "ferrule_performance"', source.read_text())
        self.assertEqual(runner.DIAGNOSTIC_LOG, 'info,ferrule_performance=debug')
        spec = ct.server_spec(ct.BASE, 'ferrule', '/fixture', UUID)
        self.assertEqual(spec['environment']['FERRULE_LOG'], 'info')

    def test_historical_build_paper_pins_do_not_replace_current_preparation(self):
        historic = {str(ct.HERE / 'runner.py'): 'a'*64,
                    str(ct.ROOT / 'scripts/_owned_subreaper.py'): 'b'*64}
        self.assertEqual(ct.build_runtime_pins({'pins': historic}),
                         {str(ct.ROOT / 'scripts/_owned_subreaper.py'): 'b'*64})
        self.assertIn(str(ct.HERE / 'runner.py'), ct.workflow_pins())

    def test_startup_monitor_failure_enters_cleanup_without_waiting_for_ready(self):
        import threading
        server = mock.Mock()
        server.poll.return_value = None
        monitor = mock.Mock(failed=threading.Event())
        guard = runner.StartupGuard(server, monitor)
        self.assertIsNone(guard.poll())
        monitor.failed.set()
        with self.assertRaisesRegex(ValueError, 'Resource monitor failed'):
            guard.poll()
        self.assertEqual(server.poll.call_count, 1)

    def test_missing_telemetry_is_not_measured_infeasibility(self):
        for observation in ({}, {'resources': {'status':'blocked'}},
                            {'resources': {'status':'blocked','errors':['query timeout']}}):
            self.assertFalse(ct.resource_infeasible(observation))
        self.assertTrue(ct.resource_infeasible({'resources': {'errors':['Sampled GPU target exceeded']}}))
        self.assertTrue(ct.resource_infeasible({'resources': {'errors':['Sampled host PSS target exceeded']}}))

    def test_gdn_kernel_fallback_is_not_offload_contract_fallback(self):
        text = 'Falling back to the Triton GDN decode path: fused op is not built'
        result = ct.runtime_evidence('vllm', text)
        self.assertEqual(result['contradictions'], [])
        self.assertEqual(result['fallback_observations'], [text])
        self.assertIn('moe_marlin', result['missing'])

    def test_runtime_missing_evidence_never_claims_match(self):
        for text in ('', 'Using MARLIN Fp8 MoE backend', 'falling back to normal kernel selection'):
            result = ct.runtime_evidence('vllm', text)
            self.assertEqual(result['status'], 'blocked')
            self.assertFalse(result['exact_arithmetic_match'])
        self.assertTrue(ct.runtime_evidence('vllm', 'falling back to normal kernel selection')['contradictions'])

    def test_stale_pin(self):
        with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp:
            p = Path(tmp) / 'input'
            p.write_text('first')
            pins = {str(p): ct.sha(p)}
            ct.check_pins(pins)
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Stale or malformed pin'):
                ct.check_pins(pins)

    def test_release_quality_requires_exact_new_binary_and_reference(self):
        with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp:
            a, b = Path(tmp) / 'reference.json', Path(tmp) / 'result.json'
            ct.write(a, {'reference': 'fixture'})
            ct.write(b, {'result': 'fixture'})
            m = dict(binary_sha256='a'*64, model_identity_sha256='b'*64, protocol_sha256='c'*64)
            q = dict(schema=ct.SCHEMA+'release-quality.v1', status='pass', **m,
                     independent_reference=True, checks=dict(raw_hello_greedy=True, reset_repeat=True, cold_repeat=True),
                     artifact_pins={str(a): ct.sha(a), str(b): ct.sha(b)})
            path = Path(tmp) / 'quality.json'
            ct.write(path, q)
            ct.quality_receipt(path, m)
            with self.assertRaisesRegex(ValueError, 'binary_sha256'):
                ct.quality_receipt(path, dict(m, binary_sha256='d'*64))

    def test_fail_stop_unknown_custody_never_starts_second_engine(self):
        from types import SimpleNamespace
        m = dict(config=ct.BASE, manifest_sha256='a'*64, binary='/pinned/release/ferrule', binary_sha256='b'*64,
                 model_identity={}, quality_history={'status': 'historical_pass'})
        args = SimpleNamespace(sole_gpu_owner_granted=True, gpu_uuid=UUID, owner_note='MOCK ONLY',
                               prepared=Path('/mock/prepared'), release_quality=None,
                               mode='feasibility', feasibility=None)
        with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp, \
                mock.patch.object(runner, 'validate_manifest', return_value=m), \
                mock.patch.object(telemetry, 'idle', return_value={'status': 'verified'}), \
                mock.patch.object(runner, 'supervise', return_value={'returncode': 0, 'timed_out': False}) as launch, \
                mock.patch.object(subprocess, 'run', side_effect=AssertionError('no GPU query')):
            args.prepared = Path(tmp) / 'prepared.json'
            ct.write(args.prepared, m)
            result = runner.execute(args, Path(tmp))
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(launch.call_count, 1)
            self.assertFalse((Path(tmp) / 'vllm').exists())
            self.assertIsNone(result['exact_speedup'])

    def test_stale_manifest_fails_before_query(self):
        from types import SimpleNamespace
        args = SimpleNamespace(sole_gpu_owner_granted=True, gpu_uuid=UUID, owner_note='MOCK ONLY', prepared='bad')
        with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp, \
                mock.patch.object(runner, 'validate_manifest', side_effect=ValueError('stale pin')), \
                mock.patch.object(telemetry, 'idle', side_effect=AssertionError('no telemetry')):
            with self.assertRaisesRegex(ValueError, 'stale'):
                runner.execute(args, Path(tmp))

    def test_schema_config_and_negative_quality(self):
        schema = ct.load(ct.HERE / 'schema.json')
        self.assertEqual(len(schema['oneOf']), 6)
        # Make installed pure-CPU packages visible without executing .pth/site hooks.
        site = ct.absolute(ct.BASE['vllm']).parent.parent / 'lib/python3.12/site-packages'
        sys.path.append(str(site))
        if importlib.util.find_spec('jsonschema'):
            import jsonschema
            jsonschema.Draft202012Validator.check_schema(schema)
            jsonschema.validate(ct.BASE, schema)
            jsonschema.validate(runner.plan(ct.BASE), schema)
            invalid = copy.deepcopy(ct.BASE)
            invalid['protocol']['seed'] = 0
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate(invalid, schema)
            with self.assertRaises(jsonschema.ValidationError):
                jsonschema.validate({'schema': ct.SCHEMA + 'release-quality.v1', 'status': 'pass'}, schema)
        else:
            self.skipTest('Existing jsonschema unavailable; no installation attempted')

    def test_no_heavy_runtime_imported(self):
        for name in ('torch', 'vllm', 'pynvml', 'cuda'):
            self.assertNotIn(name, sys.modules)

    def test_owner_lock_nonblocking(self):
        with ct.exclusive_owner_lock(UUID):
            with self.assertRaisesRegex(ValueError, 'lock is held'):
                with ct.exclusive_owner_lock(UUID):
                    self.fail('second owner acquired lock')

    def test_worker_entry_cannot_bypass_contract(self):
        with mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('must not launch')):
            with self.assertRaisesRegex(ValueError, 'Free --_worker'):
                runner.main(['--_worker'])
            with self.assertRaisesRegex(ValueError, 'config contract'):
                runner.worker_contract({'mode': 'run', 'config': ct.BASE})

    def test_vllm_fix_env_rejects_overrides(self):
        original = ct.load(ct.absolute(ct.BASE['vllm_fix_environment']))
        for key in ct.VLLM_FORBIDDEN_KEYS | {'UNKNOWN_JIT_KEY'}:
            bad = dict(original, **{key: 'bad'})
            with mock.patch.object(ct, 'load', return_value=bad):
                with self.assertRaisesRegex(ValueError, 'allowlist'):
                    ct.validate_vllm_fix_env(ct.BASE, UUID)

    def test_static_cli_values_and_false_negation(self):
        for flag, value in (('--dtype', 'bf16'), ('--gpu-memory-utilization', 'not-a-number'),
                            ('--gpu-memory-utilization', 'NaN'), ('--max-num-seqs', 'two'),
                            ('--cpu-offload-params', 'w13_weight'), ('--enforce-eager', 'true')):
            c = copy.deepcopy(ct.BASE)
            c['vllm_flags'][flag] = value
            with self.assertRaises(ValueError):
                runner.static_cli_audit(c)
        c = copy.deepcopy(ct.BASE)
        c['vllm_flags']['--no-cpu-offload-gb'] = True
        with self.assertRaisesRegex(ValueError, 'reviewed 25'):
            runner.static_cli_audit(c)

    def test_inventory_missing_group_or_added_file(self):
        with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp:
            p = Path(tmp) / 'tool'
            p.write_text('cpu fixture')
            required = {'compiler': [p], 'cutlass_tree': [p], 'runtime_python_shared': [p]}
            with mock.patch.object(ct, 'inventory_paths', return_value=required):
                good = {k: ct.inventory(v) for k, v in required.items()}
                ct.validate_inventory(ct.BASE, good, runtime=True)
                del good['runtime_python_shared']
                with self.assertRaisesRegex(ValueError, 'Mandatory'):
                    ct.validate_inventory(ct.BASE, good, runtime=True)

    def test_immutable_stage_and_hash_binding(self):
        with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp:
            root = Path(tmp)
            source = root / 'input'
            source.write_bytes(b'not executed')
            staged = Path(ct.stage_binary(source, root))
            pin = ct.sha(staged)
            try:
                self.assertEqual(ct.validate_staged_binary(staged, pin), str(staged))
                with self.assertRaises(FileExistsError):
                    ct.stage_binary(source, root)
                source.write_bytes(b'new mutable build')
                self.assertEqual(ct.sha(staged), pin)
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    ct.validate_staged_binary(staged, 'f'*64)
            finally:
                staged.parent.chmod(0o755)

    def test_staged_python_entrypoint_supports_real_spawn(self):
        with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp:
            root = Path(tmp)
            script = root / 'spawn_fixture'
            script.write_text('#!' + sys.executable + ' -S\n'
                              'import multiprocessing\n'
                              'def child(): print("CPU_SPAWN_OK", flush=True)\n'
                              'if __name__ == "__main__":\n'
                              ' p=multiprocessing.get_context("spawn").Process(target=child)\n'
                              ' p.start(); p.join(5); assert p.exitcode == 0\n')
            staged = Path(ct.stage_binary(script, root))
            try:
                spec = dict(engine='vllm', argv=[str(staged)])
                with ct.pinned_executable(staged, ct.sha(staged)) as fd:
                    argv = runner.bound_server_argv(spec, fd)
                    self.assertEqual(argv, [str(staged)])
                    result = subprocess.run(ct.isolated_argv(argv, runner.cpu_environment()),
                                            capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('CPU_SPAWN_OK', result.stdout)
                    staged.parent.chmod(0o755)
                    with self.assertRaisesRegex(ValueError, 'writable'):
                        runner.bound_server_argv(spec, fd)
            finally:
                staged.parent.chmod(0o755)

    def test_pid_namespace_unknown_and_foreign_app_block(self):
        with mock.patch.object(telemetry.os, 'readlink', return_value='pid:[nested]'):
            with self.assertRaisesRegex(ValueError, 'unprovable'):
                telemetry.app_ownership([{'pid': 444}], 123, {'worker': {}})
        worker = dict(pid=10, birth_ticks=1, ppid=9, pgrp=10, session=10)
        server = dict(pid=11, birth_ticks=2, ppid=10, pgrp=10, session=10)
        foreign = dict(pid=44, birth_ticks=3, ppid=1, pgrp=44, session=44)
        with mock.patch.object(telemetry.os, 'readlink', return_value='pid:[4026531836]'), \
                mock.patch.object(telemetry, 'identity', side_effect=lambda p: {10:worker,11:server,44:foreign}[p]):
            with self.assertRaisesRegex(ValueError, 'Foreign GPU'):
                telemetry.app_ownership([{'pid': 44}], 11, {'worker': worker})

    def test_exclusive_version_is_explicit_and_frozen(self):
        self.assertEqual(ct.identity_mode(ct.config()), 'hostpid_verified')
        self.assertEqual(ct.config(ct.EXCLUSIVE_PATH), ct.EXCLUSIVE)
        ct.validate_schema(ct.EXCLUSIVE)
        for mode in ('hostpid_verified', 'invented'):
            bad = copy.deepcopy(ct.EXCLUSIVE)
            bad['protocol']['identity_mode'] = mode
            with self.assertRaisesRegex(ValueError, 'Unknown/stale'):
                ct.config_value(bad)
        with mock.patch.object(subprocess, 'run', side_effect=AssertionError('no query')):
            self.assertEqual(runner.main(['feasibility', '--config', str(ct.EXCLUSIVE_PATH),
                                         '--identity-mode', 'device_exclusive_local_custody']), 2)
            self.assertEqual(runner.main(['feasibility', '--config', str(ct.EXCLUSIVE_PATH),
                                         '--sole-gpu-owner-granted', '--gpu-uuid', UUID,
                                         '--owner-note', 'MOCK']), 2)
            with self.assertRaisesRegex(ValueError, 'Unknown identity'):
                telemetry.authorize(True, UUID, 'MOCK', 'invented')

    def test_exclusive_unmapped_not_verified_but_local_custody_required(self):
        worker = dict(pid=10, birth_ticks=1, ppid=9, pgrp=10, session=10)
        server = dict(pid=11, birth_ticks=2, ppid=10, pgrp=11, session=11)
        with mock.patch.object(telemetry.os, 'readlink', return_value='pid:[nested]'), \
                mock.patch.object(telemetry, 'identity', side_effect=lambda p: {10:worker,11:server}[p]):
            result = telemetry.app_ownership([{'pid':444}], 11, {'worker':worker}, 'device_exclusive_local_custody')
            self.assertEqual(result['status'], 'unmapped_host_pids')
            self.assertEqual(result['owned_gpu_attribution'], 'unknown')
            self.assertEqual(result['matched'], [])
            self.assertFalse(result['absence_of_foreign_interference_proven'])
            server['ppid'] = 99
            with self.assertRaisesRegex(ValueError, 'direct child'):
                telemetry.app_ownership([], 11, {'worker':worker}, 'device_exclusive_local_custody')

    def test_exclusive_known_foreign_gpu_process_still_blocks(self):
        worker = dict(pid=10, birth_ticks=1, ppid=9, pgrp=10, session=10)
        server = dict(pid=11, birth_ticks=2, ppid=10, pgrp=11, session=11)
        foreign = dict(pid=44, birth_ticks=3, ppid=1, pgrp=44, session=44)
        with mock.patch.object(telemetry.os, 'readlink', return_value='pid:[4026531836]'), \
                mock.patch.object(telemetry, 'identity', side_effect=lambda p: {10:worker,11:server,44:foreign}[p]):
            with self.assertRaisesRegex(ValueError, 'Foreign GPU'):
                telemetry.app_ownership([{'pid':44}], 11, {'worker':worker}, 'device_exclusive_local_custody')

    def test_exclusive_all_device_idle_rejects_other_gpu_activity(self):
        sample = dict(apps=[], utilization=0, gpu_bytes=0, total_bytes=24*(1<<30),
                      devices=[dict(utilization=0,gpu_bytes=0),dict(utilization=1,gpu_bytes=0)])
        with mock.patch.object(telemetry, 'all_gpu_sample', return_value=sample), \
                mock.patch.object(telemetry.time, 'monotonic', side_effect=[0,0,100]), \
                mock.patch.object(telemetry.time, 'sleep'):
            self.assertEqual(telemetry.idle('/mock', UUID, True, 'MOCK', ct.EXCLUSIVE)['status'], 'blocked')

    def test_exclusive_schema_requires_assumption_not_verified_attribution(self):
        report = dict(schema=ct.SCHEMA+'execution.v2', status='incomplete', mode='feasibility',
                      manifest_sha256='a'*64, gpu_uuid=UUID, cases=[], artifact_pins={'/mock':'a'*64},
                      exact_speedup=None, fairmatch=False, fairness_blockers=['partial'],
                      release_quality=None, feasibility=None, observer_effect='baseline_ferrule_log_info_only', owner_lock={})
        with self.assertRaises(ValueError):
            ct.validate_schema(report)
        report.update(identity_mode='device_exclusive_local_custody', protocol_id='m7-device-exclusive-local-custody-v2',
                      sole_gpu_owner_granted=True, performance_basis='exclusive_operator_allocation_assumption',
                      owned_gpu_attribution='unknown', owned_process_gpu_peak_bytes=None,
                      absence_of_foreign_interference_proven=False)
        ct.validate_schema(report)
        for key,value in [('sole_gpu_owner_granted',False),('owned_gpu_attribution','verified'),
                          ('owned_process_gpu_peak_bytes',42),('absence_of_foreign_interference_proven',True),
                          ('exact_speedup',2),('fairmatch',True)]:
            with self.assertRaises(ValueError):
                ct.validate_schema(dict(report, **{key:value}))

    def test_finish_records_inflight_failure(self):
        import threading
        started, release = threading.Event(), threading.Event()
        def query(*args):
            started.set()
            release.wait(2)
            raise TimeoutError('deterministic in-flight failure')
        with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp:
            root = Path(tmp)
            (root / 'server.log').write_text('')
            server = mock.Mock(pid=123)
            server.poll.return_value = None
            monitor = telemetry.Monitor(root, server, '/mock', UUID, 'mock', ct.BASE, {})
            with mock.patch.object(telemetry, 'gpu_sample', side_effect=query):
                monitor.start()
                self.assertTrue(started.wait(2))
                monitor.stop.set()
                release.set()
                result = monitor.finish()
            self.assertEqual(result['status'], 'blocked')
            self.assertIn('deterministic in-flight failure', result['errors'])
            server.terminate.assert_not_called()

    def test_schema_rejects_incomplete_prepared_duplicate_engines_and_unbound_run(self):
        with self.assertRaisesRegex(ValueError, 'schema invalid'):
            ct.validate_schema(dict(schema=ct.SCHEMA+'prepared.v1', status='prepared_cpu_only', binary='/tmp/old'))
        case = dict(engine='ferrule', supervisor=good_process(), after={'status':'verified'},
                    observation={'status':'complete'}, eligible_observation=True, exact_speedup=None)
        report = dict(schema=ct.SCHEMA+'execution.v1', status='complete', mode='feasibility',
                      manifest_sha256='a'*64, gpu_uuid=UUID, cases=[case, copy.deepcopy(case)],
                      artifact_pins={'/mock':'a'*64}, exact_speedup=None, fairmatch=False,
                      fairness_blockers=['partial'], release_quality=None, feasibility=None,
                      observer_effect='baseline_ferrule_log_info_only', owner_lock={})
        with self.assertRaises(ValueError):
            ct.validate_schema(report)
        report['cases'][1]['engine']='vllm'
        ct.validate_schema(report)
        report['mode']='run'
        with self.assertRaises(ValueError):
            ct.validate_schema(report)

    def test_sealed_worker_authentication_under_real_cpu_supervisor(self):
        import launch_guard
        with tempfile.TemporaryDirectory(dir=ct.HERE) as tmp:
            payload, pin = launch_guard.seal({'directory':tmp}, tmp)
            code = ('import sys,json; sys.path.insert(0,' + repr(str(ct.HERE)) + '); '
                    'import launch_guard; p=json.load(sys.stdin); '
                    'launch_guard.authenticate(p,sys.argv[1]); print("AUTHENTICATED_CPU_ONLY")')
            env = dict(runner.cpu_environment(), FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
            argv = ct.isolated_argv([sys.executable, '-S', '-B', '-c', code, pin], env)
            result = runner.supervisor.run_process(argv, 10, payload)
            self.assertTrue(ct.successful_process(result), result)
            self.assertIn('AUTHENTICATED_CPU_ONLY', result['stdout'])
            payload['directory'] = str(Path(tmp) / 'not-the-receipt')
            with self.assertRaises(ValueError):
                launch_guard.authenticate(payload, pin)

    def test_cpu_interpreter_rejects_venv(self):
        with mock.patch.object(sys, 'executable', str(ct.absolute(ct.BASE['vllm']).parent / 'python')):
            with self.assertRaisesRegex(ValueError, 'exact base Python'):
                ct.require_cpu_python()

    @unittest.skipUnless(sys.platform == 'linux', 'Authenticated supervisor requires Linux')
    def test_real_cpu_timeout_is_failure(self):
        argv = ct.isolated_argv([sys.executable, '-I', '-S', '-B', '-c', 'import time; time.sleep(10)'],
                               runner.cpu_environment())
        result = runner.supervisor.run_process(argv, .2)
        self.assertTrue(result['timed_out'], result)
        self.assertFalse(ct.successful_process(result))
        self.assertTrue(ct.drained(result), result)

    def test_gpu_query_failure_never_becomes_idle(self):
        with mock.patch.object(subprocess, 'run', side_effect=subprocess.TimeoutExpired('MOCK_nvidia-smi', 3)):
            with self.assertRaises(subprocess.TimeoutExpired):
                telemetry.idle('/MOCK/nvidia-smi', UUID, True, 'MOCK ONLY', ct.BASE)

    @unittest.skipUnless(sys.platform == 'linux', 'Authenticated supervisor requires Linux')
    def test_real_cpu_child_authenticated_supervision(self):
        argv = ct.isolated_argv([sys.executable, '-I', '-S', '-B', '-c',
                                'import os; assert os.environ["CUDA_VISIBLE_DEVICES"] == ""; print("CPU_ONLY")'],
                               runner.cpu_environment())
        result = runner.supervisor.run_process(argv, 10)
        self.assertTrue(ct.successful_process(result), result)
        self.assertIn('CPU_ONLY', result['stdout'])


if __name__ == '__main__':
    unittest.main()
