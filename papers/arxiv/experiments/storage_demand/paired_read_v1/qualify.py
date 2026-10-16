"""Validation-only serial exact-once gates; maintained authenticated custody/idle."""
import contextlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path('/home/admin/luhuanbing.lhb/Ferrule')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'target/validation/retained-upload-57e4629-gpu-20261016-01'
OUT = ROOT / 'target/validation/retained-profile-57e4629-natural-drain-20261016-01'
PARENT = ROOT / 'target/validation/retained-performance-57e4629-20261016-01'
PYTHON = '/home/admin/wxw403883/anaconda3/bin/python3.12'
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / 'papers/arxiv/experiments'))
sys.path.insert(0, str(ROOT / 'scripts'))
import contract as ct
import launch_guard as guard
import test_vllm_qwen35 as supervisor
import gpu_idle_window_v2 as idle
sys.path.insert(0, str(ROOT / 'papers/arxiv/experiments/storage_demand'))
sys.path.insert(0, str(HERE))
import quality as lifecycle
import retained_evidence
import performance as perf
import profile_audit
PLAN = None


def is_grouped():
    return retained_evidence.grouped_quality(PLAN)


def plan_path():
    return OUT / ('grouped-plan.json' if is_grouped() else 'performance-plan.json')


def payload_argv(gate):
    if not is_grouped():
        return perf.payload_argv(PLAN, gate)
    retained_evidence.grouped_job(PLAN, gate)
    return [gate["executable"], gate["selector"], "--exact", "--ignored", "--nocapture", "--test-threads=1"]


def primary(gate):
    (retained_evidence.grouped_job if is_grouped() else perf.job)(PLAN, gate)
    if perf.profile_only(PLAN):
        binding = next(b for b in PLAN['diagnostic_primaries']
                       if perf.primary_key(b['job']) == perf.primary_key(gate))
        path = Path(binding['report'])
        ct.check_pins({binding['report']: binding['report_sha256'],
                       binding['gate_result']: binding['gate_result_sha256']})
    else:
        source = next(g for g in PLAN['jobs'] if g['phase'] == 'primary'
                      and perf.primary_key(g) == perf.primary_key(gate))
        path = OUT / 'gates' / source['id'] / 'quality.json'
    accepted = ct.load(path.parent / 'gate-result.json')
    data = ct.load(path)
    if is_grouped():
        wanted = next(g for g in PLAN['jobs'] if g['phase'] == 'primary'
                      and perf.primary_key(g) == perf.primary_key(gate))
        ct.require(accepted.get('status') == 'passed' and accepted.get('report_sha256') == ct.sha(path)
            and accepted.get('commit') == PLAN['commit'] and accepted.get('source_digest') == PLAN['source_digest']
            and accepted.get('binary_sha256') == wanted['binary_sha256']
            and ct.digest(accepted.get('job')) == ct.digest(wanted), 'Grouped primary failed/foreign policy/schedule/source/hash')
        validate_composition(data, wanted)
    else:
        perf.validate_primary(accepted, data, gate, PLAN, ct.sha(path))
    return path, ct.sha(path), data


def environment(gate):
    directory = OUT / 'gates' / gate['id']
    env = dict(PLAN['runtime_base_environment'])
    if gate['kind'] == 'model_contract':
        pass
    else:
        env.update(FERRULE_NUMERIC_BF16_FACTORY_GPU='1',
            FERRULE_NUMERIC_BF16_MODEL='/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8',
            FERRULE_NUMERIC_BF16_REFERENCE_DIR=str(ROOT / 'target/validation/numeric-bf16-gate-20261004-v2/profile-reference'),
            FERRULE_NUMERIC_BF16_REFERENCE_SHA256=PLAN['full40_reference_sha256'])
        if gate['kind'] == 'full40':
            env['FERRULE_NUMERIC_BF16_REPORT_DIR'] = str(directory / 'numeric-report')
        else:
            env.update(FERRULE_BF16_FRESH_DIR=str(ROOT / 'target/validation/bf16-fresh-quality-20261007'),
                FERRULE_BF16_FRESH_SHA256=PLAN['fresh_reference_sha256'],
                FERRULE_BF16_FRESH_REPORT=str(directory / 'quality.json'))
            if gate['kind'] in ('ordinary', 'cold'):
                path, pin, data = primary(gate)
                env['FERRULE_BF16_FRESH_PRIMARY_REPORT'] = str(path)
                if gate['kind'] == 'ordinary':
                    env.update(FERRULE_BF16_FRESH_OBSERVED_REPORT=str(path), FERRULE_BF16_FRESH_OBSERVED_REPORT_SHA256=pin)
    ct.require(env['PATH'] == '/usr/bin:/bin' and env['HOME'] == '/root'
        and env['CUDA_VISIBLE_DEVICES'] == env['NVIDIA_VISIBLE_DEVICES'] == PLAN['uuid_inventory'][0]
        and env['TMPDIR'] == '/tmp'
        and env['CUBLAS_WORKSPACE_CONFIG'] == ':0:0' and env['FERRULE_PROFILE_NVTX'] == '0'
        and env['LD_LIBRARY_PATH'] == '/home/admin/wxw403883/anaconda3/envs/tao/lib:/usr/lib64'
        and not any(k in env for k in ['CUDA_DEVICE_ORDER', 'CUDA_HOME', 'FERRULE_CUDA_ARCH', 'LD_PRELOAD']), 'Pinned runtime changed')
    return env


def verify(full=False):
    ct.check_pins(PLAN['pins'])
    (retained_evidence.validate_grouped_plan if is_grouped() else perf.validate_plan)(PLAN)
    if perf.profile_only(PLAN):
        perf.validate_completed_performance(PLAN, ct.load(Path(PLAN['completed_performance_source']['directory']) / 'FINAL.json'))
        primaries = [primary(gate)[2] for gate in PLAN['jobs']]
        if perf.grouped_profile_only(PLAN):
            for schedule in dict.fromkeys(gate['schedule'] for gate in PLAN['jobs']):
                pair = [data for gate, data in zip(PLAN['jobs'], primaries) if gate['schedule'] == schedule]
                perf.compare_work(*pair, require_fingerprints=True)
    baseline = ROOT / 'target/validation/retained-upload-57e4629-gpu-20261016-01' if is_grouped() or perf.experimental_arm(PLAN) == 'grouped' else SOURCE
    original = ct.load(baseline / 'retained-plan.json')
    ct.require(PLAN['capacity_and_options'] == original['composition_capacity_and_options']
        and PLAN['uuid_inventory'] == original['uuid_inventory']
        and PLAN['runtime_base_environment'] == original['runtime_base_environment']
        and PLAN['fresh_reference_sha256'] == original['fresh_reference_sha256']
        and PLAN['full40_reference_sha256'] == original['full40_reference_sha256'], 'Frozen source/quota/environment/reference differs')
    if perf.experimental_arm(PLAN) == 'grouped' and not is_grouped():
        proof = ct.load(retained_evidence.GROUPED_STAGE / 'FINAL.json')
        ct.require(proof['status'] == 'passed' and proof['actual_gpu_tests_issued'] == 14
            and proof['commit'] == PLAN['commit'] and proof['source_digest'] == PLAN['source_digest'],
            'Completed frozen quality14 prerequisite failed/foreign')
        runtime = ct.load(OUT / 'profile-runtime-policy.json')
        ct.require(runtime['executable'] == PLAN['executable'] and runtime['executable_sha256'] == PLAN['binary_sha256'],
                   'Profile runtime bound to a foreign binary')
        ct.check_pins(runtime['pins'])
    source = ct.load(SOURCE / 'source-freeze.json')
    ct.require(source['commit'] == PLAN['commit'] and source['source_pins_digest'] == PLAN['source_digest'], 'Wrong frozen cut')
    ct.check_pins({str(Path(source['frozen_source']) / p): h for p, h in source['pins_relative_to_frozen_source'].items()})
    refs = ct.load(SOURCE / 'reference-pins.json')
    ct.check_pins(refs['reference_files']); ct.check_pins(refs['model_metadata'])
    for name, expected in ct.load(SOURCE / 'model-shard-identities.json').items():
        st = Path(name).stat()
        ct.require([st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_dev, st.st_ino] == expected, 'Shard identity drift: ' + name)
    ct.check_pins(ct.load(SOURCE / 'runtime-library-pins.json')['pins'])
    if full:
        ct.check_pins(ct.load(SOURCE / 'external-input-pins.json')['pins'])
        ct.check_pins(ct.load(SOURCE / 'compiled-dependency-pins.json')['pins'])
        ct.check_pins(ct.load(SOURCE / 'native-archive-pins.json')['immutable_copies'])
    for gate in PLAN['gates']:
        ct.require(ct.sha(gate['executable']) == gate['binary_sha256'] and Path(gate['executable']).stat().st_mode & 0o222 == 0, 'Executable drift')


def list_one(gate, directory):
    env = dict(PLAN['runtime_base_environment'], CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
    argv = [gate['executable'], gate['selector'], '--list', '--ignored', '--exact']
    spec = dict(argv=argv, environment=env, timeout_seconds=30, gpu_hidden=True)
    ct.write(directory / 'exact-list-command.json', spec)
    started = time.monotonic()
    result = subprocess.run(ct.isolated_argv(argv, env), cwd=ROOT, capture_output=True, text=True, timeout=30)
    for name in ('stdout', 'stderr'):
        with (directory / ('exact-list.' + name)).open('x') as stream: stream.write(getattr(result, name))
    receipt = dict(**spec, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, wall_seconds=time.monotonic()-started)
    ct.write(directory / 'exact-list-receipt.json', receipt)
    ct.require(result.returncode == 0 and not result.stderr and [s.strip() for s in result.stdout.splitlines() if s.strip()]
        == [gate['selector'] + ': test', '1 test, 0 benchmarks'], 'Exact ignored selector must select one test')


def worker(seal):
    ct.require_cpu_python()
    payload = json.load(sys.stdin)
    authenticated = guard.authenticate(payload, seal)
    require_independent_review()
    ct.require(ct.sha(plan_path()) == payload['qualification_plan_sha256'], 'Plan drift')
    gate = next(g for g in PLAN['gates'] if g['id'] == payload['gate_id'])
    directory = Path(payload['directory'])
    ct.require(directory == OUT / 'gates' / gate['id'] and payload['gate'] == gate and payload['environment'] == environment(gate), 'Gate/environment drift')
    ct.require(payload['seconds'] == gate['seconds'] == 1200 and os.getpid() == os.getpgrp(), 'Wrong deadline/session')
    verify()
    ct.require(len(payload['all_uuid_locks']) == 8, 'All eight locks required')
    for uuid, lock in zip(PLAN['uuid_inventory'], payload['all_uuid_locks']): ct.validate_owner_lock(lock, uuid, payload['coordinator']['pid'])
    pre = idle.observe(PLAN['uuid_inventory'])
    ct.write(directory / 'authenticated-pre-cuda-idle.json', pre)
    ct.require(pre['status'] == 'verified', 'Authenticated pre-CUDA strict all-eight idle failed')
    argv = payload_argv(gate)
    if gate['phase'] == 'profile':
        target = [PYTHON, '-S', '-B', str(SCRIPT), '--profile-target', seal,
                          '--directory', str(OUT), '--gate-id', gate['id']]
        argv = perf.profile_argv(PLAN, gate, directory, target)
    ct.write(directory / 'authenticated-launch.json', dict(identity=authenticated, argv=argv, profiled=gate['phase'] == 'profile', environment=payload['environment'],
        gate_id=gate['id'], executable_sha256=ct.sha(gate['executable']), all_eight_uuid_locks_validated=True,
        exec_preserves_authenticated_root_pid_birth=True, seconds=1200, no_probe_no_warmup_no_retry=True))
    stdout = os.open(directory / 'test.stdout', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stderr = os.open(directory / 'test.stderr', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.dup2(stdout, 1); os.dup2(stderr, 2); os.close(stdout); os.close(stderr)
    os.execve(argv[0], argv, payload['environment'])


def monitor(directory, stop, observations):
    """Read-only authenticated PID/birth mappings/RSS; no signal or CUDA call."""
    seen = set()
    while not stop.wait(.25):
        try:
            launch = ct.load(directory / 'authenticated-launch.json')
            if launch.get('profiled'):
                return profile_monitor(directory, stop, observations)
            root = launch['identity']['worker']
            now = guard.identity(root['pid'])
            if now['birth_ticks'] != root['birth_ticks']: continue
            maps = Path(f'/proc/{root["pid"]}/maps').read_text()
            status = Path(f'/proc/{root["pid"]}/status').read_text()
            key = maps
            if key in seen: continue
            seen.add(key)
            observations.append(dict(unix_ns=time.time_ns(), identity=now, maps=maps, status=status, authority='read-only authenticated root PID/birth; not CUDA fence or certified RSS'))
        except FileNotFoundError: pass
        except Exception as error: observations.append(dict(error=type(error).__name__ + ': ' + str(error)))


def validate_constructor(report):
    lifecycle.constructor_v3(report)


def validate_config(report, gate):
    (retained_evidence.grouped_job if is_grouped() else perf.job)(PLAN, gate)
    if is_grouped():
        conf = report['test_config']
        ct.require(conf['capacity_and_options'] == PLAN['composition_capacity_and_options']
            and conf['numeric_precision'] == 'Bf16ModelMixedF32Accumulate'
            and conf['expert_read_pipeline'] == dict(schedule=gate['schedule'].capitalize(),
                max_host_bytes=64 << 20, max_metadata_bytes=8 << 20, worker_stack_bytes=2 << 20,
                metadata_and_stack_included_in_host_cap=True, device_budget_separate=True), 'Grouped fixed config differs')
        retained_evidence.validate_grouped(report, gate['schedule'])
    else:
        perf.policy(report, gate['transfer_mode'], gate['schedule'], PLAN['capacity_and_options'])


def validate_composition(report, gate):
    ct.require(report['complete'] is True and report['passed'] is True and not report['failures'], 'Numerical gate incomplete/failed')
    validate_config(report, gate)
    ct.require(report['atol'] == .00025 and report['rtol'] == .012 and report['state_bytes_after_shutdown'] == 0
        and report['quarantine'] is False and report['device_budget_bytes'] == 7<<30, 'Tolerance or cleanup/budget drift')
    ct.require(all(m.get('failures', 0) == 0 and m.get('nonfinite', 0) == 0 for m in report['metrics']), 'Metric failure/nonfinite')
    if gate['kind'] == 'full40':
        ct.require(report['reference_manifest_sha256'] == PLAN['full40_reference_sha256'] and len(report['metrics']) == 804
            and report['logits_compared'] == 1986560 and len(report['next_token_checks']) == 4
            and all(c['equal'] is True for c in report['next_token_checks']), 'Full40 numerical/greedy denominator drift')
        ct.require(report['actual_peak_reserved_bytes'] <= report['device_budget_bytes'], 'Device cap exceeded')
    else:
        cold = gate['kind'] == 'cold'; diagnostic = gate['kind'] != 'ordinary'; cases = 1 if cold else 3
        ct.require(report['oracle_manifest_sha256'] == PLAN['fresh_reference_sha256'] and report['cold_owner_restart'] is cold
            and len(report['cases']) == cases and report['peak_reserved_bytes'] <= report['device_budget_bytes'], 'Fresh config/mode/case/budget mismatch')
        ct.require(report['observer_mode'] == ('diagnostic_routes_and_states' if diagnostic else 'ordinary_no_diagnostic_observer')
            and report['ordinary_no_observer_claim'] is (not diagnostic), 'Observer scope differs')
        for case in report['cases']:
            ct.require(case['free_greedy_equal'] is True and not case['top1_failures']
                and case['free_greedy_ids'] == case['oracle_generated_ids'] == case['common_history_generated_ids']
                and len(case['next_token_checks']) == len(case['free_greedy_next_token_checks']) == 9, 'Fresh greedy/history mismatch')
            if case.get('reset_reused_state'): ct.require(case['reset_equal'] is True, 'Reset mismatch')
        if diagnostic:
            ct.require(len(report['metrics']) == (1809 if cold else 5427) and report['logits_compared'] == (3228160 if cold else 9187840), 'Diagnostic denominator drift')
        calls = report['prediction_fingerprints']['calls']
        ct.require(len(calls) == (28 if cold else 64) and not report['prediction_fingerprints']['failures'], 'Actual fingerprint denominator mismatch')
        streams = report['prediction_denominators']['streams']
        ct.require(set(streams) == {'common_history', 'free_feedback', 'reset_replay', 'fork_replay'}, 'Prediction stream denominator missing/extra')
        if not diagnostic:
            ct.require(len(report['metrics']) == 27 and report['logits_compared'] == 9187840, 'Ordinary denominator drift')
        for name, stream in streams.items():
            count = cases*9 if name in ('common_history', 'free_feedback') else 9 if name == 'reset_replay' else 1
            ct.require(stream['prediction_calls'] == stream['greedy_checks'] == stream['input_history_matched_reference_calls'] == count
                and stream['logit_values_compared'] > 0 and not stream['failures'], 'Per-stream denominator/greedy/reference mismatch')
        if gate['kind'] in ('ordinary', 'cold'):
            path, pin, prior = primary(gate)
            ct.require(report['test_config'] == prior['test_config'] and report['effective_config'] == prior['effective_config'], 'Current primary configuration mismatch')
            if gate['kind'] == 'ordinary':
                ct.require(report['diagnostic_primary_report_sha256'] == pin and calls == prior['prediction_fingerprints']['calls']
                    and report['prediction_fingerprints']['paired_exact_logits_and_greedy_equal'] is True, 'Actual diagnostic vs ordinary logits/shape/greedy fingerprints differ')


def report_summary(path):
    if not path.exists(): return dict(report_present=False, numerical_denominators='unobserved; no report exists')
    report = ct.load(path)
    return dict(report_present=True, report=str(path), report_sha256=ct.sha(path), report_complete=report.get('complete'), report_passed=report.get('passed'),
        observed_constructor_cells=len(report.get('cells', [])), observed_constructor_logit_values=report.get('total_logit_values_checked', 0),
        observed_metrics=len(report.get('metrics', [])), observed_logits=report.get('logits_compared', 0), failures=report.get('failures', []),
        diagnostic_denominators=report.get('diagnostic_denominators'), prediction_denominators=report.get('prediction_denominators'),
        fingerprint_calls=len(report.get('prediction_fingerprints', {}).get('calls', {})), expert_read_quality=report.get('expert_read_quality'),
        physical_resource_observations='source-qualified typed JSON endpoints/cells when present; no tracing subscriber required', physical_release='unobserved; ECHILD/NVML idle not physical CUDA release')


def execute_gate(gate, locks):
    directory = OUT / 'gates' / gate['id']; directory.mkdir()
    report_path = directory / ('numeric-report/full40-bf16-profile.json' if gate['kind'] == 'full40' else 'quality.json')
    result = dict(gate_id=gate['id'], selector=gate['selector'], status='blocked', dispatched=0, actual_test_issued=False, retries=0,
        job=gate, commit=PLAN['commit'], binary_sha256=gate['binary_sha256'], source_digest=PLAN['source_digest'], expected_exact_tests=1, expected_constructor_cells=4 if gate['kind']=='constructor' else None,
        physical_release='unobserved; no fence/release certificate', numerical_mismatch=False)
    custody = {}; post = None
    try:
        verify(); list_one(gate, directory)
        env = environment(gate)
        ct.write(directory / 'runtime-environment.json', env)
        pre = idle.observe(PLAN['uuid_inventory']); ct.write(directory / 'pre-idle.json', pre)
        ct.require(pre['status'] == 'verified', 'Pre-gate all-eight strict idle failed')
        payload = dict(directory=str(directory), gate_id=gate['id'], gate=gate, seconds=1200, environment=env,
            qualification_plan_sha256=ct.sha(plan_path()), all_uuid_locks=locks, granted=True,
            owner_note=('Separately reviewed grouped quality14 authority; only this sealed job, no retries/probes/warmups/profiling' if is_grouped() else
                'Separately reviewed same-ce6-cut synchronous/grouped fixed24; no retries/probes/warmups; only four declared profiles' if perf.grouped_comparison(PLAN) else
                'Separate reviewed exact profile-only authority; completed primaries reused, no timing/quality reruns or retries/probes/warmups' if perf.profile_only(PLAN) else
                'Separate reviewed fixed24 authority, only this sealed job, no retries/probes/warmups; profiling only for four declared captures'))
        payload, seal = guard.seal(payload, directory)
        cpu_env = dict(PATH='/usr/bin:/bin', HOME='/root', CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', PYTHONDONTWRITEBYTECODE='1', FERRULE_PAPER_WORKER_TOKEN=payload['worker_token'])
        argv = ct.isolated_argv([PYTHON, '-S', '-B', str(SCRIPT), '--worker', seal, '--directory', str(OUT)], cpu_env)
        ct.write(directory / 'dispatch.json', dict(argv=argv, gate=gate, seconds=1200, retries=0, probes=0, warmups=0))
        result['dispatched'] = 1
        observations = []; stop = threading.Event(); thread = threading.Thread(target=monitor, args=(directory,stop,observations), daemon=True); thread.start()
        print('START', gate['id'], 'exactly once; timeout1200s; phase=' + gate['phase'], flush=True)
        try:
            custody = supervisor.run_process(argv, 1200, payload)
        except BaseException as error:
            custody = getattr(error, 'supervisor_result', {}); custody.setdefault('error', type(error).__name__ + ': ' + str(error))
        finally:
            stop.set(); thread.join(timeout=5)
            if thread.is_alive(): observations.append(dict(error='Profile/mapping monitor did not stop'))
            ct.write(directory / 'runtime-mapping-observations.json', dict(samples=observations, sample_count=len(observations), actual_loaded_mappings_claim=bool(observations), cadence_seconds=.25, no_cuda_or_signal_authority=True))
        ct.write(directory / 'supervisor.json', custody)
        result.update(actual_test_issued=(directory/('profile-model-start.json' if gate['phase'] == 'profile' else 'authenticated-launch.json')).exists(), returncode=custody.get('returncode'), timed_out=custody.get('timed_out'),
            authenticated_drained=ct.drained(custody), echild=custody.get('custody',{}).get('echild'), recorded_owners_absent=idle.no_recorded_owner(custody),
            natural_exit=all(not row.get('signals') for row in custody.get('custody',{}).get('entries',[])))
        for uuid, lock in zip(PLAN['uuid_inventory'],locks): ct.validate_owner_lock(lock,uuid,os.getpid())
        post = idle.observe(PLAN['uuid_inventory'], custody=custody); ct.write(directory / 'post-idle.json',post)
        result.update(post_idle_status=post['status'],post_idle_samples=len(post['samples']),post_idle_accepted_sample_indices=post.get('accepted_sample_indices'))
        verify()
        ct.require(post['status']=='verified', 'Post-ECHILD bounded stable all-eight idle failed')
        stdout=(directory/'test.stdout').read_text() if (directory/'test.stdout').exists() else ''
        stderr=(directory/'test.stderr').read_text() if (directory/'test.stderr').exists() else custody.get('stderr','')
        result['failure_classification'] = 'CUDA_driver_initialization_cuInit_3_not_numeric_mismatch' if 'cuInit failed: 3' in stderr or re.search(r'cuInit[^\n]*\b3\b',stderr) else None
        ct.require(ct.successful_process(custody) and result['natural_exit'], 'Test failure/timeout/unknown drain; inspect preserved stderr/report')
        ct.require('running 1 test' in stdout and re.search(r'test result: ok\. 1 passed; 0 failed; 0 ignored;',stdout), 'One actual unignored PASS required')
        if gate['kind'] == 'model_contract':
            result.update(status='passed',model_contract_passed=True)
        else:
            report = ct.load(report_path)
            validate_composition(report,gate)
            if is_grouped():
                if gate['kind'] == 'ordinary':
                    _, _, prior = primary(gate)
                    ct.write(directory / 'same-work-own-primary.json', perf.compare_work(prior, report))
                result.update(status='passed', numerical_pass=True, latency_population=False,
                    grouped_facade_sync_count=dict(state='unobserved', reason='not exported by frozen public stats; no zero/fallback inference'))
            else:
                timing = perf.libtest_elapsed(stdout)
                perf.work_vector(report)
                if gate['phase'] != 'primary':
                    _, _, prior = primary(gate)
                    ct.write(directory / 'same-work-own-primary.json', perf.compare_work(prior, report,
                        require_fingerprints=perf.experimental_arm(PLAN) == 'grouped'))
                opposite = next(g for g in PLAN['jobs'] if g['phase'] == gate['phase']
                    and g['schedule'] == gate['schedule'] and g['transfer_mode'] != gate['transfer_mode']
                    and g['repetition'] == gate['repetition'])
                other = OUT / 'gates' / opposite['id']
                if (other / 'gate-result.json').exists():
                    accepted = ct.load(other / 'gate-result.json')
                    ct.require(accepted['status'] == 'passed' and accepted['report_sha256'] == ct.sha(other / 'quality.json'), 'Opposite arm failed/changed')
                    ct.write(directory / 'same-work-cross-policy.json', perf.compare_work(ct.load(other / 'quality.json'), report,
                        require_fingerprints=perf.experimental_arm(PLAN) == 'grouped'))
                if gate['phase'] == 'profile':
                    result['profile'] = export_profile(directory, gate, custody, observations)
                result.update(status='passed', numerical_pass=True, whole_test_elapsed=timing,
                              latency_population=gate['phase'] == 'latency')
    except BaseException as error:
        result.update(status='failed' if result['actual_test_issued'] else 'blocked', error=type(error).__name__+': '+str(error), numerical_pass=False)
    finally:
        try: result.update(report_summary(report_path))
        except Exception as error: result['report_read_error']=type(error).__name__+': '+str(error)
        result['log_pins']={str(p):ct.sha(p) for p in directory.glob('test.*') if p.is_file()}
        ct.write(directory/'gate-result.json',result)
        print(json.dumps({k:result.get(k) for k in ['gate_id','status','actual_test_issued','returncode','authenticated_drained','echild','recorded_owners_absent','post_idle_status','observed_constructor_cells','observed_metrics','observed_logits','failure_classification','error']}),flush=True)
    return result,custody,post


def require_independent_review():
    path = OUT / 'independent-review.json'
    ct.require(path.is_file() and not path.is_symlink() and path.stat().st_mode & 0o222 == 0,
               'Independent review/GO missing; no GPU locks, telemetry or launch permitted')
    review = ct.load(path)
    ct.require(review.get('schema') == ('ferrule.worker-reuse.control-review.v1' if is_grouped() else
            'ferrule.worker-reuse.performance-control-review.v1' if perf.grouped_comparison(PLAN) else
            'ferrule.worker-reuse.grouped-lookahead-profile-only-control-review.v1' if perf.lookahead_profile_only(PLAN) else
            'ferrule.worker-reuse.grouped-profile-only-control-review.v1' if perf.grouped_profile_only(PLAN) else
            'ferrule.worker-reuse.profile-only-control-review.v1')
        and review.get('status') == 'passed' and review.get('gpu_execution_authorized') is True
        and isinstance(review.get('reviewer'), str) and bool(review['reviewer'].strip())
        and review.get('commit') == PLAN['commit']
        and review.get('qualification_plan_sha256') == ct.sha(plan_path())
        and review.get('control_sha256') == ct.sha(SCRIPT)
        and ct.digest(review.get('jobs')) == ct.digest(retained_evidence.grouped_jobs() if is_grouped() else perf.planned_jobs(PLAN))
        and review.get('review_packet_sha256') == ct.sha(OUT / 'review-ready.json'),
        'Independent review/GO must bind this exact frozen cut, plan and control')
    if perf.grouped_profile_only(PLAN):
        ct.require(review.get('reviewer_id') == '089b'
            and review.get('decision') == ('GO_EXACT2_SAME_CE6_LOOKAHEAD_GROUPED_THEN_SYNCHRONOUS_PROFILE_ONLY'
                if perf.lookahead_profile_only(PLAN) else 'GO_EXACT4_SAME_CE6_SYNCHRONOUS_VERSUS_GROUPED_PROFILE_ONLY'),
            'Fresh grouped profile-only scope requires independent089b exact-scope GO')


def main():
    require_independent_review()
    ct.require_cpu_python()
    ct.require(not (OUT/'FINAL.json').exists(), 'Qualification is exact-once; final receipt already exists')
    ct.require((OUT / 'gates').is_dir() and not any((OUT / 'gates').iterdir()),
               'Fresh empty exact-once gate directory required; no resume/retry')
    final=dict(schema='ferrule.grouped-upload.current-quality-qualification.v1' if is_grouped() else
        'ferrule.grouped-upload.performance-qualification.v1' if perf.grouped_comparison(PLAN) else
        'ferrule.grouped-upload.lookahead-profile-only-execution.v1' if perf.lookahead_profile_only(PLAN) else
        'ferrule.grouped-upload.profile-only-execution.v1' if perf.grouped_profile_only(PLAN) else
        'ferrule.retained-upload.profile-only-execution.v1',commit=PLAN['commit'],source_digest=PLAN['source_digest'],scheduled_gates=len(PLAN['jobs']),
        source_archive_sha256=ct.sha(SOURCE/'source.tar'),qualification_plan_sha256=ct.sha(plan_path()),
        status='blocked',results=[],actual_gpu_tests_issued=0,retries=0,conditioning_probes=0,warmups=0,profiling_captures=0,
        artifact_sha256=retained_evidence.GROUPED_BINARIES if is_grouped() else {'composition': PLAN['binary_sha256']},binary_math_proof=False,
        profile_evidence_scope=None if is_grouped() else profile_audit.model_summary_policy(),
        relationship='Frozen ce6 grouped quality14; no performance/default/overlap claim' if is_grouped() else
        'Same-ce6-cut synchronous/grouped fixed24; whole-test elapsed only; four separate partial observed-model profiles; no default/token/TTFT/request/overlap claim' if perf.grouped_comparison(PLAN) else f'Fresh {len(PLAN["jobs"])} profile-only captures; historical stages unchanged; completed20 never rerun; no unprofiled latency population, default switch, TTFT/token/request latency or overlap claim',
        physical_release='unobserved; authenticated ECHILD and observed NVML idle are not a CUDA fence or physical-release certificate')
    last_custody=None; locks=[]
    try:
        verify(full=True)
        ct.require(ct.sha(SCRIPT)==PLAN['pins'][str(SCRIPT)],'Adapter drift')
        with contextlib.ExitStack() as stack:
            for uuid in PLAN['uuid_inventory']:
                path=ct.HERE/(uuid+'.lock'); ct.require(path.is_file() and not path.is_symlink(),'Existing lock required')
                locks.append(stack.enter_context(ct.exclusive_owner_lock(uuid)))
            for uuid,lock in zip(PLAN['uuid_inventory'],locks): ct.validate_owner_lock(lock,uuid,os.getpid())
            ct.write(OUT/'uuid-locks-held.json',dict(locks=locks,coordinator=guard.identity(os.getpid()),all_eight=True,local_only=True,global_reservation=False))
            for gate in PLAN['gates']:
                result,custody,post=execute_gate(gate,locks); final['results'].append(result)
                if custody: last_custody=custody
                final['actual_gpu_tests_issued']+=int(result['actual_test_issued'])
                final['profiling_captures']+=int(result['actual_test_issued'] and gate['phase'] == 'profile')
                if result['status']!='passed':
                    final.update(status='failed' if result['actual_test_issued'] else 'blocked',stop_gate=gate['id'],stop_reason=result.get('failure_classification') or result.get('error'));break
            else:
                final.update(status='passed', **({'all14_quality_completed': True} if is_grouped() else
                    {'all24_completed': True, 'paired_whole_test_summary': perf.pair_summary(final['results'], 'grouped')}
                    if perf.grouped_comparison(PLAN) else
                    {'all2_profiles_completed': True} if perf.lookahead_profile_only(PLAN) else {'all4_profiles_completed': True}))
            final['not_issued_gates']=[g['id'] for g in PLAN['gates'] if g['id'] not in {r['gate_id'] for r in final['results'] if r['actual_test_issued']}]
            final_idle=idle.observe(PLAN['uuid_inventory'],custody=last_custody) if last_custody else idle.observe(PLAN['uuid_inventory'])
            ct.write(OUT/'final-idle.json',final_idle)
            final.update(final_idle_status=final_idle['status'],final_idle_samples=len(final_idle['samples']),
                final_authenticated_echild=ct.drained(last_custody) if last_custody else None,
                final_recorded_owners_absent=idle.no_recorded_owner(last_custody) if last_custody else None)
            ct.require(final_idle['status']=='verified','Final idle failed')
        final['all_eight_lock_contexts_exited']=True
        ct.write(OUT/'final-lock-release.json',dict(locks=locks,context_exited=True,release_scope='Only this coordinator local flocks; no global GPU reservation or release proof'))
        with contextlib.ExitStack() as stack:
            probed=[stack.enter_context(ct.exclusive_owner_lock(uuid)) for uuid in PLAN['uuid_inventory']]
            for uuid,lock in zip(PLAN['uuid_inventory'],probed):ct.validate_owner_lock(lock,uuid,os.getpid())
            ct.write(OUT/'final-lock-probe.json',dict(locks=probed,all_eight_nonblocking_reacquired=True,context_exits_after_receipt=True))
        final['all_eight_local_locks_released_and_reprobed']=True
        verify(full=True)
    except BaseException as error: final.update(status='failed' if final['actual_gpu_tests_issued'] else 'blocked',coordinator_error=type(error).__name__+': '+str(error))
    finally:
        final['git_status_final']=subprocess.check_output(['git','--no-pager','--no-optional-locks','status','--short'],cwd=ROOT,text=True)
        ct.write(OUT/'FINAL.json',final)
        print('FINAL',json.dumps({k:final.get(k) for k in ['status','actual_gpu_tests_issued','stop_gate','stop_reason','final_idle_status','final_authenticated_echild','final_recorded_owners_absent','all_eight_local_locks_released_and_reprobed','coordinator_error']}),flush=True)
    return 0 if final['status']=='passed' else 1

def configure(directory):
    global OUT, PLAN, SOURCE
    wanted = Path(directory).resolve()
    grouped = wanted == retained_evidence.GROUPED_STAGE
    comparison = wanted == perf.GROUPED_STAGE
    lookahead_profile = wanted == perf.LOOKAHEAD_PROFILE_STAGE
    grouped_profile = wanted == perf.GROUPED_PROFILE_STAGE or lookahead_profile
    ct.require(grouped or comparison or grouped_profile or wanted == ROOT / 'target/validation/retained-profile-57e4629-natural-drain-20261016-01',
               'Only the fresh separately sealed frozen quality/profile stage is admitted')
    plan = ct.load(wanted / ('grouped-plan.json' if grouped else 'performance-plan.json'))
    if grouped:
        retained_evidence.validate_grouped_plan(plan)
    else:
        perf.validate_plan(plan)
        ct.require(perf.grouped_comparison(plan) if comparison else
            perf.lookahead_profile_only(plan) if lookahead_profile else
            perf.grouped_profile_only(plan) and not perf.lookahead_profile_only(plan) if grouped_profile else
            perf.profile_only(plan) and not perf.grouped_profile_only(plan),
                   'Exact fresh quality/comparison/profile scope required; historical24 cannot execute')
    OUT, PLAN = wanted, plan
    SOURCE = wanted if grouped else retained_evidence.GROUPED_STAGE if comparison or grouped_profile else ROOT / 'target/validation/retained-upload-57e4629-gpu-20261016-01'
    PLAN['gates'] = PLAN['jobs']


def profile_modules():
    # Private collector instances already bind the maintained proc observer and
    # descendant verifier. No new ownership, signaling or reap engine is added.
    from profile_overlap_v1 import tp1_collector
    from runnerhelpers import library_maps_v1
    return tp1_collector.live, library_maps_v1


def profile_auth(directory, seal):
    live, _ = profile_modules()
    launch_path = directory / 'worker-launch.json'
    payload = live.sealed(launch_path, seal)
    require_independent_review()
    gate = perf.job(PLAN, payload['gate'])
    ct.require(gate['phase'] == 'profile' and gate['id'] == payload['gate_id']
        and directory == OUT / 'gates' / gate['id']
        and payload['qualification_plan_sha256'] == ct.sha(plan_path())
        and payload['environment'] == environment(gate), 'Foreign profile job/environment/plan')
    launch = ct.load(directory / 'authenticated-launch.json')
    auth = launch['identity']
    for name in ('worker', 'helper', 'coordinator'):
        ct.require(guard.identity(auth[name]['pid']) == auth[name], 'Live authenticated profile ancestry changed')
    journal = ct.load(directory / 'custody.json')
    ct.require(journal['containment_verified'] is True and journal['method'] == 'linux-subreaper-waitid'
        and journal['launch_id'] == auth['launch_id'] and journal['helper_identity'] == auth['helper']
        and journal.get('root_session_ready') is True and journal.get('status') == 'tracking'
        and journal.get('echild') is False and journal.get('errors') == [] and journal.get('reporting_errors') == []
        and journal['root_identity'] == auth['worker']
        and auth['worker']['pgrp'] == auth['worker']['session'] == auth['worker']['pid'],
        'Profile root lost exact authenticated post-setsid subreaper custody')
    for uuid, lock in zip(PLAN['uuid_inventory'], payload['all_uuid_locks']):
        ct.validate_owner_lock(lock, uuid, auth['coordinator']['pid'])
    ct.require(len(payload['all_uuid_locks']) == 8, 'Profile requires all eight exact UUID locks')
    ct.require(ct.sha(f'/proc/{auth["worker"]["pid"]}/exe') == PLAN['nsys_sha256'],
               'Authenticated root is not the pinned official Nsight executable')
    return live, gate, auth


# One in-flight pin-validation pass; never extends the capture lifetime.
PROFILE_TERMINAL_ACK_SECONDS = 15
PROFILE_PROC_TERMINAL_SECONDS = .25


def profile_binding(directory, start):
    return dict(model={k: start['model']['before'][k] for k in ('pid', 'birth_ticks', 'ppid')},
                **{k: start[k] for k in ('wrapper', 'authenticated', 'job', 'plan_sha256', 'launch_sha256',
                    'capture_started_monotonic', 'capture_deadline_monotonic', 'model_observed_monotonic')},
                start_sha256=ct.sha(directory / 'profile-model-start.json'))


def profile_context(directory):
    live, _ = profile_modules()
    start = ct.load(directory / 'profile-model-start.json')
    _, gate, auth = profile_auth(directory, start['launch_sha256'])
    ct.require(start['plan_sha256'] == ct.sha(plan_path()) and start['job'] == gate
        and start['authenticated'] == auth, 'Profile role source/job/root binding differs')
    live.descendant(start['wrapper'], auth['worker'])
    ct.require(start['capture_started_monotonic'] <= start['model_observed_monotonic']
        < start['capture_deadline_monotonic'], 'Invalid profile startup/deadline binding')
    return live, start, profile_binding(directory, start)


def profile_validate_terminal(receipt, binding):
    ct.require(receipt['binding'] == binding, 'Foreign/stale profile terminal binding')
    event, deadline = receipt['waitid'], receipt['deadline_monotonic']
    times = [event[k] for k in ('started_monotonic', 'observed_monotonic')]
    ct.require(all(type(t) in (int, float) and math.isfinite(t) for t in [*times, deadline])
        and times[0] == binding['capture_started_monotonic'] and times[0] <= times[1]
        and binding['model_observed_monotonic'] <= times[1]
        and deadline == min(binding['capture_deadline_monotonic'], times[1] + PROFILE_TERMINAL_ACK_SECONDS),
        'Invalid profile terminal timing')
    ct.require(all(type(event[k]) is int for k in ('pid', 'code', 'status'))
        and event['pid'] == binding['model']['pid'] and event['code'] == os.CLD_EXITED
        and event['status'] == 0, 'Profile terminal is not the successful direct model child')


def profile_terminal(directory, receipt=None):
    live, _, binding = profile_context(directory)
    path = directory / 'profile-model-terminal.json'
    if receipt is None:
        receipt = live.sealed(path, ct.sha(path))
    profile_validate_terminal(receipt, binding)
    ct.require(receipt['waitid']['observed_monotonic'] <= time.monotonic() < receipt['deadline_monotonic'],
               'Expired profile terminal acknowledgement')
    current = live.observer.proc_identity(binding['model']['pid'])
    ct.require(all(current[k] == binding['model'][k] for k in ('pid', 'birth_ticks', 'ppid'))
        and current['state'] in ('Z', 'X'), 'Profile terminal birth anchor is missing/changed/not held')
    ct.require(not (directory / 'profile-stop').exists(), 'Profile terminal acknowledgement cancelled')
    return live, binding, receipt


def profile_validate_ack(ack, binding, receipt, digest):
    profile_validate_terminal(receipt, binding)
    stopped, sample = ack['stopped_monotonic'], ack['last_sample']
    ct.require(ack['binding'] == binding and ack['terminal_sha256'] == digest
        and ack['status'] == 'stopped' and ack['errors'] == [], 'Missing/blocked/foreign profile observer acknowledgement')
    ct.require(type(stopped) in (int, float) and math.isfinite(stopped)
        and receipt['waitid']['observed_monotonic'] <= stopped <= time.monotonic()
        and stopped < receipt['deadline_monotonic'], 'Late/invalid profile observer acknowledgement')
    if sample is not None:
        begin, end = sample['started_monotonic'], sample['ended_monotonic']
        ct.require(all(type(t) in (int, float) and math.isfinite(t) for t in (begin, end))
            and binding['model_observed_monotonic'] <= begin <= end <= stopped,
            'Invalid profile terminal sample interval')


def profile_terminal_hook(directory, child, event, deadline, active):
    live, _, binding = profile_context(directory)
    ct.require(child.pid == binding['model']['pid'] and binding['model']['ppid'] == os.getpid()
        and binding['wrapper']['pid'] == os.getpid() and deadline == binding['capture_deadline_monotonic'],
        'Profile terminal hook is not the owning wrapper/direct child')
    receipt = dict(binding=binding, waitid=event,
        deadline_monotonic=min(deadline, event['observed_monotonic'] + PROFILE_TERMINAL_ACK_SECONDS))
    profile_terminal(directory, receipt)
    digest = live.readonly(directory / 'profile-model-terminal.json', receipt)
    ack_path = directory / 'profile-observer-stopped.json'
    while True:
        active()
        profile_terminal(directory)
        if ack_path.exists():
            ack = live.sealed(ack_path, ct.sha(ack_path))
            profile_validate_ack(ack, binding, receipt, digest)
            profile_terminal(directory)  # Recheck every held ancestor after consuming the ack.
            active()
            return
        time.sleep(.05)


def profile_monitor_ack(directory, stop, observations, sample, error=None):
    ct.require(not stop.is_set(), 'Profile observer acknowledgement cancelled')
    live, start, binding = profile_context(directory)
    if error is not None:
        ct.require(isinstance(error, live.observer.CustodyLost) and error.reason == 'exited'
            and error.expected is not None and error.current is not None
            and all(error.expected[k] == error.current[k] == binding['model'][k]
                    for k in ('pid', 'birth_ticks', 'ppid')) and error.current['state'] in ('Z', 'X'),
            'Unexpected profile identity loss; not the exact terminal model')
    deadline = min(start['capture_deadline_monotonic'], time.monotonic() + PROFILE_TERMINAL_ACK_SECONDS)
    while not (directory / 'profile-model-terminal.json').exists():
        ct.require(error is not None and time.monotonic() < deadline and not stop.wait(.05),
                   'Missing profile terminal receipt/cancelled observer')
    live, binding, receipt = profile_terminal(directory)
    digest = ct.sha(directory / 'profile-model-terminal.json')
    if error is not None:
        event = receipt['waitid']
        ct.require(sample is not None and binding['model_observed_monotonic'] <= sample['started_monotonic']
            <= error.observed_monotonic <= sample['ended_monotonic'],
            'Model terminal evidence does not overlap the interrupted sample')
        kind = 'model_terminal_before_sample' if event['observed_monotonic'] < sample['started_monotonic'] else 'model_terminal_during_sample'
        observations.append(dict(kind=kind, sample=sample,
            terminal_sha256=digest, identity_event=dict(message=str(error), expected=error.expected,
                current=error.current, reason=error.reason, observed_monotonic=error.observed_monotonic,
                **({'proc_read': error.profile_proc_read} if hasattr(error, 'profile_proc_read') else {}))))
    errors = [row['error'] for row in observations if 'error' in row]
    ack = dict(binding=binding, terminal_sha256=digest, last_sample=sample,
               stopped_monotonic=time.monotonic(), status='blocked' if errors else 'stopped', errors=errors)
    if not errors:
        profile_validate_ack(ack, binding, receipt, digest)
    ct.require(not stop.is_set(), 'Profile observer acknowledgement cancelled')
    stop.set()  # This same thread has finished its final sample, before publication.
    live.readonly(directory / 'profile-observer-stopped.json', ack)


def profile_model_read(live, model, read, *, name, operation, deadline):
    allowed = {'exe': ('sha256',), 'maps': ('read_text',), 'cmdline': ('read_bytes',)}
    ct.require(name in allowed and operation in allowed[name], 'Unbound model proc acquisition site')
    path, started = f'/proc/{model["pid"]}/{name}', time.monotonic()
    try:
        return read()
    except (FileNotFoundError, ProcessLookupError) as error:
        failed = time.monotonic()
        evidence = dict(source='qualify.profile_model_read', expected=dict(model), path=path,
            operation=operation, started_monotonic=started, failed_monotonic=failed,
            exception_type=type(error).__name__, error=str(error), filename=error.filename)
        error.profile_proc_read = evidence
        # Provenance is the declared acquisition site AND the actual failing
        # syscall path, never a substring in an arbitrary exception message.
        if error.filename != path:
            raise
        until = min(deadline, failed + (PROFILE_PROC_TERMINAL_SECONDS if name in ('exe', 'maps') else 0))
        evidence['recheck_deadline_monotonic'] = until
        while True:
            if 'first_recheck' in evidence and time.monotonic() >= until:
                raise
            try:
                current = live.observer.verify_root(model, identity=live.observer.proc_identity)
            except live.observer.CustodyLost as lost:
                lost.profile_proc_read = evidence
                raise lost from error
            evidence.setdefault('first_recheck', dict(identity=current, monotonic=time.monotonic()))
            # Linux can remove mm/exe while stat still reports R/S during exit.
            # Do not retry the read or fabricate a hash: only recheck the exact
            # birth/parent briefly, then require the existing owner receipt.
            if time.monotonic() >= until:
                raise
            time.sleep(max(0, min(.005, until - time.monotonic())))


def profile_monitor(directory, stop, observations):
    live, _ = profile_modules()
    seen, last_sample = set(), None
    while not stop.wait(.25):
        sample = None
        try:
            if not (directory / 'profile-model-start.json').exists():
                continue
            if (directory / 'profile-model-terminal.json').exists():
                profile_monitor_ack(directory, stop, observations, last_sample)
                return
            ct.require(not (directory / 'profile-model-ended.json').exists(), 'Model reaped before observer acknowledgement')
            sample = dict(started_monotonic=time.monotonic())
            _, start, _ = profile_context(directory)
            model = start['model']['before']
            live.observer.verify_root(model, identity=live.observer.proc_identity)
            key = profile_model_read(live, model, lambda: Path(f'/proc/{model["pid"]}/maps').read_text(),
                name='maps', operation='read_text', deadline=start['capture_deadline_monotonic'])
            if key not in seen or len(observations) < 2:
                seen.add(key)
                observations.append(profile_observation(directory, len(observations)))
            sample['ended_monotonic'] = time.monotonic()
            last_sample = sample
        except Exception as error:
            if sample is not None:
                sample['ended_monotonic'] = time.monotonic()
            if isinstance(error, live.observer.CustodyLost) and error.reason == 'exited':
                try:
                    profile_monitor_ack(directory, stop, observations, sample, error)
                except Exception as terminal_error:
                    observations.append(dict(error=type(error).__name__ + ': ' + str(error)
                        + '; terminal acknowledgement: ' + str(terminal_error), sample=sample,
                        traceback=traceback.format_exc(),
                        **({'proc_read': error.profile_proc_read} if hasattr(error, 'profile_proc_read') else {})))
                return
            observations.append(dict(error=type(error).__name__ + ': ' + str(error), sample=sample,
                traceback=traceback.format_exc(),
                **({'proc_read': error.profile_proc_read} if hasattr(error, 'profile_proc_read') else {})))
            if sample is None:
                return


def profile_observation(directory, index):
    live, maps = profile_modules()
    _, start, _ = profile_context(directory)
    gate = start['job']
    model, wrapper = start['model']['before'], start['wrapper']
    live.descendant(model, wrapper)
    before = live.observer.verify_root(model, identity=live.observer.proc_identity)
    record = dict(before=before,
                  executable_sha256=profile_model_read(live, model, lambda: ct.sha(f'/proc/{model["pid"]}/exe'),
                      name='exe', operation='sha256', deadline=start['capture_deadline_monotonic']),
                  command=profile_model_read(live, model, lambda: Path(f'/proc/{model["pid"]}/cmdline').read_bytes(),
                      name='cmdline', operation='read_bytes', deadline=start['capture_deadline_monotonic']).rstrip(b'\0').decode().split('\0'),
                  after=live.observer.verify_root(model, identity=live.observer.proc_identity))
    perf.validate_role(record, model, gate['binary_sha256'], perf.payload_argv(PLAN, gate))
    identity_errors = []
    def capture_identity(pid):
        ct.require(pid in (model['pid'], wrapper['pid']), 'Foreign profile capture identity request')
        try:
            return live.observer.verify_root(model if pid == model['pid'] else wrapper,
                                             identity=live.observer.proc_identity)
        except live.observer.CustodyLost as error:
            identity_errors.append(error)
            raise
    snapshot = maps.capture(directory / ('profile-maps-' + str(index)), model, wrapper,
                            gate['binary_sha256'], capture_identity)
    error = None
    if identity_errors:
        error = identity_errors[0]
        ct.require(snapshot['errors'] == [type(error).__name__ + ': ' + str(error)],
                   'Additional profile map errors during terminal transition')
    elif snapshot.get('proc_read_error'):
        failed = snapshot['proc_read_error']
        allowed = {'read_bytes': f'/proc/{model["pid"]}/maps',
                   'readlink': f'/proc/{model["pid"]}/exe', 'sha256': f'/proc/{model["pid"]}/exe'}
        ct.require(all(failed['identity'][k] == model[k] for k in ('pid', 'birth_ticks', 'ppid'))
            and failed['operation'] in allowed and failed['path'] == allowed[failed['operation']]
            and failed['exception_type'] in ('FileNotFoundError', 'ProcessLookupError')
            and snapshot['errors'] == [failed['error']]
            and snapshot['acquisition_begin']['monotonic'] <= failed['observed']['monotonic']
                <= snapshot['acquisition_end']['monotonic'], 'Unbound/additional profile proc-read error')
        try:
            live.observer.verify_root(model, identity=live.observer.proc_identity)
        except live.observer.CustodyLost as lost:
            if lost.reason != 'exited':
                raise
            error = lost
    if error is not None:
        if snapshot.get('raw_maps_path'):
            # The role read may be terminal, but already saved library evidence
            # must not conceal an independent pin/deletion error.
            ct.require(ct.sha(snapshot['raw_maps_path']) == snapshot['raw_maps_sha256'], 'Saved terminal maps changed')
            policy = ct.load(OUT / 'profile-runtime-policy.json')
            ct.require(('executable_path' not in snapshot or snapshot['executable_path'] == policy['executable'])
                and ('executable_sha256' not in snapshot or snapshot['executable_sha256'] == policy['executable_sha256']),
                'Independent profile executable role/pin error during terminal transition')
            libraries = maps.assess(Path(snapshot['raw_maps_path']).read_text(), policy)
            ct.require(not libraries['errors'] and not libraries['deleted_library_paths'],
                       'Independent profile library error during terminal transition')
        raise error
    assessment = maps.validate_snapshot(snapshot, ct.load(OUT / 'profile-runtime-policy.json'))
    ct.write(directory / ('profile-maps-' + str(index)) / 'assessment.json', assessment)
    ct.require(not assessment['errors'], 'Unknown/changed/deleted profile library mapping')
    profile_auth(directory, start['launch_sha256'])
    live.observer.verify_root(model, identity=live.observer.proc_identity)
    return dict(model=record, wrapper=wrapper, map_assessment=assessment,
                plan_sha256=start['plan_sha256'], job=gate, launch_sha256=start['launch_sha256'])


def profile_target(seal, gate_id):
    import importlib.util
    ct.require(any(g['id'] == gate_id and g['phase'] == 'profile' for g in PLAN['jobs']), 'Unknown profile target job')
    directory = OUT / 'gates' / gate_id
    live, gate, auth = profile_auth(directory, seal)
    verify()
    wrapper_identity = live.observer.proc_identity(os.getpid())
    live.descendant(wrapper_identity, auth['worker'])
    for name, value in environment(gate).items():
        # Official Nsight may extend its injected loader environment only.
        if name != 'LD_LIBRARY_PATH':
            ct.require(os.environ.get(name) == value, 'Actual profile target environment differs: ' + name)
    command = perf.payload_argv(PLAN, gate)
    def spawned(child):
        before = live.observer.proc_identity(child.pid)
        record = dict(before=before, executable_sha256=ct.sha(f'/proc/{child.pid}/exe'),
                      command=Path(f'/proc/{child.pid}/cmdline').read_bytes().rstrip(b'\0').decode().split('\0'),
                      after=live.observer.proc_identity(child.pid))
        perf.validate_role(record, before, gate['binary_sha256'], command)
        ct.require(before['ppid'] == wrapper_identity['pid'], 'Model is not the traced Popen wrapper direct child')
        live.descendant(wrapper_identity, auth['worker'])
        report = ct.load(directory / 'profile-server.json')
        ct.require(report['pid'] == child.pid, 'Foreign profile wrapper spawn report')
        live.readonly(directory / 'profile-model-start.json', dict(model=record, wrapper=wrapper_identity,
            authenticated=auth, job=gate, plan_sha256=ct.sha(plan_path()), launch_sha256=seal,
            capture_started_monotonic=report['started_monotonic'],
            capture_deadline_monotonic=report['started_monotonic'] + 1000, model_observed_monotonic=time.monotonic()))
    spec = importlib.util.spec_from_file_location('_finite_maintained_capture', ROOT / 'scripts/_profile_capture.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    result = wrapper.run_target(directory, command, 1000, 45, natural_completion=True, on_spawn=spawned,
        on_terminal=lambda child, event, deadline, active: profile_terminal_hook(directory, child, event, deadline, active))
    live.readonly(directory / 'profile-model-ended.json', dict(result=result, job=gate,
        launch_sha256=seal, wrapper_report_sha256=ct.sha(directory / 'profile-server.json')))
    return result


def export_profile(directory, gate, custody, observations):
    import sqlite3
    ct.require(ct.successful_process(custody) and idle.no_recorded_owner(custody), 'Export only after proven model/collector drain')
    ct.require(not any('error' in row for row in observations), 'Sticky profile identity/observer error')
    valid = [row for row in observations if 'model' in row]
    ct.require(len(valid) >= 2, 'Two independent live birth/executable model observations required')
    start = ct.load(directory / 'profile-model-start.json')
    for row in valid:
        ct.require(row['job'] == gate and row['plan_sha256'] == ct.sha(plan_path())
            and row['wrapper'] == start['wrapper'], 'Observed model from foreign job/root')
        perf.validate_role(row['model'], start['model']['before'], gate['binary_sha256'], perf.payload_argv(PLAN, gate))
        # An early loader snapshot may lack required CUDA DSOs; unknown maps or
        # changed content are never waived by a later good snapshot.
        ct.require(not row['map_assessment']['errors'], 'Unknown/changed/deleted profile library mapping')
    ct.require(any(r['map_assessment']['status'] == 'passed' for r in valid), 'No complete pinned profile-library observation')
    live, _ = profile_modules()
    terminal_path = directory / 'profile-model-terminal.json'
    ack_path = directory / 'profile-observer-stopped.json'
    terminal = live.sealed(terminal_path, ct.sha(terminal_path))
    ack = live.sealed(ack_path, ct.sha(ack_path))
    profile_validate_ack(ack, profile_binding(directory, start), terminal, ct.sha(terminal_path))
    ended = ct.load(directory / 'profile-model-ended.json')
    wrapper = ct.load(directory / 'profile-server.json')
    ct.require(ended['result'] == 0 and ended['job'] == gate
        and ended['wrapper_report_sha256'] == ct.sha(directory / 'profile-server.json')
        and wrapper['stop_reason'] == 'natural_completion' and wrapper['returncode'] == 0,
        'Profile target did not finish naturally with the maintained direct-child wrapper')
    ct.require(wrapper['pid'] == start['model']['before']['pid']
        and wrapper['started_monotonic'] == start['capture_started_monotonic']
        and ack['stopped_monotonic'] <= wrapper['ended_monotonic'] < start['capture_deadline_monotonic'],
        'Profile final reap report does not follow this observer acknowledgement within its lifetime')
    cpu_env = dict(PATH='/usr/bin:/bin', HOME='/root', CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', TMPDIR='/tmp')
    for format in ('sqlite', 'info'):
        output = directory / ('nsys.sqlite' if format == 'sqlite' else 'nsys.info.json')
        argv = [PLAN['nsys'], 'export', '--type=' + format, '--force-overwrite=false']
        if format == 'sqlite': argv.append('--lazy=false')
        argv += ['--output=' + str(output), str(directory / 'nsys.nsys-rep')]
        ct.write(directory / ('export-' + format + '-command.json'), dict(argv=argv, timeout_seconds=300, gpu_hidden=True))
        receipt = supervisor.run_process(ct.isolated_argv(argv, cpu_env), 300)
        ct.write(directory / ('export-' + format + '-supervisor.json'), receipt)
        ct.require(ct.successful_process(receipt) and output.is_file(), 'Official CPU-only export failed; no retry')
    with sqlite3.connect('file:' + str(directory / 'nsys.sqlite') + '?mode=ro', uri=True) as db:
        audit_deadline = time.monotonic() + 300
        db.set_progress_handler(lambda: int(time.monotonic() >= audit_deadline), 10000)
        result = profile_audit.process_api_summary(db, start['model']['before']['pid'], start['wrapper']['pid'], PLAN['uuid_inventory'][0])
    ct.write(directory / 'cuda-api-observations.json', result)
    return dict(status=result['status'], dropped_event_count=None,
        api_count_exhaustiveness='unproven', observed_api_summary_sha256=ct.sha(directory / 'cuda-api-observations.json'),
        nsys_report_sha256=ct.sha(directory / 'nsys.nsys-rep'), sqlite_sha256=ct.sha(directory / 'nsys.sqlite'),
        full_trace_accepted=False, whole_test_profiled_time_not_latency_population=True,
        owner_terminal_sha256=ct.sha(terminal_path), observer_ack_sha256=ct.sha(ack_path))


def prepare(variant='retained', *, lookahead_only=False):
    """CPU-only binding of existing artifacts and maintained controls; no build."""
    global PLAN, OUT, SOURCE, PARENT
    ct.require(variant in ('retained', 'grouped'), 'Unknown profile-only arm')
    grouped = variant == 'grouped'
    ct.require(not lookahead_only or grouped, 'Lookahead-only requires the same-ce6 grouped comparison')
    if grouped:
        OUT = perf.LOOKAHEAD_PROFILE_STAGE if lookahead_only else perf.GROUPED_PROFILE_STAGE
        SOURCE, PARENT = retained_evidence.GROUPED_STAGE, perf.GROUPED_STAGE
    ct.require_cpu_python()
    from profile_overlap_v1 import runtime_contract, analyze, scope_contract
    profile_modules()
    source_plan = SOURCE / ('grouped-plan.json' if grouped else 'retained-plan.json')
    original = ct.load(source_plan)
    ct.require(ct.load(SOURCE / 'FINAL.json')['status'] == 'passed', 'Completed frozen quality prerequisite missing')
    tool_manifest = ROOT / 'target/validation/retained-upload-57e4629-gpu-20261016-01/performance-planning/official-nsight-pins.json'
    tools = ct.load(tool_manifest)
    ct.check_pins({path: row['sha256'] for path, row in tools['pins'].items()})
    package = ROOT / 'target/vendor/nsight-systems-cli-2025.3.1/NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb'
    relative = 'opt/nvidia/nsight-systems-cli/2025.3.1/target-linux-x64/'
    control, members = runtime_contract.package_members(package, [relative + n for n in (*runtime_contract.SDK_LIBRARIES, 'nsys')])
    ct.require('Package: nsight-systems-cli-2025.3.1\n' in control
        and 'Version: 2025.3.1.90-253135822126v0\n' in control, 'Official installed package identity differs')
    target = Path(tools['nsys']).parent
    for name in (*runtime_contract.SDK_LIBRARIES, 'nsys'):
        ct.require(ct.sha(target / name) == members[relative + name], 'Installed official tool differs from package')
    prior_runtime_path = PARENT / 'profile-runtime-policy.json' if grouped else ROOT / 'papers/arxiv/experiments/storage_demand/profile_overlap_v1/artifacts-go-20261015-05/runtime-policy.json'
    runtime = ct.load(prior_runtime_path)
    base = ct.load(SOURCE / 'runtime-library-pins.json')['per_role']['composition']
    ct.require(all(runtime['pins'].get(path) == sha for path, sha in base.items()), 'Maintained profile runtime base differs from same-cut binary')
    runtime.update(role='composition', executable=str(SOURCE / 'staged/composition'),
        executable_sha256=retained_evidence.GROUPED_BINARIES['composition'] if grouped else perf.BINARY_SHA)
    ct.check_pins(runtime['pins'])
    ct.write(OUT / 'profile-runtime-policy.json', runtime)
    PLAN = dict(schema='ferrule.grouped-upload.lookahead-profile-only-plan.v1' if lookahead_only else
        'ferrule.grouped-upload.profile-only-plan.v1' if grouped else 'ferrule.retained-upload.profile-only-plan.v1',
        commit=retained_evidence.GROUPED_COMMIT if grouped else perf.COMMIT,
        source_digest=original['source_digest'], source_archive_sha256=retained_evidence.GROUPED_SOURCE_SHA if grouped else perf.SOURCE_SHA,
        executable=str(SOURCE / 'staged/composition'), binary_sha256=runtime['executable_sha256'],
        jobs=perf.profile_jobs(variant)[2:] if lookahead_only else perf.profile_jobs(variant), retries=0, extra_warmups=0, calibration=0, stop_on_first_failure=True,
        atol=.00025, rtol=.012, host_options=dict(max_host_bytes=64 << 20, max_metadata_bytes=8 << 20, worker_stack_bytes=2 << 20),
        capacity_and_options=original['composition_capacity_and_options'],
        composition_capacity_and_options=original['composition_capacity_and_options'],
        runtime_base_environment=original['runtime_base_environment'], uuid_inventory=original['uuid_inventory'],
        fresh_reference_sha256=original['fresh_reference_sha256'], full40_reference_sha256=original['full40_reference_sha256'],
        nsys_version=perf.VERSION, nsys=tools['nsys'], nsys_sha256=ct.sha(tools['nsys']),
        measurement='official_nsight_process_wide_counts_and_host_durations_not_unprofiled_latency',
        observed_model_summary_policy=profile_audit.model_summary_policy(),
        source_control_relationship='Application binary/archive unchanged; current maintained controls separately pinned, no rebuild',
        previous_plan_sha256=ct.sha(source_plan), previous_final_sha256=ct.sha(SOURCE / 'FINAL.json'),
        pins=dict(original['pins']))
    PLAN['completed_performance_source'] = dict(directory=str(PARENT),
        plan_sha256=ct.sha(PARENT / 'performance-plan.json'), final_sha256=ct.sha(PARENT / 'FINAL.json'))
    PLAN['diagnostic_primaries'] = []
    for job in perf.primary_jobs(PLAN):
        directory = PARENT / 'gates' / job['id']
        binding = dict(job=job)
        for field, filename in (('report', 'quality.json'), ('gate_result', 'gate-result.json')):
            path = directory / filename
            binding[field] = str(path); binding[field + '_sha256'] = ct.sha(path)
            PLAN['pins'][str(path)] = binding[field + '_sha256']
        PLAN['diagnostic_primaries'].append(binding)
    if grouped:
        PLAN.update(policies=dict(synchronous='unchanged_default', grouped=retained_evidence.GROUPED_POLICY),
            completed_quality_source=perf.grouped_quality_source(), profiling_captures=len(PLAN['jobs']), profiling=True,
            independent_review_required=True, exact_cross_policy_fingerprints_required=True,
            historical_results_in_population=False, terminal_observer_policy=perf.terminal_observer_policy())
        PLAN['pins'].update({str(SOURCE / n): ct.sha(SOURCE / n) for n in
            ('grouped-plan.json', 'FINAL.json', 'independent-review.json')})
    if lookahead_only:
        history = perf.GROUPED_PROFILE_STAGE / 'FINAL.json'
        PLAN.update(historical_profile_source=dict(directory=str(perf.GROUPED_PROFILE_STAGE),
            final_sha256=ct.sha(history)), proc_exit_observer_policy=perf.proc_exit_observer_policy())
        PLAN['pins'][str(history)] = ct.sha(history)
    PLAN['pins'].update({str(PARENT / n): ct.sha(PARENT / n) for n in
        ('performance-plan.json', 'FINAL.json', 'independent-review.json', 'review-ready.json')})
    PLAN['pins'].update({path: row['sha256'] for path, row in tools['pins'].items()})
    paths = {SCRIPT, HERE / 'performance.py', HERE / 'retained_evidence.py', HERE / 'quality.py', HERE / 'profile_audit.py',
             HERE / 'test_performance.py', ROOT / 'scripts/_profile_capture.py', ROOT / 'scripts/tests/test_profile_capture.py',
             ROOT / 'scripts/_owned_subreaper.py', ROOT / 'scripts/tests/test_owned_descendants.py', ROOT / 'scripts/tests/test_vllm_supervisor_failures.py',
             tool_manifest, package, prior_runtime_path, OUT / 'profile-runtime-policy.json', Path(PYTHON).resolve(), SOURCE / 'source.tar'}
    if grouped:
        paths.update({OUT / 'control-source-pins.json',
            ROOT / 'papers/arxiv/experiments/storage_demand/tests/test_library_maps_v1.py',
            ROOT / 'papers/arxiv/experiments/tp2_resident/test_observer.py'})
        PLAN['control_source'] = dict(path=str(OUT / 'control-source-pins.json'),
            sha256=ct.sha(OUT / 'control-source-pins.json'))
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file and Path(file).suffix == '.py' and Path(file).resolve().is_relative_to(ROOT):
            paths.add(Path(file).resolve())
    PLAN['pins'].update({str(path): ct.sha(path) for path in paths})
    perf.validate_plan(PLAN)
    ct.write(OUT / 'performance-plan.json', PLAN)
    (OUT / 'performance-plan.json').chmod(0o444)
    PLAN['gates'] = PLAN['jobs']
    verify(full=True)
    (OUT / 'gates').mkdir(exist_ok=False)
    print(f'PREPARED{len(PLAN["jobs"])} fresh profile-only jobs, existing same-cut artifacts/completed primaries; CPU pins only; review GO still required', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--worker')
    parser.add_argument('--profile-target')
    parser.add_argument('--gate-id')
    args = parser.parse_args()
    if args.prepare:
        ct.require(not args.worker and not args.profile_target and not args.gate_id, 'Preparation cannot execute a role')
        ct.require(args.directory.resolve() in (OUT, perf.GROUPED_PROFILE_STAGE, perf.LOOKAHEAD_PROFILE_STAGE),
                   'Wrong dedicated exact-scope profile output')
        prepare('grouped' if args.directory.resolve() in (perf.GROUPED_PROFILE_STAGE, perf.LOOKAHEAD_PROFILE_STAGE) else 'retained',
                lookahead_only=args.directory.resolve() == perf.LOOKAHEAD_PROFILE_STAGE)
        raise SystemExit(0)
    import signal
    def cancelled(number, frame):
        raise KeyboardInterrupt('Owned same-cut four-profile coordinator/worker cancellation')
    signal.signal(signal.SIGTERM, cancelled)
    configure(args.directory)
    if args.worker:
        ct.require(not args.profile_target and not args.gate_id, 'Conflicting worker roles')
        worker(args.worker)
    elif args.profile_target:
        ct.require(args.gate_id, 'Exact reviewed profile job required')
        raise SystemExit(profile_target(args.profile_target, args.gate_id))
    else:
        ct.require(not args.gate_id, 'Gate ID without authenticated target')
        raise SystemExit(main())
