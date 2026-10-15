"""CPU-only frozen TP1 profile contract. Preparation is never a GPU grant."""
from __future__ import annotations


from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from paired_read_v1.final_adapter_v3 import binding
from runnerhelpers import execution_v2 as ex, library_maps_v1 as maps
from runnerhelpers.evidence_v2 import ct, require
from profile_overlap_v1 import tp1_collector as collector, runtime_contract, scope_contract

ID = 'ferrule.storage-demand.profile-overlap.v1'
PYTHON = '/home/admin/wxw403883/anaconda3/bin/python3.12'
TOOL = ct.ROOT / 'target/vendor/nsight-systems-cli-2025.3.1/opt/nvidia/nsight-systems-cli/2025.3.1'
NSYS = TOOL / 'target-linux-x64/nsys'
VERSION = '2025.3.1.90-253135822126v0'
COMMIT = '86c58a017d7c4b53490f536a3b1a133f2f7848cc'
BINARY_SHA = '7cc8e0ff6e84f27090494b8e9e27dc3a388d1d2f3fc34ae39dae416e1e607a01'
ORDER = ['blocking', 'lookahead']
GENERATION = 'artifacts-go-20261015-04'
PREDECESSORS = tuple(HERE / f'artifacts-go-20261015-{n:02d}' for n in (1, 2, 3))
TOPOLOGY = 'maintained_traced_python_wrapper_owns_unreaped_cli'
PRIOR = HERE / 'artifacts-go-20261015-01'
ORDINARY_CLOSEOUT = HERE.parent / 'paired_read_v1/artifacts-v3-mapfixed-20261015-01/paired-final.json'
LIMITS = dict(startup_seconds=120, request_seconds=60, duration_seconds=150,
              shutdown_seconds=45, export_seconds=60, supervisor_seconds=300)
FLAGS = ['--trace=cuda,nvtx,osrt', '--sample=none', '--cpuctxsw=process-tree',
         '--gpu-metrics-devices=none', '--gpuctxsw=false', '--osrt-file-access=true',
         '--osrt-threshold=1000', '--duration=150', '--kill=none', '--wait=all',
         '--stop-on-exit=true', '--export=none', '--force-overwrite=false']
NOTE = ('Separate parent-authorized one-pair TP1 profile after ordinary timing; '
        'eight instrumentation locks, model GPU0 only, no foreign signals, retries or cache conditioning')


def qa():
    rows = ct.load(HERE.parent / 'paired_read_v1/requests.json')['schedule']
    selected = [r for r in rows if r['name'] == 'canary-qa']
    require(len(selected) == 1 and selected[0]['max_tokens'] == 8
            and selected[0]['reference']['reference_ids'] == [25358, 13, 198, 760, 6511, 314, 6124, 369],
            'frozen QA canary changed')
    return selected[0]


def spec(candidate, mode):
    require(mode in ORDER, 'unregistered arm')
    case = binding.case_for_mode(mode)
    require(case['topology'] == dict(pp=1, tp=1, ep=1, rank_backend='thread', owners=1, visible_ordinals=[0]),
            'not the ordinary TP1 configuration')
    env = dict(ex.runner.cpu_environment(), **case['environment'])
    require(env.get('FERRULE_LOG_FORMAT') == 'json'
            and env.get('FERRULE_LOG') == 'info,ferrule_performance=debug,ferrule_request=debug,ferrule_token=trace,ferrule_resource_observability=info'
            and 'FERRULE_IO_TRACE' not in env, 'typed stats logging or opt-in filename trace differs')
    env.update(CUDA_VISIBLE_DEVICES=collector.UUID, NVIDIA_VISIBLE_DEVICES=collector.UUID,
               CUDA_DEVICE_ORDER='PCI_BUS_ID', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               FERRULE_PROFILE_NVTX='1')
    return dict(engine='ferrule', argv=[candidate['candidate']['binary'], *case['argv_after_binary']],
                environment=env, model='qwen35-storage-demand-paper',
                url='http://127.0.0.1:18369/v1/completions')


def profile_argv(case, digest):
    """Original authenticated target shape; only its Popen owns the model child."""
    return [str(NSYS), 'profile', *FLAGS, '--output=' + str(case / 'nsys'),
            PYTHON, '-S', '-B', str(HERE / 'run.py'), 'target', '--directory', str(case), '--digest', digest]


def export_argv(case):
    return [str(NSYS), 'export', '--type=sqlite', '--lazy=false', '--force-overwrite=false',
            '--output=' + str(case / 'nsys.sqlite'), str(case / 'nsys.nsys-rep')]


def tool_help():
    """Reuse pinned official help receipts; never execute nsys during preparation."""
    from profile_2025 import TOOL_PINS
    ct.check_pins(TOOL_PINS)
    prior = PREDECESSORS[-1]
    prepared = collector.live.sealed(prior / 'prepared.json', ct.sha(prior / 'prepared.json'))
    require(ct.sha(prior / 'tool-help.json') == prepared['tool_help_sha256'], 'prior official help changed')
    require(prepared['pins'][str(NSYS)] == ct.sha(NSYS), 'installed official tool changed')
    reports = ct.load(prior / 'tool-help.json')
    require(VERSION in reports['version']['stdout'], 'unexpected official target CLI version')
    require(all(r['returncode'] == 0 for r in reports.values()), 'official help receipt failed')
    for flag in FLAGS:
        require(flag.split('=')[0] in reports['profile']['stdout'], 'unsupported profile option: ' + flag)
    require("Possible values are 'process-tree', 'system-wide', or 'none'." in reports['profile']['stdout']
            and "If --sample=none and a target application is launched, the default is 'process-tree'" in reports['profile']['stdout'],
            'CPU context-switch/sample independence not verified in official help')
    require('--lazy=' in reports['export']['stdout'] and '--type=' in reports['export']['stdout'],
            'full SQLite export not supported')
    require('setting is ignored for APIs that interact with files' in reports['profile']['stdout'],
            'file API threshold semantics unverified')
    return dict(reports, reuse=dict(path=str(prior / 'tool-help.json'), sha256=ct.sha(prior / 'tool-help.json'),
                                   commands_executed_now=0))


def candidate_check():
    # Reuse the immutable current qualification, not a rebuild/requalification.
    # Only the separate profile binding gains the verified injection inventory.
    root = binding.evidence.ROOT
    prior_binding = PRIOR / 'candidate-binding.json'
    previous = collector.live.sealed(prior_binding, ct.sha(prior_binding))
    require(previous['qualification_directory'] == str(root), 'different qualification directory')
    review = previous['review']
    ct.check_pins(review['artifact_pins'])
    require(ct.sha(previous['candidate']['binary']) == BINARY_SHA, 'qualified CLI changed')
    identity = review['candidate_identity']
    require(identity['commit'] == COMMIT and identity['cli_binary_sha256'] == BINARY_SHA,
            'different candidate; re-review required')
    value = dict(schema=ID + '.candidate', candidate_identity=identity,
        qualification_directory=str(root), uuid_inventory=review['uuid_inventory'],
        candidate=dict(binary=str(root / 'staged-final/cli'), binary_sha256=BINARY_SHA,
            runtime_pin_manifest=str(root / 'runtime-library-pins-final.json'),
            artifact_pins=review['artifact_pins']), review=review,
        qualification_reused_from=dict(path=str(prior_binding), sha256=ct.sha(prior_binding)),
        qualification_rerun=False, ordinary_acceptance_transferred=False, gpu_authorized=False)
    build = ct.load(root / 'build-final-cli-command.json')
    args = build['argv']
    require(args[args.index('--features') + 1] == 'cuda,ferrule-common/nvtx-profile'
            and build['environment']['FERRULE_PROFILE_NVTX'] == '0', 'actual CLI NVTX feature/default not bound')
    return value


def profile_runtime(candidate):
    return runtime_contract.inventory(TOOL, VERSION, PRIOR, candidate)


def collector_preflight():
    """CPU files only: owner-safe topology plus already recorded official CPU help.

    Reuse is explicit, not a fresh permission test. An actual permission/scheduling
    failure rejects the later capture; flags never change in response to data.
    """
    prior = PREDECESSORS[-1]
    previous = collector.live.sealed(prior / 'prepared.json', ct.sha(prior / 'prepared.json'))
    path = prior / 'collector-contract.json'
    require(ct.sha(path) == previous['collector_contract_sha256'], 'previous CPU contract changed')
    old = ct.load(path)
    cpu = old['cpu_environment_check']
    prior_go = ct.load(prior / 'explicit-parent-GO.json')
    compatible = (cpu['returncode'] == 0 and not cpu['stderr']
        and 'CPU Profiling Environment (process-tree): OK' in cpu['stdout']
        and 'Linux perf_event_open syscall available: OK' in cpu['stdout']
        and prior_go['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        and old['cpu_perf_event_paranoid'] == Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip())
    binding = collector.adapter_binding()
    blockers = [] if compatible else ['recorded CPU permissions no longer applicable; no fallback or capture']
    require(ct.load(PRIOR / 'capture-result.json')['full_trace_accepted'] is False,
            'predecessor rejection must remain immutable')
    require(all(ct.load(d / 'collector-contract.json')['launch_allowed'] is False for d in PREDECESSORS[1:]),
            'blocked predecessor must remain blocked')
    return dict(status='resolved' if not blockers else 'blocked_before_gpu',
        launch_allowed=not blockers, blockers=blockers, selected_topology=TOPOLOGY,
        owner_safe_wrapper_verified=True, adapter_binding=binding,
        measurement_scope=scope_contract.policy('process-tree'), measurement_methodology_corrected=True,
        cpu_context_switch_compatible=compatible, cpu_environment_check=cpu,
        cpu_environment_check_reused=True, cpu_environment_check_executed_now=False,
        cpu_evidence_path=str(path), cpu_evidence_sha256=ct.sha(path),
        cpu_permission_failure_policy='stop and keep both acceptances false; no flag fallback',
        shutdown='profile-stop -> original wrapper Popen.terminate/wait of unreaped direct CLI -> B1 -> natural collector exit',
        forbidden_shutdown=['direct CLI under nsys', 'nsys shutdown --kill', 'numeric grandchild PID signal', 'process group signal'],
        direct_shutdown_prior=old['direct_shutdown'], direct_adapter_verified=False,
        predecessor_pins=ct.inventory([d / 'prepared.json' for d in PREDECESSORS] +
            [PRIOR / 'capture-result.json', path, prior / 'explicit-parent-GO.json', prior / 'tool-help.json']),
        prior_acceptance_transferred=False, flags_changed_to_hide_warnings=False, collector_loss_waived=False,
        parent_GO_required_after_pressure_cleanup=True, gpu_queries=0, captures_issued=0, tool_commands_executed=0)


def source_qualification(candidate):
    """Archive-bound worker dispatcher, NVTX metadata and typed log provenance."""
    import hashlib
    import tarfile
    root = Path(candidate['qualification_directory'])
    archive = root / 'source.tar'
    require(ct.sha(archive) == candidate['candidate_identity']['source_archive_sha256'], 'source archive changed')
    checks = {
        'crates/ferrule-cli/src/main.rs': ['ferrule_common::observability::init_tracing();'],
        'crates/ferrule-common/src/observability.rs': ['nvtx::runtime_enabled()', 'filter_fn(nvtx::allows)',
                                                     '.try_init()', '.with_filter(env_filter)'],
        'crates/ferrule-common/src/observability/nvtx.rs': ['attributes.record(&mut fields)',
            'fn on_enter(', 'fn on_exit(', 'value == Some("1")', 'profile.label()'],
        'crates/ferrule-model/src/checkpoint/read_pipeline.rs': ['tracing::dispatcher::get_default(Clone::clone)',
            'tracing::dispatcher::with_default(&work.dispatch', '"read_subset"', 'source_generation = ticket.source_generation.get()'],
        'crates/ferrule-model/src/transformer/standard/cuda.rs': ['"layer.expert"', '"layer.expert.upload"',
                                                               'image_generation = self.image_generation()'],
        'crates/ferrule-model/src/transformer/standard/cuda/invocation_trace.rs': ['expert_read_pipeline_delta',
            'expert_read_counter_reset_detected', 'tracing::debug!'],
        'crates/ferrule-model/src/transformer/expert_reads.rs': ['pub io_started_reads: u64',
            'pub proved_read_bytes: u64', 'tracing::trace!(target: "ferrule_performance", layer, expert, receipt'],
    }
    with tarfile.open(archive) as source:
        members = {m.name.removeprefix('./'): m for m in source.getmembers() if m.isfile()}
        pins = {}
        for name, needles in checks.items():
            require(name in members, 'required source not archived: ' + name)
            data = source.extractfile(members[name]).read()
            path = root / 'frozen-source' / name
            require(ct.sha(path) == hashlib.sha256(data).hexdigest()
                    and all(needle in data.decode() for needle in needles), 'NVTX/worker/log source qualification differs: ' + name)
            pins[str(path)] = ct.sha(path)
    return dict(source_archive=str(archive), source_archive_sha256=ct.sha(archive), source_pins=pins,
        cli_sha256=BINARY_SHA, worker_dispatch='per-submission dispatcher carried to worker; process-global CLI subscriber',
        nvtx_runtime='profile-only FERRULE_PROFILE_NVTX=1; numeric on_new_span fields retained on enter',
        read_stats='DEBUG typed per-invocation deltas plus final owner-lifetime statistics',
        per_read_receipts='TRACE-only Debug diagnostics are not enabled or parsed; not a complete producer-identity journal',
        filename_trace='existing FERRULE_IO_TRACE opt-in remains absent', production_changes=False,
        expected_read_counts='actual current typed io_started_reads and proved_read_bytes, not a guessed expert count')


def require_launch(value):
    contract = value.get('collector_contract', {})
    require(contract.get('status') == 'resolved' and contract.get('launch_allowed') is True
            and contract.get('blockers') == [], 'collector contract unresolved; stop before GPUs')


def workflow_pins():
    # Pin transitive maintained Python sources, plus data those modules read.
    paths = {Path(m.__file__).resolve() for m in tuple(sys.modules.values())
             if getattr(m, '__file__', None) and str(m.__file__).endswith('.py')
             and Path(m.__file__).resolve().is_relative_to(ct.ROOT)}
    paths.update(HERE.glob('*.py'))
    paths.update([HERE / 'fixture.sql', HERE / 'README.md', HERE / 'RUN_20261015.md', HERE / 'SCOPE_20261015.md',
                  HERE / 'GENERATION_04.md',
                  TOOL / 'documentation/UserGuide/index.html', TOOL / 'documentation/InstallationGuide/index.html'])
    paths.update(ct.ROOT / 'scripts' / n for n in ('_profile_capture.py', '_owned_subreaper.py',
                                                 'test_vllm_qwen35.py', 'gpu_idle_window_v2.py', 'run_pinned_jobs.py'))
    paths.update(ct.HERE / 'tp2_resident' / n for n in
                 ('instrumentation_live.py', 'instrumentation_scope.py', 'instrumentation_capture.py'))
    paths.update([ct.DEFAULT, ct.EXCLUSIVE_PATH, HERE.parent / 'paired_read_v1/requests.json',
                  HERE.parent / 'paired_read_v1/cases-v3.json', Path(PYTHON), NSYS,
                  TOOL / 'host-linux-x64/QdstrmImporter'])
    return ct.inventory(sorted(paths))


def plan():
    return dict(schema=ID, generation=GENERATION, predecessor=str(PRIOR),
            ordinary_closeout=dict(path=str(ORDINARY_CLOSEOUT), sha256=ct.sha(ORDINARY_CLOSEOUT)),
            collector_gate='predeclared scoped roles AND original owner-safe collector, then fresh parent GO after pressure cleanup',
                        requested_topology=TOPOLOGY, owner_safe_wrapper_verified=True, direct_cli_adapter_verified=False,
                        predecessors={str(d): 'rejected' if d == PRIOR else 'blocked' for d in PREDECESSORS},
                        overlap_scope=scope_contract.policy('process-tree'),
                        generation_delta=['process-tree CPU context switches only; still sample=none and no GPU metrics',
                                          'original traced Python wrapper owns the unreaped CLI; no new execution loop',
                                          'separate strict scoped overlap eligibility from global full-trace acceptance'],
            order=ORDER, pair_ceiling=1, captures=2, requests_per_arm=1,
        qa=qa(), hello_requests=0, warmups=0, retries=0, limits=LIMITS, nsys=str(NSYS),
        flags=FLAGS, export_flags=['--type=sqlite', '--lazy=false', '--force-overwrite=false'],
        model_uuid=collector.UUID, model_topology=dict(tp=1, pp=1, ep=1), instrumentation_lock_count=8,
        nvtx_runtime='1 only in separate profile environments; ordinary timing remains 0',
        clock='official utcEpochNs plus actual request-ID server UTC markers',
        read_scope='worker read_subset allocation/pread envelope and OSRT pread intersections reported separately',
        compute_scope='actual CUPTI kernels correlated via process key/API/TID to layer.expert; upload excluded',
        graph_policy='eager only; graph launches/nodes/unknown one-to-many rejected, no graph-node tracing',
        warning_policy='every official warning rejects full trace, including classified wrapper/sampling warnings',
        failure_policy='audit each arm before admitting next; stop first failure without retry',
        cleanup_policy='exact authenticated owned-child shutdown, collector export, ECHILD, gpu_idle_window.v2 all-eight stable original idle',
        operation_journal='final counters and last receipt only; no complete per-operation expert journal',
        memory_claim='none; logical read reservations and sampled PSS are not a full-memory bound',
        profile_cost='whole startup plus one QA, shutdown, collector/export; never ordinary performance',
        gpu_queries=0, captures_issued=0, execute_allowed=False, ready_hook='requires fresh parent GO after pressure cleanup and ordinary closeout')


def directory_check(path):
    path = Path(path).absolute()
    require(path.parent == HERE and path.resolve() == path and not path.is_symlink()
            and re.fullmatch(r'artifacts-[A-Za-z0-9_-]{1,64}', path.name), 'fresh private artifacts-NAME under profile_overlap_v1 required')
    return path


def prepare(directory):
    directory = directory_check(directory)
    require(directory.name == GENERATION, 'only the separately authorized new generation may be prepared')
    seed = directory / 'predecessor-preservation-before.json'
    existing = set(directory.iterdir()) if directory.exists() else set()
    require(existing <= {seed}, 'preparation may not overwrite prior evidence')
    if seed.exists():
        preserved = collector.live.sealed(seed, ct.sha(seed))
        ct.check_pins(preserved['files'])
    else:
        directory.mkdir(mode=0o700, exist_ok=True)
        preserved = dict(files=ct.inventory(sorted(p for d in PREDECESSORS for p in d.rglob('*') if p.is_file())),
                         dispositions={str(d): 'rejected' if d == PRIOR else 'blocked' for d in PREDECESSORS})
        collector.live.readonly(seed, preserved)
    candidate = candidate_check()
    help_receipt = tool_help()
    runtime, policy, provenance = profile_runtime(candidate)
    contract = collector_preflight()
    source = source_qualification(candidate)
    candidate['profile_loaded_library_allowlist'] = dict(provenance['new_policy_raw_assessment']['libraries'])
    require(candidate['profile_loaded_library_allowlist'] == policy['pins'], 'profile allowlist not derived from verified inventory')
    directory.mkdir(mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    maps.private_write(directory / 'nvtx-source-qualification.json', source)
    maps.private_write(directory / 'runtime-provenance.json', provenance)
    maps.private_write(directory / 'collector-contract.json', contract)
    maps.private_write(directory / 'tool-help.json', help_receipt)
    maps.private_write(directory / 'runtime-policy.json', policy)
    collector.live.readonly(directory / 'candidate-binding.json', candidate)
    value = dict(plan(), directory=str(directory), candidate=candidate, runtime=runtime,
        collector_contract=contract, collector_contract_sha256=ct.sha(directory / 'collector-contract.json'),
        runtime_provenance_sha256=ct.sha(directory / 'runtime-provenance.json'),
        binding_path=str(directory / 'candidate-binding.json'), binding_sha256=ct.sha(directory / 'candidate-binding.json'),
        specs={mode: spec(candidate, mode) for mode in ORDER},
        target_roles=dict(model='actual frozen CLI SHA, birth, direct child of traced wrapper',
                          non_cuda_wrapper='actual pinned Python SHA and prepared source pins; no model warning exemption',
                          collector='control/custody only; never a warning-exempt role'),
        source_qualification_sha256=ct.sha(directory / 'nvtx-source-qualification.json'),
        preservation_seed_sha256=ct.sha(seed) if seed.exists() else None,
        pins=workflow_pins(), tool_help_sha256=ct.sha(directory / 'tool-help.json'),
        runtime_policy_sha256=ct.sha(directory / 'runtime-policy.json'))
    collector.live.readonly(directory / 'prepared.json', value)
    return dict(status='cpu_prepared_parent_GO_required' if contract['launch_allowed'] else 'cpu_prepared_blocked_before_gpu', directory=str(directory),
                prepared_sha256=ct.sha(directory / 'prepared.json'), gpu_queries=0, captures_issued=0)


def prepared(directory):
    directory = directory_check(directory)
    require(directory.name == GENERATION, 'only the separately authorized new generation may be used')
    value = collector.live.sealed(directory / 'prepared.json', ct.sha(directory / 'prepared.json'))
    require(all(value.get(k) == v for k, v in plan().items()), 'frozen profile plan differs')
    require(value['directory'] == str(directory) and value['binding_path'] == str(directory / 'candidate-binding.json')
            and value['binding_sha256'] == ct.sha(directory / 'candidate-binding.json')
            and collector.live.sealed(directory / 'candidate-binding.json', value['binding_sha256']) == value['candidate'],
            'profile binding/directory changed')
    seed_sha = value.get('preservation_seed_sha256')
    if seed_sha:
        preserved = collector.live.sealed(directory / 'predecessor-preservation-before.json', seed_sha)
        ct.check_pins(preserved['files'])
    require(value['source_qualification_sha256'] == ct.sha(directory / 'nvtx-source-qualification.json'),
            'NVTX source qualification changed')
    ct.check_pins(ct.load(directory / 'nvtx-source-qualification.json')['source_pins'])
    collector.adapter_binding()
    ct.check_pins(value['pins'])
    ct.check_pins(value['candidate']['candidate']['artifact_pins'])
    ct.check_pins(value['runtime']['pins'])
    policy = ct.load(directory / 'runtime-policy.json')
    require(value['candidate'].get('profile_loaded_library_allowlist') == policy['pins'] == value['runtime']['pins'],
            'profile binding/role-policy loaded-library allowlist differs')
    require(value['collector_contract_sha256'] == ct.sha(directory / 'collector-contract.json')
            and value['collector_contract'] == ct.load(directory / 'collector-contract.json')
            and value['runtime_provenance_sha256'] == ct.sha(directory / 'runtime-provenance.json'),
            'collector contract or runtime provenance changed')
    ct.check_pins(value['collector_contract']['predecessor_pins'])
    require(value['candidate']['candidate']['binary_sha256'] == BINARY_SHA
            and ct.sha(value['candidate']['candidate']['binary']) == BINARY_SHA, 'CLI candidate changed')
    require(value['tool_help_sha256'] == ct.sha(directory / 'tool-help.json')
            and value['runtime_policy_sha256'] == ct.sha(directory / 'runtime-policy.json'), 'preflight evidence changed')
    require(value['specs'] == {m: spec(value['candidate'], m) for m in ORDER}, 'profile spec changed')
    return value
