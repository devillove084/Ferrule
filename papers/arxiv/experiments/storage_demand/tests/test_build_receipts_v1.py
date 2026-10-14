"""Genuine builder/report schema shapes in temporary local CPU fixtures only."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
os.environ.update(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none')
STORAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STORAGE))
from newversionedprotocol import protocol, run_pilot
from runnerhelpers import build_receipts_v1 as adapter
from runnerhelpers.quality_reports_v1 import DENOMINATORS, PRODUCER_SCHEMAS
from runnerhelpers.evidence_v2 import ct


def quality_reports(config):
    result = {}
    for label, (metric_count, logits) in DENOMINATORS.items():
        report = dict(schema='ferrule.numeric-bf16-full40-profile-gate.v1' if label == 'full40'
            else 'ferrule.bf16-fresh-quality-result.v1', passed=True, complete=True, test_config=config,
            effective_config=dict(assertions_passed=True), failures=[], quarantine=False,
            state_bytes_after_shutdown=0, observer_mode='diagnostic_routes_and_states',
            ordinary_no_observer_claim=False, concurrent_scheduler_claim=False, greedy_tie_policy='lower_token_id',
            atol=0.00025, rtol=0.012, metrics=[dict(failures=0, nonfinite=0)] * metric_count,
            logits_compared=logits, device_budget_bytes=7516192768, actual_peak_reserved_bytes=4096,
            peak_reserved_bytes=4096, diagnostic_capture_only=False, status='passed',
            oracle_manifest_sha256='a' * 64, reference_manifest_sha256='b' * 64,
            next_token_checks=[dict(actual=7, expected=7, equal=True)] * 4)
        if label != 'full40':
            report['cold_owner_restart'] = label == 'cold'
            report['cases'] = []
            for name in (['qa'] if label == 'cold' else ['qa', 'arithmetic', 'chinese']):
                ids = list(range(9))
                checks = [dict(step=i, actual=i, expected=i, equal=True) for i in ids]
                report['cases'].append(dict(case=name, oracle_generated_ids=ids, common_history_generated_ids=ids,
                    free_greedy_ids=ids, free_greedy_equal=True, top1_failures=[],
                    free_greedy_status='completed_independent_feedback', next_token_checks=checks,
                    free_greedy_next_token_checks=checks, reset_reused_state=name == 'qa', reset_equal=True,
                    fork_snapshot_equal=True, reset_generated_ids=ids, reset_next_token_checks=checks))
        result[label] = report
    return result


class BuildFixture:
    def __init__(self, root):
        self.root = root
        self.paths = {}
        self.case = dict(argv_after_binary=['serve', '/fixture-model', '--backend', 'cuda', '--ctx-size', '128'],
                         quality_test_config=dict(name='storage-middle64'))
        self.candidate = dict(schema='ferrule.storage-demand.candidate-binding.v2',
            status='quality_passed_cpu_bind_requested', protocol_sha256=ct.sha(protocol.PROTOCOL_PATH),
            build_adapter=adapter.ADAPTER, artifact_pins={}, build_evidence={})
        self.source, self.target = root / 'frozen-source', root / 'cargo-target'
        source_files = {
            'Cargo.toml': '[profile.release]\nopt-level=3\nlto="fat"\ncodegen-units=1\npanic="unwind"\nstrip="symbols"\n',
            'Cargo.lock': '# CPU fixture lock\n', 'rust-toolchain.toml': '[toolchain]\nchannel="nightly-2026-08-01"\n',
            '.cargo/config.toml': '# fixture\n', 'crates/ferrule-cli/Cargo.toml': '[package]\nname="ferrule-cli"\n',
            'crates/ferrule-cli/src/main.rs': 'fn main() {}\n',
        }
        relative = {}
        for name, text in source_files.items():
            path = self.file('source:' + name, self.source / name, text, 0o444)
            relative[name] = ct.sha(path)
        self.source.chmod(0o555)
        original = {str(ct.ROOT / n): h for n, h in relative.items()}
        self.sources = original
        archive = root / 'source.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for name in relative:
                tar.add(self.source / name, arcname=name)
        archive.chmod(0o444)
        self.pin(archive)
        self.candidate['source_archive'] = str(archive)
        self.source_receipt = dict(schema=adapter.SOURCE_SCHEMA, frozen_source=str(self.source),
            immutable_readonly_inputs=True, archive_members_exactly_match_copied_source=True,
            archive_members_read_and_hashed=len(relative), file_count=len(relative),
            archive_payload_bytes=sum((self.source / n).stat().st_size for n in relative),
            pins_relative_to_frozen_source=relative, source_archive=str(archive), source_archive_sha256=ct.sha(archive),
            source_pins_digest_original_absolute_keys=ct.digest(original), source_pins_digest_relative_keys=ct.digest(relative))
        self.candidate['source_archive_manifest'] = str(self.receipt('source', self.source_receipt))
        self.env = {k: str(root / k.lower()) for k in adapter.ENV_KEYS}
        self.env.update(CARGO_TARGET_DIR=str(self.target), CARGO_BUILD_JOBS='2', CARGO_INCREMENTAL='0',
            CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none', FERRULE_CUDA_ARCH='sm_86', FERRULE_CUDA_TEST_ORACLE='0',
            RUSTUP_TOOLCHAIN='nightly-2026-08-01', RUSTUP_AUTO_INSTALL='0', CUBLAS_WORKSPACE_CONFIG=':0:0',
            CC='/usr/bin/gcc', CXX='/usr/bin/g++', CUDAHOSTCXX='/usr/bin/g++')
        cargo = self.file('cargo', root / 'bin/cargo', 'CPU fixture cargo')
        rustc = self.file('rustc', root / 'bin/rustc', 'CPU fixture rustc')
        nvcc = self.file('nvcc', root / 'bin/nvcc', 'CPU fixture nvcc')
        self.env.update(RUSTC=str(rustc), NVCC=str(nvcc), FERRULE_CUTLASS_DIR=str(root / 'cutlass'))
        cutlass = self.file('cutlass', root / 'cutlass/include/cutlass.h', 'CPU fixture CUTLASS')
        self.argv = [str(cargo), 'build', '--locked', '--offline', '--release', '-p', 'ferrule-cli',
                     '--no-default-features', '--features', 'cuda', '--message-format=json']
        def tool(argv, stdout):
            return dict(argv=argv, stdout=stdout, stderr='', returncode=0, exact_quality_tool_output_match=True)
        self.build = dict(schema=adapter.BUILD_SCHEMA, returncode=0, timed_out=False, timeout_seconds=1800,
            argv=self.argv, outer_supervisor_argv=['/usr/bin/timeout', '1800', *self.argv], environment=self.env,
            environment_policy='env -i exact allowlist; no inherited RUSTFLAGS, wrappers, CPU reference feature or GPU visibility',
            cwd=str(self.source), compiled_cuda='12.4.99', compiled_arch='sm_86', source_frozen_before_build=True,
            shared_build_target_reused=False, gpu_tools_or_workloads_launched=False,
            release_profile=dict(opt_level=3, lto='fat', codegen_units=1, panic='unwind', strip='symbols'),
            external_input_pin_receipt_timing='Post-build reverified byte-for-byte against previously sealed quality pins; not a newly captured pre-build receipt',
            toolchain=dict(cargo=tool([str(cargo), '-V'], 'cargo 1.99.0-nightly (7c83d4cc0 2026-07-29)\n'),
                rustc=tool([str(rustc), '-vV'], 'commit-hash: ' + adapter.RUST_COMMIT + '\n'),
                nvcc=tool([str(nvcc), '--version'], 'V12.4.99\n'),
                cutlass_head=tool(['git', 'rev-parse', 'HEAD'], adapter.CUTLASS_COMMIT + '\n'),
                cutlass_status=tool(['git', 'status', '--short'], '')))
        output = self.file('output', self.target / 'release/ferrule', 'CPU fixture executable, never executed')
        binary = self.file('binary', root / 'staged/ferrule', output.read_bytes(), 0o555)
        binary.parent.chmod(0o555)
        self.candidate.update(binary=str(binary), binary_sha256=ct.sha(binary))
        cli = dict(reason='compiler-artifact', target=dict(name='ferrule', src_path=str(self.source / 'crates/ferrule-cli/src/main.rs')),
            executable=str(output), manifest_path=str(self.source / 'crates/ferrule-cli/Cargo.toml'),
            features=['cuda'], fresh=False, profile=dict(opt_level='3', debug_assertions=False, test=False), package_id='path+fixture')
        dep = self.file('dep', root / 'registry/dependency/Cargo.toml', 'CPU fixture dependency')
        dep_row = dict(reason='compiler-artifact', package_id='registry+fixture', manifest_path=str(dep))
        self.artifact = dict(compiled_artifact=cli, workspace_artifacts=[cli], staged_binary=str(binary),
            staged_binary_sha256=ct.sha(binary), staged_binary_bytes=binary.stat().st_size, independent_inode_from_build_output=True)
        self.evidence('artifact', self.artifact)
        stdout = self.file('build_stdout', root / 'build.stdout', '\n'.join(json.dumps(v) for v in
            [dep_row, cli, dict(reason='build-finished', success=True)]) + '\n')
        stderr = self.file('build_stderr', root / 'build.stderr', '')
        self.candidate['build_evidence'].update(build_stdout=str(stdout), build_stderr=str(stderr))
        self.build.update(full_stdout_sha256=ct.sha(stdout), full_stderr_sha256=ct.sha(stderr))
        self.candidate['build_receipt'] = str(self.receipt('build', self.build))
        self.dependencies = dict(artifacts=[dep_row], package_count=1, file_count=1, pins={str(dep):ct.sha(dep)})
        self.evidence('dependencies', self.dependencies)
        library = self.file('library', root / 'runtime/libcuda.so', 'CPU fixture runtime library')
        proof = {}
        groups = {'compiler-input-pins.json': {str(p):ct.sha(p) for p in (cargo, rustc, nvcc, Path('/usr/bin/gcc'), Path('/usr/bin/g++'))},
            'cutlass-tree-pins.json': {str(cutlass):ct.sha(cutlass)},
            'compiled-dependency-pins.json': {str(dep):ct.sha(dep)},
            'runtime-library-pins.json': {str(library):ct.sha(library)}}
        for name, pins in groups.items():
            ref = self.receipt('reference:' + name, dict(pins=pins))
            proof[name] = dict(pins=pins, matched=len(pins), reference_receipt=str(ref), reference_receipt_sha256=ct.sha(ref))
        self.proof = proof
        self.evidence('compiler_pins', proof)
        originals, copies = {}, {}
        for name in ('libferrule_cpu_native.a', 'libferrule_cuda_core.a', 'libferrule_cuda_cutlass.a'):
            p = self.file('native:' + name, self.target / 'release/native' / name, name)
            q = self.file('nativecopy:' + name, root / 'immutable-native' / name, name, 0o444)
            originals[str(p)], copies[str(q)] = ct.sha(p), ct.sha(q)
        self.evidence('native_archives', dict(pins=originals))
        self.evidence('immutable_native_archives', dict(pins=copies, original_build_archives=originals))
        self.runtime = dict(libraries={str(library):dict(sha256=ct.sha(library), load_path=str(library), bytes=library.stat().st_size)},
            count=1, required_ld_library_path=self.env['LD_LIBRARY_PATH'], ldd_is_cpu_only_static_loader_resolution=True,
            gpu_context_and_actual_server_maps_unobserved=True, explicit_model_vendor_dlopen_libraries_not_exercised=True,
            all_resolved_libraries_match_quality_runtime_pins=True)
        self.evidence('runtime_resolution', self.runtime)
        probe = self.probe('ldd', ['/usr/bin/ldd', str(binary)], str(library) + ' (0xfixture)\n')
        self.candidate['build_evidence'].update(ldd_receipt=probe['receipt'], ldd_stdout=probe['stdout'], ldd_stderr=probe['stderr'])
        self.candidate['build_evidence']['help'] = {}
        for name, argv, text in [('help', ['--help'], 'serve --help'), ('serve_help', ['serve','--help'], '--backend --ctx-size'),
                                  ('version', ['--version'], 'ferrule 0.2.0')]:
            self.candidate['build_evidence']['help'][name] = self.probe(name, [str(binary), *argv], text)
        self.candidate['serve_help_receipt'] = self.candidate['build_evidence']['help']['serve_help']['receipt']
        self.candidate['runtime_pin_manifest'] = str(self.receipt('run_runtime', dict(pins=groups['runtime-library-pins.json'], unknown_nccl_enabled=False)))

    def pin(self, path):
        self.candidate['artifact_pins'][str(path)] = ct.sha(path)

    def file(self, key, path, data, mode=0o644):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.encode() if isinstance(data,str) else data)
        path.chmod(mode)
        self.paths[key] = path
        self.pin(path)
        return path

    def receipt(self, key, value):
        return self.file(key, self.root / (key.replace(':','-') + '.json'), json.dumps(value))

    def evidence(self, key, value):
        self.candidate['build_evidence'][key] = str(self.receipt(key, value))

    def probe(self, key, argv, text):
        stdout = self.file(key + '_stdout', self.root / (key + '.stdout'), text)
        stderr = self.file(key + '_stderr', self.root / (key + '.stderr'), '')
        receipt = self.receipt(key, dict(argv=argv, cwd=str(self.source), gpu_hidden=True, returncode=0, timed_out=False,
            timeout_seconds=30, environment=dict(CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='none',
            LD_LIBRARY_PATH=self.env['LD_LIBRARY_PATH']), stdout_sha256=ct.sha(stdout), stderr_sha256=ct.sha(stderr)))
        return dict(receipt=str(receipt), stdout=str(stdout), stderr=str(stderr))

    def replace(self, key, value):
        p = self.paths[key]
        p.write_text(json.dumps(value))
        self.pin(p)

    def qualify(self):
        relation = dict(status='verified_same_compiled_production_sources', cli_binary_sha256=self.candidate['binary_sha256'],
            production_source_digest=ct.digest(self.sources), independent_reference=True, supervision='authenticated_drained_all_three',
            report_sha256={}, supervisor_receipts={}, test_source_manifests={}, test_executables={}, authenticated_launches={}, producer_finals={}, reference_manifests={})
        reports = quality_reports(self.case['quality_test_config'])
        for label in reports:
            ref = self.file(label + '_ref', self.root / (label + '-reference.json'), 'CPU fixture independent reference')
            reports[label]['reference_manifest_sha256' if label == 'full40' else 'oracle_manifest_sha256'] = ct.sha(ref)
            report = self.receipt(label, reports[label])
            self.candidate[label] = str(report)
            relation['report_sha256'][label] = ct.sha(report)
            relation['reference_manifests'][label] = str(ref)
            exe = self.file(label + '_exe', self.root / (label + '-test'), 'CPU fixture libtest executable')
            relation['test_executables'][label] = str(exe)
            manifest = self.receipt(label + '_sources', dict(original_source_pins=self.sources))
            relation['test_source_manifests'][label] = str(manifest)
            helper = dict(pid=5000, birth_ticks=10, ppid=1)
            worker = dict(pid=5001, birth_ticks=20, ppid=5000)
            launch_id = 'cpu-fixture-' + label
            custody = dict(returncode=0, timed_out=False, cleanup_errors=[], descendant_custody='proven_drained',
                handshake=dict(containment_verified=True, launch_id=launch_id, helper_identity=helper),
                custody=dict(containment_verified=True, launch_id=launch_id, helper_identity=helper,
                             root_identity=worker, echild=True, status='proven_drained'))
            relation['supervisor_receipts'][label] = str(self.receipt(label + '_custody', custody))
            launch = dict(identity=dict(launch_id=launch_id, helper=helper, worker=worker), argv=[str(exe),'fixture-selector'],
                binary_sha256=ct.sha(exe), exec_preserves_authenticated_root_pid_birth=True, all_eight_uuid_locks_validated=True)
            relation['authenticated_launches'][label] = str(self.receipt(label + '_launch', launch))
            process = dict(selector='fixture-selector',returncode=0,timed_out=False,
                           invocations_this_authorization=1,invocations=1,report=str(report),report_sha256=ct.sha(report))
            final = dict(schema=PRODUCER_SCHEMAS[label],wrapper_accepted=True,
                single_test=process,process=process,
                quality_report=dict(path=str(report),sha256=ct.sha(report),complete=True,passed=True),
                binary=dict(path=str(exe),sha256=ct.sha(exe)),
                artifact=dict(binary=str(exe),sha256=ct.sha(exe),source_pins_digest=ct.digest(self.sources)),
                source=dict(pins_digest=ct.digest(self.sources),source_pins_digest=ct.digest(self.sources)))
            relation['producer_finals'][label] = str(self.receipt(label + '_final',final))
        relation['cold_primary_report_sha256'] = relation['report_sha256']['fresh']
        self.relation = relation
        self.candidate['quality_source_relation'] = str(self.receipt('quality_relation', relation))
        return self.candidate


class BuildAdapterTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(ExitStack())
        for name in ('subprocess.run', 'subprocess.Popen', 'os.kill'):
            stack.enter_context(mock.patch(name, side_effect=AssertionError('CPU fixtures: no launch/query/signal')))
        stack.enter_context(mock.patch.object(ct, 'exclusive_owner_lock', side_effect=AssertionError('no GPU lock')))
        self.fixture = BuildFixture(Path(self.enterContext(tempfile.TemporaryDirectory(prefix='cli-build-contract-cpu-'))))

    def review(self):
        f = self.fixture
        return adapter.review_cli_build(f.candidate, f.build, f.case)

    def test_genuine_raw_builder_format_positive_without_manufactured_m7_or_supervisor(self):
        result = self.review()
        self.assertEqual(result['raw_schema'], adapter.BUILD_SCHEMA)
        self.assertEqual(result['sources'], self.fixture.sources)
        self.assertEqual(result['coverage']['authenticated_build_custody'], 'unobserved')
        self.assertEqual(result['coverage']['loaded_model_runtime'], 'unobserved')
        self.assertNotIn('supervisor', self.fixture.build)
        self.assertFalse(result['quality_approved'])
        self.assertFalse(result['gpu_authorized'])

    def test_explicit_adapter_opt_in_required_and_unknown_versions_fail_closed(self):
        f = self.fixture
        for mode in (None, 'auto', adapter.ADAPTER + '.unknown'):
            f.candidate['build_adapter'] = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'explicit|unknown'):
                adapter.validate_candidate_build(f.candidate, f.build, {}, f.case)

    def test_legacy_m7_delegates_unchanged(self):
        build = dict(schema='ferrule.paper.m7.build.v1', binary_sha256='a'*64, sources_before={'s':'h'})
        with mock.patch.object(ct, 'validate_build', return_value='/fixture/legacy') as legacy:
            result = adapter.validate_candidate_build({}, build, {'config':1}, {})
        legacy.assert_called_once_with(build, {'config':1})
        self.assertEqual(result['adapter'], 'legacy_m7_unchanged')

    def test_missing_archive_rejected(self):
        Path(self.fixture.candidate['source_archive']).unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.review()

    def test_wrong_archive_hash_rejected_even_with_re_pinned_receipt(self):
        f = self.fixture
        f.source_receipt['source_archive_sha256'] = '0'*64
        f.replace('source', f.source_receipt)
        with self.assertRaisesRegex(ValueError, 'archive hash'):
            self.review()

    def test_incomplete_source_manifest_rejected(self):
        f = self.fixture
        del f.source_receipt['pins_relative_to_frozen_source']['Cargo.lock']
        f.replace('source', f.source_receipt)
        with self.assertRaises(ValueError):
            self.review()

    def test_archive_payload_not_just_receipt_boolean_verified(self):
        f = self.fixture
        p = Path(f.candidate['source_archive'])
        p.chmod(0o644)
        with tarfile.open(p, 'w:gz') as stream:
            stream.add(f.source / 'Cargo.lock', arcname='Cargo.lock')
        p.chmod(0o444)
        f.pin(p)
        f.source_receipt['source_archive_sha256'] = ct.sha(p)
        f.replace('source', f.source_receipt)
        with self.assertRaisesRegex(ValueError, 'archive coverage'):
            self.review()

    def test_frozen_source_drift_rejected_without_consulting_live_tree(self):
        path = self.fixture.paths['source:Cargo.lock']
        path.chmod(0o644)
        path.write_text('drift')
        path.chmod(0o444)
        with self.assertRaises(ValueError):
            self.review()

    def test_relative_digest_cannot_replace_original_absolute_source_cut(self):
        f = self.fixture
        f.source_receipt['source_pins_digest_original_absolute_keys'] = f.source_receipt['source_pins_digest_relative_keys']
        f.replace('source', f.source_receipt)
        with self.assertRaisesRegex(ValueError, 'cut digest'):
            self.review()

    def test_build_binary_copy_drift_rejected(self):
        self.fixture.paths['output'].write_text('different executable')
        with self.assertRaisesRegex(ValueError, 'binary copy'):
            self.review()

    def test_same_inode_not_an_independent_immutable_copy(self):
        f = self.fixture
        p = f.paths['output']
        p.unlink()
        os.link(f.paths['binary'], p)
        with self.assertRaisesRegex(ValueError, 'binary copy'):
            self.review()

    def test_missing_compiler_pin_group_rejected(self):
        f = self.fixture
        del f.proof['compiler-input-pins.json']
        f.replace('compiler_pins', f.proof)
        with self.assertRaisesRegex(ValueError, 'pin groups'):
            self.review()

    def test_extra_unpinned_actual_dependency_rejected(self):
        self.fixture.paths['dep'].with_name('extra.rs').write_text('unrecorded source')
        with self.assertRaisesRegex(ValueError, 'dependency source pins'):
            self.review()

    def test_missing_native_archive_copy_rejected(self):
        f = self.fixture
        p = f.paths['immutable_native_archives']
        value = ct.load(p)
        del value['pins'][next(iter(value['pins']))]
        f.replace('immutable_native_archives', value)
        with self.assertRaisesRegex(ValueError, 'archive copies'):
            self.review()

    def test_loaded_runtime_missing_or_false_observed_claim_rejected(self):
        f = self.fixture
        for value in (None, False):
            f.runtime['gpu_context_and_actual_server_maps_unobserved'] = value
            f.replace('runtime_resolution', f.runtime)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'loaded-runtime'):
                self.review()

    def test_runtime_resolution_pin_hash_rejected(self):
        f = self.fixture
        f.runtime['libraries'][str(f.paths['library'])]['sha256'] = '0'*64
        f.replace('runtime_resolution', f.runtime)
        with self.assertRaises(ValueError):
            self.review()

    def test_ldd_missing_loaded_resolution_library_rejected(self):
        f = self.fixture
        path = f.paths['ldd_stdout']
        path.write_text('not found\n')
        f.pin(path)
        probe = ct.load(f.paths['ldd'])
        probe['stdout_sha256'] = ct.sha(path)
        f.replace('ldd',probe)
        with self.assertRaisesRegex(ValueError, 'ldd runtime'):
            self.review()

    def test_help_cpu_fields_never_imply_gpu_or_quality(self):
        facts = self.review()
        self.assertEqual(facts['coverage']['field_help'], 'observed_cpu_only')
        self.assertEqual(facts['coverage']['gpu_context'], 'unobserved')
        self.assertFalse(facts['quality_approved'])
        with self.assertRaisesRegex(ValueError, 'unknown candidate qualification'):
            protocol.prepare_candidate(dict(self.fixture.candidate, status='build_reviewed'), self.fixture.case)

    def test_help_wrong_binary_and_missing_field_rejected(self):
        f = self.fixture
        probe = ct.load(f.paths['serve_help'])
        probe['argv'][0] = '/old/binary'
        f.replace('serve_help', probe)
        with self.assertRaisesRegex(ValueError, 'probe failed/unbound'):
            self.review()

    def test_help_flag_substring_does_not_count_as_field_help(self):
        f = self.fixture
        p = f.paths['serve_help_stdout']
        p.write_text('--backend-unrelated --ctx-size-unrelated')
        f.pin(p)
        probe = ct.load(f.paths['serve_help'])
        probe['stdout_sha256'] = ct.sha(p)
        f.replace('serve_help', probe)
        with self.assertRaisesRegex(ValueError, 'field-help flags'):
            self.review()

    def test_gpu_visibility_build_timeout_failure_and_unknown_schema_rejected(self):
        f = self.fixture
        changes = [('schema','unknown'), ('timed_out',True), ('returncode',1), ('source_frozen_before_build',None),
                   ('shared_build_target_reused',True), ('compiled_cuda','12.3'), ('gpu_tools_or_workloads_launched',True)]
        for key, value in changes:
            changed = copy.deepcopy(f.build)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                adapter.review_command(changed)
        changed = copy.deepcopy(f.build)
        changed['environment']['CUDA_VISIBLE_DEVICES'] = '0'
        with self.assertRaises(ValueError):
            adapter.review_command(changed)

    def test_raw_build_stdout_hash_is_not_trusted_boolean(self):
        self.fixture.paths['build_stdout'].write_text('{}')
        with self.assertRaises(ValueError):
            self.review()

    def test_cpu_build_review_not_blocked_by_unobserved_subreaper(self):
        f = self.fixture
        f.build['supervisor'] = dict(status='unobserved')
        f.replace('build',f.build)
        self.assertEqual(self.review()['coverage']['authenticated_build_custody'], 'unobserved')

    def test_same_cut_actual_quality_producer_fixture_reaches_cpu_gate_without_gpu_authorization(self):
        f = self.fixture
        f.qualify()
        result = protocol.prepare_candidate(f.candidate, f.case)
        self.assertEqual(result['status'],'quality_passed_cpu_prepared_not_authorized')
        self.assertFalse(result['gpu_authorized'])
        self.assertEqual(result['gpu_runs_by_this_helper'],0)
        self.assertFalse(any('candidate' in p.name for p in f.root.iterdir()))

    def workflow_fixture(self):
        f = self.fixture
        unchanged = f.paths['library']
        original = dict(schema='ferrule.paper.storage-demand.cpu-prepared.v2',
            status='cpu_prepared_candidate_unbound_not_authorized',candidate_binary=None,
            sealed_final_candidate=False,gpu_queries=0,gpu_locks=0,gpu_runs=0,
            protocol_sha256=ct.sha(protocol.PROTOCOL_PATH),
            artifact_pins={str(unchanged):ct.sha(unchanged),
                           **{str(p):'a'*64 for p in protocol.REVIEWED_REPLACEMENTS}},
            reference_pins={str(f.paths['dep']):ct.sha(f.paths['dep'])})
        original['digest'] = ct.digest(original)
        path = f.receipt('baseline_cpu',original)
        self.enterContext(mock.patch.object(protocol,'PREPARED_PATH',path))
        return path

    def test_explicit_new_workflow_fixture_plan_and_check_not_stale_bypass(self):
        f = self.fixture
        old = self.workflow_fixture()
        with self.assertRaises(ValueError):
            protocol.validate_prepared(old)
        value = protocol.workflow_review()
        self.assertFalse(value['sealed_final_candidate'])
        self.assertEqual(set(value['reviewed_replacements']),{str(p) for p in protocol.REVIEWED_REPLACEMENTS})
        value['digest'] = ct.digest(value)
        path = f.receipt('new_workflow',value)
        self.assertEqual(protocol.validate_prepared(path),value)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(protocol.main(['check','--cpu-prepared',str(path)]),0)
        plan = run_pilot.plan(path)
        self.assertIsNone(plan['preparation_error'])
        self.assertEqual((plan['starts'],plan['scheduled_requests'],plan['scheduled_measured_qa']),(5,60,25))
        plan['digest'] = ct.digest(plan)
        execution = f.receipt('fixture_execution',plan)
        self.assertEqual(run_pilot.validate_execution(execution),plan)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(run_pilot.main(['check','--execution-prepared',str(execution)]),0)
        self.assertEqual(ct.sha(old),value['original_prepared_sha256'])

    def test_workflow_registration_does_not_ignore_unrelated_old_pin_drift(self):
        f = self.fixture
        self.workflow_fixture()
        f.paths['library'].write_text('unrelated drift')
        with self.assertRaises(ValueError):
            protocol.workflow_review()

    def test_new_workflow_changed_inventory_or_false_candidate_claim_rejected(self):
        f = self.fixture
        self.workflow_fixture()
        value = protocol.workflow_review()
        for field,bad in [('candidate_binary','/old/binary'),('sealed_final_candidate',True),
                          ('artifact_pins',{}),('gpu_runs',1)]:
            changed = dict(value,**{field:bad})
            changed['digest'] = ct.digest(changed)
            path = f.receipt('modified_workflow',changed)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'workflow changed'):
                protocol.validate_prepared(path)

    def test_workflow_seal_cli_is_exclusive_and_only_creates_cpu_fixture(self):
        f = self.fixture
        self.workflow_fixture()
        output = f.root/'exclusive-workflow.json'
        reviewed = protocol.workflow_review()
        with mock.patch.object(protocol,'HERE',f.root),mock.patch.object(protocol,'workflow_review',return_value=reviewed),redirect_stdout(io.StringIO()):
            self.assertEqual(protocol.main(['seal-workflow','--output',str(output)]),0)
            first_hash = ct.sha(output)
            with self.assertRaisesRegex(ValueError,'exclusive new workflow'):
                protocol.main(['seal-workflow','--output',str(output)])
        self.assertEqual(ct.sha(output),first_hash)
        self.assertFalse(ct.load(output)['sealed_final_candidate'])

    def test_read_only_review_api_does_not_bind_and_needs_no_quality_reports(self):
        f = self.fixture
        descriptor = dict(f.candidate, schema='ferrule.storage-demand.cpu-build-review-input.v1')
        path = f.receipt('review_input', descriptor)
        result = protocol.review_build(path, f.case)
        self.assertFalse(result['sealed_final_candidate'])
        self.assertFalse(result['quality_approved'])
        self.assertEqual(result['status'], 'build_evidence_reviewed_quality_pending_not_authorized')
        with mock.patch.object(protocol, 'CASE_PATH', f.receipt('fixture_case', f.case)), redirect_stdout(io.StringIO()) as out:
            self.assertEqual(protocol.main(['review-build','--build-review',str(path)]),0)
        self.assertFalse(json.loads(out.getvalue())['gpu_authorized'])

    def test_review_cli_refuses_output_and_unknown_descriptor_version(self):
        f = self.fixture
        path = f.receipt('review_input',dict(f.candidate,schema='unknown'))
        with self.assertRaisesRegex(ValueError,'unknown CPU build review input'):
            protocol.review_build(path,f.case)
        with self.assertRaisesRegex(ValueError,'no --output'):
            protocol.main(['review-build','--build-review',str(path),'--output',str(f.root/'forbidden.json')])
        self.assertFalse((f.root/'forbidden.json').exists())

    def test_stale_workflow_plan_is_not_an_acceptable_execution_seal(self):
        with mock.patch.object(protocol,'validate_prepared',side_effect=ValueError('fixture stale pins')):
            value = run_pilot.plan()
            self.assertEqual(value['status'],'cpu_plan_workflow_unsealed_candidate_unbound_not_authorized')
            self.assertEqual(value['preparation_error'],'fixture stale pins')
            self.assertEqual((value['starts'],value['scheduled_requests'],value['scheduled_measured_qa']),(5,60,25))
            with self.assertRaisesRegex(ValueError,'workflow seal stale'):
                run_pilot.main(['seal-execution','--output',str(run_pilot.HERE/'not-created-cpu-fixture.json')])
        self.assertFalse((run_pilot.HERE/'not-created-cpu-fixture.json').exists())

    def test_old_full40_source_cut_cannot_qualify_frozen_cli(self):
        f = self.fixture
        f.qualify()
        f.replace('full40_sources',dict(original_source_pins={'old-source':'a'*64}))
        with self.assertRaisesRegex(ValueError, 'compiled-source manifest mismatch: full40'):
            protocol.prepare_candidate(f.candidate,f.case)

    def test_quality_unknown_custody_cannot_be_repaired_by_cpu_build(self):
        f = self.fixture
        f.qualify()
        custody = ct.load(f.paths['fresh_custody'])
        custody['custody']['status'] = 'unknown'
        f.replace('fresh_custody',custody)
        with self.assertRaisesRegex(ValueError,'custody/failure: fresh'):
            protocol.prepare_candidate(f.candidate,f.case)

    def test_quality_real_producer_binary_relation_not_boolean(self):
        f = self.fixture
        f.qualify()
        launch = ct.load(f.paths['cold_launch'])
        launch['binary_sha256'] = '0'*64
        f.replace('cold_launch',launch)
        with self.assertRaisesRegex(ValueError,'actual producer'):
            protocol.prepare_candidate(f.candidate,f.case)

    def test_successful_subreaper_cannot_repair_failed_outer_wrapper(self):
        f = self.fixture
        f.qualify()
        final = ct.load(f.paths['fresh_final'])
        final['wrapper_accepted'] = False
        f.replace('fresh_final',final)
        with self.assertRaisesRegex(ValueError,'final wrapper failed/unknown: fresh'):
            protocol.prepare_candidate(f.candidate,f.case)

    def test_outer_wrapper_must_join_actual_report_and_source(self):
        f = self.fixture
        f.qualify()
        for field,value in [('quality_report',dict(path='/old/report',sha256='a'*64,complete=True,passed=True)),
                            ('artifact',dict(binary='/old/test',sha256='a'*64,source_pins_digest='a'*64))]:
            final = ct.load(f.paths['fresh_final'])
            original = copy.deepcopy(final)
            final[field] = value
            f.replace('fresh_final',final)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'producer final'):
                protocol.prepare_candidate(f.candidate,f.case)
            f.replace('fresh_final',original)

    def test_missing_producer_pid_birth_cannot_match_two_absent_fields(self):
        f = self.fixture
        f.qualify()
        launch = ct.load(f.paths['cold_launch'])
        custody = ct.load(f.paths['cold_custody'])
        del launch['identity']['worker']
        del custody['custody']['root_identity']
        f.replace('cold_launch',launch)
        f.replace('cold_custody',custody)
        with self.assertRaisesRegex(ValueError,'PID/birth/parent'):
            protocol.prepare_candidate(f.candidate,f.case)

    def test_quality_producer_launch_identity_must_join_authenticated_custody(self):
        f = self.fixture
        f.qualify()
        launch = ct.load(f.paths['cold_launch'])
        launch['identity']['launch_id'] = 'different-launch'
        f.replace('cold_launch',launch)
        with self.assertRaisesRegex(ValueError,'actual producer'):
            protocol.prepare_candidate(f.candidate,f.case)

    def test_run_runtime_manifest_must_cover_build_resolved_libraries(self):
        f = self.fixture
        f.qualify()
        f.replace('run_runtime',dict(pins={str(f.paths['cargo']):ct.sha(f.paths['cargo'])},unknown_nccl_enabled=False))
        with self.assertRaisesRegex(ValueError,'runtime pins omit'):
            protocol.prepare_candidate(f.candidate,f.case)


class QualityReportsTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(quality_test_config=dict(name='storage-middle64'))
        self.reports = quality_reports(self.case['quality_test_config'])

    def test_actual_rust_producer_shapes_pass_numeric_gate_only(self):
        self.assertFalse(protocol.validate_quality_reports(self.reports,self.case)['performance_acceptance'])

    def test_unknown_partial_boolean_or_changed_quality_fields_fail_closed(self):
        changes = [('full40','passed',None),('full40','complete','true'),('fresh','quarantine',None),
                   ('full40','diagnostic_capture_only',True),('fresh','observer_mode','unknown'),
                   ('cold','cold_owner_restart',False),('fresh','logits_compared',1),('full40','metrics',[]),
                   ('fresh','state_bytes_after_shutdown',None),('fresh','atol',1),('cold','greedy_tie_policy','unknown')]
        for label,key,value in changes:
            changed = copy.deepcopy(self.reports)
            changed[label][key] = value
            with self.subTest(label=label,key=key),self.assertRaises(ValueError):
                protocol.validate_quality_reports(changed,self.case)

    def test_stored_reference_identity_mismatch_is_not_current_acceptance(self):
        self.reports['cold']['oracle_manifest_sha256'] = 'c'*64
        with self.assertRaisesRegex(ValueError,'reference mismatch'):
            protocol.validate_quality_reports(self.reports,self.case)

    def test_report_with_nonfinite_metric_fails_even_if_passed_true(self):
        self.reports['fresh']['metrics'] = [dict(failures=0,nonfinite=1)]*5427
        with self.assertRaises(ValueError):
            protocol.validate_quality_reports(self.reports,self.case)

    def test_missing_actual_token_checks_fail_even_if_passed_true(self):
        self.reports['full40']['next_token_checks'] = []
        with self.assertRaises(ValueError):
            protocol.validate_quality_reports(self.reports,self.case)

    def test_free_feedback_reset_and_fork_each_required(self):
        for key,value in [('free_greedy_status','unknown'),('reset_equal',None),('fork_snapshot_equal',None),
                          ('reset_generated_ids',[]),('reset_next_token_checks',[])]:
            changed = copy.deepcopy(self.reports)
            changed['cold']['cases'][0][key] = value
            with self.subTest(key=key),self.assertRaises(ValueError):
                protocol.validate_quality_reports(changed,self.case)

    def test_actual_step_index_and_id_not_inferred_from_case_label(self):
        changed = copy.deepcopy(self.reports)
        changed['fresh']['cases'][0]['next_token_checks'][0]['step'] = 9
        with self.assertRaises(ValueError):
            protocol.validate_quality_reports(changed,self.case)


if __name__ == '__main__':
    unittest.main()
