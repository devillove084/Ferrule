"""Additional CPU-only raw-evidence and full NVTX build-review regressions."""
import copy
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.dont_write_bytecode = True
STORAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STORAGE))
sys.path.insert(0, str(STORAGE / 'tests'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paired_read_v1 import protocol as p, quality, build_adapter, run_paired as run
from runnerhelpers.evidence_v2 import ct
import test_paired_read_v1 as shared
from test_build_receipts_v1 import BuildFixture


class RawQualityTests(shared.CPUOnly):
    def interval(self):
        read = shared.LedgerTests.ledger(self)['closure']['attempts'][0]['summary']['physical']['expert_read_pipeline']['value']
        before = copy.deepcopy(read)
        for key in ('submitted_reads', 'completed_reads', 'verified_reads', 'io_started_reads', 'proved_read_bytes'):
            before['stats'][key] = 0
        before['effective_mode'] = None
        before['worker_read_observed'] = False
        def endpoint(observation):
            return {key: dict(state='observed', value=value) for key, value in dict(
                scope=dict(owner=1, generation=1), prepared_plan_id=7,
                image_generation=3, expert_read_pipeline=observation).items()}
        return dict(before=endpoint(before), after=endpoint(read),
            delta={key: read['stats'][key] - before['stats'][key] for key in read['stats'] if not key.startswith('peak_')},
            positive_worker_read=True)

    def test_both_raw_endpoints_not_aggregate_pass_boolean(self):
        row = self.interval()
        options = p.case_for_mode('blocking')['read_options']
        quality.interval_endpoints(row, options)
        for endpoint in ('before', 'after'):
            for key, value in (('enabled', False), ('requested_mode', 'Lookahead'), ('status', 'quarantined')):
                bad = copy.deepcopy(row)
                bad[endpoint]['expert_read_pipeline']['value'][key] = value
                with self.subTest(endpoint=endpoint, key=key), self.assertRaises(ValueError):
                    quality.interval_endpoints(bad, options)
            bad = copy.deepcopy(row)
            bad[endpoint]['expert_read_pipeline']['value']['stats']['unknown_bytes'] = 1
            with self.assertRaisesRegex(ValueError, 'unknown'):
                quality.interval_endpoints(bad, options)

    def test_actual_owner_identity_delta_and_positive_read_denominators(self):
        options = p.case_for_mode('blocking')['read_options']
        for field in ('scope', 'image_generation', 'prepared_plan_id'):
            bad = self.interval()
            bad['after'][field] = dict(state='unobserved')
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'identity'):
                quality.interval_endpoints(bad, options)
        bad = self.interval()
        bad['after']['scope']['value']['generation'] = 2
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            quality.interval_endpoints(bad, options)
        bad = self.interval()
        bad['delta']['submitted_reads'] = 0
        with self.assertRaisesRegex(ValueError, 'delta mismatch'):
            quality.interval_endpoints(bad, options)

    def test_current_charge_bound_and_final_pending_are_not_repaired_by_peak(self):
        value = shared.LedgerTests.ledger(self)['closure']['attempts'][0]['summary']['physical']['expert_read_pipeline']['value']
        options = p.case_for_mode('blocking')['read_options']
        value['current_charge']['host_charged_bytes'] = 1
        with self.assertRaisesRegex(ValueError, 'budget'):
            quality.read_observation(value, options)
        value['current_charge']['host_charged_bytes'] = 0
        value['status'] = 'join_pending_at_scope_end'
        with self.assertRaisesRegex(ValueError, 'pending'):
            quality.read_observation(value, options, final=True)

    def test_runtime_nvtx_off_verified_in_actual_maps_path(self):
        root = dict(pid=12345)
        with mock.patch.object(run.ex.observer, 'verify_root'), mock.patch.object(run.ex, 'library_maps', return_value={'maps': 'fixture'}), \
                mock.patch.object(Path, 'read_text', return_value='FERRULE_PROFILE_NVTX=1\0'):
            with run.maintained_worker_adapter():
                with self.assertRaisesRegex(ValueError, 'NVTX must be OFF'):
                    run.ex.library_maps(root, {})
        with mock.patch.object(run.ex.observer, 'verify_root'), mock.patch.object(run.ex, 'library_maps', return_value={'maps': 'fixture'}), \
                mock.patch.object(Path, 'read_text', return_value='FERRULE_PROFILE_NVTX=0\0'):
            with run.maintained_worker_adapter():
                self.assertEqual(run.ex.library_maps(root, {}), {'maps': 'fixture'})


class FullBuildReviewTests(shared.CPUOnly):
    def fixture(self):
        f = BuildFixture(self.directory)
        f.candidate['build_adapter'] = build_adapter.ADAPTER
        f.build['schema'] = build_adapter.BUILD_SCHEMA
        f.build['argv'][9] = 'cuda,nvtx-profile'
        f.build['outer_supervisor_argv'] = ['/usr/bin/timeout', '1800', *f.build['argv']]
        f.build['environment']['FERRULE_NVTX_INCLUDE_DIR'] = str(f.root / 'nvtx-headers')
        header = f.file('header', f.root / 'nvtx-headers/nvtx3/nvToolsExt.h', 'CPU fixture only, never compiled', 0o444)
        group = f.proof['compiler-input-pins.json']
        group['pins'][str(header)] = ct.sha(header)
        group['matched'] = len(group['pins'])
        ref = f.receipt('reference:compiler-input-pins.json', dict(pins=group['pins']))
        group['reference_receipt_sha256'] = ct.sha(ref)
        f.evidence('compiler_pins', f.proof)
        f.artifact['compiled_artifact']['features'] = ['cuda', 'nvtx-profile']
        f.evidence('artifact', f.artifact)
        stdout_path = Path(f.candidate['build_evidence']['build_stdout'])
        lines = [json.loads(line) for line in stdout_path.read_text().splitlines()]
        for row in lines:
            if row.get('reason') == 'compiler-artifact' and row.get('executable'):
                row['features'] = ['cuda', 'nvtx-profile']
        f.file('build_stdout', stdout_path, '\n'.join(json.dumps(row) for row in lines) + '\n')
        f.build['full_stdout_sha256'] = ct.sha(stdout_path)
        f.candidate['build_receipt'] = str(f.receipt('build', f.build))
        return f

    def test_full_review_reuses_source_archive_native_dependencies_and_help(self):
        f = self.fixture()
        result = build_adapter.review_cli_build(f.candidate, f.build, f.case)
        self.assertEqual(result['adapter'], build_adapter.ADAPTER)
        self.assertEqual(result['sources'], f.sources)
        self.assertFalse(result['quality_approved'])
        self.assertFalse(result['gpu_authorized'])
        self.assertEqual(result['coverage']['loaded_model_runtime'], 'unobserved')

    def test_extra_unpinned_header_rejected(self):
        f = self.fixture()
        (f.root / 'nvtx-headers/nvtx3/unpinned.h').write_text('CPU fixture extra header')
        with self.assertRaisesRegex(ValueError, 'header tree unpinned'):
            build_adapter.review_cli_build(f.candidate, f.build, f.case)

    def test_reused_cuda_only_artifact_is_not_normalized(self):
        f = self.fixture()
        f.artifact['compiled_artifact']['features'] = ['cuda']
        f.evidence('artifact', f.artifact)
        with self.assertRaisesRegex(ValueError, 'artifact mismatch'):
            build_adapter.review_cli_build(f.candidate, f.build, f.case)


if __name__ == '__main__':
    unittest.main()
