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


# Synthetic v3 fixtures below are CPU-only mutations, never GPU evidence.
def options(mode='blocking', constructor=False):
    return dict(schedule=mode, max_host_bytes=(2 if constructor else 64) << 20,
        max_metadata_bytes=(256 << 10) if constructor else 8 << 20,
        worker_stack_bytes=(1 if constructor else 2) << 20)


def observation(opts, *, configured=False, reads=6, reuses=1):
    stats = dict.fromkeys(quality.V3_COUNTERS + quality.V3_PEAKS, 0)
    charge = dict.fromkeys(quality.V3_CHARGES, 0)
    if not configured:
        idle = opts['max_metadata_bytes'] + opts['worker_stack_bytes']
        stats.update(sequences_started=1 + reuses, sequences_finished=1 + reuses,
            workers_created=1, worker_reuses=reuses, submitted_reads=reads, completed_reads=reads,
            verified_reads=reads, worker_started_reads=reads, io_started_reads=reads,
            proved_read_bytes=reads * 128, reader_interval_calls=reads, reader_interval_bytes=reads * 128,
            peak_packed_bytes=128, peak_artifact_bytes=128, peak_payload_bytes=256,
            peak_metadata_bytes=opts['max_metadata_bytes'], peak_requested_stack_bytes=opts['worker_stack_bytes'],
            peak_host_charged_bytes=idle + 256)
        charge.update(metadata_bytes=opts['max_metadata_bytes'], requested_stack_bytes=opts['worker_stack_bytes'],
            host_charged_bytes=idle)
    return dict(enabled=True, requested_mode=opts['schedule'], effective_mode=None if configured else opts['schedule'],
        worker_read_observed=not configured, status='configured' if configured else 'idle_retained',
        limits=copy.deepcopy(opts), current_charge=charge, stats=stats,
        worker_alive=not configured, worker_join_pending=False, worker_unknown=False,
        current_read=None, pending_read=None, last_receipt=None, **quality.V3_SCOPES)


def physical(read):
    return dict(scope=dict(state='observed', value=dict(owner=1, generation=1)),
        prepared_plan_id=dict(state='observed', value=7), image_generation=dict(state='observed', value=3),
        expert_read_pipeline=dict(state='observed', value=read), release=dict(state='unobserved'))


def interval(opts, before=None, after=None):
    before = observation(opts, configured=True) if before is None else before
    after = observation(opts) if after is None else after
    delta = {k: after['stats'][k] - before['stats'][k] for k in quality.V3_COUNTERS}
    delta.update({k: after['stats'][k] for k in quality.V3_PEAKS})
    positive = all(delta[k] > 0 for k in
        ('submitted_reads', 'completed_reads', 'verified_reads', 'io_started_reads', 'proved_read_bytes'))
    return dict(requested=copy.deepcopy(opts), before=physical(before), after=physical(after), delta=delta,
        operation='prediction', positive_worker_read=positive, same_owner=True, configuration_matches=True,
        within_charged_limits=True, unambiguous=True, evidence_complete=True,
        configuration_pair={name: dict(configuration_matches=True, within_charged_limits=True,
            unambiguous=True, worker_snapshot_valid=True, worker_snapshot_rejection=None)
            for name in ('before', 'after')})


def report(mode='blocking'):
    opts = options(mode)
    first = interval(opts)
    idle = copy.deepcopy(first['after']['expert_read_pipeline']['value'])
    second = interval(opts, idle, observation(opts, reads=12, reuses=3))
    return dict(expert_read_quality=dict(schema='ferrule.expert-read-quality-evidence.v1',
        interval_validation_revision=3, worker_snapshot_contract_revision=1, passed=True,
        requested=opts, sampled_operations=2, positive_worker_read_calls=2,
        charge_basis=quality.V3_SCOPES['charge_basis'], calls=[first, second]))


def constructor():
    cells = []
    for boundary in ('W8A16Bf16F32Accumulate', 'Bf16ModelF32Accumulate'):
        for mode in ('blocking', 'lookahead'):
            opts = options(mode, constructor=True)
            row = interval(opts)
            before = row['before']['expert_read_pipeline']['value']
            after = row['after']['expert_read_pipeline']['value']
            cells.append(dict(model_boundary=boundary, requested=opts, configured=copy.deepcopy(before),
                before_prediction=row['before'], after_prediction=row['after'], observed=copy.deepcopy(after),
                delta=row['delta'], passed=True, same_owner=True, positive_worker_read=True,
                charged_limits_satisfied=True, logit_values_checked=32, state_bytes_after_shutdown=0,
                quarantine=False, host_after_prediction_stage='payloads_released_worker_idle_retained',
                host_after_shutdown_before_owner_drop=copy.deepcopy(after), host_after_final_owner_drop=None,
                host_final_join_witness=quality.CONSTRUCTOR_DROP_UNOBSERVED))
    return dict(schema='ferrule.expert-read-constructor-quality.v1', host_lifecycle_contract_revision=2,
        host_charge_basis=quality.V3_SCOPES['charge_basis'],
        worker_counter_scope='full_owner_lifetime_created_reuses_joined_not_submitted_reads',
        complete=True, passed=True, expected_cells=4, total_logit_values_checked=128,
        diagnostic_trace_installed=False, ordinary_no_observer_claim=True, cells=cells)


class LifecycleV3Tests(shared.CPUOnly):
    def check(self, value, mode):
        quality.read_quality_v3(value, mode)

    def mutate_endpoint(self, mutate):
        for mode in ('blocking', 'lookahead'):
            for endpoint in ('before', 'after'):
                value = report(mode)
                # Both endpoints of the second interval are real retained owners.
                read = value['expert_read_quality']['calls'][1][endpoint]['expert_read_pipeline']['value']
                mutate(read)
                with self.subTest(mode=mode, endpoint=endpoint), self.assertRaises(ValueError):
                    self.check(value, mode)

    def test_v3_typed_field_sets_match_the_current_maintained_producer(self):
        import re
        source = (ct.ROOT / 'crates/ferrule-model/src/transformer/expert_reads.rs').read_text()
        def fields(name):
            body = source.split('pub struct ' + name + ' {', 1)[1].split('\n}', 1)[0]
            return set(re.findall(r'^\s*pub (\w+):', body, re.MULTILINE))
        self.assertEqual(set(quality.V3_COUNTERS + quality.V3_PEAKS), fields('ExpertReadPipelineStats'))
        self.assertEqual(set(quality.V3_CHARGES), fields('ExpertReadHostCharge'))

    def test_valid_reuse_and_exact_ten_mib_both_schedules(self):
        for mode in ('blocking', 'lookahead'):
            value = report(mode)
            self.check(value, mode)
            for endpoint in ('before', 'after'):
                read = value['expert_read_quality']['calls'][1][endpoint]['expert_read_pipeline']['value']
                self.assertEqual(read['current_charge']['host_charged_bytes'], 10 << 20)
                self.assertEqual((read['stats']['workers_created'], read['stats']['workers_joined']), (1, 0))

    def test_readonly_idle_samples_preserve_lifetime_peaks_and_do_not_join(self):
        for mode in ('blocking', 'lookahead'):
            value = report(mode)
            idle = copy.deepcopy(value['expert_read_quality']['calls'][0]['after']['expert_read_pipeline']['value'])
            readonly = interval(options(mode), copy.deepcopy(idle), copy.deepcopy(idle))
            readonly['operation'] = 'diagnostic_state_snapshot'
            value['expert_read_quality']['calls'][1] = readonly
            value['expert_read_quality']['positive_worker_read_calls'] = 1
            before = copy.deepcopy(value)
            self.check(value, mode)
            self.assertEqual(value, before, 'validation must not mutate/refund raw evidence')

    def test_malformed_report_and_endpoint_pair_fail_closed(self):
        for value in (None, [], {}, dict(expert_read_quality=None), dict(expert_read_quality=[])):
            with self.assertRaises(ValueError):
                self.check(value, 'blocking')
        for pair in (None, [], dict(before=None, after=None)):
            value = report()
            value['expert_read_quality']['calls'][0]['configuration_pair'] = pair
            with self.assertRaises(ValueError):
                self.check(value, 'blocking')
        value = report()
        value['expert_read_quality']['calls'][0]['configuration_pair']['after']['worker_snapshot_rejection'] = 'poisoned'
        with self.assertRaises(ValueError):
            self.check(value, 'blocking')

    def test_revision_two_missing_and_malformed_revision_never_upgrade(self):
        for field in ('interval_validation_revision', 'worker_snapshot_contract_revision'):
            for bad in (None, False, 0, 2, '3', 3.0):
                value = report()
                value['expert_read_quality'][field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    self.check(value, 'blocking')
            value = report()
            del value['expert_read_quality'][field]
            with self.assertRaises(ValueError):
                self.check(value, 'blocking')

    def test_unknown_and_pending_are_sticky_even_with_zero_unknown_bytes(self):
        for field in ('worker_unknown', 'worker_join_pending'):
            self.mutate_endpoint(lambda read: read.update({field: True}))
        for field in ('join_failed_sequences', 'join_pending_sequences', 'unknown_reads', 'unknown_bytes',
                      'unobserved_terminal_reads', 'unobserved_reader_intervals'):
            self.mutate_endpoint(lambda read: read['stats'].update({field: 1}))

    def test_status_allowlist_does_not_accept_report_flags(self):
        for status in ('active', 'closed_retained', 'quarantined', 'unknown_ledger_poisoned',
                       'unknown_observation_poisoned', 'join_pending_at_scope_end', 'disabled', 'new_status', None):
            self.mutate_endpoint(lambda read: read.update(status=status))

    def test_missing_typed_fields_rejected_at_either_endpoint(self):
        for field in ('worker_alive', 'worker_unknown', 'worker_join_pending', 'current_read', 'pending_read',
                      'stats', 'current_charge', 'limits', 'requested_mode', 'effective_mode', 'last_receipt'):
            self.mutate_endpoint(lambda read: read.pop(field))
        for field in quality.V3_COUNTERS + quality.V3_PEAKS:
            self.mutate_endpoint(lambda read: read['stats'].pop(field))

    def test_exact_charge_all_components_including_payload_slots_and_artifacts(self):
        for field in quality.V3_CHARGES:
            self.mutate_endpoint(lambda read: read['current_charge'].update({field: read['current_charge'][field] + 1}))
        for field in ('metadata_bytes', 'requested_stack_bytes', 'host_charged_bytes'):
            self.mutate_endpoint(lambda read: read['current_charge'].update({field: read['current_charge'][field] - 1}))
        for field in ('current_read', 'pending_read'):
            self.mutate_endpoint(lambda read: read.update({field: dict(operation=1)}))

    def test_worker_count_liveness_and_read_conservation_mutations(self):
        for field in ('workers_created', 'workers_joined', 'worker_reuses', 'sequences_started',
                      'sequences_finished', 'submitted_reads', 'completed_reads', 'verified_reads',
                      'worker_started_reads', 'io_started_reads'):
            self.mutate_endpoint(lambda read: read['stats'].update({field: read['stats'][field] + 1}))
        self.mutate_endpoint(lambda read: read.update(worker_alive=False))
        self.mutate_endpoint(lambda read: read.update(worker_read_observed=False))

    def test_numeric_types_and_overflow_are_not_zero_defaults(self):
        for value in (False, None, -1, 0.0, '0', 1 << 64):
            self.mutate_endpoint(lambda read: read['stats'].update(workers_created=value))
        for value in (0, 1, None, 'false'):
            self.mutate_endpoint(lambda read: read.update(worker_unknown=value))

    def test_peak_bounds_and_exact_metadata_stack_peaks(self):
        for field in quality.V3_PEAKS:
            self.mutate_endpoint(lambda read: read['stats'].update({field: 1 << 40}))
        for field in ('peak_metadata_bytes', 'peak_requested_stack_bytes', 'peak_host_charged_bytes'):
            self.mutate_endpoint(lambda read: read['stats'].update({field: 0}))

    def test_raw_identity_release_and_endpoint_flags_not_inferred(self):
        for endpoint in ('before', 'after'):
            for field in ('scope', 'prepared_plan_id', 'image_generation', 'expert_read_pipeline'):
                value = report()
                value['expert_read_quality']['calls'][1][endpoint][field] = dict(state='unobserved')
                with self.subTest(endpoint=endpoint, field=field), self.assertRaises(ValueError):
                    self.check(value, 'blocking')
            value = report()
            value['expert_read_quality']['calls'][1][endpoint]['release'] = dict(state='observed', value='released')
            with self.assertRaises(ValueError):
                self.check(value, 'blocking')
            for field in ('worker_snapshot_valid', 'worker_snapshot_rejection'):
                value = report()
                del value['expert_read_quality']['calls'][1]['configuration_pair'][endpoint][field]
                with self.assertRaises(ValueError):
                    self.check(value, 'blocking')

    def test_full_delta_required_and_all_worker_counter_resets_rejected(self):
        for field in quality.V3_COUNTERS + quality.V3_PEAKS:
            value = report()
            del value['expert_read_quality']['calls'][1]['delta'][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(value, 'blocking')
        for field in ('workers_created', 'worker_reuses', 'workers_joined', 'sequences_started'):
            value = report()
            value['expert_read_quality']['calls'][1]['delta'][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(value, 'blocking')

    def test_real_model_scopes_must_actually_demonstrate_reuse(self):
        value = report()
        opts = options()
        one_scope = observation(opts, reuses=0)
        value['expert_read_quality'].update(calls=[interval(opts, after=one_scope)],
            sampled_operations=1, positive_worker_read_calls=1)
        with self.assertRaisesRegex(ValueError, 'demonstrate worker reuse'):
            self.check(value, 'blocking')

    def test_owner_restart_and_counter_reset_between_intervals_rejected(self):
        value = report()
        row = interval(options())
        value['expert_read_quality']['calls'][1] = row
        with self.assertRaises(ValueError):
            self.check(value, 'blocking')
        value = report()
        for endpoint in ('before', 'after'):
            value['expert_read_quality']['calls'][1][endpoint]['scope']['value']['owner'] = 2
        with self.assertRaises(ValueError):
            self.check(value, 'blocking')

    def test_final_join_only_with_explicit_witness_and_exact_one_join_delta(self):
        for mode in ('blocking', 'lookahead'):
            opts = options(mode)
            idle = observation(opts)
            ended = copy.deepcopy(idle)
            ended.update(status='closed', worker_alive=False, current_charge=dict.fromkeys(quality.V3_CHARGES, 0))
            ended['stats']['workers_joined'] = 1
            quality.final_owner_join_v3(idle, ended, opts)
            for witness in (None, idle, dict(ended, worker_join_pending=True), dict(ended, worker_unknown=True)):
                with self.assertRaises(ValueError):
                    quality.final_owner_join_v3(idle, witness, opts)
            for field in quality.V3_COUNTERS + quality.V3_PEAKS:
                bad = copy.deepcopy(ended)
                bad['stats'][field] += 1
                with self.subTest(field=field), self.assertRaises(ValueError):
                    quality.final_owner_join_v3(idle, bad, opts)
            row = interval(opts, idle, ended)
            with self.assertRaises(ValueError):
                quality.interval_endpoints_v3(row, opts)

    def test_no_vacuous_positive_or_sample_denominators(self):
        for field in ('positive_worker_read_calls', 'sampled_operations'):
            for value in (0, True, None, 3):
                bad = report()
                bad['expert_read_quality'][field] = value
                with self.assertRaises(ValueError):
                    self.check(bad, 'blocking')
        bad = report()
        bad['expert_read_quality']['calls'] = []
        with self.assertRaises(ValueError):
            self.check(bad, 'blocking')

    def test_four_constructor_cells_keep_fixture_budget_and_unobserved_postdrop(self):
        value = constructor()
        quality.constructor_v3(value)
        self.assertEqual(len(value['cells']), 4)
        for cell in value['cells']:
            self.assertEqual(cell['observed']['current_charge']['host_charged_bytes'], 1280 << 10)
            self.assertIsNone(cell['host_after_final_owner_drop'])

    def _reject_constructor_zero_reader_interval(self, mode, counter):
        value = constructor()
        for cell in value['cells']:
            if cell['requested']['schedule'] != mode:
                continue
            after = cell['after_prediction']['expert_read_pipeline']['value']
            for observed in (cell['observed'], after, cell['host_after_shutdown_before_owner_drop']):
                observed['stats'][counter] = 0
            before = cell['before_prediction']['expert_read_pipeline']['value']
            cell['delta'][counter] = after['stats'][counter] - before['stats'][counter]
            self.assertEqual(cell['observed'], after)
            self.assertEqual(cell['host_after_shutdown_before_owner_drop'], after)
            self.assertEqual(cell['delta'][counter], 0)
        with self.assertRaisesRegex(ValueError, 'constructor read/worker reuse denominator differs'):
            quality.constructor_v3(value)

    def test_constructor_zero_reader_interval_calls_blocking(self):
        self._reject_constructor_zero_reader_interval('blocking', 'reader_interval_calls')

    def test_constructor_zero_reader_interval_calls_lookahead(self):
        self._reject_constructor_zero_reader_interval('lookahead', 'reader_interval_calls')

    def test_constructor_zero_reader_interval_bytes_blocking(self):
        self._reject_constructor_zero_reader_interval('blocking', 'reader_interval_bytes')

    def test_constructor_zero_reader_interval_bytes_lookahead(self):
        self._reject_constructor_zero_reader_interval('lookahead', 'reader_interval_bytes')

    def test_constructor_never_fabricates_postdrop_zero_or_join(self):
        for value in (0, {}, dict(status='closed'), dict(state='observed', value=0)):
            bad = constructor()
            bad['cells'][0]['host_after_final_owner_drop'] = value
            with self.assertRaises(ValueError):
                quality.constructor_v3(bad)
        for field in ('host_after_final_owner_drop', 'host_final_join_witness'):
            bad = constructor()
            del bad['cells'][0][field]
            with self.assertRaises(ValueError):
                quality.constructor_v3(bad)
        bad = constructor()
        bad['cells'][0]['host_final_join_witness'] = 'joined'
        with self.assertRaises(ValueError):
            quality.constructor_v3(bad)

    def test_constructor_lifecycle_copy_budget_and_stage_mutations(self):
        for field in ('configured', 'observed', 'host_after_shutdown_before_owner_drop'):
            bad = constructor()
            bad['cells'][0][field]['worker_unknown'] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                quality.constructor_v3(bad)
        for field, value in (('host_lifecycle_contract_revision', 1), ('total_logit_values_checked', 0)):
            bad = constructor()
            bad[field] = value
            with self.assertRaises(ValueError):
                quality.constructor_v3(bad)
        bad = constructor()
        bad['cells'][0]['requested']['max_host_bytes'] += 1
        with self.assertRaises(ValueError):
            quality.constructor_v3(bad)


if __name__ == '__main__':
    unittest.main()
