"""CPU-only policy tests. Synthetic in-memory reports are not candidate evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import decoded_cache_gate as gate
import decoded_cache_compare as adapter


def metric():
    return dict(exact=True, failures=0, max_abs=0)


def full_report():
    return dict(passed=True, cases={name: dict(metrics={str(i): metric() for i in range(2178)},
        greedy_ids_match=[True, True]) for name in ('hello', 'capital')})


def resources():
    return dict(owners=[dict(owner=i, cache_peak_grants=2 << 30, cache_grant=3 << 30,
        full_total_with_margins_and_trace=22_584_094_720, new_artifact_h2d_bytes=0) for i in (0, 1)])


class GatePolicy(unittest.TestCase):
    def test_manifest_is_default_off_and_has_six_recipes(self):
        value = gate.manifest()
        self.assertFalse(value['public_enabled'])
        self.assertEqual(len(gate.RECIPES), 6)
        self.assertEqual(len(gate.SMALL_CASES), 19)
        self.assertEqual(len(gate.CPU), 6)
        self.assertEqual(value['cache_grants_per_owner'], 3 << 30)
        self.assertIn('plan_only', value['status'])

    def test_arena_cpu_contract_requires_fit_set_and_retained_root_gates(self):
        required = gate.manifest()['arena_required']
        self.assertEqual(required['physical_root_cap_bytes'], 3 << 30)
        self.assertEqual(required['hot_fit_evictions'], 0)
        self.assertEqual(required['hot_fit_decodes'], 0)
        self.assertTrue(set(gate.ARENA_HOST).issubset(set(gate.HOST_SMALL)))
        names = {selector.rsplit('::', 1)[-1] for _, selector in gate.ARENA_HOST}
        self.assertIn('pair_down_fit_set_is_stable_after_warmup_and_real_3366_mib_set_evicts', names)
        self.assertIn('every_class_at_capacity_has_no_warm_eviction_with_healthy_boundaries', names)
        self.assertIn('unknown_and_partial_install_failure_retain_root_and_all_slots', names)
        self.assertIn('decoded_snapshot_partial_view_failure_partitions_installed_root_once', names)

    def test_arena_fit_gpu_selector_and_roster_are_bound_not_a_claim_of_gpu_pass(self):
        kind, selector, owners = gate.ARENA_SMALL_GATES['model-arena-fit']
        self.assertEqual(kind, 'model')
        self.assertEqual(selector, gate.MODEL_PREFIX + 'decoded_arena_fit_working_set_hot_zero_eviction_decode_fallback')
        self.assertEqual(owners, 1)
        self.assertEqual(gate.execution_devices('model-arena-fit')['selected'], gate.UUIDS[:1])
        self.assertEqual(gate.ARENA_SMALL_GATES['model-arena-lru'][2], 2)
        self.assertEqual(gate.ARENA_SMALL_GATES['model-arena-alias'][2], 1)

    def test_changed_plan_or_missing_cpu_receipts_cannot_reseal_old_preparation(self):
        receipts = [dict(path=str(i)) for i in range(len(gate.HOST_SMALL)+len(gate.CPU))]
        with patch.object(gate.ct, 'load', return_value=gate.manifest()):
            gate.validate_required_plan(Path('unused'), dict(cpu_receipts=receipts))
            for bad in (receipts[:-1], receipts[:-1]+receipts[:1]):
                with self.assertRaises(ValueError):
                    gate.validate_required_plan(Path('unused'), dict(cpu_receipts=bad))
        old = gate.manifest()
        old['small_selectors'] = gate.HISTORICAL_SMALL_GATES
        with patch.object(gate.ct, 'load', return_value=old), self.assertRaisesRegex(ValueError, 'fresh build'):
            gate.validate_required_plan(Path('unused'), dict(cpu_receipts=receipts))

    def test_backend_reuse_is_only_the_two_file_model_scope_fix_and_exact_binaries(self):
        sources = {p:'old' for p in gate.SCOPE_FIX_SOURCES}
        sources['backend'] = 'unchanged'
        old = dict(sources=sources, tools={'compiler':'same'}, binaries={
            k:dict(sha256=k) for k in ('backend','backend-slice','model')})
        current = copy.deepcopy(old)
        for p in gate.SCOPE_FIX_SOURCES: current['sources'][p] = 'new'
        current['binaries']['model']['sha256'] = 'new-model'
        self.assertEqual(set(gate.backend_reuse_relation(old,current)), set(gate.SCOPE_FIX_SOURCES))
        for mutation in ('backend-source','extra-source','backend-binary','slice-binary','old-model','tools'):
            bad = copy.deepcopy(current)
            if mutation == 'backend-source': bad['sources']['backend'] = 'changed'
            if mutation == 'extra-source': bad['sources']['other'] = 'new'
            if mutation == 'backend-binary': bad['binaries']['backend']['sha256'] = 'changed'
            if mutation == 'slice-binary': bad['binaries']['backend-slice']['sha256'] = 'changed'
            if mutation == 'old-model': bad['binaries']['model']['sha256'] = old['binaries']['model']['sha256']
            if mutation == 'tools': bad['tools']['compiler'] = 'changed'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                gate.backend_reuse_relation(old,bad)

    def test_indexed_cpu_equivalence_and_benchmark_are_explicitly_required(self):
        for selector in gate.GROUPED_CPU:
            self.assertIn(('model', selector), gate.HOST_SMALL)
        self.assertEqual(gate.IGNORED_HOST, {('model', gate.GROUPED_BENCH)})
        self.assertIn(('model', gate.GROUPED_BENCH), gate.HOST_SMALL)
        gate.validate_indexed_sources(gate.INDEXED_LRU_SOURCES)
        for path in gate.INDEXED_LRU_SOURCES:
            changed = dict(gate.INDEXED_LRU_SOURCES)
            changed[path] = '0'*64
            with self.assertRaises(ValueError): gate.validate_indexed_sources(changed)
        with self.assertRaises(ValueError): gate.validate_indexed_sources({})

    def test_cpu_benchmark_receipt_keeps_scope_and_never_gates_on_timing(self):
        rows = ['CPU only indexed='+flag+' entries=1155 classes=2 descriptors=20000 misses=50000 wall=999s (not E2E)'
                for flag in ('false','true')]
        self.assertEqual(gate.benchmark_summary('\n'.join(rows)), rows)
        for text in (rows[0], '\n'.join(reversed(rows)), '\n'.join(rows).replace('misses=50000','misses=50'),
                     '\n'.join(rows).replace('not E2E','E2E'), '\n'.join(rows+rows)):
            with self.subTest(text=text), self.assertRaises(ValueError): gate.benchmark_summary(text)

    def test_no_owner_grant_means_no_queries_locks_or_launch(self):
        with patch.object(gate, 'run_directory') as directory, patch.object(gate, 'idle') as idle:
            with self.assertRaises(ValueError):
                gate.run('unused', 'full40', None, False, 'not a grant')
            directory.assert_not_called()
            idle.assert_not_called()

    def test_empty_owner_note_or_unknown_selector_rejects_before_io(self):
        with patch.object(gate, 'run_directory') as directory:
            for name, note in [('full40', ''), ('unlisted', 'fresh explicit grant')]:
                with self.assertRaises(ValueError):
                    gate.run('unused', name, None, True, note)
            directory.assert_not_called()

    def test_historical_contract_is_frozen_but_current_selectors_follow_actual_arena_api(self):
        self.assertEqual(list(gate.ALL_GATES)[:17], [
            'backend-exact','backend-preflight','backend-typed-view','backend-arena-allocation','backend-arena-errors',
            'model-cold-hot-fit','model-fixed-pins','model-unknown-panic','model-physical-full',
            'model-arena-fit','model-arena-lru','model-arena-alias','model-fixed-unknown',
            'model-uninitialized-views','model-first-fixed-failure','model-uninitialized-slots','full40'])
        self.assertEqual(len(gate.HISTORICAL_ALL_GATES), 13)
        self.assertEqual(len(gate.ALL_GATES), 23)
        self.assertEqual(sum(v[0] == 'model' for v in gate.SMALL_GATES.values()), 11)
        self.assertIn('cold_hot_reload', gate.HISTORICAL_ALL_GATES['model-cold-hot-reload'])
        self.assertIn('cold_hot_fit', gate.ALL_GATES['model-cold-hot-fit'])
        self.assertTrue(any('hybrid_parallel::mixed_decoded_arena::' in s for _, s in gate.ARENA_HOST))
        self.assertFalse(any('standard::mixed_decoded_arena::' in s for _, s in gate.HOST_SMALL))

    def test_uninitialized_boundary_guards_and_new_microtests_are_required(self):
        required = gate.manifest()['arena_required']
        self.assertEqual(required['allocation_policy'], 'dedicated_uninitialized_root_retained_before_initialization')
        self.assertTrue(required['driver_growth_equals_full_root_grant'])
        for name in ('arena_checks_word_bytes_rounded_grant_and_budget',
                     'arena_has_no_initialization_path_before_return_or_on_error'):
            self.assertIn(('backend-arena', name), gate.HOST_SMALL)
        for name in ('dedicated_arena_handoff_precedes_any_post_allocation_observation',
                     'slot_readiness_follows_full_decode_and_completion_classification'):
            self.assertIn(('model-handoff', name), gate.HOST_SMALL)
        for name in ('backend-arena-allocation','backend-arena-errors',
                     'model-uninitialized-views','model-first-fixed-failure','model-uninitialized-slots'):
            self.assertEqual(gate.execution_devices(name)['selected'], gate.UUIDS[:1])

    def test_every_gate_uses_one_shared_visible_and_selected_roster(self):
        for name in gate.ALL_GATES:
            with self.subTest(gate=name):
                devices = gate.execution_devices(name)
                for key in ('CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES'):
                    self.assertEqual(devices['environment'][key].split(','), devices['selected'])
                self.assertEqual(gate.validate_execution_devices(name, devices), devices)
        for name in ('backend-exact', 'backend-preflight', 'model-unknown-panic', 'model-arena-fit', 'model-arena-alias', 'model-fixed-unknown', 'backend-typed-view'):
            self.assertEqual(gate.execution_devices(name)['selected'], gate.UUIDS[:1])
        for name in ('model-cold-hot-fit', 'model-fixed-pins', 'model-physical-full', 'model-arena-lru', *gate.GATES):
            self.assertEqual(gate.execution_devices(name)['selected'], gate.UUIDS)

    def test_visible_roster_mismatch_fails_before_test_launch(self):
        devices = gate.execution_devices('backend-exact')
        devices['environment']['CUDA_VISIBLE_DEVICES'] = ','.join(gate.UUIDS)
        with self.assertRaises(ValueError):
            gate.validate_execution_devices('backend-exact', devices)
        with self.assertRaises(ValueError):
            gate.validate_execution_devices('full40', gate.execution_devices('backend-exact'))
        self.assertEqual(gate.observer.NONSELECTED_DELTA_TOLERANCE_BYTES, 1 << 20)

    def test_small_receipts_cannot_be_relabelled_across_sources_binaries_or_launches(self):
        sources = {'source': 'current'}
        rec = lambda name: dict(path=name, sha256=name + '-pin')
        row = dict(selector='actual::selector', preparation=rec('prepared'), supervisor=rec('supervisor'),
                   binary=rec('binary'), test_exit=rec('exit'), evidence=rec('evidence'))
        base = {
            'qualification': dict(schema='ferrule.paper.decoded-cache-small.v1', sources=sources,
                artifact_pins={}, cases={name: row for name in sorted(gate.SMALL_CASES)}),
            'prepared': dict(sources=sources, binaries={'model': row['binary']}),
            'supervisor': dict(handshake=dict(launch_id='actual-launch')),
            'exit': dict(returncode=0, one_test_passed=True, selector=row['selector']),
            'evidence': dict(passed=True, owners=[0, 1], coverage=sorted(gate.SMALL_CASES)[0],
                binary_sha256=row['binary']['sha256'], source_digest=gate.ct.digest(sources), launch_id='actual-launch'),
        }
        changes = [('prepared', 'sources', {'source': 'old'}),
                   ('evidence', 'coverage', 'different-case'),
                   ('evidence', 'binary_sha256', 'old-binary'),
                   ('evidence', 'source_digest', 'old-source'),
                   ('evidence', 'launch_id', 'old-launch')]
        for section, key, value in changes:
            with self.subTest(key=key):
                data = copy.deepcopy(base)
                data[section][key] = value
                with patch.object(gate.ct, 'load', side_effect=lambda p: data[str(p)]), \
                     patch.object(gate.ct, 'check_pins'), \
                     patch.object(gate.ct, 'successful_process', return_value=True), \
                     patch.object(gate, 'checked', side_effect=lambda r: Path(r['path'])):
                    with self.assertRaises(ValueError):
                        gate.validate_small('qualification', sources)

    def test_full40_requires_2178_each_not_tolerance_only(self):
        report = full_report()
        gate.exact_inventory(report, 'full40')
        for field, bad in [('exact', False), ('failures', 1), ('max_abs', 1e-12)]:
            changed = copy.deepcopy(report)
            changed['cases']['capital']['metrics']['0'][field] = bad
            with self.assertRaises(ValueError):
                gate.exact_inventory(changed, 'full40')
        del report['cases']['hello']['metrics']['0']
        with self.assertRaises(ValueError):
            gate.exact_inventory(report, 'full40')

    def test_wrong_greedy_does_not_pass_exact_tensors(self):
        report = full_report()
        report['cases']['hello']['greedy_ids_match'] = [True, False]
        with self.assertRaises(ValueError):
            gate.exact_inventory(report, 'full40')

    def test_lifecycle_requires_all_calls_and_fields(self):
        row = dict(greedy_match=True, metrics={str(i): metric() for i in range(1089)})
        report = dict(passed=True, calls={str(i): copy.deepcopy(row) for i in range(10)})
        gate.exact_inventory(report, 'greedy4')
        del report['calls']['0']
        with self.assertRaises(ValueError):
            gate.exact_inventory(report, 'greedy4')

    def test_both_owner_ledgers_include_cache_full_total_and_no_artifact_upload(self):
        gate.validate_resources(resources())
        for key, value in [('cache_peak_grants', (3 << 30) + 1), ('cache_grant', 0),
                           ('full_total_with_margins_and_trace', (22 << 30) + 1),
                           ('new_artifact_h2d_bytes', 1), ('cache_peak_grants', -1)]:
            changed = resources()
            changed['owners'][1][key] = value
            with self.assertRaises(ValueError):
                gate.validate_resources(changed)
        changed = resources()
        changed['owners'].pop()
        with self.assertRaises(ValueError):
            gate.validate_resources(changed)

    def test_host_counters_or_partial_trace_are_not_gpu_phase_evidence(self):
        report = dict(measurement='host_misses', complete=True, rows=[])
        with self.assertRaises(ValueError):
            gate.validate_phase_decode(report)
        report.update(measurement='gpu_correlated_launches', complete=False)
        with self.assertRaises(ValueError):
            gate.validate_phase_decode(report)

    def test_phase_counts_must_drop_on_both_owners_and_phases(self):
        report = dict(measurement='gpu_correlated_launches', complete=True,
            rows=[dict(owner=i, phase=p, native_decode_launches=10, cache_decode_launches=2)
                  for i in (0, 1) for p in ('prefill', 'decode')], trace_pins={'example': 'pin'})
        with patch.object(gate.ct, 'check_pins') as check:
            gate.validate_phase_decode(report)
            check.assert_called_once_with(report['trace_pins'])
        report['rows'][0]['cache_decode_launches'] = 10
        with self.assertRaises(ValueError):
            gate.validate_phase_decode(report)

    def test_recreate_trace_cannot_silently_disable_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            for epoch in (0, 1):
                for lane in (0, 1):
                    root = out / 'lifecycle-raw' / ('epoch-' + str(epoch)) / ('rank-' + str(lane))
                    root.mkdir(parents=True)
                    (root / 'events.jsonl').write_text(json.dumps(dict(kind='owner_binding', binding=dict(
                        owner=lane, decoded_weight_cache=dict(capacity_grants=3 << 30,
                        pinned_grants=1340 << 20, routed_working_set_grants=768 << 20, routed_entries=20480)))) + '\n')
            self.assertEqual(len(gate.trace_budgets(out, 'recreate1')), 4)
            path = out / 'lifecycle-raw/epoch-1/rank-1/events.jsonl'
            path.write_text(json.dumps(dict(kind='owner_binding', binding=dict(owner=1))) + '\n')
            with self.assertRaises(ValueError):
                gate.trace_budgets(out, 'recreate1')

    def test_internal_snapshots_validate_actual_partitions_each_recreated_epoch(self):
        from test_decoded_cache_release import ArenaSnapshots
        rows = ArenaSnapshots().records()
        text = '\n'.join(json.dumps(r) for r in rows)
        out = gate.internal_snapshot_rows('test exact ... '+text, [None, None])
        self.assertEqual(len(out['rows']), 4)
        self.assertTrue(all(r['snapshot']['arena_backing_grants'] == 3<<30 for r in out['rows']))
        with self.assertRaises(ValueError): gate.internal_snapshot_rows(text, [None]*4)
        with self.assertRaises(ValueError): gate.internal_snapshot_rows(text+'\n'+text, [None]*4)
        second = copy.deepcopy(rows)
        for row in second:
            snapshot = json.loads(row['fields']['snapshot_json'])
            snapshot['image_generation'] += 1
            row['fields']['snapshot_json'] = json.dumps(snapshot)
        both = text+'\n'+'\n'.join(json.dumps(r) for r in second)
        self.assertEqual(len(gate.internal_snapshot_rows(both, [None]*4)['rows']), 8)
        for key, value in (('schema_version', 1), ('stage', 'shutdown_unproven'), ('owner', 3)):
            broken = copy.deepcopy(rows)
            broken[0]['fields'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                gate.internal_snapshot_rows('\n'.join(json.dumps(r) for r in broken), [None]*2)

    def test_dedicated_root_requires_actual_full_driver_growth_not_legacy_pooled_growth(self):
        from test_decoded_cache_release import ArenaSnapshots
        for growth in (0, (3<<30)-1, (3<<30)+1):
            rows = ArenaSnapshots().records()
            snapshot = json.loads(rows[0]['fields']['snapshot_json'])
            snapshot['driver_allocation_growth_bytes'] = growth
            rows[0]['fields']['snapshot_json'] = json.dumps(snapshot)
            with self.subTest(growth=growth), self.assertRaises(ValueError):
                gate.internal_snapshot_rows('\n'.join(json.dumps(r) for r in rows), [None]*2)

    def test_outputs_cannot_escape_paper_experiment_or_modify_raw_oracle(self):
        base = adapter.HERE / 'runs' / 'unit-path-policy-no-files'
        raw = base / 'full40'
        ref = adapter.ORACLE / 'runs/full40-20261009-03'
        adapter.output_paths(raw, ref, base / 'full40-canonical', base / 'full40-comparison.json')
        for canonical in (raw / 'canonical', ref / 'new-candidate', Path('/tmp/cache-candidate')):
            with self.assertRaises(ValueError):
                adapter.output_paths(raw, ref, canonical, base / 'comparison.json')

    def test_historical_reports_match_parser_without_creating_new_receipts(self):
        for name, selector in [('production-opts-01-full40-comparison.json', 'full40'),
                               ('production-opts-01-greedy4-comparison.json', 'greedy4')]:
            path = gate.ORACLE / 'runs' / name
            if path.exists():
                gate.exact_inventory(json.loads(path.read_text()), selector)


class AllocatorV3Policy(unittest.TestCase):
    def log(self, **overrides):
        values = dict(retired_blocks=8, retirement_batches=1, retirement_created_fences=2,
            pending_retirement_blocks=0, pending_retirement_bytes=0, pending_retirement_batches=0,
            pending_unfenced_bytes=0, reserved_bytes=20<<30, peak_reserved_bytes=21<<30,
            allocation_failures=0, capture_retirement_bytes=0, shutdown_quiescent=False,
            shutdown_retained=False)
        values.update(overrides)
        text = 'CudaAllocatorMetrics { ' + ', '.join(k+': '+json.dumps(v) for k,v in values.items()) + ' }'
        return '\n'.join(json.dumps(dict(fields=dict(owner=owner, device_quarantine=False,
            claim_finished=True, quiescent=True, allocator=text, message=message)))
            for owner in (0,1) for message in ('Mixed owner preparation claim completion',
                'Mixed owner shutdown fence; image retained until worker retirement'))

    def test_named_roster_and_supplemental_are_not_interchangeable(self):
        plan = gate.manifest()['allocator_v3']
        self.assertEqual(plan['named_gates'], list(gate.ALL_GATES))
        self.assertEqual(len(gate.ALL_GATES), 23)
        self.assertEqual(sum(name.startswith('backend-') for name in gate.SMALL_GATES), 5)
        self.assertEqual(sum(name.startswith('model-') for name in gate.SMALL_GATES), 11)
        self.assertTrue(plan['supplemental_not_replacement'])
        self.assertEqual(plan['pending_bytes_growth_watermark'], 256<<20)
        self.assertEqual(plan['pending_batches_growth_watermark'], 256)

    def test_boundary_pending_peak_is_not_fabricated_global_high_watermark(self):
        result = gate.allocator_v3_rows(self.log(), 1)
        self.assertEqual(len(result['rows']), 4)
        self.assertEqual(result['observed_pending_peak_bytes'], 0)
        self.assertIsNone(result['exact_pending_high_watermark_bytes'])
        self.assertEqual(result['peak_reserved_bytes'], 21<<30)
        with self.assertRaises(ValueError):
            gate.allocator_v3_rows(self.log(), 2)

    def test_pressure_quarantine_cap_or_pending_at_quiescence_is_failure(self):
        for patch in (dict(allocation_failures=1), dict(peak_reserved_bytes=(22<<30)+1),
                      dict(pending_retirement_bytes=256), dict(pending_unfenced_bytes=256),
                      dict(pending_retirement_batches=1), dict(shutdown_retained=True)):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                gate.allocator_v3_rows(self.log(**patch), 1)
        with self.assertRaisesRegex(ValueError, 'pressure'):
            gate.allocator_v3_rows(self.log()+'\ndriver growth deferred by ordinary retirement pressure',1)

    def test_missing_v3_field_or_counter_regression_is_not_accepted(self):
        with self.assertRaises(ValueError):
            gate.allocator_v3_rows(self.log().replace('retirement_created_fences: 2, ', ''), 1)
        lines=self.log().splitlines()
        lines[1]=lines[1].replace('retired_blocks: 8','retired_blocks: 7')
        with self.assertRaises(ValueError):
            gate.allocator_v3_rows('\n'.join(lines), 1)


if __name__ == '__main__':
    unittest.main()
