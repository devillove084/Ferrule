"""Matrix contracts: CPU mocks only, no server, network, GPU or inference packages."""
import copy
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest import mock

from test_bench_ferrule_vllm import bench, record


def corpus():
    return dict(tokenizer=dict(revision='cpu-fixture-only', files_sha256={'tokenizer.json': 'a'*64},
                               encoding=dict(add_special_tokens=False, template='none', thinking=False)),
                prompts=[dict(id=f'p{n}', prompt=f'CPU MOCK ONLY {n}', input_tokens=n, input_ids=[7]*n)
                         for n in bench.MATRIX_PREFILLS])


def metadata():
    result = dict(tokenizer=corpus()['tokenizer'], model=dict(checkpoint_sha256='b'*64),
        precision=dict(weights_dtype='fp8_e4m3fn', activation_dtype='f32', accumulation_dtype='f32',
                       rounding='RNE', scale_profile=dict(dtype='bf16', interpretation='multiply', block=[128,128]),
                       kv_dtype='f32', output_dtype='f32', numeric_operands_dtype='f32', state_boundary_dtype='f32',
                       gdn_recurrent_dtype='f32', gdn_convolution_dtype='f32',
                       admitted_resources=dict(gpu_uuids=['mock-only'], per_gpu_limit_bytes=[24000],
                                               host_limit_bytes=40960, pinned_limit_bytes=0)),
        topology=dict(dp=1, tp=1, ep=8, pp=1, pcp=1, rank_layout='separate-tp-ep', model_worker_count=9,
                      physical_placement=[dict(owner=i, pp=0, role='root' if i == 0 else 'ep-expert',
                                                lane=max(0, i-1), cuda_ordinal=max(0, i-1),
                                                physical_ordinal=max(0, i-1), gpu_uuid='CPU-mock-only')
                                          for i in range(9)],
                      expert_policy=dict(residency='all_owned', partition='contiguous',
                      placement='rank0-root', rank_backend='thread', transport='in-process')),
        engine_capability=dict(max_context_tokens=8448, max_concurrency=8, streaming=True,
                               raw_completions=True, eos_policy='permitted', prefix_cache=False),
        cache=dict(prefix_mode='off', os_cache='warm', state_reset='per_request'),
        quality_gate=dict(status='pass', reference_sha256='c'*64, coverage_sha256='d'*64,
                          logits=True, routes=True, recurrent_state=True, convolution_state=True))

    canonical_fixture(result)
    return result


def canonical_fixture(meta):
    """Synthetic complete CPU plan; never a declaration about a real model."""
    t = meta['topology']
    pp, tp, ep, dp, pcp = (t[k] for k in ('pp', 'tp', 'ep', 'dp', 'pcp'))
    t['canonical'] = dict(schema='ferrule.logical_topology.v1',
        device_permutation_policy='exact_canonical_slot', logical_groups=bench.logical_groups(t),
        num_layers=2*pp, moe_layers=list(range(2*pp)),
        pp_layer_ranges=[dict(pp=p, start=p*2, end=(p+1)*2) for p in range(pp)],
        tensor_shard_map=[dict(pp=p, dp=d, pcp=c, lane=i, layer=layer, tensor='q_proj.weight',
                              axis=0, start=i*4, end=(i+1)*4, size=tp*4)
                          for p in range(pp) for d in range(dp) for c in range(pcp)
                          for layer in range(p*2, (p+1)*2) for i in range(tp)],
        expert_shard_map=[dict(pp=p, lane=i, layer=layer, total_experts=ep*2, experts=[i, i+ep])
                          for p in range(pp) for layer in range(p*2, (p+1)*2) for i in range(ep)])
    shared = t['rank_layout'] == 'shared-tp-dp-pcp'
    owners = ([(p, 'shared-tp-ep', i) for p in range(pp) for i in range(tp*dp*pcp)] if shared else
              [(p, role, i) for role, width in (('tp-root', tp), ('ep-expert', ep if ep > 1 else 0))
               for p in range(pp) for i in range(width)])
    width = max(tp*dp*pcp, ep)
    t['physical_placement'] = [dict(owner=rank, pp=p, role=role, lane=i, cuda_ordinal=p*width+i,
                                   physical_ordinal=p*width+i, gpu_uuid=f'CPU-mock-{p*width+i}')
                               for rank, (p, role, i) in enumerate(owners)]
    resources = meta['precision']['admitted_resources']
    resources.update(gpu_uuids=[f'CPU-mock-{i}' for i in range(pp*width)], per_gpu_limit_bytes=[24000]*(pp*width))


def profiles():
    return [dict(id=engine, engine=engine, model='mock-model', url=f'http://127.0.0.1:{18000+i}/v1/completions',
                 argv=['/cpu-mock-not-executable', '--ctx', '{context_tokens}', '--active', '{concurrency}'],
                 ownership=dict(exclusive_resources=True, all_workers_in_process_group=True, no_daemonize=True),
                 metadata=metadata()) for i, engine in enumerate(('ferrule', 'vllm'))]


def http_report():
    item = record()
    item.update(input_length_matches=True, output_length_matches=True)
    mock_metadata = metadata()
    # Synthetic CPU-only proof for contract tests, never persisted as a real run.
    mock_metadata['runtime_contract'] = dict(schema='ferrule.runtime_contract.v1', status='verified',
        reasons=[], argv_sha256='e'*64, server_log_sha256='f'*64)
    return dict(schema='ferrule.http_bench.v1', status='complete', metadata=mock_metadata,
                corpus=['same'], workload={'max_tokens': 3},
                aggregate=bench.aggregate([item], 2, 1),
                runs=[dict(summary=dict(not_issued=0), requests=[item])])


class ManifestTests(unittest.TestCase):
    def test_complete_required_grid_budget_includes_blocked_starts_and_cleanup(self):
        manifest = bench.build_manifest(corpus(), profiles())
        budget = bench.matrix_budget(manifest)
        self.assertEqual(budget['scheduled_server_starts'], 600)
        self.assertEqual(budget['scheduled_measured_requests'], 60000)
        self.assertEqual(budget['supervisor_seconds_total'], 4443750)
        self.assertEqual(budget['serial_seconds_with_cleanup_reserve'], 4461780)
        self.assertTrue(budget['all_required_cells_selected'])
        manifest['profiles'][0]['ownership']['exclusive_resources'] = False
        self.assertEqual(bench.matrix_budget(manifest), budget)
        small = bench.build_manifest(corpus(), profiles()[:1], repetitions=2, requests=2, warmup=2,
                    cell_ids=['p8192-o32-c1'], limits=dict(startup_seconds=900, request_seconds=300,
                    measurement_seconds=600, teardown_seconds=120))
        self.assertEqual(bench.matrix_budget(small)['supervisor_seconds_total'], 5130)
        self.assertEqual(bench.matrix_budget(small)['serial_seconds_with_cleanup_reserve'], 5220)
        self.assertFalse(bench.matrix_budget(small)['all_required_cells_selected'])

    def test_small_cell_selection_limits_and_canonical_order(self):
        limits = dict(startup_seconds=900, request_seconds=120, measurement_seconds=180, teardown_seconds=120)
        ids = ['p8192-o32-c1', 'p128-o32-c2']
        manifest = bench.build_manifest(corpus(), profiles()[:1], repetitions=2, requests=2,
                                        warmup=2, cell_ids=ids, limits=limits)
        self.assertEqual(len(manifest['schedule']), 4)
        self.assertEqual(manifest['selected_cell_ids'], list(reversed(ids)))
        self.assertEqual(manifest['limits'], limits)
        for invalid in ([], ['p1-o32-c1'], [ids[0], ids[0]]):
            with self.assertRaises(bench.ProtocolError):
                bench.build_manifest(corpus(), profiles(), cell_ids=invalid)
        for invalid in ({}, dict(limits, request_seconds=301), dict(limits, request_seconds=True),
                        dict(limits, request_seconds=float('nan')), dict(limits, measurement_seconds=0)):
            with self.assertRaises(bench.ProtocolError):
                bench.build_manifest(corpus(), profiles(), limits=invalid)

    def test_selected_cli_plan_dry_run_and_oversized_execution_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'corpus.json').write_text(json.dumps(corpus()))
            (root/'profiles.json').write_text(json.dumps(profiles()))
            with mock.patch.object(bench, 'supervisor_module', side_effect=AssertionError('offline only')), \
                    mock.patch('sys.stdout', new=io.StringIO()), mock.patch('sys.stderr', new=io.StringIO()):
                self.assertEqual(bench.main(['matrix-plan', '--corpus', str(root/'corpus.json'),
                    '--profiles', str(root/'profiles.json'), '--profile-id', 'ferrule', '--cell', 'p128-o32-c1',
                    '--process-runs', '2', '--requests', '2', '--output', str(root/'small.json')]), 0)
                self.assertEqual(bench.main(['matrix-run', '--manifest', str(root/'small.json'),
                                            '--output', str(root/'dry')]), 0)
                report = json.loads((root/'dry/matrix.json').read_text())
                self.assertEqual(len(report['runs']), 2)
                self.assertTrue(all(r['profile_id'] == 'ferrule' for r in report['runs']))
                (root/'big.json').write_text(json.dumps(bench.build_manifest(corpus(), profiles())))
                with self.assertRaises(SystemExit):
                    bench.main(['matrix-run', '--manifest', str(root/'big.json'), '--execute',
                                '--output', str(root/'refused')])
                self.assertFalse((root/'refused').exists())
                with self.assertRaises(SystemExit):
                    bench.main(['matrix-run', '--manifest', str(root/'small.json'), '--cell', 'p128-o32-c2',
                                '--output', str(root/'override')])
                tampered = json.loads((root/'small.json').read_text())
                tampered['cells'][0]['context_tokens'] = 128
                (root/'tampered.json').write_text(json.dumps(tampered))
                with self.assertRaises(SystemExit):
                    bench.main(['matrix-run', '--manifest', str(root/'tampered.json'), '--output', str(root/'bad')])

    def test_full_prefill_output_concurrency_grid_context_includes_output(self):
        manifest = bench.build_manifest(corpus(), profiles())
        self.assertEqual(len(manifest['cells']), 60)
        self.assertEqual(len(manifest['schedule']), 600)
        expected = {(p, o, c) for p in (128,512,2048,4096,8192) for o in (32,128,256) for c in (1,2,4,8)}
        self.assertEqual({(c['prefill_tokens'],c['output_tokens'],c['concurrency']) for c in manifest['cells']}, expected)
        self.assertEqual(max(c['context_tokens'] for c in manifest['cells']), 8448)
        self.assertTrue(all(c['context_tokens'] == c['prefill_tokens']+c['output_tokens'] for c in manifest['cells']))
        self.assertEqual(manifest, bench.build_manifest(corpus(), profiles()))
        self.assertNotEqual(manifest['schedule'], bench.build_manifest(corpus(), profiles(), seed=1)['schedule'])
        slots = {(r['cell_id'],r['profile_id'],r['repetition']) for r in manifest['schedule']}
        self.assertEqual(len(slots), 600)
        for left, right in zip(manifest['schedule'][::2], manifest['schedule'][1::2]):
            self.assertEqual(left['cell_id'], right['cell_id'])
            self.assertNotEqual(left['profile_id'], right['profile_id'])

    def test_exact_token_ids_and_assets_required_without_loading_tokenizer(self):
        for change in ('ids', 'counts', 'lengths', 'hash', 'revision'):
            data = corpus()
            if change == 'ids':
                del data['prompts'][0]['input_ids']
            elif change == 'counts':
                data['prompts'][0]['input_tokens'] += 1
            elif change == 'lengths':
                data['prompts'].pop()
            elif change == 'hash':
                data['tokenizer']['files_sha256']['tokenizer.json'] = 'unknown'
            else:
                data['tokenizer']['revision'] = 'unknown'
            with self.subTest(change=change), self.assertRaises(bench.ProtocolError):
                bench.validate_corpus(data)

    def test_unknown_unsupported_tokenizer_and_ownership_stay_blocked(self):
        manifest = bench.build_manifest(corpus(), profiles())
        profile, cell = manifest['profiles'][0], manifest['cells'][-1]
        self.assertEqual(bench.cell_blockers(manifest, profile, cell), [])
        for mutate in (
            lambda p: p['metadata']['engine_capability'].update(max_context_tokens=8192),
            lambda p: p['metadata']['engine_capability'].update(max_concurrency=1),
            lambda p: p['metadata'].update(engine_capability={}),
            lambda p: p['metadata']['tokenizer'].update(revision='different'),
            lambda p: p['ownership'].update(no_daemonize=False),
            lambda p: p['metadata']['cache'].update(state_reset='unknown')):
            changed = copy.deepcopy(profile)
            mutate(changed)
            self.assertTrue(bench.cell_blockers(manifest, changed, cell))

    def test_prefix_cold_and_warm_are_process_local_priming_not_dropcache(self):
        manifest = bench.build_manifest(corpus(), profiles(), warmup=0)
        profile, cell = manifest['profiles'][0], manifest['cells'][0]
        profile['metadata']['cache'].update(prefix_mode='first-cold-then-warm', fresh_process_empty_prefix=True)
        profile['metadata']['engine_capability']['prefix_cache'] = True
        self.assertTrue(bench.cell_blockers(manifest, profile, cell))
        manifest['warmup'] = 8
        self.assertEqual(bench.cell_blockers(manifest, profile, cell), [])

    def test_plan_and_dry_run_cli_never_import_supervisor_or_start_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'corpus.json').write_text(json.dumps(corpus()))
            (root/'profiles.json').write_text(json.dumps(profiles()))
            with mock.patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('no process')), \
                    mock.patch.object(bench, 'supervisor_module', side_effect=AssertionError('offline only')), \
                    mock.patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(bench.main(['matrix-plan', '--corpus', str(root/'corpus.json'),
                    '--profiles', str(root/'profiles.json'), '--process-runs', '2', '--output', str(root/'manifest.json')]), 0)
                self.assertEqual(bench.main(['matrix-run', '--manifest', str(root/'manifest.json'),
                                            '--output', str(root/'dry')]), 0)
            report = json.loads((root/'dry/matrix.json').read_text())
            self.assertEqual(report['status'], 'planned')
            self.assertFalse(report['executed'])
            self.assertEqual(len(report['runs']), 240)
            self.assertTrue(all(r['status'] == 'not_run' for r in report['runs']))
            self.assertTrue(all(not c['comparability'] for c in report['comparisons']))


class ComparisonTests(unittest.TestCase):
    def test_complete_contract_is_eligible_but_never_emits_blind_speedup(self):
        left = http_report()
        result = bench.comparison_eligibility(left, copy.deepcopy(left))
        self.assertTrue(result['comparability'])
        self.assertTrue(result['fairmatch'])
        self.assertIsNone(result['speedup'])

    def test_missing_runtime_proof_cannot_promote_identical_metadata(self):
        report = http_report()
        report['metadata'].pop('runtime_contract')
        result = bench.comparison_eligibility(report, copy.deepcopy(report))
        self.assertFalse(result['fairmatch'])
        self.assertIsNone(result['speedup'])
        self.assertTrue(any('runtime argv' in r for r in result['labels']['reasons']))

    def test_bf16_w8a16_vs_f32_topology_experts_rounding_scale_resources(self):
        left = http_report()
        mutations = [lambda m: m['precision'].update(weights_dtype='w8a16', activation_dtype='bf16'),
                     lambda m: m['precision'].update(accumulation_dtype='bf16'),
                     lambda m: m['precision'].update(numeric_operands_dtype='bf16'),
                     lambda m: m['precision'].update(state_boundary_dtype='bf16'),
                     lambda m: m['precision'].update(gdn_recurrent_dtype='bf16'),
                     lambda m: m['precision'].update(gdn_convolution_dtype='bf16'),
                     lambda m: m['topology']['physical_placement'][0].update(physical_ordinal=7),
                     lambda m: m['topology'].update(pcp=8),
                     lambda m: m['precision'].update(rounding='RTZ'),
                     lambda m: m['precision']['scale_profile'].update(interpretation='reciprocal'),
                     lambda m: m['precision']['admitted_resources'].update(host_limit_bytes=81920),
                     lambda m: m['topology'].update(dp=2, tp=4),
                     lambda m: m['topology']['expert_policy'].update(residency='bounded_cache')]
        for mutation in mutations:
            right = copy.deepcopy(left)
            mutation(right['metadata'])
            with self.subTest(mutation=mutation):
                result = bench.comparison_eligibility(left, right)
                self.assertTrue(result['length_workload_eligible'])
                self.assertFalse(result['comparability'])
                self.assertFalse(result['fairmatch'])

    def test_unknown_missing_false_capability_and_stateful_gate_never_pass_identically(self):
        mutations = [lambda m: m.clear(),
                     lambda m: m['topology'].pop('pp'),
                     lambda m: m['topology'].pop('pcp'),
                     lambda m: m['topology'].pop('rank_layout'),
                     lambda m: m['topology'].pop('model_worker_count'),
                     lambda m: m['topology'].update(pcp=True),
                     lambda m: m['topology'].update(model_worker_count=8),
                     lambda m: m['precision'].pop('numeric_operands_dtype'),
                     lambda m: m['precision'].pop('state_boundary_dtype'),
                     lambda m: m['precision'].update(state_boundary_dtype='unknown'),
                     lambda m: m['topology'].update(dp=True),
                     lambda m: m['topology'].update(expert_policy='unknown'),
                     lambda m: m['precision'].update(rounding='unknown'),
                     lambda m: m['precision']['admitted_resources'].pop('pinned_limit_bytes'),
                     lambda m: m['engine_capability'].update(streaming=False),
                     lambda m: m['engine_capability'].update(status='unavailable'),
                     lambda m: m['cache'].update(os_cache='uncontrolled'),
                     lambda m: m['cache'].update(state_reset='shared-session'),
                     lambda m: m['quality_gate'].update(recurrent_state=False),
                     lambda m: m['quality_gate'].pop('convolution_state'),
                     lambda m: m['quality_gate'].update(status='fail')]
        for mutation in mutations:
            report = http_report()
            mutation(report['metadata'])
            with self.subTest(mutation=mutation):
                self.assertFalse(bench.comparison_eligibility(report, report)['comparability'])

    def test_colocated_separate_and_shared_owners_have_same_logical_topology(self):
        for pp in (1, 2):
            left, right = metadata(), metadata()
            left['topology'].update(pp=pp, tp=2, ep=2, model_worker_count=pp*4)
            right['topology'].update(pp=pp, tp=2, ep=2, rank_layout='shared-tp-dp-pcp', model_worker_count=pp*2)
            right['topology']['expert_policy'].update(rank_backend='process', transport='allgather_reducescatter',
                                                      placement='shared TP/EP ranks')
            for m in (left, right):
                canonical_fixture(m)
            audit = bench.comparison_contract(left, right)
            self.assertTrue(audit['sameprecision'])
            self.assertTrue(audit['same_logical_topology'])
            self.assertTrue(audit['sametopology'])
            self.assertFalse(audit['owner_organization_match'])
            self.assertTrue(audit['physical_resources_match'])
            self.assertEqual(audit['labels']['physical_resources']['left']['unique_gpu_count'], pp*2)
            self.assertTrue(audit['contract_match'], audit['reasons'])

    def test_vllm_flattened_ep_constraints_even_for_identical_claims(self):
        for pp, tp, dp, pcp, ep, workers, valid in (
                (1, 2, 1, 1, 2, 2, True), (2, 2, 1, 1, 2, 4, True),
                (1, 4, 1, 1, 2, 4, False), (1, 1, 1, 1, 8, 1, False),
                (1, 4, 2, 1, 8, 8, True), (1, 1, 1, 8, 8, 8, True),
                (2, 2, 1, 1, 4, 4, False), (2, 2, 1, 1, 2, 8, False)):
            with self.subTest(pp=pp, tp=tp, dp=dp, pcp=pcp, ep=ep, workers=workers):
                m = metadata()
                m['topology'].update(pp=pp, tp=tp, dp=dp, pcp=pcp, ep=ep,
                                     model_worker_count=workers, rank_layout='shared-tp-dp-pcp',
                                     physical_placement=[dict(owner=i, pp=0, lane=i, role='shared-tp-ep',
                                         cuda_ordinal=i, physical_ordinal=i, gpu_uuid='CPU-mock-only')
                                         for i in range(workers)])
                canonical_fixture(m)
                audit = bench.comparison_contract(m, m)
                self.assertTrue(audit['sametopology'])
                self.assertEqual(audit['physical_resources_match'], valid)
                self.assertEqual(audit['contract_match'], valid)


class CanonicalTopologyTests(unittest.TestCase):
    def pair(self, pp=2):
        left, right = metadata(), metadata()
        left['topology'].update(pp=pp, tp=2, ep=2, model_worker_count=pp*4)
        right['topology'].update(pp=pp, tp=2, ep=2, model_worker_count=pp*2, rank_layout='shared-tp-dp-pcp')
        right['topology']['expert_policy'].update(rank_backend='process', transport='allgather_reducescatter')
        for m in (left, right):
            canonical_fixture(m)
        return left, right

    def test_normalization_ignores_record_order_budget_order_and_local_cuda_numbering(self):
        left, right = self.pair()
        plan = right['topology']['canonical']
        for key in ('logical_groups', 'pp_layer_ranges', 'tensor_shard_map', 'expert_shard_map', 'moe_layers'):
            plan[key].reverse()
        for group in plan['logical_groups']:
            group['members'].reverse()
        for row in plan['expert_shard_map']:
            row['experts'].reverse()
        for row in right['topology']['physical_placement']:
            row['cuda_ordinal'] = 3-row['cuda_ordinal']
        resources = right['precision']['admitted_resources']
        resources['gpu_uuids'].reverse()
        resources['per_gpu_limit_bytes'].reverse()
        before = copy.deepcopy(right)
        audit = bench.comparison_contract(left, right)
        self.assertTrue(audit['contract_match'], audit['reasons'])
        self.assertEqual(right, before)

    def test_default_exact_slot_policy_rejects_physical_device_permutation(self):
        left, right = self.pair()
        right['topology']['canonical'].pop('device_permutation_policy')
        self.assertTrue(bench.comparison_contract(left, right)['contract_match'])
        for row in right['topology']['physical_placement']:
            device = 3-row['physical_ordinal']
            row.update(physical_ordinal=device, gpu_uuid=f'CPU-mock-{device}')
        audit = bench.comparison_contract(left, right)
        self.assertTrue(audit['same_logical_topology'])
        self.assertFalse(audit['physical_resources_match'])
        self.assertFalse(audit['contract_match'])
        self.assertIn('physical_resources differs', audit['reasons'])
        right['topology']['canonical']['device_permutation_policy'] = 'any_bijection'
        self.assertFalse(bench.comparison_contract(right, right)['contract_match'])

    def test_noncolocated_ep_uses_extra_unique_gpus_not_extra_worker_budgets(self):
        left, right = self.pair(pp=1)
        for row in left['topology']['physical_placement'][2:]:
            ordinal = row['physical_ordinal'] + 2
            row.update(cuda_ordinal=ordinal, physical_ordinal=ordinal, gpu_uuid=f'CPU-mock-{ordinal}')
        resources = left['precision']['admitted_resources']
        resources.update(gpu_uuids=[f'CPU-mock-{i}' for i in range(4)], per_gpu_limit_bytes=[24000]*4)
        audit = bench.comparison_contract(left, right)
        self.assertTrue(audit['same_logical_topology'])
        self.assertFalse(audit['physical_resources_match'])
        self.assertEqual(audit['labels']['physical_resources']['left']['unique_gpu_count'], 4)
        self.assertEqual(audit['labels']['physical_resources']['right']['unique_gpu_count'], 2)
        self.assertFalse(audit['contract_match'])

    def test_budget_mismatch_is_not_logical_topology_mismatch(self):
        for key in ('per_gpu_limit_bytes', 'host_limit_bytes', 'pinned_limit_bytes'):
            left, right = self.pair()
            resources = right['precision']['admitted_resources']
            if key == 'per_gpu_limit_bytes':
                resources[key][0] += 1
            else:
                resources[key] += 1
            audit = bench.comparison_contract(left, right)
            self.assertTrue(audit['same_logical_topology'])
            self.assertFalse(audit['physical_resources_match'])
            self.assertFalse(audit['contract_match'])

    def test_valid_different_tensor_or_expert_assignment_is_logical_mismatch(self):
        for kind in ('tensor', 'expert'):
            left, right = self.pair()
            plan = right['topology']['canonical']
            for row in plan[kind + '_shard_map']:
                row['lane'] = 1-row['lane']
            self.assertIsNotNone(bench.canonical_topology(right)['logical'])
            audit = bench.comparison_contract(left, right)
            self.assertFalse(audit['same_logical_topology'])
            self.assertFalse(audit['contract_match'])

    def test_valid_different_pp_layer_split_is_logical_mismatch(self):
        left, right = self.pair()
        plan = right['topology']['canonical']
        plan['pp_layer_ranges'][0]['end'] = 1
        plan['pp_layer_ranges'][1]['start'] = 1
        for key in ('tensor_shard_map', 'expert_shard_map'):
            for row in plan[key]:
                if row['layer'] == 1:
                    row['pp'] = 1
        self.assertIsNotNone(bench.canonical_topology(right)['logical'])
        self.assertFalse(bench.comparison_contract(left, right)['same_logical_topology'])

    def test_missing_unknown_and_malformed_canonical_evidence_fail_closed_even_identically(self):
        mutations = [
            lambda t: t.pop('canonical'),
            lambda t: t.update(status='unverified'),
            lambda t: t['canonical'].update(status='unknown'),
            lambda t: t['canonical'].update(schema='future'),
            lambda t: t['canonical'].update(num_layers=True),
            lambda t: t['canonical'].update(moe_layers='unknown'),
            lambda t: t['canonical']['moe_layers'].append(True),
            lambda t: t['canonical']['logical_groups'][0]['members'].append(0),
            lambda t: t['canonical']['pp_layer_ranges'][1].update(start=1),
            lambda t: t['canonical']['pp_layer_ranges'][1].update(start=3),
            lambda t: t['canonical']['tensor_shard_map'].pop(),
            lambda t: t['canonical']['tensor_shard_map'][0].update(end=3),
            lambda t: t['canonical']['tensor_shard_map'][0].update(end=5),
            lambda t: t['canonical']['tensor_shard_map'][0].update(tensor='unknown'),
            lambda t: t['canonical']['tensor_shard_map'][0].update(axis=True),
            lambda t: t['canonical']['expert_shard_map'].pop(),
            lambda t: t['canonical']['expert_shard_map'][0].update(experts=[0, 0]),
            lambda t: t['canonical']['expert_shard_map'][0].update(experts=[0, True]),
            lambda t: t['canonical']['expert_shard_map'][0].update(experts='unknown'),
        ]
        for key in ('logical_groups', 'pp_layer_ranges', 'tensor_shard_map', 'expert_shard_map'):
            for value in (None, 'unknown', [], {}, [None]):
                mutations.append(lambda t, key=key, value=value: t['canonical'].update({key: value}))
        for mutate in mutations:
            left, _ = self.pair()
            mutate(left['topology'])
            with self.subTest(mutate=mutate):
                audit = bench.comparison_contract(left, left)
                self.assertFalse(audit['same_logical_topology'])
                self.assertFalse(audit['contract_match'])

    def test_unknown_owner_device_or_budget_blocks_without_erasing_logical_match(self):
        mutations = [
            lambda m: m['topology'].update(rank_layout='unknown'),
            lambda m: m['topology'].update(model_worker_count=2),
            lambda m: m['topology']['physical_placement'].pop(),
            lambda m: m['topology']['physical_placement'][0].update(gpu_uuid='unknown'),
            lambda m: m['topology']['physical_placement'][0].update(gpu_uuid='CPU-mock-1'),
            lambda m: m['topology']['physical_placement'][0].update(physical_ordinal=1),
            lambda m: m['topology']['physical_placement'][0].update(cuda_ordinal=True),
            lambda m: m['precision']['admitted_resources'].update(gpu_uuids=['CPU-mock-0']*4),
            lambda m: m['precision']['admitted_resources'].update(per_gpu_limit_bytes=[True]*4),
            lambda m: m['precision']['admitted_resources'].update(host_limit_bytes=-1),
            lambda m: m['precision']['admitted_resources'].update(pinned_limit_bytes='unknown'),
        ]
        for mutate in mutations:
            left, _ = self.pair()
            mutate(left)
            with self.subTest(mutate=mutate):
                audit = bench.comparison_contract(left, left)
                self.assertTrue(audit['same_logical_topology'])
                self.assertFalse(audit['physical_resources_match'])
                self.assertFalse(audit['contract_match'])

    def test_logical_match_never_bypasses_precision_boundary_quality_or_runtime_gate(self):
        for group, field, value in (
                ('precision', 'rounding', 'RTZ'), ('precision', 'activation_dtype', 'bf16'),
                ('precision', 'state_boundary_dtype', 'bf16'), ('precision', 'kv_dtype', 'bf16'),
                ('precision', 'gdn_recurrent_dtype', 'bf16'), ('precision', 'gdn_convolution_dtype', 'bf16'),
                ('quality_gate', 'status', 'fail'), ('quality_gate', 'coverage_sha256', 'unknown')):
            left, right = self.pair()
            right[group][field] = value
            audit = bench.comparison_contract(left, right)
            self.assertTrue(audit['same_logical_topology'])
            self.assertFalse(audit['contract_match'])
        left, right = http_report(), http_report()
        left['metadata'], right['metadata'] = self.pair()
        audit = bench.comparison_eligibility(left, right)
        self.assertTrue(audit['labels']['same_logical_topology'])
        self.assertFalse(audit['fairmatch'])
        self.assertIsNone(audit['speedup'])
        self.assertIn('runtime argv/topology/precision/placement evidence missing or blocked', audit['labels']['reasons'])


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_math_degenerate_missing_single_and_reproducible(self):
        self.assertEqual(bench.bootstrap_ci([4,4])['low'], 4)
        self.assertEqual(bench.bootstrap_ci([4,4])['high'], 4)
        self.assertIsNone(bench.bootstrap_ci([4])['low'])
        self.assertIsNone(bench.bootstrap_ci([None,4])['estimate'])
        self.assertIsNone(bench.bootstrap_ci([])['estimate'])
        result = bench.bootstrap_ci([0,2], seed=42)
        self.assertEqual(result['estimate'], 1)
        self.assertEqual((result['low'],result['high']), (0,2))
        self.assertEqual(result, bench.bootstrap_ci([0,2], seed=42))

    def test_fixed_process_and_request_denominators_include_oom_timeout_notrun(self):
        manifest = bench.build_manifest(corpus(), profiles()[:1], repetitions=5, requests=10)
        manifest['cells'] = manifest['cells'][:1]
        cell = manifest['cells'][0]['id']
        runs = [dict(profile_id='ferrule', cell_id=cell, repetition=i, status=status)
                for i, status in enumerate(('complete','request_failure','timeout','startup_oom','not_run'))]
        runs[0]['http'] = dict(aggregate=dict(success=10, offered=10, successful_output_tokens_per_second=4))
        runs[1]['http'] = dict(aggregate=dict(success=2, offered=5, successful_output_tokens_per_second=1))
        summary = bench.summarize_matrix(manifest, runs)[0]
        self.assertEqual(summary['scheduled_process_runs'], 5)
        self.assertEqual(summary['scheduled_requests'], 50)
        self.assertEqual(summary['offered_known'], 15)
        self.assertEqual(summary['successful'], 12)
        self.assertAlmostEqual(summary['success_fraction_ci']['estimate'], 12/50)
        self.assertIsNone(summary['throughput_ci']['estimate'])

    def test_paired_summary_uses_all_independent_blocks_and_blocks_invalid_pairs(self):
        manifest = bench.build_manifest(corpus(), profiles(), repetitions=2, requests=1)
        manifest['cells'] = manifest['cells'][:1]
        cell = manifest['cells'][0]['id']
        runs = [dict(profile_id=p['id'], cell_id=cell, repetition=i, status='complete', http=http_report())
                for p in manifest['profiles'] for i in range(2)]
        for run in runs:
            if run['profile_id'] == 'vllm':
                run['http']['aggregate']['successful_output_tokens_per_second'] += 2
        result = bench.matrix_comparisons(manifest, runs)[0]
        self.assertTrue(result['comparability'])
        self.assertEqual(result['paired_rate_difference_ci']['estimate'], 2)
        self.assertEqual(result['paired_rate_difference_ci']['low'], 2)
        runs[-1]['status'] = 'timeout'
        result = bench.matrix_comparisons(manifest, runs)[0]
        self.assertFalse(result['comparability'])
        self.assertEqual(result['scheduled_pairs'], 2)
        self.assertEqual(result['eligible_pairs'], 1)
        self.assertIsNone(result['paired_rate_difference_ci'])


class SupervisorTests(unittest.TestCase):
    def test_shutdown_only_unreaped_direct_root_never_claims_descendant_drain(self):
        server = mock.Mock()
        with mock.patch.object(bench.os, 'getpgrp', return_value=111), \
                mock.patch.object(bench.os, 'getpid', return_value=222), \
                mock.patch.object(bench.os, 'killpg') as kill:
            with self.assertRaises(bench.ProtocolError):
                bench.shutdown_owned_group(server, 1)
            kill.assert_not_called()
        for timeout in (False, True):
            server.reset_mock()
            server.poll.return_value = None
            server.wait.side_effect = bench.subprocess.TimeoutExpired('root', 1) if timeout else None
            with mock.patch.object(bench.os, 'getpgrp', return_value=222), \
                    mock.patch.object(bench.os, 'getpid', return_value=222), \
                    mock.patch.object(bench.os, 'killpg') as kill:
                result = bench.shutdown_owned_group(server, 1)
            self.assertEqual(result['timed_out'], timeout)
            server.terminate.assert_called_once()
            kill.assert_not_called()
            self.assertNotEqual(result['status'], 'drained_process_group')

    def test_supervisor_timeout_halts_remaining_schedule_keeps_all_denominators(self):
        manifest = bench.build_manifest(corpus(), profiles()[:1], repetitions=2, requests=1)
        supervisor = mock.Mock()
        supervisor.custody_capabilities.return_value = dict(containment=True)  # mock proof, never host authority
        supervisor.run_process.return_value = dict(returncode=-9, timed_out=True, wall_seconds=1, stdout='', stderr='OOM')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text(json.dumps(manifest))
            with mock.patch.object(bench, 'supervisor_module', return_value=supervisor), \
                    mock.patch.object(bench.subprocess, 'Popen', side_effect=AssertionError('no server')):
                code = bench.main(['matrix-run', '--manifest', str(root/'manifest.json'),
                                   '--output', str(root/'run'), '--execute', '--max-server-runs', '120'])
            report = json.loads((root/'run/matrix.json').read_text())
        self.assertEqual(code, 1)
        supervisor.run_process.assert_called_once()
        self.assertTrue(report['halted_for_cleanup'])
        self.assertEqual(len(report['runs']), 120)
        self.assertEqual(report['runs'][0]['status'], 'supervisor_timeout')
        self.assertTrue(all(r['status'] == 'not_run' for r in report['runs'][1:]))
        command, budget, payload = supervisor.run_process.call_args.args
        self.assertIn('--_matrix_worker', command)
        self.assertGreater(budget, 900+3600+120)
        self.assertEqual(payload['manifest'], manifest)

    def test_worker_reuses_shared_main_in_inherited_group_and_records_phases(self):
        manifest = bench.build_manifest(corpus(), profiles()[:1], repetitions=2)
        server = mock.Mock()
        server.poll.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(bench.os, 'getpgrp', return_value=222), \
                    mock.patch.object(bench.os, 'getpid', return_value=222), \
                    mock.patch.object(bench, 'own_group_members', return_value=[]), \
                    mock.patch.object(bench.socket, 'socket'), \
                    mock.patch.object(bench.subprocess, 'Popen', return_value=server) as popen, \
                    mock.patch.object(bench, 'wait_ready', return_value=2), \
                    mock.patch.object(bench, 'main', return_value=0) as shared, \
                    mock.patch.object(bench, 'shutdown_owned_group', return_value=dict(status='drained_process_group',timed_out=False)), \
                    mock.patch.object(bench, 'supervisor_module') as supervisor:
                supervisor.return_value.scrub.side_effect = lambda text: text
                code = bench.matrix_worker(dict(manifest=manifest, profile=manifest['profiles'][0],
                                                cell=manifest['cells'][0], directory=directory))
            report = json.loads((Path(directory)/'lifecycle.json').read_text())
        self.assertEqual(code, 0)
        self.assertNotIn('start_new_session', popen.call_args.kwargs)
        self.assertIn('160', popen.call_args.args[0])
        self.assertIn('--first-after-ready', shared.call_args.args[0])
        self.assertEqual(shared.call_args.args[0][shared.call_args.args[0].index('--repeat')+1], '1')
        self.assertEqual(set(report['phases']), {'startup','shutdown'})
        self.assertIsNone(report['phases']['startup']['load_seconds'])


class AdditionalMatrixContracts(unittest.TestCase):
    def test_local_tokenizer_export_roundtrip_without_real_packages(self):
        tokenizer = mock.Mock()
        tokenizer.encode.side_effect = lambda text, add_special_tokens: mock.Mock(ids=[ord(c) for c in text])
        tokenizer.decode.side_effect = lambda tokens, skip_special_tokens: ''.join(chr(c) for c in tokens)
        module = mock.Mock()
        module.Tokenizer.from_file.return_value = tokenizer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'tokenizer.json').write_text('{}')
            with mock.patch.dict('sys.modules', {'tokenizers': module}):
                bench.export_corpus(root, 'cpu-mock', root/'corpus.json')
                with self.assertRaises(bench.ProtocolError):
                    bench.export_corpus(root, 'cpu-mock', root/'corpus.json')
            result = json.loads((root/'corpus.json').read_text())
        self.assertEqual(len(result['prompts']), 10)
        bench.validate_corpus(result)
        self.assertTrue(all(len(row['prompt']) == row['input_tokens'] for row in result['prompts']))
        tokenizer.no_truncation.assert_called_once()
        tokenizer.no_padding.assert_called_once()

    def test_concurrency_1_4_8_preserves_eos_event_latencies_and_fixed_requests(self):
        import threading
        import time
        from test_bench_ferrule_vllm import parse, frame, content, usage
        for concurrency in (1,4,8):
            args = bench.parser().parse_args(['--engine','ferrule','--url','http://example.invalid/v1/completions',
                '--model','mock','--output','unused.json','--window-kind','fixed-work','--requests',str(2*concurrency),
                '--concurrency',str(concurrency),'--max-tokens','32','--input-tokens','5'])
            barrier = threading.Barrier(concurrency)
            def request(config, identity, cutoff):
                sent = time.monotonic()
                barrier.wait(timeout=2)
                # EOS before cap is protocol-successful but not length-matched. SSE events != tokens.
                stream = parse(frame(content('many words')) + frame(content(' more','stop'))
                               + frame(usage(3)) + frame('[DONE]'), clock=lambda: sent+.001)
                item = stream.metrics(sent, sent+.002, 'success')
                item.update(identity, status='success', sent_at=sent, ended_at=sent+.002,
                            length_matched_success=False, input_length_matches=True, output_length_matches=False)
                return item
            clients = [mock.Mock(request=mock.Mock(side_effect=request)) for _ in range(concurrency)]
            result = bench.measure(clients, args, bench.corpus_for(args), 0)
            self.assertEqual(result['summary']['scheduled'], 2*concurrency)
            self.assertEqual(result['summary']['success'], 2*concurrency)
            self.assertEqual(result['summary']['length_matched_success'], 0)
            self.assertEqual(result['summary']['successful_output_tokens'], 6*concurrency)
            self.assertTrue(all(r['finish_reason']=='stop' and r['content_events']==2 for r in result['requests']))
            self.assertTrue(all(r['ttft_seconds'] is not None and r['token_itl_seconds'] is None for r in result['requests']))

    def test_first_after_ready_warmup_and_measure_are_disjoint_legacy_schema_retained(self):
        client = mock.Mock()
        client.request.side_effect = lambda config, identity, cutoff=None: dict(record(identity['index']),
                                                                               population=identity['population'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'http.json'
            with mock.patch.object(bench, 'Client', return_value=client), mock.patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(bench.main(['--engine','ferrule','--url','http://example.invalid/v1/completions',
                    '--model','mock','--output',str(path),'--window-kind','fixed-work', '--warmup','2',
                    '--repeat','1','--requests','3','--first-after-ready']), 0)
            result = json.loads(path.read_text())
        self.assertEqual(result['schema'], 'ferrule.http_bench.v1')
        self.assertEqual(result['aggregate']['offered'], 3)
        self.assertEqual(result['first_after_ready']['population'], 'first_after_ready')
        self.assertEqual(len(result['warmups']), 2)
        self.assertEqual(client.request.call_count, 6)
        self.assertEqual(result['phases'][0]['phase'], 'warmup')
        self.assertTrue(result['runs'][0]['measurements']['work_completed'])

    def test_startup_oom_is_retained_and_shared_measure_never_runs(self):
        manifest = bench.build_manifest(corpus(), profiles()[:1], repetitions=2)
        with tempfile.TemporaryDirectory() as directory:
            def launch(argv, **kwargs):
                kwargs['stdout'].write(b'CUDA out of memory')
                return mock.Mock()
            with mock.patch.object(bench.os,'getpgrp',return_value=123), \
                    mock.patch.object(bench.os,'getpid',return_value=123), \
                    mock.patch.object(bench,'own_group_members',return_value=[]), \
                    mock.patch.object(bench.socket,'socket'), \
                    mock.patch.object(bench.subprocess,'Popen',side_effect=launch), \
                    mock.patch.object(bench,'wait_ready',side_effect=bench.ProtocolError('Server exited before Ready')), \
                    mock.patch.object(bench,'main',side_effect=AssertionError('no measurement')), \
                    mock.patch.object(bench,'shutdown_owned_group',return_value=dict(status='drained_process_group',timed_out=False)):
                code = bench.matrix_worker(dict(manifest=manifest, profile=manifest['profiles'][0],
                                                cell=manifest['cells'][0], directory=directory))
            lifecycle = json.loads((Path(directory)/'lifecycle.json').read_text())
        self.assertEqual(code, 1)
        self.assertTrue(lifecycle['oom_observed_in_log'])
        self.assertIsNone(lifecycle['phases']['startup']['ready_at'])
        self.assertNotIn('measure', lifecycle['phases'])


if __name__ == '__main__':
    unittest.main()
