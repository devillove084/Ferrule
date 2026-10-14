"""CPU-only cache public admission, source relation and raw snapshot contracts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import decoded_cache_release as d
import optimized_release as release
import paired as p


class CacheRelease(unittest.TestCase):
    def test_frozen_thirteen_gate_summary_is_historical_not_arena_qualification(self):
        e = d.Evidence()
        summary = e.read(d.QUALITY/'acceptance-summary.json', d.SUMMARY_SHA)
        self.assertEqual({r['gate']:r['selector'] for r in summary['rows']}, d.gates.HISTORICAL_ALL_GATES)
        self.assertEqual((summary['full_model_calls'], summary['quality_metrics']), (37, 40293))
        self.assertTrue(summary['all_exact'])
        self.assertFalse(summary['public_e2e_accepted'])
        self.assertNotEqual(set(d.gates.HISTORICAL_ALL_GATES), set(d.gates.ALL_GATES))
        # Historical workflow hashes are never silently rebound to the new runner.
        with self.assertRaisesRegex(ValueError, 'Cache release evidence changed:'):
            d.quality(d.Evidence())

    def test_old_dadf_handoff_cannot_qualify_current_arena_sources(self):
        handoff = p.ct.ROOT/'target/validation/tp2-mixed-decoded-cache-public-release-20261010-03/handoff.json'
        with self.assertRaisesRegex(ValueError, 'Cache release source drift/rebuild pending'):
            release.audit(dict(path=str(handoff),
                sha256='90a8289e18334aaae4dde93f0ae763868faf3b78e666258fddedd96e6e7b09fd'))

    def test_gate_inventory_rejects_duplicate_missing_or_wrong_selector(self):
        original=p.ct.load(d.QUALITY/'acceptance-summary.json')
        for mutation in ('duplicate','missing','selector','binary'):
            summary=copy.deepcopy(original)
            if mutation=='duplicate': summary['rows'][-1]=copy.deepcopy(summary['rows'][0])
            if mutation=='missing': summary['rows'].pop()
            if mutation=='selector': summary['rows'][0]['selector']='not-the-required-test'
            if mutation=='binary': summary['rows'][0]['binary_sha256']='0'*64
            e=d.Evidence();read=e.read
            def changed(path, expected=None):
                return summary if Path(path)==d.QUALITY/'acceptance-summary.json' else read(path,expected)
            with self.subTest(mutation=mutation), mock.patch.object(e,'read',side_effect=changed):
                with self.assertRaisesRegex(ValueError, 'Required cache gate set|Cache gate selector/binary'):d.quality(e)

    def test_old_observation_only_review_cannot_cover_new_arena_algorithm(self):
        e = d.Evidence()
        q = e.read(d.QUALITY/'acceptance-summary.json', d.SUMMARY_SHA)
        with self.assertRaisesRegex(ValueError, 'Unreviewed quality source change:'):
            d.source_relation(e, q['sources'], p.pre_serving.current_release_sources(), [])

    def test_unreviewed_backend_change_and_new_production_input_block(self):
        q=p.ct.load(d.QUALITY/'acceptance-summary.json');sources=p.pre_serving.current_release_sources()
        backend=next(k for k in sources if '/ferrule-backend/' in k)
        for changed in (dict(sources,**{backend:'0'*64}),
                        dict(sources,**{str(p.ct.ROOT/'crates/ferrule-model/src/unreviewed.rs'):'0'*64})):
            with self.assertRaisesRegex(ValueError,'Unreviewed'):
                d.source_relation(d.Evidence(),q['sources'],changed,[])

    def test_cpu_metadata_is_not_gpu_observation_and_caps_are_fixed(self):
        owner=dict(cache_enabled=True,cache_capacity_grants=3<<30,pinned_grants=1340<<20,
            routed_working_set_grants=768<<20,required_workspace_bytes=3875939840,
            workspace_ceiling_bytes=4<<30,required_device_bytes=23041273856,device_ceiling_bytes=22<<30)
        good=dict(measured_on_gpu=False,owners=[dict(owner),dict(owner)])
        self.assertEqual(d.metadata_ledger(good),good)
        for key,value in (('cache_enabled',False),('workspace_ceiling_bytes',1<<30),
                          ('device_ceiling_bytes',24<<30),('cache_capacity_grants',4<<30)):
            bad=copy.deepcopy(good);bad['owners'][0][key]=value
            with self.assertRaises(ValueError):d.metadata_ledger(bad)
        good['measured_on_gpu']=True
        with self.assertRaises(ValueError):d.metadata_ledger(good)

    def test_new_schema_dispatch_does_not_use_old_19_gate_audit(self):
        with mock.patch.object(release.pre,'checked_document',return_value={'schema':d.SCHEMA}), \
                mock.patch.object(d,'audit',return_value={'new':True}) as audit:
            self.assertEqual(release.audit({'path':'fixture','sha256':'a'*64}),{'new':True})
            audit.assert_called_once()
        with mock.patch.object(release.pre,'checked_document',return_value={'schema':d.QUALITY_SCHEMA}), \
                mock.patch.object(release,'validate',return_value={'cache':True}) as validate:
            self.assertEqual(release.pre.validate_release({}),{'cache':True})
            validate.assert_called_once()

    def test_bootstrap_requires_explicit_final_handoff_hash_before_audit_or_staging(self):
        with tempfile.TemporaryDirectory(dir=p.HERE/'admission') as tmp:
            handoff=Path(tmp)/'handoff.json';p.ct.write(handoff,{'schema':d.SCHEMA})
            with mock.patch.object(p,'strict_gate'), mock.patch.object(release,'audit') as audit, \
                    mock.patch.object(p.ct,'stage_binary') as stage:
                with self.assertRaisesRegex(ValueError,'explicit final'):
                    release.bootstrap(Path(tmp)/'fresh',handoff)
                audit.assert_not_called();stage.assert_not_called()
                self.assertFalse((Path(tmp)/'fresh').exists())

    def test_cache_metadata_cannot_be_removed_or_relabelled_during_validate(self):
        q={'schema':d.QUALITY_SCHEMA}
        evidence=dict(quality=q,source_pins={'source':'a'},runtime_pins={'lib':'b'},binary_sha256='c'*64,
            evidence_pins={'artifact':'d'},build_custody='CPU-only',serving_scope=p.CACHE_WORKSPACE_SCOPE,decoded_cache=d.CACHE)
        m=dict(source_pins=evidence['source_pins'],runtime_pins=evidence['runtime_pins'],binary={'sha256':'c'*64},
            build_evidence_pins=evidence['evidence_pins'],build_receipt={'sha256':'e'*64},test_quality={'sha256':'f'*64},
            serving_scope=p.CACHE_WORKSPACE_SCOPE,decoded_cache=dict(d.CACHE))
        with mock.patch.object(release,'audit',return_value=evidence),mock.patch.object(release.pre,'checked_document',return_value=q):
            self.assertEqual(release.validate(m)['serving_observation'],'pending')
            m['decoded_cache']['enabled']=False
            with self.assertRaisesRegex(ValueError,'metadata differs'):release.validate(m)

    def test_profile_only_rankperf_environment_never_mutates_no_trace_base_spec(self):
        with tempfile.TemporaryDirectory(dir=p.HERE) as tmp:
            pair=Path(tmp);slot=pair/'runs/P1';slot.mkdir(parents=True)
            m=dict(binary={'path':'/fixture/ferrule','sha256':'c'*64},decoded_cache=d.CACHE,
                   spec=p.ferrule_spec('/fixture/ferrule',p.CACHE_WORKSPACE_SCOPE))
            prepared=dict(purpose='profiling',admission={'engines':{'ferrule':m},'profiler':p.ct.load(p.HERE/'admission/optimized-20261009/profile-admission.json').get('profiler',{})})
            # Any non-null profiler here has already passed validate_pair; it is
            # the slot-only transformation that this fixture isolates.
            p.ct.write(pair/'prepared.json',prepared)
            p.ct.write(slot/'prepared.json',dict(pair_directory=str(pair),slot='P1',pair_sha256=p.ct.sha(pair/'prepared.json')))
            p.ct.write(slot/'canaries.json',p.shared.references()[0])
            with mock.patch.object(p,'validate_pair',return_value=prepared):
                profile=p.validate_slot(slot)
            self.assertEqual(profile['spec']['environment']['FERRULE_LOG'],d.PROFILE_LOG)
            self.assertEqual(m['spec']['environment']['FERRULE_LOG'],'info')
            self.assertNotIn('RUST_LOG',m['spec']['environment'])


class ArenaRelease(unittest.TestCase):
    def test_frozen_final_handoff_audit_preserves_historical_sources_and_gaps(self):
        import arena_index_ab
        with mock.patch.object(d.pre,'current_release_sources',side_effect=AssertionError('not historical source evidence')):
            evidence = arena_index_ab.frozen_audit()
        m = arena_index_ab.frozen_metadata()
        q = d.pre.checked_document(m['test_quality'])
        self.assertFalse(evidence['current_tree_qualified'])
        self.assertEqual(evidence['source_pins'],m['source_pins'])
        self.assertEqual(evidence['binary_sha256'], d.ARENA_BINARY_SHA)
        self.assertEqual({r['gate'] for r in q['rows']}, set(d.gates.ALL_GATES))
        self.assertEqual((len(q['rows']),q['calls'],q['metrics']), (23,37,40293))
        self.assertFalse(q['numerical_pass_inherited'])
        self.assertFalse(q['source_relation']['inherited_gpu_pass'])
        self.assertEqual(len(q['source_relation']['native_cpu_source_pin_coverage_gaps']),4)
        self.assertIn('NOT covered',q['source_relation']['caveat'])
        self.assertEqual(q['kind'],'authenticated_test_binary_not_serving')
        self.assertEqual(m['serving_scope'],p.CACHE_WORKSPACE_SCOPE)
        self.assertEqual(m['decoded_cache'],d.CACHE)

    def test_arena_gate_names_selectors_binaries_and_freshness_not_count_only(self):
        original=d.ct.load(d.ARENA_QUALITY/'acceptance-summary.json')
        d.arena_gate_inventory(original)
        for change in ('duplicate','missing','name','selector','binary','reused'):
            summary=copy.deepcopy(original)
            if change=='duplicate':summary['rows'][-1]=copy.deepcopy(summary['rows'][0])
            if change=='missing':summary['rows'].pop()
            if change=='name':summary['rows'][-1]['gate']='invented-gate'
            if change=='selector':summary['rows'][0]['selector']='invented-test'
            if change=='binary':summary['rows'][0]['binary_sha256']='0'*64
            if change=='reused':summary['rows'][0]['reused_unchanged_backend']=True
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'arena gate set|Arena gate selector'):
                d.arena_gate_inventory(summary)

    def test_final_arena_hash_and_location_are_not_interchangeable(self):
        for link in (dict(path=str(d.ARENA_RELEASE/'handoff.json'),sha256='0'*64),
                     dict(path=str(d.ARENA_QUALITY/'handoff.json'),sha256=d.ARENA_HANDOFF_SHA)):
            with self.assertRaisesRegex(ValueError,'Unreviewed final arena handoff'):d.arena_audit(link)

    def test_archived_workflow_bytes_keep_original_qualification_hashes(self):
        prepared=d.ct.load(d.ARENA_QUALITY/'prepared.json')
        for name in d.ARENA_WORKFLOW_ADAPTERS:
            self.assertEqual(d.ct.sha(d.ARENA_WORKFLOW_HISTORY/(name+'.txt')),
                             prepared['pins'][str(d.pre.HERE/name)])

    def test_arena_dispatch_and_explicit_bootstrap_hash(self):
        with mock.patch.object(release.pre,'checked_document',return_value={'schema':d.ARENA_SCHEMA}), \
                mock.patch.object(d,'audit',return_value={'arena':True}) as audit:
            self.assertEqual(release.audit({'path':'fixture','sha256':'a'*64}),{'arena':True})
            audit.assert_called_once()
        with tempfile.TemporaryDirectory(dir=p.HERE/'admission') as tmp:
            handoff=Path(tmp)/'handoff.json';p.ct.write(handoff,{'schema':d.ARENA_SCHEMA})
            with mock.patch.object(p,'strict_gate'),mock.patch.object(release,'audit') as audit:
                with self.assertRaisesRegex(ValueError,'explicit final'):
                    release.bootstrap(Path(tmp)/'fresh',handoff)
                audit.assert_not_called()
                self.assertFalse((Path(tmp)/'fresh').exists())

    def test_arena_receipts_and_comparators_are_checked_not_just_summary(self):
        qualification=d.ct.load(d.ARENA_QUALITY/'handoff.json')
        mutations=(
            ('backend-exact/supervisor.json',lambda r:r['custody'].update(echild=False)),
            ('backend-exact/authentication.json',lambda r:r.update(launch_id='wrong')),
            ('backend-exact/worker-launch.json',lambda r:r.update(sources={})),
            ('backend-exact/test-exit.json',lambda r:r.update(selector='wrong')),
            ('backend-exact/release.json',lambda r:r.update(drained=False)),
            ('backend-exact/resources.json',lambda r:r.update(errors=['failed'])),
            ('backend-arena-allocation/small-receipt.json',lambda r:r.update(assertions_passed=False)),
            ('full40/comparison-receipt.json',lambda r:r.update(reference_sha256='0'*64)),
            ('full40-comparison.json',lambda r:next(iter(next(iter(r['cases'].values()))['metrics'].values())).update(rmse=1)),
        )
        for name, mutate in mutations:
            e=d.Evidence();read=e.read
            def changed(path,expected=None):
                value=read(path,expected)
                if Path(path)==d.ARENA_QUALITY/name:mutate(value)
                return value
            # Isolate semantic receipt checks; the integration test above hashes
            # the actual sources, tools, libraries and every referenced artifact.
            with self.subTest(name=name),mock.patch.object(e,'pin',side_effect=lambda path,expected=None:Path(path)), \
                    mock.patch.object(e,'read',side_effect=changed),mock.patch.object(d.ct,'validate_inventory'):
                with self.assertRaises(ValueError):d.arena_quality(e,qualification)

    def test_tool_inventory_and_library_drift_are_not_skipped(self):
        qualification=d.ct.load(d.ARENA_QUALITY/'handoff.json')
        for kind in ('tools','library'):
            e=d.Evidence()
            library=next(iter(e.linked(qualification['runtime_libraries'])['resolved_libraries']))
            def pin(path,expected=None):
                if kind=='library' and str(path)==library:raise ValueError('library drift')
                return Path(path)
            with self.subTest(kind=kind),mock.patch.object(e,'pin',side_effect=pin), \
                    mock.patch.object(d.ct,'validate_inventory',side_effect=ValueError('tools drift') if kind=='tools' else None):
                with self.assertRaisesRegex(ValueError,kind+' drift'):d.arena_quality(e,qualification)

    def test_native_cpu_gaps_cannot_be_hidden_or_relabelled_as_qualified(self):
        h=d.ct.load(d.ARENA_RELEASE/'handoff.json')
        summary=d.ct.load(d.ARENA_QUALITY/'acceptance-summary.json')
        sources=d.pre.release_sources(d.ct.load(d.ARENA_RELEASE/'source-pins-before.json'))
        deps=d.ct.load(d.ARENA_RELEASE/'actual-release-dependencies.json')
        original=d.ct.load(d.ARENA_RELEASE/'actual-dependency-quality-relation.json')
        for change in ('missing','covered','dadf','source','dependency'):
            relation=copy.deepcopy(original);current=copy.deepcopy(h);qualified=copy.deepcopy(summary);dependencies=copy.deepcopy(deps)
            if change=='missing':relation['native_cpu_source_pin_coverage_gaps'].pop()
            if change=='covered':relation['native_cpu_source_pin_coverage_gaps'][0]['quality_sha256']='0'*64
            if change=='dadf':relation['native_cpu_source_pin_coverage_gaps'][0]['dadf_source_sha256']='0'*64
            if change=='source':qualified['sources'][next(iter(sources))]='0'*64
            if change=='dependency':dependencies[0]['quality_sha256']='0'*64
            current['qualification_native_cpu_source_pin_coverage_gaps']=relation['native_cpu_source_pin_coverage_gaps']
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'Arena'):
                d.arena_source_relation(d.Evidence(),current,qualified,sources,dependencies,relation)


class CacheSnapshots(unittest.TestCase):
    def records(self, profiling=False):
        rows=[]
        for stage in ('ready','shutdown'):
            for owner in (0,1):
                snapshot=dict(image_generation=1,phase='idle',published=True,quarantined=False,
                    capacity_grants=3<<30,pin_target_grants=1340<<20,pinned_grants=1340<<20,
                    unallocated_pin_grants=0,active_routed_grants=768<<20,pooled_grants=64<<20,
                    pending_reserved_grants=0,segment_slack_bytes=64<<20,pending_segment_slack_bytes=0,
                    physical_current_bytes=2236<<20,physical_peak_bytes=2300<<20,
                    pending_allocation_count=0,pending_lease_count=0,borrowed_key_count=0,unknown_reserved_bytes=0,
                    rank_performance_enabled=profiling,rank_performance_active=False,
                    profiled_activity=(dict(profiled_invocations=9,hits=20,misses=4,decoded_bytes=200,evictions=1)
                                       if profiling and stage=='shutdown' else None))
                rows.append(dict(target='ferrule_cache',fields=dict(event='mixed_decoded_cache_snapshot',
                    schema_version=1,owner=owner,device=owner,stage=stage,snapshot_json=json.dumps(snapshot))))
        return rows

    def audit(self,rows,profiling=False):
        return d.snapshot_rows('\n'.join(json.dumps(r) for r in rows),profiling)

    def mutate(self,rows,key,value,index=0):
        s=json.loads(rows[index]['fields']['snapshot_json']);s[key]=value
        rows[index]['fields']['snapshot_json']=json.dumps(s)
        return rows

    def test_ready_shutdown_physical_totals_and_null_activity(self):
        rows=self.audit(self.records())
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r['snapshot']['profiled_activity'] is None for r in rows))
        self.assertTrue(all(r['profiled_hit_fraction'] is None for r in rows))
        self.assertTrue(all(r['snapshot']['physical_peak_bytes']<=3<<30 for r in rows))

    def test_profile_activity_is_actual_subset_not_benchmark(self):
        rows=self.audit(self.records(True),True)
        self.assertTrue(all(r['snapshot']['profiled_activity'] is None for r in rows if r['stage']=='ready'))
        self.assertTrue(all(r['snapshot']['profiled_activity']['hits']==20 for r in rows if r['stage']=='shutdown'))
        with self.assertRaises(ValueError):self.audit(self.records(True),False)

    def test_missing_duplicate_foreign_or_unproven_snapshots_rejected(self):
        for variant in ('missing','duplicate','device','owner','stage'):
            rows=self.records()
            if variant=='missing': rows.pop()
            elif variant=='duplicate': rows.append(copy.deepcopy(rows[0]))
            else: rows[0]['fields'][variant]={'device':3,'owner':2,'stage':'shutdown_unproven'}[variant]
            with self.subTest(variant=variant),self.assertRaises(ValueError):self.audit(rows)

    def test_overbudget_wrong_sum_quarantine_busy_or_fake_activity_rejected(self):
        for key,value in (('physical_peak_bytes',(3<<30)+1),('physical_current_bytes',1),
                          ('capacity_grants',4<<30),('quarantined',True),('borrowed_key_count',1),
                          ('pending_allocation_count',1),('rank_performance_enabled',True),
                          ('profiled_activity',dict(profiled_invocations=0,hits=0,misses=0,decoded_bytes=0,evictions=0))):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.audit(self.mutate(self.records(),key,value))

    def test_profile_requires_collected_activity_and_same_image(self):
        for key,value in (('profiled_activity',None),('image_generation',2)):
            with self.subTest(key=key),self.assertRaises(ValueError):
                self.audit(self.mutate(self.records(True),key,value,index=2),True)


class ArenaSnapshots(CacheSnapshots):
    def records(self, profiling=False):
        rows = super().records(profiling)
        for row in rows:
            row['fields']['schema_version'] = 2
            s = json.loads(row['fields']['snapshot_json'])
            s.update(arena_backing_grants=3<<30, free_slot_grants=896<<20,
                unused_tail_bytes=4<<20, slot_padding_bytes=1<<20,
                segment_slack_bytes=0, physical_current_bytes=3<<30, physical_peak_bytes=3<<30,
                driver_allocation_growth_bytes=3<<30, backing_reuses=0, native_fallbacks=0, evictions=0)
            row['fields']['snapshot_json'] = json.dumps(s)
        return rows

    def test_root_and_padding_are_not_additional_component_charges(self):
        rows = self.audit(self.records())
        self.assertTrue(all(r['schema_version'] == 2 for r in rows))
        s = rows[0]['snapshot']
        partition = sum(s[k] for k in ('pinned_grants','active_routed_grants','pooled_grants',
            'free_slot_grants','unused_tail_bytes','pending_reserved_grants'))
        self.assertEqual(partition, s['physical_current_bytes'])
        self.assertGreater(partition+s['arena_backing_grants']+s['slot_padding_bytes'], 3<<30)
        for name in ('physical_current_bytes', 'physical_peak_bytes'):
            with self.assertRaises(ValueError):
                self.audit(self.mutate(self.records(), name, partition+s['arena_backing_grants']))

    def test_every_actual_field_is_required_not_defaulted_to_zero(self):
        original = json.loads(self.records()[0]['fields']['snapshot_json'])
        for key in original:
            rows = self.records()
            s = dict(original)
            del s[key]
            rows[0]['fields']['snapshot_json'] = json.dumps(s)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(rows)

    def test_arena_partition_root_allocation_and_phase_bounds_reject(self):
        for key, value in (('arena_backing_grants', 0), ('arena_backing_grants', (3<<30)-1),
                ('free_slot_grants', -1), ('unused_tail_bytes', 0), ('slot_padding_bytes', 3<<30),
                ('pending_reserved_grants', 1), ('segment_slack_bytes', 1),
                ('pending_segment_slack_bytes', 1), ('driver_allocation_growth_bytes', (3<<30)+1),
                ('physical_peak_bytes', (3<<30)-1), ('unallocated_pin_grants', 1),
                ('unknown_reserved_bytes', 1), ('phase', 'preparing'), ('phase', 'executing'),
                ('phase', 'unknown'), ('published', 1), ('free_slot_grants', True)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.audit(self.mutate(self.records(), key, value))

    def test_schema_downgrade_upgrade_unknown_and_mixed_lifecycle_reject(self):
        for version in (1, 3, True, None):
            rows = self.records()
            rows[0]['fields']['schema_version'] = version
            with self.subTest(version=version), self.assertRaises(ValueError):
                self.audit(rows)
        legacy = CacheSnapshots().records()
        legacy[0]['fields']['schema_version'] = 2
        with self.assertRaises(ValueError): self.audit(legacy)
        rows = self.records()
        rows[0] = CacheSnapshots().records()[0]
        with self.assertRaises(ValueError): self.audit(rows)

    def preallocation(self):
        s = json.loads(self.records()[0]['fields']['snapshot_json'])
        s.update(phase='preparing', published=False, arena_backing_grants=0, pinned_grants=0,
            unallocated_pin_grants=s['pin_target_grants'], active_routed_grants=0,
            pooled_grants=0, free_slot_grants=0, unused_tail_bytes=0,
            pending_reserved_grants=3<<30, pending_allocation_count=1, driver_allocation_growth_bytes=0)
        return s

    def test_preallocation_and_failed_allocation_retain_full_reservation(self):
        s = self.preallocation()
        d.snapshot_accounting(s, 2)
        self.assertEqual(s['physical_current_bytes'], 3<<30)
        self.assertEqual(s['arena_backing_grants'], 0)
        for pending in (1, 0):
            failed = dict(s, phase='quarantined', quarantined=True,
                pending_allocation_count=pending, unknown_reserved_bytes=3<<30)
            d.snapshot_accounting(failed, 2)
            for key in ('physical_current_bytes', 'unknown_reserved_bytes', 'pending_reserved_grants'):
                with self.subTest(key=key, pending=pending), self.assertRaises(ValueError):
                    d.snapshot_accounting(dict(failed, **{key: 0}), 2)
        with self.assertRaises(ValueError):
            d.snapshot_accounting(dict(s, pending_allocation_count=0), 2)

    def test_partial_install_and_unknown_keep_root_but_cannot_pass_lifecycle(self):
        s = self.preallocation()
        s.update(phase='quarantined', quarantined=True, arena_backing_grants=3<<30,
            pending_allocation_count=0, pinned_grants=8<<20,
            unallocated_pin_grants=(1340-8)<<20, pending_reserved_grants=(3072-8-4)<<20,
            unused_tail_bytes=4<<20, unknown_reserved_bytes=3<<30)
        d.snapshot_accounting(s, 2)
        rows = self.records()
        rows[0]['fields']['snapshot_json'] = json.dumps(s)
        with self.assertRaises(ValueError): self.audit(rows)
        ready = json.loads(self.records()[0]['fields']['snapshot_json'])
        for phase in ('preparing', 'executing', 'quarantined'):
            state = dict(ready, phase=phase, published=phase!='preparing',
                quarantined=phase=='quarantined', unknown_reserved_bytes=(3<<30 if phase=='quarantined' else 0))
            d.snapshot_accounting(state, 2)
            rows[0]['fields']['snapshot_json'] = json.dumps(state)
            with self.subTest(phase=phase), self.assertRaises(ValueError): self.audit(rows)


if __name__=='__main__':unittest.main()
