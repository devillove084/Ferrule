"""CPU-only indexed qualification and immutable historical-plan regressions."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import decoded_cache_release as d
import optimized_release as release
import operator_retirement_ab as retirement


class IndexedRelease(unittest.TestCase):
    def link(self):
        return dict(path=str(d.INDEXED_QUALITY/'handoff.json'),sha256=d.INDEXED_QUALIFICATION_SHA)

    def test_actual_final_indexed_serving_audit_binds_delta_without_smoke_claim(self):
        link=dict(path=str(d.INDEXED_RELEASE/'handoff.json'),sha256=d.INDEXED_HANDOFF_SHA)
        with self.assertRaisesRegex(ValueError, "Cache release evidence changed"):
            release.audit(link)
        frozen = retirement.frozen_audit()
        self.assertFalse(frozen["current_tree_qualified"])
        m = retirement.frozen_metadata()
        q = retirement.pre.checked_document(m["test_quality"])
        evidence = dict(binary_sha256=m["binary"]["sha256"], evidence_pins=m["build_evidence_pins"])
        self.assertEqual(evidence['binary_sha256'],d.INDEXED_BINARY_SHA)
        self.assertEqual((len(q['rows']),q['calls'],q['metrics']),(23,37,40293))
        self.assertEqual(q['qualification_handoff_sha256'],d.INDEXED_QUALIFICATION_SHA)
        self.assertEqual(q['public_cpu_relation']['passed'],24)
        self.assertEqual(q['source_relation']['complete_production_inventory_count'],584)
        self.assertEqual(len(q['source_relation']['changed']),2)
        self.assertEqual(len(q['source_relation']['native_cpu_source_pin_coverage_gaps']),4)
        self.assertEqual(q['source_relation']['production_delta']['sha256'],d.INDEXED_DELTA_SHA)
        self.assertFalse(q['numerical_pass_inherited'])
        self.assertEqual(evidence['evidence_pins'][str(d.INDEXED_RELEASE/'production-delta.json')],d.INDEXED_DELTA_SHA)

    def test_final_indexed_serving_hash_and_location_cannot_be_relabelled(self):
        for path,sha in ((d.INDEXED_RELEASE/'handoff.json','0'*64),(d.ARENA_RELEASE/'handoff.json',d.INDEXED_HANDOFF_SHA)):
            with self.assertRaisesRegex(ValueError,'Unreviewed final arena handoff'):
                d.arena_audit(dict(path=str(path),sha256=sha))

    def test_indexed_native_cpu_selectors_remain_exact_and_gpu_hidden(self):
        h=d.ct.load(d.INDEXED_RELEASE/'handoff.json');inventory=d.ct.load(d.INDEXED_RELEASE/'cpu-test-inventory.json')
        for change in ('argv','gpu','duplicate'):
            e=d.Evidence();read=e.read
            def changed(path,expected=None):
                value=read(path,expected)
                if Path(path)==d.INDEXED_RELEASE/'cpu-preflight-27.json':
                    if change=='argv':value['argv'][1]='unexecuted-selector'
                    if change=='gpu':value['environment']['CUDA_VISIBLE_DEVICES']='0'
                if change=='duplicate' and Path(path)==d.INDEXED_RELEASE/'preflight-test-inventory.json':
                    value['tests'][-1]=value['tests'][0]
                return value
            with self.subTest(change=change),mock.patch.object(e,'read',side_effect=changed),self.assertRaisesRegex(ValueError,'Indexed CPU'):
                d.indexed_cpu_preflights(e,h,inventory)

    def test_actual_indexed_qualification_is_audited_not_serving_accepted(self):
        with self.assertRaisesRegex(ValueError, "Cache release evidence changed"):
            d.audit_qualification(self.link())
        retirement.frozen_audit()
        m = retirement.frozen_metadata()
        historical = retirement.pre.checked_document(m["test_quality"])
        handoff = retirement.pre.checked_document(self.link())
        summary = retirement.pre.checked_document(handoff["acceptance_summary"])
        # The historical receipt audit above is real; isolate only the current-tree
        # adapter here to retain its qualification-only output contract regression.
        def historical_quality(e, h):
            self.assertEqual(h, handoff)
            for name in d.ARENA_WORKFLOW_ADAPTERS:
                path = d.INDEXED_WORKFLOW_HISTORY/(name+".txt")
                e.pin(path, historical["artifact_pins"][str(path)])
            return summary, deepcopy(historical)
        with mock.patch.object(d, "arena_quality", side_effect=historical_quality):
            q=d.audit_qualification(self.link())
        self.assertEqual((len(q['rows']),q['calls'],q['metrics']),(23,37,40293))
        self.assertEqual({r['gate']:r['selector'] for r in q['rows']},d.gates.ALL_GATES)
        self.assertEqual(q['qualification_handoff_sha256'],d.INDEXED_QUALIFICATION_SHA)
        self.assertTrue(q['qualification_only'])
        self.assertFalse(q['serving_release_accepted'])
        self.assertFalse(q['numerical_pass_inherited'])
        self.assertTrue(q['public_smoke_required'])
        for name in d.ARENA_WORKFLOW_ADAPTERS:
            path=d.INDEXED_WORKFLOW_HISTORY/(name+'.txt')
            self.assertEqual(q['artifact_pins'][str(path)],d.ct.sha(path))

    def test_unindexed_and_indexed_bind_their_real_pinned_plan_not_current_manifest(self):
        for root,count in ((d.ARENA_QUALITY,33),(d.INDEXED_QUALITY,38)):
            prepared=d.ct.load(root/'prepared.json')
            with self.subTest(root=root.name),mock.patch.object(d.gates,'manifest',side_effect=AssertionError('mutable current plan forbidden')):
                plan=d.arena_prepared_plan(d.Evidence(),root,prepared)
                self.assertEqual(len(prepared['cpu_receipts']),count)
                self.assertEqual(len(plan['host_small_selectors'])+len(plan['cpu_selectors']),count)

    def test_plan_hash_cpu_inventory_and_named_selector_tampering_rejected(self):
        root=d.INDEXED_QUALITY;original=d.ct.load(root/'prepared.json')
        for change in ('hash','missing','duplicate','selector'):
            prepared=deepcopy(original);e=d.Evidence()
            if change=='hash':prepared['pins'][str(root/'plan.json')]='0'*64
            if change=='missing':prepared['cpu_receipts'].pop()
            if change=='duplicate':prepared['cpu_receipts'][-1]=prepared['cpu_receipts'][0]
            read=e.read
            def changed(path,expected=None):
                value=read(path,expected)
                if change=='selector' and Path(path)==root/'plan.json':
                    value['host_small_selectors'][0][1]='unexecuted-selector'
                return value
            with self.subTest(change=change),mock.patch.object(e,'read',side_effect=changed),self.assertRaises(ValueError):
                d.arena_prepared_plan(e,root,prepared)

    def test_final_qualification_hash_location_and_serving_schema_are_separate(self):
        for link in (dict(self.link(),sha256='0'*64),dict(self.link(),path=str(d.ARENA_QUALITY/'handoff.json'))):
            with self.assertRaisesRegex(ValueError,'Unreviewed indexed qualification'):
                d.audit_qualification(link)
        with self.assertRaisesRegex(ValueError,'Unreviewed optimized handoff'):
            release.audit(self.link())
        with mock.patch.object(d.ct,'require_cpu_python'),mock.patch.object(d,'audit_qualification') as audit:
            with self.assertRaisesRegex(ValueError,'Explicit qualification'):
                release.main(['audit-qualification','--handoff',str(d.INDEXED_QUALITY/'handoff.json')])
            audit.assert_not_called()

    def test_actual_indexed_source_drift_still_rejected(self):
        path=next(iter(d.gates.INDEXED_LRU_SOURCES))
        sha=d.ct.sha
        def changed(name):
            return '0'*64 if str(name)==path else sha(name)
        with mock.patch.object(d.ct,'sha',side_effect=changed),self.assertRaisesRegex(ValueError,'Cache release evidence changed'):
            d.audit_qualification(self.link())

    def test_cpu_benchmark_cannot_be_relabelled_gpu_or_e2e(self):
        h=d.ct.load(d.INDEXED_QUALITY/'handoff.json')
        summary=d.ct.load(d.INDEXED_QUALITY/'acceptance-summary.json')
        prepared=d.ct.load(d.INDEXED_QUALITY/'prepared.json')
        for key in ('gpu_used','end_to_end','timing_assertion'):
            e=d.Evidence();read=e.read
            def changed(path,expected=None):
                value=read(path,expected)
                if str(path)==h['cpu_benchmark']['path']:value[key]=True
                return value
            with self.subTest(key=key),mock.patch.object(e,'read',side_effect=changed),self.assertRaisesRegex(ValueError,'benchmark source or scope'):
                d.indexed_qualification_extras(e,h,summary,prepared)


if __name__=='__main__':unittest.main()
