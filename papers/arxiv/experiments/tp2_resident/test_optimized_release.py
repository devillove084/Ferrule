"""CPU-only admission and scope tests; no GPU/device locks/server launch."""
import copy
from pathlib import Path
import sys
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parent))
import optimized_release as optimized
import compare_pilots
import paired as p


class OptimizedAdmission(unittest.TestCase):
    def test_optimized_release_dispatch_is_separate_from_old_totals(self):
        m={'test_quality':{},'build_receipt':{}}
        with mock.patch.object(p.pre_serving,'checked_document',return_value={'schema':optimized.QUALITY_SCHEMA}), \
                mock.patch.object(optimized,'validate',return_value={'fixture':True}) as validate:
            self.assertEqual(p.pre_serving.validate_release(m),{'fixture':True})
        validate.assert_called_once_with(m)

    def test_full40_exactness_and_coverage_required_not_just_pass_flag(self):
        metric=dict(exact=True,failures=0,max_abs=0,rmse=0)
        groups={name:dict(metrics={str(i):dict(metric) for i in range(2178)}) for name in ('hello','capital')}
        report=dict(passed=True,status='pass',cases=groups)
        self.assertEqual(optimized.exact_comparison(report,True),(4,4356))
        report['cases']['hello']['metrics']['0']['exact']=False
        with self.assertRaisesRegex(ValueError,'not exact'):optimized.exact_comparison(report,True)
        report['cases']['hello']['metrics']['0']['exact']=True
        report['cases'].pop('capital')
        with self.assertRaisesRegex(ValueError,'coverage'):optimized.exact_comparison(report,True)

    def test_release_identity_and_quality_must_both_match(self):
        quality={'schema':optimized.QUALITY_SCHEMA,'calls':37,'metrics':40293}
        evidence=dict(quality=quality,source_pins={'source':'a'},runtime_pins={'lib':'b'},
            binary_sha256='c'*64,evidence_pins={'artifact':'d'},build_custody='unavailable')
        m=dict(source_pins=evidence['source_pins'],runtime_pins=evidence['runtime_pins'],binary={'sha256':'c'*64},
            build_evidence_pins=evidence['evidence_pins'],build_receipt={'sha256':'e'*64},test_quality={'sha256':'f'*64})
        with mock.patch.object(optimized,'audit',return_value=evidence), \
                mock.patch.object(p.pre_serving,'checked_document',return_value=quality):
            self.assertEqual(optimized.validate(m)['serving_observation'],'pending')
            m['binary']['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'identity differs'):optimized.validate(m)

    def test_promotion_does_not_borrow_old_http_smoke(self):
        old=p.ct.load(p.HERE/'admission/tp2-serving-smoke-admission.json')['engines']['ferrule']
        new=copy.deepcopy(old);new['binary']['sha256']='0'*64
        new['serving_smoke']=dict(path=str(p.HERE/'runs/tp2-serving-smoke-20261009-003/serving-smoke.json'),
            sha256=p.ct.sha(p.HERE/'runs/tp2-serving-smoke-20261009-003/serving-smoke.json'))
        with self.assertRaisesRegex(ValueError,'binding differs'):p.pre_serving.validate_smoke(new)

    def test_false_gate_blocks_bootstrap_before_build_read(self):
        with mock.patch.object(p,'strict_gate',side_effect=ValueError('closed')),mock.patch.object(optimized,'audit') as audit:
            with self.assertRaisesRegex(ValueError,'closed'):optimized.bootstrap(Path('/not-created'),Path('/not-read'))
        audit.assert_not_called()

    def pair(self):
        admission=p.ct.load(p.HERE/'admission/tp2-paired-admission.json')
        return dict(protocol=p.PROTOCOL,purpose='paired',order=list(p.SLOTS),config=p.shared.config(),admission=admission)

    def test_comparison_rejects_precision_workload_and_trace_changes(self):
        old=self.pair();new=copy.deepcopy(old)
        compare_pilots.same_scope(old,new)
        for group,key,value in [('boundaries','ssm','bfloat16'),('workload','context',256),('topology','ep',2)]:
            changed=copy.deepcopy(new);changed['admission']['engines']['ferrule'][group][key]=value
            with self.assertRaisesRegex(ValueError,'scope'):compare_pilots.same_scope(old,changed)
        changed=copy.deepcopy(new);changed['admission']['engines']['ferrule']['spec']['environment']['FERRULE_LOG']='debug'
        with self.assertRaisesRegex(ValueError,'parameters'):compare_pilots.same_scope(old,changed)
        changed=copy.deepcopy(new);changed['admission']['engines']['vllm']['binary']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'baseline changed'):compare_pilots.same_scope(old,changed)

    def test_cache_workspace_change_is_not_an_identical_flags_comparison(self):
        old = self.pair(); new = copy.deepcopy(old)
        m = new['admission']['engines']['ferrule']
        m['serving_scope'] = dict(p.CACHE_WORKSPACE_SCOPE)
        m['spec'] = p.ferrule_spec(m['binary']['path'], m['serving_scope'])
        with self.assertRaisesRegex(ValueError, 'parameters'):
            compare_pilots.same_scope(old, new)

    def test_optimized_binary_alone_may_differ_not_inference_contract(self):
        old=self.pair();new=copy.deepcopy(old)
        m=new['admission']['engines']['ferrule'];m['binary']['sha256']='0'*64
        m['spec']['argv'][0]='/fresh/staged/ferrule'
        compare_pilots.same_scope(old,new)


if __name__=='__main__':unittest.main()
