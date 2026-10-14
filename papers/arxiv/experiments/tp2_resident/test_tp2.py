"""CPU-only tests; real file locks, no runtime imports or GPU queries."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment as tp


class TP2Contracts(unittest.TestCase):
    def test_declared_no_offload_no_ep(self):
        c = tp.validate_config(tp.config())
        flags = c['vllm_flags']
        self.assertEqual(flags['--tensor-parallel-size'], '2')
        self.assertEqual(flags['--cpu-offload-gb'], '0')
        self.assertEqual(flags['--gpu-memory-utilization'], '0.8')
        self.assertFalse(any(k in flags for k in ('--cpu-offload-params','--offload-backend',
            '--offload-group-size','--offload-num-in-group','--offload-prefetch-step','--enable-expert-parallel')))
        self.assertEqual(c['resources']['gpu_target_bytes'], 22*(1<<30))
        self.assertEqual(c['resources']['host_target_bytes'], 128*(1<<30))

    def test_reject_offload_and_ep_and_topology_overrides(self):
        for key,value in (('--cpu-offload-gb','1'),('--offload-backend','uva'),
            ('--cpu-offload-params',['w13_weight']),('--offload-group-size','1'),
            ('--offload-num-in-group','1'),('--offload-prefetch-step','1'),
            ('--enable-expert-parallel',True),('--tensor-parallel-size','1')):
            with self.subTest(key=key):
                c=copy.deepcopy(tp.config())
                c['vllm_flags'][key]=value
                with self.assertRaises(ValueError):
                    tp.validate_config(c)

    def test_sorted_locks_and_bound_coordinator(self):
        with tempfile.TemporaryDirectory(dir=tp.HERE) as name, mock.patch.object(tp.ct,'HERE',Path(name)):
            with tp.owner_locks(tp.UUIDS) as locks:
                self.assertEqual(list(locks),sorted(tp.UUIDS))
                tp.validate_locks(locks,os.getpid())
                for uuid in tp.UUIDS:
                    with self.assertRaises(ValueError):
                        with tp.ct.exclusive_owner_lock(uuid):
                            self.fail('Duplicate owner acquired lock')
                with self.assertRaises(ValueError):
                    tp.validate_locks({tp.UUIDS[0]:locks[tp.UUIDS[0]]},os.getpid())
            with tp.owner_locks(tp.UUIDS):
                pass

    def test_second_lock_failure_atomically_releases_first(self):
        first,second=sorted(tp.UUIDS)
        with tempfile.TemporaryDirectory(dir=tp.HERE) as name, mock.patch.object(tp.ct,'HERE',Path(name)):
            with tp.ct.exclusive_owner_lock(second):
                with self.assertRaises(ValueError):
                    with tp.owner_locks(tp.UUIDS):
                        self.fail('Acquired occupied pair')
                with tp.ct.exclusive_owner_lock(first):
                    pass
            with tp.owner_locks(tp.UUIDS):
                pass

    def test_invalid_lock_sets_rejected_before_acquisition(self):
        for values in ((tp.UUIDS[0],), (tp.UUIDS[0],tp.UUIDS[0]), (tp.UUIDS[0],'GPU-invalid')):
            with mock.patch.object(tp.ct,'exclusive_owner_lock') as acquire:
                with self.assertRaises(ValueError):
                    with tp.owner_locks(values):
                        self.fail('Invalid lock set')
                acquire.assert_not_called()

    def test_environment_no_inherited_cuda124(self):
        with mock.patch.dict(os.environ,{'LD_LIBRARY_PATH':'/CUDA12.4/lib','LD_PRELOAD':'bad','NCCL_P2P_DISABLE':'0'}):
            env=tp.spec_for(tp.config(),'/not-executed')['environment']
        self.assertNotIn('LD_LIBRARY_PATH',env)
        self.assertNotIn('LD_PRELOAD',env)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],','.join(tp.UUIDS))
        self.assertEqual(env['NCCL_P2P_DISABLE'],'1')

    def test_installed_source_audit_and_frozen_reference(self):
        self.assertEqual(tp.audit(tp.config())['status'],'pass_static_not_device_proof')
        rows,pins=tp.references()
        self.assertEqual([r['id'] for r in rows],['qa','arithmetic','chinese'])
        self.assertEqual([r['reference_text'] for r in rows],
                         [' Tokyo.\nThe capital of Japan is',' ?\n\n7 + 8 = ','北京，北京是中国的首都。那么'])
        self.assertTrue(all(len(r['reference_ids'])==8 for r in rows))
        self.assertIn('102c798baeb5200bc8ed8386eabed6146870b3b6da2a90fc6010677041352c7e',pins.values())

    def test_canary_mismatch_retained_and_failed(self):
        refs,_=tp.references()
        with tempfile.TemporaryDirectory(dir=tp.HERE) as name:
            root=Path(name)
            case=root/'canary8'
            case.mkdir()
            tp.ct.write(root/'canaries.json',refs)
            rows=[dict(status='success',input_length_matches=True,actual_input_tokens=r['input_tokens'],
                       actual_output_tokens=8,finish_reason='length',text=r['reference_text']) for r in refs]
            rows[1]['text']='15 forced expected answer is NOT the reference'
            report=dict(status='complete',runs=[dict(requests=rows)],warmups=[])
            def http(args):
                tp.ct.write(case/'http.json',report)
                return 0
            p=dict(config=tp.config(),spec=tp.spec_for(tp.config(),'/not-executed'))
            with mock.patch.object(tp.http,'main',side_effect=http):
                gate=tp.request_block(case,p,'canary8')
            self.assertEqual(gate['status'],'fail')
            self.assertFalse(gate['checks'][1]['passed'])
            self.assertEqual(gate['checks'][1]['observed_text'],rows[1]['text'])

    def test_offload_logs_are_contradictions_not_fallback_acceptance(self):
        evidence=tp.runtime_evidence('Offloader set to UVAOffloader\nfalling back to the Triton GDN decode path')
        self.assertTrue(evidence['contradictions'])
        self.assertTrue(evidence['evidence']['gdn'])
        self.assertFalse(evidence['exact_arithmetic_match'])


if __name__=='__main__':
    unittest.main()
