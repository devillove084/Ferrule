"""Regression coverage for ESRCH proc enumeration without losing root proof."""
from pathlib import Path
import sys
import unittest
from unittest import mock
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_v2 as tp
import sampling_v2 as sampling


class Sampling(unittest.TestCase):
    def setUp(self):
        self.root = dict(pid=42,birth_ticks=7,ppid=1,pgrp=42,session=42)
        self.sample = dict(pss_bytes=4096,rss_bytes=8192,processes=[dict(pid=42,Pss=4096,Rss=8192)])

    def test_esrch_resampled_with_evidence_not_zero(self):
        sampler=mock.Mock(side_effect=[ProcessLookupError(3,'No such process'),self.sample])
        result=sampling.host_sample(42,identity=lambda pid:self.root,sampler=sampler)
        self.assertEqual(result['pss_bytes'],4096)
        self.assertEqual(len(result['proc_enumeration_resamples']),1)
        self.assertEqual(sampler.call_count,2)

    def test_persistent_esrch_fails_with_bounded_attempts(self):
        sampler=mock.Mock(side_effect=ProcessLookupError(3,'No such process'))
        with self.assertRaises(ProcessLookupError):
            sampling.host_sample(42,identity=lambda pid:self.root,sampler=sampler)
        self.assertEqual(sampler.call_count,5)

    def test_root_identity_reuse_fails(self):
        identity=mock.Mock(side_effect=[self.root,self.root,dict(self.root,birth_ticks=8)])
        with self.assertRaises(ValueError):
            sampling.host_sample(42,identity=identity,sampler=lambda pid:self.sample)

    def test_missing_live_root_never_zero_or_retry(self):
        sampler=mock.Mock(return_value=dict(pss_bytes=0,processes=[]))
        with self.assertRaises(ValueError):
            sampling.host_sample(42,identity=lambda pid:self.root,sampler=sampler)
        self.assertEqual(sampler.call_count,1)

    def test_non_esrch_errors_not_swallowed(self):
        for error in (PermissionError('denied'),ValueError('Missing RSS/PSS')):
            with self.subTest(error=error):
                sampler=mock.Mock(side_effect=error)
                with self.assertRaises(type(error)):
                    sampling.host_sample(42,identity=lambda pid:self.root,sampler=sampler)
                self.assertEqual(sampler.call_count,1)

    def test_v2_inference_flags_unchanged(self):
        import experiment
        self.assertEqual(experiment.config()['vllm_flags'],tp.config()['vllm_flags'])
        self.assertEqual(experiment.config()['transport'],tp.config()['transport'])
        self.assertEqual(experiment.config()['resources'],tp.config()['resources'])


if __name__=='__main__':
    unittest.main()
