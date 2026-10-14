"""Standard-library CPU checks for permission and authenticated paper custody."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import launch as l


class LaunchTests(unittest.TestCase):
    def test_preflight_never_reads_model_or_launches(self):
        with tempfile.TemporaryDirectory(dir=l.HERE) as root, mock.patch.object(l.supervisor,'run_process') as run, \
                mock.patch.object(l.telemetry,'gpu_sample') as sample:
            output=Path(root)/'plan'
            self.assertEqual(l.main(['--output',str(output),'--model','/not/read']),0)
            report=json.loads((output/'preflight.json').read_text())
            self.assertFalse(report['model_files_read'])
            self.assertFalse(report['full_reference_complete'])
            run.assert_not_called();sample.assert_not_called()

    def test_missing_grant_rejects_before_io_and_telemetry(self):
        with mock.patch.object(l.ct,'exclusive_owner_lock') as lock, mock.patch.object(l.telemetry,'gpu_sample') as sample:
            with self.assertRaisesRegex(ValueError,'Explicit'):
                l.main(['--execute-cuda','--model','/not/read'])
            lock.assert_not_called();sample.assert_not_called()

    def test_free_worker_without_authentication_rejected(self):
        with mock.patch.object(l.sys,'stdin',__import__('io').StringIO('{}')):
            with self.assertRaises((ValueError,KeyError)):
                l.main(['--_worker','0'*64])

    def test_protocol_rejects_slot_order_and_vllm_claim(self):
        p=l.protocol()
        self.assertIn('ascending expert ID',p['moe']['accumulation'])
        self.assertFalse(p['moe']['vllm_bitexact'])
        self.assertEqual(p['comparison']['atol'],.00025)
        self.assertEqual(p['comparison']['rtol'],.012)

    def test_existing_authenticated_helper_bounded_cpu_probe(self):
        # Probe is the maintained helper's own no-application containment check;
        # no oracle CUDA worker, locks, nvidia-smi or checkpoint is involved.
        receipt=l.supervisor.run_process(None,1)
        self.assertTrue(l.ct.successful_process(receipt),receipt)


if __name__=='__main__':unittest.main()
