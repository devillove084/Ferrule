"""Finite NVML timeout handling; no inference/configuration retry."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import experiment_v4 as experiment
import sampling_v4 as sampling

class NVMLTimeouts(unittest.TestCase):
    def test_one_retry_is_recorded_not_zero(self):
        result=dict(devices=[dict(gpu_bytes=7)],apps=[])
        error=subprocess.TimeoutExpired(['/usr/bin/nvidia-smi','--query-gpu=x'],3)
        with mock.patch.object(sampling.telemetry,'all_gpu_sample',side_effect=[error,result]) as query:
            row=sampling.all_gpu_sample_retry(experiment.UUIDS[0],experiment.NOTE)
        self.assertEqual(query.call_count,2)
        self.assertEqual(row['devices'][0]['gpu_bytes'],7)
        self.assertEqual(len(row['nvidia_smi_timeout_retries']),1)
        self.assertEqual(row['nvidia_smi_timeout_retries'][0]['timeout_seconds'],3)

    def test_exhaustion_fails_closed_with_all_attempts(self):
        error=subprocess.TimeoutExpired(['/usr/bin/nvidia-smi'],3)
        with mock.patch.object(sampling.telemetry,'all_gpu_sample',side_effect=error) as query:
            with self.assertRaises(subprocess.TimeoutExpired):
                sampling.all_gpu_sample_retry(experiment.UUIDS[0],experiment.NOTE)
        self.assertEqual(query.call_count,2)
        self.assertIn('All bounded NVML attempts',error.__notes__[0])

    def test_other_errors_never_retried(self):
        for error in (ValueError('foreign/unknown device'),PermissionError('denied')):
            with mock.patch.object(sampling.telemetry,'all_gpu_sample',side_effect=error) as query:
                with self.assertRaises(type(error)):
                    sampling.all_gpu_sample_retry(experiment.UUIDS[0],experiment.NOTE)
            self.assertEqual(query.call_count,1)

    def test_inference_transport_reference_unchanged(self):
        import experiment_v3
        self.assertEqual(experiment.config()['vllm_flags'],experiment_v3.config()['vllm_flags'])
        self.assertEqual(experiment.config()['transport'],experiment_v3.config()['transport'])
        self.assertEqual(experiment.config()['workloads'],experiment_v3.config()['workloads'])
        self.assertEqual(sampling.NONSELECTED_DELTA_TOLERANCE_BYTES,1<<20)

if __name__=='__main__': unittest.main()
