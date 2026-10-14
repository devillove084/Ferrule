import copy
import unittest
from unittest import mock
import routed_down_launch as l


class RoutedLaunchTests(unittest.TestCase):
    def test_preflight_has_no_gpu_or_payload_read(self):
        with mock.patch.object(l,'all_idle') as idle, mock.patch.object(l.rc,'bundle') as bundle:
            self.assertEqual(l.main([]),0)
            idle.assert_not_called();bundle.assert_not_called()

    def test_missing_permission_precedes_io(self):
        with mock.patch.object(l.rc,'bundle') as bundle, mock.patch.object(l,'all_idle') as idle:
            with self.assertRaisesRegex(ValueError,'Explicit'):l.main(['--execute-cuda'])
            bundle.assert_not_called();idle.assert_not_called()

    def test_hard_bounds_and_pins(self):
        p=dict(task='routed-down-m1',timeout=120,capture_mib=16,cache_mib=64,bundle_sha256=l.rc.BUNDLE_PIN,reference_sha256=l.rc.REFERENCE_PIN,directory=str(l.l.HERE/'runs/unit-only'))
        l.rc.bounds(p)
        for key,value in [('timeout',121),('capture_mib',17),('cache_mib',65),('bundle_sha256','0'*64),('directory',str(l.rc.BUNDLE/'overwrite'))]:
            with self.subTest(key=key),self.assertRaises(ValueError):l.rc.bounds(dict(p,**{key:value}))

    def test_free_worker_rejected(self):
        with mock.patch.object(l.sys,'stdin',__import__('io').StringIO('{}')):
            with self.assertRaises((KeyError,ValueError)):l.main(['--_worker','0'*64])


if __name__=='__main__':unittest.main()


class PairSwiGLULaunchTests(unittest.TestCase):
    def test_pair_preflight_has_no_gpu_or_payload_read(self):
        with mock.patch.object(l,"all_idle") as idle, mock.patch.object(l.rc,"task_bundle") as bundle:
            self.assertEqual(l.main(["--pair-swiglu"]),0)
            idle.assert_not_called();bundle.assert_not_called()

    def test_pair_fixed_bounds_and_frozen_bundle(self):
        p=dict(task="pair-swiglu-m1-m5",timeout=120,capture_mib=16,cache_mib=64,
            bundle_sha256=l.rc.PAIR_PIN,reference_sha256=l.rc.REFERENCE_PIN,directory=str(l.l.HERE/"runs/unit-pair-only"))
        l.rc.bounds(p)
        for key,value in (("timeout",121),("bundle_sha256",l.rc.BUNDLE_PIN),("directory",str(l.rc.PAIR_BUNDLE))):
            with self.subTest(key=key),self.assertRaises(ValueError):l.rc.bounds(dict(p,**{key:value}))
        m=l.rc.task_bundle(p)
        self.assertEqual([f["rows"] for f in m["fixtures"]],[1,1,1,1,5])
        self.assertEqual(m["future_cuda_calls"]["gate_up_linear"],112)
        self.assertFalse(m["cuda_executed"])
