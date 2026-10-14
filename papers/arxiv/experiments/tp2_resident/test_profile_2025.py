"""CPU-only 2025.3 flags, frozen-binary policy, and unchanged lifecycle tests."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paired as p
import profile_2025 as compat


class Profile2025(unittest.TestCase):
    def tearDown(self):
        compat.configure(None)

    def test_flags_keep_gated_existing_wrapper_and_match_official_help(self):
        compat.configure(compat.CONFIG)
        help_text = (p.HERE/'runs/nsys-official-micro-20261009-001/research/profile-help.txt').read_text()
        with tempfile.TemporaryDirectory(dir=p.HERE) as name:
            argv = p.pre_serving.profile_server_argv(['/immutable/ferrule','serve'], Path(name), 'a'*64)
        self.assertEqual(argv[0], str(compat.TOOL/'target-linux-x64/nsys'))
        self.assertIn('--trace=cuda,nvtx,osrt,cublas', argv)
        for arg in argv[2:argv.index(sys.executable)]:
            self.assertIn(arg.split('=')[0], help_text)
        for bad in ('--cuda-trace-scope=', '--cuda-trace-all-apis=', '--gpu-video-devices='):
            self.assertFalse(any(a.startswith(bad) for a in argv))
        self.assertIn('--start-later=true', argv)
        self.assertIn(str(p.HERE/'instrumentation_capture.py'), argv)
        self.assertEqual(argv[-2:], ['/immutable/ferrule','serve'])

    def test_current_release_tool_selection_is_not_frozen_qualification(self):
        compat.configure(compat.TOOL_CONFIG)
        self.assertEqual(p.pre_serving.NSYS, str(compat.TOOL/'target-linux-x64/nsys'))
        self.assertEqual(p.pre_serving.PROFILE_TRACE, 'cuda,nvtx,osrt,cublas')
        self.assertFalse(compat.is_frozen({'profiler':compat.TOOL_CONFIG}))
        self.assertFalse(compat.is_frozen({}))
        self.assertTrue(compat.is_frozen({'profiler':compat.CONFIG}))
        with self.assertRaises(ValueError):
            compat.configure(dict(compat.TOOL_CONFIG, binary_sha256=compat.BINARY_SHA))

    def test_current_tool_policy_cannot_enter_smoke_or_no_trace_pair(self):
        with mock.patch.object(p, 'strict_gate'), mock.patch.object(p.ct, 'check_pins') as pins:
            for purpose in ('serving_smoke','paired','observed'):
                with self.assertRaisesRegex(ValueError, 'profiling-only'):
                    p.validate_admission({'profiler':compat.TOOL_CONFIG}, purpose)
            pins.assert_not_called()

    def test_current_tool_policy_still_requires_release_then_same_binary_smoke(self):
        admission = copy.deepcopy(p.ct.load(compat.ADMISSION))
        admission['profiler'] = compat.TOOL_CONFIG
        m = admission['engines']['ferrule']
        with mock.patch.object(p, 'strict_gate'), \
                mock.patch.object(p.ct, 'check_pins') as pins, \
                mock.patch.object(p.ct, 'validate_staged_binary'), \
                mock.patch.object(p.contract, '_engine_blockers', return_value=[]), \
                mock.patch.object(p, 'model_contract', return_value=m['model']), \
                mock.patch.object(p, 'source_audit', return_value={'source_pins':m['source_pins']}), \
                mock.patch.object(p.pre_serving, 'current_release_sources', return_value=m['source_pins']), \
                mock.patch.object(compat, 'validate_fixed_admission', side_effect=AssertionError('no frozen bypass')), \
                mock.patch.object(p.pre_serving, 'validate_release', side_effect=ValueError('release blocked')) as release, \
                mock.patch.object(p.pre_serving, 'validate_smoke', side_effect=ValueError('smoke blocked')) as smoke:
            with self.assertRaisesRegex(ValueError, 'release blocked'):
                p.validate_admission(admission, 'profiling')
            smoke.assert_not_called()
            release.side_effect = None
            with self.assertRaisesRegex(ValueError, 'smoke blocked'):
                p.validate_admission(admission, 'profiling')
            smoke.assert_called_once_with(m)
            smoke.side_effect = None
            self.assertEqual(p.validate_admission(admission, 'profiling'), admission)
            pins.assert_any_call(compat.TOOL_PINS)

    def test_default_2026_is_restored(self):
        compat.configure(compat.CONFIG)
        compat.configure(None)
        self.assertEqual(p.pre_serving.PROFILE_TRACE, 'cuda-sw,nvtx,osrt')
        self.assertIn('--cuda-trace-all-apis=true', p.pre_serving.PROFILE_SCOPE_FLAGS)

    def test_unknown_tool_or_binary_policy_rejected(self):
        for key in compat.CONFIG:
            config = dict(compat.CONFIG, **{key:'wrong'})
            with self.assertRaises(ValueError): compat.configure(config)

    def test_frozen_policy_cannot_change_environment_binary_or_be_benchmark(self):
        old = {'engines':{'ferrule':{'binary':{'sha256':compat.BINARY_SHA},'spec':{'environment':{'CUBLAS_WORKSPACE_CONFIG':':0:0'}}}}}
        fresh = dict(old,profiler=compat.CONFIG)
        compat.check_fixed_shape(fresh,'profiling',old)
        with self.assertRaises(ValueError): compat.check_fixed_shape(fresh,'paired',old)
        changed=copy.deepcopy(fresh)
        changed['engines']['ferrule']['spec']['environment']['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        with self.assertRaises(ValueError): compat.check_fixed_shape(changed,'profiling',old)

    def test_same_existing_server_lifecycle(self):
        compat.configure(compat.CONFIG)
        lifecycle = p.lifecycle(Path('/fixture'),'ferrule','fixture','profiling')
        for method in ('run','worker','requests'):
            self.assertEqual(getattr(lifecycle,method).__code__.co_code, getattr(p.baseline,method).__code__.co_code)
        self.assertIs(lifecycle.base.supervisor,p.base.supervisor)
        self.assertEqual(lifecycle.PROTOCOL['phases'],['hello2','canary8','profile'])
        self.assertEqual(p.experiment_slots('profiling'),('P1',))


if __name__ == '__main__':
    unittest.main()
