"""Offline destination/seal tests; never launch a GPU worker."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paired_read_v1.final_adapter_v3 import mapfixed as m


class MapfixedGenerationTests(unittest.TestCase):
    def test_new_destinations_and_worker_entry_restore(self):
        old = (m.binding.OUT, m.binding.BINDING_PATH, m.run_v3.ENTRY, m.run_v3.planned.worker)
        with m.generation():
            self.assertEqual(m.binding.OUT, m.OUT)
            self.assertEqual(m.binding._base.BINDING_PATH, m.binding.BINDING_PATH)
            self.assertEqual(m.binding._base.REVIEW_PATH, m.binding.REVIEW_PATH)
            self.assertEqual(m.binding._base.EXECUTION_PATH, m.binding.EXECUTION_PATH)
            self.assertEqual(m.run_v3.ENTRY, Path(m.__file__).resolve())
            self.assertIs(m.run_v3.planned.worker, m.maps.worker)
            self.assertIn(str(m.PREREGISTRATION.relative_to(m.HERE)), m.binding.SOURCE_FILES)
            self.assertTrue(set(m.SOURCES) <= set(m.binding.SOURCE_FILES))
        self.assertEqual(old, (m.binding.OUT, m.binding.BINDING_PATH,
                              m.run_v3.ENTRY, m.run_v3.planned.worker))

    def test_old_directories_are_not_resumable(self):
        with m.generation():
            self.assertEqual(m.run_v3.directory_check(m.DIRECTORY), m.DIRECTORY)
            for name in (*m.PRIOR, 'artifacts-v3-other'):
                with self.assertRaises(ValueError):
                    m.run_v3.directory_check(m.HERE / name)

    def test_inherited_default_paths_select_new_seals(self):
        with m.generation(), patch.object(m.binding, '_validate_binding') as candidate, \
                patch.object(m.binding, '_validate_execution') as execution:
            m.binding._base.validate_binding()
            m.binding._base.validate_execution()
            candidate.assert_called_once_with(m.OUT / 'actual-binding-v3.json')
            execution.assert_called_once_with(m.OUT / 'execution-sealed-v3.json')

    def test_execution_includes_independent_preregistration(self):
        with patch.object(m.binding, 'execution_value', return_value={'artifact_pins': {}}), \
                patch.object(m, 'validate_preregistration') as validate, \
                patch.object(m.ct, 'sha', return_value='a' * 64), m.generation():
            value = m.binding._base.execution_value()
            validate.assert_called_once_with()
            self.assertEqual(value['execution_generation'], m.GENERATION)
            self.assertEqual(value['preregistration_path'], str(m.PREREGISTRATION))
            self.assertEqual(value['preregistration_sha256'], 'a' * 64)

    def test_stale_preregistration_rejected(self):
        with patch.object(m.historical, 'unseal', return_value={'digest': 'x', 'pooling': True}), \
                patch.object(m, 'preregistration_value', return_value={'pooling': False}):
            with self.assertRaisesRegex(ValueError, 'preregistration stale'):
                m.validate_preregistration()


if __name__ == '__main__':
    unittest.main()
