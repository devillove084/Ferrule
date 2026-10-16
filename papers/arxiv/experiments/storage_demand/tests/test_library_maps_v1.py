"""CPU-only proc-map regressions: no GPU query, server or inference launch."""
from contextlib import ExitStack, nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode = True
STORAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STORAGE))
from runnerhelpers import library_maps_v1 as maps, execution_v2 as ex
from paired_read_v1 import analysis
from paired_read_v1.map_evidence_v1 import adapter, calibrate


class MapEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.lib = self.directory / 'lib runtime.so.1'
        self.lib.write_bytes(b'\x7fELFfixture-only-not-a-real-executable')
        self.pin = hashlib.sha256(self.lib.read_bytes()).hexdigest()
        self.policy = dict(pins={str(self.lib): self.pin}, required={str(self.lib): ['build-role:cli']},
                           optional={}, executable='/fixture/cli', executable_sha256='a' * 64)
        self.root = dict(pid=123, birth_ticks=45, ppid=122, state='S')
        self.parent = dict(pid=122, birth_ticks=44, ppid=121, state='S')

    def line(self, path=None, deleted=False, inode=None):
        stat = self.lib.stat()
        return ('7f1234000000-7f1234001000 r-xp 00000000 '
                f'{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x} '
                f'{stat.st_ino if inode is None else inode} '
                f'{path or self.lib}' + (' (deleted)' if deleted else '') + '\n')

    def snapshot(self, raw):
        path = self.directory / 'saved.maps'
        path.write_text(raw)
        return dict(root_identity=dict(self.root), parent_identity=dict(self.parent), before=dict(self.root),
                    after=dict(self.root), after_executable=dict(self.root), parent_before=dict(self.parent),
                    parent_after=dict(self.parent), errors=[], executable_path=self.policy['executable'],
                    executable_sha256=self.policy['executable_sha256'],
                    expected_executable_sha256=self.policy['executable_sha256'],
                    raw_maps_path=str(path), raw_maps_sha256=maps.ct.sha(path), acquisition_end=maps.stamp())

    def test_deleted_anonymous_not_deleted_pinned_runtime(self):
        raw = self.line() + ''.join('7f1235000000-7f1235001000 rw-s 00000000 00:01 42 ' + p + ' (deleted)\n'
            for p in ('/dev/zero', '/memfd:allocator buffer', '/dev/shm/cuda shared buffer', '/SYSV00000000'))
        fixed = maps.assess(raw, self.policy)
        self.assertEqual(fixed['status'], 'passed')
        self.assertEqual(len(fixed['ignored_deleted_nonlibraries']), 4)
        self.assertEqual(maps.legacy_assess(raw, self.policy['pins'])['error'], 'loaded library mappings missing/deleted')
        self.assertIsNone(fixed['live_mapped_memory_sha256'])

    def test_deleted_required_path_exact_even_with_spaces(self):
        result = maps.assess(self.line(deleted=True), self.policy)
        self.assertEqual(result['deleted_library_paths'], [str(self.lib)])
        self.assertEqual(result['missing_required_paths'], [str(self.lib)])
        self.assertEqual(result['status'], 'failed')

    def test_missing_required_not_silently_dropped(self):
        extra = self.directory / 'librequired.so.2'
        self.policy['pins'][str(extra)] = 'b' * 64
        self.policy['required'][str(extra)] = ['DT_NEEDED']
        result = maps.assess(self.line(), self.policy)
        self.assertEqual(result['missing_required_paths'], [str(extra)])
        self.assertIs(result['required_pin_check'], False)

    def test_symlink_spaces_and_map_inode(self):
        link = self.directory / 'lib alias with spaces.so'
        link.symlink_to(self.lib)
        self.assertEqual(maps.assess(self.line(link), self.policy)['status'], 'passed')
        result = maps.assess(self.line(link, inode=1), self.policy)
        self.assertIn('mapped device/inode differs', str(result['errors']))
        self.assertEqual(result['status'], 'failed')

    def test_zero_observation_is_null_not_success(self):
        result = maps.assess('7f1235000000-7f1235001000 rw-p 00000000 00:00 0\n', self.policy)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['observed_library_count'], 0)
        self.assertIsNone(result['required_pin_check'])
        self.assertIsNone(analysis.arm_metrics(dict(assessment=dict(can_continue=True),
            read_ledger=dict(can_continue=True), lifecycle=dict(requests=dict(observations=[])))))

    def test_optional_lazy_absence_explicit_and_present_still_validated(self):
        optional = str(self.directory / 'liboptional.so')
        self.policy['pins'][optional] = 'b' * 64
        self.policy['optional'][optional] = 'other-role, not DT_NEEDED; checked if mapped'
        self.assertEqual(maps.assess(self.line(), self.policy)['status'], 'passed')
        self.assertEqual(maps.assess(self.line() + self.line(optional), self.policy)['status'], 'failed')
        self.policy['optional'].clear()
        with self.assertRaisesRegex(ValueError, 'classification'):
            maps.assess(self.line(), self.policy)

    def test_unbound_library_and_executable_deleted_memfd_fail(self):
        for raw in (self.line(self.directory / 'libunknown.so'),
                    self.line('/memfd:unknown executable', deleted=True)):
            with self.subTest(raw=raw):
                self.assertEqual(maps.assess(self.line() + raw, self.policy)['status'], 'failed')

    def test_malformed_and_ambiguous_map_paths_fail(self):
        for raw in ('garbage', self.line('/tmp/lib\\012newline.so')):
            with self.subTest(raw=raw):
                self.assertEqual(maps.assess(raw, self.policy)['status'], 'failed')

    def test_pid_birth_wrong_child_and_wrapper_cannot_pass(self):
        snapshot = self.snapshot(self.line())
        self.assertEqual(maps.validate_snapshot(snapshot, self.policy)['status'], 'passed')
        for key, field, value in (('after', 'birth_ticks', 46), ('after', 'ppid', 5),
                                   ('parent_after', 'birth_ticks', 50)):
            bad = copy.deepcopy(snapshot)
            bad[key][field] = value
            self.assertEqual(maps.validate_snapshot(bad, self.policy)['status'], 'failed')
        bad = copy.deepcopy(snapshot)
        bad['parent_identity']['pid'] = 124
        self.assertIn('wrong child', str(maps.validate_snapshot(bad, self.policy)['errors']))
        bad = copy.deepcopy(snapshot)
        bad['executable_path'] = '/usr/bin/python-wrapper'
        self.assertIn('not wrapper', str(maps.validate_snapshot(bad, self.policy)['errors']))

    def test_raw_saved_before_policy_failure_with_private_permissions(self):
        root = ex.observer.proc_identity(os.getpid())
        parent = ex.observer.proc_identity(root['ppid'])
        output = self.directory / 'private'
        def fail_policy(*args):
            self.assertTrue((output / 'maps.raw').is_file())
            self.assertTrue((output / 'snapshot.json').is_file())
            raise ValueError('required dependency absent')
        with mock.patch.object(maps, 'role_policy', side_effect=fail_policy):
            with self.assertRaisesRegex(ValueError, 'required dependency absent'):
                maps.library_maps(root, {}, directory=output, parent=parent, executable=sys.executable,
                    executable_sha256=maps.ct.sha(sys.executable), runtime={'pins': {}}, role='cli',
                    identity=ex.observer.proc_identity)
        self.assertEqual(output.stat().st_mode & 0o777, 0o700)
        for name in ('maps.raw', 'snapshot.json', 'validation.json'):
            self.assertEqual((output / name).stat().st_mode & 0o777, 0o600)
        snapshot = json.loads((output / 'snapshot.json').read_text())
        self.assertEqual(snapshot['root_identity'], root)
        self.assertIsNotNone(snapshot['raw_maps_sha256'])
        self.assertNotIn('environment', snapshot)

    def test_missing_model_proc_read_provenance_survives_real_capture(self):
        root = ex.observer.proc_identity(os.getpid())
        parent = ex.observer.proc_identity(root['ppid'])
        executable_hash = maps.ct.sha(sys.executable)
        read_bytes, readlink, sha = Path.read_bytes, Path.readlink, maps.ct.sha
        for name, operation in (('maps', 'read_bytes'), ('exe', 'readlink'), ('exe', 'sha256')):
            target = f'/proc/{root["pid"]}/{name}'
            def missing(path, *args, **kwargs):
                if str(path) == target: raise FileNotFoundError(2, 'Synthetic proc-read race', target)
                return {'read_bytes': read_bytes, 'readlink': readlink, 'sha256': sha}[operation](path, *args, **kwargs)
            with self.subTest(operation=operation), ExitStack() as stack:
                obj, method = (maps.ct, 'sha') if operation == 'sha256' else (Path, operation)
                stack.enter_context(mock.patch.object(obj, method, missing))
                result = maps.capture(self.directory / operation, root, parent, executable_hash, ex.observer.proc_identity)
            failed = result['proc_read_error']
            self.assertEqual(failed['identity'], root)
            self.assertEqual((failed['path'], failed['operation']), (target, operation))
            self.assertEqual(failed['exception_type'], 'FileNotFoundError')
            self.assertEqual(result['errors'], [failed['error']])
            self.assertLessEqual(result['acquisition_begin']['monotonic'], failed['observed']['monotonic'])
            self.assertLessEqual(failed['observed']['monotonic'], result['acquisition_end']['monotonic'])
            self.assertEqual(json.loads((self.directory / operation / 'snapshot.json').read_text())['proc_read_error'], failed)
            self.assertTrue(maps.validate_snapshot(result, self.policy)['errors'])

    def test_capture_persistence_failure_is_not_model_proc_read_provenance(self):
        root = ex.observer.proc_identity(os.getpid())
        parent = ex.observer.proc_identity(root['ppid'])
        write = maps.private_write
        def missing(path, value):
            if path.name == 'maps.raw': raise FileNotFoundError('Non-proc capture persistence failure')
            return write(path, value)
        with mock.patch.object(maps, 'private_write', side_effect=missing):
            result = maps.capture(self.directory / 'persistence', root, parent,
                                  maps.ct.sha(sys.executable), ex.observer.proc_identity)
        self.assertIn('persistence failure', result['errors'][0])
        self.assertNotIn('proc_read_error', result)

    def test_pid_reuse_before_capture_does_not_read_foreign_proc(self):
        output = self.directory / 'wrong-birth'
        actual = dict(self.root, birth_ticks=999)
        result = maps.capture(output, self.root, self.parent, 'a' * 64,
                              lambda pid: actual if pid == self.root['pid'] else self.parent)
        self.assertIsNone(result['raw_maps_path'])
        self.assertFalse((output / 'maps.raw').exists())
        self.assertIn('before capture', str(result['errors']))

    def test_role_and_transitive_dt_needed_not_all_pins_optional(self):
        extra = self.directory / 'libdependency.so'
        extra.write_bytes(self.lib.read_bytes())
        optional = self.directory / 'libotherrole.so'
        optional.write_bytes(self.lib.read_bytes())
        binary = self.directory / 'cli'
        binary.write_bytes(self.lib.read_bytes())
        pins = {str(p): maps.ct.sha(p) for p in (self.lib, extra, optional)}
        def elf(path):
            needed = ['lib runtime.so.1'] if str(path) == str(binary) else (
                ['libdependency.so'] if str(path) == str(self.lib) else [])
            return dict(needed=needed, soname=Path(path).name, interpreter=None)
        with mock.patch.object(maps, 'elf_dynamic', side_effect=elf):
            policy = maps.role_policy(binary, maps.ct.sha(binary),
                dict(pins=pins, per_role={'cli': {str(self.lib): self.pin}}), 'cli')
            self.assertEqual(set(policy['required']), {str(self.lib), str(extra)})
            self.assertEqual(set(policy['optional']), {str(optional)})
            del pins[str(extra)]
            with self.assertRaisesRegex(ValueError, 'DT_NEEDED lacks pinned ELF'):
                maps.role_policy(binary, maps.ct.sha(binary),
                    dict(pins=pins, per_role={'cli': {str(self.lib): self.pin}}), 'cli')

    def test_overlay_device_alias_is_explicit_not_a_memory_identity_pass(self):
        fields = self.line().split(maxsplit=5)
        fields[3] = 'ff:ff'
        raw = ' '.join(fields)
        mount = dict(filesystem='overlay', mountpoint='/', stat_device=[0, 48])
        with mock.patch.object(maps, 'overlay_mount', return_value=mount):
            result = maps.assess(raw, self.policy)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['libraries'], self.policy['pins'])
            self.assertIn('unobserved', result['overlay_device_aliases'][str(self.lib)]['backing_file_identity'])
            self.lib.write_bytes(b'\x7fELFchanged')
            self.assertEqual(maps.assess(raw, self.policy)['status'], 'failed')
        with mock.patch.object(maps, 'overlay_mount', return_value=None):
            self.assertIn('outside overlay', str(maps.assess(raw, self.policy)['errors']))

    def test_overlay_detection_requires_longest_mount_and_stat_device(self):
        mountinfo = ('1 0 0:48 / / rw - overlay overlay rw\n'
                     '2 1 253:2 / /home rw - ext4 /dev/vda2 rw\n')
        with mock.patch.object(Path, 'read_text', return_value=mountinfo):
            self.assertEqual(maps.overlay_mount('/usr/lib64/libc.so', [0, 48])['filesystem'], 'overlay')
            self.assertIsNone(maps.overlay_mount('/home/libc.so', [253, 2]))
            self.assertIsNone(maps.overlay_mount('/usr/lib64/libc.so', [9, 9]))

    def test_character_device_mappings_are_not_read(self):
        original = Path.open
        def opened(path, *args, **kwargs):
            self.assertNotEqual(str(path), '/dev/zero')
            return original(path, *args, **kwargs)
        raw = self.line() + '7f1235000000-7f1235001000 rw-s 00000000 00:01 42 /dev/zero\n'
        with mock.patch.object(Path, 'open', opened):
            self.assertEqual(maps.assess(raw, self.policy)['status'], 'passed')

    def test_birth_change_after_raw_retains_snapshot(self):
        root = ex.observer.proc_identity(os.getpid())
        parent = ex.observer.proc_identity(root['ppid'])
        calls = [0]
        def identity(pid):
            if pid != root['pid']:
                return parent
            calls[0] += 1
            return root if calls[0] == 1 else dict(root, birth_ticks=root['birth_ticks'] + 1)
        result = maps.capture(self.directory / 'raced', root, parent, maps.ct.sha(sys.executable), identity)
        self.assertTrue(Path(result['raw_maps_path']).is_file())
        self.assertNotEqual(result['before']['birth_ticks'], result['after']['birth_ticks'])
        self.assertEqual(maps.validate_snapshot(result, self.policy)['status'], 'failed')

    def test_adapter_rejects_wrong_root_and_restores_entry_points(self):
        original = ex.library_maps
        candidate = dict(binary='/fixture/cli', binary_sha256='a'*64, runtime_pin_manifest='/fixture/runtime')
        with mock.patch.object(ex, 'capture_root', return_value=(self.parent, self.root, {})), \
                mock.patch.object(adapter.ct, 'load', side_effect=[{'candidate': candidate}, {'pins': {}}]), \
                mock.patch.object(maps, 'library_maps', return_value={'libraries': {}}) as check:
            with adapter.mapped_worker(dict(directory=str(self.directory), candidate_path='/fixture/candidate')):
                ex.capture_root()
                with self.assertRaisesRegex(ValueError, 'authenticated owned server'):
                    ex.library_maps(dict(self.root, pid=999), {})
                ex.library_maps(self.root, {})
                self.assertEqual(check.call_args.kwargs['parent'], self.parent)
                self.assertEqual(check.call_args.kwargs['role'], 'cli')
            self.assertIs(ex.library_maps, original)

    def test_failed_calibration_cannot_launch_twice(self):
        output = self.directory / 'calibration'
        output.mkdir()
        prepared = dict(input_pins={}, historical_artifact_pins={}, spec={'argv': ['/fixture/cli']},
                        eight_uuids=['fixture']*8, uuid='fixture', limits=calibrate.LIMITS,
                        binary_sha256=calibrate.CLI_SHA)
        maps.private_write(output / 'prepared.json', prepared)
        maps.private_write(output / 'role-policy.json', {'pins': {}})
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(calibrate, 'OUT', output))
            for module, name, value in (
                (maps.ct, 'require_cpu_python', None), (maps.ct, 'check_pins', None),
                (maps.ct, 'sha', calibrate.CLI_SHA), (maps.ct, 'validate_owner_lock', None),
                (calibrate.pilot, 'validate_inventory', None),
                (calibrate.pilot.telemetry, 'authorize', None),
                (calibrate.pilot, 'idle', {'status': 'verified'}),
                (calibrate.pilot, 'baseline', {}),
            ):
                stack.enter_context(mock.patch.object(module, name, return_value=value))
            stack.enter_context(mock.patch.object(maps.ct, 'exclusive_owner_lock', side_effect=lambda u: nullcontext({})))
            stack.enter_context(mock.patch.object(calibrate.pilot.launch_guard, 'seal',
                side_effect=lambda p, d: (dict(p, worker_token='fixture-not-auth'), 'fixture')))
            launch = stack.enter_context(mock.patch.object(calibrate.pilot.supervisor, 'run_process',
                side_effect=ValueError('synthetic no launch completed')))
            old_umask = os.umask(0o077)
            try:
                result = calibrate.run(True)
                self.assertEqual(result['status'], 'failed')
                with self.assertRaises(FileExistsError):
                    calibrate.run(True)
            finally:
                os.umask(old_umask)
            self.assertEqual(launch.call_count, 1)


@unittest.skipUnless((calibrate.OUT / 'offline-comparison.json').is_file(), 'private one-shot calibration unavailable')
class SavedCalibrationTests(unittest.TestCase):
    def test_same_saved_snapshot_reproduces_original_and_requires_all_twelve_pins(self):
        snapshot = maps.ct.load(calibrate.OUT / 'owned-ready/ready-library-evidence/snapshot.json')
        policy = maps.ct.load(calibrate.OUT / 'role-policy.json')
        raw = Path(snapshot['raw_maps_path']).read_text()
        original = maps.legacy_assess(raw, policy['pins'])
        self.assertEqual(original['error'], 'loaded library mappings missing/deleted')
        result = maps.validate_snapshot(snapshot, policy)
        self.assertEqual(result['status'], 'passed')
        self.assertIs(result['required_pin_check'], True)
        self.assertEqual(result['observed_library_count'], 12)
        self.assertEqual(len(policy['required']), 12)
        self.assertEqual(policy['optional'], {})
        self.assertEqual(result['ignored_deleted_nonlibraries'], ['/dev/zero'])
        self.assertEqual(len(result['overlay_device_aliases']), 7)
        self.assertEqual(result['missing_required_paths'], [])
        self.assertEqual(result['deleted_library_paths'], [])
        self.assertIsNone(result['live_mapped_memory_sha256'])
        # The initial, over-strict device check remains a failed immutable receipt.
        first = maps.ct.load(calibrate.OUT / 'FINAL.json')
        self.assertEqual(first['status'], 'failed')
        self.assertTrue(first['ready'])
        self.assertTrue(first['authenticated_echild'])
        self.assertTrue(first['all_eight_idle'])
        self.assertEqual(first['inference_requests'], 0)


if __name__ == '__main__':
    unittest.main()
