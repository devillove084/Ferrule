#!/usr/bin/env python3
"""Standard-library, CPU-only fixtures; no live hardware reads in unit tests."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('platform_inventory', Path(__file__).with_name('inventory.py'))
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class ParsingTests(unittest.TestCase):
    @staticmethod
    def cpu(processor, model, socket=None, core=None):
        text = f'processor : {processor}\nmodel name : {model}\n'
        if socket is not None:
            text += f'physical id : {socket}\n'
        if core is not None:
            text += f'core id : {core}\n'
        return text

    def test_heterogeneous_models_and_smt(self):
        text = '\n'.join([self.cpu(0, 'Model A', 0, 0), self.cpu(1, 'Model A', 0, 0),
                          self.cpu(2, 'Model B', 1, 0), self.cpu(3, 'Model B', 1, 1)])
        result = inventory.parse_cpuinfo(text)
        self.assertEqual(result['model_counts'], {'Model A': 2, 'Model B': 2})
        self.assertEqual(result['logical_processor_count'], 4)
        self.assertEqual(result['visible_socket_id_count'], 2)
        self.assertEqual(result['visible_socket_core_pair_count'], 3)

    def test_identical_duplicate_is_not_extra_thread(self):
        row = self.cpu(2, 'A', 0, 0)
        result = inventory.parse_cpuinfo(row + '\n' + row)
        self.assertEqual(result['logical_processor_count'], 1)
        self.assertEqual(result['duplicate_identical_processor_records'], 1)

    def test_conflicting_duplicate_processor_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate processor'):
            inventory.parse_cpuinfo(self.cpu(0, 'A', 0, 0) + '\n' + self.cpu(0, 'B', 0, 1))

    def test_conflicting_duplicate_field_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate CPU field'):
            inventory.parse_cpuinfo(self.cpu(0, 'A') + 'model name : B\n')

    def test_missing_topology_never_invents_sockets(self):
        result = inventory.parse_cpuinfo(self.cpu(0, 'A') + '\n' + self.cpu(1, 'B', 0, 1))
        self.assertIsNone(result['visible_socket_id_count'])
        self.assertIsNone(result['visible_socket_core_pair_count'])
        self.assertEqual(result['logical_processor_count'], 2)

    def test_memtotal_is_kib_not_decimal_kb(self):
        result = inventory.parse_meminfo('MemFree: 12 kB\nMemTotal: 1048576 kB\n')
        self.assertEqual(result['bytes'], 1073741824)
        self.assertEqual(result['GiB'], 1)
        self.assertEqual(result['GB'], 1.073741824)

    def test_bad_memtotal_units_or_duplicates_rejected(self):
        for text in ['MemTotal: 4 MB', 'MemTotal: 4 KB', 'MemTotal: 4', 'MemFree: 4 kB',
                     'MemTotal: 4 kB\nMemTotal: 4 kB', 'MemTotal: 0 kB']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                inventory.parse_meminfo(text)

    def test_cpulist_unique_and_sparse(self):
        self.assertEqual(inventory.cpu_list('0-2,2,7,10-11'),
                         {'list': '0-2,2,7,10-11', 'count': 6})
        self.assertEqual(inventory.cpu_list('')['count'], 0)
        for text in ['-1', '3-1', '0;cat', '1-', '0-999999999999']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                inventory.cpu_list(text)


class ResourceTests(unittest.TestCase):
    def test_v1_unlimited_threshold_and_finite_limit(self):
        for raw in ['-1', str(1 << 60), '9223372036854771712']:
            result = inventory.memory_limit(raw, 1)
            self.assertIsNone(result['finite_bytes'])
            self.assertEqual(result['status'], 'unlimited_sentinel_at_read_scope')
        for number in [0, 8 * 1024 ** 3, (1 << 60) - 1]:
            self.assertEqual(inventory.memory_limit(str(number), 1)['finite_bytes'], number)

    def test_v2_max_is_distinct_from_numeric_limit(self):
        self.assertIsNone(inventory.memory_limit('max', 2)['finite_bytes'])
        self.assertEqual(inventory.memory_limit(str(1 << 60), 2)['finite_bytes'], 1 << 60)
        with self.assertRaises(ValueError):
            inventory.memory_limit('-1', 2)

    def test_cpu_quota_not_a_thread_count(self):
        self.assertEqual(inventory.cpu_quota('150000', '100000')['cpu_time_ratio'], 1.5)
        for quota in ['-1', 'max']:
            self.assertIsNone(inventory.cpu_quota(quota, '100000')['cpu_time_ratio'])
        with self.assertRaises(ValueError):
            inventory.cpu_quota('1000', '0')

    def test_exact_v1_membership_relative_to_mount_root(self):
        groups = '4:cpuset,cpu,cpuacct:/tenant/container/task\n10:memory:/tenant/container/task\n'
        mounts = ('21 1 0:21 /tenant/container /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpuset,cpu,cpuacct\n'
                  '22 1 0:22 /tenant/container /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n'
                  '23 1 0:22 /tenant/container /private/overlay/sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n')
        result = inventory.resolve_cgroups(groups, mounts)
        self.assertEqual(str(result['memory']['directory']), '/sys/fs/cgroup/memory/task')
        self.assertEqual(str(result['cpu']['directory']), '/sys/fs/cgroup/cpu/task')
        self.assertEqual(result['memory']['version'], 1)

    def test_v2_root_membership(self):
        result = inventory.resolve_cgroups('0::/\n', '1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n')
        self.assertEqual(set(result), {'cpu', 'cpuset', 'memory'})
        self.assertEqual(result['memory']['version'], 2)
        self.assertEqual(str(result['memory']['directory']), '/sys/fs/cgroup')

    def test_mismatched_root_not_assumed_to_be_current(self):
        result = inventory.resolve_cgroups('10:memory:/unrelated\n',
            '1 0 0:1 /container /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n')
        self.assertEqual(result, {})

    def test_mount_escape_and_traversal(self):
        self.assertEqual(inventory.unescape_mount('/a\\040b'), '/a b')
        with self.assertRaises(ValueError):
            inventory.resolve_cgroups('10:memory:/../other\n', '')

    def test_hidden_ancestors_and_private_paths_not_published(self):
        mapping = {
            '/proc/self/cgroup': '10:memory:/private-container\n',
            '/proc/self/mountinfo': '1 0 0:1 /private-container /sys/fs/cgroup/memory rw - cgroup cgroup rw,memory\n',
            '/sys/fs/cgroup/memory/memory.limit_in_bytes': '9223372036854771712',
            '/sys/fs/cgroup/memory/memory.memsw.limit_in_bytes': '9223372036854771712',
            '/sys/fs/cgroup/memory/memory.use_hierarchy': '1',
            '/sys/fs/cgroup/memory/memory.stat': 'hierarchical_memory_limit 9223372036854771712\nrss 400\n',
        }
        with mock.patch.object(inventory, 'read_text', side_effect=lambda path: mapping[str(path)]):
            result = inventory.collect_cgroups()
        memory = result['controllers']['memory']
        self.assertTrue(memory['hidden_ancestors_unobserved'])
        self.assertTrue(memory['current_equals_visible_mount_root'])
        self.assertNotIn('private-container', json.dumps(result))
        self.assertNotIn('rss', memory['levels'][0]['fields']['memory.stat']['value'])


class BindingTests(unittest.TestCase):
    BOOT = '00000000-0000-4000-8000-000000000001'
    OTHER = '00000000-0000-4000-8000-000000000002'

    def test_same_boot_uses_stripped_text_hash(self):
        digest = inventory.boot_hash(self.BOOT)
        result = inventory.compare_boot_ids([self.BOOT, self.BOOT + '\n'], digest)
        self.assertEqual(result['status'], 'same_boot')
        self.assertEqual(result['matches'], 2)
        self.assertNotIn(self.BOOT, json.dumps(result))

    def test_one_mismatch_blocks_study_binding(self):
        result = inventory.compare_boot_ids([self.BOOT, self.OTHER], inventory.boot_hash(self.BOOT))
        self.assertEqual(result['status'], 'cannot_bind_posthoc_to_study')
        self.assertEqual(result['mismatches'], 1)

    def test_missing_invalid_or_empty_never_passes(self):
        for values in [[], [None], [''], ['not-a-uuid'], [self.BOOT, None]]:
            with self.subTest(values=values):
                result = inventory.compare_boot_ids(values, inventory.boot_hash(self.BOOT))
                self.assertEqual(result['status'], 'cannot_bind_posthoc_to_study')

    def test_missing_boot_stops_before_current_inventory(self):
        study = {'binding_status': 'cannot_bind_posthoc_to_study'}
        with mock.patch.object(inventory, 'inspect_study', return_value=study), \
             mock.patch.object(inventory, 'read_text', return_value=self.BOOT), \
             mock.patch.object(inventory, 'collect_current') as current:
            result = inventory.collect(Path('.'))
        self.assertIsNone(result['current_inventory'])
        current.assert_not_called()

    def test_reboot_during_collection_invalidates_inventory(self):
        study = {'binding_status': 'POSTHOC_INVENTORY_SAME_BOOT'}
        with mock.patch.object(inventory, 'inspect_study', return_value=study), \
             mock.patch.object(inventory, 'read_text', side_effect=[self.BOOT, self.OTHER]), \
             mock.patch.object(inventory, 'collect_current', return_value={'fixture': True}):
            result = inventory.collect(Path('.'))
        self.assertEqual(result['status'], 'cannot_bind_posthoc_to_study')
        self.assertIsNone(result['current_inventory'])

    def test_json_field_search_counts_boots_not_bootstrap(self):
        values, keys = [], inventory.collections.Counter()
        inventory.inspect_fields({'bootstrap': 5, 'nested': [{'boot_id': self.BOOT}],
                                  'MemTotal': 12, 'facade_kernel_launches': 3}, values, keys)
        self.assertEqual(values, [self.BOOT])
        self.assertEqual(keys, {'MemTotal': 1})

    def test_bounded_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture'
            path.write_text('12345')
            with self.assertRaisesRegex(ValueError, 'bounded read'):
                inventory.read_bytes(path, 4)


class ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent
        cls.raw = (cls.root / 'posthoc-20261015.json').read_bytes()
        cls.receipt = json.loads(cls.raw)

    def test_receipt_pins_unchanged_collector_source(self):
        self.assertEqual(self.receipt['collector_source_sha256'],
                         inventory.sha256((self.root / 'inventory.py').read_bytes()))

    def test_all_arms_and_boot_occurrences_accounted(self):
        study = self.receipt['study_binding']
        self.assertEqual(self.receipt['status'], 'POSTHOC_INVENTORY_SAME_BOOT')
        self.assertTrue(self.receipt['boot_hash_unchanged_during_collection'])
        self.assertEqual(study['original_json_files_checked_against_public_provenance'], 1037)
        self.assertEqual(study['public_json_files_checked_against_manifest'], 597)
        self.assertEqual(study['original_mount_text_files_checked'], 300)
        comparison = study['all_original_json_boot_ids']
        self.assertEqual(comparison['occurrences'], 341)
        self.assertEqual(comparison['matches'], 341)
        self.assertEqual(comparison['mismatches'], 0)
        self.assertEqual(comparison['missing_or_invalid'], 0)
        self.assertEqual(comparison['recorded_boot_id_sha256_counts'],
                         {self.receipt['boot_id_sha256']: 341})
        self.assertEqual(len(study['phase_labels']), 30)
        self.assertEqual({row['arm'] for row in study['per_arm']},
                         {f'P{i}-{mode}' for i in range(1, 6) for mode in ('blocking', 'lookahead')})
        self.assertEqual(sum(row['phase_snapshots'] for row in study['per_arm']), 300)
        for row in study['per_arm']:
            self.assertEqual(row['phase_snapshots'], 30)
            self.assertEqual(row['boot_comparison']['matches'], 30)
            self.assertEqual(row['boot_comparison']['status'], 'same_boot')

    def test_posthoc_time_and_derived_hardware_counts(self):
        current, study = self.receipt['current_inventory'], self.receipt['study_binding']
        self.assertLess(study['study_ended']['unix_ns'], current['begin']['unix_ns'])
        self.assertLess(study['study_ended']['monotonic'] * 1e9,
                        current['begin']['monotonic_ns'])
        self.assertLess(current['begin']['monotonic_ns'], current['end']['monotonic_ns'])
        cpu = current['cpu']
        rows = cpu['topology_rows']
        self.assertEqual(len({row[0] for row in rows}), cpu['logical_processor_count'])
        self.assertEqual(len({row[1] for row in rows}), cpu['visible_socket_id_count'])
        self.assertEqual(len({(row[1], row[2]) for row in rows}), cpu['visible_socket_core_pair_count'])
        self.assertEqual(sum(cpu['model_counts'].values()), cpu['logical_processor_count'])
        memory = current['memory']
        self.assertEqual(memory['bytes'], memory['source_value'] * 1024)
        self.assertEqual(memory['GiB'], memory['bytes'] / 1024 ** 3)
        self.assertEqual(memory['GB'], memory['bytes'] / 1000 ** 3)

    def test_receipt_does_not_hide_resource_scope_unknowns(self):
        cgroups = self.receipt['current_inventory']['cgroups']
        self.assertTrue(cgroups['membership_unchanged_during_collection'])
        for group in cgroups['controllers'].values():
            self.assertTrue(group['hidden_ancestors_unobserved'])
            self.assertTrue(group['current_equals_visible_mount_root'])
            self.assertEqual(group['effective_limit_claim'], 'not_attested; local observations only')
        self.assertIsNone(cgroups['controllers']['memory']['levels'][0]['local_memory_limit']['finite_bytes'])
        self.assertIsNone(cgroups['controllers']['cpu']['levels'][0]['local_cpu_quota']['cpu_time_ratio'])
        self.assertEqual(self.receipt['study_binding']['resource_field_search']['original_matches'], {})
        self.assertEqual(self.receipt['study_binding']['resource_field_search']['public_matches'], {})

    def test_public_receipt_has_no_raw_identifiers_or_private_location_markers(self):
        self.assertIsNone(inventory.re.search(
            rb'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', self.raw))
        self.assertIsNone(inventory.re.search(rb'(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])', self.raw))
        for marker in [b'/home/', b'/mnt/', b'://', b'GPU-']:
            self.assertNotIn(marker, self.raw)
        forbidden = {'boot_id', 'hostname', 'machine_id', 'mac', 'serial', 'ip', 'environment',
                     'raw_text', 'raw_lines', 'mount_namespace', 'mountpoint', 'worker_token',
                     'nonce', 'handshake', 'password', 'credential', 'authorization'}
        def walk(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        walk(self.receipt)

    def test_total_new_file_size_below_100_kib(self):
        self.assertLess(sum((self.root / name).stat().st_size for name in
                            ('inventory.py', 'test_inventory.py', 'posthoc-20261015.json', 'README.md')),
                        100 * 1024)


if __name__ == '__main__':
    unittest.main()
