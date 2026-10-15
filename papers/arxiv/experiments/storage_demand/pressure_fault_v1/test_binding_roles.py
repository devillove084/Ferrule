"""CPU-only role and immutable-source mutation fixtures; never edit the candidate."""
import copy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import binding_roles as r
import subset_adapter as a


class RoleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.name = 'crates/fixture/src/cuda_tests.rs'
        self.live = self.root / self.name
        self.live.parent.mkdir(parents=True)
        self.live.write_text('// old test source\n')
        self.frozen = self.root / 'frozen-source'
        self.member = self.frozen / self.name
        self.member.parent.mkdir(parents=True)
        self.member.write_bytes(self.live.read_bytes())
        self.member.chmod(0o444)
        self.pin = r.sha(self.member)
        self.source = dict(frozen_source=str(self.frozen), pins_relative_to_frozen_source={self.name: self.pin})
        self.files = {self.name: self.pin}
        self.roles = {self.name: 'compiled_source'}

    def validate(self):
        return r.validate_entries(self.files, self.roles, self.root, self.source)

    def test_unrelated_live_compiled_test_edit_is_not_consumed(self):
        self.live.write_text('// new cfg(test) include\n')
        pins, relation = self.validate()
        self.assertEqual(pins, {str(self.member): self.pin})
        self.assertNotIn(str(self.live), pins)
        self.assertFalse(relation[0]['live_matches_compiled'])
        self.assertFalse(relation[0]['live_checkout_consumed'])
        self.assertEqual(relation[0]['current_live_sha256'], r.sha(self.live))

    def test_frozen_member_mutation_rejected_even_if_live_matches(self):
        self.member.chmod(0o644)
        self.member.write_text('// mutated immutable fixture\n')
        self.member.chmod(0o444)
        with self.assertRaisesRegex(ValueError, 'changed compiled member'):
            self.validate()

    def test_unarchived_member_or_wrong_pin_cannot_be_compiled_source(self):
        self.source['pins_relative_to_frozen_source'] = {}
        with self.assertRaisesRegex(ValueError, 'not in candidate archive'):
            self.validate()
        self.source['pins_relative_to_frozen_source'] = {self.name: self.pin}
        self.files[self.name] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'not in candidate archive'):
            self.validate()

    def test_mutable_member_and_unknown_or_missing_role_rejected(self):
        self.member.chmod(0o644)
        with self.assertRaisesRegex(ValueError, 'mutable compiled'):
            self.validate()
        self.member.chmod(0o444)
        for roles in ({}, {self.name: 'ignore_changed_file'}, {self.name: 'unknown'}):
            self.roles = roles
            with self.assertRaisesRegex(ValueError, 'input role'):
                self.validate()

    def test_live_host_helper_change_fails_even_when_frozen_script_exists(self):
        name = 'scripts/helper.py'
        path = self.root / name
        path.parent.mkdir()
        path.write_text('control = 1\n')
        frozen = self.frozen / name
        frozen.parent.mkdir()
        frozen.write_bytes(path.read_bytes())
        frozen.chmod(0o444)
        self.source['pins_relative_to_frozen_source'][name] = r.sha(frozen)
        self.files, self.roles = {name: r.sha(path)}, {name: 'host_control'}
        path.write_text('control = 2\n')
        with self.assertRaisesRegex(ValueError, 'changed host_control input'):
            self.validate()
        self.roles[name] = 'compiled_source'
        with self.assertRaisesRegex(ValueError, 'cannot be compiled_source'):
            self.validate()

    def test_reference_and_canonical_protocol_are_live_strict(self):
        for role in ('references', 'protocol'):
            self.roles[self.name] = role
            self.live.write_text('changed ' + role)
            with self.assertRaisesRegex(ValueError, 'changed ' + role):
                self.validate()

    def archive_fixture(self):
        cargo = self.frozen / 'Cargo.toml'
        cargo.write_text('[profile.release]\nopt-level=3\nlto="fat"\ncodegen-units=1\npanic="unwind"\nstrip="symbols"\n')
        cargo.chmod(0o444)
        members = dict(self.source['pins_relative_to_frozen_source'], **{'Cargo.toml': r.sha(cargo)})
        archive = self.root / 'source.tar'
        with tarfile.open(archive, 'w') as stream:
            for name in members:
                stream.add(self.frozen / name, arcname=name, recursive=False)
        archive.chmod(0o444)
        self.frozen.chmod(0o555)
        original = {str(a.ct.ROOT / name): pin for name, pin in members.items()}
        source = dict(schema='ferrule.expert-read.committed-source-freeze.v1',
            immutable_readonly_inputs=True, worktree_edits_excluded=True, frozen_source=str(self.frozen),
            archive=str(archive), pins_relative_to_frozen_source=members, original_source_pins=original,
            source_pins_digest_relative=r.digest(members))
        identity = dict(commit='fixture', source_members=len(members), source_digest=r.digest(original),
                        source_archive_sha256=r.sha(archive))
        (self.root / 'source-freeze.json').write_text(json.dumps(source))
        (self.root / 'archive-exact-members.json').write_text(json.dumps(dict(commit='fixture',
            file_count=len(members), members={n: dict(sha256=p) for n, p in members.items()})))
        return archive, identity, members

    def test_exact_archive_member_mutation_rejected(self):
        archive, identity, members = self.archive_fixture()
        with mock.patch.object(a.prior_evidence, 'ROOT', self.root):
            inventory, _ = a.prior_evidence.source_inventory(identity, dict(pins_relative_to_directory={}))
            self.assertTrue(inventory['archive_and_all_pinned_source_bytes_verified'])
            archive.chmod(0o644)
            with tarfile.open(archive, 'w') as stream:
                for name in members:
                    data = b'changed archive member' if name == self.name else (self.frozen / name).read_bytes()
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    stream.addfile(member, io.BytesIO(data))
            archive.chmod(0o444)
            with self.assertRaisesRegex(ValueError, 'archive identity differs'):
                a.prior_evidence.source_inventory(identity, dict(pins_relative_to_directory={}))
            # Even a freshly hashed archive cannot override the pinned member map.
            identity['source_archive_sha256'] = r.sha(archive)
            with self.assertRaisesRegex(ValueError, 'archive member bytes differ'):
                a.prior_evidence.source_inventory(identity, dict(pins_relative_to_directory={}))

    def test_model_identity_change_is_not_a_compiled_source_exception(self):
        rows = {}
        for index in range(14):
            path = self.root / ('shard-' + str(index))
            path.write_bytes(b'shard')
            st = path.stat()
            rows[str(path)] = [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_dev, st.st_ino]
        r.validate_model_identity(rows)
        path.write_bytes(b'changed shard')
        with self.assertRaisesRegex(ValueError, 'physical identity changed'):
            r.validate_model_identity(rows)

    def test_runtime_library_pin_mismatch_remains_strict(self):
        library = self.root / 'libfixture.so'
        library.write_bytes(b'library')
        pins = {str(library): r.sha(library)}
        a.ct.check_pins(pins)
        library.write_bytes(b'changed runtime')
        with self.assertRaises(ValueError):
            a.ct.check_pins(pins)


class CurrentBindingTests(unittest.TestCase):
    def test_old_seal_unchanged_new_role_map_covers_every_predecessor(self):
        spec = r.role_root()
        old = r.load(r.ROOT / spec['predecessor_binding'])
        self.assertEqual(old['schema'], 'ferrule.storage-demand.pressure-fault-inputs.v2')
        self.assertTrue(set(old['files']) <= set(spec['roles']))
        self.assertNotEqual(r.BINDING_PATH, r.ROOT / spec['predecessor_binding'])
        self.assertFalse(spec['historical_resume_permitted'])
        self.assertEqual(spec['next_directory'], 'artifacts-go-v2-20261015-02')
        self.assertEqual(len(spec['committed_sources']), 22)

    def test_actual_frozen_relation_keeps_live_runtime_modules_strict(self):
        value = r.load(r.BINDING_PATH)
        source = r.load(r.ROOT / value['source_freeze'])
        pins, relations = r.validate_entries(value['files'], value['roles'], r.ROOT, source)
        for row in relations:
            self.assertIn(row['consumed_path'], pins)
            self.assertNotIn(str(r.ROOT / row['path']), pins)
        for name in ('scripts/_owned_subreaper.py', 'scripts/_profile_capture.py',
                     'scripts/gpu_idle_window_v2.py', 'papers/arxiv/experiments/launch_guard.py'):
            self.assertEqual(value['roles'][name], 'host_control')
            self.assertIn(str(r.ROOT / name), pins)
        self.assertEqual(a.p.LIMITS, value['budget_variant']['limits'])
        self.assertEqual(value['budget_variant'], r.load(a.p.PREPARATION / 'bindings.json')['budget_variant'])

    def test_private_binding_is_exclusive_create(self):
        with mock.patch.object(r, 'source_context', return_value=(r.load(r.ROOT / r.role_root()['source_freeze']), {})):
            with self.assertRaises(FileExistsError):
                r.create_binding()


if __name__ == '__main__':
    unittest.main()
