"""CPU regression tests, including fail-closed mutations of the actual receipts."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from common import HERE, PINNED, Invalid, decode, encode, load, regular, safe_data, sha, unpack, verify_members
import check
import prespecified
import reconcile


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = check.data(HERE, 'study.json.gz')
        cls.schedule = load(HERE, 'design/requests.json')['schedule']
        cls.oracle = {r['id']: r for r in check.data(HERE, 'design/oracles.json')['references']}
        cls.rows = []
        for trial in cls.study['trials']:
            prefix = 'arms/' + trial + '/'
            row = check.data(HERE, prefix + 'journal.json')
            row['lifecycle'] = dict(requests=dict(observations=[check.data(HERE, prefix + s['name'] + '/observed.json.gz') for s in cls.schedule]))
            cls.rows.append(row)
        cls.events = [decode(line) for line in unpack(HERE / 'arms/P1-blocking/server-events.jsonl.gz').splitlines()]

    def report(self):
        return copy.deepcopy(dict(order=self.study['order'], lifecycles=self.rows))

    def request_args(self, name='qa-measured-1'):
        p = 'arms/P1-blocking/' + name + '/'
        return [check.data(HERE, p + 'observed.json.gz'), next(s for s in self.schedule if s['name'] == name),
                load(HERE, p + 'http-terminal.json'), check.data(HERE, p + 'http.json.gz'),
                load(HERE, p + 'issued.json'), copy.deepcopy(self.events), self.oracle]

    def test_full_offline_rebuild(self):
        result = check.rebuild(HERE)
        self.assertEqual(result, load(HERE, 'recomputed.json.gz'))
        self.assertEqual(result['coverage']['pipeline_invocation_jsons'], 1020)
        self.assertEqual(result['coverage']['actual_worker_generated_tokens'], 900)

    def test_all_3125_block_draws(self):
        value = prespecified.summarize(self.report())
        self.assertEqual(value, self.study['analysis'])
        for metric in value['aggregate'].values():
            self.assertEqual(metric['draws'], 3125)
            self.assertEqual(metric['ranks'], [79, 3047])

    def test_missing_pair_rejected(self):
        report = self.report()
        report['lifecycles'] = report['lifecycles'][:-2]
        with self.assertRaisesRegex(Invalid, 'ten planned'):
            prespecified.summarize(report)

    def test_duplicate_arm_rejected(self):
        report = self.report()
        report['lifecycles'][-1] = report['lifecycles'][0]
        with self.assertRaisesRegex(Invalid, 'duplicate lifecycle'):
            prespecified.summarize(report)

    def test_failed_pair_kept_aggregate_null(self):
        report = self.report()
        report['lifecycles'][3]['assessment']['can_continue'] = False
        value = prespecified.summarize(report)
        self.assertEqual(len(value['blocks']), 5)
        self.assertEqual(value['qualified_blocks'], 4)
        self.assertTrue(value['blocks'][1]['failure_pair_retained'])
        self.assertIsNone(value['aggregate'])

    def test_no_winner_only_selection(self):
        report = self.report()
        report['lifecycles'] = [row for row in report['lifecycles'] if row['mode'] == 'lookahead']
        with self.assertRaises(Invalid):
            prespecified.summarize(report)

    def test_nonfinite_measured_time_nulls_aggregate(self):
        report = self.report()
        report['lifecycles'][0]['lifecycle']['requests']['observations'][7]['actual']['request_e2e_seconds'] = float('nan')
        self.assertIsNone(prespecified.summarize(report)['aggregate'])

    def test_wrong_enqueue_ordinals_even_if_copies_agree(self):
        args = self.request_args()
        args[0]['trace']['enqueue_ordinals'][0] = 0
        for e in args[5]:
            f = e['record']['fields']
            if f['event'] == 'request_worker_token_enqueued' and f['request_id'] == 8 and f['emission_ordinal'] == 1:
                f['emission_ordinal'] = 0
        with self.assertRaisesRegex(Invalid, 'ordinals'):
            reconcile.client_tokens(*args)

    def test_wrong_actual_oracle_even_if_log_and_trace_agree(self):
        args = self.request_args()
        args[0]['trace']['actual_generated_ids'][0] += 1
        for e in args[5]:
            f = e['record']['fields']
            if f['event'] == 'request_worker_token_enqueued' and f['request_id'] == 8 and f['emission_ordinal'] == 1:
                f['actual_token_id'] += 1
        with self.assertRaisesRegex(Invalid, 'oracle mismatch'):
            reconcile.client_tokens(*args)

    def test_missing_done(self):
        args = self.request_args()
        for actual in (args[0]['actual'], args[2], args[3]['runs'][0]['requests'][0]):
            actual['raw_sse_events'].pop()
        with self.assertRaisesRegex(Invalid, 'SSE DONE'):
            reconcile.client_tokens(*args)

    def test_wrong_usage(self):
        args = self.request_args()
        for actual in (args[0]['actual'], args[2], args[3]['runs'][0]['requests'][0]):
            actual['usage']['completion_tokens'] = 7
        with self.assertRaisesRegex(Invalid, 'usage/length'):
            reconcile.client_tokens(*args)

    def test_actual_http_id_join(self):
        args = self.request_args()
        args[0]['trace']['actual_worker_request_id'] = 9
        with self.assertRaisesRegex(Invalid, 'ID join'):
            reconcile.client_tokens(*args)

    def test_pipeline_counter_reset(self):
        r = self.request_args('hello2')[0]
        previous = decode(r['logical_counters']['rows'][0]['fields']['expert_read_pipeline'])['stats']
        previous['submitted_reads'] += 1000000
        with self.assertRaisesRegex(Invalid, 'counter reset'):
            reconcile.invocations(r, self.events, 'blocking', previous, 0)

    def test_pipeline_reset_flag(self):
        r = self.request_args('hello2')[0]
        events = copy.deepcopy(self.events)
        e = next(e for e in events if e['record']['fields']['event'] == 'mixed_decoder_invocation')
        e['record']['fields']['expert_read_counter_reset_detected'] = True
        r['logical_counters']['rows'][0]['raw'] = copy.deepcopy(e['record'])
        r['logical_counters']['rows'][0]['fields'] = copy.deepcopy(e['record']['fields'])
        with self.assertRaisesRegex(Invalid, 'reset'):
            reconcile.invocations(r, events, 'blocking', None, 0)

    def test_host_peak_cap(self):
        ledger = load(HERE, 'arms/P1-blocking/read-ledger-assessment.json')['final_read_ledger']
        ledger['stats']['peak_host_charged_bytes'] = (64 << 20) + 1
        with self.assertRaisesRegex(Invalid, 'host bound'):
            reconcile.read_ledger(ledger, 'blocking')

    def test_missing_null_ledger_counter_rejected(self):
        for mutation in ('missing', 'null'):
            with self.subTest(mutation=mutation):
                ledger = load(HERE, 'arms/P1-blocking/read-ledger-assessment.json')['final_read_ledger']
                if mutation == 'missing':
                    del ledger['current_charge']['unknown_stack_bytes']
                else:
                    ledger['current_charge']['unknown_stack_bytes'] = None
                with self.assertRaises((Invalid, KeyError)):
                    reconcile.read_ledger(ledger, 'blocking')

    def test_nfs_reset_rejected_not_clamped(self):
        nfs = check.data(HERE, 'arms/P1-blocking/nfs-snapshots.json.gz')
        r = self.request_args('hello2')[0]
        nfs['hello2-post']['selected_mount']['counters']['normalreadbytes'] = 0
        with self.assertRaisesRegex(Invalid, 'reset'):
            reconcile.nfs_delta(nfs['hello2-pre'], nfs['hello2-post'], r['nfs'])

    def test_nfs_null_counter_not_zero(self):
        nfs = check.data(HERE, 'arms/P1-blocking/nfs-snapshots.json.gz')
        r = self.request_args('hello2')[0]
        nfs['hello2-post']['selected_mount']['counters']['serverreadbytes'] = None
        with self.assertRaisesRegex(Invalid, 'null'):
            reconcile.nfs_delta(nfs['hello2-pre'], nfs['hello2-post'], r['nfs'])

    def test_missing_backend_null_is_not_observation(self):
        nfs = check.data(HERE, 'arms/P1-blocking/nfs-snapshots.json.gz')
        r = self.request_args('hello2')[0]
        del r['nfs']['physical_nas_backend_bytes']
        with self.assertRaises(KeyError):
            reconcile.nfs_delta(nfs['hello2-pre'], nfs['hello2-post'], r['nfs'])

    def test_pid_birth_changed(self):
        r = self.request_args('hello2')[0]
        life = check.data(HERE, 'arms/P1-blocking/lifecycle.json.gz')
        r['process_io']['after']['root_identity']['birth_ticks'] += 1
        with self.assertRaisesRegex(Invalid, 'PID/birth'):
            reconcile.pid_io(r['process_io'], life['server_identity'], r['actual'])

    def test_missing_required_map(self):
        row = check.data(HERE, 'arms/P1-blocking/library-evidence.json')['ready']
        row['policy']['required'].pop()
        life = check.data(HERE, 'arms/P1-blocking/lifecycle.json.gz')
        runtime = check.data(HERE, 'design/source-and-build.json')['runtime_pins']
        with self.assertRaisesRegex(Invalid, '12/12'):
            reconcile.map_evidence(row, 'P1-blocking', 'ready', life['server_identity'], runtime)

    def test_retained_is_not_released(self):
        journal = check.data(HERE, 'arms/P1-blocking/journal.json')
        life = check.data(HERE, 'arms/P1-blocking/lifecycle.json.gz')
        life['closure']['final']['release'] = {'state': 'verified', 'value': 'released'}
        ledger = journal['read_ledger']['final_read_ledger']
        with self.assertRaisesRegex(Invalid, 'physical release'):
            check.qualify_close(journal, life, self.events, ledger)

    def test_deep_private_fields_and_nested_strings(self):
        for key in ('worker_token', 'original_worker_token', 'originalworker_token', 'environment', 'handshake', 'nonce'):
            with self.subTest(key=key):
                with self.assertRaises(Invalid):
                    safe_data({'nested': [{'context': json.dumps({'deeper': {key: 'synthetic-private-value'}})}]})

    def test_private_location_marker(self):
        with self.assertRaises(Invalid):
            safe_data({'context': {'url': 'https://private.invalid/receipt'}})

    def test_duplicate_json_keys(self):
        with self.assertRaisesRegex(Invalid, 'duplicate JSON'):
            decode('{"n": 1, "n": 2}')

    def test_stale_copy_hash(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            value = b'{"value": 1}\n'
            (root / 'one.json').write_bytes(value)
            (root / 'manifest.json').write_bytes(encode(dict(source_pins=PINNED, files={'one.json': dict(
                bytes=len(value), sha256='0' * 64, uncompressed_bytes=len(value), uncompressed_sha256=sha(value))})))
            with self.assertRaisesRegex(Invalid, 'stale copy hash'):
                verify_members(root)

    def test_stale_uncompressed_hash(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)
            value = b'{"value": 1}\n'
            compressed = gzip.compress(value, mtime=0)
            (root / 'one.json.gz').write_bytes(compressed)
            (root / 'manifest.json').write_bytes(encode(dict(source_pins=PINNED, files={'one.json.gz': dict(
                bytes=len(compressed), sha256=sha(compressed), uncompressed_bytes=len(value), uncompressed_sha256='0' * 64)})))
            with self.assertRaisesRegex(Invalid, 'stale uncompressed hash'):
                verify_members(root)

    def test_path_traversal(self):
        with self.assertRaisesRegex(Invalid, 'unsafe archive member'):
            regular(HERE, '../oracles.json')

    def test_isolated_copy_without_repository_or_raw_receipts(self):
        manifest = load(HERE, 'manifest.json')
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            isolated = Path(tmp) / 'portable'
            isolated.mkdir()
            for name in list(manifest['files']) + ['manifest.json']:
                target = isolated / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(HERE / name, target)
            result = subprocess.run([sys.executable, '-S', '-B', str(isolated / 'check.py')],
                                    cwd=isolated, capture_output=True, text=True, timeout=90,
                                    env={'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1'})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], 'passed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
