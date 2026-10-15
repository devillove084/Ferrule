"""Stdlib-only checks of the figure data and measured text layout."""
from copy import deepcopy
import unittest

import check
import evidence as e


class FigureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = e.load(e.HERE/'figure.json')
        cls.model = e.build_model()

    def test_published_figure_data_and_layout(self):
        check.validate_figure(self.record, self.model)
        self.assertEqual(self.record['renderer']['backend'], 'Agg')
        self.assertEqual(self.record['layout']['overlapping_text_pairs'], 0)

    def test_corrupt_all_figure_ledgers(self):
        for i in range(4):
            for field in e.LEDGER_KEYS:
                record = deepcopy(self.record)
                ledger = record['data']['cells'][i]['snapshots']['after_cleanup']['cache']
                ledger[field] += 1
                with self.subTest(cell=i, field=field), self.assertRaises(e.EvidenceError):
                    check.validate_figure(record, self.model)

    def test_false_content_and_additive_workspace(self):
        for key, value in (('state_content_bytes',0), ('aggregate_physical_retained_bytes',327174),
                           ('workspace_overlap_bytes',0)):
            record = deepcopy(self.record)
            record['data']['cells'][2]['snapshots']['after_cleanup'][key] = value
            with self.assertRaises(e.EvidenceError):
                check.validate_figure(record, self.model)
        record = deepcopy(self.record)
        record['data']['cells'][2]['committed_content']['equality'] = True
        with self.assertRaises(e.EvidenceError):
            check.validate_figure(record, self.model)

    def test_clipping_and_overlap_rejected(self):
        for kind in ('off_canvas','collision','false_count'):
            record = deepcopy(self.record)
            labels = record['layout']['labels']
            if kind == 'off_canvas':
                labels[0]['bounds'][0] = -0.01
            elif kind == 'collision':
                labels[1]['bounds'] = labels[0]['bounds'][:]
            else:
                record['layout']['text_count'] += 1
            with self.subTest(kind=kind), self.assertRaises(e.EvidenceError):
                check.validate_figure(record, self.model)

    def test_original_fixture_not_full_model(self):
        config = self.model['config']
        self.assertEqual((config['hidden'],config['experts_per_layer'],config['top_k']),(16,4,3))
        self.assertEqual(config['layers'], ['GatedDeltaNet','GQA'])
        self.assertEqual((config['history_dtype'],config['kv_dtype'],config['ssm_dtype']),('BF16','BF16','F32'))
        text = '\n'.join(x['text'] for x in self.record['layout']['labels'])
        for caveat in ('CONTENTS UNOBSERVED','subset, never added again','no device-wide release certificate',
                       'not integrated broker / CLI E5','4/4 cells PASS'):
            self.assertIn(caveat,text)


if __name__ == '__main__':
    unittest.main()
