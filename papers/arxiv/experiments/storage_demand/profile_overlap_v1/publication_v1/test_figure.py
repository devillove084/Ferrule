"""CPU rendering tests; optional Matplotlib is required only for these tests."""
import copy
import importlib.util
from pathlib import Path
import unittest

import check as c

PLOT = c.HERE.parents[3] / 'figures/read-compute-overlap.py'


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib needed for rendering tests')
class Figure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('overlap_plot', PLOT)
        cls.plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.plot)
        _, _, cls.data, cls.results = c.load()

    def test_first_window_not_maximum_positive_overlap_selection(self):
        for arm in c.ARMS:
            self.assertEqual(self.plot.zoom_bounds(self.data[arm]),
                             (self.results[arm]['zoom_start_ns'], self.results[arm]['zoom_end_ns']))
            changed = dict(self.data[arm], kernel=[])
            self.assertEqual(self.plot.zoom_bounds(changed), self.plot.zoom_bounds(self.data[arm]))

    def test_no_cpu_range_used_as_compute(self):
        for arm in c.ARMS:
            tracks = self.plot.track_data(self.data[arm])
            self.assertIs(tracks['kernel'], self.data[arm]['kernel'])
            self.assertTrue(all(r['kind'] == 'read_subset' for r in tracks['read']))
            self.assertEqual(len(tracks['kernel']), 33403)

    def test_actual_figure_layout_and_zero_overlap_markers(self):
        from matplotlib import pyplot as plt
        fig, counts = self.plot.build_figure(self.data, self.results)
        try:
            report = self.plot.layout_check(fig, c)
            self.assertEqual(report['overlapping_text'], [])
            self.assertEqual(counts['blocking']['tracks'], dict(read=6, pread=36, kernel=46))
            self.assertEqual(counts['lookahead']['tracks'], dict(read=10, pread=60, kernel=82))
            self.assertEqual(sum(t['text'] == '0 ms (0%)' for t in report['text']), 2)
        finally:
            plt.close(fig)

    def test_empty_read_case_does_not_invent_io_or_divide_by_zero(self):
        from matplotlib import pyplot as plt
        data = {arm: dict(nvtx=[], pread=[], kernel=[]) for arm in c.ARMS}
        result = copy.deepcopy(self.results)
        for arm in c.ARMS:
            for key in c.METRICS:
                result[arm][key] = 0
        fig, counts = self.plot.build_figure(data, result)
        try:
            self.assertTrue(all(v['tracks'] == dict(read=0, pread=0, kernel=0) for v in counts.values()))
            texts = [t.get_text() for ax in fig.axes for t in ax.texts]
            self.assertEqual(texts.count('0.000000 ms (denominator zero)'), 4)
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
