import unittest
import math
from datetime import datetime, timedelta

import numpy as np
from utils import get_maxima_indices, get_periods, get_maxima, get_frequencies


class TestFrequencies(unittest.TestCase):

    Value = 0.9876883405951378

    def __init__(self, *args, **kwargs):
        super(TestFrequencies, self).__init__(*args, **kwargs)
        self.period = 10
        self.values = [math.sin(i / self.period * 2. * math.pi + 0.25 * math.pi) for i in range(55)]
        self.spikes = get_maxima(self.values)

    def test_get_maxima_indices(self):
        spikes = list(get_maxima_indices(self.values))
        self.assertListEqual(spikes, [1, 11, 21, 31, 41, 51])

    def test_get_maxima(self):
        nan = np.NaN
        expected = [nan, *([TestFrequencies.Value, *([nan] * 9)] * 6)]
        for i, value in enumerate(self.spikes):
            expect = expected[i]
            print(value, expect)
            self.assertTrue(math.isnan(value) and math.isnan(expect) or (value - expect) < 1e-7)


    def test_get_periods(self):
        periods = get_periods(self.values)
        self.assertListEqual(periods, [10, 10, 10, 10, 10])

    def test_get_frequencies(self):
        timestamps = [i * 1000 for i in range(len(self.values))]
        frequencies = get_frequencies(self.spikes, timestamps)
        self.assertAlmostEquals(frequencies[timestamps[0]], 0.11110905353604564)
        frequencies = get_frequencies(self.spikes, timestamps, interval=10000)
        self.assertListEqual(list(frequencies.values()), [0.1] * 7)
        frequencies = get_frequencies(self.spikes, timestamps, interval=5000)
        self.assertListEqual(list(frequencies.values()), [*([0.2, 0.0] * 5), 0.2,0.2])

    def test_filter_small_changes(self):
        values = [(i % 2) for i in range(20)]
        x = 10
        values[x] = 10000
        expected = [np.NaN] * len(values)
        expected[x] = values[x]

        spikes = get_maxima(values, threshold=0.01)
        self.assertListEqual(spikes, expected)


if __name__ == '__main__':
    unittest.main()
