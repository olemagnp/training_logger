import unittest
import numpy as np

from unittest.mock import patch
from training_logger.visualizers import LogVisualizer

class MockLogVisualizer(LogVisualizer):
    def __init__(self):
        self.logger = None
        self.prefix = ''

class LogVisualizerTest(unittest.TestCase):

    def test_smoothing(self):
        inp = np.array([i for i in range(10)])
        window_size = 3
        sigma = 6

        lv = MockLogVisualizer()

        out = lv.smooth_data(inp, window_size, sigma)

        self.assertEqual(inp.shape[0], out.shape[0], "Input and output shapes are different")

