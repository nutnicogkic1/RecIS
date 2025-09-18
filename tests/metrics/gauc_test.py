import unittest

import torch

from recis.metrics.gauc import Gauc


class TestGauc(unittest.TestCase):
    def test_gauc(self):
        labels = torch.tensor([1.0, 0.0, 1, 0, 1])
        predictions = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        indicator = torch.tensor([0, 1, 1, 1, 2])
        gauc_metric = Gauc()
        cur, acc = gauc_metric(labels, predictions, indicator)
        self.assertEqual(cur, 0.5)
        self.assertEqual(acc, 0.5)


if __name__ == "__main__":
    unittest.main()
