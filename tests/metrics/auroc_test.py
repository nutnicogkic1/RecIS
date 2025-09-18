import unittest

import torch

from recis.metrics.auroc import AUROC


class AUROCTest(unittest.TestCase):
    def test_auroc(self):
        labels = torch.tensor([0, 0, 1, 0, 0, 0, 1, 0, 1, 1])
        prediction = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        metric = AUROC(num_thresholds=200)
        auc = metric(prediction, labels)
        self.assertAlmostEqual(float(auc), 0.7917, places=4)


if __name__ == "__main__":
    unittest.main()
