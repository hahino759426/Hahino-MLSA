import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from v9_pruning import PrunedSpatialAttention, pruned_coadapt_eta


class V9PruningTests(unittest.TestCase):
    def test_dead_links_stay_zero_during_isolation(self):
        model = PrunedSpatialAttention(num_classes=1, device="cpu")
        model.frozen_links[0][0, 0, 0] = True
        model.exc_links[0][0, 0, 0] = 0.0

        model.train_isolated_batch(torch.ones(2, 2352), torch.tensor([0]))

        self.assertEqual(float(model.exc_links[0][0, 0, 0]), 0.0)

    def test_dead_links_stay_zero_during_coadaptation(self):
        model = PrunedSpatialAttention(num_classes=1, device="cpu")
        model.frozen_links[0][0, 0, 0] = True
        model.exc_links[0][0, 0, 0] = 0.0

        model.train_coadapt_batch(torch.ones(2, 2352), torch.tensor([0]))

        self.assertEqual(float(model.exc_links[0][0, 0, 0]), 0.0)

    def test_pruned_decay_halves_in_last_five_epochs(self):
        self.assertEqual(pruned_coadapt_eta(5), 0.0001)
        self.assertEqual(pruned_coadapt_eta(6), 0.00005)
        self.assertEqual(pruned_coadapt_eta(10), 0.000003125)


if __name__ == "__main__":
    unittest.main()
