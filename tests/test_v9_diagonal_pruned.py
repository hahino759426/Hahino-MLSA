import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from v9_diagonal import PrunedDiagonalSpatialAttention


class V9PrunedDiagonalTests(unittest.TestCase):
    def test_pruned_diagonal_links_stay_zero(self):
        model = PrunedDiagonalSpatialAttention(num_classes=1, device="cpu")
        model.frozen_links[0][0, 0, 0] = True
        model.exc_links[0][0, 0, 0] = 0.0

        model.train_coadapt_batch(torch.ones(2, 3920), torch.tensor([0]))

        self.assertEqual(float(model.exc_links[0][0, 0, 0]), 0.0)

    def test_diagonal_pruning_has_five_cluster_masks(self):
        model = PrunedDiagonalSpatialAttention(num_classes=10, device="cpu")

        self.assertEqual(len(model.frozen_links), 10)
        self.assertEqual(tuple(model.frozen_links[0].shape), (10, 16, 1280))
        self.assertEqual(tuple(model.frozen_links[4].shape), (10, 16, 3920))


if __name__ == "__main__":
    unittest.main()
