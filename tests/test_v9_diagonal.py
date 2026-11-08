import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from fixed_retina_diagonal import diagonal_sobel_retina
from v9_diagonal import DiagonalSpatialAttention


class V9DiagonalTests(unittest.TestCase):
    def test_retina_adds_two_binary_diagonal_channels(self):
        x = torch.rand(4, 784)

        result = diagonal_sobel_retina(x, device="cpu")

        self.assertEqual(tuple(result.shape), (4, 3920))
        self.assertTrue(bool(((result[:, 784:] == 0.0) | (result[:, 784:] == 1.0)).all()))

    def test_cluster_shapes_match_five_channel_input(self):
        model = DiagonalSpatialAttention(num_classes=10, device="cpu")

        for links in model.exc_links[:4] + model.inh_links[:4]:
            self.assertEqual(tuple(links.shape), (10, 16, 1280))
        self.assertEqual(tuple(model.exc_links[4].shape), (10, 16, 3920))
        self.assertEqual(tuple(model.inh_links[4].shape), (10, 16, 3920))

    def test_forward_uses_five_channel_spatial_fields(self):
        model = DiagonalSpatialAttention(num_classes=1, device="cpu")

        result = model.forward(torch.rand(2, 3920))

        self.assertEqual(tuple(result["score"].shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
