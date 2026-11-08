import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from v9_8_network import MixedLinkSpatialAttention


class V98FixedTests(unittest.TestCase):
    def test_strict_configuration_clips_to_zero_and_one(self):
        model = MixedLinkSpatialAttention(num_classes=1, link_min=0.0, link_max=1.0, device="cpu")

        model.train_isolated_batch(torch.ones(1, 2352), torch.tensor([1]))

        for links in model.exc_links + model.inh_links:
            self.assertGreaterEqual(float(links.min()), 0.0)
            self.assertLessEqual(float(links.max()), 1.0)

    def test_mixed_configuration_allows_negative_links_within_bounds(self):
        model = MixedLinkSpatialAttention(num_classes=1, eta=2.0, link_min=-1.0, link_max=1.0, device="cpu")

        model.train_isolated_batch(torch.ones(1, 2352), torch.tensor([1]))

        self.assertLess(float(model.exc_links[0].min()), 0.0)
        self.assertGreaterEqual(float(model.exc_links[0].min()), -1.0)
        self.assertLessEqual(float(model.exc_links[0].max()), 1.0)

    def test_phase_3_uses_the_same_configured_bounds(self):
        model = MixedLinkSpatialAttention(num_classes=1, link_min=-1.0, link_max=1.0, device="cpu")

        model.train_coadapt_batch(torch.ones(1, 2352), torch.tensor([1]))

        for links in model.exc_links + model.inh_links:
            self.assertGreaterEqual(float(links.min()), -1.0)
            self.assertLessEqual(float(links.max()), 1.0)

    def test_attention_is_normalized_again_after_clipping(self):
        model = MixedLinkSpatialAttention(num_classes=10, eta_w=10.0, device="cpu")
        model.exc_links = [links.clone().zero_() for links in model.exc_links]
        model.inh_links = [links.clone().zero_() for links in model.inh_links]

        model.train_weight_batch(torch.ones(8, 2352), torch.full((8,), 3, dtype=torch.long))

        self.assertTrue(torch.allclose(model.exc_weights.sum(dim=1), torch.ones(10), atol=1e-5))
        self.assertTrue(torch.allclose(model.inh_weights.sum(dim=1), torch.ones(10), atol=1e-5))

    def test_final_score_remains_excitation_minus_inhibition(self):
        model = MixedLinkSpatialAttention(num_classes=1, device="cpu")
        model.inh_links = [links.clone().zero_() for links in model.inh_links]

        result = model.forward(torch.ones(1, 2352))

        self.assertEqual(float(result["score"][0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
