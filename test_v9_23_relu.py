import unittest

import torch

from v9_23_relu import FixedReluGuide, FixedReluSpatialAttention


class V923ReluTests(unittest.TestCase):
    def test_guide_is_fixed_seeded_relu_projection(self):
        guide = FixedReluGuide(seed=42, device="cpu")
        x = torch.randn(4, 2352)

        first = guide(x)
        second = guide(x)

        self.assertEqual(tuple(guide.weights.shape), (2352, 2000))
        self.assertEqual(tuple(first.shape), (4, 2000))
        self.assertTrue(torch.equal(first, second))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertFalse(guide.weights.requires_grad)

    def test_all_clusters_receive_2000_expanded_features(self):
        model = FixedReluSpatialAttention(num_classes=10, device="cpu")

        for links in model.exc_links + model.inh_links:
            self.assertEqual(tuple(links.shape), (10, 16, 2000))
        fields = model.receptive_fields(torch.ones(3, 2000))
        self.assertEqual(len(fields), 5)
        self.assertTrue(all(field is fields[0] for field in fields))

    def test_training_does_not_modify_guide(self):
        guide = FixedReluGuide(seed=42, device="cpu")
        model = FixedReluSpatialAttention(num_classes=1, device="cpu")
        before = guide.weights.clone()
        x = guide(torch.ones(1, 2352))

        model.train_isolated_batch(x, torch.tensor([0]))

        self.assertTrue(torch.equal(guide.weights, before))


if __name__ == "__main__":
    unittest.main()
