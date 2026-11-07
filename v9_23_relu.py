import torch

from v9_8_network import MixedLinkSpatialAttention


class FixedReluGuide:
    def __init__(self, input_dim=2352, output_dim=2000, std=0.05, seed=42, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        generator = torch.Generator().manual_seed(seed)
        self.weights = (torch.randn(input_dim, output_dim, generator=generator) * std).to(self.device)
        self.weights.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x, batch_size=512):
        outputs = []
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size].to(self.device).float()
            outputs.append(torch.relu(batch @ self.weights))
        return torch.cat(outputs)


class FixedReluSpatialAttention(MixedLinkSpatialAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exc_links = [torch.ones(self.num_classes, 16, 2000, device=self.device) for _ in range(5)]
        self.inh_links = [torch.ones(self.num_classes, 16, 2000, device=self.device) for _ in range(5)]

    def receptive_fields(self, x):
        return [x] * 5
