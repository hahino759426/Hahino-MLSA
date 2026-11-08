import torch

from v9_8_network import MixedLinkSpatialAttention


def diagonal_receptive_fields(x):
    images = x.reshape(-1, 5, 28, 28)
    return [
        images[:, :, :16, :16].reshape(-1, 1280),
        images[:, :, :16, 12:].reshape(-1, 1280),
        images[:, :, 12:, :16].reshape(-1, 1280),
        images[:, :, 12:, 12:].reshape(-1, 1280),
        x,
    ]


class DiagonalSpatialAttention(MixedLinkSpatialAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sizes = [1280, 1280, 1280, 1280, 3920]
        self.exc_links = [torch.ones(self.num_classes, 16, size, device=self.device) for size in sizes]
        self.inh_links = [torch.ones(self.num_classes, 16, size, device=self.device) for size in sizes]

    def receptive_fields(self, x):
        return diagonal_receptive_fields(x)


class PrunedDiagonalSpatialAttention(DiagonalSpatialAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_links = [torch.zeros_like(links, dtype=torch.bool) for links in self.exc_links + self.inh_links]

    def enforce_frozen_links(self):
        for links, frozen in zip(self.exc_links + self.inh_links, self.frozen_links):
            links.masked_fill_(frozen, 0.0)

    @torch.no_grad()
    def train_coadapt_batch(self, x, y):
        super().train_coadapt_batch(x, y)
        self.enforce_frozen_links()

    def state_dict(self):
        state = super().state_dict()
        state["frozen_links"] = [mask.cpu().tolist() for mask in self.frozen_links]
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        if "frozen_links" in state:
            for target, saved in zip(self.frozen_links, state["frozen_links"]):
                target.copy_(torch.as_tensor(saved, dtype=torch.bool, device=self.device))
