import torch

from v9_8_network import MixedLinkSpatialAttention


def pruned_coadapt_eta(epoch):
    return 0.0001 * (0.5 ** max(0, epoch - 5))


class PrunedSpatialAttention(MixedLinkSpatialAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_links = [torch.zeros_like(links, dtype=torch.bool) for links in self.exc_links + self.inh_links]

    def enforce_frozen_links(self):
        for links, frozen in zip(self.exc_links + self.inh_links, self.frozen_links):
            links.masked_fill_(frozen, 0.0)

    @torch.no_grad()
    def train_isolated_batch(self, x, y):
        super().train_isolated_batch(x, y)
        self.enforce_frozen_links()

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


@torch.no_grad()
def prune_by_utilization(model, train_x, threshold=0.01):
    fields = model.receptive_fields(train_x)
    pruned = 0
    for kind_index, links_all in enumerate((model.exc_links, model.inh_links)):
        for cluster, links in enumerate(links_all):
            field_std = torch.zeros(10, fields[cluster].shape[1], device=model.device)
            for start in range(0, len(train_x), 512):
                field_std += fields[cluster][start : start + 512].std(dim=0, unbiased=False)
            field_std /= 10.0
            utilization = links.abs() * field_std.unsqueeze(1)
            frozen = model.frozen_links[kind_index * 5 + cluster]
            frozen.copy_(utilization < threshold)
            links.masked_fill_(frozen, 0.0)
            pruned += int(frozen.sum())
    return pruned
