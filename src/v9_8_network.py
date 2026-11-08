import torch


def receptive_fields(x):
    images = x.reshape(-1, 3, 28, 28)
    return [
        images[:, :, :16, :16].reshape(-1, 768),
        images[:, :, :16, 12:].reshape(-1, 768),
        images[:, :, 12:, :16].reshape(-1, 768),
        images[:, :, 12:, 12:].reshape(-1, 768),
        x,
    ]


class MixedLinkSpatialAttention:
    def __init__(self, num_classes=10, eta=0.01, eta_w=0.001, link_min=-1.0, link_max=1.0, seed=42, device=None):
        torch.manual_seed(seed)
        self.num_classes = num_classes
        self.eta = eta
        self.eta_w = eta_w
        self.link_min = link_min
        self.link_max = link_max
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.exc_links = [torch.ones(num_classes, 16, size, device=self.device) for size in [768, 768, 768, 768, 2352]]
        self.inh_links = [torch.ones(num_classes, 16, size, device=self.device) for size in [768, 768, 768, 768, 2352]]
        self.exc_thresholds = [torch.full((num_classes, 16), 0.5, device=self.device) for _ in range(5)]
        self.inh_thresholds = [torch.full((num_classes, 16), 0.5, device=self.device) for _ in range(5)]
        self.exc_weights = torch.full((num_classes, 5), 0.2, device=self.device)
        self.inh_weights = torch.full((num_classes, 5), 0.2, device=self.device)

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device).float()
        fields = self.receptive_fields(x)
        activations = {}
        outputs = {}
        for kind, links, thresholds, weights in (
            ("exc", self.exc_links, self.exc_thresholds, self.exc_weights),
            ("inh", self.inh_links, self.inh_thresholds, self.inh_weights),
        ):
            cluster_activations = []
            for field, link, threshold in zip(fields, links, thresholds):
                spikes = torch.einsum("bi,cni->bcn", field, link) >= threshold.unsqueeze(0)
                cluster_activations.append(spikes.float().mean(dim=2))
            activation = torch.stack(cluster_activations, dim=2)
            activations[kind] = activation
            outputs[kind] = (activation * weights.unsqueeze(0)).sum(dim=2)
        score = outputs["exc"] - outputs["inh"]
        predictions = (score <= 0.0).long().squeeze(1) if self.num_classes == 1 else score.argmax(dim=1)
        return {
            "exc_activation": activations["exc"],
            "inh_activation": activations["inh"],
            "exc_output": outputs["exc"],
            "inh_output": outputs["inh"],
            "score": score,
            "predictions": predictions,
        }

    def receptive_fields(self, x):
        return receptive_fields(x)

    def targets(self, y):
        if self.num_classes == 1:
            exc_target = (y == 0).unsqueeze(1)
        else:
            classes = torch.arange(self.num_classes, device=self.device)
            exc_target = y.unsqueeze(1) == classes.unsqueeze(0)
        return exc_target, ~exc_target

    @torch.no_grad()
    def train_isolated_batch(self, x, y):
        x = x.to(self.device).float()
        y = y.to(self.device).long()
        fields = self.receptive_fields(x)
        result = self.forward(x)
        exc_target, inh_target = self.targets(y)
        for target, activation, links, thresholds in (
            (exc_target, result["exc_activation"], self.exc_links, self.exc_thresholds),
            (inh_target, result["inh_activation"], self.inh_links, self.inh_thresholds),
        ):
            for cluster in range(5):
                error = target.float() - activation[:, :, cluster]
                links[cluster] += self.eta * torch.einsum("bc,bi->ci", error, fields[cluster]).unsqueeze(1)
                links[cluster].clamp_(self.link_min, self.link_max)
                thresholds[cluster] -= self.eta * error.sum(dim=0).unsqueeze(1)

    def margin_coefficients(self, result, y, target, output):
        batch = torch.arange(len(y), device=self.device)
        predicted = result["predictions"]
        if self.num_classes == 1:
            margin = torch.zeros(len(y), device=self.device)
        else:
            margin = result["score"][batch, y] - result["score"][batch, predicted]
        active = (margin < 0.5).float()
        error = target.float() - output
        coefficient = torch.zeros(1, 1, device=self.device) if self.num_classes == 1 else torch.zeros(len(y), self.num_classes, device=self.device)
        if self.num_classes == 1:
            coefficient[0, 0] = (error[:, 0] * active).sum()
        else:
            coefficient[batch, y] += error[batch, y] * active
            coefficient[batch, predicted] -= error[batch, predicted] * active
        return coefficient

    def update_weights(self, coefficient, activation, weights):
        weights += self.eta_w * torch.einsum("bc,bcq->cq", coefficient, activation)
        updated = coefficient.abs().sum(dim=0) > 0.0
        if bool(updated.any()):
            weights[updated] /= weights[updated].sum(dim=1, keepdim=True) + 1e-6
            weights[updated] = weights[updated].clamp(0.1, 0.6)
            weights[updated] /= weights[updated].sum(dim=1, keepdim=True) + 1e-6

    @torch.no_grad()
    def train_weight_batch(self, x, y):
        x = x.to(self.device).float()
        y = y.to(self.device).long()
        result = self.forward(x)
        exc_target, inh_target = self.targets(y)
        for target, output, activation, weights in (
            (exc_target, result["exc_output"], result["exc_activation"], self.exc_weights),
            (inh_target, result["inh_output"], result["inh_activation"], self.inh_weights),
        ):
            coefficient = self.margin_coefficients(result, y, target, output)
            self.update_weights(coefficient, activation, weights)

    @torch.no_grad()
    def train_coadapt_batch(self, x, y):
        x = x.to(self.device).float()
        y = y.to(self.device).long()
        fields = self.receptive_fields(x)
        result = self.forward(x)
        exc_target, inh_target = self.targets(y)
        for target, output, activation, weights, links, thresholds in (
            (exc_target, result["exc_output"], result["exc_activation"], self.exc_weights, self.exc_links, self.exc_thresholds),
            (inh_target, result["inh_output"], result["inh_activation"], self.inh_weights, self.inh_links, self.inh_thresholds),
        ):
            coefficient = self.margin_coefficients(result, y, target, output)
            self.update_weights(coefficient, activation, weights)
            selected = (coefficient != 0.0).float()
            for cluster in range(5):
                error = (target.float() - activation[:, :, cluster]) * selected
                links[cluster] += self.eta * torch.einsum("bc,bi->ci", error, fields[cluster]).unsqueeze(1)
                links[cluster].clamp_(self.link_min, self.link_max)
                thresholds[cluster] -= self.eta * error.sum(dim=0).unsqueeze(1)

    def state_dict(self):
        return {
            "exc_links": [links.cpu().tolist() for links in self.exc_links],
            "inh_links": [links.cpu().tolist() for links in self.inh_links],
            "exc_thresholds": [thresholds.cpu().tolist() for thresholds in self.exc_thresholds],
            "inh_thresholds": [thresholds.cpu().tolist() for thresholds in self.inh_thresholds],
            "exc_weights": self.exc_weights.cpu().tolist(),
            "inh_weights": self.inh_weights.cpu().tolist(),
        }

    def load_state_dict(self, state):
        for name in ["exc_links", "inh_links", "exc_thresholds", "inh_thresholds"]:
            target = getattr(self, name)
            for index, value in enumerate(state[name]):
                target[index].copy_(torch.as_tensor(value, device=self.device))
        self.exc_weights.copy_(torch.as_tensor(state["exc_weights"], device=self.device))
        self.inh_weights.copy_(torch.as_tensor(state["inh_weights"], device=self.device))
