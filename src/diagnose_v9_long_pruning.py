import json
from pathlib import Path

import torch

from data import full_mnist_data
from fixed_retina import sobel_retina
from v9_8_network import MixedLinkSpatialAttention


OUTPUT_DIR = Path(__file__).resolve().parent / "results_long"
NAMES = ["TopLeft", "TopRight", "BottomLeft", "BottomRight", "Global"]


@torch.no_grad()
def main():
    checkpoint = json.loads((OUTPUT_DIR / "best.json").read_text(encoding="utf-8"))
    model = MixedLinkSpatialAttention(seed=42, link_min=-1.0, link_max=1.0)
    model.load_state_dict(checkpoint["model"])
    train_x, _, _, _, _, _ = full_mnist_data()
    train_x = sobel_retina(train_x, device=model.device)
    fields = model.receptive_fields(train_x)
    fire_rates = []
    link_records = []
    total_links = 0
    dead_links = 0
    absolute_sum = 0.0
    for kind, links_all, thresholds_all in (("Exc", model.exc_links, model.exc_thresholds), ("Inh", model.inh_links, model.inh_thresholds)):
        for cluster in range(5):
            links = links_all[cluster]
            thresholds = thresholds_all[cluster]
            fire = torch.zeros(10, 16, device=model.device)
            field_std = torch.zeros(10, fields[cluster].shape[1], device=model.device)
            for start in range(0, len(train_x), 512):
                field = fields[cluster][start : start + 512]
                spikes = torch.einsum("bi,cni->bcn", field, links) >= thresholds.unsqueeze(0)
                fire += spikes.sum(dim=0)
                field_std += field.std(dim=0, unbiased=False)
            field_std /= 10.0
            rates = fire / len(train_x)
            utilization = links.abs() * field_std.unsqueeze(1)
            dead = utilization < 0.01
            total = links.numel()
            total_links += total
            dead_links += int(dead.sum())
            absolute_sum += float(links.abs().sum())
            fire_rates.append({"kind": kind, "cluster": cluster, "cluster_name": NAMES[cluster], "rates": rates.cpu().tolist()})
            link_records.append({
                "kind": kind,
                "cluster": cluster,
                "cluster_name": NAMES[cluster],
                "total_links": total,
                "dead_links": int(dead.sum()),
                "dead_percent": float(dead.float().mean() * 100),
                "mean_abs_link": float(links.abs().mean()),
            })
    all_rates = torch.cat([torch.tensor(record["rates"]).reshape(-1) for record in fire_rates])
    saturated = int((all_rates > 0.90).sum())
    dead_neurons = int((all_rates < 0.10).sum())
    healthy = int(((all_rates >= 0.10) & (all_rates <= 0.90)).sum())
    results = {
        "checkpoint": "results_long/best.json",
        "training_images": len(train_x),
        "total_neurons": len(all_rates),
        "saturated_neurons": saturated,
        "dead_neurons": dead_neurons,
        "healthy_neurons": healthy,
        "total_links": total_links,
        "dead_links": dead_links,
        "dead_link_percent": 100.0 * dead_links / total_links,
        "average_absolute_link": absolute_sum / total_links,
        "fire_rates": fire_rates,
        "link_clusters": link_records,
    }
    (OUTPUT_DIR / "pruning_diagnostic.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in results.items() if key not in {"fire_rates", "link_clusters"}}, indent=2))
    print("Per-cluster dead links:")
    for record in link_records:
        print(f"{record['kind']} {record['cluster_name']}: {record['dead_links']}/{record['total_links']} ({record['dead_percent']:.2f}%)")


if __name__ == "__main__":
    main()
