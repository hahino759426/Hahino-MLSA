import json
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import full_mnist_data
from fixed_retina_diagonal import diagonal_sobel_retina
from run_v9_8 import evaluate, run_epoch
from v9_diagonal import PrunedDiagonalSpatialAttention
from v9_pruning import prune_by_utilization, pruned_coadapt_eta


OUTPUT_DIR = Path(__file__).resolve().parent / "results_diagonal_pruned"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checkpoint = json.loads(Path("results_diagonal/v9_diagonal.json").read_text(encoding="utf-8"))
    model = PrunedDiagonalSpatialAttention(seed=42, link_min=-1.0, link_max=1.0)
    model.load_state_dict(checkpoint["model"])
    train_x, train_y, _, _, test_x, test_y = full_mnist_data()
    train_x = diagonal_sobel_retina(train_x, device=model.device)
    test_x = diagonal_sobel_retina(test_x, device=model.device)
    pruned = prune_by_utilization(model, train_x)
    total_links = sum(links.numel() for links in model.exc_links + model.inh_links)
    history = []
    print(f"Pruned {pruned}/{total_links} diagonal links; starting 10-epoch co-adaptation", flush=True)
    for epoch in range(1, 11):
        model.eta = pruned_coadapt_eta(epoch)
        model.eta_w = model.eta
        run_epoch(model, train_x, train_y, "train_coadapt_batch")
        metrics = evaluate(model, test_x, test_y)
        history.append({"epoch": epoch, "accuracy": metrics["accuracy"], "eta": model.eta})
        print(f"epoch={epoch} acc={metrics['accuracy']:.4f} eta={model.eta:.8f}", flush=True)
    final = evaluate(model, test_x, test_y)
    accuracy = final["accuracy"]
    dead_nonzero = sum(int((links[frozen] != 0.0).sum()) for links, frozen in zip(model.exc_links + model.inh_links, model.frozen_links))
    success = accuracy > 0.962
    checkpoint_size = None
    if success:
        payload = {"accuracy": accuracy, "pruned_links": pruned, "model": model.state_dict()}
        (OUTPUT_DIR / "v9_diagonal_pruned.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        checkpoint_size = (OUTPUT_DIR / "v9_diagonal_pruned.json").stat().st_size
    results = {
        "experiment": "Pruned diagonal Sobel",
        "accuracy": accuracy,
        "target_96_2": success,
        "pruned_links": pruned,
        "pruned_percent": 100.0 * pruned / total_links,
        "total_links": total_links,
        "dead_links_nonzero_after_training": dead_nonzero,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_size_mb": None if checkpoint_size is None else checkpoint_size / (1024 * 1024),
        "confusion_matrix": final["confusion_matrix"],
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(10, 6))
    axis.plot([entry["epoch"] for entry in history], [entry["accuracy"] for entry in history])
    axis.axhline(0.962, color="red", linestyle="--", label="96.2% target")
    axis.set(xlabel="Co-adaptation epoch", ylabel="Accuracy", title="Pruned Diagonal Sobel")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v9_diagonal_pruned.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in results.items() if key != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
