import json
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import full_mnist_data
from fixed_retina import sobel_retina
from run_v9_8 import evaluate, run_epoch
from v9_pruning import PrunedSpatialAttention, prune_by_utilization, pruned_coadapt_eta


OUTPUT_DIR = Path(__file__).resolve().parent / "results_long"


def main():
    started = time.perf_counter()
    checkpoint = json.loads((OUTPUT_DIR / "best.json").read_text(encoding="utf-8"))
    model = PrunedSpatialAttention(seed=42, link_min=-1.0, link_max=1.0)
    model.load_state_dict(checkpoint["model"])
    original_state = model.state_dict()
    train_x, train_y, _, _, test_x, test_y = full_mnist_data()
    train_x = sobel_retina(train_x, device=model.device)
    test_x = sobel_retina(test_x, device=model.device)
    pruned = prune_by_utilization(model, train_x)
    total_links = sum(links.numel() for links in model.exc_links + model.inh_links)
    history = {1: [], 2: [], 3: []}
    print(f"Pruned {pruned}/{total_links} links; starting 5+5+10 retraining", flush=True)
    for epoch in range(1, 21):
        if epoch <= 5:
            phase = 1
            phase_epoch = epoch
            method = "train_isolated_batch"
        elif epoch <= 10:
            phase = 2
            phase_epoch = epoch - 5
            method = "train_weight_batch"
            model.eta_w = 0.01
        else:
            phase = 3
            phase_epoch = epoch - 10
            method = "train_coadapt_batch"
            model.eta = pruned_coadapt_eta(phase_epoch)
            model.eta_w = model.eta
        run_epoch(model, train_x, train_y, method)
        metrics = evaluate(model, test_x, test_y)
        history[phase].append({"epoch": phase_epoch, "accuracy": metrics["accuracy"], "eta": model.eta})
        print(f"phase={phase}.{phase_epoch} acc={metrics['accuracy']:.4f} eta={model.eta:.8f}", flush=True)

    pruned_metrics = evaluate(model, test_x, test_y)
    pruned_accuracy = pruned_metrics["accuracy"]
    dead_nonzero = 0
    for links, frozen in zip(model.exc_links + model.inh_links, model.frozen_links):
        dead_nonzero += int((links[frozen] != 0.0).sum())
    success = pruned_accuracy >= 0.945
    checkpoint_size = None
    if success:
        payload = {"accuracy": pruned_accuracy, "pruned_links": pruned, "model": model.state_dict()}
        (OUTPUT_DIR / "v9_pruned.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        checkpoint_size = (OUTPUT_DIR / "v9_pruned.json").stat().st_size
    else:
        model.load_state_dict(original_state)
    final = evaluate(model, test_x, test_y)
    results = {
        "experiment": "V9-LONG link pruning and retraining",
        "pruned_links": pruned,
        "pruned_percent": 100.0 * pruned / total_links,
        "pruned_accuracy": pruned_accuracy,
        "final_restored_accuracy": final["accuracy"],
        "pruning_succeeded": success,
        "dead_links_nonzero_after_training": dead_nonzero,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_size_mb": None if checkpoint_size is None else checkpoint_size / (1024 * 1024),
        "confusion_matrix": pruned_metrics["confusion_matrix"],
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "pruning_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(10, 6))
    for phase in [1, 2, 3]:
        entries = history[phase]
        axis.plot([entry["epoch"] for entry in entries], [entry["accuracy"] for entry in entries], label=f"Phase {phase}")
    axis.axhline(0.945, color="red", linestyle="--", label="94.5% gate")
    axis.set(xlabel="Epoch within phase", ylabel="Accuracy", title="V9-LONG Link Pruning")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v9_pruned.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in results.items() if key != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
