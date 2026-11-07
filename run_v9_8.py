import argparse
import json
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import full_mnist_data
from fixed_retina import sobel_retina
from metrics import confusion, recalls
from v9_8_network import MixedLinkSpatialAttention


OUTPUT_DIR = Path(__file__).resolve().parent / "results"


@torch.no_grad()
def evaluate(model, x, y):
    results = [model.forward(x[start : start + 1024]) for start in range(0, len(x), 1024)]
    exc = torch.cat([result["exc_output"] for result in results])
    inh = torch.cat([result["inh_output"] for result in results])
    predictions = torch.cat([result["predictions"] for result in results])
    labels = y.to(model.device).long()
    classes = torch.arange(model.num_classes, device=model.device)
    exc_target = labels.unsqueeze(1) == classes.unsqueeze(0)
    inh_target = ~exc_target
    exc_isolated = (((exc >= 0.5) == exc_target).float().mean(dim=0)).cpu().tolist()
    inh_isolated = (((inh >= 0.5) == inh_target).float().mean(dim=0)).cpu().tolist()
    return {
        "accuracy": float((predictions == labels).float().mean()),
        "exc_isolated_accuracy": exc_isolated,
        "inh_isolated_accuracy": inh_isolated,
        "per_class_recall": recalls(predictions, labels),
        "confusion_matrix": confusion(predictions, labels),
    }


def run_epoch(model, x, y, method):
    for start in range(0, len(x), 128):
        getattr(model, method)(x[start : start + 128], y[start : start + 128])


def main():
    global OUTPUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR = Path(__file__).resolve().parent / ("results_strict" if args.strict else "results_mixed")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_x, train_y, _, _, test_x, test_y = full_mnist_data()
    model = MixedLinkSpatialAttention(seed=42, link_min=0.0 if args.strict else -1.0, link_max=1.0)
    train_x = sobel_retina(train_x, device=model.device)
    test_x = sobel_retina(test_x, device=model.device)
    phase_history = {1: [], 2: [], 3: []}
    abort_reason = None
    print(f"Mixed-Link Spatial Attention, strict={args.strict}, device={model.device}", flush=True)

    phase1_epochs = 0
    for epoch in range(1, 16):
        run_epoch(model, train_x, train_y, "train_isolated_batch")
        phase1_epochs = epoch
        metrics = evaluate(model, test_x, test_y)
        phase_history[1].append({"epoch": epoch, "test_accuracy": metrics["accuracy"]})
        print(f"PHASE1 epoch={epoch} test={metrics['accuracy']:.4f}", flush=True)
        if epoch == 10:
            isolated_min = min(min(metrics["exc_isolated_accuracy"]), min(metrics["inh_isolated_accuracy"]))
            if isolated_min <= 0.95:
                print("Phase 1 gate below 95%; running 5 additional epochs", flush=True)
            else:
                break
    phase1 = evaluate(model, test_x, test_y)
    phase1["epochs"] = phase1_epochs
    phase1["history"] = phase_history[1]
    phase1["model"] = model.state_dict()
    (OUTPUT_DIR / "v9_8_phase1.json").write_text(json.dumps(phase1, indent=2), encoding="utf-8")
    if min(min(phase1["exc_isolated_accuracy"]), min(phase1["inh_isolated_accuracy"])) < 0.90:
        abort_reason = "Phase 1 isolated accuracy below 90% after 15 epochs"

    phase2 = None
    phase2_state = None
    if abort_reason is None:
        model.eta_w = 0.01
        for epoch in range(11, 16):
            run_epoch(model, train_x, train_y, "train_weight_batch")
            metrics = evaluate(model, test_x, test_y)
            phase_history[2].append({"epoch": epoch, "test_accuracy": metrics["accuracy"]})
            print(f"PHASE2 epoch={epoch} test={metrics['accuracy']:.4f}", flush=True)
        phase2 = evaluate(model, test_x, test_y)
        phase2["gate_93_06"] = phase2["accuracy"] > 0.9306
        phase2["history"] = phase_history[2]
        phase2["exc_cluster_weights"] = model.exc_weights.cpu().tolist()
        phase2["inh_cluster_weights"] = model.inh_weights.cpu().tolist()
        phase2["model"] = model.state_dict()
        phase2_state = model.state_dict()
        (OUTPUT_DIR / "v9_8_phase2.json").write_text(json.dumps(phase2, indent=2), encoding="utf-8")
        if phase2["accuracy"] < 0.92:
            abort_reason = "Phase 2 accuracy below 92%"

    phase3 = None
    if abort_reason is None:
        model.eta = 0.0001
        model.eta_w = 0.0001
        for epoch in range(16, 21):
            run_epoch(model, train_x, train_y, "train_coadapt_batch")
            metrics = evaluate(model, test_x, test_y)
            phase_history[3].append({"epoch": epoch, "test_accuracy": metrics["accuracy"]})
            print(f"PHASE3 epoch={epoch} test={metrics['accuracy']:.4f}", flush=True)
            if metrics["accuracy"] < 0.93:
                abort_reason = "Phase 3 accuracy below 93%; restored Phase 2 checkpoint"
                model.load_state_dict(phase2_state)
                break
        phase3 = evaluate(model, test_x, test_y)
        phase3["history"] = phase_history[3]
        phase3["model"] = model.state_dict()
        (OUTPUT_DIR / "v9_8_phase3.json").write_text(json.dumps(phase3, indent=2), encoding="utf-8")

    results = {
        "experiment": "V9.8 fixed clean pipeline",
        "architecture": "Mixed-Link Spatial Attention with Path-Level Push-Pull Routing",
        "configuration": "strict" if args.strict else "mixed",
        "link_bounds": [model.link_min, model.link_max],
        "phase1": {key: value for key, value in phase1.items() if key not in {"history", "model"}},
        "phase2": None if phase2 is None else {key: value for key, value in phase2.items() if key not in {"history", "model"}},
        "phase3": None if phase3 is None else {key: value for key, value in phase3.items() if key not in {"history", "model"}},
        "abort_reason": abort_reason,
        "seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "v9_8_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(10, 6))
    for phase, label in ((1, "Isolation"), (2, "Weights"), (3, "Co-adapt")):
        entries = phase_history[phase]
        if entries:
            axis.plot([entry["epoch"] for entry in entries], [entry["test_accuracy"] for entry in entries], label=label)
    axis.axhline(0.94, color="black", linestyle="--", label="94% target")
    axis.set(xlabel="Epoch", ylabel="Accuracy", title="V9.8 Fixed Three-Phase Training")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v9_8_fixed.png", dpi=160)
    plt.close(fig)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
