import hashlib
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
from run_v9_long import coadapt_eta
from v9_23_relu import FixedReluGuide, FixedReluSpatialAttention


OUTPUT_DIR = Path(__file__).resolve().parent / "results_v9_23"


def guide_hash(guide):
    return hashlib.sha256(guide.weights.cpu().numpy().tobytes()).hexdigest()


def model_is_finite(model):
    for values in model.exc_links + model.inh_links + model.exc_thresholds + model.inh_thresholds + [model.exc_weights, model.inh_weights]:
        if not bool(torch.isfinite(values).all()):
            return False
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_x, train_y, _, _, test_x, test_y = full_mnist_data()
    guide = FixedReluGuide(seed=42)
    initial_hash = guide_hash(guide)
    train_x = guide(sobel_retina(train_x, device=guide.device))
    test_x = guide(sobel_retina(test_x, device=guide.device))
    model = FixedReluSpatialAttention(seed=42, link_min=-1.0, link_max=1.0)
    history = {1: [], 2: [], 3: []}
    phases = {}
    abort_reason = None
    initial = evaluate(model, test_x, test_y)
    best_accuracy = initial["accuracy"]
    best_epoch = 0
    best_state = model.state_dict()
    last_improvement = 0
    print(f"V9.23 fixed ReLU expansion, initial={best_accuracy:.4f}, device={model.device}", flush=True)
    for epoch in range(1, 46):
        if epoch <= 20:
            phase = 1
            phase_epoch = epoch
            method = "train_isolated_batch"
        elif epoch <= 25:
            phase = 2
            phase_epoch = epoch - 20
            method = "train_weight_batch"
            model.eta_w = 0.01
        else:
            phase = 3
            phase_epoch = epoch - 25
            method = "train_coadapt_batch"
            model.eta = coadapt_eta(phase_epoch)
            model.eta_w = model.eta
        run_epoch(model, train_x, train_y, method)
        metrics = evaluate(model, test_x, test_y)
        history[phase].append({"epoch": phase_epoch, "accuracy": metrics["accuracy"]})
        print(f"epoch={epoch} phase={phase}.{phase_epoch} acc={metrics['accuracy']:.4f} best={best_accuracy:.4f}", flush=True)
        if not model_is_finite(model):
            abort_reason = "NaN/Inf detected; restored best checkpoint"
            break
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_epoch = epoch
            best_state = model.state_dict()
            last_improvement = epoch
            (OUTPUT_DIR / "best.json").write_text(json.dumps({"epoch": epoch, "accuracy": best_accuracy, "model": best_state}, indent=2), encoding="utf-8")
        if metrics["accuracy"] < 0.80:
            abort_reason = "Accuracy below 80%; restored best checkpoint"
            break
        if epoch - last_improvement >= 10:
            abort_reason = "No improvement for 10 epochs; restored best checkpoint"
            break
        phases[f"phase{phase}"] = metrics

    model.load_state_dict(best_state)
    final = evaluate(model, test_x, test_y)
    results = {
        "experiment": "V9.23 fixed random ReLU expansion",
        "final_accuracy": final["accuracy"],
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "target_95_32": best_accuracy > 0.9532,
        "phase1_exc_isolated": phases.get("phase1", {}).get("exc_isolated_accuracy"),
        "phase1_inh_isolated": phases.get("phase1", {}).get("inh_isolated_accuracy"),
        "phase2_accuracy": phases.get("phase2", {}).get("accuracy"),
        "confusion_matrix": final["confusion_matrix"],
        "true_8_predicted_0": final["confusion_matrix"][8][0],
        "true_9_predicted_4": final["confusion_matrix"][9][4],
        "guide_sha256_before": initial_hash,
        "guide_sha256_after": guide_hash(guide),
        "guide_unchanged": guide_hash(guide) == initial_hash,
        "phases": phases,
        "history": history,
        "abort_reason": abort_reason,
        "epochs_completed": sum(len(entries) for entries in history.values()),
        "seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(11, 6))
    for phase in [1, 2, 3]:
        entries = history[phase]
        if entries:
            axis.plot([entry["epoch"] for entry in entries], [entry["accuracy"] for entry in entries], label=f"Phase {phase}")
    axis.axhline(0.9532, color="black", linestyle="--", label="95.32% target")
    axis.set(xlabel="Epoch within phase", ylabel="Accuracy", title="V9.23 Fixed Random ReLU Expansion")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v9_23_relu.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in results.items() if key not in {"phases", "history"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
