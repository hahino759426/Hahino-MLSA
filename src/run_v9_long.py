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
from v9_8_network import MixedLinkSpatialAttention


OUTPUT_DIR = Path(__file__).resolve().parent / "results_long"


def coadapt_eta(epoch):
    if epoch < 16:
        return 0.0001
    return 0.0001 * (0.5 ** (epoch - 15))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_x, train_y, _, _, test_x, test_y = full_mnist_data()
    model = MixedLinkSpatialAttention(seed=42, link_min=-1.0, link_max=1.0)
    train_x = sobel_retina(train_x, device=model.device)
    test_x = sobel_retina(test_x, device=model.device)
    history = {1: [], 2: [], 3: []}
    phases = {}
    abort_reason = None
    best_accuracy = -1.0
    best_epoch = None
    print(f"V9-LONG mixed-link stability, device={model.device}", flush=True)

    phase1_epochs = 0
    for epoch in range(1, 21):
        run_epoch(model, train_x, train_y, "train_isolated_batch")
        phase1_epochs = epoch
        metrics = evaluate(model, test_x, test_y)
        history[1].append({"epoch": epoch, "accuracy": metrics["accuracy"]})
        isolated_min = min(min(metrics["exc_isolated_accuracy"]), min(metrics["inh_isolated_accuracy"]))
        print(f"PHASE1 epoch={epoch} acc={metrics['accuracy']:.4f} isolated_min={isolated_min:.4f}", flush=True)
        if isolated_min > 0.99:
            print("All paths exceeded 99%; stopping Phase 1 early", flush=True)
            break
    phases["phase1"] = evaluate(model, test_x, test_y)
    phases["phase1"]["epochs"] = phase1_epochs
    phases["phase1"]["history"] = history[1]
    phases["phase1"]["model"] = model.state_dict()
    (OUTPUT_DIR / "phase1.json").write_text(json.dumps(phases["phase1"], indent=2), encoding="utf-8")
    isolated_min = min(min(phases["phase1"]["exc_isolated_accuracy"]), min(phases["phase1"]["inh_isolated_accuracy"]))
    if isolated_min < 0.90:
        abort_reason = "Phase 1 isolated accuracy below 90%"

    phase2_state = None
    phase2_accuracy = None
    if abort_reason is None:
        model.eta_w = 0.01
        for epoch in range(1, 6):
            run_epoch(model, train_x, train_y, "train_weight_batch")
            metrics = evaluate(model, test_x, test_y)
            history[2].append({"epoch": epoch, "accuracy": metrics["accuracy"]})
            print(f"PHASE2 epoch={epoch} acc={metrics['accuracy']:.4f}", flush=True)
        phases["phase2"] = evaluate(model, test_x, test_y)
        phases["phase2"]["history"] = history[2]
        phases["phase2"]["model"] = model.state_dict()
        phase2_state = model.state_dict()
        phase2_accuracy = phases["phase2"]["accuracy"]
        (OUTPUT_DIR / "phase2.json").write_text(json.dumps(phases["phase2"], indent=2), encoding="utf-8")

    if abort_reason is None:
        for epoch in range(1, 21):
            model.eta = coadapt_eta(epoch)
            model.eta_w = model.eta
            run_epoch(model, train_x, train_y, "train_coadapt_batch")
            metrics = evaluate(model, test_x, test_y)
            history[3].append({"epoch": epoch, "accuracy": metrics["accuracy"], "eta": model.eta})
            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]
                best_epoch = epoch
                (OUTPUT_DIR / "best.json").write_text(json.dumps({"phase": 3, "epoch": epoch, "accuracy": best_accuracy, "model": model.state_dict()}, indent=2), encoding="utf-8")
            print(f"PHASE3 epoch={epoch} acc={metrics['accuracy']:.4f} eta={model.eta:.8f}", flush=True)
            if metrics["accuracy"] < 0.94 or metrics["accuracy"] < phase2_accuracy - 0.01:
                abort_reason = "Phase 3 rollback: below 94% or more than 1 point below Phase 2"
                model.load_state_dict(phase2_state)
                break
        phases["phase3"] = evaluate(model, test_x, test_y)
        phases["phase3"]["history"] = history[3]
        phases["phase3"]["model"] = model.state_dict()
        (OUTPUT_DIR / "phase3.json").write_text(json.dumps(phases["phase3"], indent=2), encoding="utf-8")

    final_phase = "phase3" if "phase3" in phases else ("phase2" if "phase2" in phases else "phase1")
    final_accuracy = phases[final_phase]["accuracy"]
    last3 = [entry["accuracy"] for entry in history[3][-3:]]
    plateau_variance = float(torch.tensor(last3).var(unbiased=False)) if len(last3) == 3 else None
    checkpoints = {}
    for epoch in [5, 10, 15, 20]:
        matches = [entry for entry in history[3] if entry["epoch"] == epoch]
        checkpoints[str(epoch)] = matches[0]["accuracy"] if matches else None
    results = {
        "experiment": "V9-LONG mixed-link stability and convergence",
        "architecture": "Mixed-Link Spatial Attention with Path-Level Push-Pull Routing",
        "link_bounds": [-1.0, 1.0],
        "phase1_isolated_exc": phases["phase1"]["exc_isolated_accuracy"],
        "phase1_isolated_inh": phases["phase1"]["inh_isolated_accuracy"],
        "phase1_epochs": phase1_epochs,
        "phase2_accuracy": phase2_accuracy,
        "phase3_checkpoints": checkpoints,
        "final_accuracy": final_accuracy,
        "target_94_45": final_accuracy >= 0.9445,
        "best_phase3_accuracy": None if best_epoch is None else best_accuracy,
        "best_phase3_epoch": best_epoch,
        "last_3_variance": plateau_variance,
        "plateaued": plateau_variance is not None and plateau_variance < 0.001,
        "confusion_matrix": phases[final_phase]["confusion_matrix"],
        "phases": phases,
        "history": history,
        "abort_reason": abort_reason,
        "seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(11, 6))
    for phase in [1, 2, 3]:
        if history[phase]:
            axis.plot([entry["epoch"] for entry in history[phase]], [entry["accuracy"] for entry in history[phase]], label=f"Phase {phase}")
    axis.axhline(0.9445, color="black", linestyle="--", label="94.45% target")
    axis.set(xlabel="Epoch within phase", ylabel="Accuracy", title="V9-LONG Mixed-Link Stability")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v9_long.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in results.items() if key not in {"phases", "history"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
