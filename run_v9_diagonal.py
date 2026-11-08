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
from run_v9_long import coadapt_eta
from v9_8_network import MixedLinkSpatialAttention
from v9_diagonal import DiagonalSpatialAttention


OUTPUT_DIR = Path(__file__).resolve().parent / "results_diagonal"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train_x, train_y, _, _, test_x, test_y = full_mnist_data()
    raw_test_x = test_x
    train_x = diagonal_sobel_retina(train_x)
    test_x = diagonal_sobel_retina(test_x)
    model = DiagonalSpatialAttention(seed=42, link_min=-1.0, link_max=1.0)
    history = {1: [], 2: [], 3: []}
    phases = {}
    initial = evaluate(model, test_x, test_y)
    best_accuracy = initial["accuracy"]
    best_epoch = 0
    best_state = model.state_dict()
    print(f"Experiment 2 diagonal Sobel from scratch, device={model.device}", flush=True)
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
        phases[f"phase{phase}"] = metrics
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_epoch = epoch
            best_state = model.state_dict()
            (OUTPUT_DIR / "best.json").write_text(json.dumps({"epoch": epoch, "accuracy": best_accuracy, "model": best_state}, indent=2), encoding="utf-8")
        print(f"epoch={epoch} phase={phase}.{phase_epoch} acc={metrics['accuracy']:.4f} best={best_accuracy:.4f}", flush=True)

    model.load_state_dict(best_state)
    final = evaluate(model, test_x, test_y)
    diagonal_accuracy = final["accuracy"]
    reverted = diagonal_accuracy < 0.95
    restored_accuracy = diagonal_accuracy
    if reverted:
        original = MixedLinkSpatialAttention(seed=42, link_min=-1.0, link_max=1.0)
        original_checkpoint = json.loads(Path("results_long/best.json").read_text(encoding="utf-8"))
        original.load_state_dict(original_checkpoint["model"])
        restored_accuracy = evaluate(original, raw_test_x, test_y)["accuracy"]
    checkpoint_size = None
    if diagonal_accuracy > 0.9532:
        payload = {"accuracy": diagonal_accuracy, "model": model.state_dict()}
        (OUTPUT_DIR / "v9_diagonal.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        checkpoint_size = (OUTPUT_DIR / "v9_diagonal.json").stat().st_size
    results = {
        "experiment": "Diagonal Sobel features",
        "diagonal_accuracy": diagonal_accuracy,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "reverted_to_v9_long": reverted,
        "final_restored_accuracy": restored_accuracy,
        "target_95_32": diagonal_accuracy > 0.9532,
        "true_9_predicted_4": final["confusion_matrix"][9][4],
        "true_7_predicted_1": final["confusion_matrix"][7][1],
        "true_8_predicted_0": final["confusion_matrix"][8][0],
        "confusion_matrix": final["confusion_matrix"],
        "phase1_exc_isolated": phases["phase1"]["exc_isolated_accuracy"],
        "phase1_inh_isolated": phases["phase1"]["inh_isolated_accuracy"],
        "phase2_accuracy": phases["phase2"]["accuracy"],
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_size_mb": None if checkpoint_size is None else checkpoint_size / (1024 * 1024),
        "phases": phases,
        "history": history,
        "seconds": time.perf_counter() - started,
    }
    (OUTPUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fig, axis = plt.subplots(figsize=(11, 6))
    for phase in [1, 2, 3]:
        entries = history[phase]
        axis.plot([entry["epoch"] for entry in entries], [entry["accuracy"] for entry in entries], label=f"Phase {phase}")
    axis.axhline(0.9532, color="black", linestyle="--", label="95.32% target")
    axis.set(xlabel="Epoch within phase", ylabel="Accuracy", title="Diagonal Sobel Features")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v9_diagonal.png", dpi=160)
    plt.close(fig)
    print(json.dumps({key: value for key, value in results.items() if key not in {"phases", "history"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
