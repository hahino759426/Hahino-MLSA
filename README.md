# V9-LONG — Mixed-Link Spatial Attention

**Mixed-Link Spatial Attention with Path-Level Push-Pull Routing** is a local-learning MNIST network that reaches **95.32% test accuracy** without backpropagation, a global loss, `nn.Linear`, or `nn.Parameter`.

## What makes this project special

- **Local learning:** every spatial cluster learns from its own classification error.
- **Mixed links:** links are bounded to `[-1,1]`, allowing local inhibition inside every path.
- **Path-level push-pull:** class scores still use `Exc_Score - Inh_Score`; the path types remain separate even though links within each path can be mixed.
- **Overlapping spatial attention:** four local views plus one global view preserve both detail and overall shape.
- **Three timescales of learning:** feature isolation, attention calibration, then gentle co-adaptation.
- **Measured stability:** the last three epochs have accuracy variance `8.67e-8`, satisfying the `<0.001` plateau criterion.

## Architecture

### Fixed Sobel retina

Each normalized 28×28 image becomes 2,352 features:

```text
784 raw pixels + 784 horizontal edges + 784 vertical edges
```

The Sobel kernels are fixed and identical during training and inference.

### Paths and clusters

- 10 excitatory paths and 10 inhibitory paths.
- Five clusters per path: top-left, top-right, bottom-left, bottom-right, and global.
- Sixteen binary neurons per cluster.
- Total: 1,600 neurons.

Local fields cover 16×16 pixels across three input channels. The global cluster receives all 2,352 features.

For cluster `k`:

```text
Activation_k = firing_neurons / 16
```

The path output is the attention-weighted sum of its five cluster activations. Final class routing is:

```text
Score_c = Exc_Output_c - Inh_Output_c
Prediction = argmax(Score)
```

## Learning rules

### Local delta links

```text
Error = Target - Cluster_Activation
Links += eta_link * Error * Input
Links = clip(Links, -1, 1)
```

### Local thresholds

The V9.8 rule is:

```text
Threshold -= eta * Error
```

### Attention weights

Weights start at `0.2`. Margin-driven updates use true- and predicted-class errors, followed by:

```text
normalize → clip to [0.1,0.6] → normalize again
```

### Three phases

| Phase | Epochs | What learns |
|---|---:|---|
| 1. Isolation | Up to 20 | Links and thresholds using independent 1-vs-9 targets; attention frozen |
| 2. Attention | 5 | Cluster weights only; links and thresholds frozen |
| 3. Co-adaptation | 20 | Links, thresholds, and weights with tiny rates |

Phase 1 stops early only if every path exceeds 99% isolated accuracy. Phase 3 uses `eta = eta_w = 0.0001`, then halves both rates during each of its final five epochs:

```text
0.00005, 0.000025, 0.0000125, 0.00000625, 0.000003125
```

Margin updates activate when `Margin < 0.5`. Phase 3 rolls back if accuracy falls below 94% or more than one point below Phase 2. In the verified run, Phase 2 finished at 93.90% and Phase 3 began at 94.86%, so neither rollback condition was met.

## Verified results

| Configuration | Accuracy |
|---|---:|
| Strict `[0,1]` links | 69.07% |
| Short mixed-link run | 94.45% |
| Original V9.8 baseline | 94.73% |
| **V9-LONG final** | **95.32%** |
| **V9-LONG best, Phase 3 epoch 13** | **95.34%** |

Phase checkpoints in the final run:

| Phase 3 epoch | Accuracy |
|---|---:|
| 5 | 95.13% |
| 10 | 95.24% |
| 15 | 95.28% |
| 20 | **95.32%** |

The final confusion matrix is included in `results_long/results.json`. The best model is `results_long/best.json`, and the training curve is `results_long/v9_long.png`.

## Repository contents

```text
v9_8_network.py     Mixed-link spatial attention model and learning rules
fixed_retina.py     Fixed Sobel preprocessing
data.py             IDX loading and normalization
metrics.py          Confusion matrix and recalls
run_v9_8.py         Standard three-phase pipeline and evaluation helpers
run_v9_long.py      Extended 20+5+20 schedule
test_*.py           Architecture and schedule tests
results_long/       Verified result, best checkpoint, and graph
```

## Setup

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Place these MNIST IDX files in `data/MNIST/raw/`:

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

Alternatively, set `MNIST_DATA_DIR` to an existing directory containing those files.

## Run tests

```bash
python -m unittest test_v9_8_fixed.py test_v9_long.py
```

## Reproduce V9-LONG

```bash
python -u run_v9_long.py
```

With a custom data directory:

```bash
MNIST_DATA_DIR=/path/to/MNIST/raw python -u run_v9_long.py
```

Training starts from scratch with seed 42 and does not load a checkpoint. The verified run took **101.53 seconds** on CUDA.

See `PAPER.md` for the ablation narrative and architecture discussion.
