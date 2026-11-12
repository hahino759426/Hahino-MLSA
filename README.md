# V9 — Mixed-Link Spatial Attention

## 96.57% Test Accuracy — Backprop-Free

**Mixed-Link Spatial Attention with Path-Level Push-Pull Routing** is a local-learning MNIST classifier that reaches **96.57% test accuracy** with diagonal Sobel features and without backpropagation, a global loss, `nn.Linear`, or `nn.Parameter`.

## A neural network that learns without backpropagation—and gets better when links can be negative

This is not another MNIST demo. It is a controlled experiment in how far purely local learning can go. The network uses fixed Sobel features, spatial clusters, and path-level competition instead of a global training signal.

The surprising part is the ablation: **forcing every link to be positive collapses accuracy to 69.07%, while allowing mixed-sign links restores performance and reaches 96.57% with diagonal features.** Local inhibition is not an implementation detail; it is a mechanism that makes the architecture work.

The project is unusual in three ways:

- **Learning happens locally.** Every cluster updates from its own error—no backpropagation, global loss, `nn.Linear`, or `nn.Parameter`.
- **The architecture is the experiment.** Mixed links, spatial receptive fields, and path-level push-pull routing are tested against a strict nonnegative control, not presented as untested biological metaphors.
- **Everything is reproducible.** The repository includes the full pipeline and tests, while checkpoints and result files are generated locally rather than stored on GitHub.

## What makes this project special

- **Local learning:** every spatial cluster learns from its own classification error.
- **Mixed links:** links are bounded to `[-1,1]`, allowing local inhibition inside every path.
- **Path-level push-pull:** class scores use `Exc_Score - Inh_Score`; path types remain separate even though links within each path can be mixed.
- **Overlapping spatial attention:** four local views plus one global view preserve both detail and overall shape.
- **Diagonal visual features:** fixed 45° and 135° Sobel filters add diagonal stroke information.
- **Three timescales of learning:** feature isolation, attention calibration, then gentle co-adaptation.
- **Measured stability:** the last three epochs of V9-LONG have accuracy variance `8.67e-8`, satisfying the `<0.001` plateau criterion.

## Architecture

### Fixed Sobel retina

Each normalized 28×28 image starts with 2,352 features:

```text
784 raw pixels + 784 horizontal edges + 784 vertical edges
```

The diagonal experiment adds two more thresholded edge maps:

```text
784 diagonal-45 edges + 784 diagonal-135 edges = 3,920 total features
```

Every edge channel is binarized with `abs(gradient) > 0.5`. The kernels are fixed and identical during training and inference.

### Paths and clusters

- 10 excitatory paths and 10 inhibitory paths.
- Five clusters per path: top-left, top-right, bottom-left, bottom-right, and global.
- Sixteen binary neurons per cluster.
- Total: 1,600 neurons.

With horizontal/vertical inputs, local fields have 768 features and the global field has 2,352. With diagonal inputs, local fields have 1,280 features and the global field has 3,920.

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

Margin updates activate when `Margin < 0.5`. Phase 3 rolls back if accuracy falls below 94% or more than one point below Phase 2. In the verified V9-LONG run, Phase 2 finished at 93.90% and Phase 3 began at 94.86%, so neither rollback condition was met.

## Verified results

| Model | Final accuracy |
|---|---:|
| Strict Push-Pull | 69.07% |
| Mixed V9 (Horizontal/Vertical Sobel) | 95.32% |
| Mixed V9 (Diagonal Sobel added) | **96.57%** |
| Pruned Diagonal (9% fewer links) | **96.54%** |

Diagonal Sobel improves accuracy by **1.25 points** over V9-LONG. Pruning removes **9.05%** of diagonal links while losing only **0.03 points** of accuracy.

No checkpoints, JSON result files, or PNG graphs are tracked by Git. Run the scripts to generate them locally.

## Repository contents

```text
src/                    Models, retinas, data loading, metrics, and runners
tests/                  Architecture, schedule, and pruning tests
docs/PAPER.md           Ablation narrative and architecture discussion
README.md               Project overview and reproduction guide
LICENSE                 Apache License 2.0
requirements.txt        Python dependencies
```

Generated checkpoints, JSON results, and PNG graphs are not tracked by Git. Run the scripts to generate them locally.

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
python -m unittest discover -s tests
```

## Reproduce the 96.57% model

```bash
python -u src/run_v9_diagonal.py
```

With a custom data directory:

```bash
MNIST_DATA_DIR=/path/to/MNIST/raw python -u src/run_v9_diagonal.py
```

Training starts from scratch with seed 42 and does not load a checkpoint. The diagonal run took **133.37 seconds** on CUDA. V9-LONG took **101.53 seconds**.

See `docs/PAPER.md` for the ablation narrative and architecture discussion.


## Related project: Hahino-LSLM

The local-learning and memory-pool research developed in this repository is used in [Hahino-LSLM](https://github.com/hahino759426/Hahino-LSLM), a standalone word-level language-model project. The LSLM repository contains the language-model implementation, analysis, and V5.5 champion results; this repository remains focused on MLSA.
