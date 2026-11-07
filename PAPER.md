# Mixed-Link Spatial Attention with Path-Level Push-Pull Routing

## Abstract

We present a spatial attention network trained entirely by local rules. The model combines fixed Sobel features, five overlapping receptive fields per path, and separate excitatory/inhibitory class routing. A controlled ablation shows that mixed-sign links are important: strict `[0,1]` links achieved 69.07%, while mixed `[-1,1]` links achieved 94.45% under the short schedule. Extending isolated feature learning and gentle co-adaptation increased final accuracy to **95.32%**, with a best checkpoint of **95.34%**. The result demonstrates that local inhibition within paths and path-level push-pull routing are compatible.

## 1. Introduction

The project investigates whether a network can learn MNIST through local delta rules and structural spatial attention rather than global backpropagation. Its key finding is that sign restrictions inside a path remove useful local inhibition. The final architecture therefore permits mixed links while retaining separate excitatory and inhibitory paths at the routing level.

## 2. Method

### Fixed visual front end

Normalized pixels are processed by fixed horizontal and vertical Sobel filters. Raw pixels and both **thresholded (binary) edge maps**, produced by `abs(gradient) > 0.5`, are concatenated into 2,352 features. The filter never changes.

### Spatial cluster population

Each of 20 paths contains five clusters of 16 binary neurons. Four clusters observe overlapping 16×16 image regions across all three channels; the fifth observes the complete input. Cluster activation is the fraction of firing neurons.

### Push-pull routing

Each class has one excitatory and one inhibitory path. Attention weights combine their cluster activations, and the final class score is excitation minus inhibition. Mixed links inside a path provide local subtractive effects without merging the path types.

### Local learning

Links use `eta × (Target − Activation) × Input` and are clipped to `[-1,1]`. Thresholds use the corresponding local error. No global loss or autograd training is used.

Attention weights learn from margin errors and follow normalize → clip → normalize. This keeps each path’s attention distribution normalized after clipping.

## 3. Extended curriculum

1. Twenty epochs of independent 1-vs-9 path isolation, with early stopping only above 99% for every path.
2. Five attention-only epochs with links and thresholds frozen.
3. Twenty gentle co-adaptation epochs at rate `0.0001`, halving rates in each final five epochs.

Phase 3 uses rollback below 94% or after a drop greater than one point below Phase 2. The reported model completes the schedule without rollback. Specifically, Phase 2 finished at 93.90%, and Phase 3 began at 94.86%; neither the 94% threshold nor the one-point drop condition was triggered.

## 4. Results

| Phase | Accuracy |
|---|---:|
| Isolation (ensemble with frozen attention) | 93.12% |
| Attention | 93.90% |
| Co-adaptation, epoch 5 | 95.13% |
| Co-adaptation, epoch 10 | 95.24% |
| Co-adaptation, epoch 15 | 95.28% |
| Co-adaptation, epoch 20 | **95.32%** |
| Best checkpoint, epoch 13 | **95.34%** |

Individual paths achieved approximately 97–99% isolated accuracy. The Phase 1 ensemble value of 93.12% is lower because attention weights were still frozen at 0.2 and had not yet learned which clusters to trust.

The final three epochs are 95.22%, 95.30%, and 95.32%. Their variance is `8.67e-8`, meeting the `<0.001` plateau criterion.

## 5. Ablation

This ablation shows empirically that enforcing strictly positive links collapses accuracy to 69.07% in this architecture, consistent with losing local inhibition needed for AND-NOT-like discrimination. It is an experimental result, not a mathematical proof that every such network is unable to learn.

| Link rule | Final accuracy |
|---|---:|
| Strict `[0,1]` | 69.07% |
| Mixed `[-1,1]`, short schedule | 94.45% |
| Mixed `[-1,1]`, V9-LONG | **95.32%** |

Strict training failed its isolated-learning gate after Phase 1. Mixed training completed all phases. Mixed links therefore provide a measured 25.38-point improvement over strict links before the long schedule, and the extended curriculum adds a further 0.87 points.

## 6. Conclusion

V9-LONG shows that fixed spatial features, local delta learning, normalized cluster attention, and path-level push-pull routing can jointly reach 95.32% on MNIST without backpropagation. Mixed-sign links are an intentional source of local inhibition and are central to the observed performance.
