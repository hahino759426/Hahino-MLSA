# Mixed-Link Spatial Attention with Path-Level Push-Pull Routing

## 96.57% Test Accuracy — Backprop-Free

## Abstract

We present a spatial attention network trained entirely by local rules. The model combines fixed Sobel features, five overlapping receptive fields per path, and separate excitatory/inhibitory class routing. A controlled ablation shows that mixed-sign links are important: strict `[0,1]` links achieved 69.07%, mixed horizontal/vertical Sobel features achieved 95.32%, and adding diagonal Sobel features increased accuracy to **96.57%**. Pruning 9.05% of the diagonal model’s links retained **96.54%** accuracy. No backpropagation, global loss, `nn.Linear`, or `nn.Parameter` is used.

## 1. Introduction

The project investigates whether a network can learn MNIST through local delta rules and structural spatial attention rather than global backpropagation. Its first finding is that sign restrictions inside a path remove useful local inhibition. Its second is that fixed diagonal edge features resolve additional stroke ambiguities without changing the learning rule.

## 2. Method

### Fixed visual front end

Normalized pixels are processed by fixed horizontal, vertical, 45°, and 135° Sobel filters. Raw pixels and four **thresholded binary edge maps**, each produced by `abs(gradient) > 0.5`, are concatenated into 3,920 features. The filters never change.

### Spatial cluster population

Each of 20 paths contains five clusters of 16 binary neurons. Four clusters observe overlapping 16×16 image regions across all five channels; the fifth observes the complete input. Cluster activation is the fraction of firing neurons.

### Push-pull routing

Each class has one excitatory and one inhibitory path. Attention weights combine their cluster activations, and the final class score is excitation minus inhibition. Mixed links inside a path provide local subtractive effects without merging the path types.

### Local learning

Links use `eta × (Target − Activation) × Input` and are clipped to `[-1,1]`. Thresholds use the corresponding local error. No global loss or autograd training is used.

Attention weights learn from margin errors and follow normalize → clip → normalize. This keeps each path’s attention distribution normalized after clipping.

## 3. Extended curriculum

1. Twenty epochs of independent 1-vs-9 path isolation, with early stopping only above 99% for every path.
2. Five attention-only epochs with links and thresholds frozen.
3. Twenty gentle co-adaptation epochs at rate `0.0001`, halving rates in each final five epochs.

Phase 3 uses rollback below 94% or after a drop greater than one point below Phase 2. The reported V9-LONG model completed the schedule without rollback: Phase 2 finished at 93.90%, and Phase 3 began at 94.86%, so neither condition was triggered.

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
| Diagonal Sobel model | **96.57%** |
| Pruned diagonal model | **96.54%** |

Individual V9-LONG paths achieved approximately 97–99% isolated accuracy. The Phase 1 ensemble value of 93.12% is lower because attention weights were still frozen at 0.2 and had not yet learned which clusters to trust.

The final three V9-LONG epochs are 95.22%, 95.30%, and 95.32%. Their variance is `8.67e-8`, meeting the `<0.001` plateau criterion.

## 5. Ablation

This ablation shows empirically that enforcing strictly positive links collapses accuracy to 69.07% in this architecture, consistent with losing local inhibition needed for AND-NOT-like discrimination. It is an experimental result, not a mathematical proof that every such network is unable to learn.

| Link/feature rule | Final accuracy |
|---|---:|
| Strict Push-Pull | 69.07% |
| Mixed V9 (Horizontal/Vertical Sobel) | 95.32% |
| Mixed V9 (Diagonal Sobel added) | **96.57%** |
| Pruned Diagonal (9% fewer links) | **96.54%** |

Strict training failed its isolated-learning gate after Phase 1. Mixed training completed all phases. The extended curriculum adds 0.87 points over the short mixed-link run, diagonal features add another 1.25 points, and pruning preserves all but 0.03 points of the diagonal result.

## 6. Conclusion

The project shows that fixed spatial features, local delta learning, normalized cluster attention, mixed-sign links, and path-level push-pull routing can jointly reach **96.57% on MNIST without backpropagation**. Mixed-sign links are an intentional source of local inhibition, and fixed diagonal features provide a further measurable improvement. Pruning then reduces the link count by 9.05% with negligible accuracy loss.
