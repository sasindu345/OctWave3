# Modelling: model choice, overfitting, and what counts as evidence

Task: 4-class image classification, 2680 training images, 480×854 cartoon frames,
metric = macro F1. Everything below follows from those four facts.

---

## 1. Which model, and why

**Chosen baseline: `tf_efficientnet_b0`, ImageNet-pretrained, 224px.**

The dataset is small — 2680 images across 4 classes, about 2144 per training fold.
With that little data, the binding constraint is not model capacity, it is overfitting.
Model choice follows from the ratio of parameters to examples.

| Candidate | Params | Verdict for this task |
|---|---|---|
| **efficientnet_b0** | 5.3M | **baseline** — best accuracy-per-parameter at this data size |
| efficientnet_b3 | 12M | worth testing, 2.3× the parameters on the same 2680 images |
| convnext_tiny | 28M | worth testing; modern, but 10× b0's parameters |
| ViT / Swin | 22M+ | avoid — transformers need far more data than we have |
| ResNet50 | 25M | fine but strictly worse than b0 per parameter |
| training from scratch | — | hopeless at 2680 images; transfer learning is mandatory |

**Why pretrained weights matter more than architecture here.** ImageNet features
(edges, colours, textures, part shapes) transfer to cartoon frames even though the
domain differs — only the classifier head has to learn Tom-vs-Jerry. Without
pretraining, 2680 images cannot train any of these networks.

**The highest-value knob is resolution, not architecture.** Source frames are 854px
wide and we downscale to 224 — a 3.8× reduction. Jerry is small in frame, so detail
is being thrown away. Expect more from 224 → 320 than from b0 → b3.

**Ladder, in order.** Each step is gated by `decide()`; stop when the deadline hits.

1. b0 @ 224, fold 0 ✅ done — CV 0.6818, LB 0.7251
2. b0 @ 224, **all 5 folds + ensemble** ← biggest reliable gain
3. **TTA** (horizontal flip) — near-free
4. b0 @ **320**
5. convnext_tiny or b3 @ 224 — only if 2-4 are done and time remains

Ties go to the simpler model.

---

## 2. How to get accuracy without overfitting

Ranked by value **for this dataset**. The first five are already on.

| Technique | Status | Why it works here |
|---|---|---|
| Transfer learning | ✅ on | the single biggest factor at 2680 images |
| Augmentation (crop, flip, shift/scale/rotate, brightness, cutout) | ✅ on | multiplies effective dataset size |
| `drop_rate = 0.3` | ✅ on | regularises the classifier head |
| `label_smoothing = 0.05` | ✅ on | train labels are noisy; stops hard memorisation |
| Early stopping on **macro F1** | ✅ on | selecting on the competition metric, not accuracy |
| **K-fold ensemble** | ⬜ next | averaging 5 folds cuts variance — the cheapest real gain left |
| TTA | ⬜ | averaging over flips reduces prediction variance |
| Mixup / CutMix | ⬜ | strong regulariser; only if overfitting turns severe |
| Weight EMA | ⬜ | smooths the final weights; small, reliable |
| More weight decay / smaller LR | ⬜ | blunt instruments; try only after the above |

**Ensembling is the safest way to raise a score without overfitting**, because it
reduces variance rather than adding capacity. That is why 5-fold sits above every
architecture change on the list.

**What NOT to do:** a bigger backbone to fix overfitting. More parameters on the same
2680 images makes it worse, not better.

---

## 3. What counts as evidence of overfitting

Opinion is not evidence. Run:

```python
from src.analysis import overfit_check
import pandas as pd
log = pd.read_json(cfg.out_dir / 'run_log.jsonl', lines=True)
overfit_check(log, 'exp01_baseline', fold=0)
```

It applies four measurable tests and returns a verdict.

| # | Test | What it measures | Overfitting when |
|---|---|---|---|
| 1 | **Val loss turn** | epochs since validation loss bottomed | bottomed ≥ 2 epochs ago and has risen > 0.01 |
| 2 | **Loss gap** | `val_loss − train_loss`, best epoch vs final | widened by > 0.05 |
| 3 | **Metric plateau** | did macro F1 improve after the turn? | no further gain (< 0.005) |
| 4 | **Best-epoch lag** | distance from best epoch to last | best == last → *undertrained*, not overfit |

**Verdicts**

- `HEALTHY` — no sustained rise in validation loss
- `MILD` — loss turned up but the metric is still improving; little real cost
- `OVERFITTING` — loss rising, gap widening, metric stopped improving → act
- `UNDERTRAINED` — best epoch is the last one → train longer

Tests 1 and 2 alone are not enough. Validation loss commonly rises while accuracy and
F1 keep improving — the model gets more confident and more wrong on a few samples
while getting more right overall. **Test 3 is what separates "overfitting" from
"overfitting that matters".**

### Three further checks beyond one run

- **Fold spread.** Across 5 folds, report `mean ± std`. A std above ~0.02 means fold
  differences are drowning your improvements — use `plot_fold_box`.
- **CV ↔ LB agreement.** Plot every submission with `plot_cv_vs_lb`. Pearson r > 0.7
  means CV is trustworthy. If CV rises while LB falls, you are overfitting your
  validation split.
- **Per-class F1.** Macro F1 averages all 4 classes equally. Class 3 has only 219
  images (~44 per fold), so its F1 is noisy and dominates your variance. Check
  `plot_per_class_f1` before believing any small improvement.

### exp01 result (2026-08-16)

Verdict **MILD**. Validation loss bottoms out and turns up, and the loss gap widens
from 0.15 to 0.57 — but macro F1 kept improving to epoch 10, so the cost is small.
Action taken: none. The fix for a small-data model is more data through the model
(5-fold ensembling), not fewer epochs.

---

## 4. The honest limits

- **CV understates test performance here.** Train labels are noisy, test labels are
  clean, so validation is measured against corrupted answers. Observed: CV 0.682 vs
  LB 0.725. CV remains useful as a *relative* signal — it ranks options correctly —
  just not as an absolute prediction of the LB.
- **~44 rare-class images per fold.** A ±0.01 macro F1 move is well inside noise.
  Use `decide()` before believing any small gain.
- **The public LB is itself a sample.** Do not chase it past what CV supports.
