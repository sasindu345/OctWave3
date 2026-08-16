# Decision log

One entry per accept/reject decision. Written **after** `decide()` runs, using its
actual output. If there is no evidence line, there is no entry.

Format (copy this block):

```
## D00 — <the question, in one line>
- **Date:**
- **Compared:** exp01 (baseline) vs exp02 (candidate)
- **Folds:** 0,1,2
- **Baseline:** 0.0000 ± 0.0000
- **Candidate:** 0.0000 ± 0.0000
- **Paired bootstrap:** diff +0.0000, 95% CI [ , ], P(B better) = 0.000
- **McNemar:** n flips to B, n to A, p = 0.000
- **Chart:** outputs/figures/xxx.png
- **VERDICT:** ADOPT / WEAK / REJECT
- **Reasoning:** <one or two sentences — what the evidence actually shows>
- **Next:** <the experiment this result makes worth running>
```

---

## D01 — Is the pipeline correct end to end?

- **Date:** 2026-08-16
- **Evidence:** exp01 trained 12 epochs on fold 0, submitted, scored on the LB.
- **CV macro F1:** 0.6818 (fold 0, epoch 10) · **LB:** 0.7251 · **all-zeros floor:** 0.1521
- **VERDICT:** CONFIRMED — pipeline works end to end.
- **Reasoning:** LB sits well above the naive floor, and above CV, which is expected
  given noisy train labels vs clean test labels. CV is conservative but usable.
- **Next:** D02 — does overfitting after epoch ~9 cost us anything?

## D02 — Is exp01 overfitting?

- **Date:** 2026-08-16
- **Evidence:** run_log fold 0, epochs 1-12.
- **Train loss:** falls throughout, 0.699 (e8) → 0.578 (e12)
- **Valid loss:** bottoms at 1.1194 (e9), then rises → 1.1431 (e12)
- **Macro F1:** plateaus 0.668 → 0.682, best 0.6818 at e10
- **Gap (train-val):** widens −0.465 (e8) → −0.565 (e12)
- **VERDICT:** MILD overfitting, onset ~epoch 9.
- **Reasoning:** the classic signature — training loss still falling while validation
  loss turns upward. But macro F1 plateaus rather than collapsing, so the cost is
  small. Early stopping did not fire because F1 kept inching up.
- **Next:** the fix is more data through the model (5 folds + ensemble), not fewer
  epochs. Revisit epochs only if 5-fold shows the same turn.

## D03 — Did the exp03 training run cause the Kaggle drop? NO — the multipliers did

- **Date:** 2026-08-16
- **Evidence:** Kaggle shows `exp03_5fold_b0_e18_mult.csv` against 0.746580. The plain
  `exp03_5fold_b0_e18.csv` (argmax) was never submitted.
- **VERDICT:** the CV↔LB comparison was CONFOUNDED. exp02-argmax vs exp03-multiplier
  differs in two ways at once.
- **Reasoning:** multipliers were fitted on OOF carrying noisy train labels, then applied
  to a clean test set. They moved 405/2798 predictions (14.5%), mostly `3 → 1`,
  cutting class-3 predictions by 46% (561 → 304) and pushing the output toward the
  training prior. Under macro F1 that costs rare-class recall directly.
- **Consequence:** "higher OOF does not mean higher Kaggle" is NOT yet demonstrated.
  Submitting the plain exp03 CSV tests it for free.

## D04 — Is the model undertrained at 12 epochs? NO (corrects an earlier read)

- **Date:** 2026-08-16
- **Evidence:** `results/figures/exp03_5fold_b0_e18_training.png`
- **Train loss:** flattens at ~0.5 by epoch 15. **Valid loss:** flat 1.2-1.45 from epoch 7.
- **VERDICT:** converged. The earlier "best epoch == last epoch" signal was macro-F1
  noise on ~44 rare-class validation samples, not continued learning.
- **Consequence:** stop pursuing longer schedules. exp03 CV +0.0137 over exp02 is
  below the 0.018 noise floor and is not a demonstrated improvement.

## D05 — Where macro F1 is actually lost: BOTH joint-decision classes

- **Date:** 2026-08-16
- **Evidence:** `results/figures/exp03_5fold_b0_e18_errors.png` (row-normalised confusion)
- Class 0 "neither" recall **0.61** — 33% leaks to classes 1/2 (hallucinates a character)
- Class 3 "both"    recall **0.53** — 39% leaks to classes 1/2 (misses the second character)
- Class 1 recall 0.78, class 2 recall 0.78 — single-character frames are fine
- Calibration ECE 0.073 — acceptable, no fix needed
- **VERDICT:** the model handles "exactly one character" well and fails on both
  *joint* decisions (both-present, both-absent).
- **Consequence:** this is a label-structure problem, not a capacity or schedule problem.
  Motivates the two-sigmoid multilabel head (exp05), where class 0 requires both
  detectors negative and class 3 requires both positive.

## D06 — exp05 multilabel vs champion: ADOPT (first proven gain in the project)

- **Date:** 2026-08-16
- **Compared:** `exp02_5fold_b0` (champion) vs `exp05_multilabel_b0`, all 5 folds, same splits
- **Champion CV:** 0.6419 ± 0.0153 · **exp05 CV:** 0.6877 ± 0.0307
- **Per-fold delta:** +0.0471, +0.0247, +0.0445, +0.0866, +0.0264 → **5/5 folds improved**
- **Paired bootstrap:** diff **+0.0453**, 95% CI **[+0.0258, +0.0659]** (excludes 0), P(B better) = 1.000
- **McNemar:** 311 flips to exp05, 157 to champion, **p = 0.0000**
- **Overfit check:** MILD on all 5 folds, macro F1 still improving after the val-loss turn
- **VERDICT: ADOPT** — the only change so far to clear the 0.018 noise floor (2.5x it)
- **Chart:** `results/figures/exp05_multilabel_b0_training.png`

### But the mechanism was NOT what was predicted

Row-normalised recall, exp03 (softmax) → exp05 (multilabel):

| class | softmax | multilabel | change |
|---|---|---|---|
| 0 neither | 0.61 | **0.54** | −0.07 |
| 1 Tom | 0.78 | **0.86** | **+0.08** |
| 2 Jerry | 0.78 | **0.86** | **+0.08** |
| 3 both | 0.53 | **0.46** | −0.07 |

The hypothesis was that pooling would fix the joint classes (0 and 3). It did the
opposite: single-character recall improved sharply, joint-class recall fell. Macro F1
still rose +0.045, which means **precision** on classes 0 and 3 must have improved
enough to outweigh the recall loss — consistent with the softmax model massively
over-predicting them (39.4% predicted class 0 vs a 13.7% prior).

**Consequence:** the two models have opposite error profiles — champion trades
precision for recall on rare classes, multilabel does the reverse. That is the
profile that ensembles well. Check the oracle ceiling before assuming it.

---

## Standing decisions

Choices already made, so they don't get re-litigated every session:

| # | Decision | Reasoning |
|---|---|---|
| S1 | Fixed `StratifiedKFold(n=5, seed=42)` for all experiments | comparisons are only valid on identical splits |
| S2 | Trust local CV over public LB when they disagree | the public LB is a small sample; chasing it overfits to it |
| S3 | Ties break toward the simpler model | complexity must earn its place with evidence |
| S4 | Adopt only on `ADOPT` verdict with ≥ 3 folds | single-fold differences are usually noise |
