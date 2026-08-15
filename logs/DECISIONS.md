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

---

## Standing decisions

Choices already made, so they don't get re-litigated every session:

| # | Decision | Reasoning |
|---|---|---|
| S1 | Fixed `StratifiedKFold(n=5, seed=42)` for all experiments | comparisons are only valid on identical splits |
| S2 | Trust local CV over public LB when they disagree | the public LB is a small sample; chasing it overfits to it |
| S3 | Ties break toward the simpler model | complexity must earn its place with evidence |
| S4 | Adopt only on `ADOPT` verdict with ≥ 3 folds | single-fold differences are usually noise |
