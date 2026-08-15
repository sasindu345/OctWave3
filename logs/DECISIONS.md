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

## D00 — Is the pipeline correct end to end? (template example, not a real result)

- **Date:** —
- **Compared:** —
- **VERDICT:** UNKNOWN — no runs yet, competition has not started
- **Next:** once data is available, run a 2-epoch smoke test on fold 0 and confirm
  train loss decreases, a checkpoint appears in Drive, and an OOF `.npz` is written.

---

## Standing decisions

Choices already made, so they don't get re-litigated every session:

| # | Decision | Reasoning |
|---|---|---|
| S1 | Fixed `StratifiedKFold(n=5, seed=42)` for all experiments | comparisons are only valid on identical splits |
| S2 | Trust local CV over public LB when they disagree | the public LB is a small sample; chasing it overfits to it |
| S3 | Ties break toward the simpler model | complexity must earn its place with evidence |
| S4 | Adopt only on `ADOPT` verdict with ≥ 3 folds | single-fold differences are usually noise |
