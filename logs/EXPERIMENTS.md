# Experiments

One row per run. Fill this in the moment a run finishes — this table is what tells you
what to try next.

**Metric: macro F1** (competition uses it because of severe class imbalance).
Always record macro F1, never accuracy — accuracy can look fine while a rare class scores 0.

| # | Date | Model | Img | BS | LR | Epochs | CV (macro F1) | LB | Notes |
|---|------|-------|-----|----|----|--------|---------------|----|-------|
| exp01 | — | tf_efficientnet_b0 | 224 | 32 | 3e-4 | 12 | — | — | baseline, fold 0, class-weighted |

## Ideas backlog — ordered for a 1-day deadline

Do them in this order. Stop when time runs out; every step leaves a submittable model.

- [ ] **exp01 baseline, fold 0** → submit immediately (~5 min on T4)
- [ ] **All 5 folds + ensemble** (~30 min) — biggest reliable gain, low risk
- [ ] **TTA** (horizontal flip) — ~free, usually +0.005 macro F1
- [ ] Bigger image size 224 → 320 — cartoon characters can be small in frame
- [ ] Stronger backbone (efficientnet_b3, convnext_tiny)
- [ ] Tune per-class thresholds on OOF — macro F1 responds strongly to this
- [ ] Mixup / CutMix — only if overfitting shows in the learning curves

Skipped deliberately: pseudo-labelling, big ensembles, long schedules. Not enough
time for them to pay off, and each one adds a way to lose the working submission.

## Notes

- Only ~2680 training images and the test set is *larger* (2798). Overfitting is the
  main risk, so: strong augmentation, `drop_rate=0.3`, small backbone first.
- Train labels are noisy, test labels are clean. `label_smoothing=0.05` is set for this.
  It also means CV will slightly *understate* true test performance.

## Notes

Keep observations here — what helped, what didn't, and why. Negative results save
the most time.
