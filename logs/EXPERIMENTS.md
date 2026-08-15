# Experiments

One row per run. Fill this in the moment a run finishes — this table is what tells you
what to try next.

**Metric: macro F1** (competition uses it because of severe class imbalance).
Always record macro F1, never accuracy — accuracy can look fine while a rare class scores 0.

| # | Date | Model | Img | BS | LR | Epochs | CV (macro F1) | LB | Notes |
|---|------|-------|-----|----|----|--------|---------------|----|-------|
| — | 2026-08-16 | all-zeros (sample_submission) | — | — | — | — | — | **0.1521** | naive floor |
| exp01 | 2026-08-16 | tf_efficientnet_b0 | 224 | 32 | 3e-4 | 12 | **0.6818** (fold 0) | **0.7251** | baseline, class-weighted |

**Real class distribution** (from train.csv, 2680 rows):
`{0 neither: 368 (13.7%), 1 Tom: 1252 (46.7%), 2 Jerry: 841 (31.4%), 3 both: 219 (8.2%)}`
Imbalance is 5.7x — moderate, not severe. Class weights `[1.18, 0.35, 0.51, 1.97]`.

**LB (0.725) > CV (0.682).** Expected: train labels are noisy, test labels are clean, so
validation understates true performance. CV is a conservative, usable proxy.

## Ideas backlog — ordered for a 1-day deadline

Do them in this order. Stop when time runs out; every step leaves a submittable model.

- [x] **exp01 baseline, fold 0** → CV 0.6818, **LB 0.7251** ✅
- [ ] **exp02: all 5 folds + ensemble** (~25 min) — biggest reliable gain, low risk
- [ ] **exp03: TTA** (horizontal flip) — ~free
- [ ] **exp04: image size 224 → 320** — 854px source downscaled 3.8×; Jerry is small
- [ ] exp05: class_weights=False — predicted test distribution is skewed far from the
      training prior (35.6% class 0 predicted vs 13.7% in train); weights may over-correct
- [ ] Stronger backbone (efficientnet_b3, convnext_tiny) — only if the above are done
- [ ] Tune per-class thresholds on OOF — macro F1 responds strongly to this
- [ ] Mixup / CutMix — only if `overfit_check` returns OVERFITTING, not MILD

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
