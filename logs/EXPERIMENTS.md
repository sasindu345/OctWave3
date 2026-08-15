# Experiments

One row per run. Fill this in the moment a run finishes — this table is what tells you
what to try next.

| # | Date | Model | Img | BS | LR | Epochs | CV (acc) | LB | Notes |
|---|------|-------|-----|----|----|--------|----------|----|-------|
| exp01 | — | tf_efficientnet_b0 | 224 | 64 | 3e-4 | 10 | — | — | baseline, fold 0 only |

## Ideas backlog

- [ ] Bigger image size (224 → 300 → 384)
- [ ] Stronger backbone (efficientnet_b3, convnext_tiny, swin_tiny)
- [ ] Full 5-fold CV + fold ensemble
- [ ] TTA (horizontal flip at inference)
- [ ] Mixup / CutMix
- [ ] Label smoothing 0.1
- [ ] Cosine schedule with warmup restarts
- [ ] Pseudo-labelling on test set (late-stage only)

## Notes

Keep observations here — what helped, what didn't, and why. Negative results save
the most time.
