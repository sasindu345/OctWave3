# Daily workflow

## The loop

```
laptop:  edit src/*.py  ->  git commit  ->  git push
colab:   run cell 4 (git pull)  ->  run cell 6 (config)  ->  run cell 7 (train)
after:   add a row to logs/EXPERIMENTS.md  ->  commit
```

You edit `.py` files in your normal editor with normal diffs. The notebook stays
unchanged, so it never conflicts.

## If you DO change the notebook in Colab

1. `Edit → Clear all outputs` (keeps the diff readable — outputs are huge base64 blobs)
2. `File → Save a copy in GitHub`
3. Target repo, branch `main`, path `notebooks/01_train_colab.ipynb`
4. Write a real commit message
5. On your laptop: `git pull` before your next edit

Save from **one place at a time**. Editing the notebook in Colab *and* locally in the
same session is the main way people create `.ipynb` conflicts.

## Long training runs

- Keep the Colab tab open and visible; idle sessions get reclaimed.
- Every epoch writes `*_last.pt` to Drive. If you get disconnected, just reconnect,
  re-run cells 3→7, and it picks up at the next epoch.
- Before a long run, smoke test: `cfg.epochs = 2` and a small `img_size`. Confirm loss
  drops and a checkpoint appears in Drive. *Then* start the real run.
- Rough T4 budget: EfficientNet-B0 @ 224px, ~20k images ≈ 3–5 min/epoch.
  Doubling image size roughly quadruples the time.

## Batch size for a 16GB T4

| Model | Image size | Batch size |
|---|---|---|
| efficientnet_b0 / resnet50 | 224 | 64 |
| efficientnet_b3 | 300 | 24 |
| efficientnet_b4 | 380 | 12 |
| convnext_tiny | 224 | 32 |

If you hit CUDA OOM: halve `batch_size` and set `grad_accum` to 2 — same effective
batch, less memory. Then `Runtime → Restart session` to actually free the GPU.

## Competition strategy order

1. Get *any* submission on the leaderboard (baseline b0, few epochs). Confirms the
   whole pipeline works end to end.
2. Build trustworthy CV — if local score and LB move together, you can iterate fast.
3. Then improve, one change at a time: bigger image size → better backbone → more
   augmentation → longer schedule → TTA → ensemble folds.
4. Log every attempt. An untracked experiment is a wasted one.
