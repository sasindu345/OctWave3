# Changelog

Code and structure changes, newest first. One line each — keep it cheap to maintain.
(For run results, use [EXPERIMENTS.md](EXPERIMENTS.md) instead.)

## 2026-08-16

- Competition opened: `oct-wave-3-0-kaggle-challenge-02`. Slug set in notebook 01.
- Added section 5b "Inspect the data" to notebook 01 — prints layout, class counts,
  imbalance ratio, image shapes and submission columns. Tested on a synthetic dataset.
- Competition details received (Tom & Jerry character classification). Configured:
  4 classes, metric **macro F1**, flat `images/` dir + train.csv/test.csv,
  submission `filename,appearance`.
- `train.py` now selects the best epoch on macro F1 (was accuracy) and applies
  inverse-frequency class weights; `label_smoothing=0.05` for noisy train labels.
- `dataset.py`: `build_test_dataframe()` and `class_weights()` added; loader rewritten
  for the flat-images layout.
- Verified end-to-end on a synthetic replica (2680 train / 200 test, 4 imbalanced
  classes): trains, writes valid `filename,appearance` submission, saves OOF.
- Still UNKNOWN until cell 5b runs on real data: actual class distribution and
  image resolution.
- Added `results/` (tracked) + Colab push cell so run logs and figures come back to git.

## 2026-08-15

- Added `AGENTS.md` + `CLAUDE.md`: evidence rules for AI agents — verify before claiming,
  no invented metrics, all model comparisons through `decide()`.
- Added `src/analysis.py`: bootstrap CIs, paired bootstrap, McNemar test, and six
  standard charts. Smoke-tested on synthetic data: correctly returns ADOPT for a
  real +8pt gain, WEAK for +1.8pt, REJECT for a reseed of the same model.
- `train.py` now saves best-epoch out-of-fold predictions to `outputs/oof/*.npz`
  and logs the train/valid loss gap each epoch (overfitting signal).
- Added `logs/DECISIONS.md` and `notebooks/02_analysis.ipynb`.
- Initial project structure: `src/` training pipeline, thin Colab notebook, docs, logs.
- Training loop resumes automatically from Drive checkpoints after a Colab disconnect.
- Placeholders left for competition specifics: `COMP` slug, `num_classes`, `data_mode`,
  metric in `train.py::validate`, submission format in `predict.py::make_submission`.
