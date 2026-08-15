# Changelog

Code and structure changes, newest first. One line each — keep it cheap to maintain.
(For run results, use [EXPERIMENTS.md](EXPERIMENTS.md) instead.)

## 2026-08-16 (Notebook & Validation Upgrades)

- Added Test-Time Augmentation (TTA, horizontal flip averaging) to `src/predict.py` for variance-reduction accuracy gain without capacity inflation.
- Upgraded `notebooks/01_train_colab.ipynb`:
  - Section 6: Configured 5-fold training by default (`cfg.train_folds = [0, 1, 2, 3, 4]`).
  - Section 7: Multi-fold training loop with automated OOF CV summary (`mean ± std`).
  - Section 8: Visual learning curves + integrated `overfit_check()` generalization diagnostic.
  - Section 9: TTA inference and automated submission integrity validation checks.
- Upgraded `notebooks/02_analysis.ipynb`: switched all metrics to competition macro F1, updated class names, and added multi-fold box plot support.
- Updated `src/analysis.py`: `evidence_report()` now dynamically formats metrics and labels based on `cfg.metric`.
- Added [docs/NOTEBOOK_GUIDE.md](../docs/NOTEBOOK_GUIDE.md) covering the 5-fold training workflow, overfitting guardrails, and submission pipeline.

## 2026-08-16 (later)

- exp01 baseline complete: CV macro F1 0.6818 (fold 0), **LB 0.7251**. All-zeros
  floor is 0.1521, so the pipeline is confirmed working end to end.
- Real class distribution measured: `{0: 368, 1: 1252, 2: 841, 3: 219}` — 5.7x
  imbalance, moderate rather than severe. Majority class is Tom (1), not "neither".
- Added `overfit_check()` to `src/analysis.py` — four measurable tests
  (val-loss turn, loss gap, metric plateau, best-epoch lag) returning
  HEALTHY / MILD / OVERFITTING / UNDERTRAINED. Validated on three known cases.
- Added [docs/MODELING.md](../docs/MODELING.md): model choice rationale, the
  anti-overfitting toolkit, and the evidence standard with thresholds.
- Four Colab environment failures fixed along the way: private-repo auth (GH_TOKEN),
  nested `images/images/` path, `sys.path` loss after runtime restart, and stale
  import caches after re-cloning.

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
