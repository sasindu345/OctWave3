# Changelog

Code and structure changes, newest first. One line each — keep it cheap to maintain.
(For run results, use [EXPERIMENTS.md](EXPERIMENTS.md) instead.)

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
