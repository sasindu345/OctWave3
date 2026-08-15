# OctWave3 — Kaggle Image Classification

Training code for an image classification competition, run on Google Colab (T4 GPU).

## Structure

```
OctWave3/
├── AGENTS.md                  # RULES - evidence standards, read this first
├── CLAUDE.md                  # short pointer to AGENTS.md
├── notebooks/
│   ├── 01_train_colab.ipynb   # thin driver - run this in Colab on a T4
│   └── 02_analysis.ipynb      # evidence + decisions (CPU is fine)
├── src/
│   ├── config.py              # every tunable setting
│   ├── dataset.py             # data loading + augmentations + CV folds
│   ├── model.py               # timm model factory
│   ├── train.py               # training loop (resumable, saves OOF preds)
│   ├── predict.py             # inference -> submission.csv
│   ├── analysis.py            # statistical tests + standard charts
│   └── utils.py               # seeding, checkpoints, logging
├── data/                      # downloaded here, gitignored
├── outputs/                   # checkpoints, oof/, figures/, submissions/ - gitignored
├── logs/
│   ├── CHANGELOG.md           # what changed in the code, by date
│   ├── EXPERIMENTS.md         # what each run scored
│   └── DECISIONS.md           # what was adopted/rejected, and on what evidence
└── docs/
    ├── SETUP.md               # first-time Colab + GitHub setup
    └── WORKFLOW.md            # the daily loop
```

## Two rules that matter most

**1. Logic goes in `src/*.py`. The notebook only calls it.**
Edit `.py` files locally → `git push` → re-run the pull cell in Colab. This keeps
notebook diffs tiny and avoids the merge conflicts that make `.ipynb` painful in git.

**2. No model is adopted without a paired statistical test.**

```python
from src.analysis import decide
decide(targets, probs_baseline, probs_candidate)   # -> ADOPT / WEAK / REJECT
```

Two accuracies (0.912 vs 0.908) are not a comparison — that gap is usually inside
the noise band. `decide()` runs a paired bootstrap and McNemar's test on the same
validation samples and tells you whether the difference is real. Full rules in
[AGENTS.md](AGENTS.md).

## Quick start

1. Push this repo to GitHub.
2. Open `notebooks/01_train_colab.ipynb` in Colab (see [docs/SETUP.md](docs/SETUP.md)).
3. Set `REPO_URL` and `COMP` in the notebook, set `cfg.num_classes`, run all cells.

## Checklist when the competition opens

- [ ] Set `COMP` (competition slug) in the notebook
- [ ] Inspect the data layout, set `cfg.data_mode` to `"folder"` or `"csv"` in `src/config.py`
- [ ] Set `cfg.num_classes` and `cfg.image_ext`
- [ ] Match the evaluation metric in `src/train.py::validate` (currently accuracy + macro F1)
- [ ] Match the submission format in `src/predict.py::make_submission`
- [ ] Run a 2-epoch smoke test on one fold before any long run
