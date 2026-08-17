# OctWave3 — Tom & Jerry Image Classification

A comprehensive, robust solution for the `oct-wave-3-0` Kaggle Image Classification competition. This repository contains the training code and analytical pipeline designed to run efficiently on Google Colab (T4 GPU).

| Notebook | Environment | Link |
|---|---|---|
| **Train** | Needs T4 GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sasindu345/OctWave3/blob/main/notebooks/01_train_colab.ipynb) |
| **Analyse** | CPU is fine | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sasindu345/OctWave3/blob/main/notebooks/02_analysis.ipynb) |

Click a badge to open that notebook in Colab. Train first, then analyse.

## 🎯 The Problem We Tackled

In the `oct-wave-3-0` competition, the objective was to build an image classification model to detect the presence of two famous cartoon characters—Tom and Jerry. The model had to categorize each image into one of four classes:
- **0:** Neither
- **1:** Tom present
- **2:** Jerry present
- **3:** Both present

### Key Challenges
1. **Severe Class Imbalance:** The training dataset is heavily skewed, with some classes appearing far less frequently than others.
2. **Noisy Labels:** The training data contains mislabeled examples, whereas the test set has clean, accurate labels.
3. **Macro F1 Metric:** The competition is evaluated on the **Macro F1** score. This penalizes poor performance on minority classes much more than standard accuracy would, demanding a highly robust classifier.
4. **Data Scarcity:** With only ~2,680 training images, the risk of overfitting is extremely high.

## 💡 Our Solution & Approach

To address these challenges, we built a modular and highly disciplined pipeline:

- **Efficient Architecture:** We utilize `tf_efficientnet_b0` as our core backbone with a 4-way CrossEntropy softmax head, offering a great balance of feature extraction capability and parameter efficiency.
- **Handling Imbalance:** We apply **inverse-frequency class weights** during training to ensure rare classes are prioritized, directly optimizing for the Macro F1 metric.
- **Combatting Noisy Labels & Overfitting:**
  - We employ **label smoothing** (0.05) to prevent the model from becoming overconfident on noisy training labels.
  - Aggressive dropout (0.3) and weight decay (1e-4) are used to regularize the network given the small dataset.
- **Accelerated Training:** We use Automatic Mixed Precision (AMP) to achieve roughly 2x faster training times on a T4 GPU.

### Rigorous Evaluation (The Core Philosophy)
We do not rely on single-fold metrics or "vibes". **No model is adopted without a paired statistical test.**
We use paired bootstrapping and McNemar's exact test on the exact same validation samples to ensure that any accuracy or F1 gap is statistically significant and not just variance noise.

```python
from src.analysis import decide
decide(targets, probs_baseline, probs_candidate)   # -> ADOPT / WEAK / REJECT
```

*See [AGENTS.md](AGENTS.md) for our strict evidence and analysis standards.*

## 📂 Repository Structure

```text
OctWave3/
├── AGENTS.md                  # RULES - evidence standards, read this first
├── CLAUDE.md                  # short pointer to AGENTS.md
├── notebooks/
│   ├── 01_train_colab.ipynb   # thin driver - run this in Colab on a T4
│   └── 02_analysis.ipynb      # evidence + decisions (CPU is fine)
├── src/
│   ├── config.py              # every tunable setting (model, data, paths)
│   ├── dataset.py             # data loading + augmentations + CV folds
│   ├── model.py               # timm model factory
│   ├── train.py               # training loop (resumable, saves OOF preds)
│   ├── predict.py             # inference -> submission.csv
│   ├── analysis.py            # statistical tests + standard charts
│   └── utils.py               # seeding, checkpoints, logging
├── data/                      # downloaded here, gitignored
├── outputs/                   # checkpoints, oof/, figures/ - lives in Drive, gitignored
├── results/                   # TRACKED - small evidence pushed back from Colab
│   ├── run_log.jsonl          #   per-epoch metrics
│   └── figures/               #   charts
├── logs/
│   ├── CHANGELOG.md           # what changed in the code, by date
│   ├── EXPERIMENTS.md         # what each run scored
│   └── DECISIONS.md           # what was adopted/rejected, and on what evidence
└── docs/
    ├── SETUP.md               # first-time Colab + GitHub setup
    └── WORKFLOW.md            # the daily loop
```

## 🚀 Quick Start

1. Push this repo to GitHub.
2. Open `notebooks/01_train_colab.ipynb` in Colab (see [docs/SETUP.md](docs/SETUP.md)).
3. Set `REPO_URL` and `COMP` in the notebook, set `cfg.num_classes`, run all cells.

### Golden Rule
**Logic goes in `src/*.py`. The notebook only calls it.**
Edit `.py` files locally → `git push` → re-run the pull cell in Colab. This keeps notebook diffs tiny and avoids the merge conflicts that make `.ipynb` painful in git.
