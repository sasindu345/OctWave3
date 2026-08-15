# Colab Notebook Guide — Training, Generalization & Submissions

This document details the updated notebook structure and training workflow for the **OctWave 3.0 Image Classification Challenge** (Tom & Jerry 4-Class Classification).

---

## 1. Overview of Notebook Upgrades

| Notebook Section | Key Upgrades | Why It Matters |
|---|---|---|
| **Section 6: Configure** | Added `cfg.train_folds = [0, 1, 2, 3, 4]`, module reload, and cache invalidation. | Allows running all 5 folds or single folds seamlessly without kernel crash issues. |
| **Section 7: Train** | Automated multi-fold loop with out-of-fold `mean ± std` summary. | 5-fold CV averages out fold variance and builds a robust 5-model ensemble. |
| **Section 8: Curves & Overfit Check** | Directly calls `overfit_check()` from `src/analysis.py`. | Gives instant diagnostic verdicts (`HEALTHY`, `MILD`, `OVERFITTING`, `UNDERTRAINED`) after training. |
| **Section 9: Predict & Submit** | Enabled Test-Time Augmentation (`tta=True`) and added submission validation checks. | Boosts Macro F1 via variance reduction (+0.01~0.03) and validates file format automatically. |
| **Section 10: Sync** | Pushes `results/run_log.jsonl` and figures back to GitHub using Colab Secret `GH_TOKEN`. | Enables offline tracking and direct analysis on your local machine. |

---

## 2. Preventing Overfitting & Ensuring Generalization

With **2,680 training images** and **4 classes** (imbalance ratio 5.7x), large neural networks easily memorize training noise. Here is how our updated pipeline prevents overfitting:

### A. 5-Fold Cross Validation (`cfg.train_folds = [0, 1, 2, 3, 4]`)
- Stratified folds guarantee identical class proportions across every validation split (each fold contains ~44 samples of the rarest class `3: both`).
- Training 5 separate models on 80/20 splits ensures that every single image is evaluated out-of-fold.
- The 5-checkpoint ensemble averages predictions across all 5 models, dramatically smoothing out decision boundaries and reducing prediction variance.

### B. Automated `overfit_check()`
In Section 8 of `notebooks/01_train_colab.ipynb`, `overfit_check()` tests:
1. **Validation loss turn**: Did validation loss bottom out and rise?
2. **Train/Val gap**: Did the train-loss vs val-loss gap widen by > 0.05?
3. **Metric progression**: Did validation Macro F1 keep improving despite loss rising?
4. **Verdict**:
   - `HEALTHY`: Stable loss and progressing score.
   - `MILD`: Validation loss turned up slightly, but Macro F1 is still climbing (no penalty).
   - `OVERFITTING`: Val loss rising, train-val gap widening, and Macro F1 flat/declining.
   - `UNDERTRAINED`: The best score occurred on the very last epoch.

### C. Test-Time Augmentation (TTA)
- For every test image, `src/predict.py` passes both the original image and its horizontal flip ($180^\circ$ mirror), averaging their softmax probabilities.
- TTA adds **zero trainable parameters** and **zero risk of overfitting**, while providing an empirical boost of +0.01 to +0.02 Macro F1.

---

## 3. Step-by-Step Execution in Google Colab

1. **Open Notebook**: Open `notebooks/01_train_colab.ipynb` in Google Colab with GPU runtime enabled (**Runtime -> Change runtime type -> T4 GPU**).
2. **Execute Setup (Cells 1-6)**:
   - Cell 1-2: Check GPU and install dependencies.
   - Cell 3: Mount Google Drive (keeps checkpoints persistent).
   - Cell 4: Pull latest code from GitHub.
   - Cell 5-5b: Download and inspect competition data.
   - Cell 6: Configure experiment parameters (`cfg.exp_name = 'exp02_5fold_b0'`, `cfg.train_folds = [0, 1, 2, 3, 4]`).
3. **Train All Folds (Cell 7)**:
   - Takes ~20-25 minutes total on T4 GPU for 5 folds (12 epochs each).
   - Automatically saves `*_best.pt` and `*_f*.npz` out-of-fold predictions.
4. **Inspect Curves & Diagnostics (Cell 8)**:
   - Review the loss and Macro F1 curves.
   - Read the output of `overfit_check()`.
5. **Generate Submission with TTA (Cell 9)**:
   - Averages all 5 fold models with TTA enabled.
   - Verifies row count (2,798), column names (`filename`, `appearance`), and valid class ranges ($0 \le \text{appearance} \le 3$).
6. **Submit to Kaggle (Cell 9.2)**:
   - Submits directly to the leaderboard using the Kaggle API.
7. **Sync Back (Cell 10)**:
   - Pushes updated logs and figures to your GitHub repo.
