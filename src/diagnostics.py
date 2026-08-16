"""Two cheap checks that run before any retraining.

CHECK 1 - LABEL NOISE
    The competition states training labels are noisy and test labels are clean.
    Every model so far has been faithfully learning those mistakes. This finds the
    training images where the model is confident the label is wrong.

CHECK 2 - NEAR-DUPLICATE FRAMES
    Images are video frames sampled at 1 FPS, and train/test were split randomly
    from the same episodes. Frames one second apart look nearly identical, so many
    test frames may have a near-twin in train - with a known label. If so, matching
    beats predicting.

Both checks measure themselves on validation data BEFORE anything is applied to
test, and both report a verdict rather than a number to eyeball.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import ImageDataset, build_transforms
from src.model import build_model
from src.utils import get_device, load_oof


# --------------------------------------------------------------------------
# CHECK 1: which training labels look wrong?
# --------------------------------------------------------------------------

def label_noise_report(out_dir, exp_name, folds, df, thresholds=(0.5, 0.7, 0.9, 0.95, 0.99)):
    """Confident-learning style scan of the OOF predictions.

    A sample is 'suspect' when the model predicts a different class AND is more
    confident in that other class than `t`. Because the predictions are
    out-of-fold, the model never saw the sample during training - so a confident
    disagreement is evidence about the LABEL, not memorisation.

    Returns the per-threshold table plus a per-sample frame for the largest threshold.
    """
    probs = np.concatenate([load_oof(out_dir, exp_name, f)[0] for f in folds])
    targets = np.concatenate([load_oof(out_dir, exp_name, f)[1] for f in folds])
    pred = probs.argmax(1)
    conf = probs.max(1)
    n = len(targets)

    rows = []
    for t in thresholds:
        m = (pred != targets) & (conf >= t)
        rows.append({"threshold": t, "suspect": int(m.sum()),
                     "pct_of_train": round(100 * m.mean(), 2)})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    hi = max(thresholds)
    m = (pred != targets) & (conf >= hi)
    print(f"\nat confidence >= {hi}: {m.sum()} samples ({100*m.mean():.2f}%) look mislabeled")

    if m.sum() > 0:
        print("\nwhich labels are most often disputed (true -> model says):")
        ct = pd.crosstab(pd.Series(targets[m], name="labelled"),
                         pd.Series(pred[m], name="model_says"))
        print(ct.to_string())

    pct = 100 * ((pred != targets) & (conf >= 0.9)).mean()
    if pct >= 8:
        verdict = "ACT"
        why = f"{pct:.1f}% of training labels are confidently disputed - cleaning is worth a retrain"
    elif pct >= 3:
        verdict = "MARGINAL"
        why = f"{pct:.1f}% disputed - cleaning may give a small gain"
    else:
        verdict = "SKIP"
        why = f"only {pct:.1f}% disputed - labels look fine, do not spend a retrain on this"
    print(f"\nVERDICT: {verdict} - {why}")

    suspects = pd.DataFrame({
        "idx": np.arange(n), "label": targets, "model_pred": pred, "confidence": conf,
    })
    suspects["suspect"] = (pred != targets) & (conf >= 0.9)
    return {"table": table, "verdict": verdict, "why": why,
            "pct_disputed": float(pct), "suspects": suspects}


# --------------------------------------------------------------------------
# CHECK 2: do near-duplicate frames exist across the split?
# --------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(cfg, ckpt_path, frame_df, batch_size=None):
    """Penultimate-layer features for every image in frame_df (needs a 'path' column).

    Nothing is trained here.

    `ckpt_path=None` uses plain ImageNet weights, which is the DEFAULT for the
    duplicate check on purpose: a fine-tuned checkpoint has seen most of these
    images and embeds them distinctively, which would inflate the similarity of
    exactly the pairs we are trying to measure. Generic features are unbiased and
    perfectly adequate for spotting near-identical frames.
    """
    device = get_device()
    model = build_model(cfg).to(device)
    if ckpt_path is not None:
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        print(f"embedding with fine-tuned weights: {Path(ckpt_path).name}")
    else:
        print("embedding with plain ImageNet weights (unbiased for duplicate search)")
    model.reset_classifier(0)          # timm: drop the head, forward() -> pooled features
    model.eval()

    ds = ImageDataset(frame_df, build_transforms(cfg.img_size, False), has_label=False)
    dl = DataLoader(ds, batch_size=batch_size or cfg.batch_size * 2,
                    shuffle=False, num_workers=cfg.num_workers)

    out = []
    for images in tqdm(dl, desc="embedding", leave=False):
        with torch.autocast("cuda", enabled=cfg.amp):
            f = model(images.to(device))
        out.append(f.float().cpu().numpy())
    emb = np.concatenate(out)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9   # cosine-ready
    print(f"embeddings: {emb.shape}")
    return emb


def _topk_neighbours(query, bank, k=1, chunk=512):
    """Cosine similarity in chunks so 5478x5478 never lands in memory at once."""
    sims, idxs = [], []
    for i in range(0, len(query), chunk):
        s = query[i:i + chunk] @ bank.T
        top = np.argpartition(-s, kth=min(k, s.shape[1] - 1), axis=1)[:, :k]
        row = np.take_along_axis(s, top, axis=1)
        order = np.argsort(-row, axis=1)
        idxs.append(np.take_along_axis(top, order, axis=1))
        sims.append(np.take_along_axis(row, order, axis=1))
    return np.concatenate(sims), np.concatenate(idxs)


def duplicate_report(emb, df, folds, bins=(0.80, 0.90, 0.95, 0.97, 0.99)):
    """For each validation image, find its nearest TRAINING image and ask whether
    the labels agree - bucketed by similarity.

    The question this answers: above what similarity is a neighbour's label a
    better guess than the model's prediction? If agreement never gets high, the
    near-duplicate hypothesis is dead and we stop.
    """
    y = df["target"].values
    rows = []
    for f in folds:
        va = np.where(df.fold.values == f)[0]
        tr = np.where(df.fold.values != f)[0]
        sims, idxs = _topk_neighbours(emb[va], emb[tr], k=1)
        rows.append(pd.DataFrame({
            "fold": f, "val_idx": va, "sim": sims[:, 0],
            "val_label": y[va], "nn_label": y[tr[idxs[:, 0]]],
        }))
    nn = pd.concat(rows, ignore_index=True)
    nn["agree"] = nn.val_label == nn.nn_label

    print(f"nearest-training-neighbour similarity: "
          f"min {nn.sim.min():.3f} | median {nn.sim.median():.3f} | max {nn.sim.max():.3f}\n")

    out = []
    for b in bins:
        m = nn.sim >= b
        out.append({"sim>=": b, "n_images": int(m.sum()),
                    "pct_of_val": round(100 * m.mean(), 1),
                    "label_agreement": round(float(nn.agree[m].mean()), 4) if m.sum() else np.nan})
    table = pd.DataFrame(out)
    print(table.to_string(index=False))

    best = table.dropna()
    hits = best[(best.label_agreement >= 0.90) & (best.pct_of_val >= 5)]
    if len(hits):
        r = hits.iloc[0]
        verdict = "ACT"
        why = (f"{r.pct_of_val}% of validation images have a training twin at "
               f"sim>={r['sim>=']} with {100*r.label_agreement:.1f}% label agreement")
    else:
        verdict = "SKIP"
        why = "no similarity band gives both high agreement and useful coverage"
    print(f"\nVERDICT: {verdict} - {why}")
    print("READ: agreement must clearly beat the model's own accuracy (~0.78) to be worth using.")
    return {"table": table, "nn": nn, "verdict": verdict, "why": why}


def apply_neighbour_labels(probs, test_emb, train_emb, train_labels, threshold, boost=0.9):
    """Override model probabilities where a test image has a confident training twin.

    Not a hard label swap - it adds mass to the neighbour's class so a very
    confident model can still disagree. Returns (new_probs, n_overridden).
    """
    sims, idxs = _topk_neighbours(test_emb, train_emb, k=1)
    sim, nn_lab = sims[:, 0], train_labels[idxs[:, 0]]
    out = probs.copy()
    m = sim >= threshold
    out[m] = (1 - boost) * out[m]
    out[m, nn_lab[m]] += boost
    out /= out.sum(1, keepdims=True)
    print(f"overrode {int(m.sum())} / {len(out)} test images ({100*m.mean():.1f}%) "
          f"at sim >= {threshold}")
    return out, int(m.sum())
