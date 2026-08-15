"""Inference -> submission.csv. Adjust column names once the comp format is known."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import ImageDataset, build_transforms
from src.model import build_model
from src.utils import get_device


@torch.no_grad()
def predict(cfg, ckpt_paths, test_df, tta=True):
    """Averages softmax probabilities over checkpoints, and optionally over flips.

    Two kinds of averaging, both variance reduction rather than added capacity -
    which is why neither can overfit more than a single model:
      - fold ensemble: average the 5 models trained on different splits
      - TTA: average each image with its horizontal mirror
    """
    device = get_device()
    ds = ImageDataset(test_df, build_transforms(cfg.img_size, False), has_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)

    if not ckpt_paths:
        raise FileNotFoundError("no checkpoints found - train a fold first")

    probs = np.zeros((len(ds), cfg.num_classes), dtype=np.float32)
    for path in ckpt_paths:
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device)["model"])
        model.eval()
        out = []
        for images in tqdm(dl, desc=Path(path).stem, leave=False):
            images = images.to(device)
            with torch.autocast("cuda", enabled=cfg.amp):
                p = model(images).softmax(1)
                if tta:
                    p = (p + model(torch.flip(images, dims=[3])).softmax(1)) / 2
            out.append(p.float().cpu().numpy())
        probs += np.concatenate(out) / len(ckpt_paths)

    print(f"predicted with {len(ckpt_paths)} checkpoint(s), TTA={'on' if tta else 'off'}")
    return probs


def make_submission(cfg, probs, test_df, out_name="submission.csv"):
    """Writes filename,appearance - appearance as a plain int 0-3.

    Labels were never remapped (they are already 0-3), so argmax is the label.
    """
    sub = pd.DataFrame({
        cfg.image_col: test_df["filename"].values,
        cfg.label_col: probs.argmax(1).astype(int),
    })
    out_path = Path(cfg.out_dir) / "submissions" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)

    print(f"wrote {out_path} ({len(sub)} rows)")
    print("predicted class distribution:\n", sub[cfg.label_col].value_counts().sort_index())
    return sub
