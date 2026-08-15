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
def predict(cfg, ckpt_paths, test_df):
    """Averages softmax probabilities over one or more checkpoints (fold ensemble)."""
    device = get_device()
    ds = ImageDataset(test_df, build_transforms(cfg.img_size, False), has_label=False)
    dl = DataLoader(ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.num_workers)

    probs = np.zeros((len(ds), cfg.num_classes), dtype=np.float32)
    for path in ckpt_paths:
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(path, map_location=device)["model"])
        model.eval()
        out = []
        for images in tqdm(dl, desc=Path(path).stem, leave=False):
            with torch.autocast("cuda", enabled=cfg.amp):
                out.append(model(images.to(device)).softmax(1).float().cpu().numpy())
        probs += np.concatenate(out) / len(ckpt_paths)
    return probs


def make_submission(cfg, probs, test_df, classes, out_name="submission.csv"):
    sub = pd.DataFrame({
        cfg.image_col: [Path(p).stem for p in test_df["path"]],
        cfg.label_col: [classes[i] for i in probs.argmax(1)],
    })
    out_path = Path(cfg.out_dir) / "submissions" / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(sub)} rows)")
    return sub
