"""Dataset + augmentations.

Supports two common Kaggle layouts, switched by cfg.data_mode:
  "folder" -> data_dir/train/<class_name>/*.jpg
  "csv"    -> data_dir/train.csv listing image ids and labels
"""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int, train: bool):
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.CoarseDropout(p=0.3),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ToTensorV2(),
    ])


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, has_label=True):
        self.paths = df["path"].tolist()
        self.labels = df["target"].tolist() if has_label else None
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]))
        if img is None:
            raise FileNotFoundError(self.paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.labels is None:
            return img
        return img, self.labels[i]


_IMG_ROOT_CACHE = {}


def _images_root(cfg) -> Path:
    """Find the directory holding the images.

    Not hardcoded because images.zip nests unpredictably - on this competition it
    lands in data/images/images/, but a re-unzip can flatten it to data/images/.
    Whichever directory holds the most .jpg files is the right one.
    """
    data_dir = Path(cfg.data_dir)
    if data_dir in _IMG_ROOT_CACHE:
        return _IMG_ROOT_CACHE[data_dir]

    counts = {}
    for p in data_dir.rglob("*.jpg"):
        counts[p.parent] = counts.get(p.parent, 0) + 1
    if not counts:
        raise FileNotFoundError(f"no .jpg files anywhere under {data_dir} - did images.zip unzip?")

    root = max(counts, key=counts.get)
    print(f"images root: {root} ({counts[root]} files)")
    _IMG_ROOT_CACHE[data_dir] = root
    return root


def build_dataframe(cfg) -> pd.DataFrame:
    """Training rows from train.csv, as columns [path, target, fold].

    Labels are already integers 0-3, so they are used directly as class indices -
    no mapping, which keeps predictions aligned with the submission format.
    """
    data_dir = Path(cfg.data_dir)
    img_root = _images_root(cfg)
    raw = pd.read_csv(data_dir / cfg.train_csv)

    missing = [c for c in (cfg.image_col, cfg.label_col) if c not in raw.columns]
    if missing:
        raise KeyError(f"{cfg.train_csv} is missing {missing}; it has {list(raw.columns)}")

    df = pd.DataFrame({
        "path": [img_root / f for f in raw[cfg.image_col]],
        "target": raw[cfg.label_col].astype(int),
    })

    absent = [p for p in df.path[:50] if not p.exists()]
    if absent:
        raise FileNotFoundError(f"listed in {cfg.train_csv} but not on disk: {absent[:3]}")

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["target"])):
        df.loc[val_idx, "fold"] = fold

    df.attrs["classes"] = sorted(df.target.unique().tolist())
    return df


def build_test_dataframe(cfg) -> pd.DataFrame:
    """Test rows from test.csv, as columns [path, filename] - no labels."""
    data_dir = Path(cfg.data_dir)
    img_root = _images_root(cfg)
    raw = pd.read_csv(data_dir / cfg.test_csv)
    return pd.DataFrame({
        "path": [img_root / f for f in raw[cfg.image_col]],
        "filename": raw[cfg.image_col],
    })


def class_weights(df: pd.DataFrame, num_classes: int):
    """Inverse-frequency weights, normalised to mean 1.

    Needed because the metric is macro F1 on a severely imbalanced set: unweighted
    training lets the model win on the majority class while scoring ~0 F1 on the
    rare ones, which macro-averaging punishes hard.
    """
    counts = np.bincount(df["target"], minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0            # never divide by zero on an absent class
    w = counts.sum() / (num_classes * counts)
    return w / w.mean()


def build_loaders(cfg, df: pd.DataFrame, fold: int):
    tr = df[df.fold != fold].reset_index(drop=True)
    va = df[df.fold == fold].reset_index(drop=True)

    train_ds = ImageDataset(tr, build_transforms(cfg.img_size, True))
    valid_ds = ImageDataset(va, build_transforms(cfg.img_size, False))

    common = dict(num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **common)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size * 2, shuffle=False, **common)
    return train_dl, valid_dl
