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


def build_dataframe(cfg) -> pd.DataFrame:
    """Returns a df with columns [path, target, fold] plus cfg.class_names side effect."""
    data_dir = Path(cfg.data_dir)

    if cfg.data_mode == "folder":
        train_root = data_dir / "train"
        classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        rows = [
            {"path": p, "target": cls_to_idx[p.parent.name]}
            for p in train_root.rglob(f"*{cfg.image_ext}")
        ]
        df = pd.DataFrame(rows)
    else:
        raw = pd.read_csv(data_dir / cfg.train_csv)
        classes = sorted(raw[cfg.label_col].unique().tolist())
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        df = pd.DataFrame({
            "path": [
                data_dir / "train" / f"{x}{cfg.image_ext}" for x in raw[cfg.image_col]
            ],
            "target": raw[cfg.label_col].map(cls_to_idx),
        })

    if df.empty:
        raise RuntimeError(f"no images found under {data_dir} (mode={cfg.data_mode})")

    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df["target"])):
        df.loc[val_idx, "fold"] = fold

    df.attrs["classes"] = classes
    return df


def build_loaders(cfg, df: pd.DataFrame, fold: int):
    tr = df[df.fold != fold].reset_index(drop=True)
    va = df[df.fold == fold].reset_index(drop=True)

    train_ds = ImageDataset(tr, build_transforms(cfg.img_size, True))
    valid_ds = ImageDataset(va, build_transforms(cfg.img_size, False))

    common = dict(num_workers=cfg.num_workers, pin_memory=True, persistent_workers=cfg.num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **common)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size * 2, shuffle=False, **common)
    return train_dl, valid_dl
