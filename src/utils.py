"""Small helpers: seeding, checkpointing, metric tracking."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def save_checkpoint(path: Path, model, optimizer, scaler, epoch, best_score, cfg):
    """Always write to a temp file first so a Colab disconnect mid-write
    cannot corrupt the checkpoint we need to resume from."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "best_score": best_score,
            "cfg": cfg.to_dict(),
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer=None, scaler=None, device="cpu"):
    """Returns (start_epoch, best_score). Safe to call when no checkpoint exists."""
    path = Path(path)
    if not path.exists():
        return 0, -float("inf")
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    if optimizer is not None and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    print(f"resumed from {path} @ epoch {ck['epoch']} (best={ck['best_score']:.4f})")
    return ck["epoch"] + 1, ck["best_score"]


def save_oof(out_dir: Path, exp_name: str, fold: int, probs, targets):
    """Out-of-fold predictions for the best epoch.

    These are the evidence base: every later comparison between two experiments
    is a paired test on these arrays, not on two headline numbers.
    """
    oof_dir = Path(out_dir) / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(oof_dir / f"{exp_name}_f{fold}.npz", probs=probs, targets=targets)


def load_oof(out_dir: Path, exp_name: str, fold: int):
    """Returns (probs, targets) for one fold of one experiment."""
    path = Path(out_dir) / "oof" / f"{exp_name}_f{fold}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no OOF predictions at {path} - train that fold first")
    d = np.load(path)
    return d["probs"], d["targets"]


def append_run_log(out_dir: Path, record: dict):
    """One JSON line per epoch -> outputs/run_log.jsonl. Cheap history you can plot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
    with open(out_dir / "run_log.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
