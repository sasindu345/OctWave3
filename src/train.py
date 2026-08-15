"""Training loop with resume-after-disconnect support.

Run from the notebook:   from src.train import run_fold; run_fold(cfg, fold=0)
Or from a terminal:      python -m src.train --epochs 10 --model_name resnet50
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

from src.config import cfg as default_cfg
from src.dataset import build_dataframe, build_loaders
from src.model import build_model
from src.utils import (
    AverageMeter, append_run_log, get_device, load_checkpoint,
    save_checkpoint, save_oof, seed_everything,
)


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, cfg):
    model.train()
    losses = AverageMeter()
    pbar = tqdm(loader, desc="train", leave=False)
    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=cfg.amp):
            loss = criterion(model(images), targets) / cfg.grad_accum

        scaler.scale(loss).backward()
        if (step + 1) % cfg.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        losses.update(loss.item() * cfg.grad_accum, images.size(0))
        pbar.set_postfix(loss=f"{losses.avg:.4f}")

    return losses.avg


@torch.no_grad()
def validate(model, loader, criterion, device, cfg):
    """Returns (loss, acc, macro_f1, probs, targets).

    The probs/targets are the raw material for every statistical claim later on
    (bootstrap CIs, McNemar, confusion matrices) - see src/analysis.py. Without
    them saved you can only compare single numbers, which is how people fool
    themselves into chasing noise.
    """
    model.eval()
    losses = AverageMeter()
    probs, gts = [], []

    for images, targets in tqdm(loader, desc="valid", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=cfg.amp):
            logits = model(images)
            loss = criterion(logits, targets)
        losses.update(loss.item(), images.size(0))
        probs.append(logits.softmax(1).float().cpu().numpy())
        gts.append(targets.cpu().numpy())

    probs, gts = np.concatenate(probs), np.concatenate(gts)
    preds = probs.argmax(1)
    return (losses.avg, accuracy_score(gts, preds),
            f1_score(gts, preds, average="macro"), probs, gts)


def run_fold(cfg, fold: int = 0):
    seed_everything(cfg.seed + fold)
    device = get_device()
    print(f"device={device} | {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    df = build_dataframe(cfg)
    print(f"{len(df)} images | {cfg.num_classes} classes | fold {fold}")
    train_dl, valid_dl = build_loaders(cfg, df, fold)

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.epochs,
        steps_per_epoch=max(len(train_dl) // cfg.grad_accum, 1),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    ckpt_dir = Path(cfg.out_dir) / "checkpoints"
    last_path = ckpt_dir / f"{cfg.exp_name}_f{fold}_last.pt"
    best_path = ckpt_dir / f"{cfg.exp_name}_f{fold}_best.pt"

    # This is what makes a Colab disconnect cost one epoch instead of the whole run.
    start_epoch, best_score = load_checkpoint(last_path, model, optimizer, scaler, device)
    bad_epochs = 0

    for epoch in range(start_epoch, cfg.epochs):
        tr_loss = train_one_epoch(model, train_dl, criterion, optimizer, scaler, scheduler, device, cfg)
        va_loss, acc, f1, probs, gts = validate(model, valid_dl, criterion, device, cfg)
        print(f"epoch {epoch+1}/{cfg.epochs} | train {tr_loss:.4f} | val {va_loss:.4f} | acc {acc:.4f} | f1 {f1:.4f}")

        append_run_log(cfg.out_dir, {
            "exp": cfg.exp_name, "fold": fold, "epoch": epoch + 1,
            "train_loss": round(tr_loss, 5), "val_loss": round(va_loss, 5),
            "acc": round(acc, 5), "f1": round(f1, 5),
            "gap": round(tr_loss - va_loss, 5),  # negative & widening = overfitting
        })

        save_checkpoint(last_path, model, optimizer, scaler, epoch, best_score, cfg)
        if acc > best_score:
            best_score, bad_epochs = acc, 0
            save_checkpoint(best_path, model, optimizer, scaler, epoch, best_score, cfg)
            save_oof(cfg.out_dir, cfg.exp_name, fold, probs, gts)
            print(f"  -> new best {best_score:.4f}")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print("early stopping")
                break

    print(f"fold {fold} done | best acc {best_score:.4f} | {best_path}")
    return best_score


def main():
    p = argparse.ArgumentParser()
    for key, val in default_cfg.to_dict().items():
        if isinstance(val, (str, int, float)) and not isinstance(val, bool):
            p.add_argument(f"--{key}", type=type(val), default=None)
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()

    cfg = default_cfg
    for key, val in vars(args).items():
        if val is not None and hasattr(cfg, key):
            setattr(cfg, key, val)
    run_fold(cfg, args.fold)


if __name__ == "__main__":
    main()
