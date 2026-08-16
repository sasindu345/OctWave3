"""Model factory + the two prediction heads.

The 4 classes are not independent categories - they are two binary questions:

    class 0 = (no Tom, no Jerry)      class 2 = (no Tom, Jerry)
    class 1 = (Tom,    no Jerry)      class 3 = (Tom,    Jerry)

A 4-way softmax throws that structure away: "Tom is present" has to be learned
separately for class 1 (1252 images) and class 3 (219 images). The multilabel head
predicts the two questions instead, so:

    "Tom present"   trains on 1252 + 219 = 1471 images
    "Jerry present" trains on  841 + 219 = 1060 images

Class 3 stops being a rare class needing its own examples and becomes the
conjunction of two well-supported detectors. Both heads emit 4-class probabilities,
so their outputs are directly comparable and can be ensembled together.
"""

import timm
import torch
import torch.nn as nn


def n_outputs(cfg) -> int:
    return 2 if getattr(cfg, "head", "softmax") == "multilabel" else cfg.num_classes


def targets_to_multilabel(targets: torch.Tensor) -> torch.Tensor:
    """(N,) ints 0-3  ->  (N,2) floats [tom_present, jerry_present]."""
    tom = (targets == 1) | (targets == 3)
    jerry = (targets == 2) | (targets == 3)
    return torch.stack([tom, jerry], dim=1).float()


def multilabel_to_probs4(logits: torch.Tensor) -> torch.Tensor:
    """(N,2) logits -> (N,4) class probabilities, assuming the two events are
    conditionally independent given the image. Rows sum to 1 by construction."""
    p = torch.sigmoid(logits)
    t, j = p[:, 0], p[:, 1]
    return torch.stack([(1 - t) * (1 - j), t * (1 - j), (1 - t) * j, t * j], dim=1)


def multilabel_pos_weight(df, device=None):
    """BCE pos_weight = negatives/positives for each of the two questions."""
    n = len(df)
    tom = int(((df.target == 1) | (df.target == 3)).sum())
    jerry = int(((df.target == 2) | (df.target == 3)).sum())
    w = torch.tensor([(n - tom) / max(tom, 1), (n - jerry) / max(jerry, 1)],
                     dtype=torch.float32)
    return w.to(device) if device is not None else w


def build_model(cfg) -> nn.Module:
    return timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=n_outputs(cfg),
        drop_rate=cfg.drop_rate,
    )
