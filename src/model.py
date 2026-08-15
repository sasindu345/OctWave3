"""Model factory. timm gives ~1000 pretrained backbones behind one call."""
import timm
import torch.nn as nn


def build_model(cfg) -> nn.Module:
    return timm.create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        drop_rate=cfg.drop_rate,
    )
