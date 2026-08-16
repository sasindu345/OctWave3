"""All tunable settings in one place. Edit this, not the training code."""

from dataclasses import dataclass, asdict, field
from pathlib import Path


@dataclass
class Config:
    # --- experiment identity (used for checkpoint/log filenames) ---
    exp_name: str = "exp01_baseline"
    seed: int = 42

    # --- paths ---
    # On Colab these get overridden to point at Drive. See notebook cell 3.
    data_dir: Path = Path("data")
    out_dir: Path = Path("outputs")

    # oct-wave-3-0: every image (train AND test) sits in one flat images/ folder;
    # train.csv and test.csv say which is which. Filenames already carry ".jpg".
    images_dir: str = "images"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    image_col: str = "filename"
    label_col: str = "appearance"

    # --- model ---
    # head: "softmax"    -> 4-way CrossEntropy (the champion)
    #       "multilabel" -> 2 sigmoids [Tom present, Jerry present], mapped back to
    #                       4 classes. Pools class 3 with classes 1 and 2 during
    #                       training instead of treating it as its own rare class.
    head: str = "softmax"
    model_name: str = "tf_efficientnet_b0"  # any timm model
    pretrained: bool = True
    num_classes: int = 4  # 0 neither, 1 Tom, 2 Jerry, 3 both
    drop_rate: float = 0.3  # only 2680 training images - regularise hard

    # --- data ---
    img_size: int = 224
    batch_size: int = 32  # T4 (16GB): b0@224 ~64, b3@300 ~24, b4@380 ~12
    num_workers: int = 2  # Colab only gives ~2 usable CPU cores
    n_folds: int = 5
    train_folds: list = field(default_factory=lambda: [0])  # [0,1,2,3,4] for full CV

    # --- training ---
    # The competition metric is MACRO F1 on a severely imbalanced training set.
    # Both settings below exist because of that, and matter more than the backbone.
    metric: str = "macro_f1"  # what early stopping and "best" are judged on
    class_weights: bool = True  # inverse-frequency weights so rare classes count
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05  # train labels are noisy; test labels are clean
    amp: bool = True  # mixed precision - roughly 2x faster on T4
    grad_accum: int = 1
    early_stop_patience: int = 5

    def to_dict(self):
        return {
            k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()
        }


cfg = Config()
