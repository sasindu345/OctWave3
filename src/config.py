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

    # Set once the competition data layout is known.
    # "folder" -> data_dir/train/<class_name>/*.jpg
    # "csv"    -> data_dir/train.csv with columns [image_col, label_col]
    data_mode: str = "folder"
    train_csv: str = "train.csv"
    image_col: str = "image_id"
    label_col: str = "label"
    image_ext: str = ".jpg"

    # --- model ---
    model_name: str = "tf_efficientnet_b0"  # any timm model
    pretrained: bool = True
    num_classes: int = 10  # UPDATE when competition starts
    drop_rate: float = 0.0

    # --- data ---
    img_size: int = 224
    batch_size: int = 64        # T4 (16GB): b0@224 ~64, b3@300 ~24, b4@380 ~12
    num_workers: int = 2        # Colab only gives ~2 usable CPU cores
    n_folds: int = 5
    train_folds: list = field(default_factory=lambda: [0])  # [0,1,2,3,4] for full CV

    # --- training ---
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    amp: bool = True            # mixed precision - roughly 2x faster on T4
    grad_accum: int = 1
    early_stop_patience: int = 5

    def to_dict(self):
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


cfg = Config()
