from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import LABEL_COL, RANDOM_SEED, TEST_SIZE


@dataclass(frozen=True)
class DataConfig:
    train_csv_path: Path = Path("data/raw/train.csv")
    label_col: str = LABEL_COL
    random_seed: int = RANDOM_SEED
    test_size: float = TEST_SIZE


@dataclass(frozen=True)
class ModelConfig:
    """
    Keep training deterministic and fast for CI.
    """

    model_type: str = "logistic"
    random_seed: int = RANDOM_SEED
    logistic_params: dict[str, Any] = field(
        default_factory=lambda: {
            "solver": "liblinear",
            "max_iter": 400,
            "C": 1.0,
        }
    )


@dataclass(frozen=True)
class ExplainerConfig:
    shap_max_background: int = 50
    shap_max_sample: int = 40


@dataclass
class PathsConfig:
    artifacts_dir: Path = Path("artifacts")
    model_path: Path = field(init=False)
    metrics_path: Path = field(init=False)
    shap_global_png: Path = field(init=False)
    shap_local_png: Path = field(init=False)

    def __post_init__(self) -> None:
        self.model_path = self.artifacts_dir / "model.joblib"
        self.metrics_path = self.artifacts_dir / "metrics.json"
        self.shap_global_png = self.artifacts_dir / "shap_global.png"
        self.shap_local_png = self.artifacts_dir / "shap_local.png"

