from __future__ import annotations

from pathlib import Path

# ----------------------------
# Dataset / business semantics
# ----------------------------

LABEL_COL: str = "Survived"

# Project framing: Survived (1) -> "good outcome".
# For a finance-style "risk" view we define default-risk as P(good) inverted.
DEFAULT_RISK_PROB_COL: str = "default_risk_prob"

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2

# ----------------------------
# Feature engineering columns
# ----------------------------

ID_COLS: tuple[str, ...] = ("PassengerId",)
DROP_COLS: tuple[str, ...] = ("Name", "Ticket")
CABIN_COL: str = "Cabin"

NUMERIC_COLS: tuple[str, ...] = ("Pclass", "Age", "SibSp", "Parch", "Fare")
CATEGORICAL_COLS: tuple[str, ...] = ("Sex", "Embarked", "CabinDeck")

DECK_COL: str = "CabinDeck"

# ----------------------------
# Artifacts
# ----------------------------

ARTIFACTS_DIR: Path = Path("artifacts")
MODEL_PATH: Path = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH: Path = ARTIFACTS_DIR / "metrics.json"

SHAP_GLOBAL_PNG: Path = ARTIFACTS_DIR / "shap_global.png"
SHAP_LOCAL_PNG: Path = ARTIFACTS_DIR / "shap_local.png"

