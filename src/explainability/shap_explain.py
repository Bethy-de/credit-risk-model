from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import shap

from ..config import ExplainerConfig, PathsConfig
from ..features.feature_engineering import clean_raw_features


def _as_numpy_2d(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def compute_shap(
    pipeline,
    raw_X,
    *,
    max_background: int,
    max_sample: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (shap_values, transformed_sample_X, feature_names).
    """

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Feature engineering must match training.
    X_clean = clean_raw_features(raw_X)
    # Deterministic subsampling for speed.
    X_sample = X_clean.sample(n=min(max_sample, len(X_clean)), random_state=random_seed)
    X_background = X_clean.sample(
        n=min(max_background, len(X_clean)), random_state=random_seed
    )

    X_trans = preprocessor.transform(X_sample)
    X_background_trans = preprocessor.transform(X_background)

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    if hasattr(X_background_trans, "toarray"):
        X_background_trans = X_background_trans.toarray()

    feature_names = list(preprocessor.get_feature_names_out())

    # LinearExplainer is stable for logistic regression + one-hot encoded features.
    explainer = shap.LinearExplainer(model, X_background_trans)
    shap_values = explainer.shap_values(X_trans)

    # Binary classification can be either array or list-of-arrays depending on SHAP version.
    if isinstance(shap_values, list):
        shap_values_arr = np.asarray(shap_values[-1])
    else:
        shap_values_arr = np.asarray(shap_values)

    return shap_values_arr, _as_numpy_2d(X_trans), feature_names


def save_shap_global_plot(
    shap_values: np.ndarray,
    transformed_X: np.ndarray,
    feature_names: list[str],
    *,
    output_path: Path,
    max_display: int = 15,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.summary_plot(
        shap_values,
        transformed_X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_shap_local_waterfall(
    pipeline,
    raw_X,
    *,
    index: int,
    output_path: Path,
    feature_max_display: int = 10,
    random_seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_clean = clean_raw_features(raw_X)
    # Build a local sample deterministically.
    X_sample = X_clean.sample(
        n=min(max(index + 1, 1), len(X_clean)), random_state=random_seed
    )
    # Ensure index is valid in the sample.
    index = int(min(index, len(X_sample) - 1))

    X_trans = preprocessor.transform(X_sample)
    X_trans_background = preprocessor.transform(X_clean.sample(n=1, random_state=random_seed))
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    if hasattr(X_trans_background, "toarray"):
        X_trans_background = X_trans_background.toarray()
    feature_names = list(preprocessor.get_feature_names_out())

    explainer = shap.LinearExplainer(model, X_trans_background)
    shap_values = explainer.shap_values(X_trans)
    if isinstance(shap_values, list):
        shap_values_arr = np.asarray(shap_values[-1])
    else:
        shap_values_arr = np.asarray(shap_values)

    # SHAP expected value differs across versions; handle both scalar/list.
    expected_value = explainer.expected_value
    base_value_arr = np.asarray(expected_value).reshape(-1)
    base_value = float(base_value_arr[-1])

    # Build a SHAP Explanation object for waterfall plotting.
    values_1 = shap_values_arr[index]
    x_1 = X_trans[index]
    exp = shap.Explanation(
        values=values_1,
        base_values=base_value,
        data=x_1,
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(exp, max_display=feature_max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_shap_artifacts(
    pipeline,
    raw_X_valid,
    *,
    data_config: Any,
    explainer_config: ExplainerConfig = ExplainerConfig(),
    paths_config: PathsConfig = PathsConfig(),
    random_seed: int,
) -> Dict[str, Any]:
    """
    Generate SHAP PNGs for portfolio evidence.
    """

    shap_values, X_trans, feature_names = compute_shap(
        pipeline,
        raw_X_valid,
        max_background=explainer_config.shap_max_background,
        max_sample=explainer_config.shap_max_sample,
        random_seed=random_seed,
    )

    save_shap_global_plot(
        shap_values=shap_values,
        transformed_X=X_trans,
        feature_names=feature_names,
        output_path=paths_config.shap_global_png,
    )

    save_shap_local_waterfall(
        pipeline,
        raw_X_valid,
        index=0,
        output_path=paths_config.shap_local_png,
        random_seed=random_seed,
    )

    return {
        "paths": asdict(paths_config),
        "feature_count_transformed": len(feature_names),
    }

