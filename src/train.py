from __future__ import annotations

import argparse
from pathlib import Path

from .config import DataConfig, ExplainerConfig, ModelConfig, PathsConfig
from .constants import ARTIFACTS_DIR
from .data_processing import load_training_data, prepare_features, split_train_test
from .explainability.shap_explain import generate_shap_artifacts
from .models.persistence import save_metrics, save_model
from .models.training import build_pipeline, train_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a credit risk model.")
    parser.add_argument("--data-path", type=Path, default=Path("data/raw/train.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    args = parser.parse_args(argv)

    data_config = DataConfig(train_csv_path=args.data_path)
    paths_config = PathsConfig(artifacts_dir=args.artifacts_dir)

    model_config = ModelConfig()
    explainer_config = ExplainerConfig()

    X, y = load_training_data(data_config.train_csv_path, label_col=data_config.label_col)
    # Feature engineering to keep preprocessing consistent.
    X = prepare_features(X)

    X_train, X_valid, y_train, y_valid = split_train_test(
        X, y, test_size=data_config.test_size, random_seed=data_config.random_seed
    )

    pipeline = build_pipeline(model_config)
    pipeline, metrics = train_model(pipeline, X_train, y_train, X_valid, y_valid)

    save_model(pipeline, paths_config.model_path)
    save_metrics(metrics, paths_config.metrics_path)

    # Evidence for finance stakeholders.
    generate_shap_artifacts(
        pipeline,
        raw_X_valid=X_valid,
        data_config=data_config,
        explainer_config=explainer_config,
        paths_config=paths_config,
        random_seed=model_config.random_seed,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

