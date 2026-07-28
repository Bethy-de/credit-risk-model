from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.preprocessing import build_preprocessor


def test_build_preprocessor_fit_transform(prepared_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X_prepared, _ = prepared_data
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X_prepared)
    assert hasattr(X_trans, "shape")
    assert len(X_trans.shape) == 2
    # Should produce a non-empty feature matrix.
    assert X_trans.shape[0] > 0
    assert X_trans.shape[1] > 0
    # ColumnTransformer typically returns a sparse matrix for OneHotEncoder.
    if hasattr(X_trans, "toarray"):
        arr = X_trans.toarray()
    else:
        arr = np.asarray(X_trans)
    assert isinstance(arr, np.ndarray)

