# Task 2 Evidence: Engineering Excellence

## Tests

Command:
```bash
py -3.11 -m pytest -q
```

Output:
```text
.....                                                                    [100%]
5 passed in 1.33s
```

## Training + Artifacts

Command:
```bash
py -3.11 -m src.train --data-path "data/raw/train.csv"
```

Generated files:
- `artifacts/model.joblib`
- `artifacts/metrics.json`
- `artifacts/shap_global.png`
- `artifacts/shap_local.png`

## Interactive Dashboard

Run:
```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard loads `artifacts/model.joblib` and displays:
- the global SHAP PNG (`artifacts/shap_global.png`)
- a local SHAP explanation generated on demand for the entered sample

