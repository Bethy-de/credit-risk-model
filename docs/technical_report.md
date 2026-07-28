# Technical Report: From Prototype to Finance-Ready Risk Scoring

## 1) Executive Summary

This project upgrades a simple machine learning prototype into a more
professional, finance-oriented portfolio piece. The main objective was to
improve reliability, transparency, and communication quality for stakeholders
who care about risk controls and decision traceability.

Final outcomes include:
- modular, typed Python codebase
- automated tests and CI checks
- interactive dashboard for decision support
- SHAP explainability artifacts for model transparency

## 2) Problem Context

In credit risk workflows, model quality alone is not enough. Teams need:
- reproducible model training
- clear validation evidence
- explainable predictions
- stakeholder-facing outputs

Early project state had business framing but lacked production engineering
artifacts (tests, CI, implemented pipeline modules, and clear usage docs).

## 3) Approach

### 3.1 Engineering Refactor

The codebase was restructured to separate concerns:
- `src/features/` for feature engineering and preprocessing
- `src/models/` for training and persistence
- `src/explainability/` for SHAP logic
- `src/api/` for serving interfaces

Python best practices applied:
- type hints on function signatures
- dataclasses for configuration (`DataConfig`, `ModelConfig`, `PathsConfig`)
- named constants to avoid magic numbers
- reusable utility functions for shared transformations

### 3.2 Testing and CI/CD

Added a minimum of five tests covering:
- feature engineering output validity
- preprocessing transformation behavior
- training metric sanity checks
- prediction probability bounds
- SHAP output structure

CI workflow runs lint and tests on push/pull request to prevent regressions.

### 3.3 Dashboard and Explainability

Built a Streamlit app that supports:
- interactive prediction input
- business-oriented risk band output
- recommendation messaging by risk level
- SHAP global importance plot
- SHAP local explanation plot for specific predictions

## 4) Results

Validation metrics from `artifacts/metrics.json`:
- ROC AUC: **0.840**
- Accuracy: **0.804**

Quality evidence:
- `pytest`: **5 passed**
- CI lint/test workflow configured in `.github/workflows/ci.yml`

Explainability evidence:
- `artifacts/shap_global.png`
- `artifacts/shap_local.png`

## 5) Visuals

### Global Feature Importance (SHAP)

![SHAP Global](../artifacts/shap_global.png)

### Local Prediction Explanation (SHAP)

![SHAP Local](../artifacts/shap_local.png)

## 6) Lessons Learned

1. Reliability is a product feature in finance.  
   Strong engineering controls (tests + CI) increase stakeholder trust.

2. Explainability must be integrated, not optional.  
   SHAP outputs helped bridge technical and non-technical communication.

3. Communication quality changes project impact.  
   Better README/report/dashboard framing makes value clearer to recruiters and
   hiring managers.

## 7) Limitations and Next Steps

Current limitations:
- Proxy dataset instead of real credit bureau + transaction features
- No calibration or rejection-inference workflow
- No production deployment infrastructure

Recommended next steps:
- add calibration and threshold optimization by business objective
- monitor drift and fairness over time
- deploy API/dashboard with container orchestration and observability
- add model governance documentation (assumptions, risk controls, changelog)

## 8) How to Reproduce

```bash
pip install -r requirements.txt
py -3.11 -m src.train --data-path "data/raw/train.csv"
py -3.11 -m pytest -q
streamlit run dashboard/streamlit_app.py
```

## 9) Portfolio Positioning Statement

This capstone demonstrates the ability to move from data-science experimentation
to reliable ML product thinking: reproducible pipelines, automated quality
controls, interpretable outputs, and business-ready communication tailored to
finance stakeholders.
