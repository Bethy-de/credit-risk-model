# Task 1: Project Selection and Gap Analysis

## Selected Project
**Project:** Credit Risk Probability Model for Alternative Data  
**Why this project:** It aligns directly with finance sector priorities (risk reduction, model transparency, and regulatory explainability) and can be upgraded into a production-style ML system with clear business value.

## Current State Snapshot
- The repository has a clear domain focus and initial structure (`src`, `tests`, `api`, `.github/workflows`), but most implementation files are empty placeholders.
- A small EDA notebook exists and a raw training dataset is present.
- CI/CD, Docker, tests, API, and core training/prediction pipelines are not yet implemented.

## Gap Analysis Checklist

| Category | Question | Status |
|---|---|---|
| Code Quality | Is the code modular and well-organized? | Partial |
| Code Quality | Are there type hints on functions? | No |
| Code Quality | Is there a clear project structure? | Partial |
| Testing | Are there unit tests for core functions? | No |
| Testing | Do tests run automatically on push? | No |
| Documentation | Is the README comprehensive? | Partial |
| Documentation | Are there docstrings on functions? | No |
| Reproducibility | Can someone else run this project? | Partial |
| Reproducibility | Are dependencies in `requirements.txt`? | Yes (needs pinning/cleanup) |
| Visualization | Is there an interactive way to explore results? | No |
| Business Impact | Is the problem clearly articulated? | Yes |
| Business Impact | Are success metrics defined? | No |

## Prioritized Improvement Plan (3-5 items)

### 1) Build a production-grade training and inference pipeline
**Estimate:** 8-10 hours  
**Scope:**
- Implement modular pipeline files in `src` for data processing, training, evaluation, and prediction.
- Add strict type hints, dataclasses/config objects, and input/output contracts.
- Persist trained model and preprocessing artifacts.

**Why high impact for finance:** Establishes a reliable, auditable foundation and reduces operational/model risk from ad hoc notebook workflows.

### 2) Add robust testing (unit + integration) with `pytest`
**Estimate:** 6-8 hours  
**Scope:**
- Unit tests for feature engineering, validation, and prediction interfaces.
- Integration test for end-to-end train-to-predict flow on a small fixture dataset.
- Coverage target for critical business logic.

**Why high impact for finance:** Demonstrates correctness and reliability, which are central for risk-sensitive environments.

### 3) Implement CI quality gates with GitHub Actions
**Estimate:** 3-4 hours  
**Scope:**
- Complete `.github/workflows/ci.yml` to run linting + tests on push/PR.
- Add fail-fast checks for style and test failures.
- Optional: add coverage report artifact.

**Why high impact for finance:** Proves disciplined engineering and prevents regressions before deployment.

### 4) Add model explainability and risk reporting
**Estimate:** 5-7 hours  
**Scope:**
- Add SHAP-based global and local explanations.
- Generate risk-segment summary outputs (e.g., high/medium/low risk distribution).
- Save explainability artifacts for reproducible reporting.

**Why high impact for finance:** Builds stakeholder trust and supports transparent decision-making under governance requirements.

### 5) Deliver stakeholder-facing interface + documentation
**Estimate:** 6-8 hours  
**Scope:**
- Build a lightweight Streamlit dashboard for score simulation and explanation display.
- Expand `README.md` with setup, run, test, assumptions, and business impact section.
- Add clear success metrics (e.g., AUC, KS, calibration, inference latency).

**Why high impact for finance:** Makes technical outcomes understandable to non-technical decision-makers and recruiters.

## Suggested Success Criteria (Project Score/Outcome)
- **Engineering reliability:** CI green; reproducible setup; tests passing.
- **Model quality:** Defined and reported metrics (AUC/ROC + calibration + stability checks).
- **Transparency:** Explainability visuals available for individual predictions and portfolio-level behavior.
- **Business clarity:** README and dashboard clearly connect model outputs to lending risk decisions.

## Proposed Execution Order
1. Production pipeline implementation
2. Testing
3. CI/CD automation
4. Explainability
5. Dashboard + polished documentation

This ordering de-risks delivery by establishing correctness first, then presentation.
