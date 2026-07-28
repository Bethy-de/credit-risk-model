from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.constants import DEFAULT_RISK_PROB_COL, MODEL_PATH, SHAP_GLOBAL_PNG, SHAP_LOCAL_PNG
from src.models.persistence import load_model
from src.predict import predict_default_risk
from src.explainability.shap_explain import save_shap_local_waterfall


def _risk_band(default_risk_prob: float) -> str:
    if default_risk_prob < 0.33:
        return "Low risk"
    if default_risk_prob < 0.66:
        return "Medium risk"
    return "High risk"


def _business_action(band: str) -> str:
    return {
        "Low risk": "Automate approval with periodic monitoring.",
        "Medium risk": "Require additional documentation / review.",
        "High risk": "Escalate to manual underwriting / safeguards.",
    }[band]


def main() -> None:
    st.set_page_config(page_title="Credit Risk Model Dashboard", layout="wide")
    st.title("Credit Risk Probability (Production-style Demo)")

    if not MODEL_PATH.exists():
        st.error("Model artifact not found. Run `python -m src.train` first.")
        st.stop()

    pipeline = load_model(MODEL_PATH)

    metrics_path = Path("artifacts/metrics.json")
    metrics: dict | None = None
    if metrics_path.exists():
        import json

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Key metrics")
        if metrics is not None:
            st.metric("ROC AUC", f"{metrics.get('roc_auc', float('nan')):.3f}")
            st.metric("Accuracy", f"{metrics.get('accuracy', float('nan')):.3f}")
        else:
            st.caption("No metrics found yet.")

        st.subheader("Interactive prediction")
        submitted = False
        with st.form("prediction_form"):
            Pclass = st.selectbox("Pclass", [1, 2, 3], index=2)
            Sex = st.selectbox("Sex", ["male", "female"])
            Age = st.number_input("Age", min_value=0.0, max_value=90.0, value=30.0, step=0.5)
            SibSp = st.number_input("SibSp", min_value=0, max_value=10, value=0, step=1)
            Parch = st.number_input("Parch", min_value=0, max_value=10, value=0, step=1)
            Fare = st.number_input("Fare", min_value=0.0, max_value=200.0, value=30.0, step=0.5)
            Embarked = st.selectbox("Embarked", ["S", "C", "Q"], index=0)
            Cabin = st.text_input("Cabin (optional)", value="C85")

            submitted = st.form_submit_button("Score")

        if submitted:
            raw = pd.DataFrame(
                [
                    {
                        "Pclass": int(Pclass),
                        "Sex": str(Sex),
                        "Age": float(Age),
                        "SibSp": int(SibSp),
                        "Parch": int(Parch),
                        "Fare": float(Fare),
                        "Embarked": str(Embarked),
                        "Cabin": str(Cabin) if Cabin.strip() else None,
                    }
                ]
            )
            preds = predict_default_risk(pipeline, raw)
            default_risk_prob = float(preds.loc[0, DEFAULT_RISK_PROB_COL])
            band = _risk_band(default_risk_prob)

            st.success(f"Default risk: {default_risk_prob:.3f} ({band})")
            st.write("Business impact recommendation:")
            st.write(_business_action(band))

    with right:
        st.subheader("Model explainability (SHAP)")
        if SHAP_GLOBAL_PNG.exists():
            st.image(str(SHAP_GLOBAL_PNG), caption="Global feature importance (SHAP summary).")
        else:
            st.caption("Global SHAP artifact missing. Run training to generate it.")

        if submitted:
            tmp_path = SHAP_LOCAL_PNG
            save_shap_local_waterfall(
                pipeline,
                raw_X=raw,
                index=0,
                output_path=tmp_path,
                random_seed=42,
            )
            if tmp_path.exists():
                st.image(str(tmp_path), caption="Local explanation for this prediction (SHAP waterfall).")

            st.caption(
                "Interpretability note: this demo uses SHAP values computed from the fitted preprocessing + XGBoost model."
            )


if __name__ == "__main__":
    main()

