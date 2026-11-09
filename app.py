"""Streamlit demo: Loan Default Prediction (German Credit dataset).

Run:  streamlit run app.py
Needs: models/xgboost.joblib produced by `python train.py` first.
The app predicts the probability that a loan applicant defaults, flags the
applicant against the tuned decision threshold from metrics.json, and shows
the top SHAP drivers for that individual prediction.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "xgboost.joblib"
METRICS_PATH = BASE / "metrics.json"
DATA_PATH = BASE / "data" / "german.data"

# Human-readable labels for the coded UCI values.
OPTIONS = {
    "checking_account_status": {
        "A11": "< 0 DM", "A12": "0–200 DM", "A13": ">= 200 DM",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all paid back",
        "A31": "all credits at this bank paid back",
        "A32": "existing credits paid back duly",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
        "A43": "radio/television", "A44": "domestic appliances",
        "A45": "repairs", "A46": "education", "A47": "vacation",
        "A48": "retraining", "A49": "business", "A410": "others",
    },
    "savings_account": {
        "A61": "< 100 DM", "A62": "100–500 DM", "A63": "500–1000 DM",
        "A64": ">= 1000 DM", "A65": "unknown / no savings account",
    },
    "employment_since": {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1–4 years",
        "A74": "4–7 years", "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male, divorced/separated", "A92": "female, divorced/separated/married",
        "A93": "male, single", "A94": "male, married/widowed",
        "A95": "female, single",
    },
    "other_debtors": {
        "A101": "none", "A102": "co-applicant", "A103": "guarantor",
    },
    "property": {
        "A121": "real estate", "A122": "building society savings / life insurance",
        "A123": "car or other", "A124": "unknown / no property",
    },
    "other_installment_plans": {
        "A141": "bank", "A142": "stores", "A143": "none",
    },
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {
        "A171": "unemployed / unskilled, non-resident",
        "A172": "unskilled, resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified / officer",
    },
    "telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}

NUMERIC_DEFAULTS = {
    "duration_months": (24, 4, 72),
    "credit_amount": (2000, 250, 20000),
    "installment_rate_pct": (3, 1, 4),
    "residence_since": (3, 1, 4),
    "age_years": (35, 18, 75),
    "existing_credits": (1, 1, 4),
    "maintenance_liable_people": (1, 1, 2),
}


@st.cache_resource
def load_artifacts():
    import json

    pipe = joblib.load(MODEL_PATH)
    metrics = json.loads(METRICS_PATH.read_text())
    return pipe, metrics


def main():
    st.title("Loan Default Prediction — Credit Risk Demo")
    st.write(
        "XGBoost model trained on the **German Credit dataset** (UCI, 1,000 "
        "applicants). Enter an applicant's details to get a default "
        "probability, a risk flag, and the reasons behind it."
    )
    if not MODEL_PATH.exists():
        st.error("Model not found. Run `python train.py` first to train it.")
        st.stop()
    pipe, metrics = load_artifacts()
    threshold = metrics["threshold"]["tuned_threshold"]
    test_auc = metrics["best_model_test_auc"]
    st.caption(
        f"Best model: XGBoost · test ROC-AUC {test_auc:.3f} · "
        f"decision threshold {threshold:.3f} (tuned for recall on defaulters)"
    )

    st.subheader("Applicant details")
    inputs = {}
    col_a, col_b = st.columns(2)
    with col_a:
        for feat in ["checking_account_status", "credit_history", "purpose",
                     "savings_account", "employment_since",
                     "personal_status_sex"]:
            opts = OPTIONS[feat]
            label = st.selectbox(
                feat.replace("_", " ").title(),
                options=list(opts.keys()),
                format_func=lambda k, o=opts: f"{k} — {o[k]}",
            )
            inputs[feat] = label
    with col_b:
        for feat in ["other_debtors", "property", "other_installment_plans",
                     "housing", "job", "telephone", "foreign_worker"]:
            opts = OPTIONS[feat]
            label = st.selectbox(
                feat.replace("_", " ").title(),
                options=list(opts.keys()),
                format_func=lambda k, o=opts: f"{k} — {o[k]}",
            )
            inputs[feat] = label
    col_c, col_d = st.columns(2)
    with col_c:
        for feat in ["duration_months", "credit_amount",
                     "installment_rate_pct", "residence_since"]:
            default, lo, hi = NUMERIC_DEFAULTS[feat]
            inputs[feat] = st.number_input(
                feat.replace("_", " ").title(), min_value=lo, max_value=hi,
                value=default)
    with col_d:
        for feat in ["age_years", "existing_credits",
                     "maintenance_liable_people"]:
            default, lo, hi = NUMERIC_DEFAULTS[feat]
            inputs[feat] = st.number_input(
                feat.replace("_", " ").title(), min_value=lo, max_value=hi,
                value=default)

    if st.button("Predict default risk"):
        row = pd.DataFrame([inputs])
        prob = float(pipe.predict_proba(row)[0, 1])
        flagged = prob >= threshold
        st.subheader("Result")
        st.metric("Probability of default", f"{prob:.1%}")
        if flagged:
            st.error(
                f"⚠️ HIGH RISK — probability {prob:.1%} is at/above the "
                f"decision threshold ({threshold:.2f}). Flag for manual review.")
        else:
            st.success(
                f"✅ LOW RISK — probability {prob:.1%} is below the decision "
                f"threshold ({threshold:.2f}).")
        st.write(
            "Why the threshold matters: it was tuned on the validation set to "
            "maximise the F2-score, so the model deliberately catches more "
            "defaulters at the cost of extra false alarms — a missed defaulter "
            "costs a bank far more than a manual review.")

        # Top SHAP drivers for this applicant.
        st.subheader("What drives this prediction (SHAP)")
        try:
            import shap

            pre = pipe.named_steps["preprocess"]
            clf = pipe.named_steps["clf"]
            names = list(pre.get_feature_names_out())
            x_enc = pre.transform(row)
            if hasattr(x_enc, "toarray"):
                x_enc = x_enc.toarray()
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(x_enc)
            if isinstance(sv, list):
                sv = sv[1]
            contrib = sorted(
                zip(names, sv[0]), key=lambda t: -abs(t[1]))[:5]
            for name, value in contrib:
                direction = "pushes risk UP" if value > 0 else "pushes risk DOWN"
                st.write(f"- `{name}`: SHAP {value:+.3f} — {direction}")
        except Exception as exc:
            st.info(f"SHAP explanation unavailable ({exc}).")
        st.caption(
            "Demo only — trained on 1,000 German Credit applicants. "
            "Not a lending decision.")


if __name__ == "__main__":
    main()
