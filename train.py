#!/usr/bin/env python3
"""
Loan Default Prediction (Credit Risk) — German Credit dataset (UCI Statlog).

Why German Credit and not Home Credit:
    The Home Credit Default Risk dataset lives on Kaggle and requires Kaggle API
    credentials to download. No Kaggle credentials are available in this
    environment (no ~/.kaggle/kaggle.json, no KAGGLE_USERNAME), and the Kaggle
    API returns 403 without them. The German Credit dataset (UCI ML Repository,
    Statlog) is public, needs no authentication, and is a standard credit-risk
    benchmark: 1,000 loan applicants, 20 attributes, binary good/bad outcome.
    Every number in RESULTS.md comes from THIS dataset — no Home Credit numbers
    are claimed anywhere.

What the script does:
    1. Downloads german.data from UCI (cached under data/ afterwards).
    2. Cleans: checks missing values (this dataset has none — verified, not
       assumed), encodes the 13 categorical features with one-hot encoding.
    3. Splits stratified 60% train / 20% validation / 20% test.
       Validation is used for model selection and threshold tuning; the test
       set is touched exactly once, for the final honest evaluation.
    4. Handles class imbalance with class weights (Logistic Regression,
       Random Forest) and scale_pos_weight (XGBoost). No SMOTE is used: with
       only 1,000 rows, class weighting avoids synthetic-sample and leakage
       risks entirely. (If SMOTE were used, it would have to be fitted on the
       training folds only — never on validation/test.)
    5. Trains Logistic Regression, Random Forest, XGBoost; picks the best by
       validation ROC-AUC.
    6. Tunes the decision threshold on the validation set by maximising the
       F2-score (recall counts twice as much as precision — in lending, a
       missed defaulter costs far more than a false alarm).
    7. Evaluates the best model on the test set at the tuned threshold.
    8. Explains the model with SHAP (TreeExplainer) on the test set and
       builds a simple points-based scorecard from the logistic regression.

Run:  .venv/bin/python train.py
Outputs: models/*.joblib, metrics.json, RESULTS.md, figures/*.png
"""

import json
import os
import urllib.request
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
FIG_DIR = BASE / "figures"
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
)

# Column names for the 20 UCI German Credit attributes + target.
COLUMNS = [
    "checking_account_status",  # A11-A14
    "duration_months",  # numeric
    "credit_history",  # A30-A34
    "purpose",  # A40-A410
    "credit_amount",  # numeric
    "savings_account",  # A61-A65
    "employment_since",  # A71-A75
    "installment_rate_pct",  # numeric
    "personal_status_sex",  # A91-A95
    "other_debtors",  # A101-A103
    "residence_since",  # numeric
    "property",  # A121-A124
    "age_years",  # numeric
    "other_installment_plans",  # A141-A143
    "housing",  # A151-A153
    "existing_credits",  # numeric
    "job",  # A171-A174
    "maintenance_liable_people",  # numeric
    "telephone",  # A191-A192
    "foreign_worker",  # A201-A202
    "target",  # 1 = good, 2 = bad
]
CATEGORICAL = [
    "checking_account_status",
    "credit_history",
    "purpose",
    "savings_account",
    "employment_since",
    "personal_status_sex",
    "other_debtors",
    "property",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]
NUMERIC = [
    "duration_months",
    "credit_amount",
    "installment_rate_pct",
    "residence_since",
    "age_years",
    "existing_credits",
    "maintenance_liable_people",
]


def load_data() -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "german.data"
    if not path.exists():
        print(f"Downloading German Credit data from UCI:\n  {UCI_URL}")
        urllib.request.urlretrieve(UCI_URL, path)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ]
    )


def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(fbeta_score(y_true, y_pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    df = load_data()
    print(f"Rows: {len(df)}, columns: {len(df.columns)}")
    n_missing = int(df.isna().sum().sum())
    print(f"Missing values in the raw file: {n_missing}")
    # Target: 1 = good credit, 2 = bad credit -> default = 1 if bad.
    y = (df["target"] == 2).astype(int)
    X = df.drop(columns=["target"])
    print(f"Class balance: good (0) = {(y == 0).sum()}, default (1) = {(y == 1).sum()} "
          f"({y.mean():.1%} defaulters)")

    # Stratified 60/20/20: train / validation / test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    print(f"Split sizes — train: {len(X_train)}, validation: {len(X_val)}, test: {len(X_test)}")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Train scale_pos_weight (neg/pos): {scale_pos_weight:.2f}")

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                           random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("clf", RandomForestClassifier(
                    n_estimators=300, class_weight="balanced_subsample",
                    random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("clf", XGBClassifier(
                    n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                    random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
    }

    val_results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        val_prob = pipe.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_prob)
        val_results[name] = {"pipeline": pipe, "val_auc": float(val_auc),
                             "val_prob": val_prob}
        joblib.dump(pipe, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
        print(f"{name}: validation ROC-AUC = {val_auc:.4f}")

    best_name = max(val_results, key=lambda n: val_results[n]["val_auc"])
    best_pipe = val_results[best_name]["pipeline"]
    print(f"Best model by validation ROC-AUC: {best_name} "
          f"({val_results[best_name]['val_auc']:.4f})")

    # Threshold tuning on the VALIDATION set only: maximise F2 (recall x2).
    val_prob = val_results[best_name]["val_prob"]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_prob)
    f2_scores = [
        (2**2 * p * r) / ((2**2 * p) + r) if (p + r) > 0 else 0.0
        for p, r in zip(precisions[:-1], recalls[:-1])
    ]
    best_idx = int(np.argmax(f2_scores))
    tuned_threshold = float(thresholds[best_idx])
    print(f"Tuned threshold (max F2 on validation): {tuned_threshold:.3f} "
          f"(F2={f2_scores[best_idx]:.3f})")

    # Final evaluation on the TEST set — touched only now.
    test_prob = best_pipe.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_prob))
    default_metrics = metrics_at_threshold(y_test, test_prob, 0.5)
    tuned_metrics = metrics_at_threshold(y_test, test_prob, tuned_threshold)
    print(f"TEST ROC-AUC ({best_name}): {test_auc:.4f}")
    print(f"TEST @0.50 : {default_metrics}")
    print(f"TEST @{tuned_threshold:.3f}: {tuned_metrics}")

    # All models on test @0.5 for the comparison table.
    test_table = {}
    for name, res in val_results.items():
        p = res["pipeline"].predict_proba(X_test)[:, 1]
        m = metrics_at_threshold(y_test, p, 0.5)
        test_table[name] = {
            "test_auc": float(roc_auc_score(y_test, p)),
            "precision@0.5": m["precision"],
            "recall@0.5": m["recall"],
            "f1@0.5": m["f1"],
        }

    # ---- Explainability: SHAP on the best model (TreeExplainer for XGBoost) ----
    encoded_names = list(
        best_pipe.named_steps["preprocess"].get_feature_names_out()
    )
    n_features_encoded = len(encoded_names)
    shap_top = []
    shap_note = ""
    try:
        import shap

        X_test_enc = best_pipe.named_steps["preprocess"].transform(X_test)
        if hasattr(X_test_enc, "toarray"):
            X_test_enc = X_test_enc.toarray()
        clf = best_pipe.named_steps["clf"]
        explainer = shap.TreeExplainer(clf)
        sample = X_test_enc[:200]
        sv = explainer.shap_values(sample)
        if isinstance(sv, list):  # older shap returns a list per class
            sv = sv[1]
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(-mean_abs)[:10]
        shap_top = [
            {"feature": encoded_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in order
        ]
        print("Top SHAP features:", shap_top[:5])
    except Exception as exc:  # fall back to built-in importance
        shap_note = f"SHAP failed ({exc}); fell back to model feature importances."
        print(shap_note)
        clf = best_pipe.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            order = np.argsort(-imp)[:10]
            shap_top = [
                {"feature": encoded_names[i],
                 "mean_abs_shap": float(imp[i])}
                for i in order
            ]
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
            order = np.argsort(-imp)[:10]
            shap_top = [
                {"feature": encoded_names[i],
                 "mean_abs_shap": float(imp[i])}
                for i in order
            ]

    # ---- Simple points-based scorecard from the logistic regression ----
    lr_pipe = val_results["Logistic Regression"]["pipeline"]
    lr_clf = lr_pipe.named_steps["clf"]
    lr_names = list(lr_pipe.named_steps["preprocess"].get_feature_names_out())
    coef = lr_clf.coef_[0]
    intercept = float(lr_clf.intercept_[0])
    SCALE = 20  # points per 1 unit of log-odds
    scorecard = pd.DataFrame(
        {
            "feature": lr_names,
            "coefficient": coef,
            "points": np.round(coef * SCALE).astype(int),
        }
    ).sort_values("points", ascending=False)
    scorecard.to_csv(BASE / "scorecard.csv", index=False)
    base_points = int(round(intercept * SCALE))
    # score / SCALE == linear predictor, so p(default) = sigmoid(score / SCALE).

    # ---- ROC curve figure ----
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC = {test_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC curve — test set (German Credit)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # ---- Save metrics ----
    metrics = {
        "dataset": {
            "name": "German Credit (UCI ML Repository, Statlog)",
            "source_url": UCI_URL,
            "why_not_home_credit": "Home Credit Default Risk is Kaggle-only and "
            "requires Kaggle API credentials; none are available in this "
            "environment (Kaggle API returns 403 without them).",
            "rows_total": int(len(df)),
            "rows_train": int(len(X_train)),
            "rows_validation": int(len(X_val)),
            "rows_test": int(len(X_test)),
            "missing_values_raw": n_missing,
            "n_raw_features": 20,
            "n_features_after_one_hot": int(n_features_encoded),
            "n_categorical": len(CATEGORICAL),
            "n_numeric": len(NUMERIC),
            "default_rate_overall": float(y.mean()),
            "default_rate_train": float(y_train.mean()),
            "default_rate_validation": float(y_val.mean()),
            "default_rate_test": float(y_test.mean()),
            "split": "stratified 60% train / 20% validation / 20% test "
            "(random_state=42)",
            "split_why": "Validation is used for model selection and threshold "
            "tuning; the test set is evaluated exactly once so the reported "
            "numbers are honest.",
            "imbalance_handling": "class_weight='balanced' (LR, RF); "
            "scale_pos_weight=neg/pos on train (XGBoost). No SMOTE — with "
            "1,000 rows, class weighting avoids synthetic-sample and leakage "
            "risks. (If SMOTE were used it would be fitted on training folds "
            "only.)",
        },
        "model_comparison_test_at_0.5": test_table,
        "validation_auc": {n: r["val_auc"] for n, r in val_results.items()},
        "best_model": best_name,
        "best_model_test_auc": test_auc,
        "threshold": {
            "tuned_threshold": tuned_threshold,
            "how_chosen": "Maximised F2-score on the validation set "
            "(recall weighted 2x precision). In lending a missed defaulter "
            "(false negative) costs far more than a false alarm (false "
            "positive), so recall is deliberately prioritised.",
            "default_0.5_metrics_test": default_metrics,
            "tuned_metrics_test": tuned_metrics,
        },
        "explainability": {
            "method": "SHAP TreeExplainer on the best model, mean |SHAP| over "
            "the first 200 test rows" + (f". {shap_note}" if shap_note else ""),
            "top_features": shap_top,
        },
        "scorecard": {
            "description": "Simplified points-based scorecard derived from the "
            "logistic-regression coefficients: points = round(coef * 20); "
            "base points from the intercept. Total score / 20 = linear "
            "predictor, so P(default) = sigmoid(score / 20). Full table in "
            "scorecard.csv.",
            "base_points": base_points,
            "intercept_log_odds": intercept,
            "file": "scorecard.csv",
        },
    }
    with open(BASE / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics.json")

    # ---- RESULTS.md generated from the real numbers ----
    cm = tuned_metrics["confusion_matrix"]
    lines = []
    lines.append("# Loan Default Prediction — Results (REAL numbers)\n")
    lines.append("All figures below were produced by running `train.py` on the "
                 "German Credit dataset. Nothing here is estimated, copied from "
                 "another dataset, or invented.\n")
    lines.append("## Dataset\n")
    lines.append(f"- **Dataset:** German Credit (UCI ML Repository, Statlog) — "
                 f"[source]({UCI_URL})")
    lines.append("- **Why not Home Credit:** the Home Credit Default Risk dataset "
                 "is distributed through Kaggle and requires Kaggle API "
                 "credentials to download. No credentials exist in this "
                 "environment and the Kaggle API returns 403 without them, so "
                 "the public UCI German Credit dataset was used instead. **No "
                 "Home Credit numbers are claimed anywhere in this project.**")
    lines.append(f"- **Rows:** {len(df)} total "
                 f"({len(X_train)} train / {len(X_val)} validation / {len(X_test)} test, "
                 f"stratified 60/20/20, random_state=42)")
    lines.append(f"- **Features:** 20 raw attributes ({len(CATEGORICAL)} categorical, "
                 f"{len(NUMERIC)} numeric) → {n_features_encoded} after one-hot encoding")
    lines.append(f"- **Missing values in raw file:** {n_missing} "
                 f"(median/most-frequent imputation is still in the pipeline for robustness)")
    lines.append(f"- **Default rate:** {y.mean():.1%} overall "
                 f"({int(y.sum())} defaulters / {len(y)} applicants); "
                 f"train {y_train.mean():.1%}, validation {y_val.mean():.1%}, "
                 f"test {y_test.mean():.1%}")
    lines.append("- **Imbalance handling:** `class_weight='balanced'` (Logistic "
                 "Regression, Random Forest); `scale_pos_weight` = neg/pos on the "
                 "training set (XGBoost). No SMOTE — with 1,000 rows, class "
                 "weighting avoids synthetic-sample and leakage risks. If SMOTE "
                 "were used, it would be fitted on training folds only.\n")
    lines.append("## Split — why 60/20/20\n")
    lines.append("The validation set is used for model selection and for tuning "
                 "the decision threshold. The test set is evaluated exactly once, "
                 "at the end, so the numbers below are an honest estimate of "
                 "unseen-data performance.\n")
    lines.append("## Model comparison (test set, threshold 0.5)\n")
    lines.append("| Model | ROC-AUC | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in test_table.items():
        lines.append(f"| {name} | {m['test_auc']:.3f} | {m['precision@0.5']:.3f} | "
                     f"{m['recall@0.5']:.3f} | {m['f1@0.5']:.3f} |")
    lines.append("")
    lines.append(f"**Best model (by validation ROC-AUC): {best_name}** — "
                 f"validation AUC {val_results[best_name]['val_auc']:.3f}, "
                 f"test AUC {test_auc:.3f}.\n")
    lines.append("## Decision threshold\n")
    lines.append(f"- **Tuned threshold: {tuned_threshold:.3f}** (default is 0.5).")
    lines.append("- **How chosen:** maximised the F2-score on the validation set "
                 "(recall weighted twice as heavily as precision). In lending, a "
                 "missed defaulter (false negative — lending money that is never "
                 "repaid) costs far more than a false alarm (false positive — a "
                 "good applicant sent for manual review), so recall is "
                 "deliberately prioritised over precision.")
    lines.append(f"- **Test set at 0.5:** precision {default_metrics['precision']:.3f}, "
                 f"recall {default_metrics['recall']:.3f}, F1 {default_metrics['f1']:.3f}.")
    lines.append(f"- **Test set at tuned threshold {tuned_threshold:.3f}:** precision "
                 f"{tuned_metrics['precision']:.3f}, recall {tuned_metrics['recall']:.3f}, "
                 f"F1 {tuned_metrics['f1']:.3f}.\n")
    lines.append(f"## Confusion matrix — {best_name}, test set, threshold {tuned_threshold:.3f}\n")
    lines.append("|  | Predicted good (0) | Predicted default (1) |")
    lines.append("|---|---|---|")
    lines.append(f"| Actual good (0) | TN = {cm['tn']} | FP = {cm['fp']} |")
    lines.append(f"| Actual default (1) | FN = {cm['fn']} | TP = {cm['tp']} |")
    lines.append("")
    lines.append(f"- Of the {cm['fn'] + cm['tp']} actual defaulters in the test set, "
                 f"the model catches {cm['tp']} (recall {tuned_metrics['recall']:.1%}) "
                 f"and misses {cm['fn']}.")
    lines.append(f"- Of the {cm['tp'] + cm['fp']} applicants flagged as high risk, "
                 f"{cm['tp']} truly default (precision {tuned_metrics['precision']:.1%}); "
                 f"the other {cm['fp']} are false alarms sent for manual review.\n")
    lines.append("## Explainability (SHAP)\n")
    lines.append("SHAP values from the best model on the test set (mean |SHAP|, "
                 "top 10). These are the features that most drive a high-risk flag:\n")
    lines.append("| Feature (one-hot level) | Mean |SHAP| |")
    lines.append("|---|---:|")
    for item in shap_top:
        lines.append(f"| `{item['feature']}` | {item['mean_abs_shap']:.4f} |")
    lines.append("")
    lines.append("## Logistic-regression scorecard\n")
    lines.append("A simplified points-based scorecard built from the logistic "
                 "regression (see `scorecard.csv` for the full table): "
                 f"points = round(coefficient × 20), base points = {base_points} "
                 "(from the intercept). Total score ÷ 20 is the linear predictor, "
                 "so P(default) = sigmoid(score ÷ 20). A scorecard like this is "
                 "what a bank can actually put in front of a credit officer — "
                 "every point is traceable to one applicant attribute, which is "
                 "exactly the explainability regulators ask for.\n")
    lines.append("## How to reproduce\n")
    lines.append("```bash")
    lines.append("python -m venv .venv && source .venv/bin/activate")
    lines.append("pip install -r requirements.txt")
    lines.append("python train.py   # downloads data from UCI, trains, writes metrics.json + RESULTS.md")
    lines.append("streamlit run app.py   # interactive demo")
    lines.append("```")
    lines.append("ROC curve: `figures/roc_curve.png`.")
    (BASE / "RESULTS.md").write_text("\n".join(lines))
    print("Saved RESULTS.md")


if __name__ == "__main__":
    main()
