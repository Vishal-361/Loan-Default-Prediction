# Loan Default Prediction — Results (REAL numbers)

All figures below were produced by running `train.py` on the German Credit dataset. Nothing here is estimated, copied from another dataset, or invented.

## Dataset

- **Dataset:** German Credit (UCI ML Repository, Statlog) — [source](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data)
- **Why not Home Credit:** the Home Credit Default Risk dataset is distributed through Kaggle and requires Kaggle API credentials to download. No credentials exist in this environment and the Kaggle API returns 403 without them, so the public UCI German Credit dataset was used instead. **No Home Credit numbers are claimed anywhere in this project.**
- **Rows:** 1000 total (600 train / 200 validation / 200 test, stratified 60/20/20, random_state=42)
- **Features:** 20 raw attributes (13 categorical, 7 numeric) → 61 after one-hot encoding
- **Missing values in raw file:** 0 (median/most-frequent imputation is still in the pipeline for robustness)
- **Default rate:** 30.0% overall (300 defaulters / 1000 applicants); train 30.0%, validation 30.0%, test 30.0%
- **Imbalance handling:** `class_weight='balanced'` (Logistic Regression, Random Forest); `scale_pos_weight` = neg/pos on the training set (XGBoost). No SMOTE — with 1,000 rows, class weighting avoids synthetic-sample and leakage risks. If SMOTE were used, it would be fitted on training folds only.

## Split — why 60/20/20

The validation set is used for model selection and for tuning the decision threshold. The test set is evaluated exactly once, at the end, so the numbers below are an honest estimate of unseen-data performance.

## Model comparison (test set, threshold 0.5)

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.790 | 0.541 | 0.767 | 0.634 |
| Random Forest | 0.776 | 0.621 | 0.300 | 0.404 |
| XGBoost | 0.754 | 0.524 | 0.550 | 0.537 |

**Best model (by validation ROC-AUC): XGBoost** — validation AUC 0.833, test AUC 0.754.

## Decision threshold

- **Tuned threshold: 0.174** (default is 0.5).
- **How chosen:** maximised the F2-score on the validation set (recall weighted twice as heavily as precision). In lending, a missed defaulter (false negative — lending money that is never repaid) costs far more than a false alarm (false positive — a good applicant sent for manual review), so recall is deliberately prioritised over precision.
- **Test set at 0.5:** precision 0.524, recall 0.550, F1 0.537.
- **Test set at tuned threshold 0.174:** precision 0.417, recall 0.800, F1 0.549.

## Confusion matrix — XGBoost, test set, threshold 0.174

|  | Predicted good (0) | Predicted default (1) |
|---|---|---|
| Actual good (0) | TN = 73 | FP = 67 |
| Actual default (1) | FN = 12 | TP = 48 |

- Of the 60 actual defaulters in the test set, the model catches 48 (recall 80.0%) and misses 12.
- Of the 115 applicants flagged as high risk, 48 truly default (precision 41.7%); the other 67 are false alarms sent for manual review.

## Explainability (SHAP)

SHAP values from the best model on the test set (mean |SHAP|, top 10). These are the features that most drive a high-risk flag:

| Feature (one-hot level) | Mean |SHAP| |
|---|---:|
| `cat__checking_account_status_A14` | 0.6087 |
| `num__duration_months` | 0.4500 |
| `num__credit_amount` | 0.4296 |
| `cat__credit_history_A34` | 0.3485 |
| `cat__employment_since_A74` | 0.2379 |
| `cat__savings_account_A61` | 0.2252 |
| `num__installment_rate_pct` | 0.2186 |
| `num__age_years` | 0.2168 |
| `cat__checking_account_status_A11` | 0.2131 |
| `cat__other_installment_plans_A143` | 0.2124 |

## Logistic-regression scorecard

A simplified points-based scorecard built from the logistic regression (see `scorecard.csv` for the full table): points = round(coefficient × 20), base points = -14 (from the intercept). Total score ÷ 20 is the linear predictor, so P(default) = sigmoid(score ÷ 20). A scorecard like this is what a bank can actually put in front of a credit officer — every point is traceable to one applicant attribute, which is exactly the explainability regulators ask for.

## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train.py   # downloads data from UCI, trains, writes metrics.json + RESULTS.md
streamlit run app.py   # interactive demo
```
ROC curve: `figures/roc_curve.png`.