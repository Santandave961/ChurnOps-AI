"""
ChurnOps AI — Model Training Script
Run this once locally or on Streamlit Cloud to generate the saved model.
Usage: python model/train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)
import xgboost as xgb
import mlflow
import mlflow.xgboost

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Synthetic Fintech Churn Dataset ─────────────────────────────────────────
def generate_data(n=3000):
    np.random.seed(SEED)
    data = {
        "age":                np.random.randint(18, 65, n),
        "tenure_months":      np.random.randint(1, 72, n),
        "monthly_balance":    np.random.uniform(500, 500000, n),
        "num_transactions":   np.random.randint(0, 200, n),
        "num_products":       np.random.randint(1, 5, n),
        "has_loan":           np.random.randint(0, 2, n),
        "has_savings":        np.random.randint(0, 2, n),
        "complaint_count":    np.random.randint(0, 10, n),
        "days_since_login":   np.random.randint(0, 90, n),
        "failed_txn_rate":    np.random.uniform(0, 0.5, n),
        "support_calls":      np.random.randint(0, 15, n),
        "account_type":       np.random.choice(["Savings", "Current", "Fixed"], n),
        "region":             np.random.choice(["Lagos", "Abuja", "PH", "Kano", "Others"], n),
    }
    df = pd.DataFrame(data)

    # Churn probability based on features
    churn_prob = (
        0.3 * (df["complaint_count"] / 10) +
        0.25 * (df["days_since_login"] / 90) +
        0.2 * (1 - df["tenure_months"] / 72) +
        0.15 * df["failed_txn_rate"] +
        0.1 * (df["support_calls"] / 15)
    )
    df["churn"] = (np.random.uniform(0, 1, n) < churn_prob).astype(int)
    return df


def train():
    df = generate_data()

    # Encode categoricals
    le_account = LabelEncoder()
    le_region  = LabelEncoder()
    df["account_type"] = le_account.fit_transform(df["account_type"])
    df["region"]       = le_region.fit_transform(df["region"])

    features = [c for c in df.columns if c != "churn"]
    X = df[features]
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # ── MLflow Experiment ────────────────────────────────────────────────────
    mlflow.set_experiment("ChurnOps_AI")

    with mlflow.start_run(run_name="XGBoost_v1"):
        params = {
            "n_estimators":     200,
            "max_depth":        5,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric":      "logloss",
            "random_state":     SEED,
        }
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
            "f1_score":  round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "xgboost_churn_model")

        print("✅ Training complete!")
        for k, v in metrics.items():
            print(f"   {k}: {v}")

    # ── Save artefacts ───────────────────────────────────────────────────────
    os.makedirs("model", exist_ok=True)

    with open("model/churn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/label_encoders.pkl", "wb") as f:
        pickle.dump({"account_type": le_account, "region": le_region}, f)
    with open("model/feature_names.pkl", "wb") as f:
        pickle.dump(features, f)

    # Save test data for performance page
    X_test_df = X_test.copy()
    X_test_df["churn"]      = y_test.values
    X_test_df["churn_prob"] = y_proba
    X_test_df.to_csv("model/test_predictions.csv", index=False)

    print("✅ Model artefacts saved to model/")


if __name__ == "__main__":
    train()