import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score
)
import xgboost as xgb

SEED = 42
np.random.seed(SEED)

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
        "support_calls":      np.random.randint(0, 15, n),
        "account_type":       np.random.choice(["Savings", "Current", "Fixed"], n),
        "region":             np.random.choice(["Lagos", "Abuja", "PH", "Kano", "Others"], n),
    }
    df = pd.DataFrame(data)

    # failed_txn_rate now DERIVED from num_transactions instead of independent
    # random noise - a customer with more transactions and more complaints is
    # more likely to also have a higher failure rate (a real-world pattern,
    # not an unrelated random number)
    base_fail_rate = np.random.uniform(0, 0.15, n)
    df["failed_txn_rate"] = np.clip(
        base_fail_rate + (df["complaint_count"] / 10) * 0.2, 0, 1
    )

    # --- Stronger, sharper churn signal ---
    # Same weighted logic as before, but scaled up before the sigmoid so the
    # seperation between "likely churner" and "likely stayer" is much wider.
    raw_score = (
        1.2 * (df["complaint_count"] / 10) +
        1.0 * (df["days_since_login"] / 90) +
        0.9 * (1 - df["tenure_months"] / 72) +
        1.1 * df["failed_txn_rate"] +
        0.7 * (df["support_calls"] / 15) +
        0.4 * (1 - df["num_transactions"] / 200) # low activity also raises churn risk
    )

    # Center and scale, then push through a sigmoid for a clean, separable
    # probability curve instead of a flat linear probability compared
    # against pure uniform noise
    centered = (raw_score - raw_score.mean()) / raw_score.std()
    churn_prob = 1 / (1 + np.exp(-2.5 * centered)) # steeper sigmoid = more seperable

    # Small amount of noise only, not a full independent random threshold -
    # keeps some realistic uncertainty without drowning the signal
    noise = np.random.normal(0, 0.05, n)
    df["churn"] = ((churn_prob + noise) > 0.5).astype(int)
    return df

def train():
    df = generate_data()
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

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=SEED
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
    }

    print("Training complete!")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs("model", exist_ok=True)
    pickle.dump(model, open("model/churn_model.pkl", "wb"))
    pickle.dump({"account_type": le_account, "region": le_region},
                open("model/label_encoders.pkl", "wb"))
    pickle.dump(features, open("model/feature_names.pkl", "wb"))

    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f)

    X_test_df = X_test.copy()
    X_test_df["churn"]      = y_test.values
    X_test_df["churn_prob"] = y_proba
    X_test_df.to_csv("model/test_predictions.csv", index=False)

    print("All model files saved!")

if __name__ == "__main__":
    train()