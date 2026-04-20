import streamlit as st
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import json
import pickle
import hashlib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              classification_report)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ChurnOps AI", layout="centered")

# ── MLflow setup ──────────────────────────────────────────────────────────────
MLFLOW_DIR = "./mlruns"
MODEL_DIR  = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)
mlflow.set_tracking_uri(MLFLOW_DIR)
EXPERIMENT_NAME = "nigerian_fintech_churn"
mlflow.set_experiment(EXPERIMENT_NAME)


# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_data(n=500):
    np.random.seed(42)
    df = pd.DataFrame({
        "customer_id":        [f"CUST{i:04d}" for i in range(n)],
        "age":                np.random.randint(18, 65, n),
        "tenure_months":      np.random.randint(1, 60, n),
        "monthly_balance":    np.round(np.random.lognormal(10, 1, n), 2),
        "num_transactions":   np.random.randint(1, 200, n),
        "num_products":       np.random.randint(1, 5, n),
        "has_loan":           np.random.randint(0, 2, n),
        "has_savings":        np.random.randint(0, 2, n),
        "support_calls":      np.random.randint(0, 10, n),
        "days_inactive":      np.random.randint(0, 90, n),
        "gender":             np.random.choice(["Male","Female"], n),
        "state":              np.random.choice(["Lagos","Abuja","Kano","Rivers","Oyo"], n),
        "account_type":       np.random.choice(["Savings","Current","Wallet"], n),
    })
    # Churn logic
    churn_score = (
        (df["days_inactive"] > 30).astype(int) * 2 +
        (df["support_calls"] > 5).astype(int) * 2 +
        (df["tenure_months"] < 6).astype(int) +
        (df["num_products"] == 1).astype(int) +
        (df["monthly_balance"] < 5000).astype(int) +
        np.random.randint(0, 3, n)
    )
    df["churned"] = (churn_score >= 4).astype(int)
    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    for col in ["gender","state","account_type"]:
        df[col] = le.fit_transform(df[col])
    features = ["age","tenure_months","monthly_balance","num_transactions",
                "num_products","has_loan","has_savings","support_calls",
                "days_inactive","gender","state","account_type"]
    X = df[features]
    y = df["churned"]
    return X, y, features


# ── Train and log with MLflow ─────────────────────────────────────────────────
def train_model(df, model_name, model_obj, params):
    X, y, features = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  model_obj)
    ])

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(params)
        mlflow.log_param("model_type",   model_name)
        mlflow.log_param("train_size",   len(X_train))
        mlflow.log_param("test_size",    len(X_test))
        mlflow.log_param("churn_rate",   round(y.mean(), 4))
        mlflow.log_param("n_features",   len(features))

        pipeline.fit(X_train, y_train)
        y_pred      = pipeline.predict(X_test)
        y_pred_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_pred_prob),
        }

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")

        run_id = run.info.run_id

    # Save model locally
    model_path = os.path.join(MODEL_DIR, f"{model_name.replace(' ','_')}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"pipeline": pipeline, "features": features,
                     "metrics": metrics, "run_id": run_id,
                     "model_name": model_name, "trained_at": str(datetime.now())}, f)

    return metrics, run_id, pipeline, X_test, y_test, y_pred, y_pred_prob, features


# ── Load saved models ─────────────────────────────────────────────────────────
def load_saved_models():
    models = []
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            with open(os.path.join(MODEL_DIR, f), "rb") as fp:
                models.append(pickle.load(fp))
    return models


# ── Get MLflow runs ───────────────────────────────────────────────────────────
def get_mlflow_runs():
    try:
        client = mlflow.tracking.MlflowClient()
        exp    = client.get_experiment_by_name(EXPERIMENT_NAME)
        if not exp:
            return pd.DataFrame()
        runs = client.search_runs(exp.experiment_id,
                                   order_by=["metrics.roc_auc DESC"])
        if not runs:
            return pd.DataFrame()
        rows = []
        for r in runs:
            rows.append({
                "Run ID":     r.info.run_id[:8],
                "Model":      r.data.params.get("model_type",""),
                "Accuracy":   round(r.data.metrics.get("accuracy", 0), 4),
                "Precision":  round(r.data.metrics.get("precision", 0), 4),
                "Recall":     round(r.data.metrics.get("recall", 0), 4),
                "F1":         round(r.data.metrics.get("f1", 0), 4),
                "ROC-AUC":    round(r.data.metrics.get("roc_auc", 0), 4),
            })
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────
st.title("ChurnOps AI")
st.caption("End-to-End MLOps Pipeline — Customer Churn · MLflow · Streamlit")
st.markdown("Train, compare, track, and deploy customer churn models with full MLflow experiment tracking.")
st.divider()

df = generate_data()

# ── Pipeline stages sidebar ───────────────────────────────────────────────────
st.sidebar.title("MLOps Pipeline")
st.sidebar.markdown("""
**Stages:**
1. Data Ingestion
2. Preprocessing
3. Model Training
4. Experiment Tracking
5. Model Comparison
6. Model Deployment
7. Live Prediction
""")
st.sidebar.divider()
st.sidebar.metric("Dataset Size",  f"{len(df)} customers")
st.sidebar.metric("Churn Rate",    f"{df['churned'].mean()*100:.1f}%")
st.sidebar.metric("Features",      "12")

page = st.sidebar.radio("Navigate", [
    "1. Data Overview",
    "2. Train Models",
    "3. Experiment Tracking",
    "4. Model Comparison",
    "5. Deploy & Predict",
])

st.sidebar.divider()
st.sidebar.caption("Powered by MLflow + scikit-learn")


# ══════════════════════════════════════════════════════════════════════════════
if page == "1. Data Overview":
    st.subheader("Stage 1 — Data Ingestion & Overview")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{len(df):,}")
    k2.metric("Churned",         f"{df['churned'].sum():,}")
    k3.metric("Retained",        f"{(df['churned']==0).sum():,}")
    k4.metric("Churn Rate",      f"{df['churned'].mean()*100:.1f}%")

    st.markdown("#### Sample Data")
    st.dataframe(df.drop("customer_id", axis=1).head(10), use_container_width=True)

    st.markdown("#### Feature Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    num_cols = ["age","tenure_months","monthly_balance","num_transactions","support_calls","days_inactive"]
    for i, col in enumerate(num_cols):
        ax = axes[i//3][i%3]
        ax.hist(df[df["churned"]==0][col], bins=20, alpha=0.6, color="#2ecc71", label="Retained")
        ax.hist(df[df["churned"]==1][col], bins=20, alpha=0.6, color="#e74c3c", label="Churned")
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Churn by Category")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for i, col in enumerate(["gender","state","account_type"]):
        churn_rate = df.groupby(col)["churned"].mean() * 100
        axes[i].bar(churn_rate.index, churn_rate.values, color="#3498db", width=0.5)
        axes[i].set_title(f"Churn Rate by {col}", fontsize=9)
        axes[i].set_ylabel("%")
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=30, fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Correlation with Churn")
    X, y, features = preprocess(df)
    corr = pd.concat([X, y], axis=1).corr()["churned"].drop("churned").sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in corr.values]
    ax.barh(corr.index, corr.values, color=colors, height=0.6)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title("Feature Correlation with Churn")
    ax.set_xlabel("Correlation")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
elif page == "2. Train Models":
    st.subheader("Stage 2 & 3 — Model Training + MLflow Logging")

    st.markdown("Select a model, configure hyperparameters, and train. All runs are logged to MLflow automatically.")

    model_choice = st.selectbox("Select Model", [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    ])

    st.markdown("#### Hyperparameters")

    params = {}
    model_obj = None

    if model_choice == "Logistic Regression":
        c_val = st.slider("C (Regularization)", 0.01, 10.0, 1.0, step=0.01)
        max_iter = st.slider("Max Iterations", 100, 1000, 200, step=100)
        params = {"C": c_val, "max_iter": max_iter}
        model_obj = LogisticRegression(C=c_val, max_iter=max_iter, random_state=42)

    elif model_choice == "Random Forest":
        n_est  = st.slider("N Estimators", 50, 500, 100, step=50)
        max_d  = st.slider("Max Depth", 2, 20, 6)
        min_s  = st.slider("Min Samples Split", 2, 20, 2)
        params = {"n_estimators": n_est, "max_depth": max_d, "min_samples_split": min_s}
        model_obj = RandomForestClassifier(n_estimators=n_est, max_depth=max_d,
                                            min_samples_split=min_s, random_state=42)

    elif model_choice == "Gradient Boosting":
        n_est  = st.slider("N Estimators", 50, 300, 100, step=50)
        lr     = st.slider("Learning Rate", 0.01, 0.5, 0.1, step=0.01)
        max_d  = st.slider("Max Depth", 2, 8, 3)
        params = {"n_estimators": n_est, "learning_rate": lr, "max_depth": max_d}
        model_obj = GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr,
                                                max_depth=max_d, random_state=42)

    if st.button("Train and Log to MLflow", use_container_width=True):
        with st.spinner(f"Training {model_choice} and logging to MLflow..."):
            metrics, run_id, pipeline, X_test, y_test, y_pred, y_pred_prob, features = \
                train_model(df, model_choice, model_obj, params)

        st.success(f"Model trained and logged! Run ID: `{run_id[:8]}`")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
        m2.metric("Precision", f"{metrics['precision']:.4f}")
        m3.metric("Recall",    f"{metrics['recall']:.4f}")
        m4.metric("F1",        f"{metrics['f1']:.4f}")
        m5.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}")

        # Confusion matrix
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
        cm = confusion_matrix(y_test, y_pred)
        im = ax1.imshow(cm, cmap="Blues")
        ax1.set_xticks([0,1]); ax1.set_yticks([0,1])
        ax1.set_xticklabels(["Retained","Churned"])
        ax1.set_yticklabels(["Retained","Churned"])
        ax1.set_xlabel("Predicted"); ax1.set_ylabel("Actual")
        ax1.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax1.text(j, i, str(cm[i,j]), ha="center", va="center",
                         color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)

        # Feature importance
        if hasattr(pipeline.named_steps["model"], "feature_importances_"):
            fi = pd.Series(pipeline.named_steps["model"].feature_importances_,
                           index=features).sort_values(ascending=True).tail(8)
            ax2.barh(fi.index, fi.values, color="#3498db", height=0.6)
            ax2.set_title("Feature Importance")
            ax2.set_xlabel("Importance")
        else:
            coef = pd.Series(abs(pipeline.named_steps["model"].coef_[0]),
                             index=features).sort_values(ascending=True).tail(8)
            ax2.barh(coef.index, coef.values, color="#3498db", height=0.6)
            ax2.set_title("Feature Coefficients")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("**MLflow Log Summary:**")
        st.json({"run_id": run_id[:8], "model": model_choice,
                 "params": params, "metrics": {k: round(v,4) for k,v in metrics.items()}})


# ══════════════════════════════════════════════════════════════════════════════
elif page == "3. Experiment Tracking":
    st.subheader("Stage 4 — MLflow Experiment Tracking")

    runs_df = get_mlflow_runs()

    if runs_df.empty:
        st.info("No runs logged yet. Go to Stage 2 to train models first.")
    else:
        st.markdown(f"**{len(runs_df)} experiment runs logged**")
        st.dataframe(runs_df, use_container_width=True, hide_index=True)

        st.markdown("#### Metrics Comparison")
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        for i, metric in enumerate(["Accuracy","F1","ROC-AUC"]):
            if metric in runs_df.columns:
                axes[i].bar(runs_df["Model"], runs_df[metric],
                            color=["#3498db","#e74c3c","#2ecc71"][:len(runs_df)], width=0.5)
                axes[i].set_title(metric, fontsize=10)
                axes[i].set_ylim(0, 1)
                plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=20, fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        best = runs_df.loc[runs_df["ROC-AUC"].idxmax()]
        st.success(f"Best model by ROC-AUC: **{best['Model']}** with ROC-AUC = {best['ROC-AUC']}")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "4. Model Comparison":
    st.subheader("Stage 5 — Model Comparison")

    saved = load_saved_models()

    if not saved:
        st.info("No models saved yet. Train at least one model in Stage 2.")
    else:
        st.markdown(f"**{len(saved)} models trained and saved**")

        comp_df = pd.DataFrame([{
            "Model":     m["model_name"],
            "Accuracy":  round(m["metrics"]["accuracy"], 4),
            "Precision": round(m["metrics"]["precision"], 4),
            "Recall":    round(m["metrics"]["recall"], 4),
            "F1":        round(m["metrics"]["f1"], 4),
            "ROC-AUC":   round(m["metrics"]["roc_auc"], 4),
            "Trained At":m["trained_at"][:19],
        } for m in saved])

        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("#### ROC-AUC Leaderboard")
        comp_sorted = comp_df.sort_values("ROC-AUC", ascending=True)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = ["#e74c3c" if v == comp_sorted["ROC-AUC"].max() else "#3498db"
                  for v in comp_sorted["ROC-AUC"]]
        bars = ax.barh(comp_sorted["Model"], comp_sorted["ROC-AUC"],
                       color=colors, height=0.5)
        ax.set_xlabel("ROC-AUC Score")
        ax.set_title("Model Leaderboard")
        ax.set_xlim(0, 1)
        for bar, val in zip(bars, comp_sorted["ROC-AUC"]):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{val:.4f}", va="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        best_model = comp_df.loc[comp_df["ROC-AUC"].idxmax(), "Model"]
        st.success(f"Recommended for deployment: **{best_model}**")


# ══════════════════════════════════════════════════════════════════════════════
elif page == "5. Deploy & Predict":
    st.subheader("Stage 6 & 7 — Model Deployment + Live Prediction")

    saved = load_saved_models()

    if not saved:
        st.info("No models trained yet. Go to Stage 2 first.")
    else:
        model_names = [m["model_name"] for m in saved]
        selected    = st.selectbox("Select Model to Deploy", model_names)
        model_data  = next(m for m in saved if m["model_name"] == selected)

        st.success(f"Model **{selected}** loaded and ready for inference")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{model_data['metrics']['accuracy']:.4f}")
        col2.metric("F1 Score", f"{model_data['metrics']['f1']:.4f}")
        col3.metric("ROC-AUC",  f"{model_data['metrics']['roc_auc']:.4f}")

        st.divider()
        st.markdown("#### Live Customer Churn Prediction")
        st.markdown("Input customer details to get a real-time churn prediction.")

        p1, p2 = st.columns(2)
        with p1:
            age           = st.slider("Age", 18, 65, 35)
            tenure        = st.slider("Tenure (months)", 1, 60, 12)
            balance       = st.number_input("Monthly Balance (NGN)", 0, 5000000, 50000)
            num_txn       = st.slider("Num Transactions", 1, 200, 20)
            num_products  = st.slider("Num Products", 1, 5, 2)
            has_loan      = st.selectbox("Has Loan?", [0, 1])
        with p2:
            has_savings   = st.selectbox("Has Savings?", [1, 0])
            support_calls = st.slider("Support Calls", 0, 10, 1)
            days_inactive = st.slider("Days Inactive", 0, 90, 10)
            gender        = st.selectbox("Gender", ["Male","Female"])
            state         = st.selectbox("State", ["Lagos","Abuja","Kano","Rivers","Oyo"])
            account_type  = st.selectbox("Account Type", ["Savings","Current","Wallet"])

        if st.button("Predict Churn Risk", use_container_width=True):
            le_gender  = LabelEncoder().fit(["Female","Male"])
            le_state   = LabelEncoder().fit(["Abuja","Kano","Lagos","Oyo","Rivers"])
            le_account = LabelEncoder().fit(["Current","Savings","Wallet"])

            X_input = pd.DataFrame([[
                age, tenure, balance, num_txn, num_products,
                has_loan, has_savings, support_calls, days_inactive,
                le_gender.transform([gender])[0],
                le_state.transform([state])[0],
                le_account.transform([account_type])[0],
            ]], columns=model_data["features"])

            pipeline = model_data["pipeline"]
            pred     = pipeline.predict(X_input)[0]
            prob     = pipeline.predict_proba(X_input)[0]

            st.divider()
            if pred == 1:
                st.error("HIGH CHURN RISK")
                verdict_color = "#e74c3c"
            else:
                st.success("LOW CHURN RISK")
                verdict_color = "#2ecc71"

            st.markdown(
                "<h2 style='text-align:center;color:" + verdict_color + "'>" +
                ("LIKELY TO CHURN" if pred==1 else "LIKELY TO STAY") + "</h2>",
                unsafe_allow_html=True
            )

            r1, r2 = st.columns(2)
            r1.metric("Churn Probability",  f"{prob[1]*100:.1f}%")
            r2.metric("Retention Probability", f"{prob[0]*100:.1f}%")

            fig, ax = plt.subplots(figsize=(6, 1))
            ax.barh([""], [prob[1]*100], color="#e74c3c", height=0.5)
            ax.barh([""], [prob[0]*100], left=[prob[1]*100], color="#2ecc71", height=0.5)
            ax.set_xlim(0,100)
            ax.set_title("Churn vs Retention Probability")
            ax.legend(["Churn","Retain"], loc="lower right", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("**Recommended Actions:**")
            if pred == 1:
                if days_inactive > 30:
                    st.warning("Send re-engagement campaign — customer has been inactive for over 30 days")
                if support_calls > 3:
                    st.warning("Escalate to customer success team — high support call volume")
                if num_products == 1:
                    st.info("Offer additional product — savings or loan to increase stickiness")
                if balance < 10000:
                    st.info("Offer cashback or loyalty reward to increase engagement")
            else:
                st.info("Customer is healthy. Consider upselling premium products.")

st.divider()
st.caption("ChurnOps AI · MLOps Pipeline · MLflow + scikit-learn + Streamlit · Built by Okparaji Wisdom")