import sys, os
sys.path.insert(0, r"C:\Users\USER\Desktop\ChurnOps AI")
import os

import streamlit as st
import pandas as pd
from utils.helpers import apply_global_css

st.set_page_config(page_title="MLflow | ChurnOps AI", page_icon=":chart_with_upwards_trend:", layout="wide")
apply_global_css()

st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800;
           background:linear-gradient(135deg,#60a5fa,#3b82f6);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    📈 MLflow Experiment Tracker
</h1>
<p style='color:#64748b;'>Review experiment runs, compare metrics, and track model versions.</p>
""", unsafe_allow_html=True)

ACCENT = "#60a5fa"
SUBTLE = "#64748b"

# ── Try loading MLflow ────────────────────────────────────────────────────────
try:
    import mlflow
    mlflow_available = True
except ImportError:
    mlflow_available = False

if not mlflow_available:
    st.error("MLflow is not installed. Add `mlflow==2.10.0` to requirements.txt.")
    st.stop()

# ── Load Runs ────────────────────────────────────────────────────────────────
try:
    mlflow.set_experiment("ChurnOps_AI")
    runs_df = mlflow.search_runs(experiment_names=["ChurnOps_AI"])

    if runs_df.empty:
        st.info("No MLflow runs found yet. Run `python model/train_model.py` to log the first experiment.")
        st.stop()

    # Clean column names
    metric_cols  = [c for c in runs_df.columns if c.startswith("metrics.")]
    param_cols   = [c for c in runs_df.columns if c.startswith("params.")]
    keep_cols    = ["run_id", "status", "start_time", "end_time"] + metric_cols + param_cols
    display_df   = runs_df[[c for c in keep_cols if c in runs_df.columns]].copy()

    display_df.columns = [
        c.replace("metrics.", "").replace("params.", "param_")
        for c in display_df.columns
    ]

    st.markdown("### 🗂️ All Experiment Runs")
    st.dataframe(display_df, use_container_width=True)

    # ── Best Run ──────────────────────────────────────────────────────────────
    if "roc_auc" in display_df.columns:
        best = display_df.loc[display_df["roc_auc"].idxmax()]

        st.markdown("### 🏆 Best Run by ROC-AUC")
        cols = st.columns(5)
        metric_keys = ["accuracy", "roc_auc", "f1_score", "precision", "recall"]
        for col, key in zip(cols, metric_keys):
            val = best.get(key, "N/A")
            display_val = f"{float(val):.4f}" if val != "N/A" else "N/A"
            col.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.7rem; font-weight:800; font-family:Syne,sans-serif;
                            color:{ACCENT};">{display_val}</div>
                <div style="color:{SUBTLE}; font-size:0.8rem; margin-top:0.3rem;">{key.replace('_',' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Metric Trend Chart ─────────────────────────────────────────────────────
    if len(display_df) > 1 and "roc_auc" in display_df.columns:
        st.markdown("### 📉 ROC-AUC Across Runs")
        chart_df = display_df[["start_time", "roc_auc"]].dropna().copy()
        chart_df = chart_df.sort_values("start_time")
        st.line_chart(chart_df.set_index("start_time")["roc_auc"])

except Exception as e:
    st.error(f"Could not load MLflow data: {e}")
    st.info("""
    **Tip:** MLflow tracking requires the model to have been trained locally.
    On Streamlit Cloud, MLflow stores runs in a local `mlruns/` folder.
    Run `python model/train_model.py` to initialise experiment logs.
    """)