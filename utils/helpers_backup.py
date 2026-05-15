import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    """Load XGBoost model, encoders, and feature names."""
    base = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base, "model")

    with open(os.path.join(model_dir, "churn_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "label_encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(model_dir, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    return model, encoders, feature_names


@st.cache_data
def load_test_data():
    path = os.path.join(MODEL_DIR, "test_predictions.csv")
    return pd.read_csv(path)


def apply_global_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Mono', monospace;
        background-color: #0a0f1e;
        color: #e2e8f0;
    }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1428 0%, #111827 100%);
        border-right: 1px solid #1e3a5f;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: white; border: none; border-radius: 6px;
        font-family: 'Syne', sans-serif; font-weight: 600;
        letter-spacing: 0.05em; transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(59,130,246,0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 1.4rem; text-align: center;
    }
    .churn-high {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #ef4444; border-radius: 12px; padding: 1.5rem;
    }
    .churn-low {
        background: linear-gradient(135deg, #052e16, #14532d);
        border: 1px solid #22c55e; border-radius: 12px; padding: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)


def churn_risk_label(prob: float) -> tuple:
    """Return (label, color) based on churn probability."""
    if prob >= 0.7:
        return "🔴 HIGH RISK", "#ef4444"
    elif prob >= 0.4:
        return "🟡 MEDIUM RISK", "#f59e0b"
    else:
        return "🟢 LOW RISK", "#22c55e"