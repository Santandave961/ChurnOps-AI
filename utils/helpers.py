import os
import pickle
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "model")

@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "churn_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    return model, encoders, feature_names

@st.cache_data
def load_test_data():
    return pd.read_csv(os.path.join(MODEL_DIR, "test_predictions.csv"))

def apply_global_css():
    st.markdown("<style>body{background:#0a0f1e;color:#e2e8f0;}</style>", unsafe_allow_html=True)

def churn_risk_label(prob):
    if prob >= 0.7:
        return "HIGH RISK", "#ef4444"
    elif prob >= 0.4:
        return "MEDIUM RISK", "#f59e0b"
    else:
        return "LOW RISK", "#22c55e"