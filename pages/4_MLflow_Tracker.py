import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import json
from utils.helpers import apply_global_css

st.set_page_config(page_title="Metrics | ChurnOps AI", page_icon=":chart_with_upwards_trend:", layout="wide")
apply_global_css()

st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800;
           background:linear-gradient(135deg,#60a5fa,#3b82f6);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    📈 Model Metrics
</h1>
<p style='color:#64748b;'>Training results and model performance summary.</p>
""", unsafe_allow_html=True)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
metrics_path = os.path.join(ROOT, "model", "metrics.json")

try:
    with open(metrics_path) as f:
        m = json.load(f)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Accuracy",  f"{m['accuracy']:.4f}")
    c2.metric("ROC-AUC",   f"{m['roc_auc']:.4f}")
    c3.metric("F1 Score",  f"{m['f1_score']:.4f}")
    c4.metric("Precision", f"{m['precision']:.4f}")
    c5.metric("Recall",    f"{m['recall']:.4f}")
except Exception:
    st.info("Model metrics will appear here after the app trains the model.")