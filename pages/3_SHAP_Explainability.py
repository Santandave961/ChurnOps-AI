import sys, os
sys.path.insert(0, r"C:\Users\USER\Desktop\ChurnOps AI")
import os

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from utils.helpers import load_model, load_test_data, apply_global_css

st.set_page_config(page_title="SHAP | ChurnOps AI", page_icon=":mag:", layout="wide")
apply_global_css()

st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800;
           background:linear-gradient(135deg,#60a5fa,#3b82f6);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    🔬 SHAP Explainability
</h1>
<p style='color:#64748b;'>Understand which features drive churn predictions.</p>
""", unsafe_allow_html=True)

DARK_BG = "#0a0f1e"
CARD_BG = "#0f172a"
BORDER  = "#1e3a5f"
BLUE    = "#3b82f6"
ACCENT  = "#60a5fa"
TEXT    = "#e2e8f0"
SUBTLE  = "#64748b"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor":   CARD_BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT,
    "xtick.color":      SUBTLE,
    "ytick.color":      SUBTLE,
    "text.color":       TEXT,
    "grid.color":       BORDER,
})

try:
    model, encoders, feature_names = load_model()
    df = load_test_data()
except Exception:
    st.warning("⚠️ Run `python model/train_model.py` first.")
    st.stop()

X_test = df[feature_names]

# ── SHAP Values ──────────────────────────────────────────────────────────────
@st.cache_data
def compute_shap(_model, X):
    explainer   = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values

with st.spinner("Computing SHAP values..."):
    explainer, shap_values = compute_shap(model, X_test)

# ── Global Feature Importance ────────────────────────────────────────────────
st.markdown("### 🌍 Global Feature Importance (Mean |SHAP|)")

mean_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean |SHAP|": mean_shap
}).sort_values("Mean |SHAP|", ascending=True)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(importance_df["Feature"], importance_df["Mean |SHAP|"],
               color=BLUE, alpha=0.85)

# Gradient color by value
max_val = importance_df["Mean |SHAP|"].max()
for bar, val in zip(bars, importance_df["Mean |SHAP|"]):
    alpha = 0.4 + 0.6 * (val / max_val)
    bar.set_alpha(alpha)

ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
ax.set_title("Feature Importance (Global)", fontsize=13,
             fontweight="bold", color=ACCENT, fontfamily="monospace")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ── SHAP Summary Plot ─────────────────────────────────────────────────────────
st.markdown("### 🎨 SHAP Summary Beeswarm Plot")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                  show=False, plot_size=None)
fig = plt.gcf()
fig.patch.set_facecolor(DARK_BG)
for ax_ in fig.axes:
    ax_.set_facecolor(CARD_BG)
plt.tight_layout()
st.pyplot(fig)
plt.close("all")

# ── Individual Customer Explanation ──────────────────────────────────────────
st.markdown("### 👤 Individual Customer Explanation")
idx = st.slider("Select customer index", 0, len(X_test) - 1, 0)

customer_row  = X_test.iloc[idx:idx+1]
customer_shap = shap_values[idx]

shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Value":   customer_row.values[0],
    "SHAP":    customer_shap,
}).sort_values("SHAP", key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_df["SHAP"]]
ax.barh(shap_df["Feature"], shap_df["SHAP"], color=colors, alpha=0.85)
ax.axvline(0, color=TEXT, lw=0.8)
ax.set_xlabel("SHAP Value (impact on churn probability)", fontsize=10)
ax.set_title(f"Customer #{idx} — Feature Contributions", fontsize=13,
             fontweight="bold", color=ACCENT, fontfamily="monospace")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# Show table
st.dataframe(
    shap_df.style.background_gradient(subset=["SHAP"], cmap="RdYlGn"),
    use_container_width=True
)