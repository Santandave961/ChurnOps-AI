import streamlit as st

st.set_page_config(
    page_title="ChurnOps AI",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
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
    color: white;
    border: none;
    border-radius: 6px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59,130,246,0.4);
}

.metric-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 1rem 2rem;">
    <h1 style="font-family:'Syne',sans-serif; font-size:3rem; font-weight:800;
               background: linear-gradient(135deg, #60a5fa, #3b82f6, #1d4ed8);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        ⚡ ChurnOps AI
    </h1>
    <p style="color:#94a3b8; font-size:1.1rem; margin-top:0.5rem;">
        MLOps-grade Customer Churn Intelligence Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards ───────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
cards = [
    ("🎯", "XGBoost", "Gradient Boosted Model"),
    ("🔬", "SHAP", "Explainability Engine"),
    ("📊", "MLflow", "Experiment Tracking"),
    ("🚀", "Real-Time", "Churn Prediction"),
]
for col, (icon, title, desc) in zip([col1, col2, col3, col4], cards):
    col.markdown(f"""
    <div class="metric-card">
        <div style="font-size:2rem;">{icon}</div>
        <div style="font-family:'Syne',sans-serif; font-weight:700;
                    color:#60a5fa; font-size:1.1rem; margin:0.4rem 0;">{title}</div>
        <div style="color:#64748b; font-size:0.8rem;">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── About ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a,#1e293b);
            border:1px solid #1e3a5f; border-radius:12px; padding:2rem; margin-top:1rem;">
    <h3 style="font-family:'Syne',sans-serif; color:#60a5fa;">About ChurnOps AI</h3>
    <p style="color:#94a3b8; line-height:1.8;">
        ChurnOps AI is an end-to-end MLOps pipeline for predicting customer churn in financial
        services. It combines <strong style="color:#e2e8f0;">XGBoost</strong> for high-accuracy
        predictions, <strong style="color:#e2e8f0;">SHAP</strong> for model transparency,
        and <strong style="color:#e2e8f0;">MLflow</strong> for full experiment tracking —
        all wrapped in a production-ready Streamlit interface.
    </p>
    <ul style="color:#94a3b8; line-height:2;">
        <li>Navigate to <strong style="color:#60a5fa;">Predict</strong> to score a customer</li>
        <li>Navigate to <strong style="color:#60a5fa;">Model Performance</strong> for metrics & charts</li>
        <li>Navigate to <strong style="color:#60a5fa;">SHAP Explainability</strong> for feature impact</li>
        <li>Navigate to <strong style="color:#60a5fa;">MLflow Tracker</strong> for experiment history</li>
    </ul>
</div>
""", unsafe_allow_html=True)