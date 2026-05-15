import sys, os
sys.path.insert(0, r"C:\Users\USER\Desktop\ChurnOps AI")
import os

import streamlit as st
import numpy as np
import pandas as pd
from utils.helpers import load_model, apply_global_css, churn_risk_label

st.set_page_config(page_title="Predict | ChurnOps AI", page_icon=":dart:", layout="wide")
apply_global_css()

st.markdown("""
<h1 style='font-family:Syne,sans-serif; font-size:2.2rem; font-weight:800;
           background:linear-gradient(135deg,#60a5fa,#3b82f6);
           -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
    🎯 Churn Prediction
</h1>
<p style='color:#64748b;'>Enter customer details to predict churn probability in real time.</p>
""", unsafe_allow_html=True)

# ── Try loading model ────────────────────────────────────────────────────────
try:
    model, encoders, feature_names = load_model()
    model_loaded = True
except Exception:
    model_loaded = False
    st.warning(
        "⚠️ Model not found. Please run `python model/train_model.py` first to generate the model.",
        icon="⚠️"
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Input Form ───────────────────────────────────────────────────────────────
with st.form("predict_form"):
    st.markdown("### 👤 Customer Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        age             = st.slider("Age", 18, 65, 32)
        tenure_months   = st.slider("Tenure (months)", 1, 72, 12)
        monthly_balance = st.number_input("Monthly Balance (₦)", 500.0, 500000.0, 50000.0, step=1000.0)
        num_transactions = st.slider("Monthly Transactions", 0, 200, 20)

    with col2:
        num_products    = st.slider("Number of Products", 1, 4, 2)
        has_loan        = st.selectbox("Has Loan?", ["No", "Yes"])
        has_savings     = st.selectbox("Has Savings Account?", ["Yes", "No"])
        complaint_count = st.slider("Complaints Filed", 0, 10, 0)

    with col3:
        days_since_login  = st.slider("Days Since Last Login", 0, 90, 5)
        failed_txn_rate   = st.slider("Failed Transaction Rate", 0.0, 0.5, 0.05, step=0.01)
        support_calls     = st.slider("Support Calls", 0, 15, 1)
        account_type      = st.selectbox("Account Type", ["Savings", "Current", "Fixed"])
        region            = st.selectbox("Region", ["Lagos", "Abuja", "PH", "Kano", "Others"])

    submitted = st.form_submit_button("⚡ Predict Churn", use_container_width=True)

# ── Prediction ───────────────────────────────────────────────────────────────
if submitted:
    if not model_loaded:
        st.error("Cannot predict — model not loaded.")
    else:
        # Encode inputs
        acc_enc = encoders["account_type"].transform([account_type])[0]
        reg_enc = encoders["region"].transform([region])[0]

        input_data = pd.DataFrame([{
            "age":              age,
            "tenure_months":    tenure_months,
            "monthly_balance":  monthly_balance,
            "num_transactions": num_transactions,
            "num_products":     num_products,
            "has_loan":         1 if has_loan == "Yes" else 0,
            "has_savings":      1 if has_savings == "Yes" else 0,
            "complaint_count":  complaint_count,
            "days_since_login": days_since_login,
            "failed_txn_rate":  failed_txn_rate,
            "support_calls":    support_calls,
            "account_type":     acc_enc,
            "region":           reg_enc,
        }])[feature_names]

        prob  = model.predict_proba(input_data)[0][1]
        label, color = churn_risk_label(prob)

        css_class = "churn-high" if prob >= 0.5 else "churn-low"

        st.markdown(f"""
        <div class="{css_class}" style="margin-top:1.5rem; text-align:center;">
            <div style="font-size:2.5rem; font-weight:800; font-family:Syne,sans-serif;
                        color:{color};">{label}</div>
            <div style="font-size:3.5rem; font-weight:800; color:{color}; margin:0.5rem 0;">
                {prob:.1%}
            </div>
            <div style="color:#94a3b8; font-size:0.9rem;">Churn Probability</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge-style progress bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(float(prob))

        # Risk factors
        st.markdown("### 🔍 Key Risk Signals")
        signals = []
        if complaint_count >= 5:   signals.append("⚠️ High complaint count")
        if days_since_login >= 30:  signals.append("⚠️ Inactive for 30+ days")
        if failed_txn_rate >= 0.2:  signals.append("⚠️ High failed transaction rate")
        if support_calls >= 5:      signals.append("⚠️ Frequent support calls")
        if tenure_months <= 3:      signals.append("⚠️ Very new customer")

        if signals:
            for s in signals:
                st.markdown(f"- {s}")
        else:
            st.success("✅ No major risk signals detected for this customer.")