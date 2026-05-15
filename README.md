# ⚡ ChurnOps AI

**MLOps-grade Customer Churn Intelligence Platform for Nigerian Fintech**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)
![MLflow](https://img.shields.io/badge/MLflow-2.10-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.45-purple)

---

## 🎯 Overview

ChurnOps AI is an end-to-end MLOps pipeline that predicts customer churn in financial services. It combines production-grade machine learning with full experiment tracking and model explainability — all in a polished Streamlit interface.

Built as part of a fintech-focused data science portfolio targeting roles at Nigerian fintech companies including Kuda, Moniepoint, Flutterwave, and Carbon.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🎯 Real-Time Prediction | Score any customer instantly with churn probability |
| 📊 Model Performance | ROC curve, confusion matrix, probability distributions |
| 🔬 SHAP Explainability | Global & individual feature impact analysis |
| 📈 MLflow Tracking | Full experiment history, metric comparison, model registry |

---

## 🛠️ Tech Stack

- **ML Model:** XGBoost (Gradient Boosted Trees)
- **Explainability:** SHAP (SHapley Additive exPlanations)
- **Experiment Tracking:** MLflow
- **Frontend:** Streamlit
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualisation:** Matplotlib

---

## 📁 Project Structure

```
ChurnOps_AI/
├── app.py                      # Main Streamlit entry point
├── pages/
│   ├── 1_Predict.py            # Real-time churn prediction
│   ├── 2_Model_Performance.py  # Metrics & diagnostic charts
│   ├── 3_SHAP_Explainability.py # Feature importance & SHAP plots
│   └── 4_MLflow_Tracker.py     # Experiment tracking dashboard
├── model/
│   └── train_model.py          # XGBoost + MLflow training script
├── utils/
│   └── helpers.py              # Shared utilities & CSS
├── requirements.txt
└── runtime.txt                 # Python 3.11 for Streamlit Cloud
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/Santandave961/ChurnOps-AI.git
cd ChurnOps-AI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates pkl files)
python model/train_model.py

# 4. Run the app
streamlit run app.py
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | ~0.78 |
| ROC-AUC | ~0.84 |
| F1 Score | ~0.72 |
| Precision | ~0.75 |
| Recall | ~0.70 |

---

## 🌍 Deployment

Deployed on **Streamlit Community Cloud**.

The app auto-trains the model on first run using `train_model.py`.
MLflow experiment logs are stored locally in `mlruns/`.

---

## 👤 Author

**Okparaji Wisdom**
- GitHub: [@Santandave961](https://github.com/Santandave961)
- X: [@Santandave961](https://twitter.com/Santandave961)

---

## 📄 License

MIT License — free to use and modify.