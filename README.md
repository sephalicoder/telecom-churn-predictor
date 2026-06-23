# 📡 Telecom Customer Churn Predictor

> End-to-end machine learning system that predicts which telecom customers will churn — and explains exactly why — with actionable retention recommendations.

🔗 **[Live App](https://telecom-churn-predictor-ha9vbj7rmec9zdsv7jmcpj.streamlit.app/)**

---

## 📌 Project Summary

A major telecom operator faces a **26.5% annual churn rate** across 7,043 customers — losing revenue silently with no visibility into who is at risk or why.

This project builds a full predictive analytics pipeline:
- Identifies **which customers will churn** with 80.3% accuracy
- Explains **why** using SHAP explainability (not just a black box)
- Delivers **boardroom-ready recommendations** — not just model metrics
- Deployed as a **live web application** for real-time customer risk scoring

---

## 🎯 Business Insights

| # | Driver | Finding | Recommendation |
|---|---|---|---|
| 1 | Contract type | Month-to-month customers churn at **3× the rate** of annual subscribers | Offer 10–15% discount to migrate to annual plans |
| 2 | Customer tenure | **Churn peaks in first 12 months** — risk drops sharply after that | Launch 90-day onboarding programme with proactive check-ins |
| 3 | Monthly charges | Churned customers pay **$74/mo vs $61/mo** for retained customers | Personalised retention offers for high-spend flexible-plan customers |
| 4 | Online security | Customers without security services churn **2× more** | Offer 30-day free trial of security + tech support bundle |
| 5 | Payment method | Electronic check correlates with **higher disengagement** | Incentivise auto-pay via bank transfer or credit card |

---

## 🤖 Model Performance

| Model | Accuracy | AUC-ROC |
|---|---|---|
| **XGBoost** ✅ | 80.3% | **0.845** |
| Logistic Regression | 80.6% | 0.842 |
| Random Forest | 78.3% | 0.818 |

XGBoost selected for deployment — highest AUC and best SHAP compatibility.

**Confusion matrix (test set):**
- Churners correctly identified: **200 of 374**
- False alarms: 104
- Estimated annual revenue protected: **~$443,000**

---

## 🏗️ Project Architecture

```
churn-consulting-project/
│
├── 01_eda.ipynb              # Exploratory data analysis
├── 02_preprocessing.ipynb    # Feature engineering & encoding
├── 03_model.ipynb            # Model training & comparison
├── 04_shap.ipynb             # SHAP explainability analysis
├── 05_dashboard.ipynb        # Executive dashboard & consulting brief
│
├── app.py                    # Streamlit web application
├── requirements.txt          # Dependencies
│
├── xgb_churn_model.pkl       # Trained XGBoost model
├── shap_explainer.pkl        # SHAP TreeExplainer
├── processed_data.pkl        # Train/test splits
└── scaler.pkl                # StandardScaler
```

---

## 🔍 SHAP Feature Importance (Global)

| Rank | Feature | Mean SHAP Value |
|---|---|---|
| 1 | Month-to-month contract | 0.744 |
| 2 | Customer tenure | 0.453 |
| 3 | Monthly charges | 0.274 |
| 4 | No online security | 0.251 |
| 5 | Fiber optic internet | 0.182 |
| 6 | No tech support | 0.174 |
| 7 | Electronic check payment | 0.156 |
| 8 | Total charges | 0.153 |

---

## 🖥️ Web Application Features

- **Real-time churn prediction** — input any customer profile, get instant risk score
- **SHAP waterfall chart** — see exactly which features drive each individual prediction
- **Risk classification** — Low / Medium / High with colour-coded visual indicators
- **6 dynamic recommendations** — auto-generated based on the customer's profile
- **Executive summary** — one-paragraph client brief that updates with every prediction
- **Revenue at risk** — annual revenue exposure calculated per customer

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Web application | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📦 Run Locally

```bash
# Clone the repo
git clone https://github.com/sephalicoder/telecom-churn-predictor.git
cd telecom-churn-predictor

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📊 Dataset

**IBM Telco Customer Churn Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Records: 7,043 customers
- Features: 20 variables (demographics, services, billing)
- Target: Churn (Yes/No) — 26.5% positive class

---

## 💼 Consulting Framing

This project is deliberately framed as a **consulting deliverable**, not just an ML exercise:

- Every model output is translated into a **business recommendation**
- Results are presented at the **executive level** (revenue impact, not just accuracy)
- The app mirrors what a **McKinsey or BCG data team** would deliver to a telecom client
- SHAP explainability makes the model **auditable and client-presentable**

---

## 👤 SEPHALI

Built as a final year project demonstrating end-to-end ML engineering + business consulting thinking.

🔗 **[Live Demo](https://telecom-churn-predictor-ha9vbj7rmec9zdsv7jmcpj.streamlit.app/)**
