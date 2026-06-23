import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Churn Intelligence", page_icon="📡", layout="wide")

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] { background:#F8F9FB; }
[data-testid="stSidebar"] {
    background:#0A1628 !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color:#CBD5E1 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] h4 {
    color:#F1F5F9 !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
    color:#94A3B8 !important;
    font-size:11px !important;
    font-weight:600;
    text-transform:uppercase;
    letter-spacing:0.06em;
}
.stSelectbox > div > div {
    background:#132039 !important;
    border:1px solid #1E3A5F !important;
    border-radius:8px !important;
    color:#E2E8F0 !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background:#2563EB !important;
    border-color:#2563EB !important;
}
.main .block-container {
    background:#F8F9FB;
    padding-top:1.5rem;
    max-width:1200px;
}
/* ── Metrics ── */
[data-testid="metric-container"] {
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:12px;
    padding:1rem 1.2rem;
    box-shadow:0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label { color:#64748B !important; font-size:12px !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color:#0F172A !important; font-size:24px !important; }
[data-testid="stMetricDelta"] { color:#2563EB !important; }
/* ── Button ── */
.stButton > button {
    background:#2563EB !important;
    color:#fff !important;
    border:none !important;
    border-radius:8px !important;
    font-weight:600 !important;
    padding:0.65rem 1.2rem !important;
    width:100%;
    font-size:13px !important;
    letter-spacing:0.02em;
}
.stButton > button:hover { background:#1D4ED8 !important; }
hr { border-color:#E2E8F0 !important; }
.stProgress > div > div > div { background:#2563EB !important; }
.stProgress > div > div { background:#E2E8F0 !important; border-radius:6px !important; }
</style>
""", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model     = joblib.load('xgb_churn_model.pkl')
    explainer = joblib.load('shap_explainer.pkl')
    _, X_test, _, _ = joblib.load('processed_data.pkl')
    return model, explainer, X_test

model, explainer, X_test = load_assets()

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding-bottom:18px;
         border-bottom:1px solid #1E3A5F;margin-bottom:16px">
        <div style="width:36px;height:36px;background:#2563EB;border-radius:8px;
             display:flex;align-items:center;justify-content:center;font-size:18px">📡</div>
        <div>
            <div style="font-size:14px;font-weight:700;color:#F1F5F9">Churn Intelligence</div>
            <div style="font-size:11px;color:#64748B">Telecom Analytics Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Account")
    contract   = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
    tenure     = st.slider("Tenure (months)", 0, 72, 12)
    monthly    = st.slider("Monthly Charges ($)", 18, 120, 65)
    total_ch   = monthly * tenure

    st.markdown("#### Services")
    internet   = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
    online_sec = st.selectbox("Online Security", ["Yes","No","No internet service"])
    online_bk  = st.selectbox("Online Backup", ["Yes","No","No internet service"])
    device_pr  = st.selectbox("Device Protection", ["Yes","No","No internet service"])
    tech_sup   = st.selectbox("Tech Support", ["Yes","No","No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
    streaming_mv = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

    st.markdown("#### Billing")
    payment    = st.selectbox("Payment Method", [
                    "Electronic check","Mailed check",
                    "Bank transfer (automatic)","Credit card (automatic)"])
    paperless  = st.selectbox("Paperless Billing", ["Yes","No"])

    st.markdown("#### Demographics")
    gender     = st.selectbox("Gender", ["Male","Female"])
    senior     = st.selectbox("Senior Citizen", ["No","Yes"])
    partner    = st.selectbox("Has Partner", ["Yes","No"])
    dependents = st.selectbox("Has Dependents", ["Yes","No"])
    phone_svc  = st.selectbox("Phone Service", ["Yes","No"])
    multi_lines= st.selectbox("Multiple Lines", ["Yes","No","No phone service"])

    st.markdown("---")
    predict_btn = st.button("Run Prediction →", type="primary")

# ── Feature Builder ───────────────────────────────────────────────
def build_features():
    row    = pd.DataFrame(0, index=[0], columns=X_test.columns)
    scaler = joblib.load('scaler.pkl')
    scaled = scaler.transform([[tenure, monthly, total_ch]])
    row['tenure']           = scaled[0][0]
    row['MonthlyCharges']   = scaled[0][1]
    row['TotalCharges']     = scaled[0][2]
    row['SeniorCitizen']    = 1 if senior=="Yes" else 0
    row['gender']           = 1 if gender=="Male" else 0
    row['Partner']          = 1 if partner=="Yes" else 0
    row['Dependents']       = 1 if dependents=="Yes" else 0
    row['PaperlessBilling'] = 1 if paperless=="Yes" else 0
    row['PhoneService']     = 1 if phone_svc=="Yes" else 0

    def s(col):
        if col in row.columns: row[col] = 1

    if contract=="Month-to-month": s('Contract_Month-to-month')
    elif contract=="One year":     s('Contract_One year')
    else:                          s('Contract_Two year')

    if multi_lines=="Yes":             s('MultipleLines_Yes')
    elif multi_lines=="No":            s('MultipleLines_No')
    else:                              s('MultipleLines_No phone service')

    if internet=="Fiber optic":    s('InternetService_Fiber optic')
    elif internet=="DSL":          s('InternetService_DSL')
    else:                          s('InternetService_No')

    for feat, val in [(online_sec,'OnlineSecurity'),(online_bk,'OnlineBackup'),
                      (device_pr,'DeviceProtection'),(tech_sup,'TechSupport'),
                      (streaming_tv,'StreamingTV'),(streaming_mv,'StreamingMovies')]:
        if feat=="Yes":                  s(f'{val}_Yes')
        elif feat=="No":                 s(f'{val}_No')
        else:                            s(f'{val}_No internet service')

    s(f'PaymentMethod_{payment}')
    return row

# ── Header ────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:6px">
    <span style="font-size:11px;font-weight:700;color:#2563EB;text-transform:uppercase;
          letter-spacing:0.1em">Predictive Analytics</span>
    <h1 style="color:#0F172A;font-size:24px;font-weight:700;margin:4px 0 2px;line-height:1.2">
        Customer Churn Risk Assessment
    </h1>
    <p style="color:#64748B;font-size:13px;margin:0">
        XGBoost model · SHAP explainability · 7,043 training records · AUC-ROC 0.845
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

k1,k2,k3,k4 = st.columns(4)
k1.metric("Model AUC-ROC",      "0.845",  "↑ above 0.80 benchmark")
k2.metric("Test Accuracy",      "80.3%",  "Held-out test set")
k3.metric("Churners Identified","200",    "of 374 in test set")
k4.metric("Est. Revenue Saved", "$443K",  "Annual, 30% retention")

st.divider()

# ── Landing ───────────────────────────────────────────────────────
if not predict_btn:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:22px;
             box-shadow:0 1px 3px rgba(0,0,0,0.05)">
            <div style="font-size:11px;font-weight:700;color:#2563EB;text-transform:uppercase;
                 letter-spacing:0.08em;margin-bottom:10px">Model architecture</div>
            <table style="width:100%;font-size:13px;border-collapse:collapse;background:transparent">
                <tr><td style="padding:8px 0;color:#64748B;border-bottom:1px solid #F1F5F9">Algorithm</td>
                    <td style="padding:8px 0;color:#0F172A;font-weight:600;text-align:right;border-bottom:1px solid #F1F5F9">XGBoost Classifier</td></tr>
                <tr><td style="padding:8px 0;color:#64748B;border-bottom:1px solid #F1F5F9">Explainability</td>
                    <td style="padding:8px 0;color:#0F172A;font-weight:600;text-align:right;border-bottom:1px solid #F1F5F9">SHAP TreeExplainer</td></tr>
                <tr><td style="padding:8px 0;color:#64748B;border-bottom:1px solid #F1F5F9">Training records</td>
                    <td style="padding:8px 0;color:#0F172A;font-weight:600;text-align:right;border-bottom:1px solid #F1F5F9">5,634 (80%)</td></tr>
                <tr><td style="padding:8px 0;color:#64748B;border-bottom:1px solid #F1F5F9">Test records</td>
                    <td style="padding:8px 0;color:#0F172A;font-weight:600;text-align:right;border-bottom:1px solid #F1F5F9">1,409 (20%)</td></tr>
                <tr><td style="padding:8px 0;color:#64748B;border-bottom:1px solid #F1F5F9">AUC-ROC</td>
                    <td style="padding:8px 0;color:#2563EB;font-weight:700;text-align:right;border-bottom:1px solid #F1F5F9">0.845</td></tr>
                <tr><td style="padding:8px 0;color:#64748B">Features</td>
                    <td style="padding:8px 0;color:#0F172A;font-weight:600;text-align:right">20 raw → 30+ encoded</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#fff;border:1px solid #E2E8F0;border-radius:14px;padding:22px;
             box-shadow:0 1px 3px rgba(0,0,0,0.05)">
            <div style="font-size:11px;font-weight:700;color:#2563EB;text-transform:uppercase;
                 letter-spacing:0.08em;margin-bottom:14px">Top churn drivers · SHAP (global)</div>
        """, unsafe_allow_html=True)

        drivers = [
            ("Month-to-month contract", 0.744, 90),
            ("Customer tenure",         0.453, 55),
            ("Monthly charges",         0.274, 34),
            ("No online security",      0.251, 31),
            ("Fiber optic internet",    0.182, 22),
            ("No tech support",         0.174, 21),
        ]
        for label, score, pct in drivers:
            st.markdown(f"""
            <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;margin-bottom:4px">
                    <span style="font-size:12px;color:#374151">{label}</span>
                    <span style="font-size:12px;font-weight:600;color:#2563EB">{score}</span>
                </div>
                <div style="height:5px;background:#F1F5F9;border-radius:3px">
                    <div style="width:{pct}%;height:100%;background:#2563EB;border-radius:3px"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:12px;
         padding:14px 18px;margin-top:16px;display:flex;align-items:center;gap:12px">
        <div style="font-size:20px">👈</div>
        <div>
            <div style="font-size:13px;font-weight:600;color:#1E40AF">
                Configure the customer profile in the sidebar and click Run Prediction
            </div>
            <div style="font-size:12px;color:#3B82F6;margin-top:2px">
                All 20 features are used — fill every field for the most accurate prediction
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────
else:
    features = build_features()
    prob     = float(model.predict_proba(features)[0][1])

    if prob < 0.3:
        risk_label  = "Low Risk"
        risk_color  = "#059669"
        risk_bg     = "#ECFDF5"
        risk_border = "#6EE7B7"
        risk_badge  = "#D1FAE5"
        risk_text   = "#065F46"
        dot         = "#059669"
    elif prob < 0.6:
        risk_label  = "Medium Risk"
        risk_color  = "#D97706"
        risk_bg     = "#FFFBEB"
        risk_border = "#FCD34D"
        risk_badge  = "#FEF3C7"
        risk_text   = "#92400E"
        dot         = "#D97706"
    else:
        risk_label  = "High Risk"
        risk_color  = "#DC2626"
        risk_bg     = "#FEF2F2"
        risk_border = "#FCA5A5"
        risk_badge  = "#FEE2E2"
        risk_text   = "#991B1B"
        dot         = "#DC2626"

    # ── Result banner ─────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{risk_bg};border:1px solid {risk_border};border-radius:14px;
         padding:20px 24px;margin-bottom:20px;display:flex;
         align-items:center;justify-content:space-between">
        <div style="display:flex;align-items:center;gap:20px">
            <div>
                <div style="display:inline-flex;align-items:center;gap:6px;background:{risk_badge};
                     padding:4px 12px;border-radius:20px;margin-bottom:8px">
                    <div style="width:7px;height:7px;border-radius:50%;background:{dot}"></div>
                    <span style="font-size:11px;font-weight:700;color:{risk_text};
                          text-transform:uppercase;letter-spacing:0.06em">{risk_label}</span>
                </div>
                <div style="font-size:42px;font-weight:800;color:{risk_color};line-height:1">{prob*100:.1f}%</div>
                <div style="font-size:12px;color:#64748B;margin-top:4px">Predicted churn probability</div>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px 24px">
            <div><div style="font-size:11px;color:#94A3B8">Contract</div>
                 <div style="font-size:13px;font-weight:600;color:#0F172A">{contract}</div></div>
            <div><div style="font-size:11px;color:#94A3B8">Tenure</div>
                 <div style="font-size:13px;font-weight:600;color:#0F172A">{tenure} months</div></div>
            <div><div style="font-size:11px;color:#94A3B8">Monthly charges</div>
                 <div style="font-size:13px;font-weight:600;color:#0F172A">${monthly}</div></div>
            <div><div style="font-size:11px;color:#94A3B8">Internet</div>
                 <div style="font-size:13px;font-weight:600;color:#0F172A">{internet}</div></div>
            <div><div style="font-size:11px;color:#94A3B8">Security</div>
                 <div style="font-size:13px;font-weight:600;color:#0F172A">{online_sec}</div></div>
            <div><div style="font-size:11px;color:#94A3B8">Payment</div>
                 <div style="font-size:13px;font-weight:600;color:#0F172A">{payment.split()[0]}</div></div>
        </div>
        <div style="text-align:center">
            <div style="font-size:11px;color:#94A3B8;margin-bottom:6px">Annual revenue at risk</div>
            <div style="font-size:28px;font-weight:700;color:{risk_color}">${monthly*12:,}</div>
            <div style="font-size:11px;color:#94A3B8;margin-top:2px">if customer churns</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SHAP + Performance ────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div style="font-size:11px;font-weight:700;color:#2563EB;text-transform:uppercase;
             letter-spacing:0.08em;margin-bottom:4px">Explainability</div>
        <div style="font-size:15px;font-weight:600;color:#0F172A;margin-bottom:2px">
            Why is this customer at risk?
        </div>
        <div style="font-size:12px;color:#64748B;margin-bottom:14px">
            Red bars increase churn risk · Green bars reduce it
        </div>
        """, unsafe_allow_html=True)

        shap_vals   = explainer.shap_values(features)
        shap_series = pd.Series(shap_vals[0], index=X_test.columns)
        top_idx     = shap_series.abs().sort_values(ascending=False).head(8).index
        top_shap    = shap_series[top_idx]

        clean = {
            'Contract_Month-to-month':       'Month-to-month contract',
            'tenure':                         'Customer tenure',
            'MonthlyCharges':                 'Monthly charges',
            'OnlineSecurity_No':              'No online security',
            'InternetService_Fiber optic':    'Fiber optic internet',
            'TechSupport_No':                 'No tech support',
            'PaymentMethod_Electronic check': 'Electronic check payment',
            'TotalCharges':                   'Total charges',
            'Contract_Two year':              'Two-year contract',
            'PaperlessBilling':               'Paperless billing',
            'OnlineBackup_No':                'No online backup',
            'DeviceProtection_No':            'No device protection',
            'SeniorCitizen':                  'Senior citizen',
            'InternetService_DSL':            'DSL internet',
            'StreamingTV_Yes':                'Streaming TV',
            'StreamingMovies_Yes':            'Streaming movies',
        }
        labels = [clean.get(f, f) for f in top_shap.index]
        colors = ['#DC2626' if v > 0 else '#059669' for v in top_shap.values]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#F8F9FB')

        ax.barh(labels, top_shap.values, color=colors, height=0.55)
        ax.axvline(0, color='#CBD5E1', linewidth=1, linestyle='-')

        ax.set_xlabel('SHAP value', color='#64748B', fontsize=10)
        ax.tick_params(colors='#374151', labelsize=9)
        ax.set_title('Feature impact on churn prediction',
                     color='#0F172A', fontsize=12, pad=10, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_color('#E2E8F0')
            spine.set_linewidth(0.8)

        legend_patches = [
            mpatches.Patch(color='#DC2626', label='Increases churn risk'),
            mpatches.Patch(color='#059669', label='Reduces churn risk'),
        ]
        ax.legend(handles=legend_patches, fontsize=9,
                  facecolor='#FFFFFF', edgecolor='#E2E8F0',
                  labelcolor='#374151', loc='lower right')

        ax.grid(axis='x', color='#F1F5F9', linewidth=0.8)
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("""
        <div style="font-size:11px;font-weight:700;color:#2563EB;text-transform:uppercase;
             letter-spacing:0.08em;margin-bottom:4px">Model performance</div>
        <div style="font-size:15px;font-weight:600;color:#0F172A;margin-bottom:12px">
            Validation metrics
        </div>
        <div style="background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px;margin-bottom:14px">
            <table style="width:100%;font-size:12px;border-collapse:collapse;background:transparent">
                <tr><td style="padding:7px 0;color:#64748B;border-bottom:1px solid #F1F5F9">AUC-ROC</td>
                    <td style="padding:7px 0;color:#2563EB;font-weight:700;text-align:right;border-bottom:1px solid #F1F5F9">0.845</td></tr>
                <tr><td style="padding:7px 0;color:#64748B;border-bottom:1px solid #F1F5F9">Accuracy</td>
                    <td style="padding:7px 0;color:#0F172A;font-weight:600;text-align:right;border-bottom:1px solid #F1F5F9">80.3%</td></tr>
                <tr><td style="padding:7px 0;color:#64748B;border-bottom:1px solid #F1F5F9">Precision (churn)</td>
                    <td style="padding:7px 0;color:#0F172A;font-weight:600;text-align:right;border-bottom:1px solid #F1F5F9">65.8%</td></tr>
                <tr><td style="padding:7px 0;color:#64748B;border-bottom:1px solid #F1F5F9">Recall (churn)</td>
                    <td style="padding:7px 0;color:#0F172A;font-weight:600;text-align:right;border-bottom:1px solid #F1F5F9">53.5%</td></tr>
                <tr><td style="padding:7px 0;color:#64748B;border-bottom:1px solid #F1F5F9">Churners caught</td>
                    <td style="padding:7px 0;color:#059669;font-weight:700;text-align:right;border-bottom:1px solid #F1F5F9">200 / 374</td></tr>
                <tr><td style="padding:7px 0;color:#64748B">False alarms</td>
                    <td style="padding:7px 0;color:#D97706;font-weight:600;text-align:right">104</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px;text-align:center">
            <div style="font-size:11px;color:#64748B;margin-bottom:6px;text-transform:uppercase;
                 letter-spacing:0.05em;font-weight:600">This customer</div>
            <div style="font-size:44px;font-weight:800;color:{risk_color};line-height:1">{prob*100:.1f}%</div>
            <div style="font-size:12px;color:#64748B;margin:6px 0 10px">churn probability</div>
            <div style="height:8px;background:#F1F5F9;border-radius:4px;overflow:hidden;margin-bottom:6px">
                <div style="width:{prob*100:.0f}%;height:100%;background:{risk_color};border-radius:4px;
                     transition:width 0.3s"></div>
            </div>
            <div style="display:flex;justify-content:space-between">
                <span style="font-size:10px;color:#94A3B8">0%</span>
                <span style="font-size:10px;color:#94A3B8">50%</span>
                <span style="font-size:10px;color:#94A3B8">100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Recommendations ───────────────────────────────────────────
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#2563EB;text-transform:uppercase;
         letter-spacing:0.08em;margin-bottom:4px">Action plan</div>
    <div style="font-size:15px;font-weight:600;color:#0F172A;margin-bottom:16px">
        Retention recommendations
    </div>
    """, unsafe_allow_html=True)

    def rec(col, icon, title, body, urgent=True):
        border = "#FCA5A5" if urgent else "#6EE7B7"
        badge_bg = "#FEE2E2" if urgent else "#D1FAE5"
        badge_col = "#991B1B" if urgent else "#065F46"
        label = "Action required" if urgent else "No action needed"
        col.markdown(f"""
        <div style="background:#fff;border:1px solid {border};border-radius:12px;
             padding:16px;height:100%;box-shadow:0 1px 2px rgba(0,0,0,0.04)">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
                <span style="font-size:18px">{icon}</span>
                <span style="font-size:10px;font-weight:700;padding:2px 8px;border-radius:20px;
                      background:{badge_bg};color:{badge_col}">{label}</span>
            </div>
            <div style="font-size:13px;font-weight:600;color:#0F172A;margin-bottom:6px">{title}</div>
            <div style="font-size:12px;color:#64748B;line-height:1.6">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    r1,r2,r3 = st.columns(3)
    with r1:
        if contract=="Month-to-month":
            rec(r1,"📋","Migrate to annual plan",
                "Month-to-month customers churn at 3× the rate of annual subscribers. Offer 10–15% discount to convert.",
                urgent=True)
        else:
            rec(r1,"✅","Contract is stable",
                f"Customer is on a {contract.lower()} contract — significantly lower churn risk from this factor.",
                urgent=False)
    with r2:
        if tenure<12:
            rec(r2,"🕐",f"Early tenure — {tenure}-month window",
                "New customers are highest risk. Assign a customer success rep. Schedule proactive check-ins at Day 7, 30, 90.",
                urgent=True)
        else:
            rec(r2,"✅","Established customer",
                f"{tenure} months tenure — churn risk from this factor is low. Loyalty is building.",
                urgent=False)
    with r3:
        if online_sec=="No" or tech_sup=="No":
            missing=[]
            if online_sec=="No": missing.append("online security")
            if tech_sup=="No":   missing.append("tech support")
            rec(r3,"🛡️","Upsell protective services",
                f"No {' or '.join(missing)} detected. Customers with both services churn 2× less. Offer 30-day free trial.",
                urgent=True)
        else:
            rec(r3,"✅","Services adopted",
                "Customer has online security and tech support — strong retention anchors in place.",
                urgent=False)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)
    r4,r5,r6 = st.columns(3)
    with r4:
        if payment=="Electronic check":
            rec(r4,"💳","Switch payment method",
                "Electronic check correlates with higher churn. Encourage auto-pay via bank transfer or credit card.",
                urgent=True)
        else:
            rec(r4,"✅","Payment method stable",
                f"{payment} is associated with lower churn. Customer shows strong billing commitment.",
                urgent=False)
    with r5:
        if monthly>65 and contract=="Month-to-month":
            rec(r5,"💰","High-value customer at risk",
                f"${monthly}/mo on a flexible plan — highest risk segment. Prioritise for personalised retention offer.",
                urgent=True)
        else:
            rec(r5,"✅","Charge level manageable",
                f"${monthly}/mo on this contract type is within acceptable risk range.",
                urgent=False)
    with r6:
        if senior=="Yes":
            rec(r6,"👴","Senior citizen — support priority",
                "Senior customers may benefit from a dedicated helpline, simplified billing, or in-person service options.",
                urgent=True)
        else:
            rec(r6,"✅","Standard demographic",
                "No elevated churn risk from demographic profile. Standard engagement strategy applies.",
                urgent=False)

    # ── Consulting Brief ──────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#2563EB;text-transform:uppercase;
         letter-spacing:0.08em;margin-bottom:8px">Executive summary</div>
    """, unsafe_allow_html=True)

    actions = []
    if contract=="Month-to-month": actions.append("migrate to annual plan")
    if tenure<12:                  actions.append("90-day onboarding programme")
    if online_sec=="No" or tech_sup=="No": actions.append("upsell security/support services")
    if payment=="Electronic check": actions.append("switch to auto-pay")
    if not actions:                actions.append("maintain current engagement strategy")

    st.markdown(f"""
    <div style="background:#fff;border-left:4px solid #2563EB;border-radius:0 12px 12px 0;
         padding:16px 20px;box-shadow:0 1px 3px rgba(0,0,0,0.05)">
        <p style="color:#374151;font-size:13px;line-height:1.8;margin:0">
            This customer presents a
            <strong style="color:{risk_color}">{risk_label.lower()} ({prob*100:.1f}%)</strong>
            churn probability, driven primarily by their
            <strong style="color:#0F172A">{contract} contract</strong>,
            <strong style="color:#0F172A">{tenure}-month tenure</strong>, and
            <strong style="color:#0F172A">${monthly}/month billing</strong>.
            Annual revenue at risk: <strong style="color:{risk_color}">${monthly*12:,}</strong>.
            Recommended actions: <strong style="color:#0F172A">{", ".join(actions)}</strong>.
            Estimated churn reduction if actions taken: <strong style="color:#059669">15–25%</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)