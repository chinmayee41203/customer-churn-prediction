"""
Customer Churn Prediction — Streamlit Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

*, html, body { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'IBM Plex Mono', monospace;
    background-color: #080a0f;
    color: #c9d1d9;
}
h1, h2, h3, h4 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.08em;
    color: #ffffff;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
div[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 14px 18px;
}
div[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
    color: #f0c040 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    color: #6e7681 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
div[data-testid="stFormSubmitButton"] button {
    background: #f0c040 !important;
    color: #080a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.1em;
    border: none !important;
    border-radius: 4px !important;
    width: 100% !important;
}
hr { border-color: #21262d !important; }
.tag {
    display: inline-block;
    background: #21262d;
    color: #8b949e;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin-right: 4px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #21262d;
    padding-bottom: 6px;
    margin-bottom: 14px;
}
.prob-display {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 28px 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE / 'models'
REPORTS_DIR = BASE / 'reports'

DARK_BG  = '#0d1117'
GRID_COL = '#21262d'
TEXT_COL = '#c9d1d9'
ACCENT   = '#f0c040'
RED      = '#f85149'
GREEN    = '#3fb950'
BLUE     = '#58a6ff'
ORANGE   = '#d29922'

def dark_fig(w=8, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.title.set_color(TEXT_COL)
    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.5)
    return fig, ax

# ── Load Artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load(MODELS_DIR / 'xgboost.pkl')
    scaler    = joblib.load(MODELS_DIR / 'scaler.pkl')
    explainer = joblib.load(MODELS_DIR / 'shap_explainer.pkl')
    X_train, X_test, y_train, y_test = joblib.load(MODELS_DIR / 'train_test_split.pkl')
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test  = X_test.apply(pd.to_numeric,  errors='coerce').fillna(0)
    return model, scaler, explainer, X_test, y_test

@st.cache_data
def load_risk_scores():
    p = REPORTS_DIR / 'customer_risk_scores.csv'
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_model_comparison():
    p = REPORTS_DIR / 'model_comparison.csv'
    return pd.read_csv(p) if p.exists() else None

try:
    model, scaler, explainer, X_test, y_test = load_artifacts()
    LOADED = True
except Exception as e:
    LOADED = False

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## CHURN PREDICTION DASHBOARD")
    st.markdown("<p style='color:#6e7681;font-size:0.72rem;margin-top:-10px'>Telco · XGBoost · SHAP</p>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("nav", ["🏠  Overview","🔮  Predict","📊  Performance","🚨  Risk Board"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("""
    <p style='color:#6e7681;font-size:0.7rem;line-height:1.8'>
    Dataset · Telco IBM<br>Records · 7,043<br>Model · XGBoost<br>CV AUC · 0.9194<br><br>
    </p>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("# CUSTOMER CHURN PREDICTION")
    st.markdown("<p class='tag'>XGBoost</p><p class='tag'>SHAP</p><p class='tag'>Streamlit</p><p class='tag'>7043 customers</p><br>", unsafe_allow_html=True)
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset",    "7,043",  "customers")
    c2.metric("Churn Rate", "26.5%",  "imbalanced")
    c3.metric("CV AUC",     "0.9194", "XGBoost")
    c4.metric("Features",   "27",     "engineered")

    st.divider()
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<p class="section-label">Pipeline</p>', unsafe_allow_html=True)
        phases = [
            ("01","EDA",                 "Missing values · Distributions · Correlations"),
            ("02","Data Cleaning",       "Drop leakage · Fix dtypes · Handle nulls"),
            ("03","Feature Engineering", "Encode · Scale · 5 new features"),
            ("04","Model Training",      "LR · RF · XGBoost · LightGBM + SMOTE"),
            ("05","Evaluation",          "ROC · PR Curve · Threshold tuning"),
            ("06","SHAP",                "Global + local explanations · Risk tiers"),
            ("07","Dashboard",           "Streamlit · Live prediction · Risk board"),
        ]
        for num, name, desc in phases:
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:12px;margin-bottom:10px;'>
                <span style='font-family:Bebas Neue,sans-serif;font-size:1.1rem;color:{ACCENT};min-width:28px'>{num}</span>
                <div>
                    <span style='color:#ffffff;font-size:0.82rem'>{name}</span><br>
                    <span style='color:#6e7681;font-size:0.7rem'>{desc}</span>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown('<p class="section-label">Model Leaderboard</p>', unsafe_allow_html=True)
        comp_df = load_model_comparison()
        if comp_df is not None:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.dataframe(pd.DataFrame({
                'Model'  : ['XGBoost','LightGBM','Logistic Regression','Random Forest'],
                'CV AUC' : [0.9194, 0.9186, 0.9149, 0.9057],
                'CV F1'  : [0.8509, 0.8512, 0.8378, 0.8334],
            }), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown('<p class="section-label">Key EDA Findings</p>', unsafe_allow_html=True)
        findings = [
            ("📋","Contract",    "Month-to-month → 43% churn vs 11% two-year"),
            ("🌐","Internet",    "Fiber optic → ~42% churn rate"),
            ("⏱️","Tenure",      "First 12 months → highest churn risk"),
            ("💳","Payment",     "Electronic check → highest churn of all methods"),
            ("🔒","No Security", "Customers without online security churn more"),
        ]
        for icon, label, detail in findings:
            st.markdown(f"""
            <div style='display:flex;gap:10px;margin-bottom:8px;align-items:flex-start'>
                <span>{icon}</span>
                <div>
                    <span style='color:#ffffff;font-size:0.8rem'>{label}</span>
                    <span style='color:#6e7681;font-size:0.75rem'> — {detail}</span>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ══════════════════════════════════════════════════════════════════════
elif page == "🔮  Predict":
    st.markdown("# LIVE CHURN PREDICTION")
    st.markdown("<p style='color:#6e7681;font-size:0.8rem'>Enter customer details → instant churn probability + SHAP explanation</p>", unsafe_allow_html=True)
    st.divider()

    if not LOADED:
        st.error("Model artifacts not found. Run notebooks 01–06 first.")
        st.stop()

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown('<p class="section-label">Customer Details</p>', unsafe_allow_html=True)
        with st.form("pred_form"):
            c1, c2 = st.columns(2)
            with c1:
                tenure   = st.slider("Tenure (months)", 0, 72, 12)
                monthly  = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
                contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
                internet = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
                payment  = st.selectbox("Payment Method", [
                    "Electronic check","Mailed check",
                    "Bank transfer (automatic)","Credit card (automatic)"])
            with c2:
                gender     = st.selectbox("Gender", ["Male","Female"])
                senior     = st.selectbox("Senior Citizen", ["No","Yes"])
                partner    = st.selectbox("Partner", ["Yes","No"])
                dependents = st.selectbox("Dependents", ["No","Yes"])
                paperless  = st.selectbox("Paperless Billing", ["Yes","No"])

            st.markdown("<p class='section-label' style='margin-top:12px'>Services</p>", unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                phone       = st.checkbox("Phone Service",    value=True)
                multi_lines = st.checkbox("Multiple Lines")
                online_sec  = st.checkbox("Online Security")
                online_back = st.checkbox("Online Backup")
            with sc2:
                device_prot = st.checkbox("Device Protection")
                tech_sup    = st.checkbox("Tech Support")
                stream_tv   = st.checkbox("Streaming TV")
                stream_mov  = st.checkbox("Streaming Movies")

            submitted = st.form_submit_button("PREDICT CHURN", use_container_width=True)

    with col_result:
        st.markdown('<p class="section-label">Result</p>', unsafe_allow_html=True)
        if submitted:
            contract_map  = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            total_charges = monthly * (tenure + 1)
            num_services  = sum([phone, multi_lines, online_sec, online_back,
                                  device_prot, tech_sup, stream_tv, stream_mov])
            charges_pm    = total_charges / (tenure + 1)
            tenure_group  = int(pd.cut([tenure], bins=[0,12,24,48,72],
                                        labels=[0,1,2,3], include_lowest=True)[0])
            raw = {
                'Gender'             : 1 if gender == 'Male' else 0,
                'Senior Citizen'     : 1 if senior == 'Yes' else 0,
                'Partner'            : 1 if partner == 'Yes' else 0,
                'Dependents'         : 1 if dependents == 'Yes' else 0,
                'Tenure Months'      : tenure,
                'Phone Service'      : int(phone),
                'Multiple Lines'     : int(multi_lines),
                'Online Security'    : int(online_sec),
                'Online Backup'      : int(online_back),
                'Device Protection'  : int(device_prot),
                'Tech Support'       : int(tech_sup),
                'Streaming TV'       : int(stream_tv),
                'Streaming Movies'   : int(stream_mov),
                'Contract'           : contract_map[contract],
                'Paperless Billing'  : 1 if paperless == 'Yes' else 0,
                'Monthly Charges'    : monthly,
                'Total Charges'      : total_charges,
                'Charges Per Month'  : charges_pm,
                'Num Services'       : num_services,
                'Tenure Group'       : tenure_group,
                'High Value Customer': 1 if monthly > 79.0 else 0,
                'Auto Payment'       : 1 if 'automatic' in payment else 0,
                'Internet Service_DSL'                     : 1 if internet == 'DSL' else 0,
                'Internet Service_Fiber optic'             : 1 if internet == 'Fiber optic' else 0,
                'Internet Service_No'                      : 1 if internet == 'No' else 0,
                'Payment Method_Bank transfer (automatic)' : 1 if payment == 'Bank transfer (automatic)' else 0,
                'Payment Method_Credit card (automatic)'   : 1 if payment == 'Credit card (automatic)' else 0,
                'Payment Method_Electronic check'          : 1 if payment == 'Electronic check' else 0,
                'Payment Method_Mailed check'              : 1 if payment == 'Mailed check' else 0,
            }
            input_df = pd.DataFrame([raw])
            for col in X_test.columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[X_test.columns]
            num_cols = ['Tenure Months','Monthly Charges','Total Charges','Charges Per Month']
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            prob = model.predict_proba(input_df)[0, 1]
            pred = int(prob >= 0.5)

            if prob >= 0.6:
                tier_label, tier_color = "HIGH RISK",    RED
            elif prob >= 0.3:
                tier_label, tier_color = "MEDIUM RISK",  ORANGE
            else:
                tier_label, tier_color = "LOW RISK",     GREEN

            st.markdown(f"""
            <div class="prob-display">
                <p style='font-size:0.7rem;color:#6e7681;text-transform:uppercase;letter-spacing:0.15em;margin:0'>Churn Probability</p>
                <p style='font-family:Bebas Neue,sans-serif;font-size:4.5rem;color:{tier_color};margin:4px 0;line-height:1'>{prob:.1%}</p>
                <p style='font-family:Bebas Neue,sans-serif;font-size:1.4rem;color:{tier_color};margin:0;letter-spacing:0.1em'>{tier_label}</p>
                <p style='font-size:0.75rem;color:#8b949e;margin-top:8px'>{"⚠️ Will Churn" if pred else "✅ Will Stay"} &nbsp;·&nbsp; {num_services} services subscribed</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("<p class='section-label' style='margin-top:20px'>Why this prediction?</p>", unsafe_allow_html=True)
            shap_vals = explainer.shap_values(input_df)
            explanation = shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=X_test.columns.tolist()
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor(DARK_BG)
            shap.waterfall_plot(explanation, show=False, max_display=10)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.markdown(f"""
            <div style='background:{DARK_BG};border:1px dashed {GRID_COL};border-radius:8px;
                        padding:60px 20px;text-align:center;color:#6e7681;margin-top:10px'>
                <p style='font-family:Bebas Neue,sans-serif;font-size:3rem;color:{GRID_COL}'>🔮</p>
                <p style='font-size:0.8rem'>Fill in the form and click PREDICT CHURN</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════
elif page == "📊  Performance":
    st.markdown("# MODEL PERFORMANCE")
    st.divider()

    if not LOADED:
        st.error("Model artifacts not found.")
        st.stop()

    from sklearn.metrics import (
        confusion_matrix, classification_report, roc_curve, auc,
        precision_recall_curve, ConfusionMatrixDisplay,
        f1_score, precision_score, recall_score, roc_auc_score
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC",   f"{roc_auc_score(y_test, y_prob):.4f}")
    c2.metric("F1 Score",  f"{f1_score(y_test, y_pred):.4f}")
    c3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    c4.metric("Recall",    f"{recall_score(y_test, y_pred):.4f}")

    st.divider()
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<p class="section-label">Confusion Matrix</p>', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        ConfusionMatrixDisplay(cm, display_labels=['No Churn','Churn']).plot(
            ax=ax, cmap='YlOrRd', colorbar=False)
        ax.tick_params(colors=TEXT_COL)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        for text in ax.texts:
            text.set_color('#080a0f')
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<p class="section-label" style="margin-top:16px">Classification Report</p>', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred,
                    target_names=['No Churn','Churn'], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    with col_r:
        st.markdown('<p class="section-label">ROC Curves — All Models</p>', unsafe_allow_html=True)
        model_files = {
            'XGBoost'            : MODELS_DIR / 'xgboost.pkl',
            'LightGBM'           : MODELS_DIR / 'lightgbm.pkl',
            'Logistic Regression': MODELS_DIR / 'logistic_regression.pkl',
            'Random Forest'      : MODELS_DIR / 'random_forest.pkl',
        }
        fig, ax = dark_fig(6, 5)
        for (name, path), color in zip(model_files.items(), [ACCENT, BLUE, GREEN, ORANGE]):
            try:
                m = joblib.load(path)
                prob = m.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, prob)
                ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} ({auc(fpr,tpr):.4f})')
            except:
                pass
        ax.plot([0,1],[0,1],'--', color=GRID_COL, lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend(facecolor=DARK_BG, labelcolor=TEXT_COL, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<p class="section-label" style="margin-top:16px">Precision-Recall Curve</p>', unsafe_allow_html=True)
        prec_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
        fig, ax = dark_fig(6, 4)
        ax.plot(rec_vals, prec_vals, color=ACCENT, lw=2, label=f'PR-AUC = {auc(rec_vals,prec_vals):.4f}')
        ax.axhline(y_test.mean(), color=GRID_COL, linestyle='--', label=f'Baseline ({y_test.mean():.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(facecolor=DARK_BG, labelcolor=TEXT_COL, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.divider()
    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<p class="section-label">Threshold Tuning</p>', unsafe_allow_html=True)
        threshold_results = []
        for t in np.arange(0.2, 0.7, 0.05):
            y_pred_t = (y_prob >= t).astype(int)
            threshold_results.append({
                'Threshold': round(t, 2),
                'Precision': round(precision_score(y_test, y_pred_t), 4),
                'Recall'   : round(recall_score(y_test, y_pred_t), 4),
                'F1'       : round(f1_score(y_test, y_pred_t), 4),
            })
        thresh_df = pd.DataFrame(threshold_results)
        best_t = thresh_df.loc[thresh_df['F1'].idxmax(), 'Threshold']
        fig, ax = dark_fig(6, 4)
        ax.plot(thresh_df['Threshold'], thresh_df['Precision'], marker='o', color=BLUE,  label='Precision')
        ax.plot(thresh_df['Threshold'], thresh_df['Recall'],    marker='o', color=RED,   label='Recall')
        ax.plot(thresh_df['Threshold'], thresh_df['F1'],        marker='o', color=GREEN, label='F1')
        ax.axvline(0.5,    color=GRID_COL, linestyle='--', lw=1, label='Default 0.5')
        ax.axvline(best_t, color=ACCENT,   linestyle='--', lw=1, label=f'Best F1 @ {best_t}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Threshold Tuning')
        ax.legend(facecolor=DARK_BG, labelcolor=TEXT_COL, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.caption(f"Best threshold by F1: **{best_t}**")

    with col_b:
        st.markdown('<p class="section-label">Top 15 Feature Importances</p>', unsafe_allow_html=True)
        importance = pd.Series(
            model.feature_importances_, index=X_test.columns
        ).sort_values(ascending=False).head(15)
        fig, ax = dark_fig(6, 5)
        importance.sort_values().plot(kind='barh', ax=ax, color=ACCENT, edgecolor=DARK_BG, width=0.7)
        ax.set_xlabel('Importance Score')
        ax.set_title('XGBoost Feature Importance')
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — RISK BOARD
# ══════════════════════════════════════════════════════════════════════
elif page == "🚨  Risk Board":
    st.markdown("# CUSTOMER RISK BOARD")
    st.markdown("<p style='color:#6e7681;font-size:0.8rem'>All test customers scored and tiered by churn probability</p>", unsafe_allow_html=True)
    st.divider()

    risk_df = load_risk_scores()
    if risk_df is None:
        st.error("Run 06_SHAP.ipynb first to generate customer_risk_scores.csv")
        st.stop()

    high   = (risk_df['Risk Tier'] == 'High').sum()
    medium = (risk_df['Risk Tier'] == 'Medium').sum()
    low    = (risk_df['Risk Tier'] == 'Low').sum()
    total  = len(risk_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total",          total)
    c2.metric("🔴 High Risk",   high,   f"{high/total:.1%}")
    c3.metric("🟠 Medium Risk", medium, f"{medium/total:.1%}")
    c4.metric("🟢 Low Risk",    low,    f"{low/total:.1%}")

    st.divider()
    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown('<p class="section-label">Risk Tier Distribution</p>', unsafe_allow_html=True)
        tier_counts = risk_df['Risk Tier'].value_counts().reindex(['High','Medium','Low'])
        fig, ax = dark_fig(5, 3)
        bars = ax.bar(['High','Medium','Low'], tier_counts.values,
                      color=[RED, ORANGE, GREEN], edgecolor=DARK_BG, width=0.5)
        ax.set_ylabel('Customers')
        ax.set_title('Risk Tier Distribution')
        for bar, val in zip(bars, tier_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(val), ha='center', color=TEXT_COL, fontsize=9, fontweight='bold')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_r:
        st.markdown('<p class="section-label">Score Distribution</p>', unsafe_allow_html=True)
        fig, ax = dark_fig(5, 3)
        ax.hist(risk_df[risk_df['Actual Churn']==0]['Churn Probability'],
                bins=30, alpha=0.7, color=BLUE, label='No Churn', edgecolor=DARK_BG)
        ax.hist(risk_df[risk_df['Actual Churn']==1]['Churn Probability'],
                bins=30, alpha=0.7, color=RED,  label='Churned',  edgecolor=DARK_BG)
        ax.axvline(0.5, color=ACCENT, linestyle='--', lw=1.5, label='Threshold 0.5')
        ax.set_xlabel('Churn Probability')
        ax.set_ylabel('Count')
        ax.set_title('Score Distribution by Actual Label')
        ax.legend(facecolor=DARK_BG, labelcolor=TEXT_COL, fontsize=8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.divider()
    st.markdown('<p class="section-label">Customer Table</p>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        filter_tier = st.selectbox("Filter by Tier", ["All","High","Medium","Low"])
    with col_f2:
        prob_min, prob_max = st.slider("Probability Range", 0.0, 1.0, (0.0, 1.0), step=0.05)

    display_df = risk_df.copy()
    if filter_tier != "All":
        display_df = display_df[display_df['Risk Tier'] == filter_tier]
    display_df = display_df[
        (display_df['Churn Probability'] >= prob_min) &
        (display_df['Churn Probability'] <= prob_max)
    ]
    show_cols = [c for c in ['Churn Probability','Risk Tier','Actual Churn',
                 'Tenure Months','Monthly Charges','Contract',
                 'Num Services','SHAP Magnitude'] if c in display_df.columns]
    st.dataframe(
        display_df[show_cols].sort_values('Churn Probability', ascending=False)
        .head(25).reset_index(drop=True),
        use_container_width=True
    )
    st.caption(f"Showing {min(25, len(display_df))} of {len(display_df)} customers")
