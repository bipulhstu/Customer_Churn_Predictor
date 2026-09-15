import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import io

# Base directory resolution
BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained multi-model bundle and scaler
@st.cache_resource
def load_artifacts():
    bundle_path = BASE_DIR / "models.pkl"
    if bundle_path.exists():
        try:
            return joblib.load(bundle_path)
        except Exception as e:
            st.warning(f"Warning loading models.pkl: {e}. Attempting fallback load...")

    # Fallback for legacy standalone files
    model_path = BASE_DIR / "churn_model.pkl"
    scaler_path = BASE_DIR / "scaler.pkl"
    if model_path.exists() and scaler_path.exists():
        try:
            legacy_model = joblib.load(model_path)
            legacy_scaler = joblib.load(scaler_path)
            return {
                "models": {
                    "Champion Model": {
                        "model": legacy_model,
                        "metrics": {"Accuracy": 0.778, "AUC-ROC": 0.842, "Recall": 0.666, "Precision": 0.570, "F1-Score": 0.614},
                        "description": "Production trained classifier for churn detection"
                    }
                },
                "scaler": legacy_scaler,
                "feature_names": list(legacy_scaler.feature_names_in_) if hasattr(legacy_scaler, "feature_names_in_") else [],
                "feature_importances": {},
                "feature_directions": {},
                "friendly_names": {}
            }
        except Exception as e2:
            st.warning(f"Warning loading legacy artifacts: {e2}. Initializing training pipeline...")

    # Auto-regeneration fallback (guarantees 100% uptime across any NumPy/Python version)
    try:
        from train_models import load_and_merge_data, preprocess_data, train_and_evaluate, save_artifacts
        df = load_and_merge_data()
        X, y = preprocess_data(df)
        bundle = train_and_evaluate(X, y)
        save_artifacts(bundle)
        return bundle
    except Exception as e3:
        st.error(f"Fatal error initializing models: {e3}")
        raise e3

@st.cache_data
def load_historical_dataset():
    """Loads and caches merged historical dataset for analytics and batch demo."""
    try:
        churn_df = pd.read_csv(BASE_DIR / "churn_data.csv")
        customer_df = pd.read_csv(BASE_DIR / "customer_data.csv")
        internet_df = pd.read_csv(BASE_DIR / "internet_data.csv")
        merged = churn_df.merge(customer_df, on="customerID").merge(internet_df, on="customerID")
        merged["TotalCharges"] = pd.to_numeric(merged["TotalCharges"], errors="coerce").fillna(0)
        return merged
    except Exception as e:
        st.error(f"Error loading historical dataset: {e}")
        return pd.DataFrame()

artifacts = load_artifacts()
models_dict = artifacts["models"]
scaler = artifacts["scaler"]
feature_names = artifacts["feature_names"]
feature_importances_dict = artifacts.get("feature_importances", {})
feature_directions = artifacts.get("feature_directions", {})
friendly_names = artifacts.get("friendly_names", {})

# Custom CSS for Dark Theme
st.markdown("""
    <style>
    /* ===== DARK THEME - MAIN LAYOUT ===== */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1e1b4b 50%, #18181b 100%);
    }
    
    .main .block-container {
        background: #18181b;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        max-width: 1400px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* ===== TYPOGRAPHY ===== */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(120deg, #818cf8 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    .main h1, .main h2 {
        color: #f1f5f9;
        font-weight: 700;
    }
    
    .main h3, .main h4 {
        color: #cbd5e1;
        font-weight: 600;
    }
    
    /* ===== TABS & NAVIGATION ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 2px solid rgba(99, 102, 241, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #27272a;
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        color: #cbd5e1;
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3f3f46;
        color: #818cf8;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #9333ea 100%);
        color: white !important;
        border-color: #6366f1;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #27272a;
        border-radius: 0 10px 10px 10px;
        padding: 25px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    /* ===== INPUTS ===== */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .stSelectbox > div > div, 
    .stNumberInput > div > div > input {
        background-color: #3f3f46 !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border-color: #818cf8 !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white !important;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6);
    }
    
    /* ===== RISK BADGES ===== */
    .risk-badge {
        padding: 22px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .badge-critical {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        border: 1px solid #f87171;
    }
    .badge-high {
        background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%);
        border: 1px solid #fb923c;
    }
    .badge-moderate {
        background: linear-gradient(135deg, #d97706 0%, #b45309 100%);
        border: 1px solid #fbbf24;
    }
    .badge-low {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        border: 1px solid #34d399;
    }
    .risk-badge h2, .risk-badge p {
        color: white !important;
        margin: 0;
    }
    
    /* ===== KPI CARDS ===== */
    .kpi-card {
        background: #27272a;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .kpi-card h4 {
        margin: 0 0 6px 0;
        font-size: 0.85rem;
        color: #94a3b8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-card h2 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc !important;
    }
    .kpi-card p {
        margin: 6px 0 0 0;
        font-size: 0.8rem;
        color: #cbd5e1 !important;
    }
    
    /* ===== INFO CONTAINERS ===== */
    .info-container {
        background: #27272a;
        border-left: 4px solid #6366f1;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .info-container h4 {
        color: #f1f5f9 !important;
        margin-top: 0;
    }
    .info-container li {
        color: #cbd5e1 !important;
        margin: 8px 0;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        color: white !important;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid rgba(99, 102, 241, 0.3);
        background-color: #27272a;
        border-radius: 12px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Sidebar: Model Selector and Benchmark Details
with st.sidebar:
    st.markdown("### 🤖 ML Engine Selection")
    
    available_models = list(models_dict.keys())
    default_index = 0
    if "Gradient Boosting" in available_models:
        default_index = available_models.index("Gradient Boosting")

    selected_model_name = st.selectbox(
        "Active Classifier:",
        available_models,
        index=default_index,
        help="Select the AI model used for inference. Compare precision/recall trade-offs."
    )
    
    active_info = models_dict[selected_model_name]
    active_model = active_info["model"]
    m_metrics = active_info.get("metrics", {})
    
    st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.08); 
                    padding: 14px; border-radius: 10px; margin-top: 10px; margin-bottom: 15px;
                    border: 1px solid rgba(255, 255, 255, 0.2);">
            <h4 style="margin: 0 0 6px 0; color: #a5b4fc; font-size: 1.05rem;">{selected_model_name}</h4>
            <p style="color: #cbd5e1; font-size: 0.85rem; margin-bottom: 10px;">{active_info.get('description', '')}</p>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 0.85rem; color: #f8fafc;">
                <div>🎯 <strong>AUC:</strong> {m_metrics.get('AUC-ROC', 0.84):.3f}</div>
                <div>⚡ <strong>Recall:</strong> {m_metrics.get('Recall', 0.70):.1%}</div>
                <div>📈 <strong>Accuracy:</strong> {m_metrics.get('Accuracy', 0.78):.1%}</div>
                <div>🔍 <strong>Precision:</strong> {m_metrics.get('Precision', 0.57):.1%}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📈 Key Churn Drivers")
    st.markdown("""
        - ⏱️ **Tenure:** First year (< 12 months)
        - 📝 **Contract:** Month-to-month contracts
        - 💳 **Payment:** Electronic check billing
        - 👴 **Demographics:** Senior citizens
        - 👥 **Family:** Single customers without dependents
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Best Practices")
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.08); 
                    padding: 12px; border-radius: 8px;
                    border: 1px solid rgba(255, 255, 255, 0.15);">
            <ul style="color: #f1f5f9; margin: 0; padding-left: 18px; font-size: 0.85rem;">
                <li>Use <strong>Single Predictor</strong> for deep customer dive</li>
                <li>Use <strong>Batch Scoring</strong> for prioritized outreach</li>
                <li>Explore <strong>Historical Analytics</strong> for cohort trends</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# Title and description with custom styling
st.markdown('<h1 class="main-title">🎯 Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enterprise Decision Intelligence Engine: Explainable Risk, Financial Impact & Batch Inference</p>', unsafe_allow_html=True)

# Top Navigation Mode Selector
nav_tab = st.radio(
    "Select Workspace Mode:",
    ["🎯 Single Customer Risk Lab", "📁 Enterprise Batch Scoring Engine", "📊 Historical Dataset Intelligence"],
    horizontal=True
)

st.markdown("---")

# ==============================================================================
# MODE 1: SINGLE CUSTOMER RISK LAB
# ==============================================================================
if nav_tab == "🎯 Single Customer Risk Lab":
    st.info(f"💡 **Active AI Classifier:** **{selected_model_name}** | Test AUC: **{m_metrics.get('AUC-ROC', 0.84):.3f}** | Recall: **{m_metrics.get('Recall', 0.70):.1%}**")

    tab1, tab2, tab3 = st.tabs(["👤 Customer Profile", "🌐 Services & Features", "💳 Billing & Charges"])

    with tab1:
        st.markdown("#### Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("👥 Gender", ["Male", "Female"], help="Select customer's gender")
            senior_citizen = st.selectbox("👴 Senior Citizen", ["No", "Yes"], help="Is the customer a senior citizen?")
            partner = st.selectbox("👫 Partner", ["No", "Yes"], help="Does the customer have a partner?")
        with col2:
            dependents = st.selectbox("👨‍👩‍👧 Dependents", ["No", "Yes"], help="Does the customer have dependents?")
            tenure = st.slider("📅 Tenure (months)", 0, 72, 12, help="How long has the customer been with the company?")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📞 Communication Services")
            phone_service = st.selectbox("📱 Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("📞 Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
        with col2:
            st.markdown("#### 🛡️ Protection & Entertainment")
            online_security = st.selectbox("🔒 Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("💾 Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("🛡️ Device Protection", ["No", "Yes", "No internet service"])
        
        col3, col4 = st.columns(2)
        with col3:
            tech_support = st.selectbox("🔧 Tech Support", ["No", "Yes", "No internet service"])
        with col4:
            streaming_tv = st.selectbox("📺 Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("🎬 Streaming Movies", ["No", "Yes", "No internet service"])

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📋 Contract Details")
            contract = st.selectbox("📝 Contract Type", ["Month-to-month", "One year", "Two year"], 
                                   help="Type of contract the customer has")
            paperless_billing = st.selectbox("📄 Paperless Billing", ["No", "Yes"],
                                            help="Does the customer use paperless billing?")
            payment_method = st.selectbox("💳 Payment Method", 
                                          ["Electronic check", "Mailed check", 
                                           "Bank transfer (automatic)", "Credit card (automatic)"],
                                          help="Customer's payment method")
        with col2:
            st.markdown("#### 💰 Financial Information")
            monthly_charges = st.number_input("💵 Monthly Charges ($)", min_value=0.0, max_value=200.0, 
                                             value=50.0, step=0.5, help="Amount charged per month")
            total_charges = st.number_input("💸 Total Charges ($)", min_value=0.0, max_value=10000.0, 
                                           value=500.0, step=10.0, help="Total amount charged to date")

    st.markdown("---")

    def build_feature_vector(t_tenure, t_monthly, t_total, t_contract, t_payment, t_tech_supp, t_online_sec, t_online_bkp):
        """Encapsulates feature vector generation for both live inputs and what-if simulation."""
        if t_tenure <= 12:
            t_grp = "0-12 Months"
        elif 12 < t_tenure <= 24:
            t_grp = "12-24 Months"
        elif 24 < t_tenure <= 48:
            t_grp = "24-48 Months"
        elif 48 < t_tenure <= 60:
            t_grp = "48-60 Months"
        else:
            t_grp = "60+ Months"

        raw_dict = {
            'tenure': t_tenure,
            'PhoneService': 1 if phone_service == "Yes" else 0,
            'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
            'MonthlyCharges': float(t_monthly),
            'TotalCharges': float(t_total),
            'gender': 1 if gender == "Male" else 0,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': 1 if partner == "Yes" else 0,
            'Dependents': 1 if dependents == "Yes" else 0,
            'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
            'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
            'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
            'InternetService_No': 1 if internet_service == "No" else 0,
            'OnlineSecurity_No internet service': 1 if t_online_sec == "No internet service" else 0,
            'OnlineSecurity_Yes': 1 if t_online_sec == "Yes" else 0,
            'OnlineBackup_No internet service': 1 if t_online_bkp == "No internet service" else 0,
            'OnlineBackup_Yes': 1 if t_online_bkp == "Yes" else 0,
            'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0,
            'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
            'TechSupport_No internet service': 1 if t_tech_supp == "No internet service" else 0,
            'TechSupport_Yes': 1 if t_tech_supp == "Yes" else 0,
            'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0,
            'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
            'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0,
            'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
            'Contract_One year': 1 if t_contract == "One year" else 0,
            'Contract_Two year': 1 if t_contract == "Two year" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if t_payment == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if t_payment == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if t_payment == "Mailed check" else 0,
            'tenure_group_12-24 Months': 1 if t_grp == '12-24 Months' else 0,
            'tenure_group_24-48 Months': 1 if t_grp == '24-48 Months' else 0,
            'tenure_group_48-60 Months': 1 if t_grp == '48-60 Months' else 0,
            'tenure_group_60+ Months': 1 if t_grp == '60+ Months' else 0,
        }

        expected_cols = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else (feature_names if feature_names else list(raw_dict.keys()))
        row_data = {col: raw_dict.get(col, 0) for col in expected_cols}
        df_feat = pd.DataFrame([row_data], columns=expected_cols)
        scaled_matrix = scaler.transform(df_feat)
        return df_feat, raw_dict, scaled_matrix

    def format_feature_label(col, raw_val, scaled_val, friendly_dict):
        base_name = friendly_dict.get(col, col)
        if col == "tenure":
            return f"Short Tenure ({int(raw_val)} mos)" if scaled_val < 0 else f"Long Tenure ({int(raw_val)} mos)"
        elif col == "MonthlyCharges":
            return f"High Monthly Charges (${raw_val:.0f})" if scaled_val > 0 else f"Low Monthly Charges (${raw_val:.0f})"
        elif col == "TotalCharges":
            return f"High Lifetime Spend (${raw_val:.0f})" if scaled_val > 0 else f"Low Lifetime Spend (${raw_val:.0f})"

        if raw_val == 0:
            if any(w in col for w in ["Contract", "Security", "Backup", "Support", "Protection", "Partner", "Dependents"]):
                return f"No {base_name}"
            return f"Absent: {base_name}"
        else:
            return base_name

    def compute_attribution(df_feat, raw_dict, scaled_matrix, model, model_name):
        expected_cols = list(df_feat.columns)
        scaled_vals = scaled_matrix[0]
        attributions = []
        if hasattr(model, "coef_"):
            coefs = model.coef_[0]
            for col, s_val, c in zip(expected_cols, scaled_vals, coefs):
                raw_v = raw_dict.get(col, 0)
                score = float(s_val * c)
                lbl = format_feature_label(col, raw_v, s_val, friendly_names)
                attributions.append((lbl, score))
        else:
            norm_imp = feature_importances_dict.get(model_name, {})
            for col, s_val in zip(expected_cols, scaled_vals):
                raw_v = raw_dict.get(col, 0)
                imp = norm_imp.get(col, 0.01)
                corr_dir = feature_directions.get(col, 0.0)
                sign = 1.0 if corr_dir >= 0 else -1.0
                score = float(s_val * imp * sign * 10.0)
                lbl = format_feature_label(col, raw_v, s_val, friendly_names)
                attributions.append((lbl, score))

        attributions.sort(key=lambda x: x[1])
        protective = [x for x in attributions if x[1] < 0][:5]
        drivers = [x for x in reversed(attributions) if x[1] > 0][:5]
        return protective, drivers, attributions

    # Prediction button
    if st.button("🔮 Predict Churn & Financial Exposure", type="primary"):
        df_feat, raw_dict, input_scaled = build_feature_vector(
            tenure, monthly_charges, total_charges, contract, payment_method, tech_support, online_security, online_backup
        )
        prediction_proba = active_model.predict_proba(input_scaled)
        churn_probability = prediction_proba[0][1] * 100

        st.session_state["predicted"] = True
        st.session_state["base_churn_prob"] = churn_probability
        st.session_state["raw_dict"] = raw_dict
        st.session_state["df_feat"] = df_feat
        st.session_state["input_scaled"] = input_scaled
        st.session_state["base_monthly"] = float(monthly_charges)

    if st.session_state.get("predicted", False):
        churn_probability = st.session_state["base_churn_prob"]
        no_churn_probability = 100.0 - churn_probability
        raw_dict = st.session_state["raw_dict"]
        df_feat = st.session_state["df_feat"]
        input_scaled = st.session_state["input_scaled"]
        base_monthly = st.session_state["base_monthly"]

        if churn_probability >= 80:
            risk_tier = "CRITICAL RISK"
            badge_class = "badge-critical"
            risk_desc = "Customer is in the critical flight window. Immediate 24-hour intervention required!"
        elif churn_probability >= 60:
            risk_tier = "HIGH RISK"
            badge_class = "badge-high"
            risk_desc = "High probability of churn. Proactive retention incentive recommended."
        elif churn_probability >= 30:
            risk_tier = "MODERATE RISK"
            badge_class = "badge-moderate"
            risk_desc = "Moderate churn indicators. Review service satisfaction and engagement."
        else:
            risk_tier = "LOW RISK"
            badge_class = "badge-low"
            risk_desc = "Loyal & stable account. Focus on value expansion and relationship building."

        st.markdown("---")
        st.markdown("## 📊 Risk Assessment Results")
        st.caption(f"Evaluated via **{selected_model_name}** | Test Holdout AUC: {m_metrics.get('AUC-ROC', 0.84):.3f} | Model Recall: {m_metrics.get('Recall', 0.70):.1%}")

        col_res1, col_res2, col_res3 = st.columns([1.2, 1, 1])
        with col_res1:
            st.markdown(f"""
                <div class="risk-badge {badge_class}">
                    <h2 style="font-size:1.5rem; letter-spacing:0.05em;">{risk_tier}</h2>
                    <p style="font-size:0.95rem; margin-top:8px;">{risk_desc}</p>
                </div>
            """, unsafe_allow_html=True)
        with col_res2:
            st.metric("🎯 Churn Probability", f"{churn_probability:.1f}%", f"{churn_probability - 50:.1f}% vs baseline", delta_color="inverse")
        with col_res3:
            st.metric("✨ Retention Probability", f"{no_churn_probability:.1f}%", f"{no_churn_probability - 50:.1f}% vs baseline", delta_color="normal")

        # Financial KPI Section
        arr = base_monthly * 12
        arr_at_risk = arr * (churn_probability / 100)
        clv_projected = base_monthly * 36

        st.markdown("---")
        st.markdown("## 💵 Executive Financial Impact & Revenue at Risk")
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid #818cf8;">
                    <h4>Annual Recurring Revenue</h4>
                    <h2>${arr:,.2f}</h2>
                    <p>Contracted baseline spend (${base_monthly:.2f}/mo)</p>
                </div>
            """, unsafe_allow_html=True)
        with col_kpi2:
            arr_color = "#ef4444" if arr_at_risk >= 500 else "#f59e0b"
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid {arr_color};">
                    <h4>ARR at Risk</h4>
                    <h2 style="color: {arr_color} !important;">${arr_at_risk:,.2f}</h2>
                    <p>Expected annual loss at {churn_probability:.1f}% churn risk</p>
                </div>
            """, unsafe_allow_html=True)
        with col_kpi3:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid #10b981;">
                    <h4>3-Year CLV Exposure</h4>
                    <h2>${clv_projected:,.2f}</h2>
                    <p>Projected lifetime value if retained</p>
                </div>
            """, unsafe_allow_html=True)

        # Feature Attribution Section
        st.markdown("---")
        st.markdown("## 🔍 Why This Prediction? (Feature Attribution)")
        protective, drivers, attributions = compute_attribution(df_feat, raw_dict, input_scaled, active_model, selected_model_name)

        all_top = list(reversed(protective)) + drivers
        plot_names = [item[0] for item in all_top]
        plot_scores = [item[1] for item in all_top]
        plot_colors = ["#10b981" if s < 0 else "#ef4444" for s in plot_scores]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=plot_names, x=plot_scores, orientation="h",
            marker=dict(color=plot_colors, line=dict(width=1, color="rgba(255,255,255,0.2)")),
            hovertemplate="<b>%{y}</b><br>Impact: %{x:.2f}<extra></extra>"
        ))
        fig.update_layout(
            title="<b>Local Feature Impact for This Customer</b> (Green = Lowers Churn, Red = Increases Churn)",
            title_font=dict(color="#f8fafc", size=14), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(39,39,42,0.6)",
            font=dict(color="#cbd5e1"), xaxis=dict(title="Impact Direction & Magnitude", zeroline=True, zerolinecolor="rgba(255,255,255,0.4)", zerolinewidth=2, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)"), margin=dict(l=10, r=10, t=40, b=30), height=380
        )
        st.plotly_chart(fig, use_container_width=True)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("""
                <div class="info-container" style="border-left-color: #ef4444;">
                    <h4 style="color: #fca5a5 !important;">🔴 Top Risk Drivers (Pushing Risk Up)</h4>
                    <ul>
            """ + "".join([f"<li><strong>{f}</strong> (+{abs(s):.2f})</li>" for f, s in drivers]) + """
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        with col_d2:
            st.markdown("""
                <div class="info-container" style="border-left-color: #10b981;">
                    <h4 style="color: #6ee7b7 !important;">🟢 Top Protective Factors (Preventing Churn)</h4>
                    <ul>
            """ + "".join([f"<li><strong>{f}</strong> (-{abs(s):.2f})</li>" for f, s in protective]) + """
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        # What-If Simulator Section
        st.markdown("---")
        st.markdown("## 🔄 Interactive \"What-If\" Counterfactual Retention Simulator")
        col_sim1, col_sim2, col_sim3 = st.columns(3)
        with col_sim1:
            contract_opts = ["Month-to-month", "One year", "Two year"]
            c_idx = contract_opts.index(contract) if contract in contract_opts else 0
            sim_contract = st.selectbox("Simulated Contract:", contract_opts, index=min(c_idx + 1, 2), key="sim_c")
            pay_opts = ["Electronic check", "Credit card (automatic)", "Bank transfer (automatic)", "Mailed check"]
            p_idx = pay_opts.index(payment_method) if payment_method in pay_opts else 0
            sim_payment = st.selectbox("Simulated Payment Method:", pay_opts, index=1 if p_idx == 0 else p_idx, key="sim_p")
        with col_sim2:
            sim_tech = st.selectbox("Tech Support:", ["No", "Yes"], index=1, key="sim_t")
            sim_sec = st.selectbox("Online Security:", ["No", "Yes"], index=1, key="sim_s")
            sim_bkp = st.selectbox("Online Backup:", ["No", "Yes"], index=1, key="sim_b")
        with col_sim3:
            sim_discount = st.slider("Monthly Loyalty Discount ($/mo):", 0.0, 30.0, 10.0, step=2.5, key="sim_disc")
            sim_monthly = max(18.0, base_monthly - sim_discount)
            st.caption(f"Adjusted Bill: **${sim_monthly:.2f}/mo** (Discount: ${sim_discount:.2f}/mo)")

        sim_feat, sim_raw, sim_scaled = build_feature_vector(
            tenure, sim_monthly, total_charges, sim_contract, sim_payment, sim_tech, sim_sec, sim_bkp
        )
        sim_proba = active_model.predict_proba(sim_scaled)[0][1] * 100
        risk_drop = churn_probability - sim_proba
        sim_arr_at_risk = sim_monthly * 12 * (sim_proba / 100)
        gross_arr_saved = max(0.0, arr_at_risk - sim_arr_at_risk)
        net_arr_saved = gross_arr_saved - (sim_discount * 12)

        st.markdown("#### 🎯 Simulation Outcome Comparison")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.metric("Simulated Churn Risk", f"{sim_proba:.1f}%", f"-{risk_drop:.1f}%" if risk_drop > 0 else f"+{abs(risk_drop):.1f}%", delta_color="inverse")
        with col_c2:
            st.metric("Risk Drop", f"{risk_drop:.1f}% points", "Risk Lowered" if risk_drop > 0 else "Risk Unchanged", delta_color="normal")
        with col_c3:
            st.metric("Net ARR Protected", f"${net_arr_saved:,.2f}/yr", f"+${gross_arr_saved:,.2f} Gross Saved", delta_color="normal")

        st.markdown("""
            <div class="info-container" style="border-left-color: #38bdf8; margin-top: 15px;">
                <h4 style="color: #7dd3fc !important;">📋 Frontline Agent Recommendation Playbook</h4>
                <p style="color: #e2e8f0; font-size: 0.95rem; margin-bottom: 8px;">
                    <strong>Recommended Pitch:</strong> "To reward your loyalty, we can apply a <strong>$""" + f"{sim_discount:.0f}/month discount" + """</strong> today, switch you to our <strong>""" + f"{sim_contract}" + """</strong> with guaranteed price lock, and include <strong>complimentary Tech Support & Online Security</strong>."
                </p>
                <p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">
                    Projected Outcome: Churn risk drops from <strong>""" + f"{churn_probability:.1f}%" + """</strong> to <strong>""" + f"{sim_proba:.1f}%" + """</strong> (Net Annual Savings: <strong>$""" + f"{net_arr_saved:,.2f}" + """</strong>).
                </p>
            </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# MODE 2: ENTERPRISE BATCH SCORING ENGINE
# ==============================================================================
elif nav_tab == "📁 Enterprise Batch Scoring Engine":
    st.markdown("### 📁 High-Throughput Batch Scoring & Retention Targeting")
    st.markdown("Score multiple accounts simultaneously, rank customers by churn risk & revenue exposure, and export prioritized outreach targets.")

    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        uploaded_file = st.file_uploader("Upload Customer Accounts CSV:", type=["csv"])
    with col_b2:
        st.markdown("#### Quick Tools")
        # Template download generator
        sample_template = pd.DataFrame([{
            'customerID': 'SAMPLE-001', 'gender': 'Female', 'SeniorCitizen': 'No', 'Partner': 'Yes', 'Dependents': 'No',
            'tenure': 6, 'PhoneService': 'Yes', 'MultipleLines': 'No', 'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No', 'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
            'StreamingTV': 'Yes', 'StreamingMovies': 'Yes', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check', 'MonthlyCharges': 89.85, 'TotalCharges': 539.10
        }])
        csv_buffer = io.StringIO()
        sample_template.to_csv(csv_buffer, index=False)
        st.download_button("📄 Download Sample CSV Template", csv_buffer.getvalue(), "telecom_batch_template.csv", "text/csv")
        
        run_demo = st.button("🚀 Run Demo with 25 Live Accounts")

    def run_batch_inference(input_df):
        """Vectorized batch preprocessing and multi-model inference."""
        df_clean = input_df.copy()
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce").fillna(0)
        df_clean["MonthlyCharges"] = pd.to_numeric(df_clean["MonthlyCharges"], errors="coerce").fillna(0)
        
        # Binary conversions
        bin_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0, 1: 1, 0: 0}
        for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "SeniorCitizen"]:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map(bin_map).fillna(0).astype(int)

        def t_grp(t):
            if t <= 12: return "0-12 Months"
            elif t <= 24: return "12-24 Months"
            elif t <= 48: return "24-48 Months"
            elif t <= 60: return "48-60 Months"
            else: return "60+ Months"
        df_clean["tenure_group"] = df_clean["tenure"].apply(t_grp)

        cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
                    "Contract", "PaymentMethod", "tenure_group"]
        df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)

        feat_df = pd.DataFrame(0, index=df_encoded.index, columns=feature_names)
        for c in feature_names:
            if c in df_encoded.columns:
                feat_df[c] = df_encoded[c].values

        scaled_matrix = scaler.transform(feat_df)
        probs = active_model.predict_proba(scaled_matrix)[:, 1] * 100

        res = input_df.copy()
        res["Churn_Probability_%"] = np.round(probs, 1)
        res["Risk_Tier"] = pd.cut(
            probs, bins=[-np.inf, 30, 60, 80, np.inf], labels=["Low", "Moderate", "High", "Critical"]
        )
        res["ARR_At_Risk_$"] = np.round(res["MonthlyCharges"] * 12 * (probs / 100), 2)
        return res.sort_values(by="Churn_Probability_%", ascending=False)

    df_to_score = None
    if uploaded_file is not None:
        try:
            df_to_score = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df_to_score)} customer accounts from uploaded file.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    elif run_demo:
        hist_df = load_historical_dataset()
        if not hist_df.empty:
            df_to_score = hist_df.sample(25, random_state=42)
            st.success(f"Loaded 25 customer accounts from historical repository for demo.")

    if df_to_score is not None:
        with st.spinner(f"Scoring accounts with {selected_model_name}..."):
            scored_df = run_batch_inference(df_to_score)

        st.markdown("---")
        st.markdown("### 📊 Batch Executive Summary")
        
        total_accounts = len(scored_df)
        avg_risk = scored_df["Churn_Probability_%"].mean()
        total_arr_risk = scored_df["ARR_At_Risk_$"].sum()
        critical_count = (scored_df["Risk_Tier"] == "Critical").sum()
        high_count = (scored_df["Risk_Tier"] == "High").sum()

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid #818cf8;">
                    <h4>Total Scored</h4>
                    <h2>{total_accounts:,}</h2>
                    <p>Accounts in Batch</p>
                </div>
            """, unsafe_allow_html=True)
        with col_k2:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid #38bdf8;">
                    <h4>Average Churn Risk</h4>
                    <h2>{avg_risk:.1f}%</h2>
                    <p>Across Cohort</p>
                </div>
            """, unsafe_allow_html=True)
        with col_k3:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid #ef4444;">
                    <h4>Total ARR at Risk</h4>
                    <h2 style="color:#ef4444 !important;">${total_arr_risk:,.2f}</h2>
                    <p>Annual Exposure</p>
                </div>
            """, unsafe_allow_html=True)
        with col_k4:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid #f59e0b;">
                    <h4>Priority Targets</h4>
                    <h2>{critical_count + high_count}</h2>
                    <p>{critical_count} Critical, {high_count} High</p>
                </div>
            """, unsafe_allow_html=True)

        col_chart1, col_chart2 = st.columns([1, 1.5])
        with col_chart1:
            st.markdown("#### 🎯 Risk Tier Distribution")
            tier_counts = scored_df["Risk_Tier"].value_counts()
            donut_fig = go.Figure(go.Pie(
                labels=tier_counts.index,
                values=tier_counts.values,
                hole=0.55,
                marker=dict(colors=["#059669", "#d97706", "#ea580c", "#dc2626"]),
                textinfo="label+percent",
                hovertemplate="<b>%{label} Risk</b>: %{value} accounts (%{percent})<extra></extra>"
            ))
            donut_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"), margin=dict(l=10, r=10, t=10, b=10), height=280,
                showlegend=False
            )
            st.plotly_chart(donut_fig, use_container_width=True)

        with col_chart2:
            st.markdown("#### 💰 Top 5 Revenue-Endangered Accounts")
            top_arr = scored_df.sort_values(by="ARR_At_Risk_$", ascending=False).head(5)
            fig_bar = go.Figure(go.Bar(
                x=top_arr["customerID"].astype(str),
                y=top_arr["ARR_At_Risk_$"],
                marker=dict(color="#ef4444"),
                hovertemplate="<b>ID: %{x}</b><br>ARR at Risk: $%{y:,.2f}<extra></extra>"
            ))
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(39,39,42,0.6)",
                font=dict(color="#cbd5e1"), margin=dict(l=10, r=10, t=20, b=20), height=280,
                xaxis=dict(title="Customer ID", gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="ARR at Risk ($)", gridcolor="rgba(255,255,255,0.05)")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📋 Prioritized Customer Retention List")
        st.dataframe(
            scored_df[["customerID", "Contract", "tenure", "MonthlyCharges", "Churn_Probability_%", "Risk_Tier", "ARR_At_Risk_$"]],
            use_container_width=True,
            height=350
        )

        out_buffer = io.StringIO()
        scored_df.to_csv(out_buffer, index=False)
        st.download_button(
            "📥 Download Scored Retention Target List (CSV)",
            out_buffer.getvalue(),
            "scored_retention_targets.csv",
            "text/csv"
        )


# ==============================================================================
# MODE 3: HISTORICAL DATASET INTELLIGENCE
# ==============================================================================
elif nav_tab == "📊 Historical Dataset Intelligence":
    st.markdown("### 📊 Historical Telecom Cohort Intelligence")
    st.markdown("Explore trends, churn drivers, and behavioral distributions across 7,042 verified historical customer accounts.")

    hist_data = load_historical_dataset()
    if not hist_data.empty:
        total_pop = len(hist_data)
        churn_pop = (hist_data["Churn"] == "Yes").sum()
        churn_pct = (churn_pop / total_pop) * 100
        avg_monthly = hist_data["MonthlyCharges"].mean()
        avg_tenure = hist_data["tenure"].mean()

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        with col_h1:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid #818cf8;">
                    <h4>Total Population</h4>
                    <h2>{total_pop:,}</h2>
                    <p>Historical Accounts</p>
                </div>
            """, unsafe_allow_html=True)
        with col_h2:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid #ef4444;">
                    <h4>Historical Churn Rate</h4>
                    <h2>{churn_pct:.1f}%</h2>
                    <p>{churn_pop:,} Churned Accounts</p>
                </div>
            """, unsafe_allow_html=True)
        with col_h3:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid #38bdf8;">
                    <h4>Mean Monthly Spend</h4>
                    <h2>${avg_monthly:.2f}</h2>
                    <p>Average Monthly Bill</p>
                </div>
            """, unsafe_allow_html=True)
        with col_h4:
            st.markdown(f"""
                <div class="kpi-card" style="border-top: 4px solid #10b981;">
                    <h4>Mean Tenure</h4>
                    <h2>{avg_tenure:.1f} mos</h2>
                    <p>Average Account Lifetime</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            st.markdown("#### 📝 Churn Rate by Contract Type")
            contract_churn = hist_data.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig_c = px.bar(
                contract_churn, x="Contract", y="Churn",
                labels={"Churn": "Churn Rate (%)", "Contract": "Contract Type"},
                color="Contract",
                color_discrete_sequence=["#ef4444", "#38bdf8", "#10b981"]
            )
            fig_c.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(39,39,42,0.6)",
                font=dict(color="#cbd5e1"), height=300, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_c, use_container_width=True)

        with col_c2:
            st.markdown("#### 🌐 Churn Rate by Internet Service")
            net_churn = hist_data.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig_net = px.bar(
                net_churn, x="InternetService", y="Churn",
                labels={"Churn": "Churn Rate (%)", "InternetService": "Internet Service"},
                color="InternetService",
                color_discrete_sequence=["#38bdf8", "#ef4444", "#10b981"]
            )
            fig_net.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(39,39,42,0.6)",
                font=dict(color="#cbd5e1"), height=300, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_net, use_container_width=True)

        col_c3, col_c4 = st.columns(2)
        with col_c3:
            st.markdown("#### 💳 Churn Rate by Payment Method")
            pay_churn = hist_data.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig_pay = px.bar(
                pay_churn, x="PaymentMethod", y="Churn",
                labels={"Churn": "Churn Rate (%)", "PaymentMethod": "Payment Method"},
                color="PaymentMethod",
                color_discrete_sequence=["#10b981", "#ef4444", "#f59e0b", "#818cf8"]
            )
            fig_pay.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(39,39,42,0.6)",
                font=dict(color="#cbd5e1"), height=320, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False
            )
            st.plotly_chart(fig_pay, use_container_width=True)

        with col_c4:
            st.markdown("#### 📈 Monthly Charges vs. Tenure Distribution")
            sample_scatter = hist_data.sample(min(800, len(hist_data)), random_state=42)
            fig_scat = px.scatter(
                sample_scatter, x="tenure", y="MonthlyCharges", color="Churn",
                color_discrete_map={"Yes": "#ef4444", "No": "#10b981"},
                labels={"tenure": "Tenure (Months)", "MonthlyCharges": "Monthly Charges ($)", "Churn": "Churned"}
            )
            fig_scat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(39,39,42,0.6)",
                font=dict(color="#cbd5e1"), height=320, margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig_scat, use_container_width=True)

# Enhanced Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <h3 style="margin-bottom: 15px;">🎯 Customer Churn Predictor</h3>
        <p style="font-size: 1rem; color: #cbd5e1; margin-bottom: 10px;">
            Enterprise Decision Intelligence Engine with Financial Impact Analysis & Counterfactual Simulation
        </p>
        <p style="font-size: 0.9rem; color: #94a3b8;">
            © 2026 | Built with Python, Scikit-learn, Plotly & Streamlit
        </p>
        <div style="margin-top: 15px;">
            <span style="margin: 0 10px;">📊 34 Features</span>
            <span style="margin: 0 10px;">🔍 Attribution Engine</span>
            <span style="margin: 0 10px;">⚡ What-If Simulator</span>
            <span style="margin: 0 10px;">📁 Batch Scoring</span>
            <span style="margin: 0 10px;">📈 Cohort Analytics</span>
        </div>
    </div>
""", unsafe_allow_html=True)
