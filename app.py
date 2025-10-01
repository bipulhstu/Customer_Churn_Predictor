import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme
st.markdown("""
    <style>
    /* ===== DARK THEME - MAIN LAYOUT ===== */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    .main .block-container {
        background: #1a1a2e;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        max-width: 1400px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* ===== TYPOGRAPHY ===== */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .main h1, .main h2 {
        color: #e2e8f0;
        font-weight: 700;
    }
    
    .main h3, .main h4 {
        color: #cbd5e0;
        font-weight: 600;
    }
    
    .main p, .main li {
        color: #a0aec0;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748;
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 600;
        color: #cbd5e0;
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #374151;
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: #667eea;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #2d3748;
        border-radius: 0 10px 10px 10px;
        padding: 25px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* ===== INPUTS ===== */
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    
    .stSelectbox > div > div, 
    .stNumberInput > div > div > input {
        background-color: #374151 !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    
    .stSelectbox > div > div:hover,
    .stNumberInput > div > div > input:hover {
        border-color: #667eea !important;
    }
    
    .stSlider {
        padding: 10px 0;
    }
    
    .stSlider > div > div > div {
        background-color: #374151;
    }
    
    .stSlider > div > div > div > div {
        color: #e2e8f0;
    }
    
    /* ===== BUTTON ===== */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* ===== RESULTS BOXES ===== */
    .success-box {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4);
    }
    
    .success-box h2, .success-box p {
        color: white !important;
        margin: 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(245, 101, 101, 0.4);
    }
    
    .error-box h2, .error-box p {
        color: white !important;
        margin: 0;
    }
    
    /* ===== INFO CONTAINERS ===== */
    .info-container {
        background: #2d3748;
        border-left: 4px solid #667eea;
        padding: 20px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    
    .info-container h4 {
        color: #e2e8f0 !important;
        margin-top: 0;
    }
    
    .info-container ul {
        margin: 10px 0;
        padding-left: 20px;
    }
    
    .info-container li {
        color: #cbd5e0 !important;
        margin: 8px 0;
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0aec0;
        font-weight: 600;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600;
    }
    
    /* ===== PROGRESS BARS ===== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .stProgress > div > div {
        background-color: #374151;
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        background-color: #2d3748 !important;
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        color: #e2e8f0;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] strong {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white !important;
    }
    
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
    }
    
    /* ===== DIVIDERS ===== */
    .main hr {
        border: none;
        border-top: 2px solid rgba(102, 126, 234, 0.3);
        margin: 2rem 0;
    }
    
    /* ===== FOOTER ===== */
    .footer {
        text-align: center;
        color: #a0aec0;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid rgba(102, 126, 234, 0.3);
        background-color: #2d3748;
        border-radius: 10px;
    }
    
    .footer h3 {
        color: #e2e8f0 !important;
    }
    
    .footer p, .footer span {
        color: #a0aec0 !important;
    }
    
    /* ===== MARKDOWN STYLING ===== */
    .main .stMarkdown {
        color: #a0aec0;
    }
    
    /* ===== SELECT DROPDOWN ===== */
    [data-baseweb="select"] > div {
        background-color: #374151 !important;
        color: #e2e8f0 !important;
    }
    
    /* ===== HIDE STREAMLIT BRANDING ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Title and description with custom styling
st.markdown('<h1 class="main-title">🎯 Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict customer churn probability using advanced machine learning</p>', unsafe_allow_html=True)

# Information banner
st.info("💡 **How it works:** Enter customer details below and our AI model will predict the likelihood of churn with actionable insights!")
st.markdown("---")

# Create input sections with tabs for better organization
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

# Prediction button
if st.button("🔮 Predict Churn", type="primary"):
    # Create a dictionary with user inputs
    input_data = {
        'gender': 1 if gender == "Male" else 0,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
        'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
        'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0,
        'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
    }

    # Create tenure group features
    if tenure <= 12:
        tenure_group = '0-12 Months'
    elif 12 < tenure <= 24:
        tenure_group = '12-24 Months'
    elif 24 < tenure <= 48:
        tenure_group = '24-48 Months'
    elif 48 < tenure <= 60:
        tenure_group = '48-60 Months'
    else:
        tenure_group = '60+ Months'

    input_data['tenure_group_12-24 Months'] = 1 if tenure_group == '12-24 Months' else 0
    input_data['tenure_group_24-48 Months'] = 1 if tenure_group == '24-48 Months' else 0
    input_data['tenure_group_48-60 Months'] = 1 if tenure_group == '48-60 Months' else 0
    input_data['tenure_group_60+ Months'] = 1 if tenure_group == '60+ Months' else 0

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Get the expected feature names from the scaler
    expected_features = scaler.feature_names_in_
    
    # Create a DataFrame with all expected features, filling missing ones with 0
    final_input = pd.DataFrame(0, index=[0], columns=expected_features)
    
    # Fill in the features we have
    for col in input_df.columns:
        if col in final_input.columns:
            final_input[col] = input_df[col].values[0]
    
    # Scale the input data
    input_scaled = scaler.transform(final_input)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    # Display results with enhanced visuals
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")
    
    # Calculate probability
    churn_probability = prediction_proba[0][1] * 100
    no_churn_probability = prediction_proba[0][0] * 100
    
    # Create metrics row
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if prediction[0] == 1:
            st.markdown("""
                <div class="error-box">
                    <h2 style="margin:0;">⚠️ High Risk</h2>
                    <p style="font-size:1.1rem; margin-top:10px;">Customer Likely to Churn</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="success-box">
                    <h2 style="margin:0;">✅ Low Risk</h2>
                    <p style="font-size:1.1rem; margin-top:10px;">Customer Likely to Stay</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            label="🎯 Churn Probability",
            value=f"{churn_probability:.1f}%",
            delta=f"{churn_probability - 50:.1f}% vs baseline",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="✨ Retention Probability", 
            value=f"{no_churn_probability:.1f}%",
            delta=f"{no_churn_probability - 50:.1f}% vs baseline",
            delta_color="normal"
        )
    
    # Visual probability bar
    st.markdown("#### 📈 Probability Distribution")
    col_bar1, col_bar2 = st.columns(2)
    
    with col_bar1:
        st.markdown(f"**Will Churn:** {churn_probability:.2f}%")
        st.progress(churn_probability / 100)
    
    with col_bar2:
        st.markdown(f"**Will Stay:** {no_churn_probability:.2f}%")
        st.progress(no_churn_probability / 100)
    
    # Recommendations Section
    st.markdown("---")
    st.markdown("## 💡 Actionable Recommendations")
    
    if prediction[0] == 1:
        # High Risk - Detailed recommendations
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown("""
                <div class="info-container">
                    <h4>🎯 Immediate Actions</h4>
                    <ul>
                        <li>📞 Priority outreach call within 24 hours</li>
                        <li>💰 Offer 15-20% retention discount</li>
                        <li>📝 Upgrade to annual contract with incentives</li>
                        <li>🎁 Provide loyalty rewards or free services</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("""
                <div class="info-container">
                    <h4>🔧 Long-term Strategies</h4>
                    <ul>
                        <li>💳 Switch to automatic payment methods</li>
                        <li>🛡️ Add value with security/backup services</li>
                        <li>👥 Assign dedicated account manager</li>
                        <li>📊 Schedule quarterly satisfaction reviews</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # Risk Level Indicator
        if churn_probability >= 75:
            st.error("🚨 **CRITICAL RISK** - Immediate intervention required!")
        elif churn_probability >= 60:
            st.warning("⚠️ **HIGH RISK** - Proactive retention measures recommended")
        else:
            st.info("📋 **MODERATE RISK** - Monitor and engage regularly")
    
    else:
        # Low Risk - Growth recommendations
        col_rec1, col_rec2 = st.columns(2)
        
        with col_rec1:
            st.markdown("""
                <div class="info-container">
                    <h4>🌟 Growth Opportunities</h4>
                    <ul>
                        <li>📈 Upsell premium services or packages</li>
                        <li>🎯 Cross-sell complementary products</li>
                        <li>💎 Introduce to loyalty/VIP programs</li>
                        <li>🤝 Request referrals and testimonials</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col_rec2:
            st.markdown("""
                <div class="info-container">
                    <h4>💪 Retention Reinforcement</h4>
                    <ul>
                        <li>✅ Continue excellent customer service</li>
                        <li>📧 Regular engagement via newsletters</li>
                        <li>🎁 Surprise with occasional perks</li>
                        <li>📊 Annual satisfaction check-ins</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        st.success("✨ **STABLE CUSTOMER** - Focus on value addition and engagement!")

# Sidebar with additional information
with st.sidebar:
    st.markdown("### 📊 Model Information")
    st.markdown("---")
    
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.15); 
                    padding: 15px; border-radius: 10px; margin-bottom: 15px;
                    border: 1px solid rgba(255, 255, 255, 0.3);">
            <h4 style="margin-top:0; color: white;">🤖 AI Model Details</h4>
            <p style="color: white;"><strong>Algorithm:</strong> Logistic Regression</p>
            <p style="color: white;"><strong>Accuracy:</strong> ~79.4%</p>
            <p style="color: white;"><strong>AUC-ROC:</strong> 0.835</p>
            <p style="color: white;"><strong>Training Data:</strong> 7,042 customers</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📈 Key Churn Factors")
    st.markdown("""
        - ⏱️ **Tenure:** New customers (0-12 months)
        - 📝 **Contract:** Month-to-month
        - 💳 **Payment:** Electronic check
        - 👴 **Demographics:** Senior citizens
        - 👥 **Family:** No partner/dependents
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.15); 
                    padding: 15px; border-radius: 10px;
                    border: 1px solid rgba(255, 255, 255, 0.3);">
            <p style="color: white; margin: 0;"><strong>Best Practices:</strong></p>
            <ul style="color: white; margin-top: 10px;">
                <li>Enter accurate customer data</li>
                <li>Review all tabs before predicting</li>
                <li>Use insights for targeted retention</li>
                <li>Update predictions regularly</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📞 Support")
    st.markdown("""
        Need help? Contact:
        - 📧 Email: bipulhstu@gmail.com
        - 🌐 Web: www.bipulhstu.github.io
    """)

# Enhanced Footer
st.markdown("---")
st.markdown("""
    <div class="footer">
        <h3 style="margin-bottom: 15px;">🎯 Customer Churn Predictor</h3>
        <p style="font-size: 1rem; color: #6c757d; margin-bottom: 10px;">
            Powered by Advanced Machine Learning | Built with ❤️ using Streamlit
        </p>
        <p style="font-size: 0.9rem; color: #adb5bd;">
            © 2025 | Made with Python, Scikit-learn & Streamlit
        </p>
        <div style="margin-top: 15px;">
            <span style="margin: 0 10px;">📊 Data Science</span>
            <span style="margin: 0 10px;">🤖 Machine Learning</span>
            <span style="margin: 0 10px;">🚀 Deployment</span>
        </div>
    </div>
""", unsafe_allow_html=True)
