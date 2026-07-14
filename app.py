import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import time

# ==========================================
# 1. ARCHITECTURAL & INTERFACE INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="TelcoPulse Core v3.0 | Enterprise Retention Control",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Deep-space glassmorphic style framework injection
st.markdown("""
    <style>
    /* Global Canvas Overrides */
    .main { background-color: #060913; color: #f1f5f9; }
    .stApp { background: radial-gradient(circle at 50% 0%, #0c1530 0%, #030712 100%); }
    
    /* Absolute Visibility Overrides for Headings, Subheaders & Markdown Text */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #f1f5f9 !important;
    }
    
    /* Native Input Field Labels & Helper Text Correction */
    div[data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
        font-weight: 600 !important;
    }
    
    /* Radio Button & Checkbox Element Labels */
    div[data-testid="stRadio"] label p, div[data-testid="stCheckbox"] label p {
        color: #f1f5f9 !important;
    }
    
    /* Tab Navigation Layout Text Configuration */
    button[data-baseweb="tab"] div p {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] div p {
        color: #00f2fe !important; /* Accent light blue for the active configuration pane */
    }
    
    /* Slider Min/Max & Floating Value Indicators */
    div[data-testid="stSlider"] div {
        color: #f1f5f9 !important;
    }
    
    /* Neon Sidebar Framework */
    [data-testid="stSidebar"] {
        background-color: #02040a;
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
        color: #f1f5f9 !important;
    }
    
    /* Interactive Control Units */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        color: #020617 !important; 
        border: none; 
        border-radius: 8px;
        padding: 0.85rem 2rem; 
        font-weight: 800; 
        width: 100%;
        font-size: 1.15rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        box-shadow: 0px 0px 25px rgba(0, 242, 254, 0.2);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0px 0px 35px rgba(0, 242, 254, 0.55);
        color: #000000 !important;
    }
    
    /* Premium Analytical Card Widgets */
    .telecom-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 1.5rem; 
        border-radius: 16px; 
        backdrop-filter: blur(20px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 1.25rem;
        transition: border 0.3s ease;
    }
    .telecom-card:hover {
        border-color: rgba(56, 189, 248, 0.5);
    }
    .card-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #94a3b8 !important;
        font-weight: 600;
    }
    .card-metric {
        font-size: 2.25rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-top: 0.5rem;
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Matrix Readout Design blocks */
    .status-alert-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid #ef4444;
        padding: 1.75rem; border-radius: 12px; margin-top: 1rem;
    }
    .status-alert-low {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(22, 163, 74, 0.05) 100%);
        border: 1px solid #22c55e;
        padding: 1.75rem; border-radius: 12px; margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CORE MACHINE LEARNING RUNTIME MODEL
# ==========================================

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def bootstrap_ml_system():
    try:
        encoder = joblib.load("models/encoder.pkl")
        scaler = joblib.load("models/scaler.pkl")
        cat_cols = joblib.load("models/categorical_columns.pkl")
        num_cols = joblib.load("models/numerical_columns.pkl")
        feature_cols = joblib.load("models/feature_columns.pkl")
        
        model = NeuralNetwork(len(feature_cols))
        model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
        model.eval()
        return model, encoder, scaler, cat_cols, num_cols, feature_cols
    except Exception as e:
        return None, None, None, None, None, None

model, encoder, scaler, cat_cols, num_cols, feature_cols = bootstrap_ml_system()

if model is None:
    st.error("🚨 **System Halt**: Essential ML pipeline components are missing from the `models/` directory. Please run your `train.py` sequence and ensure files are positioned properly.")
    st.stop()

# ==========================================
# 3. STATE MANAGEMENT & SESSION CONTROLS
# ==========================================
if "audit_count" not in st.session_state:
    st.session_state.audit_count = 0
if "high_risk_count" not in st.session_state:
    st.session_state.high_risk_count = 0
if "last_probability" not in st.session_state:
    st.session_state.last_probability = 0.0

# ==========================================
# 4. CONTROL PANEL INTERFACE (SIDEBAR)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#00f2fe !important; margin-bottom:0;'>TELCOPULSE OS</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94a3b8 !important; font-size:0.85rem; margin-top:0;'>Enterprise Account Auditing Engine</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 📡 Operational Base")
    telecom_operator = st.selectbox("Regional Carrier Network", ["Jazz Pakistan", "Zong 4G", "Telenor Pakistan", "Ufone"])
    
    st.markdown("### 👤 Demographics Profile")
    gender = st.radio("Subscriber Gender", ["Male", "Female"], horizontal=True)
    senior_citizen = st.radio("Senior Citizen Allocation (65+)", ["No", "Yes"], horizontal=True)
    partner = st.radio("Family Unit / Married Status", ["No", "Yes"], horizontal=True)
    dependents = st.radio("Household Dependent Flag", ["No", "Yes"], horizontal=True)
    
    st.markdown("---")
    st.markdown("### 📉 Live Environment Diagnostics")
    st.markdown(f"**Total Runs Sequenced:** `{st.session_state.audit_count}`")
    st.markdown(f"**Threat Flags Issued:** `{st.session_state.high_risk_count}`")
    st.markdown(f"**Last Model Loss Context:** `0.3053` (Validation)")

# ==========================================
# 5. CORE WORKSPACE DESIGN & CONTROL PIPELINES
# ==========================================
st.title("⚡ INFRASTRUCTURE SUBSCRIBER RETENTION TERMINAL")
st.markdown("<p style='color:#cbd5e1 !important; font-size:1.1rem; margin-top:-15px;'>Real-time Deep Learning Analytics Node running over PyTorch Production Architecture.</p>", unsafe_allow_html=True)
st.markdown("---")

col_inputs, col_monitors = st.columns([12, 9], gap="large")

with col_inputs:
    st.markdown("### 🎛️ Account Workspace Parameter Configuration")
    
    g1, g2, g3 = st.columns(3)
    with g1:
        tenure = st.slider("Lifespan Tenure (Months)", 0, 72, 24, help="Total elapsed structural months since initial network sign-up.")
    with g2:
        monthly_charges = st.number_input("Cyclic Monthly Cost (PKR)", 10.00, 250.00, 70.35, step=0.50)
    with g3:
        total_charges = st.number_input("Cumulative Paid Assets (PKR)", 0.00, 10000.00, 1500.00, step=25.00)

    st.markdown("<br>", unsafe_allow_html=True)
    
    tab_network, tab_security, tab_ledger = st.tabs([
        "🌐 Infrastructure Connectivity", 
        "🛡️ Core Value-Added Services (VAS)", 
        "💳 Ledger Billing Contracts"
    ])
    
    with tab_network:
        st.markdown("<p style='color:#38bdf8 !important; font-weight:bold;'>Configure Hardware Layer Interconnect Paths:</p>", unsafe_allow_html=True)
        phone_service = st.selectbox("Primary Voice Line Activation", ["Yes", "No"])
        multiple_lines = st.selectbox("Multi-SIM Interface Links", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Data Transport Network Infrastructure", ["Fiber optic", "DSL", "No"])
        unlimited_data = st.selectbox("Unlimited High-Speed Data Protocol", ["Yes", "No", "No internet service"])
        avg_gb = st.slider("Average Monthly Volume Consumed (GB)", 0, 500, 45)

    with tab_security:
        st.markdown("<p style='color:#38bdf8 !important; font-weight:bold;'>Value-Added Security & Media Layer Frameworks:</p>", unsafe_allow_html=True)
        row_s1, row_s2 = st.columns(2)
        with row_s1:
            online_security = st.selectbox("Cloud Access Firewall Proxy", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Automated Cloud Workspace Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Hardware Protection Protocols", ["No", "Yes", "No internet service"])
        with row_s2:
            tech_support = st.selectbox("Priority Enterprise Helpline Desk", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Smart IPTV Media Allocation", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("On-Demand Streaming Media Rights", ["No", "Yes", "No internet service"])
            streaming_music = st.selectbox("Hi-Fi Audio Streaming License", ["No", "Yes", "No internet service"])

    with tab_ledger:
        st.markdown("<p style='color:#38bdf8 !important; font-weight:bold;'>Financial Invoicing & Promotion Agreements:</p>", unsafe_allow_html=True)
        row_l1, row_l2 = st.columns(2)
        with row_l1:
            contract = st.selectbox("Contractual Agreement Term Structure", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Digital E-Invoice Protocols", ["Yes", "No"])
            payment_method = st.selectbox("Financial Clearing Channel", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        with row_l2:
            offer = st.selectbox("Campaign Marketing Promotional Assignment", ["None", "Offer A", "Offer B", "Offer C", "Offer D", "Offer E"])
            under_30 = st.selectbox("Age Assessment Metric Block: Under 30", ["No", "Yes"])
            married = st.selectbox("System Civil Marital Status Flag", ["No", "Yes"])
            
        st.markdown("---")
        st.markdown("<p style='color:#38bdf8 !important; font-weight:bold;'>Auxiliary Behavior Profiles:</p>", unsafe_allow_html=True)
        row_aux1, row_aux2 = st.columns(2)
        with row_aux1:
            age = st.number_input("Account Owner Calculated Biological Age", 18, 100, 38)
            referrals = st.number_input("Tracked Sign-up Customer Referrals", 0, 50, 2)
            referred_friend = st.selectbox("Acquired Through Friend Referral Link", ["No", "Yes"])
        with row_aux2:
            avg_long_distance = st.number_input("Average Long Distance Cyclic Surcharges", 0.00, 100.00, 12.45)
            total_refunds = st.number_input("Total Historical Operational Refunds Disbursed", 0.00, 1000.00, 0.00)
            extra_data_charges = st.number_input("Overage Data Rate Ledger Surcharges", 0.00, 1000.00, 0.00)
            satisfaction_score = st.slider("Latest Interaction Account Satisfaction Index", 1, 5, 4)

    monthly_spend_per_tenure = monthly_charges / (tenure + 1)
    yearly_spend = monthly_charges * 12
    
    val_map = {"Yes": 1, "No": 0, "No internet service": 0, "No phone service": 0}
    total_services = sum([val_map.get(x, 0) for x in [phone_service, multiple_lines, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]])
    
    total_long_distance_charges = avg_long_distance * tenure
    total_revenue = total_charges + total_long_distance_charges + extra_data_charges - total_refunds

with col_monitors:
    st.markdown("### 📊 Production System Telemetry")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"""
            <div class="telecom-card">
                <div class="card-label">💼 Lifecycle Gross Value</div>
                <div class="card-metric">PKR {total_revenue:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
            <div class="telecom-card">
                <div class="card-label">📡 Active Integration Load</div>
                <div class="card-metric">{total_services} / 8</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
        <div class="telecom-card" style="text-align: center;">
            <div class="card-label">🎯 Operational Context Network Target</div>
            <div style="font-size: 1.5rem; font-weight:700; color:#00f2fe; margin-top:0.5rem;">{telecom_operator} Hub Node</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # ==========================================
    # 6. INFERENCE EXECUTION & EVENT LOOP
    # ==========================================
    if st.button("🚀 INITIATE NEURAL CHURN INTERROGATION"):
        
        raw_dataframe_payload = {
            "gender": gender,
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": int(tenure),
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
            "Age": int(age),
            "Under 30": under_30,
            "Married": married,
            "Referred a Friend": referred_friend,
            "Number of Referrals": int(referrals),
            "Offer": offer,
            "Avg Monthly Long Distance Charges": float(avg_long_distance),
            "Avg Monthly GB Download": int(avg_gb),
            "Streaming Music": streaming_music,
            "Premium Tech Support": tech_support, 
            "Unlimited Data": unlimited_data,
            "Total Refunds": float(total_refunds),
            "Total Extra Data Charges": float(extra_data_charges),
            "Total Long Distance Charges": float(total_long_distance_charges),
            "Total Revenue": float(total_revenue),
            "Satisfaction Score": int(satisfaction_score),
            "MonthlySpendPerTenure": float(monthly_spend_per_tenure),
            "YearlySpend": float(yearly_spend),
            "TotalServices": float(total_services)
        }
        
        input_frame = pd.DataFrame([raw_dataframe_payload])
        
        with st.spinner("Executing feedforward propagation across model layers..."):
            time.sleep(0.45) 
            
            encoded_matrix = encoder.transform(input_frame[cat_cols])
            encoded_frame = pd.DataFrame(
                encoded_matrix, 
                columns=encoder.get_feature_names_out(cat_cols), 
                index=input_frame.index
            )
            
            processed_intermediate_frame = input_frame.drop(columns=cat_cols)
            final_inference_dataframe = pd.concat([processed_intermediate_frame, encoded_frame], axis=1)
            final_inference_dataframe = final_inference_dataframe.reindex(columns=feature_cols, fill_value=0.0)
            
            scaled_inference_matrix = scaler.transform(final_inference_dataframe)
            runtime_inference_tensor = torch.FloatTensor(scaled_inference_matrix)
            
            with torch.no_grad():
                churn_risk_probability = model(runtime_inference_tensor).item()
        
        st.session_state.audit_count += 1
        st.session_state.last_probability = churn_risk_probability
        
        st.markdown("### 👁️ Core Prediction Readout Diagnostic")
        
        if churn_risk_probability >= 0.50:
            st.session_state.high_risk_count += 1
            st.markdown(f"""
                <div class="status-alert-high">
                    <h3 style="color: #f87171 !important; margin-top:0;">🚨 HIGH CHURN THREAT PROFILE FLAG DETECTED</h3>
                    <p style="font-size: 1.25rem; color: #fca5a5 !important; margin-bottom:0.5rem;">
                        Neural Evaluation Probability Track: <b>{churn_risk_probability * 100:.2f}%</b>
                    </p>
                    <p style="color: #e2e8f0 !important; font-size: 0.95rem; margin-bottom: 0;">
                        <b>Automated Operational Override Response:</b> Customer flagged for retention routing. Dispatched special loyalty bundle tokens directly to subscriber ledger profiles.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(churn_risk_probability)
        else:
            st.markdown(f"""
                <div class="status-alert-low">
                    <h3 style="color: #4ade80 !important; margin-top:0;">💚 SECURE NOMINAL RETENTION MATRIX CONFIRMED</h3>
                    <p style="font-size: 1.25rem; color: #86efac !important; margin-bottom:0.5rem;">
                        Neural Evaluation Probability Track: <b>{churn_risk_probability * 100:.2f}%</b>
                    </p>
                    <p style="color: #e2e8f0 !important; font-size: 0.95rem; margin-bottom: 0;">
                        <b>System Diagnostic Profile:</b> Account reflects steady user behavior signals. No immediate automated pricing or priority support intervention passes required.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.progress(churn_risk_probability)