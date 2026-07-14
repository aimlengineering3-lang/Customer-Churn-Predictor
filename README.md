# ⚡ TelcoPulse Core v3.0 | Deep Learning Retention Terminal

An enterprise-grade subscriber churn prediction and risk auditing framework. This system utilizes a deep feedforward PyTorch Neural Network to compute risk matrices over subscriber account variables, rendered through a high-fidelity, real-time glassmorphic Streamlit executive dashboard.

---

## 📂 Project Architecture

For the runtime environments to map data vectors correctly without dimension or path crashes, maintain your directory layout exactly like this:

```text
Telco_Churn_Project/
│
├── data/
│   └── Telco_customer_churn.csv      # Raw IBM/Kaggle Dataset
│
├── train.py                          # PyTorch Pipeline & Optimization Script
├── app.py                            # Cinematic Streamlit UI Dashboard
├── requirements.txt                  # Python Dependency Manifest
├── README.md                         # Operations Terminal Manual
│
│   /* Automatically Generated Pipeline Artifacts */
├── best_model.pth                    # Evaluated Neural Network Weights
├── scaler.pkl                        # Fitted StandardScaler Instance
├── encoder.pkl                       # Fitted OneHotEncoder Instance
├── feature_columns.pkl               # Sorted Structural Feature Matrix Map
├── categorical_columns.pkl           # Categorical Feature Name Registry
└── numerical_columns.pkl             # Numerical Feature Name Registry