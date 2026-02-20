import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import shap
import os
import time
from dotenv import load_dotenv
import plotly.express as px

load_dotenv()

# --- SECURITY: AUTHENTICATION ---
USERNAME = os.getenv("AUTH_USERNAME")
PASSWORD = os.getenv("AUTH_PASSWORD")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    st.title("🔒 FinGuard AI Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == USERNAME and pwd == PASSWORD:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# --- ARCHITECTURE SETUP ---
st.set_page_config(page_title="FinGuard AI", layout="wide")

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec, enc

# --- CACHED LOADERS ---
@st.cache_resource
def load_models():
    scaler = pickle.load(open('models/saved_models/scaler.pkl', 'rb'))
    encoders = pickle.load(open('models/saved_models/encoders.pkl', 'rb'))
    iso = pickle.load(open('models/saved_models/isolation_forest.pkl', 'rb'))
    lof = pickle.load(open('models/saved_models/lof.pkl', 'rb'))
    user_profiles = pickle.load(open('models/saved_models/user_profiles.pkl', 'rb'))

    input_dim = 8
    ae = DeepAutoencoder(input_dim)
    ae.load_state_dict(torch.load('models/saved_models/autoencoder.pth', map_location=torch.device('cpu')))
    ae.eval()

    return scaler, encoders, iso, lof, ae, user_profiles

scaler, encoders, iso, lof, ae, user_profiles = load_models()

# --- AI INVESTIGATOR (MOCKED FOR DEMO TO AVOID API COSTS) ---
def generate_report(tx_data, risk_score):
    import random
    import time

    case_id = f"CASE-{random.randint(10000, 99999)}"

    report = f"""
    **Case ID:** {case_id}

    **Risk Breakdown:** This transaction has been flagged with a severe risk score of **{risk_score:.2f}**.
    The primary drivers for this anomaly are:
    1. Extreme deviation from the user's historical transaction amounts.
    2. Significant behavioral drift detected by the deep learning autoencoder.
    3. Unexpected merchant category and currency combination.

    **Recommended Action:** IMMEDIATE ACTION REQUIRED. Temporarily freeze the profit center account and escalate to a human fraud analyst for verification.

    **Confidence Level:** 92%
    """

    time.sleep(1.5)
    return report

# --- UI DASHBOARD ---
st.title("🛡️ FinGuard AI: Real-Time Fraud Surveillance")

st.sidebar.header("Control Panel")

if st.sidebar.button("Simulate Incoming Transaction"):

    sample_tx = pd.DataFrame([{
        "WAERS": "C99",
        "BUKRS": "C10",
        "KTOSL": "C5",
        "PRCTR": "C20",
        "BSCHL": "A9",
        "HKONT": "B9",
        "DMBTR": 999999999.00,
        "WRBTR": 999999999.00
    }])

    FEATURE_COLUMNS = [
        "WAERS","BUKRS","KTOSL","PRCTR",
        "BSCHL","HKONT","DMBTR","WRBTR"
    ]

    proc_tx = sample_tx[FEATURE_COLUMNS].copy()

    for col in ['WAERS', 'BUKRS', 'KTOSL', 'PRCTR', 'BSCHL', 'HKONT']:
        try:
            proc_tx[col] = encoders[col].transform(proc_tx[col].astype(str))
        except:
            proc_tx[col] = 0

    proc_tx[['DMBTR', 'WRBTR']] = scaler.transform(proc_tx[['DMBTR', 'WRBTR']])

    proc_tx = proc_tx.apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
    X_tensor = torch.tensor(proc_tx.values, dtype=torch.float32)

    iso_sc = -iso.score_samples(proc_tx)[0]
    lof_sc = -lof.score_samples(proc_tx)[0]

    with torch.no_grad():
        recon, embed = ae(X_tensor)
        ae_sc = torch.mean((X_tensor - recon)**2).item()

    norm_ae = min(ae_sc / 5.0, 1.0)
    norm_iso = min(iso_sc, 1.0)
    norm_lof = min(lof_sc, 1.0)

    base_score = 0.5 * norm_ae + 0.3 * norm_iso + 0.2 * norm_lof

    from scipy.spatial.distance import cosine
    user = sample_tx['PRCTR'][0]
    behavior_score = 0

    if user in user_profiles:
        baseline = user_profiles[user]
        behavior_score = cosine(embed.numpy()[0], baseline)
        if np.isnan(behavior_score):
            behavior_score = 0

    final_score = 0.4 * base_score + 0.3 * behavior_score + 0.3 * norm_ae

    risk_category = "Low"
    color = "green"

    if final_score > 0.4:
        risk_category = "Medium"
        color = "orange"

    if final_score > 0.7:
        risk_category = "High Risk"
        color = "red"
        st.error("🚨 HIGH RISK TRANSACTION DETECTED!")
        try:
            st.audio("src/utils/buzzer.wav", autoplay=True)
        except:
            pass

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk Score", f"{final_score:.3f}")
    col2.markdown(f"<h2 style='color: {color}'>{risk_category}</h2>", unsafe_allow_html=True)
    col3.metric("Behavioral Drift", f"{behavior_score:.3f}")

    st.write("### Transaction Details", sample_tx)

    st.write("### AI Explainability (Feature Importance)")
    explainer = shap.TreeExplainer(iso)
    shap_values = explainer.shap_values(proc_tx)
    fig = px.bar(x=proc_tx.columns, y=shap_values[0], title="Factors Driving Anomaly Score")
    st.plotly_chart(fig)

    if final_score > 0.4:
        with st.spinner("Autonomous Investigator analyzing case..."):
            report = generate_report(sample_tx, final_score)
            st.info("📝 **Autonomous Investigation Report**")
            st.write(report)