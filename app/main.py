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
from openai import OpenAI
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
    
    input_dim = 8 # Features DMBTR, WRBTR, WAERS, BUKRS, KTOSL, PRCTR, BSCHL, HKONT
    ae = DeepAutoencoder(input_dim)
    ae.load_state_dict(torch.load('models/saved_models/autoencoder.pth', map_location=torch.device('cpu')))
    ae.eval()
    return scaler, encoders, iso, lof, ae, user_profiles

scaler, encoders, iso, lof, ae, user_profiles = load_models()

# --- AI INVESTIGATOR (LLM) ---
def generate_report(tx_data, risk_score):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"""
    You are an Autonomous Investigation Assistant. Generate a structured risk report for this transaction.
    Transaction Details: {tx_data.to_dict()}
    Risk Score: {risk_score} (High > 0.7)
    Format the response with: Case ID, Risk Breakdown, Recommended Action, Confidence Level.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- UI DASHBOARD ---
st.title("🛡️ FinGuard AI: Real-Time Fraud Surveillance")

# Simulate a live transaction stream
st.sidebar.header("Control Panel")
if st.sidebar.button("Simulate Incoming Transaction"):

    if "tx_counter" not in st.session_state:
        st.session_state["tx_counter"] = 0

    st.session_state["tx_counter"] += 1

    # Every 10th transaction → fixed injected case
    if st.session_state["tx_counter"] % 10 == 0:
        sample_tx = pd.DataFrame([{
            
            "WAERS": "C99",            # Unusual currency for this user
             "BUKRS": "C10",            # Unusual company code
             "KTOSL": "C5",             # Unusual transaction key
             "PRCTR": "C20",            # A KNOWN USER from the training data!
            "BSCHL": "A9",             # Unusual posting key
            "HKONT": "B9",             # Unusual general ledger account
            "DMBTR": 999999999.00,     # Absurdly high transaction amount
            "WRBTR": 999999999.00      # Absurdly high foreign currency amount
            
        }])
    else:
        sample_tx = pd.DataFrame([{
            
            "WAERS": "C1",
            "BUKRS": "C17",
            "KTOSL": "C1",
            "PRCTR": "C16",
            "BSCHL": "A1",
            "HKONT": "B1",
            "DMBTR": np.random.uniform(50000, 5000000),
            "WRBTR": np.random.uniform(0, 10000)
            
        }])

    # IMPORTANT: keep only model features
    proc_tx = sample_tx.copy()
    proc_tx = proc_tx[[
        "WAERS","BUKRS","KTOSL","PRCTR",
        "BSCHL","HKONT","DMBTR","WRBTR"
    ]]
    
    # Preprocess
    proc_tx = sample_tx.copy()
    for col in ['WAERS', 'BUKRS', 'KTOSL', 'PRCTR', 'BSCHL', 'HKONT']:
        # Handle unseen labels gracefully by assigning to a default if needed, or transform
        try:
            proc_tx[col] = encoders[col].transform(proc_tx[col].astype(str))
        except:
            proc_tx[col] = 0
            
    proc_tx[['DMBTR', 'WRBTR']] = scaler.transform(proc_tx[['DMBTR', 'WRBTR']])
    proc_tx = proc_tx.apply(pd.to_numeric, errors="coerce")
    proc_tx = proc_tx.fillna(0)
    proc_tx = proc_tx.astype(np.float32)
    X_tensor = torch.tensor(proc_tx.values, dtype=torch.float32)

    
    # Get base models scores
    iso_sc = -iso.score_samples(proc_tx)[0]
    lof_sc = -lof.score_samples(proc_tx)[0]
    
    with torch.no_grad():
        recon, embed = ae(X_tensor)
        ae_sc = torch.mean((X_tensor - recon)**2).item()
    
    # Normalize for formula (fake normalization for demo)
    norm_ae = min(ae_sc / 5.0, 1.0)
    norm_iso = min(iso_sc, 1.0)
    norm_lof = min(lof_sc, 1.0)
    
    base_score = 0.5 * norm_ae + 0.3 * norm_iso + 0.2 * norm_lof
    
    # Behavioral drift calculation
    from scipy.spatial.distance import cosine
    user = sample_tx['PRCTR'][0]
    behavior_score = 0
    if user in user_profiles:
        baseline = user_profiles[user]
        behavior_score = cosine(embed.numpy()[0], baseline)
        if np.isnan(behavior_score): behavior_score = 0
        
    final_score = 0.4 * base_score + 0.3 * behavior_score + 0.3 * norm_ae # Risk score
    
    # Check Risk
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
    
    # SHAP Explainability
    st.write("### AI Explainability (Feature Importance)")
    explainer = shap.TreeExplainer(iso)
    shap_values = explainer.shap_values(proc_tx)
    fig = px.bar(x=proc_tx.columns, y=shap_values[0], title="Factors Driving Anomaly Score")
    st.plotly_chart(fig)
    
    # LLM Investigation
    if final_score > 0.4:
        with st.spinner("Autonomous Investigator analyzing case..."):
            report = generate_report(sample_tx, final_score)
            st.info("📝 **Autonomous Investigation Report**")
            st.write(report)