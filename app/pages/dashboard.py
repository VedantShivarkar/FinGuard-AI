import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from src.models.autoencoder import DeepAutoencoder
from src.utils.email_sender import send_fraud_alert
from src.explainability.shap_engine import generate_shap_explanation
from src.investigation.report_generator import generate_fraud_report

warnings.filterwarnings("ignore", category=UserWarning)

@st.cache_resource
def load_ml_models():
    try:
        base = "models/saved_models/"
        scaler = pickle.load(open(base + 'scaler.pkl', 'rb'))
        encoders = pickle.load(open(base + 'encoders.pkl', 'rb'))
        iso = pickle.load(open(base + 'isolation_forest.pkl', 'rb'))
        lof = pickle.load(open(base + 'lof.pkl', 'rb'))
        ae = DeepAutoencoder(8) 
        ae.load_state_dict(torch.load(base + 'autoencoder.pth', map_location=torch.device('cpu')))
        ae.eval()
        return scaler, encoders, iso, lof, ae
    except:
        return None, None, None, None, None

def predict_risk(df, scaler, encoders, iso, lof, ae):
    proc_df = df.copy()
    cat_cols = ['WAERS', 'BUKRS', 'KTOSL', 'PRCTR', 'BSCHL', 'HKONT']
    for col in cat_cols:
        proc_df[col] = proc_df[col].astype(str).apply(
            lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
        )
        proc_df[col] = encoders[col].transform(proc_df[col])
    proc_df[['DMBTR', 'WRBTR']] = scaler.transform(proc_df[['DMBTR', 'WRBTR']])
    iso_preds = -iso.score_samples(proc_df)
    lof_preds = -lof.score_samples(proc_df)
    X_tensor = torch.FloatTensor(proc_df.values)
    with torch.no_grad():
        recon, _ = ae(X_tensor)
        ae_preds = torch.mean((X_tensor - recon)**2, dim=1).numpy()
    final_scores = (0.5 * np.clip(ae_preds/5,0,1)) + (0.3 * np.clip(iso_preds,0,1)) + (0.2 * np.clip(lof_preds,0,1))
    return final_scores, proc_df

def show_dashboard():
    st.title("📊 FinGuard AI Dashboard")
    user_email = str(st.session_state.get('user_email', ''))
    st.write(f"Logged in as: **{user_email}**")
    
    scaler, encoders, iso, lof, ae = load_ml_models()
    if not scaler:
        st.error("Model files missing!")
        st.stop()

    tab1, tab2 = st.tabs(["📝 Manual Entry", "📂 Batch Upload"])

    with tab1:
        with st.form("manual_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                waers, bukrs = st.text_input("Currency", "USD"), st.text_input("Company", "1000")
                dmbtr = st.number_input("Amount", value=9999999.0)
            with col2:
                ktosl, prctr = st.text_input("Key", "XYZ"), st.text_input("User ID", "U123")
                wrbtr = st.number_input("Foreign Amount", value=0.0)
            with col3:
                bschl, hkont = st.text_input("Posting", "40"), st.text_input("Account", "100100")
            submit = st.form_submit_button("Run Analysis", use_container_width=True)

        if submit:
            input_dict = {'WAERS': waers, 'BUKRS': bukrs, 'KTOSL': ktosl, 'PRCTR': prctr, 'BSCHL': bschl, 'HKONT': hkont, 'DMBTR': dmbtr, 'WRBTR': wrbtr}
            scores, proc_df = predict_risk(pd.DataFrame([input_dict]), scaler, encoders, iso, lof, ae)
            score = scores[0]
            
            if score > 0.7:
                st.error("🚨 HIGH RISK DETECTED")
                # 1. Buzzer
                try:
                    audio_path = os.path.join(os.getcwd(), "src", "utils", "buzzer.wav")
                    with open(audio_path, 'rb') as f:
                        st.audio(f.read(), format='audio/wav', autoplay=True)
                except: pass
                # 2. Alert Email
                if user_email:
                    with st.spinner("Dispatching alert..."):
                        if send_fraud_alert(user_email, input_dict):
                            st.success(f"📧 Alert sent to {user_email}")

            # Risk Gauge
            fig = go.Figure(go.Indicator(mode="gauge+number", value=score, gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "red" if score > 0.7 else "green"}}))
            st.plotly_chart(fig)

            if score > 0.4:
                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    fig_s, sum_s = generate_shap_explanation(iso, proc_df, list(input_dict.keys()))
                    st.plotly_chart(fig_s)
                with col_b:
                    st.info(generate_fraud_report(input_dict, score, sum_s))

    with tab2:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded and st.button("Batch Analysis"):
            df = pd.read_csv(uploaded)
            feats = df[['WAERS', 'BUKRS', 'KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'DMBTR', 'WRBTR']]
            scores, _ = predict_risk(feats, scaler, encoders, iso, lof, ae)
            df['Risk_Score'] = scores
            st.dataframe(df.sort_values(by="Risk_Score", ascending=False))
            if any(df['Risk_Score'] > 0.7):
                send_fraud_alert(user_email, {"Status": "Multiple anomalies found in batch upload."})