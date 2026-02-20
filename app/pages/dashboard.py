import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import plotly.express as px
import plotly.graph_objects as go
import os
from src.models.autoencoder import DeepAutoencoder
from src.utils.email_sender import send_fraud_alert

# --- MODEL LOADING (CACHED FOR SPEED) ---
@st.cache_resource
def load_ml_models():
    """Loads the pre-trained models from the saved_models folder."""
    try:
        base = "models/saved_models/"
        scaler = pickle.load(open(base + 'scaler.pkl', 'rb'))
        encoders = pickle.load(open(base + 'encoders.pkl', 'rb'))
        iso = pickle.load(open(base + 'isolation_forest.pkl', 'rb'))
        lof = pickle.load(open(base + 'lof.pkl', 'rb'))
        
        ae = DeepAutoencoder(8) # 8 features used during training
        ae.load_state_dict(torch.load(base + 'autoencoder.pth', map_location=torch.device('cpu')))
        ae.eval()
        return scaler, encoders, iso, lof, ae
    except FileNotFoundError:
        return None, None, None, None, None

def predict_risk(df, scaler, encoders, iso, lof, ae):
    """Runs the ensemble AI models on a dataframe."""
    proc_df = df.copy()
    
    # 1. Encode Categorical Variables safely
    cat_cols = ['WAERS', 'BUKRS', 'KTOSL', 'PRCTR', 'BSCHL', 'HKONT']
    for col in cat_cols:
        # If a new category appears, default to the first known category to prevent crashes
        proc_df[col] = proc_df[col].astype(str).apply(
            lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0]
        )
        proc_df[col] = encoders[col].transform(proc_df[col])
        
    # 2. Scale Numerical Variables
    proc_df[['DMBTR', 'WRBTR']] = scaler.transform(proc_df[['DMBTR', 'WRBTR']])
    
    # 3. Get Scikit-Learn Scores
    iso_preds = -iso.score_samples(proc_df)
    lof_preds = -lof.score_samples(proc_df)
    
    # 4. Get PyTorch Autoencoder Scores
    X_tensor = torch.FloatTensor(proc_df.values)
    with torch.no_grad():
        recon, _ = ae(X_tensor)
        ae_preds = torch.mean((X_tensor - recon)**2, dim=1).numpy()
        
    # 5. Ensemble Risk Calculation
    norm_ae = np.clip(ae_preds / 5.0, 0, 1)
    norm_iso = np.clip(iso_preds, 0, 1)
    norm_lof = np.clip(lof_preds, 0, 1)
    
    final_scores = (0.5 * norm_ae) + (0.3 * norm_iso) + (0.2 * norm_lof)
    return final_scores

def show_dashboard():
    st.title("📊 FinGuard AI: Anomaly Detection Engine")
    st.write(f"Logged in securely as: **{st.session_state.get('user_email')}**")
    
    scaler, encoders, iso, lof, ae = load_ml_models()
    
    if scaler is None:
        st.error("🚨 Model files missing! Please ensure you have placed scaler.pkl, encoders.pkl, isolation_forest.pkl, lof.pkl, and autoencoder.pth inside the 'models/saved_models/' folder.")
        st.stop()

    tab1, tab2 = st.tabs(["📝 Manual Data Entry", "📂 Batch CSV Upload"])

    # --- TAB 1: MANUAL ENTRY ---
    with tab1:
        st.subheader("Analyze Single Transaction")
        with st.form("manual_input_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                waers = st.text_input("Currency (WAERS)", "USD")
                bukrs = st.text_input("Company Code (BUKRS)", "1000")
                dmbtr = st.number_input("Amount in Local Currency (DMBTR)", min_value=0.0, value=500.0)
            with col2:
                ktosl = st.text_input("Transaction Key (KTOSL)", "XYZ")
                prctr = st.text_input("Profit Center / User ID (PRCTR)", "U123")
                wrbtr = st.number_input("Amount in Foreign Currency (WRBTR)", min_value=0.0, value=0.0)
            with col3:
                bschl = st.text_input("Posting Key (BSCHL)", "40")
                hkont = st.text_input("G/L Account (HKONT)", "100100")
                
            submit_manual = st.form_submit_button("Run AI Analysis", use_container_width=True)
            
        if submit_manual:
            input_data = pd.DataFrame([{
                'WAERS': waers, 'BUKRS': bukrs, 'KTOSL': ktosl, 'PRCTR': prctr, 
                'BSCHL': bschl, 'HKONT': hkont, 'DMBTR': dmbtr, 'WRBTR': wrbtr
            }])
            
            with st.spinner("AI is analyzing transaction..."):
                score = predict_risk(input_data, scaler, encoders, iso, lof, ae)[0]
                
            risk_label = "Low Risk"
            color = "green"
            if score > 0.4:
                risk_label = "Medium Risk"
                color = "orange"
            if score > 0.7:
                risk_label = "High Risk (Fraud Alert)"
                color = "red"
                st.audio("src/utils/buzzer.wav", autoplay=True) # Ensure buzzer.wav is in src/utils
                # Send Email Alert!
                send_fraud_alert(st.session_state["user_email"], input_data.to_dict('records')[0])
                st.error("🚨 Fraud Alert sent to your email!")

            # GRAPH 1: Gauge Chart for Risk Score
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                title={'text': f"Transaction Risk Level: {risk_label}"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 0.4], 'color': "lightgreen"},
                        {'range': [0.4, 0.7], 'color': "lightyellow"},
                        {'range': [0.7, 1.0], 'color': "lightcoral"}
                    ]
                }
            ))
            st.plotly_chart(fig1, use_container_width=True)

    # --- TAB 2: BATCH CSV UPLOAD ---
    with tab2:
        st.subheader("Bulk Anomaly Detection")
        uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:", df.head())
            
            if st.button("Run Batch Analysis"):
                with st.spinner("Processing thousands of transactions through the Ensemble Model..."):
                    # We only need the 8 features for prediction
                    features_df = df[['WAERS', 'BUKRS', 'KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'DMBTR', 'WRBTR']]
                    scores = predict_risk(features_df, scaler, encoders, iso, lof, ae)
                    
                    df['Risk_Score'] = scores
                    df['Risk_Category'] = pd.cut(df['Risk_Score'], bins=[-1, 0.4, 0.7, 2], labels=['Low', 'Medium', 'High'])
                    
                    high_risk_count = len(df[df['Risk_Category'] == 'High'])
                    if high_risk_count > 0:
                        st.error(f"🚨 {high_risk_count} High-Risk transactions detected!")
                        send_fraud_alert(st.session_state["user_email"], f"Batch analysis found {high_risk_count} critical anomalies.")

                st.success("Analysis Complete!")
                
                col1, col2 = st.columns(2)
                
                # GRAPH 2: Pie Chart of Risk Distribution
                with col1:
                    risk_counts = df['Risk_Category'].value_counts().reset_index()
                    risk_counts.columns = ['Risk Category', 'Count']
                    fig2 = px.pie(risk_counts, values='Count', names='Risk Category', 
                                  title="Risk Distribution Across Dataset",
                                  color='Risk Category',
                                  color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                    st.plotly_chart(fig2, use_container_width=True)
                
                # GRAPH 3: Scatter Plot (Transaction Amount vs Risk Score)
                
                with col2:
                    fig3 = px.scatter(df, x='DMBTR', y='Risk_Score', color='Risk_Category',
                                      title="Transaction Amount vs. AI Risk Score",
                                      hover_data=['PRCTR', 'WAERS'],
                                      color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                    st.plotly_chart(fig3, use_container_width=True)

                st.write("### Detailed Results")
                st.dataframe(df.sort_values(by="Risk_Score", ascending=False))
                
                # --- CSV EXPORT FEATURE ---
                csv_export = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Analyzed Report (CSV)",
                    data=csv_export,
                    file_name="FinGuard_Risk_Report.csv",
                    mime="text/csv",
                )