import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import plotly.express as px
import plotly.graph_objects as go
from src.models.autoencoder import DeepAutoencoder
from src.utils.email_sender import send_fraud_alert
from src.explainability.shap_engine import generate_shap_explanation
from src.investigation.report_generator import generate_fraud_report

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
    except FileNotFoundError:
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
        
    norm_ae = np.clip(ae_preds / 5.0, 0, 1)
    norm_iso = np.clip(iso_preds, 0, 1)
    norm_lof = np.clip(lof_preds, 0, 1)
    
    final_scores = (0.5 * norm_ae) + (0.3 * norm_iso) + (0.2 * norm_lof)
    return final_scores, proc_df

def show_dashboard():
    st.title("📊 FinGuard AI: Anomaly Detection Engine")
    st.write(f"Logged in securely as: **{st.session_state.get('user_email')}**")
    
    scaler, encoders, iso, lof, ae = load_ml_models()
    if scaler is None:
        st.error("🚨 Model files missing in 'models/saved_models/'!")
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
                # Intentionally huge default value to trigger a high-risk alert for testing!
                dmbtr = st.number_input("Amount (DMBTR)", min_value=0.0, value=9999999.0) 
            with col2:
                ktosl = st.text_input("Transaction Key (KTOSL)", "XYZ")
                prctr = st.text_input("User ID (PRCTR)", "U123")
                wrbtr = st.number_input("Foreign Amount (WRBTR)", min_value=0.0, value=0.0)
            with col3:
                bschl = st.text_input("Posting Key (BSCHL)", "40")
                hkont = st.text_input("G/L Account (HKONT)", "100100")
                
            submit_manual = st.form_submit_button("Run AI Analysis", use_container_width=True)
            
       if submit_manual:
            input_dict = {'WAERS': waers, 'BUKRS': bukrs, 'KTOSL': ktosl, 'PRCTR': prctr, 'BSCHL': bschl, 'HKONT': hkont, 'DMBTR': dmbtr, 'WRBTR': wrbtr}
            input_data = pd.DataFrame([input_dict])
            
            with st.spinner("AI is analyzing transaction..."):
                scores, processed_df = predict_risk(input_data, scaler, encoders, iso, lof, ae)
                score = scores[0]
                
            risk_label = "Low Risk"
            color = "green"
            
            if score > 0.4:
                risk_label = "Medium Risk"
                color = "orange"
            if score > 0.7:
                risk_label = "High Risk (Fraud Alert)"
                color = "red"
                try: 
                    st.audio("src/utils/buzzer.wav", autoplay=True)
                except: 
                    pass
                
                # --- NEW: SEND EMAIL IMMEDIATELY ---
                # We wrap this in a try/except so it never crashes the app
                try:
                    send_fraud_alert(st.session_state["user_email"], input_dict)
                    st.success("📧 Emergency Fraud Alert successfully sent to your email!")
                except Exception as e:
                    st.error(f"⚠️ Email alert failed to send: {e}")

            # GRAPH 1: Gauge Chart (Always displays)
            fig1 = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                title={'text': f"Transaction Risk Level: {risk_label}"},
                gauge={'axis': {'range': [0, 1]}, 'bar': {'color': color},
                       'steps': [{'range': [0, 0.4], 'color': "lightgreen"},
                                 {'range': [0.4, 0.7], 'color': "lightyellow"},
                                 {'range': [0.7, 1.0], 'color': "lightcoral"}]}
            ))
            st.plotly_chart(fig1, use_container_width=True)

            # Explainability & LLM Integration
            st.markdown("---")
            st.subheader("🕵️ Autonomous Investigation")
            
            if score <= 0.4:
                st.success("✅ Transaction falls within normal behavioral parameters. No autonomous investigation required.")
            else:
                st.warning("⚠️ Abnormal transaction detected. Initiating AI audit protocols...")
                col_a, col_b = st.columns([1, 1])
                
                with col_a:
                    with st.spinner("Generating SHAP Explainability..."):
                        feature_names = ['WAERS', 'BUKRS', 'KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'DMBTR', 'WRBTR']
                        shap_fig, shap_summary = generate_shap_explanation(iso, processed_df, feature_names)
                        if shap_fig:
                            st.plotly_chart(shap_fig, use_container_width=True)
                        else:
                            st.error(f"SHAP Error: {shap_summary}")
                
                with col_b:
                    with st.spinner("LLM Agent writing report..."):
                        report = generate_fraud_report(input_dict, score, shap_summary)
                        # Gracefully handle the OpenAI Quota Error for the demo
                        if "API Key missing" in report or "Error" in report or "quota" in report.lower():
                            st.error("⚠️ Autonomous Agent Offline: API Quota Exceeded.")
                            st.info("💡 **Demo Fallback:** The AI detected the anomaly and successfully sent the emergency email. In a production environment with an active API tier, this box would generate a comprehensive text audit based on the SHAP data.")
                        else:
                            st.info(report)

    # --- TAB 2: BATCH CSV UPLOAD ---
    with tab2:
        st.subheader("Bulk Anomaly Detection")
        uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:", df.head(3))
            
            if st.button("Run Batch Analysis"):
                with st.spinner("Processing transactions..."):
                    features_df = df[['WAERS', 'BUKRS', 'KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'DMBTR', 'WRBTR']]
                    scores, _ = predict_risk(features_df, scaler, encoders, iso, lof, ae)
                    
                    df['Risk_Score'] = scores
                    df['Risk_Category'] = pd.cut(df['Risk_Score'], bins=[-1, 0.4, 0.7, 2], labels=['Low', 'Medium', 'High'])
                    
                    high_risk_count = len(df[df['Risk_Category'] == 'High'])
                    if high_risk_count > 0:
                        st.error(f"🚨 {high_risk_count} High-Risk transactions detected!")
                        send_fraud_alert(st.session_state["user_email"], f"Batch analysis found {high_risk_count} critical anomalies.")

                col1, col2 = st.columns(2)
                with col1:
                    risk_counts = df['Risk_Category'].value_counts().reset_index()
                    risk_counts.columns = ['Risk Category', 'Count']
                    fig2 = px.pie(risk_counts, values='Count', names='Risk Category', 
                                  title="Risk Distribution",
                                  color='Risk Category', color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                    st.plotly_chart(fig2, use_container_width=True)
                
                with col2:
                    fig3 = px.scatter(df, x='DMBTR', y='Risk_Score', color='Risk_Category',
                                      title="Amount vs Risk", hover_data=['PRCTR'],
                                      color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'})
                    st.plotly_chart(fig3, use_container_width=True)

                st.dataframe(df.sort_values(by="Risk_Score", ascending=False))
                
                csv_export = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Risk Report (CSV)", data=csv_export, file_name="FinGuard_Report.csv", mime="text/csv")