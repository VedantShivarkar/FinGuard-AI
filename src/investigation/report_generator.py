import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_fraud_report(transaction_data, risk_score, shap_summary):
    """Uses an LLM to write a professional investigation report."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_actual_openai_api_key_here":
        return "⚠️ **OpenAI API Key missing.** Please add your actual key to the .env file to view the autonomous AI report."
    
    try:
        client = OpenAI(api_key=api_key)
        prompt = f"""
        You are an expert Financial Fraud Investigator AI. 
        Analyze the following transaction which was flagged with a risk score of {risk_score:.2f} (High Risk is > 0.70).
        
        Transaction Details:
        {transaction_data}
        
        Key Contributing Factors (SHAP Analysis):
        {shap_summary}
        
        Generate a highly professional, concise investigation report containing exactly these 3 sections:
        1. 📌 Case Summary
        2. 🔍 Risk Breakdown (Why it was flagged, referencing the SHAP features)
        3. 🛡️ Recommended Actions (e.g., Block account, Call customer, etc.)
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are FinGuard AI's Autonomous Investigator agent."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3 # Keep it analytical and factual
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ LLM Generation Error: {str(e)}"