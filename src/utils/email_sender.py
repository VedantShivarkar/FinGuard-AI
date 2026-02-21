import smtplib
import os
import streamlit as st
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

def send_email(to_email, subject, body):
    S_EMAIL = os.getenv("EMAIL_SENDER")
    S_PASS = os.getenv("EMAIL_PASSWORD")

    if not S_EMAIL or not S_PASS:
        st.error("❌ Email Credentials Missing in Secrets!")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = S_EMAIL
        msg['To'] = str(to_email) # Force string conversion
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))

        # Using SSL Port 465 for better reliability on Cloud
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(S_EMAIL, S_PASS)
        server.send_message(msg)
        server.quit()
        
        print(f"DEBUG: Email successfully sent to {to_email}")
        return True
    except Exception as e:
        st.error(f"📧 SMTP Error: {str(e)}")
        print(f"DEBUG: Email failed: {e}")
        return False

def send_otp(to_email, otp):
    subject = "FinGuard AI - Verification Code"
    body = f"<div style='font-family:sans-serif;'><h2>Your OTP is: <span style='color:blue;'>{otp}</span></h2></div>"
    return send_email(to_email, subject, body)

def send_fraud_alert(to_email, transaction_details):
    subject = "🚨 ALERT: High-Risk Transaction Detected"
    # Create a clean HTML list for the details
    details_html = "".join([f"<li><b>{k}</b>: {v}</li>" for k, v in transaction_details.items()])
    body = f"""
    <div style='font-family:sans-serif; border:2px solid red; padding:20px;'>
        <h2 style='color:red;'>Suspicious Activity Detected</h2>
        <p>The FinGuard AI engine has flagged the following transaction:</p>
        <ul>{details_html}</ul>
        <p><b>Action:</b> Immediate manual audit required.</p>
    </div>
    """
    return send_email(to_email, subject, body)