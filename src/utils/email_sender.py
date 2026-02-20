import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv

load_dotenv()

SENDER_EMAIL = os.getenv("EMAIL_SENDER")
SENDER_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_email(to_email, subject, body):
    """Sends an email using Gmail SMTP."""
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'html'))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def send_otp(to_email, otp):
    subject = "FinGuard AI - Your Verification OTP"
    body = f"<h3>Welcome to FinGuard AI!</h3><p>Your verification OTP is: <strong>{otp}</strong></p><p>Please do not share this code with anyone.</p>"
    return send_email(to_email, subject, body)

def send_fraud_alert(to_email, transaction_details):
    subject = "🚨 ALERT: High-Risk Transaction Detected"
    body = f"<h3>FinGuard AI Alert</h3><p>A suspicious transaction was just detected.</p><p><strong>Details:</strong> {transaction_details}</p><p>Please log in to the dashboard immediately to investigate.</p>"
    return send_email(to_email, subject, body)