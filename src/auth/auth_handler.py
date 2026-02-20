import random
import string
import requests
import os
from src.database.db_manager import add_user, get_user, check_password, verify_user
from src.utils.email_sender import send_otp
from dotenv import load_dotenv

load_dotenv()

def generate_otp():
    """Generates a 6-digit random OTP."""
    return ''.join(random.choices(string.digits, k=6))

def signup_user(email, password):
    """Checks domain and initiates signup."""
    allowed_domain = os.getenv("ALLOWED_DOMAIN", "@ycce.in")
    if not email.endswith(allowed_domain):
        return False, f"Access restricted to {allowed_domain} emails only."
    
    user = get_user(email)
    if user:
        return False, "Email already registered."
    
    # Save user to DB but unverified
    success = add_user(email, password)
    if success:
        otp = generate_otp()
        send_otp(email, otp)
        return True, otp
    return False, "Database error."

def login_user(email, password):
    """Verifies login credentials."""
    user = get_user(email)
    if not user:
        return False, "User not found."
    if not check_password(password, user[2]): # user[2] is password_hash
        return False, "Incorrect password."
    if not user[3]: # user[3] is is_verified
        return False, "Account not verified. Please sign up again to verify OTP."
    return True, "Login successful."

def reset_password_request(email):
    """Sends OTP for password reset."""
    user = get_user(email)
    if not user:
        return False, "User not found."
    
    otp = generate_otp()
    send_otp(email, otp)
    return True, otp

def get_google_login_url():
    """Generates the Google OAuth login URL."""
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    redirect_uri = os.getenv("REDIRECT_URI")
    scope = "openid email profile"
    return f"https://accounts.google.com/o/oauth2/v2/auth?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&scope={scope}"

def verify_google_login(auth_code):
    """Exchanges Google auth code for user email."""
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": auth_code,
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uri": os.getenv("REDIRECT_URI"),
        "grant_type": "authorization_code"
    }
    response = requests.post(token_url, data=data)
    if response.status_code == 200:
        access_token = response.json().get("access_token")
        user_info = requests.get("https://www.googleapis.com/oauth2/v2/userinfo", headers={"Authorization": f"Bearer {access_token}"})
        return True, user_info.json().get("email")
    return False, None