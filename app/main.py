import sys
import os

# --- Ensure project root is discoverable ---
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
from app.pages.login import show_login_page
from src.auth.auth_handler import verify_google_login


# 1. Page Configuration
st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# 2. Session State Initialization
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if "theme" not in st.session_state:
    st.session_state["theme"] = "Dark"

if "user_email" not in st.session_state:
    st.session_state["user_email"] = None


# 3. Handle Google OAuth Redirect
query_params = st.query_params

if "code" in query_params and not st.session_state["authenticated"]:
    success, email = verify_google_login(query_params["code"])

    if success:
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = email
        st.query_params.clear()
        st.success(f"Successfully logged in via Google as {email}")
    else:
        st.error("Google Login Failed.")


# 4. Sidebar Settings
st.sidebar.title("Settings ⚙️")

theme_choice = st.sidebar.radio(
    "App Theme",
    ["Dark", "Light"],
    index=0 if st.session_state["theme"] == "Dark" else 1
)

if theme_choice != st.session_state["theme"]:
    st.session_state["theme"] = theme_choice
    st.rerun()


# 5. Light Theme Styling
if st.session_state["theme"] == "Light":
    light_css = """
    <style>
        .stApp { background-color: #F0F2F6; color: black; }
        h1, h2, h3, p, label { color: black !important; }
    </style>
    """
    st.markdown(light_css, unsafe_allow_html=True)


# 6. Routing
if not st.session_state["authenticated"]:
    show_login_page()

else:
    st.title("📊 FinGuard AI Dashboard")
    st.write(f"Welcome, **{st.session_state.get('user_email', 'User')}**!")
    st.success("Authentication Module Complete! Ready to build the anomaly detection upload feature.")

    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["user_email"] = None
        st.rerun()