import streamlit as st
from src.auth.auth_handler import signup_user, login_user, reset_password_request, get_google_login_url
from src.database.db_manager import verify_user, init_db
import sqlite3
import bcrypt

def show_login_page():
    st.title("🛡️ FinGuard AI Gateway")
    st.markdown("Secure access for authorized institutional members only.")

    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])

    # --- LOGIN TAB ---
    with tab1:
        st.subheader("Login to your account")
        login_email = st.text_input("Email", key="login_email")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", use_container_width=True):
            success, msg = login_user(login_email, login_pass)
            if success:
                st.session_state["authenticated"] = True
                st.session_state["user_email"] = login_email
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)
                
        st.markdown("---")
        # Google Login Button
        google_url = get_google_login_url()
        st.link_button("🌐 Continue with Google", google_url, use_container_width=True)

    # --- SIGN UP TAB (OTP Flow) ---
    with tab2:
        st.subheader("Register New Account")
        signup_email = st.text_input("Institutional Email (@ycce.in)", key="signup_email")
        signup_pass = st.text_input("Create Password", type="password", key="signup_pass")
        
        if "signup_otp_sent" not in st.session_state:
            st.session_state["signup_otp_sent"] = False
            
        if not st.session_state["signup_otp_sent"]:
            if st.button("Send OTP"):
                success, response = signup_user(signup_email, signup_pass)
                if success:
                    st.session_state["signup_otp_sent"] = True
                    st.session_state["current_otp"] = response
                    st.session_state["pending_email"] = signup_email
                    st.success("OTP sent to your email! Please check your inbox.")
                    st.rerun()
                else:
                    st.error(response)
        else:
            entered_otp = st.text_input("Enter 6-digit OTP")
            if st.button("Verify & Create Account"):
                if entered_otp == st.session_state["current_otp"]:
                    verify_user(st.session_state["pending_email"])
                    st.success("Account verified! You can now log in.")
                    st.session_state["signup_otp_sent"] = False
                else:
                    st.error("Invalid OTP. Try again.")

    # --- FORGOT PASSWORD TAB ---
    with tab3:
        st.subheader("Reset Password")
        reset_email = st.text_input("Registered Email", key="reset_email")
        
        if "reset_otp_sent" not in st.session_state:
            st.session_state["reset_otp_sent"] = False
            
        if not st.session_state["reset_otp_sent"]:
            if st.button("Send Reset OTP"):
                success, response = reset_password_request(reset_email)
                if success:
                    st.session_state["reset_otp_sent"] = True
                    st.session_state["reset_otp"] = response
                    st.session_state["reset_email_target"] = reset_email
                    st.success("Reset OTP sent to your email!")
                    st.rerun()
                else:
                    st.error(response)
        else:
            entered_reset_otp = st.text_input("Enter Reset OTP")
            new_password = st.text_input("New Password", type="password")
            if st.button("Update Password"):
                if entered_reset_otp == st.session_state["reset_otp"]:
                    # Update password in DB
                    conn = sqlite3.connect("finguard.db")
                    cursor = conn.cursor()
                    hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                    cursor.execute("UPDATE users SET password_hash = ? WHERE email = ?", (hashed, st.session_state["reset_email_target"]))
                    conn.commit()
                    conn.close()
                    st.success("Password updated successfully! You can now log in.")
                    st.session_state["reset_otp_sent"] = False
                else:
                    st.error("Invalid OTP.")