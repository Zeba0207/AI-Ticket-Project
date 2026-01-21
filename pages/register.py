import streamlit as st
from scripts.auth import register

st.set_page_config(page_title="Register", layout="centered")
st.title("📝 Register New Account")

username = st.text_input("Username")
password = st.text_input("Password", type="password")
confirm = st.text_input("Confirm Password", type="password")

if st.button("Create Account"):
    if not username or not password or not confirm:
        st.warning("All fields required")

    elif password != confirm:
        st.error("Passwords do not match")

    else:
        try:
            register(username, password)
            st.success("Account created successfully")
            st.switch_page("pages/login.py")

        except ValueError as e:
            if str(e) == "USERNAME_EXISTS":
                st.error("Username already exists")
            else:
                st.error(str(e))

        except Exception as e:
            st.error(f"Unexpected error: {e}")

if st.button("Back to Login"):
    st.switch_page("pages/login.py")
