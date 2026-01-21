import streamlit as st
from scripts.auth import login

st.set_page_config(page_title="Login", layout="centered")

st.title("🔐 HelpDesk Login")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

col1, col2 = st.columns(2)

with col1:
    if st.button("Login"):
        user = login(username, password)

        if user:
            user_id, role = user
            st.session_state.logged_in = True
            st.session_state.user_id = user_id
            st.session_state.role = role
            st.success("Login successful")
            st.switch_page("pages/dashboard.py")
        else:
            st.error("Invalid username or password")

with col2:
    if st.button("Register"):
        st.switch_page("pages/register.py")
