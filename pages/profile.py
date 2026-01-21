import streamlit as st

st.set_page_config(page_title="Profile", layout="centered")

if not st.session_state.get("logged_in"):
    st.switch_page("pages/login.py")

st.title("👤 Profile")

st.write(f"**User ID:** {st.session_state.user_id}")
st.write(f"**Role:** {st.session_state.role}")

if st.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.role = None
    st.switch_page("pages/login.py")
