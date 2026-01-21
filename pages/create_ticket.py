import streamlit as st
from scripts.db import insert_ticket
from scripts.ai_logic import predict_ticket

st.set_page_config(page_title="Create Ticket", layout="centered")

if not st.session_state.get("logged_in"):
    st.switch_page("pages/login.py")

st.title("➕ Create New Ticket")

user_input = st.text_area(
    "Describe your issue",
    placeholder="e.g. Laptop not turning on, urgent for client demo"
)

if st.button("Generate & Save Ticket"):
    if not user_input.strip():
        st.warning("Please enter issue description.")
    else:
        category, priority = predict_ticket(user_input)

        insert_ticket(
            title=f"{category.capitalize()} Issue",
            description=user_input,
            category=category,
            priority=priority
        )

        st.success("🎫 Ticket created successfully")
        st.switch_page("pages/dashboard.py")
