import streamlit as st
from scripts.db import fetch_closed_tickets

st.set_page_config(page_title="Closed Tickets", layout="wide")

if not st.session_state.get("logged_in"):
    st.switch_page("pages/login.py")

st.title("🗄 Closed Tickets")

tickets = fetch_closed_tickets()

if not tickets:
    st.info("No closed tickets.")
else:
    for t in tickets:
        tid, title, desc, cat, pr, status, created_at, updated_at = t
        st.markdown(
            f"""
            **🎫 Ticket #{tid}**  
            Category: `{cat}` | Priority: `{pr}`  
            Created: {created_at}
            """
        )
        st.write(desc)
        st.divider()
