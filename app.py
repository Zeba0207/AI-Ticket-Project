import streamlit as st
import joblib
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from scripts.clean_text import clean_text
from scripts.entity_extraction import extract_entities
from scripts.db import (
    create_table,
    insert_ticket,
    fetch_active_tickets,
    fetch_closed_tickets,
    update_status,
    get_counts
)

# =====================================
# INITIALIZE DATABASE
# =====================================
create_table()

# =====================================
# BASE DIRECTORY
# =====================================
BASE_DIR = Path(__file__).resolve().parent

# =====================================
# LOAD MODELS
# =====================================
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")
category_model = joblib.load(BASE_DIR / "models" / "category_model.pkl")
priority_model = joblib.load(BASE_DIR / "models" / "priority_model.pkl")

category_encoder = joblib.load(BASE_DIR / "models" / "category_encoder.pkl")
priority_encoder = joblib.load(BASE_DIR / "models" / "priority_encoder.pkl")

# =====================================
# RULE-BASED HELPERS
# =====================================
def rule_based_category(text):
    t = text.lower()
    if any(k in t for k in ["purchase", "buy", "request"]):
        return "purchase"
    if any(k in t for k in ["hr", "leave", "salary"]):
        return "hr support"
    if any(k in t for k in ["login", "signin", "otp"]):
        return "access"
    if any(k in t for k in ["vpn", "wifi", "network"]):
        return "network"
    if any(k in t for k in ["laptop", "keyboard", "printer"]):
        return "hardware"
    return None

def detect_urgent_intent(text):
    urgent = ["urgent", "immediately", "asap", "system down", "not working"]
    return any(k in text.lower() for k in urgent)

# =====================================
# STREAMLIT UI
# =====================================
st.set_page_config(page_title="AI Ticket System", layout="wide")
st.title("🎫 AI-Powered Ticket Management System")

# =====================================
# ANALYTICS DASHBOARD
# =====================================
stats = get_counts()
c1, c2, c3, c4 = st.columns(4)

c1.metric("🎟 Total Tickets", stats["total"])
c2.metric("📂 Open Tickets", stats["open"])
c3.metric("🔥 High Priority", stats["high"])
c4.metric("✅ Closed Tickets", stats["closed"])

st.divider()

# =====================================
# TICKET CREATION
# =====================================
st.subheader("➕ Create New Ticket")

user_input = st.text_area(
    "Describe your issue",
    placeholder="e.g. Laptop not turning on, urgent for client demo"
)

if st.button("Generate & Save Ticket"):
    if not user_input.strip():
        st.warning("Please enter issue description.")
        st.stop()

    cleaned_text = clean_text(user_input)
    X = vectorizer.transform([cleaned_text])

    rule_cat = rule_based_category(cleaned_text)
    if rule_cat:
        category = rule_cat
    else:
        category = category_encoder.inverse_transform(
            category_model.predict(X)
        )[0]

    priority = priority_encoder.inverse_transform(
        priority_model.predict(X)
    )[0]

    if detect_urgent_intent(user_input):
        priority = "High"

    insert_ticket(
        title=f"{category.capitalize()} Issue",
        description=user_input,
        category=category,
        priority=priority
    )

    st.success("🎫 Ticket created and stored successfully!")

st.divider()

# =====================================
# TABS
# =====================================
tab1, tab2 = st.tabs(["📂 Active Tickets", "🗄 Closed Tickets"])

# =====================================
# ACTIVE TICKETS
# =====================================
with tab1:
    tickets = fetch_active_tickets()

    st.subheader("📋 Support Team View")

    if tickets:
        df = pd.DataFrame(
            tickets,
            columns=[
                "ID",
                "Title",
                "Description",
                "Category",
                "Priority",
                "Status",
                "Created Time",
                "Updated Time"
            ]
        )

        # REQUIRED COLUMN VIEW (MENTOR ASKED)
        st.dataframe(
            df[[
                "ID",
                "Description",
                "Category",
                "Priority",
                "Status",
                "Created Time"
            ]],
            use_container_width=True
        )
    else:
        st.info("No active tickets available.")

    st.divider()

    # DETAILED VIEW
    for t in tickets:
        ticket_id, title, desc, cat, pr, status, created_at, updated_at = t

        with st.expander(f"🎫 Ticket #{ticket_id} — {pr.upper()}"):
            st.write("**Description:**", desc)
            st.write("**Category:**", cat)
            st.write("**Priority:**", pr)
            st.write("**Status:**", status)

            created = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
            hours = (datetime.now() - created).total_seconds() / 3600

            if hours < 2:
                st.success(f"⏱ {hours:.1f} hrs (Within SLA)")
            elif hours < 6:
                st.warning(f"⏱ {hours:.1f} hrs (Approaching SLA)")
            else:
                st.error(f"⏱ {hours:.1f} hrs (SLA Breached)")

            new_status = st.selectbox(
                "Update Status",
                ["Open", "In Progress", "Resolved", "Closed"],
                index=["Open", "In Progress", "Resolved", "Closed"].index(status),
                key=f"status_{ticket_id}"
            )

            if st.button("Save Status", key=f"btn_{ticket_id}"):
                update_status(ticket_id, new_status)
                st.success("Status updated")
                st.rerun()

            if st.checkbox("Show Ticket JSON", key=f"json_{ticket_id}"):
                st.json({
                    "id": ticket_id,
                    "description": desc,
                    "category": cat,
                    "priority": pr,
                    "status": status,
                    "created_at": created_at
                })

# =====================================
# CLOSED TICKETS
# =====================================
with tab2:
    closed = fetch_closed_tickets()

    if not closed:
        st.info("No closed tickets.")
    else:
        for t in closed:
            ticket_id, title, desc, cat, pr, status, created_at, updated_at = t
            st.markdown(
                f"""
                **🎫 Ticket #{ticket_id}**  
                Category: `{cat}` | Priority: `{pr}`  
                Status: `{status}`  
                Created: {created_at}
                """
            )
            st.write(desc)
            st.divider()
