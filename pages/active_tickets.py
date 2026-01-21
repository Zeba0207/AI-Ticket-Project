import streamlit as st
import pandas as pd
from datetime import datetime
import json

from scripts.db import fetch_active_tickets, update_status

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Active Tickets", layout="wide")

# =====================================
# AUTH CHECK
# =====================================
if not st.session_state.get("logged_in"):
    st.switch_page("pages/login.py")

# =====================================
# PAGE TITLE
# =====================================
st.title("📂 Active Tickets")

st.markdown(
    "Monitor ongoing support tickets, track SLA, update status, and inspect ticket JSON."
)

# =====================================
# DEVELOPER MODE (JSON VISIBILITY)
# =====================================
show_json = st.toggle("👩‍💻 Developer Mode (Show Ticket JSON)", value=False)

# =====================================
# FETCH DATA
# =====================================
tickets = fetch_active_tickets()

if not tickets:
    st.info("No active tickets available.")
    st.stop()

# =====================================
# TABLE VIEW (SUMMARY)
# =====================================
df = pd.DataFrame(
    tickets,
    columns=[
        "ID", "Title", "Description",
        "Category", "Priority", "Status",
        "Created", "Updated"
    ]
)

st.subheader("📋 Active Ticket Summary")

st.dataframe(
    df[["ID", "Description", "Category", "Priority", "Status", "Created"]],
    use_container_width=True
)

st.divider()

# =====================================
# DETAILED TICKET VIEW
# =====================================
for t in tickets:
    tid, title, desc, cat, pr, status, created_at, updated_at = t

    with st.expander(f"🎫 Ticket #{tid} — {pr.upper()}"):

        # -------------------------
        # Ticket Description
        # -------------------------
        st.markdown("**📝 Description**")
        st.write(desc)

        # -------------------------
        # SLA TIMER
        # -------------------------
        created = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
        hours = (datetime.now() - created).total_seconds() / 3600

        if hours < 2:
            st.success(f"⏱️ {hours:.1f} hrs — Within SLA")
        elif hours < 6:
            st.warning(f"⏱️ {hours:.1f} hrs — Approaching SLA")
        else:
            st.error(f"⏱️ {hours:.1f} hrs — SLA Breached")

        # -------------------------
        # STATUS UPDATE
        # -------------------------
        new_status = st.selectbox(
            "📌 Update Status",
            ["Open", "In Progress", "Resolved", "Closed"],
            index=["Open", "In Progress", "Resolved", "Closed"].index(status),
            key=f"status_{tid}"
        )

        if st.button("💾 Save Status", key=f"save_{tid}"):
            update_status(tid, new_status)
            st.success("✅ Status updated successfully")
            st.rerun()

        # -------------------------
        # JSON VISIBILITY (MENTOR REQUIREMENT)
        # -------------------------
        if show_json:
            ticket_json = {
                "id": tid,
                "title": title,
                "description": desc,
                "category": cat,
                "priority": pr,
                "status": status,
                "created_at": created_at,
                "updated_at": updated_at
            }

            with st.expander("🧾 View Ticket JSON"):
                st.json(ticket_json)
