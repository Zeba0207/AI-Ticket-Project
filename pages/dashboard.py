import streamlit as st
from scripts.db import get_counts

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="Dashboard", layout="wide")

# =====================================
# AUTH CHECK
# =====================================
if not st.session_state.get("logged_in"):
    st.switch_page("pages/login.py")

# =====================================
# TITLE
# =====================================
st.title("📊 Ticket Analytics Dashboard")
st.caption("Real-time overview of support workload")

# =====================================
# FETCH ANALYTICS FROM DATABASE
# =====================================
stats = get_counts()

# =====================================
# METRIC CARDS
# =====================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("🎟 Total Tickets", stats["total"])
c2.metric("📂 Open Tickets", stats["open"])
c3.metric("🔥 High Priority Tickets", stats["high"])
c4.metric("✅ Closed Tickets", stats["closed"])

st.divider()

# =====================================
# QUICK ACTIONS
# =====================================
st.subheader("🚀 Quick Actions")

a1, a2, a3 = st.columns(3)

with a1:
    if st.button("➕ Create Ticket", use_container_width=True):
        st.switch_page("pages/create_ticket.py")

with a2:
    if st.button("📂 View Active Tickets", use_container_width=True):
        st.switch_page("pages/active_tickets.py")

with a3:
    if st.button("🗄 View Closed Tickets", use_container_width=True):
        st.switch_page("pages/closed_tickets.py")

st.divider()

# =====================================
# SMART INSIGHTS (MENTOR-READY)
# =====================================
st.subheader("📈 System Insights")

if stats["high"] > 0:
    st.warning("⚠️ High priority tickets need immediate attention.")
else:
    st.success("✅ No high priority tickets currently.")

if stats["open"] > 10:
    st.info("📌 Ticket load is high. Consider scaling support.")
elif stats["open"] > 0:
    st.info("📌 Ticket load is manageable.")
else:
    st.success("🎉 No open tickets — system is clear!")
