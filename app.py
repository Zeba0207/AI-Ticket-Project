import streamlit as st
from pathlib import Path

# =====================================
# DATABASE INITIALIZATION (CRITICAL)
# =====================================
from scripts.db import create_table, create_user_table

# These MUST run when the app starts
create_table()        # tickets table
create_user_table()   # users table

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="AI Ticket Management System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================
# LOAD CUSTOM CSS (PROFESSIONAL UI)
# =====================================
def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()  # <-- LOAD CSS ONCE

# =====================================
# SIDEBAR NAVIGATION
# =====================================
st.sidebar.title("🎫 AI Ticket System")

st.sidebar.markdown("""
Welcome to the **AI-Powered Ticket Management System**  
Use the menu below to navigate.
""")

st.sidebar.divider()

if st.sidebar.button("🏠 Dashboard", use_container_width=True):
    st.switch_page("pages/dashboard.py")

if st.sidebar.button("➕ Create Ticket", use_container_width=True):
    st.switch_page("pages/create_ticket.py")

if st.sidebar.button("📂 Active Tickets", use_container_width=True):
    st.switch_page("pages/active_tickets.py")

if st.sidebar.button("🗄 Closed Tickets", use_container_width=True):
    st.switch_page("pages/closed_tickets.py")

st.sidebar.divider()

if st.sidebar.button("🔐 Login", use_container_width=True):
    st.switch_page("pages/login.py")

if st.sidebar.button("📝 Register", use_container_width=True):
    st.switch_page("pages/register.py")

# =====================================
# MAIN LANDING PAGE (HERO SECTION)
# =====================================
st.markdown(
    """
    <div class="glass-card">
        <div class="hero-title">
            Intelligent Support<br/>
            Powered by AI
        </div>
        <div class="hero-subtitle">
            Transform your helpdesk with AI-powered ticket classification,
            priority detection, SLA tracking, and real-time analytics.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =====================================
# FEATURE CARDS
# =====================================
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        """
        <div class="glass-card">
            🤖 <b>AI Ticket Intelligence</b><br/>
            Automatically classify category and priority using NLP models.
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        """
        <div class="glass-card">
            ⏱ <b>SLA Monitoring</b><br/>
            Real-time SLA tracking with color-coded alerts.
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        """
        <div class="glass-card">
            📊 <b>Live Analytics</b><br/>
            Track ticket workload and resolution metrics instantly.
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

st.info("📝 New users should register first before creating tickets.")
