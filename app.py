import streamlit as st
import joblib
import json
import uuid
from pathlib import Path
from datetime import datetime

from scripts.clean_text import clean_text
from scripts.entity_extraction import extract_entities

# =====================================
# BASE DIRECTORY
# =====================================
BASE_DIR = Path(__file__).resolve().parent

# =====================================
# LOAD MODELS, VECTORIZER, ENCODERS
# =====================================
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")
category_model = joblib.load(BASE_DIR / "models" / "category_model.pkl")
priority_model = joblib.load(BASE_DIR / "models" / "priority_model.pkl")

category_encoder = joblib.load(BASE_DIR / "models" / "category_encoder.pkl")
priority_encoder = joblib.load(BASE_DIR / "models" / "priority_encoder.pkl")

# =====================================
# RULE-BASED HELPERS (HIGH CONFIDENCE)
# =====================================
def rule_based_category(text: str):
    t = text.lower()

    # INTENT FIRST (important)
    if any(k in t for k in ["purchase", "buy", "request", "procure"]):
        return "purchase"

    if any(k in t for k in ["hr", "leave", "salary", "payroll"]):
        return "hr support"

    if any(k in t for k in ["login", "signin", "access denied", "otp", "credential"]):
        return "access"

    if any(k in t for k in ["vpn", "wifi", "network", "disconnect", "slow internet"]):
        return "network"

    if any(k in t for k in ["laptop", "mouse", "keyboard", "printer", "screen"]):
        return "hardware"

    return None


def detect_urgent_intent(text: str):
    urgent_keywords = [
        "unable to access", "system down", "blocked",
        "critical", "immediately", "asap", "not working",
        "urgent", "failure", "crash"
    ]
    return any(k in text.lower() for k in urgent_keywords)


# =====================================
# STREAMLIT UI
# =====================================
st.set_page_config(page_title="AI Ticket Generator", layout="centered")

st.title("🎫 AI-Powered Ticket Generation System")
st.write(
    "Automatically converts user issues into structured IT support tickets "
    "using NLP and Machine Learning."
)

user_input = st.text_area(
    "Describe your issue:",
    placeholder="e.g. Laptop not turning on after Windows update, urgent for client demo"
)

# =====================================
# BUTTON ACTION
# =====================================
if st.button("Generate Ticket"):

    # ---------------- INPUT VALIDATION ----------------
    if not user_input.strip():
        st.warning("Please enter an issue description.")
        st.stop()

    if len(user_input.split()) < 4:
        st.warning(
            "Please provide more details (issue + context + urgency) for accurate prediction."
        )
        st.stop()

    # ---------------- PREPROCESS ----------------
    cleaned_text = clean_text(user_input)

    if not cleaned_text.strip():
        st.warning("Input text could not be processed.")
        st.stop()

    # ---------------- VECTORIZE ----------------
    X = vectorizer.transform([cleaned_text])

    # ---------------- CATEGORY ----------------
    rule_cat = rule_based_category(cleaned_text)

    if rule_cat:
        category = rule_cat
        confidence = 1.0
    else:
        scores = category_model.decision_function(X)
        confidence = float(scores.max())

        category = category_encoder.inverse_transform(
            category_model.predict(X)
        )[0]

    # ---------------- PRIORITY ----------------
    priority = priority_encoder.inverse_transform(
        priority_model.predict(X)
    )[0]

    # Urgent intent override
    if detect_urgent_intent(user_input):
        priority = "high"

    # ---------------- ENTITY EXTRACTION ----------------
    entities = extract_entities(user_input)

    # ---------------- TICKET JSON ----------------
    ticket = {
        "ticket_id": str(uuid.uuid4())[:8],
        "title": f"{category.capitalize()} Issue",
        "description": user_input,
        "cleaned_description": cleaned_text,
        "category": category,
        "priority": priority,
        "confidence_score": round(confidence, 3),
        "entities": entities,
        "status": "open",
        "created_at": datetime.now().isoformat()
    }

    # =====================================
    # DISPLAY OUTPUT
    # =====================================
    st.success("Ticket generated successfully ✅")

    st.subheader("📌 Ticket Summary")
    st.write("**Category:**", category)
    st.write("**Priority:**", priority)
    st.write("**Confidence Score:**", round(confidence, 3))

    st.subheader("🧾 Generated Ticket (JSON)")
    st.json(ticket)

    # Optional download
    st.download_button(
        label="📥 Download Ticket as JSON",
        data=json.dumps(ticket, indent=4),
        file_name="generated_ticket.json",
        mime="application/json"
    )
