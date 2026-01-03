import streamlit as st
import joblib
import json
import uuid
from scripts.clean_text import clean_text

# -----------------------------------
# Load vectorizer and trained models
# -----------------------------------
vectorizer = joblib.load("models/tfidf.pkl")
category_model = joblib.load("models/category_model_random_forest.pkl")
priority_model = joblib.load("models/priority_model_random_forest.pkl")

# -----------------------------------
# Allowed labels (consistency)
# -----------------------------------
VALID_CATEGORIES = ["Network", "Hardware", "Software", "Billing", "Account", "Other"]
VALID_PRIORITIES = ["Low", "Medium", "High"]

# -----------------------------------
# Keyword-based category mapping
# -----------------------------------
CATEGORY_MAPPING = {
    "wifi": "Network",
    "internet": "Network",
    "network": "Network",
    "vpn": "Network",

    "laptop": "Hardware",
    "battery": "Hardware",
    "screen": "Hardware",
    "keyboard": "Hardware",
    "printer": "Hardware",

    "outlook": "Software",
    "application": "Software",
    "software": "Software",
    "email": "Software",

    "login": "Account",
    "password": "Account",
    "account": "Account",

    "bill": "Billing",
    "invoice": "Billing",
    "payment": "Billing",
    "charged": "Billing"
}

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.set_page_config(page_title="AI Ticket Generator")
st.title("🎫 AI-Powered Ticket Generation System")

user_input = st.text_area(
    "Describe your issue:",
    placeholder="e.g. My office WiFi keeps disconnecting every 10 minutes"
)

# -----------------------------------
# Ticket generation
# -----------------------------------
if st.button("Generate Ticket"):

    if user_input.strip() == "":
        st.warning("Please enter an issue description.")
        st.stop()

    # Clean text
    cleaned_text = clean_text(user_input)

    # Edge-case handling
    if len(cleaned_text.split()) < 3:
        st.warning(
            "Please provide more details about the issue for accurate ticket generation."
        )
        st.stop()

    # Vectorize
    vectorized_text = vectorizer.transform([cleaned_text])

    # ML Predictions
    category = category_model.predict(vectorized_text)[0]
    priority = priority_model.predict(vectorized_text)[0]

    # Confidence scores
    category_probs = category_model.predict_proba(vectorized_text)[0]
    priority_probs = priority_model.predict_proba(vectorized_text)[0]

    category_confidence = max(category_probs)
    priority_confidence = max(priority_probs)

    # -----------------------------------
    # Rule-based category correction
    # -----------------------------------
    for keyword, mapped_category in CATEGORY_MAPPING.items():
        if keyword in cleaned_text:
            category = mapped_category
            break

    # Final safety check
    if category not in VALID_CATEGORIES:
        category = "Other"

    if priority not in VALID_PRIORITIES:
        priority = "Medium"

    # -----------------------------------
    # Structured JSON Ticket (Module 3)
    # -----------------------------------
    ticket = {
        "ticket_id": str(uuid.uuid4())[:8],
        "title": f"{category} Issue",
        "description": user_input,
        "category": category,
        "priority": priority,
        "category_confidence": round(category_confidence, 2),
        "priority_confidence": round(priority_confidence, 2)
    }

    # -----------------------------------
    # UI Output
    # -----------------------------------
    st.success("Ticket Generated Successfully ✅")

    st.write("**Category:**", category)
    st.write("**Priority:**", priority)
    st.write("**Category Confidence:**", f"{category_confidence:.2f}")
    st.write("**Priority Confidence:**", f"{priority_confidence:.2f}")

    st.subheader("📄 Generated Ticket (JSON)")
    st.json(ticket)

    # Optional: save ticket locally
    with open("generated_ticket.json", "w") as f:
        json.dump(ticket, f, indent=4)
