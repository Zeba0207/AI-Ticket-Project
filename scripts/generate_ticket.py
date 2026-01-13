import joblib
from pathlib import Path
from datetime import datetime

from clean_text import clean_text
from entity_extraction import extract_entities

# ======================================
# Base project directory
# ======================================
BASE_DIR = Path(__file__).resolve().parents[1]

# ======================================
# Load vectorizer, models, encoders
# ======================================
vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")

category_model = joblib.load(BASE_DIR / "models" / "category_model.pkl")
priority_model = joblib.load(BASE_DIR / "models" / "priority_model.pkl")

category_encoder = joblib.load(BASE_DIR / "models" / "category_encoder.pkl")
priority_encoder = joblib.load(BASE_DIR / "models" / "priority_encoder.pkl")

print("Models, vectorizer, and encoders loaded successfully.")

# ======================================
# High-confidence intent-based category
# ======================================
def rule_based_category(text: str):
    t = text.lower()

    # ----- INTENT RULES (FIRST) -----
    if any(k in t for k in ["purchase", "buy", "request", "procure"]):
        return "purchase"

    if any(k in t for k in [
        "hr", "leave", "salary", "payroll", "payslip", "reimbursement"
    ]):
        return "hr support"

    # ----- ACCESS -----
    if any(k in t for k in [
        "login", "signin", "access denied", "otp", "credential"
    ]):
        return "access"

    # ----- NETWORK -----
    if any(k in t for k in [
        "vpn", "wifi", "network", "disconnect", "slow internet"
    ]):
        return "network"

    # ----- HARDWARE -----
    if any(k in t for k in [
        "laptop", "mouse", "keyboard", "printer", "screen"
    ]):
        return "hardware"

    return None


# ======================================
# Urgency detection (Priority override)
# ======================================
def detect_urgent_intent(text: str):
    urgent_keywords = [
        "unable to access",
        "system down",
        "blocked",
        "critical",
        "immediately",
        "asap",
        "not working"
    ]
    return any(word in text.lower() for word in urgent_keywords)


# ======================================
# Ticket Generation Engine
# ======================================
def generate_ticket(text: str):
    """
    End-to-end inference flow:
    Input → Cleaning → Vectorization → Prediction → Confidence
    """

    # -------- PREPROCESSING --------
    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        return {
            "title": "Invalid Ticket",
            "description": text,
            "category": "unknown",
            "priority": "low",
            "confidence_score": 0.0,
            "entities": {},
            "created_at": datetime.now().isoformat(),
            "status": "open"
        }

    # -------- VECTORIZATION --------
    X = vectorizer.transform([cleaned_text])

    # -------- CATEGORY PREDICTION --------
    rule_category = rule_based_category(cleaned_text)

    if rule_category:
        category = rule_category
        confidence = 1.0   # rule-based = full confidence
    else:
        pred_label = category_model.predict(X)
        category = category_encoder.inverse_transform(pred_label)[0]

        # Confidence from SVM decision function
        scores = category_model.decision_function(X)
        confidence = float(scores.max())

    # -------- PRIORITY PREDICTION --------
    priority = priority_encoder.inverse_transform(
        priority_model.predict(X)
    )[0]

    # Urgency override
    if detect_urgent_intent(text):
        priority = "high"

    # -------- ENTITY EXTRACTION --------
    entities = extract_entities(text)

    # -------- TITLE --------
    title = f"{category.capitalize()} Issue"

    # -------- FINAL STRUCTURED TICKET --------
    ticket = {
        "title": title,
        "description": text,
        "cleaned_description": cleaned_text,
        "category": category,
        "priority": priority,
        "confidence_score": round(confidence, 3),
        "entities": entities,
        "created_at": datetime.now().isoformat(),
        "status": "open"
    }

    return ticket


# ======================================
# CLI Runner
# ======================================
if __name__ == "__main__":
    print("\n=== AI Ticket Generation Engine ===")
    user_input = input("Enter ticket description: ")

    ticket = generate_ticket(user_input)

    print("\n--- GENERATED TICKET (JSON) ---")
    for key, value in ticket.items():
        print(f"{key}: {value}")
    print("------------------------------\n")
