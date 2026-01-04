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
# High-confidence rule-based category
# ======================================
def rule_based_category(text: str):
    t = text.lower()

    if any(k in t for k in [
        "login", "signin", "access denied", "otp", "credential"
    ]):
        return "access"

    if any(k in t for k in [
        "vpn", "wifi", "network", "disconnect", "slow internet"
    ]):
        return "network"

    if any(k in t for k in [
        "purchase", "buy", "request", "procure"
    ]):
        return "purchase"

    if any(k in t for k in [
        "laptop", "mouse", "keyboard", "printer", "screen"
    ]):
        return "hardware"

    return None


# ======================================
# Ticket Generation Engine (Module 3)
# ======================================
def generate_ticket(text: str):
    """
    Generates a structured ticket JSON object
    """

    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        return {
            "title": "Invalid Ticket",
            "description": text,
            "category": "unknown",
            "priority": "low",
            "entities": {},
            "created_at": datetime.now().isoformat()
        }

    # Vectorize text
    X = vectorizer.transform([cleaned_text])

    # -------- CATEGORY (Hybrid Logic) --------
    rule_category = rule_based_category(cleaned_text)
    if rule_category:
        category = rule_category
    else:
        category = category_encoder.inverse_transform(
            category_model.predict(X)
        )[0]

    # -------- PRIORITY --------
    priority = priority_encoder.inverse_transform(
        priority_model.predict(X)
    )[0]

    # -------- ENTITY EXTRACTION --------
    entities = extract_entities(text)

    # -------- TITLE GENERATION --------
    title = f"{category.capitalize()} Issue"

    # -------- FINAL TICKET JSON --------
    ticket = {
        "title": title,
        "description": text,
        "cleaned_description": cleaned_text,
        "category": category,
        "priority": priority,
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

    print("\n--- GENERATED TICKET ---")
    for key, value in ticket.items():
        print(f"{key}: {value}")
    print("------------------------\n")
