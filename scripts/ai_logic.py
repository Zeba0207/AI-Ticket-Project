import joblib
from pathlib import Path
from scripts.clean_text import clean_text

# =====================================
# BASE DIRECTORY (PROJECT ROOT)
# =====================================
BASE_DIR = Path(__file__).resolve().parent.parent

# =====================================
# LOAD MODELS (ONCE AT STARTUP)
# =====================================
try:
    vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")
    category_model = joblib.load(BASE_DIR / "models" / "category_model.pkl")
    priority_model = joblib.load(BASE_DIR / "models" / "priority_model.pkl")

    category_encoder = joblib.load(BASE_DIR / "models" / "category_encoder.pkl")
    priority_encoder = joblib.load(BASE_DIR / "models" / "priority_encoder.pkl")

except FileNotFoundError as e:
    raise RuntimeError(
        f"❌ Model file not found. Please check the models folder.\n{e}"
    )

# =====================================
# RULE-BASED CATEGORY (FALLBACK LOGIC)
# =====================================
def rule_based_category(text: str):
    """
    Uses keyword-based rules for faster and more accurate classification
    in obvious cases.
    """
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


# =====================================
# URGENT INTENT DETECTION
# =====================================
def detect_urgent_intent(text: str) -> bool:
    urgent_keywords = [
        "urgent", "immediately", "asap",
        "system down", "not working", "critical"
    ]
    return any(k in text.lower() for k in urgent_keywords)


# =====================================
# MAIN PREDICTION FUNCTION
# =====================================
def predict_ticket(text: str):
    """
    Predicts ticket category and priority using:
    - Text cleaning
    - Rule-based classification
    - ML model (fallback)
    - Urgency override
    """

    if not text or not text.strip():
        raise ValueError("Ticket description cannot be empty")

    # Clean text
    cleaned_text = clean_text(text)

    # Vectorize
    X = vectorizer.transform([cleaned_text])

    # Category prediction
    rule_cat = rule_based_category(cleaned_text)
    if rule_cat:
        category = rule_cat
    else:
        category = category_encoder.inverse_transform(
            category_model.predict(X)
        )[0]

    # Priority prediction
    priority = priority_encoder.inverse_transform(
        priority_model.predict(X)
    )[0]

    # Urgency override
    if detect_urgent_intent(text):
        priority = "High"

    return category, priority
