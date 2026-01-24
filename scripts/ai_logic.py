import joblib
from pathlib import Path
from scripts.clean_text import clean_text


# =====================================
# PROJECT ROOT & MODELS DIR
# =====================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# =====================================
# LOAD MODELS (ONCE)
# =====================================
try:
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.pkl")

    category_model = joblib.load(MODELS_DIR / "category_model.pkl")
    priority_model = joblib.load(MODELS_DIR / "priority_model.pkl")

    category_encoder = joblib.load(MODELS_DIR / "category_encoder.pkl")
    priority_encoder = joblib.load(MODELS_DIR / "priority_encoder.pkl")

except Exception as e:
    raise RuntimeError(f"❌ Failed to load model files:\n{e}")


# =====================================
# RULE-BASED CATEGORY (FAST PATH)
# =====================================
def rule_based_category(text: str):
    t = text.lower()

    if any(k in t for k in ["purchase", "buy", "order", "procure"]):
        return "Purchase"
    if any(k in t for k in ["hr", "leave", "salary"]):
        return "HR Support"
    if any(k in t for k in ["login", "password", "otp"]):
        return "Access"
    if any(k in t for k in ["vpn", "wifi", "network"]):
        return "Network"
    if any(k in t for k in ["laptop", "printer", "keyboard"]):
        return "Hardware"

    return None


# =====================================
# URGENCY DETECTION
# =====================================
def detect_urgent_intent(text: str) -> bool:
    return any(
        k in text.lower()
        for k in ["urgent", "asap", "immediately", "critical", "system down"]
    )


# =====================================
# MAIN PREDICTION FUNCTION
# =====================================
def predict_ticket(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ticket description cannot be empty")

    cleaned_text = clean_text(text)
    X = vectorizer.transform([cleaned_text])

    # CATEGORY
    rule_cat = rule_based_category(cleaned_text)
    if rule_cat:
        category = rule_cat
    else:
        cat_pred = category_model.predict(X)[0]
        category = category_encoder.inverse_transform([cat_pred])[0]

    # PRIORITY
    pr_pred = priority_model.predict(X)[0]
    priority = priority_encoder.inverse_transform([pr_pred])[0].capitalize()

    if detect_urgent_intent(text):
        priority = "High"

    return category, priority
