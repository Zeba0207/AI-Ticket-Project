import joblib
from pathlib import Path
from scripts.clean_text import clean_text

# =====================================
# PROJECT ROOT
# =====================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# =====================================
# HELPER: LOAD MODEL WITH FALLBACK
# =====================================
def load_model(primary, fallback):
    path_primary = MODELS_DIR / primary
    path_fallback = MODELS_DIR / fallback

    if path_primary.exists():
        return joblib.load(path_primary)
    elif path_fallback.exists():
        return joblib.load(path_fallback)
    else:
        raise FileNotFoundError(
            f"Missing model files: {primary} or {fallback}"
        )

# =====================================
# LOAD MODELS
# =====================================
try:
    vectorizer = load_model(
        "tfidf_vectorizer.pkl",
        "tfidf_vectorizer_final.pkl"
    )

    category_model = load_model(
        "category_model.pkl",
        "category_model_final.pkl"
    )

    priority_model = load_model(
        "priority_model.pkl",
        "priority_model_final.pkl"
    )

    category_encoder = joblib.load(MODELS_DIR / "category_encoder.pkl")
    priority_encoder = joblib.load(MODELS_DIR / "priority_encoder.pkl")

except Exception as e:
    raise RuntimeError(
        f"❌ Model file not found or failed to load.\n{e}"
    )

# =====================================
# RULE-BASED CATEGORY
# =====================================
def rule_based_category(text: str):
    text = text.lower()

    if any(k in text for k in ["purchase", "buy", "order"]):
        return "purchase"
    if any(k in text for k in ["hr", "leave", "salary"]):
        return "hr support"
    if any(k in text for k in ["login", "otp", "password"]):
        return "access"
    if any(k in text for k in ["vpn", "wifi", "network"]):
        return "network"
    if any(k in text for k in ["laptop", "keyboard", "printer"]):
        return "hardware"

    return None

# =====================================
# URGENCY DETECTION
# =====================================
def detect_urgent_intent(text: str) -> bool:
    urgent_keywords = [
        "urgent", "asap", "critical", "system down", "not working"
    ]
    return any(k in text.lower() for k in urgent_keywords)

# =====================================
# MAIN PREDICTION
# =====================================
def predict_ticket(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ticket description cannot be empty")

    cleaned_text = clean_text(text)
    X = vectorizer.transform([cleaned_text])

    # Category
    rule_cat = rule_based_category(cleaned_text)
    if rule_cat:
        category = rule_cat
    else:
        cat_pred = category_model.predict(X)[0]
        category = category_encoder.inverse_transform([cat_pred])[0]

    # Priority (safe handling)
    pr_pred = priority_model.predict(X)[0]
    if isinstance(pr_pred, str):
        priority = pr_pred.capitalize()
    else:
        priority = priority_encoder.inverse_transform([pr_pred])[0]

    if detect_urgent_intent(text):
        priority = "High"

    return category, priority
