import joblib
from pathlib import Path
from scripts.clean_text import clean_text

# =====================================
# PROJECT ROOT & MODELS DIR
# =====================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"


# =====================================
# SAFE MODEL LOADER WITH FALLBACK
# =====================================
def load_model(primary_name: str, fallback_name: str):
    primary_path = MODELS_DIR / primary_name
    fallback_path = MODELS_DIR / fallback_name

    if primary_path.exists():
        return joblib.load(primary_path)

    if fallback_path.exists():
        return joblib.load(fallback_path)

    raise FileNotFoundError(
        f"Missing model files: {primary_name} OR {fallback_name}"
    )


# =====================================
# LOAD MODELS ONCE (STREAMLIT SAFE)
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
# RULE-BASED CATEGORY (FAST PATH)
# =====================================
def rule_based_category(text: str):
    t = text.lower()

    if any(k in t for k in ["purchase", "buy", "order", "procure", "request"]):
        return "Purchase"

    if any(k in t for k in ["hr", "leave", "salary", "payroll"]):
        return "HR Support"

    if any(k in t for k in ["login", "password", "otp", "signin", "access"]):
        return "Access"

    if any(k in t for k in ["vpn", "wifi", "network", "internet"]):
        return "Network"

    if any(k in t for k in ["laptop", "keyboard", "printer", "mouse"]):
        return "Hardware"

    return None


# =====================================
# URGENCY DETECTION
# =====================================
def detect_urgent_intent(text: str) -> bool:
    urgent_keywords = [
        "urgent",
        "asap",
        "immediately",
        "critical",
        "system down",
        "not working",
        "client demo"
    ]
    return any(k in text.lower() for k in urgent_keywords)


# =====================================
# MAIN PREDICTION FUNCTION
# =====================================
def predict_ticket(text: str):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Ticket description cannot be empty")

    # -----------------------
    # CLEAN TEXT
    # -----------------------
    cleaned_text = clean_text(text)

    # -----------------------
    # VECTORIZE
    # -----------------------
    X = vectorizer.transform([cleaned_text])

    # -----------------------
    # CATEGORY
    # -----------------------
    rule_cat = rule_based_category(cleaned_text)

    if rule_cat:
        category = rule_cat
    else:
        cat_pred = category_model.predict(X)[0]
        category = category_encoder.inverse_transform([cat_pred])[0]

    # -----------------------
    # PRIORITY
    # -----------------------
    pr_pred = priority_model.predict(X)[0]

    if isinstance(pr_pred, str):
        priority = pr_pred.capitalize()
    else:
        priority = priority_encoder.inverse_transform([pr_pred])[0].capitalize()

    # Urgency override
    if detect_urgent_intent(text):
        priority = "High"

    return category, priority
