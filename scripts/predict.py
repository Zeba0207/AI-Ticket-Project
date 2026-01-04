import joblib
from pathlib import Path
from clean_text import clean_text

# ======================================
# BASE PATH
# ======================================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

# ======================================
# LOAD MODELS & ENCODERS
# ======================================
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

category_model = joblib.load(MODEL_DIR / "category_model.pkl")
priority_model = joblib.load(MODEL_DIR / "priority_model.pkl")

category_encoder = joblib.load(MODEL_DIR / "category_encoder.pkl")
priority_encoder = joblib.load(MODEL_DIR / "priority_encoder.pkl")

print("Models & vectorizer loaded successfully.")

# ======================================
# PREDICTION FUNCTION
# ======================================
def predict_ticket(text: str):
    """
    Predicts category and priority for a ticket description
    """

    # 1️⃣ Clean input text (same as training)
    cleaned = clean_text(text)

    if not cleaned.strip():
        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "predicted_category": "miscellaneous",
            "predicted_priority": "low"
        }

    # 2️⃣ Vectorize
    X = vectorizer.transform([cleaned])

    # 3️⃣ CATEGORY prediction (with safety fallback)
    scores = category_model.decision_function(X)

    if scores.max() < 0.2:
        category = "miscellaneous"
    else:
        cat_label = category_model.predict(X)
        category = category_encoder.inverse_transform(cat_label)[0]

    # 4️⃣ PRIORITY prediction
    pri_label = priority_model.predict(X)
    priority = priority_encoder.inverse_transform(pri_label)[0]

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "predicted_category": category,
        "predicted_priority": priority
    }

# ======================================
# CLI TEST
# ======================================
if __name__ == "__main__":
    print("\n=== AI Ticket Prediction ===")
    user_text = input("Enter ticket description: ")

    result = predict_ticket(user_text)

    print("\n--- PREDICTION RESULT ---")
    print("Original Text :", result["original_text"])
    print("Cleaned Text  :", result["cleaned_text"])
    print("Category      :", result["predicted_category"])
    print("Priority      :", result["predicted_priority"])
    print("-------------------------\n")
