import joblib
from pathlib import Path
from clean_text import clean_text

# Base project directory
base = Path(__file__).resolve().parents[1]

# Load unified TF-IDF vectorizer
vectorizer = joblib.load(base / "models" / "tfidf.pkl")

# Load Random Forest trained models
category_model = joblib.load(base / "models" / "category_model_random_forest.pkl")
priority_model = joblib.load(base / "models" / "priority_model_random_forest.pkl")

def generate_ticket(text: str):
    """
    Takes a raw ticket description and predicts:
    - Category
    - Priority
    """

    # 1️⃣ Clean input text
    cleaned = clean_text(text)

    # 2️⃣ Vectorize
    X = vectorizer.transform([cleaned])

    # 3️⃣ Predict
    category = category_model.predict(X)[0]
    priority = priority_model.predict(X)[0]

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "predicted_category": category,
        "predicted_priority": priority
    }


if __name__ == "__main__":
    print("\n=== AI Ticket Prediction (Random Forest) ===")
    user_text = input("Enter ticket description: ")

    result = generate_ticket(user_text)

    print("\n--- PREDICTION RESULT ---")
    print("Original Text:", result["original_text"])
    print("Cleaned Text:", result["cleaned_text"])
    print("Predicted Category:", result["predicted_category"])
    print("Predicted Priority:", result["predicted_priority"])
    print("---------------------------\n")
