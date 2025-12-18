import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib, json

# Base project directory
base = Path(__file__).resolve().parents[1]

# Load dataset splits
train = pd.read_csv(base / "data/splits/train.csv")
test = pd.read_csv(base / "data/splits/test.csv")

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=8000)
X_train = vectorizer.fit_transform(train["text_clean"])
X_test = vectorizer.transform(test["text_clean"])

y_train_cat = train["category"]
y_test_cat = test["category"]

y_train_pri = train["priority"]
y_test_pri = test["priority"]


# ==============================
# CATEGORY MODEL - RANDOM FOREST
# ==============================
print("\nTraining Category Model → Random Forest")

category_model = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
category_model.fit(X_train, y_train_cat)
category_preds = category_model.predict(X_test)

category_acc = accuracy_score(y_test_cat, category_preds)

# Print & save category accuracy
print("Category Accuracy:", category_acc)
with open(base / "models/category_metrics.json", "w") as f:
    json.dump({"random_forest_accuracy": category_acc}, f, indent=4)

# Save category model
joblib.dump(category_model, base / "models/category_model_random_forest.pkl")


# ==============================
# PRIORITY MODEL - RANDOM FOREST
# ==============================
print("\nTraining Priority Model → Random Forest")

priority_model = RandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
priority_model.fit(X_train, y_train_pri)
priority_preds = priority_model.predict(X_test)

priority_acc = accuracy_score(y_test_pri, priority_preds)

# Print & save priority accuracy
print("Priority Accuracy:", priority_acc)
with open(base / "models/priority_metrics.json", "w") as f:
    json.dump({"random_forest_accuracy": priority_acc}, f, indent=4)

# Save priority model
joblib.dump(priority_model, base / "models/priority_model_random_forest.pkl")


# ==============================
# Save Vectorizer
# ==============================
joblib.dump(vectorizer, base / "models/tfidf.pkl")

print("\n✔ Training Complete! Random Forest models saved.")
