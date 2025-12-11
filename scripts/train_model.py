import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib, json

# Base project directory
base = Path(__file__).resolve().parents[1]

# Load splits
train = pd.read_csv(base / "data/splits/train.csv")
val = pd.read_csv(base / "data/splits/val.csv")
test = pd.read_csv(base / "data/splits/test.csv")

# ---------------------------------------------------
# 1️⃣ TF-IDF VECTORIZER (unified for BOTH models)
# ---------------------------------------------------
vectorizer = TfidfVectorizer(max_features=8000)

X_train = vectorizer.fit_transform(train["text_clean"])
X_test = vectorizer.transform(test["text_clean"])

y_train_cat = train["category"]
y_test_cat = test["category"]

y_train_pri = train["priority"]
y_test_pri = test["priority"]

# ---------------------------------------------------
# 2️⃣ CATEGORY MODEL
# ---------------------------------------------------
category_model = LogisticRegression(
    max_iter=400,
    class_weight='balanced'
)

category_model.fit(X_train, y_train_cat)
category_preds = category_model.predict(X_test)

cat_report = classification_report(y_test_cat, category_preds, output_dict=True)
cat_acc = accuracy_score(y_test_cat, category_preds)

# Save category metrics
(cat_metrics_path := base / "models" / "category_metrics.json").parent.mkdir(exist_ok=True)
with open(cat_metrics_path, "w") as f:
    json.dump(cat_report, f, indent=4)

print("Category Accuracy:", cat_acc)

# ---------------------------------------------------
# 3️⃣ PRIORITY MODEL (NEW)
# ---------------------------------------------------
priority_model = LogisticRegression(
    max_iter=400,
    class_weight='balanced'
)

priority_model.fit(X_train, y_train_pri)
priority_preds = priority_model.predict(X_test)

pri_report = classification_report(y_test_pri, priority_preds, output_dict=True)
pri_acc = accuracy_score(y_test_pri, priority_preds)

# Save priority metrics
with open(base / "models" / "priority_metrics.json", "w") as f:
    json.dump(pri_report, f, indent=4)

print("Priority Accuracy:", pri_acc)

# ---------------------------------------------------
# 4️⃣ SAVE MODELS + VECTORIZER
# ---------------------------------------------------
joblib.dump(vectorizer, base / "models" / "tfidf.pkl")
joblib.dump(category_model, base / "models" / "category_model.pkl")
joblib.dump(priority_model, base / "models" / "priority_model.pkl")

print("\nAll models and vectorizer saved successfully!")
