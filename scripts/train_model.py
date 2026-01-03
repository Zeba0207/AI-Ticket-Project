import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import joblib, json

# ===================================
# Base project directory
# ===================================
base = Path(__file__).resolve().parents[1]

# ===================================
# Load dataset splits
# ===================================
train = pd.read_csv(base / "data/splits/train.csv")
test = pd.read_csv(base / "data/splits/test.csv")

# ===================================
# STEP 1: CATEGORY NORMALIZATION (NO NOISE)
# ===================================
CATEGORY_MAP = {
    "Service Request": "Software",
    "Software Bug": "Software",
    "Account/Access Issue": "Access",
    "Hardware Issue": "Hardware",
    "Network Problem": "Network",
    "Security": "Security"
}

train["category"] = train["category"].map(CATEGORY_MAP)
test["category"] = test["category"].map(CATEGORY_MAP)

# Remove rows with undefined / noisy categories
train = train.dropna(subset=["category"])
test = test.dropna(subset=["category"])

# ===================================
# STEP 2: TF-IDF (OPTIMIZED FOR IT TEXT)
# ===================================
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.8,
    stop_words="english",
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(train["text_clean"])
X_test = vectorizer.transform(test["text_clean"])

# Labels
y_train_cat = train["category"]
y_test_cat = test["category"]

y_train_pri = train["priority"]
y_test_pri = test["priority"]

print("\nCategory distribution (TRAIN):")
print(y_train_cat.value_counts())

# ===================================
# STEP 3: CATEGORY MODEL → LINEAR SVM
# ===================================
print("\nTraining Category Model → Linear SVM")

category_model = LinearSVC(
    C=1.5,                  # tuned for overlap
    class_weight="balanced"
)

category_model.fit(X_train, y_train_cat)
category_preds = category_model.predict(X_test)

category_acc = accuracy_score(y_test_cat, category_preds)
category_f1 = f1_score(y_test_cat, category_preds, average="macro")

print("\nCategory Accuracy (SVM):", round(category_acc, 4))
print("Category Macro F1:", round(category_f1, 4))
print("\nCategory Classification Report:\n")
print(classification_report(y_test_cat, category_preds))

with open(base / "models/category_metrics.json", "w") as f:
    json.dump(
        {
            "model": "LinearSVC",
            "accuracy": round(category_acc, 4),
            "macro_f1": round(category_f1, 4),
            "labels": sorted(y_train_cat.unique().tolist()),
            "notes": "Linear SVM + optimized TF-IDF, noisy categories removed"
        },
        f,
        indent=4
    )

joblib.dump(
    category_model,
    base / "models/category_model.pkl"
)

# ===================================
# STEP 4: PRIORITY MODEL → RANDOM FOREST
# ===================================
print("\nTraining Priority Model → Random Forest")

priority_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

priority_model.fit(X_train, y_train_pri)
priority_preds = priority_model.predict(X_test)

priority_acc = accuracy_score(y_test_pri, priority_preds)

print("Priority Accuracy (RF):", round(priority_acc, 4))

with open(base / "models/priority_metrics.json", "w") as f:
    json.dump(
        {
            "model": "RandomForest",
            "accuracy": round(priority_acc, 4)
        },
        f,
        indent=4
    )

joblib.dump(
    priority_model,
    base / "models/priority_model.pkl"
)

# ===================================
# Save Vectorizer
# ===================================
joblib.dump(vectorizer, base / "models/tfidf.pkl")

print("\n✔ Training Complete! (Category: Linear SVM | Priority: Random Forest)")
