import pickle
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# IMPORT CLEANING FUNCTION
# ==============================
from clean_text import clean_text

# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cleaned" / "final_dataset_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ==============================
# LOAD DATASET
# ==============================
df = pd.read_csv(DATA_PATH)
print("Initial columns:", df.columns.tolist())

# ==============================
# CREATE text_clean IF MISSING
# ==============================
if "text_clean" not in df.columns:
    if "text" in df.columns:
        source_col = "text"
    elif "description" in df.columns:
        source_col = "description"
    else:
        raise ValueError("No text column found (expected 'text' or 'description')")

    print(f"Creating text_clean from '{source_col}'")
    df["text_clean"] = df[source_col].astype(str).apply(clean_text)

# ==============================
# CLEAN & NORMALISE
# ==============================
required_cols = ["text_clean", "category", "priority"]
df = df.dropna(subset=required_cols)

df["category"] = df["category"].astype(str).str.strip().str.lower()
df["priority"] = df["priority"].astype(str).str.strip().str.lower()

print("Dataset size after cleaning:", df.shape)

# ==============================
# LABEL ENCODING
# ==============================
category_encoder = LabelEncoder()
priority_encoder = LabelEncoder()

df["category_label"] = category_encoder.fit_transform(df["category"])
df["priority_label"] = priority_encoder.fit_transform(df["priority"])

pickle.dump(category_encoder, open(MODEL_DIR / "category_encoder.pkl", "wb"))
pickle.dump(priority_encoder, open(MODEL_DIR / "priority_encoder.pkl", "wb"))

print("Category classes:", list(category_encoder.classes_))
print("Priority classes:", list(priority_encoder.classes_))

# ==============================
# TF-IDF FEATURE EXTRACTION
# ==============================
tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    stop_words="english",
    sublinear_tf=True
)

X_tfidf = tfidf.fit_transform(df["text_clean"])
pickle.dump(tfidf, open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb"))

# ==============================
# -------- CATEGORY MODEL -------
# ==============================
y_cat = df["category_label"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_tfidf,
    y_cat,
    test_size=0.2,
    stratify=y_cat,
    random_state=42
)

print("\nRunning hyperparameter tuning for CATEGORY (Linear SVM)...")

param_grid = {"C": [0.5, 1.0, 1.5, 2.0]}

svm = LinearSVC(class_weight="balanced")
grid = GridSearchCV(
    svm,
    param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1
)

grid.fit(Xc_train, yc_train)
category_model = grid.best_estimator_

print("Best SVM parameters:", grid.best_params_)

yc_pred = category_model.predict(Xc_test)

print("\nCATEGORY RESULTS")
print("Accuracy:", round(accuracy_score(yc_test, yc_pred), 4))
print(classification_report(
    yc_test,
    yc_pred,
    target_names=category_encoder.classes_
))

pickle.dump(category_model, open(MODEL_DIR / "category_model.pkl", "wb"))

# ---- Category Confusion Matrix
cm_cat = confusion_matrix(yc_test, yc_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm_cat,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=category_encoder.classes_,
    yticklabels=category_encoder.classes_
)
plt.title("Category Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ==============================
# -------- PRIORITY MODEL -------
# ==============================
y_pri = df["priority_label"]

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_tfidf,
    y_pri,
    test_size=0.2,
    stratify=y_pri,
    random_state=42
)

priority_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining PRIORITY model (Logistic Regression)...")
priority_model.fit(Xp_train, yp_train)

yp_pred = priority_model.predict(Xp_test)

print("\nPRIORITY RESULTS")
print("Accuracy:", round(accuracy_score(yp_test, yp_pred), 4))
print(classification_report(
    yp_test,
    yp_pred,
    target_names=priority_encoder.classes_
))

pickle.dump(priority_model, open(MODEL_DIR / "priority_model.pkl", "wb"))

# ---- Priority Confusion Matrix
cm_pri = confusion_matrix(yp_test, yp_pred)

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm_pri,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=priority_encoder.classes_,
    yticklabels=priority_encoder.classes_
)
plt.title("Priority Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ==============================
# MODEL COMPARISON (CATEGORY)
# ==============================
print("\n--- CATEGORY MODEL COMPARISON ---")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

for name, model in models.items():
    print(f"\n{name}")
    model.fit(Xc_train, yc_train)
    preds = model.predict(Xc_test)
    print("Accuracy:", round(accuracy_score(yc_test, preds), 4))

print("\n✅ Training complete with tuning, comparison, and confusion matrices.")
