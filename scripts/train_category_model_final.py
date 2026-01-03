import sys
import re
from html import unescape
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "cleaned" / "final_dataset_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# =====================================================
# TEXT CLEANING FUNCTION (FINAL – DO NOT MODIFY)
# =====================================================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unescape(text)
    text = text.lower()

    # Remove personal information
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
    text = re.sub(r'\b\d{10}\b', ' ', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' ', text)

    # Keep letters + numbers
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =====================================================
# LOAD DATASET
# =====================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

# Drop useless column if exists
if "SNO" in df.columns:
    df = df.drop(columns=["SNO"])

# =====================================================
# CREATE CLEAN_TEXT COLUMN
# =====================================================
print("Cleaning text...")
df["clean_text"] = df["text"].apply(clean_text)

# =====================================================
# VERIFY CATEGORIES
# =====================================================
print("\nCategory distribution:")
print(df["category"].value_counts())

# =====================================================
# TRAIN–TEST SPLIT (STRATIFIED)
# =====================================================
X = df["clean_text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# =====================================================
# TF-IDF VECTORIZATION (MAX ACCURACY CONFIG)
# =====================================================
print("\nVectorizing text...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorized shape:", X_train_vec.shape)

# =====================================================
# TRAIN LINEAR SVM
# =====================================================
print("\nTraining Linear SVM...")
model = LinearSVC(C=1.0)
model.fit(X_train_vec, y_train)

# =====================================================
# EVALUATION
# =====================================================
print("\nEvaluating model...")
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =====================================================
# SAVE FINAL MODELS
# =====================================================
print("\nSaving models...")
joblib.dump(model, MODEL_DIR / "category_model_final.pkl")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer_final.pkl")

print("\n✅ Training complete. Final models saved.")
