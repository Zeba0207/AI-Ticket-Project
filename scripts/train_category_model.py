import re
from html import unescape
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# =====================================================
# PATH SETUP
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cleaned" / "final_dataset_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# =====================================================
# TEXT CLEANING FUNCTION (DEPLOYMENT SAFE)
# =====================================================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unescape(text).lower()

    # Remove emails, phone numbers, IPs
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
    text = re.sub(r'\b\d{10}\b', ' ', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' ', text)

    # Remove special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =====================================================
# LOAD DATASET
# =====================================================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

if "SNO" in df.columns:
    df.drop(columns=["SNO"], inplace=True)

# =====================================================
# CLEAN TEXT
# =====================================================
df["clean_text"] = df["text"].apply(clean_text)

# =====================================================
# ENCODE CATEGORY LABELS (IMPORTANT)
# =====================================================
label_encoder = LabelEncoder()
df["category_encoded"] = label_encoder.fit_transform(df["category"])

print("\nCategory mapping:")
for idx, label in enumerate(label_encoder.classes_):
    print(idx, "→", label)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X = df["clean_text"]
y = df["category_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# TF-IDF VECTORIZATION
# =====================================================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    stop_words="english",
    sublinear_tf=True
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =====================================================
# TRAIN LINEAR SVM
# =====================================================
model = LinearSVC(C=1.0)
model.fit(X_train_vec, y_train)

# =====================================================
# EVALUATION
# =====================================================
y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# =====================================================
# SAVE MODELS (DEPLOYMENT FRIENDLY NAMES)
# =====================================================
joblib.dump(model, MODEL_DIR / "category_model_final.pkl")
joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer_final.pkl")
joblib.dump(label_encoder, MODEL_DIR / "category_encoder.pkl")

print("\n✅ Category model training completed successfully")
