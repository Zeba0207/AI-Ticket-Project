import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# ✅ Correct import (same folder)
from clean_text import clean_text

# =====================================================
# PATHS
# =====================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cleaned" / "final_dataset_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================
df = pd.read_csv(DATA_PATH)

# Required columns check
required_cols = ["category", "priority"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# =====================================================
# CREATE text_clean
# =====================================================
if "text_clean" not in df.columns:
    if "text" in df.columns:
        source_col = "text"
    elif "description" in df.columns:
        source_col = "description"
    else:
        raise ValueError("No text column found")

    df["text_clean"] = df[source_col].astype(str).apply(clean_text)

df = df.dropna(subset=["text_clean", "category", "priority"])

# Normalize labels
df["category"] = df["category"].astype(str).str.strip().str.lower()
df["priority"] = df["priority"].astype(str).str.strip().str.lower()

print("\nDataset size:", df.shape)

# =====================================================
# LABEL ENCODING
# =====================================================
category_encoder = LabelEncoder()
priority_encoder = LabelEncoder()

df["category_label"] = category_encoder.fit_transform(df["category"])
df["priority_label"] = priority_encoder.fit_transform(df["priority"])

pickle.dump(category_encoder, open(MODEL_DIR / "category_encoder.pkl", "wb"))
pickle.dump(priority_encoder, open(MODEL_DIR / "priority_encoder.pkl", "wb"))

# =====================================================
# TF-IDF
# =====================================================
tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.85,
    stop_words="english",
    sublinear_tf=True
)

X = tfidf.fit_transform(df["text_clean"])
pickle.dump(tfidf, open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb"))

# =====================================================
# CATEGORY MODEL
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, df["category_label"],
    test_size=0.2,
    stratify=df["category_label"],
    random_state=42
)

category_model = LinearSVC(C=1.5, class_weight="balanced", random_state=42)
category_model.fit(X_train, y_train)

y_pred = category_model.predict(X_test)
print("\nCATEGORY MODEL RESULTS")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=category_encoder.classes_))

pickle.dump(category_model, open(MODEL_DIR / "category_model.pkl", "wb"))

# =====================================================
# PRIORITY MODEL
# =====================================================
Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X, df["priority_label"],
    test_size=0.2,
    stratify=df["priority_label"],
    random_state=42
)

priority_model = LogisticRegression(
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced"
)

priority_model.fit(Xp_train, yp_train)
yp_pred = priority_model.predict(Xp_test)

print("\nPRIORITY MODEL RESULTS")
print("Accuracy:", round(accuracy_score(yp_test, yp_pred), 4))
print(classification_report(yp_test, yp_pred, target_names=priority_encoder.classes_))

pickle.dump(priority_model, open(MODEL_DIR / "priority_model.pkl", "wb"))

print("\n✅ Training complete for CATEGORY and PRIORITY models.")
