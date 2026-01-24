import pickle
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from clean_text import clean_text


# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cleaned" / "final_dataset_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(DATA_PATH)

df["text_clean"] = df["text"].astype(str).apply(clean_text)
df["category"] = df["category"].astype(str).str.strip().str.lower()
df["priority"] = df["priority"].astype(str).str.strip().str.lower()


# ==============================
# LABEL ENCODING
# ==============================
category_encoder = LabelEncoder()
priority_encoder = LabelEncoder()

y_category = category_encoder.fit_transform(df["category"])
y_priority = priority_encoder.fit_transform(df["priority"])

pickle.dump(category_encoder, open(MODEL_DIR / "category_encoder.pkl", "wb"))
pickle.dump(priority_encoder, open(MODEL_DIR / "priority_encoder.pkl", "wb"))


# ==============================
# TF-IDF (ONE VECTOR SPACE)
# ==============================
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    stop_words="english",
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["text_clean"])
pickle.dump(vectorizer, open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb"))


# ==============================
# CATEGORY MODEL (SVM)
# ==============================
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X,
    y_category,
    test_size=0.2,
    stratify=y_category,
    random_state=42
)

category_model = LinearSVC(class_weight="balanced")
category_model.fit(Xc_train, yc_train)

yc_pred = category_model.predict(Xc_test)

print("\nCATEGORY RESULTS")
print("Accuracy:", accuracy_score(yc_test, yc_pred))
print(classification_report(yc_test, yc_pred, target_names=category_encoder.classes_))

pickle.dump(category_model, open(MODEL_DIR / "category_model.pkl", "wb"))


# ==============================
# PRIORITY MODEL (LOGISTIC)
# ==============================
Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X,
    y_priority,
    test_size=0.2,
    stratify=y_priority,
    random_state=42
)

priority_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)

priority_model.fit(Xp_train, yp_train)

yp_pred = priority_model.predict(Xp_test)

print("\nPRIORITY RESULTS")
print("Accuracy:", accuracy_score(yp_test, yp_pred))
print(classification_report(yp_test, yp_pred, target_names=priority_encoder.classes_))

pickle.dump(priority_model, open(MODEL_DIR / "priority_model.pkl", "wb"))


print("\n✅ Training completed successfully")
