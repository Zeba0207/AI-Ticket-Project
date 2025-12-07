import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load data
train_df = pd.read_csv("../data/splits/train.csv")
test_df = pd.read_csv("../data/splits/test.csv")

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=3000)

X_train = vectorizer.fit_transform(train_df["clean_text"])
X_test = vectorizer.transform(test_df["clean_text"])

# CATEGORY MODEL
category_model = LogisticRegression(max_iter=200)
category_model.fit(X_train, train_df["category"])

category_preds = category_model.predict(X_test)
print("\n=== CATEGORY CLASSIFICATION ===")
print("Accuracy:", accuracy_score(test_df["category"], category_preds))
print(classification_report(test_df["category"], category_preds))

# PRIORITY MODEL
priority_model = LogisticRegression(max_iter=200)
priority_model.fit(X_train, train_df["priority"])

priority_preds = priority_model.predict(X_test)
print("\n=== PRIORITY CLASSIFICATION ===")
print("Accuracy:", accuracy_score(test_df["priority"], priority_preds))
print(classification_report(test_df["priority"], priority_preds))

# Save models
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(category_model, "../models/category_model.pkl")
joblib.dump(priority_model, "../models/priority_model.pkl")

print("\nModels saved in /models folder")
