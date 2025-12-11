import joblib
from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parents[1]

vectorizer = joblib.load(base/"models"/"tfidf.pkl")
model = joblib.load(base/"models"/"category_model.pkl")

def predict(text):
    clean = pd.Series([text])
    X = vectorizer.transform(clean)
    pred = model.predict(X)[0]
    return pred

if __name__=="__main__":
    t = input("Enter ticket description: ")
    print("\nPredicted Category:", predict(t))
