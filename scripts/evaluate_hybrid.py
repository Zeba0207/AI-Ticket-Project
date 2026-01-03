import pandas as pd
from pathlib import Path
import joblib

# ==============================
# Load paths & models
# ==============================
base = Path(__file__).resolve().parents[1]

vectorizer = joblib.load(base / "models" / "tfidf.pkl")
model = joblib.load(base / "models" / "category_model.pkl")

test = pd.read_csv(base / "data/splits/test.csv")

# ==============================
# CATEGORY NORMALIZATION
# (MUST MATCH TRAINING)
# ==============================
CATEGORY_MAP = {
    "Service Request": "Software",
    "Software Bug": "Software",
    "Account/Access Issue": "Access",
    "Hardware Issue": "Hardware",
    "Network Problem": "Network",
    "Security": "Security"
}

test["category"] = test["category"].map(CATEGORY_MAP)
test = test.dropna(subset=["category"])

# ==============================
# SAME rule-based logic as predict.py
# ==============================
def rule_based_category(text):
    t = text.lower()

    if any(k in t for k in [
        "unable to access", "cannot access", "can't access",
        "login", "log in", "log into", "signin", "sign in",
        "access denied", "credentials", "otp"
    ]):
        return "Access"

    if any(k in t for k in [
        "vpn", "wifi", "wi-fi", "internet", "network",
        "lan", "connection", "disconnect", "slow internet"
    ]):
        return "Network"

    if any(k in t for k in [
        "laptop", "desktop", "tablet",
        "keyboard", "mouse", "screen",
        "monitor", "printer", "hardware"
    ]):
        return "Hardware"

    if any(k in t for k in [
        "application", "app", "software",
        "outlook", "zoom", "crm", "portal",
        "not opening", "fails to load",
        "crashing", "error"
    ]):
        return "Software"

    if any(k in t for k in [
        "virus", "malware", "security",
        "firewall", "phishing", "ransomware"
    ]):
        return "Security"

    return None

# ==============================
# Hybrid prediction
# ==============================
def hybrid_predict(text):
    rule_pred = rule_based_category(text)
    if rule_pred:
        return rule_pred

    X = vectorizer.transform(pd.Series([text]))
    return model.predict(X)[0]

# ==============================
# Evaluate hybrid accuracy
# ==============================
correct = 0
total = len(test)

for _, row in test.iterrows():
    pred = hybrid_predict(row["text_clean"])
    if pred == row["category"]:
        correct += 1

accuracy = correct / total
print("Hybrid Category Accuracy:", round(accuracy, 4))
