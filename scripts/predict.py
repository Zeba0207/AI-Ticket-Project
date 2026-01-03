import joblib
from pathlib import Path
import pandas as pd

# ==============================
# Load paths & models
# ==============================
base = Path(__file__).resolve().parents[1]

vectorizer = joblib.load(base / "models" / "tfidf.pkl")
model = joblib.load(base / "models" / "category_model.pkl")

# ==============================
# Rule-based category detection
# (DATASET-ALIGNED, ORDER MATTERS)
# ==============================
def rule_based_category(text):
    t = text.lower()

    # -------- ACCESS --------
    if any(k in t for k in [
        "unable to access", "cannot access", "can't access",
        "login", "log in", "log into", "signin", "sign in",
        "access denied", "credentials", "otp"
    ]):
        return "Access"

    # -------- NETWORK --------
    if any(k in t for k in [
        "vpn", "wifi", "wi-fi", "internet", "network",
        "lan", "connection", "disconnect", "slow internet"
    ]):
        return "Network"

    # -------- HARDWARE --------
    if any(k in t for k in [
        "laptop", "desktop", "tablet",
        "keyboard", "mouse", "screen",
        "monitor", "printer", "hardware"
    ]):
        return "Hardware"

    # -------- SOFTWARE --------
    if any(k in t for k in [
        "application", "app", "software",
        "outlook", "zoom", "crm", "portal",
        "not opening", "fails to load",
        "crashing", "error"
    ]):
        return "Software"

    # -------- SECURITY --------
    if any(k in t for k in [
        "virus", "malware", "security",
        "firewall", "phishing", "ransomware"
    ]):
        return "Security"

    return None

# ==============================
# Hybrid prediction (Rules + ML)
# ==============================
def predict(text):
    rule_pred = rule_based_category(text)
    if rule_pred:
        return rule_pred

    X = vectorizer.transform(pd.Series([text]))
    return model.predict(X)[0]

# ==============================
# CLI test
# ==============================
if __name__ == "__main__":
    t = input("Enter ticket description: ")
    print("\nPredicted Category:", predict(t))
