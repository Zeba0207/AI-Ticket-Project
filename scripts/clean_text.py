import re
from html import unescape
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------------------------
# SAFE NLTK RESOURCE LOADING (DEPLOYMENT)
# --------------------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4", quiet=True)

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Cleans and normalizes input text for NLP models.
    Safe for local + Streamlit Cloud deployment.
    """

    if not isinstance(text, str):
        return ""

    # 1️⃣ Decode HTML entities & lowercase
    text = unescape(text).lower()

    # 2️⃣ Remove PII (emails, phone numbers, IP addresses)
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)
    text = re.sub(r'\b\d{10}\b', ' ', text)
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' ', text)

    # 3️⃣ Remove punctuation & special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 4️⃣ Tokenization (simple & deployment-safe)
    tokens = text.split()

    # 5️⃣ Stopword removal & short-token filtering
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 6️⃣ Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    # 7️⃣ Remove duplicate tokens (order preserved)
    tokens = list(dict.fromkeys(tokens))

    return " ".join(tokens)
