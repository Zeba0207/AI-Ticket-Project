import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resources on first run
nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("../data/annotated/milestone1_labeled.csv")

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'[^a-zA-Z ]', ' ', text)  
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

df["clean_text"] = df["text"].astype(str).apply(clean_text)

df.to_csv("../data/cleaned/cleaned_dataset.csv", index=False)

print("Saved cleaned file to data/cleaned/")
