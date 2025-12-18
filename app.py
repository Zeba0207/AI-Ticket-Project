import streamlit as st
import pickle
from scripts.clean_text import clean_text

# Load vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load models
with open("models/category_model.pkl", "rb") as f:
    category_model = pickle.load(f)

with open("models/priority_model.pkl", "rb") as f:
    priority_model = pickle.load(f)

st.set_page_config(page_title="AI Ticket Generator")

st.title("🎫 AI-Powered Ticket Generation System")

user_input = st.text_area(
    "Describe your issue:",
    placeholder="e.g. My internet is not working since morning"
)

if st.button("Generate Ticket"):
    if user_input.strip() == "":
        st.warning("Please enter an issue description.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        category = category_model.predict(vectorized_text)[0]
        priority = priority_model.predict(vectorized_text)[0]

        st.success("Ticket Generated Successfully ✅")
        st.write("**Category:**", category)
        st.write("**Priority:**", priority)
