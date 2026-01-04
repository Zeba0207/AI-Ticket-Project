# 🚀 AI-Powered Ticket Creation & Categorization System  

Modern helpdesks receive thousands of IT support messages every day. These messages are often unstructured and require manual effort to read, classify, and convert into support tickets.

This project automates the entire process using **Natural Language Processing (NLP)** and **Machine Learning**, enabling faster, consistent, and reliable ticket creation with minimal human intervention.

---

## 📌 Problem Statement

Support agents manually read and classify thousands of incoming user messages daily, which leads to:

- Delays in ticket creation  
- Human errors and inconsistent tagging  
- Increased workload for support teams  

### 🎯 Goal
Automatically analyze user messages and generate **structured IT support tickets** with **minimum human involvement**.

---

## 🎯 Objectives

- Clean and preprocess raw user messages (PII masking + NLP pipeline)  
- Classify messages into predefined ticket categories  
- Predict ticket priority (Low / Medium / High)  
- Extract relevant entities (devices, usernames, error codes)  
- Generate a complete, structured ticket in JSON-ready format  
- Enable predictions for new messages using a CLI-based ticket generator  

---

## 📂 Project Structure

```bash
AI-Ticket-Project/
│── data/
│   ├── raw/                   # Raw input data
│   ├── cleaned/               # Final cleaned dataset
│   ├── splits/                # Train/Validation/Test splits
│   └── annotated/             # Annotated data from Label Studio
│
│── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── category_encoder.pkl
│   └── priority_encoder.pkl
│
│── scripts/
│   ├── clean_text.py          # Text preprocessing module
│   ├── entity_extraction.py   # Named Entity Extraction (NER)
│   ├── make_splits.py         # Dataset splitting logic
│   ├── train_model.py         # Model training & evaluation
│   ├── generate_ticket.py     # Ticket Generation Engine
│   └── predict.py             # CLI-based prediction utility
│
│── notebooks/                 # Exploratory analysis
│── docs/                      # Documentation and notes
└── README.md
---
## 📊 Dataset

The dataset contains realistic IT support messages such as:

- Hardware issues  
- Login and access failures  
- Network connectivity problems  
- Software / application errors  
- Purchase and service requests  

### Dataset Fields

- **text** – Raw user message  
- **text_clean** – Cleaned and normalized text  
- **category** – Issue category label  
- **priority** – Ticket priority level  

Dataset annotation was performed using **Label Studio** following predefined guidelines.

---

⚖️ Category Distribution & Imbalance Handling

The dataset showed moderate class imbalance across issue categories.

Steps taken to address this:

Class Weights: Applied during model training

Stratified Splits: Used for Train/Validation/Test data

Evaluation Metrics: Precision, Recall, and F1-score monitored per class

No synthetic oversampling (e.g., SMOTE) was applied to avoid introducing artificial text samples.
---
🧠 NLP Models Used
🔹 Feature Extraction

TF-IDF Vectorizer

Uni-grams, Bi-grams, Tri-grams

Stopword removal and sublinear TF scaling

🔹 Category Classification

Linear Support Vector Machine (LinearSVC)

Hyperparameter tuning using GridSearchCV

Balanced class weights for robustness

🔹 Priority Prediction

Logistic Regression

Balanced class weights

Predicts Low / Medium / High priority

🔹 Named Entity Recognition (NER)

Pattern-based extraction of:

Devices (laptop, mouse, printer, etc.)

User references

Error codes
---
🔁 End-to-End Pipeline
User Message
     ↓
Text Cleaning & Normalization
     ↓
TF-IDF Feature Extraction
     ↓
Category & Priority Prediction
     ↓
Entity Extraction (NER)
     ↓
Structured Ticket Generation (JSON)
---
## 🛠 Technologies Used

| Category          | Tools / Libraries                          |
|-------------------|--------------------------------------------|
| Programming       | Python                                     |
| NLP               | Scikit-learn, Regex                        |
| Machine Learning  | Linear SVM, Logistic Regression            |
| Data Handling     | Pandas, NumPy                              |
| Annotation        | Label Studio                               |
| Evaluation        | Accuracy, Precision, Recall, F1-score     |
## ✅ Modules Completed

| Module   | Description                         | Status        |
|----------|-------------------------------------|---------------|
| Module 1 | Data Collection & Preprocessing     | ✅ Completed  |
| Module 2 | NLP Model Development + NER          | ✅ Completed  |
| Module 3 | Ticket Generation Engine             | ✅ Completed  |
| Module 4 | UI & Integration Layer               | ⏳ Planned    |

🧪 Current Project Status

Dataset cleaned and standardized

Models trained and evaluated

Confusion matrices generated

Hybrid rule-based + ML classification implemented

Ticket generation engine validated

JSON-ready structured ticket output achieved

This results in a fully functional AI-powered IT ticketing system.

🚀 How to Run the Project
Train the Models
python scripts/train_model.py

Generate a Ticket (CLI)
python scripts/generate_ticket.py

🧾 Example Output (JSON-ready)
{
  "title": "Purchase Issue",
  "category": "purchase",
  "priority": "low",
  "entities": {
    "devices": ["mouse"],
    "usernames": [],
    "error_codes": []
  },
  "status": "open",
  "created_at": "2026-01-04T22:34:12"
}

🚀 Future Enhancements

Streamlit / Flask Web UI (Module 4)

Transformer-based models (BERT)

REST API using FastAPI

Database integration (MongoDB / PostgreSQL)

Prediction confidence scores

Multi-language support

👩‍💻 Author

Shaik Zeba
AI-Powered Ticket Creation & Categorization System
2025
