# 🚀 AI-Powered Ticket Creation & Categorization System  

Modern helpdesks receive thousands of IT support messages every day. Most messages are unstructured and require manual reading, classification, and ticket creation by support teams.

This project automates that entire process using **NLP + Machine Learning**, enabling faster and more consistent ticket creation.

---

## 📌 Problem Statement
Support agents manually read and classify thousands of incoming user messages daily, which results in:

- Delays in ticket creation  
- Human errors  
- Inconsistent tagging  
- Increased workload  

### 🎯 Goal  
Automatically analyze user messages and generate **clean, structured tickets** with **minimum human involvement**.

---

## 🎯 Objectives
✔ Classify user messages into predefined categories (e.g., Hardware Issue, Network Issue, Software Bug, etc.)  
✔ Predict priority level (Low / Medium / High / Critical)  
✔ Clean & preprocess user text (PII masking + NLP pipeline)  
✔ Auto-generate ticket fields (category, priority, cleaned text)  
✔ Allow prediction for new messages (CLI ticket generator)

---

## 📂 Project Structure

```bash
AI-Ticket-Project/
│── data/
│   ├── raw/
│   ├── cleaned/
│   ├── splits/
│   └── annotated/
│
│── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   ├── tfidf.pkl
│   ├── category_metrics.json
│   └── priority_metrics.json
│
│── scripts/
│   ├── clean_text.py
│   ├── preprocess.py
│   ├── make_splits.py
│   ├── train_model.py
│   ├── generate_ticket.py
│   ├── predict.py
│   ├── check_vectorizers.py
│   └── distribution_check.py
│
│── notebooks/
│── annotation_guidelines/
│── docs/
└── README.md
```

---

## 📊 Dataset
The dataset contains realistic IT support messages including:

- Hardware issues  
- Login/access issues  
- Network failures  
- System/application errors  
- Password reset requests  
- Service requests  

### Dataset Fields
- **text** – raw user message  
- **clean_text** – processed text  
- **category** – assigned issue category  
- **priority** – low/medium/high/critical  

Annotation was completed using **Label Studio**.

---

## 🧠 Models Used

### 🔹 Text Classification Models
- **TF-IDF Vectorizer (8000 features)**  
- **Logistic Regression (balanced class weights)**  
- Metrics saved as JSON for documentation  

### 🔹 Priority Prediction
- Logistic Regression model  
- Uses same TF-IDF features  

### 🔹 Preprocessing & Cleaning
Performed by:
- `clean_text.py`
- `preprocess.py`

Includes:
- Lowercasing  
- Special character removal  
- PII masking (email, phone, IP)  
- Tokenization  
- Stopword removal  
- Lemmatization  

---

## 🔁 End-to-End Pipeline

```
User Message
     ↓
Text Preprocessing (clean_text.py)
     ↓
Train/Val/Test Split (make_splits.py)
     ↓
Model Training (train_model.py)
     ↓
Category + Priority Prediction
     ↓
Ticket Generation (generate_ticket.py)
```

---

## 🛠 Technologies Used

| Category         | Tools / Libraries                   |
|------------------|--------------------------------------|
| Machine Learning | Scikit-Learn, Logistic Regression    |
| NLP              | spaCy, Regex, Lemmatization         |
| Annotation       | Label Studio                        |
| Programming      | Python, Pandas, NumPy               |
| Storage          | CSV, JSON                           |

---

## 📅 Milestones Completed

### ✅ **Milestone 1 – Dataset & Annotation**
✔ Folder structure created  
✔ Raw & sample data explored  
✔ Label Studio setup  
✔ Annotation guidelines prepared  
✔ Labeled dataset exported  

### ✅ **Milestone 2 – Preprocessing & Text Cleaning**
✔ PII masking  
✔ Lemmatization + stopword removal  
✔ Cleaned dataset generated  
✔ Consistency checks performed  

### ✅ **Milestone 3 – Model Development & Ticket Prediction**
✔ TF-IDF vectorizer created  
✔ Train/Val/Test split  
✔ Category model trained  
✔ Priority model trained  
✔ Evaluation metrics saved  
✔ Ticket prediction script working  
✔ Distribution checks added  

You have successfully reproduced the full ML pipeline and prediction workflow 🎉

---

## 🧪 Current Status
✔ Dataset cleaned  
✔ Train/Val/Test split complete  
✔ Models trained and evaluated  
✔ Category & Priority prediction working  
✔ Ticket generator tested and validated  

You now have a **fully functional AI Ticket Classification System**.

---

## 🚀 Future Enhancements
- BERT / Transformer-based text classifier  
- Flask/Streamlit UI for real-time predictions  
- Integration with ServiceNow / Jira  
- Confusion matrix visualization  
- Multi-language support  
- API deployment (FastAPI)  

---

## 👩‍💻 Author
**Shaik Zeba**  
AI Ticket Project – 2025  

