# 🚀 AI-Powered Ticket Management System

Modern IT helpdesks receive thousands of unstructured support requests every day.  
Manually reading, classifying, prioritizing, and tracking these requests leads to delays, human errors, and SLA breaches.

This project implements a **full-stack AI-powered helpdesk system** that automatically converts free-text user issues into **structured, persistent support tickets** and provides a **real-world support team dashboard** to manage the complete ticket lifecycle.

---

## 📌 Problem Statement

Support teams manually read and classify thousands of incoming user messages daily, which leads to:

- ⏳ Delays in ticket creation  
- ❌ Human errors and inconsistent categorization  
- 📈 Increased workload for support engineers  

---

## 🎯 Goal

To automatically analyze user messages and generate structured IT support tickets with:

- Minimal human intervention  
- AI-based categorization and priority detection  
- Persistent storage and lifecycle tracking  

---

## 🧠 What This Application Does

### 👤 For Users
- Secure registration and login
- Submit support issues using natural language
- Tickets are automatically:
  - Categorized (Hardware, Network, HR, Access, etc.)
  - Assigned priority (Low / Medium / High)
- Track ticket status

### 🧑‍💻 For Support Teams
- View all tickets in a central dashboard
- Monitor SLA timers with color-coded alerts
- Update ticket status through the lifecycle:
Open → In Progress → Resolved → Closed
- Inspect tickets in JSON format (Developer Mode)
- View analytics and workload distribution

> All tickets are persistently stored in the database and never disappear on refresh.

---

## 🏗️ System Architecture

User Input  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

Streamlit User Interface  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

Text Cleaning & NLP Processing  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

Machine Learning Models  
(Category Classification & Priority Prediction)  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

SQLite Database  

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓

Dashboard & SLA Monitoring


---

## 📂 Project Structure

```bash
AI-Ticket-Project/
│
├── app.py                      # Main Streamlit entry point
│
├── pages/
│   ├── dashboard.py            # Ticket analytics dashboard
│   ├── create_ticket.py        # Ticket creation page
│   ├── active_tickets.py       # Active tickets (support team)
│   ├── closed_tickets.py       # Closed tickets archive
│   ├── login.py                # Login page
│   ├── register.py             # User registration
│   └── profile.py              # User profile
│
├── scripts/
│   ├── db.py                   # SQLite database operations
│   ├── auth.py                 # Authentication logic
│   ├── ai_logic.py             # Category & priority prediction
│   ├── clean_text.py           # NLP preprocessing
│   └── entity_extraction.py    # Named Entity Recognition
│
├── models/
│   ├── category_model.pkl
│   ├── priority_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── category_encoder.pkl
│   └── priority_encoder.pkl
│
├── assets/
│   └── style.css               # Custom UI styling
│
├── tickets.db                  # SQLite database
├── requirements.txt
└── README.md

```

---
## 🧠 NLP & Machine Learning Models

This section describes the Natural Language Processing and Machine Learning techniques used to automatically analyze, categorize, and prioritize support tickets.

### 🔹 Feature Engineering
- TF-IDF Vectorization (Unigrams + Bigrams)
- Stopword removal
- Text normalization

### 🔹 Category Classification
- Linear Support Vector Machine (LinearSVC)
- Balanced class weights
- Rule-based overrides for critical keywords

### 🔹 Priority Prediction
- Logistic Regression
- Predicts **Low / Medium / High**
- Urgent keywords trigger escalation (e.g., *urgent, ASAP, system down*)

### 🔹 Entity Extraction
- Device names (laptop, printer, keyboard)
- Error-related keywords
- User references

---

💾 Database Design (SQLite)

Each ticket is stored as a row in the database:
```text
+-------------+-----------------------------------------------+
| Field       | Description                                   |
+-------------+-----------------------------------------------+
| id          | Auto-increment primary key                    |
| title       | Short summary of the issue                    |
| description | Original user input (free text)               |
| category    | AI-predicted ticket category                  |
| priority    | AI-predicted priority (Low / Medium / High)   |
| status      | Open / In Progress / Resolved / Closed         |
| created_at  | Ticket creation timestamp                     |
| updated_at  | Last status update timestamp                  |
+-------------+-----------------------------------------------+

```

Users are stored in a separate table with hashed passwords.



---
## ⏱ SLA Monitoring

🟢 Green → Less than 2 hours

🟡 Yellow → 2–6 hours

🔴 Red → More than 6 hours

This simulates enterprise-grade SLA enforcement.


---

## 📊 Dashboard Analytics

Support teams can monitor:

- Total tickets
- Open tickets
- High-priority tickets
- Closed tickets

All metrics update dynamically.



---

## 🧪 Example Ticket (JSON View)
```
{
  "id": 8,
  "description": "VPN disconnects every 10 minutes while working remotely",
  "category": "Network",
  "priority": "High",
  "status": "In Progress",
  "created_at": "2026-01-21 17:03:30"
}
```
---


## 🧑‍💻 Support Team Workflow

1. User submits a support request
2. AI classifies and prioritizes the ticket
3. Ticket is stored in the database
4. Support team processes the ticket
5. Status is updated through the ticket lifecycle
6. Resolved tickets are archived

---

## 🚀 How to Run Locally
```
pip install -r requirements.txt
streamlit run app.py
```
---
## 📦 Requirements

Core dependencies:
- streamlit
- pandas
- numpy
- scikit-learn
- joblib
- sqlite3

All dependencies are listed in `requirements.txt`.

---
## 🌍 Deployment
- Recommended
  - Streamlit Community Cloud (best for demos)
  - SQLite is sufficient for demo and evaluation

- Production-Ready Upgrade
  - PostgreSQL instead of SQLite
  - Role-based access (Admin / Support Agent)
  - FastAPI backend

---

### Streamlit Deployment Notes

- Entry file: `app.py`
- Python version: 3.9+
- Ensure all `.pkl` model files are committed
- SQLite database initializes automatically
---

🔮 Future Enhancements

-Agent assignment

-Notification system for high-priority tickets

-Chat-based ticket creation

-Transformer-based NLP models (BERT)

-REST API integration

-Multi-language support

---
## 👩‍💻 Author

**Shaik Zeba**  
AI-Powered Ticket Management System  
© 2026
