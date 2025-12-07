# AI-Powered Ticket Creation & Categorization System  

Modern helpdesks receive thousands of IT support messages every day. Most messages are unstructured and require manual reading, classification, and ticket creation by support teams.  

This project automates that process using NLP and Machine Learning.

---

## 📌 Problem Statement
Support agents manually read user messages and classify them into ticket categories, which leads to:

- Delays
- Human errors
- Inconsistent ticket tagging
- Extra workload

The goal is to automatically analyze user messages and generate structured tickets with minimum human involvement.

---

## 🎯 Objectives
✔ Classify user messages into predefined categories  
✔ Assign priority level  
✔ Extract key information using NER  
✔ Auto-generate ticket structure  
✔ Display generated ticket (UI)  

---

## 📂 Project Structure
AI-Ticket-Project/
│
├── data/
│ ├── raw
│ ├── cleaned
│ ├── annotated
│
├── scripts/
│ ├── clean_text.py
│ ├── sample_for_annotation.py
│
├── annotation_guidelines/
├── notebooks/
├── models/
├── docs/

---

## 📊 Dataset
The dataset contains real support text such as:
- hardware issues
- login issues
- network problems
- application errors
- password reset requests

Fields include:
- text
- clean_text
- category
- priority

Annotation done using **Label Studio**.

---

## 🧠 Models Used
### 🔹 Text Classification
- Logistic Regression / SVM / Random Forest
- BERT (optional next step)

### 🔹 NER (Named Entity Recognition)
Extracts:
- user name
- system
- error codes
- dates

### 🔹 Priority Prediction
Rule-based or ML based

---

## 🔁 Pipeline
User Message
→ Text Preprocessing
→ ML Classification
→ NER Extraction
→ Priority Prediction
→ Ticket Creation
→ UI Display

---

## 🛠 Technologies
| Category         | Tools                      |
| ---------------- | -------------------------- |
| Machine Learning | Scikit-Learn, Transformers |
| Text Processing  | NLP, Regex                 |
| Annotation       | Label Studio               |
| UI               | Flask, Streamlit           |
| Programming      | Python, Pandas, NumPy      |
| Storage          | CSV, JSON                  |


---

## 📅 Milestones Completed
### ✅ Milestone 1 (DONE)
✔ folder structure  
✔ cleaned dataset  
✔ annotation setup  
✔ sample dataset labeling  

### 🔜 Milestone 2
Model development  

---

## 🧩 Challenges
- lack of real ticket data
- inconsistent user text
- designing entity extraction rules
- balancing rule-based + ML techniques  

---

## 🚀 Future Improvements
- Jira/ServiceNow integration
- voice-to-ticket
- multi-language support
- advanced transformer models
- real-time ticket generation

---

## 🧪 Current Status
Dataset prepared, annotation done, ready to start model training 😄  

---

## 🔗 Author
**Shaik Zeba**  
AI-Ticket-Project – 2025  
