# 🛡️ FinGuard AI: Autonomous Financial Anomaly Detection

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM%20Investigator-412991.svg)

**FinGuard AI** is an enterprise-grade, multi-model unsupervised anomaly detection engine designed to identify fraudulent financial transactions in real-time. Built for the **HackWhack3** Hackathon.

---

## 🚀 Key Features

* **Ensemble AI Engine:** Combines **PyTorch Deep Autoencoders**, **Isolation Forests**, and **Local Outlier Factor (LOF)** to drastically reduce false positives compared to standard rule-based systems.
* **Explainable AI (XAI):** Integrates **SHAP (SHapley Additive exPlanations)** to crack open the "black box" and visually highlight the exact transaction features that triggered a risk alert.
* **Autonomous LLM Investigator:** Automatically synthesizes SHAP data and risk scores using OpenAI's GPT models to generate structured, professional audit reports instantly.
* **Real-Time Alerting:** Triggers immediate browser audio alarms and dispatches SMTP email alerts to security teams the second a high-risk transaction (>0.70 threshold) is detected.
* **Secure Authentication:** Integrated with Google OAuth 2.0 and a custom OTP verification system for institutional security.

---

## 🧠 System Architecture

The system learns "normal" transaction behavior and flags deviations across multiple financial parameters such as Currency, Company Code, Posting Keys, and Transaction Amounts.



### The Tech Stack
* **Backend & ML:** Python, PyTorch, Scikit-Learn, Pandas, NumPy
* **Explainability:** SHAP (TreeExplainer)
* **Generative AI:** OpenAI API (gpt-4o / gpt-3.5-turbo)
* **Frontend Dashboard:** Streamlit, Plotly Express
* **Security:** Authlib (OAuth 2.0), Bcrypt, SQLite3

---

## ⚙️ Local Installation & Setup

To run this application locally for development or auditing:

**1. Clone the repository**
```bash
git clone [https://github.com/VedantShivarkar/FinGuard-AI.git](https://github.com/VedantShivarkar/FinGuard-AI.git)
cd FinGuard-AI
