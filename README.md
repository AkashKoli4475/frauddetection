# 💳 Online Payments Fraud Detection System

A Machine Learning + Web Application project that detects fraudulent online transactions using XGBoost and Flask.

---

## 🚀 Project Overview

Online payments have made transactions easy, but fraud cases have also increased.  
This project detects whether a transaction is **fraudulent or safe** using Machine Learning.

User enters transaction details → system predicts:
- ✅ Safe Transaction  
- ⚠️ Fraud Transaction  

---

## 🎯 Key Features

- Real-time fraud detection
- Machine Learning model (XGBoost)
- Clean and responsive UI
- Confidence score display
- Color-based result (Green = Safe, Red = Fraud)
- Handles imbalanced dataset (SMOTE)

---

## 🧠 Technologies Used

| Technology | Purpose |
|----------|--------|
| Python | Backend |
| Flask | Web Framework |
| XGBoost | ML Model |
| Pandas & NumPy | Data Processing |
| Scikit-learn | Preprocessing |
| HTML/CSS | Frontend UI |

---

## 📊 Dataset Information

- Dataset: PaySim (Kaggle)
- Records: ~6.3 Million transactions

### Features:

| Feature | Description |
|--------|------------|
| step | Time (1 step = 1 hour) |
| type | Transaction type |
| amount | Transaction amount |
| oldbalanceOrg | Sender balance before |
| newbalanceOrig | Sender balance after |
| oldbalanceDest | Receiver balance before |
| newbalanceDest | Receiver balance after |
| isFraud | Target (0 = Safe, 1 = Fraud) |

---

## ⚙️ Data Preprocessing

- Removed irrelevant columns (`nameOrig`, `nameDest`)
- Handled missing values
- Encoded categorical feature (`type`)
- Applied **log transformation on amount**
- Balanced dataset using **SMOTE**

---

## 🤖 Machine Learning Model

- Model: XGBoost Classifier
- Accuracy: ~99%
- Focus: High recall for fraud detection

---

## 🧪 Prediction Logic

```python
prob = model.predict_proba(input)[0][1]

if prob > 0.8:
    "Fraud"
else:
    "Safe"
```

---

## 📁 Project Structure

```
fraud-detection/
│
├── app.py
├── fraud_model.json
├── type_encoder.pkl
├── requirements.txt
│
├── templates/
│   └── index.html
│
├── static/
│
└── README.md
```

---

## 🖥️ How to Run Locally

### 1. Clone repository
```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Flask app
```bash
python app.py
```

### 4. Open browser
```
http://127.0.0.1:5000
```

---

## ☁️ Deployment

You can deploy this project on:

- **Render** (recommended)
- **Railway**

---

## 🧪 Example Test Inputs

### ✅ Safe Transaction
| Field | Value |
|-------|-------|
| Type | PAYMENT |
| Amount | 1000 |
| Old Balance | 5000 |
| New Balance | 4000 |

### ⚠️ Fraud Transaction
| Field | Value |
|-------|-------|
| Type | CASH_OUT |
| Amount | 500000 |
| Old Balance | 500000 |
| New Balance | 0 |
