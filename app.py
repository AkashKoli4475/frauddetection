from flask import Flask, render_template, request
import numpy as np
import pickle
import xgboost as xgb

app = Flask(__name__)

# Load model
model = xgb.XGBClassifier()
model.load_model("fraud_model.json")

# Load encoder
try:
    with open("type_encoder.pkl", "rb") as f:
        type_encoder = pickle.load(f)
    print("✅ Encoder loaded")
except:
    type_encoder = None
    print("⚠️ Using manual encoding")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Inputs
        step = float(request.form.get('step', 0))
        txn_type = request.form.get('type')
        amount = float(request.form.get('amount', 0))
        oldbalanceOrg = float(request.form.get('oldbalanceOrg', 0))
        newbalanceOrig = float(request.form.get('newbalanceOrig', 0))
        oldbalanceDest = float(request.form.get('oldbalanceDest', 0))
        newbalanceDest = float(request.form.get('newbalanceDest', 0))

        # 🔥 IMPORTANT FIX → log transform
        amount = np.log(amount + 1)

        # Encode type
        if type_encoder:
            type_val = type_encoder.transform([txn_type])[0]
        else:
            type_val = encode_type(txn_type)

        # Input format (same as training)
        input_data = [[step, type_val, amount,
                       oldbalanceOrg, newbalanceOrig,
                       oldbalanceDest, newbalanceDest]]

        input_array = np.array(input_data)

        # Prediction probability
        prob = model.predict_proba(input_array)[0][1]

        print("Fraud Probability:", prob)

        # 🔥 Better threshold
        threshold = 0.8

        if prob > threshold:
            result = f"⚠️ Fraud Transaction Detected (Confidence: {prob*100:.2f}%)"
        else:
            result = f"✅ Safe Transaction (Confidence: {(1-prob)*100:.2f}%)"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return f"Error: {str(e)}"


def encode_type(txn_type):
    mapping = {
        "CASH_IN": 0,
        "CASH_OUT": 1,
        "DEBIT": 2,
        "PAYMENT": 3,
        "TRANSFER": 4
    }
    return mapping.get(txn_type, 3)


if __name__ == '__main__':
    app.run(debug=True)