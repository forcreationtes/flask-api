from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

app = Flask(__name__)

# Load trained model
model = joblib.load("drop_predictor.pkl")

@app.route("/")
def home():
    return "✅ ML Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("🔍 Received data:", data)

        # Create DataFrame from JSON payload
        df = pd.DataFrame([data])

        # Add technical indicators
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()

        # Fill NaNs that may occur due to small input window
        df.fillna(method='bfill', inplace=True)

        # Select model input features
        X = df[['rsi', 'ema9', 'ema21']]

        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(prob, 4)
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
