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
        # Parse incoming JSON
        data = request.get_json()
        print("📥 Incoming JSON data:", data)

        # Check if data is missing
        if not data:
            raise ValueError("❌ No JSON data received in request.")

        # Create DataFrame
        df = pd.DataFrame([data])
        print("📊 Converted to DataFrame:", df)

        # Ensure numeric conversion just in case
        df = df.astype(float)

        # Add technical indicators
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()

        # Fill NaNs that may occur
        df.fillna(method='bfill', inplace=True)

        # Debug final features
        print("🔍 Model Features:", df[['rsi', 'ema9', 'ema21']])

        # Extract features
        X = df[['rsi', 'ema9', 'ema21']]

        # Make prediction
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        print(f"✅ Prediction: {prediction}, Probability: {prob}")

        return jsonify({
            "prediction": int(prediction),
            "probability": round(prob, 4)
        })

    except Exception as e:
        print("❌ ERROR in /predict:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
