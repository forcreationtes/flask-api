from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

app = Flask(__name__)
model = joblib.load("drop_predictor.pkl")

@app.route("/")
def home():
    return "✅ ML Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print("📥 Received data:", data)

        # Step 1: Build DataFrame with raw input
        df = pd.DataFrame([data])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        # Step 2: Calculate features (same as model was trained on)
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()
        df.fillna(method='bfill', inplace=True)

        # Step 3: Only pass trained-on features
        X = df[['rsi', 'ema9', 'ema21']]

        # Step 4: Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(probability, 4)
        })

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
