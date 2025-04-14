from flask import Flask, request, jsonify
import pandas as pd
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
    data = request.get_json()
    features = np.array([[data['open'], data['high'], data['low'], data['close'], data['volume']]])
    prediction = model.predict(features)
    return jsonify({'drop_predicted': int(prediction[0])})

    try:
        # Create DataFrame from JSON payload
        df = pd.DataFrame([data])

        # Add features
        df['rsi'] = RSIIndicator(df['close'], window=14).rsi()
        df['ema9'] = EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema21'] = EMAIndicator(df['close'], window=21).ema_indicator()

        # Select features model expects
        X = df[['rsi', 'ema9', 'ema21']]

        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": round(prob, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
