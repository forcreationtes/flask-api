import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data
df = pd.read_csv("NVDA_1m_7d.csv")

# Clean up data types
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna().reset_index(drop=True)

# Add RSI
df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

# Create label: 1 = price will go down next bar, 0 = price will go up or stay
df['future_return'] = df['close'].shift(-1) - df['close']
df['label'] = (df['future_return'] < 0).astype(int)

# Features for prediction
features = ['open', 'high', 'low', 'close', 'volume', 'rsi']
X = df[features]
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("📊 Model Evaluation:")
print(classification_report(y_test, model.predict(X_test)))

# Save model
joblib.dump(model, "drop_predictor.pkl")
print("✅ Model saved as drop_predictor.pkl")
