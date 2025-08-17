# train_and_predict.py
import yfinance as yf
import pandas as pd
import ta
import joblib
from xgboost import XGBRegressor
from datetime import datetime, timedelta

# 1. Fetch BTC data from Yahoo Finance (full history since 2014)
print("📥 Fetching Bitcoin data from Yahoo Finance...")
df = yf.download("BTC-USD", start="2014-01-01")

# Reset index for easier handling
df.reset_index(inplace=True)

# 2. Add Technical Indicators
print("⚙️ Adding technical indicators...")

# RSI
df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()

# MACD
macd = ta.trend.MACD(df["Close"])
df["MACD"] = macd.macd()
df["Signal"] = macd.macd_signal()

# Bollinger Bands
bb = ta.volatility.BollingerBands(df["Close"])
df["BB_upper"] = bb.bollinger_hband()
df["BB_lower"] = bb.bollinger_lband()

# Moving Averages
df["SMA_20"] = df["Close"].rolling(window=20).mean()
df["EMA_20"] = df["Close"].ewm(span=20).mean()

# Drop NA (from indicators)
df.dropna(inplace=True)

# 3. Save enriched dataset
df.to_csv("btc_dataset.csv", index=False)
print("✅ Saved dataset with indicators: btc_dataset.csv")

# 4. Prepare data for training
features = ["RSI", "MACD", "Signal", "SMA_20", "EMA_20", "BB_upper", "BB_lower"]
target = "Close"

# Create shifted target for 5-day ahead prediction
df["Target_Close"] = df[target].shift(-5)
df.dropna(inplace=True)

X = df[features]
y = df["Target_Close"]

# 5. Train Model
print("🤖 Training model...")
model = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42)
model.fit(X, y)

# Save trained model
joblib.dump(model, "btc_xgb_5day_model_latest.joblib")
print("✅ Model saved: btc_xgb_5day_model_latest.joblib")

# 6. Predict 5-day ahead prices
print("🔮 Making predictions...")
df["Predicted_Close"] = model.predict(X)

# Save predictions separately
pred_df = df[["Date", "Predicted_Close"]]
pred_df.to_csv("btc_predictions.csv", index=False)
print("✅ Predictions saved: btc_predictions.csv")
