# train_and_predict.py
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# ================================
# 1. Fetch Data
# ================================
print("📥 Fetching BTC-USD data from Yahoo Finance...")
btc = yf.download("BTC-USD", start="2014-01-01")

# ================================
# 2. Add Technical Indicators
# ================================
btc["RSI"] = ta.momentum.RSIIndicator(btc["Close"]).rsi()
btc["MACD"] = ta.trend.MACD(btc["Close"]).macd()
btc["Signal"] = ta.trend.MACD(btc["Close"]).macd_signal()
btc["Volume_Change"] = btc["Volume"].pct_change()
btc = btc.dropna()

# ================================
# 3. Features & Labels
# ================================
btc["Target"] = btc["Close"].shift(-5)   # Predict 5 days ahead
features = ["Close", "RSI", "MACD", "Signal", "Volume", "Volume_Change"]
X = btc[features].iloc[:-5]
y = btc["Target"].dropna()

# ================================
# 4. Train/Test Split
# ================================
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ================================
# 5. Train Model (or Load Latest)
# ================================
model_file = "btc_xgb_5day_model_latest.joblib"
if os.path.exists(model_file):
    print("🔄 Loading existing model...")
    model = joblib.load(model_file)
else:
    print("🆕 Training new model...")
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)

# Always retrain with full dataset (continuous learning)
model.fit(X, y)
joblib.dump(model, model_file)

# ================================
# 6. Predictions
# ================================
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Model RMSE: {rmse:.2f}")

# Predict next 5 days
latest_data = btc[features].iloc[-5:]
future_pred = model.predict(latest_data)

future_dates = pd.date_range(start=btc.index[-1] + pd.Timedelta(days=1), periods=5)
pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_pred})
pred_df.to_csv("btc_predictions.csv", index=False)

# ================================
# 7. Plot Predictions
# ================================
plt.figure(figsize=(12, 6))
plt.plot(btc.index[-100:], btc["Close"].iloc[-100:], label="Actual Price")
plt.plot(pred_df["Date"], pred_df["Predicted_Close"], "r--", label="Predicted (5-day)")
plt.legend()
plt.title("BTC Price Prediction (Next 5 Days)")
plt.savefig("btc_prediction.png")
plt.close()

# ================================
# 8. Plot RSI
# ================================
plt.figure(figsize=(12, 4))
plt.plot(btc.index[-200:], btc["RSI"].iloc[-200:], label="RSI", color="purple")
plt.axhline(70, linestyle="--", color="red")
plt.axhline(30, linestyle="--", color="green")
plt.legend()
plt.title("BTC RSI (Relative Strength Index)")
plt.savefig("btc_rsi.png")
plt.close()

# ================================
# 9. Plot MACD
# ================================
plt.figure(figsize=(12, 4))
plt.plot(btc.index[-200:], btc["MACD"].iloc[-200:], label="MACD", color="blue")
plt.plot(btc.index[-200:], btc["Signal"].iloc[-200:], label="Signal", color="orange")
plt.legend()
plt.title("BTC MACD (Moving Average Convergence Divergence)")
plt.savefig("btc_macd.png")
plt.close()

# ================================
# 10. Plot Volume
# ================================
plt.figure(figsize=(12, 4))
plt.bar(btc.index[-200:], btc["Volume"].iloc[-200:], color="gray")
plt.title("BTC Trading Volume")
plt.savefig("btc_volume.png")
plt.close()

print("📊 Plots saved: btc_prediction.png, btc_rsi.png, btc_macd.png, btc_volume.png")
from generate_report import generate_report

# After training and plots:
generate_report(rmse)
