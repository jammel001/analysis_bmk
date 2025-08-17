import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Fetch BTC data
# -----------------------------
btc = yf.download("BTC-USD", start="2014-01-01", auto_adjust=False)

# Ensure we have Adj Close (fallback to Close if missing)
if "Adj Close" in btc.columns:
    btc["Price"] = btc["Adj Close"]
else:
    print("⚠️ Warning: 'Adj Close' not found, using 'Close' instead")
    btc["Price"] = btc["Close"]

# -----------------------------
# 2. Feature engineering
# -----------------------------
btc["Return"] = btc["Price"].pct_change()
btc["SMA_10"] = btc["Price"].rolling(window=10).mean()
btc["SMA_50"] = btc["Price"].rolling(window=50).mean()

# RSI
delta = btc["Price"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
btc["RSI"] = 100 - (100 / (1 + rs))

# MACD
ema12 = btc["Price"].ewm(span=12, adjust=False).mean()
ema26 = btc["Price"].ewm(span=26, adjust=False).mean()
btc["MACD"] = ema12 - ema26
btc["Signal"] = btc["MACD"].ewm(span=9, adjust=False).mean()

# Drop NaN rows
btc = btc.dropna()

# -----------------------------
# 3. Prepare dataset
# -----------------------------
X = btc[["Return", "SMA_10", "SMA_50", "RSI", "MACD", "Signal"]]
y = btc["Price"].shift(-5)  # predict 5 days ahead
btc = btc.dropna()

X = X.iloc[:-5]
y = y.dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# -----------------------------
# 4. Train model
# -----------------------------
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=200)
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate
# -----------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("✅ RMSE:", rmse)

# -----------------------------
# 6. Save model
# -----------------------------
joblib.dump(model, "btc_xgb_5day_model_latest.joblib")

# -----------------------------
# 7. Plot prediction vs actual
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="Actual Price", color="blue")
plt.plot(y_test.index, y_pred, label="Predicted Price", color="red")
plt.title("BTC 5-Day Ahead Prediction")
plt.legend()
plt.savefig("btc_prediction.png")
plt.close()
