import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# === SETTINGS ===
MODEL_FILE = "btc_xgb_5day_model_latest.joblib"
PREDICTION_PNG = "btc_prediction.png"
PREDICTIONS_FILE = "btc_predictions.csv"

# === Fetch Bitcoin Data (since 2014) ===
btc = yf.download("BTC-USD", start="2014-01-01")

# === Feature Engineering ===
btc["Return"] = btc["Adj Close"].pct_change()
btc["SMA_10"] = btc["Adj Close"].rolling(window=10).mean()
btc["SMA_50"] = btc["Adj Close"].rolling(window=50).mean()
btc["RSI"] = 100 - (100 / (1 + btc["Return"].rolling(14).mean()))
btc = btc.dropna()

# Shift target (predict 5 days ahead)
btc["Target"] = btc["Adj Close"].shift(-5)
btc = btc.dropna()

# Features and Target
X = btc[["Adj Close", "SMA_10", "SMA_50", "RSI", "Volume"]]
y = btc["Target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === Load or Train Model ===
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    print("✅ Loaded existing model")
else:
    model = XGBRegressor(objective="reg:squarederror", n_estimators=500, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_FILE)
    print("✅ Trained and saved new model")

# === Evaluate Model ===
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Model Performance → RMSE: {rmse:.2f}, R²: {r2:.4f}")

# === Predict Next 5 Days ===
last_data = X.tail(5).copy()
future_predictions = model.predict(last_data)

future_dates = pd.date_range(datetime.today() + timedelta(days=1), periods=5)
pred_df = pd.DataFrame({"Date": future_dates, "Predicted_BTC_Price": future_predictions})

# Save predictions to CSV
pred_df.to_csv(PREDICTIONS_FILE, index=False)
print(f"✅ Saved predictions to {PREDICTIONS_FILE}")

# === Plot Predictions ===
plt.figure(figsize=(10, 5))
plt.plot(btc.index[-200:], btc["Adj Close"].tail(200), label="Historical")
plt.plot(future_dates, future_predictions, "ro--", label="Predicted (5 days)")
plt.xlabel("Date")
plt.ylabel("BTC Price (USD)")
plt.title("Bitcoin Price Prediction (5 days ahead)")
plt.legend()
plt.grid(True)
plt.savefig(PREDICTION_PNG)
plt.close()
print(f"✅ Prediction chart saved to {PREDICTION_PNG}")
