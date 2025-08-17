# train_and_predict.py
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import joblib
from datetime import datetime, timedelta
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# =========================
# 1. Fetch BTC Data
# =========================
def get_btc_data():
    df = yf.download("BTC-USD", start="2014-01-01")
    df.reset_index(inplace=True)

    # Add Technical Indicators
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["MACD"] = ta.trend.MACD(df["Close"]).macd()
    df["Signal_Line"] = ta.trend.MACD(df["Close"]).macd_signal()
    df["EMA_20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["EMA_50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    df["Volume_Change"] = df["Volume"].pct_change()

    df.dropna(inplace=True)
    return df

# =========================
# 2. Train Model & Predict
# =========================
def train_and_predict(df):
    # Create future target (5-day ahead price)
    df["Target"] = df["Close"].shift(-5)
    df.dropna(inplace=True)

    features = ["Close", "RSI", "MACD", "Signal_Line", "EMA_20", "EMA_50", "Volume_Change"]
    X = df[features]
    y = df["Target"]

    # Train-test split
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Train XGBoost
    model = XGBRegressor(objective="reg:squarederror", n_estimators=200)
    model.fit(X_train, y_train)

    # Save Model
    joblib.dump(model, "btc_xgb_5day_model_latest.joblib")

    # Predict future
    last_data = df[features].iloc[-5:]
    future_preds = model.predict(last_data)

    future_dates = pd.date_range(start=df["Date"].iloc[-1] + timedelta(days=1), periods=5)
    pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": future_preds})

    return pred_df, model, df

# =========================
# 3. Visualization
# =========================
def plot_results(df, pred_df):
    plt.figure(figsize=(12,6))
    plt.plot(df["Date"], df["Close"], label="Historical Price", color="blue")
    plt.plot(pred_df["Date"], pred_df["Predicted_Close"], label="Predicted Price", color="red", linestyle="--")
    plt.title("Bitcoin Price Prediction (5 Days Ahead)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.savefig("btc_prediction.png")
    plt.close()

# =========================
# 4. Run Script
# =========================
if __name__ == "__main__":
    df = get_btc_data()
    pred_df, model, df = train_and_predict(df)
    plot_results(df, pred_df)

    # Save prediction CSV
    pred_df.to_csv("btc_predictions.csv", index=False)

    print("✅ Training & Prediction Complete")
    print(pred_df)
