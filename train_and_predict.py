import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib
from datetime import datetime, timedelta

# ==============================
# Download BTC data
# ==============================
def fetch_data():
    end = datetime.today()
    start = end - timedelta(days=365*2)  # last 2 years
    df = yf.download("BTC-USD", start=start, end=end)
    df.to_csv("btc_data.csv")
    return df

# ==============================
# Feature Engineering
# ==============================
def add_features(df):
    df["Return"] = df["Close"].pct_change()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["Signal"] = compute_macd(df["Close"])
    df["Volume_Change"] = df["Volume"].pct_change()
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# ==============================
# Train & Predict
# ==============================
def train_and_predict():
    df = fetch_data()
    df = add_features(df)

    # target: next 5-day average close
    df["Target"] = df["Close"].shift(-5).rolling(5).mean()
    df.dropna(inplace=True)

    X = df[["Close", "Return", "RSI", "MACD", "Signal", "Volume_Change"]]
    y = df["Target"]

    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5)
    model.fit(X, y)

    joblib.dump(model, "btc_xgb_5day_model_latest.joblib")

    # Prediction for last known values
    latest_features = X.iloc[-1:].values
    prediction = model.predict(latest_features)[0]
    print(f"Predicted BTC price (5-day ahead): {prediction:.2f} USD")

    # Save plots
    plot_prediction(df, prediction)
    plot_rsi(df)
    plot_macd(df)
    plot_volume(df)

    return prediction

# ==============================
# Plot functions
# ==============================
def plot_prediction(df, prediction):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="BTC Close Price")
    plt.axhline(prediction, color="r", linestyle="--", label="5-day Prediction")
    plt.title("BTC Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.savefig("btc_prediction.png")
    plt.close()

def plot_rsi(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["RSI"], label="RSI", color="orange")
    plt.axhline(70, linestyle="--", color="red")
    plt.axhline(30, linestyle="--", color="green")
    plt.title("Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.savefig("btc_rsi.png")
    plt.close()

def plot_macd(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["MACD"], label="MACD", color="blue")
    plt.plot(df.index, df["Signal"], label="Signal Line", color="red")
    plt.title("MACD Indicator")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig("btc_macd.png")
    plt.close()

def plot_volume(df):
    plt.figure(figsize=(10, 4))
    plt.bar(df.index, df["Volume"], color="purple", alpha=0.6)
    plt.title("BTC Trading Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.savefig("btc_volume.png")
    plt.close()

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    train_and_predict()
