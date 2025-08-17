import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib
import os
from datetime import datetime, timedelta


# ========= Compute Indicators =========
def compute_rsi(series, period=14):
    """Compute Relative Strength Index (RSI)"""
    delta = series.diff()

    gain = delta.clip(lower=0)   # positive changes
    loss = -delta.clip(upper=0)  # negative changes as positive

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_indicators(df):
    """Add RSI, MACD, and Volume indicators to dataframe"""
    # RSI
    df["RSI"] = compute_rsi(df["Close"], 14)

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2

    # Volume (already exists, but ensure column name is consistent)
    df["Volume"] = df["Volume"]

    return df


# ========= Train and Predict =========
def train_and_predict():
    print("📥 Downloading BTC-USD data...")
    end = datetime.today()
    start = end - timedelta(days=5 * 365)  # last 5 years

    df = yf.download("BTC-USD", start=start, end=end)

    if df.empty:
        raise ValueError("❌ No data downloaded from yfinance. Try again later.")

    print("✅ Data downloaded:", df.shape)

    # Add indicators
    df = add_indicators(df)

    # Drop missing values
    df = df.dropna()

    if df.empty:
        raise ValueError("❌ Dataframe empty after adding indicators and dropping NaN.")

    # Features and target
    X = df[["RSI", "MACD", "Volume"]]
    y = df["Close"].shift(-5)  # predict 5 days ahead

    # Drop NaNs caused by shifting
    X = X[:-5]
    y = y[:-5]

    if X.empty or y.empty:
        raise ValueError("❌ Not enough data for training after shift.")

    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train model
    print("🤖 Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    model_filename = "btc_xgb_5day_model_latest.joblib"
    joblib.dump(model, model_filename)
    print(f"✅ Model saved as {model_filename}")

    # Predictions
    y_pred = model.predict(X_test)

    # Plot Predictions
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label="Actual")
    plt.plot(y_test.index, y_pred, label="Predicted")
    plt.legend()
    plt.title("BTC Price Prediction (5 days ahead)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("btc_prediction.png")
    print("📊 Prediction plot saved as btc_prediction.png")

    # Plot RSI
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["RSI"], label="RSI")
    plt.axhline(70, color="red", linestyle="--")
    plt.axhline(30, color="green", linestyle="--")
    plt.legend()
    plt.title("BTC RSI")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("btc_rsi.png")
    print("📊 RSI plot saved as btc_rsi.png")

    # Plot MACD
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["MACD"], label="MACD", color="blue")
    plt.legend()
    plt.title("BTC MACD")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("btc_macd.png")
    print("📊 MACD plot saved as btc_macd.png")

    # Plot Volume
    plt.figure(figsize=(10, 4))
    plt.bar(df.index, df["Volume"], label="Volume", color="gray")
    plt.legend()
    plt.title("BTC Trading Volume")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("btc_volume.png")
    print("📊 Volume plot saved as btc_volume.png")


if __name__ == "__main__":
    train_and_predict()
