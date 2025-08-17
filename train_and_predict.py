import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import datetime as dt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


# =========================
# Technical Indicator Functions
# =========================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain = pd.Series(gain.flatten(), index=series.index)
    loss = pd.Series(loss.flatten(), index=series.index)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def add_indicators(df):
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["MACD"], df["Signal"] = compute_macd(df["Close"])
    return df.dropna()


# =========================
# Main Training & Prediction
# =========================
def train_and_predict():
    print("📥 Downloading BTC-USD data...")
    end = dt.datetime.today()
    start = end - dt.timedelta(days=5 * 365)  # 5 years of data
    df = yf.download("BTC-USD", start=start, end=end)

    if df.empty:
        raise ValueError("No data downloaded from Yahoo Finance!")

    print(f"✅ Data downloaded: {df.shape}")
    df = add_indicators(df)

    # Features and Target
    X = df[["RSI", "MACD", "Signal"]]
    y = df["Close"].shift(-5)  # predict 5 days ahead
    df = df.dropna()

    X = df[["RSI", "MACD", "Signal"]]
    y = df["Close"]

    if len(X) == 0:
        raise ValueError("Not enough data after adding indicators!")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train model
    print("🤖 Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "btc_xgb_5day_model_latest.joblib")
    print("✅ Model saved as btc_xgb_5day_model_latest.joblib")

    # Predictions
    y_pred = model.predict(X_test)

    # =========================
    # Plot Predictions
    # =========================
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test, label="Actual Price", color="blue")
    plt.plot(y_test.index, y_pred, label="Predicted Price", color="red")
    plt.legend()
    plt.title("BTC Price Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("btc_prediction.png")
    print("📊 Prediction plot saved as btc_prediction.png")

    # =========================
    # Plot RSI
    # =========================
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["RSI"], label="RSI", color="purple")
    plt.axhline(70, linestyle="--", color="red")
    plt.axhline(30, linestyle="--", color="green")
    plt.legend()
    plt.title("BTC RSI")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("btc_rsi.png")
    print("📊 RSI plot saved as btc_rsi.png")

    # =========================
    # Plot MACD
    # =========================
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["MACD"], label="MACD", color="blue")
    plt.plot(df.index, df["Signal"], label="Signal", color="orange")
    plt.legend()
    plt.title("BTC MACD")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("btc_macd.png")
    print("📊 MACD plot saved as btc_macd.png")

    # =========================
    # Plot Volume (fixed with line plot)
    # =========================
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df["Volume"], label="Volume", color="gray")
    plt.legend()
    plt.title("BTC Trading Volume")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("btc_volume.png")
    print("📊 Volume plot saved as btc_volume.png")


# =========================
# Run Script
# =========================
if __name__ == "__main__":
    train_and_predict()
