import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# =========================
# Helper functions
# =========================
def add_indicators(df):
    # Moving averages with min_periods=1 (keeps early rows)
    df["MA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["MA200"] = df["Close"].rolling(window=200, min_periods=1).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-10)  # avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def plot_prices(df):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Close"], label="BTC Close", color="blue")
    plt.plot(df.index, df["MA50"], label="MA50", color="orange")
    plt.plot(df.index, df["MA200"], label="MA200", color="red")
    plt.title("BTC Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rsi(df):
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df["RSI"], label="RSI", color="green")
    plt.axhline(70, color="red", linestyle="--")
    plt.axhline(30, color="blue", linestyle="--")
    plt.title("Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_volume(df):
    plt.figure(figsize=(12,4))
    plt.bar(df.index, df["Volume"], color="purple", alpha=0.6, width=1.0)
    plt.title("BTC Trading Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.grid(True)
    plt.show()

# =========================
# Main pipeline
# =========================
def train_and_predict():
    start = "2014-01-01"
    end = datetime.datetime.today().strftime("%Y-%m-%d")

    # Download BTC data
    df = yf.download("BTC-USD", start=start, end=end)

    # Add indicators
    df = add_indicators(df)
    df.dropna(inplace=True)  # should keep early rows now

    if df.empty or len(df) < 100:
        raise ValueError("Not enough data after indicators. Check date range or reduce MA windows.")

    # Features and target
    X = df[["MA50", "MA200", "RSI"]]
    y = df["Close"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print("Model Performance:")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.2f}")

    # Predict 5 days ahead
    last_features = X.iloc[-1].values.reshape(1, -1)
    future_price = model.predict(last_features)[0]
    print(f"Predicted BTC price (5-day ahead): {future_price:.2f} USD")

    # Generate plots
    plot_prices(df)
    plot_rsi(df)
    plot_volume(df)


if __name__ == "__main__":
    train_and_predict()
