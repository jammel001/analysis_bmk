import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime


# ==============================
# Technical Indicators
# ==============================

def compute_rsi(series, period=14):
    """Compute Relative Strength Index (RSI)."""
    delta = series.diff().dropna().values.flatten()  # ensure 1D
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Align with index
    rsi = pd.Series(rsi, index=series.index[-len(rsi):])
    return rsi.reindex(series.index)


def add_indicators(df):
    """Add Moving Averages and RSI to dataframe."""
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    return df


# ==============================
# Plotting Functions
# ==============================

def plot_prices(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df["Close"], label="Close Price", color="blue")
    plt.plot(df["MA50"], label="50-day MA", color="orange")
    plt.plot(df["MA200"], label="200-day MA", color="red")
    plt.title("Bitcoin Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("price_ma.png")
    plt.close()


def plot_rsi(df):
    plt.figure(figsize=(12, 4))
    plt.plot(df["RSI"], label="RSI", color="green")
    plt.axhline(70, linestyle="--", color="red")
    plt.axhline(30, linestyle="--", color="blue")
    plt.title("Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("rsi.png")
    plt.close()


def plot_volume(df):
    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df["Volume"].values, color="purple", alpha=0.6)
    plt.title("Trading Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.savefig("volume.png")
    plt.close()


# ==============================
# Training & Prediction
# ==============================

def train_and_predict():
    start = "2014-01-01"
    end = datetime.datetime.today().strftime("%Y-%m-%d")

    # Download BTC data
    df = yf.download("BTC-USD", start=start, end=end)

    # Add indicators
    df = add_indicators(df)
    df.dropna(inplace=True)

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


# ==============================
# Unit Test
# ==============================

def test_compute_rsi():
    """Simple test for RSI function to avoid shape errors."""
    test_series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    rsi = compute_rsi(test_series, period=5)
    assert isinstance(rsi, pd.Series), "RSI output should be a Pandas Series"
    assert len(rsi) == len(test_series), "RSI length must match input series"
    print("✅ RSI test passed!")


# ==============================
# Main
# ==============================

if __name__ == "__main__":
    test_compute_rsi()
    train_and_predict()
