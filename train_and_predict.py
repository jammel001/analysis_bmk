import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# ===============================
# Download BTC data
# ===============================
def load_data():
    end = datetime.today()
    start = end - timedelta(days=365 * 2)  # last 2 years
    df = yf.download("BTC-USD", start=start, end=end)
    df.dropna(inplace=True)
    return df

# ===============================
# Technical indicators
# ===============================
def add_indicators(df):
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ===============================
# Plot functions
# ===============================
def plot_price(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Close"], label="Close Price", color="blue")
    plt.plot(df.index, df["SMA_20"], label="SMA 20", color="orange")
    plt.plot(df.index, df["SMA_50"], label="SMA 50", color="green")
    plt.title("BTC Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("btc_price.png")
    plt.close()

def plot_rsi(df):
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["RSI"], label="RSI", color="red")
    plt.axhline(70, linestyle="--", color="gray")
    plt.axhline(30, linestyle="--", color="gray")
    plt.title("BTC Relative Strength Index (RSI)")
    plt.xlabel("Date")
    plt.ylabel("RSI Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("btc_rsi.png")
    plt.close()

def plot_volume(df):
    plt.figure(figsize=(10, 6))
    if len(df) > 200:  # fallback to line plot for large datasets
        plt.plot(df.index, df["Volume"].values, color="purple", alpha=0.6, label="Volume")
        plt.legend()
    else:
        plt.bar(df.index.astype(str), df["Volume"].values, color="purple", alpha=0.6)
    plt.title("BTC Trading Volume")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("btc_volume.png")
    plt.close()

# ===============================
# Model training and prediction
# ===============================
def prepare_features(df):
    df = df.dropna().copy()
    features = df[["SMA_20", "SMA_50", "RSI"]]
    target = df["Close"].shift(-5)  # predict 5 days ahead
    features = features[:-5]
    target = target[:-5]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    return X_scaled, target, scaler

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_future(df, model, scaler):
    last_row = df.iloc[-1][["SMA_20", "SMA_50", "RSI"]].values.reshape(1, -1)
    last_scaled = scaler.transform(last_row)
    prediction = model.predict(last_scaled)
    return prediction[0]

# ===============================
# Main execution
# ===============================
def train_and_predict():
    df = load_data()
    df = add_indicators(df)

    # Generate plots
    plot_price(df)
    plot_rsi(df)
    plot_volume(df)

    # Prepare ML data
    X, y, scaler = prepare_features(df)
    model = train_model(X, y)

    # Make prediction
    future_price = predict_future(df, model, scaler)
    print(f"Predicted BTC price (5-day ahead): {future_price:.2f} USD")

if __name__ == "__main__":
    train_and_predict()
