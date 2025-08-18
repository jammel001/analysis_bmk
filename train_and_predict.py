import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import datetime as dt
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Create docs directory if not exists
os.makedirs("docs", exist_ok=True)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def train_and_predict():
    print("📥 Downloading BTC-USD data...")
    end = dt.datetime.today()
    start = end - dt.timedelta(days=5*365)
    df = yf.download("BTC-USD", start=start, end=end)
    
    if df.empty:
        raise ValueError("No data downloaded!")
    
    print("✅ Data downloaded")
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['Signal'] = compute_macd(df['Close'])
    df = df.dropna()

    X = df[['RSI', 'MACD', 'Signal']]
    y = df['Close']
    
    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, "btc_xgb_5day_model_latest.joblib")
    
    # Save all plots to docs/
    plt.figure(figsize=(10,6))
    plt.plot(df.index, df['Close'], label='Actual')
    plt.title('BTC Price Prediction')
    plt.grid(True)
    plt.savefig('docs/btc_prediction.png')
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['RSI'], label='RSI', color='purple')
    plt.axhline(70, linestyle='--', color='red')
    plt.axhline(30, linestyle='--', color='green')
    plt.title('BTC RSI')
    plt.grid(True)
    plt.savefig('docs/btc_rsi.png')
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['MACD'], label='MACD')
    plt.plot(df.index, df['Signal'], label='Signal')
    plt.title('BTC MACD')
    plt.grid(True)
    plt.savefig('docs/btc_macd.png')
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(df.index, df['Volume'], label='Volume', color='gray')
    plt.title('BTC Volume')
    plt.grid(True)
    plt.savefig('docs/btc_volume.png')
    plt.close()

    print("✅ All charts saved to docs/")

if __name__ == "__main__":
    train_and_predict()
