import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import datetime as dt
import numpy as np

# --- Indicator Functions ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)  # avoid division by zero
    return 100 - (100 / (1 + rs))

def compute_macd(series, short=12, long=26, signal=9):
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def add_indicators(df):
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["MACD"], df["Signal_Line"] = compute_macd(df["Close"])
    return df

# --- Load model prediction ---
def get_latest_prediction():
    # Fetch fresh BTC data
    end = dt.datetime.today()
    start = end - dt.timedelta(days=365*5)
    print("📥 Downloading fresh BTC data...")
    df = yf.download("BTC-USD", start=start, end=end)
    print(f"✅ Data downloaded: {df.shape}")

    df = add_indicators(df)

    # Prepare features
    df = df.dropna()
    X = df[["RSI", "MACD", "Signal_Line"]]

    # Load trained model
    model = joblib.load("btc_xgb_5day_model_latest.joblib")

    # Predict
    df["Prediction"] = model.predict(X)
    latest = df.iloc[-1]
    return latest

# --- Report Generator ---
def generate_report():
    latest = get_latest_prediction()

    report = f"""
    📊 BTC Prediction Report
    ------------------------
    Date: {latest.name.date()}
    Close Price: {latest['Close']:.2f}
    RSI: {latest['RSI']:.2f}
    MACD: {latest['MACD']:.2f}
    Signal Line: {latest['Signal_Line']:.2f}
    🔮 Predicted Price (5-day): {latest['Prediction']:.2f}
    """

    print(report)

    # Save to file
    with open("btc_report.txt", "w") as f:
        f.write(report)

    print("✅ Report saved as btc_report.txt")

# --- Run ---
if __name__ == "__main__":
    generate_report()
