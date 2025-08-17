import pandas as pd
import yfinance as yf
import datetime
import joblib

# -------------------------------
# Helper Functions
# -------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)  # avoid div by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_indicators(df):
    df["RSI"] = compute_rsi(df["Close"], 14)
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    return df

def get_latest_prediction():
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=5 * 365)

    print("📥 Downloading fresh BTC data...")
    df = yf.download("BTC-USD", start=start, end=end)
    print(f"✅ Data downloaded: {df.shape}")

    df = add_indicators(df)
    latest = df.iloc[-1]

    # Load model if exists
    try:
        model = joblib.load("btc_xgb_5day_model_latest.joblib")
        features = ["Close", "RSI", "SMA_20", "EMA_20"]
        pred_price = model.predict([latest[features].values])[0]
    except Exception as e:
        print("⚠️ No model found, skipping prediction:", e)
        pred_price = None

    return latest, pred_price

# -------------------------------
# Generate Report
# -------------------------------
def generate_report():
    latest, pred_price = get_latest_prediction()

    # Save CSV of predictions
    csv_data = {
        "Date": [latest.name.strftime("%Y-%m-%d")],
        "Close": [latest["Close"]],
        "RSI": [latest["RSI"]],
        "SMA_20": [latest["SMA_20"]],
        "EMA_20": [latest["EMA_20"]],
        "Predicted_Close": [pred_price],
    }
    pd.DataFrame(csv_data).to_csv("btc_predictions.csv", index=False)
    print("✅ Predictions saved as btc_predictions.csv")

    # Save text report
    report = f"""
📊 Bitcoin Report ({latest.name.strftime("%Y-%m-%d")})

- Latest Close Price: ${latest['Close']:.2f}
- RSI (14): {latest['RSI']:.2f}
- SMA(20): ${latest['SMA_20']:.2f}
- EMA(20): ${latest['EMA_20']:.2f}
- Predicted Close (5-day XGB model): {"$"+str(round(pred_price,2)) if pred_price else "N/A"}
"""
    with open("btc_report.txt", "w") as f:
        f.write(report.strip())
    print("✅ Report saved as btc_report.txt")


if __name__ == "__main__":
    generate_report()
