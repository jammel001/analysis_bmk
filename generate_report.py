import pandas as pd
import yfinance as yf
import datetime
import joblib
import os

os.makedirs("docs", exist_ok=True)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def get_latest_prediction():
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=30)  # Shorter period for report
    df = yf.download("BTC-USD", start=start, end=end)
    
    df['RSI'] = compute_rsi(df['Close'])
    latest = df.iloc[-1]
    
    try:
        model = joblib.load("btc_xgb_5day_model_latest.joblib")
        pred = model.predict([[latest['RSI'], latest['Close']]])[0]
        return latest, float(pred)
    except Exception as e:
        print(f"⚠️ Prediction failed: {e}")
        return latest, None

def generate_report():
    latest, pred = get_latest_prediction()
    report_data = {
        "Date": [latest.name.strftime("%Y-%m-%d")],
        "Close": [latest['Close']],
        "RSI": [latest['RSI']],
        "Prediction": [pred] if pred else ["N/A"]
    }
    
    pd.DataFrame(report_data).to_csv("docs/btc_predictions.csv", index=False)
    
    with open("docs/btc_report.txt", "w") as f:
        f.write(f"BTC Report {datetime.date.today()}\n")
        f.write(f"Close: ${latest['Close']:.2f}\n")
        f.write(f"RSI: {latest['RSI']:.2f}\n")
        f.write(f"5-Day Prediction: ${pred:.2f}" if pred else "Prediction: N/A")
    
    print("✅ Report generated in docs/")

if __name__ == "__main__":
    generate_report()
