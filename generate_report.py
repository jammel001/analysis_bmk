from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import joblib
import pandas as pd
import os

# ==============================
# Load latest prediction from model
# ==============================
def get_latest_prediction():
    # Load last row of btc_data.csv for features
    df = pd.read_csv("btc_data.csv")
    model = joblib.load("btc_xgb_5day_model_latest.joblib")

    # Feature Engineering (must match train script)
    df["Return"] = df["Close"].pct_change()
    df["RSI"] = compute_rsi(df["Close"])
    df["MACD"], df["Signal"] = compute_macd(df["Close"])
    df["Volume_Change"] = df["Volume"].pct_change()
    df.dropna(inplace=True)

    latest_features = df[["Close", "Return", "RSI", "MACD", "Signal", "Volume_Change"]].iloc[-1:].values
    prediction = model.predict(latest_features)[0]
    return prediction

# ==============================
# RSI and MACD helpers
# ==============================
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
# Report generator
# ==============================
def generate_report():
    prediction = get_latest_prediction()

    doc = SimpleDocTemplate("btc_analysis_report.pdf", pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Bitcoin Analysis Report", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Prediction text
    elements.append(Paragraph(
        f"<b>Predicted BTC Price (5-day ahead):</b> ${prediction:,.2f} USD",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 12))

    # Add plots if available
    plots = [
        ("btc_prediction.png", "BTC Price Prediction"),
        ("btc_rsi.png", "Relative Strength Index (RSI)"),
        ("btc_macd.png", "MACD Indicator"),
        ("btc_volume.png", "BTC Trading Volume"),
    ]

    for plot_file, caption in plots:
        if os.path.exists(plot_file):
            elements.append(Paragraph(caption, styles["Heading2"]))
            elements.append(Image(plot_file, width=400, height=200))
            elements.append(Spacer(1, 12))

    doc.build(elements)
    print("✅ Report generated: btc_analysis_report.pdf")

# ==============================
# Run
# ==============================
if __name__ == "__main__":
    generate_report()
