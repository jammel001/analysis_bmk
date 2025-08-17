import shutil
import os
from datetime import datetime
import pandas as pd

# === SETTINGS ===
PREDICTIONS_FILE = "btc_predictions.csv"   # we’ll save predictions into this file from train_and_predict.py
REPORT_FILE = "btc_report.pdf"
DOCS_DIR = "docs"
DASHBOARD_FILE = os.path.join(DOCS_DIR, "btc_dashboard.html")

# === Ensure /docs exists ===
os.makedirs(DOCS_DIR, exist_ok=True)

# === Copy the PDF report into /docs ===
try:
    shutil.copy(REPORT_FILE, os.path.join(DOCS_DIR, REPORT_FILE))
    print("✅ Report copied to docs/")
except Exception as e:
    print(f"⚠️ Could not copy report: {e}")

# === Load Predictions ===
pred_table_html = "<p><i>No predictions available</i></p>"
if os.path.exists(PREDICTIONS_FILE):
    try:
        preds = pd.read_csv(PREDICTIONS_FILE)
        pred_table_html = preds.to_html(index=False, border=0, classes="pred-table")
        print("✅ Predictions loaded for dashboard")
    except Exception as e:
        print(f"⚠️ Could not load predictions file: {e}")

# === Timestamp ===
last_updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

# === Dashboard HTML ===
dashboard_html = f"""
<html>
<head>
    <title>Bitcoin Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
        h1 {{ color: #333; }}
        .chart {{ margin-bottom: 30px; }}
        .pred-table {{
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .pred-table th, .pred-table td {{
            border: 1px solid #ccc;
            padding: 8px 12px;
            text-align: center;
        }}
        .pred-table th {{
            background: #007BFF;
            color: white;
        }}
        a.download {{
            display: inline-block;
            margin-top: 20px;
            padding: 10px 15px;
            background: #007BFF;
            color: white;
            text-decoration: none;
            border-radius: 5px;
        }}
        a.download:hover {{ background: #0056b3; }}
        .timestamp {{ color: #555; font-size: 14px; margin-top: 30px; }}
    </style>
</head>
<body>
    <h1>📊 Bitcoin Daily Dashboard</h1>
    <p>This dashboard updates automatically every day with the latest Bitcoin data, technical indicators, and predictions.</p>

    <div class="chart">
        <h2>Prediction (Next 5 days)</h2>
        <img src="../btc_prediction.png" width="700">
        <h3>Predicted Prices</h3>
        {pred_table_html}
    </div>
    
    <div class="chart">
        <h2>RSI (Relative Strength Index)</h2>
        <img src="../btc_rsi.png" width="700">
    </div>
    
    <div class="chart">
        <h2>MACD</h2>
        <img src="../btc_macd.png" width="700">
    </div>
    
    <div class="chart">
        <h2>Volume</h2>
        <img src="../btc_volume.png" width="700">
    </div>

    <h2>📄 Report</h2>
    <p>You can download the full BTC analysis report (PDF) with all charts and metrics:</p>
    <a href="btc_report.pdf" class="download">⬇️ Download Report</a>

    <div class="timestamp">⏰ Last updated: {last_updated}</div>
</body>
</html>
"""

# === Save dashboard in /docs ===
with open(DASHBOARD_FILE, "w", encoding="utf-8") as f:
    f.write(dashboard_html)

print(f"✅ Dashboard updated: {DASHBOARD_FILE}")
