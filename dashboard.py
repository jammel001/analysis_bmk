# dashboard.py
import pandas as pd
from jinja2 import Template

# =========================
# 1. Load Predictions
# =========================
pred_df = pd.read_csv("btc_predictions.csv")

from datetime import datetime
import shutil

# Copy PDF report into docs
try:
    shutil.copy("btc_report.pdf", "docs/btc_report.pdf")
except:
    pass

# Add last update timestamp
last_updated = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

dashboard_html = f"""
<html>
<head>
    <title>Bitcoin Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
        h1 {{ color: #333; }}
        .chart {{ margin-bottom: 30px; }}
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
        .timestamp {{ margin-top: 20px; color: #555; font-size: 14px; }}
    </style>
</head>
<body>
    <h1>📊 Bitcoin Daily Dashboard</h1>
    <p>This dashboard updates automatically every day with the latest Bitcoin data, technical indicators, and predictions.</p>

    <div class="timestamp">Last updated: {last_updated}</div>
    
    <div class="chart">
        <h2>Prediction (Next 5 days)</h2>
        <img src="../btc_prediction.png" width="700">
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
</body>
</html>
"""

with open("docs/btc_dashboard.html", "w") as f:
    f.write(dashboard_html)

print("✅ Dashboard updated with timestamp")
 =========================
# 3. Render HTML
# =========================
template = Template(html_template)
html_content = template.render(
    predictions=pred_df.to_dict(orient="records"),
    generated_date=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
)

# =========================
# 4. Save HTML in /docs
# =========================
with open("docs/btc_dashboard.html", "w") as f:
    f.write(html_content)

print("✅ Dashboard updated at docs/btc_dashboard.html")
