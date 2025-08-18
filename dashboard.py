import os
from datetime import datetime
import pandas as pd

os.makedirs("docs", exist_ok=True)

def create_dashboard():
    # Try to load predictions
    try:
        preds = pd.read_csv("docs/btc_predictions.csv")
        pred_table = preds.to_html(index=False, classes="table")
    except:
        pred_table = "<p>No prediction data available</p>"
    
    # Try to load report
    try:
        with open("docs/btc_report.txt") as f:
            report = "<pre>" + f.read() + "</pre>"
    except:
        report = "<p>No report available</p>"
    
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>BTC Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        .chart {{ margin: 20px 0; }}
        .table {{ width: 100%; border-collapse: collapse; }}
        .table th, .table td {{ padding: 8px; border: 1px solid #ddd; }}
        .table th {{ background: #f2f2f2; }}
        pre {{ background: #f5f5f5; padding: 15px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Bitcoin Analysis Dashboard</h1>
        <p>Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
        
        <div class="chart">
            <h2>Price Prediction</h2>
            <img src="btc_prediction.png" width="100%">
            {pred_table}
        </div>
        
        <div class="chart">
            <h2>Technical Indicators</h2>
            <img src="btc_rsi.png" width="49%">
            <img src="btc_macd.png" width="49%">
            <img src="btc_volume.png" width="49%">
        </div>
        
        <div class="chart">
            <h2>Analysis Report</h2>
            {report}
        </div>
    </div>
</body>
</html>
    """
    
    with open("docs/index.html", "w") as f:
        f.write(html)
    print("✅ Dashboard created at docs/index.html")

if __name__ == "__main__":
    create_dashboard()
