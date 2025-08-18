# enhanced_dashboard.py
import os
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt

os.makedirs("docs", exist_ok=True)

def load_prediction_history():
    """Load prediction tracking history"""
    try:
        with open("docs/prediction_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def create_accuracy_chart():
    """Create prediction accuracy visualization"""
    history = load_prediction_history()
    
    if len(history) < 5:
        return "<p>Not enough prediction history yet. Check back after a few days!</p>"
    
    # This would normally create a chart, but for now return a placeholder
    return """
    <div style="background: #f0f8ff; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h4>📈 Prediction Accuracy Tracking</h4>
        <p>Accuracy analysis will appear here once we have more historical data.</p>
        <p><strong>Note:</strong> The model tracks predictions vs actual prices daily to improve accuracy.</p>
    </div>
    """

def create_enhanced_dashboard():
    """Create comprehensive Bitcoin analysis dashboard"""
    
    # Load predictions
    try:
        predictions_df = pd.read_csv("docs/btc_predictions.csv")
        pred_table = predictions_df.to_html(index=False, classes="prediction-table", table_id="predTable")
    except FileNotFoundError:
        pred_table = "<p class='warning'>⚠️ No prediction data available. Run daily_tracker.py first!</p>"
    
    # Load report
    try:
        with open("docs/btc_report.txt", "r") as f:
            report = f.read()
    except FileNotFoundError:
        report = "No analysis report available. Run daily_tracker.py first!"
    
    # Load model metrics
    try:
        with open("docs/model_metrics.txt", "r") as f:
            metrics = f.read()
    except FileNotFoundError:
        metrics = "Model performance metrics not available."
    
    # Get prediction history stats
    history = load_prediction_history()
    history_stats = f"""
    <div class="stats-card">
        <h3>📊 Prediction Statistics</h3>
        <p><strong>Total Prediction Sets:</strong> {len(history)}</p>
        <p><strong>Days Tracked:</strong> {len(history) * 5 if history else 0}</p>
        <p><strong>Last Updated:</strong> {history[-1]['prediction_made_on'] if history else 'Never'}</p>
    </div>
    """
    
    accuracy_chart = create_accuracy_chart()
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Bitcoin Analysis Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            text-align: center;
        }}
        
        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #f39c12, #e74c3c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .header p {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}
        
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            transition: transform 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
        }}
        
        .card h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.4em;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        
        .prediction-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        
        .prediction-table th {{
            background: linear-gradient(45deg, #3498db, #2980b9);
            color: white;
            padding: 12px 8px;
            text-align: left;
            font-weight: 600;
        }}
        
        .prediction-table td {{
            padding: 10px 8px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .prediction-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .prediction-table tr:hover {{
            background: #e3f2fd;
        }}
        
        .stats-card {{
            background: linear-gradient(45deg, #1abc9c, #16a085);
            color: white;
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
        }}
        
        .stats-card h3 {{
            margin-bottom: 15px;
        }}
        
        .chart-container {{
            text-align: center;
            margin: 20px 0;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        .report-section {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 25px;
            border-radius: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            white-space: pre-wrap;
            overflow-x: auto;
        }}
        
        .warning {{
            background: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-weight: bold;
        }}
        
        .success {{
            background: #27ae60;
            color: white;
            padding: 15px;
            border-radius: 8px;
            font-weight: bold;
        }}
        
        .info-box {{
            background: #3498db;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: rgba(255,255,255,0.7);
        }}
        
        @media (max-width: 768px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
            .header h1 {{
                font-size: 2em;
            }}
            .prediction-table {{
                font-size: 0.8em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Enhanced Bitcoin Analysis Dashboard</h1>
            <p>Advanced ML-Powered 5-Day Bitcoin Price Predictions</p>
            <p><strong>Last Updated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="dashboard-grid">
            <div class="card">
                <h2>🎯 5-Day Price Predictions</h2>
                <div class="info-box">
                    <strong>🤖 AI Model Features:</strong><br>
                    • 20+ Technical Indicators<br>
                    • Random Forest Algorithm<br>
                    • Trained on 10+ Years Data<br>
                    • Daily Accuracy Tracking
                </div>
                {pred_table}
            </div>
            
            <div class="card">
                <h2>📊 Prediction Analytics</h2>
                {history_stats}
                {accuracy_chart}
            </div>
        </div>
        
        <div class="card">
            <h2>📈 Technical Analysis Charts</h2>
            <div class="dashboard-grid">
                <div class="chart-container">
                    <h3>Price Predictions vs Historical</h3>
                    <img src="btc_predictions.png" alt="Bitcoin Predictions" 
                         onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                    <p style="display:none; color:#e74c3c;">Chart not available - run enhanced_train_predict.py</p>
                </div>
                <div class="chart-container">
                    <h3>Technical Indicators</h3>
                    <img src="btc_technical_analysis.png" alt="Technical Analysis" 
                         onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                    <p style="display:none; color:#e74c3c;">Chart not available - run enhanced_train_predict.py</p>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h2>📋 Detailed Analysis Report</h2>
            <div class="report-section">{report}</div>
        </div>
        
        <div class="card">
            <h2>🎯 Model Performance Metrics</h2>
            <div class="report-section">{metrics}</div>
        </div>
        
        <div class="footer">
            <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Not financial advice.</p>
            <p>🔄 Dashboard updates automatically when scripts are run</p>
        </div>
    </div>
    
    <script>
        // Add some interactivity
        document.addEventListener('DOMContentLoaded', function() {{
            // Add click-to-highlight for prediction table rows
            const rows = document.querySelectorAll('#predTable tr');
            rows.forEach(row => {{
                row.addEventListener('click', function() {{
                    rows.forEach(r => r.style.background = '');
                    this.style.background = '#fff3cd';
                }});
            }});
            
            // Auto-refresh every 5 minutes
            setTimeout(() => {{
                location.reload();
            }}, 300000);
        }});
    </script>
</body>
</html>
    """
    
    with open("docs/index.html", "w") as f:
        f.write(html)
    
    print("✅ Enhanced dashboard created at docs/index.html")

if __name__ == "__main__":
    create_enhanced_dashboard()
