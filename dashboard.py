# dashboard.py
import pandas as pd
from jinja2 import Template

# =========================
# 1. Load Predictions
# =========================
pred_df = pd.read_csv("btc_predictions.csv")

# =========================
# 2. HTML Template
# =========================
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Bitcoin Price Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f5f5f5; text-align: center; }
    h1 { color: #333; }
    table { margin: auto; border-collapse: collapse; width: 50%; background: white; }
    th, td { border: 1px solid #ccc; padding: 10px; text-align: center; }
    th { background: #333; color: white; }
    img { margin: 15px; max-width: 80%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .footer { margin-top: 20px; font-size: 14px; color: #777; }
  </style>
</head>
<body>
  <h1>📈 Bitcoin Price Prediction Dashboard</h1>
  <p>Latest 5-day forecast (auto-updated daily)</p>

  <table>
    <tr>
      <th>Date</th>
      <th>Predicted Close (USD)</th>
    </tr>
    {% for row in predictions %}
    <tr>
      <td>{{ row.Date }}</td>
      <td>{{ "%.2f"|format(row.Predicted_Close) }}</td>
    </tr>
    {% endfor %}
  </table>

  <h2>📊 Prediction Chart</h2>
  <img src="../btc_prediction.png" alt="Prediction Chart">

  <h2>📊 Technical Indicators</h2>
  <img src="../btc_rsi.png" alt="RSI">
  <img src="../btc_macd.png" alt="MACD">
  <img src="../btc_volume.png" alt="Volume">

  <div class="footer">
    <p>Auto-generated on {{ generated_date }}</p>
  </div>
</body>
</html>
"""

# =========================
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
