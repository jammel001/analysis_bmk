from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pandas as pd

# === Settings ===
REPORT_FILE = "btc_report.pdf"
PREDICTIONS_FILE = "btc_predictions.csv"

# Charts generated earlier
charts = [
    ("Bitcoin 5-Day Price Prediction", "btc_prediction.png"),
    ("Relative Strength Index (RSI)", "btc_rsi.png"),
    ("MACD Indicator", "btc_macd.png"),
    ("Trading Volume", "btc_volume.png"),
]

# === Create PDF ===
def generate_report():
    doc = SimpleDocTemplate(REPORT_FILE, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("📊 Bitcoin Price Prediction Report", styles['Title']))
    story.append(Spacer(1, 12))

    # Intro
    story.append(Paragraph(
        "This report provides technical analysis of Bitcoin (BTC-USD) using "
        "machine learning and technical indicators. The model predicts the next 5 days of BTC price trends.",
        styles['BodyText']
    ))
    story.append(Spacer(1, 12))

    # Load Predictions
    try:
        predictions = pd.read_csv(PREDICTIONS_FILE)
        story.append(Paragraph("📅 Next 5-Day BTC Price Forecast", styles['Heading2']))

        # Convert to table
        data = [["Date", "Predicted BTC Price (USD)"]]
        for _, row in predictions.iterrows():
            data.append([row["Date"], f"${row['Predicted_BTC_Price']:.2f}"])

        table = Table(data, colWidths=[200, 200])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#003366")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,0), 8),
            ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.25, colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))
    except Exception as e:
        story.append(Paragraph(f"⚠️ Could not load predictions: {e}", styles['BodyText']))
        story.append(Spacer(1, 20))

    # Add Charts
    story.append(Paragraph("📉 Technical Analysis Charts", styles['Heading2']))
    for title, path in charts:
        try:
            story.append(Paragraph(title, styles['Heading3']))
            story.append(Image(path, width=400, height=200))
            story.append(Spacer(1, 12))
        except Exception as e:
            story.append(Paragraph(f"⚠️ Could not load chart {path}: {e}", styles['BodyText']))

    # Build PDF
    doc.build(story)
    print(f"✅ Report generated and saved as {REPORT_FILE}")

if __name__ == "__main__":
    generate_report()
