# generate_report.py
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(rmse, pred_file="btc_predictions.csv"):
    # Load styles
    styles = getSampleStyleSheet()
    story = []

    # Create PDF
    doc = SimpleDocTemplate("btc_report.pdf", pagesize=letter)

    # Title
    story.append(Paragraph("📊 Bitcoin Price Prediction Report", styles['Title']))
    story.append(Spacer(1, 20))

    # Model performance
    story.append(Paragraph(f"✅ Model RMSE: {rmse:.2f}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "This report includes Bitcoin's recent performance with technical indicators "
        "(RSI, MACD, Volume) and a 5-day forward prediction generated using XGBoost.",
        styles['Normal']
    ))
    story.append(Spacer(1, 20))

    # Add prediction chart
    story.append(Paragraph("📈 BTC Price Forecast (5 days ahead)", styles['Heading2']))
    story.append(Image("btc_prediction.png", width=480, height=240))
    story.append(Spacer(1, 20))

    # RSI chart
    story.append(Paragraph("📉 RSI (Relative Strength Index)", styles['Heading2']))
    story.append(Image("btc_rsi.png", width=480, height=240))
    story.append(Spacer(1, 20))

    # MACD chart
    story.append(Paragraph("📊 MACD (Moving Average Convergence Divergence)", styles['Heading2']))
    story.append(Image("btc_macd.png", width=480, height=240))
    story.append(Spacer(1, 20))

    # Volume chart
    story.append(Paragraph("📦 Trading Volume", styles['Heading2']))
    story.append(Image("btc_volume.png", width=480, height=240))
    story.append(Spacer(1, 20))

    # Save PDF
    doc.build(story)
    print("📄 Report generated: btc_report.pdf")
