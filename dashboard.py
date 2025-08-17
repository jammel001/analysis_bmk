# dashboard.py
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Load dataset + predictions
df = pd.read_csv("btc_dataset.csv", parse_dates=["Date"])
pred_df = pd.read_csv("btc_predictions.csv", parse_dates=["Date"])

# Merge predictions into main dataset
df = df.merge(pred_df, on="Date", how="left")

# Create figure with subplots
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.45, 0.15, 0.2, 0.2],
    subplot_titles=("Bitcoin Price + Predictions + Bollinger Bands", "Volume", "RSI", "MACD")
)

# --- 1. Bitcoin Price + Predictions + Bollinger Bands ---
fig.add_trace(
    go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="BTC Close"),
    row=1, col=1
)

# Bollinger Bands (Upper & Lower)
fig.add_trace(
    go.Scatter(x=df["Date"], y=df["BB_upper"], mode="lines",
               line=dict(width=1, color="rgba(255,0,0,0.4)"),
               name="BB Upper", showlegend=True),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=df["Date"], y=df["BB_lower"], mode="lines",
               line=dict(width=1, color="rgba(0,0,255,0.4)"),
               name="BB Lower", fill="tonexty", fillcolor="rgba(173,216,230,0.2)"),
    row=1, col=1
)

# Add predicted future prices
preds = df.dropna(subset=["Predicted_Close"])
if not preds.empty:
    fig.add_trace(
        go.Scatter(x=preds["Date"], y=preds["Predicted_Close"],
                   mode="lines+markers", name="Predicted 5-Day Ahead"),
        row=1, col=1
    )

# --- 2. Trading Volume ---
fig.add_trace(
    go.Bar(x=df["Date"], y=df["Volume"], name="Volume", marker_color="rgba(0,0,150,0.5)"),
    row=2, col=1
)

# --- 3. RSI ---
fig.add_trace(
    go.Scatter(x=df["Date"], y=df["RSI"], mode="lines", name="RSI"),
    row=3, col=1
)
fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)

# --- 4. MACD ---
fig.add_trace(
    go.Scatter(x=df["Date"], y=df["MACD"], mode="lines", name="MACD"),
    row=4, col=1
)
fig.add_trace(
    go.Scatter(x=df["Date"], y=df["Signal"], mode="lines", name="Signal"),
    row=4, col=1
)

# Layout
fig.update_layout(
    height=1100,
    title="Bitcoin Daily Analysis Dashboard",
    template="plotly_white",
    legend=dict(orientation="h", y=-0.3)
)

# Save dashboard
fig.write_html("btc_dashboard.html")
print("✅ Dashboard saved as btc_dashboard.html")
