# train_and_predict.py
# Fetch BTC-USD (since 2014), compute indicators, train XGB model, and save artifacts.

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import ta
import joblib
from xgboost import XGBRegressor


START_DATE = "2014-01-01"
HORIZON_DAYS = 5
DATASET_PATH = "btc_dataset.csv"
MODEL_PATH = "btc_xgb_5day_model_latest.joblib"
PREDICTIONS_PATH = "btc_predictions.csv"


def fetch_btc_data(start_date: str = START_DATE) -> pd.DataFrame:
    df = yf.download("BTC-USD", start=start_date, interval="1d", progress=False)
    if df.empty:
        raise RuntimeError("Yahoo Finance returned an empty dataframe. Try again later.")
    df = df.reset_index()  # keep Date column
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # --- Trend ---
    d["sma_20"] = d["Close"].rolling(20).mean()
    d["sma_50"] = d["Close"].rolling(50).mean()
    d["sma_200"] = d["Close"].rolling(200).mean()

    d["ema_12"] = ta.trend.EMAIndicator(d["Close"], window=12).ema_indicator()
    d["ema_26"] = ta.trend.EMAIndicator(d["Close"], window=26).ema_indicator()

    macd = ta.trend.MACD(d["Close"], window_slow=26, window_fast=12, window_sign=9)
    d["macd"] = macd.macd()
    d["macd_signal"] = macd.macd_signal()
    d["macd_hist"] = macd.macd_diff()

    d["adx"] = ta.trend.ADXIndicator(d["High"], d["Low"], d["Close"], window=14).adx()

    # --- Momentum ---
    d["rsi"] = ta.momentum.RSIIndicator(d["Close"], window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(d["High"], d["Low"], d["Close"], window=14, smooth_window=3)
    d["stoch_k"] = stoch.stoch()
    d["stoch_d"] = stoch.stoch_signal()
    d["williams_r"] = ta.momentum.WilliamsRIndicator(d["High"], d["Low"], d["Close"], lbp=14).williams_r()

    # --- Volatility ---
    bb = ta.volatility.BollingerBands(d["Close"], window=20, window_dev=2.0)
    d["bb_upper"] = bb.bollinger_hband()
    d["bb_middle"] = bb.bollinger_mavg()
    d["bb_lower"] = bb.bollinger_lband()
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_middle"]

    atr = ta.volatility.AverageTrueRange(d["High"], d["Low"], d["Close"], window=14)
    d["atr"] = atr.average_true_range()
    d["atr_pct"] = d["atr"] / d["Close"]
    # Normalized position inside the Bollinger envelope (~ -1 .. +1)
    d["bb_pos"] = (d["Close"] - d["bb_middle"]) / (0.5 * (d["bb_upper"] - d["bb_lower"]))

    # --- Volume ---
    d["obv"] = ta.volume.OnBalanceVolumeIndicator(d["Close"], d["Volume"]).on_balance_volume()
    d["mfi"] = ta.volume.MFIIndicator(d["High"], d["Low"], d["Close"], d["Volume"], window=14).money_flow_index()

    return d


def build_target(df: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    d = df.copy()
    d["target"] = d["Close"].shift(-horizon)
    return d


def split_features_target(df: pd.DataFrame):
    # Drop columns that are not features
    drop_cols = ["Date", "Adj Close", "target"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    y = df["target"]
    # Remove any remaining non-numeric columns just in case
    X = X.select_dtypes(include="number")
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    # Drop rows with missing values (after indicator/shift creation)
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X_train = X[mask]
    y_train = y[mask]

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def export_recent_predictions(df_full: pd.DataFrame, model: XGBRegressor, out_path=PREDICTIONS_PATH):
    # Build features aligned with df_full
    X_full, _ = split_features_target(df_full)
    # Predict for all rows where features are complete
    mask = ~X_full.isna().any(axis=1)
    preds = pd.Series(index=df_full.index, dtype="float64")
    preds.loc[mask] = model.predict(X_full[mask])

    # Target date for each prediction = Date + horizon
    df_preds = pd.DataFrame({
        "asof_date": pd.to_datetime(df_full["Date"]),
        "predicted_close_t_plus_5d": preds,
        "target_date": pd.to_datetime(df_full["Date"]) + timedelta(days=HORIZON_DAYS),
        "close_asof": df_full["Close"]
    })
    # Keep the last 120 predictions for a light file
    df_preds = df_preds.dropna().tail(120).reset_index(drop=True)
    df_preds.to_csv(out_path, index=False)


def main():
    print("📥 Fetching BTC-USD from Yahoo Finance (daily, since 2014-01-01)…")
    raw = fetch_btc_data(START_DATE)

    print("🧮 Adding technical indicators…")
    feats = add_indicators(raw)

    print("🎯 Building 5-day ahead target…")
    data = build_target(feats, HORIZON_DAYS)

    # Drop early rows with NaNs from indicators
    data = data.dropna().reset_index(drop=True)

    print("🔧 Preparing features and labels…")
    X, y = split_features_target(data)

    print("🤖 Training XGBoost model…")
    model = train_model(X, y)

    print(f"💾 Saving dataset → {DATASET_PATH}")
    data.to_csv(DATASET_PATH, index=False)

    print(f"💾 Saving model → {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)

    print(f"📝 Exporting recent predictions → {PREDICTIONS_PATH}")
    export_recent_predictions(data, model, PREDICTIONS_PATH)

    # Final sanity check: show latest prediction
    latest_pred = float(model.predict(X.tail(1))[0])
    last_close = float(data["Close"].iloc[-1])
    print(f"✅ Latest 5-day ahead prediction: {latest_pred:,.2f} USD (last close: {last_close:,.2f} USD)")


if __name__ == "__main__":
    main()
