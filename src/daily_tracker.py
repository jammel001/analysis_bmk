
# daily_tracker.py
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import datetime as dt
import json
import os
from sklearn.preprocessing import StandardScaler

os.makedirs("docs", exist_ok=True)

def load_models():
    """Load trained models and scalers"""
    try:
        models = joblib.load("docs/btc_models.joblib")
        scalers = joblib.load("docs/btc_scalers.joblib")
        feature_columns = joblib.load("docs/feature_columns.joblib")
        return models, scalers, feature_columns
    except FileNotFoundError:
        print("❌ Models not found! Please run enhanced_train_predict.py first")
        return None, None, None

def compute_technical_indicators(df):
    """Same technical indicators as training"""
    
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast).mean()
        exp2 = series.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def bollinger_bands(series, period=20):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        bb_width = (upper - lower) / sma
        bb_position = (series - lower) / (upper - lower)
        return upper, lower, bb_width, bb_position
    
    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # Technical Indicators
    df['RSI'] = rsi(df['Close'])
    df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = macd(df['Close'])
    df['BB_Upper'], df['BB_Lower'], df['BB_Width'], df['BB_Position'] = bollinger_bands(df['Close'])
    
    # Price-based features
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5d'] = df['Close'].pct_change(5)
    df['Volatility_10d'] = df['Price_Change'].rolling(window=10).std()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Momentum indicators
    df['Momentum_5'] = df['Close'] / df['Close'].shift(5)
    df['Momentum_10'] = df['Close'] / df['Close'].shift(10)
    
    # Support/Resistance levels
    df['High_20'] = df['High'].rolling(window=20).max()
    df['Low_20'] = df['Low'].rolling(window=20).min()
    df['Close_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
    
    return df

def get_latest_data():
    """Get latest Bitcoin data"""
    end = dt.datetime.now()
    start = end - dt.timedelta(days=100)  # Get enough data for indicators
    
    df = yf.download("BTC-USD", start=start, end=end)
    df = compute_technical_indicators(df)
    df = df.dropna()
    
    return df

def make_predictions(df, models, scalers, feature_columns):
    """Make 5-day ahead predictions"""
    
    latest_features = df[feature_columns].iloc[-1:].values
    current_price = df['Close'].iloc[-1]
    
    predictions = {}
    prediction_dates = []
    prediction_prices = []
    
    for day in range(1, 6):
        model = models[f'model_{day}d']
        scaler = scalers[f'scaler_{day}d']
        
        features_scaled = scaler.transform(latest_features)
        pred_price = model.predict(features_scaled)[0]
        
        pred_date = (df.index[-1] + pd.Timedelta(days=day)).strftime('%Y-%m-%d')
        predictions[f'Day_{day}'] = {
            'date': pred_date,
            'predicted_price': round(pred_price, 2),
            'change_from_current': round(pred_price - current_price, 2),
            'change_percent': round((pred_price - current_price) / current_price * 100, 2)
        }
        
        prediction_dates.append(pred_date)
        prediction_prices.append(pred_price)
    
    return predictions, current_price

def load_tracking_history():
    """Load historical predictions for tracking accuracy"""
    try:
        with open("docs/prediction_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_tracking_history(history):
    """Save prediction history"""
    with open("docs/prediction_history.json", "w") as f:
        json.dump(history, f, indent=2)

def track_prediction_accuracy():
    """Track how accurate previous predictions were"""
    history = load_tracking_history()
    
    if not history:
        return None
    
    # Get recent actual prices for comparison
    end = dt.datetime.now()
    start = end - dt.timedelta(days=30)
    actual_df = yf.download("BTC-USD", start=start, end=end)
    
    accuracy_report = []
    
    for record in history[-5:]:  # Check last 5 prediction sets
        prediction_date = record['prediction_made_on']
        predictions = record['predictions']
        
        for day_pred in predictions.values():
            target_date = day_pred['date']
            predicted_price = day_pred['predicted_price']
            
            # Find actual price for that date
            try:
                actual_price = actual_df.loc[target_date]['Close']
                error = abs(predicted_price - actual_price)
                error_percent = (error / actual_price) * 100
                
                accuracy_report.append({
                    'prediction_date': prediction_date,
                    'target_date': target_date,
                    'predicted': predicted_price,
                    'actual': actual_price,
                    'error': round(error, 2),
                    'error_percent': round(error_percent, 2)
                })
            except KeyError:
                continue  # Date not found in actual data
    
    return accuracy_report

def generate_daily_report():
    """Generate comprehensive daily prediction report"""
    
    print("🔍 Loading models...")
    models, scalers, feature_columns = load_models()
    
    if models is None:
        return
    
    print("📥 Getting latest Bitcoin data...")
    df = get_latest_data()
    
    print("🎯 Making 5-day predictions...")
    predictions, current_price = make_predictions(df, models, scalers, feature_columns)
    
    print("📊 Tracking prediction accuracy...")
    accuracy_report = track_prediction_accuracy()
    
    # Save current predictions to history
    history = load_tracking_history()
    today_record = {
        'prediction_made_on': dt.datetime.now().strftime('%Y-%m-%d'),
        'current_price': current_price,
        'predictions': predictions
    }
    history.append(today_record)
    save_tracking_history(history)
    
    # Generate CSV report for dashboard
    prediction_data = []
    for day, pred in predictions.items():
        prediction_data.append({
            'Day': day.replace('_', ' '),
            'Date': pred['date'],
            'Predicted Price': f"${pred['predicted_price']:,.2f}",
            'Change $': f"${pred['change_from_current']:+,.2f}",
            'Change %': f"{pred['change_percent']:+.1f}%"
        })
    
    pd.DataFrame(prediction_data).to_csv("docs/btc_predictions.csv", index=False)
    
    # Generate text report
    with open("docs/btc_report.txt", "w") as f:
        f.write(f"Bitcoin Analysis Report - {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Current BTC Price: ${current_price:,.2f}\n")
        f.write(f"Latest RSI: {df['RSI'].iloc[-1]:.1f}\n")
        f.write(f"Latest MACD: {df['MACD'].iloc[-1]:.2f}\n")
        f.write(f"24h Volume: {df['Volume'].iloc[-1]:,.0f}\n\n")
        
        f.write("5-DAY PRICE PREDICTIONS:\n")
        f.write("-" * 30 + "\n")
        for day, pred in predictions.items():
            f.write(f"{day.replace('_', ' ')}: ${pred['predicted_price']:,.2f} ")
            f.write(f"({pred['change_percent']:+.1f}%)\n")
        
        f.write(f"\nTechnical Indicators:\n")
        f.write("-" * 20 + "\n")
        latest = df.iloc[-1]
        
        # RSI Signal
        if latest['RSI'] > 70:
            rsi_signal = "Overbought (Sell Signal)"
        elif latest['RSI'] < 30:
            rsi_signal = "Oversold (Buy Signal)"
        else:
            rsi_signal = "Neutral"
        f.write(f"RSI Signal: {rsi_signal}\n")
        
        # MACD Signal
        if latest['MACD'] > latest['MACD_Signal']:
            macd_signal = "Bullish"
        else:
            macd_signal = "Bearish"
        f.write(f"MACD Signal: {macd_signal}\n")
        
        # Bollinger Bands
        if latest['BB_Position'] > 0.8:
            bb_signal = "Near Upper Band (Resistance)"
        elif latest['BB_Position'] < 0.2:
            bb_signal = "Near Lower Band (Support)"
        else:
            bb_signal = "Within Normal Range"
        f.write(f"Bollinger Bands: {bb_signal}\n")
        
        # Prediction accuracy from history
        if accuracy_report:
            f.write(f"\nRecent Prediction Accuracy:\n")
            f.write("-" * 25 + "\n")
            avg_error = np.mean([r['error_percent'] for r in accuracy_report[-10:]])
            f.write(f"Average Error: {avg_error:.1f}%\n")
            f.write(f"Predictions Tracked: {len(accuracy_report)}\n")
    
    print("✅ Daily report generated successfully!")
    print(f"📊 Current BTC Price: ${current_price:,.2f}")
    print("🔮 5-Day Predictions:")
    for day, pred in predictions.items():
        print(f"   {day}: ${pred['predicted_price']:,.2f} ({pred['change_percent']:+.1f}%)")

if __name__ == "__main__":
    generate_daily_report()

