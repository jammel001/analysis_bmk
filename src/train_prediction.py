# enhanced_train_predict.py
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import datetime as dt
import os
import warnings
warnings.filterwarnings("ignore")

os.makedirs("docs", exist_ok=True)

def compute_technical_indicators(df):
    """Compute comprehensive technical indicators"""
    
    # RSI
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    # MACD
    def macd(series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast).mean()
        exp2 = series.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    # Bollinger Bands
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

def create_prediction_targets(df, days_ahead=5):
    """Create multiple day-ahead prediction targets"""
    targets = {}
    for day in range(1, days_ahead + 1):
        targets[f'Close_+{day}d'] = df['Close'].shift(-day)
        targets[f'Direction_+{day}d'] = (df['Close'].shift(-day) > df['Close']).astype(int)
    
    for key, value in targets.items():
        df[key] = value
    
    return df

def prepare_features():
    """Download data and prepare features"""
    print("📥 Downloading BTC-USD data from 2014...")
    
    # Download from 2014 to present
    start = dt.datetime(2014, 1, 1)
    end = dt.datetime.now()
    df = yf.download("BTC-USD", start=start, end=end)
    
    if df.empty:
        raise ValueError("No data downloaded!")
    
    print(f"✅ Downloaded {len(df)} days of data from {start.date()} to {end.date()}")
    
    # Add technical indicators
    df = compute_technical_indicators(df)
    
    # Create prediction targets
    df = create_prediction_targets(df, days_ahead=5)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    print(f"✅ Processed data with {len(df)} valid rows")
    return df

def train_models(df):
    """Train models for 1-5 day predictions"""
    
    # Select features for training
    feature_columns = [
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Width', 'BB_Position', 'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26', 'Price_Change', 'Price_Change_5d',
        'Volatility_10d', 'High_Low_Ratio', 'Volume_Change', 'Volume_Ratio',
        'Momentum_5', 'Momentum_10', 'Close_Position'
    ]
    
    X = df[feature_columns][:-5]  # Remove last 5 rows (no targets)
    
    models = {}
    scalers = {}
    metrics = {}
    
    # Train separate models for each prediction horizon
    for day in range(1, 6):
        print(f"🤖 Training model for {day}-day prediction...")
        
        y = df[f'Close_+{day}d'][:-5]
        
        # Split data (last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        models[f'model_{day}d'] = model
        scalers[f'scaler_{day}d'] = scaler
        metrics[f'{day}d'] = {'MAE': mae, 'R2': r2}
        
        print(f"   ✅ {day}-day model: MAE=${mae:.2f}, R²={r2:.3f}")
    
    # Save models and scalers
    joblib.dump(models, "docs/btc_models.joblib")
    joblib.dump(scalers, "docs/btc_scalers.joblib")
    joblib.dump(feature_columns, "docs/feature_columns.joblib")
    
    return models, scalers, metrics, feature_columns

def create_visualizations(df, models, scalers, feature_columns):
    """Create comprehensive visualizations"""
    
    # 1. Price chart with moving averages
    plt.figure(figsize=(15, 8))
    plt.subplot(2, 2, 1)
    recent_data = df.tail(365)  # Last year
    plt.plot(recent_data.index, recent_data['Close'], label='Close Price', linewidth=2)
    plt.plot(recent_data.index, recent_data['SMA_20'], label='SMA 20', alpha=0.7)
    plt.plot(recent_data.index, recent_data['SMA_50'], label='SMA 50', alpha=0.7)
    plt.title('BTC Price with Moving Averages (Last Year)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. RSI
    plt.subplot(2, 2, 2)
    plt.plot(recent_data.index, recent_data['RSI'], color='purple', linewidth=2)
    plt.axhline(70, color='red', linestyle='--', alpha=0.7, label='Overbought')
    plt.axhline(30, color='green', linestyle='--', alpha=0.7, label='Oversold')
    plt.title('RSI (Relative Strength Index)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Volume analysis
    plt.subplot(2, 2, 3)
    plt.bar(recent_data.index, recent_data['Volume'], alpha=0.6, color='gray')
    plt.plot(recent_data.index, recent_data['Volume_SMA'], color='red', linewidth=2, label='Volume SMA')
    plt.title('Trading Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. MACD
    plt.subplot(2, 2, 4)
    plt.plot(recent_data.index, recent_data['MACD'], label='MACD', linewidth=2)
    plt.plot(recent_data.index, recent_data['MACD_Signal'], label='Signal', linewidth=2)
    plt.bar(recent_data.index, recent_data['MACD_Hist'], alpha=0.3, label='Histogram')
    plt.title('MACD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('docs/btc_technical_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create prediction visualization
    latest_data = df.tail(100)  # Last 100 days
    X_recent = latest_data[feature_columns]
    
    predictions_chart = plt.figure(figsize=(12, 6))
    
    # Plot recent prices
    plt.plot(latest_data.index, latest_data['Close'], 
             label='Historical Price', linewidth=2, color='blue')
    
    # Create future predictions
    latest_features = X_recent.iloc[-1:].values
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=5)
    
    predictions = []
    for day in range(1, 6):
        scaler = scalers[f'scaler_{day}d']
        model = models[f'model_{day}d']
        features_scaled = scaler.transform(latest_features)
        pred = model.predict(features_scaled)[0]
        predictions.append(pred)
    
    # Plot predictions
    plt.plot(future_dates, predictions, 
             'ro-', label='5-Day Predictions', linewidth=2, markersize=8)
    
    plt.title('BTC Price: Recent History + 5-Day Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('docs/btc_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Visualizations saved to docs/")

def main():
    print("🚀 Starting Enhanced BTC Prediction System...")
    
    try:
        # Prepare data
        df = prepare_features()
        
        # Train models
        models, scalers, metrics, feature_columns = train_models(df)
        
        # Create visualizations
        create_visualizations(df, models, scalers, feature_columns)
        
        # Save model performance metrics
        with open("docs/model_metrics.txt", "w") as f:
            f.write("BTC Prediction Model Performance\n")
            f.write("=" * 40 + "\n\n")
            for day, metric in metrics.items():
                f.write(f"{day} Prediction:\n")
                f.write(f"  Mean Absolute Error: ${metric['MAE']:.2f}\n")
                f.write(f"  R² Score: {metric['R2']:.3f}\n\n")
        
        print("🎉 Training completed successfully!")
        print(f"📊 Models trained on {len(df)} days of data")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
