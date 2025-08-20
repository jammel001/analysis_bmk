# run_analysis.py - Automated Bitcoin Analysis Workflow (GitHub-structured)
import os
import subprocess
import argparse

# Paths
SRC_DIR = "src"
DATA_DIR = "data"
REPORTS_DIR = "reports"
DOCS_DIR = "docs"

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")

    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ Error in {description}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Exception in {description}: {e}")
        return False

def install_requirements():
    """Install required packages"""
    requirements = [
        "yfinance", "pandas", "numpy",
        "matplotlib", "scikit-learn", "joblib"
    ]
    print("📦 Installing required packages...")
    for package in requirements:
        run_command(f"pip install {package}", f"Installing {package}")

def full_analysis():
    """Run complete Bitcoin analysis workflow"""
    print("""
    🚀 ENHANCED BITCOIN ANALYSIS SYSTEM
    ====================================
    1. Train ML models
    2. Generate predictions
    3. Create visualizations
    4. Save reports and dashboard
    """)

    # Ensure folders exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)

    # Step 1: Train models
    if not run_command(f"python {SRC_DIR}/train_predict.py",
                       "Training ML Models"):
        return False

    # Step 2: Generate daily predictions
    if not run_command(f"python {SRC_DIR}/daily_tracker.py",
                       "Generating Daily Predictions"):
        return False

    # Step 3: Create dashboard
    if not run_command(f"python {SRC_DIR}/dashboard.py",
                       "Creating Dashboard"):
        return False

    print(f"""
    🎉 ANALYSIS COMPLETE!
    =====================

    📊 Files Generated:
    ├── {DOCS_DIR}/btc_dashboard.html
    ├── {DATA_DIR}/btc_predictions.csv
    ├── {REPORTS_DIR}/btc_report.txt
    ├── {REPORTS_DIR}/btc_report.pdf
    ├── {DOCS_DIR}/btc_prediction.png
    ├── {DOCS_DIR}/btc_macd.png
    ├── {DOCS_DIR}/btc_rsi.png
    ├── {DOCS_DIR}/btc_volume.png

    🌐 Dashboard URL: https://YOUR_USERNAME.github.io/YOUR_REPO/
    """)
    return True

def daily_update():
    """Run daily update only (predictions + dashboard)"""
    print("🔄 Running Daily Bitcoin Analysis Update...")
    ok1 = run_command(f"python {SRC_DIR}/daily_tracker.py", "Daily Predictions")
    ok2 = run_command(f"python {SRC_DIR}/dashboard.py", "Updating Dashboard")
    return ok1 and ok2

def main():
    parser = argparse.ArgumentParser(description="Bitcoin Analysis Automation")
    parser.add_argument("--mode", choices=["full", "daily"], default="full",
                        help="Analysis mode")
    parser.add_argument("--install", action="store_true", help="Install requirements")
    args = parser.parse_args()

    if args.install:
        install_requirements()
    if args.mode == "full":
        full_analysis()
    elif args.mode == "daily":
        daily_update()

if __name__ == "__main__":
    main()
