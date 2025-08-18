# run_analysis.py - Automated Bitcoin Analysis Workflow
import os
import subprocess
import sys
import datetime as dt
import argparse

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
        "yfinance",
        "pandas", 
        "numpy",
        "matplotlib",
        "scikit-learn",
        "joblib"
    ]
    
    print("📦 Installing required packages...")
    for package in requirements:
        run_command(f"pip install {package}", f"Installing {package}")

def full_analysis():
    """Run complete Bitcoin analysis workflow"""
    print("""
    🚀 ENHANCED BITCOIN ANALYSIS SYSTEM
    ====================================
    This system will:
    1. Download Bitcoin data from 2014 to present
    2. Train ML models with 20+ technical indicators  
    3. Generate 5-day price predictions
    4. Create comprehensive visualizations
    5. Track prediction accuracy daily
    6. Deploy interactive dashboard
    """)
    
    # Create docs directory
    os.makedirs("docs", exist_ok=True)
    
    # Step 1: Train models (run weekly or when needed)
    success = run_command(
        "python enhanced_train_predict.py", 
        "Training ML Models with Historical Data (2014-Present)"
    )
    
    if not success:
        print("❌ Model training failed! Check enhanced_train_predict.py")
        return False
    
    # Step 2: Generate daily predictions and tracking
    success = run_command(
        "python daily_tracker.py",
        "Generating Daily Predictions & Tracking Accuracy"
    )
    
    if not success:
        print("❌ Daily tracking failed! Check daily_tracker.py")
        return False
    
    # Step 3: Create dashboard
    success = run_command(
        "python enhanced_dashboard.py",
        "Creating Interactive Dashboard"
    )
    
    if not success:
        print("❌ Dashboard creation failed! Check enhanced_dashboard.py")
        return False
    
    print(f"""
    🎉 ANALYSIS COMPLETE!
    =====================
    
    📊 Files Generated:
    ├── docs/
    │   ├── index.html                    # Interactive Dashboard
    │   ├── btc_predictions.csv          # 5-day predictions  
    │   ├── btc_report.txt               # Detailed analysis
    │   ├── btc_models.joblib            # Trained ML models
    │   ├── btc_predictions.png          # Prediction charts
    │   ├── btc_technical_analysis.png   # Technical indicators
    │   ├── prediction_history.json     # Accuracy tracking
    │   └── model_metrics.txt            # Performance metrics
    
    🌐 Dashboard URL: https://YOUR_USERNAME.github.io/YOUR_REPO/
    
    📈 Next Steps:
    1. Commit and push to GitHub
    2. Enable GitHub Pages
    3. Set up daily automation (GitHub Actions)
    4. Monitor prediction accuracy
    
    ⏰ Recommended Schedule:
    • Daily: Run daily_tracker.py + enhanced_dashboard.py  
    • Weekly: Run enhanced_train_predict.py (retrain models)
    """)
    
    return True

def daily_update():
    """Run daily update only (predictions + dashboard)"""
    print("🔄 Running Daily Bitcoin Analysis Update...")
    
    success1 = run_command("python daily_tracker.py", "Daily Predictions & Tracking")
    success2 = run_command("python enhanced_dashboard.py", "Updating Dashboard")
    
    if success1 and success2:
        print("✅ Daily update completed successfully!")
        return True
    else:
        print("❌ Daily update failed!")
        return False

def create_github_actions():
    """Create GitHub Actions workflow for automation"""
    
    os.makedirs(".github/workflows", exist_ok=True)
    
    workflow = """
name: Bitcoin Analysis Automation

on:
  schedule:
    # Run daily at 9 AM UTC (after markets open)
    - cron: '0 9 * * *'
  workflow_dispatch:  # Allow manual trigger

jobs:
  update-analysis:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install yfinance pandas numpy matplotlib scikit-learn joblib
    
    - name: Run daily analysis
      run: |
        python daily_tracker.py
        python enhanced_dashboard.py
    
    - name: Commit and push changes
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add docs/
        git commit -m "Auto-update Bitcoin analysis $(date)" || exit 0
        git push
"""
    
    with open(".github/workflows/bitcoin_analysis.yml", "w") as f:
        f.write(workflow)
    
    print("✅ GitHub Actions workflow created!")
    print("📁 File: .github/workflows/bitcoin_analysis.yml")

def main():
    parser = argparse.ArgumentParser(description='Bitcoin Analysis Automation')
    parser.add_argument('--mode', choices=['full', 'daily', 'setup'], 
                       default='full', help='Analysis mode')
    parser.add_argument('--install', action='store_true', 
                       help='Install required packages')
    parser.add_argument('--github-actions', action='store_true',
                       help='Create GitHub Actions workflow')
    
    args = parser.parse_args()
    
    if args.install:
        install_requirements()
    
    if args.github_actions:
        create_github_actions()
    
    if args.mode == 'full':
        full_analysis()
    elif args.mode == 'daily':
        daily_update()
    elif args.mode == 'setup':
        install_requirements()
        create_github_actions()
        print("🎯 Setup complete! Run 'python run_analysis.py --mode full' to start analysis.")

if __name__ == "__main__":
    main()
